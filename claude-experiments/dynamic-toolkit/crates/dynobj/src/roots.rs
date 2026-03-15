use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Trait for types that can enumerate GC root references.
///
/// The visitor receives `*mut u64` so the GC can update slots in-place
/// (e.g., for moving/forwarding pointers). `&self` enables composability —
/// multiple root sources can be scanned through shared references, behind
/// `Arc`, etc. Interior mutability (`Cell` for per-thread, `Mutex` for shared)
/// makes this safe.
pub trait RootSource {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64));
}

// Blanket impl: scan a slice of root sources.
impl<T: RootSource> RootSource for [T] {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for source in self {
            source.scan_roots(visitor);
        }
    }
}

// ─── Shadow Frame Chain (Fil-C style) ──────────────────────────────

/// Header for a shadow frame in the linked frame chain.
///
/// Each function that holds GC roots allocates a frame on its stack
/// (typically via `RootFrame<N>`). The frame header links into a
/// per-thread chain so the GC can walk all live roots.
///
/// Layout mirrors Fil-C's `filc_frame`:
/// ```text
/// ┌──────────────┐
/// │ parent       │  → previous frame (or null)
/// │ slot_count   │  → number of root slots in this frame
/// │ slots[0]     │  → u64 root values (GC-updateable)
/// │ slots[1]     │
/// │ ...          │
/// └──────────────┘
/// ```
#[repr(C)]
pub struct FrameHeader {
    parent: Cell<*mut FrameHeader>,
    slot_count: usize,
    // Followed by `slot_count` × Cell<u64> slots (in RootFrame<N>)
}

/// Stack-allocated shadow frame with `N` root slots.
///
/// Created by [`FrameChain::push`], which returns a [`FrameGuard`]
/// that pops the frame on drop.
///
/// The slots use `Cell<u64>` for interior mutability — the GC can
/// update them in-place during collection (forwarding pointers)
/// while the mutator holds shared references.
#[repr(C)]
pub struct RootFrame<const N: usize> {
    pub header: FrameHeader,
    pub slots: [Cell<u64>; N],
}

impl FrameHeader {
    /// Create a new frame header with the given slot count and null parent.
    pub fn new(slot_count: usize) -> Self {
        FrameHeader {
            parent: Cell::new(core::ptr::null_mut()),
            slot_count,
        }
    }
}

impl<const N: usize> RootFrame<N> {
    /// Create a new frame with all slots zeroed.
    pub fn new() -> Self {
        RootFrame {
            header: FrameHeader::new(N),
            slots: [const { Cell::new(0) }; N],
        }
    }
}

/// Per-thread chain of shadow frames.
///
/// Holds a pointer to the topmost frame. Functions push frames
/// at entry and pop them at return (via RAII [`FrameGuard`]).
///
/// The GC walks the chain by following parent pointers, visiting
/// all slots in each frame. Since slots are `Cell<u64>`, the GC
/// can update them in-place for pointer forwarding.
///
/// # Example (compiler-emitted pattern)
/// ```rust,ignore
/// let chain = FrameChain::new();
///
/// fn foo(chain: &FrameChain) {
///     let mut frame = RootFrame::<2>::new();
///     let guard = chain.push(&mut frame);
///     frame.slots[0].set(some_heap_ptr as u64);
///     frame.slots[1].set(another_ptr as u64);
///     // ... do work, may trigger GC which walks chain ...
///     // guard drops here, popping frame from chain
/// }
/// ```
pub struct FrameChain {
    top: Cell<*mut FrameHeader>,
}

impl FrameChain {
    pub fn new() -> Self {
        FrameChain {
            top: Cell::new(core::ptr::null_mut()),
        }
    }

    /// Push a frame onto the chain. Returns a guard that pops it on drop.
    ///
    /// # Safety contract
    /// The returned `FrameGuard` borrows the frame mutably, ensuring:
    /// - The frame outlives the guard (stack discipline)
    /// - Only one guard exists per frame
    /// Push a frame onto the chain. Returns a guard that pops it on drop.
    ///
    /// Takes `&RootFrame` because all mutable fields use `Cell`.
    pub fn push<'a, const N: usize>(&'a self, frame: &'a RootFrame<N>) -> FrameGuard<'a> {
        frame.header.parent.set(self.top.get());
        // Cast from the whole RootFrame pointer (not &frame.header) so the
        // raw pointer's provenance covers the trailing slots, not just the header.
        self.top.set(
            (frame as *const RootFrame<N>)
                .cast::<FrameHeader>()
                .cast_mut(),
        );
        FrameGuard { chain: self }
    }

    /// Push a raw FrameHeader pointer onto the chain.
    ///
    /// This supports dynamically-sized frames where `N` isn't known at
    /// compile time. The header must point to a contiguous block of
    /// `[FrameHeader][Cell<u64>; slot_count]`.
    ///
    /// # Safety
    /// - `header` must point to valid memory with `slot_count` Cell<u64> slots
    ///   immediately following the FrameHeader.
    /// - The memory must remain valid until the returned `FrameGuard` is dropped.
    pub unsafe fn push_raw<'a>(&'a self, header: *mut FrameHeader) -> FrameGuard<'a> {
        unsafe { (*header).parent.set(self.top.get()) };
        self.top.set(header);
        FrameGuard { chain: self }
    }

    /// Push a raw FrameHeader without returning a guard.
    ///
    /// The caller is responsible for calling [`pop_raw`] later to maintain
    /// stack discipline. This is intended for interpreters that manage
    /// frame lifetimes dynamically rather than via RAII guards.
    ///
    /// # Safety
    /// - `header` must point to valid memory with `slot_count` Cell<u64> slots
    ///   immediately following the FrameHeader.
    /// - The memory must remain valid until `pop_raw` is called.
    pub unsafe fn push_raw_unguarded(&self, header: *mut FrameHeader) {
        unsafe { (*header).parent.set(self.top.get()) };
        self.top.set(header);
    }

    /// Pop the top frame from the chain without a guard.
    ///
    /// # Safety
    /// - The chain must not be empty.
    /// - Must be called in LIFO order matching `push_raw_unguarded` calls.
    pub unsafe fn pop_raw(&self) {
        unsafe {
            let top = self.top.get();
            assert!(!top.is_null(), "pop_raw: chain is empty");
            self.top.set((*top).parent.get());
        }
    }

    /// Number of frames currently in the chain.
    pub fn depth(&self) -> usize {
        let mut count = 0;
        let mut cursor = self.top.get();
        while !cursor.is_null() {
            count += 1;
            cursor = unsafe { (*cursor).parent.get() };
        }
        count
    }
}

impl RootSource for FrameChain {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        let mut cursor = self.top.get();
        while !cursor.is_null() {
            unsafe {
                let header = &*cursor;
                // Slots start right after the FrameHeader
                let slots_base = cursor.add(1) as *mut Cell<u64>;
                for i in 0..header.slot_count {
                    let cell = &*slots_base.add(i);
                    visitor(cell.as_ptr());
                }
                cursor = header.parent.get();
            }
        }
    }
}

/// RAII guard that pops a frame from the chain on drop.
///
/// Ensures stack discipline: frames are always popped in reverse
/// push order (guaranteed by Rust's drop order for locals).
pub struct FrameGuard<'a> {
    chain: &'a FrameChain,
}

impl<'a> Drop for FrameGuard<'a> {
    fn drop(&mut self) {
        let top = self.chain.top.get();
        assert!(!top.is_null(), "FrameGuard::drop: chain is empty");
        unsafe {
            self.chain.top.set((*top).parent.get());
        }
    }
}

// ─── DynRootFrame (runtime-sized) ────────────────────────────────────

/// A dynamically-sized root frame compatible with [`FrameChain`].
///
/// In compiled code, each function allocates a `RootFrame<N>` on its stack
/// where N is known at compile time. For the interpreter, we don't know N
/// until we inspect the function's IR, so we heap-allocate a frame with
/// the right number of slots.
///
/// Memory layout (matches `RootFrame<N>`):
/// ```text
/// ┌───────────────────┐
/// │  FrameHeader      │  16 bytes (parent + slot_count)
/// ├───────────────────┤
/// │  slots[0]: u64    │  slot_count × 8 bytes
/// │  slots[1]: u64    │
/// │  ...              │
/// └───────────────────┘
/// ```
pub struct DynRootFrame {
    /// Raw allocation: [FrameHeader][Cell<u64> × slot_count]
    backing: Vec<u64>,
    slot_count: usize,
}

impl DynRootFrame {
    /// Create a new dynamic root frame with `slot_count` GC root slots.
    ///
    /// All slots are initialized to 0 (null).
    pub fn new(slot_count: usize) -> Self {
        let header_words = std::mem::size_of::<FrameHeader>() / 8;
        debug_assert_eq!(header_words, 2);
        let total_words = header_words + slot_count;
        let mut backing = vec![0u64; total_words];

        // Initialize the FrameHeader in-place.
        let header_ptr = backing.as_mut_ptr() as *mut FrameHeader;
        unsafe {
            header_ptr.write(FrameHeader::new(slot_count));
        }

        DynRootFrame {
            backing,
            slot_count,
        }
    }

    /// Get a raw pointer to the FrameHeader (for pushing onto a FrameChain).
    pub fn header_ptr(&self) -> *mut FrameHeader {
        self.backing.as_ptr() as *mut FrameHeader
    }

    /// Push this frame onto a FrameChain. Returns a guard that pops it on drop.
    ///
    /// # Safety
    /// The DynRootFrame must outlive the returned FrameGuard.
    pub fn push_onto<'a>(&'a self, chain: &'a FrameChain) -> FrameGuard<'a> {
        unsafe { chain.push_raw(self.header_ptr()) }
    }

    /// Number of root slots.
    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Get the value in slot `i`.
    pub fn get(&self, i: usize) -> u64 {
        assert!(
            i < self.slot_count,
            "slot index {i} >= slot_count {}",
            self.slot_count
        );
        let header_words = std::mem::size_of::<FrameHeader>() / 8;
        let slot_ptr = unsafe {
            (self.backing.as_ptr().add(header_words + i) as *const Cell<u64>)
                .as_ref()
                .unwrap()
        };
        slot_ptr.get()
    }

    /// Set the value in slot `i`.
    pub fn set(&self, i: usize, val: u64) {
        assert!(
            i < self.slot_count,
            "slot index {i} >= slot_count {}",
            self.slot_count
        );
        let header_words = std::mem::size_of::<FrameHeader>() / 8;
        let slot_ptr = unsafe {
            (self.backing.as_ptr().add(header_words + i) as *const Cell<u64>)
                .as_ref()
                .unwrap()
        };
        slot_ptr.set(val);
    }

    /// Zero all slots (used before mirroring live values at safepoints).
    pub fn clear_all(&self) {
        for i in 0..self.slot_count {
            self.set(i, 0);
        }
    }
}

// ─── RootSet (for globals, constants, pinned roots) ────────────────

/// A growable set of GC roots for globals, constants, or other
/// long-lived values that don't follow stack discipline.
///
/// Uses `Cell<u64>` slots so the GC can update values in-place.
pub struct RootSet {
    slots: Vec<Cell<u64>>,
}

impl RootSet {
    pub fn new() -> Self {
        RootSet { slots: Vec::new() }
    }

    /// Add a root value, returning its index for later access.
    pub fn add(&mut self, bits: u64) -> usize {
        let index = self.slots.len();
        self.slots.push(Cell::new(bits));
        index
    }

    /// Read a root's current value.
    pub fn get(&self, index: usize) -> u64 {
        self.slots[index].get()
    }

    /// Update a root's value.
    pub fn set(&self, index: usize, bits: u64) {
        self.slots[index].set(bits);
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

impl RootSource for RootSet {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for cell in &self.slots {
            visitor(cell.as_ptr());
        }
    }
}

// ─── AtomicRootSet (thread-safe globals) ────────────────────────────

/// Thread-safe set of GC roots for globals and shared constants.
///
/// Uses `AtomicU64` for slots (lock-free read/write/scan) and a
/// `Mutex` to synchronize growth (`add`). Safe to share across
/// threads via `Arc` or `&'static`.
///
/// The GC scans this during STW, so atomic ordering on individual
/// slot access can be `Relaxed` — the safepoint handshake provides
/// the necessary happens-before.
pub struct AtomicRootSet {
    slots: Mutex<Vec<AtomicU64>>,
}

// AtomicRootSet is Send + Sync by construction:
// - Vec<AtomicU64> behind Mutex for growth
// - AtomicU64 for individual slot access
unsafe impl Sync for AtomicRootSet {}

impl AtomicRootSet {
    pub fn new() -> Self {
        AtomicRootSet {
            slots: Mutex::new(Vec::new()),
        }
    }

    /// Add a root value, returning its index for later access.
    ///
    /// Takes the lock to grow the vector. The returned index
    /// can be used with `get`/`set` without locking.
    pub fn add(&self, bits: u64) -> usize {
        let mut slots = self.slots.lock().unwrap();
        let index = slots.len();
        slots.push(AtomicU64::new(bits));
        index
    }

    /// Read a root's current value (lock-free).
    pub fn get(&self, index: usize) -> u64 {
        let slots = self.slots.lock().unwrap();
        slots[index].load(Ordering::Relaxed)
    }

    /// Update a root's value (lock-free for the atomic store,
    /// but takes lock to access the vec).
    pub fn set(&self, index: usize, bits: u64) {
        let slots = self.slots.lock().unwrap();
        slots[index].store(bits, Ordering::Relaxed);
    }

    pub fn len(&self) -> usize {
        self.slots.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.lock().unwrap().is_empty()
    }
}

impl RootSource for AtomicRootSet {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        let slots = self.slots.lock().unwrap();
        for atomic in slots.iter() {
            visitor(atomic.as_ptr() as *mut u64);
        }
    }
}
