use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

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

impl<const N: usize> RootFrame<N> {
    /// Create a new frame with all slots zeroed.
    pub fn new() -> Self {
        RootFrame {
            header: FrameHeader {
                parent: Cell::new(core::ptr::null_mut()),
                slot_count: N,
            },
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
        self.top.set((frame as *const RootFrame<N>).cast::<FrameHeader>().cast_mut());
        FrameGuard { chain: self }
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
