use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

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

    /// Direct access to slot `i`. The slot is the GC's canonical storage
    /// for that root: the collector may rewrite it during a moving
    /// collection, so the cell type guarantees interior mutability is safe
    /// for shared (`&self`) borrows. Used by the `Rooted` / `RootScope`
    /// abstraction to hand out slot references that are stable across
    /// GC points.
    pub fn slot(&self, i: usize) -> &Cell<u64> {
        assert!(
            i < self.slot_count,
            "slot index {i} >= slot_count {}",
            self.slot_count
        );
        let header_words = std::mem::size_of::<FrameHeader>() / 8;
        unsafe {
            (self.backing.as_ptr().add(header_words + i) as *const Cell<u64>)
                .as_ref()
                .unwrap()
        }
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

// ─── Rooted<T> + RootScope (high-level handle abstraction) ─────────
//
// `Rooted<'scope, T>` and `RootScope` are the safe, hard-to-misuse face
// of the rooting machinery. Frontends that hold u64 GC handles in Rust
// across allocation points (every allocator extern called from JIT, every
// host-side helper that walks heap-allocated trees) should pin them in a
// `RootScope`. The compiler's borrow checker enforces that:
//
//   * a `Rooted` cannot outlive its scope (the slot would be stale),
//   * a `Rooted` cannot be fabricated from a raw u64 (no public ctor),
//   * the only way to *read* the current value is via `.get()`, which
//     re-loads through the slot — so a moving GC that rewrote the slot
//     in place is observed correctly on the next access.
//
// The scope's storage is a `DynRootFrame` registered with the active
// `FrameChain`. The collector traces the chain (via the
// `RootSource for FrameChain` impl), so every live `Rooted` is a real
// GC root for the duration of its scope.

use std::marker::PhantomData;

/// A heap value pinned for the GC. Holding `Rooted<'scope, T>` keeps the
/// underlying slot scannable: the GC sees the slot as a root and rewrites
/// it in place across moving collections. Reading via `.get()` always
/// observes the current (possibly relocated) value.
///
/// `Rooted` is bound to a `RootScope` by lifetime — it cannot outlive the
/// scope it was created from. There is no public constructor that takes a
/// raw `u64`; the only way to mint a `Rooted` is through `RootScope::root`.
/// That eliminates the "I cached the raw pointer somewhere" footgun at
/// the API layer.
///
/// `T` is a phantom type tag for documentation (e.g., `NanBox`, `GcPtr`).
/// It plays no runtime role today; future work can give it `Rootable` /
/// `from_bits` semantics for type-safe access.
#[must_use = "a Rooted must be held in a binding to keep the slot rooted"]
pub struct Rooted<'scope, T: ?Sized> {
    slot: &'scope Cell<u64>,
    _phantom: PhantomData<*const T>,
}

// `*const T` makes Rooted !Send + !Sync regardless of T — the underlying
// FrameChain is per-thread.

impl<'scope, T: ?Sized> Rooted<'scope, T> {
    /// Read the current value through the slot. Re-loads each call: a
    /// moving GC that ran since the last read will have already written
    /// the relocated address into the slot.
    #[inline]
    pub fn get(&self) -> u64 {
        self.slot.get()
    }

    /// Update the rooted value. The GC will see the new value at the next
    /// collection.
    #[inline]
    pub fn set(&self, value: u64) {
        self.slot.set(value);
    }
}

impl<'scope, T> std::fmt::Debug for Rooted<'scope, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rooted(0x{:016x})", self.slot.get())
    }
}

/// A LIFO-scoped batch of GC roots backed by a `DynRootFrame` registered
/// with the active `FrameChain`. Bump-allocates slots up to a fixed
/// capacity. The frame is unregistered (popped) when the scope drops.
///
/// Construct one per logical work unit that needs to hold heap handles
/// across allocations. Typical uses: an extern's body, a recursive walker
/// in the host frontend, a code-emission helper that holds tree nodes.
pub struct RootScope<'chain> {
    /// Owned frame storage. Boxed so its address is stable across moves
    /// of the `RootScope` itself; `Rooted` borrows directly into it.
    frame: Box<DynRootFrame>,
    /// Bump cursor — next free slot index.
    cursor: Cell<usize>,
    _guard: FrameGuard<'chain>,
}

impl<'chain> RootScope<'chain> {
    /// Create a new scope with `capacity` slots. The scope registers
    /// itself with `chain` for the lifetime of the returned value; it is
    /// popped on drop.
    pub fn new(chain: &'chain FrameChain, capacity: usize) -> Self {
        let frame = Box::new(DynRootFrame::new(capacity));
        let _guard = unsafe { chain.push_raw(frame.header_ptr()) };
        RootScope {
            frame,
            cursor: Cell::new(0),
            _guard,
        }
    }

    /// Mint a fresh `Rooted` initialized to `value`. The handle borrows
    /// from this scope and cannot outlive it. Panics if the scope's slot
    /// capacity is exhausted.
    pub fn root<T>(&self, value: u64) -> Rooted<'_, T> {
        let i = self.cursor.get();
        assert!(
            i < self.frame.slot_count(),
            "RootScope: capacity {} exhausted",
            self.frame.slot_count()
        );
        let slot = self.frame.slot(i);
        slot.set(value);
        self.cursor.set(i + 1);
        Rooted {
            slot,
            _phantom: PhantomData,
        }
    }

    pub fn capacity(&self) -> usize {
        self.frame.slot_count()
    }

    pub fn used(&self) -> usize {
        self.cursor.get()
    }
}

// ─── Thread-local active FrameChain ─────────────────────────────────
//
// Most callers don't want to thread a `&FrameChain` through every
// function signature. The runtime/Engine installs a chain once per
// thread; helpers that need rooting reach for it via `with_scope`.

thread_local! {
    static ACTIVE_CHAIN: Cell<*const FrameChain> = const { Cell::new(std::ptr::null()) };
}

/// Install `chain` as the thread-local active frame chain. The previous
/// installation is restored when the returned guard drops.
pub fn install_chain<'a>(chain: &'a FrameChain) -> InstallChainGuard<'a> {
    let prev = ACTIVE_CHAIN.with(|c| {
        let p = c.get();
        c.set(chain as *const _);
        p
    });
    InstallChainGuard {
        prev,
        _phantom: PhantomData,
    }
}

/// RAII guard restoring the previous thread-local chain on drop.
pub struct InstallChainGuard<'a> {
    prev: *const FrameChain,
    _phantom: PhantomData<&'a FrameChain>,
}

impl Drop for InstallChainGuard<'_> {
    fn drop(&mut self) {
        ACTIVE_CHAIN.with(|c| c.set(self.prev));
    }
}

/// Run `f` within a fresh `RootScope` of `capacity` slots, attached to
/// the thread's active frame chain. Panics if no chain is installed.
///
/// Typical pattern at an FFI / extern boundary:
/// ```ignore
/// extern "C" fn ml_cons(car: u64, cdr: u64) -> u64 {
///     dynobj::roots::with_scope(2, |scope| {
///         let car = scope.root::<NanBox>(car);
///         let cdr = scope.root::<NanBox>(cdr);
///         // ... can call gc.alloc here; car/cdr re-read via .get() ...
///     })
/// }
/// ```
pub fn with_scope<R>(capacity: usize, f: impl FnOnce(&RootScope<'_>) -> R) -> R {
    let chain_ptr = ACTIVE_CHAIN.with(|c| c.get());
    assert!(
        !chain_ptr.is_null(),
        "with_scope: no FrameChain installed on this thread; \
         call dynobj::roots::install_chain(&chain) first",
    );
    // Safety: `install_chain` recorded a borrow tied to its guard, which
    // outlives any scope created here per the runtime contract.
    let chain: &FrameChain = unsafe { &*chain_ptr };
    let scope = RootScope::new(chain, capacity);
    f(&scope)
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
/// Lock-free reads via RCU (Read-Copy-Update) pattern:
/// - `get`/`set` load a raw pointer (Relaxed) and do an atomic
///   load/store on the slot. No mutex, no RwLock — just pointer
///   arithmetic + atomic access (~1-2ns).
/// - `add` takes a mutex to grow the backing Vec, then publishes
///   the (possibly new) buffer pointer with Release ordering.
/// - `scan_roots` iterates slots during STW — no concurrent mutation.
///
/// Safety relies on the single-threaded mutator invariant: `add()`
/// and `get()`/`set()` are never called concurrently. The GC only
/// calls `scan_roots` during STW pauses. See per-method safety notes.
pub struct AtomicRootSet {
    /// Raw pointer to the Vec's backing buffer. Hot-path reads
    /// go through this pointer — no lock needed.
    ptr: AtomicPtr<AtomicU64>,

    /// Current number of live slots.
    len: AtomicUsize,

    /// Mutex protects the Vec (which owns the allocation) during
    /// growth. Never touched on the hot path (get/set).
    backing: Mutex<Vec<AtomicU64>>,
}

// AtomicRootSet is Send + Sync:
// - AtomicPtr/AtomicUsize are Send+Sync
// - Mutex<Vec<AtomicU64>> is Send+Sync
// - Raw pointer dereferences in get/set are safe due to
//   single-threaded mutator invariant (no concurrent add+get)
unsafe impl Sync for AtomicRootSet {}

impl AtomicRootSet {
    pub fn new() -> Self {
        AtomicRootSet {
            ptr: AtomicPtr::new(core::ptr::null_mut()),
            len: AtomicUsize::new(0),
            backing: Mutex::new(Vec::new()),
        }
    }

    /// Add a root value, returning its index for later access.
    ///
    /// Takes the mutex to grow the vector, then publishes the
    /// (possibly reallocated) buffer pointer with Release ordering.
    ///
    /// # Safety invariant
    /// Must not be called concurrently with `get`/`set` on the same
    /// indices that might be affected by reallocation. In practice,
    /// the single-threaded mutator guarantees this.
    pub fn add(&self, bits: u64) -> usize {
        let mut vec = self.backing.lock().unwrap();
        let index = vec.len();
        vec.push(AtomicU64::new(bits));
        // Publish the (possibly new) buffer pointer and length.
        // Release ordering ensures the new slot's data is visible
        // before any reader sees the updated ptr/len.
        self.ptr.store(vec.as_mut_ptr(), Ordering::Release);
        self.len.store(vec.len(), Ordering::Release);
        index
    }

    /// Read a root's current value. **Lock-free.**
    ///
    /// Two instructions on the hot path: load ptr, indexed atomic load.
    #[inline]
    pub fn get(&self, index: usize) -> u64 {
        let ptr = self.ptr.load(Ordering::Relaxed);
        debug_assert!(!ptr.is_null(), "AtomicRootSet::get on empty set");
        debug_assert!(
            index < self.len.load(Ordering::Relaxed),
            "AtomicRootSet::get index {index} out of bounds"
        );
        unsafe { (*ptr.add(index)).load(Ordering::Relaxed) }
    }

    /// Update a root's value. **Lock-free.**
    #[inline]
    pub fn set(&self, index: usize, bits: u64) {
        let ptr = self.ptr.load(Ordering::Relaxed);
        debug_assert!(!ptr.is_null(), "AtomicRootSet::set on empty set");
        debug_assert!(
            index < self.len.load(Ordering::Relaxed),
            "AtomicRootSet::set index {index} out of bounds"
        );
        unsafe { (*ptr.add(index)).store(bits, Ordering::Relaxed) }
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl RootSource for AtomicRootSet {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        let ptr = self.ptr.load(Ordering::Relaxed);
        let len = self.len.load(Ordering::Relaxed);
        for i in 0..len {
            unsafe {
                visitor((*ptr.add(i)).as_ptr());
            }
        }
    }
}

// ─── AtomicRootSet tests ───────────────────────────────────────────

#[cfg(test)]
mod atomic_root_set_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::thread;

    // --- Basic functionality ---

    #[test]
    fn add_and_get() {
        let rs = AtomicRootSet::new();
        assert!(rs.is_empty());
        assert_eq!(rs.len(), 0);

        let i0 = rs.add(100);
        let i1 = rs.add(200);
        let i2 = rs.add(300);

        assert_eq!(i0, 0);
        assert_eq!(i1, 1);
        assert_eq!(i2, 2);
        assert_eq!(rs.len(), 3);
        assert!(!rs.is_empty());

        assert_eq!(rs.get(0), 100);
        assert_eq!(rs.get(1), 200);
        assert_eq!(rs.get(2), 300);
    }

    #[test]
    fn set_updates_value() {
        let rs = AtomicRootSet::new();
        rs.add(42);
        assert_eq!(rs.get(0), 42);

        rs.set(0, 99);
        assert_eq!(rs.get(0), 99);
    }

    #[test]
    fn scan_roots_visits_all_slots() {
        let rs = AtomicRootSet::new();
        rs.add(10);
        rs.add(20);
        rs.add(30);

        let mut visited = Vec::new();
        rs.scan_roots(&mut |slot_ptr| {
            visited.push(unsafe { *slot_ptr });
        });
        assert_eq!(visited, vec![10, 20, 30]);
    }

    #[test]
    fn scan_roots_can_update_in_place() {
        let rs = AtomicRootSet::new();
        rs.add(100);
        rs.add(200);

        // Simulate GC forwarding: double every root value
        rs.scan_roots(&mut |slot_ptr| {
            unsafe {
                let old = *slot_ptr;
                *slot_ptr = old * 2;
            }
        });

        assert_eq!(rs.get(0), 200);
        assert_eq!(rs.get(1), 400);
    }

    #[test]
    fn many_adds_trigger_reallocation() {
        let rs = AtomicRootSet::new();
        for i in 0..1000u64 {
            rs.add(i * 7);
        }
        assert_eq!(rs.len(), 1000);
        for i in 0..1000u64 {
            assert_eq!(rs.get(i as usize), i * 7);
        }
    }

    // --- Multi-threaded tests ---

    #[test]
    fn concurrent_readers_no_contention() {
        // Pre-populate, then spawn many reader threads.
        // This is the hot-path scenario: many reads, no writes.
        let rs = Arc::new(AtomicRootSet::new());
        let n = 100;
        for i in 0..n {
            rs.add(i as u64 * 11);
        }

        let barrier = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let rs = Arc::clone(&rs);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..100_000 {
                        for i in 0..n {
                            assert_eq!(rs.get(i), i as u64 * 11);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn concurrent_readers_and_writer() {
        // One writer thread does set() on existing slots.
        // Multiple reader threads do get().
        // Readers should always see either old or new value, never garbage.
        let rs = Arc::new(AtomicRootSet::new());
        let n = 64;
        for i in 0..n {
            rs.add(i as u64);
        }

        let barrier = Arc::new(Barrier::new(5));
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Writer: continuously update slot values
        let writer = {
            let rs = Arc::clone(&rs);
            let barrier = Arc::clone(&barrier);
            let done = Arc::clone(&done);
            thread::spawn(move || {
                barrier.wait();
                for round in 1u64..=50_000 {
                    for i in 0..n {
                        rs.set(i, round * 1000 + i as u64);
                    }
                }
                done.store(true, Ordering::Release);
            })
        };

        // Readers: continuously read and verify consistency per-slot
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let rs = Arc::clone(&rs);
                let barrier = Arc::clone(&barrier);
                let done = Arc::clone(&done);
                thread::spawn(move || {
                    barrier.wait();
                    let mut reads = 0u64;
                    while !done.load(Ordering::Acquire) {
                        for i in 0..n {
                            let val = rs.get(i);
                            // Value is either initial (i) or round*1000+i
                            if val >= 1000 {
                                let slot = val % 1000;
                                assert_eq!(
                                    slot, i as u64,
                                    "corrupted read: slot {i} had value {val}"
                                );
                            } else {
                                assert_eq!(
                                    val, i as u64,
                                    "corrupted read: slot {i} had value {val}"
                                );
                            }
                            reads += 1;
                        }
                    }
                    reads
                })
            })
            .collect();

        writer.join().unwrap();
        let total_reads: u64 = readers.into_iter().map(|h| h.join().unwrap()).sum();
        assert!(
            total_reads > 0,
            "readers should have done work"
        );
    }

    #[test]
    fn concurrent_add_from_single_thread_then_read_from_many() {
        // Simulates init phase (single-threaded add) followed by
        // multi-threaded read phase (JIT execution).
        let rs = Arc::new(AtomicRootSet::new());

        // Init phase: add a bunch of roots
        for i in 0..500u64 {
            rs.add(i * 3 + 7);
        }

        // Read phase: many threads verify all values
        let barrier = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let rs = Arc::clone(&rs);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..10_000 {
                        for i in 0..500usize {
                            assert_eq!(rs.get(i), i as u64 * 3 + 7);
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn simulated_gc_scan_while_readers_active() {
        // Simulates the GC scanning pattern: one thread does scan_roots
        // (forwarding pointers) while other threads read via get().
        // In production this happens during STW, but we test that
        // the atomic operations are correct even without STW.
        let rs = Arc::new(AtomicRootSet::new());
        let n = 32;
        for i in 0..n {
            rs.add(i as u64 + 1); // values 1..=32
        }

        let barrier = Arc::new(Barrier::new(5));
        let gc_done = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // GC thread: repeatedly scan and "forward" (add 1000 to each value)
        let gc_thread = {
            let rs = Arc::clone(&rs);
            let barrier = Arc::clone(&barrier);
            let gc_done = Arc::clone(&gc_done);
            thread::spawn(move || {
                barrier.wait();
                for _cycle in 0..100 {
                    rs.scan_roots(&mut |slot_ptr| {
                        unsafe {
                            // Atomic store through the raw pointer
                            let atomic = slot_ptr as *const AtomicU64;
                            let old = (*atomic).load(Ordering::Relaxed);
                            (*atomic).store(old + 1000, Ordering::Relaxed);
                        }
                    });
                }
                gc_done.store(true, Ordering::Release);
            })
        };

        // Reader threads: read values, verify they're sensible
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let rs = Arc::clone(&rs);
                let barrier = Arc::clone(&barrier);
                let gc_done = Arc::clone(&gc_done);
                thread::spawn(move || {
                    barrier.wait();
                    while !gc_done.load(Ordering::Acquire) {
                        for i in 0..n {
                            let val = rs.get(i);
                            // Value should be (i+1) + k*1000 for some k >= 0
                            let base = val % 1000;
                            assert_eq!(
                                base,
                                i as u64 + 1,
                                "corrupted: slot {i} = {val}, base = {base}"
                            );
                        }
                    }
                })
            })
            .collect();

        gc_thread.join().unwrap();
        for h in readers {
            h.join().unwrap();
        }

        // After 100 GC cycles, each value should have been incremented
        // by 100*1000 = 100000
        for i in 0..n {
            assert_eq!(rs.get(i), i as u64 + 1 + 100_000);
        }
    }

    #[test]
    fn stress_add_then_concurrent_get_set() {
        // Stress test: large root set, concurrent get/set from many threads
        let rs = Arc::new(AtomicRootSet::new());
        let n = 2048;
        for i in 0..n {
            rs.add(i as u64);
        }

        let barrier = Arc::new(Barrier::new(16));
        let handles: Vec<_> = (0..16)
            .map(|tid| {
                let rs = Arc::clone(&rs);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    // Each thread works on its own slice of indices
                    let start = (tid * n) / 16;
                    let end = ((tid + 1) * n) / 16;
                    for _ in 0..10_000 {
                        for i in start..end {
                            let val = rs.get(i);
                            rs.set(i, val.wrapping_add(1));
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Each slot was incremented 10_000 times by exactly one thread
        for i in 0..n {
            assert_eq!(rs.get(i), i as u64 + 10_000);
        }
    }

    #[test]
    fn scan_roots_empty_set() {
        let rs = AtomicRootSet::new();
        let mut count = 0;
        rs.scan_roots(&mut |_| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn add_interleaved_with_get_set_single_thread() {
        // Simulates the define_global pattern: add() during execution,
        // interleaved with get()/set() on existing indices.
        let rs = AtomicRootSet::new();

        let i0 = rs.add(10);
        assert_eq!(rs.get(i0), 10);
        rs.set(i0, 11);
        assert_eq!(rs.get(i0), 11);

        let i1 = rs.add(20);
        // Old slot still accessible after growth
        assert_eq!(rs.get(i0), 11);
        assert_eq!(rs.get(i1), 20);

        // Add many more to force reallocation
        for i in 0..100 {
            rs.add(i * 5);
        }
        // Original slots still correct
        assert_eq!(rs.get(i0), 11);
        assert_eq!(rs.get(i1), 20);
        // New slots correct
        assert_eq!(rs.get(2), 0);
        assert_eq!(rs.get(50), 48 * 5);
    }
}
