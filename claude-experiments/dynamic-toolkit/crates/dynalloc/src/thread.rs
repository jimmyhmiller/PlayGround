use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use dynobj::{FrameChain, ObjHeader, RootSource, TypeInfo};

use crate::barrier::SATBBuffer;
use crate::heap::Heap;
use crate::semi_space::PtrPolicy;

// ─── Thread states ──────────────────────────────────────────────────

pub const STATE_RUNNING: u8 = 0;
pub const STATE_AT_SAFEPOINT: u8 = 1;

// ─── ThreadState ────────────────────────────────────────────────────

/// Per-thread GC coordination state.
///
/// Each mutator thread owns a `ThreadState` that contains:
/// - A `FrameChain` for shadow frame roots (per-thread, `Cell`-based)
/// - Safepoint synchronization primitives
///
/// The GC coordinator signals threads to reach safepoints, then scans
/// their roots while they're suspended.
pub struct ThreadState {
    /// Shadow frame chain for this thread's stack roots.
    pub frame_chain: FrameChain,

    /// Current state: STATE_RUNNING or STATE_AT_SAFEPOINT.
    state: AtomicU8,

    /// Mutex + Condvar for safepoint suspension.
    safepoint_lock: Mutex<bool>, // bool = "gc_done"
    safepoint_cond: Condvar,

    /// SATB write barrier buffer. Only accessed by the owning thread
    /// during mutation, or by the GC thread after safepoint is reached.
    /// UnsafeCell because it needs interior mutability without Sync.
    satb_buffer: UnsafeCell<SATBBuffer>,
}

// Safety: ThreadState is designed to be shared between the owning
// mutator thread and the GC coordinator thread.
// - `frame_chain` uses Cell (not Sync), but it's only accessed by:
//   (a) the owning thread during normal execution, or
//   (b) the GC thread after the owning thread has reached a safepoint
//       (happens-before established by the atomic state transition)
unsafe impl Sync for ThreadState {}

impl ThreadState {
    pub fn new() -> Self {
        ThreadState {
            frame_chain: FrameChain::new(),
            state: AtomicU8::new(STATE_RUNNING),
            safepoint_lock: Mutex::new(false),
            safepoint_cond: Condvar::new(),
            satb_buffer: UnsafeCell::new(SATBBuffer::new(256)),
        }
    }

    /// Access the SATB buffer.
    ///
    /// # Safety
    /// Must only be called by the owning thread during mutation,
    /// or by the GC thread after the owning thread is at a safepoint.
    pub unsafe fn satb_buffer(&self) -> &mut SATBBuffer {
        unsafe { &mut *self.satb_buffer.get() }
    }

    /// Enter safepoint: mark this thread as suspended and wait
    /// for the GC to finish.
    pub fn enter_safepoint(&self) {
        self.state.store(STATE_AT_SAFEPOINT, Ordering::Release);

        let mut gc_done = self.safepoint_lock.lock().unwrap();
        while !*gc_done {
            gc_done = self.safepoint_cond.wait(gc_done).unwrap();
        }

        *gc_done = false;
        self.state.store(STATE_RUNNING, Ordering::Release);
    }

    /// Resume this thread after GC completes.
    pub fn resume(&self) {
        let mut gc_done = self.safepoint_lock.lock().unwrap();
        *gc_done = true;
        self.safepoint_cond.notify_one();
        drop(gc_done);
    }

    /// Check if this thread is at a safepoint.
    pub fn is_at_safepoint(&self) -> bool {
        self.state.load(Ordering::Acquire) == STATE_AT_SAFEPOINT
    }
}

impl RootSource for ThreadState {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        self.frame_chain.scan_roots(visitor);
    }
}

// ─── MutatorThread ──────────────────────────────────────────────────

/// RAII handle for a mutator thread registered with a `Heap`.
///
/// This is the main API for mutator code:
/// - `alloc()` / `alloc_obj()` — allocate objects, auto-trigger GC if needed
/// - `safepoint()` — poll for GC requests (call at loop backedges, etc.)
/// - `frame_chain()` — access the shadow frame chain for rooting
///
/// On drop, deregisters from the heap.
///
/// # Example
/// ```rust,ignore
/// let heap = Arc::new(Heap::new::<Compact>(65536));
/// let mut mt = MutatorThread::register(heap.clone());
///
/// let frame = RootFrame::<2>::new();
/// let _guard = mt.frame_chain().push(&frame);
///
/// let obj = mt.alloc_obj::<Compact>(&PAIR_INFO, 0);
/// frame.slots[0].set(obj as u64);
///
/// mt.safepoint(); // check for GC
/// ```
pub struct MutatorThread<P: PtrPolicy> {
    heap: Arc<Heap>,
    state: Arc<ThreadState>,
    thread_id: usize,
    _marker: core::marker::PhantomData<P>,
}

impl<P: PtrPolicy> MutatorThread<P> {
    /// Register a new mutator thread with the heap.
    pub fn register(heap: Arc<Heap>) -> Self {
        let (state, thread_id) = heap.register_thread();
        MutatorThread {
            heap,
            state,
            thread_id,
            _marker: core::marker::PhantomData,
        }
    }

    /// Access the thread's shadow frame chain for rooting.
    pub fn frame_chain(&self) -> &FrameChain {
        &self.state.frame_chain
    }

    /// Access the thread state (for advanced usage).
    pub fn state(&self) -> &Arc<ThreadState> {
        &self.state
    }

    /// Access the heap.
    pub fn heap(&self) -> &Arc<Heap> {
        &self.heap
    }

    /// Poll for GC. If collection has been requested, this thread
    /// enters a safepoint (suspends) until the GC completes.
    ///
    /// Also flushes the SATB buffer if it's full.
    ///
    /// Call this at loop backedges and other natural safepoints.
    /// Fast path: single relaxed atomic load + branch.
    #[inline(always)]
    pub fn safepoint(&self) {
        // Flush SATB buffer if full (before entering safepoint)
        let buf = unsafe { self.state.satb_buffer() };
        if buf.should_flush() {
            self.heap.satb_queue.push(buf.drain());
        }
        self.heap.safepoint(&self.state);
    }

    /// Write barrier: call BEFORE overwriting a pointer field.
    ///
    /// Logs the old value in the thread-local SATB buffer so the
    /// concurrent GC can discover objects disconnected by the mutator.
    ///
    /// Fast path when no GC is active: single atomic load + branch.
    /// When GC is active: push to thread-local Vec (no synchronization).
    #[inline(always)]
    pub fn write_barrier(&self, old_value: u64) {
        if !self.heap.barriers_active() {
            return;
        }
        let buf = unsafe { self.state.satb_buffer() };
        buf.log(old_value);
    }

    /// Read barrier: follow forwarding pointer if object was relocated.
    ///
    /// During concurrent GC, objects may have been copied to to-space.
    /// This returns the up-to-date address.
    ///
    /// Fast path when no GC is active: returns ptr unchanged.
    ///
    /// # Safety
    /// - `ptr` must point to a valid heap object or be null.
    #[inline(always)]
    pub unsafe fn read_barrier(&self, ptr: *mut u8) -> *mut u8 {
        unsafe { self.heap.read_barrier(ptr) }
    }

    /// Allocate an object from the heap. If from-space is full,
    /// triggers a STW collection and retries once.
    ///
    /// Returns null only if allocation still fails after GC
    /// (i.e., live data exceeds half the heap).
    pub fn alloc(&self, info: &'static TypeInfo, varlen_len: usize) -> *mut u8 {
        let ptr = self.heap.alloc(info, varlen_len);
        if !ptr.is_null() {
            return ptr;
        }
        self.alloc_slow_path(info, varlen_len)
    }

    /// Allocate and initialize header + varlen count.
    /// Triggers GC if from-space is full.
    pub fn alloc_obj<H: ObjHeader>(
        &self,
        info: &'static TypeInfo,
        varlen_len: usize,
    ) -> *mut u8 {
        let ptr = self.heap.alloc_obj::<H>(info, varlen_len);
        if !ptr.is_null() {
            return ptr;
        }
        self.alloc_obj_slow_path::<H>(info, varlen_len)
    }

    #[cold]
    fn alloc_slow_path(&self, info: &'static TypeInfo, varlen_len: usize) -> *mut u8 {
        self.trigger_gc();
        self.heap.alloc(info, varlen_len)
    }

    #[cold]
    fn alloc_obj_slow_path<H: ObjHeader>(
        &self,
        info: &'static TypeInfo,
        varlen_len: usize,
    ) -> *mut u8 {
        self.trigger_gc();
        self.heap.alloc_obj::<H>(info, varlen_len)
    }

    /// Trigger a GC from this mutator thread.
    ///
    /// If a concurrent GC is already in progress, yields and waits for
    /// it to finish (participating in safepoints as needed). Otherwise,
    /// runs a full STW collection via `mutator_triggered_gc`.
    fn trigger_gc(&self) {
        if self.heap.barriers_active() {
            // Concurrent GC is running. Participate in safepoints
            // and wait until space becomes available.
            // The concurrent GC's STW pauses will request safepoints;
            // we just need to poll and yield.
            while self.heap.barriers_active() {
                self.heap.safepoint(&self.state);
                std::thread::yield_now();
            }
            return;
        }
        unsafe { (*self.heap).mutator_triggered_gc::<P>(&self.state) };
    }
}

impl<P: PtrPolicy> Drop for MutatorThread<P> {
    fn drop(&mut self) {
        self.heap.deregister_thread(self.thread_id);
    }
}
