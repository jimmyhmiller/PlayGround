use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use dynobj::{FrameChain, ObjHeader, RootSource, TypeInfo};

use crate::barrier::SATBBuffer;
use crate::heap::Heap;
use crate::semi_space::PtrPolicy;
use crate::statemap::TraceState;

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

    /// Incremented each time the thread enters a safepoint.
    /// Used by the GC to distinguish old safepoints from new ones.
    safepoint_gen: AtomicU64,

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
            safepoint_gen: AtomicU64::new(0),
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
        self.safepoint_gen.fetch_add(1, Ordering::AcqRel);
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

    /// Check if this thread is at a safepoint (atomic state only).
    pub fn is_at_safepoint(&self) -> bool {
        self.state.load(Ordering::Acquire) == STATE_AT_SAFEPOINT
    }

    /// Check if this thread is truly blocked at a safepoint with no
    /// stale resume pending. This acquires the safepoint lock to verify
    /// `gc_done` is false, meaning the thread is genuinely waiting on the
    /// condvar and won't spontaneously wake up.
    ///
    /// Use this instead of `is_at_safepoint()` when you need to ensure
    /// the thread is safe to scan and won't leave its safepoint until
    /// explicitly resumed.
    pub fn is_safely_at_safepoint(&self) -> bool {
        if self.state.load(Ordering::Acquire) != STATE_AT_SAFEPOINT {
            return false;
        }
        let gc_done = self.safepoint_lock.lock().unwrap();
        !*gc_done
    }

    /// Current safepoint generation (incremented each enter_safepoint call).
    pub fn safepoint_gen(&self) -> u64 {
        self.safepoint_gen.load(Ordering::Acquire)
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
        if self.heap.gc_requested() {
            if let Some(ref tracer) = self.heap.tracer {
                tracer.record_thread(self.thread_id, TraceState::AtSafepoint);
            }
            self.state.enter_safepoint();
            if let Some(ref tracer) = self.heap.tracer {
                tracer.record_thread(self.thread_id, TraceState::Running);
            }
        }
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

    /// Replication barrier: if `obj` has been forwarded during concurrent
    /// GC, replicate a field write to the to-space copy.
    ///
    /// Call this AFTER writing to a from-space object's field. If the
    /// object has been copied to to-space, the to-space copy's field
    /// will be updated to match, preventing the GC's STW re-scan from
    /// seeing stale data.
    ///
    /// Fast path when no GC is active: single atomic load + branch.
    ///
    /// # Safety
    /// - `obj` must point to a valid heap object.
    /// - `field_offset` must be a valid offset within the object.
    #[inline(always)]
    pub unsafe fn replication_barrier(&self, obj: *mut u8, field_offset: usize, new_bits: u64) {
        unsafe { self.heap.replication_barrier(obj, field_offset, new_bits) };
    }

    /// Generational write barrier: if a tenured object stores a nursery
    /// pointer, mark the corresponding card dirty.
    ///
    /// Call this AFTER writing to a field. The card table ensures minor
    /// GC can find old→young pointers without scanning all of tenured space.
    ///
    /// Fast path when no nursery: single branch on Option.
    ///
    /// # Safety
    /// - `obj` must point to a valid heap object.
    #[inline(always)]
    pub unsafe fn generational_write_barrier(&self, obj: *mut u8, new_bits: u64) {
        if !self.heap.has_nursery() {
            return;
        }
        if let Some(new_ptr) = P::try_decode_ptr(new_bits) {
            if self.heap.is_nursery(new_ptr) && self.heap.is_tenured(obj as *const u8) {
                self.heap.mark_card_dirty(obj as *const u8);
            }
        }
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

    /// Allocate an object from the heap. For generational heaps, allocates
    /// from the nursery first. If space is exhausted, triggers minor GC
    /// (generational) or major GC (non-generational) and retries.
    ///
    /// Returns null only if allocation still fails after GC.
    pub fn alloc(&self, info: &'static TypeInfo, varlen_len: usize) -> *mut u8 {
        if self.heap.gc_every_alloc() {
            self.trigger_gc();
        }
        let ptr = self.heap.alloc_nursery(info, varlen_len);
        if !ptr.is_null() {
            return ptr;
        }
        self.alloc_slow_path(info, varlen_len)
    }

    /// Allocate and initialize header + varlen count.
    /// Triggers GC if from-space is full.
    pub fn alloc_obj<H: ObjHeader>(&self, info: &'static TypeInfo, varlen_len: usize) -> *mut u8 {
        if self.heap.gc_every_alloc() {
            self.trigger_gc();
        }
        let ptr = self.heap.alloc_nursery_obj::<H>(info, varlen_len);
        if !ptr.is_null() {
            return ptr;
        }
        self.alloc_obj_slow_path::<H>(info, varlen_len)
    }

    #[cold]
    fn alloc_slow_path(&self, info: &'static TypeInfo, varlen_len: usize) -> *mut u8 {
        if self.heap.has_nursery() {
            // Try minor GC first → retry nursery
            self.trigger_minor_gc();
            let ptr = self.heap.alloc_nursery(info, varlen_len);
            if !ptr.is_null() {
                return ptr;
            }
            // Nursery still full (shouldn't happen after reset) → try major GC
            self.trigger_gc();
            let ptr = self.heap.alloc_nursery(info, varlen_len);
            if !ptr.is_null() {
                return ptr;
            }
            // Last resort: allocate directly in tenured
            self.heap.alloc_tenured(info, varlen_len)
        } else {
            self.trigger_gc();
            self.heap.alloc(info, varlen_len)
        }
    }

    #[cold]
    fn alloc_obj_slow_path<H: ObjHeader>(
        &self,
        info: &'static TypeInfo,
        varlen_len: usize,
    ) -> *mut u8 {
        if self.heap.has_nursery() {
            // Try minor GC first → retry nursery
            self.trigger_minor_gc();
            let ptr = self.heap.alloc_nursery_obj::<H>(info, varlen_len);
            if !ptr.is_null() {
                return ptr;
            }
            // Major GC
            self.trigger_gc();
            let ptr = self.heap.alloc_nursery_obj::<H>(info, varlen_len);
            if !ptr.is_null() {
                return ptr;
            }
            // Last resort: tenured
            self.heap.alloc_obj::<H>(info, varlen_len)
        } else {
            self.trigger_gc();
            self.heap.alloc_obj::<H>(info, varlen_len)
        }
    }

    /// Trigger a minor (nursery) GC from this mutator thread.
    ///
    /// If a concurrent major GC is in progress, waits for it to finish
    /// instead. If tenured space doesn't have enough room for worst-case
    /// promotion, triggers a major GC first to reclaim tenured space.
    fn trigger_minor_gc(&self) {
        if self.heap.barriers_active() {
            // Concurrent major GC in progress — wait for it
            while self.heap.barriers_active() {
                if self.heap.gc_requested() {
                    self.state.enter_safepoint();
                } else {
                    std::thread::yield_now();
                }
            }
            return;
        }
        // Check if tenured has enough room for worst-case promotion.
        // If not, run a major GC first to reclaim tenured space.
        if self.heap.nursery_used() > self.heap.from_remaining() {
            unsafe { self.heap.mutator_triggered_gc::<P>(&self.state) };
        }
        unsafe { self.heap.mutator_triggered_minor_gc::<P>(&self.state) };
    }

    /// Trigger a GC from this mutator thread.
    ///
    /// If a concurrent GC is already in progress, yields and waits for
    /// it to finish (participating in safepoints as needed). Otherwise,
    /// runs a full STW collection via `mutator_triggered_gc`.
    fn trigger_gc(&self) {
        if self.heap.barriers_active() {
            if let Some(ref tracer) = self.heap.tracer {
                tracer.record_thread(self.thread_id, TraceState::WaitingForGc);
            }
            while self.heap.barriers_active() {
                if self.heap.gc_requested() {
                    if let Some(ref tracer) = self.heap.tracer {
                        tracer.record_thread(self.thread_id, TraceState::AtSafepoint);
                    }
                    self.state.enter_safepoint();
                } else {
                    std::thread::yield_now();
                }
            }
            if let Some(ref tracer) = self.heap.tracer {
                tracer.record_thread(self.thread_id, TraceState::Running);
            }
            return;
        }
        unsafe { (*self.heap).mutator_triggered_gc::<P>(&self.state) };
    }
}

impl<P: PtrPolicy> Drop for MutatorThread<P> {
    fn drop(&mut self) {
        self.heap.safe_deregister_thread(&self.state);
    }
}
