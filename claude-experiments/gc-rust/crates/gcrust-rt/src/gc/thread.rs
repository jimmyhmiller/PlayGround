use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::gc::header::ObjHeader;
use crate::gc::roots::{FrameChain, RootSource};
use crate::gc::type_info::TypeInfo;

use crate::gc::barrier::SATBBuffer;
use crate::gc::heap::Heap;
use crate::gc::semi_space::PtrPolicy;
use crate::gc::statemap::TraceState;

// ─── Thread states ──────────────────────────────────────────────────

pub const STATE_RUNNING: u8 = 0;
pub const STATE_AT_SAFEPOINT: u8 = 1;
/// Thread is parked on an external blocking call (mutex, condvar, IO).
/// The GC treats this state as "safepoint-equivalent": it scans the
/// thread's roots without waiting for the thread to poll. The thread
/// must transition back to RUNNING via `exit_blocked` before resuming
/// any mutator work.
pub const STATE_BLOCKED: u8 = 2;

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

    /// Frame pointer of the JIT call this thread is currently parked
    /// at. Set by the JIT safepoint handler **before** parking, so a
    /// concurrent GC running on another thread can walk this thread's
    /// JIT frames as roots. Cleared (set null) on resume.
    ///
    /// Without this, a concurrent GC scans only the parked thread's
    /// `frame_chain` (host-side roots), missing the JIT-frame spill
    /// slots — and so doesn't update them when relocating objects.
    /// The parked thread then resumes and reads stale `from`-space
    /// pointers (SIGBUS).
    ///
    /// `AtomicPtr` because both the owning thread (set/clear) and
    /// the GC coordinator (read) touch it; the value is a raw FP
    /// that points into the owning thread's stack — valid only while
    /// the owning thread is parked.
    parked_jit_fp: AtomicPtr<u8>,

    /// Pointer to the owning runtime `Thread`'s `state` byte — the flag
    /// the JIT inline safepoint poll reads. The GC coordinator stores a
    /// non-zero value here (via [`request_poll`](Self::request_poll))
    /// when it wants this thread to park, so a thread spinning in JIT'd
    /// code (e.g. an allocation-free self-tail loop) reaches a safepoint.
    /// Null for threads with no runtime `Thread` (e.g. pure-Rust
    /// `MutatorThread`s that poll `gc_requested` directly).
    ///
    /// Set once via [`set_poll_flag`](Self::set_poll_flag) right after
    /// the owning `Thread` is constructed; the address is stable for the
    /// `Thread`'s lifetime (it lives in a `Box`).
    poll_flag: AtomicPtr<u8>,

    /// The OS thread that owns (and registered) this state. Recorded at
    /// construction — registration always happens on the owning thread.
    /// Used by [`Heap::pause_world`](crate::gc::Heap::pause_world) to
    /// exclude the requesting thread from the set it parks.
    os_thread: std::thread::ThreadId,

    /// Scratch GC-root STACK for runtime functions that hold heap pointers
    /// across an allocation (which may collect + relocate). A fn marks the
    /// current depth, pushes its live heap-pointer args, allocates, re-reads
    /// the (relocated) values, then resets to its mark. A stack (not fixed
    /// slots) is required because these calls NEST — e.g. `atom_swap`'s
    /// updater closure calls `string_concat`, which also needs scratch — so
    /// the inner use pushes above the outer's roots and resets back without
    /// clobbering them. Slots `[0..depth)` are scanned by `scan_roots`;
    /// values above `depth` are ignored. Touched only by the owning thread
    /// (between safepoints) or the collector (while it's parked).
    gc_scratch: [std::sync::atomic::AtomicU64; GC_SCRATCH_DEPTH],
    scratch_depth: std::sync::atomic::AtomicUsize,
}

/// Maximum nesting of runtime alloc fns parking scratch roots at once.
pub const GC_SCRATCH_DEPTH: usize = 32;

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
            parked_jit_fp: AtomicPtr::new(std::ptr::null_mut()),
            poll_flag: AtomicPtr::new(std::ptr::null_mut()),
            os_thread: std::thread::current().id(),
            gc_scratch: Default::default(),
            scratch_depth: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Remember the current scratch-stack depth so a nested run of
    /// allocations can be unwound back to here with [`scratch_reset`]. Call
    /// at the top of a runtime fn that parks roots; pass the result to
    /// `scratch_reset` before returning.
    pub fn scratch_mark(&self) -> usize {
        self.scratch_depth.load(Ordering::Acquire)
    }

    /// Push a heap pointer onto the scratch stack so it survives (and is
    /// relocated by) a collection during a subsequent allocation. Returns
    /// the slot index; re-read the (possibly relocated) pointer with
    /// [`scratch_at`](Self::scratch_at) after each alloc. Panics if the
    /// stack overflows `GC_SCRATCH_DEPTH` (runtime-fn nesting is shallow).
    pub fn push_scratch(&self, ptr: *const u8) -> usize {
        let i = self.scratch_depth.load(Ordering::Acquire);
        assert!(
            i < GC_SCRATCH_DEPTH,
            "GC scratch stack overflow (depth {GC_SCRATCH_DEPTH}); runtime-fn nesting too deep"
        );
        self.gc_scratch[i].store(ptr as u64, Ordering::Release);
        self.scratch_depth.store(i + 1, Ordering::Release);
        i
    }

    /// Read scratch slot `i` (the post-relocation pointer).
    pub fn scratch_at(&self, i: usize) -> *mut u8 {
        self.gc_scratch[i].load(Ordering::Acquire) as *mut u8
    }

    /// Pop the scratch stack back to a depth previously returned by
    /// [`scratch_mark`](Self::scratch_mark), so a later collection doesn't
    /// pin stale pointers.
    pub fn scratch_reset(&self, mark: usize) {
        self.scratch_depth.store(mark, Ordering::Release);
    }

    /// Open a scoped scratch region that auto-resets on drop. Preferred over
    /// manual `scratch_mark`/`scratch_reset` in Rust code with `?`
    /// early-returns (wire decode, arg builders): the `Drop` runs on every
    /// exit path, so a decode error can't leak stale scratch roots that a
    /// later collection would wrongly pin/scan.
    pub fn scratch_scope(&self) -> ScratchScope<'_> {
        ScratchScope {
            ts: self,
            mark: self.scratch_mark(),
        }
    }

    /// The OS thread that owns this state.
    pub fn os_thread(&self) -> std::thread::ThreadId {
        self.os_thread
    }

    /// Point this thread's safepoint poll flag at the owning runtime
    /// `Thread`'s `state` byte. Called once, right after the `Thread` is
    /// built, before the thread starts running JIT'd code.
    pub fn set_poll_flag(&self, state_byte: *mut u8) {
        self.poll_flag.store(state_byte, Ordering::Release);
    }

    /// Ask this thread to reach a safepoint by raising its JIT poll flag.
    /// No-op if the thread has no runtime `Thread` (null flag). The store
    /// is a benign single-byte race with the JIT's plain-load poll: the
    /// thread is guaranteed to observe it on a subsequent poll (cache
    /// coherence), which is all liveness requires.
    pub fn request_poll(&self) {
        let p = self.poll_flag.load(Ordering::Acquire);
        if !p.is_null() {
            unsafe { (*(p as *const AtomicU8)).store(1, Ordering::Release) };
        }
    }

    /// Lower this thread's JIT poll flag. Called by the coordinator after
    /// the thread has been resumed, so a stale `1` doesn't make the next
    /// poll re-park. (The JIT slow path also clears it; this is the
    /// belt-and-braces clear for threads that parked some other way.)
    pub fn clear_poll(&self) {
        let p = self.poll_flag.load(Ordering::Acquire);
        if !p.is_null() {
            unsafe { (*(p as *const AtomicU8)).store(0, Ordering::Release) };
        }
    }

    /// Set the JIT frame pointer that should be exposed to GC root
    /// scans while this thread is parked at a safepoint. Cleared
    /// (with `clear_parked_jit_fp`) on resume.
    pub fn set_parked_jit_fp(&self, fp: *const u8) {
        self.parked_jit_fp.store(fp as *mut u8, Ordering::Release);
    }

    /// Clear the parked JIT FP pointer.
    pub fn clear_parked_jit_fp(&self) {
        self.parked_jit_fp
            .store(std::ptr::null_mut(), Ordering::Release);
    }

    /// Read the parked JIT frame pointer. Returns null if this
    /// thread isn't currently parked at a JIT safepoint.
    pub fn parked_jit_fp(&self) -> *const u8 {
        self.parked_jit_fp.load(Ordering::Acquire) as *const u8
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
    ///
    /// `STATE_BLOCKED` threads are also considered safely at safepoint:
    /// they cannot transition back to RUNNING without first observing
    /// that no GC is in progress, so it is sound to scan their roots.
    pub fn is_safely_at_safepoint(&self) -> bool {
        match self.state.load(Ordering::Acquire) {
            STATE_BLOCKED => true,
            STATE_AT_SAFEPOINT => {
                let gc_done = self.safepoint_lock.lock().unwrap();
                // Re-read the state under the lock: enter_safepoint
                // consumes gc_done and stores RUNNING while holding it,
                // so the pre-lock state load above can be stale — seeing
                // (AT_SAFEPOINT, gc_done=false) for a thread that has
                // already resumed mutating.
                !*gc_done && self.state.load(Ordering::Acquire) == STATE_AT_SAFEPOINT
            }
            _ => false,
        }
    }

    /// True if this thread is currently in the BLOCKED state.
    pub fn is_blocked(&self) -> bool {
        self.state.load(Ordering::Acquire) == STATE_BLOCKED
    }

    /// Current safepoint generation (incremented each enter_safepoint call).
    pub fn safepoint_gen(&self) -> u64 {
        self.safepoint_gen.load(Ordering::Acquire)
    }

    /// Enter a blocking region. The thread is treated as if it had
    /// reached a safepoint, so the GC can scan its roots and proceed
    /// without waiting for the thread to poll.
    ///
    /// MUST be paired with `exit_blocked`. Nesting is not supported —
    /// callers must be in `STATE_RUNNING` when calling.
    ///
    /// # Preconditions
    /// - All currently-live GC pointers must already be rooted in
    ///   `frame_chain`. The GC may scan immediately after this returns.
    /// - The caller must not mutate the heap or root set while in the
    ///   blocked region. Allocations and root-set changes are only
    ///   allowed after `exit_blocked` returns.
    ///
    /// # Safety
    /// Logically — not type-system — unsafe: violating the preconditions
    /// causes UB at the GC level (missed roots, dangling pointers).
    pub fn enter_blocked(&self) {
        self.safepoint_gen.fetch_add(1, Ordering::AcqRel);
        let prev = self.state.swap(STATE_BLOCKED, Ordering::AcqRel);
        debug_assert_eq!(
            prev, STATE_RUNNING,
            "enter_blocked: thread was not in STATE_RUNNING (was {prev})"
        );
    }

    /// Exit a blocking region. Waits for any in-progress GC to finish
    /// (so the thread does not start running with stale references after
    /// a moving collector compacted), then transitions back to RUNNING.
    ///
    /// Also clears any stale `gc_done` set by `resume()` calls that
    /// fired during the blocked window — without this, the next
    /// `enter_safepoint` would observe the residual flag and exit early
    /// without participating in the next collection.
    pub fn exit_blocked(&self, heap: &Heap) {
        loop {
            // Wait for any active STW window to fully close before
            // resuming. Concurrent GC's `barriers_active` is also a
            // window during which we shouldn't blindly transition: the
            // collector may still need our state stable.
            while heap.gc_requested() || heap.barriers_active() {
                std::thread::yield_now();
            }
            // The check above and the swap below must be atomic w.r.t.
            // a collector's safepoint census: a collector that raises
            // gc_requested between them samples us as BLOCKED ("safely
            // parked, scan in place") while we resume mutating — its
            // relocations then race our reads/writes and leave stale
            // from-space pointers behind (garbage type_id one or two
            // collections later). `transition_outside_gc` revalidates
            // under the threads lock, which every collector holds while
            // raising the flag.
            let transitioned = heap.transition_outside_gc(|| {
                // Drain any stale resume flag. resume() during our BLOCKED
                // window left gc_done = true; if we don't consume it, the
                // next enter_safepoint short-circuits and the next GC
                // busy-waits for us until we poll a fresh safepoint.
                {
                    let mut gc_done = self.safepoint_lock.lock().unwrap();
                    *gc_done = false;
                }
                let prev = self.state.swap(STATE_RUNNING, Ordering::AcqRel);
                debug_assert_eq!(
                    prev, STATE_BLOCKED,
                    "exit_blocked: thread was not in STATE_BLOCKED (was {prev})"
                );
            });
            if transitioned {
                return;
            }
        }
    }

    /// Permanently park this state on behalf of an owner thread that has
    /// handed its runtime off to another OS thread and will never mutate
    /// the heap through it again.
    ///
    /// Without this, a handed-off runtime's home `ThreadState` stays
    /// `STATE_RUNNING` forever while its OS thread runs unrelated code —
    /// it never polls this heap's safepoints, so every stop-the-world
    /// collection spins in `is_safely_at_safepoint` waiting for it
    /// (livelock). Parking it as `STATE_BLOCKED` lets the GC scan its
    /// (empty) frame chain in place, exactly like a thread blocked in a
    /// syscall.
    ///
    /// CAS `RUNNING → BLOCKED`; idempotent (no-op unless currently
    /// RUNNING, so concurrent/repeated calls and an in-progress safepoint
    /// are all safe). May be called from any thread.
    ///
    /// # Preconditions (logical, like `enter_blocked`)
    /// The owning thread must have no live un-rooted heap pointers and
    /// must never run mutator code on this state again. Violating this
    /// is a GC-level soundness bug.
    pub fn park_handed_off(&self) {
        if self
            .state
            .compare_exchange(
                STATE_RUNNING,
                STATE_BLOCKED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.safepoint_gen.fetch_add(1, Ordering::AcqRel);
        }
    }
}

/// RAII scratch-stack region. Push heap pointers with [`push`](Self::push)
/// to root them across a subsequent allocation; re-read the relocated
/// pointer with [`get`](Self::get) after each alloc. The region pops back to
/// its opening depth on drop, so every exit path (including `?` errors)
/// cleans up.
pub struct ScratchScope<'a> {
    ts: &'a ThreadState,
    mark: usize,
}

impl<'a> ScratchScope<'a> {
    /// Push a heap pointer; returns its slot index for later [`get`](Self::get).
    pub fn push(&self, ptr: *const u8) -> usize {
        self.ts.push_scratch(ptr)
    }

    /// Read the (possibly relocated) pointer at slot index `i`.
    pub fn get(&self, i: usize) -> *mut u8 {
        self.ts.scratch_at(i)
    }
}

impl Drop for ScratchScope<'_> {
    fn drop(&mut self) {
        self.ts.scratch_reset(self.mark);
    }
}

impl RootSource for ThreadState {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        self.frame_chain.scan_roots(visitor);
        // Scratch roots: heap pointers a runtime fn parked here across an
        // allocation. Only the live prefix `[0..depth)` is scanned; slots
        // above `depth` hold stale pointers a popped frame left behind.
        let depth = self.scratch_depth.load(Ordering::Acquire);
        for s in &self.gc_scratch[..depth] {
            visitor(s.as_ptr() as *mut u64);
        }
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

    /// Acquire a `Mutex` while keeping the GC unblocked.
    ///
    /// Naively calling `mutex.lock()` parks the OS thread without
    /// transitioning to a safepoint state. If another thread triggers
    /// GC while this one is parked, the GC busy-waits forever for our
    /// thread to reach a safepoint — deadlock.
    ///
    /// This wrapper transitions to `STATE_BLOCKED` before locking
    /// (telling the GC it can scan our roots and proceed), then
    /// transitions back via `exit_blocked` after the lock is acquired
    /// (which waits for any in-flight GC to finish before resuming).
    ///
    /// # Preconditions
    /// - All currently-live GC pointers must be rooted in `frame_chain`
    ///   before calling. The GC may scan immediately after we park.
    /// - The caller must currently be in `STATE_RUNNING` (no nested
    ///   `lock_safe` calls inside another blocking region).
    ///
    /// After this returns the thread is RUNNING again and the caller
    /// should poll `safepoint()` reasonably soon — a fresh GC may have
    /// started while we were exiting the blocked window.
    pub fn lock_safe<'a, T>(&self, mutex: &'a Mutex<T>) -> MutexGuard<'a, T> {
        self.state.enter_blocked();
        let guard = match mutex.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                // Even on poison we must restore state, otherwise the
                // thread leaks the BLOCKED state and the GC will treat
                // it as safepoint-equivalent forever.
                self.state.exit_blocked(&self.heap);
                panic!("lock_safe: mutex poisoned: {poisoned}");
            }
        };
        self.state.exit_blocked(&self.heap);
        guard
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
    pub fn alloc(&self, info: &TypeInfo, varlen_len: usize) -> *mut u8 {
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
    pub fn alloc_obj<H: ObjHeader>(&self, info: &TypeInfo, varlen_len: usize) -> *mut u8 {
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
    fn alloc_slow_path(&self, info: &TypeInfo, varlen_len: usize) -> *mut u8 {
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
        info: &TypeInfo,
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
