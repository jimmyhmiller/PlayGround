//! Runtime support for compiled gc-rust code.
//!
//! This is the ABI boundary between LLVM-compiled gc-rust code and the
//! [`crate::gc`] collector. It is a deliberately *trimmed* descendant of
//! ai-lang's runtime: we keep only the pieces the GC protocol requires and
//! drop everything ai-lang-specific (content-addressed code tables, the
//! uniform-boxed-Int machinery, atom/prim-array shapes, closure code-hash
//! dispatch). gc-rust monomorphizes, so there is no uniform-boxed
//! representation to support here.
//!
//! ## What compiled code sees
//!
//! - [`Thread`] — passed as the first parameter of every compiled function.
//!   Holds the safepoint `state` byte, the head of the per-thread shadow-stack
//!   chain ([`Thread::top_frame`]), a pointer to the GC [`Heap`], and a pointer
//!   to the heap's inline-allocation [`AllocWindow`]. Compiled IR reads these
//!   at the fixed byte offsets in [`thread_offsets`].
//! - [`Frame`] / [`FrameOrigin`] — per-call shadow-stack frames. Compiled code
//!   alloca's a `{ Frame, [*mut u8; N] }` on the native stack, links it into
//!   the chain on entry, zeroes its root slots, and unlinks on return. The GC
//!   walks the chain to find live heap pointers — this is what makes our roots
//!   *precise* (no stack scanning, no false retention).
//!
//! ## How the GC finds our roots
//!
//! The collector can run only at a safepoint. Before a mutator does anything
//! that can trigger a collection (any `ai_gc_alloc_*`, any poll trap), it
//! publishes its current [`Thread::top_frame`] into the owning
//! [`ThreadState::set_parked_jit_fp`]. At collection time the GC reads each
//! parked thread's fp and invokes [`walk_gc_frames`] (registered via
//! [`Heap::set_jit_frame_walker`]) to scan the `Frame` chain. The
//! `state`/`top_frame`/`heap`/`alloc_window` layout is load-bearing ABI; the
//! `const _: ()` asserts below fail the build if it drifts.

use crate::gc::{AllocWindow, Full, Heap, IdentityPtrPolicy, ThreadState, TypeInfo};
use std::sync::Arc;
use std::cell::Cell;

thread_local! {
    /// THIS thread's current mutator `Thread*`. Set when a thread's gc-rust entry
    /// starts (the main thread at program entry; a spawned thread in
    /// `ai_thread_spawn`). Read by FFI **callback trampolines**, which are plain C
    /// functions invoked by foreign code and therefore receive no `Thread*`
    /// argument — they recover it from here. Thread-local so each mutator gets its
    /// own (crucial once multiple threads run). See `docs/ffi.md`, `docs/threads.md`.
    static CURRENT_THREAD: Cell<*mut Thread> = const { Cell::new(core::ptr::null_mut()) };
}

/// Publish `t` as THIS thread's current mutator thread (called at each thread's
/// gc-rust entry: the AOT/JIT driver for main, `ai_thread_spawn` for children).
pub fn set_current_thread(t: *mut Thread) {
    CURRENT_THREAD.with(|c| c.set(t));
}

/// This thread's current mutator `Thread*`. Used by callback trampolines. Returns
/// null if no gc-rust entry is on this thread's stack.
///
/// # Safety
/// The returned pointer is valid only while a gc-rust entry is on this thread's
/// stack.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_current_thread() -> *mut Thread {
    CURRENT_THREAD.with(|c| c.get())
}

// =============================================================================
// Thread
// =============================================================================

/// Per-thread state visible to compiled code. Passed as the first argument of
/// every compiled gc-rust function.
///
/// Accessed from LLVM IR by fixed byte offset — see [`thread_offsets`]. The
/// asserts below enforce the ABI.
#[repr(C)]
pub struct Thread {
    /// Safepoint flag. `0` = running normally. Non-zero means the GC has
    /// requested a safepoint; the next safepoint poll traps into
    /// [`ai_gc_pollcheck_slow`].
    pub state: u8,
    _pad: [u8; 7],

    /// Head of this thread's shadow-stack chain. Each compiled function
    /// alloca's its own [`Frame`] on entry, links it here, and unlinks on
    /// return. The GC walks this to find live roots.
    pub top_frame: *mut Frame,

    /// Pointer to the GC heap. Compiled code calls `ai_gc_alloc_*`, which
    /// dereference this to allocate.
    pub heap: *mut Heap,

    /// Pointer to the [`ThreadState`] in the `gc` module that this `Thread`
    /// shadows. Used to publish/clear `parked_jit_fp` around allocations so the
    /// GC walks our chain, and to enter the safepoint on a poll trap.
    pub dyna_thread: *const ThreadState,

    /// The heap's inline-allocation window (`cursor` / `base` / `limit` of the
    /// active from-space, re-pointed at flips under stop-the-world). The
    /// compiled inline fast path reads it; `limit == 0` (stress mode) closes it
    /// so every allocation takes the out-of-line slow path.
    pub alloc_window: *const AllocWindow,
}

pub mod thread_offsets {
    //! Byte offsets within [`super::Thread`]. Mirrored in LLVM codegen.
    pub const STATE: usize = 0;
    pub const TOP_FRAME: usize = 8;
    pub const HEAP: usize = 16;
    pub const DYNA_THREAD: usize = 24;
    pub const ALLOC_WINDOW: usize = 32;
}

const _: () = {
    assert!(core::mem::offset_of!(Thread, state) == thread_offsets::STATE);
    assert!(core::mem::offset_of!(Thread, top_frame) == thread_offsets::TOP_FRAME);
    assert!(core::mem::offset_of!(Thread, heap) == thread_offsets::HEAP);
    assert!(core::mem::offset_of!(Thread, dyna_thread) == thread_offsets::DYNA_THREAD);
    assert!(core::mem::offset_of!(Thread, alloc_window) == thread_offsets::ALLOC_WINDOW);
};

// =============================================================================
// Frame + FrameOrigin
// =============================================================================

/// Per-call shadow-stack frame header.
///
/// Compiled code allocates a `{ Frame, [*mut u8; N] }` on the native stack
/// (N = the function's GC-typed local/temporary count), links it into the
/// thread chain on entry, and unlinks on return.
///
/// Layout (load-bearing — matches LLVM IR `{ ptr, ptr, [N x ptr] }`):
///
/// ```text
/// offset 0  : parent  : *mut Frame
/// offset 8  : origin  : *const FrameOrigin
/// offset 16 : roots   : [*mut u8; N]   (variable, walked via origin.num_roots)
/// ```
#[repr(C)]
pub struct Frame {
    pub parent: *mut Frame,
    pub origin: *const FrameOrigin,
    // followed by [*mut u8; num_roots] starting at offset 16
}

pub mod frame_offsets {
    pub const PARENT: usize = 0;
    pub const ORIGIN: usize = 8;
    /// First root slot. `num_roots` slots of 8 bytes each follow.
    pub const ROOTS: usize = 16;
}

const _: () = {
    assert!(core::mem::offset_of!(Frame, parent) == frame_offsets::PARENT);
    assert!(core::mem::offset_of!(Frame, origin) == frame_offsets::ORIGIN);
    // The trailing root array is outside the struct (variable-length), so the
    // declared size is exactly the header.
    assert!(core::mem::size_of::<Frame>() == frame_offsets::ROOTS);
};

/// Static per-function descriptor. One emitted per compiled function as a
/// private constant global; each frame's `origin` field points at it. The GC
/// reads `num_roots` to know how many root slots follow the frame header, then
/// `num_indirect` *indirect* slots after them.
///
/// A **direct** root slot holds a GC pointer the collector reads and updates in
/// place. An **indirect** root slot holds the *address* of a GC pointer that
/// lives elsewhere on the stack — namely a reference embedded in a flattened
/// `#[value]` local's alloca (the alloca's interior). The collector dereferences
/// the indirect slot once to reach the real pointer and updates it in place.
/// This is how value-with-ref locals are kept precise across collections without
/// moving the value out of its alloca.
#[repr(C)]
pub struct FrameOrigin {
    pub num_roots: u32,
    pub num_indirect: u32,
    /// Function name, NUL-terminated, for debugging / GC tracing. May be null.
    pub name: *const u8,
}

impl FrameOrigin {
    pub const fn new(num_roots: u32, name: *const u8) -> Self {
        FrameOrigin {
            num_roots,
            num_indirect: 0,
            name,
        }
    }
}

// =============================================================================
// GC frame walker
// =============================================================================

/// Walk one parked thread's shadow-stack chain, visiting every live root slot.
///
/// Registered with the heap via [`Heap::set_jit_frame_walker`]. `jit_fp` is the
/// value the mutator published into its `ThreadState::parked_jit_fp` — i.e. the
/// top [`Frame`] of its chain. We follow `parent` links to the root of the
/// chain, and for each frame scan `origin.num_roots` pointer slots.
///
/// # Safety
/// `jit_fp` must be a valid `*const Frame` (or null) and the chain it heads must
/// be stable for the duration of the call. The safepoint protocol guarantees
/// this: the owning mutator is parked with its chain published and is not
/// mutating it.
pub unsafe fn walk_gc_frames(jit_fp: *const u8, visitor: &mut dyn FnMut(*mut u64)) {
    let mut frame = jit_fp as *const Frame;
    while !frame.is_null() {
        unsafe {
            let origin = (*frame).origin;
            let (num_roots, num_indirect) = if origin.is_null() {
                (0, 0)
            } else {
                ((*origin).num_roots as usize, (*origin).num_indirect as usize)
            };
            // Direct root slots start at byte offset ROOTS past the frame header.
            let slots = (frame as *const u8).add(frame_offsets::ROOTS) as *mut u64;
            for i in 0..num_roots {
                visitor(slots.add(i));
            }
            // Indirect slots follow the direct ones: each holds the address of a
            // GC pointer inside a value-with-ref local's alloca. Deref once, then
            // visit that pointer's slot so the collector updates it in place.
            for j in 0..num_indirect {
                let cell = slots.add(num_roots + j);
                let target = (*cell) as *mut u64;
                if !target.is_null() {
                    visitor(target);
                }
            }
            frame = (*frame).parent;
        }
    }
}

// =============================================================================
// RuntimeContext — owns the heap + the main mutator Thread
// =============================================================================

/// Owns a [`Heap`] and one registered mutator [`Thread`], wired together so
/// compiled code can run against it: the GC frame walker is installed, the
/// thread's safepoint poll flag points at the `Thread::state` byte, and the
/// `alloc_window` is pointed at the heap's inline-allocation window.
///
/// The `Heap` always uses the [`Full`] (16-byte) header — it carries the GC
/// word the copying collector parks forwarding pointers in — and the
/// [`IdentityPtrPolicy`] (root/field slot bits are raw pointers; `0` is the
/// null non-pointer). Codegen targets exactly this configuration.
pub struct RuntimeContext {
    /// Boxed so its address is stable: compiled code holds a `*mut Thread`.
    thread: Box<Thread>,
    heap: Arc<Heap>,
    dyna: Arc<ThreadState>,
}

impl RuntimeContext {
    /// Build a runtime over a single-generation semi-space heap of `space_size`
    /// bytes per space, with `type_table` describing every heap shape compiled
    /// code can allocate (`type_id` is the index into this table).
    pub fn new(space_size: usize, type_table: Vec<TypeInfo>) -> Self {
        Self::from_heap(Heap::new::<Full>(space_size, type_table))
    }

    /// Build a runtime over a GENERATIONAL heap: a `nursery_size`-byte young
    /// generation plus a `tenured_size`-byte (per-space) old generation.
    /// New allocations land in the nursery; a cheap minor GC scavenges it and
    /// promotes survivors to tenured; a major GC collects tenured when it fills.
    pub fn new_generational(nursery_size: usize, tenured_size: usize, type_table: Vec<TypeInfo>) -> Self {
        Self::from_heap(Heap::new_generational::<Full>(nursery_size, tenured_size, type_table))
    }

    /// Shared wiring: register the mutator thread, install the JIT frame walker,
    /// and point the safepoint poll flag. Used by both `new` and
    /// `new_generational`.
    fn from_heap(heap: Heap) -> Self {
        let heap = Arc::new(heap);
        // The GC walks our shadow-stack chain through this walker whenever a
        // parked thread has published its top frame.
        heap.set_jit_frame_walker(walk_gc_frames);

        let (dyna, _id) = heap.register_thread();
        let alloc_window = heap.alloc_window_ptr();

        let mut ctx = RuntimeContext {
            thread: Box::new(Thread {
                state: 0,
                _pad: [0; 7],
                top_frame: core::ptr::null_mut(),
                heap: Arc::as_ptr(&heap) as *mut Heap,
                dyna_thread: Arc::as_ptr(&dyna),
                alloc_window,
            }),
            heap,
            dyna,
        };

        // Point the safepoint poll flag at the live Thread::state byte so a GC
        // requested from another thread flips the byte this mutator polls.
        ctx.dyna
            .set_poll_flag(&mut ctx.thread.state as *mut u8);
        ctx
    }

    /// Raw pointer to the mutator `Thread`, passed as the first argument of
    /// every compiled gc-rust function.
    pub fn thread_ptr(&mut self) -> *mut Thread {
        &mut *self.thread as *mut Thread
    }

    pub fn heap(&self) -> &Arc<Heap> {
        &self.heap
    }

    /// Force a collection driven *by this mutator* — the model real compiled
    /// code uses when it hits allocation exhaustion. The calling thread becomes
    /// the GC thread (it parks every *other* registered mutator, excludes
    /// itself, and scans every parked thread's published frame chain). The
    /// caller MUST have published its own top frame into `parked_jit_fp` first
    /// (so its roots are scanned), via `thread`.
    ///
    /// # Safety
    /// `thread` must be this context's live mutator thread, with its current
    /// frame chain published. All live roots must already be in frame slots.
    pub unsafe fn force_collect(&self, thread: &Thread) {
        unsafe {
            let dyna = &*thread.dyna_thread;
            dyna.set_parked_jit_fp(thread.top_frame as *const u8);
            self.heap.mutator_triggered_gc::<IdentityPtrPolicy>(dyna);
            dyna.clear_parked_jit_fp();
        }
    }
}

impl Drop for RuntimeContext {
    fn drop(&mut self) {
        self.heap.safe_deregister_thread(&self.dyna);
    }
}

// =============================================================================
// Threads: spawn / join
// =============================================================================

/// The ABI of a compiled no-arg gc-rust closure body: `(thread, env) -> u64`.
/// The u64 is the closure's result (a scalar, or a GC pointer bit-pattern).
type ClosureFn = unsafe extern "C" fn(*mut Thread, *mut u8) -> u64;

/// A spawned thread's join handle, owned on the Rust side and handed to compiled
/// code as an opaque `*mut`. Keeps the heap `Arc` alive for the child's lifetime.
pub struct JoinHandle {
    handle: Option<std::thread::JoinHandle<u64>>,
}

/// Spawn a new OS thread that runs gc-rust closure code against the SAME heap as
/// `parent`. Codegen passes the closure's `env` object (a GC pointer) and its
/// `code` pointer (read from the env by the caller, which knows the layout). The
/// child registers its own mutator `Thread` (so stop-the-world can pause it and
/// the collector scans its roots), publishes its thread-local current thread,
/// invokes `code(child_thread, env)`, captures the `u64` result, and deregisters.
/// Returns an opaque `*mut JoinHandle`.
///
/// # Safety
/// `parent` must be the caller's live mutator thread; `env` a valid closure env
/// object; `code` its body with ABI `(thread, env) -> u64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_thread_spawn(parent: *mut Thread, env: *mut u8, code: *const u8) -> *mut JoinHandle {
    let heap: Arc<Heap> = unsafe {
        // An owning Arc handle to the shared heap (bump strong count; don't drop
        // the parent's reference).
        let p = (*parent).heap as *const Heap;
        Arc::increment_strong_count(p);
        Arc::from_raw(p)
    };
    // Hand off the env object via a GLOBAL ROOT slot. This runs on the PARENT
    // thread while `env` is still live in the parent's frame, so registering it
    // as a global root keeps it alive and relocated across any GC during the
    // handoff — closing the race where the child reads a stale (moved) pointer.
    // The child reads the (relocated) env via `get`, then clears the slot.
    let root_idx = heap.globals.add(env as u64);
    let code_bits = code as usize;

    let join = std::thread::spawn(move || -> u64 {
        // Register this child as a mutator thread on the shared heap.
        let (dyna, _id) = heap.register_thread();
        let alloc_window = heap.alloc_window_ptr();
        let mut thread = Box::new(Thread {
            state: 0,
            _pad: [0; 7],
            top_frame: core::ptr::null_mut(),
            heap: Arc::as_ptr(&heap) as *mut Heap,
            dyna_thread: Arc::as_ptr(&dyna),
            alloc_window,
        });
        dyna.set_poll_flag(&mut thread.state as *mut u8);
        let tptr = &mut *thread as *mut Thread;
        set_current_thread(tptr);

        // Read the (possibly relocated) env from the global root, then move it
        // into our own scratch root so it survives the call's allocations, and
        // release the global slot.
        let mark = dyna.scratch_mark();
        let slot = dyna.push_scratch(heap.globals.get(root_idx) as *const u8);
        heap.globals.set(root_idx, 0); // un-root the global handoff slot
        let env = dyna.scratch_at(slot);

        let f: ClosureFn = unsafe { core::mem::transmute(code_bits) };
        let result = unsafe { f(tptr, env) };

        dyna.scratch_reset(mark);
        set_current_thread(core::ptr::null_mut());
        // ORDERING INVARIANT (load-bearing): deregister BEFORE `thread` (the
        // Box) drops at end of closure. The poll flag points into `thread.state`
        // (this Box), NOT into the `Arc<ThreadState>` — so a GC snapshot's Arc
        // clone does NOT keep the poll-flag storage alive. `safe_deregister_thread`
        // removes us from the registry atomically w.r.t. the GC census, so after
        // it returns no collector can `request_poll` into the about-to-free Box.
        // Reordering (dropping `thread` first) would be a use-after-free.
        heap.safe_deregister_thread(&dyna);
        result
    });

    Box::into_raw(Box::new(JoinHandle { handle: Some(join) }))
}

/// Run `body` while this mutator is transitioned to `STATE_BLOCKED`, so a
/// stop-the-world collection on another thread can proceed *without* this thread
/// reaching a safepoint (a blocked thread counts as already-parked, and is
/// scanned in place). Without this, any blocking call (join, sleep) that outlasts
/// another thread's GC trigger DEADLOCKS: the collector spins waiting for a
/// safepoint this thread — parked in libc — will never poll.
///
/// We PUBLISH the caller's current JIT frame chain (`top_frame`) before blocking,
/// so the collector scans the joining/sleeping thread's live GC roots (its
/// caller's spill slots — e.g. `main`'s `Vec` locals across `t.join()`). Compiled
/// code spills live roots into frame slots before any call, so `top_frame` is a
/// complete root set. Without this, a GC during the block would miss those roots
/// and the thread would read relocated-away pointers on resume.
///
/// # Safety
/// `thread` must be this thread's live mutator `Thread` with a valid heap + dyna.
unsafe fn blocking_region<R>(thread: *mut Thread, body: impl FnOnce() -> R) -> R {
    unsafe {
        if thread.is_null() {
            return body();
        }
        let t = &*thread;
        let dyna = &*t.dyna_thread;
        let heap = &*t.heap;
        // Publish our frame chain so the collector finds our roots while blocked.
        dyna.set_parked_jit_fp(t.top_frame as *const u8);
        dyna.enter_blocked();

        // Restore RUNNING + clear the published frame even if `body` unwinds —
        // otherwise a panic would leave this thread permanently BLOCKED and every
        // future GC would scan its now-invalid frame. (Mirrors `lock_safe`'s
        // poison handling.)
        struct Unblock<'a>(&'a ThreadState, &'a Heap);
        impl Drop for Unblock<'_> {
            fn drop(&mut self) {
                self.0.exit_blocked(self.1);
                self.0.clear_parked_jit_fp();
            }
        }
        let _guard = Unblock(dyna, heap);
        body()
    }
}

/// Join a spawned thread, returning its `u64` result. Consumes the handle.
/// Transitions the joining thread to BLOCKED so a collection triggered by the
/// child (or any other thread) can proceed while we wait.
///
/// # Safety
/// `thread` must be the caller's live mutator thread; `handle` a `*mut JoinHandle`
/// from `ai_thread_spawn`, not yet joined.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_thread_join(thread: *mut Thread, handle: *mut JoinHandle) -> u64 {
    unsafe {
        let mut h = Box::from_raw(handle);
        match h.handle.take() {
            Some(j) => blocking_region(thread, move || j.join().unwrap_or(0)),
            None => 0,
        }
    }
}

/// `Thread.sleep(ms)`. Transitions to BLOCKED so a sleeping mutator doesn't stall
/// every other thread's GC for the sleep duration.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_thread_sleep(thread: *mut Thread, millis: i64) {
    if millis > 0 {
        unsafe {
            blocking_region(thread, || {
                std::thread::sleep(std::time::Duration::from_millis(millis as u64));
            });
        }
    }
}

/// `Thread.yield_now()`.
#[unsafe(no_mangle)]
pub extern "C" fn ai_thread_yield(_thread: *mut Thread) {
    std::thread::yield_now();
}

// =============================================================================
// Channels — Sender<T> / Receiver<T> over a bounded, blocking, on-heap queue
// =============================================================================
//
// Design (mirrors Go's hchan / Java's ArrayBlockingQueue): the element BUFFER is
// an ON-HEAP object (a varlen pointer array) so the GC traces + relocates the
// queued values for free — nothing lives off-heap except the indices + sync. A
// Rust control block holds the mutex + two condvars + head/tail/count/closed.
// `recv`/`send` block by PARKING on a condvar (never spin), transitioning to
// BLOCKED so a stop-the-world GC can proceed. The buffer pointer is read fresh
// from the (frame-rooted) channel object on each access — never held across a
// wait — so relocation during a park can't dangle it. See `docs/threads.md`.

use std::sync::{Condvar, Mutex};

struct ChanState {
    head: usize,
    tail: usize,
    count: usize,
    cap: usize,
    closed: bool,
    senders: usize, // live Sender handles; channel auto-closes at 0
}

pub struct ChanCtrl {
    state: Mutex<ChanState>,
    not_empty: Condvar,
    not_full: Condvar,
}

/// `Channel::new(cap)` → a `RawPtr` to a fresh control block for a `cap`-slot
/// channel (the element buffer is allocated separately, on the heap).
#[unsafe(no_mangle)]
pub extern "C" fn ai_chan_new(_thread: *mut Thread, cap: i64) -> *mut ChanCtrl {
    let cap = cap.max(1) as usize;
    Box::into_raw(Box::new(ChanCtrl {
        state: Mutex::new(ChanState { head: 0, tail: 0, count: 0, cap, closed: false, senders: 1 }),
        not_empty: Condvar::new(),
        not_full: Condvar::new(),
    }))
}

/// Register a cloned Sender (so the channel doesn't auto-close while it lives).
///
/// # Safety
/// `ctrl` must be a live channel control block.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_chan_sender_clone(_thread: *mut Thread, ctrl: *const ChanCtrl) {
    unsafe { (*ctrl).state.lock().unwrap().senders += 1; }
}

/// Drop a Sender; when the last one goes, close the channel and wake receivers.
///
/// # Safety
/// As [`ai_chan_sender_clone`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_chan_sender_drop(_thread: *mut Thread, ctrl: *const ChanCtrl) {
    unsafe {
        let c = &*ctrl;
        let mut st = c.state.lock().unwrap();
        st.senders -= 1;
        if st.senders == 0 {
            st.closed = true;
            c.not_empty.notify_all();
        }
    }
}

/// `sender.send(buf, value)`: store the GC pointer `value` into the heap buffer
/// `buf` at the tail slot, blocking while the channel is full. Yields i64 0.
///
/// # Safety
/// `thread` valid; `buf` a live varlen-pointer-array heap object with at least
/// `cap` slots; `ctrl` a live control block; `value` a GC pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_chan_send(thread: *mut Thread, buf: *mut u8, ctrl: *const ChanCtrl, value: *mut u8) -> i64 {
    unsafe {
        let c = &*ctrl;
        let t = &*thread;
        let dyna = &*t.dyna_thread;
        let heap = &*t.heap;
        // Root `value` AND `buf` in scratch so a GC during the park relocates
        // them (they're held as raw Rust args, not in the published frame). We
        // re-read both from scratch after waking.
        let mark = dyna.scratch_mark();
        let vslot = dyna.push_scratch(value as *const u8);
        let bslot = dyna.push_scratch(buf as *const u8);
        let mut st = c.state.lock().unwrap();
        // Block while full. Publish our frame + go BLOCKED so a GC can run while
        // we park.
        while st.count == st.cap && !st.closed {
            dyna.set_parked_jit_fp(t.top_frame as *const u8);
            dyna.enter_blocked();
            st = c.not_full.wait(st).unwrap();
            dyna.exit_blocked(heap);
            dyna.clear_parked_jit_fp();
        }
        if st.closed { dyna.scratch_reset(mark); return 0; }
        // Re-read the (possibly relocated) buffer + value from scratch.
        let buf = dyna.scratch_at(bslot);
        let value = dyna.scratch_at(vslot);
        let slot = buf.add(STR_DATA_OFF + st.tail * 8) as *mut *mut u8;
        *slot = value;
        // Generational write barrier: buffer (maybe old) ← value (maybe young).
        if heap.has_nursery() && !value.is_null() && heap.is_nursery(value) && heap.is_tenured(buf) {
            heap.mark_card_dirty(buf);
        }
        dyna.scratch_reset(mark);
        st.tail = (st.tail + 1) % st.cap;
        st.count += 1;
        c.not_empty.notify_one();
        0
    }
}

/// `receiver.recv(buf)`: pop the head element from the heap buffer `buf`,
/// blocking while empty. Returns the GC pointer (or null when the channel is
/// closed and drained — the language wraps this as `None`).
///
/// # Safety
/// As [`ai_chan_send`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_chan_recv(thread: *mut Thread, buf: *mut u8, ctrl: *const ChanCtrl) -> *mut u8 {
    unsafe {
        let c = &*ctrl;
        let t = &*thread;
        let dyna = &*t.dyna_thread;
        let heap = &*t.heap;
        let mark = dyna.scratch_mark();
        let bslot = dyna.push_scratch(buf as *const u8);
        let mut st = c.state.lock().unwrap();
        while st.count == 0 && !st.closed {
            dyna.set_parked_jit_fp(t.top_frame as *const u8);
            dyna.enter_blocked();
            st = c.not_empty.wait(st).unwrap();
            dyna.exit_blocked(heap);
            dyna.clear_parked_jit_fp();
        }
        if st.count == 0 {
            dyna.scratch_reset(mark);
            return core::ptr::null_mut(); // closed + drained → None
        }
        let buf = dyna.scratch_at(bslot);
        let slot = buf.add(STR_DATA_OFF + st.head * 8) as *mut *mut u8;
        let v = *slot;
        *slot = core::ptr::null_mut(); // clear so the GC can reclaim it after recv
        st.head = (st.head + 1) % st.cap;
        st.count -= 1;
        c.not_full.notify_one();
        dyna.scratch_reset(mark);
        v
    }
}

// =============================================================================
// AtomicI64 — a lock-free shared integer cell
// =============================================================================
//
// The cell lives OFF the GC heap (a leaked Box), so its address is stable: the
// relocating collector never moves it, and concurrent CAS/fetch-add operate on a
// fixed location. The gc-rust handle is a `RawPtr` to this `AtomicI64`. (Scalars
// aren't GC pointers, so the cell needs no root registration.)

use std::sync::atomic::{AtomicI64, Ordering};

/// `AtomicI64::new(v)` → a `RawPtr` to a fresh off-heap atomic cell.
#[unsafe(no_mangle)]
pub extern "C" fn ai_atomic_i64_new(_thread: *mut Thread, v: i64) -> *mut AtomicI64 {
    Box::into_raw(Box::new(AtomicI64::new(v)))
}

/// `a.load()` (SeqCst).
///
/// # Safety
/// `a` must be a live cell from `ai_atomic_i64_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_atomic_i64_load(_thread: *mut Thread, a: *const AtomicI64) -> i64 {
    unsafe { (*a).load(Ordering::SeqCst) }
}

/// `a.store(v)` (SeqCst). Yields 0.
///
/// # Safety
/// As [`ai_atomic_i64_load`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_atomic_i64_store(_thread: *mut Thread, a: *const AtomicI64, v: i64) -> i64 {
    unsafe { (*a).store(v, Ordering::SeqCst) };
    0
}

/// `a.fetch_add(delta)` → the PREVIOUS value (SeqCst).
///
/// # Safety
/// As [`ai_atomic_i64_load`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_atomic_i64_fetch_add(_thread: *mut Thread, a: *const AtomicI64, delta: i64) -> i64 {
    unsafe { (*a).fetch_add(delta, Ordering::SeqCst) }
}

/// `a.compare_and_set(expected, new)` → 1 if the swap happened, else 0 (SeqCst).
///
/// # Safety
/// As [`ai_atomic_i64_load`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_atomic_i64_compare_and_set(
    _thread: *mut Thread,
    a: *const AtomicI64,
    expected: i64,
    new: i64,
) -> i64 {
    unsafe {
        match (*a).compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => 1,
            Err(_) => 0,
        }
    }
}

/// A stable per-thread id (`Thread.current_id()`), derived from the OS thread.
#[unsafe(no_mangle)]
pub extern "C" fn ai_thread_current_id(_thread: *mut Thread) -> i64 {
    // ThreadId isn't directly an integer; hash it to a stable-per-thread value.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    std::thread::current().id().hash(&mut h);
    (h.finish() & 0x7fff_ffff_ffff_ffff) as i64
}

// =============================================================================
// AOT entry point
// =============================================================================

/// One heap-shape descriptor as emitted into an AOT object file by codegen.
///
/// This is the *source* data for a `gc::TypeInfo` (the same fields
/// `codegen::layouts_to_type_infos` derives a `TypeInfo` from), serialized as a
/// fixed `#[repr(C)]` record so an AOT-compiled binary can hand its per-program
/// layout table to the runtime at startup. We deliberately do NOT serialize a
/// `gc::TypeInfo` directly: that type is not `#[repr(C)]`, so its in-memory
/// field order is unspecified. Instead the runtime rebuilds each `TypeInfo`
/// here with the exact same logic the JIT path uses, guaranteeing the GC
/// scanner sees identical shapes regardless of compilation mode.
///
/// `varlen`: 0 = None, 1 = Values, 2 = Bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AotLayout {
    pub ptr_fields: u16,
    pub raw_bytes: u16,
    pub varlen: u8,
    pub _pad: [u8; 3],
}

/// AOT program entry, called from the native `main` emitted into the object
/// file. Builds a [`RuntimeContext`] over the program's layout table (passed as
/// a `[AotLayout; ti_count]` blob in the binary), then invokes the compiled
/// program entry (`gcrust_entry`) with a live `Thread*` and returns its `i64`.
///
/// The `RuntimeContext` setup is identical to the JIT driver's: the GC frame
/// walker is installed, the safepoint poll flag is wired to `Thread::state`, and
/// the alloc window is pointed at the heap. This MUST match the JIT path or the
/// GC ABI breaks.
///
/// `meta`/`meta_len` carry the reflection metadata blob (encoded by
/// `gc::reflect::encode`); it is decoded into the heap's `TypeMeta` table so
/// heap-exploration tooling and in-language reflection can recover type/field
/// names. An empty blob (`meta_len == 0`) simply installs no metadata.
///
/// # Safety
/// `layouts` must point at `ti_count` valid `AotLayout` records, `meta` at
/// `meta_len` bytes produced by `gc::reflect::encode`, and `entry` must be the
/// compiled program entry with signature `extern "C" fn(*mut Thread) -> i64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gcr_runtime_main(
    layouts: *const AotLayout,
    ti_count: usize,
    meta: *const u8,
    meta_len: usize,
    entry: extern "C" fn(*mut Thread) -> i64,
) -> i64 {
    use crate::gc::{Full, ObjHeader, TypeInfo};
    let slice = if ti_count == 0 {
        &[][..]
    } else {
        assert!(!layouts.is_null(), "gcr_runtime_main: null layout table");
        unsafe { std::slice::from_raw_parts(layouts, ti_count) }
    };
    // Decode the metadata blob FIRST: besides the reflection tables it carries
    // the per-type interior-pointer offsets, which must be baked into the
    // `TypeInfo`s as we build the table (so the GC traces refs embedded in
    // flattened value fields). The slices are leaked — the type table lives for
    // the whole program, a bounded one-time leak, matching the JIT path.
    let (meta_types, meta_values, interior) = if meta_len > 0 {
        assert!(!meta.is_null(), "gcr_runtime_main: null metadata blob");
        let bytes = unsafe { std::slice::from_raw_parts(meta, meta_len) };
        crate::gc::reflect::decode(bytes)
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };

    // Rebuild the type table exactly as `codegen::layouts_to_type_infos` does:
    // pointer fields first (traced), then raw bytes, then any varlen tail, plus
    // interior pointers for flattened value-with-ref fields.
    let type_table: Vec<TypeInfo> = slice
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let mut ti = TypeInfo::for_header(Full::SIZE)
                .with_type_id(i as u16)
                .with_fields(l.ptr_fields)
                .with_raw_bytes(l.raw_bytes);
            ti = match l.varlen {
                0 => ti,
                1 => ti.with_varlen_values(l.ptr_fields),
                2 => ti.with_varlen_bytes(l.ptr_fields),
                other => panic!("gcr_runtime_main: invalid varlen kind {}", other),
            };
            if let Some(offs) = interior.get(i) {
                if !offs.is_empty() {
                    let leaked: &'static [u16] = Box::leak(offs.clone().into_boxed_slice());
                    ti = ti.with_interior_ptrs(leaked);
                }
            }
            ti
        })
        .collect();

    // Generational heap: a small nursery (collected cheaply and often) over a
    // large tenured generation. Most objects die young, so minor GCs reclaim the
    // bulk of garbage without touching tenured space.
    //
    // The nursery is intentionally SMALL (1 MB) so real minor collections happen
    // during ordinary workloads — including the self-compile — rather than being
    // sized past the workload to avoid collecting. The old 256 MB nursery was an
    // AVOIDANCE of a (since-disproven) "int-or-pointer mis-move" concern: on this
    // moving path the monomorphizing front end is precise (every scalar lives in
    // the untraced raw region; unused enum pointer-slots are zeroed), so a traced
    // slot only ever holds a pointer-or-null and an int can never be mis-moved.
    // See docs/FUTURE_WORK.md (P0 int/pointer entry) — that residual unsoundness
    // is DORMANT here and only becomes live if the self-hosted toolchain ever
    // links this moving GC instead of the non-moving gcr_rt.c (the M-RT split).
    // The collector arms a precise-layout detector under debug / --gc-stress and
    // under GCR_GC_VERIFY=1 (see gc::heap::gc_verify_armed).
    let nursery = 1 << 20;     // 1 MB young generation — collect for real
    let tenured = 256 << 20;  // 256 MB old generation (per space)
    let mut rt = RuntimeContext::new_generational(nursery, tenured, type_table);
    // Install the reflection metadata decoded above (type/field names + types)
    // so heap-exploration tooling and in-language reflection have nominal info.
    if !meta_types.is_empty() || !meta_values.is_empty() {
        rt.heap().set_type_meta(meta_types);
        rt.heap().set_value_meta(meta_values);
    }
    let thread = rt.thread_ptr();
    // Publish the current thread so FFI callback trampolines can recover it.
    set_current_thread(thread);
    let result = entry(thread);
    // Opt-in heap dump: render the live object graph with reflection metadata,
    // after the program returns (heap quiescent). `GCR_HEAP_DUMP=json` emits a
    // structured snapshot for tooling; any other non-empty value emits text.
    if let Some(mode) = std::env::var_os("GCR_HEAP_DUMP") {
        if mode == "json" {
            eprint!("{}", unsafe { crate::gc::dump::dump_heap_json(rt.heap()) });
        } else {
            eprint!("{}", unsafe { crate::gc::dump::dump_heap_text(rt.heap()) });
        }
    }
    result
}

// =============================================================================
// In-language structural reflection (field iteration)
// =============================================================================

/// The reflection fields *active* for `obj` right now: a struct's fields, or the
/// payload fields of an enum's currently-tagged variant. Empty for opaque
/// builtins or when no metadata is installed. Borrows the heap's (immortal)
/// metadata table.
fn active_fields<'h>(heap: &'h Heap, obj: *const u8) -> &'h [crate::gc::FieldMeta] {
    use crate::gc::TypeKind;
    let tid = unsafe { heap.obj_type_id(obj) };
    match heap.type_meta_by_id(tid) {
        Some(m) => match &m.kind {
            TypeKind::Struct { fields } => fields,
            TypeKind::Enum { tag_offset, variants } => {
                let tag = unsafe { (obj.add(*tag_offset as usize) as *const u32).read_unaligned() };
                variants
                    .iter()
                    .find(|v| v.tag == tag)
                    .map(|v| v.fields.as_slice())
                    .unwrap_or(&[])
            }
            TypeKind::Opaque => &[],
        },
        None => &[],
    }
}

/// `field_count(obj)` — number of reflectable fields of `obj` (a struct's fields,
/// or the active enum variant's payload). 0 for opaque/builtin objects.
///
/// # Safety
/// `thread` valid with a live heap; `obj` a valid heap object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_reflect_field_count(thread: *mut Thread, obj: *const u8) -> i64 {
    unsafe {
        let heap = &*(*thread).heap;
        active_fields(heap, obj).len() as i64
    }
}

/// Bounds-check a reflection field index, panicking with a clear message (an
/// out-of-range index is a caller bug — `field_count` is the guard).
fn field_at<'a>(fields: &'a [crate::gc::FieldMeta], idx: i64, op: &str) -> &'a crate::gc::FieldMeta {
    if idx < 0 || idx as usize >= fields.len() {
        panic!("reflect::{op}: field index {idx} out of range (0..{})", fields.len());
    }
    &fields[idx as usize]
}

/// `field_kind(obj, i)` — the kind of field `i`: 0=ref, 1=int, 2=float, 3=bool,
/// 4=char, 5=value-aggregate. Lets in-language code decide how to read it.
///
/// # Safety
/// As [`ai_reflect_field_count`]; `i` must be in `0..field_count`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_reflect_field_kind(thread: *mut Thread, obj: *const u8, i: i64) -> i64 {
    use crate::gc::{FieldTy, ScalarKind};
    unsafe {
        let heap = &*(*thread).heap;
        let f = field_at(active_fields(heap, obj), i, "field_kind");
        match f.ty {
            FieldTy::Ref(_) => 0,
            FieldTy::Scalar(ScalarKind::F32 | ScalarKind::F64) => 2,
            FieldTy::Scalar(ScalarKind::Bool) => 3,
            FieldTy::Scalar(ScalarKind::Char) => 4,
            FieldTy::Scalar(_) => 1,
            FieldTy::Value(_) => 5,
        }
    }
}

/// `field_i64(obj, i)` — field `i`'s value widened to i64: a scalar decoded by
/// its kind (sign-extended for signed ints, float bits for floats), or a ref's
/// pointer bits. Pairs with `field_kind` so the caller interprets it.
///
/// # Safety
/// As [`ai_reflect_field_count`]; `i` must be in `0..field_count`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_reflect_field_i64(thread: *mut Thread, obj: *const u8, i: i64) -> i64 {
    use crate::gc::{FieldTy, ScalarKind};
    unsafe {
        let heap = &*(*thread).heap;
        let f = field_at(active_fields(heap, obj), i, "field_i64");
        let at = obj.add(f.offset as usize);
        match f.ty {
            FieldTy::Ref(_) | FieldTy::Value(_) => (at as *const u64).read_unaligned() as i64,
            FieldTy::Scalar(k) => match k {
                ScalarKind::I8 => (at as *const i8).read_unaligned() as i64,
                ScalarKind::I16 => (at as *const i16).read_unaligned() as i64,
                ScalarKind::I32 => (at as *const i32).read_unaligned() as i64,
                ScalarKind::I64 => (at as *const i64).read_unaligned(),
                ScalarKind::U8 | ScalarKind::Bool => at.read_unaligned() as i64,
                ScalarKind::U16 => (at as *const u16).read_unaligned() as i64,
                ScalarKind::U32 | ScalarKind::Char => (at as *const u32).read_unaligned() as i64,
                ScalarKind::U64 | ScalarKind::Ptr => (at as *const u64).read_unaligned() as i64,
                ScalarKind::F32 => (at as *const f32).read_unaligned().to_bits() as i64,
                ScalarKind::F64 => (at as *const f64).read_unaligned().to_bits() as i64,
            },
        }
    }
}

/// `field_name(obj, i)` — a fresh `String` holding field `i`'s source name.
/// `str_type_id` is the `String` layout id used to allocate the result.
///
/// # Safety
/// As [`ai_reflect_field_count`]; `i` must be in `0..field_count`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_reflect_field_name(
    thread: *mut Thread,
    str_type_id: i64,
    obj: *const u8,
    i: i64,
) -> *mut u8 {
    unsafe {
        let heap = &*(*thread).heap;
        // Clone the name out before allocating (the alloc may move the heap, but
        // the metadata table is off-heap and immortal, so the &str stays valid;
        // we copy to be explicit and avoid borrowing across the alloc).
        let name = field_at(active_fields(heap, obj), i, "field_name").name.clone();
        alloc_string_from_bytes(thread, str_type_id as u32, name.as_bytes())
    }
}

// =============================================================================
// Runtime extern functions called by compiled code
// =============================================================================

/// Allocate a fixed-size (non-varlen) heap object of the shape described by the
/// `TypeInfo` at index `type_id` in the heap's type table. Returns a pointer to
/// the object (past nothing — the header is at offset 0). The object's header is
/// initialized; value-field slots are zeroed.
///
/// Compiled code spills all live roots into its frame before calling this,
/// because a collection triggered here can relocate every live object.
///
/// # Safety
/// `thread` must be a valid `*mut Thread` whose `heap` points at a live `Heap`,
/// and `type_id` must index a registered `TypeInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_gc_alloc_fixed(thread: *mut Thread, type_id: u32) -> *mut u8 {
    unsafe {
        let t = &*thread;
        let heap = &*t.heap;
        let info: &TypeInfo = heap.type_info_by_id(type_id as u16);
        alloc_with_published_frame(t, heap, info, 0)
    }
}

/// Read a closure env object's CODE POINTER. The code pointer is stored at
/// `HEADER + ptr_fields*8` (right after the captured GC-pointer slots), but
/// `ptr_fields` varies per closure and is NOT known at a polymorphic call/spawn
/// site (where the closure is typed as a bare `fn()` and lowers to a placeholder
/// layout with ptr_fields=0). So we recover it at runtime from the env object's
/// own header `type_id` → `TypeInfo.value_field_count`. This is the single source
/// of truth for the code-pointer offset; codegen's `gen_call_closure` and
/// `ThreadSpawn` both use it instead of a static (possibly-wrong) offset. See
/// `docs/threads.md`.
///
/// # Safety
/// `thread` valid; `env` a valid closure env object allocated by `gen_make_closure`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_closure_code_ptr(thread: *mut Thread, env: *const u8) -> *const u8 {
    unsafe {
        let heap = &*(*thread).heap;
        // Header: [gc_word: u64 @0][type_id: u16 @8]. (Full::SIZE = 16.)
        let type_id = *(env.add(8) as *const u16);
        let info: &TypeInfo = heap.type_info_by_id(type_id);
        let ptr_fields = info.value_field_count as usize;
        let code_off = 16 + ptr_fields * 8; // 16 = Full header size
        *(env.add(code_off) as *const *const u8)
    }
}

/// Allocate a variable-length heap object (array / string / bytes shape) with
/// `varlen_len` trailing elements, per the `TypeInfo` at `type_id`.
///
/// # Safety
/// As [`ai_gc_alloc_fixed`]; additionally the `TypeInfo` must be a varlen shape.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_gc_alloc_varlen(
    thread: *mut Thread,
    type_id: u32,
    varlen_len: u64,
) -> *mut u8 {
    unsafe {
        let t = &*thread;
        let heap = &*t.heap;
        let info: &TypeInfo = heap.type_info_by_id(type_id as u16);
        alloc_with_published_frame(t, heap, info, varlen_len as usize)
    }
}

/// Generational write barrier. Compiled code calls this AFTER storing the
/// pointer `new_val` into a field of object `obj`. If `obj` lives in tenured
/// space and `new_val` points into the nursery, the card covering `obj` is
/// marked dirty so the next minor GC treats `obj` as a root (finding the
/// old→young pointer without scanning all of tenured space).
///
/// Fast path: for a non-generational heap (no nursery) this is a single load +
/// branch and returns immediately. Null/non-nursery writes also return cheaply.
///
/// # Safety
/// `thread` valid; `obj` a valid heap object; `new_val` a heap pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_gc_write_barrier(thread: *mut Thread, obj: *mut u8, new_val: *mut u8) {
    unsafe {
        let heap = &*(*thread).heap;
        // Cheapest possible bail-out for the common (non-generational) case.
        if !heap.has_nursery() {
            return;
        }
        if !new_val.is_null() && heap.is_nursery(new_val) && heap.is_tenured(obj) {
            heap.mark_card_dirty(obj);
        }
    }
}

/// FFI `as_c_bytes`: copy a `String`'s UTF-8 bytes followed by a NUL terminator
/// into the caller-provided buffer `dst`, which must have room for
/// `str_len(s) + 1` bytes. The caller (compiled code) allocates `dst` as a
/// dynamically-sized stack alloca, so the resulting C pointer is stable for the
/// duration of the enclosing extern call and needs no pinning. Returns the byte
/// length (excluding the NUL). See `docs/ffi.md`.
///
/// # Safety
/// `s` must be a valid `String` object; `dst` must point to at least
/// `str_len(s) + 1` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_copy_to_buf(_thread: *mut Thread, s: *const u8, dst: *mut u8) -> i64 {
    unsafe {
        let bytes = str_bytes(s);
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        *dst.add(bytes.len()) = 0; // NUL terminator for C string consumers
        bytes.len() as i64
    }
}

/// FFI `as_c_bytes` (generic): copy `byte_len` bytes from a varlen object's data
/// region (a `String` or scalar `Array`, both laid out as
/// `[header][count][data…]`) into the caller-provided stack buffer `dst`. Used
/// for both strings and arrays; the caller passes the exact byte length (for a
/// String this is its length, for an `Array<T>` it is `len * sizeof(T)`).
///
/// # Safety
/// `obj` must be a valid varlen object with at least `byte_len` data bytes; `dst`
/// must point to at least `byte_len` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_buf_copy_in(_thread: *mut Thread, obj: *const u8, dst: *mut u8, byte_len: i64) {
    unsafe {
        let src = obj.add(STR_DATA_OFF);
        core::ptr::copy_nonoverlapping(src, dst, byte_len as usize);
    }
}

/// FFI copy-out: copy `byte_len` bytes from the stack buffer `src` BACK into a
/// varlen object's data region. Pairs with [`ai_buf_copy_in`] for a `mut` array
/// argument that the C function filled in place (e.g. `read(fd, buf, n)`).
///
/// # Safety
/// `obj` must be a valid varlen object with room for `byte_len` data bytes; `src`
/// must point to at least `byte_len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_buf_copy_out(_thread: *mut Thread, obj: *mut u8, src: *const u8, byte_len: i64) {
    unsafe {
        let dst = obj.add(STR_DATA_OFF);
        core::ptr::copy_nonoverlapping(src, dst, byte_len as usize);
    }
}

/// FFI boundary: enter a foreign (`extern "C"`) call. Publishes this thread's
/// current frame chain so that a garbage collection occurring while we are
/// inside native code (triggered by another mutator thread) can still find this
/// thread's live roots — i.e. the thread is treated as parked at a safepoint for
/// the duration of the call. This is the "managed → native" transition.
///
/// Compiled code calls this immediately BEFORE every `extern "C"` call and pairs
/// it with [`ai_ffi_leave`] immediately after. Because roots are published, a GC
/// may safely run during the call: the FFI is *safe by default*, not dependent
/// on the foreign function promising not to allocate. Per the FFI design, only
/// non-moving data (scalars, pointers to stack-resident value structs) crosses
/// the boundary, so the collector never needs to update anything native code
/// holds. See `docs/ffi.md`.
///
/// # Safety
/// `thread` must be a valid `*mut Thread` with a live `dyna_thread`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_ffi_enter(thread: *mut Thread) {
    unsafe {
        let t = &*thread;
        let dyna = &*t.dyna_thread;
        dyna.set_parked_jit_fp(t.top_frame as *const u8);
    }
}

/// FFI boundary: leave a foreign call (the "native → managed" transition).
/// Clears the published frame pointer set by [`ai_ffi_enter`]. In a future
/// multi-threaded runtime this is also where a thread returning from native code
/// blocks until an in-progress stop-the-world collection completes; today, with
/// a single mutator, the clear is sufficient.
///
/// # Safety
/// As [`ai_ffi_enter`]; must be paired with a preceding `ai_ffi_enter`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_ffi_leave(thread: *mut Thread) {
    unsafe {
        let dyna = &*(*thread).dyna_thread;
        dyna.clear_parked_jit_fp();
    }
}

/// Callback boundary: the inverse transition, for a gc-rust function invoked as a
/// C **callback** while the thread is parked "in native" (inside an outer
/// `extern` call). The callback re-enters managed code and may allocate, so it
/// must re-acquire managed state — clear the published frame pointer — on entry
/// ([`ai_ffi_reenter`]) and re-publish it on exit ([`ai_ffi_exit`]) since control
/// returns to the foreign caller. A callback trampoline brackets the real
/// gc-rust call with this pair. See `docs/ffi.md`.
///
/// # Safety
/// `thread` must be the live current mutator thread (from `ai_current_thread`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_ffi_reenter(thread: *mut Thread) {
    unsafe {
        let dyna = &*(*thread).dyna_thread;
        dyna.clear_parked_jit_fp();
    }
}

/// Callback boundary: re-publish the frame pointer before returning from a
/// callback into foreign code. Pairs with [`ai_ffi_reenter`].
///
/// # Safety
/// As [`ai_ffi_reenter`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_ffi_exit(thread: *mut Thread) {
    unsafe {
        let t = &*thread;
        let dyna = &*t.dyna_thread;
        dyna.set_parked_jit_fp(t.top_frame as *const u8);
    }
}

/// Shared allocation helper: publish the current frame chain so a GC triggered
/// inside `heap.alloc` can find our roots, allocate, then clear the published
/// fp. Allocation may move objects, so callers must have spilled live roots to
/// frame slots already.
#[inline]
unsafe fn alloc_with_published_frame(
    t: &Thread,
    heap: &Heap,
    info: &TypeInfo,
    varlen_len: usize,
) -> *mut u8 {
    unsafe {
        let dyna = &*t.dyna_thread;
        dyna.set_parked_jit_fp(t.top_frame as *const u8);

        // Stress mode: collect on EVERY allocation, maximally exercising precise
        // rooting and relocation. The frame is published above, so all live roots
        // are scannable (ANF guarantees no GC value is stranded in a register, and
        // construction sites reload GC operands after the alloc). Sound for both
        // single- and multi-threaded programs. Minor GC for generational heaps; a
        // full collection for the semi-space heap the `--gc-stress` driver uses.
        if heap.gc_every_alloc() {
            if heap.has_nursery() {
                heap.mutator_triggered_minor_gc::<IdentityPtrPolicy>(dyna);
            } else {
                heap.mutator_triggered_gc::<IdentityPtrPolicy>(dyna);
            }
        }

        // `alloc_obj::<Full>` (not bare `alloc`) stamps the object header with
        // `info.type_id`; bare `alloc` only zeroes, leaving type_id 0, which
        // breaks the GC scanner.
        let p = if heap.has_nursery() {
            // Generational: allocate in the nursery; on full, a MINOR GC (cheap,
            // scavenges only the young generation), retry, then fall back to a
            // major GC, and finally to tenured space directly.
            let mut p = heap.alloc_nursery_obj::<Full>(info, varlen_len);
            if p.is_null() {
                heap.mutator_triggered_minor_gc::<IdentityPtrPolicy>(dyna);
                p = heap.alloc_nursery_obj::<Full>(info, varlen_len);
            }
            if p.is_null() {
                heap.mutator_triggered_gc::<IdentityPtrPolicy>(dyna);
                p = heap.alloc_nursery_obj::<Full>(info, varlen_len);
            }
            if p.is_null() {
                // The object doesn't fit the (just-cleared) nursery — allocate it
                // straight into tenured space (large objects, or nursery too small).
                p = heap.alloc_obj::<Full>(info, varlen_len);
            }
            p
        } else {
            // Non-generational semi-space: from-space, then a major GC + retry.
            let mut p = heap.alloc_obj::<Full>(info, varlen_len);
            if p.is_null() {
                heap.mutator_triggered_gc::<IdentityPtrPolicy>(dyna);
                p = heap.alloc_obj::<Full>(info, varlen_len);
            }
            p
        };

        if p.is_null() {
            dyna.clear_parked_jit_fp();
            eprintln!(
                "gc-rust: out of memory — live set exceeds the heap \
                 (object type_id {}, {} varlen elems)",
                info.type_id, varlen_len,
            );
            std::process::abort();
        }
        dyna.clear_parked_jit_fp();
        p
    }
}

/// Print a signed 64-bit integer followed by a newline. A minimal IO primitive
/// so compiled programs can emit output. Returns 0.
#[unsafe(no_mangle)]
pub extern "C" fn ai_print_int(_thread: *mut Thread, v: i64) -> i64 {
    println!("{}", v);
    0
}

/// Print a 64-bit float followed by a newline. Returns 0.
#[unsafe(no_mangle)]
pub extern "C" fn ai_print_float(_thread: *mut Thread, v: f64) -> i64 {
    println!("{}", v);
    0
}

// ---- String primitives ----------------------------------------------------
//
// A gc-rust `String` is a varlen `Bytes` object: [ObjHeader (Full::SIZE)] then a
// u64 byte-length count word, then the UTF-8 bytes. This matches the layout
// codegen emits for string literals and what `String`'s registered `TypeInfo`
// (varlen bytes) describes. The constants below mirror `codegen`'s `HEADER`.

/// Byte offset from an object pointer to its varlen count word.
const STR_COUNT_OFF: usize = 16; // Full::SIZE
/// Byte offset from an object pointer to the first UTF-8 byte.
const STR_DATA_OFF: usize = 24; // Full::SIZE + 8 (count word)

/// Borrow a `String` object's bytes as a slice. The pointer must be live (not
/// mid-relocation) for the duration of the borrow.
///
/// # Safety
/// `s` must point to a valid gc-rust `String` object.
unsafe fn str_bytes<'a>(s: *const u8) -> &'a [u8] {
    unsafe {
        let len = *(s.add(STR_COUNT_OFF) as *const u64) as usize;
        core::slice::from_raw_parts(s.add(STR_DATA_OFF), len)
    }
}

/// `print_str`: write a `String`'s bytes followed by a newline. Returns 0.
///
/// # Safety
/// `s` must point to a valid gc-rust `String` object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_print_str(_thread: *mut Thread, s: *const u8) -> i64 {
    unsafe {
        // Lossy is fine for output; gc-rust strings are UTF-8 by construction.
        let text = String::from_utf8_lossy(str_bytes(s));
        println!("{}", text);
    }
    0
}

/// `print`: write a `String`'s bytes with NO trailing newline, then flush so
/// partial-line output (prompts, progress) appears immediately. Returns 0.
///
/// # Safety
/// `s` must point to a valid gc-rust `String` object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_print_str_raw(_thread: *mut Thread, s: *const u8) -> i64 {
    use std::io::Write;
    unsafe {
        let bytes = str_bytes(s);
        let mut out = std::io::stdout();
        let _ = out.write_all(bytes);
        let _ = out.flush();
    }
    0
}

/// `str_len`: the byte length of a `String`.
///
/// # Safety
/// `s` must point to a valid gc-rust `String` object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_len(_thread: *mut Thread, s: *const u8) -> i64 {
    unsafe { *(s.add(STR_COUNT_OFF) as *const u64) as i64 }
}

/// `str_eq`: byte-wise equality of two `String`s. Returns 1 if equal, else 0.
///
/// # Safety
/// `a` and `b` must point to valid gc-rust `String` objects.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_eq(_thread: *mut Thread, a: *const u8, b: *const u8) -> i64 {
    unsafe { (str_bytes(a) == str_bytes(b)) as i64 }
}

/// `str_concat`: allocate a fresh `String` (whose varlen `Bytes` shape is the
/// `TypeInfo` at `type_id`) holding `a` followed by `b`.
///
/// The source bytes are copied into a Rust-owned buffer *before* allocating, so
/// a collection triggered by the allocation (which may relocate `a`/`b`) cannot
/// invalidate the copy. Codegen passes the String layout's `type_id` (the same
/// one it uses for string literals), mirroring `ai_gc_alloc_varlen`.
///
/// # Safety
/// `a` and `b` must point to valid gc-rust `String` objects; `thread` valid;
/// `type_id` must index the registered `String` `TypeInfo`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_concat(
    thread: *mut Thread,
    type_id: u32,
    a: *const u8,
    b: *const u8,
) -> *mut u8 {
    unsafe {
        // Copy source bytes out first — after this point a GC may move a/b.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(str_bytes(a));
        buf.extend_from_slice(str_bytes(b));
        alloc_string_from_bytes(thread, type_id, &buf)
    }
}

/// Allocate a fresh `String` (varlen `Bytes`, `TypeInfo` at `type_id`) holding
/// `bytes`. The caller must already hold `bytes` in Rust-owned memory (not in a
/// GC object), since the allocation may relocate the heap.
///
/// # Safety
/// `thread` valid; `type_id` indexes the registered `String` `TypeInfo`.
unsafe fn alloc_string_from_bytes(thread: *mut Thread, type_id: u32, bytes: &[u8]) -> *mut u8 {
    unsafe {
        let t = &*thread;
        let heap = &*t.heap;
        let info: &TypeInfo = heap.type_info_by_id(type_id as u16);
        let obj = alloc_with_published_frame(t, heap, info, bytes.len());
        *(obj.add(STR_COUNT_OFF) as *mut u64) = bytes.len() as u64;
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), obj.add(STR_DATA_OFF), bytes.len());
        obj
    }
}

/// `str_get`: the byte at index `i` (as an i64), or -1 if out of range.
///
/// # Safety
/// `s` must point to a valid gc-rust `String` object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_get(_thread: *mut Thread, s: *const u8, i: i64) -> i64 {
    unsafe {
        let bytes = str_bytes(s);
        if i < 0 || i as usize >= bytes.len() { -1 } else { bytes[i as usize] as i64 }
    }
}

/// `str_substring`: a fresh `String` with bytes `[start, end)` of `s`. Indices
/// are clamped to `[0, len]` and an empty range yields the empty string.
///
/// # Safety
/// `s` valid `String`; `thread`/`type_id` as `ai_str_concat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_substring(
    thread: *mut Thread,
    type_id: u32,
    s: *const u8,
    start: i64,
    end: i64,
) -> *mut u8 {
    unsafe {
        let bytes = str_bytes(s);
        let len = bytes.len() as i64;
        let lo = start.clamp(0, len) as usize;
        let hi = end.clamp(start.max(0), len) as usize;
        let slice: Vec<u8> = bytes[lo..hi].to_vec();
        alloc_string_from_bytes(thread, type_id, &slice)
    }
}

/// `read_file`: a fresh `String` with the bytes of the file at `path` (empty if
/// it can't be read). Enables a self-hosted compiler driver to read source files.
///
/// # Safety
/// `path` valid `String`; `thread`/`type_id` as `ai_str_concat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_read_file(thread: *mut Thread, type_id: u32, path: *const u8) -> *mut u8 {
    unsafe {
        let bytes = str_bytes(path);
        let path_str = std::str::from_utf8(bytes).unwrap_or("");
        let content = std::fs::read(path_str).unwrap_or_default();
        alloc_string_from_bytes(thread, type_id, &content)
    }
}

/// `str_from_int`: a fresh `String` holding the decimal rendering of `v`.
///
/// # Safety
/// `thread`/`type_id` as `ai_str_concat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_from_int(thread: *mut Thread, type_id: u32, v: i64) -> *mut u8 {
    unsafe {
        let s = v.to_string();
        alloc_string_from_bytes(thread, type_id, s.as_bytes())
    }
}

/// `type_name_of(obj)`: a fresh `String` holding the source type name of `obj`,
/// looked up in the heap's reflection metadata table by the object's header
/// `type_id`. `str_type_id` is the `String` layout id used to allocate the
/// result. Falls back to `<type N>` when no metadata is installed for the type.
///
/// # Safety
/// `thread` valid with a live heap; `obj` a valid heap object with an
/// initialized header; `str_type_id` the registered `String` layout.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_type_name(thread: *mut Thread, str_type_id: u32, obj: *const u8) -> *mut u8 {
    unsafe {
        let heap = &*(*thread).heap;
        let obj_tid = heap.obj_type_id(obj);
        let name = heap
            .type_meta_by_id(obj_tid)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| format!("<type {}>", obj_tid));
        alloc_string_from_bytes(thread, str_type_id, name.as_bytes())
    }
}

/// `char_to_str`: a fresh `String` holding the UTF-8 encoding of the Unicode
/// scalar value `cp` (1–4 bytes). Invalid scalars — negative, above U+10FFFF, or
/// a surrogate (U+D800..U+DFFF) — become U+FFFD, so this is total and always
/// produces valid UTF-8.
///
/// # Safety
/// `thread`/`type_id` as `ai_str_concat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_char_to_str(thread: *mut Thread, type_id: u32, cp: i64) -> *mut u8 {
    let ch = u32::try_from(cp).ok().and_then(char::from_u32).unwrap_or('\u{FFFD}');
    let mut buf = [0u8; 4];
    let s = ch.encode_utf8(&mut buf);
    unsafe { alloc_string_from_bytes(thread, type_id, s.as_bytes()) }
}

/// `str_from_float`: a fresh `String` holding the rendering of `v`.
///
/// # Safety
/// `thread`/`type_id` as `ai_str_concat`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_from_float(thread: *mut Thread, type_id: u32, v: f64) -> *mut u8 {
    unsafe {
        let s = v.to_string();
        alloc_string_from_bytes(thread, type_id, s.as_bytes())
    }
}

/// `str_hash`: a stable 63-bit FNV-1a hash of a `String`'s bytes (non-negative
/// so it can be used as an i64 array index after a modulo). Does not allocate.
///
/// # Safety
/// `s` must point to a valid gc-rust `String` object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_hash(_thread: *mut Thread, s: *const u8) -> i64 {
    unsafe {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in str_bytes(s) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        // Mask to 63 bits so the result is always non-negative as an i64.
        (h & 0x7fff_ffff_ffff_ffff) as i64
    }
}

/// `str_to_float`: parse a `String` as f64, returning 0.0 on a malformed input.
/// Does not allocate.
///
/// # Safety
/// `s` must point to a valid gc-rust `String` object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_to_float(_thread: *mut Thread, s: *const u8) -> f64 {
    unsafe {
        match core::str::from_utf8(str_bytes(s)) {
            Ok(text) => text.trim().parse::<f64>().unwrap_or(0.0),
            Err(_) => 0.0,
        }
    }
}

/// Safepoint poll slow path. Compiled code inlines a load of `thread.state` at
/// loop back-edges; when it is non-zero, it traps here. We publish our frame
/// chain and park at a safepoint until the GC that requested the stop completes,
/// then return so the mutator resumes.
///
/// # Safety
/// `thread` must be a valid `*mut Thread` with a live `dyna_thread`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_gc_pollcheck_slow(thread: *mut Thread) {
    unsafe {
        let t = &*thread;
        let dyna = &*t.dyna_thread;
        dyna.set_parked_jit_fp(t.top_frame as *const u8);
        dyna.enter_safepoint();
        dyna.clear_parked_jit_fp();
    }
}
