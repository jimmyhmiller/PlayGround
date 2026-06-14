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
/// reads `num_roots` to know how many root slots follow the frame header.
#[repr(C)]
pub struct FrameOrigin {
    pub num_roots: u32,
    _pad: u32,
    /// Function name, NUL-terminated, for debugging / GC tracing. May be null.
    pub name: *const u8,
}

impl FrameOrigin {
    pub const fn new(num_roots: u32, name: *const u8) -> Self {
        FrameOrigin {
            num_roots,
            _pad: 0,
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
            let num_roots = if origin.is_null() {
                0
            } else {
                (*origin).num_roots as usize
            };
            // Root slots start at byte offset ROOTS past the frame header.
            let slots = (frame as *const u8).add(frame_offsets::ROOTS) as *mut u64;
            for i in 0..num_roots {
                visitor(slots.add(i));
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
        let heap = Arc::new(Heap::new::<Full>(space_size, type_table));
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

    /// Force a stop-the-world collection. The calling thread must be at a
    /// safepoint or unregistered; here it is driven from test/host code with
    /// the mutator parked, so it is safe.
    ///
    /// # Safety
    /// No other registered mutator may be running outside a safepoint.
    pub unsafe fn force_collect(&self) {
        unsafe { self.heap.stw_collect::<IdentityPtrPolicy>() }
    }
}

impl Drop for RuntimeContext {
    fn drop(&mut self) {
        self.heap.safe_deregister_thread(&self.dyna);
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
        let p = heap.alloc(info, varlen_len);
        dyna.clear_parked_jit_fp();
        p
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
