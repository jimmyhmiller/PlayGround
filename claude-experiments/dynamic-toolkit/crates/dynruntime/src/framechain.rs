use std::marker::PhantomData;

use dynalloc::{Heap, PtrPolicy};
use dynexec::{ConservativeWordRoots, PreciseStackRoots, RootTransport, ValueLayout};
use dynir::InterpRootManager;
use dynobj::{Compact, DynRootFrame, FrameChain, ObjHeader, TypeInfo};

// ─── FrameChainRootManager ──────────────────────────────────────────

/// Root manager for [`ModuleInterpreter`] using [`FrameChain`], generic over [`PtrPolicy`].
///
/// Uses the real [`DynRootFrame`] + [`FrameChain`] infrastructure from dynobj.
/// Each pushed frame is a [`DynRootFrame`] linked into the chain. The GC
/// walks the chain via [`FrameChain`]'s [`RootSource`] impl to find all roots.
///
/// Frame lifetimes are managed manually via [`FrameChain::push_raw_unguarded`]
/// and [`FrameChain::pop_raw`] (no RAII guards) since the interpreter manages
/// push/pop dynamically rather than via stack discipline.
///
/// Note: `DynRootFrame` is safe to move (e.g. when the `Vec` reallocates)
/// because its `header_ptr()` points into its internal `Vec<u64>` heap
/// allocation, not into the struct itself.
pub struct FrameChainRootManager<P: PtrPolicy> {
    heap: Heap,
    chain: FrameChain,
    frames: std::cell::RefCell<Vec<DynRootFrame>>,
    _policy: PhantomData<P>,
}

impl<P: PtrPolicy> FrameChainRootManager<P> {
    /// Create a new root manager with the given semi-space size.
    pub fn new(space_size: usize) -> Self {
        FrameChainRootManager {
            heap: Heap::new::<Compact>(space_size),
            chain: FrameChain::new(),
            frames: std::cell::RefCell::new(Vec::new()),
            _policy: PhantomData,
        }
    }

    /// Create with a custom header type.
    pub fn new_with_header<H: ObjHeader>(space_size: usize) -> Self {
        FrameChainRootManager {
            heap: Heap::new::<H>(space_size),
            chain: FrameChain::new(),
            frames: std::cell::RefCell::new(Vec::new()),
            _policy: PhantomData,
        }
    }

    /// Allocate a heap object.
    pub fn alloc(&self, info: &'static TypeInfo, varlen_len: usize) -> *mut u8 {
        self.heap.alloc_obj::<Compact>(info, varlen_len)
    }

    /// Allocate a heap object with a custom header type.
    pub fn alloc_with_header<H: ObjHeader>(
        &self,
        info: &'static TypeInfo,
        varlen_len: usize,
    ) -> *mut u8 {
        self.heap.alloc_obj::<H>(info, varlen_len)
    }

    /// Number of collections so far.
    pub fn collections(&self) -> usize {
        self.heap.collections()
    }

    /// Bytes used in from-space.
    pub fn from_used(&self) -> usize {
        self.heap.from_used()
    }
}

impl<P, L, Transport> InterpRootManager<L, PreciseStackRoots, Transport> for FrameChainRootManager<P>
where
    P: PtrPolicy,
    L: ValueLayout,
    Transport: RootTransport<L, PreciseStackRoots>,
{
    fn push_frame(&self, gc_slot_count: usize) -> usize {
        let mut frames = self.frames.borrow_mut();
        let idx = frames.len();
        let frame = DynRootFrame::new(gc_slot_count);
        unsafe {
            self.chain.push_raw_unguarded(frame.header_ptr());
        }
        frames.push(frame);
        idx
    }

    fn pop_frame(&self) {
        // Pop from chain first (while frame memory is still valid),
        // then deallocate the frame.
        unsafe {
            self.chain.pop_raw();
        }
        self.frames.borrow_mut().pop().expect("no frame to pop");
    }

    fn set_root(&self, frame: usize, slot: usize, value: u64) {
        self.frames.borrow()[frame].set(slot, value);
    }

    fn get_root(&self, frame: usize, slot: usize) -> u64 {
        self.frames.borrow()[frame].get(slot)
    }

    fn clear_frame(&self, frame: usize) {
        self.frames.borrow()[frame].clear_all();
    }

    fn collect(&self) {
        // FrameChain implements RootSource — it walks the linked list
        // of DynRootFrames, visiting all slots.
        unsafe {
            self.heap.collect::<P>(&[&self.chain]);
        }
    }
}

impl<P, L, Transport> InterpRootManager<L, ConservativeWordRoots, Transport>
    for FrameChainRootManager<P>
where
    P: PtrPolicy,
    L: ValueLayout,
    Transport: RootTransport<L, ConservativeWordRoots>,
{
    fn push_frame(&self, gc_slot_count: usize) -> usize {
        let mut frames = self.frames.borrow_mut();
        let idx = frames.len();
        let frame = DynRootFrame::new(gc_slot_count);
        unsafe {
            self.chain.push_raw_unguarded(frame.header_ptr());
        }
        frames.push(frame);
        idx
    }

    fn pop_frame(&self) {
        unsafe {
            self.chain.pop_raw();
        }
        self.frames.borrow_mut().pop().expect("no frame to pop");
    }

    fn set_root(&self, frame: usize, slot: usize, value: u64) {
        self.frames.borrow()[frame].set(slot, value);
    }

    fn get_root(&self, frame: usize, slot: usize) -> u64 {
        self.frames.borrow()[frame].get(slot)
    }

    fn clear_frame(&self, frame: usize) {
        self.frames.borrow()[frame].clear_all();
    }

    fn collect(&self) {
        unsafe {
            self.heap.collect::<P>(&[&self.chain]);
        }
    }
}
