use std::cell::RefCell;
use std::marker::PhantomData;

use dynalloc::{Heap, Mutator, PtrPolicy, Root, RootScope};
use dynexec::{ConservativeWordRoots, PreciseStackRoots, RootTransport, ValueLayout};
use dynir::InterpRootManager;
use dynobj::{Compact, ObjHeader, TypeInfo};

use crate::jit::GcPolicy;

// ─── MutatorRootManager ─────────────────────────────────────────────

/// Per-frame scope info for the MutatorRootManager.
struct MutatorFrameScope {
    scope: RootScope,
    roots: Vec<Root>,
}

/// Root manager for [`ModuleInterpreter`] using [`Mutator`]-based root tracking,
/// generic over [`PtrPolicy`].
///
/// Each pushed frame creates new Root handles in the Mutator via `save`/`restore`
/// scoping. The GC traces the Mutator's root slots to find all live heap pointers.
///
/// Implements [`InterpRootManager`] so it can be stored in a [`ModuleInterpreter`].
pub struct MutatorRootManager<P: PtrPolicy> {
    heap: Heap,
    mutator: RefCell<Mutator>,
    frame_scopes: RefCell<Vec<MutatorFrameScope>>,
    gc_policy: GcPolicy,
    _policy: PhantomData<P>,
}

impl<P: PtrPolicy> MutatorRootManager<P> {
    /// Create a new root manager with the given semi-space size.
    /// Defaults to [`GcPolicy::NeverAuto`]; use [`Self::with_gc_policy`]
    /// to opt into pressure-based or stress-mode collection.
    pub fn new(space_size: usize, type_table: Vec<TypeInfo>) -> Self {
        MutatorRootManager {
            heap: Heap::new::<Compact>(space_size, type_table),
            mutator: RefCell::new(Mutator::new()),
            frame_scopes: RefCell::new(Vec::new()),
            gc_policy: GcPolicy::NeverAuto,
            _policy: PhantomData,
        }
    }

    /// Create with a custom header type.
    pub fn new_with_header<H: ObjHeader>(space_size: usize, type_table: Vec<TypeInfo>) -> Self {
        MutatorRootManager {
            heap: Heap::new::<H>(space_size, type_table),
            mutator: RefCell::new(Mutator::new()),
            frame_scopes: RefCell::new(Vec::new()),
            gc_policy: GcPolicy::NeverAuto,
            _policy: PhantomData,
        }
    }

    /// Set the collection policy. See [`GcPolicy`].
    pub fn with_gc_policy(mut self, policy: GcPolicy) -> Self {
        self.gc_policy = policy;
        self
    }

    /// Allocate a heap object.
    pub fn alloc(&self, info: &TypeInfo, varlen_len: usize) -> *mut u8 {
        self.heap.alloc_obj::<Compact>(info, varlen_len)
    }

    /// Allocate a heap object with a custom header type.
    pub fn alloc_with_header<H: ObjHeader>(
        &self,
        info: &TypeInfo,
        varlen_len: usize,
    ) -> *mut u8 {
        self.heap.alloc_obj::<H>(info, varlen_len)
    }

    /// Access the underlying heap (e.g. for sharing with a runtime that allocates).
    pub fn heap(&self) -> &Heap {
        &self.heap
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

impl<P, L, Transport> InterpRootManager<L, PreciseStackRoots, Transport> for MutatorRootManager<P>
where
    P: PtrPolicy,
    L: ValueLayout,
    Transport: RootTransport<L, PreciseStackRoots>,
{
    fn push_frame(&self, gc_slot_count: usize) -> usize {
        let mut mutator = self.mutator.borrow_mut();
        let scope = mutator.save();
        let roots: Vec<Root> = (0..gc_slot_count).map(|_| mutator.root(0)).collect();
        let mut scopes = self.frame_scopes.borrow_mut();
        let idx = scopes.len();
        scopes.push(MutatorFrameScope { scope, roots });
        idx
    }

    fn pop_frame(&self) {
        let frame_scope = self
            .frame_scopes
            .borrow_mut()
            .pop()
            .expect("no frame to pop");
        self.mutator.borrow_mut().restore(frame_scope.scope);
    }

    fn set_root(&self, frame: usize, slot: usize, value: u64) {
        let scopes = self.frame_scopes.borrow();
        let root = &scopes[frame].roots[slot];
        self.mutator.borrow().set(root, value);
    }

    fn get_root(&self, frame: usize, slot: usize) -> u64 {
        let scopes = self.frame_scopes.borrow();
        let root = &scopes[frame].roots[slot];
        self.mutator.borrow().get(root).bits()
    }

    fn clear_frame(&self, frame: usize) {
        let scopes = self.frame_scopes.borrow();
        let mutator = self.mutator.borrow();
        for root in &scopes[frame].roots {
            mutator.set(root, 0);
        }
    }

    fn collect(&self) {
        if !self
            .gc_policy
            .should_collect(self.heap.from_used(), self.heap.space_size())
        {
            return;
        }
        let mutator = self.mutator.borrow();
        unsafe {
            self.heap.collect::<P>(&[&*mutator]);
        }
    }
}

impl<P, L, Transport> InterpRootManager<L, ConservativeWordRoots, Transport> for MutatorRootManager<P>
where
    P: PtrPolicy,
    L: ValueLayout,
    Transport: RootTransport<L, ConservativeWordRoots>,
{
    fn push_frame(&self, gc_slot_count: usize) -> usize {
        let mut mutator = self.mutator.borrow_mut();
        let scope = mutator.save();
        let roots: Vec<Root> = (0..gc_slot_count).map(|_| mutator.root(0)).collect();
        let mut scopes = self.frame_scopes.borrow_mut();
        let idx = scopes.len();
        scopes.push(MutatorFrameScope { scope, roots });
        idx
    }

    fn pop_frame(&self) {
        let frame_scope = self
            .frame_scopes
            .borrow_mut()
            .pop()
            .expect("no frame to pop");
        self.mutator.borrow_mut().restore(frame_scope.scope);
    }

    fn set_root(&self, frame: usize, slot: usize, value: u64) {
        let scopes = self.frame_scopes.borrow();
        let root = &scopes[frame].roots[slot];
        self.mutator.borrow().set(root, value);
    }

    fn get_root(&self, frame: usize, slot: usize) -> u64 {
        let scopes = self.frame_scopes.borrow();
        let root = &scopes[frame].roots[slot];
        self.mutator.borrow().get(root).bits()
    }

    fn clear_frame(&self, frame: usize) {
        let scopes = self.frame_scopes.borrow();
        let mutator = self.mutator.borrow();
        for root in &scopes[frame].roots {
            mutator.set(root, 0);
        }
    }

    fn collect(&self) {
        if !self
            .gc_policy
            .should_collect(self.heap.from_used(), self.heap.space_size())
        {
            return;
        }
        let mutator = self.mutator.borrow();
        unsafe {
            self.heap.collect::<P>(&[&*mutator]);
        }
    }
}
