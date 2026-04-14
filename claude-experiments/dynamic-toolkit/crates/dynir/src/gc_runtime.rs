//! Real GC-backed root manager + allocator for the dynir interpreter.
//!
//! This module bridges three separate pieces:
//!
//! 1. A `dynalloc::SemiSpace` heap (the allocator and collector).
//! 2. The `dynexec::InterpRootManager` trait — the contract the
//!    interpreter uses to push/pop/update GC root frames.
//! 3. The `dynalloc::Alloc` trait — the contract the continuation
//!    capture/read functions use for allocation.
//!
//! One type, [`GcInterpCtx`], implements both traits. The test harness
//! (or embedding language) constructs a `GcInterpCtx` holding a
//! `RefCell<SemiSpace>`, a type table (including the continuation
//! types registered via `ContinuationTypes::register_into`), and
//! passes `&ctx` to `ModuleInterpreter::new`.
//!
//! This is the first type in the tree that plugs a real `SemiSpace`
//! into the reference interpreter. Before this, the only existing
//! `InterpRootManager` impl was `NoGcRoots`, a no-op.

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;

use dynalloc::{Alloc, PtrPolicy, SemiSpace};
use dynobj::{ObjHeader, RootSource, TypeInfo};
use dynexec::{
    capture_continuation, read_continuation, CapturedStackBuilder, ContinuationContext,
    ContinuationTypes, ContinuationView, RootPrecision, RootStrategy, RootTransport,
    ValueLayout,
};

use crate::interp::InterpRootManager;

/// A real GC-backed interpreter context.
///
/// Owns a `RefCell<SemiSpace>` (so `collect()` can take `&mut` through
/// a `&self` method), the full type table including continuation
/// types, a copy of the `ContinuationTypes` descriptor for fast access,
/// and a stack of root frames for the interpreter's per-frame GC slots.
///
/// Generic over `P: PtrPolicy` so the embedding language can bring its
/// own pointer tag scheme (NanBox, LowBit, etc.).
pub struct GcInterpCtx<H: ObjHeader, P: PtrPolicy> {
    heap: RefCell<SemiSpace>,
    type_table: Vec<TypeInfo>,
    cont_types: ContinuationTypes,
    root_frames: RefCell<Vec<Vec<Cell<u64>>>>,
    /// Extra root sources registered by the embedding (e.g., the JIT's
    /// continuation store). Raw pointers — the caller is responsible
    /// for ensuring they outlive the `collect()` call.
    extra_roots: RefCell<Vec<*const dyn RootSource>>,
    /// Number of allocations since the last collection. Bumped by
    /// `Alloc::alloc`, checked by `should_collect`.
    alloc_count: Cell<usize>,
    /// Allocations-since-last-gc threshold for auto-collection. When
    /// `alloc_count >= gc_threshold`, the next call to `needs_gc`
    /// returns `true`. Set to `usize::MAX` to disable auto-gc.
    gc_threshold: Cell<usize>,
    _phantom: PhantomData<(H, P)>,
}

// SAFETY: GcInterpCtx is single-threaded (uses Cell/RefCell). The
// extra_roots raw pointers are only dereferenced during collect()
// which runs on the same thread.
unsafe impl<H: ObjHeader, P: PtrPolicy> Send for GcInterpCtx<H, P> {}

impl<H: ObjHeader, P: PtrPolicy> GcInterpCtx<H, P> {
    /// Build a new GC-backed context.
    ///
    /// The caller must have already appended the continuation types to
    /// `type_table` via `ContinuationTypes::register_into` and passed
    /// the returned `cont_types` here alongside the augmented table.
    /// The `heap` should have been constructed with header type `H`
    /// matching what the interpreter uses.
    pub fn new(
        heap: SemiSpace,
        type_table: Vec<TypeInfo>,
        cont_types: ContinuationTypes,
    ) -> Self {
        GcInterpCtx {
            heap: RefCell::new(heap),
            type_table,
            cont_types,
            root_frames: RefCell::new(Vec::new()),
            extra_roots: RefCell::new(Vec::new()),
            alloc_count: Cell::new(0),
            // Default: disable auto-gc. Tests and real programs can
            // call `set_gc_threshold` to turn it on.
            gc_threshold: Cell::new(usize::MAX),
            _phantom: PhantomData,
        }
    }

    /// Register an additional root source that the GC should scan
    /// during collection. Used by the JIT engine to register its
    /// `JitFrameSliceRuntime` (which holds Vec-backed continuation
    /// snapshots with heap-pointer slots) alongside the interpreter's
    /// root frames.
    ///
    /// # Safety
    /// The caller must ensure the pointed-to `RootSource` outlives
    /// every `collect()` call. Typically the source is a field on the
    /// `ExecutionEngine` which outlives the `GcInterpCtx`.
    pub unsafe fn register_extra_roots(&self, source: *const dyn RootSource) {
        self.extra_roots.borrow_mut().push(source);
    }

    /// Set the auto-gc threshold: after this many `alloc` calls, the
    /// next `needs_gc()` returns `true`, which the interpreter checks
    /// at instruction boundaries and uses to drive a `collect()`.
    /// Pass `usize::MAX` to disable auto-gc.
    pub fn set_gc_threshold(&self, threshold: usize) {
        self.gc_threshold.set(threshold);
    }

    /// Number of allocations since the last collection.
    pub fn alloc_count_since_gc(&self) -> usize {
        self.alloc_count.get()
    }

    /// Whether an auto-collection is due. Called by the interpreter's
    /// `needs_gc()` at instruction boundaries — the only safe points
    /// for collection.
    pub fn should_auto_collect(&self) -> bool {
        self.alloc_count.get() >= self.gc_threshold.get()
    }

    pub fn cont_types(&self) -> &ContinuationTypes {
        &self.cont_types
    }

    pub fn type_table(&self) -> &[TypeInfo] {
        &self.type_table
    }

    // ── Inherent root-frame operations ──────────────────────────────
    //
    // These mirror the `InterpRootManager` trait methods but avoid the
    // `L / Roots / Transport` generic parameters, so tests and other
    // direct callers can use them without disambiguation gymnastics.
    // The trait impl below just forwards to these.

    pub fn push_root_frame(&self, gc_slot_count: usize) -> usize {
        let mut frames = self.root_frames.borrow_mut();
        let handle = frames.len();
        frames.push(vec![Cell::new(0); gc_slot_count]);
        handle
    }

    pub fn pop_root_frame(&self) {
        let mut frames = self.root_frames.borrow_mut();
        frames.pop().expect("pop_root_frame on empty stack");
    }

    pub fn set_root_slot(&self, frame: usize, slot: usize, value: u64) {
        let frames = self.root_frames.borrow();
        frames[frame][slot].set(value);
    }

    pub fn get_root_slot(&self, frame: usize, slot: usize) -> u64 {
        let frames = self.root_frames.borrow();
        frames[frame][slot].get()
    }

    pub fn clear_root_frame(&self, frame: usize) {
        let frames = self.root_frames.borrow();
        for cell in &frames[frame] {
            cell.set(0);
        }
    }

    /// Read the number of collections performed so far. Useful for tests
    /// that want to assert that GC actually ran.
    pub fn collection_count(&self) -> usize {
        self.heap.borrow().collections()
    }

    /// Snapshot of `from-space` byte usage.
    pub fn from_used(&self) -> usize {
        self.heap.borrow().from_used()
    }

    /// Run a garbage collection explicitly. The root frames and all
    /// FrameSlice handles / GcPtr values held in them are visited;
    /// the GC traces through captured continuations via standard
    /// `scan_object` rules.
    pub fn collect(&self) {
        let frames_guard = self.root_frames.borrow();
        let src = AllFramesRootSource { frames: &*frames_guard };

        // Build a root-source list: interpreter frame roots plus
        // any extra root sources registered by the embedding (e.g.,
        // the JIT's continuation store).
        let extras_guard = self.extra_roots.borrow();
        let mut sources: Vec<&dyn RootSource> = vec![&src];
        for &ptr in extras_guard.iter() {
            // SAFETY: the caller of `register_extra_roots` guaranteed
            // the pointer remains valid for all `collect()` calls.
            sources.push(unsafe { &*ptr });
        }

        unsafe {
            self.heap
                .borrow_mut()
                .collect::<P>(&self.type_table, &mut sources);
        }
        // Reset the alloc counter: the next threshold window starts
        // from here.
        self.alloc_count.set(0);
    }
}

impl<H: ObjHeader, P: PtrPolicy> Alloc for GcInterpCtx<H, P> {
    fn alloc(&self, info: &TypeInfo, varlen_len: usize) -> *mut u8 {
        // Bump the alloc counter. This does NOT trigger a collection —
        // `should_auto_collect` is checked at instruction boundaries
        // by the interpreter, never mid-alloc. That's critical because
        // `capture_continuation` makes two allocations in sequence
        // while holding a raw pointer to the first — a GC mid-sequence
        // would turn that pointer stale.
        self.alloc_count.set(self.alloc_count.get() + 1);
        self.heap.borrow().alloc(info, varlen_len)
    }
}

impl<H: ObjHeader, P: PtrPolicy> ContinuationContext for GcInterpCtx<H, P> {
    fn capture(&self, builder: &CapturedStackBuilder) -> Option<u64> {
        capture_continuation::<H, Self, P>(self, &self.cont_types, builder)
    }
    fn read<'h>(&'h self, handle: u64) -> Option<ContinuationView<'h>> {
        read_continuation::<Self, P>(self, &self.cont_types, handle)
    }
}

impl<H, P, L, Roots, Transport> InterpRootManager<L, Roots, Transport> for GcInterpCtx<H, P>
where
    H: ObjHeader,
    P: PtrPolicy,
    L: ValueLayout,
    Roots: RootStrategy<L>,
    Transport: RootTransport<L, Roots>,
{
    fn push_frame(&self, gc_slot_count: usize) -> usize {
        self.push_root_frame(gc_slot_count)
    }
    fn pop_frame(&self) {
        self.pop_root_frame()
    }
    fn set_root(&self, frame: usize, slot: usize, value: u64) {
        self.set_root_slot(frame, slot, value)
    }
    fn get_root(&self, frame: usize, slot: usize) -> u64 {
        self.get_root_slot(frame, slot)
    }
    fn clear_frame(&self, frame: usize) {
        self.clear_root_frame(frame)
    }
    fn collect(&self) {
        GcInterpCtx::collect(self);
    }
    fn should_collect(&self) -> bool {
        self.should_auto_collect()
    }
    fn register_root_source_raw(&self, source: *const dyn RootSource) {
        // SAFETY: delegated from the caller who guarantees the source
        // outlives all collect() calls.
        unsafe { self.register_extra_roots(source) }
    }
    fn root_precision(&self) -> RootPrecision {
        RootPrecision::PreciseSlots
    }
}

/// `RootSource` impl that visits every cell in every frame of a stack
/// of root frames. Used internally by `collect()` to feed the GC.
struct AllFramesRootSource<'a> {
    frames: &'a [Vec<Cell<u64>>],
}

impl<'a> RootSource for AllFramesRootSource<'a> {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for frame in self.frames {
            for cell in frame {
                visitor(cell.as_ptr());
            }
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use dynexec::{
        capture_continuation, read_continuation, BuilderFrame, FrameResume,
        CapturedStackBuilder,
    };
    use dynobj::{init_header, write_varlen_count, Compact, ObjHeader};

    /// Trivial LowBit<3> pointer policy: tag 0 = pointer, tag 1 = fixnum.
    struct TestPolicy;
    impl PtrPolicy for TestPolicy {
        fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
            if bits == 0 {
                return None;
            }
            if bits & 0b111 == 0 {
                Some(bits as *mut u8)
            } else {
                None
            }
        }
        fn encode_ptr(ptr: *mut u8) -> u64 {
            debug_assert_eq!((ptr as u64) & 0b111, 0);
            ptr as u64
        }
    }

    fn build_ctx() -> (GcInterpCtx<Compact, TestPolicy>, TypeInfo) {
        // User "byte buffer" type for test data.
        let user_type_id: u16 = 0;
        let user_type = TypeInfo::for_header(Compact::SIZE)
            .with_varlen_bytes(0)
            .with_type_id(user_type_id);

        let mut type_table: Vec<TypeInfo> = vec![user_type];
        let cont_types = ContinuationTypes::register_into::<Compact>(&mut type_table);
        let heap = SemiSpace::new::<Compact>(64 * 1024);
        let ctx = GcInterpCtx::<Compact, TestPolicy>::new(heap, type_table, cont_types);
        (ctx, user_type)
    }

    #[test]
    fn push_pop_set_get_root_frames() {
        let (ctx, _) = build_ctx();
        let f0 = ctx.push_root_frame(3);
        assert_eq!(f0, 0);
        ctx.set_root_slot(f0, 1, 0xdeadbeef);
        assert_eq!(ctx.get_root_slot(f0, 1), 0xdeadbeef);
        let f1 = ctx.push_root_frame(2);
        assert_eq!(f1, 1);
        ctx.pop_root_frame();
        ctx.pop_root_frame();
    }

    /// End-to-end: allocate a user object, capture a continuation
    /// referencing it, register the handle as a root, run GC via the
    /// root manager, verify the captured pointer is forwarded.
    ///
    /// This is the same property as the gc_forwards_pointers test in
    /// `dynexec::cont_heap::tests`, but driven through `GcInterpCtx`
    /// end-to-end instead of calling SemiSpace directly. If this
    /// passes, Phase 2 is working.
    #[test]
    fn gc_via_root_manager_forwards_captured_pointer() {
        let (ctx, user_type) = build_ctx();

        // Allocate a user object with some magic bytes.
        const MAGIC: [u8; 5] = [1, 2, 3, 4, 5];
        let user_ptr = ctx.alloc(&user_type, MAGIC.len());
        assert!(!user_ptr.is_null());
        unsafe {
            init_header::<Compact>(user_ptr, user_type.type_id);
            write_varlen_count(user_ptr, &user_type, MAGIC.len());
            let base = user_type.varlen_count_offset() + 8;
            core::ptr::copy_nonoverlapping(MAGIC.as_ptr(), user_ptr.add(base), MAGIC.len());
        }
        let user_tagged = TestPolicy::encode_ptr(user_ptr);

        // Capture a single-frame continuation whose value slot holds
        // the tagged user pointer.
        let builder = CapturedStackBuilder {
            prompt_id: 3,
            frames: vec![BuilderFrame {
                func_idx: 0,
                block_idx: 0,
                inst_idx: 0,
                values: vec![user_tagged],
                active_prompts: vec![3],
                root_indices: vec![0],
                resume_arg_slot: None,
                caller_resume: FrameResume::TopLevel,
            }],
        };
        let handle_before = capture_continuation::<Compact, _, TestPolicy>(
            &ctx,
            ctx.cont_types(),
            &builder,
        )
        .expect("capture OOM");

        // Store the handle in a root frame.
        let handle_frame = ctx.push_root_frame(1);
        ctx.set_root_slot(handle_frame, 0, handle_before);

        // Run collection via the inherent method (the trait impl
        // delegates to this same function).
        let collections_before = ctx.collection_count();
        ctx.collect();
        assert_eq!(ctx.collection_count(), collections_before + 1);

        // Read the (possibly forwarded) handle back out.
        let handle_after = ctx.get_root_slot(handle_frame, 0);
        assert_ne!(
            handle_after, handle_before,
            "handle should have been forwarded"
        );

        // Decode and read the captured continuation.
        let view = read_continuation::<_, TestPolicy>(&ctx, ctx.cont_types(), handle_after)
            .expect("read after GC failed");
        let f = view.frame(0);
        let captured_user_after = f.values[0];
        assert_ne!(
            captured_user_after, user_tagged,
            "captured user pointer should have been forwarded"
        );

        // Decode the new user pointer and verify the magic bytes.
        let new_user = TestPolicy::try_decode_ptr(captured_user_after).unwrap();
        let base = user_type.varlen_count_offset() + 8;
        let bytes =
            unsafe { core::slice::from_raw_parts(new_user.add(base) as *const u8, MAGIC.len()) };
        assert_eq!(bytes, &MAGIC);
    }

    /// Sanity check: straight-line alloc/write/read externs work
    /// through the reference interpreter.
    #[test]
    fn externs_straight_line_alloc_write_read() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicPtr, Ordering};
        use dynvalue::LowBit;
        use crate::builder::ModuleBuilder;
        use crate::interp::{ExternCallResult, InterpResult, ModuleInterpreter};
        use crate::types::{Signature, Type};

        let user_type_id: u16 = 0;
        let user_type = TypeInfo::for_header(Compact::SIZE)
            .with_varlen_bytes(0)
            .with_type_id(user_type_id);
        let mut type_table: Vec<TypeInfo> = vec![user_type];
        let cont_types =
            ContinuationTypes::register_into::<Compact>(&mut type_table);
        let heap = SemiSpace::new::<Compact>(64 * 1024);
        let ctx: GcInterpCtx<Compact, TestPolicy> =
            GcInterpCtx::new(heap, type_table, cont_types);

        let mut mb = ModuleBuilder::new();
        let f_alloc = mb.declare_extern(
            "alloc_bytes",
            Signature { params: vec![Type::I64], ret: Some(Type::GcPtr) },
        );
        let f_write = mb.declare_extern(
            "write_byte",
            Signature { params: vec![Type::GcPtr, Type::I64, Type::I64], ret: None },
        );
        let f_read = mb.declare_extern(
            "read_byte",
            Signature { params: vec![Type::GcPtr, Type::I64], ret: Some(Type::I64) },
        );
        let f_main = mb.declare_func("main", &[], Some(Type::I64));

        let mut fb = mb.define_func(f_main);
        let _entry = fb.entry_block();
        let five = fb.iconst(Type::I64, 5);
        let p = fb.call(f_alloc, &[five]).unwrap();
        let zero = fb.iconst(Type::I64, 0);
        let magic = fb.iconst(Type::I64, 42);
        fb.call(f_write, &[p, zero, magic]);
        let byte = fb.call(f_read, &[p, zero]).unwrap();
        fb.ret(byte);
        mb.finish_func(f_main, fb);
        let module = mb.build();

        let ctx_ptr: *const GcInterpCtx<Compact, TestPolicy> = &ctx;
        let user_type_for_closures = user_type;
        let ctx_ptr: Arc<AtomicPtr<GcInterpCtx<Compact, TestPolicy>>> =
            Arc::new(AtomicPtr::new(ctx_ptr as *mut _));

        let mut interp =
            ModuleInterpreter::<LowBit<3>, _>::new(&module, &ctx);
        interp.set_cont_ctx(&ctx);

        let ctx_a = ctx_ptr.clone();
        interp.bind(f_alloc, move |args| {
            let len = args[0] as usize;
            let ctx = unsafe { &*ctx_a.load(Ordering::SeqCst) };
            let raw = ctx.alloc(&user_type_for_closures, len);
            assert!(!raw.is_null());
            unsafe {
                dynobj::init_header::<Compact>(raw, user_type_for_closures.type_id);
                dynobj::write_varlen_count(raw, &user_type_for_closures, len);
            }
            ExternCallResult::Value(Some(TestPolicy::encode_ptr(raw)))
        });
        let ut_write = user_type;
        interp.bind(f_write, move |args| {
            let ptr = args[0] as *mut u8;
            let idx = args[1] as usize;
            let byte = args[2] as u8;
            let base = ut_write.varlen_count_offset() + 8;
            unsafe { *ptr.add(base + idx) = byte; }
            ExternCallResult::Value(None)
        });
        let ut_read = user_type;
        interp.bind(f_read, move |args| {
            let ptr = args[0] as *mut u8;
            assert!(!ptr.is_null(), "read_byte null");
            let idx = args[1] as usize;
            let base = ut_read.varlen_count_offset() + 8;
            let byte = unsafe { *ptr.add(base + idx) };
            ExternCallResult::Value(Some(byte as u64))
        });

        match interp.run(f_main, &[]).unwrap() {
            InterpResult::Value(v) => assert_eq!(v, 42),
            other => panic!("{:?}", other),
        }
    }

    /// Auto-GC: set a low threshold on `GcInterpCtx`, capture enough
    /// continuations in a loop to cross it, and verify the interpreter
    /// triggers a collection on its own without any explicit
    /// `force_gc` call. Uses contlang-style shift/reset programs so
    /// we exercise the whole dispatch path.
    #[test]
    fn auto_gc_triggers_from_allocation_threshold() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicPtr, Ordering};
        use dynvalue::LowBit;
        use crate::builder::ModuleBuilder;
        use crate::interp::{ExternCallResult, InterpResult, ModuleInterpreter};
        use crate::types::{Signature, Type};

        // Byte-buffer user type.
        let user_type_id: u16 = 0;
        let user_type = TypeInfo::for_header(Compact::SIZE)
            .with_varlen_bytes(0)
            .with_type_id(user_type_id);
        let mut type_table: Vec<TypeInfo> = vec![user_type];
        let cont_types =
            ContinuationTypes::register_into::<Compact>(&mut type_table);
        let heap = SemiSpace::new::<Compact>(64 * 1024);
        let ctx: GcInterpCtx<Compact, TestPolicy> =
            GcInterpCtx::new(heap, type_table, cont_types);

        // Low threshold: auto-gc every 10 allocations.
        ctx.set_gc_threshold(10);

        // Build a module with:
        //   extern alloc_bytes(n) -> GcPtr   (allocates, discards result)
        //   fn main():
        //     loop { alloc_bytes(8); alloc_bytes(8); ... }  — 100 times
        //     ret 0
        // Each loop iteration runs two allocs. With a threshold of 10,
        // auto-gc should fire several times before we finish.
        let mut mb = ModuleBuilder::new();
        let f_alloc = mb.declare_extern(
            "alloc_bytes",
            Signature { params: vec![Type::I64], ret: Some(Type::GcPtr) },
        );
        let f_main = mb.declare_func("main", &[], Some(Type::I64));

        let mut fb = mb.define_func(f_main);
        // Straight-line 40 calls to alloc_bytes, then return.
        let n = fb.iconst(Type::I64, 8);
        for _ in 0..40 {
            let _p = fb.call(f_alloc, &[n]).unwrap();
        }
        let zero = fb.iconst(Type::I64, 0);
        fb.ret(zero);
        mb.finish_func(f_main, fb);
        let module = mb.build();

        // Bind alloc_bytes.
        let user_type_for_closures = user_type;
        let ctx_ptr: *const GcInterpCtx<Compact, TestPolicy> = &ctx;
        let ctx_ptr: Arc<AtomicPtr<GcInterpCtx<Compact, TestPolicy>>> =
            Arc::new(AtomicPtr::new(ctx_ptr as *mut _));

        let mut interp =
            ModuleInterpreter::<LowBit<3>, _>::new(&module, &ctx);
        interp.set_cont_ctx(&ctx);

        let ctx_a = ctx_ptr.clone();
        interp.bind(f_alloc, move |args| {
            let len = args[0] as usize;
            let ctx = unsafe { &*ctx_a.load(Ordering::SeqCst) };
            let raw = ctx.alloc(&user_type_for_closures, len);
            assert!(!raw.is_null());
            unsafe {
                dynobj::init_header::<Compact>(raw, user_type_for_closures.type_id);
                dynobj::write_varlen_count(raw, &user_type_for_closures, len);
            }
            ExternCallResult::Value(Some(TestPolicy::encode_ptr(raw)))
        });

        let collections_before = ctx.collection_count();
        let result = interp.run(f_main, &[]).expect("interp error");
        assert!(matches!(result, InterpResult::Value(0)));

        // With threshold 10 and 40 allocations in straight-line main,
        // at least 3 collections should have triggered automatically.
        // (The interpreter polls `needs_gc` at outer-loop iterations,
        // which happen per FrameAction — i.e., between extern calls.)
        let collected = ctx.collection_count() - collections_before;
        assert!(
            collected >= 3,
            "expected auto-gc to fire at least 3 times; got {}",
            collected
        );
    }

    /// **Phase 4**: end-to-end GC-during-dormant-continuation through the
    /// full interpreter pipeline. Hand-built IR program:
    ///
    /// ```text
    ///   fn main() -> i64 {
    ///     push_prompt P
    ///     p = alloc_bytes(5)             // GcPtr
    ///     write_byte(p, 0, 42)
    ///     capture_slice P -> shift_handler(k), captured_resume_entry(v)
    ///
    ///   shift_handler(k: FrameSlice):
    ///     force_gc()                     // <-- GC runs with k and p both rooted
    ///     resume_slice k([0]) -> shift_handler_cont(r)
    ///
    ///   shift_handler_cont(r: i64):
    ///     ret r                          // result lands here from FromResume
    ///
    ///   captured_resume_entry(v: i64):
    ///     // p here is the CAPTURED p, forwarded during force_gc
    ///     byte = read_byte(p, 0)
    ///     abort_to_prompt P, [byte]
    ///
    ///   reset_handler(aborted: i64):
    ///     ret aborted
    ///   }
    /// ```
    ///
    /// This verifies that during `force_gc`, the GC:
    ///   1. Forwards `p` in main's live frame (a rooted GcPtr slot).
    ///   2. Forwards the ContObj handle `k` in main's live frame.
    ///   3. Traces INTO the ContObj and forwards the captured copy of
    ///      `p` in its varlen Values tail.
    ///
    /// Then on resume, `splice_from_view` copies the forwarded captured
    /// `p` into the fresh live frame, so `read_byte` at the resumer's
    /// new PC follows a valid pointer and reads the 42 byte.
    ///
    /// If any step is wrong (GC doesn't trace the ContObj, splice
    /// copies stale bytes, roots aren't properly synced), the test
    /// either panics in the GC or reads garbage.
    #[test]
    fn end_to_end_gc_during_dormant_continuation() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicPtr, Ordering};

        use dynvalue::LowBit;

        use crate::builder::ModuleBuilder;
        use crate::interp::{ExternCallResult, InterpResult, ModuleInterpreter};
        use crate::ir::{BlockId, Module, PromptId};
        use crate::types::{Signature, Type};

        // ── Set up the user byte-array type + heap ──────────────────
        let user_type_id: u16 = 0;
        let user_type = TypeInfo::for_header(Compact::SIZE)
            .with_varlen_bytes(0)
            .with_type_id(user_type_id);
        let mut type_table: Vec<TypeInfo> = vec![user_type];
        let cont_types =
            ContinuationTypes::register_into::<Compact>(&mut type_table);
        let heap = SemiSpace::new::<Compact>(64 * 1024);
        let ctx: GcInterpCtx<Compact, TestPolicy> =
            GcInterpCtx::new(heap, type_table, cont_types);
        // Aggressive auto-gc: trigger on every single alloc. Exercises
        // the "GC mid-capture is suppressed" invariant — if the GC
        // fired between ContMeta and ContObj allocations inside
        // `capture_continuation`, the raw `meta_ptr` Rust local would
        // turn stale and we'd write a dangling pointer into the
        // ContObj's field 0. The safe-point discipline is: alloc
        // bumps the counter but NEVER triggers; collection only fires
        // at instruction boundaries (where no raw heap pointers are
        // held in Rust locals).
        ctx.set_gc_threshold(1);

        // ── Build the IR module ─────────────────────────────────────
        let mut mb = ModuleBuilder::new();
        let f_alloc = mb.declare_extern(
            "alloc_bytes",
            Signature { params: vec![Type::I64], ret: Some(Type::GcPtr) },
        );
        let f_write = mb.declare_extern(
            "write_byte",
            Signature { params: vec![Type::GcPtr, Type::I64, Type::I64], ret: None },
        );
        let f_read = mb.declare_extern(
            "read_byte",
            Signature { params: vec![Type::GcPtr, Type::I64], ret: Some(Type::I64) },
        );
        let f_gc = mb.declare_extern(
            "force_gc",
            Signature { params: vec![], ret: None },
        );
        let f_main = mb.declare_func("main", &[], Some(Type::I64));

        let mut fb = mb.define_func(f_main);
        let entry_bb = fb.entry_block();
        let prompt: PromptId = fb.create_prompt();

        // Create all target blocks up front.
        let shift_handler_bb: BlockId = fb.create_block(&[Type::FrameSlice]);
        let shift_handler_cont_bb: BlockId = fb.create_block(&[Type::I64]);
        let captured_resume_entry_bb: BlockId = fb.create_block(&[Type::I64]);
        let reset_handler_bb: BlockId = fb.create_block(&[Type::I64]);

        // --- entry --- (already the current block)
        let _ = entry_bb;
        fb.push_prompt(prompt, reset_handler_bb);
        let five = fb.iconst(Type::I64, 5);
        let p = fb
            .call(f_alloc, &[five])
            .expect("alloc_bytes returns a value");
        let zero = fb.iconst(Type::I64, 0);
        let magic = fb.iconst(Type::I64, 42);
        fb.call(f_write, &[p, zero, magic]);
        fb.capture_slice_term(prompt, shift_handler_bb, captured_resume_entry_bb);

        // --- shift_handler(k) ---
        fb.switch_to_block(shift_handler_bb);
        let k = fb.block_param(shift_handler_bb, 0);
        fb.call(f_gc, &[]);
        let zero2 = fb.iconst(Type::I64, 0);
        fb.resume_slice(k, &[zero2], shift_handler_cont_bb, &[]);

        // --- shift_handler_cont(r) ---
        fb.switch_to_block(shift_handler_cont_bb);
        let r = fb.block_param(shift_handler_cont_bb, 0);
        fb.ret(r);

        // --- captured_resume_entry(v) ---
        fb.switch_to_block(captured_resume_entry_bb);
        let _v = fb.block_param(captured_resume_entry_bb, 0);
        let zero3 = fb.iconst(Type::I64, 0);
        let byte = fb
            .call(f_read, &[p, zero3])
            .expect("read_byte returns a value");
        fb.abort_to_prompt(prompt, &[byte]);

        // --- reset_handler(aborted) ---
        fb.switch_to_block(reset_handler_bb);
        let aborted = fb.block_param(reset_handler_bb, 0);
        fb.ret(aborted);

        mb.finish_func(f_main, fb);
        let module: Module = mb.build();

        // ── Bind the externs ────────────────────────────────────────
        // Each closure captures a shared pointer to the ctx via an
        // `AtomicPtr` held in an `Arc`, because the `bind` closures
        // must be `'static` from the borrow checker's perspective and
        // the interpreter's borrow on `&ctx` overlaps with them.
        let ctx_ptr: *const GcInterpCtx<Compact, TestPolicy> = &ctx;
        let user_type_for_closures = user_type;
        let ctx_ptr: Arc<AtomicPtr<GcInterpCtx<Compact, TestPolicy>>> =
            Arc::new(AtomicPtr::new(ctx_ptr as *mut _));

        let mut interp =
            ModuleInterpreter::<LowBit<3>, _>::new(&module, &ctx);
        interp.set_cont_ctx(&ctx);

        let ctx_for_alloc = ctx_ptr.clone();
        interp.bind(f_alloc, move |args| {
            let len = args[0] as usize;
            let ctx = unsafe { &*ctx_for_alloc.load(Ordering::SeqCst) };
            let raw = ctx.alloc(&user_type_for_closures, len);
            assert!(!raw.is_null(), "alloc_bytes: OOM");
            unsafe {
                dynobj::init_header::<Compact>(raw, user_type_for_closures.type_id);
                dynobj::write_varlen_count(raw, &user_type_for_closures, len);
                // Zero the bytes explicitly (alloc is expected to
                // return zeroed memory, but be explicit).
                let base = user_type_for_closures.varlen_count_offset() + 8;
                core::ptr::write_bytes(raw.add(base), 0, len);
            }
            ExternCallResult::Value(Some(TestPolicy::encode_ptr(raw)))
        });

        let ut_write = user_type;
        interp.bind(f_write, move |args| {
            let ptr = args[0] as *mut u8;
            let idx = args[1] as usize;
            let byte = args[2] as u8;
            assert!(!ptr.is_null());
            let base = ut_write.varlen_count_offset() + 8;
            unsafe {
                *ptr.add(base + idx) = byte;
            }
            ExternCallResult::Value(None)
        });

        let ut_read = user_type;
        interp.bind(f_read, move |args| {
            let ptr = args[0] as *mut u8;
            let idx = args[1] as usize;
            assert!(!ptr.is_null(), "read_byte called with null pointer");
            let base = ut_read.varlen_count_offset() + 8;
            let byte = unsafe { *ptr.add(base + idx) };
            ExternCallResult::Value(Some(byte as u64))
        });

        let ctx_for_gc = ctx_ptr.clone();
        interp.bind(f_gc, move |_args| {
            let ctx = unsafe { &*ctx_for_gc.load(Ordering::SeqCst) };
            ctx.collect();
            ExternCallResult::Value(None)
        });

        // ── Run and verify ──────────────────────────────────────────
        let collections_before = ctx.collection_count();
        let result = interp.run(f_main, &[]).expect("interp error");
        let value = match result {
            InterpResult::Value(v) => v,
            other => panic!("expected Value, got {:?}", other),
        };
        assert_eq!(value, 42, "main should return the magic byte");
        assert!(
            ctx.collection_count() > collections_before,
            "force_gc should have triggered at least one collection"
        );
    }
}
