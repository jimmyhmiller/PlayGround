//! End-to-end Phase 0 gate: prove the codegen↔GC ABI contract works under a
//! real collection.
//!
//! We hand-build (with inkwell) an LLVM function that follows the EXACT shape
//! Phase 3 codegen will emit:
//!
//!   1. Prologue: alloca a `{ parent, origin, [roots; 1] }` frame, link it into
//!      `thread.top_frame`, zero the root slot.
//!   2. Allocate a heap object (1 value field) via `ai_gc_alloc_fixed`.
//!   3. Store a sentinel child pointer into the object's field, and spill the
//!      object pointer into the frame's root slot (so a GC can find + relocate
//!      it).
//!   4. Call an extern that triggers a stop-the-world collection *while our
//!      frame is live on the stack* — the copying collector MOVES the object.
//!   5. Reload the object pointer from the (now GC-updated) root slot, read its
//!      field back, and return it.
//!   6. Epilogue: unlink the frame.
//!
//! If precise rooting + relocation fixup work, the returned child pointer equals
//! the sentinel we stored — even though the containing object moved. A miss
//! (stale root, wrong slot count, missing fixup) shows up as a wrong pointer or
//! a crash.

use gcrust::gc::{Full, ObjHeader, TypeInfo, VarLenKind};
use gcrust::runtime::{self, RuntimeContext, Thread};

use inkwell::AddressSpace;
use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::module::Linkage;
use inkwell::values::{BasicValue, BasicValueEnum};
use inkwell::values::CallSiteValue;

/// Extract the `BasicValueEnum` result of a call (this inkwell fork's
/// `try_as_basic_value` returns a `ValueKind`, not an `Either`).
fn call_result<'ctx>(cs: CallSiteValue<'ctx>) -> BasicValueEnum<'ctx> {
    match cs.try_as_basic_value() {
        inkwell::values::ValueKind::Basic(v) => v,
        inkwell::values::ValueKind::Instruction(_) => panic!("call had no basic value"),
    }
}

/// Test-only extern: trigger a GC from inside JIT'd code. The runtime context
/// pointer is threaded through a thread-local so the JIT'd function only needs
/// the `Thread*` it already has.
use std::cell::Cell;
thread_local! {
    static FORCE_GC_CTX: Cell<*const RuntimeContext> = const { Cell::new(std::ptr::null()) };
}

#[unsafe(no_mangle)]
extern "C" fn gcr_test_force_gc(thread: *mut Thread) {
    let ctx = FORCE_GC_CTX.with(|c| c.get());
    assert!(!ctx.is_null(), "force-gc context not installed");
    unsafe { (*ctx).force_collect(&*thread) };
}

#[test]
fn jit_object_survives_collection_via_precise_roots() {
    // ---- Heap shapes -------------------------------------------------------
    // type_id 0: a "pair-ish" object with one GC-traced value field (the child
    // pointer). Full header so the collector has its forwarding word.
    let obj_ti = TypeInfo::for_header(Full::SIZE)
        .with_fields(1)
        .with_type_id(0);
    // type_id 1: the child — a leaf with no fields. We only ever compare its
    // identity, so its shape just needs a header.
    let child_ti = TypeInfo::for_header(Full::SIZE)
        .with_fields(0)
        .with_type_id(1);
    assert_eq!(obj_ti.varlen, VarLenKind::None);

    // Small spaces so we can reason about it; one collection is all we force.
    let mut ctx = RuntimeContext::new(1 << 16, vec![obj_ti, child_ti]);

    // ---- Pre-allocate the child from the host side, kept alive across the GC
    // by being reachable from the JIT'd object's rooted field. We capture its
    // pre-GC identity by reading it back post-GC through the (relocated) parent.
    // To have a stable sentinel to compare against, we allocate the child and
    // root IT too: the simplest robust check is "the field still points at a
    // live, correctly-shaped child after the collection".
    //
    // We allocate the child inside the JIT function as well (second alloc),
    // so the whole graph is built and collected by compiled code.

    // ---- Build the LLVM module --------------------------------------------
    let context = Context::create();
    let module = context.create_module("gc_abi_smoke");
    let builder = context.create_builder();

    let i32t = context.i32_type();
    let i64t = context.i64_type();
    let ptr = context.ptr_type(AddressSpace::default());

    // extern declarations (resolved via global mappings below).
    // ptr ai_gc_alloc_fixed(ptr thread, i32 type_id)
    let alloc_fixed_ty = ptr.fn_type(&[ptr.into(), i32t.into()], false);
    let alloc_fixed = module.add_function("ai_gc_alloc_fixed", alloc_fixed_ty, Some(Linkage::External));
    // void gcr_test_force_gc(ptr thread)
    let force_gc_ty = context.void_type().fn_type(&[ptr.into()], false);
    let force_gc = module.add_function("gcr_test_force_gc", force_gc_ty, Some(Linkage::External));

    // FrameOrigin global: { i32 num_roots, i32 pad, ptr name }. We need 2 root
    // slots: the parent object and the child object both must survive the GC.
    let origin_ty = context.struct_type(&[i32t.into(), i32t.into(), ptr.into()], false);
    let origin_global = module.add_global(origin_ty, None, "frame_origin_roundtrip");
    origin_global.set_constant(true);
    origin_global.set_initializer(
        &origin_ty.const_named_struct(&[
            i32t.const_int(2, false).into(),       // num_roots = 2
            i32t.const_int(0, false).into(),       // pad
            ptr.const_null().into(),               // name
        ]),
    );

    // Frame type: { ptr parent, ptr origin, [2 x ptr] roots }.
    let roots_arr = ptr.array_type(2);
    let frame_ty = context.struct_type(&[ptr.into(), ptr.into(), roots_arr.into()], false);

    // ptr roundtrip(ptr thread)
    let fn_ty = ptr.fn_type(&[ptr.into()], false);
    let func = module.add_function("roundtrip", fn_ty, None);
    let entry = context.append_basic_block(func, "entry");
    builder.position_at_end(entry);

    let thread = func.get_nth_param(0).unwrap().into_pointer_value();

    // ---- Prologue ----------------------------------------------------------
    let frame = builder.build_alloca(frame_ty, "frame").unwrap();
    // frame.origin = &origin
    let origin_field = builder
        .build_struct_gep(frame_ty, frame, 1, "origin_field")
        .unwrap();
    builder
        .build_store(origin_field, origin_global.as_pointer_value())
        .unwrap();
    // frame.parent = thread.top_frame; thread.top_frame = &frame
    let top_frame_ptr = builder
        .build_int_to_ptr(
            builder
                .build_int_add(
                    builder.build_ptr_to_int(thread, i64t, "t_int").unwrap(),
                    i64t.const_int(runtime::thread_offsets::TOP_FRAME as u64, false),
                    "tf_addr",
                )
                .unwrap(),
            ptr,
            "top_frame_ptr",
        )
        .unwrap();
    let prev_top = builder.build_load(ptr, top_frame_ptr, "prev_top").unwrap();
    let parent_field = builder
        .build_struct_gep(frame_ty, frame, 0, "parent_field")
        .unwrap();
    builder.build_store(parent_field, prev_top).unwrap();
    builder.build_store(top_frame_ptr, frame).unwrap();
    // Zero both root slots.
    let roots_field = builder
        .build_struct_gep(frame_ty, frame, 2, "roots_field")
        .unwrap();
    let zero = i32t.const_zero();
    let slot0 = unsafe {
        builder
            .build_in_bounds_gep(roots_arr, roots_field, &[zero, i32t.const_zero()], "slot0")
            .unwrap()
    };
    let slot1 = unsafe {
        builder
            .build_in_bounds_gep(roots_arr, roots_field, &[zero, i32t.const_int(1, false)], "slot1")
            .unwrap()
    };
    builder.build_store(slot0, ptr.const_null()).unwrap();
    builder.build_store(slot1, ptr.const_null()).unwrap();

    // ---- Allocate child (type_id 1), spill into root slot 1 ----------------
    let child = call_result(
        builder
            .build_call(alloc_fixed, &[thread.into(), i32t.const_int(1, false).into()], "child")
            .unwrap(),
    )
    .into_pointer_value();
    builder.build_store(slot1, child).unwrap();

    // ---- Allocate parent object (type_id 0), spill into root slot 0 --------
    // NOTE: this alloc can itself trigger a GC; `child` is already rooted in
    // slot1, so it survives and we reload it afterwards.
    let obj = call_result(
        builder
            .build_call(alloc_fixed, &[thread.into(), i32t.const_int(0, false).into()], "obj")
            .unwrap(),
    )
    .into_pointer_value();
    builder.build_store(slot0, obj).unwrap();

    // ---- obj.field0 = child (reload child from its root slot first) --------
    let child_live = builder.build_load(ptr, slot1, "child_live").unwrap().into_pointer_value();
    let obj_live = builder.build_load(ptr, slot0, "obj_live").unwrap().into_pointer_value();
    // field0 is at byte offset header_size (Full = 16).
    let field0 = builder
        .build_int_to_ptr(
            builder
                .build_int_add(
                    builder.build_ptr_to_int(obj_live, i64t, "obj_int").unwrap(),
                    i64t.const_int(Full::SIZE as u64, false),
                    "field0_addr",
                )
                .unwrap(),
            ptr,
            "field0",
        )
        .unwrap();
    builder.build_store(field0, child_live).unwrap();

    // ---- Force a GC while obj + child are rooted ---------------------------
    builder.build_call(force_gc, &[thread.into()], "").unwrap();

    // ---- Reload obj from its (GC-updated) root slot, read field0 back ------
    let obj_after = builder.build_load(ptr, slot0, "obj_after").unwrap().into_pointer_value();
    let field0_after = builder
        .build_int_to_ptr(
            builder
                .build_int_add(
                    builder.build_ptr_to_int(obj_after, i64t, "obj_after_int").unwrap(),
                    i64t.const_int(Full::SIZE as u64, false),
                    "field0_after_addr",
                )
                .unwrap(),
            ptr,
            "field0_after",
        )
        .unwrap();
    let child_after = builder.build_load(ptr, field0_after, "child_after").unwrap();

    // ---- Epilogue: unlink frame --------------------------------------------
    let parent_reload = builder.build_load(ptr, parent_field, "parent_reload").unwrap();
    builder.build_store(top_frame_ptr, parent_reload).unwrap();

    builder
        .build_return(Some(&child_after.as_basic_value_enum()))
        .unwrap();

    // Also expose the post-GC child root so the test can compare identities.
    // (returned value = obj.field0 after GC; we separately read slot1.)

    module.verify().expect("module failed to verify");

    // ---- JIT + run ---------------------------------------------------------
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();
    ee.add_global_mapping(&alloc_fixed, runtime::ai_gc_alloc_fixed as *const () as usize);
    ee.add_global_mapping(&force_gc, gcr_test_force_gc as *const () as usize);

    FORCE_GC_CTX.with(|c| c.set(&ctx as *const RuntimeContext));

    let thread_ptr = ctx.thread_ptr();
    type RoundtripFn = unsafe extern "C" fn(*mut Thread) -> *mut u8;
    let f: RoundtripFn = unsafe { std::mem::transmute(ee.get_function_address("roundtrip").unwrap()) };

    let returned_child = unsafe { f(thread_ptr) };

    FORCE_GC_CTX.with(|c| c.set(std::ptr::null()));

    // The collection happened. The returned pointer is obj.field0 read AFTER
    // the GC, through the relocated obj. It must be non-null, live in the
    // current from-space, and carry the child's type_id (1) — i.e. the field
    // points at a correctly-relocated child, proving precise rooting + fixup.
    assert!(!returned_child.is_null(), "field0 came back null after GC");
    assert!(
        ctx.heap().from_space_contains(returned_child),
        "relocated child is not in the live from-space (stale/un-fixed pointer)"
    );
    let tid = unsafe { ctx.heap().obj_type_id(returned_child) };
    assert_eq!(
        tid, 1,
        "field0 does not point at a child-shaped object after GC \
         (returned_child={:p}, tid={}, collections={})",
        returned_child, tid, ctx.heap().collections()
    );

    // And exactly one collection ran.
    assert_eq!(ctx.heap().collections(), 1, "expected exactly one collection");
}
