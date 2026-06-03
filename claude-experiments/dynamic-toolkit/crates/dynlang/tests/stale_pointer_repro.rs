//! Diagnostic reproducer for the suspected stale-pointer bug.
//!
//! Hypothesis: when an extern fn called from JIT internally triggers a
//! moving GC (via `gc_alloc_thunk` overflowing the nursery), the
//! triggering thread's own JIT frame chain is NOT walked. The
//! `LiteralPool` IS walked (it's registered as a permanent extra
//! root), so its slots get rewritten to forwarded pointers. But the
//! JIT frame's spill slots — which hold copies of those same pointers,
//! materialized before the call — are left stale.
//!
//! Post-call, the JIT reloads the value from its spill slot (stale
//! bits → from-space address) and the field load reads garbage or
//! segfaults.
//!
//! The test:
//!   1. JIT-compiles a function `f` that loads a `gc_literal(0)`, calls
//!      an extern that burns the nursery, then reads a field of the
//!      loaded value.
//!   2. Pre-allocates a `Pair` cell with field 0 set to `SENTINEL`,
//!      pushes its NanBox into pool[0].
//!   3. Runs `f()`. If the bug is real, the spill-slot reload sees
//!      stale bits and the test fails / crashes. If fixed, returns
//!      `SENTINEL`.

use std::sync::Arc;

use dynir::builder::ModuleBuilder;
use dynir::ir::LiteralRef;
use dynir::types::{Signature, Type};
use dynlang::gc::{gc_alloc_thunk, DynGcRuntime};
use dynlang::{GcConfig, GcPolicy, NanBoxTags};
use dynlower::{Arm64Backend, CallMode, JitModule, JitOutcome};
use dynobj::{Compact, ObjHeader, RootSource, TypeInfo};
use dynruntime::active_jit_safepoint_handler;
use dynvalue::{NanBox, TagScheme};

/// Pair: Compact header + 2 value fields.
const PAIR_TYPE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(2);

/// Sentinel stored in pair.field[0] before the call.
const SENTINEL: i64 = 0x4242_4242_4242_4242u64 as i64;

/// PAIR's type_id once registered. Set at test setup.
static mut PAIR_TYPE_ID: u64 = u64::MAX;

/// Snapshot of pool[0] taken inside `burn_nursery_observe` AFTER the
/// GC has fired. The test asserts this differs from the pre-call
/// value (proving the pool slot WAS rewritten) — pinning the bug
/// specifically to spill-slot non-walking, not pool non-walking.
static mut POOL_AFTER_GC: u64 = 0;
/// The pool's base address, captured at setup so the extern can
/// read pool[0] directly.
static mut POOL_BASE: *const u64 = std::ptr::null();

/// Extern called from the JIT. Allocates a flood of pairs through
/// `gc_alloc_thunk` so the nursery fills, triggering
/// `mutator_triggered_gc` from inside the extern — no JIT safepoint
/// is involved in the trigger.
extern "C" fn burn_nursery() -> u64 {
    let tid = unsafe { PAIR_TYPE_ID };
    for _ in 0..512 {
        let _ = gc_alloc_thunk(tid, 0);
    }
    // Snapshot pool[0] AFTER all the allocs+GCs. If the alloc-triggered
    // GC walked the literal pool, this differs from the pre-call value.
    unsafe {
        POOL_AFTER_GC = *POOL_BASE;
    }
    0
}

#[test]
fn pool_value_survives_alloc_triggered_gc() {
    // Small nursery so burn_nursery() overflows it reliably.
    let gc_config = GcConfig::Generational {
        heap_size: 256 * 1024,
        nursery_size: Some(8 * 1024),
    };
    let tags = NanBoxTags {
        nil: 0,
        bool_tag: 1,
        ptr: 2,
    };
    let obj_types = vec![dynlang::ObjType {
        name: "Pair".into(),
        type_info: Box::leak(Box::new(PAIR_TYPE)),
        field_offsets: std::collections::HashMap::new(),
        varlen: dynobj::VarLenKind::None,
    }];
    let gc = Arc::new(DynGcRuntime::new(&gc_config, &tags, &obj_types));

    // Make the type_id available to the extern.
    let pair_type_id = gc.type_info(0).type_id as u64;
    unsafe {
        PAIR_TYPE_ID = pair_type_id;
    }

    // Install thread state so gc_alloc_thunk + NanBoxPolicy can find
    // the runtime.
    let _guard = gc.install_thread();

    // Build the module.
    let mut mb = ModuleBuilder::new();
    let f_burn = mb.declare_extern(
        "burn_nursery",
        Signature {
            params: vec![],
            ret: Some(Type::I64),
        },
    );
    let f_main = mb.declare_func("read_through_literal", &[], Some(Type::I64));
    {
        let mut fb = mb.define_func(f_main);
        let v = fb.gc_literal(LiteralRef::from_u32(0));
        let _ = fb.call(f_burn, &[]).unwrap();
        let raw = fb.payload(v);
        let off = fb.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        let addr = fb.add(raw, off);
        let loaded = fb.load(Type::I64, addr, 0);
        fb.ret(loaded);
        mb.finish_func(f_main, fb);
    }
    let module = mb.build();

    // Wire externs by FuncRef index.
    let mut externs: Vec<*const u8> = vec![std::ptr::null(); module.func_table.len()];
    externs[f_burn.index()] = burn_nursery as *const u8;

    // Compile via the public linear-scan path (what production uses).
    let mut jit = JitModule::compile_with_gc_linear_scan::<NanBox>(
        &module,
        &externs,
        active_jit_safepoint_handler,
    );

    // Pre-allocate the Pair we'll thread through pool[0].
    let pair_ptr = gc.alloc(0, 0);
    assert!(!pair_ptr.is_null());
    unsafe {
        let field0 = pair_ptr.add(PAIR_TYPE.value_field_offset(0)) as *mut i64;
        *field0 = SENTINEL;
    }
    let pair_nb = NanBox::encode_tagged(tags.ptr, pair_ptr as u64);
    let idx = jit.literal_pool().push(pair_nb);
    assert_eq!(idx, 0);
    unsafe {
        POOL_BASE = jit.literal_pool().base() as *const u64;
    }
    let pre_call_pool0 = jit.literal_pool().get(0);

    // Register the pool as a PERMANENT extra root so the
    // alloc-triggered GC path walks it (matches clojure-jvm /
    // microlisp setup).
    let pool_ptr: *const dyn RootSource = jit.literal_pool();
    unsafe {
        gc.register_extra_root_source(pool_ptr);
    }

    // Run f. The post-call field load must return SENTINEL.
    let outcome = gc.run_jit(&jit, f_main, &[], GcPolicy::EveryPoint);
    let result = match outcome {
        JitOutcome::Value(v) => v as i64,
        other => panic!("unexpected outcome: {:?}", other),
    };

    let post_call_pool0 = jit.literal_pool().get(0);
    let observed_inside = unsafe { POOL_AFTER_GC };
    eprintln!("pre_call pool[0]    = 0x{:x}", pre_call_pool0);
    eprintln!("observed inside ext = 0x{:x}", observed_inside);
    eprintln!("post_call pool[0]   = 0x{:x}", post_call_pool0);
    eprintln!("returned value      = 0x{:x}", result);

    // Did the GC actually walk the pool?
    assert_ne!(
        observed_inside, pre_call_pool0,
        "pool[0] was NOT rewritten by the alloc-triggered GC inside \
         burn_nursery — the bug is in pool walking, not spill walking"
    );

    assert_eq!(
        result, SENTINEL,
        "post-call load returned 0x{:x}, expected 0x{:x} — \
         the spill-slot reload saw a stale (pre-GC) pointer, \
         indicating the alloc-triggered GC did not walk the JIT \
         frame's stack-map roots",
        result, SENTINEL,
    );
}

/// Compile-time use of `JitModule` to silence unused-import warnings
/// if `compile_with_config_and_gc_linear_scan` ever moves.
#[allow(dead_code)]
fn _unused(_: JitModule) {}

/// Likewise.
#[allow(dead_code)]
const _: CallMode = CallMode::FastCall;
#[allow(dead_code)]
fn _backend() -> std::marker::PhantomData<Arm64Backend> {
    std::marker::PhantomData
}
