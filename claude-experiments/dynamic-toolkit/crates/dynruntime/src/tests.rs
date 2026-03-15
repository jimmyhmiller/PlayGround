use dynir::builder::{FunctionBuilder, ModuleBuilder};
use dynir::types::{Signature, Type};
use dynir::{ExternCallResult, FuncRef, InterpResult, ModuleInterpreter};
use dynobj::{Compact, ObjHeader, TypeInfo};
use dynvalue::{LowBit, NanBox, TagScheme};

use crate::framechain::FrameChainRootManager;
use crate::ptr_policy::{LowBitPtrPolicy, NanBoxPtrPolicy};
use crate::stackmap::MutatorRootManager;

// ─── Object layout ──────────────────────────────────────────────────

/// Simple object: Compact header + 1 GC-traced value field.
/// Total: 8 (header) + 8 (field) = 16 bytes.
static PAIR_TYPE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);

// ─── Helper: wrap a single Function into a Module ───────────────────

/// Build a module containing a single function plus an "alloc" extern.
/// Returns (module, entry FuncRef, alloc FuncRef).
fn single_func_module(
    name: &str,
    params: &[Type],
    ret: Option<Type>,
    build: impl FnOnce(&mut FunctionBuilder, FuncRef),
) -> (dynir::Module, FuncRef, FuncRef) {
    let mut mb = ModuleBuilder::new();
    let f_alloc = mb.declare_extern(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );
    let f_entry = mb.declare_func(name, params, ret);
    let mut fb = mb.define_func(f_entry);
    build(&mut fb, f_alloc);
    mb.finish_func(f_entry, fb);
    (mb.build(), f_entry, f_alloc)
}

// ─── Alloc + safepoint + load (LowBit) ──────────────────────────────

#[test]
fn alloc_store_safepoint_load_lowbit() {
    let (module, f_entry, f_alloc) =
        single_func_module("test_gc", &[], Some(Type::I64), |b, f_alloc| {
            let obj = b.call(f_alloc, &[]).unwrap();
            let magic = b.iconst(Type::I64, 42);
            let field_offset = PAIR_TYPE.value_field_offset(0) as i64;
            let offset_val = b.iconst(Type::I64, field_offset);
            let field_addr = b.add(obj, offset_val);
            b.store(magic, field_addr, 0);
            b.safepoint(&[obj]);
            let field_addr2 = b.add(obj, offset_val);
            let loaded = b.load(Type::I64, field_addr2, 0);
            b.ret(loaded);
        });

    let roots = MutatorRootManager::<LowBitPtrPolicy<3>>::new(4096);
    let mut interp = ModuleInterpreter::<LowBit<3>, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "allocation failed");
        ExternCallResult::Value(Some(ptr as u64))
    });

    let result = interp.run(f_entry, &[]).unwrap();
    assert_eq!(result, InterpResult::Value(42));
    assert_eq!(roots.collections(), 1);
}

// ─── Alloc + safepoint + load (NanBox) ──────────────────────────────

#[test]
fn alloc_store_safepoint_load_nanbox() {
    let (module, f_entry, f_alloc) =
        single_func_module("test_gc_nan", &[], Some(Type::I64), |b, f_alloc| {
            let obj = b.call(f_alloc, &[]).unwrap();
            let magic = b.iconst(Type::I64, 42);
            let field_offset = PAIR_TYPE.value_field_offset(0) as i64;
            let raw_ptr = b.payload(obj);
            let offset_val = b.iconst(Type::I64, field_offset);
            let field_addr = b.add(raw_ptr, offset_val);
            b.store(magic, field_addr, 0);
            b.safepoint(&[obj]);
            let raw_ptr2 = b.payload(obj);
            let field_addr2 = b.add(raw_ptr2, offset_val);
            let loaded = b.load(Type::I64, field_addr2, 0);
            b.ret(loaded);
        });

    let roots = FrameChainRootManager::<NanBoxPtrPolicy>::new(4096);
    let mut interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        ExternCallResult::Value(Some(NanBox::encode_tagged(0, ptr as u64)))
    });

    let result = interp.run(f_entry, &[]).unwrap();
    assert_eq!(result, InterpResult::Value(42));
    assert_eq!(roots.collections(), 1);
}

// ─── Multiple objects survive GC ─────────────────────────────────────

#[test]
fn multiple_objects_survive_gc_lowbit() {
    let (module, f_entry, f_alloc) =
        single_func_module("two_obj", &[], Some(Type::I64), |b, f_alloc| {
            let obj1 = b.call(f_alloc, &[]).unwrap();
            let obj2 = b.call(f_alloc, &[]).unwrap();
            let val1 = b.iconst(Type::I64, 111);
            let val2 = b.iconst(Type::I64, 222);
            let field_offset = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
            let addr1 = b.add(obj1, field_offset);
            b.store(val1, addr1, 0);
            let addr2 = b.add(obj2, field_offset);
            b.store(val2, addr2, 0);
            b.safepoint(&[obj1, obj2]);
            let addr1b = b.add(obj1, field_offset);
            let loaded1 = b.load(Type::I64, addr1b, 0);
            let addr2b = b.add(obj2, field_offset);
            let loaded2 = b.load(Type::I64, addr2b, 0);
            let sum = b.add(loaded1, loaded2);
            b.ret(sum);
        });

    let roots = MutatorRootManager::<LowBitPtrPolicy<3>>::new(4096);
    let mut interp = ModuleInterpreter::<LowBit<3>, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        ExternCallResult::Value(Some(ptr as u64))
    });

    let result = interp.run(f_entry, &[]).unwrap();
    assert_eq!(result, InterpResult::Value(333));
}

// ─── GC in loop ─────────────────────────────────────────────────────

#[test]
fn gc_in_loop_lowbit() {
    let mut mb = ModuleBuilder::new();
    let f_alloc = mb.declare_extern(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );
    let f_entry = mb.declare_func("gc_loop", &[Type::I64], Some(Type::I64));

    {
        let mut b = mb.define_func(f_entry);
        let entry = b.entry_block();
        let n = b.block_param(entry, 0);

        let loop_bb = b.create_block(&[Type::I64, Type::I64]); // (i, acc)
        let exit_bb = b.create_block(&[Type::I64]);

        let zero = b.iconst(Type::I64, 0);
        b.jump(loop_bb, &[zero, zero]);

        b.switch_to_block(loop_bb);
        let i = b.block_param(loop_bb, 0);
        let acc = b.block_param(loop_bb, 1);

        let obj = b.call(f_alloc, &[]).unwrap();
        let field_offset = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        let addr = b.add(obj, field_offset);
        b.store(i, addr, 0);
        b.safepoint(&[obj]);
        let addr2 = b.add(obj, field_offset);
        let loaded = b.load(Type::I64, addr2, 0);
        let new_acc = b.add(acc, loaded);

        let one = b.iconst(Type::I64, 1);
        let new_i = b.add(i, one);
        let cond = b.icmp(dynir::CmpOp::Slt, new_i, n);
        b.br_if(cond, loop_bb, &[new_i, new_acc], exit_bb, &[new_acc]);

        b.switch_to_block(exit_bb);
        let result = b.block_param(exit_bb, 0);
        b.ret(result);

        mb.finish_func(f_entry, b);
    }

    let module = mb.build();
    let roots = MutatorRootManager::<LowBitPtrPolicy<3>>::new(8192);
    let mut interp = ModuleInterpreter::<LowBit<3>, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        ExternCallResult::Value(Some(ptr as u64))
    });

    // Sum 0..5 = 0+1+2+3+4 = 10
    let result = interp.run(f_entry, &[5]).unwrap();
    assert_eq!(result, InterpResult::Value(10));
    assert!(
        roots.collections() >= 5,
        "should have collected at each iteration"
    );
}

// ─── Proof tests: GC really moves objects ───────────────────────────

#[test]
fn proves_object_moved_lowbit() {
    use std::cell::Cell;

    let (module, f_entry, f_alloc) =
        single_func_module("prove_move", &[], Some(Type::I64), |b, f_alloc| {
            let obj = b.call(f_alloc, &[]).unwrap();
            let magic = b.iconst(Type::I64, 42);
            let offset = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
            let addr = b.add(obj, offset);
            b.store(magic, addr, 0);
            b.safepoint(&[obj]);
            let ptr_as_i64 = b.bitcast(obj, Type::I64);
            b.ret(ptr_as_i64);
        });

    let original_ptr = Cell::new(0u64);
    let roots = MutatorRootManager::<LowBitPtrPolicy<3>>::new(4096);
    let mut interp = ModuleInterpreter::<LowBit<3>, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        original_ptr.set(ptr as u64);
        ExternCallResult::Value(Some(ptr as u64))
    });

    let result = interp.run(f_entry, &[]).unwrap();
    let returned_ptr = match result {
        InterpResult::Value(v) => v,
        other => panic!("expected Value, got {:?}", other),
    };

    let orig = original_ptr.get();
    assert_ne!(orig, 0, "alloc should have been called");
    assert_ne!(
        returned_ptr, orig,
        "GC should have moved the object: original={orig:#x} returned={returned_ptr:#x}"
    );

    // The stored value should still be readable at the new location
    let field_addr = returned_ptr + PAIR_TYPE.value_field_offset(0) as u64;
    let stored = unsafe { *(field_addr as *const u64) };
    assert_eq!(stored, 42, "value should survive the move");
}

#[test]
fn proves_object_moved_nanbox() {
    use std::cell::Cell;

    let (module, f_entry, f_alloc) =
        single_func_module("prove_move_nan", &[], Some(Type::I64), |b, f_alloc| {
            let obj = b.call(f_alloc, &[]).unwrap();
            let magic = b.iconst(Type::I64, 42);
            let offset = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
            let raw = b.payload(obj);
            let addr = b.add(raw, offset);
            b.store(magic, addr, 0);
            b.safepoint(&[obj]);
            let raw2 = b.payload(obj);
            let raw_as_i64 = b.bitcast(raw2, Type::I64);
            b.ret(raw_as_i64);
        });

    let original_ptr = Cell::new(0u64);
    let roots = FrameChainRootManager::<NanBoxPtrPolicy>::new(4096);
    let mut interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        original_ptr.set(ptr as u64);
        ExternCallResult::Value(Some(NanBox::encode_tagged(0, ptr as u64)))
    });

    let result = interp.run(f_entry, &[]).unwrap();
    let returned_ptr = match result {
        InterpResult::Value(v) => v,
        other => panic!("expected Value, got {:?}", other),
    };

    let orig = original_ptr.get();
    assert_ne!(orig, 0, "alloc should have been called");
    assert_ne!(
        returned_ptr, orig,
        "GC should have moved the object: original={orig:#x} returned={returned_ptr:#x}"
    );
}

// ─── Dead objects reclaimed ──────────────────────────────────────────

#[test]
fn dead_objects_reclaimed_lowbit() {
    use std::cell::Cell;

    let mut mb = ModuleBuilder::new();
    let f_alloc = mb.declare_extern(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );
    let f_entry = mb.declare_func("reclaim", &[], Some(Type::I64));

    {
        let mut b = mb.define_func(f_entry);
        let entry = b.entry_block();
        let loop_bb = b.create_block(&[Type::I64, Type::GcPtr]);
        let exit_bb = b.create_block(&[Type::GcPtr]);

        let first = b.call(f_alloc, &[]).unwrap();
        let zero = b.iconst(Type::I64, 0);
        b.jump(loop_bb, &[zero, first]);

        b.switch_to_block(loop_bb);
        let i = b.block_param(loop_bb, 0);
        let _prev = b.block_param(loop_bb, 1);

        let obj = b.call(f_alloc, &[]).unwrap();
        let val = b.iconst(Type::I64, 999);
        let offset = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        let addr = b.add(obj, offset);
        b.store(val, addr, 0);

        // Only keep the latest object alive
        b.safepoint(&[obj]);

        let one = b.iconst(Type::I64, 1);
        let new_i = b.add(i, one);
        let limit = b.iconst(Type::I64, 10);
        let cond = b.icmp(dynir::CmpOp::Slt, new_i, limit);
        b.br_if(cond, loop_bb, &[new_i, obj], exit_bb, &[obj]);

        b.switch_to_block(exit_bb);
        let last_obj = b.block_param(exit_bb, 0);
        let addr2 = b.add(last_obj, offset);
        let loaded = b.load(Type::I64, addr2, 0);
        b.ret(loaded);

        mb.finish_func(f_entry, b);
    }

    let module = mb.build();
    let alloc_count = Cell::new(0usize);
    let roots = MutatorRootManager::<LowBitPtrPolicy<3>>::new(4096);
    let mut interp = ModuleInterpreter::<LowBit<3>, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        alloc_count.set(alloc_count.get() + 1);
        ExternCallResult::Value(Some(ptr as u64))
    });

    let result = interp.run(f_entry, &[]).unwrap();

    // All 11 objects were allocated (1 initial + 10 loop)
    assert_eq!(alloc_count.get(), 11);

    // After GC, only 1 object survives
    let used_after = roots.from_used();
    let one_obj_size = PAIR_TYPE.allocation_size(0);
    assert_eq!(
        used_after, one_obj_size,
        "only 1 of 11 objects should survive GC: used={used_after} expected={one_obj_size}"
    );

    assert_eq!(result, InterpResult::Value(999));
}

// ─── Boxed fibonacci (LowBit) ───────────────────────────────────────

#[test]
fn boxed_fib_lowbit() {
    let mut mb = ModuleBuilder::new();
    let f_alloc = mb.declare_extern(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );
    let f_entry = mb.declare_func("boxed_fib", &[Type::I64], Some(Type::I64));

    {
        let mut b = mb.define_func(f_entry);
        let entry = b.entry_block();
        let n = b.block_param(entry, 0);

        let loop_bb = b.create_block(&[Type::I64, Type::GcPtr, Type::GcPtr]);
        let exit_bb = b.create_block(&[Type::GcPtr]);

        let box_a = b.call(f_alloc, &[]).unwrap();
        let zero = b.iconst(Type::I64, 0);
        let field_off = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        let addr_a = b.add(box_a, field_off);
        b.store(zero, addr_a, 0);

        let box_b = b.call(f_alloc, &[]).unwrap();
        let one = b.iconst(Type::I64, 1);
        let addr_b = b.add(box_b, field_off);
        b.store(one, addr_b, 0);

        let i_zero = b.iconst(Type::I64, 0);
        b.jump(loop_bb, &[i_zero, box_a, box_b]);

        b.switch_to_block(loop_bb);
        let i = b.block_param(loop_bb, 0);
        let a_box = b.block_param(loop_bb, 1);
        let b_box = b.block_param(loop_bb, 2);

        let cond = b.icmp(dynir::CmpOp::Sge, i, n);
        let continue_bb = b.create_block(&[]);
        b.br_if(cond, exit_bb, &[a_box], continue_bb, &[]);

        b.switch_to_block(continue_bb);
        let a_addr = b.add(a_box, field_off);
        let a_val = b.load(Type::I64, a_addr, 0);
        let b_addr = b.add(b_box, field_off);
        let b_val = b.load(Type::I64, b_addr, 0);
        let tmp = b.add(a_val, b_val);

        let new_box = b.call(f_alloc, &[]).unwrap();
        let new_addr = b.add(new_box, field_off);
        b.store(tmp, new_addr, 0);

        b.safepoint(&[b_box, new_box]);

        let one_i = b.iconst(Type::I64, 1);
        let new_i = b.add(i, one_i);
        b.jump(loop_bb, &[new_i, b_box, new_box]);

        b.switch_to_block(exit_bb);
        let result_box = b.block_param(exit_bb, 0);
        let result_addr = b.add(result_box, field_off);
        let result = b.load(Type::I64, result_addr, 0);
        b.ret(result);

        mb.finish_func(f_entry, b);
    }

    let module = mb.build();
    let roots = MutatorRootManager::<LowBitPtrPolicy<3>>::new(8192);
    let mut interp = ModuleInterpreter::<LowBit<3>, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "heap exhausted");
        ExternCallResult::Value(Some(ptr as u64))
    });

    let result = interp.run(f_entry, &[10]).unwrap();
    assert_eq!(result, InterpResult::Value(55));
    assert!(roots.collections() >= 10);

    let result = interp.run(f_entry, &[20]).unwrap();
    assert_eq!(result, InterpResult::Value(6765));
}

// ─── Boxed fibonacci (NanBox) ───────────────────────────────────────

#[test]
fn boxed_fib_nanbox() {
    let mut mb = ModuleBuilder::new();
    let f_alloc = mb.declare_extern(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );
    let f_entry = mb.declare_func("boxed_fib_nan", &[Type::I64], Some(Type::I64));

    {
        let mut b = mb.define_func(f_entry);
        let entry = b.entry_block();
        let n = b.block_param(entry, 0);

        let loop_bb = b.create_block(&[Type::I64, Type::GcPtr, Type::GcPtr]);
        let exit_bb = b.create_block(&[Type::GcPtr]);

        let box_a = b.call(f_alloc, &[]).unwrap();
        let zero = b.iconst(Type::I64, 0);
        let field_off = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        let raw_a = b.payload(box_a);
        let addr_a = b.add(raw_a, field_off);
        b.store(zero, addr_a, 0);

        let box_b = b.call(f_alloc, &[]).unwrap();
        let one = b.iconst(Type::I64, 1);
        let raw_b = b.payload(box_b);
        let addr_b = b.add(raw_b, field_off);
        b.store(one, addr_b, 0);

        let i_zero = b.iconst(Type::I64, 0);
        b.jump(loop_bb, &[i_zero, box_a, box_b]);

        b.switch_to_block(loop_bb);
        let i = b.block_param(loop_bb, 0);
        let a_box = b.block_param(loop_bb, 1);
        let b_box = b.block_param(loop_bb, 2);

        let cond = b.icmp(dynir::CmpOp::Sge, i, n);
        let continue_bb = b.create_block(&[]);
        b.br_if(cond, exit_bb, &[a_box], continue_bb, &[]);

        b.switch_to_block(continue_bb);
        let a_raw = b.payload(a_box);
        let a_addr = b.add(a_raw, field_off);
        let a_val = b.load(Type::I64, a_addr, 0);
        let b_raw = b.payload(b_box);
        let b_addr = b.add(b_raw, field_off);
        let b_val = b.load(Type::I64, b_addr, 0);
        let tmp = b.add(a_val, b_val);

        let new_box = b.call(f_alloc, &[]).unwrap();
        let new_raw = b.payload(new_box);
        let new_addr = b.add(new_raw, field_off);
        b.store(tmp, new_addr, 0);

        b.safepoint(&[b_box, new_box]);

        let one_i = b.iconst(Type::I64, 1);
        let new_i = b.add(i, one_i);
        b.jump(loop_bb, &[new_i, b_box, new_box]);

        b.switch_to_block(exit_bb);
        let result_box = b.block_param(exit_bb, 0);
        let result_raw = b.payload(result_box);
        let result_addr = b.add(result_raw, field_off);
        let result = b.load(Type::I64, result_addr, 0);
        b.ret(result);

        mb.finish_func(f_entry, b);
    }

    let module = mb.build();
    let roots = FrameChainRootManager::<NanBoxPtrPolicy>::new(8192);
    let mut interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "heap exhausted");
        ExternCallResult::Value(Some(NanBox::encode_tagged(0, ptr as u64)))
    });

    let result = interp.run(f_entry, &[10]).unwrap();
    assert_eq!(result, InterpResult::Value(55));
    assert!(roots.collections() >= 10);

    let result = interp.run(f_entry, &[20]).unwrap();
    assert_eq!(result, InterpResult::Value(6765));
}

// ─── DynRootFrame unit tests ────────────────────────────────────────

#[test]
fn dyn_root_frame_basic() {
    use dynobj::DynRootFrame;
    use dynobj::FrameChain;

    let frame = DynRootFrame::new(3);
    assert_eq!(frame.slot_count(), 3);

    frame.set(0, 100);
    frame.set(1, 200);
    frame.set(2, 300);
    assert_eq!(frame.get(0), 100);
    assert_eq!(frame.get(1), 200);
    assert_eq!(frame.get(2), 300);

    let chain = FrameChain::new();
    let _guard = frame.push_onto(&chain);
    assert_eq!(chain.depth(), 1);

    let mut count = 0;
    use dynobj::RootSource;
    chain.scan_roots(&mut |slot| {
        let val = unsafe { *slot };
        match count {
            0 => assert_eq!(val, 100),
            1 => assert_eq!(val, 200),
            2 => assert_eq!(val, 300),
            _ => panic!("too many roots"),
        }
        count += 1;
    });
    assert_eq!(count, 3);
}

#[test]
fn dyn_root_frame_nested() {
    use dynobj::DynRootFrame;
    use dynobj::{FrameChain, RootSource};

    let chain = FrameChain::new();

    let frame1 = DynRootFrame::new(2);
    frame1.set(0, 10);
    frame1.set(1, 20);
    let _g1 = frame1.push_onto(&chain);

    let frame2 = DynRootFrame::new(1);
    frame2.set(0, 30);
    let _g2 = frame2.push_onto(&chain);

    assert_eq!(chain.depth(), 2);

    let mut values = vec![];
    chain.scan_roots(&mut |slot| {
        values.push(unsafe { *slot });
    });
    assert_eq!(values, vec![30, 10, 20]);
}

#[test]
fn dyn_root_frame_clear() {
    use dynobj::DynRootFrame;

    let frame = DynRootFrame::new(3);
    frame.set(0, 100);
    frame.set(1, 200);
    frame.set(2, 300);

    frame.clear_all();

    assert_eq!(frame.get(0), 0);
    assert_eq!(frame.get(1), 0);
    assert_eq!(frame.get(2), 0);
}

// ─── PtrPolicy tests ────────────────────────────────────────────────

#[test]
fn lowbit_ptr_policy_roundtrip() {
    use crate::ptr_policy::LowBitPtrPolicy;
    use dynalloc::PtrPolicy;

    let ptr = 0x1000_0000_0008 as *mut u8;
    let bits = LowBitPtrPolicy::<3>::encode_ptr(ptr);
    assert_eq!(bits, ptr as u64);

    let decoded = LowBitPtrPolicy::<3>::try_decode_ptr(bits);
    assert_eq!(decoded, Some(ptr));

    let fixnum = LowBit::<3>::encode_tagged(1, 42);
    assert_eq!(LowBitPtrPolicy::<3>::try_decode_ptr(fixnum), None);

    assert_eq!(LowBitPtrPolicy::<3>::try_decode_ptr(0), None);
}

#[test]
fn nanbox_ptr_policy_roundtrip() {
    use crate::ptr_policy::NanBoxPtrPolicy;
    use dynalloc::PtrPolicy;

    let ptr = 0x0000_1234_5678 as *mut u8;
    let bits = NanBoxPtrPolicy::encode_ptr(ptr);

    assert!(NanBox::has_tag(bits, 0));
    assert_eq!(NanBox::extract_payload(bits), ptr as u64);

    let decoded = NanBoxPtrPolicy::try_decode_ptr(bits);
    assert_eq!(decoded, Some(ptr));

    let float_bits = 3.14f64.to_bits();
    assert_eq!(NanBoxPtrPolicy::try_decode_ptr(float_bits), None);

    let fixnum = NanBox::encode_tagged(1, 42);
    assert_eq!(NanBoxPtrPolicy::try_decode_ptr(fixnum), None);

    let null_tagged = NanBox::encode_tagged(0, 0);
    assert_eq!(NanBoxPtrPolicy::try_decode_ptr(null_tagged), None);
}

// ─── ModuleInterpreter + GC: callee allocates ───────────────────────

/// Build a module where callee allocates a GC object, stores a value, and returns it.
/// Caller calls callee, then hits a safepoint (GC may move objects), then loads the value.
fn build_callee_allocs_module() -> (dynir::Module, FuncRef, FuncRef, FuncRef) {
    let mut mb = ModuleBuilder::new();
    let f_alloc = mb.declare_extern(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );
    let f_callee = mb.declare_func("callee", &[], Some(Type::GcPtr));
    let f_main = mb.declare_func("main", &[], Some(Type::I64));

    {
        let mut fb = mb.define_func(f_callee);
        let obj = fb.call(f_alloc, &[]).unwrap();
        let val = fb.iconst(Type::I64, 42);
        let field_off = fb.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        let addr = fb.add(obj, field_off);
        fb.store(val, addr, 0);
        fb.ret(obj);
        mb.finish_func(f_callee, fb);
    }
    {
        let mut fb = mb.define_func(f_main);
        let obj = fb.call(f_callee, &[]).unwrap();
        fb.safepoint(&[obj]);
        let field_off = fb.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        let addr = fb.add(obj, field_off);
        let loaded = fb.load(Type::I64, addr, 0);
        fb.ret(loaded);
        mb.finish_func(f_main, fb);
    }

    (mb.build(), f_main, f_alloc, f_callee)
}

#[test]
fn module_gc_callee_allocs_lowbit() {
    let (module, f_main, f_alloc, _) = build_callee_allocs_module();

    let roots = MutatorRootManager::<LowBitPtrPolicy<3>>::new(4096);
    let mut interp = ModuleInterpreter::<LowBit<3>, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "allocation failed");
        ExternCallResult::Value(Some(ptr as u64))
    });

    let result = interp.run(f_main, &[]).unwrap();
    assert_eq!(result, InterpResult::Value(42));
    assert_eq!(roots.collections(), 1);
}

fn build_callee_allocs_module_nanbox() -> (dynir::Module, FuncRef, FuncRef, FuncRef) {
    let mut mb = ModuleBuilder::new();
    let f_alloc = mb.declare_extern(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );
    let f_callee = mb.declare_func("callee", &[], Some(Type::GcPtr));
    let f_main = mb.declare_func("main", &[], Some(Type::I64));

    {
        let mut fb = mb.define_func(f_callee);
        let obj = fb.call(f_alloc, &[]).unwrap();
        let val = fb.iconst(Type::I64, 42);
        let field_off = fb.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        let raw = fb.payload(obj);
        let addr = fb.add(raw, field_off);
        fb.store(val, addr, 0);
        fb.ret(obj);
        mb.finish_func(f_callee, fb);
    }
    {
        let mut fb = mb.define_func(f_main);
        let obj = fb.call(f_callee, &[]).unwrap();
        fb.safepoint(&[obj]);
        let field_off = fb.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        let raw = fb.payload(obj);
        let addr = fb.add(raw, field_off);
        let loaded = fb.load(Type::I64, addr, 0);
        fb.ret(loaded);
        mb.finish_func(f_main, fb);
    }

    (mb.build(), f_main, f_alloc, f_callee)
}

#[test]
fn module_gc_callee_allocs_nanbox() {
    let (module, f_main, f_alloc, _) = build_callee_allocs_module_nanbox();

    let roots = FrameChainRootManager::<NanBoxPtrPolicy>::new(4096);
    let mut interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "allocation failed");
        ExternCallResult::Value(Some(NanBox::encode_tagged(0, ptr as u64)))
    });

    let result = interp.run(f_main, &[]).unwrap();
    assert_eq!(result, InterpResult::Value(42));
    assert_eq!(roots.collections(), 1);
}

// ─── ModuleInterpreter + GC: caller roots survive callee GC ─────────

fn build_caller_roots_survive_module() -> (dynir::Module, FuncRef, FuncRef) {
    let mut mb = ModuleBuilder::new();
    let f_alloc = mb.declare_extern(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );
    let f_callee = mb.declare_func("callee", &[Type::GcPtr], None);
    let f_main = mb.declare_func("main", &[], Some(Type::I64));

    {
        let mut fb = mb.define_func(f_callee);
        let entry = fb.entry_block();
        let obj = fb.block_param(entry, 0);
        let _j1 = fb.call(f_alloc, &[]).unwrap();
        let _j2 = fb.call(f_alloc, &[]).unwrap();
        let _j3 = fb.call(f_alloc, &[]).unwrap();
        fb.safepoint(&[obj]);
        fb.ret_void();
        mb.finish_func(f_callee, fb);
    }
    {
        let mut fb = mb.define_func(f_main);
        let obj = fb.call(f_alloc, &[]).unwrap();
        let val = fb.iconst(Type::I64, 99);
        let field_off = fb.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        let addr = fb.add(obj, field_off);
        fb.store(val, addr, 0);
        fb.call(f_callee, &[obj]);
        let addr2 = fb.add(obj, field_off);
        let loaded = fb.load(Type::I64, addr2, 0);
        fb.ret(loaded);
        mb.finish_func(f_main, fb);
    }

    (mb.build(), f_main, f_alloc)
}

#[test]
fn module_gc_caller_roots_survive_callee_gc_lowbit() {
    let (module, f_main, f_alloc) = build_caller_roots_survive_module();

    let roots = MutatorRootManager::<LowBitPtrPolicy<3>>::new(4096);
    let mut interp = ModuleInterpreter::<LowBit<3>, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "allocation failed");
        ExternCallResult::Value(Some(ptr as u64))
    });

    let result = interp.run(f_main, &[]).unwrap();
    assert_eq!(result, InterpResult::Value(99));
    assert!(roots.collections() >= 1);
}

fn build_caller_roots_survive_module_nanbox() -> (dynir::Module, FuncRef, FuncRef) {
    let mut mb = ModuleBuilder::new();
    let f_alloc = mb.declare_extern(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );
    let f_callee = mb.declare_func("callee", &[Type::GcPtr], None);
    let f_main = mb.declare_func("main", &[], Some(Type::I64));

    {
        let mut fb = mb.define_func(f_callee);
        let entry = fb.entry_block();
        let obj = fb.block_param(entry, 0);
        let _j1 = fb.call(f_alloc, &[]).unwrap();
        let _j2 = fb.call(f_alloc, &[]).unwrap();
        let _j3 = fb.call(f_alloc, &[]).unwrap();
        fb.safepoint(&[obj]);
        fb.ret_void();
        mb.finish_func(f_callee, fb);
    }
    {
        let mut fb = mb.define_func(f_main);
        let obj = fb.call(f_alloc, &[]).unwrap();
        let val = fb.iconst(Type::I64, 99);
        let field_off = fb.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        let raw = fb.payload(obj);
        let addr = fb.add(raw, field_off);
        fb.store(val, addr, 0);
        fb.call(f_callee, &[obj]);
        let raw2 = fb.payload(obj);
        let addr2 = fb.add(raw2, field_off);
        let loaded = fb.load(Type::I64, addr2, 0);
        fb.ret(loaded);
        mb.finish_func(f_main, fb);
    }

    (mb.build(), f_main, f_alloc)
}

#[test]
fn module_gc_caller_roots_survive_callee_gc_nanbox() {
    let (module, f_main, f_alloc) = build_caller_roots_survive_module_nanbox();

    let roots = FrameChainRootManager::<NanBoxPtrPolicy>::new(4096);
    let mut interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "allocation failed");
        ExternCallResult::Value(Some(NanBox::encode_tagged(0, ptr as u64)))
    });

    let result = interp.run(f_main, &[]).unwrap();
    assert_eq!(result, InterpResult::Value(99));
    assert!(roots.collections() >= 1);
}

// ─── ModuleInterpreter + GC: recursive ──────────────────────────────

fn build_recursive_gc_module() -> (dynir::Module, FuncRef, FuncRef) {
    let mut mb = ModuleBuilder::new();
    let f_alloc = mb.declare_extern(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );
    let f_rec = mb.declare_func("recurse", &[Type::I64], Some(Type::I64));

    {
        let mut fb = mb.define_func(f_rec);
        let entry = fb.entry_block();
        let n = fb.block_param(entry, 0);

        let base_bb = fb.create_block(&[]);
        let rec_bb = fb.create_block(&[]);

        let zero = fb.iconst(Type::I64, 0);
        let cond = fb.icmp(dynir::CmpOp::Eq, n, zero);
        fb.br_if(cond, base_bb, &[], rec_bb, &[]);

        fb.switch_to_block(base_bb);
        let zero2 = fb.iconst(Type::I64, 0);
        fb.ret(zero2);

        fb.switch_to_block(rec_bb);
        let obj = fb.call(f_alloc, &[]).unwrap();
        let field_off = fb.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        let addr = fb.add(obj, field_off);
        fb.store(n, addr, 0);
        fb.safepoint(&[obj]);
        let one = fb.iconst(Type::I64, 1);
        let nm1 = fb.sub(n, one);
        let sub_result = fb.call(f_rec, &[nm1]).unwrap();
        let addr2 = fb.add(obj, field_off);
        let n_back = fb.load(Type::I64, addr2, 0);
        let result = fb.add(n_back, sub_result);
        fb.ret(result);

        mb.finish_func(f_rec, fb);
    }

    (mb.build(), f_rec, f_alloc)
}

#[test]
fn module_gc_recursive_lowbit() {
    let (module, f_rec, f_alloc) = build_recursive_gc_module();

    let roots = MutatorRootManager::<LowBitPtrPolicy<3>>::new(8192);
    let mut interp = ModuleInterpreter::<LowBit<3>, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "allocation failed");
        ExternCallResult::Value(Some(ptr as u64))
    });

    let result = interp.run(f_rec, &[5]).unwrap();
    assert_eq!(result, InterpResult::Value(15));
    assert!(roots.collections() >= 5);
}

fn build_recursive_gc_module_nanbox() -> (dynir::Module, FuncRef, FuncRef) {
    let mut mb = ModuleBuilder::new();
    let f_alloc = mb.declare_extern(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );
    let f_rec = mb.declare_func("recurse", &[Type::I64], Some(Type::I64));

    {
        let mut fb = mb.define_func(f_rec);
        let entry = fb.entry_block();
        let n = fb.block_param(entry, 0);

        let base_bb = fb.create_block(&[]);
        let rec_bb = fb.create_block(&[]);

        let zero = fb.iconst(Type::I64, 0);
        let cond = fb.icmp(dynir::CmpOp::Eq, n, zero);
        fb.br_if(cond, base_bb, &[], rec_bb, &[]);

        fb.switch_to_block(base_bb);
        let zero2 = fb.iconst(Type::I64, 0);
        fb.ret(zero2);

        fb.switch_to_block(rec_bb);
        let obj = fb.call(f_alloc, &[]).unwrap();
        let field_off = fb.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        let raw = fb.payload(obj);
        let addr = fb.add(raw, field_off);
        fb.store(n, addr, 0);
        fb.safepoint(&[obj]);
        let one = fb.iconst(Type::I64, 1);
        let nm1 = fb.sub(n, one);
        let sub_result = fb.call(f_rec, &[nm1]).unwrap();
        let raw2 = fb.payload(obj);
        let addr2 = fb.add(raw2, field_off);
        let n_back = fb.load(Type::I64, addr2, 0);
        let result = fb.add(n_back, sub_result);
        fb.ret(result);

        mb.finish_func(f_rec, fb);
    }

    (mb.build(), f_rec, f_alloc)
}

#[test]
fn module_gc_recursive_nanbox() {
    let (module, f_rec, f_alloc) = build_recursive_gc_module_nanbox();

    let roots = FrameChainRootManager::<NanBoxPtrPolicy>::new(8192);
    let mut interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "allocation failed");
        ExternCallResult::Value(Some(NanBox::encode_tagged(0, ptr as u64)))
    });

    let result = interp.run(f_rec, &[5]).unwrap();
    assert_eq!(result, InterpResult::Value(15));
    assert!(roots.collections() >= 5);
}

// ─── ModuleInterpreter + GC: proves object moved across frames ──────

#[test]
fn module_gc_proves_object_moved_lowbit() {
    let (module, f_main, f_alloc) = build_caller_roots_survive_module();

    let roots = MutatorRootManager::<LowBitPtrPolicy<3>>::new(512);
    let mut interp = ModuleInterpreter::<LowBit<3>, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "allocation failed");
        ExternCallResult::Value(Some(ptr as u64))
    });

    let result = interp.run(f_main, &[]).unwrap();
    assert_eq!(result, InterpResult::Value(99));
    assert!(roots.collections() >= 1);
}

#[test]
fn module_gc_proves_object_moved_nanbox() {
    let (module, f_main, f_alloc) = build_caller_roots_survive_module_nanbox();

    let roots = FrameChainRootManager::<NanBoxPtrPolicy>::new(512);
    let mut interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "allocation failed");
        ExternCallResult::Value(Some(NanBox::encode_tagged(0, ptr as u64)))
    });

    let result = interp.run(f_main, &[]).unwrap();
    assert_eq!(result, InterpResult::Value(99));
    assert!(roots.collections() >= 1);
}

// ─── Cross-combination tests: prove generics actually work ──────────

#[test]
fn cross_framechain_with_lowbit() {
    // FrameChain root strategy + LowBit pointer policy
    let (module, f_main, f_alloc, _) = build_callee_allocs_module();

    let roots = FrameChainRootManager::<LowBitPtrPolicy<3>>::new(4096);
    let mut interp = ModuleInterpreter::<LowBit<3>, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        ExternCallResult::Value(Some(ptr as u64))
    });

    let result = interp.run(f_main, &[]).unwrap();
    assert_eq!(result, InterpResult::Value(42));
    assert_eq!(roots.collections(), 1);
}

#[test]
fn cross_mutator_with_nanbox() {
    // Mutator root strategy + NanBox pointer policy
    let (module, f_main, f_alloc, _) = build_callee_allocs_module_nanbox();

    let roots = MutatorRootManager::<NanBoxPtrPolicy>::new(4096);
    let mut interp = ModuleInterpreter::<NanBox, _>::new(&module, &roots);
    interp.bind(f_alloc, |_args| {
        let ptr = roots.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        ExternCallResult::Value(Some(NanBox::encode_tagged(0, ptr as u64)))
    });

    let result = interp.run(f_main, &[]).unwrap();
    assert_eq!(result, InterpResult::Value(42));
    assert_eq!(roots.collections(), 1);
}
