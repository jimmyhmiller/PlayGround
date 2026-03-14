use std::rc::Rc;

use dynir::builder::FunctionBuilder;
use dynir::types::{Signature, Type};
use dynir::{ExternCallResult, InterpResult, Interpreter};
use dynobj::{Compact, ObjHeader, TypeInfo};
use dynvalue::{LowBit, NanBox, TagScheme};

use crate::framechain::FrameChainGcInterp;
use crate::stackmap::StackmapGcInterp;

// ─── Object layout ──────────────────────────────────────────────────

/// Simple object: Compact header + 1 GC-traced value field.
/// Total: 8 (header) + 8 (field) = 16 bytes.
static PAIR_TYPE: TypeInfo = TypeInfo::for_header(Compact::SIZE).with_fields(1);

// ─── Stackmap tests (LowBit<3> tagged pointers) ─────────────────────

#[test]
fn stackmap_alloc_store_safepoint_load() {
    let mut b = FunctionBuilder::new("test_gc", &[], Some(Type::I64));

    let alloc_fn = b.declare_func(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );

    let obj = b.call(alloc_fn, &[]).unwrap();

    let magic = b.iconst(Type::I64, 42);
    let field_offset = PAIR_TYPE.value_field_offset(0) as i64;
    let offset_val = b.iconst(Type::I64, field_offset);
    let field_addr = b.add(obj, offset_val);
    b.store(magic, field_addr, 0);

    b.safepoint(&[obj]);

    let field_addr2 = b.add(obj, offset_val);
    let loaded = b.load(Type::I64, field_addr2, 0);

    b.ret(loaded);

    let func = b.build();

    let gc = Rc::new(StackmapGcInterp::<3>::new(4096));

    let mut interp = Interpreter::<LowBit<3>>::new(&func);
    let gc_clone = gc.clone();
    interp.bind(alloc_fn, move |_args| {
        let ptr = gc_clone.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "allocation failed");
        ExternCallResult::Value(Some(ptr as u64))
    });

    let result = gc.run(&interp, &func, &[]).unwrap();
    match result {
        InterpResult::Value(v) => assert_eq!(v, 42, "GC should preserve the stored value"),
        other => panic!("expected Value(42), got {:?}", other),
    }

    assert_eq!(gc.collections(), 1, "should have collected once at safepoint");
}

#[test]
fn stackmap_multiple_objects_survive_gc() {
    let mut b = FunctionBuilder::new("two_obj", &[], Some(Type::I64));

    let alloc_fn = b.declare_func(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );

    let obj1 = b.call(alloc_fn, &[]).unwrap();
    let obj2 = b.call(alloc_fn, &[]).unwrap();

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

    let func = b.build();

    let gc = Rc::new(StackmapGcInterp::<3>::new(4096));
    let mut interp = Interpreter::<LowBit<3>>::new(&func);
    let gc_clone = gc.clone();
    interp.bind(alloc_fn, move |_args| {
        let ptr = gc_clone.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        ExternCallResult::Value(Some(ptr as u64))
    });

    let result = gc.run(&interp, &func, &[]).unwrap();
    match result {
        InterpResult::Value(v) => assert_eq!(v, 333, "111 + 222 = 333"),
        other => panic!("expected Value(333), got {:?}", other),
    }
}

#[test]
fn stackmap_gc_in_loop() {
    let mut b = FunctionBuilder::new("gc_loop", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let n = b.block_param(entry, 0);

    let alloc_fn = b.declare_func(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );

    let loop_bb = b.create_block(&[Type::I64, Type::I64]); // (i, acc)
    let exit_bb = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    b.jump(loop_bb, &[zero, zero]);

    b.switch_to_block(loop_bb);
    let i = b.block_param(loop_bb, 0);
    let acc = b.block_param(loop_bb, 1);

    let obj = b.call(alloc_fn, &[]).unwrap();
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

    let func = b.build();

    let gc = Rc::new(StackmapGcInterp::<3>::new(8192));
    let mut interp = Interpreter::<LowBit<3>>::new(&func);
    let gc_clone = gc.clone();
    interp.bind(alloc_fn, move |_args| {
        let ptr = gc_clone.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        ExternCallResult::Value(Some(ptr as u64))
    });

    // Sum 0..5 = 0+1+2+3+4 = 10
    let result = gc.run(&interp, &func, &[5]).unwrap();
    match result {
        InterpResult::Value(v) => assert_eq!(v, 10),
        other => panic!("expected Value(10), got {:?}", other),
    }

    assert!(gc.collections() >= 5, "should have collected at each iteration");
}

// ─── Frame chain tests (NaN-boxing) ─────────────────────────────────

#[test]
fn framechain_alloc_store_safepoint_load() {
    let mut b = FunctionBuilder::new("test_gc_nan", &[], Some(Type::I64));

    let alloc_fn = b.declare_func(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );

    let obj = b.call(alloc_fn, &[]).unwrap();

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

    let func = b.build();

    let gc = Rc::new(FrameChainGcInterp::new(4096));
    let mut interp = Interpreter::<NanBox>::new(&func);
    let gc_clone = gc.clone();
    interp.bind(alloc_fn, move |_args| {
        let ptr = gc_clone.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        let tagged = NanBox::encode_tagged(0, ptr as u64);
        ExternCallResult::Value(Some(tagged))
    });

    let result = gc.run(&interp, &func, &[]).unwrap();
    match result {
        InterpResult::Value(v) => assert_eq!(v, 42, "GC should preserve the stored value"),
        other => panic!("expected Value(42), got {:?}", other),
    }

    assert_eq!(gc.collections(), 1);
}

#[test]
fn framechain_multiple_objects_survive_gc() {
    let mut b = FunctionBuilder::new("two_obj_nan", &[], Some(Type::I64));

    let alloc_fn = b.declare_func(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );

    let obj1 = b.call(alloc_fn, &[]).unwrap();
    let obj2 = b.call(alloc_fn, &[]).unwrap();

    let val1 = b.iconst(Type::I64, 111);
    let val2 = b.iconst(Type::I64, 222);
    let field_offset = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);

    let raw1 = b.payload(obj1);
    let addr1 = b.add(raw1, field_offset);
    b.store(val1, addr1, 0);

    let raw2 = b.payload(obj2);
    let addr2 = b.add(raw2, field_offset);
    b.store(val2, addr2, 0);

    b.safepoint(&[obj1, obj2]);

    let raw1b = b.payload(obj1);
    let addr1b = b.add(raw1b, field_offset);
    let loaded1 = b.load(Type::I64, addr1b, 0);

    let raw2b = b.payload(obj2);
    let addr2b = b.add(raw2b, field_offset);
    let loaded2 = b.load(Type::I64, addr2b, 0);

    let sum = b.add(loaded1, loaded2);
    b.ret(sum);

    let func = b.build();

    let gc = Rc::new(FrameChainGcInterp::new(4096));
    let mut interp = Interpreter::<NanBox>::new(&func);
    let gc_clone = gc.clone();
    interp.bind(alloc_fn, move |_args| {
        let ptr = gc_clone.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        ExternCallResult::Value(Some(NanBox::encode_tagged(0, ptr as u64)))
    });

    let result = gc.run(&interp, &func, &[]).unwrap();
    match result {
        InterpResult::Value(v) => assert_eq!(v, 333),
        other => panic!("expected Value(333), got {:?}", other),
    }
}

#[test]
fn framechain_gc_in_loop() {
    let mut b = FunctionBuilder::new("gc_loop_nan", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let n = b.block_param(entry, 0);

    let alloc_fn = b.declare_func(
        "alloc",
        Signature {
            params: vec![],
            ret: Some(Type::GcPtr),
        },
    );

    let loop_bb = b.create_block(&[Type::I64, Type::I64]);
    let exit_bb = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    b.jump(loop_bb, &[zero, zero]);

    b.switch_to_block(loop_bb);
    let i = b.block_param(loop_bb, 0);
    let acc = b.block_param(loop_bb, 1);

    let obj = b.call(alloc_fn, &[]).unwrap();
    let field_offset = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
    let raw = b.payload(obj);
    let addr = b.add(raw, field_offset);
    b.store(i, addr, 0);

    b.safepoint(&[obj]);

    let raw2 = b.payload(obj);
    let addr2 = b.add(raw2, field_offset);
    let loaded = b.load(Type::I64, addr2, 0);
    let new_acc = b.add(acc, loaded);

    let one = b.iconst(Type::I64, 1);
    let new_i = b.add(i, one);
    let cond = b.icmp(dynir::CmpOp::Slt, new_i, n);
    b.br_if(cond, loop_bb, &[new_i, new_acc], exit_bb, &[new_acc]);

    b.switch_to_block(exit_bb);
    let result = b.block_param(exit_bb, 0);
    b.ret(result);

    let func = b.build();

    let gc = Rc::new(FrameChainGcInterp::new(8192));
    let mut interp = Interpreter::<NanBox>::new(&func);
    let gc_clone = gc.clone();
    interp.bind(alloc_fn, move |_args| {
        let ptr = gc_clone.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        ExternCallResult::Value(Some(NanBox::encode_tagged(0, ptr as u64)))
    });

    let result = gc.run(&interp, &func, &[5]).unwrap();
    match result {
        InterpResult::Value(v) => assert_eq!(v, 10),
        other => panic!("expected Value(10), got {:?}", other),
    }

    assert!(gc.collections() >= 5);
}

// ─── Proof tests: GC really moves objects ───────────────────────────

#[test]
fn stackmap_proves_object_moved() {
    // Build IR that:
    //   %obj = call @alloc()
    //   store 42 at obj.field[0]
    //   safepoint [%obj]        ← GC copies obj to to-space
    //   return %obj             ← return the (updated) pointer
    //
    // If GC really moved the object, the returned pointer differs from
    // the one alloc originally handed out.
    use std::cell::Cell;

    let mut b = FunctionBuilder::new("prove_move", &[], Some(Type::I64));
    let alloc_fn = b.declare_func("alloc", Signature { params: vec![], ret: Some(Type::GcPtr) });

    let obj = b.call(alloc_fn, &[]).unwrap();
    let magic = b.iconst(Type::I64, 42);
    let offset = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
    let addr = b.add(obj, offset);
    b.store(magic, addr, 0);
    b.safepoint(&[obj]);
    // Return the raw pointer value — bitcast GcPtr → I64 so the type checker is happy
    // (For LowBit<3>, the tagged value IS the aligned pointer, so this is identity)
    let ptr_as_i64 = b.bitcast(obj, Type::I64);
    b.ret(ptr_as_i64);

    let func = b.build();
    let gc = Rc::new(StackmapGcInterp::<3>::new(4096));

    let original_ptr = Rc::new(Cell::new(0u64));
    let orig_clone = original_ptr.clone();

    let mut interp = Interpreter::<LowBit<3>>::new(&func);
    let gc_clone = gc.clone();
    interp.bind(alloc_fn, move |_args| {
        let ptr = gc_clone.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        orig_clone.set(ptr as u64);
        ExternCallResult::Value(Some(ptr as u64))
    });

    let result = gc.run(&interp, &func, &[]).unwrap();
    match result {
        InterpResult::Value(new_ptr) => {
            let old_ptr = original_ptr.get();
            assert_ne!(old_ptr, 0, "alloc should have been called");
            assert_ne!(new_ptr, old_ptr,
                "GC should have moved the object: old={old_ptr:#x} new={new_ptr:#x}");
            assert_ne!(new_ptr, 0, "new pointer should not be null");

            // The stored value should survive the move
            let field_addr = new_ptr as usize + PAIR_TYPE.value_field_offset(0);
            let loaded = unsafe { *(field_addr as *const u64) };
            assert_eq!(loaded, 42, "field value should survive GC move");
        }
        other => panic!("expected Value, got {:?}", other),
    }
    assert_eq!(gc.collections(), 1);
}

#[test]
fn framechain_proves_object_moved() {
    // Same test but with NaN-boxing + frame chain.
    // Returns the raw pointer payload (not the tagged value).
    use std::cell::Cell;

    let mut b = FunctionBuilder::new("prove_move_nan", &[], Some(Type::I64));
    let alloc_fn = b.declare_func("alloc", Signature { params: vec![], ret: Some(Type::GcPtr) });

    let obj = b.call(alloc_fn, &[]).unwrap();
    let magic = b.iconst(Type::I64, 42);
    let offset = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
    let raw = b.payload(obj);
    let addr = b.add(raw, offset);
    b.store(magic, addr, 0);
    b.safepoint(&[obj]);
    // Extract the raw pointer after GC (may have changed)
    let raw_after = b.payload(obj);
    b.ret(raw_after);

    let func = b.build();
    let gc = Rc::new(FrameChainGcInterp::new(4096));

    let original_ptr = Rc::new(Cell::new(0u64));
    let orig_clone = original_ptr.clone();

    let mut interp = Interpreter::<NanBox>::new(&func);
    let gc_clone = gc.clone();
    interp.bind(alloc_fn, move |_args| {
        let ptr = gc_clone.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        orig_clone.set(ptr as u64);
        ExternCallResult::Value(Some(NanBox::encode_tagged(0, ptr as u64)))
    });

    let result = gc.run(&interp, &func, &[]).unwrap();
    match result {
        InterpResult::Value(new_ptr) => {
            let old_ptr = original_ptr.get();
            assert_ne!(old_ptr, 0, "alloc should have been called");
            assert_ne!(new_ptr, old_ptr,
                "GC should have moved the object: old={old_ptr:#x} new={new_ptr:#x}");

            let field_addr = new_ptr as usize + PAIR_TYPE.value_field_offset(0);
            let loaded = unsafe { *(field_addr as *const u64) };
            assert_eq!(loaded, 42, "field value should survive GC move");
        }
        other => panic!("expected Value, got {:?}", other),
    }
    assert_eq!(gc.collections(), 1);
}

#[test]
fn stackmap_dead_objects_reclaimed() {
    // Allocate 10 objects, but only keep the last one alive at safepoint.
    // After GC, from_used should be just one object's worth.
    use std::cell::Cell;

    let mut b = FunctionBuilder::new("dead_reclaim", &[], Some(Type::I64));
    let alloc_fn = b.declare_func("alloc", Signature { params: vec![], ret: Some(Type::GcPtr) });

    // Allocate 10 objects (only keep the last one alive)
    let mut last_obj = b.call(alloc_fn, &[]).unwrap();
    for _ in 1..10 {
        last_obj = b.call(alloc_fn, &[]).unwrap();
    }

    // Store a value in the last object
    let magic = b.iconst(Type::I64, 999);
    let offset = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
    let addr = b.add(last_obj, offset);
    b.store(magic, addr, 0);

    // Only last_obj is live at safepoint — the other 9 are dead
    b.safepoint(&[last_obj]);

    // Load from the surviving object
    let addr2 = b.add(last_obj, offset);
    let loaded = b.load(Type::I64, addr2, 0);
    b.ret(loaded);

    let func = b.build();
    let gc = Rc::new(StackmapGcInterp::<3>::new(4096));

    let alloc_count = Rc::new(Cell::new(0usize));
    let count_clone = alloc_count.clone();

    let mut interp = Interpreter::<LowBit<3>>::new(&func);
    let gc_clone = gc.clone();
    interp.bind(alloc_fn, move |_args| {
        let ptr = gc_clone.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null());
        count_clone.set(count_clone.get() + 1);
        ExternCallResult::Value(Some(ptr as u64))
    });

    let used_before_run = gc.from_used();
    assert_eq!(used_before_run, 0, "nothing allocated yet");

    let result = gc.run(&interp, &func, &[]).unwrap();

    // All 10 objects were allocated
    assert_eq!(alloc_count.get(), 10);

    // After GC, only 1 object survives — from_used should reflect just that one
    let used_after = gc.from_used();
    let one_obj_size = PAIR_TYPE.allocation_size(0);
    assert_eq!(used_after, one_obj_size,
        "only 1 of 10 objects should survive GC: used={used_after} expected={one_obj_size}");

    match result {
        InterpResult::Value(v) => assert_eq!(v, 999, "surviving object's field intact"),
        other => panic!("expected Value(999), got {:?}", other),
    }
}

#[test]
fn both_schemes_same_ir_different_encoding() {
    // Build ONE piece of IR (allocate, store, GC, load, return).
    // Run it with BOTH tagging schemes. Both should produce the same result.
    // This proves the IR is encoding-agnostic.

    // --- Stackmap (LowBit<3>) ---
    {
        let mut b = FunctionBuilder::new("dual", &[], Some(Type::I64));
        let alloc_fn = b.declare_func("alloc", Signature { params: vec![], ret: Some(Type::GcPtr) });
        let obj = b.call(alloc_fn, &[]).unwrap();
        let val = b.iconst(Type::I64, 777);
        let offset = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        // LowBit: raw pointer IS the tagged value, no payload extraction needed
        let addr = b.add(obj, offset);
        b.store(val, addr, 0);
        b.safepoint(&[obj]);
        let addr2 = b.add(obj, offset);
        let loaded = b.load(Type::I64, addr2, 0);
        b.ret(loaded);
        let func = b.build();

        let gc = Rc::new(StackmapGcInterp::<3>::new(4096));
        let mut interp = Interpreter::<LowBit<3>>::new(&func);
        let gc_clone = gc.clone();
        interp.bind(alloc_fn, move |_args| {
            let ptr = gc_clone.alloc(&PAIR_TYPE, 0);
            ExternCallResult::Value(Some(ptr as u64))
        });

        let result = gc.run(&interp, &func, &[]).unwrap();
        match result {
            InterpResult::Value(v) => assert_eq!(v, 777, "LowBit scheme: value survives GC"),
            other => panic!("LowBit: expected Value(777), got {:?}", other),
        }
        assert_eq!(gc.collections(), 1, "LowBit: GC ran");
    }

    // --- FrameChain (NaN-boxing) ---
    {
        let mut b = FunctionBuilder::new("dual", &[], Some(Type::I64));
        let alloc_fn = b.declare_func("alloc", Signature { params: vec![], ret: Some(Type::GcPtr) });
        let obj = b.call(alloc_fn, &[]).unwrap();
        let val = b.iconst(Type::I64, 777);
        let offset = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
        // NaN-box: must extract raw pointer via payload before pointer arithmetic
        let raw = b.payload(obj);
        let addr = b.add(raw, offset);
        b.store(val, addr, 0);
        b.safepoint(&[obj]);
        let raw2 = b.payload(obj);
        let addr2 = b.add(raw2, offset);
        let loaded = b.load(Type::I64, addr2, 0);
        b.ret(loaded);
        let func = b.build();

        let gc = Rc::new(FrameChainGcInterp::new(4096));
        let mut interp = Interpreter::<NanBox>::new(&func);
        let gc_clone = gc.clone();
        interp.bind(alloc_fn, move |_args| {
            let ptr = gc_clone.alloc(&PAIR_TYPE, 0);
            ExternCallResult::Value(Some(NanBox::encode_tagged(0, ptr as u64)))
        });

        let result = gc.run(&interp, &func, &[]).unwrap();
        match result {
            InterpResult::Value(v) => assert_eq!(v, 777, "NanBox scheme: value survives GC"),
            other => panic!("NanBox: expected Value(777), got {:?}", other),
        }
        assert_eq!(gc.collections(), 1, "NanBox: GC ran");
    }
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

// ─── Wasm through GC interpreters ───────────────────────────────────

/// Translate WAT source to our IR Function.
fn wat_to_func(wat: &str) -> dynir::Function {
    let wasm = wat::parse_str(wat).expect("failed to parse WAT");
    let (func, _imports) = wasm2dynir::translate_wasm(&wasm).expect("failed to translate");
    dynir::verify::verify(&func).expect("IR verification failed");
    func
}

// Fibonacci WAT — used across multiple tests
const FIB_WAT: &str = r#"(module
    (func (export "fib") (param $n i32) (result i32)
        (local $a i32)
        (local $b i32)
        (local $i i32)
        (local $tmp i32)
        i32.const 0
        local.set $a
        i32.const 1
        local.set $b
        i32.const 0
        local.set $i
        block $exit
            loop $loop
                local.get $i
                local.get $n
                i32.ge_s
                br_if $exit
                local.get $a
                local.get $b
                i32.add
                local.set $tmp
                local.get $b
                local.set $a
                local.get $tmp
                local.set $b
                local.get $i
                i32.const 1
                i32.add
                local.set $i
                br $loop
            end
        end
        local.get $a))"#;

// Factorial WAT
const FACT_WAT: &str = r#"(module
    (func (export "fact") (param $n i32) (result i32)
        (local $result i32)
        i32.const 1
        local.set $result
        block $exit
            loop $loop
                local.get $n
                i32.const 1
                i32.le_s
                br_if $exit
                local.get $result
                local.get $n
                i32.mul
                local.set $result
                local.get $n
                i32.const 1
                i32.sub
                local.set $n
                br $loop
            end
        end
        local.get $result))"#;

#[test]
fn wasm_fib_through_stackmap_interp() {
    // Real wasm fibonacci running through the LowBit<3> stackmap GC interpreter.
    // No GcPtr values exist, so the GC is idle — but the interpreter pipeline is real.
    let func = wat_to_func(FIB_WAT);
    let gc = Rc::new(StackmapGcInterp::<3>::new(4096));
    let interp = Interpreter::<LowBit<3>>::new(&func);

    let result = gc.run(&interp, &func, &[10]).unwrap();
    assert_eq!(result, InterpResult::Value(55), "fib(10) = 55");

    let result = gc.run(&interp, &func, &[20]).unwrap();
    assert_eq!(result, InterpResult::Value(6765), "fib(20) = 6765");

    assert_eq!(gc.collections(), 0, "no GcPtr values → no safepoints → no collections");
}

#[test]
fn wasm_fib_through_framechain_interp() {
    // Same fibonacci but through NaN-boxing frame chain interpreter.
    let func = wat_to_func(FIB_WAT);
    let gc = Rc::new(FrameChainGcInterp::new(4096));
    let interp = Interpreter::<NanBox>::new(&func);

    let result = gc.run(&interp, &func, &[10]).unwrap();
    assert_eq!(result, InterpResult::Value(55));

    let result = gc.run(&interp, &func, &[20]).unwrap();
    assert_eq!(result, InterpResult::Value(6765));

    assert_eq!(gc.collections(), 0);
}

#[test]
fn wasm_factorial_through_both_interps() {
    let func = wat_to_func(FACT_WAT);

    // Stackmap (LowBit<3>)
    {
        let gc = Rc::new(StackmapGcInterp::<3>::new(4096));
        let interp = Interpreter::<LowBit<3>>::new(&func);
        let result = gc.run(&interp, &func, &[10]).unwrap();
        assert_eq!(result, InterpResult::Value(3628800), "10! via stackmap");
    }

    // FrameChain (NaN-boxing)
    {
        let gc = Rc::new(FrameChainGcInterp::new(4096));
        let interp = Interpreter::<NanBox>::new(&func);
        let result = gc.run(&interp, &func, &[10]).unwrap();
        assert_eq!(result, InterpResult::Value(3628800), "10! via framechain");
    }
}

#[test]
fn wasm_fib_with_gc_boxing_stackmap() {
    // The real deal: wasm-style fibonacci where every intermediate value
    // is boxed in a GC-allocated object. Each loop iteration:
    //   1. Allocates a box for the new fib value
    //   2. Stores the value in it
    //   3. Hits a safepoint (GC runs, box may move)
    //   4. Loads the value back from the (possibly moved) box
    //
    // This is what a dynamic language runtime actually does:
    // computation + allocation + GC, all interleaved.

    let mut b = FunctionBuilder::new("boxed_fib", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let n = b.block_param(entry, 0);

    let alloc_fn = b.declare_func("alloc", Signature { params: vec![], ret: Some(Type::GcPtr) });

    // loop(i, a, b) where a,b are boxed GC pointers
    let loop_bb = b.create_block(&[Type::I64, Type::GcPtr, Type::GcPtr]);
    let exit_bb = b.create_block(&[Type::GcPtr]);

    // Box the initial values: a=0, b=1
    let box_a = b.call(alloc_fn, &[]).unwrap();
    let zero = b.iconst(Type::I64, 0);
    let field_off = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
    let addr_a = b.add(box_a, field_off);
    b.store(zero, addr_a, 0);

    let box_b = b.call(alloc_fn, &[]).unwrap();
    let one = b.iconst(Type::I64, 1);
    let addr_b = b.add(box_b, field_off);
    b.store(one, addr_b, 0);

    let i_zero = b.iconst(Type::I64, 0);
    b.jump(loop_bb, &[i_zero, box_a, box_b]);

    // Loop body
    b.switch_to_block(loop_bb);
    let i = b.block_param(loop_bb, 0);
    let a_box = b.block_param(loop_bb, 1);
    let b_box = b.block_param(loop_bb, 2);

    // Check loop condition
    let cond = b.icmp(dynir::CmpOp::Sge, i, n);
    let continue_bb = b.create_block(&[]);
    b.br_if(cond, exit_bb, &[a_box], continue_bb, &[]);

    b.switch_to_block(continue_bb);

    // Unbox a and b
    let a_addr = b.add(a_box, field_off);
    let a_val = b.load(Type::I64, a_addr, 0);
    let b_addr = b.add(b_box, field_off);
    let b_val = b.load(Type::I64, b_addr, 0);

    // tmp = a + b
    let tmp = b.add(a_val, b_val);

    // Box the new value
    let new_box = b.call(alloc_fn, &[]).unwrap();
    let new_addr = b.add(new_box, field_off);
    b.store(tmp, new_addr, 0);

    // Safepoint: keep b_box (becomes new a) and new_box (becomes new b)
    b.safepoint(&[b_box, new_box]);

    let one_i = b.iconst(Type::I64, 1);
    let new_i = b.add(i, one_i);
    b.jump(loop_bb, &[new_i, b_box, new_box]);

    // Exit: unbox the result
    b.switch_to_block(exit_bb);
    let result_box = b.block_param(exit_bb, 0);
    let result_addr = b.add(result_box, field_off);
    let result = b.load(Type::I64, result_addr, 0);
    b.ret(result);

    let func = b.build();

    let gc = Rc::new(StackmapGcInterp::<3>::new(8192));
    let mut interp = Interpreter::<LowBit<3>>::new(&func);
    let gc_clone = gc.clone();
    interp.bind(alloc_fn, move |_args| {
        let ptr = gc_clone.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "heap exhausted");
        ExternCallResult::Value(Some(ptr as u64))
    });

    // fib(10) = 55
    let result = gc.run(&interp, &func, &[10]).unwrap();
    assert_eq!(result, InterpResult::Value(55));
    let collections = gc.collections();
    assert!(collections >= 10, "GC should run at each safepoint ({collections} collections)");

    // fib(20) = 6765
    let result = gc.run(&interp, &func, &[20]).unwrap();
    assert_eq!(result, InterpResult::Value(6765));
}

#[test]
fn wasm_fib_with_gc_boxing_framechain() {
    // Same boxed fibonacci but with NaN-boxing + frame chain.
    // The IR uses Payload to extract raw pointers for memory ops.

    let mut b = FunctionBuilder::new("boxed_fib_nan", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let n = b.block_param(entry, 0);

    let alloc_fn = b.declare_func("alloc", Signature { params: vec![], ret: Some(Type::GcPtr) });

    let loop_bb = b.create_block(&[Type::I64, Type::GcPtr, Type::GcPtr]);
    let exit_bb = b.create_block(&[Type::GcPtr]);

    // Box initial values
    let box_a = b.call(alloc_fn, &[]).unwrap();
    let zero = b.iconst(Type::I64, 0);
    let field_off = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);
    let raw_a = b.payload(box_a);
    let addr_a = b.add(raw_a, field_off);
    b.store(zero, addr_a, 0);

    let box_b = b.call(alloc_fn, &[]).unwrap();
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

    // Unbox a and b (extract raw pointers via payload)
    let a_raw = b.payload(a_box);
    let a_addr = b.add(a_raw, field_off);
    let a_val = b.load(Type::I64, a_addr, 0);
    let b_raw = b.payload(b_box);
    let b_addr = b.add(b_raw, field_off);
    let b_val = b.load(Type::I64, b_addr, 0);

    let tmp = b.add(a_val, b_val);

    // Box the new value
    let new_box = b.call(alloc_fn, &[]).unwrap();
    let new_raw = b.payload(new_box);
    let new_addr = b.add(new_raw, field_off);
    b.store(tmp, new_addr, 0);

    // Safepoint
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

    let func = b.build();

    let gc = Rc::new(FrameChainGcInterp::new(8192));
    let mut interp = Interpreter::<NanBox>::new(&func);
    let gc_clone = gc.clone();
    interp.bind(alloc_fn, move |_args| {
        let ptr = gc_clone.alloc(&PAIR_TYPE, 0);
        assert!(!ptr.is_null(), "heap exhausted");
        ExternCallResult::Value(Some(NanBox::encode_tagged(0, ptr as u64)))
    });

    let result = gc.run(&interp, &func, &[10]).unwrap();
    assert_eq!(result, InterpResult::Value(55));
    let collections = gc.collections();
    assert!(collections >= 10, "GC should run at each safepoint ({collections} collections)");

    let result = gc.run(&interp, &func, &[20]).unwrap();
    assert_eq!(result, InterpResult::Value(6765));
}

// ─── IR comparison: how the two schemes differ ──────────────────────

#[test]
fn print_both_boxed_fib_ir() {
    // LowBit version: pointer IS the value, arithmetic on GcPtr directly
    let lowbit_ir = {
        let mut b = FunctionBuilder::new("boxed_fib_lowbit", &[Type::I64], Some(Type::I64));
        let entry = b.entry_block();
        let n = b.block_param(entry, 0);
        let alloc_fn = b.declare_func("alloc", Signature { params: vec![], ret: Some(Type::GcPtr) });
        let field_off = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);

        let box_a = b.call(alloc_fn, &[]).unwrap();
        let zero = b.iconst(Type::I64, 0);
        let addr_a = b.add(box_a, field_off);
        b.store(zero, addr_a, 0);

        let box_b = b.call(alloc_fn, &[]).unwrap();
        let one = b.iconst(Type::I64, 1);
        let addr_b = b.add(box_b, field_off);
        b.store(one, addr_b, 0);

        let loop_bb = b.create_block(&[Type::I64, Type::GcPtr, Type::GcPtr]);
        let exit_bb = b.create_block(&[Type::GcPtr]);
        let i_zero = b.iconst(Type::I64, 0);
        b.jump(loop_bb, &[i_zero, box_a, box_b]);

        b.switch_to_block(loop_bb);
        let i = b.block_param(loop_bb, 0);
        let a_box = b.block_param(loop_bb, 1);
        let b_box = b.block_param(loop_bb, 2);
        let cond = b.icmp(dynir::CmpOp::Sge, i, n);
        let body_bb = b.create_block(&[]);
        b.br_if(cond, exit_bb, &[a_box], body_bb, &[]);

        b.switch_to_block(body_bb);
        // *** KEY: add directly on GcPtr — no payload extraction ***
        let a_addr = b.add(a_box, field_off);
        let a_val = b.load(Type::I64, a_addr, 0);
        let b_addr = b.add(b_box, field_off);
        let b_val = b.load(Type::I64, b_addr, 0);
        let tmp = b.add(a_val, b_val);
        let new_box = b.call(alloc_fn, &[]).unwrap();
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
        b.build()
    };

    // NaN-box version: must extract payload before arithmetic
    let nanbox_ir = {
        let mut b = FunctionBuilder::new("boxed_fib_nanbox", &[Type::I64], Some(Type::I64));
        let entry = b.entry_block();
        let n = b.block_param(entry, 0);
        let alloc_fn = b.declare_func("alloc", Signature { params: vec![], ret: Some(Type::GcPtr) });
        let field_off = b.iconst(Type::I64, PAIR_TYPE.value_field_offset(0) as i64);

        let box_a = b.call(alloc_fn, &[]).unwrap();
        let zero = b.iconst(Type::I64, 0);
        let raw_a = b.payload(box_a);
        let addr_a = b.add(raw_a, field_off);
        b.store(zero, addr_a, 0);

        let box_b = b.call(alloc_fn, &[]).unwrap();
        let one = b.iconst(Type::I64, 1);
        let raw_b = b.payload(box_b);
        let addr_b = b.add(raw_b, field_off);
        b.store(one, addr_b, 0);

        let loop_bb = b.create_block(&[Type::I64, Type::GcPtr, Type::GcPtr]);
        let exit_bb = b.create_block(&[Type::GcPtr]);
        let i_zero = b.iconst(Type::I64, 0);
        b.jump(loop_bb, &[i_zero, box_a, box_b]);

        b.switch_to_block(loop_bb);
        let i = b.block_param(loop_bb, 0);
        let a_box = b.block_param(loop_bb, 1);
        let b_box = b.block_param(loop_bb, 2);
        let cond = b.icmp(dynir::CmpOp::Sge, i, n);
        let body_bb = b.create_block(&[]);
        b.br_if(cond, exit_bb, &[a_box], body_bb, &[]);

        b.switch_to_block(body_bb);
        // *** KEY: payload() extracts raw pointer before arithmetic ***
        let a_raw = b.payload(a_box);
        let a_addr = b.add(a_raw, field_off);
        let a_val = b.load(Type::I64, a_addr, 0);
        let b_raw = b.payload(b_box);
        let b_addr = b.add(b_raw, field_off);
        let b_val = b.load(Type::I64, b_addr, 0);
        let tmp = b.add(a_val, b_val);
        let new_box = b.call(alloc_fn, &[]).unwrap();
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
        b.build()
    };

    println!("=== LowBit<3> (tagged pointer) ===");
    println!("{lowbit_ir}");
    println!("=== NaN-Boxing ===");
    println!("{nanbox_ir}");
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
