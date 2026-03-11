use crate::builder::FunctionBuilder;
use crate::interp::*;
use crate::ir::*;
use crate::types::{Signature, Type};

fn run_simple(func: &Function, args: &[u64]) -> u64 {
    let interp = Interpreter::new(func);
    match interp.run(args).unwrap() {
        InterpResult::Value(v) => v,
        other => panic!("expected Value, got {:?}", other),
    }
}

// ---- Arithmetic ----

#[test]
fn return_const() {
    let mut b = FunctionBuilder::new("ret42", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 42);
    b.ret(v);
    let func = b.build();
    assert_eq!(run_simple(&func, &[]), 42);
}

#[test]
fn add_sub_mul() {
    let mut b = FunctionBuilder::new("arith", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let sum = b.add(a, bb);
    let diff = b.sub(sum, bb);
    let prod = b.mul(diff, bb);
    b.ret(prod);
    let func = b.build();
    // (10 + 3 - 3) * 3 = 30
    assert_eq!(run_simple(&func, &[10, 3]), 30);
}

#[test]
fn sdiv_udiv() {
    // sdiv: -10 / 3 = -3
    let mut b = FunctionBuilder::new("sdiv", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let q = b.sdiv(a, bb);
    b.ret(q);
    let func = b.build();
    assert_eq!(run_simple(&func, &[(-10i64) as u64, 3]), (-3i64) as u64);

    // udiv: large unsigned / 2
    let mut b = FunctionBuilder::new("udiv", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let q = b.udiv(a, bb);
    b.ret(q);
    let func = b.build();
    assert_eq!(run_simple(&func, &[100, 5]), 20);
}

#[test]
fn float_arithmetic() {
    let mut b = FunctionBuilder::new("fadd", &[Type::F64, Type::F64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let sum = b.fadd(a, bb);
    b.ret(sum);
    let func = b.build();
    let result = f64::from_bits(run_simple(&func, &[1.5f64.to_bits(), 2.5f64.to_bits()]));
    assert_eq!(result, 4.0);
}

#[test]
fn bitwise_ops() {
    let mut b = FunctionBuilder::new("bits", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.and(a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_simple(&func, &[0xFF, 0x0F]), 0x0F);

    let mut b = FunctionBuilder::new("or", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.or(a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_simple(&func, &[0xF0, 0x0F]), 0xFF);

    let mut b = FunctionBuilder::new("xor", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.xor(a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_simple(&func, &[0xFF, 0xFF]), 0);
}

#[test]
fn shift_ops() {
    let mut b = FunctionBuilder::new("shl", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.shl(a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_simple(&func, &[1, 4]), 16);
}

#[test]
fn unary_neg_not() {
    let mut b = FunctionBuilder::new("neg", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.neg(a);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_simple(&func, &[5]), (-5i64) as u64);

    let mut b = FunctionBuilder::new("fneg", &[Type::F64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.fneg(a);
    b.ret(r);
    let func = b.build();
    let result = f64::from_bits(run_simple(&func, &[3.0f64.to_bits()]));
    assert_eq!(result, -3.0);
}

// ---- Comparison ----

#[test]
fn icmp_all_ops() {
    fn test_icmp(op: CmpOp, a: i64, b_val: i64) -> u64 {
        let mut b = FunctionBuilder::new("cmp", &[Type::I64, Type::I64], Some(Type::I8));
        let entry = b.entry_block();
        let va = b.block_param(entry, 0);
        let vb = b.block_param(entry, 1);
        let r = b.icmp(op, va, vb);
        b.ret(r);
        let func = b.build();
        run_simple(&func, &[a as u64, b_val as u64])
    }

    assert_eq!(test_icmp(CmpOp::Eq, 5, 5), 1);
    assert_eq!(test_icmp(CmpOp::Eq, 5, 6), 0);
    assert_eq!(test_icmp(CmpOp::Ne, 5, 6), 1);
    assert_eq!(test_icmp(CmpOp::Slt, -1, 1), 1);
    assert_eq!(test_icmp(CmpOp::Sle, 5, 5), 1);
    assert_eq!(test_icmp(CmpOp::Sgt, 1, -1), 1);
    assert_eq!(test_icmp(CmpOp::Sge, 5, 5), 1);
    assert_eq!(test_icmp(CmpOp::Ult, 1, 2), 1);
    assert_eq!(test_icmp(CmpOp::Ule, 2, 2), 1);
    assert_eq!(test_icmp(CmpOp::Ugt, 3, 2), 1);
    assert_eq!(test_icmp(CmpOp::Uge, 2, 2), 1);
}

#[test]
fn fcmp() {
    let mut b = FunctionBuilder::new("fcmp", &[Type::F64, Type::F64], Some(Type::I8));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.fcmp(CmpOp::Slt, a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(
        run_simple(&func, &[1.0f64.to_bits(), 2.0f64.to_bits()]),
        1
    );
    assert_eq!(
        run_simple(&func, &[2.0f64.to_bits(), 1.0f64.to_bits()]),
        0
    );
}

// ---- Conversions ----

#[test]
fn sext_zext_trunc() {
    // sext i8 -1 -> i64 -1
    let mut b = FunctionBuilder::new("sext", &[Type::I8], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.sext(a, Type::I64);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_simple(&func, &[0xFF]), (-1i64) as u64); // 0xFF = -1 as i8

    // zext i8 0xFF -> i64 255
    let mut b = FunctionBuilder::new("zext", &[Type::I8], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.zext(a, Type::I64);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_simple(&func, &[0xFF]), 0xFF);

    // trunc i64 0x1234 -> i8 0x34
    let mut b = FunctionBuilder::new("trunc", &[Type::I64], Some(Type::I8));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.trunc(a, Type::I8);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_simple(&func, &[0x1234]), 0x34);
}

#[test]
fn int_to_float_and_back() {
    let mut b = FunctionBuilder::new("itof", &[Type::I64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.int_to_float(a);
    b.ret(r);
    let func = b.build();
    let result = f64::from_bits(run_simple(&func, &[42]));
    assert_eq!(result, 42.0);

    let mut b = FunctionBuilder::new("ftoi", &[Type::F64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.float_to_int(a);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_simple(&func, &[42.9f64.to_bits()]), 42);
}

#[test]
fn bitcast_i64_f64() {
    let mut b = FunctionBuilder::new("bc", &[Type::I64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.bitcast(a, Type::F64);
    b.ret(r);
    let func = b.build();
    let bits = 3.14f64.to_bits();
    assert_eq!(run_simple(&func, &[bits]), bits);
}

// ---- Control flow ----

#[test]
fn diamond_br_if() {
    // if arg != 0 then 1 else 2
    let mut b = FunctionBuilder::new("diamond", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);

    let then_bb = b.create_block(&[]);
    let else_bb = b.create_block(&[]);
    let merge = b.create_block(&[Type::I64]);
    let zero = b.iconst(Type::I64, 0);
    let cond = b.icmp(CmpOp::Ne, arg, zero);
    b.br_if(cond, then_bb, &[], else_bb, &[]);

    b.switch_to_block(then_bb);
    let one = b.iconst(Type::I64, 1);
    b.jump(merge, &[one]);

    b.switch_to_block(else_bb);
    let two = b.iconst(Type::I64, 2);
    b.jump(merge, &[two]);

    b.switch_to_block(merge);
    let result = b.block_param(merge, 0);
    b.ret(result);

    let func = b.build();
    assert_eq!(run_simple(&func, &[1]), 1);
    assert_eq!(run_simple(&func, &[0]), 2);
}

#[test]
fn loop_sum() {
    // sum 0..n
    let mut b = FunctionBuilder::new("sum", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let n = b.block_param(entry, 0);

    let loop_bb = b.create_block(&[Type::I64, Type::I64]); // (i, acc)
    let exit_bb = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    b.jump(loop_bb, &[zero, zero]);

    b.switch_to_block(loop_bb);
    let i = b.block_param(loop_bb, 0);
    let acc = b.block_param(loop_bb, 1);
    let cond = b.icmp(CmpOp::Slt, i, n);
    let new_acc = b.add(acc, i);
    let one = b.iconst(Type::I64, 1);
    let new_i = b.add(i, one);
    b.br_if(cond, loop_bb, &[new_i, new_acc], exit_bb, &[acc]);

    b.switch_to_block(exit_bb);
    let result = b.block_param(exit_bb, 0);
    b.ret(result);

    let func = b.build();
    assert_eq!(run_simple(&func, &[10]), 45); // 0+1+...+9
    assert_eq!(run_simple(&func, &[0]), 0);
}

#[test]
fn switch_dispatch() {
    // switch(val) { 1 => 10, 2 => 20, default => 99 }
    let mut b = FunctionBuilder::new("sw", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let val = b.block_param(entry, 0);

    let case1 = b.create_block(&[]);
    let case2 = b.create_block(&[]);
    let default = b.create_block(&[]);
    let merge = b.create_block(&[Type::I64]);

    b.switch(val, &[(1, case1, &[]), (2, case2, &[])], default, &[]);

    b.switch_to_block(case1);
    let ten = b.iconst(Type::I64, 10);
    b.jump(merge, &[ten]);

    b.switch_to_block(case2);
    let twenty = b.iconst(Type::I64, 20);
    b.jump(merge, &[twenty]);

    b.switch_to_block(default);
    let nn = b.iconst(Type::I64, 99);
    b.jump(merge, &[nn]);

    b.switch_to_block(merge);
    let r = b.block_param(merge, 0);
    b.ret(r);

    let func = b.build();
    assert_eq!(run_simple(&func, &[1]), 10);
    assert_eq!(run_simple(&func, &[2]), 20);
    assert_eq!(run_simple(&func, &[3]), 99);
}

// ---- Memory ----

#[test]
fn store_load_i64() {
    let mut b = FunctionBuilder::new("mem", &[Type::Ptr], Some(Type::I64));
    let entry = b.entry_block();
    let ptr = b.block_param(entry, 0);
    let val = b.iconst(Type::I64, 0xDEAD_BEEF);
    b.store(val, ptr, 0);
    let loaded = b.load(Type::I64, ptr, 0);
    b.ret(loaded);
    let func = b.build();

    let mut buf: [u64; 1] = [0];
    let ptr_val = buf.as_mut_ptr() as u64;
    assert_eq!(run_simple(&func, &[ptr_val]), 0xDEAD_BEEF);
}

#[test]
fn store_load_i32_with_offset() {
    let mut b = FunctionBuilder::new("mem32", &[Type::Ptr], Some(Type::I32));
    let entry = b.entry_block();
    let ptr = b.block_param(entry, 0);
    let val = b.iconst(Type::I32, 42);
    b.store(val, ptr, 4); // store at offset 4
    let loaded = b.load(Type::I32, ptr, 4);
    b.ret(loaded);
    let func = b.build();

    let mut buf: [u32; 4] = [0; 4];
    let ptr_val = buf.as_mut_ptr() as u64;
    assert_eq!(run_simple(&func, &[ptr_val]), 42);
}

// ---- Tagged values ----

#[test]
fn tagged_roundtrip() {
    let mut b = FunctionBuilder::new("tag", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let payload_in = b.block_param(entry, 0);

    let tagged = b.make_tagged(7, payload_in);
    let tag = b.tag_of(tagged);
    let payload_out = b.payload(tagged);

    // Return (tag << 48) | payload to verify both
    let tag_ext = b.zext(tag, Type::I64);
    let forty_eight = b.iconst(Type::I64, 48);
    let shifted = b.shl(tag_ext, forty_eight);
    let combined = b.or(shifted, payload_out);
    b.ret(combined);
    let func = b.build();

    let input = 0x1234_5678_9ABCu64;
    let result = run_simple(&func, &[input]);
    assert_eq!(result >> 48, 7);
    assert_eq!(result & ((1u64 << 48) - 1), input);
}

#[test]
fn is_tag_check() {
    let mut b = FunctionBuilder::new("istag", &[Type::I64], Some(Type::I8));
    let entry = b.entry_block();
    let val = b.block_param(entry, 0);
    let result = b.is_tag(val, 3);
    b.ret(result);
    let func = b.build();

    let tagged_3 = (3u64 << 48) | 42;
    let tagged_5 = (5u64 << 48) | 42;
    assert_eq!(run_simple(&func, &[tagged_3]), 1);
    assert_eq!(run_simple(&func, &[tagged_5]), 0);
}

// ---- Guard / Deopt ----

#[test]
fn guard_passes() {
    let mut b = FunctionBuilder::new("guard_ok", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);
    let deopt = b.create_deopt(100, "test guard");
    let one = b.iconst(Type::I8, 1);
    b.guard(one, deopt, &[arg]);
    b.ret(arg);
    let func = b.build();

    assert_eq!(run_simple(&func, &[42]), 42);
}

#[test]
fn guard_fails_deopt() {
    let mut b = FunctionBuilder::new("guard_fail", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);
    let deopt = b.create_deopt(200, "type check");
    let zero = b.iconst(Type::I8, 0);
    b.guard(zero, deopt, &[arg]);
    b.ret(arg);
    let func = b.build();

    let interp = Interpreter::new(&func);
    match interp.run(&[99]).unwrap() {
        InterpResult::Deopt {
            deopt_id,
            resume_point,
            live_values,
        } => {
            assert_eq!(deopt_id.index(), 0);
            assert_eq!(resume_point, 200);
            assert_eq!(live_values, vec![99]);
        }
        other => panic!("expected Deopt, got {:?}", other),
    }
}

// ---- Calls ----

#[test]
fn call_extern() {
    let mut b = FunctionBuilder::new("caller", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);
    let fref = b.declare_func(
        "double",
        Signature {
            params: vec![Type::I64],
            ret: Some(Type::I64),
        },
    );
    let result = b.call(fref, &[arg]).unwrap();
    b.ret(result);
    let func = b.build();

    let mut interp = Interpreter::new(&func);
    interp.bind(fref, |args| ExternCallResult::Value(Some(args[0] * 2)));
    match interp.run(&[21]).unwrap() {
        InterpResult::Value(v) => assert_eq!(v, 42),
        other => panic!("expected Value, got {:?}", other),
    }
}

#[test]
fn call_void_extern() {
    let mut b = FunctionBuilder::new("caller", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);
    let fref = b.declare_func(
        "side_effect",
        Signature {
            params: vec![Type::I64],
            ret: None,
        },
    );
    b.call(fref, &[arg]);
    b.ret(arg);
    let func = b.build();

    let mut interp = Interpreter::new(&func);
    interp.bind(fref, |_args| ExternCallResult::Value(None));
    assert_eq!(
        match interp.run(&[7]).unwrap() {
            InterpResult::Value(v) => v,
            other => panic!("expected Value, got {:?}", other),
        },
        7
    );
}

#[test]
fn call_indirect() {
    let mut b = FunctionBuilder::new("indirect", &[Type::Ptr], Some(Type::I64));
    let entry = b.entry_block();
    let callee = b.block_param(entry, 0);
    let arg = b.iconst(Type::I64, 10);
    let result = b.call_indirect(callee, &[arg], Some(Type::I64)).unwrap();
    b.ret(result);
    let func = b.build();

    let mut interp = Interpreter::new(&func);
    interp.bind_indirect(|_callee, args| ExternCallResult::Value(Some(args[0] + 5)));
    match interp.run(&[0xCAFE]).unwrap() {
        InterpResult::Value(v) => assert_eq!(v, 15),
        other => panic!("expected Value, got {:?}", other),
    }
}

// ---- Invoke ----

#[test]
fn invoke_normal_path() {
    let mut b = FunctionBuilder::new("inv", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);
    let fref = b.declare_func(
        "maybe_throw",
        Signature {
            params: vec![Type::I64],
            ret: Some(Type::I64),
        },
    );
    let normal = b.create_block(&[Type::I64]); // receives return value
    let exception = b.create_block(&[]);
    b.invoke(fref, &[arg], normal, &[], exception, &[]);

    b.switch_to_block(normal);
    let ret_val = b.block_param(normal, 0);
    b.ret(ret_val);

    b.switch_to_block(exception);
    let neg = b.iconst(Type::I64, -1i64);
    b.ret(neg);

    let func = b.build();

    let mut interp = Interpreter::new(&func);
    interp.bind(fref, |args| ExternCallResult::Value(Some(args[0] * 3)));
    match interp.run(&[14]).unwrap() {
        InterpResult::Value(v) => assert_eq!(v, 42),
        other => panic!("expected Value, got {:?}", other),
    }
}

#[test]
fn invoke_exception_path() {
    // Exception path: exception block receives no implicit exception value,
    // it just gets exception_args. We pass the arg through to verify control flow.
    let mut b = FunctionBuilder::new("inv_exc", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let arg = b.block_param(entry, 0);
    let fref = b.declare_func(
        "always_throw",
        Signature {
            params: vec![Type::I64],
            ret: Some(Type::I64),
        },
    );
    let normal = b.create_block(&[Type::I64]);
    let exception = b.create_block(&[Type::I64]); // receives arg via exception_args
    b.invoke(fref, &[arg], normal, &[], exception, &[arg]);

    b.switch_to_block(normal);
    let ret_val = b.block_param(normal, 0);
    b.ret(ret_val);

    b.switch_to_block(exception);
    let exc_val = b.block_param(exception, 0);
    // Return a sentinel to prove we took the exception path
    let sentinel = b.iconst(Type::I64, 999);
    let result = b.add(exc_val, sentinel);
    b.ret(result);

    let func = b.build();

    let mut interp = Interpreter::new(&func);
    interp.bind(fref, |_args| ExternCallResult::Exception(0));
    match interp.run(&[1]).unwrap() {
        // 1 (arg passed via exception_args) + 999 = 1000
        InterpResult::Value(v) => assert_eq!(v, 1000),
        other => panic!("expected Value, got {:?}", other),
    }
}

// ---- Overflow ----

#[test]
fn overflow_check_no_overflow() {
    let mut b = FunctionBuilder::new("ov", &[Type::I64, Type::I64], Some(Type::I8));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.overflow_check(OverflowOp::SAdd, a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_simple(&func, &[1, 2]), 0); // no overflow
}

#[test]
fn overflow_check_overflows() {
    let mut b = FunctionBuilder::new("ov", &[Type::I64, Type::I64], Some(Type::I8));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.overflow_check(OverflowOp::SAdd, a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_simple(&func, &[i64::MAX as u64, 1]), 1); // overflow
}

// ---- Select ----

#[test]
fn select_true() {
    let mut b = FunctionBuilder::new("sel", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let one = b.iconst(Type::I8, 1);
    let r = b.select(one, a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_simple(&func, &[10, 20]), 10);
}

#[test]
fn select_false() {
    let mut b = FunctionBuilder::new("sel", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let zero = b.iconst(Type::I8, 0);
    let r = b.select(zero, a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_simple(&func, &[10, 20]), 20);
}

// ---- Errors ----

#[test]
fn error_unreachable() {
    let mut b = FunctionBuilder::new("unreach", &[], Some(Type::I64));
    b.unreachable();
    let func = b.build();
    let interp = Interpreter::new(&func);
    match interp.run(&[]) {
        Err(InterpError::Unreachable) => {}
        other => panic!("expected Unreachable, got {:?}", other),
    }
}

#[test]
fn error_unbound_extern() {
    let mut b = FunctionBuilder::new("unbound", &[], Some(Type::I64));
    let fref = b.declare_func(
        "missing",
        Signature {
            params: vec![],
            ret: Some(Type::I64),
        },
    );
    let r = b.call(fref, &[]).unwrap();
    b.ret(r);
    let func = b.build();
    let interp = Interpreter::new(&func);
    match interp.run(&[]) {
        Err(InterpError::UnknownExternFunc(name)) => assert_eq!(name, "missing"),
        other => panic!("expected UnknownExternFunc, got {:?}", other),
    }
}

#[test]
fn error_divide_by_zero() {
    let mut b = FunctionBuilder::new("dbz", &[], Some(Type::I64));
    let a = b.iconst(Type::I64, 10);
    let zero = b.iconst(Type::I64, 0);
    let r = b.sdiv(a, zero);
    b.ret(r);
    let func = b.build();
    let interp = Interpreter::new(&func);
    match interp.run(&[]) {
        Err(InterpError::DivideByZero) => {}
        other => panic!("expected DivideByZero, got {:?}", other),
    }
}

#[test]
fn error_uncaught_exception() {
    let mut b = FunctionBuilder::new("uncaught", &[], Some(Type::I64));
    let fref = b.declare_func(
        "throws",
        Signature {
            params: vec![],
            ret: Some(Type::I64),
        },
    );
    let r = b.call(fref, &[]).unwrap();
    b.ret(r);
    let func = b.build();
    let mut interp = Interpreter::new(&func);
    interp.bind(fref, |_| ExternCallResult::Exception(42));
    match interp.run(&[]) {
        Err(InterpError::UncaughtException(v)) => assert_eq!(v, 42),
        other => panic!("expected UncaughtException, got {:?}", other),
    }
}

// ---- Integration ----

#[test]
fn fib_loop() {
    // fib(n): iterative fibonacci
    let mut b = FunctionBuilder::new("fib", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let n = b.block_param(entry, 0);

    let loop_bb = b.create_block(&[Type::I64, Type::I64, Type::I64]); // (i, a, b)
    let exit = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    let one = b.iconst(Type::I64, 1);
    b.jump(loop_bb, &[zero, zero, one]);

    b.switch_to_block(loop_bb);
    let i = b.block_param(loop_bb, 0);
    let a = b.block_param(loop_bb, 1);
    let fib_b = b.block_param(loop_bb, 2);
    let cond = b.icmp(CmpOp::Slt, i, n);
    let next = b.add(a, fib_b);
    let i_plus = b.add(i, one);
    b.br_if(cond, loop_bb, &[i_plus, fib_b, next], exit, &[a]);

    b.switch_to_block(exit);
    let result = b.block_param(exit, 0);
    b.ret(result);

    let func = b.build();
    assert_eq!(run_simple(&func, &[0]), 0);
    assert_eq!(run_simple(&func, &[1]), 1);
    assert_eq!(run_simple(&func, &[10]), 55);
    assert_eq!(run_simple(&func, &[20]), 6765);
}

#[test]
fn pic_add_tagged() {
    // Polymorphic inline cache: if both args are int-tagged (tag=1), fast path add.
    // Otherwise return -1 (slow path placeholder).
    let tag_int: u32 = 1;

    let mut b = FunctionBuilder::new("pic_add", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb_val = b.block_param(entry, 1);

    let fast = b.create_block(&[]);
    let slow = b.create_block(&[]);
    let merge = b.create_block(&[Type::I64]);

    let a_is_int = b.is_tag(a, tag_int);
    let b_is_int = b.is_tag(bb_val, tag_int);
    let both = b.and(a_is_int, b_is_int);
    b.br_if(both, fast, &[], slow, &[]);

    b.switch_to_block(fast);
    let pa = b.payload(a);
    let pb = b.payload(bb_val);
    let sum = b.add(pa, pb);
    let result = b.make_tagged(tag_int, sum);
    b.jump(merge, &[result]);

    b.switch_to_block(slow);
    let neg_one = b.iconst(Type::I64, -1i64);
    b.jump(merge, &[neg_one]);

    b.switch_to_block(merge);
    let r = b.block_param(merge, 0);
    b.ret(r);

    let func = b.build();

    // Both int-tagged: (tag=1, payload=10) + (tag=1, payload=32) = (tag=1, payload=42)
    let a_tagged = (1u64 << 48) | 10;
    let b_tagged = (1u64 << 48) | 32;
    let result = run_simple(&func, &[a_tagged, b_tagged]);
    assert_eq!(result >> 48, 1);
    assert_eq!(result & ((1u64 << 48) - 1), 42);

    // Mixed tags: slow path
    let a_str = (2u64 << 48) | 10;
    let result = run_simple(&func, &[a_str, b_tagged]);
    assert_eq!(result, (-1i64) as u64);
}

#[test]
fn sum_array_memory() {
    // Sum an array of i64 values using pointer arithmetic and load.
    let mut b = FunctionBuilder::new("sum_arr", &[Type::Ptr, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let base = b.block_param(entry, 0);
    let len = b.block_param(entry, 1);

    let loop_bb = b.create_block(&[Type::I64, Type::I64]); // (i, acc)
    let exit = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    b.jump(loop_bb, &[zero, zero]);

    b.switch_to_block(loop_bb);
    let i = b.block_param(loop_bb, 0);
    let acc = b.block_param(loop_bb, 1);
    let cond = b.icmp(CmpOp::Slt, i, len);

    let eight = b.iconst(Type::I64, 8);
    let offset = b.mul(i, eight);
    let addr = b.add(base, offset);
    let elem = b.load(Type::I64, addr, 0);
    let new_acc = b.add(acc, elem);
    let one = b.iconst(Type::I64, 1);
    let new_i = b.add(i, one);
    b.br_if(cond, loop_bb, &[new_i, new_acc], exit, &[acc]);

    b.switch_to_block(exit);
    let result = b.block_param(exit, 0);
    b.ret(result);

    let func = b.build();

    let data: Vec<u64> = vec![10, 20, 30, 40];
    let result = run_simple(&func, &[data.as_ptr() as u64, data.len() as u64]);
    assert_eq!(result, 100);
}

#[test]
fn bind_by_name() {
    let mut b = FunctionBuilder::new("caller", &[], Some(Type::I64));
    let fref = b.declare_func(
        "get_value",
        Signature {
            params: vec![],
            ret: Some(Type::I64),
        },
    );
    let r = b.call(fref, &[]).unwrap();
    b.ret(r);
    let func = b.build();

    let mut interp = Interpreter::new(&func);
    interp.bind_by_name("get_value", |_| ExternCallResult::Value(Some(123)));
    match interp.run(&[]).unwrap() {
        InterpResult::Value(v) => assert_eq!(v, 123),
        other => panic!("expected Value, got {:?}", other),
    }
}

#[test]
fn void_return() {
    let mut b = FunctionBuilder::new("void_fn", &[], None);
    b.ret_void();
    let func = b.build();
    let interp = Interpreter::new(&func);
    match interp.run(&[]).unwrap() {
        InterpResult::Void => {}
        other => panic!("expected Void, got {:?}", other),
    }
}

#[test]
fn i32_arithmetic() {
    let mut b = FunctionBuilder::new("i32add", &[Type::I32, Type::I32], Some(Type::I32));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let sum = b.add(a, bb);
    b.ret(sum);
    let func = b.build();
    // Test wrapping at 32 bits
    assert_eq!(run_simple(&func, &[0xFFFF_FFFF, 1]), 0);
}

#[test]
fn invoke_indirect_normal() {
    let mut b = FunctionBuilder::new("inv_ind", &[Type::Ptr], Some(Type::I64));
    let entry = b.entry_block();
    let callee = b.block_param(entry, 0);
    let arg = b.iconst(Type::I64, 5);
    let normal = b.create_block(&[Type::I64]);
    let exception = b.create_block(&[]);
    b.invoke_indirect(callee, &[arg], Some(Type::I64), normal, &[], exception, &[]);

    b.switch_to_block(normal);
    let ret = b.block_param(normal, 0);
    b.ret(ret);

    b.switch_to_block(exception);
    let neg = b.iconst(Type::I64, -1i64);
    b.ret(neg);

    let func = b.build();

    let mut interp = Interpreter::new(&func);
    interp.bind_indirect(|_callee, args| ExternCallResult::Value(Some(args[0] * 10)));
    match interp.run(&[0xBEEF]).unwrap() {
        InterpResult::Value(v) => assert_eq!(v, 50),
        other => panic!("expected Value, got {:?}", other),
    }
}

#[test]
fn guard_with_multiple_live_values() {
    let mut b = FunctionBuilder::new("guard_multi", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let deopt = b.create_deopt(42, "multi guard");
    let zero = b.iconst(Type::I8, 0);
    b.guard(zero, deopt, &[a, bb]);
    b.ret(a);
    let func = b.build();

    let interp = Interpreter::new(&func);
    match interp.run(&[10, 20]).unwrap() {
        InterpResult::Deopt { live_values, .. } => {
            assert_eq!(live_values, vec![10, 20]);
        }
        other => panic!("expected Deopt, got {:?}", other),
    }
}

#[test]
fn ashr_sign_extension() {
    let mut b = FunctionBuilder::new("ashr", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.ashr(a, bb);
    b.ret(r);
    let func = b.build();

    // Arithmetic right shift of negative number preserves sign
    let neg_eight = (-8i64) as u64;
    let result = run_simple(&func, &[neg_eight, 1]) as i64;
    assert_eq!(result, -4);
}

#[test]
fn float_sub_mul_div() {
    // (a - b) * b / a
    let mut b = FunctionBuilder::new("fops", &[Type::F64, Type::F64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let diff = b.fsub(a, bb);
    let prod = b.fmul(diff, bb);
    let quot = b.fdiv(prod, a);
    b.ret(quot);
    let func = b.build();

    let result = f64::from_bits(run_simple(
        &func,
        &[10.0f64.to_bits(), 3.0f64.to_bits()],
    ));
    assert!((result - 2.1).abs() < 1e-10);
}
