use crate::{call_jit, JitFunction};
use dynir::builder::FunctionBuilder;
use dynir::ir::*;
use dynir::types::Type;

fn run_jit(func: &dynir::Function, args: &[u64]) -> u64 {
    let jit = JitFunction::compile(func, &[]);
    unsafe { call_jit(jit.as_ptr(), args) }
}

// ── Phase 1: return_const ──────────────────────────────────────────

#[test]
fn return_const() {
    let mut b = FunctionBuilder::new("ret42", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 42);
    b.ret(v);
    let func = b.build();
    assert_eq!(run_jit(&func, &[]), 42);
}

#[test]
fn return_const_i32() {
    let mut b = FunctionBuilder::new("ret99", &[], Some(Type::I32));
    let v = b.iconst(Type::I32, 99);
    b.ret(v);
    let func = b.build();
    assert_eq!(run_jit(&func, &[]) as i32, 99);
}

#[test]
fn return_zero() {
    let mut b = FunctionBuilder::new("ret0", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 0);
    b.ret(v);
    let func = b.build();
    assert_eq!(run_jit(&func, &[]), 0);
}

#[test]
fn return_large_const() {
    let mut b = FunctionBuilder::new("retlarge", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 0x1234_5678_9ABC_DEF0u64 as i64);
    b.ret(v);
    let func = b.build();
    assert_eq!(run_jit(&func, &[]), 0x1234_5678_9ABC_DEF0);
}

// ── Phase 2: Arithmetic + args ─────────────────────────────────────

#[test]
fn add_two() {
    let mut b = FunctionBuilder::new("add", &[Type::I32, Type::I32], Some(Type::I32));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let sum = b.add(a, bb);
    b.ret(sum);
    let func = b.build();
    assert_eq!(run_jit(&func, &[10, 32]) as i32, 42);
}

#[test]
fn arithmetic() {
    let mut b = FunctionBuilder::new("calc", &[Type::I32, Type::I32], Some(Type::I32));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let prod = b.mul(a, bb);
    let sum = b.add(prod, a);
    b.ret(sum);
    let func = b.build();
    // 5 * 7 + 5 = 40
    assert_eq!(run_jit(&func, &[5, 7]) as i32, 40);
}

#[test]
fn sub_mul() {
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
    assert_eq!(run_jit(&func, &[10, 3]), 30);
}

#[test]
fn sdiv_udiv() {
    let mut b = FunctionBuilder::new("sdiv", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let q = b.sdiv(a, bb);
    b.ret(q);
    let func = b.build();
    assert_eq!(run_jit(&func, &[(-10i64) as u64, 3]), (-3i64) as u64);

    let mut b = FunctionBuilder::new("udiv", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let q = b.udiv(a, bb);
    b.ret(q);
    let func = b.build();
    assert_eq!(run_jit(&func, &[100, 5]), 20);
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
    assert_eq!(run_jit(&func, &[0xFF, 0x0F]), 0x0F);

    let mut b = FunctionBuilder::new("or", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.or(a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[0xF0, 0x0F]), 0xFF);

    let mut b = FunctionBuilder::new("xor", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let r = b.xor(a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[0xFF, 0xFF]), 0);
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
    assert_eq!(run_jit(&func, &[1, 4]), 16);
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
    let neg_eight = (-8i64) as u64;
    let result = run_jit(&func, &[neg_eight, 1]) as i64;
    assert_eq!(result, -4);
}

#[test]
fn unary_neg() {
    let mut b = FunctionBuilder::new("neg", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.neg(a);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[5]), (-5i64) as u64);
}

// ── Phase 3: Control flow ──────────────────────────────────────────

#[test]
fn if_else() {
    // if arg0 > arg1 then arg0 else arg1
    let wat = r#"(module
        (func (export "max") (param i32) (param i32) (result i32)
            local.get 0
            local.get 1
            i32.gt_s
            if (result i32)
                local.get 0
            else
                local.get 1
            end))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[10, 20]) as i32, 20);
    assert_eq!(run_jit(&func, &[30, 20]) as i32, 30);
}

#[test]
fn diamond_br_if() {
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
    assert_eq!(run_jit(&func, &[1]), 1);
    assert_eq!(run_jit(&func, &[0]), 2);
}

// ── Phase 4: Loops + spilling ──────────────────────────────────────

#[test]
fn simple_loop() {
    let wat = r#"(module
        (func (export "sum") (param $n i32) (result i32)
            (local $i i32)
            (local $acc i32)
            i32.const 0
            local.set $i
            i32.const 0
            local.set $acc
            block $exit
                loop $loop
                    local.get $i
                    local.get $n
                    i32.ge_s
                    br_if $exit
                    local.get $acc
                    local.get $i
                    i32.add
                    local.set $acc
                    local.get $i
                    i32.const 1
                    i32.add
                    local.set $i
                    br $loop
                end
            end
            local.get $acc))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[10]) as i32, 45);
    assert_eq!(run_jit(&func, &[0]) as i32, 0);
    assert_eq!(run_jit(&func, &[100]) as i32, 4950);
}

#[test]
fn fibonacci() {
    let wat = r#"(module
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
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[0]) as i32, 0);
    assert_eq!(run_jit(&func, &[1]) as i32, 1);
    assert_eq!(run_jit(&func, &[10]) as i32, 55);
    assert_eq!(run_jit(&func, &[20]) as i32, 6765);
}

#[test]
fn factorial() {
    let wat = r#"(module
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
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[1]) as i32, 1);
    assert_eq!(run_jit(&func, &[5]) as i32, 120);
    assert_eq!(run_jit(&func, &[10]) as i32, 3628800);
}

#[test]
fn nested_if() {
    let wat = r#"(module
        (func (export "clamp") (param $x i32) (param $lo i32) (param $hi i32) (result i32)
            local.get $x
            local.get $lo
            i32.lt_s
            if (result i32)
                local.get $lo
            else
                local.get $x
                local.get $hi
                i32.gt_s
                if (result i32)
                    local.get $hi
                else
                    local.get $x
                end
            end))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[5, 0, 10]) as i32, 5);
    assert_eq!(run_jit(&func, &[(-5i32) as u64, 0, 10]) as i32, 0);
    assert_eq!(run_jit(&func, &[15, 0, 10]) as i32, 10);
}

#[test]
fn void_if() {
    let wat = r#"(module
        (func (export "abs") (param $x i32) (result i32)
            local.get $x
            i32.const 0
            i32.lt_s
            if
                i32.const 0
                local.get $x
                i32.sub
                local.set $x
            end
            local.get $x))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[5]) as i32, 5);
    assert_eq!(run_jit(&func, &[(-5i32) as u64 & 0xFFFFFFFF]) as i32, 5);
}

#[test]
fn local_tee() {
    let wat = r#"(module
        (func (export "test") (param $x i32) (result i32)
            local.get $x
            i32.const 10
            i32.add
            local.tee $x
            local.get $x
            i32.add))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(run_jit(&func, &[5]) as i32, 30);
}

// ── Phase 5: 64-bit ────────────────────────────────────────────────

#[test]
fn i64_arithmetic() {
    let wat = r#"(module
        (func (export "add64") (param i64) (param i64) (result i64)
            local.get 0
            local.get 1
            i64.add))"#;
    let wasm = wat::parse_str(wat).expect("parse WAT");
    let (func, _) = wasm2dynir::translate_wasm(&wasm).expect("translate");
    assert_eq!(
        run_jit(&func, &[1_000_000_000_000, 2_000_000_000_000]) as i64,
        3_000_000_000_000
    );
}

// ── Phase 6: Floats ────────────────────────────────────────────────

#[test]
fn float_arithmetic() {
    let mut b = FunctionBuilder::new("fadd", &[Type::F64, Type::F64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let sum = b.fadd(a, bb);
    b.ret(sum);
    let func = b.build();
    let result = f64::from_bits(run_jit(&func, &[1.5f64.to_bits(), 2.5f64.to_bits()]));
    assert_eq!(result, 4.0);
}

#[test]
fn float_sub_mul_div() {
    let mut b = FunctionBuilder::new("fops", &[Type::F64, Type::F64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let diff = b.fsub(a, bb);
    let prod = b.fmul(diff, bb);
    let quot = b.fdiv(prod, a);
    b.ret(quot);
    let func = b.build();
    let result = f64::from_bits(run_jit(
        &func,
        &[10.0f64.to_bits(), 3.0f64.to_bits()],
    ));
    assert!((result - 2.1).abs() < 1e-10);
}

#[test]
fn float_neg() {
    let mut b = FunctionBuilder::new("fneg", &[Type::F64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.fneg(a);
    b.ret(r);
    let func = b.build();
    let result = f64::from_bits(run_jit(&func, &[3.0f64.to_bits()]));
    assert_eq!(result, -3.0);
}

#[test]
fn int_to_float_and_back() {
    let mut b = FunctionBuilder::new("itof", &[Type::I64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.int_to_float(a);
    b.ret(r);
    let func = b.build();
    let result = f64::from_bits(run_jit(&func, &[42]));
    assert_eq!(result, 42.0);

    let mut b = FunctionBuilder::new("ftoi", &[Type::F64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.float_to_int(a);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[42.9f64.to_bits()]), 42);
}

#[test]
fn f64_const() {
    let mut b = FunctionBuilder::new("fconst", &[], Some(Type::F64));
    let v = b.f64const(3.14);
    b.ret(v);
    let func = b.build();
    let result = f64::from_bits(run_jit(&func, &[]));
    assert!((result - 3.14).abs() < 1e-10);
}

// ── Conversions ────────────────────────────────────────────────────

#[test]
fn sext_zext_trunc() {
    // sext i8 -1 -> i64 -1
    let mut b = FunctionBuilder::new("sext", &[Type::I8], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.sext(a, Type::I64);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[0xFF]), (-1i64) as u64);

    // zext i8 0xFF -> i64 255
    let mut b = FunctionBuilder::new("zext", &[Type::I8], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.zext(a, Type::I64);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[0xFF]), 0xFF);

    // trunc i64 -> i8
    let mut b = FunctionBuilder::new("trunc", &[Type::I64], Some(Type::I8));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let r = b.trunc(a, Type::I8);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[0x1234]), 0x34);
}

// ── Select ─────────────────────────────────────────────────────────

#[test]
fn select_true_false() {
    let mut b = FunctionBuilder::new("sel", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let one = b.iconst(Type::I8, 1);
    let r = b.select(one, a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[10, 20]), 10);

    let mut b = FunctionBuilder::new("sel", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bb = b.block_param(entry, 1);
    let zero = b.iconst(Type::I8, 0);
    let r = b.select(zero, a, bb);
    b.ret(r);
    let func = b.build();
    assert_eq!(run_jit(&func, &[10, 20]), 20);
}

// ── Comparison ─────────────────────────────────────────────────────

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
        run_jit(&func, &[a as u64, b_val as u64])
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

// ── Switch ─────────────────────────────────────────────────────────

#[test]
fn switch_dispatch() {
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
    assert_eq!(run_jit(&func, &[1]), 10);
    assert_eq!(run_jit(&func, &[2]), 20);
    assert_eq!(run_jit(&func, &[3]), 99);
}

// ── Loop with dynir builder ────────────────────────────────────────

#[test]
fn loop_sum_dynir() {
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
    assert_eq!(run_jit(&func, &[10]), 45);
    assert_eq!(run_jit(&func, &[0]), 0);
}

#[test]
fn as_fib() {
    let wasm = include_bytes!("../../wasm2dynir/as-programs/fib.wasm");
    let (func, _) = wasm2dynir::translate_wasm(wasm).expect("translate");
    assert_eq!(run_jit(&func, &[0]) as i32, 0);
    assert_eq!(run_jit(&func, &[1]) as i32, 1);
    assert_eq!(run_jit(&func, &[10]) as i32, 55);
    assert_eq!(run_jit(&func, &[20]) as i32, 6765);
    assert_eq!(run_jit(&func, &[30]) as i32, 832040);
}

#[test]
fn as_collatz() {
    let wasm = include_bytes!("../../wasm2dynir/as-programs/collatz.wasm");
    let (func, _) = wasm2dynir::translate_wasm(wasm).expect("translate");
    assert_eq!(run_jit(&func, &[1]) as i32, 0);
    assert_eq!(run_jit(&func, &[2]) as i32, 1);
    assert_eq!(run_jit(&func, &[3]) as i32, 7);
    assert_eq!(run_jit(&func, &[6]) as i32, 8);
    assert_eq!(run_jit(&func, &[27]) as i32, 111);
}

#[test]
fn as_gcd() {
    let wasm = include_bytes!("../../wasm2dynir/as-programs/gcd.wasm");
    let (func, _) = wasm2dynir::translate_wasm(wasm).expect("translate");
    assert_eq!(run_jit(&func, &[12, 8]) as i32, 4);
    assert_eq!(run_jit(&func, &[100, 75]) as i32, 25);
    assert_eq!(run_jit(&func, &[17, 13]) as i32, 1);
    assert_eq!(run_jit(&func, &[0, 5]) as i32, 5);
}

#[test]
fn as_power() {
    let wasm = include_bytes!("../../wasm2dynir/as-programs/power.wasm");
    let (func, _) = wasm2dynir::translate_wasm(wasm).expect("translate");
    assert_eq!(run_jit(&func, &[2, 0]) as i64, 1);
    assert_eq!(run_jit(&func, &[2, 10]) as i64, 1024);
    assert_eq!(run_jit(&func, &[3, 5]) as i64, 243);
    assert_eq!(run_jit(&func, &[10, 9]) as i64, 1_000_000_000);
}

#[test]
fn as_primes() {
    let wasm = include_bytes!("../../wasm2dynir/as-programs/primes.wasm");
    let (func, _) = wasm2dynir::translate_wasm(wasm).expect("translate");
    assert_eq!(run_jit(&func, &[1]) as i32, 0);
    assert_eq!(run_jit(&func, &[10]) as i32, 4);
    assert_eq!(run_jit(&func, &[100]) as i32, 25);
    assert_eq!(run_jit(&func, &[1000]) as i32, 168);
}

#[test]
fn fib_loop_dynir() {
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
    assert_eq!(run_jit(&func, &[0]), 0);
    assert_eq!(run_jit(&func, &[1]), 1);
    assert_eq!(run_jit(&func, &[10]), 55);
    assert_eq!(run_jit(&func, &[20]), 6765);
}

#[test]
fn bench_primes() {
    let wasm = include_bytes!("../../wasm2dynir/as-programs/primes.wasm");
    let (func, _) = wasm2dynir::translate_wasm(wasm).expect("translate");

    // JIT compile
    let start = std::time::Instant::now();
    let jit = crate::JitFunction::compile(&func, &[]);
    let compile_time = start.elapsed();

    // JIT run
    let start = std::time::Instant::now();
    let jit_result = unsafe { crate::call_jit(jit.as_ptr(), &[10000]) } as i32;
    let jit_time = start.elapsed();

    // Interpreter run
    use dynir::interp::*;
    use dynvalue::LowBit;
    let interp = Interpreter::<LowBit<3>>::new(&func);
    let start = std::time::Instant::now();
    let interp_result = match interp.run(&[10000]).unwrap() {
        InterpResult::Value(v) => v as i32,
        other => panic!("{:?}", other),
    };
    let interp_time = start.elapsed();

    assert_eq!(jit_result, interp_result);
    eprintln!("count_primes(10000) = {}", jit_result);
    eprintln!("  compile:     {:?}", compile_time);
    eprintln!("  JIT run:     {:?}", jit_time);
    eprintln!("  interp run:  {:?}", interp_time);
    eprintln!("  speedup:     {:.1}x", interp_time.as_secs_f64() / jit_time.as_secs_f64());
}

#[test]
fn test_payload_nanbox() {
    use dynir::builder::FunctionBuilder;
    use dynir::types::Type;
    let mut b = FunctionBuilder::new("test", &[Type::I64], Some(Type::I64));
    let x = b.block_param(b.entry_block(), 0);
    let p = b.payload(x);
    b.ret(p);
    let func = b.build();
    let jit = JitFunction::compile(&func, &[]);
    let input = 0x7FFC_0001_0000_1234u64;
    let result = unsafe { call_jit(jit.as_ptr(), &[input]) };
    assert_eq!(result, 0x0000_0001_0000_1234u64);
}

#[test]
fn test_payload_then_load() {
    use dynir::builder::FunctionBuilder;
    use dynir::types::Type;
    // fn(ptr_nanbox: I64) -> I64
    // result = Load(I64, Payload(ptr_nanbox), 0)
    let mut b = FunctionBuilder::new("test", &[Type::I64], Some(Type::I64));
    let x = b.block_param(b.entry_block(), 0);
    let p = b.payload(x);
    let loaded = b.load(Type::I64, p, 0);
    b.ret(loaded);
    let func = b.build();
    let jit = JitFunction::compile(&func, &[]);

    // Allocate a u64 on the heap, encode as NanBox TAG_PTR
    let val: Box<u64> = Box::new(0xDEAD_BEEF_CAFE_BABEu64);
    let ptr = Box::into_raw(val) as u64;
    let nanbox = 0x7FFC_0000_0000_0000u64 | (ptr & 0x0000_FFFF_FFFF_FFFF);
    let result = unsafe { call_jit(jit.as_ptr(), &[nanbox]) };
    assert_eq!(result, 0xDEAD_BEEF_CAFE_BABEu64);
    unsafe { drop(Box::from_raw(ptr as *mut u64)); }
}

#[test]
fn test_extern_with_payload() {
    use dynir::builder::FunctionBuilder;
    use dynir::types::{Type, Signature};

    // fn(nanbox: I64) -> I64
    // p = Payload(nanbox)
    // result = call double_extern(p)
    extern "C" fn double_val(x: u64) -> u64 { x * 2 }

    let mut b = FunctionBuilder::new("test", &[Type::I64], Some(Type::I64));
    let fref = b.declare_func("double", Signature { params: vec![Type::I64], ret: Some(Type::I64) });
    let x = b.block_param(b.entry_block(), 0);
    let p = b.payload(x);
    let result = b.call(fref, &[p]).unwrap();
    b.ret(result);
    let func = b.build();
    let externs = vec![double_val as *const u8];
    let jit = JitFunction::compile(&func, &externs);
    let input = 0x7FFC_0000_0000_0005u64; // payload = 5
    let result = unsafe { call_jit(jit.as_ptr(), &[input]) };
    assert_eq!(result, 10);
}

#[test]
fn test_extern_two_params() {
    use dynir::builder::FunctionBuilder;
    use dynir::types::{Type, Signature};

    // fn(a: I64, b: I64) -> I64
    // Call extern sub(a, b)
    extern "C" fn sub_fn(a: u64, b: u64) -> u64 { a.wrapping_sub(b) }

    let mut b = FunctionBuilder::new("test", &[Type::I64, Type::I64], Some(Type::I64));
    let fref = b.declare_func("sub", Signature { params: vec![Type::I64, Type::I64], ret: Some(Type::I64) });
    let a = b.block_param(b.entry_block(), 0);
    let bv = b.block_param(b.entry_block(), 1);
    let result = b.call(fref, &[a, bv]).unwrap();
    b.ret(result);
    let func = b.build();
    let externs = vec![sub_fn as *const u8];
    let jit = JitFunction::compile(&func, &externs);
    let result = unsafe { call_jit(jit.as_ptr(), &[10, 3]) };
    assert_eq!(result, 7);
}

#[test]
fn test_extern_skip_first_param() {
    use dynir::builder::FunctionBuilder;
    use dynir::types::{Type, Signature};

    // fn(unused: I64, a: I64, b: I64) -> I64
    // Call extern sub(a, b)  -- skipping param 0
    extern "C" fn sub_fn(a: u64, b: u64) -> u64 { a.wrapping_sub(b) }

    let mut b = FunctionBuilder::new("test", &[Type::I64, Type::I64, Type::I64], Some(Type::I64));
    let fref = b.declare_func("sub", Signature { params: vec![Type::I64, Type::I64], ret: Some(Type::I64) });
    let _unused = b.block_param(b.entry_block(), 0);
    let a = b.block_param(b.entry_block(), 1);
    let bv = b.block_param(b.entry_block(), 2);
    let result = b.call(fref, &[a, bv]).unwrap();
    b.ret(result);
    let func = b.build();
    let externs = vec![sub_fn as *const u8];
    let jit = JitFunction::compile(&func, &externs);
    // Pass unused=99, a=10, b=3
    let result = unsafe { call_jit(jit.as_ptr(), &[99, 10, 3]) };
    assert_eq!(result, 7);
}

#[test]
fn test_branch_preserves_params() {
    // Mimics: fn(closure: I64, n: I64) -> I64
    //   result = call extern_check(n)
    //   if result == 1 then return 42
    //   else return call extern_identity(n)  <-- n must survive the branch
    use dynir::builder::FunctionBuilder;
    use dynir::types::{Type, Signature};
    use dynir::ir::CmpOp;

    extern "C" fn check_fn(x: u64) -> u64 { if x > 5 { 1 } else { 0 } }
    extern "C" fn identity_fn(x: u64) -> u64 { x }

    let mut b = FunctionBuilder::new("test", &[Type::I64, Type::I64], Some(Type::I64));
    let check = b.declare_func("check", Signature { params: vec![Type::I64], ret: Some(Type::I64) });
    let ident = b.declare_func("identity", Signature { params: vec![Type::I64], ret: Some(Type::I64) });

    let closure = b.block_param(b.entry_block(), 0);
    let n = b.block_param(b.entry_block(), 1);

    // Call extern check(n)
    let result = b.call(check, &[n]).unwrap();
    let one = b.iconst(Type::I64, 1);
    let cond = b.icmp(CmpOp::Eq, result, one);

    // Create then/else blocks with 2 params each (closure, n)
    let then_block = b.create_block(&[Type::I64, Type::I64]);
    let else_block = b.create_block(&[Type::I64, Type::I64]);

    b.br_if(cond, then_block, &[closure, n], else_block, &[closure, n]);

    // Then block: return 42
    b.switch_to_block(then_block);
    let val42 = b.iconst(Type::I64, 42);
    b.ret(val42);

    // Else block: return identity(n)
    b.switch_to_block(else_block);
    let else_n = b.block_param(else_block, 1);
    let else_result = b.call(ident, &[else_n]).unwrap();
    b.ret(else_result);

    let func = b.build();
    let externs = vec![check_fn as *const u8, identity_fn as *const u8];
    let jit = JitFunction::compile(&func, &externs);

    // n=10 > 5, so check returns 1, should go to then_block and return 42
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[99, 10]) }, 42);
    // n=3 <= 5, so check returns 0, should go to else_block and return 3
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[99, 3]) }, 3);
}

#[test]
fn test_simple_branch_params_no_extern() {
    // fn(a: I64, b: I64) -> I64
    // if a == 0 then return 42 else return b
    // No extern calls - tests pure branching with params
    use dynir::builder::FunctionBuilder;
    use dynir::types::Type;
    use dynir::ir::CmpOp;

    let mut b = FunctionBuilder::new("test", &[Type::I64, Type::I64], Some(Type::I64));
    let a = b.block_param(b.entry_block(), 0);
    let bv = b.block_param(b.entry_block(), 1);

    let zero = b.iconst(Type::I64, 0);
    let cond = b.icmp(CmpOp::Eq, a, zero);

    let then_block = b.create_block(&[Type::I64, Type::I64]);
    let else_block = b.create_block(&[Type::I64, Type::I64]);
    b.br_if(cond, then_block, &[a, bv], else_block, &[a, bv]);

    b.switch_to_block(then_block);
    let val42 = b.iconst(Type::I64, 42);
    b.ret(val42);

    b.switch_to_block(else_block);
    let else_b = b.block_param(else_block, 1);
    b.ret(else_b);

    let func = b.build();
    let jit = JitFunction::compile(&func, &[]);

    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[0, 99]) }, 42); // a==0 → return 42
    assert_eq!(unsafe { call_jit(jit.as_ptr(), &[1, 77]) }, 77); // a!=0 → return b=77
}
