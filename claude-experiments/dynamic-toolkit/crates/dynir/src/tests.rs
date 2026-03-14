use crate::*;
use proptest::prelude::*;

// ── Unit tests ─────────────────────────────────────────────────

#[test]
fn test_return_const() {
    let mut b = FunctionBuilder::new("ret42", &[], Some(Type::I64));
    let v = b.iconst(Type::I64, 42);
    b.ret(v);
    let func = b.build();

    assert_eq!(func.blocks.len(), 1);
    assert_eq!(func.blocks[0].insts.len(), 1);
    verify(&func).unwrap();
    let s = func.to_string();
    assert!(s.contains("iconst.i64 42"));
    assert!(s.contains("ret v0"));
}

#[test]
fn test_add_two_params() {
    let mut b = FunctionBuilder::new("add", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);
    let sum = b.add(a, bv);
    b.ret(sum);
    let func = b.build();

    verify(&func).unwrap();
    assert!(func.to_string().contains("add v0, v1"));
}

#[test]
fn test_diamond_cfg() {
    let mut b = FunctionBuilder::new("diamond", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);

    let then_bb = b.create_block(&[Type::I64]);
    let else_bb = b.create_block(&[Type::I64]);
    let merge_bb = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    let cmp = b.icmp(CmpOp::Eq, x, zero);
    b.br_if(cmp, then_bb, &[x], else_bb, &[x]);

    b.switch_to_block(then_bb);
    let then_val = b.block_param(then_bb, 0);
    let one = b.iconst(Type::I64, 1);
    let result1 = b.add(then_val, one);
    b.jump(merge_bb, &[result1]);

    b.switch_to_block(else_bb);
    let else_val = b.block_param(else_bb, 0);
    let two = b.iconst(Type::I64, 2);
    let result2 = b.mul(else_val, two);
    b.jump(merge_bb, &[result2]);

    b.switch_to_block(merge_bb);
    let merged = b.block_param(merge_bb, 0);
    b.ret(merged);

    let func = b.build();
    verify(&func).unwrap();

    // Should have 4 blocks
    assert_eq!(func.blocks.len(), 4);
}

#[test]
fn test_loop() {
    // sum = 0; for i in 0..n: sum += i
    let mut b = FunctionBuilder::new("sum_to_n", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let n = b.block_param(entry, 0);

    let loop_bb = b.create_block(&[Type::I64, Type::I64]); // (i, sum)
    let exit_bb = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    b.jump(loop_bb, &[zero, zero]);

    b.switch_to_block(loop_bb);
    let i = b.block_param(loop_bb, 0);
    let sum = b.block_param(loop_bb, 1);
    let cmp = b.icmp(CmpOp::Slt, i, n);
    let body_bb = b.create_block(&[Type::I64, Type::I64]);
    b.br_if(cmp, body_bb, &[i, sum], exit_bb, &[sum]);

    b.switch_to_block(body_bb);
    let bi = b.block_param(body_bb, 0);
    let bsum = b.block_param(body_bb, 1);
    let new_sum = b.add(bsum, bi);
    let one = b.iconst(Type::I64, 1);
    let new_i = b.add(bi, one);
    b.jump(loop_bb, &[new_i, new_sum]);

    b.switch_to_block(exit_bb);
    let result = b.block_param(exit_bb, 0);
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
}

#[test]
fn test_tagged_value_ops() {
    let mut b = FunctionBuilder::new("add_tagged", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);

    let a_pay = b.payload(a);
    let b_pay = b.payload(bv);
    let sum = b.add(a_pay, b_pay);
    let result = b.make_tagged(1, sum);
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
    let s = func.to_string();
    assert!(s.contains("payload"));
    assert!(s.contains("make_tagged"));
}

#[test]
fn test_tag_check_with_branch() {
    let mut b = FunctionBuilder::new("check_tag", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let v = b.block_param(entry, 0);

    let fast = b.create_block(&[Type::I64]);
    let slow = b.create_block(&[Type::I64]);

    let is_int = b.is_tag(v, 1);
    b.br_if(is_int, fast, &[v], slow, &[v]);

    b.switch_to_block(fast);
    let fv = b.block_param(fast, 0);
    let pay = b.payload(fv);
    let result = b.make_tagged(1, pay);
    b.ret(result);

    b.switch_to_block(slow);
    let sv = b.block_param(slow, 0);
    b.ret(sv);

    let func = b.build();
    verify(&func).unwrap();
}

#[test]
fn test_call() {
    let mut b = FunctionBuilder::new("caller", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);

    let callee = b.declare_func(
        "double",
        Signature {
            params: vec![Type::I64],
            ret: Some(Type::I64),
        },
    );
    let result = b.call(callee, &[x]).unwrap();
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
    assert!(func.to_string().contains("call @f0"));
}

#[test]
fn test_void_function() {
    let mut b = FunctionBuilder::new("nop", &[], None);
    b.ret_void();
    let func = b.build();
    verify(&func).unwrap();
}

#[test]
fn test_select() {
    let mut b = FunctionBuilder::new("abs", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);
    let zero = b.iconst(Type::I64, 0);
    let neg = b.sub(zero, x);
    let is_neg = b.icmp(CmpOp::Slt, x, zero);
    let result = b.select(is_neg, neg, x);
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
}

#[test]
fn test_store_no_result() {
    let mut b = FunctionBuilder::new("writer", &[Type::Ptr, Type::I64], None);
    let entry = b.entry_block();
    let addr = b.block_param(entry, 0);
    let val = b.block_param(entry, 1);
    b.store(val, addr, 0);
    b.ret_void();

    let func = b.build();
    verify(&func).unwrap();
}

#[test]
fn test_load_store_roundtrip() {
    let mut b = FunctionBuilder::new("read_write", &[Type::Ptr], Some(Type::I64));
    let entry = b.entry_block();
    let addr = b.block_param(entry, 0);
    let val = b.iconst(Type::I64, 99);
    b.store(val, addr, 0);
    let loaded = b.load(Type::I64, addr, 0);
    b.ret(loaded);

    let func = b.build();
    verify(&func).unwrap();
}

#[test]
fn test_conversions() {
    let mut b = FunctionBuilder::new("conv", &[Type::I32], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);
    let wide = b.sext(x, Type::I64);
    b.ret(wide);

    let func = b.build();
    verify(&func).unwrap();
}

#[test]
fn test_display_output() {
    let mut b = FunctionBuilder::new("example", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);
    let sum = b.add(a, bv);
    b.ret(sum);

    let func = b.build();
    let s = func.to_string();
    assert!(s.contains("fn example(i64, i64) -> i64 {"));
    assert!(s.contains("bb0(v0: i64, v1: i64):"));
    assert!(s.contains("v2: i64 = add v0, v1"));
    assert!(s.contains("ret v2"));
}

#[test]
fn test_predecessors() {
    let mut b = FunctionBuilder::new("preds", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);

    let bb1 = b.create_block(&[Type::I64]);
    let bb2 = b.create_block(&[Type::I64]);

    let zero = b.iconst(Type::I64, 0);
    let cmp = b.icmp(CmpOp::Eq, x, zero);
    b.br_if(cmp, bb1, &[x], bb2, &[x]);

    b.switch_to_block(bb1);
    let v1 = b.block_param(bb1, 0);
    b.ret(v1);

    b.switch_to_block(bb2);
    let v2 = b.block_param(bb2, 0);
    b.ret(v2);

    let func = b.build();
    let preds = func.predecessors();
    assert!(preds[0].is_empty()); // entry has no preds
    assert_eq!(preds[1].len(), 1); // bb1 has entry as pred
    assert_eq!(preds[2].len(), 1); // bb2 has entry as pred
}

#[test]
fn test_multiple_cmp_ops() {
    let mut b = FunctionBuilder::new("cmps", &[Type::I64, Type::I64], Some(Type::I8));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);

    let _ = b.icmp(CmpOp::Eq, a, bv);
    let _ = b.icmp(CmpOp::Ne, a, bv);
    let _ = b.icmp(CmpOp::Slt, a, bv);
    let _ = b.icmp(CmpOp::Sle, a, bv);
    let _ = b.icmp(CmpOp::Sgt, a, bv);
    let _ = b.icmp(CmpOp::Sge, a, bv);
    let _ = b.icmp(CmpOp::Ult, a, bv);
    let _ = b.icmp(CmpOp::Ule, a, bv);
    let _ = b.icmp(CmpOp::Ugt, a, bv);
    let result = b.icmp(CmpOp::Uge, a, bv);
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();
}

#[test]
fn test_unreachable() {
    let mut b = FunctionBuilder::new("trap", &[], None);
    b.unreachable();
    let func = b.build();
    verify(&func).unwrap();
}

#[test]
fn test_float_ops() {
    let mut b = FunctionBuilder::new("float_math", &[Type::F64, Type::F64], Some(Type::F64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);
    let sum = b.fadd(a, bv);
    let diff = b.fsub(a, bv);
    let prod = b.fmul(sum, diff);
    let quot = b.fdiv(prod, bv);
    b.ret(quot);

    let func = b.build();
    verify(&func).unwrap();
}

#[test]
fn test_bitwise_ops() {
    let mut b = FunctionBuilder::new("bits", &[Type::I64, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);
    let x = b.and(a, bv);
    let y = b.or(x, bv);
    let z = b.xor(y, a);
    let w = b.shl(z, bv);
    let u = b.lshr(w, bv);
    let v = b.ashr(u, a);
    b.ret(v);

    let func = b.build();
    verify(&func).unwrap();
}

// ── Builder error tests ────────────────────────────────────────

#[test]
#[should_panic(expected = "integer binop requires matching int/ptr types")]
fn test_type_mismatch_add() {
    let mut b = FunctionBuilder::new("bad", &[Type::I32, Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    let bv = b.block_param(entry, 1);
    b.add(a, bv); // I32 + I64 = panic
}

#[test]
#[should_panic(expected = "integer binop requires matching int/ptr types")]
fn test_float_in_int_op() {
    let mut b = FunctionBuilder::new("bad", &[Type::F64], Some(Type::I64));
    let entry = b.entry_block();
    let a = b.block_param(entry, 0);
    b.add(a, a); // F64 in int op
}

#[test]
#[should_panic(expected = "return type mismatch")]
fn test_wrong_return_type() {
    let mut b = FunctionBuilder::new("bad", &[], Some(Type::I64));
    let v = b.iconst(Type::I32, 42);
    b.ret(v);
}

#[test]
#[should_panic(expected = "branch to bb1 arg count mismatch")]
fn test_branch_arg_count_mismatch() {
    let mut b = FunctionBuilder::new("bad", &[], None);
    let target = b.create_block(&[Type::I64]);
    b.jump(target, &[]); // target expects 1 arg
}

#[test]
#[should_panic(expected = "branch to bb1 arg 0 type mismatch")]
fn test_branch_arg_type_mismatch() {
    let mut b = FunctionBuilder::new("bad", &[], None);
    let target = b.create_block(&[Type::I64]);
    let v = b.iconst(Type::I32, 1);
    b.jump(target, &[v]); // I32 vs I64
}

#[test]
#[should_panic(expected = "block already has a terminator")]
fn test_double_terminator() {
    let mut b = FunctionBuilder::new("bad", &[], None);
    b.ret_void();
    b.ret_void();
}

#[test]
#[should_panic(expected = "current block bb0 must be terminated before switching")]
fn test_switch_without_terminator() {
    let mut b = FunctionBuilder::new("bad", &[], None);
    let bb1 = b.create_block(&[]);
    b.switch_to_block(bb1); // bb0 not terminated
}

#[test]
#[should_panic(expected = "select cond must be i8")]
fn test_select_bad_cond() {
    let mut b = FunctionBuilder::new("bad", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);
    b.select(x, x, x); // I64 as cond
}

#[test]
#[should_panic(expected = "icmp requires matching int types")]
fn test_icmp_float() {
    let mut b = FunctionBuilder::new("bad", &[Type::F64], Some(Type::I8));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);
    b.icmp(CmpOp::Eq, x, x); // F64 in icmp
}

#[test]
#[should_panic(expected = "block bb0 is not terminated")]
fn test_build_unterminated() {
    let b = FunctionBuilder::new("bad", &[], None);
    b.build();
}

// ── Verifier tests (catch issues in hand-built IR) ─────────────

#[test]
fn test_verify_catches_type_mismatch() {
    // Build valid IR then mangle it
    let mut b = FunctionBuilder::new("ok", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let x = b.block_param(entry, 0);
    let y = b.iconst(Type::I64, 1);
    let z = b.add(x, y);
    b.ret(z);
    let mut func = b.build();

    // Change y's type to I32 behind the builder's back
    func.value_types[1] = Type::I32;

    let errs = verify(&func).unwrap_err();
    assert!(!errs.is_empty());
}

// ── Property-based tests ───────────────────────────────────────

/// Decision-driven random function builder.
struct DecisionReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> DecisionReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn next_byte(&mut self) -> u8 {
        if self.data.is_empty() {
            return 0;
        }
        let v = self.data[self.pos % self.data.len()];
        self.pos += 1;
        v
    }

    fn pick<'b, T>(&mut self, choices: &'b [T]) -> &'b T {
        &choices[self.next_byte() as usize % choices.len()]
    }

    fn pick_index(&mut self, len: usize) -> usize {
        if len == 0 {
            0
        } else {
            self.next_byte() as usize % len
        }
    }
}

const ALL_INT_TYPES: [Type; 3] = [Type::I8, Type::I32, Type::I64];
const MAIN_TYPES: [Type; 4] = [Type::I64, Type::I32, Type::I8, Type::F64];

fn build_random_single_block(decisions: &[u8]) -> Function {
    let mut dr = DecisionReader::new(decisions);
    let n_params = (dr.next_byte() % 4) as usize;
    let params: Vec<Type> = (0..n_params).map(|_| *dr.pick(&MAIN_TYPES)).collect();
    let ret_ty = *dr.pick(&MAIN_TYPES);

    let mut b = FunctionBuilder::new("random", &params, Some(ret_ty));
    let entry = b.entry_block();

    // Collect available values by type
    let mut int_vals: Vec<Value> = Vec::new();
    let mut float_vals: Vec<Value> = Vec::new();
    let mut all_vals: Vec<Value> = Vec::new();

    for i in 0..n_params {
        let v = b.block_param(entry, i);
        match params[i] {
            Type::F64 => float_vals.push(v),
            _ if params[i].is_int() => int_vals.push(v),
            _ => {}
        }
        all_vals.push(v);
    }

    let n_insts = (dr.next_byte() % 20 + 1) as usize;

    for _ in 0..n_insts {
        let op = dr.next_byte() % 10;
        match op {
            0..=2 => {
                // iconst
                let ty = *dr.pick(&ALL_INT_TYPES);
                let val = dr.next_byte() as i64;
                let v = b.iconst(ty, val);
                int_vals.push(v);
                all_vals.push(v);
            }
            3 => {
                // f64const
                let v = b.f64const(dr.next_byte() as f64);
                float_vals.push(v);
                all_vals.push(v);
            }
            4..=6 if int_vals.len() >= 2 => {
                // int binop
                // pick two vals of same type
                let ty = b.value_type(int_vals[0]);
                let same_ty: Vec<Value> = int_vals
                    .iter()
                    .copied()
                    .filter(|&v| b.value_type(v) == ty)
                    .collect();
                if same_ty.len() >= 2 {
                    let ai = dr.pick_index(same_ty.len());
                    let bi = dr.pick_index(same_ty.len());
                    let a = same_ty[ai];
                    let bv = same_ty[bi];
                    let v = match dr.next_byte() % 5 {
                        0 => b.add(a, bv),
                        1 => b.sub(a, bv),
                        2 => b.mul(a, bv),
                        3 => b.and(a, bv),
                        _ => b.or(a, bv),
                    };
                    int_vals.push(v);
                    all_vals.push(v);
                }
            }
            7 if float_vals.len() >= 2 => {
                let ai = dr.pick_index(float_vals.len());
                let bi = dr.pick_index(float_vals.len());
                let a = float_vals[ai];
                let bv = float_vals[bi];
                let v = match dr.next_byte() % 4 {
                    0 => b.fadd(a, bv),
                    1 => b.fsub(a, bv),
                    2 => b.fmul(a, bv),
                    _ => b.fdiv(a, bv),
                };
                float_vals.push(v);
                all_vals.push(v);
            }
            8 if int_vals.len() >= 2 => {
                // icmp
                let ty = b.value_type(int_vals[0]);
                let same_ty: Vec<Value> = int_vals
                    .iter()
                    .copied()
                    .filter(|&v| b.value_type(v) == ty)
                    .collect();
                if same_ty.len() >= 2 {
                    let ai = dr.pick_index(same_ty.len());
                    let bi = dr.pick_index(same_ty.len());
                    let v = b.icmp(CmpOp::Eq, same_ty[ai], same_ty[bi]);
                    int_vals.push(v); // I8 is int
                    all_vals.push(v);
                }
            }
            9 if !all_vals.is_empty() => {
                // tagged ops (scheme-agnostic)
                let vi = dr.pick_index(all_vals.len());
                let v = all_vals[vi];
                match dr.next_byte() % 4 {
                    0 => {
                        let r = b.tag_of(v);
                        int_vals.push(r);
                        all_vals.push(r);
                    }
                    1 => {
                        let r = b.payload(v);
                        int_vals.push(r);
                        all_vals.push(r);
                    }
                    2 => {
                        let tag = (dr.next_byte() as u32) % 8;
                        let r = b.is_tag(v, tag);
                        int_vals.push(r);
                        all_vals.push(r);
                    }
                    _ => {
                        let tag = (dr.next_byte() as u32) % 8;
                        let r = b.make_tagged(tag, v);
                        int_vals.push(r);
                        all_vals.push(r);
                    }
                }
            }
            _ => {
                // fallback: iconst
                let v = b.iconst(Type::I64, dr.next_byte() as i64);
                int_vals.push(v);
                all_vals.push(v);
            }
        }
    }

    // Find a value matching ret_ty, or create one
    let ret_val = match ret_ty {
        Type::F64 => {
            if let Some(&v) = float_vals.last() {
                v
            } else {
                b.f64const(0.0)
            }
        }
        ty if ty.is_int() => {
            let matching: Vec<Value> = int_vals
                .iter()
                .copied()
                .filter(|&v| b.value_type(v) == ty)
                .collect();
            if let Some(&v) = matching.last() {
                v
            } else {
                b.iconst(ty, 0)
            }
        }
        _ => b.iconst(Type::I64, 0),
    };
    b.ret(ret_val);
    b.build()
}

fn build_random_multi_block(decisions: &[u8]) -> Function {
    let mut dr = DecisionReader::new(decisions);
    let n_params = (dr.next_byte() % 3 + 1) as usize;
    let params: Vec<Type> = (0..n_params).map(|_| Type::I64).collect();

    let mut b = FunctionBuilder::new("multi", &params, Some(Type::I64));
    let entry = b.entry_block();

    let n_extra_blocks = (dr.next_byte() % 4 + 1) as usize;
    let mut extra_blocks: Vec<BlockId> = Vec::new();

    // Create extra blocks — all take one I64 param for simplicity
    for _ in 0..n_extra_blocks {
        let bb = b.create_block(&[Type::I64]);
        extra_blocks.push(bb);
    }

    // Entry block: do some computation, then branch
    let mut val = b.block_param(entry, 0);
    let n_entry_insts = (dr.next_byte() % 5 + 1) as usize;
    for _ in 0..n_entry_insts {
        let c = b.iconst(Type::I64, dr.next_byte() as i64);
        val = b.add(val, c);
    }

    if extra_blocks.len() >= 2 {
        let zero = b.iconst(Type::I64, 0);
        let cmp = b.icmp(CmpOp::Slt, val, zero);
        b.br_if(cmp, extra_blocks[0], &[val], extra_blocks[1], &[val]);
    } else {
        b.jump(extra_blocks[0], &[val]);
    }

    // Fill extra blocks — each does some work and either returns or jumps
    for (i, &bb) in extra_blocks.iter().enumerate() {
        b.switch_to_block(bb);
        let param = b.block_param(bb, 0);
        let c = b.iconst(Type::I64, (i as i64) + 1);
        let result = b.add(param, c);

        let should_return = i == extra_blocks.len() - 1 || dr.next_byte() % 3 == 0;
        if should_return {
            b.ret(result);
        } else {
            // Jump to next block
            let next = extra_blocks[(i + 1) % extra_blocks.len()];
            b.jump(next, &[result]);
        }
    }

    b.build()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_single_block_always_verifies(decisions in proptest::collection::vec(any::<u8>(), 10..100)) {
        let func = build_random_single_block(&decisions);
        verify(&func).unwrap();
    }

    #[test]
    fn prop_multi_block_always_verifies(decisions in proptest::collection::vec(any::<u8>(), 20..100)) {
        let func = build_random_multi_block(&decisions);
        verify(&func).unwrap();
    }

    #[test]
    fn prop_display_roundtrip_no_panic(decisions in proptest::collection::vec(any::<u8>(), 10..100)) {
        let func = build_random_single_block(&decisions);
        let _ = func.to_string(); // should not panic
    }

    #[test]
    fn prop_value_types_consistent(decisions in proptest::collection::vec(any::<u8>(), 10..100)) {
        let func = build_random_single_block(&decisions);
        // Every value referenced in instructions should have a valid type
        for block in &func.blocks {
            for (v, ty) in &block.params {
                assert_eq!(func.value_type(*v), *ty);
            }
            for inst_node in &block.insts {
                if let Some(v) = inst_node.value {
                    let ty = func.value_type(v);
                    let computed = inst_node
                        .inst
                        .result_type(|v| func.value_type(v), &func.extern_funcs);
                    assert_eq!(Some(ty), computed, "value type mismatch for {v}");
                }
            }
        }
    }

    #[test]
    fn prop_predecessors_consistent(decisions in proptest::collection::vec(any::<u8>(), 20..100)) {
        let func = build_random_multi_block(&decisions);
        let preds = func.predecessors();

        // Entry block should have no predecessors
        assert!(preds[0].is_empty());

        // For each edge src->dst, dst's preds should include src
        for (bi, block) in func.blocks.iter().enumerate() {
            let src = BlockId(bi as u32);
            for succ in block.terminator.successors() {
                assert!(
                    preds[succ.index()].contains(&src),
                    "bb{} -> bb{} not reflected in predecessors",
                    bi,
                    succ.index()
                );
            }
        }
    }

    #[test]
    fn prop_ssa_single_definition(decisions in proptest::collection::vec(any::<u8>(), 10..100)) {
        let func = build_random_single_block(&decisions);
        let mut seen = std::collections::HashSet::new();
        for block in &func.blocks {
            for (v, _) in &block.params {
                assert!(seen.insert(*v), "duplicate value definition: {v}");
            }
            for inst_node in &block.insts {
                if let Some(v) = inst_node.value {
                    assert!(seen.insert(v), "duplicate value definition: {v}");
                }
            }
        }
    }

    #[test]
    fn prop_for_each_value_covers_all(decisions in proptest::collection::vec(any::<u8>(), 10..100)) {
        let func = build_random_single_block(&decisions);
        for block in &func.blocks {
            for inst_node in &block.insts {
                let mut count = 0;
                inst_node.inst.for_each_value(|_| count += 1);
                // Just verify it doesn't panic and visits some values
                // (constants have 0 operands, binops have 2, etc.)
                match &inst_node.inst {
                    Inst::Iconst(_, _) | Inst::F64Const(_) => assert_eq!(count, 0),
                    Inst::Add(_, _) | Inst::Sub(_, _) | Inst::Mul(_, _)
                    | Inst::And(_, _) | Inst::Or(_, _) => assert_eq!(count, 2),
                    Inst::TagOf(_) | Inst::Payload(_) | Inst::IsTag(_, _)
                    | Inst::MakeTagged(_, _) => assert_eq!(count, 1),
                    _ => {} // other ops have varying counts
                }
            }
        }
    }

    #[test]
    fn prop_block_terminators_target_valid_blocks(decisions in proptest::collection::vec(any::<u8>(), 20..100)) {
        let func = build_random_multi_block(&decisions);
        let n = func.blocks.len();
        for block in &func.blocks {
            for succ in block.terminator.successors() {
                assert!(succ.index() < n, "terminator targets invalid block {}", succ);
            }
        }
    }
}

// ── Fibonacci end-to-end test ──────────────────────────────────

#[test]
fn test_fib_ir() {
    let mut b = FunctionBuilder::new("fib", &[Type::I64], Some(Type::I64));
    let entry = b.entry_block();
    let n = b.block_param(entry, 0);

    let base_bb = b.create_block(&[Type::I64]);
    let rec_bb = b.create_block(&[Type::I64]);

    let two = b.iconst(Type::I64, 2);
    let cmp = b.icmp(CmpOp::Slt, n, two);
    b.br_if(cmp, base_bb, &[n], rec_bb, &[n]);

    // Base case: return n
    b.switch_to_block(base_bb);
    let base_n = b.block_param(base_bb, 0);
    b.ret(base_n);

    // Recursive case
    b.switch_to_block(rec_bb);
    let rec_n = b.block_param(rec_bb, 0);

    let fib_func = b.declare_func(
        "fib",
        Signature {
            params: vec![Type::I64],
            ret: Some(Type::I64),
        },
    );

    let one = b.iconst(Type::I64, 1);
    let n_minus_1 = b.sub(rec_n, one);
    let r1 = b.call(fib_func, &[n_minus_1]).unwrap();

    let two2 = b.iconst(Type::I64, 2);
    let n_minus_2 = b.sub(rec_n, two2);
    let r2 = b.call(fib_func, &[n_minus_2]).unwrap();

    let result = b.add(r1, r2);
    b.ret(result);

    let func = b.build();
    verify(&func).unwrap();

    let s = func.to_string();
    assert!(s.contains("fn fib(i64) -> i64"));
    assert!(s.contains("call @f0"));
    assert!(s.contains("br_if"));
}
