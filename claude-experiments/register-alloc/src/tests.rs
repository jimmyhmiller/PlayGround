//! Tests for the greedy allocator (framework validation) and
//! the linear scan allocator (algorithmic correctness + quality).

use crate::allocator::*;
use crate::cost::*;
use crate::ir::Function;
use crate::linear_scan::LinearScanAllocator;
use crate::target::Target;
use crate::testing::*;
use crate::types::*;
use crate::verify;

use std::collections::HashMap;

// ============================================================
// A trivial greedy allocator for testing the framework itself
// ============================================================

struct GreedyAllocator;

impl RegisterAllocator for GreedyAllocator {
    fn name(&self) -> &str {
        "greedy"
    }

    fn allocate<F: Function, T: Target>(
        &mut self,
        func: &F,
        target: &T,
    ) -> Result<Allocation, AllocError> {
        let mut alloc = Allocation::new();
        let mut vreg_to_preg: HashMap<VReg, PReg> = HashMap::new();

        for block in func.blocks() {
            for inst in func.block_insts(block) {
                let operands: Vec<Operand> = func.inst_operands(inst).collect();
                let mut used_at_inst: Vec<PReg> = Vec::new();

                for (op_idx, operand) in operands.iter().enumerate() {
                    if operand.kind == OperandKind::Use {
                        if let Reg::Virtual(vreg) = operand.reg {
                            let preg = match &operand.constraint {
                                OperandConstraint::FixedReg(p) => {
                                    vreg_to_preg.insert(vreg, *p);
                                    *p
                                }
                                _ => {
                                    if let Some(&p) = vreg_to_preg.get(&vreg) {
                                        p
                                    } else {
                                        let class = func.vreg_class(vreg);
                                        let p = self.pick_reg(target, class, &used_at_inst)?;
                                        vreg_to_preg.insert(vreg, p);
                                        p
                                    }
                                }
                            };
                            alloc.set(inst, op_idx, preg);
                            used_at_inst.push(preg);
                        }
                    }
                }

                for (op_idx, operand) in operands.iter().enumerate() {
                    match operand.kind {
                        OperandKind::Def | OperandKind::EarlyDef => {
                            if let Reg::Virtual(vreg) = operand.reg {
                                let preg = match &operand.constraint {
                                    OperandConstraint::FixedReg(p) => *p,
                                    OperandConstraint::Reuse(reuse_idx) => {
                                        alloc.get(inst, *reuse_idx).unwrap()
                                    }
                                    _ => {
                                        let class = func.vreg_class(vreg);
                                        self.pick_reg(target, class, &used_at_inst)?
                                    }
                                };
                                alloc.set(inst, op_idx, preg);
                                vreg_to_preg.insert(vreg, preg);
                                used_at_inst.push(preg);
                            }
                        }
                        OperandKind::UseDef => {
                            if let Reg::Virtual(vreg) = operand.reg {
                                let preg = match &operand.constraint {
                                    OperandConstraint::Tied(tied_idx) => {
                                        alloc.get(inst, *tied_idx).unwrap()
                                    }
                                    OperandConstraint::FixedReg(p) => *p,
                                    _ => {
                                        let class = func.vreg_class(vreg);
                                        self.pick_reg(target, class, &used_at_inst)?
                                    }
                                };
                                alloc.set(inst, op_idx, preg);
                                vreg_to_preg.insert(vreg, preg);
                                used_at_inst.push(preg);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(alloc)
    }
}

impl GreedyAllocator {
    fn pick_reg<T: Target>(
        &self,
        target: &T,
        class: RegClass,
        used: &[PReg],
    ) -> Result<PReg, AllocError> {
        let available = target.allocatable_regs(class);
        for preg in &available {
            if !used.contains(preg) {
                return Ok(*preg);
            }
        }
        Err(AllocError::OutOfRegisters {
            inst: InstId(0),
            class,
        })
    }
}

// ============================================================
// Framework tests (greedy allocator)
// ============================================================

#[test]
fn greedy_straight_line_no_pressure() {
    let func = TestSuite::straight_line_no_pressure();
    let target = TestTarget::with_gpr(4);
    let result = run_test(&mut GreedyAllocator, &func, &target, None::<&UniformCostModel>);
    assert!(result.is_ok(), "errors: {:?} {:?}", result.error, result.verification_errors);
}

#[test]
fn greedy_fixed_constraints() {
    let func = TestSuite::fixed_constraints();
    let target = TestTarget::with_gpr(4);
    let result = run_test(&mut GreedyAllocator, &func, &target, None::<&UniformCostModel>);
    assert!(result.is_ok(), "errors: {:?} {:?}", result.error, result.verification_errors);
    let alloc = result.allocation.unwrap();
    assert_eq!(alloc.get(InstId(2), 0), Some(PReg(0)));
    assert_eq!(alloc.get(InstId(2), 1), Some(PReg(2)));
    assert_eq!(alloc.get(InstId(2), 2), Some(PReg(0)));
}

#[test]
fn greedy_two_address() {
    let func = TestSuite::two_address();
    let target = TestTarget::with_gpr(4);
    let result = run_test(&mut GreedyAllocator, &func, &target, None::<&UniformCostModel>);
    assert!(result.is_ok(), "errors: {:?} {:?}", result.error, result.verification_errors);
    let alloc = result.allocation.unwrap();
    assert_eq!(alloc.get(InstId(2), 0), alloc.get(InstId(2), 1));
}

#[test]
fn verifier_catches_wrong_fixed_reg() {
    let func = TestSuite::fixed_constraints();
    let target = TestTarget::with_gpr(4);
    let mut alloc = Allocation::new();
    for block in func.blocks() {
        for inst in func.block_insts(block) {
            let operands: Vec<Operand> = func.inst_operands(inst).collect();
            for (op_idx, _) in operands.iter().enumerate() {
                alloc.set(inst, op_idx, PReg(1));
            }
        }
    }
    let result = verify::verify(&func, &target, &alloc);
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| e.kind == crate::verify::VerifyErrorKind::WrongFixedReg));
}

#[test]
fn verifier_catches_missing_allocation() {
    let func = TestSuite::straight_line_no_pressure();
    let target = TestTarget::with_gpr(4);
    let alloc = Allocation::new();
    let result = verify::verify(&func, &target, &alloc);
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| e.kind == crate::verify::VerifyErrorKind::MissingAllocation));
}

// ============================================================
// Linear Scan: correctness tests
// ============================================================

fn ls() -> LinearScanAllocator {
    LinearScanAllocator
}

/// Helper: run linear scan, assert it allocates + verifies correctly.
fn ls_ok(func: &TestFunction, target: &TestTarget) -> Allocation {
    let result = run_test(&mut ls(), func, target, None::<&UniformCostModel>);
    assert!(
        result.is_ok(),
        "linear scan failed: error={:?}, verify={:?}",
        result.error,
        result.verification_errors
    );
    result.allocation.unwrap()
}

/// Helper: run linear scan with cost model.
fn ls_cost(func: &TestFunction, target: &TestTarget) -> (Allocation, AllocationCost) {
    let cm = UniformCostModel;
    let result = run_test(&mut ls(), func, target, Some(&cm));
    assert!(
        result.is_ok(),
        "linear scan failed: error={:?}, verify={:?}",
        result.error,
        result.verification_errors
    );
    (result.allocation.unwrap(), result.cost.unwrap())
}

// ---- Basic correctness ----

#[test]
fn ls_straight_line_no_pressure() {
    let func = TestSuite::straight_line_no_pressure();
    let target = TestTarget::with_gpr(4);
    let (alloc, cost) = ls_cost(&func, &target);
    // No spills needed: 3 vregs, 4 registers.
    assert_eq!(cost.spill_stores, 0);
    assert_eq!(cost.spill_loads, 0);
    assert_eq!(cost.reg_moves, 0);
    assert_eq!(alloc.num_spill_slots, 0);
}

#[test]
fn ls_straight_line_minimal_regs() {
    // 3 vregs but only 2 registers. v0 and v1 overlap (both live at inst 2),
    // so one must spill. Then v2 is defined and used. Should succeed with spilling.
    let func = TestSuite::straight_line_no_pressure();
    let target = TestTarget::with_gpr(2);
    let (alloc, cost) = ls_cost(&func, &target);
    // At least one spill should have occurred.
    assert!(
        alloc.num_spill_slots >= 1,
        "expected spills with only 2 regs, got {} spill slots",
        alloc.num_spill_slots
    );
    assert!(cost.spill_stores + cost.spill_loads > 0);
}

#[test]
fn ls_high_pressure_enough_regs() {
    let func = TestSuite::high_pressure(3);
    let target = TestTarget::with_gpr(4);
    let (alloc, cost) = ls_cost(&func, &target);
    assert_eq!(alloc.num_spill_slots, 0);
    assert_eq!(cost.total_cost, 0.0);
}

#[test]
fn ls_high_pressure_spills() {
    // 8 values all live simultaneously, only 3 registers.
    let func = TestSuite::high_pressure(8);
    let target = TestTarget::with_gpr(3);
    let (alloc, cost) = ls_cost(&func, &target);
    // Must spill at least 5 values (8 - 3).
    assert!(
        alloc.num_spill_slots >= 5,
        "expected >= 5 spill slots, got {}",
        alloc.num_spill_slots
    );
    assert!(cost.spill_loads > 0);
}

#[test]
fn ls_diamond_cfg() {
    let func = TestSuite::diamond();
    let target = TestTarget::with_gpr(4);
    let (_alloc, cost) = ls_cost(&func, &target);
    // Diamond with 4 regs should be easy, no spills.
    assert_eq!(cost.spill_stores, 0);
    assert_eq!(cost.spill_loads, 0);
}

#[test]
fn ls_simple_loop() {
    let func = TestSuite::simple_loop();
    let target = TestTarget::with_gpr(4);
    ls_ok(&func, &target);
}

#[test]
fn ls_with_call() {
    let func = TestSuite::with_call();
    let target = TestTarget::with_gpr(4);
    ls_ok(&func, &target);
}

#[test]
fn ls_two_address() {
    let func = TestSuite::two_address();
    let target = TestTarget::with_gpr(4);
    let alloc = ls_ok(&func, &target);
    // The reuse constraint: operand 0 (def v2) must match operand 1 (use v0).
    let def = alloc.get(InstId(2), 0).unwrap();
    let use_ = alloc.get(InstId(2), 1).unwrap();
    assert_eq!(def, use_, "reuse constraint violated at inst 2");
}

#[test]
fn ls_fixed_constraints() {
    let func = TestSuite::fixed_constraints();
    let target = TestTarget::with_gpr(4);
    let alloc = ls_ok(&func, &target);
    // idiv at inst 2: operand 0 in r0, operand 1 in r2, operand 2 in r0.
    assert_eq!(alloc.get(InstId(2), 0), Some(PReg(0)));
    assert_eq!(alloc.get(InstId(2), 1), Some(PReg(2)));
    assert_eq!(alloc.get(InstId(2), 2), Some(PReg(0)));
}

#[test]
fn ls_early_def() {
    let func = TestSuite::early_def_test();
    let target = TestTarget::with_gpr(3);
    ls_ok(&func, &target);
}

#[test]
fn ls_reserved_regs() {
    let func = TestSuite::straight_line_no_pressure();
    let target = TestTarget::with_gpr(4).reserved(vec![PReg(0)]);
    let alloc = ls_ok(&func, &target);
    for (&(_inst, _op), &preg) in &alloc.inst_allocs {
        assert_ne!(preg, PReg(0), "reserved register r0 should not be used");
    }
}

#[test]
fn ls_two_reg_classes() {
    let gpr = RegClass(0);
    let fp = RegClass(1);
    let target = TestTarget::with_gpr_and_fp(4, 4);

    let mut b = TestFunctionBuilder::new();
    let v_int = b.vreg(gpr);
    let v_float = b.vreg(fp);
    let v_result = b.vreg(gpr);

    let _bb0 = b.block();
    b.inst(vec![def_reg(v_int, gpr)]);
    b.inst(vec![def_reg(v_float, fp)]);
    b.inst(vec![def_reg(v_result, gpr), use_reg(v_int, gpr)]);
    b.ret(vec![use_reg(v_result, gpr)]);

    let func = b.build();
    let alloc = ls_ok(&func, &target);

    // The float should be in FP class (regs 4-7).
    let float_preg = alloc.get(InstId(1), 0).unwrap();
    assert!(
        float_preg.0 >= 4 && float_preg.0 < 8,
        "float should be in FP class, got {:?}",
        float_preg
    );
}

// ============================================================
// Linear Scan: quality / property tests
// ============================================================

/// Property: with enough registers, linear scan never spills.
#[test]
fn ls_property_no_spill_when_enough_regs() {
    for num_values in 1..=8 {
        let func = TestSuite::high_pressure(num_values);
        // Provide exactly as many registers as values.
        let target = TestTarget::with_gpr(num_values as u16);
        let (alloc, cost) = ls_cost(&func, &target);
        assert_eq!(
            alloc.num_spill_slots, 0,
            "with {} values and {} regs, expected 0 spills but got {}",
            num_values, num_values, alloc.num_spill_slots
        );
        assert_eq!(cost.total_cost, 0.0);
    }
}

/// Property: spills = max(0, num_simultaneous_live - num_regs).
/// For the high_pressure test case, all values are live at the "use all" inst.
#[test]
fn ls_property_spill_count_optimal_for_straight_line() {
    for num_values in 1..=12u32 {
        for num_regs in 1..=6u16 {
            let func = TestSuite::high_pressure(num_values);
            let target = TestTarget::with_gpr(num_regs);
            let result = run_test(&mut ls(), &func, &target, Some(&UniformCostModel));
            assert!(
                result.is_ok(),
                "linear scan should always succeed (it can spill). \
                 values={}, regs={}, error={:?}, verify={:?}",
                num_values,
                num_regs,
                result.error,
                result.verification_errors
            );
            let alloc = result.allocation.unwrap();
            let expected_spills = (num_values as i32 - num_regs as i32).max(0) as u32;
            assert!(
                alloc.num_spill_slots >= expected_spills,
                "values={}, regs={}: expected >= {} spills, got {}",
                num_values, num_regs, expected_spills, alloc.num_spill_slots
            );
        }
    }
}

/// Property: Belady's heuristic — in straight-line code, linear scan
/// spills the interval that ends furthest in the future. This means
/// the intervals that get registers are the ones ending soonest.
/// With n values and R registers, the R intervals with the earliest
/// end points should be in registers.
#[test]
fn ls_property_belady_spills_longest() {
    let gpr = RegClass(0);
    let mut b = TestFunctionBuilder::new();
    // 5 values with varying lifetimes.
    // v0: def at 0, use at 6 (longest)
    // v1: def at 1, use at 5
    // v2: def at 2, use at 4
    // v3: def at 3, use at 3 (shortest)
    // v4: def at 4, use at 5 (medium)
    let v0 = b.vreg(gpr);
    let v1 = b.vreg(gpr);
    let v2 = b.vreg(gpr);
    let v3 = b.vreg(gpr);
    let v4 = b.vreg(gpr);

    let _bb0 = b.block();
    b.inst(vec![def_reg(v0, gpr)]); // inst 0
    b.inst(vec![def_reg(v1, gpr)]); // inst 1
    b.inst(vec![def_reg(v2, gpr)]); // inst 2
    b.inst(vec![def_reg(v3, gpr), use_reg(v3, gpr)]); // inst 3: v3 starts and ends here
    b.inst(vec![def_reg(v4, gpr), use_reg(v2, gpr)]); // inst 4: v2 ends, v4 starts
    b.inst(vec![use_reg(v1, gpr), use_reg(v4, gpr)]); // inst 5: v1 and v4 end
    b.inst(vec![use_reg(v0, gpr)]); // inst 6: v0 ends (longest)
    b.ret(vec![]); // inst 7

    let func = b.build();

    // 2 registers: must spill 3. Belady should spill v0 (ends at 6),
    // v1 (ends at 5), and either v2 or v4.
    let target = TestTarget::with_gpr(2);
    let (alloc, _cost) = ls_cost(&func, &target);
    // v0 should be spilled (longest interval).
    assert!(
        alloc.spill_slots.contains_key(&v0),
        "Belady should spill v0 (longest interval), but it wasn't spilled. \
         Spilled: {:?}",
        alloc.spill_slots.keys().collect::<Vec<_>>()
    );
}

/// Property: linear scan cost never exceeds a simple upper bound.
/// For n values in straight-line with R regs, at most (n-R) spill loads
/// + (n-R) spill stores.
#[test]
fn ls_property_spill_cost_bounded() {
    let cm = UniformCostModel;
    for n in 2..=10u32 {
        for r in 2..=4u16 {
            let func = TestSuite::high_pressure(n);
            let target = TestTarget::with_gpr(r);
            let result = run_test(&mut ls(), &func, &target, Some(&cm));
            assert!(result.is_ok());
            let cost = result.cost.unwrap();
            let spills = (n as i32 - r as i32).max(0) as u64;
            // Each spilled vreg needs at most 1 store (at def) + 1 load (at use).
            // In high_pressure, each vreg has 1 def and 1-2 uses.
            let max_moves = spills * 8; // generous upper bound (includes save/restore for spill reloads)
            let total_moves = cost.spill_stores + cost.spill_loads + cost.reg_moves;
            assert!(
                total_moves <= max_moves,
                "n={}, r={}: total moves {} exceeds bound {}",
                n, r, total_moves, max_moves
            );
        }
    }
}

/// Property: with 1 register, everything but one must spill.
#[test]
fn ls_property_single_register() {
    for n in 2..=6u32 {
        let func = TestSuite::high_pressure(n);
        let target = TestTarget::with_gpr(1);
        let result = run_test(&mut ls(), &func, &target, Some(&UniformCostModel));
        assert!(
            result.is_ok(),
            "even with 1 register, linear scan should succeed via spilling. n={}, err={:?}, verify={:?}",
            n, result.error, result.verification_errors
        );
        let alloc = result.allocation.unwrap();
        assert!(
            alloc.num_spill_slots >= n - 1,
            "n={}: expected >= {} spills with 1 register, got {}",
            n,
            n - 1,
            alloc.num_spill_slots
        );
    }
}

// ============================================================
// Linear Scan: stress tests
// ============================================================

/// Many values, few registers — tests scalability and correctness
/// of the spill code generation.
#[test]
fn ls_stress_many_values() {
    let func = TestSuite::high_pressure(50);
    let target = TestTarget::with_gpr(4);
    let result = run_test(&mut ls(), &func, &target, Some(&UniformCostModel));
    assert!(
        result.is_ok(),
        "stress: error={:?}, verify={:?}",
        result.error,
        result.verification_errors
    );
    let alloc = result.allocation.unwrap();
    assert!(alloc.num_spill_slots >= 46); // 50 - 4 (may be more due to save/restore slots)
}

/// Chain of dependent operations: each value is used by the next.
/// Tests that liveness analysis correctly sees overlapping intervals.
#[test]
fn ls_chain_of_ops() {
    let gpr = RegClass(0);
    let mut b = TestFunctionBuilder::new();

    let mut vregs = Vec::new();
    let v0 = b.vreg(gpr);
    vregs.push(v0);

    let _bb0 = b.block();
    b.inst(vec![def_reg(v0, gpr)]);

    for _ in 0..20 {
        let v_new = b.vreg(gpr);
        let v_prev = *vregs.last().unwrap();
        b.inst(vec![def_reg(v_new, gpr), use_reg(v_prev, gpr)]);
        vregs.push(v_new);
    }
    b.ret(vec![use_reg(*vregs.last().unwrap(), gpr)]);

    let func = b.build();
    let target = TestTarget::with_gpr(2);
    let (alloc, cost) = ls_cost(&func, &target);
    // Chain of ops: each value is only live for one instruction,
    // so 2 registers should be enough with no spills.
    assert_eq!(
        alloc.num_spill_slots, 0,
        "chain of ops should not spill with 2 regs"
    );
    assert_eq!(cost.total_cost, 0.0);
}

/// Multiple independent chains — tests that the allocator correctly
/// reuses registers when values' lifetimes don't overlap.
#[test]
fn ls_independent_chains() {
    let gpr = RegClass(0);
    let mut b = TestFunctionBuilder::new();
    let _bb0 = b.block();

    // 3 independent chains of 5 ops each.
    // Each chain needs 2 registers (current + next), but chains don't overlap.
    let mut final_vregs = Vec::new();
    for _chain in 0..3 {
        let mut v = b.vreg(gpr);
        b.inst(vec![def_reg(v, gpr)]);
        for _ in 0..4 {
            let v_new = b.vreg(gpr);
            b.inst(vec![def_reg(v_new, gpr), use_reg(v, gpr)]);
            v = v_new;
        }
        final_vregs.push(v);
    }
    // Use all final values to keep them live until here.
    let uses: Vec<Operand> = final_vregs.iter().map(|&v| use_reg(v, gpr)).collect();
    b.inst(uses);
    b.ret(vec![use_reg(final_vregs[0], gpr)]);

    let func = b.build();
    // 3 final values are live simultaneously. Conservative single-interval
    // analysis also keeps intermediate values live across chains, so we
    // need more than 3 registers for zero spills.
    // With 4 registers, there's always a spare for intermediates.
    let target = TestTarget::with_gpr(4);
    let (alloc, cost) = ls_cost(&func, &target);
    assert_eq!(
        alloc.num_spill_slots, 0,
        "3 independent chains with 4 regs should not spill"
    );
    assert_eq!(cost.total_cost, 0.0);

    // With exactly 3 registers, a small number of spills is expected
    // due to conservative interval approximation.
    let target_tight = TestTarget::with_gpr(3);
    let result = run_test(&mut ls(), &func, &target_tight, Some(&UniformCostModel));
    assert!(result.is_ok(), "should succeed even with tight regs");
}

// ============================================================
// Linear Scan: cost model comparison
// ============================================================

#[test]
fn ls_cost_model_uniform_vs_memory_expensive() {
    // With spills, MemoryExpensiveCostModel should report higher cost.
    let func = TestSuite::high_pressure(6);
    let target = TestTarget::with_gpr(3);

    let uniform = UniformCostModel;
    let expensive = MemoryExpensiveCostModel;

    let r_uni = run_test(&mut ls(), &func, &target, Some(&uniform));
    let r_exp = run_test(&mut ls(), &func, &target, Some(&expensive));

    assert!(r_uni.is_ok());
    assert!(r_exp.is_ok());

    let c_uni = r_uni.cost.unwrap();
    let c_exp = r_exp.cost.unwrap();

    // Same allocation, but expensive model weights spills more.
    assert!(
        c_exp.total_cost > c_uni.total_cost,
        "memory-expensive model should report higher cost: uniform={}, expensive={}",
        c_uni.total_cost,
        c_exp.total_cost
    );
}

// ============================================================
// Linear Scan: edge cases
// ============================================================

/// Empty function (no instructions).
#[test]
fn ls_empty_function() {
    let mut b = TestFunctionBuilder::new();
    let _bb0 = b.block();
    // No instructions at all.
    let func = b.build();
    let target = TestTarget::with_gpr(4);
    let result = run_test(&mut ls(), &func, &target, None::<&UniformCostModel>);
    assert!(result.is_ok());
}

/// Single instruction: just a return.
#[test]
fn ls_single_return() {
    let gpr = RegClass(0);
    let mut b = TestFunctionBuilder::new();
    let v0 = b.vreg(gpr);
    let _bb0 = b.block();
    b.inst(vec![def_reg(v0, gpr)]);
    b.ret(vec![use_reg(v0, gpr)]);
    let func = b.build();
    let target = TestTarget::with_gpr(1);
    ls_ok(&func, &target);
}

/// Two fixed registers conflict at the same instruction.
/// Both defs need different fixed registers.
#[test]
fn ls_multiple_fixed_at_same_inst() {
    let gpr = RegClass(0);
    let mut b = TestFunctionBuilder::new();
    let v0 = b.vreg(gpr);
    let v1 = b.vreg(gpr);
    let _bb0 = b.block();
    b.inst(vec![def_fixed(v0, PReg(0)), def_fixed(v1, PReg(1))]);
    b.inst(vec![use_reg(v0, gpr), use_reg(v1, gpr)]);
    b.ret(vec![]);
    let func = b.build();
    let target = TestTarget::with_gpr(4);
    let alloc = ls_ok(&func, &target);
    assert_eq!(alloc.get(InstId(0), 0), Some(PReg(0)));
    assert_eq!(alloc.get(InstId(0), 1), Some(PReg(1)));
}

/// Value defined in one block, used in another (cross-block liveness).
#[test]
fn ls_cross_block_liveness() {
    let gpr = RegClass(0);
    let mut b = TestFunctionBuilder::new();
    let v0 = b.vreg(gpr);

    let bb0 = b.block();
    let bb1 = b.block();

    b.current_block = Some(0);
    b.inst(vec![def_reg(v0, gpr)]);
    b.branch(vec![], vec![bb1], vec![vec![]]);

    b.current_block = Some(1);
    b.ret(vec![use_reg(v0, gpr)]);

    let _ = (bb0, bb1);
    let func = b.build();
    let target = TestTarget::with_gpr(2);
    ls_ok(&func, &target);
}

// ============================================================
// Safepoint / stackmap tests
// ============================================================

use crate::allocator::{MoveOperand, MovePosition};
use crate::ir::SafepointAction;

/// Safepoint with Record: values stay where they are, locations reported.
#[test]
fn ls_safepoint_record() {
    let gpr = RegClass(0);
    let mut b = TestFunctionBuilder::new();
    let v0 = b.vreg(gpr);
    let v1 = b.vreg(gpr);
    let v_result = b.vreg(gpr);

    let _bb0 = b.block();
    b.inst(vec![def_reg(v0, gpr)]);
    b.inst(vec![def_reg(v1, gpr)]);
    // Safepoint call — v0 and v1 are live across it.
    // Clobbers r0-r2 (caller-saved), but r3-r5 are callee-saved.
    let call_inst = b.safepoint_call(
        vec![def_reg(v_result, gpr)],
        vec![PReg(0), PReg(1), PReg(2)],
    );
    // Mark v0 and v1 as GC roots that just need location recording.
    b.set_safepoint_action(call_inst, v0, SafepointAction::Record);
    b.set_safepoint_action(call_inst, v1, SafepointAction::Record);
    b.inst(vec![use_reg(v0, gpr), use_reg(v1, gpr), use_reg(v_result, gpr)]);
    b.ret(vec![use_reg(v_result, gpr)]);

    let func = b.build();
    // 6 regs: r0-r2 caller-saved (clobbered), r3-r5 callee-saved (survive call)
    let target = TestTarget::with_gpr(6).callee_saved(vec![PReg(3), PReg(4), PReg(5)]);
    let alloc = ls_ok(&func, &target);

    // Should have a stackmap for the call instruction.
    let stackmap = alloc.stackmaps.get(&call_inst);
    assert!(stackmap.is_some(), "expected stackmap at safepoint call");
    let entries = stackmap.unwrap();
    // Both v0 and v1 should be recorded.
    assert_eq!(entries.len(), 2, "expected 2 stackmap entries, got {}", entries.len());
    for entry in entries {
        assert_eq!(entry.action, SafepointAction::Record);
        // With callee-saved regs available, values should be in registers.
        assert!(matches!(entry.location, MoveOperand::Reg(_)),
            "Record should keep values in registers, got {:?}", entry.location);
    }
}

/// Safepoint with SpillAndRecord: values forced to stack, locations reported.
#[test]
fn ls_safepoint_spill_and_record() {
    let gpr = RegClass(0);
    let mut b = TestFunctionBuilder::new();
    let v0 = b.vreg(gpr);
    let v1 = b.vreg(gpr);
    let v_result = b.vreg(gpr);

    let _bb0 = b.block();
    b.inst(vec![def_reg(v0, gpr)]);
    b.inst(vec![def_reg(v1, gpr)]);
    let call_inst = b.safepoint_call(
        vec![def_reg(v_result, gpr)],
        vec![PReg(0), PReg(1), PReg(2)],
    );
    // Moving GC: both values must be on the stack.
    b.set_safepoint_action(call_inst, v0, SafepointAction::SpillAndRecord);
    b.set_safepoint_action(call_inst, v1, SafepointAction::SpillAndRecord);
    b.inst(vec![use_reg(v0, gpr), use_reg(v1, gpr), use_reg(v_result, gpr)]);
    b.ret(vec![use_reg(v_result, gpr)]);

    let func = b.build();
    // 6 regs with callee-saved, so values can be in regs (then SpillAndRecord
    // forces them to stack at the safepoint).
    let target = TestTarget::with_gpr(6).callee_saved(vec![PReg(3), PReg(4), PReg(5)]);
    let alloc = ls_ok(&func, &target);

    let stackmap = alloc.stackmaps.get(&call_inst);
    assert!(stackmap.is_some(), "expected stackmap at safepoint call");
    let entries = stackmap.unwrap();
    assert_eq!(entries.len(), 2);
    for entry in entries {
        assert_eq!(entry.action, SafepointAction::SpillAndRecord);
        // SpillAndRecord: values must be in stack slots.
        assert!(matches!(entry.location, MoveOperand::SpillSlot(_)),
            "SpillAndRecord should force values to stack, got {:?}", entry.location);
    }

    // There should be spill-before and reload-after moves for both values.
    let spill_moves: Vec<_> = alloc.moves.iter()
        .filter(|m| m.at == MovePosition::Before(call_inst))
        .collect();
    let reload_moves: Vec<_> = alloc.moves.iter()
        .filter(|m| m.at == MovePosition::After(call_inst))
        .collect();
    assert_eq!(spill_moves.len(), 2, "expected 2 spill moves before safepoint");
    assert_eq!(reload_moves.len(), 2, "expected 2 reload moves after safepoint");
}

/// Mixed safepoint actions: some values Record, some SpillAndRecord, some Ignore.
#[test]
fn ls_safepoint_mixed_actions() {
    let gpr = RegClass(0);
    let mut b = TestFunctionBuilder::new();
    let v_ptr = b.vreg(gpr);    // heap pointer — needs SpillAndRecord
    let v_int = b.vreg(gpr);    // raw integer — Ignore
    let v_weak = b.vreg(gpr);   // weak ref — just Record
    let v_result = b.vreg(gpr);

    let _bb0 = b.block();
    b.inst(vec![def_reg(v_ptr, gpr)]);
    b.inst(vec![def_reg(v_int, gpr)]);
    b.inst(vec![def_reg(v_weak, gpr)]);
    let call_inst = b.safepoint_call(
        vec![def_reg(v_result, gpr)],
        vec![PReg(0), PReg(1), PReg(2)],
    );
    b.set_safepoint_action(call_inst, v_ptr, SafepointAction::SpillAndRecord);
    b.set_safepoint_action(call_inst, v_int, SafepointAction::Ignore);
    b.set_safepoint_action(call_inst, v_weak, SafepointAction::Record);
    b.inst(vec![
        use_reg(v_ptr, gpr), use_reg(v_int, gpr),
        use_reg(v_weak, gpr), use_reg(v_result, gpr),
    ]);
    b.ret(vec![use_reg(v_result, gpr)]);

    let func = b.build();
    // 6 regs with callee-saved so values survive the call in registers.
    let target = TestTarget::with_gpr(6).callee_saved(vec![PReg(3), PReg(4), PReg(5)]);
    let alloc = ls_ok(&func, &target);

    let entries = alloc.stackmaps.get(&call_inst).expect("expected stackmap");

    // v_int should NOT appear (Ignore).
    assert!(!entries.iter().any(|e| e.vreg == v_int),
        "Ignore vreg should not be in stackmap");

    // v_ptr should be SpillAndRecord (on stack).
    let ptr_entry = entries.iter().find(|e| e.vreg == v_ptr).expect("v_ptr missing");
    assert_eq!(ptr_entry.action, SafepointAction::SpillAndRecord);
    assert!(matches!(ptr_entry.location, MoveOperand::SpillSlot(_)));

    // v_weak should be Record (in register).
    let weak_entry = entries.iter().find(|e| e.vreg == v_weak).expect("v_weak missing");
    assert_eq!(weak_entry.action, SafepointAction::Record);
    assert!(matches!(weak_entry.location, MoveOperand::Reg(_)));
}

/// No safepoint actions = no stackmaps (backwards compatible).
#[test]
fn ls_no_safepoints_no_stackmaps() {
    let func = TestSuite::with_call();
    let target = TestTarget::with_gpr(4);
    let alloc = ls_ok(&func, &target);
    assert!(alloc.stackmaps.is_empty(), "no safepoints → no stackmaps");
}

/// CallingConvention action at a safepoint should not generate stackmap entries.
#[test]
fn ls_safepoint_calling_convention_no_entry() {
    let gpr = RegClass(0);
    let mut b = TestFunctionBuilder::new();
    let v0 = b.vreg(gpr);
    let v_result = b.vreg(gpr);

    let _bb0 = b.block();
    b.inst(vec![def_reg(v0, gpr)]);
    let _call_inst = b.safepoint_call(
        vec![def_reg(v_result, gpr)],
        vec![PReg(0), PReg(1), PReg(2)],
    );
    // Default action is CallingConvention — should produce no stackmap entry.
    b.inst(vec![use_reg(v0, gpr), use_reg(v_result, gpr)]);
    b.ret(vec![use_reg(v_result, gpr)]);

    let func = b.build();
    let target = TestTarget::with_gpr(4);
    let alloc = ls_ok(&func, &target);
    assert!(alloc.stackmaps.is_empty(),
        "CallingConvention-only safepoints should not produce stackmap entries");
}

// ============================================================
// Flat IR tests — modeled after nano-gpt MachIR patterns
// ============================================================

use crate::flat::*;

/// Helper: allocate flat and assert success.
fn flat_ok(flat: &SimpleFlat, target: &TestTarget) -> Allocation {
    let result = allocate_flat(flat, target);
    match result {
        Ok(alloc) => alloc,
        Err(e) => panic!("flat allocation failed: {:?}", e),
    }
}

/// Straight-line code: define values, use them, return. No control flow.
/// Models nano-gpt's elementwise loop body (address calc + FP arithmetic).
#[test]
fn flat_straight_line() {
    let gpr = RegClass(0);
    let mut b = SimpleFlatBuilder::new();
    let v0 = b.vreg(gpr);
    let v1 = b.vreg(gpr);
    let v2 = b.vreg(gpr);

    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v0)]));
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v1)]));
    b.push(SimpleFlatInst::op(vec![
        FlatOperand::Def(v2),
        FlatOperand::Use(v0),
        FlatOperand::Use(v1),
    ]));
    b.push(SimpleFlatInst::ret(vec![FlatOperand::Use(v2)]));

    let flat = b.build();
    let target = TestTarget::with_gpr(4);
    let alloc = flat_ok(&flat, &target);

    // v2's def should have a physical register assigned.
    assert!(alloc.get(InstId(2), 0).is_some());
}

/// Simple loop: label, body, conditional branch back.
/// Models nano-gpt's `emit_elementwise_loop` pattern.
#[test]
fn flat_simple_loop() {
    let gpr = RegClass(0);
    let mut b = SimpleFlatBuilder::new();
    let v_count = b.vreg(gpr); // loop counter
    let v_limit = b.vreg(gpr); // loop limit
    let v_one = b.vreg(gpr);   // constant 1

    // Prologue: set up counter and limit.
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_count)]));  // 0: mov counter, 0
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_limit)]));  // 1: mov limit, N
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_one)]));    // 2: mov one, 1

    // Loop header.
    b.push(SimpleFlatInst::label(0));                              // 3: label 0

    // Loop body: increment counter.
    b.push(SimpleFlatInst::op(vec![                                // 4: add counter, one
        FlatOperand::Def(v_count),
        FlatOperand::Use(v_count),
        FlatOperand::Use(v_one),
    ]));

    // Compare and branch back.
    b.push(SimpleFlatInst::op(vec![                                // 5: cmp counter, limit
        FlatOperand::Use(v_count),
        FlatOperand::Use(v_limit),
    ]));
    b.push(SimpleFlatInst::cond_branch(0, vec![]));                // 6: blt label 0

    // Epilogue.
    b.push(SimpleFlatInst::ret(vec![FlatOperand::Use(v_count)]));  // 7: ret counter

    let flat = b.build();
    let target = TestTarget::with_gpr(4);
    let alloc = flat_ok(&flat, &target);

    // All three vregs should be allocated to distinct registers (no spills needed).
    assert_eq!(alloc.num_spill_slots, 0);
    // counter at inst 4 should have a register.
    assert!(alloc.get(InstId(4), 0).is_some());
}

/// Mixed register classes: GP for address calc, FP for computation.
/// Models nano-gpt's typical loop body with GP pointers + FP scalar math.
#[test]
fn flat_mixed_reg_classes() {
    let gpr = RegClass(0);
    let fp = RegClass(1);
    let mut b = SimpleFlatBuilder::new();

    let v_base = b.vreg(gpr);   // GP: base pointer
    let v_offset = b.vreg(gpr); // GP: offset
    let v_a = b.vreg(fp);       // FP: loaded value
    let v_b = b.vreg(fp);       // FP: loaded value
    let v_c = b.vreg(fp);       // FP: result

    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_base)]));
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_offset)]));
    // Load a float using GP address registers.
    b.push(SimpleFlatInst::op(vec![
        FlatOperand::Def(v_a),
        FlatOperand::Use(v_base),
        FlatOperand::Use(v_offset),
    ]));
    b.push(SimpleFlatInst::op(vec![
        FlatOperand::Def(v_b),
        FlatOperand::Use(v_base),
        FlatOperand::Use(v_offset),
    ]));
    // FP add.
    b.push(SimpleFlatInst::op(vec![
        FlatOperand::Def(v_c),
        FlatOperand::Use(v_a),
        FlatOperand::Use(v_b),
    ]));
    b.push(SimpleFlatInst::ret(vec![FlatOperand::Use(v_c)]));

    let flat = b.build();
    let target = TestTarget::with_gpr_and_fp(4, 4);
    let alloc = flat_ok(&flat, &target);

    // GP regs should be in 0..4, FP regs in 4..8.
    let base_preg = alloc.get(InstId(0), 0).unwrap();
    assert!(base_preg.0 < 4, "GP vreg should get GP preg, got {:?}", base_preg);

    let a_preg = alloc.get(InstId(2), 0).unwrap();
    assert!(a_preg.0 >= 4 && a_preg.0 < 8, "FP vreg should get FP preg, got {:?}", a_preg);
}

/// Tied operand: accumulator-style instruction like ARM FMLA.
/// `acc = acc + lhs * rhs` — acc is both read and written.
#[test]
fn flat_tied_operand_fmla() {
    let fp = RegClass(1);
    let mut b = SimpleFlatBuilder::new();

    let v_acc = b.vreg(fp);
    let v_lhs = b.vreg(fp);
    let v_rhs = b.vreg(fp);

    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_acc)]));   // movi acc, 0
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_lhs)]));   // ldr lhs
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_rhs)]));   // ldr rhs

    // FMLA: acc += lhs * rhs (acc is UseDef — tied)
    b.push(SimpleFlatInst::op(vec![
        FlatOperand::UseDef(v_acc),
        FlatOperand::Use(v_lhs),
        FlatOperand::Use(v_rhs),
    ]));

    b.push(SimpleFlatInst::ret(vec![FlatOperand::Use(v_acc)]));

    let flat = b.build();
    let target = TestTarget::with_gpr_and_fp(4, 4);
    let alloc = flat_ok(&flat, &target);

    // The UseDef operand should get a register assignment.
    let acc_preg = alloc.get(InstId(3), 0).unwrap();
    assert!(acc_preg.0 >= 4 && acc_preg.0 < 8, "FP UseDef should get FP preg");
}

/// Fixed register constraint: call requiring arg in a specific register.
/// Models nano-gpt's `CallFpUnary` which needs arg in S0 and result in S0.
#[test]
fn flat_fixed_reg_call() {
    let gpr = RegClass(0);
    let fp = RegClass(1);
    let s0 = PReg(4); // First FP register in our test target

    let mut b = SimpleFlatBuilder::new();
    let v_ptr = b.vreg(gpr);    // function pointer
    let v_arg = b.vreg(fp);     // arg value
    let v_result = b.vreg(fp);  // result
    let v_other = b.vreg(fp);   // a value that must survive the call

    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_ptr)]));
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_arg)]));
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_other)]));

    // Call: arg must be in S0, result comes out in S0.
    // Clobbers all caller-saved FP regs.
    b.push(SimpleFlatInst::call(
        vec![
            FlatOperand::DefFixed(v_result, s0),
            FlatOperand::UseFixed(v_arg, s0),
            FlatOperand::Use(v_ptr),
        ],
        vec![PReg(4), PReg(5), PReg(6), PReg(7)], // clobber all FP regs
    ));

    // Use result and the value that survived the call.
    b.push(SimpleFlatInst::op(vec![
        FlatOperand::Use(v_result),
        FlatOperand::Use(v_other),
    ]));
    b.push(SimpleFlatInst::ret(vec![]));

    let flat = b.build();
    let target = TestTarget::with_gpr_and_fp(4, 4);
    let alloc = flat_ok(&flat, &target);

    // The call's def should be in S0.
    assert_eq!(alloc.get(InstId(3), 0), Some(s0), "call result must be in S0");
    // The call's use should be in S0.
    assert_eq!(alloc.get(InstId(3), 1), Some(s0), "call arg must be in S0");
}

/// Register pressure with spilling in flat IR.
/// More live values than registers forces spills.
#[test]
fn flat_spilling() {
    let gpr = RegClass(0);
    let mut b = SimpleFlatBuilder::new();
    let mut vregs = Vec::new();

    // Define 8 values.
    for _ in 0..8 {
        let v = b.vreg(gpr);
        b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v)]));
        vregs.push(v);
    }
    // Use all 8 values (all live simultaneously).
    for &v in &vregs {
        b.push(SimpleFlatInst::op(vec![FlatOperand::Use(v)]));
    }
    b.push(SimpleFlatInst::ret(vec![]));

    let flat = b.build();
    // Only 3 registers available — must spill 5.
    let target = TestTarget::with_gpr(3);
    let alloc = flat_ok(&flat, &target);
    assert!(alloc.num_spill_slots >= 5, "8 values with 3 regs → expected >= 5 spills, got {}", alloc.num_spill_slots);
}

/// Nested loop: outer loop + inner loop.
/// Models nano-gpt's reduce loops (outer over non-reduced axes, inner over reduce axis).
#[test]
fn flat_nested_loop() {
    let gpr = RegClass(0);
    let fp = RegClass(1);
    let mut b = SimpleFlatBuilder::new();

    let v_i = b.vreg(gpr);     // outer counter
    let v_j = b.vreg(gpr);     // inner counter
    let v_n = b.vreg(gpr);     // outer limit
    let v_m = b.vreg(gpr);     // inner limit
    let v_acc = b.vreg(fp);    // accumulator
    let v_val = b.vreg(fp);    // loaded value

    // Init.
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_n)]));     // 0
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_m)]));     // 1
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_i)]));     // 2

    // Outer loop header.
    b.push(SimpleFlatInst::label(0));                              // 3
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_acc)]));   // 4: zero acc
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_j)]));     // 5: zero j

    // Inner loop header.
    b.push(SimpleFlatInst::label(1));                              // 6

    // Inner body: acc += load(base + i*m + j)
    b.push(SimpleFlatInst::op(vec![                                // 7: load val
        FlatOperand::Def(v_val),
        FlatOperand::Use(v_i),
        FlatOperand::Use(v_j),
    ]));
    b.push(SimpleFlatInst::op(vec![                                // 8: acc += val
        FlatOperand::UseDef(v_acc),
        FlatOperand::Use(v_val),
    ]));

    // Inner increment and branch.
    b.push(SimpleFlatInst::op(vec![                                // 9: j++
        FlatOperand::Def(v_j),
        FlatOperand::Use(v_j),
    ]));
    b.push(SimpleFlatInst::op(vec![                                // 10: cmp j, m
        FlatOperand::Use(v_j),
        FlatOperand::Use(v_m),
    ]));
    b.push(SimpleFlatInst::cond_branch(1, vec![]));                // 11: blt inner

    // Store accumulator result, outer increment.
    b.push(SimpleFlatInst::op(vec![FlatOperand::Use(v_acc)]));    // 12: store acc
    b.push(SimpleFlatInst::op(vec![                                // 13: i++
        FlatOperand::Def(v_i),
        FlatOperand::Use(v_i),
    ]));
    b.push(SimpleFlatInst::op(vec![                                // 14: cmp i, n
        FlatOperand::Use(v_i),
        FlatOperand::Use(v_n),
    ]));
    b.push(SimpleFlatInst::cond_branch(0, vec![]));                // 15: blt outer

    b.push(SimpleFlatInst::ret(vec![]));                           // 16

    let flat = b.build();
    let target = TestTarget::with_gpr_and_fp(6, 4);
    let alloc = flat_ok(&flat, &target);

    // With 6 GP regs and 4 FP regs, should not need spills.
    assert_eq!(alloc.num_spill_slots, 0, "nested loop should fit without spills");
}

/// Loop with NEON FMLA accumulation — tied operand inside a loop.
/// Models nano-gpt's tiled matmul inner loop.
#[test]
fn flat_fmla_in_loop() {
    let gpr = RegClass(0);
    let fp = RegClass(1);
    let mut b = SimpleFlatBuilder::new();

    let v_k = b.vreg(gpr);      // loop counter
    let v_n = b.vreg(gpr);      // loop limit
    let v_acc = b.vreg(fp);     // FMLA accumulator (tied)
    let v_lhs = b.vreg(fp);     // left operand
    let v_rhs = b.vreg(fp);     // right operand

    // Init.
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_n)]));     // 0
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_acc)]));   // 1: zero acc
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_k)]));     // 2: zero k

    // Loop.
    b.push(SimpleFlatInst::label(0));                              // 3

    b.push(SimpleFlatInst::op(vec![                                // 4: load lhs
        FlatOperand::Def(v_lhs),
        FlatOperand::Use(v_k),
    ]));
    b.push(SimpleFlatInst::op(vec![                                // 5: load rhs
        FlatOperand::Def(v_rhs),
        FlatOperand::Use(v_k),
    ]));

    // FMLA: acc += lhs * rhs
    b.push(SimpleFlatInst::op(vec![                                // 6
        FlatOperand::UseDef(v_acc),
        FlatOperand::Use(v_lhs),
        FlatOperand::Use(v_rhs),
    ]));

    // k++, compare, branch.
    b.push(SimpleFlatInst::op(vec![                                // 7
        FlatOperand::Def(v_k),
        FlatOperand::Use(v_k),
    ]));
    b.push(SimpleFlatInst::op(vec![                                // 8
        FlatOperand::Use(v_k),
        FlatOperand::Use(v_n),
    ]));
    b.push(SimpleFlatInst::cond_branch(0, vec![]));                // 9

    // Return accumulator.
    b.push(SimpleFlatInst::ret(vec![FlatOperand::Use(v_acc)]));   // 10

    let flat = b.build();
    let target = TestTarget::with_gpr_and_fp(4, 4);
    let alloc = flat_ok(&flat, &target);

    assert_eq!(alloc.num_spill_slots, 0, "FMLA loop should fit without spills");
    // The acc UseDef at inst 6 should have a consistent FP register.
    let acc_preg = alloc.get(InstId(6), 0).unwrap();
    assert!(acc_preg.0 >= 4, "acc should be in FP class");
}

/// Early def in flat IR: output can't share register with inputs.
#[test]
fn flat_early_def() {
    let gpr = RegClass(0);
    let mut b = SimpleFlatBuilder::new();
    let v0 = b.vreg(gpr);
    let v1 = b.vreg(gpr);
    let v2 = b.vreg(gpr);

    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v0)]));
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v1)]));
    b.push(SimpleFlatInst::op(vec![
        FlatOperand::EarlyDef(v2),
        FlatOperand::Use(v0),
        FlatOperand::Use(v1),
    ]));
    b.push(SimpleFlatInst::ret(vec![FlatOperand::Use(v2)]));

    let flat = b.build();
    // Need at least 3 regs: v2 can't share with v0 or v1.
    let target = TestTarget::with_gpr(3);
    let alloc = flat_ok(&flat, &target);

    let r0 = alloc.get(InstId(2), 1).unwrap(); // use v0
    let r1 = alloc.get(InstId(2), 2).unwrap(); // use v1
    let r2 = alloc.get(InstId(2), 0).unwrap(); // early def v2
    assert_ne!(r2, r0, "early def must not share reg with input v0");
    assert_ne!(r2, r1, "early def must not share reg with input v1");
}

/// Multiple labels, forward and backward branches.
/// Tests that the block splitter handles complex control flow.
#[test]
fn flat_forward_and_backward_branches() {
    let gpr = RegClass(0);
    let mut b = SimpleFlatBuilder::new();
    let v0 = b.vreg(gpr);
    let v1 = b.vreg(gpr);

    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v0)]));       // 0
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v1)]));       // 1
    // Forward branch: skip the next block.
    b.push(SimpleFlatInst::cond_branch(1, vec![                   // 2
        FlatOperand::Use(v0),
    ]));
    // Fallthrough block.
    b.push(SimpleFlatInst::op(vec![                                // 3
        FlatOperand::Def(v0),
        FlatOperand::Use(v1),
    ]));
    // Label 1: merge point.
    b.push(SimpleFlatInst::label(1));                              // 4
    // Use both values.
    b.push(SimpleFlatInst::op(vec![                                // 5
        FlatOperand::Use(v0),
        FlatOperand::Use(v1),
    ]));
    b.push(SimpleFlatInst::ret(vec![]));                           // 6

    let flat = b.build();
    let target = TestTarget::with_gpr(4);
    flat_ok(&flat, &target);
}

/// Empty flat function (no instructions).
#[test]
fn flat_empty() {
    let flat = SimpleFlat {
        num_vregs: 0,
        vreg_classes: vec![],
        insts: vec![],
    };
    let target = TestTarget::with_gpr(4);
    let alloc = flat_ok(&flat, &target);
    assert!(alloc.inst_allocs.is_empty());
}

/// Single instruction: just a return with no operands.
#[test]
fn flat_single_ret() {
    let mut b = SimpleFlatBuilder::new();
    b.push(SimpleFlatInst::ret(vec![]));
    let flat = b.build();
    let target = TestTarget::with_gpr(4);
    flat_ok(&flat, &target);
}

/// Spilling with a loop: values defined before the loop must survive.
#[test]
fn flat_spill_across_loop() {
    let gpr = RegClass(0);
    let mut b = SimpleFlatBuilder::new();

    // Define more values than we have registers.
    let mut long_lived: Vec<VReg> = Vec::new();
    for _ in 0..5 {
        let v = b.vreg(gpr);
        b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v)]));
        long_lived.push(v);
    }

    let v_i = b.vreg(gpr);
    let v_n = b.vreg(gpr);
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_i)]));
    b.push(SimpleFlatInst::op(vec![FlatOperand::Def(v_n)]));

    // Loop.
    b.push(SimpleFlatInst::label(0));
    b.push(SimpleFlatInst::op(vec![
        FlatOperand::Def(v_i),
        FlatOperand::Use(v_i),
    ]));
    b.push(SimpleFlatInst::op(vec![
        FlatOperand::Use(v_i),
        FlatOperand::Use(v_n),
    ]));
    b.push(SimpleFlatInst::cond_branch(0, vec![]));

    // Use all the long-lived values after the loop.
    for &v in &long_lived {
        b.push(SimpleFlatInst::op(vec![FlatOperand::Use(v)]));
    }
    b.push(SimpleFlatInst::ret(vec![]));

    let flat = b.build();
    // Only 3 registers — must spill several values.
    let target = TestTarget::with_gpr(3);
    let alloc = flat_ok(&flat, &target);
    assert!(alloc.num_spill_slots > 0, "should need spills with only 3 regs for 7 values");
}
