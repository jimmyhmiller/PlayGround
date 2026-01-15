//! Integration tests for the optimization framework.
//!
//! These tests use the concrete implementation to verify the optimization passes work correctly.

use ssa_test::concrete::instruction::{Instruction, InstructionBuilder};
use ssa_test::concrete::value::Value;
use ssa_test::concrete::ast::BinaryOperator;
use ssa_test::types::{BlockId, SsaVariable};
use ssa_test::translator::SSATranslator;
use ssa_test::optim::prelude::*;

type TestTranslator = SSATranslator<Value, Instruction, InstructionBuilder>;

/// Helper to create a test translator with some instructions
fn create_simple_translator() -> TestTranslator {
    let mut translator = TestTranslator::new();

    // Block 0: entry
    // v0 := 10
    // v1 := 20
    // v2 := v0 + v1
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Literal(20),
    });
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v2"),
        left: Value::Var(SsaVariable::new("v0")),
        op: BinaryOperator::Add,
        right: Value::Var(SsaVariable::new("v1")),
    });

    // Seal the entry block
    translator.seal_block(BlockId(0));

    translator
}

// ============================================================================
// Dead Code Elimination Tests
// ============================================================================

#[test]
fn test_dce_removes_dead_instruction() {
    let mut translator = TestTranslator::new();

    // v0 := 10 (dead - never used)
    // v1 := 20
    // print v1
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Literal(20),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v1")),
    });

    translator.seal_block(BlockId(0));

    let initial_count = translator.blocks[0].instructions.len();
    assert_eq!(initial_count, 3);

    let mut cache = AnalysisCache::new();
    let mut dce = DeadCodeElimination::new();
    let result = dce.run(&mut translator, &mut cache);

    assert!(result.changed);
    assert_eq!(result.stats.instructions_removed, 1);
    assert_eq!(translator.blocks[0].instructions.len(), 2);
}

#[test]
fn test_dce_keeps_side_effects() {
    let mut translator = TestTranslator::new();

    // v0 := 10
    // print v0  (has side effects)
    // v1 := 20 (dead)
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Literal(20),
    });

    translator.seal_block(BlockId(0));

    let mut cache = AnalysisCache::new();
    let mut dce = DeadCodeElimination::new();
    let result = dce.run(&mut translator, &mut cache);

    assert!(result.changed);
    // Should keep v0 (used by print), print (side effect), remove v1 (dead)
    assert_eq!(translator.blocks[0].instructions.len(), 2);
}

#[test]
fn test_dce_keeps_transitive_uses() {
    let mut translator = TestTranslator::new();

    // v0 := 10
    // v1 := v0 + 5
    // print v1
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v1"),
        left: Value::Var(SsaVariable::new("v0")),
        op: BinaryOperator::Add,
        right: Value::Literal(5),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v1")),
    });

    translator.seal_block(BlockId(0));

    let mut cache = AnalysisCache::new();
    let mut dce = DeadCodeElimination::new();
    let result = dce.run(&mut translator, &mut cache);

    // Nothing should be removed - all are used
    assert!(!result.changed);
    assert_eq!(translator.blocks[0].instructions.len(), 3);
}

// ============================================================================
// Copy Propagation Tests
// ============================================================================

#[test]
fn test_copy_prop_simple() {
    let mut translator = TestTranslator::new();

    // v0 := 10
    // v1 := v0  (copy)
    // print v1  -> should become print v0
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v1")),
    });

    translator.seal_block(BlockId(0));

    let mut cache = AnalysisCache::new();
    let mut copy_prop = CopyPropagation::new();
    let result = copy_prop.run(&mut translator, &mut cache);

    assert!(result.changed);
    assert!(result.stats.values_propagated > 0);

    // Check that the print now uses v0
    if let Instruction::Print { value } = &translator.blocks[0].instructions[2] {
        assert_eq!(value, &Value::Var(SsaVariable::new("v0")));
    } else {
        panic!("Expected Print instruction");
    }
}

#[test]
fn test_copy_prop_chain() {
    let mut translator = TestTranslator::new();

    // v0 := 10
    // v1 := v0
    // v2 := v1
    // print v2  -> should become print v0
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v2"),
        value: Value::Var(SsaVariable::new("v1")),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v2")),
    });

    translator.seal_block(BlockId(0));

    let mut cache = AnalysisCache::new();
    let mut copy_prop = CopyPropagation::new();
    let result = copy_prop.run(&mut translator, &mut cache);

    assert!(result.changed);

    // Check that print now uses v0 (the original)
    if let Instruction::Print { value } = &translator.blocks[0].instructions[3] {
        assert_eq!(value, &Value::Var(SsaVariable::new("v0")));
    } else {
        panic!("Expected Print instruction");
    }
}

// ============================================================================
// Constant Propagation Tests
// ============================================================================

#[test]
fn test_const_prop_simple() {
    let mut translator = TestTranslator::new();

    // v0 := 10
    // v1 := v0 + 5  -> after const prop: v1 := 10 + 5
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v1"),
        left: Value::Var(SsaVariable::new("v0")),
        op: BinaryOperator::Add,
        right: Value::Literal(5),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v1")),
    });

    translator.seal_block(BlockId(0));

    let mut cache = AnalysisCache::new();
    let mut const_prop = ConstantPropagation::new();
    let result = const_prop.run(&mut translator, &mut cache);

    assert!(result.changed);

    // Check that the binary op now has constant left operand
    if let Instruction::BinaryOp { left, .. } = &translator.blocks[0].instructions[1] {
        assert_eq!(left, &Value::Literal(10));
    } else {
        panic!("Expected BinaryOp instruction");
    }
}

// ============================================================================
// Constant Folding Tests
// ============================================================================

#[test]
fn test_const_fold_add() {
    let mut translator = TestTranslator::new();

    // v0 := 10 + 20  -> should become v0 := 30
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v0"),
        left: Value::Literal(10),
        op: BinaryOperator::Add,
        right: Value::Literal(20),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v0")),
    });

    translator.seal_block(BlockId(0));

    let mut cache = AnalysisCache::new();
    let mut const_fold = ConstantFolding::new();
    let result = const_fold.run(&mut translator, &mut cache);

    assert!(result.changed);
    assert_eq!(result.stats.expressions_folded, 1);

    // Check that it's now an assignment of 30
    if let Instruction::Assign { value, .. } = &translator.blocks[0].instructions[0] {
        assert_eq!(value, &Value::Literal(30));
    } else {
        panic!("Expected Assign instruction after folding");
    }
}

#[test]
fn test_const_fold_comparison() {
    let mut translator = TestTranslator::new();

    // v0 := 10 < 20  -> should become v0 := 1 (true)
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v0"),
        left: Value::Literal(10),
        op: BinaryOperator::LessThan,
        right: Value::Literal(20),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v0")),
    });

    translator.seal_block(BlockId(0));

    let mut cache = AnalysisCache::new();
    let mut const_fold = ConstantFolding::new();
    let result = const_fold.run(&mut translator, &mut cache);

    assert!(result.changed);

    // Check that it's now 1
    if let Instruction::Assign { value, .. } = &translator.blocks[0].instructions[0] {
        assert_eq!(value, &Value::Literal(1));
    } else {
        panic!("Expected Assign instruction after folding");
    }
}

#[test]
fn test_const_fold_no_divide_by_zero() {
    let mut translator = TestTranslator::new();

    // v0 := 10 / 0  -> should NOT fold (would be division by zero)
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v0"),
        left: Value::Literal(10),
        op: BinaryOperator::Divide,
        right: Value::Literal(0),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v0")),
    });

    translator.seal_block(BlockId(0));

    let mut cache = AnalysisCache::new();
    let mut const_fold = ConstantFolding::new();
    let result = const_fold.run(&mut translator, &mut cache);

    // Should not fold due to divide by zero
    assert!(!result.changed);
}

// ============================================================================
// CSE Tests
// ============================================================================

#[test]
fn test_cse_simple() {
    let mut translator = TestTranslator::new();

    // v0 := a
    // v1 := b
    // v2 := v0 + v1
    // v3 := v0 + v1  (same expression - should reuse v2)
    // print v3
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Literal(20),
    });
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v2"),
        left: Value::Var(SsaVariable::new("v0")),
        op: BinaryOperator::Add,
        right: Value::Var(SsaVariable::new("v1")),
    });
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v3"),
        left: Value::Var(SsaVariable::new("v0")),
        op: BinaryOperator::Add,
        right: Value::Var(SsaVariable::new("v1")),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v3")),
    });

    translator.seal_block(BlockId(0));

    let mut cache = AnalysisCache::new();
    let mut cse = CommonSubexpressionElimination::new();
    let result = cse.run(&mut translator, &mut cache);

    assert!(result.changed);
    assert_eq!(result.stats.cse_eliminations, 1);

    // The print should now use v2 instead of v3
    if let Instruction::Print { value } = &translator.blocks[0].instructions[4] {
        assert_eq!(value, &Value::Var(SsaVariable::new("v2")));
    } else {
        panic!("Expected Print instruction");
    }
}

// ============================================================================
// Pipeline Tests
// ============================================================================

#[test]
fn test_pipeline_standard() {
    let mut translator = TestTranslator::new();

    // v0 := 10
    // v1 := v0  (copy, will be propagated)
    // v2 := 20 (dead, will be eliminated)
    // print v1
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v2"),
        value: Value::Literal(20),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v1")),
    });

    translator.seal_block(BlockId(0));

    let mut pipeline: OptimizationPipeline<Value, Instruction, InstructionBuilder> =
        OptimizationPipeline::standard();

    let result = pipeline.run(&mut translator);

    assert!(result.changed);
    // After copy prop + DCE: v0 := 10, print v0 (v1 and v2 eliminated)
    // The exact count depends on pass order
}

#[test]
fn test_pipeline_aggressive() {
    let mut translator = TestTranslator::new();

    // v0 := 5 + 5     (will be folded to 10)
    // v1 := v0        (copy, will be propagated)
    // v2 := v0 + v0   (will compute from folded constant)
    // v3 := v0 + v0   (CSE: same as v2)
    // print v3
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v0"),
        left: Value::Literal(5),
        op: BinaryOperator::Add,
        right: Value::Literal(5),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v2"),
        left: Value::Var(SsaVariable::new("v0")),
        op: BinaryOperator::Add,
        right: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v3"),
        left: Value::Var(SsaVariable::new("v0")),
        op: BinaryOperator::Add,
        right: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v3")),
    });

    translator.seal_block(BlockId(0));

    let mut pipeline: OptimizationPipeline<Value, Instruction, InstructionBuilder> =
        OptimizationPipeline::aggressive();

    let result = pipeline.run_until_fixed_point(&mut translator, 5);

    assert!(result.changed);
    // Should have folded 5+5=10, eliminated CSE, etc.
}

#[test]
fn test_pipeline_fixed_point() {
    let mut translator = TestTranslator::new();

    // This setup benefits from multiple iterations:
    // v0 := 2 + 3    (fold to 5)
    // v1 := v0       (copy prop makes v1 -> v0)
    // v2 := v1 + v1  (after copy prop: v0 + v0, then const prop makes 5 + 5, then fold to 10)
    // print v2
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v0"),
        left: Value::Literal(2),
        op: BinaryOperator::Add,
        right: Value::Literal(3),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v2"),
        left: Value::Var(SsaVariable::new("v1")),
        op: BinaryOperator::Add,
        right: Value::Var(SsaVariable::new("v1")),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v2")),
    });

    translator.seal_block(BlockId(0));

    let mut pipeline: OptimizationPipeline<Value, Instruction, InstructionBuilder> =
        OptimizationPipeline::aggressive();

    let result = pipeline.run_until_fixed_point(&mut translator, 10);

    // Multiple iterations should converge
    assert!(result.iterations >= 1);
}

// ============================================================================
// Analysis Tests
// ============================================================================

#[test]
fn test_liveness_analysis() {
    let mut translator = TestTranslator::new();

    // v0 := 10
    // v1 := 20
    // v2 := v0 + v1
    // print v2
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Literal(20),
    });
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v2"),
        left: Value::Var(SsaVariable::new("v0")),
        op: BinaryOperator::Add,
        right: Value::Var(SsaVariable::new("v1")),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v2")),
    });

    translator.seal_block(BlockId(0));

    let mut cache: AnalysisCache<Value, Instruction> = AnalysisCache::new();
    let liveness = cache.liveness(&translator);

    // Basic structure checks
    assert!(liveness.live_in.contains_key(&BlockId(0)), "Block 0 should have live_in");
    assert!(liveness.live_out.contains_key(&BlockId(0)), "Block 0 should have live_out");
    assert!(liveness.defs.contains_key(&BlockId(0)), "Block 0 should have defs");

    // Check that defs includes v0, v1, v2 (all defined in block 0)
    let defs = liveness.defs.get(&BlockId(0)).unwrap();
    assert!(defs.contains(&SsaVariable::new("v0")), "v0 should be in defs");
    assert!(defs.contains(&SsaVariable::new("v1")), "v1 should be in defs");
    assert!(defs.contains(&SsaVariable::new("v2")), "v2 should be in defs");

    // Cache should now have liveness
    assert!(cache.has_liveness());
}

#[test]
fn test_use_def_chains() {
    let translator = create_simple_translator();

    let mut cache: AnalysisCache<Value, Instruction> = AnalysisCache::new();
    let use_def = cache.use_def(&translator);

    // v0 should be defined
    let v0 = SsaVariable::new("v0");
    assert!(use_def.get_def(&v0).is_some());

    // v0 should have uses (used by v2's computation)
    assert!(use_def.has_uses(&v0));
}

#[test]
fn test_analysis_cache_invalidation() {
    let translator = create_simple_translator();

    let mut cache: AnalysisCache<Value, Instruction> = AnalysisCache::new();

    // Compute liveness
    let _ = cache.liveness(&translator);
    assert!(cache.has_liveness());

    // Invalidate
    cache.invalidate(&Invalidations::all());
    assert!(!cache.has_liveness());
    assert!(!cache.has_use_def());
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_translator() {
    let mut translator = TestTranslator::new();
    translator.seal_block(BlockId(0));

    let mut cache = AnalysisCache::new();
    let mut dce = DeadCodeElimination::new();
    let result = dce.run(&mut translator, &mut cache);

    // No changes on empty program
    assert!(!result.changed);
}

#[test]
fn test_all_dead_code() {
    let mut translator = TestTranslator::new();

    // All dead code - no side effects
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Literal(20),
    });
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v2"),
        left: Value::Var(SsaVariable::new("v0")),
        op: BinaryOperator::Add,
        right: Value::Var(SsaVariable::new("v1")),
    });

    translator.seal_block(BlockId(0));

    let mut cache = AnalysisCache::new();
    let mut dce = DeadCodeElimination::new();
    let result = dce.run(&mut translator, &mut cache);

    assert!(result.changed);
    assert_eq!(result.stats.instructions_removed, 3);
    assert!(translator.blocks[0].instructions.is_empty());
}

// ============================================================================
// Verification Tests
// ============================================================================

#[test]
fn test_pipeline_with_verification() {
    let mut translator = TestTranslator::new();

    // v0 := 10
    // v1 := v0  (copy, will be propagated)
    // v2 := 20 (dead, will be eliminated)
    // print v1
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v2"),
        value: Value::Literal(20),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v1")),
    });

    translator.seal_block(BlockId(0));

    // Use pipeline with verification enabled
    let mut pipeline: OptimizationPipeline<Value, Instruction, InstructionBuilder> =
        OptimizationPipeline::new_with_verify();
    pipeline.add_pass(CopyPropagation::new());
    pipeline.add_pass(ConstantPropagation::new());
    pipeline.add_pass(DeadCodeElimination::new());

    let result = pipeline.run(&mut translator);

    assert!(result.changed);
    // Verification should pass - if not, this test will fail
    result.assert_valid();
}

#[test]
fn test_aggressive_pipeline_with_verification() {
    let mut translator = TestTranslator::new();

    // v0 := 5 + 5     (will be folded to 10)
    // v1 := v0        (copy, will be propagated)
    // v2 := v0 + v0   (will compute from folded constant)
    // v3 := v0 + v0   (CSE: same as v2)
    // print v3
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v0"),
        left: Value::Literal(5),
        op: BinaryOperator::Add,
        right: Value::Literal(5),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v2"),
        left: Value::Var(SsaVariable::new("v0")),
        op: BinaryOperator::Add,
        right: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v3"),
        left: Value::Var(SsaVariable::new("v0")),
        op: BinaryOperator::Add,
        right: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v3")),
    });

    translator.seal_block(BlockId(0));

    // Use aggressive pipeline with verification enabled
    let mut pipeline: OptimizationPipeline<Value, Instruction, InstructionBuilder> =
        OptimizationPipeline::new_with_verify();
    pipeline.add_pass(CopyPropagation::new());
    pipeline.add_pass(ConstantPropagation::new());
    pipeline.add_pass(ConstantFolding::new());
    pipeline.add_pass(CommonSubexpressionElimination::new());
    pipeline.add_pass(DeadCodeElimination::new());

    let result = pipeline.run_until_fixed_point(&mut translator, 5);

    assert!(result.changed);
    // Verification should pass after all passes
    result.assert_valid();
}

// ============================================================================
// New Safe Optimizer API Tests
// ============================================================================

use ssa_test::optim::{Optimizer, optimize, optimize_aggressive};

#[test]
fn test_optimizer_simple() {
    let mut translator = TestTranslator::new();

    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v1")),
    });

    translator.seal_block(BlockId(0));

    // New safe API - returns Result, DCE auto-included, verification on
    let result = Optimizer::new(&mut translator)
        .copy_propagation()
        .run()
        .expect("optimization should succeed");

    assert!(result.changed);
}

#[test]
fn test_optimizer_with_constant_folding() {
    let mut translator = TestTranslator::new();

    // v0 := 5 + 5 (foldable)
    // print v0
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v0"),
        left: Value::Literal(5),
        op: BinaryOperator::Add,
        right: Value::Literal(5),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v0")),
    });

    translator.seal_block(BlockId(0));

    let result = Optimizer::new(&mut translator)
        .constant_folding()
        .run()
        .expect("optimization should succeed");

    assert!(result.changed);
    assert!(result.stats.expressions_folded > 0);
}

#[test]
fn test_optimizer_fixed_point() {
    let mut translator = TestTranslator::new();

    // Setup that benefits from multiple iterations
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v0"),
        left: Value::Literal(2),
        op: BinaryOperator::Add,
        right: Value::Literal(3),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v1")),
    });

    translator.seal_block(BlockId(0));

    let result = Optimizer::new(&mut translator)
        .copy_propagation()
        .constant_propagation()
        .constant_folding()
        .run_to_fixed_point(10)
        .expect("optimization should succeed");

    assert!(result.changed);
}

#[test]
fn test_optimize_convenience_function() {
    let mut translator = TestTranslator::new();

    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(42),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v0")),
    });

    translator.seal_block(BlockId(0));

    // Convenience function - easiest path
    let result = optimize(&mut translator).expect("optimization should succeed");

    // No changes expected (already optimal), but should still succeed
    assert!(!result.has_validation_errors());
}

#[test]
fn test_optimize_aggressive_convenience_function() {
    let mut translator = TestTranslator::new();

    // Complex setup
    translator.emit(Instruction::BinaryOp {
        dest: SsaVariable::new("v0"),
        left: Value::Literal(5),
        op: BinaryOperator::Add,
        right: Value::Literal(5),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v1")),
    });

    translator.seal_block(BlockId(0));

    let result = optimize_aggressive(&mut translator, 5)
        .expect("optimization should succeed");

    assert!(result.changed);
}

#[test]
fn test_optimizer_no_auto_cleanup() {
    let mut translator = TestTranslator::new();

    // v0 := 10
    // v1 := v0 (dead after copy prop)
    // print v0
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Var(SsaVariable::new("v0")),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v0")),
    });

    translator.seal_block(BlockId(0));

    // Explicitly disable auto-cleanup
    let result = Optimizer::new(&mut translator)
        .copy_propagation()
        .no_auto_cleanup()  // Explicit opt-out
        .dce()              // Manually add DCE
        .run()
        .expect("optimization should succeed");

    assert!(result.changed);
}

#[test]
fn test_optimizer_unchecked_mode() {
    let mut translator = TestTranslator::new();

    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("v0")),
    });

    translator.seal_block(BlockId(0));

    // Use unchecked mode (returns PipelineResult, not Result)
    let result = Optimizer::new(&mut translator)
        .copy_propagation()
        .no_verification()
        .run_unchecked();

    // Can inspect raw result
    assert!(!result.has_validation_errors()); // Empty because verification disabled
}

// ============================================================================
// Trampoline Elimination Tests
// ============================================================================

use ssa_test::optim::eliminate_trampolines;

/// Helper to create a CFG with a trampoline block:
/// B0: if cond goto B1 else goto B2
/// B1: (real code) goto B3
/// B2: goto B3  <- TRAMPOLINE
/// B3: (real code)
fn create_cfg_with_trampoline() -> TestTranslator {
    let mut translator = TestTranslator::new();

    // Block 0: entry with conditional
    let cond = SsaVariable::new("cond");
    translator.emit(Instruction::Assign {
        dest: cond.clone(),
        value: Value::Literal(1),
    });
    translator.emit(Instruction::ConditionalJump {
        condition: Value::Var(cond),
        true_target: BlockId(1),
        false_target: BlockId(2),
    });
    translator.seal_block(BlockId(0));

    // Block 1: real code
    let b1 = translator.create_block();
    assert_eq!(b1, BlockId(1));
    translator.blocks[1].predecessors.push(BlockId(0));
    translator.current_block = b1;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Literal(100),
    });
    translator.emit(Instruction::Jump { target: BlockId(3) });
    translator.seal_block(b1);

    // Block 2: TRAMPOLINE (only a jump)
    let b2 = translator.create_block();
    assert_eq!(b2, BlockId(2));
    translator.blocks[2].predecessors.push(BlockId(0));
    translator.current_block = b2;
    translator.emit(Instruction::Jump { target: BlockId(3) });
    translator.seal_block(b2);

    // Block 3: merge point
    let b3 = translator.create_block();
    assert_eq!(b3, BlockId(3));
    translator.blocks[3].predecessors.push(BlockId(1));
    translator.blocks[3].predecessors.push(BlockId(2));
    translator.current_block = b3;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("result"),
        value: Value::Literal(42),
    });
    translator.seal_block(b3);

    translator
}

#[test]
fn test_eliminate_trampolines_basic() {
    let mut translator = create_cfg_with_trampoline();

    // Verify initial state: B2 is a trampoline
    assert_eq!(translator.blocks[2].instructions.len(), 1);
    assert_eq!(translator.blocks[2].instructions[0].jump_targets(), vec![BlockId(3)]);

    // Eliminate trampolines
    let count = eliminate_trampolines(&mut translator);

    // Should have eliminated 1 trampoline
    assert_eq!(count, 1, "Should eliminate 1 trampoline");

    // B2 should now be empty
    assert!(translator.blocks[2].instructions.is_empty(), "Trampoline should be cleared");
    assert!(translator.blocks[2].predecessors.is_empty(), "Trampoline preds should be cleared");

    // B0's conditional should now jump directly to B3 (not B2)
    let b0_targets = translator.blocks[0].instructions.last().unwrap().jump_targets();
    assert!(b0_targets.contains(&BlockId(3)), "B0 should jump directly to B3, not B2");
    assert!(!b0_targets.contains(&BlockId(2)), "B0 should not reference trampoline B2");

    // B3's predecessors should include B0 (replacing B2)
    assert!(translator.blocks[3].predecessors.contains(&BlockId(0)),
        "B3 should have B0 as predecessor (was B2)");
}

#[test]
fn test_eliminate_trampolines_chain() {
    // Create a chain: B0 -> B1 -> B2 -> B3 where B1 and B2 are trampolines
    let mut translator = TestTranslator::new();

    // Block 0: entry
    translator.emit(Instruction::Jump { target: BlockId(1) });
    translator.seal_block(BlockId(0));

    // Block 1: trampoline
    let b1 = translator.create_block();
    translator.blocks[1].predecessors.push(BlockId(0));
    translator.current_block = b1;
    translator.emit(Instruction::Jump { target: BlockId(2) });
    translator.seal_block(b1);

    // Block 2: trampoline
    let b2 = translator.create_block();
    translator.blocks[2].predecessors.push(BlockId(1));
    translator.current_block = b2;
    translator.emit(Instruction::Jump { target: BlockId(3) });
    translator.seal_block(b2);

    // Block 3: real destination
    let b3 = translator.create_block();
    translator.blocks[3].predecessors.push(BlockId(2));
    translator.current_block = b3;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("x"),
        value: Value::Literal(1),
    });
    translator.seal_block(b3);

    // Eliminate trampolines
    let count = eliminate_trampolines(&mut translator);

    // Should eliminate 2 trampolines
    assert_eq!(count, 2, "Should eliminate 2 trampolines in chain");

    // B0 should now jump directly to B3
    let b0_targets = translator.blocks[0].instructions.last().unwrap().jump_targets();
    assert_eq!(b0_targets, vec![BlockId(3)], "B0 should jump directly to B3");
}

#[test]
fn test_eliminate_trampolines_none() {
    // Create a CFG with no trampolines
    let mut translator = TestTranslator::new();

    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Jump { target: BlockId(1) });
    translator.seal_block(BlockId(0));

    let b1 = translator.create_block();
    translator.blocks[1].predecessors.push(BlockId(0));
    translator.current_block = b1;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Literal(20),
    });
    // B1 has real code, not just a jump
    translator.seal_block(b1);

    let count = eliminate_trampolines(&mut translator);

    assert_eq!(count, 0, "Should not eliminate any blocks");
}

#[test]
fn test_eliminate_trampolines_preserves_entry() {
    // Entry block should never be considered a trampoline even if it's just a jump
    let mut translator = TestTranslator::new();

    // Block 0: entry (just a jump, but should NOT be eliminated)
    translator.emit(Instruction::Jump { target: BlockId(1) });
    translator.seal_block(BlockId(0));

    let b1 = translator.create_block();
    translator.blocks[1].predecessors.push(BlockId(0));
    translator.current_block = b1;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("x"),
        value: Value::Literal(1),
    });
    translator.seal_block(b1);

    let count = eliminate_trampolines(&mut translator);

    // Entry block should not be eliminated
    assert_eq!(count, 0, "Entry block should not be eliminated");
    assert!(!translator.blocks[0].instructions.is_empty(), "Entry block should still have instructions");
}

#[test]
fn test_phi_elimination_with_cleanup() {
    use ssa_test::optim::PhiElimination;

    // Create a simple diamond CFG that will have phis
    let mut translator = TestTranslator::new();

    // Block 0: entry with conditional
    let cond = SsaVariable::new("cond");
    translator.emit(Instruction::Assign {
        dest: cond.clone(),
        value: Value::Literal(1),
    });
    translator.emit(Instruction::ConditionalJump {
        condition: Value::Var(cond),
        true_target: BlockId(1),
        false_target: BlockId(2),
    });
    translator.seal_block(BlockId(0));

    // Block 1: then branch
    let b1 = translator.create_block();
    translator.blocks[1].predecessors.push(BlockId(0));
    translator.current_block = b1;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("x"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Jump { target: BlockId(3) });
    translator.seal_block(b1);

    // Block 2: else branch
    let b2 = translator.create_block();
    translator.blocks[2].predecessors.push(BlockId(0));
    translator.current_block = b2;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("y"),
        value: Value::Literal(20),
    });
    translator.emit(Instruction::Jump { target: BlockId(3) });
    translator.seal_block(b2);

    // Block 3: merge
    let b3 = translator.create_block();
    translator.blocks[3].predecessors.push(BlockId(1));
    translator.blocks[3].predecessors.push(BlockId(2));
    translator.current_block = b3;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("result"),
        value: Value::Literal(42),
    });
    translator.seal_block(b3);

    // Use the combined eliminate + cleanup
    let trampolines_removed = PhiElimination::eliminate_with_cleanup(&mut translator);

    // After phi elimination, there should be no phis
    assert!(translator.phis.is_empty(), "All phis should be eliminated");

    // The function should return 0 or more depending on if any trampolines were created
    // (In this simple case, likely 0 since no critical edges needed splitting)
    // Just verify it completes without panic - any value is valid
    let _ = trampolines_removed;
}

// ============================================================================
// CfgCleanup Bug Reproduction Tests (BUG_REPORT_CFG_CLEANUP.md)
// ============================================================================

/// Test that CfgCleanup doesn't corrupt phi nodes when predecessor lists
/// are out of sync with terminators.
///
/// Bug: When rebuild_predecessors_from_terminators discovers NEW predecessors
/// that weren't in the old list, it updates the predecessor list but doesn't
/// add corresponding phi operands, causing a mismatch.
#[test]
fn test_cfg_cleanup_doesnt_add_phantom_predecessors() {
    use ssa_test::optim::passes::CfgCleanup;

    let mut translator = TestTranslator::new();

    // Create a diamond CFG:
    // Block 0 (entry) -> Block 1 or Block 2 -> Block 3 (merge with phi)

    // Block 0: entry with conditional
    let cond = SsaVariable::new("cond");
    translator.emit(Instruction::Assign {
        dest: cond.clone(),
        value: Value::Literal(1),
    });
    translator.emit(Instruction::ConditionalJump {
        condition: Value::Var(cond),
        true_target: BlockId(1),
        false_target: BlockId(2),
    });
    translator.seal_block(BlockId(0));

    // Block 1: then branch
    let b1 = translator.create_block();
    translator.blocks[1].predecessors.push(BlockId(0));
    translator.current_block = b1;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("x"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Jump { target: BlockId(3) });
    translator.seal_block(b1);

    // Block 2: else branch
    let b2 = translator.create_block();
    translator.blocks[2].predecessors.push(BlockId(0));
    translator.current_block = b2;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("y"),
        value: Value::Literal(20),
    });
    translator.emit(Instruction::Jump { target: BlockId(3) });
    translator.seal_block(b2);

    // Block 3: merge point
    let b3 = translator.create_block();
    translator.blocks[3].predecessors.push(BlockId(1));
    translator.blocks[3].predecessors.push(BlockId(2));
    translator.current_block = b3;

    // Create a phi node manually that merges values from blocks 1 and 2
    let phi_id = translator.create_phi(b3);
    translator.phis.get_mut(&phi_id).unwrap().operands.push(Value::Literal(10)); // from block 1
    translator.phis.get_mut(&phi_id).unwrap().operands.push(Value::Literal(20)); // from block 2

    translator.emit(Instruction::Print {
        value: Value::Phi(phi_id),
    });
    translator.seal_block(b3);

    // Verify initial state
    assert_eq!(translator.blocks[3].predecessors.len(), 2, "Block 3 should have 2 predecessors");
    assert_eq!(translator.phis[&phi_id].operands.len(), 2, "Phi should have 2 operands");

    // Run CfgCleanup
    let mut cache = AnalysisCache::new();
    let mut cfg_cleanup = CfgCleanup::new();
    let _result = cfg_cleanup.run(&mut translator, &mut cache);

    // After CfgCleanup, the phi operand count must still match predecessor count
    let pred_count = translator.blocks[3].predecessors.len();
    let phi_operand_count = translator.phis.get(&phi_id).map(|p| p.operands.len()).unwrap_or(0);

    assert_eq!(
        pred_count, phi_operand_count,
        "Phi operand count ({}) must match predecessor count ({}) after CfgCleanup",
        phi_operand_count, pred_count
    );
}

/// Test that CfgCleanup preserves phi/predecessor correspondence when predecessors
/// are intentionally incomplete (simulating out-of-sync state from other passes).
///
/// This test creates a scenario where:
/// - Block 4 has 2 predecessors listed (Block 1, Block 2)
/// - Phi has 2 operands
/// - But Block 3 ALSO jumps to Block 4 (not listed as predecessor)
/// - After CfgCleanup: 3 predecessors but still only 2 phi operands = BUG!
#[test]
fn test_cfg_cleanup_with_incomplete_predecessors() {
    use ssa_test::optim::passes::CfgCleanup;

    let mut translator = TestTranslator::new();

    // Create a CFG where we intentionally set predecessors incorrectly
    // to simulate what happens after certain optimization passes
    //
    // Blocks:
    // 0 (entry) -> branches to 1, 2, 3
    // 1 -> 4
    // 2 -> 4
    // 3 -> 4 (but not listed as predecessor of 4!)
    // 4: merge with phi (2 operands for blocks 1 and 2, but NOT for 3)

    // Block 0: entry with 3-way branch (simulated via two conditionals)
    let cond = SsaVariable::new("cond");
    translator.emit(Instruction::Assign {
        dest: cond.clone(),
        value: Value::Literal(1),
    });
    translator.emit(Instruction::ConditionalJump {
        condition: Value::Var(cond.clone()),
        true_target: BlockId(1),
        false_target: BlockId(2),
    });
    translator.seal_block(BlockId(0));

    // Block 1 -> Block 4
    let b1 = translator.create_block();
    translator.blocks[1].predecessors.push(BlockId(0));
    translator.current_block = b1;
    translator.emit(Instruction::Jump { target: BlockId(4) });
    translator.seal_block(b1);

    // Block 2 -> Block 3 (intermediate block that also goes to 4)
    let b2 = translator.create_block();
    translator.blocks[2].predecessors.push(BlockId(0));
    translator.current_block = b2;
    translator.emit(Instruction::ConditionalJump {
        condition: Value::Var(cond.clone()),
        true_target: BlockId(3),
        false_target: BlockId(4),
    });
    translator.seal_block(b2);

    // Block 3 -> Block 4 (this edge is NOT reflected in Block 4's predecessor list!)
    let b3 = translator.create_block();
    translator.blocks[3].predecessors.push(BlockId(2));
    translator.current_block = b3;
    translator.emit(Instruction::Jump { target: BlockId(4) });
    translator.seal_block(b3);

    // Block 4: merge point
    // INTENTIONALLY only list Block 1 and Block 2 as predecessors
    // (missing Block 3 even though Block 3 DOES jump here!)
    let b4 = translator.create_block();
    translator.blocks[4].predecessors.push(BlockId(1));
    translator.blocks[4].predecessors.push(BlockId(2));
    // Note: Block 3 is NOT listed as predecessor, but it DOES jump to Block 4!
    translator.current_block = b4;

    // Create a phi with 2 operands (matching the incomplete predecessor list)
    let phi_id = translator.create_phi(b4);
    translator.phis.get_mut(&phi_id).unwrap().operands.push(Value::Literal(10)); // from block 1
    translator.phis.get_mut(&phi_id).unwrap().operands.push(Value::Literal(20)); // from block 2
    // Note: no operand for Block 3!

    translator.emit(Instruction::Print {
        value: Value::Phi(phi_id),
    });
    translator.seal_block(b4);

    // Verify initial broken state: 2 predecessors listed, but 3 blocks actually jump here
    assert_eq!(translator.blocks[4].predecessors.len(), 2, "Initial: Block 4 has 2 listed predecessors");
    assert_eq!(translator.phis[&phi_id].operands.len(), 2, "Initial: Phi has 2 operands");

    // Run CfgCleanup - this should NOT corrupt the phi
    let mut cache = AnalysisCache::new();
    let mut cfg_cleanup = CfgCleanup::new();
    let _result = cfg_cleanup.run(&mut translator, &mut cache);

    // After CfgCleanup: either
    // 1. Predecessors stay at 2 (don't add new ones), OR
    // 2. Predecessors become 3 AND phi operands also become 3
    // The BUG is: predecessors = 3 but phi operands = 2

    let pred_count = translator.blocks[4].predecessors.len();
    let phi = translator.phis.get(&phi_id);

    // The phi should still exist (it has 2 operands, not 1)
    assert!(phi.is_some(), "Phi should still exist after CfgCleanup (had 2 operands)");

    let phi = phi.unwrap();
    let operand_count = phi.operands.len();

    assert_eq!(
        pred_count, operand_count,
        "BUG DETECTED: Predecessor count ({}) != phi operand count ({}) after CfgCleanup. \
        CfgCleanup added predecessors without adding phi operands!",
        pred_count, operand_count
    );
}

/// Test running CfgCleanup multiple times in a fixed-point loop doesn't corrupt phis.
/// This specifically tests the scenario from the bug report.
#[test]
fn test_cfg_cleanup_multiple_iterations_preserves_phis() {
    use ssa_test::optim::passes::CfgCleanup;

    let mut translator = TestTranslator::new();

    // Create the CFG from the bug report:
    // fn test(x, args) {
    //     let result = if x { null } else { read_field(args, 0) }
    //     return result
    // }

    // Block 0: entry with conditional check
    let x = SsaVariable::new("x");
    translator.emit(Instruction::Assign {
        dest: x.clone(),
        value: Value::Literal(1), // true condition
    });
    translator.emit(Instruction::ConditionalJump {
        condition: Value::Var(x),
        true_target: BlockId(1),  // then: return null
        false_target: BlockId(2), // else: return read_field
    });
    translator.seal_block(BlockId(0));

    // Block 1: then branch (result = null/0)
    let b1 = translator.create_block();
    translator.blocks[1].predecessors.push(BlockId(0));
    translator.current_block = b1;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("then_val"),
        value: Value::Literal(0), // null
    });
    translator.emit(Instruction::Jump { target: BlockId(3) });
    translator.seal_block(b1);

    // Block 2: else branch (result = read_field(args, 0))
    let b2 = translator.create_block();
    translator.blocks[2].predecessors.push(BlockId(0));
    translator.current_block = b2;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("else_val"),
        value: Value::Literal(42), // simulated read_field result
    });
    translator.emit(Instruction::Jump { target: BlockId(3) });
    translator.seal_block(b2);

    // Block 3: merge point with phi
    let b3 = translator.create_block();
    translator.blocks[3].predecessors.push(BlockId(1));
    translator.blocks[3].predecessors.push(BlockId(2));
    translator.current_block = b3;

    // Create phi merging the two branches
    let phi_id = translator.create_phi(b3);
    translator.phis.get_mut(&phi_id).unwrap().operands.push(Value::Literal(0));  // from block 1
    translator.phis.get_mut(&phi_id).unwrap().operands.push(Value::Literal(42)); // from block 2

    translator.emit(Instruction::Print {
        value: Value::Phi(phi_id),
    });
    translator.seal_block(b3);

    // Run CfgCleanup multiple times (simulating fixed-point iteration)
    let mut cache = AnalysisCache::new();
    let mut cfg_cleanup = CfgCleanup::new();

    for iteration in 1..=5 {
        let result = cfg_cleanup.run(&mut translator, &mut cache);

        // Check invariant after each iteration
        for (phi_id, phi) in &translator.phis {
            let block = &translator.blocks[phi.block_id.0];
            let pred_count = block.predecessors.len();
            let operand_count = phi.operands.len();

            assert_eq!(
                pred_count, operand_count,
                "Iteration {}: Phi {:?} in block {:?} has {} operands but block has {} predecessors",
                iteration, phi_id, phi.block_id, operand_count, pred_count
            );
        }

        if !result.changed {
            break;
        }
    }
}

/// Test that JumpThreading + CfgCleanup correctly maintains predecessors.
///
/// This tests the interaction after fixing CfgCleanup to not add new predecessors.
/// JumpThreading must now properly update predecessors itself.
#[test]
fn test_jump_threading_with_cfg_cleanup_maintains_predecessors() {
    use ssa_test::optim::passes::{JumpThreading, CfgCleanup};

    let mut translator = TestTranslator::new();

    // Create a simple chain with a trivial block:
    // Block 0 (entry) -> Block 1 (trivial, just jumps to 2) -> Block 2 (target)
    //
    // After JumpThreading:
    // Block 0 -> Block 2 directly
    // Block 1 becomes unreachable
    //
    // Block 2 should have predecessor [Block 0], not []

    // Block 0: entry, jumps to Block 1
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("x"),
        value: Value::Literal(42),
    });
    translator.emit(Instruction::Jump { target: BlockId(1) });
    translator.seal_block(BlockId(0));

    // Block 1: trivial block (single unconditional jump)
    let b1 = translator.create_block();
    translator.blocks[1].predecessors.push(BlockId(0));
    translator.current_block = b1;
    translator.emit(Instruction::Jump { target: BlockId(2) });
    translator.seal_block(b1);

    // Block 2: target block
    let b2 = translator.create_block();
    translator.blocks[2].predecessors.push(BlockId(1));
    translator.current_block = b2;
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("x")),
    });
    translator.seal_block(b2);

    // Verify initial state
    assert_eq!(translator.blocks[2].predecessors, vec![BlockId(1)], "Initial: Block 2 has predecessor [1]");

    // Run JumpThreading
    let mut cache = AnalysisCache::new();
    let mut jump_threading = JumpThreading::new();
    let jt_result = jump_threading.run(&mut translator, &mut cache);
    assert!(jt_result.changed, "JumpThreading should have made changes");

    // Run CfgCleanup
    let mut cfg_cleanup = CfgCleanup::new();
    let _cc_result = cfg_cleanup.run(&mut translator, &mut cache);

    // After both passes, Block 2 should have Block 0 as predecessor
    // (Block 0 now jumps directly to Block 2, bypassing the trivial Block 1)
    let block2_preds = &translator.blocks[2].predecessors;

    assert!(
        block2_preds.contains(&BlockId(0)),
        "Block 2 should have Block 0 as predecessor after JumpThreading + CfgCleanup. \
        Actual predecessors: {:?}",
        block2_preds
    );

    // Block 2 should NOT have Block 1 as predecessor anymore (it's unreachable)
    assert!(
        !block2_preds.contains(&BlockId(1)),
        "Block 2 should NOT have Block 1 as predecessor (it's unreachable). \
        Actual predecessors: {:?}",
        block2_preds
    );
}

/// Test that JumpThreading correctly copies phi operands when bypassing trivial blocks.
///
/// Bug: When JumpThreading adds a new predecessor to the ultimate target,
/// it was adding V::undefined() as the phi operand instead of copying
/// the phi operand from the trivial block being bypassed.
///
/// This causes incorrect code generation on 2+ iterations because:
/// 1. First iteration: adds undefined operand to phi
/// 2. Second iteration: phi uses undefined value, produces wrong result
#[test]
fn test_jump_threading_copies_phi_operands_for_bypassed_blocks() {
    use ssa_test::optim::passes::JumpThreading;

    let mut translator = TestTranslator::new();

    // Create a CFG where:
    // Block 0 (entry) -> Block 1 (trivial, just jumps)
    //                    Block 1 -> Block 2 (has phi)
    // Block 3 (alt path) ---------> Block 2
    //
    // Block 2 has phi merging values from Block 1 and Block 3
    //
    // After JumpThreading:
    // Block 0 -> Block 2 directly (bypassing Block 1)
    // Block 2's phi should have operand from Block 0 = same as operand from Block 1

    // Block 0: entry, defines value, jumps to Block 1 (trivial)
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("entry_val"),
        value: Value::Literal(100),
    });
    translator.emit(Instruction::Jump { target: BlockId(1) });
    translator.seal_block(BlockId(0));

    // Block 1: trivial block (single unconditional jump to Block 2)
    let b1 = translator.create_block();
    translator.blocks[1].predecessors.push(BlockId(0));
    translator.current_block = b1;
    translator.emit(Instruction::Jump { target: BlockId(2) });
    translator.seal_block(b1);

    // Block 2: merge point with phi
    let b2 = translator.create_block();
    translator.blocks[2].predecessors.push(BlockId(1)); // from trivial block
    translator.current_block = b2;

    // Create phi merging values - the key test data
    let phi_id = translator.create_phi(b2);
    // Operand from Block 1 (which came from Block 0 via trivial block): value 100
    translator.phis.get_mut(&phi_id).unwrap().operands.push(Value::Literal(100));

    translator.emit(Instruction::Print {
        value: Value::Phi(phi_id),
    });
    translator.seal_block(b2);

    // Verify initial state
    assert_eq!(translator.blocks[2].predecessors.len(), 1);
    assert_eq!(translator.phis[&phi_id].operands.len(), 1);
    assert_eq!(translator.phis[&phi_id].operands[0], Value::Literal(100));

    // Run JumpThreading
    let mut cache = AnalysisCache::new();
    let mut jump_threading = JumpThreading::new();
    let result = jump_threading.run(&mut translator, &mut cache);

    assert!(result.changed, "JumpThreading should have made changes");

    // After JumpThreading:
    // - Block 0 now jumps directly to Block 2 (bypassing Block 1)
    // - Block 2 should have Block 0 as a predecessor
    // - The phi operand for Block 0 should be the same as the operand for Block 1 (Value::Literal(100))
    //   NOT Value::Undefined!

    let block2_preds = &translator.blocks[2].predecessors;
    assert!(
        block2_preds.contains(&BlockId(0)),
        "Block 2 should have Block 0 as predecessor. Actual: {:?}",
        block2_preds
    );

    // The critical assertion: phi operand count should match predecessor count
    let pred_count = translator.blocks[2].predecessors.len();
    let operand_count = translator.phis[&phi_id].operands.len();
    assert_eq!(
        pred_count, operand_count,
        "Phi operand count ({}) should match predecessor count ({})",
        operand_count, pred_count
    );

    // THE BUG: The phi operand for the new predecessor (Block 0) was being set to undefined
    // instead of copying the operand from the trivial block (Block 1)
    // Check that NO operands are undefined
    for (i, operand) in translator.phis[&phi_id].operands.iter().enumerate() {
        assert!(
            !matches!(operand, Value::Undefined),
            "Bug detected: Phi operand {} is undefined. This happens when JumpThreading \
            adds V::undefined() instead of copying the operand from the bypassed trivial block. \
            All operands: {:?}",
            i,
            translator.phis[&phi_id].operands
        );
    }
}

/// Test that DCE handles recursive-like control flow with phi nodes correctly.
///
/// This tests the scenario from BUG_REPORT_DCE.md where a recursive function
/// that returns a boolean had its return value corrupted by DCE.
///
/// The issue may be subtle - related to how DCE handles phi nodes that merge
/// values from different branches, especially when some branches have side effects.
#[test]
fn test_dce_preserves_phi_in_recursive_pattern() {
    let mut translator = TestTranslator::new();

    // Simulate a recursive function pattern like:
    // fn check(n) {
    //     if n <= 0 {
    //         true               // base case
    //     } else {
    //         call side_effect() // side effect with unused result
    //         check(n - 1)       // recursive call (simulated as returning a value)
    //     }
    // }
    //
    // CFG:
    // Block 0 (entry): cond = n <= 0; br cond, Block1, Block2
    // Block 1 (base): result1 = true; jump Block3
    // Block 2 (recursive): _ = call(); result2 = false; jump Block3
    // Block 3 (merge): result = phi(result1, result2); return result

    // Block 0: entry with condition
    let n = SsaVariable::new("n");
    translator.emit(Instruction::Assign {
        dest: n.clone(),
        value: Value::Literal(5),
    });
    let cond = SsaVariable::new("cond");
    translator.emit(Instruction::BinaryOp {
        dest: cond.clone(),
        left: Value::Var(n),
        op: BinaryOperator::LessThanOrEqual,
        right: Value::Literal(0),
    });
    translator.emit(Instruction::ConditionalJump {
        condition: Value::Var(cond),
        true_target: BlockId(1),
        false_target: BlockId(2),
    });
    translator.seal_block(BlockId(0));

    // Block 1: base case - return true (1)
    let b1 = translator.create_block();
    translator.blocks[1].predecessors.push(BlockId(0));
    translator.current_block = b1;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("result1"),
        value: Value::Literal(1), // true
    });
    translator.emit(Instruction::Jump { target: BlockId(3) });
    translator.seal_block(b1);

    // Block 2: recursive case - call with side effect, then "recursive result"
    let b2 = translator.create_block();
    translator.blocks[2].predecessors.push(BlockId(0));
    translator.current_block = b2;
    // Side-effecting call with UNUSED result (the core of the bug)
    translator.emit(Instruction::Call {
        dest: SsaVariable::new("unused_call_result"),
        func: "swap".to_string(),
        args: vec![Value::Literal(1)],
    });
    // Simulated recursive result
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("result2"),
        value: Value::Literal(0), // would be recursive call result
    });
    translator.emit(Instruction::Jump { target: BlockId(3) });
    translator.seal_block(b2);

    // Block 3: merge point with phi for return value
    let b3 = translator.create_block();
    translator.blocks[3].predecessors.push(BlockId(1));
    translator.blocks[3].predecessors.push(BlockId(2));
    translator.current_block = b3;

    // Phi merging the two branches
    let phi_id = translator.create_phi(b3);
    translator.phis.get_mut(&phi_id).unwrap().operands.push(
        Value::Var(SsaVariable::new("result1"))
    );
    translator.phis.get_mut(&phi_id).unwrap().operands.push(
        Value::Var(SsaVariable::new("result2"))
    );
    translator.phis.get_mut(&phi_id).unwrap().dest = Some(SsaVariable::new("final_result"));

    // Use the phi result (simulating return)
    translator.emit(Instruction::Print {
        value: Value::Var(SsaVariable::new("final_result")),
    });
    translator.seal_block(b3);

    // Run DCE multiple times (simulating fixed-point iteration)
    let mut cache = AnalysisCache::new();
    let mut dce = DeadCodeElimination::new();

    for iteration in 1..=3 {
        let _result = dce.run(&mut translator, &mut cache);

        // Verify critical invariants after each iteration:

        // 1. The Call instruction must still exist (side effect)
        let has_call = translator.blocks.iter().any(|b| {
            b.instructions.iter().any(|i| matches!(i, Instruction::Call { .. }))
        });
        assert!(
            has_call,
            "Iteration {}: Call instruction was incorrectly removed!",
            iteration
        );

        // 2. The phi must still exist
        assert!(
            translator.phis.contains_key(&phi_id),
            "Iteration {}: Phi was incorrectly removed!",
            iteration
        );

        // 3. The phi must have 2 operands (one for each predecessor)
        let phi = &translator.phis[&phi_id];
        assert_eq!(
            phi.operands.len(), 2,
            "Iteration {}: Phi should have 2 operands, has {}",
            iteration, phi.operands.len()
        );

        // 4. The assignments feeding the phi must exist
        let has_result1 = translator.blocks[1].instructions.iter().any(|i| {
            matches!(i, Instruction::Assign { dest, .. } if dest.name() == "result1")
        });
        let has_result2 = translator.blocks[2].instructions.iter().any(|i| {
            matches!(i, Instruction::Assign { dest, .. } if dest.name() == "result2")
        });
        assert!(
            has_result1,
            "Iteration {}: result1 assignment was incorrectly removed!",
            iteration
        );
        assert!(
            has_result2,
            "Iteration {}: result2 assignment was incorrectly removed!",
            iteration
        );
    }
}

/// Test that DCE preserves side-effecting Call instructions even when their result is unused.
///
/// This tests the bug from BUG_REPORT_DCE.md where swap! operations were being
/// removed because their return values weren't used, even though they had side effects.
#[test]
fn test_dce_preserves_call_with_unused_result() {
    let mut translator = TestTranslator::new();

    // Create a scenario where a Call instruction's result is not used:
    // result := call side_effecting_function()
    // print 42  // result is never used!
    //
    // DCE should NOT remove the call because it has side effects

    // Call instruction with unused result
    translator.emit(Instruction::Call {
        dest: SsaVariable::new("unused_result"),
        func: "swap".to_string(),
        args: vec![Value::Literal(1)],
    });

    // Print something else (doesn't use the call result)
    translator.emit(Instruction::Print {
        value: Value::Literal(42),
    });

    translator.seal_block(BlockId(0));

    let initial_count = translator.blocks[0].instructions.len();
    assert_eq!(initial_count, 2, "Should have 2 instructions initially");

    // Run DCE
    let mut cache = AnalysisCache::new();
    let mut dce = DeadCodeElimination::new();
    let result = dce.run(&mut translator, &mut cache);

    // DCE should NOT have removed anything - both instructions have side effects
    assert!(
        !result.changed,
        "DCE should not remove side-effecting instructions. Removed {} instructions",
        result.stats.instructions_removed
    );
    assert_eq!(
        translator.blocks[0].instructions.len(),
        2,
        "Should still have 2 instructions after DCE"
    );

    // Verify the Call instruction is still there
    let has_call = translator.blocks[0].instructions.iter().any(|instr| {
        matches!(instr, Instruction::Call { .. })
    });
    assert!(
        has_call,
        "Call instruction should be preserved even with unused result"
    );
}

/// Test that DCE removes dead code but preserves the inputs to side-effecting calls.
#[test]
fn test_dce_preserves_inputs_to_side_effecting_calls() {
    let mut translator = TestTranslator::new();

    // Create:
    // v0 := 10
    // v1 := 20       <- dead (not used by anything)
    // result := call func(v0)  <- side effect, uses v0
    //
    // DCE should remove v1 but keep v0 and the call

    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v0"),
        value: Value::Literal(10),
    });
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("v1"),
        value: Value::Literal(20),
    });
    translator.emit(Instruction::Call {
        dest: SsaVariable::new("result"),
        func: "side_effect".to_string(),
        args: vec![Value::Var(SsaVariable::new("v0"))],
    });

    translator.seal_block(BlockId(0));

    let mut cache = AnalysisCache::new();
    let mut dce = DeadCodeElimination::new();
    let result = dce.run(&mut translator, &mut cache);

    // Should remove v1 (dead) but keep v0 (used by call) and call (side effect)
    assert!(result.changed, "DCE should have removed v1");
    assert_eq!(result.stats.instructions_removed, 1, "Should remove exactly 1 instruction (v1)");
    assert_eq!(translator.blocks[0].instructions.len(), 2, "Should have 2 instructions left");

    // Verify v0 assignment is preserved
    let has_v0 = translator.blocks[0].instructions.iter().any(|instr| {
        matches!(instr, Instruction::Assign { dest, .. } if dest.name() == "v0")
    });
    assert!(has_v0, "v0 assignment should be preserved (used by call)");

    // Verify Call is preserved
    let has_call = translator.blocks[0].instructions.iter().any(|instr| {
        matches!(instr, Instruction::Call { .. })
    });
    assert!(has_call, "Call instruction should be preserved");
}

/// Test that the validation catches undefined phi operands.
///
/// This test verifies that the UndefinedPhiOperand validation works correctly
/// to catch bugs like the one fixed in jump threading.
#[test]
fn test_validation_catches_undefined_phi_operands() {
    use ssa_test::validation::validate_ssa;
    use ssa_test::validation::SSAViolation;

    let mut translator = TestTranslator::new();

    // Create a simple CFG with a phi that has an undefined operand
    // This simulates what would happen if jump threading had the bug

    // Block 0: entry with condition
    let cond = SsaVariable::new("cond");
    translator.emit(Instruction::Assign {
        dest: cond.clone(),
        value: Value::Literal(1),
    });
    translator.emit(Instruction::ConditionalJump {
        condition: Value::Var(cond),
        true_target: BlockId(1),
        false_target: BlockId(2),
    });
    translator.seal_block(BlockId(0));

    // Block 1: true branch
    let b1 = translator.create_block();
    translator.blocks[1].predecessors.push(BlockId(0));
    translator.current_block = b1;
    translator.emit(Instruction::Jump { target: BlockId(2) });
    translator.seal_block(b1);

    // Block 2: merge with phi that has an undefined operand (simulating the bug)
    let b2 = translator.create_block();
    translator.blocks[2].predecessors.push(BlockId(0));
    translator.blocks[2].predecessors.push(BlockId(1));
    translator.current_block = b2;

    let phi_id = translator.create_phi(b2);
    translator.phis.get_mut(&phi_id).unwrap().operands.push(Value::Literal(1)); // Good operand
    translator.phis.get_mut(&phi_id).unwrap().operands.push(Value::Undefined);  // Bug: undefined!

    translator.emit(Instruction::Print {
        value: Value::Phi(phi_id),
    });
    translator.seal_block(b2);

    // Run validation - it should catch the undefined operand
    let violations = validate_ssa(&translator);

    // Find the UndefinedPhiOperand violation
    let has_undefined_violation = violations.iter().any(|v| {
        matches!(v, SSAViolation::UndefinedPhiOperand { .. })
    });

    assert!(
        has_undefined_violation,
        "Validation should catch undefined phi operand. Violations: {:?}",
        violations
    );
}

/// Test that JumpThreading works correctly across multiple iterations.
///
/// This specifically tests the bug from BUG_REPORT_JUMP_THREADING.md
/// where running JumpThreading 2+ times in a fixed-point loop causes
/// incorrect boolean return values.
#[test]
fn test_jump_threading_multiple_iterations_preserves_correctness() {
    use ssa_test::optim::passes::JumpThreading;

    let mut translator = TestTranslator::new();

    // Create a more complex CFG similar to the recursive check function:
    // Block 0: entry, conditional branch
    // Block 1: true branch, returns true
    // Block 2: false branch, has nested check
    // Block 3: trivial block jumping to Block 4
    // Block 4: merge block with phi

    // Block 0: entry with condition
    let cond = SsaVariable::new("cond");
    translator.emit(Instruction::Assign {
        dest: cond.clone(),
        value: Value::Literal(1), // true
    });
    translator.emit(Instruction::ConditionalJump {
        condition: Value::Var(cond),
        true_target: BlockId(1),
        false_target: BlockId(2),
    });
    translator.seal_block(BlockId(0));

    // Block 1: true branch - assign true (1) and jump to merge
    let b1 = translator.create_block();
    translator.blocks[1].predecessors.push(BlockId(0));
    translator.current_block = b1;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("result_true"),
        value: Value::Literal(1), // true
    });
    translator.emit(Instruction::Jump { target: BlockId(3) }); // Go through trivial block
    translator.seal_block(b1);

    // Block 2: false branch - assign false (0) and jump to merge
    let b2 = translator.create_block();
    translator.blocks[2].predecessors.push(BlockId(0));
    translator.current_block = b2;
    translator.emit(Instruction::Assign {
        dest: SsaVariable::new("result_false"),
        value: Value::Literal(0), // false
    });
    translator.emit(Instruction::Jump { target: BlockId(4) }); // Direct to merge
    translator.seal_block(b2);

    // Block 3: trivial block (just jumps to Block 4)
    let b3 = translator.create_block();
    translator.blocks[3].predecessors.push(BlockId(1));
    translator.current_block = b3;
    translator.emit(Instruction::Jump { target: BlockId(4) });
    translator.seal_block(b3);

    // Block 4: merge point with phi
    let b4 = translator.create_block();
    translator.blocks[4].predecessors.push(BlockId(3)); // from trivial block
    translator.blocks[4].predecessors.push(BlockId(2)); // from false branch
    translator.current_block = b4;

    // Create phi merging the boolean results
    let phi_id = translator.create_phi(b4);
    translator.phis.get_mut(&phi_id).unwrap().operands.push(Value::Literal(1)); // from Block 3 (true)
    translator.phis.get_mut(&phi_id).unwrap().operands.push(Value::Literal(0)); // from Block 2 (false)

    translator.emit(Instruction::Print {
        value: Value::Phi(phi_id),
    });
    translator.seal_block(b4);

    // Verify initial state
    assert_eq!(translator.blocks[4].predecessors.len(), 2);
    assert_eq!(translator.phis[&phi_id].operands.len(), 2);

    // Run JumpThreading MULTIPLE times (simulating fixed-point iteration)
    let mut cache = AnalysisCache::new();
    let mut jump_threading = JumpThreading::new();

    for iteration in 1..=3 {
        let result = jump_threading.run(&mut translator, &mut cache);

        // Check invariant after each iteration
        let pred_count = translator.blocks[4].predecessors.len();
        let operand_count = translator.phis[&phi_id].operands.len();

        assert_eq!(
            pred_count, operand_count,
            "Iteration {}: Block 4 has {} predecessors but phi has {} operands",
            iteration, pred_count, operand_count
        );

        // Check no undefined values
        for (i, operand) in translator.phis[&phi_id].operands.iter().enumerate() {
            assert!(
                !matches!(operand, Value::Undefined),
                "Iteration {}: Phi operand {} is undefined! All operands: {:?}",
                iteration, i, translator.phis[&phi_id].operands
            );
        }

        if !result.changed {
            break;
        }
    }

    // Final check: phi should only have values 0 or 1 (the boolean results), never undefined
    for operand in &translator.phis[&phi_id].operands {
        match operand {
            Value::Literal(0) | Value::Literal(1) => {} // OK
            Value::Undefined => panic!(
                "Bug: Phi has undefined operand! This causes wrong boolean returns. Operands: {:?}",
                translator.phis[&phi_id].operands
            ),
            _ => {} // Other values are OK in general
        }
    }
}
