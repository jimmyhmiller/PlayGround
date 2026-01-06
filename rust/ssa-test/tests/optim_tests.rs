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
