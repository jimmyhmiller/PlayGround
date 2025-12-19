/// SSA Property Tests
///
/// Based on "Simple and Efficient Construction of Static Single Assignment Form"
/// by Braun et al. (CC 2013)
///
/// Key properties that valid SSA must satisfy:
/// 1. No trivial phis - a phi is trivial if all operands are the same (ignoring self-references)
/// 2. Phi operand count matches predecessor count
/// 3. All phi references are valid (no dangling references)
/// 4. Phis only exist at join points (blocks with multiple predecessors)
/// 5. All blocks are sealed after construction

use crate::instruction::{BlockId, PhiId, Value};
use crate::SSATranslator;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq)]
pub enum SSAViolation {
    /// Phi has all same operands (should have been removed as trivial)
    TrivialPhi {
        phi_id: PhiId,
        block_id: BlockId,
        unique_operand: Value,
    },
    /// Phi operand count doesn't match block's predecessor count
    OperandCountMismatch {
        phi_id: PhiId,
        block_id: BlockId,
        operand_count: usize,
        predecessor_count: usize,
    },
    /// Phi references another phi that doesn't exist
    DanglingPhiReference {
        phi_id: PhiId,
        referenced_phi: PhiId,
    },
    /// Phi exists in a block with 0 or 1 predecessors (not a join point)
    PhiInNonJoinBlock {
        phi_id: PhiId,
        block_id: BlockId,
        predecessor_count: usize,
    },
    /// Block is not sealed after construction
    UnsealedBlock {
        block_id: BlockId,
    },
    /// Phi has no operands (incomplete)
    EmptyPhi {
        phi_id: PhiId,
        block_id: BlockId,
    },
    /// Self-referential phi with no other operands (undefined behavior)
    OnlySelfReferencePhi {
        phi_id: PhiId,
        block_id: BlockId,
    },
    /// Phi operand references undefined value where it shouldn't
    UndefinedOperand {
        phi_id: PhiId,
        block_id: BlockId,
    },
}

impl std::fmt::Display for SSAViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SSAViolation::TrivialPhi { phi_id, block_id, unique_operand } => {
                write!(f, "Trivial phi {:?} in block {:?}: all operands are {:?}",
                    phi_id, block_id, unique_operand)
            }
            SSAViolation::OperandCountMismatch { phi_id, block_id, operand_count, predecessor_count } => {
                write!(f, "Phi {:?} in block {:?} has {} operands but block has {} predecessors",
                    phi_id, block_id, operand_count, predecessor_count)
            }
            SSAViolation::DanglingPhiReference { phi_id, referenced_phi } => {
                write!(f, "Phi {:?} references non-existent phi {:?}", phi_id, referenced_phi)
            }
            SSAViolation::PhiInNonJoinBlock { phi_id, block_id, predecessor_count } => {
                write!(f, "Phi {:?} exists in block {:?} which has only {} predecessors (not a join point)",
                    phi_id, block_id, predecessor_count)
            }
            SSAViolation::UnsealedBlock { block_id } => {
                write!(f, "Block {:?} is not sealed after construction", block_id)
            }
            SSAViolation::EmptyPhi { phi_id, block_id } => {
                write!(f, "Phi {:?} in block {:?} has no operands", phi_id, block_id)
            }
            SSAViolation::OnlySelfReferencePhi { phi_id, block_id } => {
                write!(f, "Phi {:?} in block {:?} only references itself", phi_id, block_id)
            }
            SSAViolation::UndefinedOperand { phi_id, block_id } => {
                write!(f, "Phi {:?} in block {:?} has undefined operand (may indicate bug)",
                    phi_id, block_id)
            }
        }
    }
}

/// Validates all SSA properties and returns a list of violations
pub fn validate_ssa(translator: &SSATranslator) -> Vec<SSAViolation> {
    let mut violations = Vec::new();

    // Collect all valid phi IDs
    let valid_phi_ids: HashSet<PhiId> = translator.phis.keys().copied().collect();

    // Check each phi
    for (phi_id, phi) in &translator.phis {
        let block = &translator.blocks[phi.block_id.0];

        // Property 1: No trivial phis
        if let Some(violation) = check_trivial_phi(*phi_id, phi, &valid_phi_ids) {
            violations.push(violation);
        }

        // Property 2: Operand count matches predecessor count
        // (Only check for sealed blocks with operands)
        if !phi.operands.is_empty() && block.predecessors.len() > 1 {
            if phi.operands.len() != block.predecessors.len() {
                violations.push(SSAViolation::OperandCountMismatch {
                    phi_id: *phi_id,
                    block_id: phi.block_id,
                    operand_count: phi.operands.len(),
                    predecessor_count: block.predecessors.len(),
                });
            }
        }

        // Property 3: All phi references are valid
        for operand in &phi.operands {
            if let Value::Phi(ref_phi_id) = operand {
                if !valid_phi_ids.contains(ref_phi_id) {
                    violations.push(SSAViolation::DanglingPhiReference {
                        phi_id: *phi_id,
                        referenced_phi: *ref_phi_id,
                    });
                }
            }
        }

        // Property 4: Phis only at join points (2+ predecessors)
        // Exception: entry block can have phis for parameters, but this simple language doesn't have params
        if block.predecessors.len() < 2 && !phi.operands.is_empty() {
            violations.push(SSAViolation::PhiInNonJoinBlock {
                phi_id: *phi_id,
                block_id: phi.block_id,
                predecessor_count: block.predecessors.len(),
            });
        }

        // Check for empty phis (incomplete construction)
        if phi.operands.is_empty() && translator.sealed_blocks.contains(&phi.block_id) {
            violations.push(SSAViolation::EmptyPhi {
                phi_id: *phi_id,
                block_id: phi.block_id,
            });
        }
    }

    // Property 5: All blocks should be sealed
    for block in &translator.blocks {
        if !block.sealed {
            violations.push(SSAViolation::UnsealedBlock {
                block_id: block.id,
            });
        }
    }

    violations
}

/// Check if a phi is trivial (all operands are the same, excluding self-references)
fn check_trivial_phi(
    phi_id: PhiId,
    phi: &crate::instruction::Phi,
    _valid_phi_ids: &HashSet<PhiId>,
) -> Option<SSAViolation> {
    if phi.operands.is_empty() {
        return None; // Empty phis are caught elsewhere
    }

    let mut unique_value: Option<&Value> = None;
    let mut has_non_self_operand = false;

    for operand in &phi.operands {
        // Skip self-references
        if let Value::Phi(op_phi_id) = operand {
            if *op_phi_id == phi_id {
                continue;
            }
        }

        has_non_self_operand = true;

        match unique_value {
            None => unique_value = Some(operand),
            Some(existing) => {
                if existing != operand {
                    // Found two different operands, not trivial
                    return None;
                }
            }
        }
    }

    // If all operands were self-references
    if !has_non_self_operand {
        return Some(SSAViolation::OnlySelfReferencePhi {
            phi_id,
            block_id: phi.block_id,
        });
    }

    // If we found a unique value (all non-self operands are the same), it's trivial
    if let Some(value) = unique_value {
        return Some(SSAViolation::TrivialPhi {
            phi_id,
            block_id: phi.block_id,
            unique_operand: value.clone(),
        });
    }

    None
}

/// Helper to assert no violations occurred
pub fn assert_valid_ssa(translator: &SSATranslator) {
    let violations = validate_ssa(translator);
    if !violations.is_empty() {
        let mut msg = String::from("SSA validation failed:\n");
        for v in &violations {
            msg.push_str(&format!("  - {}\n", v));
        }
        panic!("{}", msg);
    }
}

/// Print a detailed report of SSA state (useful for debugging)
pub fn debug_ssa_state(translator: &SSATranslator) {
    println!("=== SSA State Debug ===");
    println!("\nBlocks ({}):", translator.blocks.len());
    for block in &translator.blocks {
        println!("  Block {:?} (sealed: {}, predecessors: {:?})",
            block.id, block.sealed, block.predecessors);
        for (i, instr) in block.instructions.iter().enumerate() {
            println!("    [{}] {:?}", i, instr);
        }
    }

    println!("\nPhis ({}):", translator.phis.len());
    for (phi_id, phi) in &translator.phis {
        println!("  {:?} in block {:?}: operands={:?}", phi_id, phi.block_id, phi.operands);
    }

    println!("\nIncomplete Phis:");
    for (block_id, phis) in &translator.incomplete_phis {
        println!("  Block {:?}: {:?}", block_id, phis);
    }

    println!("\nSealed Blocks: {:?}", translator.sealed_blocks);

    let violations = validate_ssa(translator);
    println!("\nViolations ({}):", violations.len());
    for v in &violations {
        println!("  - {}", v);
    }
    println!("=== End Debug ===\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{program, ast};

    /// Test 1: Simple straight-line code (no phis needed)
    #[test]
    fn test_straight_line_no_phis() {
        let program = program! {
            (set x 1)
            (set y 2)
            (set z (+ (var x) (var y)))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);

        // Should have no phis in straight-line code
        let non_empty_phis: Vec<_> = translator.phis.values()
            .filter(|p| !p.operands.is_empty())
            .collect();
        assert!(non_empty_phis.is_empty(),
            "Straight-line code should have no phis, found: {:?}", non_empty_phis);

        assert_valid_ssa(&translator);
    }

    /// Test 2: If-then-else with different assignments (phi needed)
    #[test]
    fn test_if_else_needs_phi() {
        let program = program! {
            (set x 0)
            (if (> 5 3)
                (set x 1)
                (set x 2))
            (print (var x))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);

        // Should have exactly one non-trivial phi for x at the merge point
        let non_trivial_phis: Vec<_> = translator.phis.values()
            .filter(|p| p.operands.len() >= 2)
            .filter(|p| {
                let unique: HashSet<_> = p.operands.iter()
                    .filter(|op| !matches!(op, Value::Phi(id) if *id == p.id))
                    .collect();
                unique.len() > 1
            })
            .collect();

        assert!(!non_trivial_phis.is_empty(),
            "If-else with different assignments should have a phi");
    }

    /// Test 3: If-then-else with same assignment (phi should be trivial and removed)
    #[test]
    fn test_if_else_same_value_no_phi() {
        let program = program! {
            (set x 0)
            (if (> 5 3)
                (set x 1)
                (set x 1))
            (print (var x))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);

        // The phi for x should be trivial (both branches assign 1) and removed
        let violations = validate_ssa(&translator);
        let trivial_violations: Vec<_> = violations.iter()
            .filter(|v| matches!(v, SSAViolation::TrivialPhi { .. }))
            .collect();

        assert!(trivial_violations.is_empty(),
            "Same value in both branches should result in trivial phi removal: {:?}",
            trivial_violations);
    }

    /// Test 4: While loop (phi needed at loop header)
    #[test]
    fn test_while_loop_needs_phi() {
        let program = program! {
            (set x 10)
            (while (> (var x) 0)
                (set x (- (var x) 1)))
            (print (var x))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }

    /// Test 5: Variable used only in condition, not modified (no phi)
    #[test]
    fn test_unmodified_variable_no_phi() {
        let program = program! {
            (set x 5)
            (set y 0)
            (if (> (var x) 3)
                (set y 1)
                (set y 2))
            (print (var x))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }

    /// Test 6: Nested if statements
    #[test]
    fn test_nested_if() {
        let program = program! {
            (set x 0)
            (if (> 5 3)
                (if (> 2 1)
                    (set x 1)
                    (set x 2))
                (set x 3))
            (print (var x))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }

    /// Test 7: Multiple variables with different phi requirements
    #[test]
    fn test_multiple_variables() {
        let program = program! {
            (set x 1)
            (set y 2)
            (if (> (var x) 0)
                (begin
                    (set x 10)
                    (set y 20))
                (begin
                    (set x 100)))
            (print (var x))
            (print (var y))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }

    /// Test 8: While loop with multiple variables
    #[test]
    fn test_while_multiple_variables() {
        let program = program! {
            (set i 0)
            (set sum 0)
            (while (< (var i) 10)
                (begin
                    (set sum (+ (var sum) (var i)))
                    (set i (+ (var i) 1))))
            (print (var sum))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }

    /// Test 9: If without else (one branch doesn't modify)
    #[test]
    fn test_if_without_else() {
        let program = program! {
            (set x 1)
            (if (> 5 3)
                (set x 2))
            (print (var x))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }

    /// Test 10: Complex control flow - if inside while
    #[test]
    fn test_if_inside_while() {
        let program = program! {
            (set x 10)
            (set y 0)
            (while (> (var x) 0)
                (begin
                    (if (> (var x) 5)
                        (set y (+ (var y) 2))
                        (set y (+ (var y) 1)))
                    (set x (- (var x) 1))))
            (print (var y))
        };

        let mut translator = SSATranslator::new();
        translator.translate(&program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }
}
