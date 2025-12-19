//! Example usage of the SSA library with the concrete implementation.

use ssa_test::concrete::ast::Ast;
use ssa_test::concrete::instruction::{Instruction, InstructionBuilder};
use ssa_test::concrete::value::Value;
use ssa_test::prelude::*;
use ssa_test::visualizer::SSAVisualizer;
use ssa_test::{ast, program};

/// Type alias for convenience
type ConcreteTranslator = SSATranslator<Value, Instruction, InstructionBuilder>;

/// Translate an AST node to SSA form
fn translate(translator: &mut ConcreteTranslator, ast: &Ast) -> Value {
    let current_block = translator.current_block;
    match ast {
        Ast::Literal(value) => {
            let temp_var = translator.get_temp_variable("lit");
            translator.emit(Instruction::Assign {
                dest: temp_var.clone(),
                value: Value::Literal(*value),
            });
            Value::Var(temp_var)
        }
        Ast::Variable(name) => translator.read_variable(name.clone(), current_block),
        Ast::Block(statements) => {
            let mut value = Value::Undefined;
            for statement in statements {
                value = translate(translator, statement);
            }
            value
        }
        Ast::Assignment { variable, value } => {
            let rhs = translate(translator, value);
            translator.write_variable(variable.clone(), current_block, rhs.clone());
            rhs
        }
        Ast::BinaryOp { left, op, right } => {
            let left_value = translate(translator, left);
            let right_value = translate(translator, right);
            let dest_var = translator.get_temp_variable(&format!("{:?}", op));
            let instruction = Instruction::BinaryOp {
                dest: dest_var.clone(),
                left: left_value,
                op: op.clone(),
                right: right_value,
            };
            translator.emit(instruction);
            Value::Var(dest_var)
        }
        Ast::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let condition_value = translate(translator, condition);

            let then_block = translator.create_block();
            let else_block = translator.create_block();

            translator.add_predecessor(then_block, current_block);
            translator.add_predecessor(else_block, current_block);
            translator.seal_block(current_block);
            translator.seal_block(then_block);
            translator.seal_block(else_block);

            translator.emit(Instruction::ConditionalJump {
                condition: condition_value,
                true_target: then_block,
                false_target: else_block,
            });

            let merge_block = translator.create_block();
            translator.current_block = then_block;
            for stmt in then_branch {
                translate(translator, stmt);
            }

            translator.add_predecessor(merge_block, translator.current_block);
            translator.emit(Instruction::Jump { target: merge_block });

            translator.current_block = else_block;
            if let Some(else_stmts) = else_branch {
                for stmt in else_stmts {
                    translate(translator, stmt);
                }
            }
            translator.add_predecessor(merge_block, translator.current_block);
            translator.emit(Instruction::Jump { target: merge_block });

            translator.seal_block(merge_block);
            translator.current_block = merge_block;
            Value::Undefined
        }
        Ast::Print(value) => {
            let print_value = translate(translator, value);
            let temp_var = translator.get_temp_variable("print");
            translator.emit(Instruction::Assign {
                dest: temp_var.clone(),
                value: print_value,
            });
            translator.emit(Instruction::Print {
                value: Value::Var(temp_var.clone()),
            });
            Value::Var(temp_var)
        }
        Ast::While { condition, body } => {
            let loop_start = translator.create_block();
            let loop_body = translator.create_block();
            let loop_end = translator.create_block();

            // Add predecessor from entry to loop start
            translator.add_predecessor(loop_start, current_block);

            // Jump to loop start
            translator.emit(Instruction::Jump { target: loop_start });

            // Seal the current block since all its successors are known
            translator.seal_block(current_block);

            // Switch to loop start block
            translator.current_block = loop_start;

            // Translate condition
            let condition_value = translate(translator, condition);

            // Add conditional jump
            translator.emit(Instruction::ConditionalJump {
                condition: condition_value,
                true_target: loop_body,
                false_target: loop_end,
            });

            // Add predecessors for loop body and loop end
            translator.add_predecessor(loop_body, loop_start);
            translator.add_predecessor(loop_end, loop_start);

            // Seal loop_end now - it won't get any more predecessors
            translator.seal_block(loop_end);

            // Process loop body
            translator.current_block = loop_body;
            for stmt in body {
                translate(translator, stmt);
            }

            // Jump back to loop start from body
            translator.emit(Instruction::Jump { target: loop_start });

            // Add the back-edge predecessor
            translator.add_predecessor(loop_start, translator.current_block);

            // Seal loop_body after processing it
            translator.seal_block(loop_body);

            // Seal loop_start last, after the back-edge is added
            translator.seal_block(loop_start);

            translator.current_block = loop_end;
            Value::Undefined
        }
        ast => {
            println!("Unsupported AST node: {:?}", ast);
            unimplemented!();
        }
    }
}

fn main() {
    let program_lisp = program! {
        (set x 10)
        (set y 5)
        (set sum (+ (var x) (var y)))
        (if (> (var sum) 10)
            (if (> (var y) 0)
                (set result 1)
                (set result 2))
            (set result 1))
        (print (var result))
        (while (> (var x) 0)
            (set x (- (var x) 1)))
        (print (var x))
    };

    let mut ssa_translator = ConcreteTranslator::new();
    translate(&mut ssa_translator, &program_lisp);

    println!("{:#?}", program_lisp);

    debug_ssa_state(&ssa_translator);

    let visualizer = SSAVisualizer::new(&ssa_translator);

    if let Err(e) = visualizer.render_to_file("ssa_graph.dot") {
        eprintln!("Failed to write dot file: {}", e);
    }

    if let Err(e) = visualizer.render_to_png("ssa_graph.png") {
        eprintln!("Failed to render SSA PNG: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Test 1: Simple straight-line code (no phis needed)
    #[test]
    fn test_straight_line_no_phis() {
        let program = program! {
            (set x 1)
            (set y 2)
            (set z (+ (var x) (var y)))
        };

        let mut translator = ConcreteTranslator::new();
        translate(&mut translator, &program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);

        // Should have no phis in straight-line code
        let non_empty_phis: Vec<_> = translator
            .phis
            .values()
            .filter(|p| !p.operands.is_empty())
            .collect();
        assert!(
            non_empty_phis.is_empty(),
            "Straight-line code should have no phis, found: {:?}",
            non_empty_phis
        );

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

        let mut translator = ConcreteTranslator::new();
        translate(&mut translator, &program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);

        // Should have exactly one non-trivial phi for x at the merge point
        let non_trivial_phis: Vec<_> = translator
            .phis
            .values()
            .filter(|p| p.operands.len() >= 2)
            .filter(|p| {
                let unique: HashSet<_> = p
                    .operands
                    .iter()
                    .filter(|op| !op.is_same_phi(p.id))
                    .collect();
                unique.len() > 1
            })
            .collect();

        assert!(
            !non_trivial_phis.is_empty(),
            "If-else with different assignments should have a phi"
        );
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

        let mut translator = ConcreteTranslator::new();
        translate(&mut translator, &program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);

        // The phi for x should be trivial (both branches assign 1) and removed
        // First check there are no trivial phi violations specifically
        let violations = validate_ssa(&translator);
        let trivial_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::TrivialPhi { .. }))
            .collect();

        assert!(
            trivial_violations.is_empty(),
            "Same value in both branches should result in trivial phi removal: {:?}",
            trivial_violations
        );

        // Also validate all other SSA properties
        assert_valid_ssa(&translator);
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

        let mut translator = ConcreteTranslator::new();
        translate(&mut translator, &program);
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

        let mut translator = ConcreteTranslator::new();
        translate(&mut translator, &program);
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

        let mut translator = ConcreteTranslator::new();
        translate(&mut translator, &program);
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

        let mut translator = ConcreteTranslator::new();
        translate(&mut translator, &program);
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

        let mut translator = ConcreteTranslator::new();
        translate(&mut translator, &program);
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

        let mut translator = ConcreteTranslator::new();
        translate(&mut translator, &program);
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

        let mut translator = ConcreteTranslator::new();
        translate(&mut translator, &program);
        translator.seal_block(translator.current_block);

        debug_ssa_state(&translator);
        assert_valid_ssa(&translator);
    }

    // ========================================================================
    // NEGATIVE TESTS - Verify validation catches each type of violation
    // ========================================================================

    /// Negative test: Detect unsealed block
    #[test]
    fn test_negative_unsealed_block() {
        let translator = ConcreteTranslator::new();
        // Don't seal the initial block
        let violations = validate_ssa(&translator);

        let unsealed_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::UnsealedBlock { .. }))
            .collect();

        assert!(
            !unsealed_violations.is_empty(),
            "Should detect unsealed block"
        );
    }

    /// Negative test: Detect trivial phi (all operands same)
    #[test]
    fn test_negative_trivial_phi() {
        use ssa_test::types::Phi;

        let mut translator = ConcreteTranslator::new();
        translator.seal_block(translator.current_block);

        // Manually create a trivial phi with all same operands
        let block_id = translator.current_block;
        let phi_id = PhiId(999);
        let mut phi = Phi::new(phi_id, block_id);
        phi.operands.push(Value::Literal(42));
        phi.operands.push(Value::Literal(42)); // Same value = trivial
        translator.phis.insert(phi_id, phi);

        // Add a fake predecessor so it looks like a join point
        translator.blocks[block_id.0].predecessors.push(BlockId(99));
        translator.blocks[block_id.0].predecessors.push(BlockId(98));

        let violations = validate_ssa(&translator);
        let trivial_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::TrivialPhi { .. }))
            .collect();

        assert!(
            !trivial_violations.is_empty(),
            "Should detect trivial phi with all same operands"
        );
    }

    /// Negative test: Detect operand count mismatch
    #[test]
    fn test_negative_operand_count_mismatch() {
        use ssa_test::types::Phi;

        let mut translator = ConcreteTranslator::new();
        translator.seal_block(translator.current_block);

        let block_id = translator.current_block;
        let phi_id = PhiId(999);
        let mut phi = Phi::new(phi_id, block_id);
        // Add 3 operands
        phi.operands.push(Value::Literal(1));
        phi.operands.push(Value::Literal(2));
        phi.operands.push(Value::Literal(3));
        translator.phis.insert(phi_id, phi);

        // But only 2 predecessors
        translator.blocks[block_id.0].predecessors.push(BlockId(99));
        translator.blocks[block_id.0].predecessors.push(BlockId(98));

        let violations = validate_ssa(&translator);
        let mismatch_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::OperandCountMismatch { .. }))
            .collect();

        assert!(
            !mismatch_violations.is_empty(),
            "Should detect operand count mismatch"
        );
    }

    /// Negative test: Detect dangling phi reference in phi operand
    #[test]
    fn test_negative_dangling_phi_reference() {
        use ssa_test::types::Phi;

        let mut translator = ConcreteTranslator::new();
        translator.seal_block(translator.current_block);

        let block_id = translator.current_block;
        let phi_id = PhiId(999);
        let mut phi = Phi::new(phi_id, block_id);
        // Reference a phi that doesn't exist
        phi.operands.push(Value::Phi(PhiId(12345)));
        phi.operands.push(Value::Literal(1));
        translator.phis.insert(phi_id, phi);

        translator.blocks[block_id.0].predecessors.push(BlockId(99));
        translator.blocks[block_id.0].predecessors.push(BlockId(98));

        let violations = validate_ssa(&translator);
        let dangling_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::DanglingPhiReference { .. }))
            .collect();

        assert!(
            !dangling_violations.is_empty(),
            "Should detect dangling phi reference"
        );
    }

    /// Negative test: Detect phi in non-join block
    #[test]
    fn test_negative_phi_in_non_join_block() {
        use ssa_test::types::Phi;

        let mut translator = ConcreteTranslator::new();
        translator.seal_block(translator.current_block);

        let block_id = translator.current_block;
        let phi_id = PhiId(999);
        let mut phi = Phi::new(phi_id, block_id);
        phi.operands.push(Value::Literal(1));
        translator.phis.insert(phi_id, phi);

        // Block has only 1 predecessor (or 0) - not a join point
        translator.blocks[block_id.0].predecessors.push(BlockId(99));

        let violations = validate_ssa(&translator);
        let non_join_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::PhiInNonJoinBlock { .. }))
            .collect();

        assert!(
            !non_join_violations.is_empty(),
            "Should detect phi in non-join block"
        );
    }

    /// Negative test: Detect empty phi
    #[test]
    fn test_negative_empty_phi() {
        use ssa_test::types::Phi;

        let mut translator = ConcreteTranslator::new();
        translator.seal_block(translator.current_block);

        let block_id = translator.current_block;
        let phi_id = PhiId(999);
        let phi = Phi::new(phi_id, block_id); // No operands
        translator.phis.insert(phi_id, phi);

        let violations = validate_ssa(&translator);
        let empty_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::EmptyPhi { .. }))
            .collect();

        assert!(
            !empty_violations.is_empty(),
            "Should detect empty phi"
        );
    }

    /// Negative test: Detect self-referential phi with no other operands
    #[test]
    fn test_negative_only_self_reference_phi() {
        use ssa_test::types::Phi;

        let mut translator = ConcreteTranslator::new();
        translator.seal_block(translator.current_block);

        let block_id = translator.current_block;
        let phi_id = PhiId(999);
        let mut phi = Phi::new(phi_id, block_id);
        // Only references itself
        phi.operands.push(Value::Phi(phi_id));
        phi.operands.push(Value::Phi(phi_id));
        translator.phis.insert(phi_id, phi);

        translator.blocks[block_id.0].predecessors.push(BlockId(99));
        translator.blocks[block_id.0].predecessors.push(BlockId(98));

        let violations = validate_ssa(&translator);
        let self_ref_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::OnlySelfReferencePhi { .. }))
            .collect();

        assert!(
            !self_ref_violations.is_empty(),
            "Should detect phi with only self-references"
        );
    }

    /// Negative test: Detect dangling phi in instruction
    #[test]
    fn test_negative_dangling_phi_in_instruction() {
        let mut translator = ConcreteTranslator::new();
        translator.seal_block(translator.current_block);

        // Add an instruction that references a non-existent phi
        translator.emit(Instruction::Assign {
            dest: SsaVariable::new("v0"),
            value: Value::Phi(PhiId(12345)), // Doesn't exist
        });

        let violations = validate_ssa(&translator);
        let dangling_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::DanglingPhiInInstruction { .. }))
            .collect();

        assert!(
            !dangling_violations.is_empty(),
            "Should detect dangling phi reference in instruction"
        );
    }

    /// Negative test: Detect phi used directly as operand (not in assign)
    #[test]
    fn test_negative_phi_used_directly() {
        use ssa_test::types::Phi;
        use ssa_test::concrete::ast::BinaryOperator;

        let mut translator = ConcreteTranslator::new();
        translator.seal_block(translator.current_block);

        // Create a valid phi first
        let block_id = translator.current_block;
        let phi_id = PhiId(999);
        let mut phi = Phi::new(phi_id, block_id);
        phi.operands.push(Value::Literal(1));
        phi.operands.push(Value::Literal(2));
        translator.phis.insert(phi_id, phi);

        translator.blocks[block_id.0].predecessors.push(BlockId(99));
        translator.blocks[block_id.0].predecessors.push(BlockId(98));

        // Use phi directly in a BinaryOp (not allowed - should be assigned first)
        translator.emit(Instruction::BinaryOp {
            dest: SsaVariable::new("result"),
            left: Value::Phi(phi_id), // Direct phi use!
            op: BinaryOperator::Add,
            right: Value::Literal(1),
        });

        let violations = validate_ssa(&translator);
        let direct_use_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::PhiUsedDirectlyAsOperand { .. }))
            .collect();

        assert!(
            !direct_use_violations.is_empty(),
            "Should detect phi used directly as operand"
        );
    }

    /// Negative test: Detect phi assignment not at block start
    #[test]
    fn test_negative_phi_not_at_block_start() {
        use ssa_test::types::Phi;

        let mut translator = ConcreteTranslator::new();
        translator.seal_block(translator.current_block);

        // Create a valid phi
        let block_id = translator.current_block;
        let phi_id = PhiId(999);
        let mut phi = Phi::new(phi_id, block_id);
        phi.operands.push(Value::Literal(1));
        phi.operands.push(Value::Literal(2));
        translator.phis.insert(phi_id, phi);

        translator.blocks[block_id.0].predecessors.push(BlockId(99));
        translator.blocks[block_id.0].predecessors.push(BlockId(98));

        // Add a non-phi instruction first
        translator.emit(Instruction::Assign {
            dest: SsaVariable::new("x"),
            value: Value::Literal(42),
        });

        // Then add phi assignment (wrong order!)
        translator.emit(Instruction::Assign {
            dest: SsaVariable::new("phi_result"),
            value: Value::Phi(phi_id),
        });

        let violations = validate_ssa(&translator);
        let order_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::PhiNotAtBlockStart { .. }))
            .collect();

        assert!(
            !order_violations.is_empty(),
            "Should detect phi assignment not at block start"
        );
    }

    /// Negative test: Detect multiple definitions of same variable
    #[test]
    fn test_negative_multiple_definitions() {
        let mut translator = ConcreteTranslator::new();
        translator.seal_block(translator.current_block);

        // Define the same variable twice
        translator.emit(Instruction::Assign {
            dest: SsaVariable::new("v0"),
            value: Value::Literal(1),
        });
        translator.emit(Instruction::Assign {
            dest: SsaVariable::new("v0"), // Same variable!
            value: Value::Literal(2),
        });

        let violations = validate_ssa(&translator);
        let multi_def_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::MultipleDefinitions { .. }))
            .collect();

        assert!(
            !multi_def_violations.is_empty(),
            "Should detect multiple definitions of same variable"
        );
    }

    /// Negative test: Detect undefined variable use
    #[test]
    fn test_negative_undefined_variable_use() {
        let mut translator = ConcreteTranslator::new();
        translator.seal_block(translator.current_block);

        // Use a variable that was never defined
        translator.emit(Instruction::Assign {
            dest: SsaVariable::new("result"),
            value: Value::Var(SsaVariable::new("undefined_var")),
        });

        let violations = validate_ssa(&translator);
        let undef_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::UndefinedVariableUse { .. }))
            .collect();

        assert!(
            !undef_violations.is_empty(),
            "Should detect undefined variable use"
        );
    }

    /// Negative test: Detect dominance violation (use before def in same block)
    #[test]
    fn test_negative_dominance_violation_same_block() {
        let mut translator = ConcreteTranslator::new();
        translator.seal_block(translator.current_block);

        // Use v0 before defining it
        translator.emit(Instruction::Assign {
            dest: SsaVariable::new("result"),
            value: Value::Var(SsaVariable::new("v0")), // Use before def
        });
        translator.emit(Instruction::Assign {
            dest: SsaVariable::new("v0"), // Defined after use
            value: Value::Literal(42),
        });

        let violations = validate_ssa(&translator);
        let dom_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::DominanceViolation { .. }))
            .collect();

        assert!(
            !dom_violations.is_empty(),
            "Should detect dominance violation (use before def)"
        );
    }

    /// Negative test: Detect dead phi (phi with no non-phi users)
    #[test]
    fn test_negative_dead_phi() {
        use ssa_test::types::Phi;

        let mut translator = ConcreteTranslator::new();
        translator.seal_block(translator.current_block);

        let block_id = translator.current_block;
        let phi_id = PhiId(999);
        let mut phi = Phi::new(phi_id, block_id);
        phi.operands.push(Value::Literal(1));
        phi.operands.push(Value::Literal(2));
        translator.phis.insert(phi_id, phi);

        // Add predecessors so it's a valid join point
        translator.blocks[block_id.0].predecessors.push(BlockId(99));
        translator.blocks[block_id.0].predecessors.push(BlockId(98));

        // Assign phi to a variable but never use that variable
        translator.emit(Instruction::Assign {
            dest: SsaVariable::new("dead_phi_result"),
            value: Value::Phi(phi_id),
        });

        // Add some other instruction that doesn't use the phi result
        translator.emit(Instruction::Assign {
            dest: SsaVariable::new("other"),
            value: Value::Literal(42),
        });

        let violations = validate_ssa(&translator);
        let dead_phi_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::DeadPhi { .. }))
            .collect();

        assert!(
            !dead_phi_violations.is_empty(),
            "Should detect dead phi (phi with no non-phi users)"
        );
    }

    /// Negative test: Detect phi operand dominance violation
    /// (phi operand's definition doesn't dominate the corresponding predecessor)
    #[test]
    fn test_negative_phi_operand_dominance_violation() {
        use ssa_test::types::Phi;

        let mut translator = ConcreteTranslator::new();

        // Create a control flow structure:
        // Block 0 -> Block 1 (then branch)
        // Block 0 -> Block 2 (else branch)
        // Block 1 -> Block 3 (merge)
        // Block 2 -> Block 3 (merge)
        //
        // If we have a phi in Block 3 that uses a variable defined in Block 1
        // as the operand for the Block 2 predecessor, that's a dominance violation
        // because Block 1 doesn't dominate Block 2.

        let block0 = translator.current_block; // BlockId(0)
        let block1 = translator.create_block(); // BlockId(1)
        let block2 = translator.create_block(); // BlockId(2)
        let block3 = translator.create_block(); // BlockId(3) - merge point

        // Set up predecessors
        translator.add_predecessor(block1, block0);
        translator.add_predecessor(block2, block0);
        translator.add_predecessor(block3, block1);
        translator.add_predecessor(block3, block2);

        // Seal all blocks
        translator.seal_block(block0);
        translator.seal_block(block1);
        translator.seal_block(block2);
        translator.seal_block(block3);

        // Define v_in_block1 only in Block 1
        translator.current_block = block1;
        translator.emit(Instruction::Assign {
            dest: SsaVariable::new("v_in_block1"),
            value: Value::Literal(42),
        });

        // Define v_in_block2 only in Block 2
        translator.current_block = block2;
        translator.emit(Instruction::Assign {
            dest: SsaVariable::new("v_in_block2"),
            value: Value::Literal(100),
        });

        // Create a phi in Block 3 with WRONG operand order:
        // - operand[0] corresponds to predecessor block1
        // - operand[1] corresponds to predecessor block2
        // But we'll put v_in_block2 (defined in block2) as operand[0] (for block1)
        // This is a dominance violation: block2 doesn't dominate block1
        let phi_id = PhiId(999);
        let mut phi = Phi::new(phi_id, block3);
        phi.operands.push(Value::Var(SsaVariable::new("v_in_block2"))); // Wrong! block2 doesn't dominate block1
        phi.operands.push(Value::Var(SsaVariable::new("v_in_block1"))); // Wrong! block1 doesn't dominate block2
        translator.phis.insert(phi_id, phi);

        // Assign phi to a variable and use it
        translator.current_block = block3;
        translator.emit(Instruction::Assign {
            dest: SsaVariable::new("phi_result"),
            value: Value::Phi(phi_id),
        });
        translator.emit(Instruction::Print {
            value: Value::Var(SsaVariable::new("phi_result")),
        });

        let violations = validate_ssa(&translator);
        let phi_dom_violations: Vec<_> = violations
            .iter()
            .filter(|v| matches!(v, SSAViolation::PhiOperandDominanceViolation { .. }))
            .collect();

        assert!(
            !phi_dom_violations.is_empty(),
            "Should detect phi operand dominance violation. Violations: {:?}",
            violations
        );
    }
}
