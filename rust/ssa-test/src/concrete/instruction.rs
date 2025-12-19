//! Concrete instruction type implementing SsaInstruction and InstructionFactory.

use crate::concrete::ast::{BinaryOperator, UnaryOperator};
use crate::concrete::value::Value;
use crate::traits::{InstructionFactory, SsaInstruction};
use crate::types::{BlockId, PhiId, SsaVariable};
use crate::visualizer::{FormatInstruction, FormatValue};

/// Concrete instruction type for the example IR.
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    /// Assignment: dest := value
    Assign {
        dest: SsaVariable,
        value: Value,
    },

    /// Binary operation: dest := left op right
    BinaryOp {
        dest: SsaVariable,
        left: Value,
        op: BinaryOperator,
        right: Value,
    },

    /// Unary operation: dest := op operand
    UnaryOp {
        dest: SsaVariable,
        op: UnaryOperator,
        operand: Value,
    },

    /// Unconditional jump
    Jump {
        target: BlockId,
    },

    /// Conditional jump
    ConditionalJump {
        condition: Value,
        true_target: BlockId,
        false_target: BlockId,
    },

    /// Print instruction (for debugging)
    Print {
        value: Value,
    },
}

impl SsaInstruction for Instruction {
    type Value = Value;

    fn visit_values<F: FnMut(&Self::Value)>(&self, mut visitor: F) {
        match self {
            Instruction::Assign { value, .. } => visitor(value),
            Instruction::BinaryOp { left, right, .. } => {
                visitor(left);
                visitor(right);
            }
            Instruction::UnaryOp { operand, .. } => visitor(operand),
            Instruction::ConditionalJump { condition, .. } => visitor(condition),
            Instruction::Print { value } => visitor(value),
            Instruction::Jump { .. } => {}
        }
    }

    fn visit_values_mut<F: FnMut(&mut Self::Value)>(&mut self, mut visitor: F) {
        match self {
            Instruction::Assign { value, .. } => visitor(value),
            Instruction::BinaryOp { left, right, .. } => {
                visitor(left);
                visitor(right);
            }
            Instruction::UnaryOp { operand, .. } => visitor(operand),
            Instruction::ConditionalJump { condition, .. } => visitor(condition),
            Instruction::Print { value } => visitor(value),
            Instruction::Jump { .. } => {}
        }
    }

    fn destination(&self) -> Option<&SsaVariable> {
        match self {
            Instruction::Assign { dest, .. }
            | Instruction::BinaryOp { dest, .. }
            | Instruction::UnaryOp { dest, .. } => Some(dest),
            _ => None,
        }
    }

    fn is_phi_assignment(&self) -> bool {
        matches!(self, Instruction::Assign { value: Value::Phi(_), .. })
    }

    fn get_phi_assignment(&self) -> Option<PhiId> {
        match self {
            Instruction::Assign { value: Value::Phi(phi_id), .. } => Some(*phi_id),
            _ => None,
        }
    }
}

/// Factory for creating instructions.
///
/// This struct exists solely to implement InstructionFactory.
pub struct InstructionBuilder;

impl InstructionFactory for InstructionBuilder {
    type Instr = Instruction;

    fn create_phi_assign(dest: SsaVariable, phi_id: PhiId) -> Instruction {
        Instruction::Assign {
            dest,
            value: Value::Phi(phi_id),
        }
    }

    fn create_copy(dest: SsaVariable, value: Value) -> Instruction {
        Instruction::Assign { dest, value }
    }
}

impl FormatInstruction for Instruction {
    fn format_for_display(&self) -> String {
        match self {
            Instruction::Assign { dest, value } => {
                format!("{} := {}", dest.name(), value.format_for_display())
            }
            Instruction::BinaryOp { dest, left, op, right } => {
                let op_str = match op {
                    BinaryOperator::Add => "+",
                    BinaryOperator::Subtract => "-",
                    BinaryOperator::Multiply => "*",
                    BinaryOperator::Divide => "/",
                    BinaryOperator::Equal => "==",
                    BinaryOperator::NotEqual => "!=",
                    BinaryOperator::LessThan => "<",
                    BinaryOperator::LessThanOrEqual => "<=",
                    BinaryOperator::GreaterThan => ">",
                    BinaryOperator::GreaterThanOrEqual => ">=",
                };
                format!(
                    "{} := {} {} {}",
                    dest.name(),
                    left.format_for_display(),
                    op_str,
                    right.format_for_display()
                )
            }
            Instruction::UnaryOp { dest, op, operand } => {
                let op_str = match op {
                    UnaryOperator::Negate => "-",
                    UnaryOperator::Not => "!",
                };
                format!(
                    "{} := {} {}",
                    dest.name(),
                    op_str,
                    operand.format_for_display()
                )
            }
            Instruction::Jump { target } => {
                format!("jump block_{}", target.0)
            }
            Instruction::ConditionalJump { condition, true_target, false_target } => {
                format!(
                    "if {} then block_{} else block_{}",
                    condition.format_for_display(),
                    true_target.0,
                    false_target.0
                )
            }
            Instruction::Print { value } => {
                format!("print {}", value.format_for_display())
            }
        }
    }
}
