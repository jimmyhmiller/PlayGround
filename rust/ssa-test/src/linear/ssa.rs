//! SSA IR for the linear language.
//!
//! This module defines the SSA representation output by translation.
//! All variables are `SsaVariable` (v0, v1, ...), not string names.

use std::fmt;

use crate::linear::input::BinOp;
use crate::traits::{InstructionFactory, SsaInstruction, SsaValue as SsaValueTrait};
use crate::types::{BlockId, PhiId, SsaVariable};
use crate::visualizer::{FormatInstruction, FormatValue};

/// A value in SSA form.
///
/// All variable references are SSA variables, not string names.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SsaValue {
    /// SSA variable reference (v0, v1, ...)
    Var(SsaVariable),
    /// Integer constant
    Const(i64),
    /// Phi function reference (used internally during construction)
    Phi(PhiId),
    /// Undefined value (used for uninitialized variables)
    Undefined,
}

impl SsaValue {
    pub fn var(v: SsaVariable) -> Self {
        SsaValue::Var(v)
    }

    pub fn constant(n: i64) -> Self {
        SsaValue::Const(n)
    }
}

impl fmt::Display for SsaValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SsaValue::Var(v) => write!(f, "{}", v),
            SsaValue::Const(n) => write!(f, "{}", n),
            SsaValue::Phi(id) => write!(f, "Φ{}", id.0),
            SsaValue::Undefined => write!(f, "undef"),
        }
    }
}

// Implement the SsaValue trait from traits.rs
impl SsaValueTrait for SsaValue {
    fn from_phi(phi_id: PhiId) -> Self {
        SsaValue::Phi(phi_id)
    }

    fn from_var(var: SsaVariable) -> Self {
        SsaValue::Var(var)
    }

    fn undefined() -> Self {
        SsaValue::Undefined
    }

    fn as_phi(&self) -> Option<PhiId> {
        match self {
            SsaValue::Phi(id) => Some(*id),
            _ => None,
        }
    }

    fn as_var(&self) -> Option<&SsaVariable> {
        match self {
            SsaValue::Var(v) => Some(v),
            _ => None,
        }
    }
}

impl FormatValue for SsaValue {
    fn format_for_display(&self) -> String {
        self.to_string()
    }
}

/// An instruction in SSA form.
///
/// All destinations are `SsaVariable`, ensuring single assignment property.
#[derive(Debug, Clone, PartialEq)]
pub enum SsaInstr {
    /// Simple assignment: dest := value
    Assign { dest: SsaVariable, value: SsaValue },

    /// Binary operation: dest := left op right
    BinOp {
        dest: SsaVariable,
        left: SsaValue,
        op: BinOp,
        right: SsaValue,
    },

    /// Phi assignment: dest := Φ(phi_id)
    /// The actual phi operands are stored in the translator's phi map.
    PhiAssign { dest: SsaVariable, phi_id: PhiId },

    /// Unconditional jump to a block
    Jump { target: BlockId },

    /// Conditional jump: if cond then true_target else false_target
    Branch {
        cond: SsaValue,
        then_block: BlockId,
        else_block: BlockId,
    },

    /// Return from function
    Return { value: SsaValue },
}

impl fmt::Display for SsaInstr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SsaInstr::Assign { dest, value } => write!(f, "{} := {}", dest, value),
            SsaInstr::BinOp { dest, left, op, right } => {
                write!(f, "{} := {} {} {}", dest, left, op, right)
            }
            SsaInstr::PhiAssign { dest, phi_id } => write!(f, "{} := Φ{}", dest, phi_id.0),
            SsaInstr::Jump { target } => write!(f, "jump B{}", target.0),
            SsaInstr::Branch { cond, then_block, else_block } => {
                write!(f, "if {} then B{} else B{}", cond, then_block.0, else_block.0)
            }
            SsaInstr::Return { value } => write!(f, "return {}", value),
        }
    }
}

// Implement SsaInstruction trait
impl SsaInstruction for SsaInstr {
    type Value = SsaValue;

    fn visit_values<F: FnMut(&Self::Value)>(&self, mut visitor: F) {
        match self {
            SsaInstr::Assign { value, .. } => visitor(value),
            SsaInstr::BinOp { left, right, .. } => {
                visitor(left);
                visitor(right);
            }
            SsaInstr::Branch { cond, .. } => visitor(cond),
            SsaInstr::Return { value } => visitor(value),
            SsaInstr::PhiAssign { .. } | SsaInstr::Jump { .. } => {}
        }
    }

    fn visit_values_mut<F: FnMut(&mut Self::Value)>(&mut self, mut visitor: F) {
        match self {
            SsaInstr::Assign { value, .. } => visitor(value),
            SsaInstr::BinOp { left, right, .. } => {
                visitor(left);
                visitor(right);
            }
            SsaInstr::Branch { cond, .. } => visitor(cond),
            SsaInstr::Return { value } => visitor(value),
            SsaInstr::PhiAssign { .. } | SsaInstr::Jump { .. } => {}
        }
    }

    fn destination(&self) -> Option<&SsaVariable> {
        match self {
            SsaInstr::Assign { dest, .. }
            | SsaInstr::BinOp { dest, .. }
            | SsaInstr::PhiAssign { dest, .. } => Some(dest),
            _ => None,
        }
    }

    fn is_phi_assignment(&self) -> bool {
        matches!(self, SsaInstr::PhiAssign { .. })
    }

    fn get_phi_assignment(&self) -> Option<PhiId> {
        match self {
            SsaInstr::PhiAssign { phi_id, .. } => Some(*phi_id),
            _ => None,
        }
    }
}

impl FormatInstruction for SsaInstr {
    fn format_for_display(&self) -> String {
        self.to_string()
    }
}

/// Factory for creating SSA instructions.
#[derive(Debug)]
pub struct SsaInstrFactory;

impl InstructionFactory for SsaInstrFactory {
    type Instr = SsaInstr;

    fn create_phi_assign(dest: SsaVariable, phi_id: PhiId) -> SsaInstr {
        SsaInstr::PhiAssign { dest, phi_id }
    }

    fn create_copy(dest: SsaVariable, value: SsaValue) -> SsaInstr {
        SsaInstr::Assign { dest, value }
    }
}
