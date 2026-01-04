//! Input IR for the linear language.
//!
//! This module defines the input representation that users write.
//! Variables are referenced by string names, not SSA variables.

use std::fmt;

use crate::cfg::{CfgInstruction, ControlFlow};

/// A label for jump targets.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label(pub String);

impl Label {
    pub fn new(name: &str) -> Self {
        Label(name.to_string())
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
        };
        write!(f, "{}", s)
    }
}

/// A value in the input IR.
///
/// Variables are referenced by name (String), not SSA variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputValue {
    /// Variable reference by name
    Var(String),
    /// Integer constant
    Const(i64),
}

impl InputValue {
    pub fn var(name: &str) -> Self {
        InputValue::Var(name.to_string())
    }

    pub fn constant(n: i64) -> Self {
        InputValue::Const(n)
    }
}

impl fmt::Display for InputValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputValue::Var(name) => write!(f, "{}", name),
            InputValue::Const(n) => write!(f, "{}", n),
        }
    }
}

/// An instruction in the input IR.
///
/// This is the user-facing representation. Variables are referenced by
/// string names, not SSA variables.
#[derive(Debug, Clone, PartialEq)]
pub enum InputInstr {
    /// Simple assignment: dest := value
    Assign { dest: String, value: InputValue },

    /// Binary operation: dest := left op right
    BinOp {
        dest: String,
        left: InputValue,
        op: BinOp,
        right: InputValue,
    },

    /// Label definition (marks a jump target)
    Label(Label),

    /// Unconditional jump
    Jump(Label),

    /// Conditional jump: if cond is truthy, jump to target; else fall through
    JumpIf { cond: InputValue, target: Label },

    /// Return from function
    Return(InputValue),
}

impl fmt::Display for InputInstr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputInstr::Assign { dest, value } => write!(f, "{} := {}", dest, value),
            InputInstr::BinOp { dest, left, op, right } => {
                write!(f, "{} := {} {} {}", dest, left, op, right)
            }
            InputInstr::Label(label) => write!(f, "{}:", label),
            InputInstr::Jump(label) => write!(f, "jump {}", label),
            InputInstr::JumpIf { cond, target } => write!(f, "if {} jump {}", cond, target),
            InputInstr::Return(value) => write!(f, "return {}", value),
        }
    }
}

// Implement CfgInstruction to enable CFG construction from input IR
impl CfgInstruction for InputInstr {
    type Label = Label;

    fn as_label(&self) -> Option<&Label> {
        match self {
            InputInstr::Label(l) => Some(l),
            _ => None,
        }
    }

    fn control_flow(&self) -> ControlFlow<Label> {
        match self {
            InputInstr::Jump(target) => ControlFlow::Jump(target.clone()),
            InputInstr::JumpIf { target, .. } => ControlFlow::ConditionalJump(target.clone()),
            InputInstr::Return(_) => ControlFlow::Terminate,
            _ => ControlFlow::FallThrough,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::CfgBuilder;

    #[test]
    fn test_simple_cfg() {
        let program = vec![
            InputInstr::Assign {
                dest: "x".into(),
                value: InputValue::Const(1),
            },
            InputInstr::Return(InputValue::Var("x".into())),
        ];

        let cfg = CfgBuilder::build(program);
        assert_eq!(cfg.blocks.len(), 1);
        assert_eq!(cfg.blocks[0].instructions.len(), 2);
    }

    #[test]
    fn test_branch_cfg() {
        let program = vec![
            InputInstr::Assign {
                dest: "x".into(),
                value: InputValue::Const(1),
            },
            InputInstr::JumpIf {
                cond: InputValue::Var("x".into()),
                target: Label::new("then"),
            },
            InputInstr::Assign {
                dest: "y".into(),
                value: InputValue::Const(2),
            },
            InputInstr::Jump(Label::new("end")),
            InputInstr::Label(Label::new("then")),
            InputInstr::Assign {
                dest: "y".into(),
                value: InputValue::Const(3),
            },
            InputInstr::Label(Label::new("end")),
            InputInstr::Return(InputValue::Var("y".into())),
        ];

        let cfg = CfgBuilder::build(program);
        assert_eq!(cfg.blocks.len(), 4);
    }
}
