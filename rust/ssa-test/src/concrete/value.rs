//! Concrete value type implementing SsaValue.

use crate::traits::SsaValue;
use crate::types::{PhiId, SsaVariable};
use crate::visualizer::FormatValue;

/// Concrete value type for the example IR.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Value {
    /// Integer literal
    Literal(i32),
    /// Variable reference
    Var(SsaVariable),
    /// Phi function reference
    Phi(PhiId),
    /// Undefined value
    Undefined,
}

impl Value {
    pub fn literal(val: i32) -> Self {
        Value::Literal(val)
    }

    pub fn var(name: &str) -> Self {
        Value::Var(SsaVariable::new(name))
    }

    pub fn new_phi(phi_id: PhiId) -> Self {
        Value::Phi(phi_id)
    }

    pub fn get_phi_id(&self) -> PhiId {
        match self {
            Value::Phi(phi_id) => *phi_id,
            _ => panic!("Value is not a Phi"),
        }
    }
}

impl SsaValue for Value {
    fn from_phi(phi_id: PhiId) -> Self {
        Value::Phi(phi_id)
    }

    fn from_var(var: SsaVariable) -> Self {
        Value::Var(var)
    }

    fn undefined() -> Self {
        Value::Undefined
    }

    fn as_phi(&self) -> Option<PhiId> {
        match self {
            Value::Phi(id) => Some(*id),
            _ => None,
        }
    }

    fn as_var(&self) -> Option<&SsaVariable> {
        match self {
            Value::Var(v) => Some(v),
            _ => None,
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Literal(n) => write!(f, "{}", n),
            Value::Var(v) => write!(f, "{}", v),
            Value::Phi(id) => write!(f, "{}", id),
            Value::Undefined => write!(f, "undef"),
        }
    }
}

impl FormatValue for Value {
    fn format_for_display(&self) -> String {
        match self {
            Value::Literal(n) => n.to_string(),
            Value::Var(v) => v.name().to_string(),
            Value::Phi(id) => format!("Φ{}", id.0),
            Value::Undefined => "⊥".to_string(),
        }
    }
}
