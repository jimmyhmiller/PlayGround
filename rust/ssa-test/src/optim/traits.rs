//! Traits for optimization-enabled IR types.
//!
//! Users implement these traits on their value and instruction types to enable
//! the generic optimization passes.

use std::fmt::Debug;
use std::hash::Hash;

use crate::traits::{SsaValue, SsaInstruction, InstructionFactory};
use crate::types::SsaVariable;

/// Trait for value types that can participate in optimizations.
///
/// Extends `SsaValue` with the ability to represent and extract constants.
///
/// # Example
/// ```ignore
/// impl OptimizableValue for MyValue {
///     type Constant = i64;
///
///     fn as_constant(&self) -> Option<&i64> {
///         match self {
///             MyValue::Const(n) => Some(n),
///             _ => None,
///         }
///     }
///
///     fn from_constant(c: i64) -> Self {
///         MyValue::Const(c)
///     }
/// }
/// ```
pub trait OptimizableValue: SsaValue {
    /// The constant type (e.g., i64, f64, or a custom constant enum)
    type Constant: Clone + Eq + Hash + Debug;

    /// Extract a constant value if this is a constant.
    fn as_constant(&self) -> Option<&Self::Constant>;

    /// Create a value from a constant.
    fn from_constant(c: Self::Constant) -> Self;

    /// Check if this value is a constant.
    fn is_constant(&self) -> bool {
        self.as_constant().is_some()
    }
}

/// Key for identifying equivalent expressions in CSE.
///
/// Expressions with the same key compute the same value and can be deduplicated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExpressionKey<V: OptimizableValue> {
    /// Binary operation: (left, op_name, right)
    BinaryOp {
        left: V,
        op: String,
        right: V,
    },
    /// Unary operation: (op_name, operand)
    UnaryOp {
        op: String,
        operand: V,
    },
    /// Custom expression key for user-defined operations
    Custom(String, Vec<V>),
}

/// Trait for instruction types that can participate in optimizations.
///
/// Extends `SsaInstruction` with methods needed by optimization passes.
///
/// # Example
/// ```ignore
/// impl OptimizableInstruction for MyInstr {
///     fn has_side_effects(&self) -> bool {
///         matches!(self, MyInstr::Print(_) | MyInstr::Store { .. })
///     }
///
///     fn is_terminator(&self) -> bool {
///         matches!(self, MyInstr::Jump(_) | MyInstr::Branch { .. } | MyInstr::Return(_))
///     }
///
///     fn as_copy(&self) -> Option<(&SsaVariable, &Self::Value)> {
///         match self {
///             MyInstr::Assign { dest, value } if !value.is_phi() => Some((dest, value)),
///             _ => None,
///         }
///     }
///
///     fn try_fold(&self) -> Option<<Self::Value as OptimizableValue>::Constant> {
///         // ... evaluate if all operands are constants
///     }
///
///     fn expression_key(&self) -> Option<ExpressionKey<Self::Value>> {
///         // ... return hashable key for pure expressions
///     }
/// }
/// ```
pub trait OptimizableInstruction: SsaInstruction
where
    Self::Value: OptimizableValue,
{
    /// Returns true if this instruction has observable side effects.
    ///
    /// Instructions with side effects cannot be eliminated even if their
    /// result is unused. Examples: print, store, I/O, function calls with
    /// side effects.
    fn has_side_effects(&self) -> bool;

    /// Returns true if this instruction is a terminator (ends a basic block).
    ///
    /// Terminators control flow and cannot be eliminated.
    /// Examples: jump, branch, return, unreachable.
    fn is_terminator(&self) -> bool;

    /// If this is a simple copy/move instruction, returns (dest, src).
    ///
    /// A copy is `dest := src` where src is a single value (not a computation).
    /// Used by copy propagation to replace uses of dest with src.
    ///
    /// Note: Phi assignments should NOT be returned as copies.
    fn as_copy(&self) -> Option<(&SsaVariable, &Self::Value)>;

    /// Try to evaluate this instruction at compile time.
    ///
    /// If all operands are constants and the operation can be evaluated,
    /// returns the result constant. Used by constant folding.
    ///
    /// Returns `None` if:
    /// - Not all operands are constants
    /// - The operation cannot be safely evaluated (e.g., divide by zero)
    /// - The instruction is not a pure computation
    fn try_fold(&self) -> Option<<Self::Value as OptimizableValue>::Constant>;

    /// Returns a hashable key identifying this expression for CSE.
    ///
    /// Two instructions with the same expression key compute the same value
    /// (given the same inputs). Used by common subexpression elimination.
    ///
    /// Returns `None` for:
    /// - Instructions with side effects
    /// - Instructions that don't compute a value
    /// - Operations that shouldn't be deduplicated
    fn expression_key(&self) -> Option<ExpressionKey<Self::Value>>;
}

/// Factory trait for creating optimization-related instructions.
///
/// Extends `InstructionFactory` with the ability to create constant assignments.
pub trait InstructionMutator: InstructionFactory
where
    <Self::Instr as SsaInstruction>::Value: OptimizableValue,
{
    /// Create an instruction that assigns a constant to a variable.
    ///
    /// `dest := constant`
    fn create_constant_assign(
        dest: SsaVariable,
        constant: <<Self::Instr as SsaInstruction>::Value as OptimizableValue>::Constant,
    ) -> Self::Instr;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test that ExpressionKey can be used in HashMaps
    #[test]
    fn test_expression_key_hash() {
        use std::collections::HashSet;

        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        struct TestValue(i32);

        impl SsaValue for TestValue {
            fn from_phi(_: crate::types::PhiId) -> Self { TestValue(0) }
            fn from_var(_: SsaVariable) -> Self { TestValue(0) }
            fn undefined() -> Self { TestValue(-1) }
            fn as_phi(&self) -> Option<crate::types::PhiId> { None }
            fn as_var(&self) -> Option<&SsaVariable> { None }
        }

        impl OptimizableValue for TestValue {
            type Constant = i32;
            fn as_constant(&self) -> Option<&i32> { Some(&self.0) }
            fn from_constant(c: i32) -> Self { TestValue(c) }
        }

        let mut set: HashSet<ExpressionKey<TestValue>> = HashSet::new();

        let key1 = ExpressionKey::BinaryOp {
            left: TestValue(1),
            op: "add".to_string(),
            right: TestValue(2),
        };
        let key2 = ExpressionKey::BinaryOp {
            left: TestValue(1),
            op: "add".to_string(),
            right: TestValue(2),
        };
        let key3 = ExpressionKey::BinaryOp {
            left: TestValue(1),
            op: "add".to_string(),
            right: TestValue(3), // Different!
        };

        set.insert(key1.clone());
        assert!(set.contains(&key2)); // Same as key1
        assert!(!set.contains(&key3)); // Different
    }
}
