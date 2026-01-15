//! Traits for optimization-enabled IR types.
//!
//! Users implement these traits on their value and instruction types to enable
//! the generic optimization passes.

use std::fmt::Debug;
use std::hash::Hash;

use crate::traits::{SsaValue, SsaInstruction, InstructionFactory};
use crate::types::{BlockId, SsaVariable};

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

/// Hint about branch probability/semantics for block layout optimization.
///
/// These hints allow instructions to communicate expected branch behavior
/// to the block layout algorithm, enabling better code placement decisions.
///
/// # Example
/// ```ignore
/// // For a guard instruction where failure is unlikely:
/// fn branch_hints(&self) -> Vec<BranchHint> {
///     match self {
///         MyInstr::Guard { fail_target, .. } => {
///             // fail_target is cold (unlikely), fall-through is hot (likely)
///             vec![BranchHint::Unlikely, BranchHint::Likely]
///         }
///         _ => vec![],
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BranchHint {
    /// No hint - use default heuristics (position-based: first=likely, second=unlikely)
    None,
    /// This edge is likely to be taken (hot path, should be fall-through candidate)
    Likely,
    /// This edge is unlikely to be taken (cold path, error handling, GC triggers)
    Unlikely,
    /// This is a loop back-edge (prefer loop exit as fall-through)
    LoopBack,
}

impl Default for BranchHint {
    fn default() -> Self {
        BranchHint::None
    }
}

/// Result of attempting to simplify a control-flow instruction.
///
/// Used by `try_simplify_control_flow()` to indicate how an instruction
/// should be transformed when its operands are known constants.
///
/// Generic over `V` (the Value type) to allow `PassThrough` to carry the
/// source value directly (which could be a variable or constant).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlFlowSimplification<V: OptimizableValue> {
    /// No simplification possible (operands not constant, or not a control-flow instruction)
    NoChange,

    /// Remove this instruction entirely (for instructions with no destination).
    ///
    /// Use this when a guard/check always passes and execution should
    /// continue to the next instruction, and the instruction produces no value.
    Remove,

    /// Replace terminator with unconditional jump (for ConditionalJump-style instructions).
    ///
    /// Use this when a conditional branch has a known outcome.
    ///
    /// - `target`: The block to jump to
    /// - `dead_targets`: Blocks that were targets but will no longer be reached
    Jump {
        target: BlockId,
        dead_targets: Vec<BlockId>,
    },

    /// Guard passes - replace with copy instruction.
    ///
    /// Use this when a guard always passes and produces a value.
    /// `dest := guard_int(value) -> fail` becomes `dest := value`
    ///
    /// The source is the Value type directly, so it works for both variables and constants.
    ///
    /// - `dest`: The destination variable that receives the value
    /// - `source`: The value to copy (can be variable or constant)
    /// - `dead_target`: The fail target edge that should be removed from CFG
    /// - `fall_through_target`: The block to jump to after the copy (makes control flow explicit)
    PassThrough {
        dest: SsaVariable,
        source: V,
        dead_target: BlockId,
        fall_through_target: BlockId,
    },

    /// Guard fails - becomes unconditional jump, rest of block is dead.
    ///
    /// Use this for non-terminator guards that always fail.
    /// The guard becomes a jump, and all instructions after it in the block are removed.
    ///
    /// - `target`: The block to jump to (the guard's fail target)
    /// - `fall_through_target`: The block that was the implicit fall-through (needs phi cleanup)
    FailJump {
        target: BlockId,
        fall_through_target: BlockId,
    },
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

    /// Returns the block IDs that this instruction may jump to.
    ///
    /// Used by CFG analysis and cleanup passes to rebuild predecessor lists.
    /// Should return all possible successors for control-flow instructions:
    /// - Jump: the single target
    /// - ConditionalJump/Branch: both targets
    /// - Guard: both the fail target AND the fall-through target
    /// - Return/non-control-flow: empty vec
    ///
    /// Default implementation returns empty vec (for non-control-flow instructions).
    fn jump_targets(&self) -> Vec<BlockId> {
        vec![]
    }

    /// Returns branch probability hints for each jump target.
    ///
    /// The returned vec should have the same length as `jump_targets()`.
    /// Used by block layout algorithms to make better placement decisions.
    ///
    /// This is particularly important for:
    /// - **Guards**: The fail target should be `Unlikely`, success path `Likely`
    /// - **GC checks**: The GC-triggered path should be `Unlikely`
    /// - **Error handling**: Exception/error paths should be `Unlikely`
    ///
    /// Default implementation returns `BranchHint::None` for all targets,
    /// which tells the layout algorithm to use position-based heuristics
    /// (first target = likely, second = unlikely).
    ///
    /// # Example
    /// ```ignore
    /// fn branch_hints(&self) -> Vec<BranchHint> {
    ///     match self {
    ///         // Guard: fail_target (cold), then fall-through (hot)
    ///         MyInstr::GuardInt { .. } => vec![BranchHint::Unlikely, BranchHint::Likely],
    ///         // GC check: gc_path (cold), normal_path (hot)
    ///         MyInstr::GCCheck { .. } => vec![BranchHint::Unlikely, BranchHint::Likely],
    ///         // Normal conditional: no hint, use default heuristics
    ///         MyInstr::Branch { .. } => vec![BranchHint::None, BranchHint::None],
    ///         _ => vec![],
    ///     }
    /// }
    /// ```
    fn branch_hints(&self) -> Vec<BranchHint> {
        vec![BranchHint::None; self.jump_targets().len()]
    }

    /// Rewrite a jump target from `old` to `new`.
    ///
    /// Used by jump threading to bypass trivial jump blocks.
    /// Should update any occurrence of `old` in the instruction's targets to `new`.
    ///
    /// Returns true if any target was rewritten.
    ///
    /// Default implementation returns false (no targets to rewrite).
    fn rewrite_jump_target(&mut self, _old: BlockId, _new: BlockId) -> bool {
        false
    }

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

    /// Try to simplify this control-flow instruction based on constant operands.
    ///
    /// Called by the control-flow simplification pass on all instructions.
    /// Override this method to enable simplification of guards, conditional
    /// branches, or other control-flow instructions when their operands are
    /// known constants.
    ///
    /// # Returns
    /// - `NoChange`: Keep the instruction as-is (default)
    /// - `Remove`: Delete the instruction (no destination to preserve)
    /// - `Jump`: Replace terminator with unconditional jump
    /// - `PassThrough`: Guard passes, replace with copy (preserves destination)
    /// - `FailJump`: Guard fails mid-block, becomes jump (rest of block is dead)
    ///
    /// # Example
    /// ```ignore
    /// fn try_simplify_control_flow(&self) -> ControlFlowSimplification<Self::Value> {
    ///     match self {
    ///         // Terminator: conditional branch with constant condition
    ///         MyInstr::Branch { cond, then_block, else_block } => {
    ///             if let Some(c) = cond.as_constant() {
    ///                 if is_truthy(*c) {
    ///                     ControlFlowSimplification::Jump {
    ///                         target: *then_block,
    ///                         dead_targets: vec![*else_block],
    ///                     }
    ///                 } else {
    ///                     ControlFlowSimplification::Jump {
    ///                         target: *else_block,
    ///                         dead_targets: vec![*then_block],
    ///                     }
    ///                 }
    ///             } else {
    ///                 ControlFlowSimplification::NoChange
    ///             }
    ///         }
    ///         // Non-terminator guard that produces a value
    ///         // dest := guard_int(value) -> fail_target, falls through to next_block
    ///         MyInstr::GuardInt { dest, value, fail_target, next_block } => {
    ///             if value_is_known_int(value) {
    ///                 // Guard passes - replace with copy, remove edge to fail_target
    ///                 ControlFlowSimplification::PassThrough {
    ///                     dest: *dest,
    ///                     source: value.clone(),
    ///                     dead_target: *fail_target,
    ///                 }
    ///             } else if value_is_known_not_int(value) {
    ///                 // Guard fails - becomes jump, rest of block is dead
    ///                 ControlFlowSimplification::FailJump {
    ///                     target: *fail_target,
    ///                     fall_through_target: *next_block,
    ///                 }
    ///             } else {
    ///                 ControlFlowSimplification::NoChange
    ///             }
    ///         }
    ///         _ => ControlFlowSimplification::NoChange,
    ///     }
    /// }
    /// ```
    fn try_simplify_control_flow(&self) -> ControlFlowSimplification<Self::Value> {
        ControlFlowSimplification::NoChange
    }
}

/// Factory trait for creating optimization-related instructions.
///
/// Extends `InstructionFactory` with the ability to create constant assignments
/// and unconditional jumps.
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

    /// Create an unconditional jump instruction.
    ///
    /// Used by control-flow simplification to replace conditional branches
    /// when the condition is a known constant.
    fn create_jump(target: BlockId) -> Self::Instr;
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
            fn is_undefined(&self) -> bool { self.0 == -1 }
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
