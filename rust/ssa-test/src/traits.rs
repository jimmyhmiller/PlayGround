//! Traits that users implement for their IR types.
//!
//! To use the SSA library with your own IR, implement these traits:
//! - `SsaValue` for your value/operand type
//! - `SsaInstruction` for your instruction type
//! - `InstructionFactory` for creating SSA-specific instructions

use crate::types::{BlockId, PhiId, SsaVariable};

/// Trait for value types that can participate in SSA construction.
///
/// Users implement this on their value enum to enable SSA transformation.
///
/// # Example
/// ```ignore
/// #[derive(Clone, PartialEq, Eq, Hash)]
/// enum MyValue {
///     Int(i64),
///     Var(SsaVariable),
///     Phi(PhiId),
///     Undef,
/// }
///
/// impl SsaValue for MyValue {
///     fn from_phi(id: PhiId) -> Self { MyValue::Phi(id) }
///     fn from_var(v: SsaVariable) -> Self { MyValue::Var(v) }
///     fn undefined() -> Self { MyValue::Undef }
///     fn as_phi(&self) -> Option<PhiId> {
///         match self { MyValue::Phi(id) => Some(*id), _ => None }
///     }
///     fn as_var(&self) -> Option<&SsaVariable> {
///         match self { MyValue::Var(v) => Some(v), _ => None }
///     }
/// }
/// ```
pub trait SsaValue: Clone + PartialEq + Eq + std::hash::Hash + std::fmt::Debug {
    /// Create a phi reference value
    fn from_phi(phi_id: PhiId) -> Self;

    /// Create a variable reference value
    fn from_var(var: SsaVariable) -> Self;

    /// Create an undefined/undef value
    fn undefined() -> Self;

    /// Check if this value is a phi reference, returning the PhiId if so
    fn as_phi(&self) -> Option<PhiId>;

    /// Check if this value is a variable reference, returning the SsaVariable if so
    fn as_var(&self) -> Option<&SsaVariable>;

    /// Check if this value is a phi with the given id
    fn is_same_phi(&self, phi_id: PhiId) -> bool {
        self.as_phi() == Some(phi_id)
    }

    /// Check if this value is any phi reference
    fn is_phi(&self) -> bool {
        self.as_phi().is_some()
    }

    /// Check if this value is undefined
    ///
    /// This is used by validation to detect bugs in CFG transformations
    /// that incorrectly add undefined values to phi operands.
    fn is_undefined(&self) -> bool;
}

/// Trait for instruction types that can participate in SSA construction.
///
/// Users implement this on their instruction enum to enable SSA transformation.
/// The key methods allow the SSA builder to:
/// - Find what variable an instruction defines (if any)
/// - Visit and mutate values in the instruction (for phi replacement)
/// - Identify phi assignments
pub trait SsaInstruction: Clone + std::fmt::Debug {
    type Value: SsaValue;

    /// Visit all value uses in this instruction (for reading).
    ///
    /// This should call the visitor for every value operand in the instruction.
    fn visit_values<F: FnMut(&Self::Value)>(&self, visitor: F);

    /// Visit all value uses in this instruction mutably (for phi replacement).
    ///
    /// This should call the visitor for every value operand that can be mutated.
    fn visit_values_mut<F: FnMut(&mut Self::Value)>(&mut self, visitor: F);

    /// Get the destination variable if this instruction defines one.
    ///
    /// Returns `Some(&var)` for instructions like `var := expr`, `None` for jumps, etc.
    fn destination(&self) -> Option<&SsaVariable>;

    /// Check if this is a phi assignment instruction (e.g., `var := phi`).
    ///
    /// Returns true if the instruction assigns a phi value to a variable.
    fn is_phi_assignment(&self) -> bool;

    /// Get the phi id if this is a phi assignment
    fn get_phi_assignment(&self) -> Option<PhiId>;
}

/// Factory trait for creating SSA-specific instructions.
///
/// The SSA builder needs to create phi assignment instructions.
/// Users implement this to provide the appropriate instruction construction.
pub trait InstructionFactory: Sized {
    type Instr: SsaInstruction;

    /// Create a phi assignment instruction: `dest := phi_id`
    fn create_phi_assign(
        dest: SsaVariable,
        phi_id: PhiId,
    ) -> Self::Instr;

    /// Create a copy/move instruction: `dest := value`
    fn create_copy(
        dest: SsaVariable,
        value: <Self::Instr as SsaInstruction>::Value,
    ) -> Self::Instr;
}

/// Trait for types that can represent control flow targets.
///
/// This is automatically implemented for BlockId but allows users
/// to use their own block identifier types if needed.
pub trait BlockIdentifier: Copy + Eq + std::hash::Hash + std::fmt::Debug {
    fn as_block_id(&self) -> BlockId;
}

impl BlockIdentifier for BlockId {
    fn as_block_id(&self) -> BlockId {
        *self
    }
}
