//! Register constraints for instructions.
//!
//! Some instructions require operands to be in specific registers:
//! - Division on x86 requires RAX/RDX
//! - Function calls require arguments in specific registers
//! - Some instructions clobber certain registers
//!
//! This module provides types for describing these constraints.

use super::target::PhysicalRegister;

/// Constraint on where a value must reside.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegisterConstraint<R: PhysicalRegister> {
    /// Any register in the appropriate class is acceptable.
    Any,

    /// Must be in this specific physical register.
    ///
    /// Examples:
    /// - x86 `div` instruction: quotient must go to RAX, remainder to RDX
    /// - x86 `shl` with variable shift: shift amount must be in CL
    Fixed(R),

    /// Must be in the same register as another operand (by index).
    ///
    /// Used for two-address instructions where the destination
    /// must be the same register as one of the sources.
    ///
    /// Example: x86 `add rax, rbx` means rax = rax + rbx,
    /// so the destination is constrained to be the same as source 0.
    SameAs(usize),
}

impl<R: PhysicalRegister> RegisterConstraint<R> {
    /// Returns true if this constraint requires a specific register.
    pub fn is_fixed(&self) -> bool {
        matches!(self, RegisterConstraint::Fixed(_))
    }

    /// Returns the fixed register if this is a Fixed constraint.
    pub fn fixed_register(&self) -> Option<R> {
        match self {
            RegisterConstraint::Fixed(r) => Some(*r),
            _ => None,
        }
    }
}

/// Constraints for all operands of an instruction.
#[derive(Debug, Clone)]
pub struct OperandConstraints<R: PhysicalRegister> {
    /// Constraint for the destination operand (if any).
    pub dest: Option<RegisterConstraint<R>>,

    /// Constraints for each source operand.
    pub sources: Vec<RegisterConstraint<R>>,

    /// Registers clobbered (destroyed) by this instruction.
    ///
    /// These registers cannot hold live values across this instruction.
    /// Common examples:
    /// - Call instructions clobber caller-saved registers
    /// - x86 `div` clobbers RAX and RDX
    pub clobbers: Vec<R>,

    /// Early clobbers: registers written before inputs are read.
    ///
    /// The destination register cannot be the same as any input register.
    /// Used when an instruction writes its destination before reading all inputs.
    pub early_clobbers: Vec<R>,
}

impl<R: PhysicalRegister> Default for OperandConstraints<R> {
    fn default() -> Self {
        Self {
            dest: None,
            sources: Vec::new(),
            clobbers: Vec::new(),
            early_clobbers: Vec::new(),
        }
    }
}

impl<R: PhysicalRegister> OperandConstraints<R> {
    /// Create unconstrained operand constraints.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the destination constraint.
    pub fn with_dest(mut self, constraint: RegisterConstraint<R>) -> Self {
        self.dest = Some(constraint);
        self
    }

    /// Add a source operand constraint.
    pub fn with_source(mut self, constraint: RegisterConstraint<R>) -> Self {
        self.sources.push(constraint);
        self
    }

    /// Add a clobbered register.
    pub fn with_clobber(mut self, reg: R) -> Self {
        self.clobbers.push(reg);
        self
    }

    /// Add an early clobber.
    pub fn with_early_clobber(mut self, reg: R) -> Self {
        self.early_clobbers.push(reg);
        self
    }

    /// Check if any fixed constraints exist.
    pub fn has_fixed_constraints(&self) -> bool {
        self.dest.as_ref().map_or(false, |c| c.is_fixed())
            || self.sources.iter().any(|c| c.is_fixed())
    }

    /// Get all fixed registers required by this instruction.
    pub fn fixed_registers(&self) -> Vec<R> {
        let mut regs = Vec::new();
        if let Some(RegisterConstraint::Fixed(r)) = &self.dest {
            regs.push(*r);
        }
        for source in &self.sources {
            if let RegisterConstraint::Fixed(r) = source {
                regs.push(*r);
            }
        }
        regs
    }
}

/// Trait for instructions that have register constraints.
///
/// Implement this trait on your instruction type to specify which
/// operands require specific registers.
///
/// # Example
/// ```ignore
/// impl HasRegisterConstraints for MyInstruction {
///     type Register = X86Reg;
///
///     fn register_constraints(&self) -> Option<OperandConstraints<X86Reg>> {
///         match self {
///             MyInstruction::Div { .. } => Some(
///                 OperandConstraints::new()
///                     .with_dest(RegisterConstraint::Fixed(X86Reg::RAX))
///                     .with_clobber(X86Reg::RDX)
///             ),
///             MyInstruction::Call { .. } => Some(
///                 OperandConstraints::new()
///                     .with_clobber(X86Reg::RAX)
///                     .with_clobber(X86Reg::RCX)
///                     // ... other caller-saved registers
///             ),
///             _ => None,  // No special constraints
///         }
///     }
/// }
/// ```
pub trait HasRegisterConstraints {
    type Register: PhysicalRegister;

    /// Return register constraints for this instruction.
    ///
    /// Returns `None` if the instruction has no special register requirements
    /// (all operands can be in any register of their class).
    fn register_constraints(&self) -> Option<OperandConstraints<Self::Register>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TestReg {
        R0,
        R1,
        R2,
    }

    impl PhysicalRegister for TestReg {
        fn id(&self) -> usize {
            match self {
                TestReg::R0 => 0,
                TestReg::R1 => 1,
                TestReg::R2 => 2,
            }
        }

        fn name(&self) -> &'static str {
            match self {
                TestReg::R0 => "r0",
                TestReg::R1 => "r1",
                TestReg::R2 => "r2",
            }
        }
    }

    #[test]
    fn test_constraint_types() {
        let any: RegisterConstraint<TestReg> = RegisterConstraint::Any;
        let fixed: RegisterConstraint<TestReg> = RegisterConstraint::Fixed(TestReg::R0);
        let same_as: RegisterConstraint<TestReg> = RegisterConstraint::SameAs(0);

        assert!(!any.is_fixed());
        assert!(fixed.is_fixed());
        assert!(!same_as.is_fixed());

        assert_eq!(fixed.fixed_register(), Some(TestReg::R0));
        assert_eq!(any.fixed_register(), None);
    }

    #[test]
    fn test_operand_constraints_builder() {
        let constraints: OperandConstraints<TestReg> = OperandConstraints::new()
            .with_dest(RegisterConstraint::Fixed(TestReg::R0))
            .with_source(RegisterConstraint::Any)
            .with_source(RegisterConstraint::Fixed(TestReg::R1))
            .with_clobber(TestReg::R2);

        assert!(constraints.has_fixed_constraints());
        assert_eq!(constraints.fixed_registers(), vec![TestReg::R0, TestReg::R1]);
        assert_eq!(constraints.clobbers, vec![TestReg::R2]);
    }
}
