//! Target architecture traits for register allocation.
//!
//! Users implement these traits to describe their target architecture,
//! including available registers, register classes, and calling conventions.

use std::fmt::Debug;
use std::hash::Hash;

/// A physical register on the target machine.
///
/// Users implement this trait for their register type (e.g., an enum of x86 registers).
///
/// # Example
/// ```ignore
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// enum X86Reg {
///     RAX, RBX, RCX, RDX, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15,
/// }
///
/// impl PhysicalRegister for X86Reg {
///     fn id(&self) -> usize {
///         *self as usize
///     }
///     fn name(&self) -> &'static str {
///         match self {
///             X86Reg::RAX => "rax",
///             // ...
///         }
///     }
/// }
/// ```
pub trait PhysicalRegister: Copy + Clone + Eq + Hash + Debug {
    /// Get a unique numeric ID for this register.
    ///
    /// Used for efficient storage in bit vectors and arrays.
    /// IDs should be contiguous starting from 0.
    fn id(&self) -> usize;

    /// Human-readable name for this register (e.g., "rax", "r0").
    fn name(&self) -> &'static str;
}

/// Classification of registers into functional groups.
///
/// Register classes separate registers by their capabilities, such as:
/// - General purpose registers (for integer operations)
/// - Floating point registers (for FP operations)
/// - Vector registers (for SIMD operations)
///
/// # Example
/// ```ignore
/// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// enum X86RegClass {
///     GP,  // General purpose
///     FP,  // Floating point (XMM)
/// }
///
/// impl RegisterClass for X86RegClass {
///     type Register = X86Reg;
///
///     fn name(&self) -> &'static str {
///         match self {
///             X86RegClass::GP => "gp",
///             X86RegClass::FP => "fp",
///         }
///     }
///
///     fn allocatable_registers(&self) -> &'static [X86Reg] {
///         match self {
///             X86RegClass::GP => &[X86Reg::RAX, X86Reg::RBX, /* ... */],
///             X86RegClass::FP => &[/* xmm0, xmm1, ... */],
///         }
///     }
/// }
/// ```
pub trait RegisterClass: Copy + Clone + Eq + Hash + Debug {
    type Register: PhysicalRegister;

    /// Name of this register class (e.g., "gp", "fp", "vec").
    fn name(&self) -> &'static str;

    /// All registers in this class that are available for allocation.
    ///
    /// This should exclude reserved registers like the stack pointer.
    fn allocatable_registers(&self) -> &'static [Self::Register];

    /// Number of allocatable registers in this class.
    fn num_registers(&self) -> usize
    where
        Self::Register: 'static,
    {
        self.allocatable_registers().len()
    }
}

/// Describes the target architecture for register allocation.
///
/// This is the main trait users implement to configure the register allocator
/// for their target machine.
///
/// # Example
/// ```ignore
/// #[derive(Debug, Clone)]
/// struct X86_64;
///
/// impl TargetArchitecture for X86_64 {
///     type Register = X86Reg;
///     type Class = X86RegClass;
///
///     fn register_classes(&self) -> &'static [X86RegClass] {
///         &[X86RegClass::GP, X86RegClass::FP]
///     }
///
///     fn default_class(&self) -> X86RegClass {
///         X86RegClass::GP
///     }
///
///     fn stack_slot_size(&self) -> usize {
///         8  // 64-bit slots
///     }
/// }
/// ```
pub trait TargetArchitecture: Clone + Debug {
    type Register: PhysicalRegister;
    type Class: RegisterClass<Register = Self::Register>;

    /// All register classes available on this architecture.
    fn register_classes(&self) -> &'static [Self::Class];

    /// The default register class for values without specific requirements.
    ///
    /// Typically this is the general-purpose integer register class.
    fn default_class(&self) -> Self::Class;

    /// Size of a single stack slot in bytes.
    ///
    /// Used for calculating spill slot offsets.
    fn stack_slot_size(&self) -> usize;

    /// Stack alignment requirement in bytes.
    ///
    /// The stack frame will be aligned to this boundary.
    fn stack_alignment(&self) -> usize {
        self.stack_slot_size()
    }

    /// Registers reserved for special purposes (not available for allocation).
    ///
    /// Examples: stack pointer, frame pointer, thread-local storage pointer.
    fn reserved_registers(&self) -> &'static [Self::Register] {
        &[]
    }

    /// Caller-saved registers (must be saved by caller across calls).
    ///
    /// These registers may be clobbered by function calls.
    fn caller_saved(&self) -> &'static [Self::Register] {
        &[]
    }

    /// Callee-saved registers (must be preserved by called function).
    ///
    /// The register allocator should prefer these for long-lived values
    /// that span function calls.
    fn callee_saved(&self) -> &'static [Self::Register] {
        &[]
    }

    /// Total number of registers across all classes.
    fn total_registers(&self) -> usize
    where
        Self::Class: 'static,
        <Self::Class as RegisterClass>::Register: 'static,
    {
        self.register_classes()
            .iter()
            .map(|c| c.num_registers())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test architecture with 4 GP registers
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TestReg {
        R0,
        R1,
        R2,
        R3,
    }

    impl PhysicalRegister for TestReg {
        fn id(&self) -> usize {
            match self {
                TestReg::R0 => 0,
                TestReg::R1 => 1,
                TestReg::R2 => 2,
                TestReg::R3 => 3,
            }
        }

        fn name(&self) -> &'static str {
            match self {
                TestReg::R0 => "r0",
                TestReg::R1 => "r1",
                TestReg::R2 => "r2",
                TestReg::R3 => "r3",
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TestRegClass {
        GP,
    }

    impl RegisterClass for TestRegClass {
        type Register = TestReg;

        fn name(&self) -> &'static str {
            "gp"
        }

        fn allocatable_registers(&self) -> &'static [TestReg] {
            &[TestReg::R0, TestReg::R1, TestReg::R2, TestReg::R3]
        }
    }

    #[derive(Debug, Clone)]
    struct TestArch;

    impl TargetArchitecture for TestArch {
        type Register = TestReg;
        type Class = TestRegClass;

        fn register_classes(&self) -> &'static [TestRegClass] {
            &[TestRegClass::GP]
        }

        fn default_class(&self) -> TestRegClass {
            TestRegClass::GP
        }

        fn stack_slot_size(&self) -> usize {
            4
        }
    }

    #[test]
    fn test_register_properties() {
        assert_eq!(TestReg::R0.id(), 0);
        assert_eq!(TestReg::R3.name(), "r3");
    }

    #[test]
    fn test_register_class() {
        let class = TestRegClass::GP;
        assert_eq!(class.num_registers(), 4);
        assert_eq!(class.allocatable_registers()[0], TestReg::R0);
    }

    #[test]
    fn test_architecture() {
        let arch = TestArch;
        assert_eq!(arch.total_registers(), 4);
        assert_eq!(arch.stack_slot_size(), 4);
        assert_eq!(arch.default_class(), TestRegClass::GP);
    }
}
