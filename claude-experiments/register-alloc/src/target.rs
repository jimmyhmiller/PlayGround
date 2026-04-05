//! Traits describing the target machine's register file and calling conventions.

use crate::types::*;

/// Describes the physical register file of a target architecture.
pub trait RegInfo {
    /// Iterator over physical registers in a class.
    type RegIter<'a>: Iterator<Item = PReg>
    where
        Self: 'a;

    /// All register classes.
    fn reg_classes(&self) -> &[RegClass];

    /// The physical registers belonging to a class, in allocation-preference order.
    /// Putting preferred registers first lets allocators try the best options first.
    fn class_regs(&self, class: RegClass) -> Self::RegIter<'_>;

    /// How many physical registers are in this class.
    fn class_size(&self, class: RegClass) -> usize;

    /// The register class that a physical register belongs to.
    fn reg_class_of(&self, reg: PReg) -> RegClass;

    /// Human-readable name for a physical register (e.g., "rax", "x0").
    fn reg_name(&self, reg: PReg) -> &str;

    /// Human-readable name for a register class (e.g., "GPR", "FP").
    fn class_name(&self, class: RegClass) -> &str;

    /// The spill size in bytes for values in a register class.
    /// Used for stack slot allocation.
    fn spill_size(&self, class: RegClass) -> u32;

    /// The spill alignment in bytes for values in a register class.
    fn spill_align(&self, class: RegClass) -> u32;

    /// Whether `sub` is a sub-register of `sup`. For architectures
    /// like x86 where `eax` overlaps `rax`. If your architecture has
    /// no sub-register relationships, always return false.
    fn is_sub_reg(&self, sub: PReg, sup: PReg) -> bool {
        let _ = (sub, sup);
        false
    }

    /// Physical registers that overlap with the given register
    /// (including itself). For x86, `rax` overlaps `eax`, `ax`, `al`, `ah`.
    /// Default: just returns the register itself.
    fn overlapping_regs(&self, reg: PReg) -> &[PReg] {
        let _ = reg;
        &[]
    }
}

/// Calling convention: which registers are preserved across calls,
/// how arguments/returns are passed, etc.
pub trait CallingConvention {
    /// Registers preserved across function calls (callee-saved).
    /// The allocator knows these survive calls.
    fn callee_saved(&self) -> &[PReg];

    /// Registers NOT preserved across calls (caller-saved / volatile).
    /// The allocator must save these around call sites if they're live.
    fn caller_saved(&self) -> &[PReg];

    /// Fixed registers used for passing arguments, in order.
    /// e.g., on x86-64 SysV: [rdi, rsi, rdx, rcx, r8, r9] for GPR args.
    fn arg_regs(&self, class: RegClass) -> &[PReg];

    /// Fixed registers used for return values.
    /// e.g., on x86-64: [rax, rdx] for GPR returns.
    fn ret_regs(&self, class: RegClass) -> &[PReg];

    /// The stack pointer register, if there is one.
    /// The allocator will never assign a vreg to the stack pointer.
    fn stack_pointer(&self) -> Option<PReg> {
        None
    }

    /// The frame pointer register, if reserved.
    fn frame_pointer(&self) -> Option<PReg> {
        None
    }

    /// Registers that are always reserved and never available for allocation.
    /// This includes the stack pointer, frame pointer, and any other
    /// platform-reserved registers.
    fn reserved_regs(&self) -> &[PReg];
}

/// Complete target description: register file + calling convention.
/// This is what the allocator receives to know about the hardware.
pub trait Target: RegInfo + CallingConvention {
    /// Registers available for allocation in the given class.
    /// This is class_regs minus reserved_regs. Default impl computes this.
    fn allocatable_regs(&self, class: RegClass) -> Vec<PReg> {
        let reserved = self.reserved_regs();
        self.class_regs(class)
            .filter(|r| !reserved.contains(r))
            .collect()
    }
}

/// Blanket impl: anything that is RegInfo + CallingConvention is a Target.
impl<T: RegInfo + CallingConvention> Target for T {}
