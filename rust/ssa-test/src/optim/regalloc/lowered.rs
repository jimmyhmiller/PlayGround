//! Lowered IR with explicit register assignments.
//!
//! After register allocation, the IR is "lowered" to use physical registers
//! and stack slots instead of SSA virtual registers. This module provides
//! types to represent this lowered form.

use std::collections::HashMap;
use std::fmt::Debug;

use crate::types::{BlockId, SsaVariable};

use super::interval::Location;
use super::linear_scan::AllocationResult;
use super::target::PhysicalRegister;

/// An operand in lowered IR with explicit physical location.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LoweredOperand<R: PhysicalRegister> {
    /// A physical register.
    Register(R),

    /// A stack slot (offset from frame pointer or stack pointer).
    StackSlot(usize),

    /// An immediate/constant value.
    Immediate(i64),

    /// Undefined value.
    Undefined,
}

impl<R: PhysicalRegister> LoweredOperand<R> {
    /// Check if this operand is a register.
    pub fn is_register(&self) -> bool {
        matches!(self, LoweredOperand::Register(_))
    }

    /// Check if this operand is a stack slot.
    pub fn is_stack_slot(&self) -> bool {
        matches!(self, LoweredOperand::StackSlot(_))
    }

    /// Get the register if this is a Register operand.
    pub fn as_register(&self) -> Option<R> {
        match self {
            LoweredOperand::Register(r) => Some(*r),
            _ => None,
        }
    }

    /// Get the stack slot if this is a StackSlot operand.
    pub fn as_stack_slot(&self) -> Option<usize> {
        match self {
            LoweredOperand::StackSlot(s) => Some(*s),
            _ => None,
        }
    }
}

/// A lowered instruction with register assignments.
///
/// This wraps an original instruction with information about where
/// its operands are located.
#[derive(Debug, Clone)]
pub struct LoweredInstruction<R: PhysicalRegister, I: Clone + Debug> {
    /// The original instruction.
    pub original: I,

    /// Destination operand location (if any).
    pub dest: Option<LoweredOperand<R>>,

    /// Source operand locations.
    pub sources: Vec<LoweredOperand<R>>,
}

impl<R: PhysicalRegister, I: Clone + Debug> LoweredInstruction<R, I> {
    /// Create a new lowered instruction.
    pub fn new(original: I) -> Self {
        LoweredInstruction {
            original,
            dest: None,
            sources: Vec::new(),
        }
    }

    /// Set the destination operand.
    pub fn with_dest(mut self, dest: LoweredOperand<R>) -> Self {
        self.dest = Some(dest);
        self
    }

    /// Add a source operand.
    pub fn with_source(mut self, source: LoweredOperand<R>) -> Self {
        self.sources.push(source);
        self
    }
}

/// A lowered basic block.
#[derive(Debug, Clone)]
pub struct LoweredBlock<R: PhysicalRegister, I: Clone + Debug> {
    /// Block identifier.
    pub id: BlockId,

    /// Lowered instructions in this block.
    pub instructions: Vec<LoweredInstruction<R, I>>,
}

impl<R: PhysicalRegister, I: Clone + Debug> LoweredBlock<R, I> {
    /// Create a new lowered block.
    pub fn new(id: BlockId) -> Self {
        LoweredBlock {
            id,
            instructions: Vec::new(),
        }
    }

    /// Add an instruction to the block.
    pub fn add_instruction(&mut self, instr: LoweredInstruction<R, I>) {
        self.instructions.push(instr);
    }
}

/// A complete lowered function after register allocation.
#[derive(Debug, Clone)]
pub struct LoweredFunction<R: PhysicalRegister, I: Clone + Debug> {
    /// Lowered basic blocks.
    pub blocks: Vec<LoweredBlock<R, I>>,

    /// Total stack frame size in bytes.
    pub stack_frame_size: usize,

    /// Callee-saved registers that need to be preserved.
    pub callee_saved_used: Vec<R>,

    /// Map from original SSA variables to their locations.
    pub register_map: HashMap<SsaVariable, LoweredOperand<R>>,
}

impl<R: PhysicalRegister, I: Clone + Debug> LoweredFunction<R, I> {
    /// Create an empty lowered function.
    pub fn new() -> Self {
        LoweredFunction {
            blocks: Vec::new(),
            stack_frame_size: 0,
            callee_saved_used: Vec::new(),
            register_map: HashMap::new(),
        }
    }

    /// Add a block to the function.
    pub fn add_block(&mut self, block: LoweredBlock<R, I>) {
        self.blocks.push(block);
    }

    /// Get the location for a variable.
    pub fn get_location(&self, var: &SsaVariable) -> Option<&LoweredOperand<R>> {
        self.register_map.get(var)
    }
}

impl<R: PhysicalRegister, I: Clone + Debug> Default for LoweredFunction<R, I> {
    fn default() -> Self {
        Self::new()
    }
}

/// Map from SSA variables to their physical locations.
///
/// This is a simpler alternative to full IR lowering when you just
/// need to know where each variable lives.
#[derive(Debug, Clone)]
pub struct RegisterMap<R: PhysicalRegister> {
    /// Map from variable to location.
    assignments: HashMap<SsaVariable, LoweredOperand<R>>,

    /// All registers in the allocatable set.
    registers: Vec<R>,
}

impl<R: PhysicalRegister> RegisterMap<R> {
    /// Create a register map from allocation results.
    ///
    /// The `registers` slice should contain all physical registers in order
    /// by their ID, so that `registers[id]` gives the register for that ID.
    pub fn from_allocation(allocation: &AllocationResult, registers: &[R]) -> Self {
        let assignments = allocation.assignments
            .iter()
            .map(|(var, loc)| {
                let operand = match loc {
                    Location::Register(id) => {
                        if *id < registers.len() {
                            LoweredOperand::Register(registers[*id])
                        } else {
                            // Invalid register ID - shouldn't happen
                            LoweredOperand::Undefined
                        }
                    }
                    Location::StackSlot(slot) => LoweredOperand::StackSlot(*slot),
                };
                (var.clone(), operand)
            })
            .collect();

        RegisterMap {
            assignments,
            registers: registers.to_vec(),
        }
    }

    /// Get the location for a variable.
    pub fn get(&self, var: &SsaVariable) -> Option<&LoweredOperand<R>> {
        self.assignments.get(var)
    }

    /// Get all variables assigned to registers.
    pub fn register_assigned_vars(&self) -> impl Iterator<Item = (&SsaVariable, R)> {
        self.assignments.iter().filter_map(|(var, op)| {
            if let LoweredOperand::Register(r) = op {
                Some((var, *r))
            } else {
                None
            }
        })
    }

    /// Get all variables spilled to stack.
    pub fn stack_assigned_vars(&self) -> impl Iterator<Item = (&SsaVariable, usize)> {
        self.assignments.iter().filter_map(|(var, op)| {
            if let LoweredOperand::StackSlot(s) = op {
                Some((var, *s))
            } else {
                None
            }
        })
    }

    /// Count variables in registers.
    pub fn num_in_registers(&self) -> usize {
        self.assignments.values()
            .filter(|op| op.is_register())
            .count()
    }

    /// Count variables on stack.
    pub fn num_on_stack(&self) -> usize {
        self.assignments.values()
            .filter(|op| op.is_stack_slot())
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::linear_scan::AllocationStats;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum TestReg { R0, R1, R2 }

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
    fn test_lowered_operand() {
        let reg: LoweredOperand<TestReg> = LoweredOperand::Register(TestReg::R1);
        let stack: LoweredOperand<TestReg> = LoweredOperand::StackSlot(5);
        let imm: LoweredOperand<TestReg> = LoweredOperand::Immediate(42);

        assert!(reg.is_register());
        assert!(!reg.is_stack_slot());
        assert_eq!(reg.as_register(), Some(TestReg::R1));

        assert!(stack.is_stack_slot());
        assert!(!stack.is_register());
        assert_eq!(stack.as_stack_slot(), Some(5));

        assert!(!imm.is_register());
        assert!(!imm.is_stack_slot());
    }

    #[test]
    fn test_register_map() {
        let mut assignments = HashMap::new();
        assignments.insert(SsaVariable::new("a"), Location::Register(0));
        assignments.insert(SsaVariable::new("b"), Location::Register(2));
        assignments.insert(SsaVariable::new("c"), Location::StackSlot(0));

        let allocation = AllocationResult {
            assignments,
            stack_slots_used: 1,
            stats: AllocationStats::default(),
        };

        let registers = [TestReg::R0, TestReg::R1, TestReg::R2];
        let map = RegisterMap::from_allocation(&allocation, &registers);

        assert_eq!(
            map.get(&SsaVariable::new("a")),
            Some(&LoweredOperand::Register(TestReg::R0))
        );
        assert_eq!(
            map.get(&SsaVariable::new("b")),
            Some(&LoweredOperand::Register(TestReg::R2))
        );
        assert_eq!(
            map.get(&SsaVariable::new("c")),
            Some(&LoweredOperand::StackSlot(0))
        );

        assert_eq!(map.num_in_registers(), 2);
        assert_eq!(map.num_on_stack(), 1);
    }
}
