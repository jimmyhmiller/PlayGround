#![allow(dead_code)]
use std::collections::{BTreeMap, HashMap};

use crate::ir::{Instruction, IrValue, VirtualRegister};

/// Linear scan register allocator
///
/// This is adapted from Beagle's linear scan implementation.
/// It assigns physical ARM64 registers (x19-x28) to virtual registers,
/// and spills to stack when necessary.
pub struct LinearScan {
    /// Lifetime intervals for each virtual register (start, end)
    pub lifetimes: HashMap<VirtualRegister, (usize, usize)>,

    /// IR instructions being processed
    pub instructions: Vec<Instruction>,

    /// Mapping from virtual registers to physical registers
    /// Uses BTreeMap for deterministic iteration
    pub allocated_registers: BTreeMap<VirtualRegister, VirtualRegister>,

    /// Available physical registers
    pub free_registers: Vec<VirtualRegister>,

    /// Spilled registers and their stack locations
    pub spill_locations: HashMap<VirtualRegister, usize>,

    /// Next available stack slot
    pub next_stack_slot: usize,

    /// Maximum number of physical registers available
    pub max_registers: usize,
}

/// Create a physical register
fn physical(index: usize) -> VirtualRegister {
    VirtualRegister {
        index,
        is_argument: false,
    }
}

impl LinearScan {
    /// Create a new linear scan allocator
    ///
    /// num_locals: number of stack slots already used for local variables
    pub fn new(instructions: Vec<Instruction>, num_locals: usize) -> Self {
        let lifetimes = Self::compute_lifetimes(&instructions);

        // Use ARM64 callee-saved registers (x19-x28)
        // These are safe to use across function calls
        let physical_registers: Vec<VirtualRegister> = (19..=28).map(physical).collect();
        let max_registers = physical_registers.len();

        LinearScan {
            lifetimes,
            instructions,
            allocated_registers: BTreeMap::new(),
            free_registers: physical_registers,
            max_registers,
            spill_locations: HashMap::new(),
            next_stack_slot: num_locals,
        }
    }

    /// Compute lifetime intervals for all virtual registers
    ///
    /// A register's lifetime is from its first definition to its last use.
    fn compute_lifetimes(
        instructions: &[Instruction],
    ) -> HashMap<VirtualRegister, (usize, usize)> {
        let mut result: HashMap<VirtualRegister, (usize, usize)> = HashMap::new();

        // Scan backwards to find first definition and last use
        for (index, instruction) in instructions.iter().enumerate().rev() {
            for register in Self::get_registers_in_instruction(instruction) {
                if let Some((_start, end)) = result.get(&register) {
                    // Extend lifetime backwards
                    result.insert(register, (index, *end));
                } else {
                    // First time seeing this register (scanning backwards)
                    result.insert(register, (index, index));
                }
            }
        }
        result
    }

    /// Extract all virtual registers used in an instruction
    fn get_registers_in_instruction(inst: &Instruction) -> Vec<VirtualRegister> {
        let mut regs = Vec::new();

        match inst {
            Instruction::AddInt(dst, src1, src2)
            | Instruction::Sub(dst, src1, src2)
            | Instruction::Mul(dst, src1, src2)
            | Instruction::Div(dst, src1, src2) => {
                if let IrValue::Register(r) = dst { regs.push(*r); }
                if let IrValue::Register(r) = src1 { regs.push(*r); }
                if let IrValue::Register(r) = src2 { regs.push(*r); }
            }

            Instruction::Compare(dst, src1, src2, _) => {
                if let IrValue::Register(r) = dst { regs.push(*r); }
                if let IrValue::Register(r) = src1 { regs.push(*r); }
                if let IrValue::Register(r) = src2 { regs.push(*r); }
            }

            Instruction::Tag(dst, src, _tag) => {
                if let IrValue::Register(r) = dst { regs.push(*r); }
                if let IrValue::Register(r) = src { regs.push(*r); }
            }

            Instruction::Untag(dst, src) => {
                if let IrValue::Register(r) = dst { regs.push(*r); }
                if let IrValue::Register(r) = src { regs.push(*r); }
            }

            Instruction::LoadConstant(dst, _)
            | Instruction::LoadVar(dst, _)
            | Instruction::LoadTrue(dst)
            | Instruction::LoadFalse(dst) => {
                if let IrValue::Register(r) = dst { regs.push(*r); }
            }

            Instruction::StoreVar(var_ptr, value) => {
                // StoreVar uses both var_ptr and value registers
                if let IrValue::Register(r) = var_ptr { regs.push(*r); }
                if let IrValue::Register(r) = value { regs.push(*r); }
            }

            Instruction::Assign(dst, src) => {
                if let IrValue::Register(r) = dst { regs.push(*r); }
                if let IrValue::Register(r) = src { regs.push(*r); }
            }

            Instruction::JumpIf(_, _, val1, val2) => {
                if let IrValue::Register(r) = val1 { regs.push(*r); }
                if let IrValue::Register(r) = val2 { regs.push(*r); }
            }

            Instruction::Ret(val) => {
                if let IrValue::Register(r) = val { regs.push(*r); }
            }

            Instruction::Label(_) | Instruction::Jump(_) => {
                // No registers
            }

            Instruction::PushBinding(var_ptr, value) => {
                if let IrValue::Register(r) = var_ptr { regs.push(*r); }
                if let IrValue::Register(r) = value { regs.push(*r); }
            }

            Instruction::PopBinding(var_ptr) => {
                if let IrValue::Register(r) = var_ptr { regs.push(*r); }
            }

            Instruction::SetVar(var_ptr, value) => {
                if let IrValue::Register(r) = var_ptr { regs.push(*r); }
                if let IrValue::Register(r) = value { regs.push(*r); }
            }
        }

        regs
    }

    /// Run the linear scan register allocation algorithm
    ///
    /// This is the main algorithm from "Linear Scan Register Allocation" by Poletto & Sarkar.
    ///
    /// Algorithm:
    /// 1. Sort intervals by start point
    /// 2. For each interval:
    ///    - Expire old intervals (free their registers)
    ///    - If all registers are in use, spill
    ///    - Otherwise, allocate a free register
    pub fn allocate(&mut self) {
        // Create sorted list of intervals (start, end, register)
        let mut intervals: Vec<(usize, usize, VirtualRegister)> = self
            .lifetimes
            .iter()
            .map(|(register, (start, end))| (*start, *end, *register))
            .collect();

        intervals.sort_by_key(|(start, _, _)| *start);

        // Active intervals (currently live)
        let mut active: Vec<(usize, usize, VirtualRegister)> = Vec::new();

        for interval in intervals.iter() {
            let (start, end, vreg) = *interval;

            // Free registers that are no longer live
            self.expire_old_intervals(start, &mut active);

            // Check if we need to spill
            if active.len() >= self.max_registers {
                self.spill_at_interval(start, end, vreg, &mut active);
            } else {
                // Allocate a free register
                if let Some(physical_reg) = self.free_registers.pop() {
                    self.allocated_registers.insert(vreg, physical_reg);
                    active.push((start, end, vreg));
                    active.sort_by_key(|(_, end, _)| *end);
                } else {
                    // This shouldn't happen if active.len() < max_registers
                    panic!("No free registers available!");
                }
            }
        }

        // Replace virtual registers with allocated physical registers in instructions
        self.apply_allocation();
    }

    /// Expire old intervals - free registers that are no longer live
    fn expire_old_intervals(
        &mut self,
        current_start: usize,
        active: &mut Vec<(usize, usize, VirtualRegister)>,
    ) {
        // Sort by end point
        active.sort_by_key(|(_, end, _)| *end);

        // Remove intervals that have expired
        let mut i = 0;
        while i < active.len() {
            let (_, end, vreg) = active[i];

            if end >= current_start {
                // This interval is still active
                break;
            }

            // Interval has expired - free its register
            if let Some(&physical_reg) = self.allocated_registers.get(&vreg) {
                self.free_register(physical_reg);
            }

            active.remove(i);
        }
    }

    /// Spill a register when all physical registers are in use
    fn spill_at_interval(
        &mut self,
        _start: usize,
        end: usize,
        vreg: VirtualRegister,
        active: &mut Vec<(usize, usize, VirtualRegister)>,
    ) {
        // Sort by end point (descending)
        active.sort_by_key(|(_, end, _)| *end);

        // Get the interval that ends last (spill candidate)
        let spill = *active.last().unwrap();
        let (_, spill_end, spill_vreg) = spill;

        if spill_end > end {
            // Spill the interval that ends last, allocate its register to current interval
            let physical_reg = *self.allocated_registers.get(&spill_vreg).unwrap();
            self.allocated_registers.insert(vreg, physical_reg);

            // Mark spilled register
            let stack_slot = self.allocate_stack_slot();
            self.spill_locations.insert(spill_vreg, stack_slot);

            // Remove from active and add current interval
            active.retain(|x| *x != spill);
            active.push((_start, end, vreg));
            active.sort_by_key(|(_, end, _)| *end);
        } else {
            // Current interval is shorter, spill it instead
            let stack_slot = self.allocate_stack_slot();
            self.spill_locations.insert(vreg, stack_slot);
        }
    }

    /// Free a physical register
    fn free_register(&mut self, register: VirtualRegister) {
        self.free_registers.push(register);
    }

    /// Allocate a new stack slot for spilling
    fn allocate_stack_slot(&mut self) -> usize {
        let slot = self.next_stack_slot;
        self.next_stack_slot += 1;
        slot
    }

    /// Apply the register allocation to instructions
    fn apply_allocation(&mut self) {
        for inst in self.instructions.iter_mut() {
            Self::replace_registers_in_instruction(inst, &self.allocated_registers);
        }
    }

    /// Replace virtual registers with allocated physical registers in an instruction
    fn replace_registers_in_instruction(
        inst: &mut Instruction,
        allocation: &BTreeMap<VirtualRegister, VirtualRegister>,
    ) {
        let replace = |val: &mut IrValue| {
            if let IrValue::Register(vreg) = val {
                if let Some(&physical) = allocation.get(vreg) {
                    *val = IrValue::Register(physical);
                }
            }
        };

        match inst {
            Instruction::AddInt(dst, src1, src2)
            | Instruction::Sub(dst, src1, src2)
            | Instruction::Mul(dst, src1, src2)
            | Instruction::Div(dst, src1, src2) => {
                replace(dst);
                replace(src1);
                replace(src2);
            }

            Instruction::Compare(dst, src1, src2, _) => {
                replace(dst);
                replace(src1);
                replace(src2);
            }

            Instruction::Tag(dst, src, _) => {
                replace(dst);
                replace(src);
            }

            Instruction::Untag(dst, src) => {
                replace(dst);
                replace(src);
            }

            Instruction::LoadConstant(dst, _)
            | Instruction::LoadVar(dst, _)
            | Instruction::LoadTrue(dst)
            | Instruction::LoadFalse(dst) => {
                replace(dst);
            }

            Instruction::StoreVar(var_ptr, value) => {
                replace(var_ptr);
                replace(value);
            }

            Instruction::Assign(dst, src) => {
                replace(dst);
                replace(src);
            }

            Instruction::JumpIf(_, _, val1, val2) => {
                replace(val1);
                replace(val2);
            }

            Instruction::Ret(val) => {
                replace(val);
            }

            Instruction::Label(_) | Instruction::Jump(_) => {
                // No registers to replace
            }

            Instruction::PushBinding(var_ptr, value) => {
                replace(var_ptr);
                replace(value);
            }

            Instruction::PopBinding(var_ptr) => {
                replace(var_ptr);
            }

            Instruction::SetVar(var_ptr, value) => {
                replace(var_ptr);
                replace(value);
            }
        }
    }

    /// Get the allocated instructions (consumes self)
    pub fn finish(self) -> Vec<Instruction> {
        self.instructions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::IrBuilder;

    #[test]
    fn test_linear_scan_simple() {
        let mut builder = IrBuilder::new();

        let r0 = builder.new_register();
        let r1 = builder.new_register();
        let r2 = builder.new_register();

        builder.emit(Instruction::LoadConstant(r0, IrValue::TaggedConstant(8)));
        builder.emit(Instruction::LoadConstant(r1, IrValue::TaggedConstant(16)));
        builder.emit(Instruction::Untag(r0, r0));
        builder.emit(Instruction::Untag(r1, r1));
        builder.emit(Instruction::AddInt(r2, r0, r1));
        builder.emit(Instruction::Tag(r2, r2, IrValue::TaggedConstant(0)));
        builder.emit(Instruction::Ret(r2));

        let instructions = builder.finish();
        let mut allocator = LinearScan::new(instructions, 0);
        allocator.allocate();

        let allocated = allocator.finish();

        println!("\nAllocated instructions:");
        for (i, inst) in allocated.iter().enumerate() {
            println!("  {}: {:?}", i, inst);
        }

        // Verify all registers are now physical (x19-x28)
        for inst in &allocated {
            for reg in LinearScan::get_registers_in_instruction(inst) {
                assert!(reg.index >= 19 && reg.index <= 28,
                    "Register {} should be physical (19-28)", reg.index);
            }
        }
    }

    #[test]
    fn test_linear_scan_many_registers() {
        let mut builder = IrBuilder::new();

        // Create more registers than we have physical ones (10 available: x19-x28)
        let mut regs = Vec::new();
        for _ in 0..12 {
            regs.push(builder.new_register());
        }

        // Use all registers
        for reg in &regs {
            builder.emit(Instruction::LoadConstant(*reg, IrValue::TaggedConstant(42)));
        }

        // Add them all together
        let mut sum = regs[0];
        for reg in &regs[1..] {
            let next_sum = builder.new_register();
            builder.emit(Instruction::AddInt(next_sum, sum, *reg));
            sum = next_sum;
        }

        builder.emit(Instruction::Ret(sum));

        let instructions = builder.finish();
        let mut allocator = LinearScan::new(instructions, 0);
        allocator.allocate();

        println!("\nSpill locations: {:?}", allocator.spill_locations);
        println!("Allocated registers: {}", allocator.allocated_registers.len());

        // Some registers should be spilled
        assert!(allocator.spill_locations.len() > 0, "Should have spilled some registers");
    }
}
