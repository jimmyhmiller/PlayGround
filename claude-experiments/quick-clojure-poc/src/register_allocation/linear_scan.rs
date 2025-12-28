#![allow(dead_code)]
use std::collections::{BTreeMap, HashMap};

use crate::ir::{CallTarget, Instruction, IrValue, VirtualRegister};

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

    /// Result register (protected from spilling)
    result_register: Option<VirtualRegister>,
}

/// Create a physical register (x19-x28 per ARM64 calling convention)
fn physical(index: usize) -> VirtualRegister {
    VirtualRegister::Temp(index)
}

impl LinearScan {
    /// Create a new linear scan allocator
    ///
    /// instructions: IR instructions to allocate registers for
    /// max_registers: maximum number of physical registers (0 = default 10)
    pub fn new(instructions: Vec<Instruction>, max_registers: usize) -> Self {
        let lifetimes = Self::compute_lifetimes(&instructions);

        // Determine number of registers to use
        let max_registers = if max_registers == 0 {
            10
        } else {
            max_registers
        };

        // Use ARM64 registers x19-x28 (preserved via CallWithSaves, not prologue)
        let physical_registers: Vec<VirtualRegister> =
            (19..(19 + max_registers)).map(physical).collect();

        LinearScan {
            lifetimes,
            instructions,
            allocated_registers: BTreeMap::new(),
            free_registers: physical_registers,
            max_registers,
            spill_locations: HashMap::new(),
            next_stack_slot: 0, // Start stack slots at 0
            result_register: None,
        }
    }

    /// Mark a register as live until the end (for result registers)
    /// Also protects this register from being spilled.
    pub fn mark_live_until_end(&mut self, register: VirtualRegister) {
        let end_index = self.instructions.len().saturating_sub(1);
        if let Some((start, _)) = self.lifetimes.get(&register) {
            self.lifetimes.insert(register, (*start, end_index));
        } else {
            // Register not seen - make it live for the whole function
            self.lifetimes.insert(register, (0, end_index));
        }
        // Mark as the result register (protected from spilling)
        self.result_register = Some(register);
    }

    /// Compute lifetime intervals for all virtual registers
    ///
    /// A register's lifetime is from its first definition to its last use.
    /// Special handling for loops: registers used within a loop body must have
    /// their lifetime extended to cover the entire loop (including back edges).
    fn compute_lifetimes(instructions: &[Instruction]) -> HashMap<VirtualRegister, (usize, usize)> {
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

        // Argument registers are live from function entry (instruction 0),
        // even if their first explicit use is later. This ensures they're
        // properly saved across any calls that precede their first use.
        for (reg, (start, _end)) in result.iter_mut() {
            if let VirtualRegister::Argument(_) = reg
                && *start > 0 {
                    *start = 0;
                }
        }

        // IMPORTANT: Extend lifetimes for registers used within loops
        // If a register is used inside a loop (between label and back-edge jump),
        // its lifetime must extend to the jump instruction to ensure
        // the physical register isn't reused before we jump back.
        // A back-edge is a Jump to a label that appears BEFORE the jump.
        for (jump_idx, instruction) in instructions.iter().enumerate() {
            if let Instruction::Jump(label) = instruction {
                // Find the target label index
                if let Some(label_idx) = instructions
                    .iter()
                    .position(|instr| matches!(instr, Instruction::Label(l) if l == label))
                {
                    // Only process back-edges (jumps to earlier labels = loops)
                    if label_idx < jump_idx {
                        // Extend lifetime of any register used between label and jump
                        for (_reg, (start, end)) in result.iter_mut() {
                            // If register is defined before or at jump, and used within the loop
                            // (end >= label_idx), extend its lifetime to cover the jump
                            if *start <= jump_idx && *end >= label_idx && *end < jump_idx {
                                *end = jump_idx;
                            }
                        }
                    }
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
            | Instruction::Div(dst, src1, src2)
            | Instruction::AddFloat(dst, src1, src2)
            | Instruction::SubFloat(dst, src1, src2)
            | Instruction::MulFloat(dst, src1, src2)
            | Instruction::DivFloat(dst, src1, src2)
            | Instruction::BitAnd(dst, src1, src2)
            | Instruction::BitOr(dst, src1, src2)
            | Instruction::BitXor(dst, src1, src2)
            | Instruction::BitShiftLeft(dst, src1, src2)
            | Instruction::BitShiftRight(dst, src1, src2)
            | Instruction::UnsignedBitShiftRight(dst, src1, src2) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
                if let IrValue::Register(r) = src1 {
                    regs.push(*r);
                }
                if let IrValue::Register(r) = src2 {
                    regs.push(*r);
                }
            }

            Instruction::IntToFloat(dst, src)
            | Instruction::GetTag(dst, src)
            | Instruction::LoadFloat(dst, src)
            | Instruction::BitNot(dst, src)
            | Instruction::Untag(dst, src) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
                if let IrValue::Register(r) = src {
                    regs.push(*r);
                }
            }

            Instruction::HeapStore(ptr, _, value) => {
                if let IrValue::Register(r) = ptr {
                    regs.push(*r);
                }
                if let IrValue::Register(r) = value {
                    regs.push(*r);
                }
            }

            Instruction::Compare(dst, src1, src2, _) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
                if let IrValue::Register(r) = src1 {
                    regs.push(*r);
                }
                if let IrValue::Register(r) = src2 {
                    regs.push(*r);
                }
            }

            Instruction::Tag(dst, src, _tag) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
                if let IrValue::Register(r) = src {
                    regs.push(*r);
                }
            }

            // Note: Untag is handled above with the other dst/src instructions
            Instruction::LoadConstant(dst, _)
            | Instruction::LoadTrue(dst)
            | Instruction::LoadFalse(dst) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
            }

            // Note: LoadVarBySymbol, LoadVarBySymbolDynamic, StoreVarBySymbol, EnsureVarBySymbol,
            // and LoadKeyword are now builtin function calls handled by Call instruction
            Instruction::Assign(dst, src) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
                if let IrValue::Register(r) = src {
                    regs.push(*r);
                }
            }

            Instruction::JumpIf(_, _, val1, val2) => {
                if let IrValue::Register(r) = val1 {
                    regs.push(*r);
                }
                if let IrValue::Register(r) = val2 {
                    regs.push(*r);
                }
            }

            Instruction::Ret(val) => {
                if let IrValue::Register(r) = val {
                    regs.push(*r);
                }
            }

            Instruction::Label(_) | Instruction::Jump(_) => {
                // No registers
            }

            Instruction::MakeFunctionPtr(dst, _code_ptr, closure_values) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
                for val in closure_values {
                    if let IrValue::Register(r) = val {
                        regs.push(*r);
                    }
                }
            }

            Instruction::LoadClosure(dst, fn_obj, _index) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
                if let IrValue::Register(r) = fn_obj {
                    regs.push(*r);
                }
            }

            Instruction::Call(dst, fn_val, args) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
                if let IrValue::Register(r) = fn_val {
                    regs.push(*r);
                }
                for arg in args {
                    if let IrValue::Register(r) = arg {
                        regs.push(*r);
                    }
                }
            }

            Instruction::CallWithSaves(dst, target, args, saves) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
                // Extract registers from CallTarget
                match target {
                    CallTarget::External(_) => {} // No registers in address
                    CallTarget::Dynamic(fn_val) => {
                        if let IrValue::Register(r) = fn_val {
                            regs.push(*r);
                        }
                    }
                }
                for arg in args {
                    if let IrValue::Register(r) = arg {
                        regs.push(*r);
                    }
                }
                for save in saves {
                    if let IrValue::Register(r) = save {
                        regs.push(*r);
                    }
                }
            }

            // NOTE: MakeType/MakeTypeWithSaves have been refactored out.
            // Deftype construction now uses ExternalCall + HeapStore + Tag.

            // NOTE: LoadTypeField and StoreTypeField have been refactored out.
            // Field access now uses ExternalCall with pre-interned symbol IDs.

            // NOTE: Println removed - now uses ExternalCall

            // Note: LoadKeyword is now a builtin function call handled by Call instruction
            Instruction::PushExceptionHandler(_label, _slot) => {
                // No registers - uses pre-allocated stack slot
            }

            Instruction::PopExceptionHandler => {
                // No registers
            }

            Instruction::Throw(exc) => {
                if let IrValue::Register(r) = exc {
                    regs.push(*r);
                }
            }

            Instruction::LoadExceptionLocal(dest, _label) => {
                if let IrValue::Register(r) = dest {
                    regs.push(*r);
                }
            }

            // External calls (trampolines) - transformed to CallWithSaves by register allocator
            Instruction::ExternalCall(dst, _func_addr, args) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
                for arg in args {
                    if let IrValue::Register(r) = arg {
                        regs.push(*r);
                    }
                }
            }

            // Multi-arity function instructions
            Instruction::MakeMultiArityFn(dst, _arities, _variadic_min, _variadic_index, closure_values) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
                for val in closure_values {
                    if let IrValue::Register(r) = val {
                        regs.push(*r);
                    }
                }
            }

            Instruction::LoadClosureMultiArity(dst, fn_obj, _arity_count, _index) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
                if let IrValue::Register(r) = fn_obj {
                    regs.push(*r);
                }
            }

            Instruction::CollectRestArgs(dst, _fixed_count, _param_offset) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
            }

            // Assertion instructions
            Instruction::AssertPre(cond, _index) => {
                if let IrValue::Register(r) = cond {
                    regs.push(*r);
                }
            }

            Instruction::AssertPost(cond, _index) => {
                if let IrValue::Register(r) = cond {
                    regs.push(*r);
                }
            }

            // Debug instruction - no registers
            Instruction::Breakpoint => {}

            // Local variable operations - StoreLocal stores to stack, LoadLocal loads from stack
            Instruction::StoreLocal(_slot, value) => {
                if let IrValue::Register(r) = value {
                    regs.push(*r);
                }
            }
            Instruction::LoadLocal(dst, _slot) => {
                if let IrValue::Register(r) = dst {
                    regs.push(*r);
                }
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
        // Debug output disabled
        // eprintln!("DEBUG LinearScan: {} virtual registers, {} physical registers available",
        //           self.lifetimes.len(), self.max_registers);

        // PRE-ALLOCATE ARGUMENT REGISTERS
        // ARM64 calling convention: arguments are passed in x0-x7
        // Virtual registers with Argument variant should map directly to their index
        // eprintln!("DEBUG: Pre-allocation phase - checking {} virtual registers", self.lifetimes.len());
        for vreg in self.lifetimes.keys() {
            if let VirtualRegister::Argument(n) = vreg {
                if *n <= 7 {
                    // Map argument virtual register to corresponding physical register (x0-x7)
                    // Physical registers are represented as Temp(index) since they're not function args
                    let physical_reg = VirtualRegister::Temp(*n);
                    self.allocated_registers.insert(*vreg, physical_reg);
                    // eprintln!("DEBUG: Pre-allocated argument register v_arg{} -> x{}", n, n);
                } else {
                    // eprintln!("DEBUG: Argument register v_arg{} NOT pre-allocated (index > 7, will use stack)", n);
                }
            }
        }

        // DEBUG: Check allocation map IMMEDIATELY after pre-allocation
        // eprintln!("DEBUG: Allocation map AFTER pre-allocation ({} entries):", self.allocated_registers.len());
        // for (vreg, physical) in &self.allocated_registers {
        //     eprintln!("DEBUG:   {} -> x{}", vreg.display_name(), physical.index());
        // }

        // Create sorted list of intervals (start, end, register)
        let mut intervals: Vec<(usize, usize, VirtualRegister)> = self
            .lifetimes
            .iter()
            .filter(|(vreg, _)| !matches!(vreg, VirtualRegister::Argument(_))) // Skip argument registers - already allocated
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
                    panic!(
                        "No free registers available! active={}, max={}",
                        active.len(),
                        self.max_registers
                    );
                }
            }
        }

        // DEBUG: Check allocation map AFTER all allocation (disabled)
        // eprintln!("DEBUG: Allocation map AFTER all allocation ({} entries):", self.allocated_registers.len());
        // for (vreg, physical) in &self.allocated_registers {
        //     eprintln!("DEBUG:   {} -> x{}", vreg.display_name(), physical.index());
        // }

        // Replace spilled registers with Spill values
        self.replace_spilled_registers();

        // DON'T replace allocated registers in the IR!
        // This causes namespace collisions when physical registers (Temp(N)) conflict
        // with virtual registers (also Temp(N)). The code generator will look up
        // each virtual register in register_map instead.
        // self.replace_allocated_registers();

        // Debug output disabled
        // eprintln!("\nFinal allocation:");
        // eprintln!("Spilled: {:?}", self.spill_locations);
        // eprintln!("Allocated: {:?}", self.allocated_registers);
    }

    /// Replace spilled registers with Spill IR values
    fn replace_spilled_registers(&mut self) {
        for inst in self.instructions.iter_mut() {
            Self::replace_spilled_in_instruction(inst, &self.spill_locations);
        }
    }

    /// Replace allocated registers with physical registers
    fn replace_allocated_registers(&mut self) {
        for inst in self.instructions.iter_mut() {
            Self::replace_registers_in_instruction(inst, &self.allocated_registers);
        }
    }

    /// Replace spilled registers in an instruction with Spill values
    fn replace_spilled_in_instruction(
        inst: &mut Instruction,
        spill_locations: &HashMap<VirtualRegister, usize>,
    ) {
        let replace = |val: &mut IrValue| {
            if let IrValue::Register(vreg) = val
                && let Some(&stack_offset) = spill_locations.get(vreg)
            {
                *val = IrValue::Spill(*vreg, stack_offset);
            }
        };

        match inst {
            Instruction::AddInt(dst, src1, src2)
            | Instruction::Sub(dst, src1, src2)
            | Instruction::Mul(dst, src1, src2)
            | Instruction::Div(dst, src1, src2)
            | Instruction::AddFloat(dst, src1, src2)
            | Instruction::SubFloat(dst, src1, src2)
            | Instruction::MulFloat(dst, src1, src2)
            | Instruction::DivFloat(dst, src1, src2)
            | Instruction::BitAnd(dst, src1, src2)
            | Instruction::BitOr(dst, src1, src2)
            | Instruction::BitXor(dst, src1, src2)
            | Instruction::BitShiftLeft(dst, src1, src2)
            | Instruction::BitShiftRight(dst, src1, src2)
            | Instruction::UnsignedBitShiftRight(dst, src1, src2) => {
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

            Instruction::Untag(dst, src)
            | Instruction::IntToFloat(dst, src)
            | Instruction::GetTag(dst, src)
            | Instruction::LoadFloat(dst, src)
            | Instruction::BitNot(dst, src) => {
                replace(dst);
                replace(src);
            }

            Instruction::HeapStore(ptr, _, value) => {
                replace(ptr);
                replace(value);
            }

            Instruction::LoadConstant(dst, _)
            | Instruction::LoadTrue(dst)
            | Instruction::LoadFalse(dst) => {
                replace(dst);
            }

            // Note: LoadVarBySymbol, LoadVarBySymbolDynamic, StoreVarBySymbol, EnsureVarBySymbol,
            // and LoadKeyword are now builtin function calls handled by Call instruction
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
                // No registers
            }

            Instruction::MakeFunctionPtr(dst, _code_ptr, closure_values) => {
                replace(dst);
                for val in closure_values {
                    replace(val);
                }
            }

            Instruction::LoadClosure(dst, fn_obj, _index) => {
                replace(dst);
                replace(fn_obj);
            }

            Instruction::Call(dst, fn_val, args) => {
                replace(dst);
                replace(fn_val);
                for arg in args {
                    replace(arg);
                }
            }

            Instruction::CallWithSaves(dst, target, args, saves) => {
                replace(dst);
                // Replace registers in CallTarget
                match target {
                    CallTarget::External(_) => {} // No registers in address
                    CallTarget::Dynamic(fn_val) => {
                        replace(fn_val);
                    }
                }
                for arg in args {
                    replace(arg);
                }
                for save in saves {
                    replace(save);
                }
            }

            // NOTE: MakeType/MakeTypeWithSaves removed - now use ExternalCall + HeapStore + Tag
            // NOTE: LoadTypeField/StoreTypeField removed - now use ExternalCall with symbol IDs

            // NOTE: Println removed - now uses ExternalCall

            // Note: LoadKeyword is now a builtin function call handled by Call instruction
            Instruction::PushExceptionHandler(_label, _slot) => {
                // No registers - uses pre-allocated stack slot
            }

            Instruction::PopExceptionHandler => {
                // No registers
            }

            Instruction::Throw(exc) => {
                replace(exc);
            }

            Instruction::LoadExceptionLocal(dest, _slot) => {
                replace(dest);
            }

            // External calls (trampolines)
            Instruction::ExternalCall(dst, _func_addr, args) => {
                replace(dst);
                for arg in args {
                    replace(arg);
                }
            }

            // Multi-arity function instructions
            Instruction::MakeMultiArityFn(dst, _arities, _variadic_min, _variadic_index, closure_values) => {
                replace(dst);
                for val in closure_values {
                    replace(val);
                }
            }

            Instruction::LoadClosureMultiArity(dst, fn_obj, _arity_count, _index) => {
                replace(dst);
                replace(fn_obj);
            }

            Instruction::CollectRestArgs(dst, _fixed_count, _param_offset) => {
                replace(dst);
            }

            // Assertion instructions
            Instruction::AssertPre(cond, _index) => {
                replace(cond);
            }

            Instruction::AssertPost(cond, _index) => {
                replace(cond);
            }

            // Debug instruction - no registers
            Instruction::Breakpoint => {}

            // Local variable operations
            Instruction::StoreLocal(_slot, value) => {
                replace(value);
            }
            Instruction::LoadLocal(dst, _slot) => {
                replace(dst);
            }
        }
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
        // Keep removing element 0 until we hit an active interval
        while !active.is_empty() {
            let (_, end, vreg) = active[0];

            if end >= current_start {
                // This interval is still active
                break;
            }

            // Interval has expired - free its register
            // IMPORTANT: We free the physical register but DON'T remove from allocated_registers!
            // The allocated_registers map is used by code generation to look up physical registers.
            // Removing expired registers would break lookups for non-overlapping intervals that
            // share the same physical register.
            if let Some(&physical_reg) = self.allocated_registers.get(&vreg) {
                self.free_register(physical_reg);
                // DON'T remove from allocated_registers - needed for code generation lookups!
            }

            active.remove(0);
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

        // Check if current interval is the result register (protected from spilling)
        let current_is_result = self.result_register == Some(vreg);

        // Get the interval that ends last (spill candidate)
        // IMPORTANT: Never spill argument registers (x0-x7)! They're reserved for calling convention
        // Also never spill the result register!
        let spill_candidate_index = active.iter().rposition(|(_, _, v)| {
            !matches!(v, VirtualRegister::Argument(_)) && self.result_register != Some(*v)
        });

        if spill_candidate_index.is_none() {
            // All active registers are argument registers or the result register
            if current_is_result {
                // Can't spill the result register - this is a register allocation failure
                // In practice, we should try to spill something else or increase register count
                panic!(
                    "Cannot allocate result register: all physical registers are in use by protected registers"
                );
            }
            // Spill current interval instead
            let stack_slot = self.allocate_stack_slot();
            // eprintln!("DEBUG LinearScan: Spilling {} to stack slot {} (all active are protected)", vreg.display_name(), stack_slot);
            self.spill_locations.insert(vreg, stack_slot);
            return;
        }

        let spill = active[spill_candidate_index.unwrap()];
        let (_, spill_end, spill_vreg) = spill;

        // If current interval is the result register, always prefer spilling something else
        if current_is_result || spill_end > end {
            // Spill the interval that ends last, allocate its register to current interval
            let physical_reg = *self.allocated_registers.get(&spill_vreg).unwrap();

            // Remove the spilled register from allocated_registers and mark it as spilled
            self.allocated_registers.remove(&spill_vreg);
            let stack_slot = self.allocate_stack_slot();
            // eprintln!("DEBUG LinearScan: Spilling {} to stack slot {}", spill_vreg.display_name(), stack_slot);
            self.spill_locations.insert(spill_vreg, stack_slot);

            // Allocate the freed physical register to current interval
            self.allocated_registers.insert(vreg, physical_reg);

            // Remove from active and add current interval
            active.retain(|x| *x != spill);
            active.push((_start, end, vreg));
            active.sort_by_key(|(_, end, _)| *end);
        } else {
            // Current interval is shorter, spill it instead
            let stack_slot = self.allocate_stack_slot();
            // eprintln!("DEBUG LinearScan: Spilling {} to stack slot {} (short interval)", vreg.display_name(), stack_slot);
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

    /// Replace virtual registers with allocated physical registers in an instruction
    fn replace_registers_in_instruction(
        inst: &mut Instruction,
        allocation: &BTreeMap<VirtualRegister, VirtualRegister>,
    ) {
        let replace = |val: &mut IrValue| {
            if let IrValue::Register(vreg) = val {
                // IMPORTANT: Don't replace Argument registers!
                // They're already physical registers (x0-x7) and should not be looked up in the allocation map.
                // The allocator maps Argument(n) -> Temp(n), but Temp(n) might be a virtual register
                // with a different allocation, causing wrong physical register lookups.
                if matches!(vreg, VirtualRegister::Argument(_)) {
                    return; // Keep Argument registers as-is
                }
                if let Some(&physical) = allocation.get(vreg) {
                    *val = IrValue::Register(physical);
                }
            }
        };

        match inst {
            Instruction::AddInt(dst, src1, src2)
            | Instruction::Sub(dst, src1, src2)
            | Instruction::Mul(dst, src1, src2)
            | Instruction::Div(dst, src1, src2)
            | Instruction::AddFloat(dst, src1, src2)
            | Instruction::SubFloat(dst, src1, src2)
            | Instruction::MulFloat(dst, src1, src2)
            | Instruction::DivFloat(dst, src1, src2)
            | Instruction::BitAnd(dst, src1, src2)
            | Instruction::BitOr(dst, src1, src2)
            | Instruction::BitXor(dst, src1, src2)
            | Instruction::BitShiftLeft(dst, src1, src2)
            | Instruction::BitShiftRight(dst, src1, src2)
            | Instruction::UnsignedBitShiftRight(dst, src1, src2) => {
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

            Instruction::Untag(dst, src)
            | Instruction::IntToFloat(dst, src)
            | Instruction::GetTag(dst, src)
            | Instruction::LoadFloat(dst, src)
            | Instruction::BitNot(dst, src) => {
                replace(dst);
                replace(src);
            }

            Instruction::HeapStore(ptr, _, value) => {
                replace(ptr);
                replace(value);
            }

            Instruction::LoadConstant(dst, _)
            | Instruction::LoadTrue(dst)
            | Instruction::LoadFalse(dst) => {
                replace(dst);
            }

            // Note: LoadVarBySymbol, LoadVarBySymbolDynamic, StoreVarBySymbol, EnsureVarBySymbol,
            // and LoadKeyword are now builtin function calls handled by Call instruction
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

            Instruction::MakeFunctionPtr(dst, _code_ptr, closure_values) => {
                replace(dst);
                for val in closure_values {
                    replace(val);
                }
            }

            Instruction::LoadClosure(dst, fn_obj, _index) => {
                replace(dst);
                replace(fn_obj);
            }

            Instruction::Call(dst, fn_val, args) => {
                replace(dst);
                replace(fn_val);
                for arg in args {
                    replace(arg);
                }
            }

            Instruction::CallWithSaves(dst, target, args, saves) => {
                replace(dst);
                // Replace registers in CallTarget
                match target {
                    CallTarget::External(_) => {} // No registers in address
                    CallTarget::Dynamic(fn_val) => {
                        replace(fn_val);
                    }
                }
                for arg in args {
                    replace(arg);
                }
                for save in saves {
                    replace(save);
                }
            }

            // NOTE: MakeType/MakeTypeWithSaves removed - now use ExternalCall + HeapStore + Tag
            // NOTE: LoadTypeField/StoreTypeField removed - now use ExternalCall with symbol IDs

            // NOTE: Println removed - now uses ExternalCall

            // Note: LoadKeyword is now a builtin function call handled by Call instruction
            Instruction::PushExceptionHandler(_label, _slot) => {
                // No registers - uses pre-allocated stack slot
            }

            Instruction::PopExceptionHandler => {
                // No registers
            }

            Instruction::Throw(exc) => {
                replace(exc);
            }

            Instruction::LoadExceptionLocal(dest, _slot) => {
                replace(dest);
            }

            // External calls (trampolines)
            Instruction::ExternalCall(dst, _func_addr, args) => {
                replace(dst);
                for arg in args {
                    replace(arg);
                }
            }

            // Multi-arity function instructions
            Instruction::MakeMultiArityFn(dst, _arities, _variadic_min, _variadic_index, closure_values) => {
                replace(dst);
                for val in closure_values {
                    replace(val);
                }
            }

            Instruction::LoadClosureMultiArity(dst, fn_obj, _arity_count, _index) => {
                replace(dst);
                replace(fn_obj);
            }

            Instruction::CollectRestArgs(dst, _fixed_count, _param_offset) => {
                replace(dst);
            }

            // Assertion instructions
            Instruction::AssertPre(cond, _index) => {
                replace(cond);
            }

            Instruction::AssertPost(cond, _index) => {
                replace(cond);
            }

            // Local variable operations
            Instruction::StoreLocal(_slot, value) => {
                replace(value);
            }
            Instruction::LoadLocal(dst, _slot) => {
                replace(dst);
            }

            // Debug instruction - no registers
            Instruction::Breakpoint => {}
        }
    }

    /// Transform Call and ExternalCall instructions to their WithSaves variants
    ///
    /// This analyzes which registers are live across each call site and generates
    /// CallWithSaves instructions with explicit register preservation.
    /// NOTE: Recur is no longer handled here - it's lowered to Assign + Jump in the compiler.
    fn transform_calls_to_saves(&mut self) {
        for i in 0..self.instructions.len() {
            // Check if this is a Call or ExternalCall instruction
            let is_call = matches!(self.instructions[i], Instruction::Call(_, _, _));
            let is_external_call =
                matches!(self.instructions[i], Instruction::ExternalCall(_, _, _));

            if !is_call && !is_external_call {
                continue;
            }

            // Extract destination register for exclusion from saves
            let dest = match &self.instructions[i] {
                Instruction::Call(d, _, _) => *d,
                Instruction::ExternalCall(d, _, _) => *d,
                _ => continue,
            };

            // Find registers live across this call
            let mut saves = Vec::new();

            for (vreg, (start, end)) in &self.lifetimes {
                // For Call/ExternalCall: standard liveness check
                // Register is live across if it starts before and ends after
                let is_live_across =
                    *start <= i && *end > i && !self.spill_locations.contains_key(vreg);

                if is_live_across {
                    // Get the physical register
                    if let Some(&physical_reg) = self.allocated_registers.get(vreg) {
                        // Don't save the destination register (it's about to be overwritten)
                        let is_dest = match &dest {
                            IrValue::Register(dest_vreg) => self
                                .allocated_registers
                                .get(dest_vreg)
                                .map(|&dest_phys| dest_phys == physical_reg)
                                .unwrap_or(false),
                            _ => false,
                        };

                        if !is_dest {
                            // IMPORTANT: Push the VIRTUAL register, not the physical register!
                            // The codegen will look up the physical register from the allocation map.
                            // If we pushed the physical register (e.g., Temp(1) meaning x1), it would
                            // get confused with any IR virtual register that happens to have the same name.
                            saves.push(IrValue::Register(*vreg));
                        }
                    }
                }
            }

            // Remove duplicates - sort by physical register and dedup
            // Multiple virtual registers might map to the same physical register
            saves.sort_by_key(|v| match v {
                IrValue::Register(vreg) => self
                    .allocated_registers
                    .get(vreg)
                    .map(|phys| phys.index())
                    .unwrap_or(vreg.index()),
                _ => 0,
            });
            // Dedup by physical register, not by virtual register
            let mut seen_physical = std::collections::HashSet::new();
            saves.retain(|v| match v {
                IrValue::Register(vreg) => {
                    let phys_idx = self
                        .allocated_registers
                        .get(vreg)
                        .map(|phys| phys.index())
                        .unwrap_or(vreg.index());
                    seen_physical.insert(phys_idx)
                }
                _ => true,
            });

            // Note: saves are always Registers, never Spills
            // (line 1205 already excludes spilled values via !self.spill_locations.contains_key)

            // Transform to unified CallWithSaves with appropriate CallTarget
            if is_call {
                let (func, args) = if let Instruction::Call(_, f, a) = &self.instructions[i] {
                    (*f, a.clone())
                } else {
                    continue;
                };
                self.instructions[i] =
                    Instruction::CallWithSaves(dest, CallTarget::Dynamic(func), args, saves);
            } else if is_external_call {
                let (func_addr, args) =
                    if let Instruction::ExternalCall(_, addr, a) = &self.instructions[i] {
                        (*addr, a.clone())
                    } else {
                        continue;
                    };
                self.instructions[i] =
                    Instruction::CallWithSaves(dest, CallTarget::External(func_addr), args, saves);
            }
        }
    }

    /// Get the number of stack slots used for spilling
    pub fn num_stack_slots(&self) -> usize {
        self.next_stack_slot
    }

    /// Get the allocated instructions (consumes self)
    pub fn finish(mut self) -> Vec<Instruction> {
        // Transform Call instructions to CallWithSaves after register allocation
        self.transform_calls_to_saves();

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

        let instructions = builder.take_instructions();
        let mut allocator = LinearScan::new(instructions, 0);
        allocator.allocate();

        // Note: Register replacement in IR is disabled (see replace_allocated_registers comment).
        // The code generator looks up virtual registers in allocated_registers map instead.
        // Verify that all virtual registers have been assigned physical registers in the map.
        for inst in &allocator.instructions {
            for reg in LinearScan::get_registers_in_instruction(&inst) {
                // Either the register should be in allocated_registers, or it should be spilled
                let is_allocated = allocator.allocated_registers.contains_key(&reg);
                let is_spilled = allocator.spill_locations.contains_key(&reg);
                assert!(
                    is_allocated || is_spilled,
                    "Register {} should be either allocated or spilled",
                    reg.display_name()
                );

                // If allocated, verify it maps to a physical register (x19-x28)
                if let Some(physical) = allocator.allocated_registers.get(&reg) {
                    let idx = physical.index();
                    assert!(
                        idx >= 19 && idx <= 28,
                        "Register {} mapped to {} which is not physical (19-28)",
                        reg.display_name(),
                        physical.display_name()
                    );
                }
            }
        }

        let allocated = allocator.finish();

        println!("\nAllocated instructions:");
        for (i, inst) in allocated.iter().enumerate() {
            println!("  {}: {:?}", i, inst);
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

        let instructions = builder.take_instructions();
        let mut allocator = LinearScan::new(instructions, 0);
        allocator.allocate();

        println!("\nSpill locations: {:?}", allocator.spill_locations);
        println!(
            "Allocated registers: {}",
            allocator.allocated_registers.len()
        );

        // Some registers should be spilled
        assert!(
            allocator.spill_locations.len() > 0,
            "Should have spilled some registers"
        );
    }
}
