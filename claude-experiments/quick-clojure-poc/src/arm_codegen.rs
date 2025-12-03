use crate::ir::{Instruction, IrValue, VirtualRegister, Condition, Label};
use crate::register_allocation::linear_scan::LinearScan;
use crate::trampoline::Trampoline;
use std::collections::HashMap;

/// ARM64 code generator - compiles IR to ARM64 machine code
///
/// This is based on Beagle's ARM64 backend but simplified for our needs.
pub struct Arm64CodeGen {
    /// Generated ARM64 machine code (32-bit instructions)
    code: Vec<u32>,

    /// Map from virtual registers to physical ARM64 registers (x0-x15)
    register_map: HashMap<VirtualRegister, usize>,

    /// Next physical register to allocate
    next_physical_reg: usize,

    /// Map from labels to code positions (for fixups)
    label_positions: HashMap<Label, usize>,

    /// Pending jump fixups: (code_index, label)
    pending_fixups: Vec<(usize, Label)>,

    /// Pool of temporary registers for spill loads (x9, x10, x11)
    temp_register_pool: Vec<usize>,
}

impl Default for Arm64CodeGen {
    fn default() -> Self {
        Self::new()
    }
}

impl Arm64CodeGen {
    pub fn new() -> Self {
        Arm64CodeGen {
            code: Vec::new(),
            register_map: HashMap::new(),
            next_physical_reg: 0,
            label_positions: HashMap::new(),
            pending_fixups: Vec::new(),
            temp_register_pool: vec![11, 10, 9],  // Start with x11, x10, x9 available
        }
    }

    /// Compile IR instructions to ARM64 machine code
    ///
    /// # Parameters
    /// - `instructions`: IR instructions to compile
    /// - `result_reg`: The register containing the final result
    /// - `num_registers`: Number of registers available (0 = default/unlimited)
    pub fn compile(&mut self, instructions: &[Instruction], result_reg: &IrValue, num_registers: usize) -> Result<Vec<u32>, String> {
        // Reset state
        self.code.clear();
        self.register_map.clear();
        self.next_physical_reg = 0;
        self.label_positions.clear();
        self.pending_fixups.clear();

        // Run linear scan register allocation
        let mut allocator = LinearScan::new(instructions.to_vec(), num_registers);

        // Mark result register as live until the end
        // This is critical - without this, the register allocator may reuse
        // the physical register for the result, causing wrong values to be returned
        if let IrValue::Register(vreg) = result_reg {
            allocator.mark_live_until_end(*vreg);
        }

        allocator.allocate();

        // Debug output BEFORE consuming allocator
        let num_stack_slots = allocator.next_stack_slot;
        let num_spills = allocator.spill_locations.len();
        eprintln!("DEBUG: {} spills, {} total stack slots", num_spills, num_stack_slots);
        eprintln!("DEBUG: Spill locations from codegen allocator:");
        for (vreg, slot) in &allocator.spill_locations {
            eprintln!("  v{} -> slot {}", vreg.index, slot);
        }

        // Find the physical register for the result (before consuming allocator)
        let result_physical = if let IrValue::Register(vreg) = result_reg {
            allocator.allocated_registers.get(vreg)
                .ok_or_else(|| format!("Result register {:?} not allocated", vreg))?
                .index
        } else {
            return Err(format!("Expected register for result, got {:?}", result_reg));
        };

        // Count spills to determine stack space needed
        // Add 8 bytes padding so spills are above SP (ARM64 requirement)
        let stack_space = if num_stack_slots > 0 {
            num_stack_slots * 8 + 8
        } else {
            0
        };

        eprintln!("DEBUG: Allocating {} bytes of stack space", stack_space);

        let allocated_instructions = allocator.finish();

        // Emit function prologue
        // Note: Callee-saved registers (x19-x28) are saved by the trampoline, not here
        // Save FP and LR
        self.emit_stp(29, 30, 31, -2);  // stp x29, x30, [sp, #-16]!
        self.emit_mov(29, 31);           // mov x29, sp (set frame pointer)

        // Allocate stack space for spills if needed
        if stack_space > 0 {
            // sub sp, sp, #stack_space
            self.emit_sub_sp_imm(stack_space as i64);
        }

        // Compile each instruction (now with physical registers)
        for inst in &allocated_instructions {
            self.compile_instruction(inst)?;
        }

        // Apply jump fixups
        self.apply_fixups()?;

        // Move result to x0 (keep it tagged)
        if result_physical != 0 {
            self.emit_mov(0, result_physical);
        }

        // Deallocate stack space for spills if needed
        if stack_space > 0 {
            // add sp, sp, #stack_space
            self.emit_add_sp_imm(stack_space as i64);
        }

        // Emit function epilogue
        // Note: Callee-saved registers (x19-x28) are restored by the trampoline, not here
        // Restore FP and LR
        self.emit_ldp(29, 30, 31, 2);    // ldp x29, x30, [sp], #16

        // Emit return instruction
        self.emit_ret();

        Ok(self.code.clone())
    }

    fn compile_instruction(&mut self, inst: &Instruction) -> Result<(), String> {
        match inst {
            Instruction::LoadConstant(dst, value) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                match value {
                    IrValue::TaggedConstant(c) => {
                        self.emit_mov_imm(dst_reg, *c as i64);
                    }
                    IrValue::True => {
                        // true: (1 << 3) | 0b011 = 11
                        self.emit_mov_imm(dst_reg, 11);
                    }
                    IrValue::False => {
                        // false: (0 << 3) | 0b011 = 3
                        self.emit_mov_imm(dst_reg, 3);
                    }
                    IrValue::Null => {
                        // nil: 0b111 = 7
                        self.emit_mov_imm(dst_reg, 7);
                    }
                    _ => return Err(format!("Invalid constant: {:?}", value)),
                }
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::LoadVar(dst, var_ptr) => {
                // LoadVar: call trampoline to check dynamic bindings
                // ARM64 calling convention:
                // - x0 = argument (var_ptr, tagged)
                // - x0 = return value (tagged)
                // - x30 = link register (return address)
                // - x19-x28 are callee-saved (our allocator uses these, so they're safe)
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;

                match var_ptr {
                    IrValue::TaggedConstant(tagged_ptr) => {
                        // Load tagged var_ptr into x0 (first argument)
                        self.emit_mov_imm(0, *tagged_ptr as i64);

                        // Load trampoline function address into x15
                        let func_addr = crate::trampoline::trampoline_var_get_value_dynamic as usize;
                        self.emit_mov_imm(15, func_addr as i64);

                        // Call the trampoline
                        // BLR x15 - stores return address in x30, jumps to x15
                        self.emit_blr(15);

                        // Result is in x0, move to destination if needed
                        if dst_reg != 0 {
                            self.emit_mov(dst_reg, 0);
                        }
                    }
                    _ => return Err(format!("LoadVar requires constant var pointer: {:?}", var_ptr)),
                }
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::StoreVar(var_ptr, value) => {
                // StoreVar: store value into var at runtime
                // Var layout: [header(8)] [ns_ptr(8)] [symbol_ptr(8)] [value(8)]
                // We want to write to field 2 (value) which is at offset 24 bytes (3 * 8)
                let value_reg = self.get_physical_reg_for_irvalue(value, false)?;

                match var_ptr {
                    IrValue::TaggedConstant(tagged_ptr) => {
                        // Untag the var pointer (shift right by 3)
                        let untagged_ptr = (*tagged_ptr as usize) >> 3;

                        // Load var pointer into a temp register
                        // Use x15 as temp register - it's the highest general purpose register
                        // and unlikely to be allocated by our simple allocator
                        let temp_reg = 15;
                        self.emit_mov_imm(temp_reg, untagged_ptr as i64);

                        // Store value into var (offset 24 = header + ns_ptr + symbol_ptr)
                        // str value_reg, [temp_reg, #24]
                        self.emit_str_offset(value_reg, temp_reg, 24);
                    }
                    _ => return Err(format!("StoreVar requires constant var pointer: {:?}", var_ptr)),
                }
            }

            Instruction::LoadTrue(dst) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                // true: (1 << 3) | 0b011 = 11
                self.emit_mov_imm(dst_reg, 11);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::LoadFalse(dst) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                // false: (0 << 3) | 0b011 = 3
                self.emit_mov_imm(dst_reg, 3);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Untag(dst, src) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                // Untag: arithmetic right shift by 3
                self.emit_asr_imm(dst_reg, src_reg, 3);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Tag(dst, src, _tag) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                // Tag: left shift by 3 (int tag is 000)
                self.emit_lsl_imm(dst_reg, src_reg, 3);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::AddInt(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                self.emit_add(dst_reg, src1_reg, src2_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Sub(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                self.emit_sub(dst_reg, src1_reg, src2_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Mul(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                self.emit_mul(dst_reg, src1_reg, src2_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Div(dst, src1, src2) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;
                self.emit_sdiv(dst_reg, src1_reg, src2_reg);
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Assign(dst, src) => {
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src_reg = self.get_physical_reg_for_irvalue(src, false)?;
                if dst_reg != src_reg {
                    self.emit_mov(dst_reg, src_reg);
                }
                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::PushBinding(var_ptr, value) => {
                // PushBinding: call trampoline to push a dynamic binding
                // ARM64 calling convention:
                // - x0 = var_ptr (tagged)
                // - x1 = value (tagged)
                // - x0 = return value (0 = success, 1 = error)

                match var_ptr {
                    IrValue::TaggedConstant(tagged_ptr) => {
                        // Get the value register
                        let value_reg = self.get_physical_reg_for_irvalue(value, false)?;

                        // Load tagged var_ptr into x0 (first argument)
                        self.emit_mov_imm(0, *tagged_ptr as i64);

                        // Move value to x1 (second argument) if not already there
                        if value_reg != 1 {
                            self.emit_mov(1, value_reg);
                        }

                        // Load trampoline function address into x15
                        let func_addr = crate::trampoline::trampoline_push_binding as usize;
                        self.emit_mov_imm(15, func_addr as i64);

                        // Call the trampoline
                        self.emit_blr(15);

                        // Return value in x0 (0 = success, 1 = error)
                        // For now, we ignore errors (could add error handling later)
                    }
                    _ => return Err(format!("PushBinding requires constant var pointer: {:?}", var_ptr)),
                }
            }

            Instruction::PopBinding(var_ptr) => {
                // PopBinding: call trampoline to pop a dynamic binding
                // ARM64 calling convention:
                // - x0 = var_ptr (tagged)
                // - x0 = return value (0 = success, 1 = error)

                match var_ptr {
                    IrValue::TaggedConstant(tagged_ptr) => {
                        // Load tagged var_ptr into x0 (first argument)
                        self.emit_mov_imm(0, *tagged_ptr as i64);

                        // Load trampoline function address into x15
                        let func_addr = crate::trampoline::trampoline_pop_binding as usize;
                        self.emit_mov_imm(15, func_addr as i64);

                        // Call the trampoline
                        self.emit_blr(15);

                        // Return value in x0 (0 = success, 1 = error)
                        // For now, we ignore errors (could add error handling later)
                    }
                    _ => return Err(format!("PopBinding requires constant var pointer: {:?}", var_ptr)),
                }
            }

            Instruction::SetVar(var_ptr, value) => {
                // SetVar: call trampoline to modify a thread-local binding (for set!)
                // ARM64 calling convention:
                // - x0 = var_ptr (tagged)
                // - x1 = value (tagged)
                // - x0 = return value (0 = success, 1 = error)

                match var_ptr {
                    IrValue::TaggedConstant(tagged_ptr) => {
                        // Get the value register
                        let value_reg = self.get_physical_reg_for_irvalue(value, false)?;

                        // Load tagged var_ptr into x0 (first argument)
                        self.emit_mov_imm(0, *tagged_ptr as i64);

                        // Move value to x1 (second argument) if not already there
                        if value_reg != 1 {
                            self.emit_mov(1, value_reg);
                        }

                        // Load trampoline function address into x15
                        let func_addr = crate::trampoline::trampoline_set_binding as usize;
                        self.emit_mov_imm(15, func_addr as i64);

                        // Call the trampoline
                        self.emit_blr(15);

                        // Return value in x0 (0 = success, 1 = error)
                        // For now, we ignore errors (could add error handling later)
                    }
                    _ => return Err(format!("SetVar requires constant var pointer: {:?}", var_ptr)),
                }
            }

            Instruction::Label(label) => {
                // Record position of this label
                self.label_positions.insert(label.clone(), self.code.len());
            }

            Instruction::Jump(label) => {
                // Emit unconditional branch
                // We'll fix up the offset later
                let fixup_index = self.code.len();
                self.pending_fixups.push((fixup_index, label.clone()));
                // Placeholder - will be patched in apply_fixups
                self.code.push(0x14000000); // B #0
            }

            Instruction::JumpIf(label, cond, src1, src2) => {
                // Compare src1 and src2
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;

                // Emit CMP instruction
                self.emit_cmp(src1_reg, src2_reg);

                // Emit conditional branch
                let fixup_index = self.code.len();
                self.pending_fixups.push((fixup_index, label.clone()));

                // Placeholder conditional branch - will be patched in apply_fixups
                let branch_cond = match cond {
                    Condition::Equal => 0,       // EQ
                    Condition::NotEqual => 1,    // NE
                    Condition::LessThan => 11,   // LT
                    Condition::LessThanOrEqual => 13, // LE
                    Condition::GreaterThan => 12, // GT
                    Condition::GreaterThanOrEqual => 10, // GE
                };

                // B.cond #0 (placeholder)
                self.code.push(0x54000000 | branch_cond);
            }

            Instruction::Compare(dst, src1, src2, cond) => {
                // Compare and set result to true/false (tagged bools: 11 or 3)
                let dest_spill = self.dest_spill(dst);
                let dst_reg = self.get_physical_reg_for_irvalue(dst, true)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1, false)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2, false)?;

                // CMP src1, src2
                self.emit_cmp(src1_reg, src2_reg);

                // CSET dst, condition (sets dst to 1 if condition is true, 0 otherwise)
                let cond_code = match cond {
                    Condition::Equal => 0,       // EQ
                    Condition::NotEqual => 1,    // NE
                    Condition::LessThan => 11,   // LT
                    Condition::LessThanOrEqual => 13, // LE
                    Condition::GreaterThan => 12, // GT
                    Condition::GreaterThanOrEqual => 10, // GE
                };

                // CSET is CSINC dst, XZR, XZR, invert(cond)
                // This sets dst to 1 if true, 0 if false
                let inverted_cond = cond_code ^ 1; // Invert the condition
                let instruction = 0x9A9F07E0 | (inverted_cond << 12) | (dst_reg as u32);
                self.code.push(instruction);

                // Now convert 0/1 to tagged bools: 3 (false) or 11 (true)
                // LSL dst, dst, #3  - Shift left by 3: 0→0, 1→8
                self.emit_lsl_imm(dst_reg, dst_reg, 3);
                // ADD dst, dst, #3  - Add 3: 0→3, 8→11
                self.emit_add_imm(dst_reg, dst_reg, 3);

                self.store_spill(dst_reg, dest_spill);
            }

            Instruction::Ret(value) => {
                // Move result to x0 (return register)
                let src_reg = self.get_physical_reg_for_irvalue(value, false)?;
                if src_reg != 0 {
                    self.emit_mov(0, src_reg);
                }
            }
        }

        // Clear temporary registers after each instruction (like Beagle does)
        self.clear_temp_registers();

        Ok(())
    }

    /// Get physical register for an IR value
    /// If is_dest=true and value is Spill, returns temp register without loading
    /// If is_dest=false and value is Spill, loads from stack into temp register
    fn get_physical_reg_for_irvalue(&mut self, value: &IrValue, is_dest: bool) -> Result<usize, String> {
        match value {
            IrValue::Register(vreg) => {
                Ok(self.get_physical_reg(vreg))
            }
            IrValue::Spill(_vreg, stack_offset) => {
                // Allocate a temporary register from the pool
                let temp_reg = self.allocate_temp_register();

                if !is_dest {
                    // Load spilled value from stack
                    // Stack layout after prologue:
                    //   [FP + 8]:  saved x30 (LR)
                    //   [FP + 0]:  saved x29 (old FP) <- x29 points here
                    //   [FP - 8]:  spill slot 0
                    //   [FP - 16]: spill slot 1
                    //   [FP - 24]: spill slot 2
                    //   ...
                    //   [FP - (N+1)*8]: spill slot N
                    let offset = -((*stack_offset as i32 + 1) * 8);
                    self.emit_load_from_fp(temp_reg, offset);
                }
                // For destination, just return temp_reg without loading
                Ok(temp_reg)
            }
            _ => Err(format!("Expected register or spill, got {:?}", value)),
        }
    }

    fn get_physical_reg(&mut self, vreg: &VirtualRegister) -> usize {
        // After linear scan allocation, all registers are already physical
        // Just return the register index directly
        vreg.index
    }

    /// Check if a destination is a spill and return its stack offset
    fn dest_spill(&self, dest: &IrValue) -> Option<usize> {
        match dest {
            IrValue::Spill(_, stack_offset) => Some(*stack_offset),
            _ => None,
        }
    }

    /// Store a register to its spill location if needed
    fn store_spill(&mut self, src_reg: usize, dest_spill: Option<usize>) {
        if let Some(stack_offset) = dest_spill {
            // Stack layout after prologue:
            //   [FP + 8]:  saved x30 (LR)
            //   [FP + 0]:  saved x29 (old FP) <- x29 points here
            //   [FP - 8]:  spill slot 0
            //   [FP - 16]: spill slot 1
            //   [FP - 24]: spill slot 2
            //   ...
            //   [FP - (N+1)*8]: spill slot N
            //   [FP - stack_space]: SP
            let offset = -((stack_offset as i32 + 1) * 8);
            eprintln!("DEBUG store_spill: slot {} -> offset {}", stack_offset, offset);
            self.emit_store_to_fp(src_reg, offset);
        }
    }

    /// Allocate a temporary register for loading spills
    fn allocate_temp_register(&mut self) -> usize {
        self.temp_register_pool
            .pop()
            .expect("Out of temporary registers! Need to clear temps between instructions")
    }

    /// Reset temporary register pool (called after each instruction)
    fn clear_temp_registers(&mut self) {
        self.temp_register_pool = vec![11, 10, 9];
    }

    fn apply_fixups(&mut self) -> Result<(), String> {
        for (code_index, label) in &self.pending_fixups {
            let target_pos = self.label_positions.get(label)
                .ok_or_else(|| format!("Undefined label: {}", label))?;

            // Calculate offset in instructions (not bytes)
            let offset = (*target_pos as isize) - (*code_index as isize);

            // Check if offset fits in the instruction encoding
            if !(-1048576..=1048575).contains(&offset) {
                return Err(format!("Jump offset too large: {}", offset));
            }

            // Patch the instruction
            let instruction = self.code[*code_index];

            // Check if it's a conditional branch (B.cond) or unconditional branch (B)
            if (instruction & 0xFF000000) == 0x54000000 {
                // B.cond - 19-bit signed offset in bits [23:5]
                let offset_bits = (offset as u32) & 0x7FFFF; // 19 bits
                self.code[*code_index] = (instruction & 0xFF00001F) | (offset_bits << 5);
            } else if (instruction & 0xFC000000) == 0x14000000 {
                // B - 26-bit signed offset in bits [25:0]
                let offset_bits = (offset as u32) & 0x03FFFFFF; // 26 bits
                self.code[*code_index] = (instruction & 0xFC000000) | offset_bits;
            } else {
                return Err(format!("Unknown branch instruction at {}: {:08x}", code_index, instruction));
            }
        }
        Ok(())
    }

    // ARM64 instruction encoding

    fn emit_mov(&mut self, dst: usize, src: usize) {
        // Special handling when either source OR destination is register 31 (SP)
        // Following Beagle's pattern: check both directions
        // ORR treats register 31 as XZR, but we need it as SP
        // Use ADD instruction which properly interprets register 31 as SP
        if dst == 31 || src == 31 {
            self.emit_mov_sp(dst, src);
        } else {
            self.emit_mov_reg(dst, src);
        }
    }

    /// Generate MOV for regular registers (uses ORR)
    /// Based on Beagle's mov_reg pattern
    fn emit_mov_reg(&mut self, dst: usize, src: usize) {
        // MOV is ORR Xd, XZR, Xm
        // This works for normal registers but treats register 31 as XZR
        let instruction = 0xAA0003E0 | ((src as u32) << 16) | (dst as u32);
        self.code.push(instruction);
    }

    /// Generate MOV involving SP (uses ADD with immediate 0)
    /// Based on Beagle's mov_sp pattern
    fn emit_mov_sp(&mut self, dst: usize, src: usize) {
        // ADD Xd, Xn, #0
        // Works for both MOV from SP and MOV to SP
        // ADD instruction properly interprets register 31 as SP, not XZR
        let instruction = 0x910003E0 | ((src as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_mov_imm(&mut self, dst: usize, imm: i64) {
        let imm = imm as u64;  // Treat as unsigned for bitwise ops

        // Extract 16-bit chunks
        let chunk0 = (imm & 0xFFFF) as u32;
        let chunk1 = ((imm >> 16) & 0xFFFF) as u32;
        let chunk2 = ((imm >> 32) & 0xFFFF) as u32;
        let chunk3 = ((imm >> 48) & 0xFFFF) as u32;

        // MOVZ Xd, #chunk0 (always emit this)
        let movz = 0xD2800000 | (chunk0 << 5) | (dst as u32);
        self.code.push(movz);

        // MOVK Xd, #chunk1, LSL #16 (if non-zero)
        if chunk1 != 0 {
            let movk = 0xF2A00000 | (chunk1 << 5) | (dst as u32);
            self.code.push(movk);
        }

        // MOVK Xd, #chunk2, LSL #32 (if non-zero)
        if chunk2 != 0 {
            let movk = 0xF2C00000 | (chunk2 << 5) | (dst as u32);
            self.code.push(movk);
        }

        // MOVK Xd, #chunk3, LSL #48 (if non-zero)
        if chunk3 != 0 {
            let movk = 0xF2E00000 | (chunk3 << 5) | (dst as u32);
            self.code.push(movk);
        }
    }

    fn emit_add(&mut self, dst: usize, src1: usize, src2: usize) {
        // ADD Xd, Xn, Xm
        let instruction = 0x8B000000 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_sub(&mut self, dst: usize, src1: usize, src2: usize) {
        // SUB Xd, Xn, Xm
        let instruction = 0xCB000000 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_sub_sp_imm(&mut self, imm: i64) {
        // SUB sp, sp, #imm
        let instruction = 0xD10003FF | ((imm as u32 & 0xFFF) << 10);
        self.code.push(instruction);
    }

    fn emit_add_sp_imm(&mut self, imm: i64) {
        // ADD sp, sp, #imm
        let instruction = 0x910003FF | ((imm as u32 & 0xFFF) << 10);
        self.code.push(instruction);
    }

    fn emit_add_imm(&mut self, dst: usize, src: usize, imm: i64) {
        // ADD Xd, Xn, #imm
        let instruction = 0x91000000 | ((imm as u32 & 0xFFF) << 10) | ((src as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_mul(&mut self, dst: usize, src1: usize, src2: usize) {
        // MUL Xd, Xn, Xm (MADD with XZR)
        let instruction = 0x9B007C00 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_sdiv(&mut self, dst: usize, src1: usize, src2: usize) {
        // SDIV Xd, Xn, Xm - signed division
        let instruction = 0x9AC00C00 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_lsl_imm(&mut self, dst: usize, src: usize, shift: u32) {
        // LSL Xd, Xn, #shift (logical shift left)
        // This is actually UBFM (Unsigned Bitfield Move)
        // LSL #shift is: UBFM Xd, Xn, #(-shift mod 64), #(63-shift)
        let shift = shift & 0x3F; // 6 bits
        let immr = (64 - shift) & 0x3F;
        let imms = 63 - shift;
        let instruction = 0xD3400000 | (immr << 16) | (imms << 10) | ((src as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_asr_imm(&mut self, dst: usize, src: usize, shift: u32) {
        // ASR Xd, Xn, #shift (arithmetic shift right)
        // This is SBFM (Signed Bitfield Move)
        // ASR #shift is: SBFM Xd, Xn, #shift, #63
        let shift = shift & 0x3F; // 6 bits
        let instruction = 0x9340FC00 | (shift << 16) | ((src as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_cmp(&mut self, src1: usize, src2: usize) {
        // CMP Xn, Xm (compare - this is SUBS XZR, Xn, Xm)
        let instruction = 0xEB00001F | ((src2 as u32) << 16) | ((src1 as u32) << 5);
        self.code.push(instruction);
    }

    fn emit_str_offset(&mut self, src: usize, base: usize, offset: i32) {
        // STR Xt, [Xn, #offset]
        // Offset is in bytes, needs to be divided by 8 for encoding (unsigned 12-bit)
        let offset_scaled = (offset / 8) as u32;
        let instruction = 0xF9000000 | (offset_scaled << 10) | ((base as u32) << 5) | (src as u32);
        self.code.push(instruction);
    }

    fn emit_load_from_fp(&mut self, dst: usize, offset: i32) {
        // LDR Xd, [x29, #offset] with signed offset
        // Using LDUR for signed 9-bit offset
        let offset_bits = (offset as u32) & 0x1FF; // 9-bit signed
        let instruction = 0xF8400000 | (offset_bits << 12) | (29 << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_store_to_fp(&mut self, src: usize, offset: i32) {
        // STR Xt, [x29, #offset] with signed offset
        // Using STUR for signed 9-bit offset
        let offset_bits = (offset as u32) & 0x1FF; // 9-bit signed
        let instruction = 0xF8000000 | (offset_bits << 12) | (29 << 5) | (src as u32);
        eprintln!("DEBUG emit_store_to_fp: offset={}, offset_bits={:03x}, instruction={:08x}",
                  offset, offset_bits, instruction);
        self.code.push(instruction);
    }

    fn emit_stp(&mut self, rt: usize, rt2: usize, rn: usize, offset: i32) {
        // STP Xt, Xt2, [Xn, #offset]! (pre-index)
        // offset is in 8-byte units for STP, range -512 to 504
        let offset_scaled = ((offset & 0x7F) as u32) << 15;  // 7-bit signed offset
        let instruction = 0xA9800000 | offset_scaled | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32);
        self.code.push(instruction);
    }

    fn emit_ldp(&mut self, rt: usize, rt2: usize, rn: usize, offset: i32) {
        // LDP Xt, Xt2, [Xn], #offset (post-index)
        // offset is in 8-byte units for LDP, range -512 to 504
        let offset_scaled = ((offset & 0x7F) as u32) << 15;  // 7-bit signed offset
        let instruction = 0xA8C00000 | offset_scaled | ((rt2 as u32) << 10) | ((rn as u32) << 5) | (rt as u32);
        self.code.push(instruction);
    }

    fn emit_ret(&mut self) {
        // RET (returns to address in X30/LR)
        self.code.push(0xD65F03C0);
    }

    fn emit_blr(&mut self, rn: usize) {
        // BLR Xn - Branch with Link to Register
        // Calls function at address in Xn, stores return address in X30
        let instruction = 0xD63F0000 | ((rn as u32) << 5);
        self.code.push(instruction);
    }

    /// Execute the compiled code (for testing)
    ///
    /// Uses a trampoline to safely execute JIT code with proper stack management
    pub fn execute(&self) -> Result<i64, String> {
        eprintln!("DEBUG: execute() called with {} instructions", self.code.len());
        eprintln!("DEBUG: JIT code:");
        for (i, inst) in self.code.iter().enumerate() {
            eprintln!("  {:04x}: {:08x}", i * 4, inst);
        }
        let code_size = self.code.len() * 4;

        unsafe {
            // Allocate memory with mmap
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                code_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                return Err("mmap failed".to_string());
            }

            // Copy code to executable memory
            let code_bytes = std::slice::from_raw_parts(
                self.code.as_ptr() as *const u8,
                code_size,
            );
            std::ptr::copy_nonoverlapping(code_bytes.as_ptr(), ptr as *mut u8, code_size);

            // Make memory executable
            if libc::mprotect(ptr, code_size, libc::PROT_READ | libc::PROT_EXEC) != 0 {
                libc::munmap(ptr, code_size);
                return Err("mprotect failed".to_string());
            }

            // Clear instruction cache (required on ARM64)
            #[cfg(target_os = "macos")]
            {
                unsafe extern "C" {
                    fn sys_icache_invalidate(start: *const libc::c_void, size: libc::size_t);
                }
                sys_icache_invalidate(ptr, code_size);
            }

            // Execute through trampoline for safety
            eprintln!("DEBUG: Creating trampoline...");
            let trampoline = Trampoline::new(64 * 1024); // 64KB stack
            eprintln!("DEBUG: Calling trampoline.execute()...");
            let result = trampoline.execute(ptr as *const u8);
            eprintln!("DEBUG: Trampoline returned: {}", result);

            // Explicitly drop trampoline before cleaning up JIT code
            drop(trampoline);

            // Clean up
            libc::munmap(ptr, code_size);

            // Return the tagged result (caller is responsible for untagging if needed)
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::read;
    use crate::clojure_ast::analyze;
    use crate::compiler::Compiler;
    use crate::gc_runtime::GCRuntime;
    use std::sync::Arc;
    use std::cell::UnsafeCell;

    #[test]
    fn test_arm64_codegen_add() {
        let code = "(+ 1 2)";
        let val = read(code).unwrap();
        let ast = analyze(&val).unwrap();

        let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
        let mut compiler = Compiler::new(runtime);
        let result_reg = compiler.compile(&ast).unwrap();
        let instructions = compiler.take_instructions();

        let mut codegen = Arm64CodeGen::new();
        let machine_code = codegen.compile(&instructions, &result_reg, 0).unwrap();

        println!("\nGenerated {} ARM64 instructions for (+ 1 2)", machine_code.len());
        for (i, inst) in machine_code.iter().enumerate() {
            println!("  {:04x}: {:08x}", i * 4, inst);
        }

        let result = codegen.execute().unwrap();
        // Result is tagged: 3 << 3 = 24
        assert_eq!(result, 24);
    }

    #[test]
    fn test_arm64_codegen_nested() {
        let code = "(+ (* 2 3) 4)";
        let val = read(code).unwrap();
        let ast = analyze(&val).unwrap();

        let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
        let mut compiler = Compiler::new(runtime);
        let result_reg = compiler.compile(&ast).unwrap();
        let instructions = compiler.take_instructions();

        let mut codegen = Arm64CodeGen::new();
        let machine_code = codegen.compile(&instructions, &result_reg, 0).unwrap();

        println!("\nGenerated {} ARM64 instructions for (+ (* 2 3) 4)", machine_code.len());

        let result = codegen.execute().unwrap();
        // Result is tagged: 10 << 3 = 80
        assert_eq!(result, 80);
    }
}
