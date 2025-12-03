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
}

impl Arm64CodeGen {
    pub fn new() -> Self {
        Arm64CodeGen {
            code: Vec::new(),
            register_map: HashMap::new(),
            next_physical_reg: 0,
            label_positions: HashMap::new(),
            pending_fixups: Vec::new(),
        }
    }

    /// Compile IR instructions to ARM64 machine code
    pub fn compile(&mut self, instructions: &[Instruction], result_reg: &IrValue) -> Result<Vec<u32>, String> {
        // Reset state
        self.code.clear();
        self.register_map.clear();
        self.next_physical_reg = 0;
        self.label_positions.clear();
        self.pending_fixups.clear();

        // Run linear scan register allocation
        let mut allocator = LinearScan::new(instructions.to_vec(), 0);
        allocator.allocate();

        // Find the physical register for the result (before consuming allocator)
        let result_physical = if let IrValue::Register(vreg) = result_reg {
            allocator.allocated_registers.get(vreg)
                .ok_or_else(|| format!("Result register {:?} not allocated", vreg))?
                .index
        } else {
            return Err(format!("Expected register for result, got {:?}", result_reg));
        };

        let allocated_instructions = allocator.finish();

        // Emit function prologue (save FP and LR)
        self.emit_stp(29, 30, 31, -2);  // stp x29, x30, [sp, #-16]!
        self.emit_mov(29, 31);           // mov x29, sp (set frame pointer)

        // Compile each instruction (now with physical registers)
        for inst in &allocated_instructions {
            self.compile_instruction(inst)?;
        }

        // Apply jump fixups
        self.apply_fixups()?;

        // Move result to x0 and untag it for return
        if result_physical != 0 {
            self.emit_mov(0, result_physical);
        }
        // Untag the result (shift right by 3)
        self.emit_asr_imm(0, 0, 3);

        // Emit function epilogue (restore FP and LR)
        self.emit_ldp(29, 30, 31, 2);    // ldp x29, x30, [sp], #16

        // Emit return instruction
        self.emit_ret();

        Ok(self.code.clone())
    }

    fn compile_instruction(&mut self, inst: &Instruction) -> Result<(), String> {
        match inst {
            Instruction::LoadConstant(dst, value) => {
                let dst_reg = self.get_physical_reg_for_irvalue(dst)?;
                match value {
                    IrValue::TaggedConstant(c) => {
                        self.emit_mov_imm(dst_reg, *c as i64);
                    }
                    IrValue::True => {
                        self.emit_mov_imm(dst_reg, 1);
                    }
                    IrValue::False => {
                        self.emit_mov_imm(dst_reg, 0);
                    }
                    IrValue::Null => {
                        self.emit_mov_imm(dst_reg, 0);
                    }
                    _ => return Err(format!("Invalid constant: {:?}", value)),
                }
            }

            Instruction::LoadVar(dst, var_ptr) => {
                // LoadVar: call trampoline to check dynamic bindings
                // ARM64 calling convention:
                // - x0 = argument (var_ptr, tagged)
                // - x0 = return value (tagged)
                // - x30 = link register (return address)
                // - x19-x28 are callee-saved (our allocator uses these, so they're safe)
                let dst_reg = self.get_physical_reg_for_irvalue(dst)?;

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
            }

            Instruction::StoreVar(var_ptr, value) => {
                // StoreVar: store value into var at runtime
                // Var layout: [header(8)] [ns_ptr(8)] [symbol_ptr(8)] [value(8)]
                // We want to write to field 2 (value) which is at offset 24 bytes (3 * 8)
                let value_reg = self.get_physical_reg_for_irvalue(value)?;

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
                let dst_reg = self.get_physical_reg_for_irvalue(dst)?;
                self.emit_mov_imm(dst_reg, 1);
            }

            Instruction::LoadFalse(dst) => {
                let dst_reg = self.get_physical_reg_for_irvalue(dst)?;
                self.emit_mov_imm(dst_reg, 0);
            }

            Instruction::Untag(dst, src) => {
                let dst_reg = self.get_physical_reg_for_irvalue(dst)?;
                let src_reg = self.get_physical_reg_for_irvalue(src)?;
                // Untag: arithmetic right shift by 3
                self.emit_asr_imm(dst_reg, src_reg, 3);
            }

            Instruction::Tag(dst, src, _tag) => {
                let dst_reg = self.get_physical_reg_for_irvalue(dst)?;
                let src_reg = self.get_physical_reg_for_irvalue(src)?;
                // Tag: left shift by 3 (int tag is 000)
                self.emit_lsl_imm(dst_reg, src_reg, 3);
            }

            Instruction::AddInt(dst, src1, src2) => {
                let dst_reg = self.get_physical_reg_for_irvalue(dst)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2)?;
                self.emit_add(dst_reg, src1_reg, src2_reg);
            }

            Instruction::Sub(dst, src1, src2) => {
                let dst_reg = self.get_physical_reg_for_irvalue(dst)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2)?;
                self.emit_sub(dst_reg, src1_reg, src2_reg);
            }

            Instruction::Mul(dst, src1, src2) => {
                let dst_reg = self.get_physical_reg_for_irvalue(dst)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2)?;
                self.emit_mul(dst_reg, src1_reg, src2_reg);
            }

            Instruction::Div(dst, src1, src2) => {
                let dst_reg = self.get_physical_reg_for_irvalue(dst)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2)?;
                self.emit_sdiv(dst_reg, src1_reg, src2_reg);
            }

            Instruction::Assign(dst, src) => {
                let dst_reg = self.get_physical_reg_for_irvalue(dst)?;
                let src_reg = self.get_physical_reg_for_irvalue(src)?;
                if dst_reg != src_reg {
                    self.emit_mov(dst_reg, src_reg);
                }
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
                        let value_reg = self.get_physical_reg_for_irvalue(value)?;

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
                        let value_reg = self.get_physical_reg_for_irvalue(value)?;

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
                let src1_reg = self.get_physical_reg_for_irvalue(src1)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2)?;

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
                // Compare and set result to true/false
                let dst_reg = self.get_physical_reg_for_irvalue(dst)?;
                let src1_reg = self.get_physical_reg_for_irvalue(src1)?;
                let src2_reg = self.get_physical_reg_for_irvalue(src2)?;

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
                let inverted_cond = cond_code ^ 1; // Invert the condition
                let instruction = 0x9A9F07E0 | (inverted_cond << 12) | (dst_reg as u32);
                self.code.push(instruction);
            }

            Instruction::Ret(value) => {
                // Move result to x0 (return register)
                let src_reg = self.get_physical_reg_for_irvalue(value)?;
                if src_reg != 0 {
                    self.emit_mov(0, src_reg);
                }
            }
        }

        Ok(())
    }

    fn get_physical_reg_for_irvalue(&mut self, value: &IrValue) -> Result<usize, String> {
        match value {
            IrValue::Register(vreg) => {
                Ok(self.get_physical_reg(vreg))
            }
            _ => Err(format!("Expected register, got {:?}", value)),
        }
    }

    fn get_physical_reg(&mut self, vreg: &VirtualRegister) -> usize {
        // After linear scan allocation, all registers are already physical
        // Just return the register index directly
        vreg.index
    }

    fn apply_fixups(&mut self) -> Result<(), String> {
        for (code_index, label) in &self.pending_fixups {
            let target_pos = self.label_positions.get(label)
                .ok_or_else(|| format!("Undefined label: {}", label))?;

            // Calculate offset in instructions (not bytes)
            let offset = (*target_pos as isize) - (*code_index as isize);

            // Check if offset fits in the instruction encoding
            if offset < -1048576 || offset > 1048575 {
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
        // MOV is ORR Xd, XZR, Xm
        let instruction = 0xAA0003E0 | ((src as u32) << 16) | (dst as u32);
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
        let instruction = 0xD3400000 | ((immr as u32) << 16) | ((imms as u32) << 10) | ((src as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_asr_imm(&mut self, dst: usize, src: usize, shift: u32) {
        // ASR Xd, Xn, #shift (arithmetic shift right)
        // This is SBFM (Signed Bitfield Move)
        // ASR #shift is: SBFM Xd, Xn, #shift, #63
        let shift = shift & 0x3F; // 6 bits
        let instruction = 0x9340FC00 | ((shift as u32) << 16) | ((src as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_cmp(&mut self, src1: usize, src2: usize) {
        // CMP Xn, Xm (compare - this is SUBS XZR, Xn, Xm)
        let instruction = 0xEB00001F | ((src2 as u32) << 16) | ((src1 as u32) << 5);
        self.code.push(instruction);
    }

    fn emit_ldr_offset(&mut self, dst: usize, base: usize, offset: i32) {
        // LDR Xd, [Xn, #offset]
        // Offset is in bytes, needs to be divided by 8 for encoding (unsigned 12-bit)
        let offset_scaled = (offset / 8) as u32;
        let instruction = 0xF9400000 | (offset_scaled << 10) | ((base as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_str_offset(&mut self, src: usize, base: usize, offset: i32) {
        // STR Xt, [Xn, #offset]
        // Offset is in bytes, needs to be divided by 8 for encoding (unsigned 12-bit)
        let offset_scaled = (offset / 8) as u32;
        let instruction = 0xF9000000 | (offset_scaled << 10) | ((base as u32) << 5) | (src as u32);
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
            let trampoline = Trampoline::new(64 * 1024); // 64KB stack
            let result = trampoline.execute(ptr as *const u8);

            // Explicitly drop trampoline before cleaning up JIT code
            drop(trampoline);

            // Clean up
            libc::munmap(ptr, code_size);

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
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_arm64_codegen_add() {
        let code = "(+ 1 2)";
        let val = read(code).unwrap();
        let ast = analyze(&val).unwrap();

        let runtime = Arc::new(Mutex::new(GCRuntime::new()));
        let mut compiler = Compiler::new(runtime);
        let result_reg = compiler.compile(&ast).unwrap();
        let instructions = compiler.finish();

        let mut codegen = Arm64CodeGen::new();
        let machine_code = codegen.compile(&instructions, &result_reg).unwrap();

        println!("\nGenerated {} ARM64 instructions for (+ 1 2)", machine_code.len());
        for (i, inst) in machine_code.iter().enumerate() {
            println!("  {:04x}: {:08x}", i * 4, inst);
        }

        let result = codegen.execute().unwrap();
        assert_eq!(result, 3);
    }

    #[test]
    fn test_arm64_codegen_nested() {
        let code = "(+ (* 2 3) 4)";
        let val = read(code).unwrap();
        let ast = analyze(&val).unwrap();

        let runtime = Arc::new(Mutex::new(GCRuntime::new()));
        let mut compiler = Compiler::new(runtime);
        let result_reg = compiler.compile(&ast).unwrap();
        let instructions = compiler.finish();

        let mut codegen = Arm64CodeGen::new();
        let machine_code = codegen.compile(&instructions, &result_reg).unwrap();

        println!("\nGenerated {} ARM64 instructions for (+ (* 2 3) 4)", machine_code.len());

        let result = codegen.execute().unwrap();
        assert_eq!(result, 10);
    }
}
