use crate::ir::{Instruction, IrValue, VirtualRegister, Condition, Label};
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
    pub fn compile(&mut self, instructions: &[Instruction]) -> Result<Vec<u32>, String> {
        // Reset state
        self.code.clear();
        self.register_map.clear();
        self.next_physical_reg = 0;
        self.label_positions.clear();
        self.pending_fixups.clear();

        // Track the last result register
        let mut last_result_reg: Option<usize> = None;

        // Compile each instruction
        for inst in instructions {
            if let Some(result_reg) = self.compile_instruction(inst)? {
                last_result_reg = Some(result_reg);
            }
        }

        // Apply jump fixups
        self.apply_fixups()?;

        // Move final result to x0 if not already there
        if let Some(result_reg) = last_result_reg {
            if result_reg != 0 {
                self.emit_mov(0, result_reg);
            }
        }

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

            Instruction::Label(label) => {
                // Record position of this label
                self.label_positions.insert(label.clone(), self.code.len());
            }

            Instruction::Jump(_label) => {
                // For now, just emit a placeholder
                // We'll fix this up later
                return Err("Jump not yet implemented".to_string());
            }

            Instruction::JumpIf(_label, _cond, _src1, _src2) => {
                return Err("JumpIf not yet implemented".to_string());
            }

            Instruction::Compare(_dst, _src1, _src2, _cond) => {
                return Err("Compare not yet implemented".to_string());
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
        if let Some(&physical) = self.register_map.get(vreg) {
            physical
        } else {
            // Allocate a new physical register
            let physical = self.next_physical_reg;
            self.next_physical_reg += 1;
            if physical >= 16 {
                panic!("Out of registers! Need to implement spilling");
            }
            self.register_map.insert(*vreg, physical);
            physical
        }
    }

    fn apply_fixups(&mut self) -> Result<(), String> {
        // For now, we don't have any jumps to fix up
        if !self.pending_fixups.is_empty() {
            return Err("Cannot apply fixups - jumps not yet implemented".to_string());
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
        // MOVZ Xd, #imm for 16-bit immediates
        if imm >= 0 && imm < 65536 {
            let instruction = 0xD2800000 | ((imm as u32) << 5) | (dst as u32);
            self.code.push(instruction);
        } else {
            // For larger values, use MOVZ/MOVK sequence
            let low = (imm & 0xFFFF) as u32;
            let high = ((imm >> 16) & 0xFFFF) as u32;

            // MOVZ Xd, #low
            let movz = 0xD2800000 | (low << 5) | (dst as u32);
            self.code.push(movz);

            if high != 0 {
                // MOVK Xd, #high, LSL #16
                let movk = 0xF2A00000 | (high << 5) | (dst as u32);
                self.code.push(movk);
            }
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

    fn emit_ret(&mut self) {
        // RET (returns to address in X30/LR)
        self.code.push(0xD65F03C0);
    }

    /// Execute the compiled code (for testing)
    pub fn execute(&self) -> Result<i64, String> {
        use std::mem;

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

            // Execute!
            let func: extern "C" fn() -> i64 = mem::transmute(ptr);
            let result = func();

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

    #[test]
    fn test_arm64_codegen_add() {
        let code = "(+ 1 2)";
        let val = read(code).unwrap();
        let ast = analyze(&val).unwrap();

        let mut compiler = Compiler::new();
        compiler.compile(&ast).unwrap();
        let instructions = compiler.finish();

        let mut codegen = Arm64CodeGen::new();
        let machine_code = codegen.compile(&instructions).unwrap();

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

        let mut compiler = Compiler::new();
        compiler.compile(&ast).unwrap();
        let instructions = compiler.finish();

        let mut codegen = Arm64CodeGen::new();
        let machine_code = codegen.compile(&instructions).unwrap();

        println!("\nGenerated {} ARM64 instructions for (+ (* 2 3) 4)", machine_code.len());

        let result = codegen.execute().unwrap();
        assert_eq!(result, 10);
    }
}
