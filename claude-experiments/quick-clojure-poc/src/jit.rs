use crate::clojure_ast::Expr;
use crate::value::Value;
use std::collections::HashMap;

/// Minimal JIT compiler - compiles simple expressions directly to ARM64
///
/// This is a simplified version to get compilation working without
/// all of Beagle's infrastructure. We'll integrate the full backend later.
pub struct JitCompiler {
    /// Global variables
    globals: HashMap<String, i64>,

    /// Generated ARM64 machine code
    code: Vec<u32>,

    /// Next register to allocate (x0-x15)
    next_reg: usize,
}

impl JitCompiler {
    pub fn new() -> Self {
        JitCompiler {
            globals: HashMap::new(),
            code: Vec::new(),
            next_reg: 0,
        }
    }

    /// Compile an expression and return the machine code (for inspection)
    pub fn get_machine_code(&mut self, expr: &Expr) -> Result<Vec<u32>, String> {
        // Reset state
        self.code.clear();
        self.next_reg = 0;

        // Compile expression
        let result_reg = self.compile_expr(expr)?;

        // Move result to x0 (return register) if not already there
        if result_reg != 0 {
            self.emit_mov(0, result_reg);
        }

        // Emit return instruction
        self.emit_ret();

        Ok(self.code.clone())
    }

    /// Compile and execute an expression
    pub fn compile_and_run(&mut self, expr: &Expr) -> Result<i64, String> {
        // Compile to get machine code
        self.get_machine_code(expr)?;

        // Execute the compiled code
        self.execute()
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<usize, String> {
        match expr {
            Expr::Literal(Value::Int(n)) => {
                let reg = self.alloc_reg();
                self.emit_mov_imm(reg, *n);
                Ok(reg)
            }

            Expr::Literal(Value::Bool(true)) => {
                let reg = self.alloc_reg();
                self.emit_mov_imm(reg, 1);
                Ok(reg)
            }

            Expr::Literal(Value::Bool(false)) => {
                let reg = self.alloc_reg();
                self.emit_mov_imm(reg, 0);
                Ok(reg)
            }

            Expr::Var(name) => {
                let value = *self.globals.get(name)
                    .ok_or_else(|| format!("Undefined variable: {}", name))?;
                let reg = self.alloc_reg();
                self.emit_mov_imm(reg, value);
                Ok(reg)
            }

            Expr::Def { name, value } => {
                // For now, just evaluate and store
                // In a real implementation, this would need more sophisticated handling
                let val_reg = self.compile_expr(value)?;
                // We can't extract the value at compile time for complex expressions
                // For now, just return the register
                // TODO: This needs proper runtime support
                Err("def not yet fully supported in JIT".to_string())
            }

            Expr::Call { func, args } => {
                if let Expr::Var(name) = &**func {
                    match name.as_str() {
                        "+" => self.compile_add(args),
                        "-" => self.compile_sub(args),
                        "*" => self.compile_mul(args),
                        _ => Err(format!("Function {} not yet supported in JIT", name)),
                    }
                } else {
                    Err("Only symbol function calls supported".to_string())
                }
            }

            _ => Err(format!("Expression type not yet supported in JIT: {:?}", expr)),
        }
    }

    fn compile_add(&mut self, args: &[Expr]) -> Result<usize, String> {
        if args.len() != 2 {
            return Err("+ requires 2 arguments".to_string());
        }

        let left_reg = self.compile_expr(&args[0])?;
        let right_reg = self.compile_expr(&args[1])?;
        let result_reg = self.alloc_reg();

        self.emit_add(result_reg, left_reg, right_reg);

        Ok(result_reg)
    }

    fn compile_sub(&mut self, args: &[Expr]) -> Result<usize, String> {
        if args.len() != 2 {
            return Err("- requires 2 arguments".to_string());
        }

        let left_reg = self.compile_expr(&args[0])?;
        let right_reg = self.compile_expr(&args[1])?;
        let result_reg = self.alloc_reg();

        self.emit_sub(result_reg, left_reg, right_reg);

        Ok(result_reg)
    }

    fn compile_mul(&mut self, args: &[Expr]) -> Result<usize, String> {
        if args.len() != 2 {
            return Err("* requires 2 arguments".to_string());
        }

        let left_reg = self.compile_expr(&args[0])?;
        let right_reg = self.compile_expr(&args[1])?;
        let result_reg = self.alloc_reg();

        self.emit_mul(result_reg, left_reg, right_reg);

        Ok(result_reg)
    }

    fn alloc_reg(&mut self) -> usize {
        let reg = self.next_reg;
        self.next_reg += 1;
        if reg >= 16 {
            panic!("Out of registers!");
        }
        reg
    }

    // ARM64 instruction encoding helpers

    fn emit_mov(&mut self, dst: usize, src: usize) {
        // MOV is actually an alias for ORR Xd, XZR, Xm
        // ORR encoding: 0xAA0003E0 | (src << 16) | dst
        let instruction = 0xAA0003E0 | ((src as u32) << 16) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_mov_imm(&mut self, dst: usize, imm: i64) {
        // MOVZ Xd, #imm - Move wide with zero
        // For simplicity, only handle 16-bit immediates for now
        if imm >= 0 && imm < 65536 {
            let instruction = 0xD2800000 | ((imm as u32) << 5) | (dst as u32);
            self.code.push(instruction);
        } else {
            // For larger values, we'd need multiple instructions
            // For now, just use MOVZ/MOVK sequence
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
        // MUL Xd, Xn, Xm (actually MADD with XZR)
        let instruction = 0x9B007C00 | ((src2 as u32) << 16) | ((src1 as u32) << 5) | (dst as u32);
        self.code.push(instruction);
    }

    fn emit_ret(&mut self) {
        // RET (returns to address in X30/LR)
        self.code.push(0xD65F03C0);
    }

    fn execute(&self) -> Result<i64, String> {
        use std::mem;

        // Allocate executable memory
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

    #[test]
    fn test_jit_literal() {
        let mut jit = JitCompiler::new();
        let val = read("42").unwrap();
        let ast = analyze(&val).unwrap();
        let result = jit.compile_and_run(&ast).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_jit_add() {
        let mut jit = JitCompiler::new();
        let val = read("(+ 1 2)").unwrap();
        let ast = analyze(&val).unwrap();
        let result = jit.compile_and_run(&ast).unwrap();
        assert_eq!(result, 3);
    }

    #[test]
    fn test_jit_nested() {
        let mut jit = JitCompiler::new();
        let val = read("(+ (* 2 3) 4)").unwrap();
        let ast = analyze(&val).unwrap();
        let result = jit.compile_and_run(&ast).unwrap();
        assert_eq!(result, 10);
    }
}
