use crate::clojure_ast::Expr;
use crate::compiler::Compiler;
use crate::beagle::ir::{Instruction, Ir};
use crate::beagle::machine_code::arm_codegen::ArmCodeGen;
use crate::beagle::code_memory::CodeMemory;
use crate::value::Value;
use std::collections::HashMap;

/// Clojure runtime with JIT compilation to ARM64
pub struct Runtime {
    /// Compiled functions
    code_memory: CodeMemory,

    /// Global variables
    globals: HashMap<String, Value>,
}

impl Runtime {
    pub fn new() -> Self {
        Runtime {
            code_memory: CodeMemory::new(),
            globals: HashMap::new(),
        }
    }

    /// Compile and execute a Clojure expression
    pub fn compile_and_run(&mut self, expr: &Expr) -> Result<Value, String> {
        // Create compiler
        let mut compiler = Compiler::new();

        // Compile expression to IR
        let result_reg = compiler.compile(expr)?;

        // Get IR instructions
        let instructions = compiler.get_instructions().to_vec();

        // Add return instruction
        let mut ir_instructions = instructions;
        ir_instructions.push(Instruction::Ret(result_reg));

        // Create IR
        let mut ir = Ir::new();
        for instruction in ir_instructions {
            ir.push(instruction);
        }

        // Generate ARM64 code
        let mut codegen = ArmCodeGen::new();
        let code = codegen.compile(&ir)?;

        // Allocate executable memory and copy code
        let func_ptr = unsafe {
            self.code_memory.allocate_and_write(&code)
        };

        // Execute the compiled code
        let result = unsafe {
            let func: extern "C" fn() -> usize = std::mem::transmute(func_ptr);
            func()
        };

        // Convert result back to Value
        // For now, just return an Int
        // TODO: proper unmarshaling based on type tag
        Ok(Value::Int(result as i64))
    }
}
