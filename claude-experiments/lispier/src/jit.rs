use melior::{
    ir::Module,
    pass::PassManager,
    utility::{parse_pass_pipeline, register_all_passes},
    ExecutionEngine,
};
use thiserror::Error;

use crate::dialect::DialectRegistry;
use crate::value_ffi::get_value_ffi_functions;

#[derive(Debug, Error)]
pub enum JitError {
    #[error("pass pipeline failed: {0}")]
    PassPipelineFailed(String),

    #[error("pass manager run failed")]
    PassManagerRunFailed,

    #[error("execution engine creation failed: {0}")]
    ExecutionEngineFailed(String),

    #[error("invocation failed: {0}")]
    InvocationFailed(String),

    #[error("function not found: {0}")]
    FunctionNotFound(String),
}

/// JIT compiler for MLIR modules
pub struct Jit {
    engine: ExecutionEngine,
}

impl Jit {
    /// Create a new JIT compiler from an MLIR module
    ///
    /// This lowers the module to LLVM dialect and creates an execution engine.
    pub fn new(
        registry: &DialectRegistry,
        module: &mut Module,
    ) -> Result<Self, JitError> {
        // Lower to LLVM dialect
        Self::lower_to_llvm(registry, module)?;

        // Create execution engine
        let engine = ExecutionEngine::new(module, 2, &[], false);

        Ok(Self { engine })
    }

    /// Lower a module from high-level dialects to LLVM dialect
    fn lower_to_llvm(registry: &DialectRegistry, module: &mut Module) -> Result<(), JitError> {
        let context = registry.context();

        // Register all passes so we can use them by name
        register_all_passes();

        // Create pass manager
        let pm = PassManager::new(context);

        // Enable verification after each pass
        pm.enable_verifier(true);

        // Use pass pipeline with llvm-request-c-wrappers to generate _mlir_ciface_ wrappers
        // This enables invoke_packed to work properly
        // llvm-request-c-wrappers runs on func.func, so it must be nested
        let pipeline = "builtin.module(func.func(llvm-request-c-wrappers),convert-func-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,convert-index-to-llvm,finalize-memref-to-llvm,reconcile-unrealized-casts)";

        parse_pass_pipeline(pm.as_operation_pass_manager(), pipeline)
            .map_err(|e| JitError::PassPipelineFailed(format!("{:?}", e)))?;

        // Run the pass manager on the module
        pm.run(module).map_err(|_| JitError::PassManagerRunFailed)?;

        Ok(())
    }

    /// Invoke a function by name with packed arguments
    ///
    /// The function must have the `llvm.emit_c_interface` attribute.
    /// Arguments are passed as pointers to the actual values.
    ///
    /// # Safety
    /// The caller must ensure that the argument types match the function signature.
    pub unsafe fn invoke_packed(
        &self,
        name: &str,
        args: &mut [*mut ()],
    ) -> Result<(), JitError> {
        unsafe {
            self.engine
                .invoke_packed(name, args)
                .map_err(|e| JitError::InvocationFailed(format!("{:?}", e)))
        }
    }

    /// Look up a function pointer by name
    pub fn lookup(&self, name: &str) -> Option<*mut ()> {
        let ptr = self.engine.lookup(name);
        if ptr.is_null() {
            None
        } else {
            Some(ptr)
        }
    }

    /// Register an external symbol with the JIT
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid for the lifetime of the JIT.
    pub unsafe fn register_symbol(&self, name: &str, ptr: *mut ()) {
        unsafe {
            self.engine.register_symbol(name, ptr);
        }
    }

    /// Dump the compiled module to an object file
    pub fn dump_to_object_file(&self, path: &str) {
        self.engine.dump_to_object_file(path);
    }

    /// Register all Value FFI functions with the JIT
    ///
    /// This makes functions like `value_list_new`, `value_symbol_new`, etc.
    /// available for calling from compiled code.
    ///
    /// Registers both the bare name and the `_mlir_ciface_` prefixed version
    /// to support both direct LLVM calls and func dialect calls.
    ///
    /// # Safety
    /// Must be called before invoking any function that uses FFI symbols.
    pub unsafe fn register_value_ffi(&self) {
        for ffi_fn in get_value_ffi_functions() {
            unsafe {
                // Register the bare name
                self.register_symbol(ffi_fn.name, ffi_fn.ptr);
                // Also register with _mlir_ciface_ prefix for func dialect compatibility
                let ciface_name = format!("_mlir_ciface_{}", ffi_fn.name);
                self.register_symbol(&ciface_name, ffi_fn.ptr);
            }
        }
    }
}

/// Print an MLIR module for debugging
pub fn print_module(module: &Module) {
    println!("{}", module.as_operation());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_types_compile() {
        // This test just verifies that the JIT types compile correctly
        let registry = DialectRegistry::new();
        let _module = registry.create_module();
    }

    #[test]
    fn test_print_empty_module() {
        let registry = DialectRegistry::new();
        let module = registry.create_module();
        // Just verify it doesn't panic
        let _ = format!("{}", module.as_operation());
    }
}
