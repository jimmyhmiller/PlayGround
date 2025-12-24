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

    /// Look up a macro function pointer by name
    ///
    /// Returns the function pointer if found, or None if not found.
    /// The function is expected to have signature `fn(*const Value) -> *mut Value`.
    pub fn lookup_macro_fn(
        &self,
        name: &str,
    ) -> Option<crate::macros::JitMacroFn> {
        // Try the ciface version first (generated by llvm-request-c-wrappers)
        let ciface_name = format!("_mlir_ciface_{}", name);
        let ptr = self.engine.lookup(&ciface_name);
        if !ptr.is_null() {
            return Some(unsafe { std::mem::transmute(ptr) });
        }

        // Fall back to bare name
        let ptr = self.engine.lookup(name);
        if !ptr.is_null() {
            return Some(unsafe { std::mem::transmute(ptr) });
        }

        None
    }
}

/// Print an MLIR module for debugging
pub fn print_module(module: &Module) {
    println!("{}", module.as_operation());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_gen::IRGenerator;
    use crate::macros::{JitMacro, Macro};
    use crate::module_loader::ModuleLoader;
    use crate::runtime::extract_defmacros;
    use crate::value::Value;
    use std::path::Path;

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

    #[test]
    fn test_jit_macro_integration() {
        // Load and compile the macro module
        let macro_file = Path::new("examples/macro_double.lisp");
        let project_root = macro_file.parent().unwrap().parent().unwrap();
        let mut loader = ModuleLoader::new(project_root);
        let nodes = loader.load(macro_file).expect("Failed to load macro module");

        // Extract defmacro declarations
        let defmacros = extract_defmacros(&nodes);
        assert!(!defmacros.is_empty(), "Expected at least one defmacro");
        assert_eq!(defmacros[0].name, "double");

        // Generate MLIR and compile
        let registry = DialectRegistry::new();
        let generator = IRGenerator::new(&registry);
        let mut module = generator.generate(&nodes).expect("Failed to generate IR");

        let jit = Jit::new(&registry, &mut module).expect("Failed to create JIT");

        // Register FFI symbols
        unsafe {
            jit.register_value_ffi();
        }

        // Look up the macro function
        let macro_fn = jit
            .lookup_macro_fn("double")
            .expect("Failed to find double function");

        // Create the JitMacro
        let jit_macro = unsafe { JitMacro::new("double", macro_fn) };

        // Test the macro: expand (double 21) should return (arith.addi 21 21)
        let args = vec![Value::Number(21.0)];
        let result = jit_macro.expand(&args).expect("Macro expansion failed");

        // The result should be a list (arith.addi 21 21)
        match result {
            Value::List(items) => {
                assert_eq!(items.len(), 3, "Expected 3 items in list");
                // First item should be the symbol arith.addi
                if let Value::Symbol(sym) = &items[0] {
                    assert_eq!(sym.name, "arith.addi");
                } else {
                    panic!("Expected symbol, got {:?}", items[0]);
                }
                // Second and third items should be the number 21
                if let Value::Number(n) = &items[1] {
                    assert_eq!(*n, 21.0);
                } else {
                    panic!("Expected number, got {:?}", items[1]);
                }
                if let Value::Number(n) = &items[2] {
                    assert_eq!(*n, 21.0);
                } else {
                    panic!("Expected number, got {:?}", items[2]);
                }
            }
            _ => panic!("Expected list result, got {:?}", result),
        }
    }
}
