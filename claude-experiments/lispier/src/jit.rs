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

/// Default pass pipeline for CPU execution
const DEFAULT_PIPELINE: &str = "builtin.module(func.func(llvm-request-c-wrappers),convert-scf-to-cf,convert-func-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,convert-index-to-llvm,finalize-memref-to-llvm,reconcile-unrealized-casts)";

impl Jit {
    /// Create a new JIT compiler from an MLIR module
    ///
    /// This lowers the module to LLVM dialect and creates an execution engine.
    pub fn new(
        registry: &DialectRegistry,
        module: &mut Module,
    ) -> Result<Self, JitError> {
        Self::with_libraries(registry, module, &[])
    }

    /// Create a new JIT compiler with shared libraries to link
    ///
    /// Library paths can be:
    /// - Absolute paths to .so/.dylib files
    /// - Library names that will be searched in standard paths
    pub fn with_libraries(
        registry: &DialectRegistry,
        module: &mut Module,
        library_paths: &[&str],
    ) -> Result<Self, JitError> {
        Self::with_pipeline_and_libraries(registry, module, DEFAULT_PIPELINE, library_paths)
    }

    /// Create a new JIT compiler with a custom pass pipeline and shared libraries
    ///
    /// The pipeline string should be in MLIR pass pipeline syntax.
    /// Library paths can be absolute paths to .so/.dylib files.
    pub fn with_pipeline_and_libraries(
        registry: &DialectRegistry,
        module: &mut Module,
        pipeline: &str,
        library_paths: &[&str],
    ) -> Result<Self, JitError> {
        // Run the pass pipeline
        Self::run_pipeline(registry, module, pipeline)?;

        // Create execution engine with shared libraries
        let engine = ExecutionEngine::new(module, 2, library_paths, false);

        Ok(Self { engine })
    }

    /// Run a pass pipeline on the module
    pub fn run_pipeline(
        registry: &DialectRegistry,
        module: &mut Module,
        pipeline: &str,
    ) -> Result<(), JitError> {
        let context = registry.context();

        // Register all passes so we can use them by name
        register_all_passes();

        // Create pass manager
        let pm = PassManager::new(context);

        // Enable verification after each pass
        pm.enable_verifier(true);

        parse_pass_pipeline(pm.as_operation_pass_manager(), pipeline)
            .map_err(|e| JitError::PassPipelineFailed(format!("{:?}", e)))?;

        // Run the pass manager on the module
        pm.run(module).map_err(|_| JitError::PassManagerRunFailed)?;

        Ok(())
    }

    /// Lower a module from high-level dialects to LLVM dialect (default pipeline)
    fn lower_to_llvm(registry: &DialectRegistry, module: &mut Module) -> Result<(), JitError> {
        Self::run_pipeline(registry, module, DEFAULT_PIPELINE)
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

    /// Register GPU runtime symbols from a shared library
    ///
    /// This loads mgpu* functions from the given library and registers them
    /// with the `_mlir_ciface_` prefix needed by compiled code.
    ///
    /// # Safety
    /// Must be called before invoking any function that uses GPU runtime symbols.
    pub unsafe fn register_gpu_runtime(&self, lib_path: &str) {
        // GPU runtime functions that need ciface wrappers
        let gpu_fns = [
            "mgpuMemAlloc",
            "mgpuMemcpy",
            "mgpuMemFree",
            "mgpuMemGetDeviceMemRef1dFloat",
            "mgpuMemGetDeviceMemRef1dInt32",
            "mgpuMemHostRegister",
            "mgpuMemHostRegisterMemRef",
            "mgpuMemHostUnregister",
            "mgpuMemHostUnregisterMemRef",
            "mgpuMemset16",
            "mgpuMemset32",
            "mgpuStreamCreate",
            "mgpuStreamDestroy",
            "mgpuStreamSynchronize",
            "mgpuStreamWaitEvent",
            "mgpuEventCreate",
            "mgpuEventDestroy",
            "mgpuEventRecord",
            "mgpuEventSynchronize",
            "mgpuLaunchKernel",
            "mgpuModuleLoad",
            "mgpuModuleUnload",
            "mgpuModuleGetFunction",
        ];

        // Use null-terminated path for dlopen
        let path_cstr = std::ffi::CString::new(lib_path).unwrap();
        let handle = libc::dlopen(path_cstr.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL);
        if handle.is_null() {
            eprintln!("Warning: Failed to dlopen {}", lib_path);
            return;
        }

        for name in gpu_fns {
            let name_cstr = std::ffi::CString::new(name).unwrap();
            let ptr = libc::dlsym(handle, name_cstr.as_ptr());
            if !ptr.is_null() {
                // Register both bare name and ciface version
                // Add null terminators for MLIR's symbol lookup
                eprintln!("  Registering {} at {:p}", name, ptr);
                let name_nul = format!("{}\0", name);
                self.register_symbol(&name_nul, ptr as *mut ());
                let ciface_name = format!("_mlir_ciface_{}\0", name);
                self.register_symbol(&ciface_name, ptr as *mut ());
            } else {
                eprintln!("  Symbol {} not found", name);
            }
        }
    }

    /// Register common libc functions with the JIT
    ///
    /// This makes functions like `malloc`, `free`, `memcpy`, etc.
    /// available for calling from compiled code.
    ///
    /// # Safety
    /// Must be called before invoking any function that uses libc symbols.
    pub unsafe fn register_libc(&self) {
        let libc_fns: &[(&str, *mut ())] = &[
            ("malloc", libc::malloc as *mut ()),
            ("free", libc::free as *mut ()),
            ("calloc", libc::calloc as *mut ()),
            ("realloc", libc::realloc as *mut ()),
            ("memcpy", libc::memcpy as *mut ()),
            ("memset", libc::memset as *mut ()),
            ("memmove", libc::memmove as *mut ()),
            ("strlen", libc::strlen as *mut ()),
            ("printf", libc::printf as *mut ()),
            // printf_N aliases for fixed-arity calls (workaround for variadic)
            ("printf_1", libc::printf as *mut ()),
            ("printf_2", libc::printf as *mut ()),
            ("printf_3", libc::printf as *mut ()),
            ("printf_4", libc::printf as *mut ()),
            ("puts", libc::puts as *mut ()),
            ("putchar", libc::putchar as *mut ()),
            ("exit", libc::exit as *mut ()),
            ("abort", libc::abort as *mut ()),
            ("atoi", libc::atoi as *mut ()),
            ("atol", libc::atol as *mut ()),
            ("atof", libc::atof as *mut ()),
            ("strtol", libc::strtol as *mut ()),
            ("strtod", libc::strtod as *mut ()),
        ];

        for (name, ptr) in libc_fns {
            unsafe {
                self.register_symbol(name, *ptr);
                let ciface_name = format!("_mlir_ciface_{}", name);
                self.register_symbol(&ciface_name, *ptr);
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
    use crate::macros::Macro;
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
        use crate::macro_compiler::MacroCompiler;

        // Load and compile the macro module using MacroCompiler
        // (MacroCompiler is the proper way to compile macro modules)
        let macro_file = Path::new("examples/macro_double.lisp");
        let compiler = MacroCompiler::new();
        let compiled = compiler
            .compile_file(macro_file)
            .expect("Failed to compile macro module");

        assert!(!compiled.macros.is_empty(), "Expected at least one macro");
        let jit_macro = &compiled.macros[0];
        assert_eq!(jit_macro.name(), "double");

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
