use melior::{
    Context,
    ExecutionEngine,
    ir::{Location, Module},
    pass::{Pass, PassManager},
};

/// JIT compilation and execution engine
pub struct JitEngine<'c> {
    context: &'c Context,
    engine: Option<ExecutionEngine>,
}

impl<'c> JitEngine<'c> {
    pub fn new(context: &'c Context) -> Self {
        Self {
            context,
            engine: None,
        }
    }

    /// Initialize the JIT engine with a module
    pub fn initialize(&mut self, module: &Module<'c>) -> Result<(), Box<dyn std::error::Error>> {
        // Apply optimization passes before JIT compilation
        self.apply_optimization_passes(module)?;
        
        // Create execution engine
        let engine = ExecutionEngine::new(module, 2, &[], false);
        self.engine = Some(engine);
        
        println!("✅ JIT engine initialized successfully");
        Ok(())
    }

    /// Apply optimization passes to improve generated code
    fn apply_optimization_passes(&self, module: &Module<'c>) -> Result<(), Box<dyn std::error::Error>> {
        let pm = PassManager::new(self.context);

        unsafe {
            // Add basic optimization passes
            let canonicalize = Pass::from_raw(mlir_sys::mlirCreateTransformsCanonicalizer());
            pm.add_pass(canonicalize);

            let cse = Pass::from_raw(mlir_sys::mlirCreateTransformsCSE());
            pm.add_pass(cse);

            // Note: LLVM optimization pass not available in this version
        }

        match pm.run(module) {
            Ok(_) => {
                println!("✅ Applied optimization passes");
                Ok(())
            }
            Err(e) => {
                println!("⚠️  Optimization passes failed: {:?}", e);
                // Continue anyway, optimizations are optional
                Ok(())
            }
        }
    }

    /// Execute a function by name with no arguments, returning an i32
    pub fn execute_function(&self, function_name: &str) -> Result<i32, Box<dyn std::error::Error>> {
        let engine = self.engine.as_ref()
            .ok_or("JIT engine not initialized")?;

        println!("🔥 Executing function: {}", function_name);

        // Look up the function
        let function_ptr = engine.lookup(function_name);
        
        let function_ptr = match function_ptr {
            Some(ptr) => ptr,
            None => return Err(format!("Function '{}' not found", function_name).into()),
        };

        println!("✅ Found function at address: {:p}", function_ptr);

        // Cast to function pointer and call
        // This is unsafe because we're calling JIT-compiled code
        unsafe {
            let func: extern "C" fn() -> i32 = std::mem::transmute(function_ptr);
            let result = func();
            println!("🎯 Function returned: {}", result);
            Ok(result)
        }
    }

    /// Verify that the module is valid before JIT compilation
    pub fn verify_module(module: &Module<'c>) -> Result<(), Box<dyn std::error::Error>> {
        if module.as_operation().verify() {
            println!("✅ Module verification passed");
            Ok(())
        } else {
            Err("Module verification failed".into())
        }
    }

    /// Print module IR for debugging
    pub fn print_module(module: &Module<'c>, stage: &str) {
        println!("\n📋 Module IR after {}:", stage);
        println!("{}", module.as_operation());
        println!("");
    }
}

/// High-level compilation pipeline
pub struct CompilationPipeline<'c> {
    context: &'c Context,
    jit_engine: JitEngine<'c>,
}

impl<'c> CompilationPipeline<'c> {
    pub fn new(context: &'c Context) -> Self {
        Self {
            context,
            jit_engine: JitEngine::new(context),
        }
    }

    /// Full compilation pipeline: Custom Dialect -> Standard Dialects -> LLVM -> JIT -> Execute
    pub fn compile_and_execute(
        &mut self,
        custom_module: Module<'c>,
        function_name: &str,
    ) -> Result<i32, Box<dyn std::error::Error>> {
        println!("🚀 Starting full compilation pipeline");
        
        // Stage 1: Print original module
        JitEngine::print_module(&custom_module, "custom dialect creation");

        // Stage 2: Transform custom dialect to standard dialects
        println!("🔄 Stage 1: Transforming custom dialect to standard dialects");
        let standard_module = crate::dialect_transforms::CalcToStandardTransform::transform_module(
            self.context,
            &custom_module,
        )?;
        JitEngine::print_module(&standard_module, "dialect transformation");

        // Stage 3: Lower standard dialects to LLVM
        println!("🔄 Stage 2: Lowering standard dialects to LLVM");
        crate::dialect_transforms::LLVMLowering::lower_to_llvm(self.context, &standard_module)?;
        JitEngine::print_module(&standard_module, "LLVM lowering");

        // Stage 4: Verify module
        println!("🔄 Stage 3: Verifying module");
        JitEngine::verify_module(&standard_module)?;

        // Stage 5: JIT compile
        println!("🔄 Stage 4: JIT compilation");
        self.jit_engine.initialize(&standard_module)?;

        // Stage 6: Execute
        println!("🔄 Stage 5: Execution");
        let result = self.jit_engine.execute_function(function_name)?;
        
        println!("✨ Compilation pipeline completed successfully!");
        println!("🎯 Final result: {}", result);
        
        Ok(result)
    }

    /// Compile and execute with detailed timing information
    pub fn compile_and_execute_timed(
        &mut self,
        custom_module: Module<'c>,
        function_name: &str,
    ) -> Result<(i32, std::time::Duration), Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();
        
        let result = self.compile_and_execute(custom_module, function_name)?;
        
        let total_time = start_time.elapsed();
        
        println!("⏱️  Total compilation and execution time: {:?}", total_time);
        
        Ok((result, total_time))
    }
}

/// Utilities for testing and benchmarking
pub struct BenchmarkUtils;

impl BenchmarkUtils {
    /// Run a function multiple times and collect statistics
    pub fn benchmark_function<F>(
        name: &str,
        iterations: usize,
        func: F,
    ) -> Result<BenchmarkResults, Box<dyn std::error::Error>>
    where
        F: Fn() -> Result<i32, Box<dyn std::error::Error>>,
    {
        println!("📊 Benchmarking function '{}' with {} iterations", name, iterations);
        
        let mut times = Vec::new();
        let mut results = Vec::new();
        
        for i in 0..iterations {
            let start = std::time::Instant::now();
            let result = func()?;
            let duration = start.elapsed();
            
            times.push(duration);
            results.push(result);
            
            if i == 0 {
                println!("✅ First run result: {}", result);
            }
        }
        
        // Verify all results are the same
        let first_result = results[0];
        for result in &results {
            if *result != first_result {
                return Err("Inconsistent results across benchmark runs".into());
            }
        }
        
        let avg_time = times.iter().sum::<std::time::Duration>() / times.len() as u32;
        let min_time = times.iter().min().unwrap();
        let max_time = times.iter().max().unwrap();
        
        let stats = BenchmarkResults {
            function_name: name.to_string(),
            iterations,
            result: first_result,
            average_time: avg_time,
            min_time: *min_time,
            max_time: *max_time,
        };
        
        println!("📈 Benchmark results:");
        println!("   Average time: {:?}", stats.average_time);
        println!("   Min time: {:?}", stats.min_time);
        println!("   Max time: {:?}", stats.max_time);
        println!("   Result: {}", stats.result);
        
        Ok(stats)
    }
}

/// Benchmark results structure
#[derive(Debug)]
pub struct BenchmarkResults {
    pub function_name: String,
    pub iterations: usize,
    pub result: i32,
    pub average_time: std::time::Duration,
    pub min_time: std::time::Duration,
    pub max_time: std::time::Duration,
}