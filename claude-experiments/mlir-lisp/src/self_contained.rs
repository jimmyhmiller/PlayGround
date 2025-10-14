/// Self-Contained Lisp Compilation System
///
/// This module implements a fully self-contained compilation pipeline
/// where everything is defined and controlled via Lisp code.
///
/// No Rust API needed - just write Lisp and it works!

use crate::{
    parser::{self, Value},
    macro_expander::MacroExpander,
    dialect_registry::DialectRegistry,
    mlir_context::MlirContext,
    transform_interpreter,
    emitter::Emitter,
    function_registry::FunctionRegistry,
    expr_compiler::ExprCompiler,
    irdl_emitter::IrdlEmitter,
    transform_emitter::TransformEmitter,
};
use melior::{
    Context,
    ir::{Module, Location, Block, Region, RegionLike, BlockLike, operation::{OperationBuilder, OperationLike}},
    pass::PassManager,
    ExecutionEngine,
    utility::load_irdl_dialects,
    dialect::transform::{TransformOptions, apply_named_sequence},
};
use std::collections::HashMap;

/// Self-contained compiler that processes Lisp source end-to-end
pub struct SelfContainedCompiler {
    mlir_ctx: MlirContext,
    expander: MacroExpander,
    registry: DialectRegistry,
    loaded_files: std::collections::HashSet<String>,
    search_paths: Vec<String>,
    functions: HashMap<String, Value>, // Store function definitions as AST
}

impl SelfContainedCompiler {
    pub fn new() -> Self {
        Self {
            mlir_ctx: MlirContext::new(),
            expander: MacroExpander::new(),
            registry: DialectRegistry::new(),
            loaded_files: std::collections::HashSet::new(),
            search_paths: vec![
                ".".to_string(),
                "./dialects".to_string(),
                "./lib".to_string(),
            ],
            functions: HashMap::new(),
        }
    }

    /// Add a directory to the search path for imports
    pub fn add_search_path(&mut self, path: String) {
        self.search_paths.push(path);
    }

    /// Evaluate a Lisp expression in the compiler context
    /// This handles special forms like defirdl-dialect, deftransform, etc.
    pub fn eval(&mut self, expr: &Value) -> Result<Value, String> {
        // Check if this is a definition form that needs to be registered
        if let Value::List(elements) = expr {
            if let Some(Value::Symbol(s)) = elements.first() {
                println!("Evaluating form: {}", s);
                match s.as_str() {
                    "defirdl-dialect" => {
                        // Expand IRDL dialect definition
                        let expanded = self.expander.expand(expr)?;
                        self.registry.process_expanded_form(&expanded)?;

                        // Generate and load IRDL module
                        if let Value::List(def_elements) = &expanded {
                            if def_elements.len() >= 2 {
                                if let Value::String(dialect_name) = &def_elements[1] {
                                    if let Some(dialect) = self.registry.get_dialect(dialect_name) {
                                        let irdl_ir = IrdlEmitter::emit_dialect(dialect);
                                        println!("\nGenerated IRDL IR:");
                                        println!("{}", irdl_ir);

                                        // Parse and load the IRDL dialect
                                        match Module::parse(self.mlir_ctx.context(), &irdl_ir) {
                                            Some(irdl_module) => {
                                                let loaded = load_irdl_dialects(&irdl_module);
                                                println!("IRDL dialects loaded: {}", loaded);
                                            }
                                            None => {
                                                eprintln!("Failed to parse IRDL module");
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if elements.len() >= 2 {
                            return Ok(elements[1].clone());
                        }
                        return Ok(Value::Symbol("ok".to_string()));
                    }
                    "deftransform" | "defpdl-pattern" => {
                        // Expand the macro
                        let expanded = self.expander.expand(expr)?;

                        println!("Registering pattern: {:?}", s);

                        // Register with the dialect registry
                        self.registry.process_expanded_form(&expanded)?;

                        println!("Pattern registered successfully");

                        // Return the name of what was defined
                        if elements.len() >= 2 {
                            return Ok(elements[1].clone());
                        }
                        return Ok(Value::Symbol("ok".to_string()));
                    }
                    "import-dialect" | "import" => {
                        // Import a file or dialect
                        if elements.len() >= 2 {
                            let import_name = match &elements[1] {
                                Value::Symbol(s) | Value::String(s) => s.clone(),
                                _ => return Err("import requires a name".into()),
                            };

                            // Try to load the file
                            let result = self.load_import(&import_name)?;
                            return Ok(Value::List(vec![
                                Value::Symbol("imported".to_string()),
                                Value::String(import_name),
                                result,
                            ]));
                        }
                        return Err("import requires a name".into());
                    }
                    "list-dialects" => {
                        let dialects: Vec<Value> = self.registry.list_dialects()
                            .into_iter()
                            .map(|s| Value::String(s.to_string()))
                            .collect();
                        return Ok(Value::Vector(dialects));
                    }
                    "list-transforms" => {
                        let transforms: Vec<Value> = self.registry.list_transforms()
                            .into_iter()
                            .map(|s| Value::String(s.to_string()))
                            .collect();
                        return Ok(Value::Vector(transforms));
                    }
                    "list-patterns" => {
                        let patterns: Vec<Value> = self.registry.list_patterns()
                            .into_iter()
                            .map(|s| Value::String(s.to_string()))
                            .collect();
                        return Ok(Value::Vector(patterns));
                    }
                    "get-dialect" => {
                        if elements.len() >= 2 {
                            if let Value::String(name) = &elements[1] {
                                if let Some(dialect) = self.registry.get_dialect(name) {
                                    // Return dialect info as a map
                                    return Ok(Value::Map(vec![
                                        (Value::Keyword("name".to_string()), Value::String(dialect.name.clone())),
                                        (Value::Keyword("namespace".to_string()), Value::String(dialect.namespace.clone())),
                                        (Value::Keyword("description".to_string()), Value::String(dialect.description.clone())),
                                        (Value::Keyword("operations".to_string()), Value::Vector(
                                            dialect.operations.iter()
                                                .map(|op| Value::String(op.name.clone()))
                                                .collect()
                                        )),
                                    ]));
                                }
                            }
                        }
                        return Err("get-dialect requires a dialect name".into());
                    }
                    "apply-transform" => {
                        // (apply-transform transform-module-name target-module-name)
                        if elements.len() < 3 {
                            return Err("apply-transform requires transform-module-name and target-module-name".into());
                        }

                        let transform_name = match &elements[1] {
                            Value::Symbol(s) | Value::String(s) => s,
                            _ => return Err("Transform module name must be a symbol or string".into()),
                        };

                        let target_name = match &elements[2] {
                            Value::Symbol(s) | Value::String(s) => s,
                            _ => return Err("Target module name must be a symbol or string".into()),
                        };

                        // For now, transform application is not fully implemented
                        return Err("apply-transform not yet fully implemented".into());
                    }
                    "store-module" => {
                        // (store-module name module-ir)
                        // For now, just acknowledge it
                        if elements.len() >= 2 {
                            if let Value::Symbol(name) | Value::String(name) = &elements[1] {
                                println!("Would store module: {}", name);
                                return Ok(Value::Symbol("module-stored".to_string()));
                            }
                        }
                        return Err("store-module requires a module name".into());
                    }
                    "println" => {
                        // (println "message" value1 value2 ...)
                        // Print all arguments
                        for arg in &elements[1..] {
                            print!("{:?} ", arg);
                        }
                        println!();
                        return Ok(Value::Symbol("ok".to_string()));
                    }
                    "defn" => {
                        // (defn name [args] return-type body)
                        // Store the function definition for later execution
                        if elements.len() < 5 {
                            return Err("defn requires: name, args, return-type, body".into());
                        }

                        let name = match &elements[1] {
                            Value::Symbol(s) => s.clone(),
                            _ => return Err("Function name must be a symbol".into()),
                        };

                        // Store the entire defn form
                        self.functions.insert(name.clone(), expr.clone());

                        // Also compile and show the IR
                        return self.compile_and_show_function(expr);
                    }
                    "jit-execute" => {
                        // (jit-execute function-name)
                        if elements.len() < 2 {
                            return Err("jit-execute requires a function name".into());
                        }

                        let func_name = match &elements[1] {
                            Value::Symbol(s) | Value::String(s) => s,
                            _ => return Err("Function name must be a symbol or string".into()),
                        };

                        return self.jit_execute(func_name);
                    }
                    _ => {}
                }
            }
        }

        // For other expressions, just expand macros
        self.expander.expand(expr)
    }

    /// Compile a function definition and show its IR (without executing)
    fn compile_and_show_function(&mut self, form: &Value) -> Result<Value, String> {
        // Parse (defn name [args] return-type body)
        if let Value::List(elements) = form {
            if elements.len() < 5 {
                return Err("defn requires: name, args, return-type, body".into());
            }

            let name = match &elements[1] {
                Value::Symbol(s) => s.clone(),
                _ => return Err("Function name must be a symbol".into()),
            };

            // Create a new module for this function using our context
            let module = self.mlir_ctx.create_module();
            self.mlir_ctx.context().set_allow_unregistered_dialects(true);

            let mut emitter = Emitter::new(&self.mlir_ctx);
            let func_registry = FunctionRegistry::new();

            // Parse return type
            let return_type = match &elements[3] {
                Value::Symbol(s) => emitter.parse_type(s)?,
                _ => return Err("Return type must be a symbol".into()),
            };

            // Create function type
            let func_type = melior::ir::r#type::FunctionType::new(
                self.mlir_ctx.context(),
                &[],
                &[return_type],
            );

            let region = Region::new();
            let entry_block = Block::new(&[]);
            region.append_block(entry_block);
            let block_ref = region.first_block()
                .ok_or("Failed to get entry block")?;

            // Compile the body with dialect registry
            let body_expr = &elements[4];
            let result_name = ExprCompiler::compile_expr(
                &mut emitter,
                &block_ref,
                body_expr,
                &func_registry,
                Some(&self.registry),
            )?;

            // Emit return
            if let Some(result_val) = emitter.get_value(&result_name) {
                let return_op = OperationBuilder::new("func.return", self.mlir_ctx.location())
                    .add_operands(&[result_val])
                    .build()
                    .map_err(|e| format!("Failed to build return: {:?}", e))?;
                block_ref.append_operation(return_op);
            }

            // Create function operation
            let func_op = OperationBuilder::new("func.func", self.mlir_ctx.location())
                .add_attributes(&[
                    (
                        melior::ir::Identifier::new(self.mlir_ctx.context(), "sym_name"),
                        melior::ir::attribute::StringAttribute::new(self.mlir_ctx.context(), &name).into(),
                    ),
                    (
                        melior::ir::Identifier::new(self.mlir_ctx.context(), "function_type"),
                        melior::ir::attribute::TypeAttribute::new(func_type.into()).into(),
                    ),
                ])
                .add_regions([region])
                .build()
                .map_err(|e| format!("Failed to build function: {:?}", e))?;

            module.body().append_operation(func_op);

            // Print the generated IR
            println!("\n{}", "=".repeat(60));
            println!("Generated MLIR for '{}':", name);
            println!("{}", "=".repeat(60));
            println!("{}", module.as_operation());
            println!();

            // Don't store module (it gets dropped here, which is fine)
            // The AST is already stored in self.functions

            Ok(Value::Map(vec![
                (Value::Keyword("function".to_string()), Value::String(name.clone())),
                (Value::Keyword("status".to_string()), Value::String("compiled".to_string())),
            ]))
        } else {
            Err("Expected list for defn".into())
        }
    }

    /// JIT execute a function
    fn jit_execute(&mut self, func_name: &str) -> Result<Value, String> {
        // Get the function definition
        let func_def = self.functions.get(func_name)
            .ok_or(format!("Function '{}' not defined", func_name))?
            .clone();

        println!("\n{}", "=".repeat(60));
        println!("JIT Executing: {}", func_name);
        println!("{}", "=".repeat(60));

        // Recompile the function fresh and execute it immediately
        // We do this in a nested scope to avoid borrow checker issues
        {
            let mut module = self.compile_function_to_module(&func_def)?;

            println!("\n1. Original IR:");
            println!("{}", module.as_operation());

            // Verify module before lowering
            if !module.as_operation().verify() {
                return Err("Module verification failed".into());
            }

            // Step 1.5: Apply transform patterns if any are registered
            let pattern_refs = self.registry.get_all_patterns();
            if !pattern_refs.is_empty() {
                println!("\n1.5. Applying transform patterns...");
                println!("   Patterns: {}", pattern_refs.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", "));

                // Convert Vec<&PdlPattern> to Vec<PdlPattern> for the emitter
                let patterns: Vec<_> = pattern_refs.iter().map(|p| (*p).clone()).collect();

                // Generate transform module
                match TransformEmitter::emit_transform_module(&patterns) {
                    Ok(transform_ir) => {
                        println!("\nGenerated Transform IR:");
                        println!("{}", transform_ir);

                        // Parse transform module
                        match Module::parse(self.mlir_ctx.context(), &transform_ir) {
                            Some(transform_module) => {
                                if let Some(transform_op) = transform_module.body().first_operation() {
                                    let options = TransformOptions::new();

                                    // Apply the transform!
                                    match apply_named_sequence(
                                        &module.as_operation(),
                                        &transform_op,
                                        &transform_module.as_operation(),
                                        &options,
                                    ) {
                                        Ok(_) => {
                                            println!("\n✅ Transform applied successfully!");
                                            println!("\nTransformed IR:");
                                            println!("{}", module.as_operation());
                                        }
                                        Err(e) => {
                                            eprintln!("⚠️  Transform application failed: {:?}", e);
                                        }
                                    }
                                }
                            }
                            None => {
                                eprintln!("Failed to parse transform module");
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to emit transform module: {}", e);
                    }
                }
            }

            // Step 2: Apply standard lowering passes
            println!("\n2. Applying lowering passes...");
            println!("   - convert-arith-to-llvm");
            println!("   - convert-func-to-llvm");
            println!("   - reconcile-unrealized-casts");

            let pm = PassManager::new(self.mlir_ctx.context());
            pm.enable_verifier(true);

            // Add conversion passes
            pm.add_pass(melior::pass::conversion::create_arith_to_llvm());
            pm.add_pass(melior::pass::conversion::create_func_to_llvm());
            pm.add_pass(melior::pass::conversion::create_reconcile_unrealized_casts());

            // Run the passes
            pm.run(&mut module).map_err(|e| format!("Pass manager failed: {:?}", e))?;

            println!("\n3. Lowered IR:");
            println!("{}", module.as_operation());

            // Verify module after lowering
            if !module.as_operation().verify() {
                return Err("Module verification failed after lowering".into());
            }

            // Step 2: Create execution engine and run
            println!("\n4. Creating ExecutionEngine and executing...");

            // Create the engine with the module reference
            let engine = ExecutionEngine::new(&module, 2, &[], false);

            let result = unsafe {
                let func_ptr = engine.lookup(func_name);
                if func_ptr.is_null() {
                    return Err(format!("Function '{}' not found in module", func_name));
                }

                // Cast to function pointer and call
                let func: extern "C" fn() -> i32 = std::mem::transmute(func_ptr);
                func()
            };

            println!("\n{}", "=".repeat(60));
            println!("✅ Execution Result: {}", result);
            println!("{}", "=".repeat(60));

            return Ok(Value::Integer(result as i64));
        }
    }

    /// Compile a function definition to a module (helper for jit_execute)
    fn compile_function_to_module(&self, form: &Value) -> Result<melior::ir::Module, String> {
        if let Value::List(elements) = form {
            if elements.len() < 5 {
                return Err("defn requires: name, args, return-type, body".into());
            }

            let name = match &elements[1] {
                Value::Symbol(s) => s.clone(),
                _ => return Err("Function name must be a symbol".into()),
            };

            // Create a new module for this function
            let module = self.mlir_ctx.create_module();
            self.mlir_ctx.context().set_allow_unregistered_dialects(true);

            let mut emitter = Emitter::new(&self.mlir_ctx);
            let func_registry = FunctionRegistry::new();

            // Parse return type
            let return_type = match &elements[3] {
                Value::Symbol(s) => emitter.parse_type(s)?,
                _ => return Err("Return type must be a symbol".into()),
            };

            // Create function type
            let func_type = melior::ir::r#type::FunctionType::new(
                self.mlir_ctx.context(),
                &[],
                &[return_type],
            );

            let region = Region::new();
            let entry_block = Block::new(&[]);
            region.append_block(entry_block);
            let block_ref = region.first_block()
                .ok_or("Failed to get entry block")?;

            // Compile the body with dialect registry
            let body_expr = &elements[4];
            let result_name = ExprCompiler::compile_expr(
                &mut emitter,
                &block_ref,
                body_expr,
                &func_registry,
                Some(&self.registry),
            )?;

            // Emit return
            if let Some(result_val) = emitter.get_value(&result_name) {
                let return_op = OperationBuilder::new("func.return", self.mlir_ctx.location())
                    .add_operands(&[result_val])
                    .build()
                    .map_err(|e| format!("Failed to build return: {:?}", e))?;
                block_ref.append_operation(return_op);
            }

            // Create function operation
            let func_op = OperationBuilder::new("func.func", self.mlir_ctx.location())
                .add_attributes(&[
                    (
                        melior::ir::Identifier::new(self.mlir_ctx.context(), "sym_name"),
                        melior::ir::attribute::StringAttribute::new(self.mlir_ctx.context(), &name).into(),
                    ),
                    (
                        melior::ir::Identifier::new(self.mlir_ctx.context(), "function_type"),
                        melior::ir::attribute::TypeAttribute::new(func_type.into()).into(),
                    ),
                ])
                .add_regions([region])
                .build()
                .map_err(|e| format!("Failed to build function: {:?}", e))?;

            module.body().append_operation(func_op);

            Ok(module)
        } else {
            Err("Expected list for defn".into())
        }
    }

    /// Load and evaluate a Lisp file
    pub fn load_file(&mut self, path: &str) -> Result<Value, String> {
        // Check if already loaded
        let canonical_path = std::fs::canonicalize(path)
            .map_err(|e| format!("Failed to resolve path {}: {}", path, e))?;

        let path_str = canonical_path.to_string_lossy().to_string();

        if self.loaded_files.contains(&path_str) {
            return Ok(Value::Symbol("already-loaded".to_string()));
        }

        let source = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file {}: {}", path, e))?;

        self.loaded_files.insert(path_str);
        self.eval_string(&source)
    }

    /// Load an import by searching for it in search paths
    fn load_import(&mut self, name: &str) -> Result<Value, String> {
        // Try different file extensions and search paths
        let extensions = vec!["", ".lisp", ".mlir-lisp"];

        for search_path in &self.search_paths.clone() {
            for ext in &extensions {
                let file_path = format!("{}/{}{}", search_path, name, ext);

                if std::path::Path::new(&file_path).exists() {
                    return self.load_file(&file_path);
                }
            }
        }

        Err(format!("Could not find import '{}' in search paths {:?}", name, self.search_paths))
    }

    /// Evaluate a string of Lisp code
    pub fn eval_string(&mut self, source: &str) -> Result<Value, String> {
        let (_, values) = parser::parse(source)
            .map_err(|e| format!("Parse error: {:?}", e))?;

        println!("Parsed {} forms from source", values.len());

        let mut result = Value::Symbol("nil".to_string());

        for value in values {
            result = self.eval(&value)?;
        }

        Ok(result)
    }

    /// Get the dialect registry (for inspection)
    pub fn registry(&self) -> &DialectRegistry {
        &self.registry
    }

    /// Get the MLIR context
    pub fn mlir_context(&self) -> &MlirContext {
        &self.mlir_ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_contained_dialect_definition() {
        let mut compiler = SelfContainedCompiler::new();

        let source = r#"
(defirdl-dialect test
  :namespace "test"
  :description "Test dialect"

  (defirdl-op foo
    :summary "Foo operation"
    :results [(result AnyInteger)]))
"#;

        let result = compiler.eval_string(source);
        assert!(result.is_ok());

        // Check that the dialect was registered
        assert!(compiler.registry().get_dialect("test").is_some());
    }
}
