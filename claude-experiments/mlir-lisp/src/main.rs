use mlir_lisp::{
    parser,
    mlir_context::MlirContext,
    emitter::Emitter,
    macro_expander::MacroExpander,
    expr_compiler::ExprCompiler,
    function_registry::FunctionRegistry,
    self_contained::SelfContainedCompiler,
};
use melior::{
    Context,
    pass::PassManager,
    ExecutionEngine,
    ir::r#type::Type,
};
use std::fs;
use std::env;

/// Parse function signature from defn form (for registration)
/// Returns: (name, arg_types, return_type)
fn parse_defn_signature<'c>(
    emitter: &Emitter<'c>,
    args: &[parser::Value],
) -> Result<Option<(String, Vec<melior::ir::Type<'c>>, melior::ir::Type<'c>)>, Box<dyn std::error::Error>> {
    if args.len() < 2 {
        return Ok(None);
    }

    // Parse function name
    let name = match &args[0] {
        parser::Value::Symbol(s) => s.clone(),
        _ => return Ok(None),
    };

    // Parse argument types
    let mut arg_types = vec![];
    if let parser::Value::Vector(params) = &args[1] {
        for param in params {
            if let parser::Value::Symbol(param_str) = param {
                let parts: Vec<&str> = param_str.split(':').collect();
                if parts.len() == 2 {
                    arg_types.push(emitter.parse_type(parts[1])?);
                } else {
                    // Default to i32
                    arg_types.push(emitter.parse_type("i32")?);
                }
            }
        }
    }

    // Parse return type (if specified)
    let ret_type = if args.len() > 2 {
        if let parser::Value::Symbol(s) = &args[2] {
            match emitter.parse_type(s) {
                Ok(t) => t,
                Err(_) => emitter.parse_type("i32")?,
            }
        } else {
            emitter.parse_type("i32")?
        }
    } else {
        emitter.parse_type("i32")?
    };

    Ok(Some((name, arg_types, ret_type)))
}

/// Process a defmacro form
/// Syntax: (defmacro name [params...] body)
fn process_defmacro(
    macro_expander: &mut MacroExpander,
    args: &[parser::Value],
) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() < 3 {
        return Err("defmacro requires: name, params vector, body".into());
    }

    // Parse macro name
    let name = match &args[0] {
        parser::Value::Symbol(s) => s.clone(),
        _ => return Err("defmacro name must be a symbol".into()),
    };

    // Parse parameters
    let mut params = vec![];
    if let parser::Value::Vector(param_vals) = &args[1] {
        for param in param_vals {
            if let parser::Value::Symbol(s) = param {
                params.push(s.clone());
            } else {
                return Err("defmacro parameters must be symbols".into());
            }
        }
    } else {
        return Err("defmacro parameters must be a vector".into());
    }

    // Body is the rest (for now, just the third element)
    let body = args[2].clone();

    macro_expander.define_macro(name, params, body);

    Ok(())
}

/// Emit a function definition from a defn form
/// Syntax: (defn name [arg:type ...] return-type body...)
fn emit_defn<'c>(
    emitter: &mut Emitter<'c>,
    module: &melior::ir::Module<'c>,
    args: &[parser::Value],
    registry: &FunctionRegistry<'c>,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() < 3 {
        return Err("defn requires at least: name, args vector, return type".into());
    }

    // Parse function name
    let name = match &args[0] {
        parser::Value::Symbol(s) => s.as_str(),
        _ => return Err("defn name must be a symbol".into()),
    };

    // Parse arguments [x:i32 y:i32] or [x y] (defaults to i32)
    let mut arg_names = vec![];
    let mut arg_types = vec![];
    if let parser::Value::Vector(params) = &args[1] {
        for param in params {
            if let parser::Value::Symbol(param_str) = param {
                // Parse "x:i32" format or just "x" (defaults to i32)
                let parts: Vec<&str> = param_str.split(':').collect();
                if parts.len() == 2 {
                    arg_names.push(parts[0].to_string());
                    arg_types.push(emitter.parse_type(parts[1])?);
                } else if parts.len() == 1 {
                    // Default to i32
                    arg_names.push(parts[0].to_string());
                    arg_types.push(emitter.parse_type("i32")?);
                } else {
                    return Err(format!("Argument must be name or name:type, got: {}", param_str).into());
                }
            } else {
                return Err("Arguments must be symbols".into());
            }
        }
    } else {
        return Err("defn arguments must be a vector".into());
    }

    // Parse return type - if args[2] is a symbol that looks like a type, use it
    // Otherwise, default to i32 and args[2] is part of the body
    let (ret_type, body_start) = if args.len() > 2 {
        if let parser::Value::Symbol(s) = &args[2] {
            // Check if this looks like a type
            match emitter.parse_type(s) {
                Ok(t) => (t, 3),
                Err(_) => (emitter.parse_type("i32")?, 2), // Not a type, part of body
            }
        } else {
            // Not a symbol, must be part of body
            (emitter.parse_type("i32")?, 2)
        }
    } else {
        // Only 2 args means args[2] onwards is the body (starting at index 2)
        (emitter.parse_type("i32")?, 2)
    };

    // Body starts at body_start
    let body = &args[body_start..];

    if body.is_empty() {
        return Err("defn requires a body".into());
    }

    // Check if body uses blocks or is a single expression
    let has_blocks = body.iter().any(|v| {
        if let parser::Value::List(elements) = v {
            if let Some(parser::Value::Symbol(s)) = elements.first() {
                return s == "block";
            }
        }
        false
    });

    // Check if body is a single expression (natural syntax)
    let is_single_expr = body.len() == 1 && !has_blocks;

    if is_single_expr {
        // Expression-based function
        emit_expr_function(emitter, module, name, &arg_names, &arg_types, ret_type, &body[0], registry)?;
    } else if has_blocks {
        emitter.emit_function_with_blocks_and_args(module, name, &arg_names, &arg_types, ret_type, body)?;
    } else {
        emitter.emit_function_with_args(module, name, &arg_names, &arg_types, ret_type, body)?;
    }

    Ok(())
}

/// Emit a function with a single expression body (natural syntax)
fn emit_expr_function<'c>(
    emitter: &mut Emitter<'c>,
    module: &melior::ir::Module<'c>,
    name: &str,
    arg_names: &[String],
    arg_types: &[melior::ir::Type<'c>],
    ret_type: melior::ir::Type<'c>,
    body_expr: &parser::Value,
    registry: &FunctionRegistry<'c>,
) -> Result<(), Box<dyn std::error::Error>> {
    use melior::ir::{
        attribute::TypeAttribute,
        operation::OperationBuilder,
        r#type::FunctionType,
        Block, BlockLike, Identifier, Location, Region, RegionLike,
    };

    // Create function type
    let func_type = FunctionType::new(emitter.context(), arg_types, &[ret_type]);

    // Create function
    let region = Region::new();
    let block_args: Vec<_> = arg_types.iter()
        .map(|t| (*t, Location::unknown(emitter.context())))
        .collect();
    let entry_block_value = Block::new(&block_args);
    region.append_block(entry_block_value);

    // Get the block reference
    let entry_block = region.first_block()
        .ok_or("Failed to get entry block")?;

    // Register function arguments in symbol table
    for (i, arg_name) in arg_names.iter().enumerate() {
        if let Ok(arg_val) = entry_block.argument(i) {
            emitter.register_value(arg_name.clone(), arg_val.into());
        }
    }

    // Compile the expression
    let result_name = ExprCompiler::compile_expr(emitter, &entry_block, body_expr, registry, None)?;

    // Emit return
    let return_val = emitter.get_value(&result_name)
        .ok_or(format!("Cannot find result value: {}", result_name))?;

    let return_op = OperationBuilder::new("func.return", Location::unknown(emitter.context()))
        .add_operands(&[return_val])
        .build()
        .map_err(|e| format!("Failed to build return: {:?}", e))?;

    unsafe { entry_block.append_operation(return_op); }

    let function = OperationBuilder::new("func.func", Location::unknown(emitter.context()))
        .add_attributes(&[
            (
                Identifier::new(emitter.context(), "sym_name"),
                melior::ir::attribute::StringAttribute::new(emitter.context(), name).into(),
            ),
            (
                Identifier::new(emitter.context(), "function_type"),
                TypeAttribute::new(func_type.into()).into(),
            ),
        ])
        .add_regions([region])
        .build()
        .map_err(|e| format!("Failed to build function: {:?}", e))?;

    module.body().append_operation(function);

    Ok(())
}

/// Detect the return type by finding the type of value being returned
fn detect_return_type<'c>(emitter: &Emitter<'c>, values: &[parser::Value]) -> Option<Type<'c>> {
    // Find all operations that produce results and track their types
    let mut value_types: std::collections::HashMap<String, String> = std::collections::HashMap::new();

    fn scan_for_types(values: &[parser::Value], types_map: &mut std::collections::HashMap<String, String>) {
        for value in values {
            if let parser::Value::List(elements) = value {
                if let Some(parser::Value::Symbol(s)) = elements.first() {
                    if s == "op" {
                        // Look for :as and :results
                        let mut result_name = None;
                        let mut result_type = None;

                        for i in 0..elements.len() - 1 {
                            if let parser::Value::Keyword(kw) = &elements[i] {
                                if kw == "as" {
                                    if let Some(parser::Value::Symbol(name)) = elements.get(i + 1) {
                                        result_name = Some(name.clone());
                                    }
                                } else if kw == "results" {
                                    if let Some(parser::Value::Vector(types)) = elements.get(i + 1) {
                                        if let Some(parser::Value::Symbol(type_name)) = types.first() {
                                            result_type = Some(type_name.clone());
                                        }
                                    }
                                }
                            }
                        }

                        if let (Some(name), Some(ty)) = (result_name, result_type) {
                            types_map.insert(name, ty);
                        }
                    } else if s == "block" {
                        // Recurse into block operations
                        scan_for_types(&elements[2..], types_map);
                    }
                }
            }
        }
    }

    scan_for_types(values, &mut value_types);

    // Now find func.return and see what it's returning
    fn find_return_type(values: &[parser::Value], types_map: &std::collections::HashMap<String, String>) -> Option<String> {
        for value in values {
            if let parser::Value::List(elements) = value {
                if let Some(parser::Value::Symbol(s)) = elements.first() {
                    if s == "op" {
                        // Check if this is func.return
                        if let Some(parser::Value::Symbol(op_name)) = elements.get(1) {
                            if op_name == "func.return" {
                                // Find the operand being returned
                                for i in 0..elements.len() - 1 {
                                    if let parser::Value::Keyword(kw) = &elements[i] {
                                        if kw == "operands" {
                                            if let Some(parser::Value::Vector(operands)) = elements.get(i + 1) {
                                                if let Some(parser::Value::Symbol(operand_name)) = operands.first() {
                                                    return types_map.get(operand_name).cloned();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if s == "block" {
                        if let Some(ty) = find_return_type(&elements[2..], types_map) {
                            return Some(ty);
                        }
                    }
                }
            }
        }
        None
    }

    if let Some(type_name) = find_return_type(values, &value_types) {
        return emitter.parse_type(&type_name).ok();
    }

    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for file argument
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        // Use SelfContainedCompiler for file-based execution
        let filename = &args[1];
        println!("Loading: {}\n", filename);

        let code = fs::read_to_string(filename)?;

        // Create a new compiler (it creates its own context internally)
        let mut compiler = SelfContainedCompiler::new();

        // Evaluate the file
        match compiler.eval_string(&code) {
            Ok(_) => {
                println!("\n✅ Program completed successfully!");
                return Ok(());
            }
            Err(e) => {
                eprintln!("\n❌ Error: {}", e);
                return Err(e.into());
            }
        }
    }

    // Legacy path: No file specified, run built-in example
    println!("No file specified, running built-in example");
    println!("Usage: mlir-lisp <file.lisp>\n");

    let code = r#"
        (op arith.constant
            :attrs {:value 10}
            :results [i32]
            :as %ten)
        (op arith.constant
            :attrs {:value 32}
            :results [i32]
            :as %thirty_two)
        (op arith.addi
            :operands [%ten %thirty_two]
            :results [i32]
            :as %result)
        (op func.return
            :operands [%result])
        "#.to_string();

    // Parse, emit, and JIT execute
    let ctx = MlirContext::new();
    let mut module = ctx.create_module();
    let mut emitter = Emitter::new(&ctx);

    let (rest, values) = parser::parse(&code)
        .map_err(|e| format!("Parse error: {:?}", e))?;

    println!("Parsed {} expressions", values.len());

    // Process defmacro forms and expand macros
    let mut macro_expander = MacroExpander::new();

    // First pass: collect macro definitions
    for value in &values {
        if let parser::Value::List(elements) = value {
            if let Some(parser::Value::Symbol(s)) = elements.first() {
                if s == "defmacro" {
                    process_defmacro(&mut macro_expander, &elements[1..])?;
                }
            }
        }
    }

    // Second pass: expand all macros
    let values = macro_expander.expand_all(&values)?;

    println!("After macro expansion: {} expressions", values.len());
    if !rest.is_empty() {
        println!("WARNING: Unparsed input remaining ({} bytes)", rest.len());
        println!("First 200 chars:\n{}", &rest[..200.min(rest.len())]);
    }
    for (i, val) in values.iter().enumerate() {
        if let parser::Value::List(elements) = val {
            if let Some(parser::Value::Symbol(s)) = elements.first() {
                if s == "block" {
                    if let Some(parser::Value::Symbol(name)) = elements.get(1) {
                        println!("  {}: block {}", i, name);
                        continue;
                    }
                }
            }
        }
        println!("  {}: {:?}", i, &val);
    }
    println!();

    // PHASE 1: Register all function signatures (for forward references and recursion)
    let mut registry = FunctionRegistry::new();
    for value in &values {
        if let parser::Value::List(elements) = value {
            if let Some(parser::Value::Symbol(s)) = elements.first() {
                if s == "defn" {
                    // Parse function signature and register it
                    match parse_defn_signature(&emitter, &elements[1..]) {
                        Ok(Some((name, arg_types, ret_type))) => {
                            registry.register(name, arg_types, ret_type);
                        }
                        Ok(None) => {}
                        Err(e) => {
                            return Err(format!("Error parsing defn signature: {}", e).into());
                        }
                    }
                }
            }
        }
    }

    // Check if we have defn forms (function definitions)
    let has_defn = values.iter().any(|v| {
        if let parser::Value::List(elements) = v {
            if let Some(parser::Value::Symbol(s)) = elements.first() {
                return s == "defn";
            }
        }
        false
    });

    let main_return_type = if has_defn {
        // Process each defn form and track main's return type
        let mut main_ret_type = None;
        for value in &values {
            if let parser::Value::List(elements) = value {
                if let Some(parser::Value::Symbol(s)) = elements.first() {
                    if s == "defn" {
                        // Check if this is the main function
                        if let Some(parser::Value::Symbol(func_name)) = elements.get(1) {
                            if func_name == "main" && elements.len() >= 4 {
                                // Parse return type (3rd argument)
                                if let parser::Value::Symbol(ret_type_str) = &elements[3] {
                                    main_ret_type = emitter.parse_type(ret_type_str).ok();
                                }
                            }
                        }
                        emit_defn(&mut emitter, &module, &elements[1..], &registry)?;
                    }
                }
            }
        }
        main_ret_type.unwrap_or_else(|| emitter.parse_type("i32").unwrap())
    } else {
        // Legacy mode: treat all top-level forms as main function body
        let ret_type = detect_return_type(&emitter, &values)
            .unwrap_or_else(|| emitter.parse_type("i32").unwrap());

        // Check if we have block-based control flow
        let has_blocks = values.iter().any(|v| {
            if let parser::Value::List(elements) = v {
                if let Some(parser::Value::Symbol(s)) = elements.first() {
                    return s == "block";
                }
            }
            false
        });

        // Create function with appropriate method
        if has_blocks {
            emitter.emit_function_with_blocks(&module, "main", &[], ret_type, &values)?;
        } else {
            emitter.emit_function(&module, "main", &[], ret_type, &values)?;
        }
        ret_type
    };

    println!("Generated MLIR:");
    println!("{}\n", module.as_operation());

    // Lower to LLVM dialect and execute
    println!("Lowering to LLVM...");
    let pm = PassManager::new(ctx.context());
    // First convert SCF (Structured Control Flow) to CF (Control Flow)
    pm.add_pass(melior::pass::conversion::create_scf_to_control_flow());
    // Then lower everything to LLVM
    pm.add_pass(melior::pass::conversion::create_arith_to_llvm());
    pm.add_pass(melior::pass::conversion::create_control_flow_to_llvm());
    pm.add_pass(melior::pass::conversion::create_func_to_llvm());
    pm.run(&mut module)?;

    println!("After LLVM lowering:");
    println!("{}\n", module.as_operation());

    // JIT compile and execute
    println!("JIT compiling and executing...");
    let engine = ExecutionEngine::new(&module, 2, &[], false);

    unsafe {
        let func_ptr = engine.lookup("main");
        if func_ptr.is_null() {
            return Err("Failed to find main function".into());
        }

        // Execute based on return type
        let ret_type_str = format!("{}", main_return_type);
        if ret_type_str.contains("f64") {
            type MainFn = unsafe extern "C" fn() -> f64;
            let main_fn: MainFn = std::mem::transmute(func_ptr);
            let result = main_fn();
            println!("✨ Execution result: {}", result);
        } else if ret_type_str.contains("f32") {
            type MainFn = unsafe extern "C" fn() -> f32;
            let main_fn: MainFn = std::mem::transmute(func_ptr);
            let result = main_fn();
            println!("✨ Execution result: {}", result);
        } else if ret_type_str.contains("i1") {
            type MainFn = unsafe extern "C" fn() -> bool;
            let main_fn: MainFn = std::mem::transmute(func_ptr);
            let result = main_fn();
            println!("✨ Execution result: {}", result);
        } else {
            // Default to i32 for integer types
            type MainFn = unsafe extern "C" fn() -> i32;
            let main_fn: MainFn = std::mem::transmute(func_ptr);
            let result = main_fn();
            println!("✨ Execution result: {}", result);
        }

        println!("✅ Program executed successfully!");
    }

    Ok(())
}
