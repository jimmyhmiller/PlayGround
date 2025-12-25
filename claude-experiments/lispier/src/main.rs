use std::env;
use std::path::Path;
use std::process::ExitCode;

use lispier::{
    extract_compilation, extract_externs, extract_link_libraries, find_project_root,
    AttributeValue, DialectRegistry, IRGenerator, Jit, MacroExpander, ModuleLoader, Node,
    Operation, Parser, Reader, RuntimeEnv, Tokenizer,
};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("run") => {
            if let Some(file) = args.get(2) {
                run_file(file)
            } else {
                eprintln!("Usage: lispier run <file>");
                ExitCode::FAILURE
            }
        }
        Some("show-ast") => {
            if let Some(file) = args.get(2) {
                show_ast(file)
            } else {
                eprintln!("Usage: lispier show-ast <file>");
                ExitCode::FAILURE
            }
        }
        Some("show-ir") => {
            if let Some(file) = args.get(2) {
                show_ir(file)
            } else {
                eprintln!("Usage: lispier show-ir <file>");
                ExitCode::FAILURE
            }
        }
        Some("show-expanded") => {
            if let Some(file) = args.get(2) {
                show_expanded(file)
            } else {
                eprintln!("Usage: lispier show-expanded <file>");
                ExitCode::FAILURE
            }
        }
        Some("eval") => {
            if let Some(expr) = args.get(2) {
                eval_expr(expr)
            } else {
                eprintln!("Usage: lispier eval <expression>");
                ExitCode::FAILURE
            }
        }
        Some("--help") | Some("-h") | None => {
            print_usage();
            ExitCode::SUCCESS
        }
        Some(cmd) => {
            eprintln!("Unknown command: {}", cmd);
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!("Lispier - A Lisp-to-MLIR Compiler");
    println!();
    println!("USAGE:");
    println!("    lispier <command> [args]");
    println!();
    println!("COMMANDS:");
    println!("    run <file>           Compile and execute a file");
    println!("    show-ast <file>      Show the AST for a file");
    println!("    show-ir <file>       Show the generated MLIR for a file");
    println!("    show-expanded <file> Show macro-expanded source");
    println!("    eval <expr>          Evaluate an expression");
    println!("    --help, -h           Show this help message");
}

fn run_file(path: &str) -> ExitCode {
    let file_path = Path::new(path);
    let project_root = find_project_root(file_path)
        .unwrap_or_else(|| file_path.parent().unwrap_or(Path::new(".")).to_path_buf());

    let mut loader = ModuleLoader::new(project_root);

    let nodes = match loader.load(file_path) {
        Ok(nodes) => nodes,
        Err(e) => {
            eprintln!("Module loading error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Check if there's a compilation spec - if so, use external pipeline
    if let Some(compilation) = extract_compilation(&nodes) {
        return run_with_compilation_spec(&nodes, &compilation);
    }

    // No compilation spec - use internal JIT
    match compile_and_run_nodes(&nodes) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}

/// Run with an external compilation pipeline (for GPU, etc.)
fn run_with_compilation_spec(nodes: &[lispier::Node], compilation: &lispier::Compilation) -> ExitCode {
    // Detect runtime environment
    let runtime = match RuntimeEnv::detect() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Runtime detection error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Check if we have a matching target
    if compilation.get_target(runtime.backend.name()).is_none() {
        eprintln!(
            "No compilation target for backend '{}' in file",
            runtime.backend.name()
        );
        eprintln!("Available targets: {:?}",
            compilation.targets.iter().map(|t| &t.backend).collect::<Vec<_>>()
        );
        return ExitCode::FAILURE;
    }

    eprintln!("Using {} backend", runtime.backend.name());
    if let Some(ref chip) = runtime.chip {
        eprintln!("Detected chip: {}", chip);
    }

    // Generate initial MLIR
    let mlir_ir = match generate_ir_from_nodes(nodes) {
        Ok(ir) => ir,
        Err(e) => {
            eprintln!("IR generation error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Compile and run
    match runtime.compile_and_run(&mlir_ir, compilation) {
        Ok(output) => {
            print!("{}", output);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Execution error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn show_ast(path: &str) -> ExitCode {
    let file_path = Path::new(path);
    let project_root = find_project_root(file_path)
        .unwrap_or_else(|| file_path.parent().unwrap_or(Path::new(".")).to_path_buf());

    let mut loader = ModuleLoader::new(project_root);

    match loader.load(file_path) {
        Ok(nodes) => {
            for node in &nodes {
                println!("{:#?}", node);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Module loading error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn show_ir(path: &str) -> ExitCode {
    let file_path = Path::new(path);
    let project_root = find_project_root(file_path)
        .unwrap_or_else(|| file_path.parent().unwrap_or(Path::new(".")).to_path_buf());

    let mut loader = ModuleLoader::new(project_root);

    match loader.load(file_path) {
        Ok(nodes) => match generate_ir_from_nodes(&nodes) {
            Ok(ir_str) => {
                println!("{}", ir_str);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("Module loading error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn show_expanded(path: &str) -> ExitCode {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read file: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Tokenize
    let mut tokenizer = Tokenizer::new(&source);
    let tokens = match tokenizer.tokenize() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Tokenizer error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Read
    let mut reader = Reader::new(&tokens);
    let values = match reader.read() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Reader error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Macro expansion
    let mut expander = MacroExpander::new();
    let expanded = match expander.expand_all(&values) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Macro error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Print expanded forms
    for value in &expanded {
        println!("{}", value);
    }

    ExitCode::SUCCESS
}

fn eval_expr(expr: &str) -> ExitCode {
    match compile_and_run(expr) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}

/// Parse source code to AST nodes
fn parse_to_ast(source: &str) -> Result<Vec<lispier::Node>, String> {
    // Tokenize
    let mut tokenizer = Tokenizer::new(source);
    let tokens = tokenizer.tokenize().map_err(|e| format!("Tokenizer error: {}", e))?;

    // Read
    let mut reader = Reader::new(&tokens);
    let values = reader.read().map_err(|e| format!("Reader error: {}", e))?;

    // Parse
    let mut parser = Parser::new();
    let nodes = parser.parse(&values).map_err(|e| format!("Parser error: {}", e))?;

    Ok(nodes)
}

/// Compile and run source code
fn compile_and_run(source: &str) -> Result<(), String> {
    let nodes = parse_to_ast(source)?;
    compile_and_run_nodes(&nodes)
}

/// Return type of a main function
#[derive(Debug, Clone)]
enum MainReturnType {
    I32,
    I64,
    F32,
    F64,
    Void,
}

/// Find the main function in AST nodes and extract its return type
fn find_main_return_type(nodes: &[Node]) -> Option<MainReturnType> {
    for node in nodes {
        if let Some(ret_type) = find_main_in_node(node) {
            return Some(ret_type);
        }
    }
    None
}

/// Recursively search for main function in a node
fn find_main_in_node(node: &Node) -> Option<MainReturnType> {
    match node {
        Node::Operation(op) => {
            if is_main_function(op) {
                return extract_return_type(op);
            }
            // Search in regions
            for region in &op.regions {
                for block in &region.blocks {
                    for child in &block.operations {
                        if let Some(ret_type) = find_main_in_node(child) {
                            return Some(ret_type);
                        }
                    }
                }
            }
            None
        }
        Node::Module(module) => {
            for child in &module.body {
                if let Some(ret_type) = find_main_in_node(child) {
                    return Some(ret_type);
                }
            }
            None
        }
        _ => None,
    }
}

/// Check if an operation is a main function
fn is_main_function(op: &Operation) -> bool {
    if op.qualified_name() != "func.func" {
        return false;
    }

    match op.attributes.get("sym_name") {
        Some(AttributeValue::String(name)) => name == "main",
        _ => false,
    }
}

/// Extract return type from a function operation
fn extract_return_type(op: &Operation) -> Option<MainReturnType> {
    let func_type = match op.attributes.get("function_type") {
        Some(AttributeValue::FunctionType(ft)) => ft,
        _ => return Some(MainReturnType::Void),
    };

    if func_type.return_types.is_empty() {
        return Some(MainReturnType::Void);
    }

    // Get the first return type
    let ret_type_name = &func_type.return_types[0].name;

    match ret_type_name.as_str() {
        "i32" => Some(MainReturnType::I32),
        "i64" => Some(MainReturnType::I64),
        "f32" => Some(MainReturnType::F32),
        "f64" => Some(MainReturnType::F64),
        _ => {
            eprintln!("Warning: unsupported main return type '{}', treating as void", ret_type_name);
            Some(MainReturnType::Void)
        }
    }
}

/// Compile and run pre-parsed nodes
fn compile_and_run_nodes(nodes: &[Node]) -> Result<(), String> {
    // Find main's return type before generating IR
    let return_type = find_main_return_type(nodes);

    // Extract extern declarations
    let externs = extract_externs(nodes);

    // Extract link library declarations
    let link_libs = extract_link_libraries(nodes);

    // Generate MLIR
    let registry = DialectRegistry::new();
    let generator = IRGenerator::new(&registry);
    let mut module = generator
        .generate(nodes)
        .map_err(|e| format!("IR generation error: {}", e))?;

    // Separate link libraries into well-known ones (like :c) and file paths
    let mut needs_libc = false;
    let mut file_libs: Vec<String> = Vec::new();
    for lib in &link_libs {
        match lib.library.as_str() {
            ":c" | ":libc" | ":m" | ":libm" => needs_libc = true,
            _ => file_libs.push(lib.resolve_path()),
        }
    }

    // Create JIT with optional library paths (for actual library files)
    let jit = if file_libs.is_empty() {
        Jit::new(&registry, &mut module)
            .map_err(|e| format!("JIT creation error: {}", e))?
    } else {
        let lib_path_refs: Vec<&str> = file_libs.iter().map(|s| s.as_str()).collect();
        Jit::with_libraries(&registry, &mut module, &lib_path_refs)
            .map_err(|e| format!("JIT creation error: {}", e))?
    };

    // Register libc symbols if needed
    if needs_libc {
        unsafe {
            jit.register_libc();
        }
    }

    // Register FFI symbols based on extern declarations
    for ext in &externs {
        // Keywords may be stored with or without the leading colon
        let lib = ext.library.trim_start_matches(':');
        match lib {
            "value-ffi" => {
                unsafe {
                    jit.register_value_ffi();
                }
            }
            other => {
                return Err(format!("Unknown extern library: {}", other));
            }
        }
    }

    // Use invoke_packed for uniform calling convention
    // The llvm-request-c-wrappers pass generates _mlir_ciface_main which invoke_packed uses
    match return_type {
        Some(MainReturnType::I32) => {
            let mut result: i32 = 0;
            unsafe {
                jit.invoke_packed("main", &mut [&mut result as *mut i32 as *mut ()])
                    .map_err(|e| format!("Invocation error: {}", e))?;
            }
            println!("{}", result);
        }
        Some(MainReturnType::I64) => {
            let mut result: i64 = 0;
            unsafe {
                jit.invoke_packed("main", &mut [&mut result as *mut i64 as *mut ()])
                    .map_err(|e| format!("Invocation error: {}", e))?;
            }
            println!("{}", result);
        }
        Some(MainReturnType::F32) => {
            let mut result: f32 = 0.0;
            unsafe {
                jit.invoke_packed("main", &mut [&mut result as *mut f32 as *mut ()])
                    .map_err(|e| format!("Invocation error: {}", e))?;
            }
            println!("{}", result);
        }
        Some(MainReturnType::F64) => {
            let mut result: f64 = 0.0;
            unsafe {
                jit.invoke_packed("main", &mut [&mut result as *mut f64 as *mut ()])
                    .map_err(|e| format!("Invocation error: {}", e))?;
            }
            println!("{}", result);
        }
        Some(MainReturnType::Void) | None => {
            unsafe {
                jit.invoke_packed("main", &mut [])
                    .map_err(|e| format!("Invocation error: {}", e))?;
            }
        }
    }

    Ok(())
}

/// Generate MLIR IR from pre-parsed nodes
fn generate_ir_from_nodes(nodes: &[lispier::Node]) -> Result<String, String> {
    // Generate MLIR
    let registry = DialectRegistry::new();
    let generator = IRGenerator::new(&registry);
    let module = generator
        .generate(nodes)
        .map_err(|e| format!("IR generation error: {}", e))?;

    Ok(format!("{}", module.as_operation()))
}

