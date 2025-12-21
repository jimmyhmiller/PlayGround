use std::env;
use std::path::Path;
use std::process::ExitCode;

use lispier::{
    extract_compilation, find_project_root, DialectRegistry, IRGenerator, Jit, ModuleLoader,
    Parser, Reader, RuntimeEnv, Tokenizer,
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
    println!("    run <file>         Compile and execute a file");
    println!("    show-ast <file>    Show the AST for a file");
    println!("    show-ir <file>     Show the generated MLIR for a file");
    println!("    eval <expr>        Evaluate an expression");
    println!("    --help, -h         Show this help message");
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

/// Compile and run pre-parsed nodes
fn compile_and_run_nodes(nodes: &[lispier::Node]) -> Result<(), String> {
    // Generate MLIR
    let registry = DialectRegistry::new();
    let generator = IRGenerator::new(&registry);
    let mut module = generator
        .generate(nodes)
        .map_err(|e| format!("IR generation error: {}", e))?;

    // Create JIT and run
    let jit = Jit::new(&registry, &mut module)
        .map_err(|e| format!("JIT creation error: {}", e))?;

    // Look for a main function to invoke
    if let Some(_main_ptr) = jit.lookup("main") {
        // TODO: Invoke main function
        println!("Found main function, but invocation not yet implemented");
    } else {
        println!("No main function found");
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

