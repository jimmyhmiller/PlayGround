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
                // Collect any additional arguments after the file path
                let program_args: Vec<String> = args.iter().skip(3).cloned().collect();
                run_file(file, &program_args)
            } else {
                eprintln!("Usage: lispier run <file> [args...]");
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
    println!("    run <file> [args...] Compile and execute a file with optional arguments");
    println!("    show-ast <file>      Show the AST for a file");
    println!("    show-ir <file>       Show the generated MLIR for a file");
    println!("    show-expanded <file> Show macro-expanded source");
    println!("    eval <expr>          Evaluate an expression");
    println!("    --help, -h           Show this help message");
}

fn run_file(path: &str, program_args: &[String]) -> ExitCode {
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
    match compile_and_run_nodes(&nodes, program_args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}

/// Run with a compilation pipeline (GPU, custom passes, etc.) using JIT
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
    let target = match compilation.get_target(runtime.backend.name()) {
        Some(t) => t,
        None => {
            eprintln!(
                "No compilation target for backend '{}' in file",
                runtime.backend.name()
            );
            eprintln!("Available targets: {:?}",
                compilation.targets.iter().map(|t| &t.backend).collect::<Vec<_>>()
            );
            return ExitCode::FAILURE;
        }
    };

    eprintln!("Using {} backend", runtime.backend.name());
    if let Some(ref chip) = runtime.chip {
        eprintln!("Detected chip: {}", chip);
    }

    // Build pass pipeline from compilation spec
    // Some passes need to run inside gpu.module scope
    let runtime_attrs = runtime.runtime_attrs();

    // Passes that run on gpu.module (must be nested)
    // Note: rocdl-attach-target and nvvm-attach-target run on builtin.module level
    let gpu_module_passes = ["convert-gpu-to-rocdl", "convert-gpu-to-nvvm"];

    let mut before_gpu: Vec<String> = Vec::new();
    let mut gpu_passes: Vec<String> = Vec::new();
    let mut after_gpu: Vec<String> = Vec::new();
    let mut seen_gpu_pass = false;

    for pass in &target.passes {
        let pass_str = pass.to_pipeline_string(&runtime_attrs);
        let is_gpu_pass = gpu_module_passes.iter().any(|&p| pass.name == p);

        if is_gpu_pass {
            seen_gpu_pass = true;
            gpu_passes.push(pass_str);
        } else if !seen_gpu_pass {
            before_gpu.push(pass_str);
        } else {
            after_gpu.push(pass_str);
        }
    }

    // Construct pipeline string for JIT
    // Wrap everything in builtin.module() and add the c-wrappers pass
    // Nest gpu.module passes properly
    let pipeline = if gpu_passes.is_empty() {
        format!(
            "builtin.module(func.func(llvm-request-c-wrappers),{})",
            before_gpu.into_iter().chain(after_gpu).collect::<Vec<_>>().join(",")
        )
    } else {
        format!(
            "builtin.module(func.func(llvm-request-c-wrappers),{},gpu.module({}),{})",
            before_gpu.join(","),
            gpu_passes.join(","),
            after_gpu.join(",")
        )
    };

    eprintln!("Pipeline: {}", pipeline);

    // Generate MLIR from AST
    let registry = DialectRegistry::new();
    let generator = IRGenerator::new(&registry);
    let mut module = match generator.generate(nodes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("IR generation error: {}", e);
            return ExitCode::FAILURE;
        }
    };

    // Get runtime library paths
    // Add null terminators since MLIR's library loading expects null-terminated strings
    let mut lib_paths: Vec<String> = runtime.runtime_libs
        .iter()
        .map(|p| {
            let mut s = p.to_string_lossy().to_string();
            s.push('\0');
            s
        })
        .collect();

    // Add the ciface shim library for GPU runtime wrappers
    if matches!(runtime.backend, lispier::runtime::Backend::Rocm) {
        let shim_path = std::env::var("HOME")
            .map(|h| format!("{}/libgpu_ciface_shim.so\0", h))
            .unwrap_or_else(|_| "/home/jimmyhmiller/libgpu_ciface_shim.so\0".to_string());
        lib_paths.push(shim_path);
    }

    eprintln!("Library paths: {:?}", lib_paths);
    let lib_path_refs: Vec<&str> = lib_paths.iter().map(|s| s.as_str()).collect();

    // Pre-load GPU runtime library with RTLD_GLOBAL so symbols are available during JIT linking
    if matches!(runtime.backend, lispier::runtime::Backend::Rocm) {
        for lib in &runtime.runtime_libs {
            let lib_str = lib.to_string_lossy();
            if lib_str.contains("rocm_runtime") {
                eprintln!("Pre-loading: {}", lib_str);
                let path_cstr = std::ffi::CString::new(lib_str.as_ref()).unwrap();
                unsafe {
                    let handle = libc::dlopen(path_cstr.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL);
                    if handle.is_null() {
                        eprintln!("Warning: Failed to pre-load {}", lib_str);
                    } else {
                        eprintln!("Successfully pre-loaded: {}", lib_str);
                    }
                }
            }
        }

        // Pre-load the ciface shim library
        let shim_path = std::env::var("HOME")
            .map(|h| format!("{}/libgpu_ciface_shim.so", h))
            .unwrap_or_else(|_| "/home/jimmyhmiller/libgpu_ciface_shim.so".to_string());
        eprintln!("Pre-loading shim: {}", shim_path);
        let path_cstr = std::ffi::CString::new(shim_path.as_str()).unwrap();
        unsafe {
            let handle = libc::dlopen(path_cstr.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL);
            if handle.is_null() {
                eprintln!("Warning: Failed to pre-load shim library");
            } else {
                eprintln!("Successfully pre-loaded shim library");
            }
        }
    }
    eprintln!("Pre-loading complete");

    // Create JIT with custom pipeline and libraries
    eprintln!("Running pass pipeline...");
    use std::io::Write;
    std::io::stderr().flush().unwrap();
    if let Err(e) = Jit::run_pipeline(&registry, &mut module, &pipeline) {
        eprintln!("Pass pipeline error: {}", e);
        return ExitCode::FAILURE;
    }
    eprintln!("Pass pipeline completed successfully");
    eprintln!("Module after passes:\n{}", module.as_operation());

    eprintln!("Creating execution engine...");
    let jit = match Jit::with_pipeline_and_libraries(&registry, &mut module, "", &lib_path_refs) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("JIT creation error: {}", e);
            return ExitCode::FAILURE;
        }
    };
    eprintln!("JIT created successfully");

    // Register GPU runtime symbols if using GPU backend
    if matches!(runtime.backend, lispier::runtime::Backend::Rocm) {
        // Find the rocm runtime library and register its symbols
        // Use the original paths (without null terminator) for dlopen
        for lib in &runtime.runtime_libs {
            let lib_str = lib.to_string_lossy();
            if lib_str.contains("rocm_runtime") {
                eprintln!("Registering GPU runtime from: {}", lib_str);
                unsafe {
                    jit.register_gpu_runtime(&lib_str);
                }
                break;
            }
        }
    }

    // Verify symbol registration worked
    let test_symbol = "_mlir_ciface_mgpuMemGetDeviceMemRef1dFloat";
    if let Some(ptr) = jit.lookup(test_symbol) {
        eprintln!("Successfully looked up {} at {:p}", test_symbol, ptr);
    } else {
        eprintln!("Failed to look up {}", test_symbol);
    }

    // Invoke main
    unsafe {
        match jit.invoke_packed("main", &mut []) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("Execution error: {}", e);
                ExitCode::FAILURE
            }
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
    compile_and_run_nodes(&nodes, &[])
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

/// Signature of a main function - return type and whether it takes argc/argv
#[derive(Debug, Clone)]
struct MainSignature {
    return_type: MainReturnType,
    takes_args: bool, // true if main accepts (i32, !llvm.ptr) or (i64, !llvm.ptr)
}

/// Find the main function in AST nodes and extract its signature
fn find_main_signature(nodes: &[Node]) -> Option<MainSignature> {
    for node in nodes {
        if let Some(sig) = find_main_in_node(node) {
            return Some(sig);
        }
    }
    None
}

/// Recursively search for main function in a node
fn find_main_in_node(node: &Node) -> Option<MainSignature> {
    match node {
        Node::Operation(op) => {
            if is_main_function(op) {
                return extract_signature(op);
            }
            // Search in regions
            for region in &op.regions {
                for block in &region.blocks {
                    for child in &block.operations {
                        if let Some(sig) = find_main_in_node(child) {
                            return Some(sig);
                        }
                    }
                }
            }
            None
        }
        Node::Module(module) => {
            for child in &module.body {
                if let Some(sig) = find_main_in_node(child) {
                    return Some(sig);
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

/// Extract full signature from a function operation
fn extract_signature(op: &Operation) -> Option<MainSignature> {
    let func_type = match op.attributes.get("function_type") {
        Some(AttributeValue::FunctionType(ft)) => ft,
        _ => return Some(MainSignature {
            return_type: MainReturnType::Void,
            takes_args: false,
        }),
    };

    // Check if main takes argc/argv arguments
    // Expected signature: (i32, !llvm.ptr) or (i64, !llvm.ptr)
    let takes_args = if func_type.arg_types.len() == 2 {
        let arg0 = &func_type.arg_types[0].name;
        let arg1 = &func_type.arg_types[1].name;
        (arg0 == "i32" || arg0 == "i64") && arg1 == "!llvm.ptr"
    } else {
        false
    };

    // Extract return type
    let return_type = if func_type.return_types.is_empty() {
        MainReturnType::Void
    } else {
        let ret_type_name = &func_type.return_types[0].name;
        match ret_type_name.as_str() {
            "i32" => MainReturnType::I32,
            "i64" => MainReturnType::I64,
            "f32" => MainReturnType::F32,
            "f64" => MainReturnType::F64,
            _ => {
                eprintln!("Warning: unsupported main return type '{}', treating as void", ret_type_name);
                MainReturnType::Void
            }
        }
    };

    Some(MainSignature { return_type, takes_args })
}

/// Compile and run pre-parsed nodes with optional program arguments
fn compile_and_run_nodes(nodes: &[Node], program_args: &[String]) -> Result<(), String> {
    // Find main's signature before generating IR
    let signature = find_main_signature(nodes);

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

    // Prepare argc/argv if main takes arguments
    let takes_args = signature.as_ref().map_or(false, |s| s.takes_args);

    // Build argv array: null-terminated array of null-terminated strings
    // We need to keep the CStrings alive for the duration of the call
    let c_strings: Vec<std::ffi::CString> = program_args
        .iter()
        .map(|s| std::ffi::CString::new(s.as_str()).unwrap())
        .collect();
    let mut argv_ptrs: Vec<*const i8> = c_strings.iter().map(|s| s.as_ptr()).collect();
    argv_ptrs.push(std::ptr::null()); // null-terminate the array

    let mut argc: i64 = program_args.len() as i64;
    let mut argv: *const *const i8 = argv_ptrs.as_ptr();

    // Use invoke_packed for uniform calling convention
    // The llvm-request-c-wrappers pass generates _mlir_ciface_main which invoke_packed uses
    let return_type = signature.as_ref().map(|s| s.return_type.clone());

    match return_type {
        Some(MainReturnType::I32) => {
            let mut result: i32 = 0;
            unsafe {
                if takes_args {
                    jit.invoke_packed("main", &mut [
                        &mut argc as *mut i64 as *mut (),
                        &mut argv as *mut *const *const i8 as *mut (),
                        &mut result as *mut i32 as *mut ()
                    ]).map_err(|e| format!("Invocation error: {}", e))?;
                } else {
                    jit.invoke_packed("main", &mut [&mut result as *mut i32 as *mut ()])
                        .map_err(|e| format!("Invocation error: {}", e))?;
                }
            }
            println!("{}", result);
        }
        Some(MainReturnType::I64) => {
            let mut result: i64 = 0;
            unsafe {
                if takes_args {
                    jit.invoke_packed("main", &mut [
                        &mut argc as *mut i64 as *mut (),
                        &mut argv as *mut *const *const i8 as *mut (),
                        &mut result as *mut i64 as *mut ()
                    ]).map_err(|e| format!("Invocation error: {}", e))?;
                } else {
                    jit.invoke_packed("main", &mut [&mut result as *mut i64 as *mut ()])
                        .map_err(|e| format!("Invocation error: {}", e))?;
                }
            }
            println!("{}", result);
        }
        Some(MainReturnType::F32) => {
            let mut result: f32 = 0.0;
            unsafe {
                if takes_args {
                    jit.invoke_packed("main", &mut [
                        &mut argc as *mut i64 as *mut (),
                        &mut argv as *mut *const *const i8 as *mut (),
                        &mut result as *mut f32 as *mut ()
                    ]).map_err(|e| format!("Invocation error: {}", e))?;
                } else {
                    jit.invoke_packed("main", &mut [&mut result as *mut f32 as *mut ()])
                        .map_err(|e| format!("Invocation error: {}", e))?;
                }
            }
            println!("{}", result);
        }
        Some(MainReturnType::F64) => {
            let mut result: f64 = 0.0;
            unsafe {
                if takes_args {
                    jit.invoke_packed("main", &mut [
                        &mut argc as *mut i64 as *mut (),
                        &mut argv as *mut *const *const i8 as *mut (),
                        &mut result as *mut f64 as *mut ()
                    ]).map_err(|e| format!("Invocation error: {}", e))?;
                } else {
                    jit.invoke_packed("main", &mut [&mut result as *mut f64 as *mut ()])
                        .map_err(|e| format!("Invocation error: {}", e))?;
                }
            }
            println!("{}", result);
        }
        Some(MainReturnType::Void) | None => {
            unsafe {
                if takes_args {
                    jit.invoke_packed("main", &mut [
                        &mut argc as *mut i64 as *mut (),
                        &mut argv as *mut *const *const i8 as *mut ()
                    ]).map_err(|e| format!("Invocation error: {}", e))?;
                } else {
                    jit.invoke_packed("main", &mut [])
                        .map_err(|e| format!("Invocation error: {}", e))?;
                }
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

