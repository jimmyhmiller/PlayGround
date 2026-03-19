mod ast;
mod codegen;
mod lexer;
mod parser;
mod typecheck;

use melior::ir::operation::OperationLike;
use std::collections::HashMap;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: simd-lang compile <file.simd> [-o <output_dir>]");
        eprintln!("       simd-lang compile <file.simd> --emit-mlir");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "compile" => cmd_compile(&args[2..]),
        other => {
            eprintln!("Unknown command: {}", other);
            std::process::exit(1);
        }
    }
}

fn cmd_compile(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: simd-lang compile <file.simd> [-o <output_dir>] [--emit-mlir]");
        std::process::exit(1);
    }

    let input_path = PathBuf::from(&args[0]);
    let stem = input_path
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    let mut output_dir = PathBuf::from(".");
    let mut emit_mlir = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--emit-mlir" => {
                emit_mlir = true;
            }
            other => {
                eprintln!("Unknown option: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Read source
    let source = std::fs::read_to_string(&input_path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", input_path.display(), e);
        std::process::exit(1);
    });

    // Parse
    let items = parser::parse(&source);

    // Type check
    if let Err(errors) = typecheck::typecheck(&items, &HashMap::new(), 8) {
        for e in &errors {
            eprintln!("{}: {}", input_path.display(), e);
        }
        std::process::exit(1);
    }

    // Compile to MLIR
    let ctx = codegen::create_context();
    let mut module = codegen::compile_module(&ctx, &items, &HashMap::new(), 8);

    if !module.as_operation().verify() {
        eprintln!("MLIR verification failed:");
        eprintln!("{}", module.as_operation());
        std::process::exit(1);
    }

    if emit_mlir {
        println!("{}", module.as_operation());
        return;
    }

    // Lower to LLVM
    codegen::lower_to_llvm(&ctx, &mut module).unwrap_or_else(|e| {
        eprintln!("LLVM lowering failed: {}", e);
        std::process::exit(1);
    });

    // Create output directory
    std::fs::create_dir_all(&output_dir).unwrap_or_else(|e| {
        eprintln!("Error creating output directory: {}", e);
        std::process::exit(1);
    });

    // Produce object file via ExecutionEngine
    let obj_path = output_dir.join(format!("{}.o", stem));
    let engine = melior::ExecutionEngine::new(&module, 2, &[], true);
    engine.dump_to_object_file(obj_path.to_str().unwrap());

    // Archive into static library
    let lib_path = output_dir.join(format!("lib{}.a", stem));
    let status = std::process::Command::new("ar")
        .args(["rcs", lib_path.to_str().unwrap(), obj_path.to_str().unwrap()])
        .status()
        .expect("failed to run ar");
    if !status.success() {
        eprintln!("ar failed");
        std::process::exit(1);
    }

    // Generate Rust bindings
    let rs_path = output_dir.join(format!("{}.rs", stem));
    let bindings = generate_rust_bindings(&stem, &items);
    std::fs::write(&rs_path, bindings).unwrap();

    // Clean up object file
    let _ = std::fs::remove_file(&obj_path);

    eprintln!("Compiled {} →", input_path.display());
    eprintln!("  {}", lib_path.display());
    eprintln!("  {}", rs_path.display());
}

/// Generate safe Rust wrapper code for all functions in a .simd file.
fn generate_rust_bindings(module_name: &str, items: &[ast::Item]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "//! Auto-generated bindings for {}.simd\n",
        module_name
    ));
    out.push_str("//! Do not edit — regenerate with: simd-lang compile\n\n");

    // Collect function signatures
    let fns: Vec<&ast::FnDef> = items
        .iter()
        .filter_map(|item| match item {
            ast::Item::Fn(f) => Some(f),
            _ => None,
        })
        .collect();

    if fns.is_empty() {
        return out;
    }

    // Extern block with mangled names to avoid conflicts with safe wrappers
    out.push_str(&format!(
        "#[link(name = \"{}\", kind = \"static\")]\n",
        module_name
    ));
    out.push_str("extern \"C\" {\n");

    for f in &fns {
        let (extern_params, _) = build_extern_params(f);
        let extern_ret = build_extern_return(f);
        out.push_str(&format!(
            "    #[link_name = \"{}\"]\n    fn __simd_{}({}){};\n",
            f.name, f.name, extern_params, extern_ret
        ));
    }
    out.push_str("}\n\n");

    // Safe wrappers
    for f in &fns {
        generate_safe_wrapper(&mut out, f);
    }

    out
}

/// Determine whether a parameter is a ptr type (memref) or a vector type.
fn is_ptr_param(p: &ast::Param) -> bool {
    p.ty.name == "ptr"
}

/// Get the element type name for a ptr[T] parameter.
fn ptr_elem(p: &ast::Param) -> &str {
    match &p.ty.width {
        Some(ast::Width::Param(name)) => name.as_str(),
        _ => "u8",
    }
}

/// Map a simd-lang scalar type name to Rust type.
fn scalar_to_rust(name: &str) -> &str {
    match name {
        "u8" => "u8",
        "u16" => "u16",
        "u32" => "u32",
        "u64" => "u64",
        "i8" => "i8",
        "i16" => "i16",
        "i32" => "i32",
        "i64" => "i64",
        "f32" => "f32",
        "f64" => "f64",
        "bool" => "u8", // i1 in LLVM, passed as i8
        _ => "u8",
    }
}

/// Build the extern "C" parameter list for a function.
/// ptr[T] expands to 5 params (alloc, align, offset, size, stride).
/// Vector types are passed directly.
fn build_extern_params(f: &ast::FnDef) -> (String, Vec<String>) {
    let mut params = Vec::new();
    let mut param_names = Vec::new();

    for p in &f.params {
        if is_ptr_param(p) {
            let elem = ptr_elem(p);
            let rust_ty = scalar_to_rust(elem);
            let is_mut = true; // conservative: treat all ptrs as mut
            let ptr_ty = if is_mut {
                format!("*mut {}", rust_ty)
            } else {
                format!("*const {}", rust_ty)
            };
            params.push(format!("{}_alloc: {}", p.name, ptr_ty));
            params.push(format!("{}_align: {}", p.name, ptr_ty));
            params.push(format!("{}_offset: i64", p.name));
            params.push(format!("{}_size: i64", p.name));
            params.push(format!("{}_stride: i64", p.name));
            param_names.push(p.name.clone());
        } else {
            // Vector param — for now, pass as the LLVM vector type
            // On ARM64, small vectors are passed in NEON registers
            let rust_ty = vector_to_rust_type(&p.ty);
            params.push(format!("{}: {}", p.name, rust_ty));
            param_names.push(p.name.clone());
        }
    }

    (params.join(", "), param_names)
}

/// Map a simd-lang type to Rust type for extern "C" ABI.
fn vector_to_rust_type(ty: &ast::Type) -> String {
    let width = match &ty.width {
        Some(ast::Width::Fixed(n)) => *n,
        _ => 1,
    };
    if width == 1 {
        // Width-1 vectors are scalars in the ABI
        // vector<1xi32> is passed in a NEON register as f32
        "f32".to_string()
    } else {
        // SIMD vectors — use fixed-size arrays for FFI
        // LLVM passes these in NEON registers
        let elem = scalar_to_rust(&ty.name);
        let total_bytes = width as usize * match ty.name.as_str() {
            "u8" | "i8" | "bool" => 1,
            "u16" | "i16" => 2,
            "u32" | "i32" | "f32" => 4,
            "u64" | "i64" | "f64" => 8,
            _ => 1,
        };
        // Use u128 for 16-byte vectors (common case)
        if total_bytes == 16 {
            "u128".to_string()
        } else {
            format!("[{}; {}]", elem, width)
        }
    }
}

/// Build the extern "C" return type.
fn build_extern_return(f: &ast::FnDef) -> String {
    match &f.ret_ty {
        None => String::new(),
        Some(ty) => {
            let rust_ty = vector_to_rust_type(ty);
            format!(" -> {}", rust_ty)
        }
    }
}

/// Generate a safe Rust wrapper function.
fn generate_safe_wrapper(out: &mut String, f: &ast::FnDef) {
    let has_ptr_params = f.params.iter().any(|p| is_ptr_param(p));
    let has_return = f.ret_ty.is_some();

    // Build safe parameter list
    let mut safe_params = Vec::new();
    for p in &f.params {
        if is_ptr_param(p) {
            let elem = ptr_elem(p);
            let rust_ty = scalar_to_rust(elem);
            safe_params.push(format!("{}: &mut [{}]", p.name, rust_ty));
        } else {
            let rust_ty = vector_to_rust_type(&p.ty);
            safe_params.push(format!("{}: {}", p.name, rust_ty));
        }
    }

    // Return type
    let safe_ret = match &f.ret_ty {
        None => String::new(),
        Some(ty) => {
            let width = match &ty.width {
                Some(ast::Width::Fixed(n)) => *n,
                _ => 1,
            };
            if width == 1 && !ty.name.starts_with('f') {
                // vector<1xi32> returns as f32 in ABI, convert to actual type
                format!(" -> {}", scalar_to_rust(&ty.name))
            } else {
                format!(" -> {}", vector_to_rust_type(ty))
            }
        }
    };

    // Doc comment
    out.push_str(&format!("/// `{}` from {}.simd\n", f.name, f.name));
    out.push_str(&format!(
        "pub fn {}({}){} {{\n",
        f.name,
        safe_params.join(", "),
        safe_ret
    ));
    out.push_str("    unsafe {\n");

    // Build call arguments
    let mut call_args = Vec::new();
    for p in &f.params {
        if is_ptr_param(p) {
            call_args.push(format!("{}.as_mut_ptr()", p.name));
            call_args.push(format!("{}.as_mut_ptr()", p.name));
            call_args.push("0".to_string());
            call_args.push(format!("{}.len() as i64", p.name));
            call_args.push("1".to_string());
        } else {
            call_args.push(p.name.clone());
        }
    }

    // Call and return
    let call = format!("__simd_{}({})", f.name, call_args.join(", "));
    match &f.ret_ty {
        None => {
            out.push_str(&format!("        {};\n", call));
        }
        Some(ty) => {
            let width = match &ty.width {
                Some(ast::Width::Fixed(n)) => *n,
                _ => 1,
            };
            if width == 1 && !ty.name.starts_with('f') {
                // vector<1xi32/u64/etc> returns as f32 bits
                let target = scalar_to_rust(&ty.name);
                out.push_str(&format!(
                    "        f32::to_bits({}) as {}\n",
                    call, target
                ));
            } else {
                out.push_str(&format!("        {}\n", call));
            }
        }
    }

    out.push_str("    }\n");
    out.push_str("}\n\n");
}
