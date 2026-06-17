//! Coil — a low-level Lisp where calling convention (and, later, allocation) is
//! part of the type system. This crate is the M0/M1 skeleton: reader → core →
//! convention-aware checks → LLVM codegen that JIT-runs `main`.
//!
//! See `docs/DESIGN.md` for the full design.

pub mod ast;
pub mod check;
pub mod codegen;
pub mod convention;
pub mod macros;
pub mod parse;
pub mod reader;

use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::OptimizationLevel;
use std::path::Path;
use std::process::Command;

/// Read + macro-expand a source string into the macro-free top-level forms.
fn read_and_expand(src: &str) -> Result<Vec<reader::Sexp>, String> {
    let forms = reader::read_all(src)?;
    macros::expand_program(&forms)
}

/// The shared front end: read → expand → parse → check → LLVM module.
fn build_module<'ctx>(ctx: &'ctx Context, src: &str) -> Result<Module<'ctx>, String> {
    let forms = read_and_expand(src)?;
    let program = parse::parse_program(&forms)?;
    check::check(&program)?;
    codegen::compile(ctx, &program)
}

/// Macro-expand and pretty-print the resulting forms (for `--expand`).
pub fn expand_to_string(src: &str) -> Result<String, String> {
    let forms = read_and_expand(src)?;
    Ok(forms
        .iter()
        .map(|f| f.to_string())
        .collect::<Vec<_>>()
        .join("\n"))
}

/// Parse + check + emit textual LLVM IR (no JIT). Useful in tests and for
/// inspecting how conventions lower.
pub fn emit_ir(src: &str) -> Result<String, String> {
    let ctx = Context::create();
    let module = build_module(&ctx, src)?;
    Ok(module.print_to_string().to_string())
}

/// AOT: compile to a native object file. This is the language's primary output
/// — no runtime dependency on LLVM, links with a real linker, and the `:shim`
/// trampolines become ordinary relocations the system toolchain resolves.
pub fn compile_to_object(src: &str, obj_path: &Path) -> Result<(), String> {
    Target::initialize_native(&InitializationConfig::default())?;
    let ctx = Context::create();
    let module = build_module(&ctx, src)?;

    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let tm = target
        .create_target_machine(
            &triple,
            "generic",
            "",
            OptimizationLevel::Default,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or("could not create target machine")?;

    module.set_triple(&triple);
    module.set_data_layout(&tm.get_target_data().get_data_layout());
    tm.write_to_file(&module, FileType::Object, obj_path)
        .map_err(|e| e.to_string())
}

/// AOT: compile to an object and link a native executable with `cc`. The Coil
/// `main` (i64, no args) becomes the process entry; its return value is the
/// exit code (low 8 bits).
pub fn build_executable(src: &str, out_path: &Path) -> Result<(), String> {
    let obj_path = out_path.with_extension("o");
    compile_to_object(src, &obj_path)?;
    let status = Command::new("cc")
        .arg(&obj_path)
        .arg("-o")
        .arg(out_path)
        .status()
        .map_err(|e| format!("failed to invoke linker (cc): {e}"))?;
    let _ = std::fs::remove_file(&obj_path);
    if !status.success() {
        return Err(format!("linker (cc) failed with {status}"));
    }
    Ok(())
}

/// JIT-evaluate `main` (no args, returns i64) and return the full i64 result.
/// This is the `eval`/REPL convenience path — AOT (`build_executable`) is the
/// real output. `eval` exists because a process exit code is only 8 bits, so it
/// can report results like 1024 that an executable's exit status cannot.
pub fn run_source(src: &str) -> Result<i64, String> {
    Target::initialize_native(&InitializationConfig::default())?;
    let ctx = Context::create();
    let module = build_module(&ctx, src)?;
    if module.get_function("main").is_none() {
        return Err("no `main` function to run".to_string());
    }
    let ee = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .map_err(|e| e.to_string())?;
    unsafe {
        let f = ee
            .get_function::<unsafe extern "C" fn() -> i64>("main")
            .map_err(|e| e.to_string())?;
        Ok(f.call())
    }
}
