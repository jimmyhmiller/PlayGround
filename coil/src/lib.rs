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
pub mod mono;
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
    macros::expand_program(&forms, &host_target())
}

/// The compile-time target description handed to the macro evaluator. Derived
/// from the host triple (Coil currently AOT-targets the host).
fn host_target() -> macros::TargetInfo {
    let triple = TargetMachine::get_default_triple()
        .as_str()
        .to_string_lossy()
        .into_owned();
    let mut parts = triple.split('-');
    let arch = parts.next().unwrap_or("unknown").to_string();
    let _vendor = parts.next();
    let os = parts.next().unwrap_or("unknown").to_string();
    let pointer_width = match arch.as_str() {
        "i386" | "i686" | "arm" | "armv7" | "thumbv7" | "wasm32" | "riscv32" | "mips" | "mipsel" => {
            32
        }
        _ => 64,
    };
    macros::TargetInfo {
        arch,
        os,
        triple,
        pointer_width,
    }
}

/// The shared front end: read → expand → parse → monomorphize → check → module.
fn build_module<'ctx>(ctx: &'ctx Context, src: &str) -> Result<Module<'ctx>, String> {
    let forms = read_and_expand(src)?;
    let program = mono::monomorphize(parse::parse_program(&forms)?)?;
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

/// Front end only: read → expand → parse → check. No codegen, no LLVM
/// execution — just diagnostics. (Coil has no `eval`/JIT: the only way to run a
/// program is to AOT-compile it.)
pub fn check_source(src: &str) -> Result<(), String> {
    let forms = read_and_expand(src)?;
    let program = mono::monomorphize(parse::parse_program(&forms)?)?;
    check::check(&program)
}
