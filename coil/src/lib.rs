//! Coil — a low-level Lisp where calling convention (and, later, allocation) is
//! part of the type system. This crate is the M0/M1 skeleton: reader → core →
//! convention-aware checks → LLVM codegen that JIT-runs `main`.
//!
//! See `docs/DESIGN.md` for the full design.

pub mod abi;
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

/// The shared front end: read → expand → parse → check (elaborate + infer) →
/// monomorphize → module. The checker runs *before* monomorphization now: it
/// types polymorphic code and infers/fills the generic type arguments, so the
/// monomorphizer is a pure specializer over fully-explicit type args.
fn build_module<'ctx>(ctx: &'ctx Context, src: &str) -> Result<Module<'ctx>, String> {
    build_module_for(ctx, src, codegen::target_triple())
}

/// `build_module` for an explicitly chosen target triple (cross-targeting).
fn build_module_for<'ctx>(
    ctx: &'ctx Context,
    src: &str,
    triple: inkwell::targets::TargetTriple,
) -> Result<Module<'ctx>, String> {
    let forms = read_and_expand(src)?;
    let program = parse::parse_program(&forms)?;
    let program = check::check(&program)?;
    let program = mono::monomorphize(program)?;
    codegen::compile_for(ctx, &program, triple)
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

/// `emit_ir` for an explicitly chosen target triple — used to inspect a
/// program's ABI lowering for a non-host target (e.g. verifying the x86-64 SysV
/// struct coercion from an arm64 host).
pub fn emit_ir_for(src: &str, triple: &str) -> Result<String, String> {
    let ctx = Context::create();
    let module = build_module_for(&ctx, src, inkwell::targets::TargetTriple::create(triple))?;
    Ok(module.print_to_string().to_string())
}

/// AOT: compile to a native object file. This is the language's primary output
/// — no runtime dependency on LLVM, links with a real linker, and the `:shim`
/// trampolines become ordinary relocations the system toolchain resolves.
pub fn compile_to_object(src: &str, obj_path: &Path) -> Result<(), String> {
    compile_to_object_for(src, obj_path, codegen::target_triple())
}

/// `compile_to_object` for an explicitly chosen target triple. Used to
/// cross-compile (e.g. an x86-64 object on an arm64 host to exercise the SysV
/// struct ABI under Rosetta); the IR and the emitted machine code share `triple`.
pub fn compile_to_object_for(
    src: &str,
    obj_path: &Path,
    triple: inkwell::targets::TargetTriple,
) -> Result<(), String> {
    Target::initialize_all(&InitializationConfig::default());
    let ctx = Context::create();
    let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
    let module = build_module_for(&ctx, src, triple)?;
    let triple = module.get_triple();
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
    build_executable_for(src, out_path, codegen::target_triple())
}

/// `build_executable` for an explicitly chosen target triple (cross-compile).
/// The object is emitted for `triple` *and* the linker is told to produce an
/// executable for that triple's architecture (`cc -arch <arch>`), so the result
/// is a real binary for the chosen target (runnable under Rosetta on macOS for a
/// cross arch). The triple's arch must be one codegen supports.
pub fn build_executable_for(
    src: &str,
    out_path: &Path,
    triple: inkwell::targets::TargetTriple,
) -> Result<(), String> {
    let triple_str = triple.as_str().to_string_lossy().into_owned();
    let obj_path = out_path.with_extension("o");
    compile_to_object_for(src, &obj_path, triple)?;
    let mut cc = Command::new("cc");
    if let Some(arch) = link_arch_flag(&triple_str) {
        cc.arg("-arch").arg(arch);
    }
    let status = cc
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

/// The `cc -arch` value for a target triple's architecture, or `None` if it is
/// the host arch (no `-arch` needed). An unsupported arch is a hard error so a
/// `--target` typo doesn't silently link for the host.
fn link_arch_flag(triple: &str) -> Option<&'static str> {
    let arch = triple.split('-').next().unwrap_or("");
    let host = TargetMachine::get_default_triple()
        .as_str()
        .to_string_lossy()
        .split('-')
        .next()
        .unwrap_or("")
        .to_string();
    match arch {
        "x86_64" | "amd64" => (host != "x86_64").then_some("x86_64"),
        "aarch64" | "arm64" | "arm64e" => (host != "aarch64" && host != "arm64").then_some("arm64"),
        _ => None,
    }
}

/// Front end only: read → expand → parse → check. No codegen, no LLVM
/// execution — just diagnostics. (Coil has no `eval`/JIT: the only way to run a
/// program is to AOT-compile it.)
pub fn check_source(src: &str) -> Result<(), String> {
    let forms = read_and_expand(src)?;
    let program = parse::parse_program(&forms)?;
    let program = check::check(&program)?;
    // Run monomorphization too, so specialization-time errors (if any) surface
    // in a check-only pass as well.
    mono::monomorphize(program)?;
    Ok(())
}
