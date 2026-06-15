//! Coil — a low-level Lisp where calling convention (and, later, allocation) is
//! part of the type system. This crate is the M0/M1 skeleton: reader → core →
//! convention-aware checks → LLVM codegen that JIT-runs `main`.
//!
//! See `docs/DESIGN.md` for the full design.

pub mod ast;
pub mod check;
pub mod codegen;
pub mod convention;
pub mod parse;
pub mod reader;

use inkwell::context::Context;
use inkwell::targets::{InitializationConfig, Target};
use inkwell::OptimizationLevel;

/// Parse + check + emit textual LLVM IR (no JIT). Useful in tests and for
/// inspecting how conventions lower.
pub fn emit_ir(src: &str) -> Result<String, String> {
    let forms = reader::read_all(src)?;
    let program = parse::parse_program(&forms)?;
    check::check(&program)?;
    let ctx = Context::create();
    let module = codegen::compile(&ctx, &program)?;
    Ok(module.print_to_string().to_string())
}

/// Parse + check + JIT-compile, then run `main` (which must take no args and
/// return i64) and return its result.
pub fn run_source(src: &str) -> Result<i64, String> {
    let forms = reader::read_all(src)?;
    let program = parse::parse_program(&forms)?;
    check::check(&program)?;

    let main = program
        .func("main")
        .ok_or("no `main` function to run")?;
    if !main.params.is_empty() {
        return Err("`main` must take no arguments".to_string());
    }

    Target::initialize_native(&InitializationConfig::default())?;
    let ctx = Context::create();
    let module = codegen::compile(&ctx, &program)?;
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
