//! MLIR-style CFG + SSA layer over the reversible JSIR IR, plus the start of the
//! React-compiler-style analyses that need it.
//!
//! Pipeline: `source` -> JSIR IR (`jsir-swc`) -> pre-SSA CFG ([`lower`]) ->
//! SSA CFG ([`ssa::construct`]). Correctness is checked two ways: an executable
//! oracle ([`interp`]) diffed against Node, and a dominance-based SSA verifier
//! ([`verify`]).

pub mod cfg;
pub mod codegen;
pub mod interp;
pub mod lower;
pub mod mutability;
pub mod print;
pub mod scopes;
pub mod ssa;
pub mod verify;

pub use cfg::Cfg;

/// Compile a source snippet's first function (or its program body) to a pre-SSA
/// CFG.
pub fn lower(src: &str) -> Result<Cfg, String> {
    let ir = jsir_swc::source_to_ir(src)?;
    lower::lower_function(&ir)
}

/// Compile a source snippet to SSA form.
pub fn compile_ssa(src: &str) -> Result<Cfg, String> {
    let mut cfg = lower(src)?;
    ssa::construct(&mut cfg);
    Ok(cfg)
}
