//! lambda-Tally: a dependent + linear (quantitative) language for memory-safe
//! low-level code.
//!
//! Pipeline: lexer -> parser -> ast -> check (the linear/permission checker).
//! Permissions, regions, and ghosts are ERASED, so the backend (codegen, behind
//! the `llvm` feature) lowers an already-checked program with no notion of them.

pub mod ast;
pub mod check;
pub mod lexer;
pub mod parser;

#[cfg(feature = "llvm")]
pub mod codegen;

/// Parse + check a source string. `Ok(())` means it is accepted; `Err` carries
/// either a parse error (one string) or the list of type/permission diagnostics.
pub fn check_str(src: &str) -> Result<(), Vec<String>> {
    let prog = parser::parse(src).map_err(|e| vec![format!("parse error: {e}")])?;
    let diags = check::check(&prog);
    if diags.is_empty() {
        Ok(())
    } else {
        Err(diags)
    }
}
