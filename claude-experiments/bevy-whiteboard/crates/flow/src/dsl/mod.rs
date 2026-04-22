//! Flow DSL: a small textual syntax for authoring sims. Parses a
//! source string into an AST, then lowers to a live `Sim`.
//!
//! Grammar highlights:
//!   params     { NAME : EXPR; ... }
//!   node NAME  { slots { NAME: TYPE = EXPR; ... } rule NAME { on PAT when EXPR do { STMT* } } ... }
//!   compound NAME { in { PORT: INNER } out { PORT: INNER } }
//!   edges      { NODE[.PORT] -> NODE[.PORT] : EXPR ; ... }
//!   scenario   { at TIME : ACTION ; ... }
//!
//! See `examples/dsl_compound_pool.flow` for a worked example.

pub mod ast;
pub mod lex;
pub mod parse;
pub mod lower;

pub use parse::parse;
pub use lower::lower;

/// Parse and lower a DSL source string in one step.
pub fn load(src: &str, seed: u64) -> Result<crate::sim::Sim, String> {
    let file = parse(src)?;
    lower(&file, seed)
}
