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
pub use lower::{lower, lower_into, Lowered};

/// Parse and lower a DSL source string in one step.
pub fn load(src: &str, seed: u64) -> Result<crate::sim::Sim, String> {
    let file = parse(src)?;
    lower(&file, seed)
}

/// Register every `node` block in the DSL source as a named class on
/// `sim`. Nothing is instantiated; use [`crate::sim::Sim::instantiate`]
/// to create working instances afterwards.
///
/// Items other than `node` blocks in the source are rejected — this
/// entry point is for authoring reusable classes, not for full sim
/// programs. Use [`load`] for that.
pub fn register_classes(sim: &mut crate::sim::Sim, src: &str) -> Result<Vec<String>, String> {
    let file = parse(src)?;
    let mut names = Vec::new();
    for item in &file.items {
        match item {
            ast::Item::Node(n) => {
                let tpl = lower::build_class_template(n)?;
                names.push(n.name.clone());
                sim.register_template(tpl);
            }
            other => {
                return Err(format!(
                    "register_classes: only `node` blocks are allowed, \
                     got {:?}",
                    std::mem::discriminant(other)
                ));
            }
        }
    }
    Ok(names)
}
