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
pub mod expand;
pub mod lower;

pub use parse::parse;
pub use lower::{lower, lower_into, Lowered};

/// Parse, expand, and lower a DSL source string in one step.
///
/// The expansion pass (between parse and lower) is the **single
/// construction phase** that resolves `for` loops, name templates
/// (`Cell_{x}_{y}`), and parametric compounds. Lowering only ever sees
/// a fully residual AST; if you want to inspect the residual, call
/// `parse + expand` separately.
pub fn load(src: &str, seed: u64) -> Result<crate::sim::Sim, String> {
    let file = parse(src)?;
    let residual = expand::expand(&file)?;
    lower(&residual, seed)
}

/// Register every `node` and `compound` block in the DSL source as a
/// named class on `sim`. Nothing is instantiated; use
/// [`crate::sim::Sim::instantiate`] to create working instances
/// afterwards.
///
/// `node` blocks are parsed + expanded + lowered into a leaf template
/// stored in `sim.templates`. `compound` blocks are stored *unexpanded*
/// in `sim.compound_templates` and expanded on demand at
/// instantiation time — that's what makes a `compound` first-class:
/// the palette can spawn one with `sim.instantiate("WorkerComposite",
/// "w1")` exactly like spawning a leaf class.
///
/// Items other than `node` / `compound` blocks are rejected — this
/// entry point is for authoring reusable classes, not for full sim
/// programs. Use [`load`] for that.
pub fn register_classes(sim: &mut crate::sim::Sim, src: &str) -> Result<Vec<String>, String> {
    let file = parse(src)?;
    // NB: don't run the full `expand` pass here — it would splice
    // every compound's body into the surrounding scope, defeating the
    // point of having compounds as registered classes. The class
    // registration only needs surface-level shape; the per-instance
    // expansion is deferred to `Sim::instantiate`.
    let mut names = Vec::new();
    for item in &file.items {
        match item {
            ast::Item::Node(n) => {
                // Nodes still get the full expand pass so their internal
                // `for` loops / name templates resolve. Compounds DON'T
                // get pre-expanded — see above.
                let single = ast::File { items: vec![ast::Item::Node(n.clone())] };
                let expanded = expand::expand(&single)?;
                let n = match &expanded.items[..] {
                    [ast::Item::Node(n)] => n,
                    _ => return Err(format!(
                        "register_classes: node `{}` expanded to non-node items",
                        n.name.as_plain().unwrap_or("?")
                    )),
                };
                let name = n.name.as_plain().ok_or_else(||
                    "register_classes: node name still contains unresolved `{...}` interpolations".to_string()
                )?.to_string();
                let tpl = lower::build_class_template(n)?;
                names.push(name);
                sim.register_template(tpl);
            }
            ast::Item::Compound(c) => {
                if c.name.is_empty() {
                    return Err(
                        "register_classes: anonymous compound has no class name".to_string()
                    );
                }
                names.push(c.name.clone());
                sim.compound_templates.insert(c.name.clone(), c.clone());
            }
            other => {
                return Err(format!(
                    "register_classes: only `node` and `compound` blocks are allowed, \
                     got {:?}",
                    std::mem::discriminant(other)
                ));
            }
        }
    }
    Ok(names)
}
