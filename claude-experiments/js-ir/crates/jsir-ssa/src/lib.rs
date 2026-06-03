//! MLIR-style CFG + SSA layer over the reversible JSIR IR, plus the start of the
//! React-compiler-style analyses that need it.
//!
//! Pipeline: `source` -> JSIR IR (`jsir-swc`) -> pre-SSA CFG ([`lower`]) ->
//! SSA CFG ([`ssa::construct`]). Correctness is checked two ways: an executable
//! oracle ([`interp`]) diffed against Node, and a dominance-based SSA verifier
//! ([`verify`]).

pub mod aliasing_ranges;
pub mod cfg;
pub mod codegen;
pub mod constfold;
pub mod detect;
pub mod effects;
pub mod fidelity;
pub mod infer_effects;
pub mod infer_types;
pub mod interp;
pub mod lower;
pub mod memoize_plan;
pub mod mutability;
pub mod print;
pub mod scopes;
pub mod ssa;
pub mod types;
pub mod validate;
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

/// The analysis output an IR-rewrite memoizer needs, with **no** string codegen.
///
/// This is the façade the `jsir-transforms` memoize pass calls into: it runs the
/// same analysis chain as [`codegen::compile`]'s memoized branch
/// (`lower_function` -> `ssa::construct` -> `aliasing_ranges::analyze` ->
/// `scopes::analyze`) and hands back the reactive-scope infos plus the SSA CFG.
/// The CFG's instructions carry [`cfg::SrcRef`] provenance (see [`lower`]), which
/// is what lets the transform map scope `Value`s back to the JSIR statements it
/// must relocate.
pub struct MemoPlan {
    /// The function's declared name (empty if anonymous).
    pub fn_name: String,
    /// Reactive-scope analysis, one entry per scope.
    pub infos: Vec<scopes::ScopeInfo>,
    /// The SSA CFG. Each surviving `Instr` retains its `src` provenance.
    pub cfg: Cfg,
    /// Whether the function is straight-line (single block ending in `return`),
    /// the only shape the straight-line memoize path can currently rewrite.
    pub single_block: bool,
}

/// Produce a [`MemoPlan`] for the first function declaration in `file_fn` (the
/// program op-tree from `jsir_swc::source_to_ir`). Analysis-only: it does not
/// emit or rewrite anything.
pub fn plan(file_fn: &jsir_ir::Op) -> Result<MemoPlan, String> {
    let mut cfg = lower::lower_function(file_fn)?;
    ssa::construct(&mut cfg);
    constfold::fold_constants(&mut cfg);
    let fn_name = cfg.fn_name.clone().unwrap_or_default();
    let ranges = aliasing_ranges::analyze(&cfg);
    let infos = scopes::analyze(&cfg, &ranges);
    let single_block = cfg.blocks.len() == 1 && matches!(cfg.blocks[0].term, cfg::Term::Ret(_));
    Ok(MemoPlan { fn_name, infos, cfg, single_block })
}
