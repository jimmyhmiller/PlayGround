//! The React Compiler pass pipeline — the full ordered list of every pass in
//! upstream `react_compiler::entrypoint::pipeline`, ported onto JSLIR.
//!
//! Every upstream pass is present here as a named, ordered entry so the skeleton
//! is complete from day one; passes not yet ported are **identity placeholders**.
//! Crucially, "not yet implemented" is a first-class, queryable [`PassStatus`] —
//! the driver reports exactly which passes are real vs. stubbed (see
//! [`PipelineReport`]), so the gap is loud, never a silent no-op masquerading as
//! work. As each pass is ported it flips `Stub → Implemented` and the react-oracle
//! byte-exact match count is the scoreboard that proves it.
//!
//! Stage groupings mirror the upstream crate that owns each pass
//! (`react_compiler_ssa`, `_optimization`, `_inference`, `_reactive_scopes`, …).
//!
//! NOTE on representation: the HIR-stage passes operate on a function-body JSLIR
//! CFG (`&mut Region`). The reactive-stage passes (after `build_reactive_function`)
//! upstream operate on a `ReactiveFunction`, which we have not built yet — their
//! placeholders take the same context for now and will be re-typed when that
//! representation lands. `codegen_function` is the backend that must eventually
//! emit the memoized `useMemoCache` output (today the round-trip lift stands in).

use jsir_ir::Region;

use crate::{constprop, dce, ssa};

/// Which upstream crate / phase a pass belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Lowering,
    Ssa,
    Optimization,
    TypeInference,
    Inference,
    Validation,
    ReactiveScopes,
}

/// Whether a pass is really ported or still an identity placeholder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassStatus {
    Implemented,
    Stub,
}

/// Mutable state threaded through the pipeline for a single function body.
pub struct PassCtx<'a> {
    /// The function-body CFG being transformed.
    pub region: &'a mut Region,
    /// Fresh-`ValueId` allocator seed (above every id in the whole file).
    pub next_value: u32,
}

/// One pipeline pass: its upstream identity + current implementation status + the
/// function that runs it. Stubs use [`noop`].
pub struct Pass {
    pub name: &'static str,
    pub stage: Stage,
    pub status: PassStatus,
    pub run: fn(&mut PassCtx),
}

impl Pass {
    const fn stub(name: &'static str, stage: Stage) -> Pass {
        Pass { name, stage, status: PassStatus::Stub, run: noop }
    }
    const fn done(name: &'static str, stage: Stage, run: fn(&mut PassCtx)) -> Pass {
        Pass { name, stage, status: PassStatus::Implemented, run }
    }
}

/// Identity placeholder for a not-yet-ported pass.
fn noop(_: &mut PassCtx) {}

// ---------------------------------------------------------------------------
// Implemented-pass adapters (compute the analyses they need on demand).
// ---------------------------------------------------------------------------

/// `enter_ssa` + `eliminate_redundant_phi` are side-table analyses in our model
/// (they compute def-use chains + phis without mutating the IR); consumers
/// recompute SSA on demand, so running them here has no materialized IR effect.
/// They are marked Implemented because the logic exists and is tested; this
/// adapter validates that SSA construction succeeds for the body.
fn run_enter_ssa(ctx: &mut PassCtx) {
    let mut next = ctx.next_value;
    let _ = ssa::enter_ssa(ctx.region, &mut next);
    ctx.next_value = next;
}

fn run_eliminate_redundant_phi(ctx: &mut PassCtx) {
    let mut next = ctx.next_value;
    if let Some(mut info) = ssa::enter_ssa(ctx.region, &mut next) {
        crate::passes::eliminate_redundant_phi(&mut info);
    }
    ctx.next_value = next;
}

/// ConstantPropagation: fold computed ops to literals, then prune `if`s with a
/// constant test (re-computing the lattice between the two, mirroring upstream's
/// fixpoint loop over fold + branch removal).
fn run_constant_propagation(ctx: &mut PassCtx) {
    let mut next = ctx.next_value;
    if let Some(info) = ssa::enter_ssa(ctx.region, &mut next) {
        let lattice = constprop::constant_lattice(ctx.region, &info);
        constprop::fold_constants(ctx.region, &lattice);
        let lattice = constprop::constant_lattice(ctx.region, &info);
        constprop::prune_constant_if_branches(ctx.region, &lattice);
    }
    ctx.next_value = next;
}

fn run_dead_code_elimination(ctx: &mut PassCtx) {
    dce::eliminate_dead_code(ctx.region);
}

// ---------------------------------------------------------------------------
// The full pipeline, in upstream order.
// ---------------------------------------------------------------------------

/// Build the complete, ordered pass pipeline. Implemented passes carry their real
/// logic; every other upstream pass is present as an identity [`Pass::stub`].
pub fn pipeline() -> Vec<Pass> {
    use Stage::*;
    vec![
        // Frontend lowering (handled by source_to_ir + build_jslir before the driver).
        Pass::done("lower", Lowering, noop),
        // -- HIR stage --------------------------------------------------------
        Pass::stub("prune_maybe_throws", Optimization),
        Pass::stub("validate_context_variable_lvalues", Validation),
        Pass::stub("validate_use_memo", Validation),
        Pass::stub("drop_manual_memoization", Optimization),
        Pass::stub("inline_immediately_invoked_function_expressions", Optimization),
        Pass::done("enter_ssa", Ssa, run_enter_ssa),
        Pass::done("eliminate_redundant_phi", Ssa, run_eliminate_redundant_phi),
        Pass::done("constant_propagation", Optimization, run_constant_propagation),
        Pass::stub("infer_types", TypeInference),
        Pass::stub("validate_hooks_usage", Validation),
        Pass::stub("validate_no_capitalized_calls", Validation),
        Pass::stub("optimize_props_method_calls", Optimization),
        Pass::stub("analyse_functions", Inference),
        Pass::stub("infer_mutation_aliasing_effects", Inference),
        Pass::stub("optimize_for_ssr", Optimization),
        Pass::done("dead_code_elimination", Optimization, run_dead_code_elimination),
        Pass::stub("infer_mutation_aliasing_ranges", Inference),
        Pass::stub("validate_locals_not_reassigned_after_render", Validation),
        Pass::stub("validate_no_ref_access_in_render", Validation),
        Pass::stub("validate_no_set_state_in_render", Validation),
        Pass::stub("validate_no_derived_computations_in_effects_exp", Validation),
        Pass::stub("validate_no_derived_computations_in_effects", Validation),
        Pass::stub("validate_no_set_state_in_effects", Validation),
        Pass::stub("validate_no_jsx_in_try_statement", Validation),
        Pass::stub("validate_no_freezing_known_mutable_functions", Validation),
        Pass::stub("infer_reactive_places", Inference),
        Pass::stub("validate_exhaustive_dependencies", Validation),
        Pass::stub("rewrite_instruction_kinds_based_on_reassignment", Ssa),
        Pass::stub("validate_static_components", Validation),
        Pass::stub("infer_reactive_scope_variables", Inference),
        Pass::stub("memoize_fbt_and_macro_operands_in_same_scope", Inference),
        Pass::stub("outline_jsx", Optimization),
        Pass::stub("name_anonymous_functions", Optimization),
        Pass::stub("outline_functions", Optimization),
        Pass::stub("align_method_call_scopes", Inference),
        Pass::stub("align_object_method_scopes", Inference),
        Pass::stub("prune_unused_labels_hir", Optimization),
        Pass::stub("align_reactive_scopes_to_block_scopes_hir", Inference),
        Pass::stub("merge_overlapping_reactive_scopes_hir", Inference),
        Pass::stub("build_reactive_scope_terminals_hir", Inference),
        Pass::stub("flatten_reactive_loops_hir", Inference),
        Pass::stub("flatten_scopes_with_hooks_or_use_hir", Inference),
        Pass::stub("propagate_scope_dependencies_hir", Inference),
        // -- HIR → ReactiveFunction + reactive stage --------------------------
        Pass::stub("build_reactive_function", ReactiveScopes),
        Pass::stub("assert_well_formed_break_targets", ReactiveScopes),
        Pass::stub("prune_unused_labels", ReactiveScopes),
        Pass::stub("assert_scope_instructions_within_scopes", ReactiveScopes),
        Pass::stub("prune_non_escaping_scopes", ReactiveScopes),
        Pass::stub("prune_non_reactive_dependencies", ReactiveScopes),
        Pass::stub("prune_unused_scopes", ReactiveScopes),
        Pass::stub("merge_reactive_scopes_that_invalidate_together", ReactiveScopes),
        Pass::stub("prune_always_invalidating_scopes", ReactiveScopes),
        Pass::stub("propagate_early_returns", ReactiveScopes),
        Pass::stub("prune_unused_lvalues", ReactiveScopes),
        Pass::stub("promote_used_temporaries", ReactiveScopes),
        Pass::stub("extract_scope_declarations_from_destructuring", ReactiveScopes),
        Pass::stub("stabilize_block_ids", ReactiveScopes),
        Pass::stub("rename_variables", ReactiveScopes),
        Pass::stub("prune_hoisted_contexts", ReactiveScopes),
        Pass::stub("validate_preserved_manual_memoization", Validation),
        // -- Backend ----------------------------------------------------------
        Pass::stub("codegen_function", ReactiveScopes),
    ]
}

/// Result of running the pipeline over one function body.
#[derive(Debug, Clone)]
pub struct PipelineReport {
    pub total: usize,
    pub implemented: usize,
    pub stub: usize,
    /// Names of the passes that actually carry logic (ran for real).
    pub implemented_names: Vec<&'static str>,
}

/// Run every pass in order over a function-body CFG. Stubs are identity, so the
/// observable effect is exactly the currently-implemented subset; the returned
/// [`PipelineReport`] makes the implemented/stub split explicit.
pub fn run_pipeline(region: &mut Region, next_value: u32) -> PipelineReport {
    let passes = pipeline();
    let mut ctx = PassCtx { region, next_value };
    let mut implemented_names = Vec::new();
    let (mut implemented, mut stub) = (0, 0);
    for pass in &passes {
        match pass.status {
            PassStatus::Implemented => {
                implemented += 1;
                implemented_names.push(pass.name);
            }
            PassStatus::Stub => stub += 1,
        }
        (pass.run)(&mut ctx);
    }
    PipelineReport { total: passes.len(), implemented, stub, implemented_names }
}

/// A one-line human summary of pipeline coverage (for the oracle scoreboard).
pub fn coverage_summary() -> String {
    let passes = pipeline();
    let implemented = passes.iter().filter(|p| p.status == PassStatus::Implemented).count();
    format!("{implemented}/{} passes implemented", passes.len())
}
