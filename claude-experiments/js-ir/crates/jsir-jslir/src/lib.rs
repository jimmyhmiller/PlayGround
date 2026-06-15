//! JSLIR — a CFG/SSA dialect on the generic `jsir-ir` op substrate, for porting
//! the React Compiler passes onto our IR.
//!
//! Pipeline position:
//! ```text
//!   source ─source_to_ir→ JSHIR ─build_jslir→ JSLIR ─[passes]→ JSLIR ─lift_jslir→ JSHIR ─ir_to_source→ JS
//! ```
//! `build_jslir` flattens JSHIR's structured control flow (`jshir.if_statement`,
//! `jshir.while_statement`, …) into a CFG of basic blocks + `jslir.*` terminators;
//! `lift_jslir` rebuilds the structured form so the existing, byte-exact
//! `hir2ast`/`ast2source` path emits the JS. Block instructions reuse the
//! AST-faithful `jsir.*` ops verbatim (coarse blocks for now; within-block
//! flattening to a fully-scalar instruction stream is a later refinement).
//!
//! Coverage grows incrementally. A function `build_jslir` cannot yet lower is
//! **passed through unchanged** (still valid JSHIR), so the round-trip never
//! regresses; [`Stats`] reports how many functions actually became a CFG.

pub mod cfg;
pub mod constprop;
pub mod dce;
pub mod dialect;
pub mod passes;
pub mod pipeline;
pub mod ssa;
pub mod verify;
mod build;
mod expr_flatten;
mod lift;

use jsir_ir::{Block, Op, Region};

/// Function-like op names whose body we lower to a CFG.
const FUNCTION_OPS: &[&str] = &[
    "jsir.function_declaration",
    "jsir.function_expression",
    "jsir.arrow_function_expression",
    "jsir.object_method",
    "jsir.class_method",
    "jsir.class_private_method",
];

fn is_function(name: &str) -> bool {
    FUNCTION_OPS.contains(&name)
}

/// Statement ops that introduce control flow. Their presence means the body is
/// not yet lowerable by the straight-line builder, so we pass the function
/// through. (These are the next increments: if → while → for → switch/try.)
const CONTROL_FLOW_STMTS: &[&str] = &[
    "jshir.if_statement",
    "jshir.while_statement",
    "jshir.do_while_statement",
    "jshir.for_in_statement",
    "jshir.for_of_statement",
    "jshir.switch_statement",
    "jshir.labeled_statement",
    "jshir.try_statement",
    "jsir.throw_statement",
];

fn is_control_flow_stmt(name: &str) -> bool {
    CONTROL_FLOW_STMTS.contains(&name)
}

/// How many function bodies `build_jslir` lowered to a CFG vs passed through.
#[derive(Debug, Default, Clone, Copy)]
pub struct Stats {
    pub functions: usize,
    pub lowered: usize,
    pub passed_through: usize,
}

// ---------------------------------------------------------------------------
// Generic post-order rebuild: clone the op tree, applying `f` to every op AFTER
// its children have been rebuilt. Used by both build and lift.
// ---------------------------------------------------------------------------

fn rebuild(op: &Op, f: &mut impl FnMut(&mut Op)) -> Op {
    let regions = op
        .regions
        .iter()
        .map(|r| Region {
            blocks: r
                .blocks
                .iter()
                .map(|b| Block {
                    id: b.id,
                    args: b.args.clone(),
                    ops: b.ops.iter().map(|o| rebuild(o, f)).collect(),
                })
                .collect(),
        })
        .collect();
    let mut new = Op {
        name: op.name.clone(),
        operands: op.operands.clone(),
        attrs: op.attrs.clone(),
        regions,
        results: op.results.clone(),
        successors: op.successors.clone(),
        trivia: op.trivia.clone(),
        node_id: op.node_id,
    };
    f(&mut new);
    new
}

// ---------------------------------------------------------------------------
// build_jslir: JSHIR -> JSLIR
// ---------------------------------------------------------------------------

/// Lower every function body in `file` from structured JSHIR to a JSLIR CFG where
/// possible; pass through what isn't supported yet.
pub fn build_jslir(file: &Op) -> (Op, Stats) {
    let mut stats = Stats::default();
    // Expression flattening mints fresh phi ValueIds; seed them above every id in
    // the whole file so they're globally unique (block ids stay region-scoped).
    let mut next_value = lift::max_value_id(file) + 1;
    let out = rebuild(file, &mut |op| {
        if is_function(&op.name) {
            stats.functions += 1;
            if cfgify_function(op, &mut next_value) {
                stats.lowered += 1;
            } else {
                stats.passed_through += 1;
            }
        }
    });
    (out, stats)
}

/// The index of a function op's BODY region. `function_declaration`/
/// `function_expression` are `[params, body]`; arrows are a single `[body]`
/// region (their params are operands).
fn body_region_index(name: &str) -> usize {
    match name {
        // Arrows and methods are a single `[body]` region (params are operands).
        "jsir.arrow_function_expression"
        | "jsir.object_method"
        | "jsir.class_method"
        | "jsir.class_private_method" => 0,
        // function_declaration / function_expression are `[params, body]`.
        _ => 1,
    }
}

/// Replace a function's body region with a CFG, in place. Returns whether it was
/// lowered (false = left as-is / passed through). Handles block bodies (normal
/// functions and block-body arrows) and arrow expression bodies.
fn cfgify_function(func: &mut Op, next_value: &mut u32) -> bool {
    let idx = body_region_index(&func.name);
    let Some(body_block) = func.regions.get(idx).and_then(|r| r.blocks.first()) else {
        return false;
    };

    let built = if body_block.ops.len() == 1 && body_block.ops[0].name == "jshir.block_statement" {
        // Block body: { block_statement }.
        let Some(stmts) = block_statement_body(&body_block.ops[0]) else {
            return false;
        };
        build::build_cfg(&stmts, next_value)
    } else if body_block.ops.last().map_or(false, |o| o.name == "jsir.expr_region_end") {
        // Arrow expression body: { <expr ops> ; expr_region_end(%v) }.
        build::build_expr_body_cfg(&body_block.ops, next_value)
    } else {
        return false;
    };

    match built {
        Ok(cfg) => {
            // A malformed CFG is a builder bug: fail loudly in debug/tests, and in
            // release fall back to passing the function through.
            let problems = verify::verify_cfg(&cfg);
            debug_assert!(problems.is_empty(), "JSLIR verify failed: {problems:?}");
            if !problems.is_empty() {
                return false;
            }
            func.regions[idx] = cfg;
            true
        }
        Err(_) => false,
    }
}

/// The statement ops inside a `jshir.block_statement` op (its body region).
fn block_statement_body(block_stmt: &Op) -> Option<Vec<Op>> {
    let stmts_region = block_stmt.regions.first()?;
    let stmts_block = stmts_region.blocks.first()?;
    Some(stmts_block.ops.clone())
}

// ---------------------------------------------------------------------------
// lift_jslir: JSLIR -> JSHIR
// ---------------------------------------------------------------------------

/// Rebuild structured JSHIR from a JSLIR CFG so the existing hir2ast/ast2source
/// path can emit JS. Functions that were passed through by `build_jslir` (no
/// JSLIR terminators in their body) are left untouched.
pub fn lift_jslir(file: &Op) -> Op {
    // Seed the id allocator above every existing ValueId so synthesized ops
    // (store_local → variable_declaration) never collide.
    let mut alloc = lift::Alloc::new(lift::max_value_id(file) + 1);
    rebuild(file, &mut |op| {
        if is_function(&op.name) {
            lift_function(op, &mut alloc);
        }
    })
}

fn lift_function(func: &mut Op, alloc: &mut lift::Alloc) {
    let idx = body_region_index(&func.name);
    let Some(body) = func.regions.get(idx) else {
        return;
    };
    if !dialect::region_is_cfg(body) {
        return; // passed-through JSHIR body
    }
    // Arrow expression body: a single block ending in an `arrow_expr` return →
    // restore an expression body region, not `{ block_statement }`.
    if let Some((ops, val)) = lift::as_expr_body(body, alloc) {
        func.regions[idx] = lift::expr_body_region(ops, val);
    } else if let Some(stmts) = lift::lift_cfg(body, alloc) {
        func.regions[idx] = Region::with_block(Block::leaf(vec![make_block_statement(stmts)]));
    }
}

/// A `jshir.block_statement` wrapping `stmts` (body region) + an empty directives
/// region (which prints as `^bb0:`).
fn make_block_statement(stmts: Vec<Op>) -> Op {
    let mut bs = Op::new("jshir.block_statement");
    bs.regions.push(Region::with_block(Block::leaf(stmts)));
    bs.regions.push(Region::with_block(Block::leaf(Vec::new())));
    bs
}

// ---------------------------------------------------------------------------
// Convenience round-trip (used by the oracle and tests).
// ---------------------------------------------------------------------------

/// JSHIR file op → build → lift → JSHIR file op. Should be a faithful round-trip
/// for every function `build_jslir` lowers; passed-through functions are
/// untouched. Returns the lifted op and the build stats.
pub fn roundtrip(file: &Op) -> (Op, Stats) {
    let (jslir, stats) = build_jslir(file);
    (lift_jslir(&jslir), stats)
}

/// The full compile: build JSLIR → run the [`pipeline`] over every lowered
/// function body → lift back to JSHIR. This is the oracle seam — as stubs in the
/// pipeline become real passes, the emitted JS converges on upstream's output.
pub fn compile(file: &Op) -> (Op, Stats) {
    let (mut jslir, stats) = build_jslir(file);
    // Seed the fresh-id allocator above every id in the file (passes that build
    // SSA mint ephemeral phi ids from here).
    let base = lift::max_value_id(&jslir) + 1;
    run_pipeline_on_functions(&mut jslir, base);
    (lift_jslir(&jslir), stats)
}

/// Run the pass pipeline over every function body that lowered to a CFG.
fn run_pipeline_on_functions(op: &mut Op, base: u32) {
    if is_function(&op.name) {
        let idx = body_region_index(&op.name);
        if let Some(body) = op.regions.get_mut(idx) {
            if dialect::region_is_cfg(body) {
                pipeline::run_pipeline(body, base);
            }
        }
    }
    for r in &mut op.regions {
        for b in &mut r.blocks {
            for o in &mut b.ops {
                run_pipeline_on_functions(o, base);
            }
        }
    }
}
