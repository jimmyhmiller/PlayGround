//! `DeadCodeElimination` — mark-and-sweep removal of instructions whose values are
//! unused, preserving side effects (upstream `Optimization/DeadCodeElimination.ts`
//! / the Rust port's `dead_code_elimination.rs`). Runs after the
//! mutation/aliasing passes upstream; we port the structural core here because it
//! is what cleans up the dead reads/stores `fold_constants` leaves behind.
//!
//! Operates directly on a JSLIR function-body CFG, in place. It exploits two facts
//! about our IR + lift:
//! - within-block flattening means every side-effecting subexpression is its *own*
//!   block-level op (a `call_expression` etc. is never hidden inside a "pure" op),
//!   so an operand-based mark/sweep is sound; and
//! - `hir2ast` emits a block op as a statement only when it has no result —
//!   value ops are inlined at use sites and an *unused* value op simply vanishes.
//!
//! The second fact makes pruning pure dead ops trivial (they vanish on their own
//! once unreferenced) but means a *side-effecting* op orphaned by DCE (e.g. the
//! `g()` in a dropped `let x = g();`) would silently lose its effect. So a final
//! pass re-wraps every kept, side-effecting, now-unused value op in a
//! `jsir.expression_statement` — exactly the `g();` upstream emits.
//!
//! Liveness mirrors upstream `pruneable_value`: pure/read-only ops
//! (literals, identifier loads, binary/unary, array/object/member, template,
//! function/class expressions) are removable when unused; everything else (calls,
//! `new`, assignments, updates, `await`, deletes/stores, tagged templates,
//! `debugger`, …) is retained for its effects. A `store_local` is removable when
//! its variable name is never read.

use std::collections::HashSet;

use jsir_ir::{Op, Region, ValueId};

use crate::dialect;

/// Read-only / side-effect-free op names — safe to drop when their result is
/// unused. Mirrors the `true` arm of upstream `pruneable_value`. Everything not
/// listed is conservatively retained.
const PURE_OPS: &[&str] = &[
    "jsir.numeric_literal",
    "jsir.string_literal",
    "jsir.boolean_literal",
    "jsir.null_literal",
    "jsir.bigint_literal",
    "jsir.regexp_literal",
    "jsir.identifier",
    "jsir.binary_expression",
    "jsir.unary_expression",
    "jsir.array_expression",
    "jsir.object_expression",
    "jsir.member_expression",
    "jsir.template_literal",
    "jsir.function_expression",
    "jsir.arrow_function_expression",
    "jsir.class_expression",
];

fn is_pure(op: &Op) -> bool {
    PURE_OPS.contains(&op.name.as_str())
}

/// Side-effecting *expression* ops that are valid as a standalone
/// `expression_statement` (`g();`, `x = 1;`, `i++;`, …). Only these are
/// re-stated when DCE orphans them; structural debris (lvalue refs, spreads,
/// pattern refs) that loses its parent is just dropped — it carries no effect.
const EFFECTFUL_EXPR_OPS: &[&str] = &[
    "jsir.call_expression",
    "jsir.optional_call_expression",
    "jsir.new_expression",
    "jsir.assignment_expression",
    "jsir.update_expression",
    "jsir.await_expression",
    "jsir.yield_expression",
    "jsir.tagged_template_expression",
];

fn is_effectful_expr(op: &Op) -> bool {
    EFFECTFUL_EXPR_OPS.contains(&op.name.as_str())
}

/// The variable name a `jsir.identifier` read refers to (its resolved symbol).
fn identifier_name(op: &Op) -> Option<String> {
    if op.name != "jsir.identifier" {
        return None;
    }
    op.trivia
        .as_ref()
        .and_then(|t| t.referenced_symbol.as_ref())
        .map(|s| s.name.clone())
}

/// Eliminate dead code in a function-body CFG, in place. Returns the number of ops
/// removed. Pure dead value ops are dropped (the lift omits them); dead
/// `store_local`s are dropped; side effects are always preserved (orphaned
/// side-effecting ops are re-stated as `expression_statement`s).
pub fn eliminate_dead_code(region: &mut Region) -> usize {
    let (live_vals, live_names) = mark(region);
    let before = count_ops(region);
    sweep(region, &live_vals, &live_names);
    let removed = before.saturating_sub(count_ops(region));
    restate_orphans(region); // only adds expression_statements; not counted as removals
    removed
}

fn count_ops(region: &Region) -> usize {
    region.blocks.iter().map(|b| b.ops.len()).sum()
}

/// Phase 1 — mark referenced values + used variable names, to a fixpoint.
fn mark(region: &Region) -> (HashSet<ValueId>, HashSet<String>) {
    // Map each ValueId to its defining op (for expression-statement liveness).
    let mut val2op: std::collections::HashMap<ValueId, &Op> = std::collections::HashMap::new();
    for b in &region.blocks {
        for op in &b.ops {
            for r in &op.results {
                val2op.insert(*r, op);
            }
        }
    }

    let mut live_vals: HashSet<ValueId> = HashSet::new();
    let mut live_names: HashSet<String> = HashSet::new();

    // Roots: terminator operands + successor (phi) args are always live.
    for b in &region.blocks {
        if let Some(t) = b.ops.last() {
            if dialect::is_terminator(&t.name) {
                live_vals.extend(t.operands.iter().copied());
                for s in &t.successors {
                    live_vals.extend(s.args.iter().copied());
                }
                // A `for`-header carries its update-expression value in an attr (not
                // an operand); the lift needs it, so it's a root.
                if let Some(u) = dialect::for_update_val(t) {
                    live_vals.insert(u);
                }
            }
        }
    }

    loop {
        let before = (live_vals.len(), live_names.len());
        for b in &region.blocks {
            let n = b.ops.len();
            for (i, op) in b.ops.iter().enumerate() {
                if i == n - 1 && dialect::is_terminator(&op.name) {
                    continue;
                }
                if op_is_live(op, &live_vals, &live_names, &val2op) {
                    live_vals.extend(op.operands.iter().copied());
                    if let Some(name) = identifier_name(op) {
                        live_names.insert(name);
                    }
                }
            }
        }
        if (live_vals.len(), live_names.len()) == before {
            break;
        }
    }
    (live_vals, live_names)
}

/// Is a (non-terminator) block op live — i.e. must be kept?
fn op_is_live(
    op: &Op,
    live_vals: &HashSet<ValueId>,
    live_names: &HashSet<String>,
    val2op: &std::collections::HashMap<ValueId, &Op>,
) -> bool {
    if dialect::is_store_local(op) {
        // Live iff the declared/assigned name is read somewhere live.
        return dialect::store_local_parts(op)
            .map(|(name, _)| live_names.contains(name))
            .unwrap_or(true);
    }
    if op.name == "jsir.expression_statement" {
        // A bare `expr;` is kept only for its side effects: live iff its expression
        // op is non-pure. (A pure `1;` is dead.)
        return op
            .operands
            .first()
            .and_then(|v| val2op.get(v))
            .map(|def| !is_pure(def))
            .unwrap_or(true);
    }
    match op.results.first() {
        // A value op: live if its result is used, or it has side effects.
        Some(r) => live_vals.contains(r) || !is_pure(op),
        // A no-result op that isn't a store_local / expression_statement (other
        // statement forms): retain conservatively.
        None => true,
    }
}

/// Phase 2 — drop ops that aren't live.
fn sweep(region: &mut Region, live_vals: &HashSet<ValueId>, live_names: &HashSet<String>) {
    let val2op_owned: std::collections::HashMap<ValueId, Op> = region
        .blocks
        .iter()
        .flat_map(|b| &b.ops)
        .flat_map(|op| op.results.iter().map(move |r| (*r, op.clone())))
        .collect();
    let val2op: std::collections::HashMap<ValueId, &Op> =
        val2op_owned.iter().map(|(k, v)| (*k, v)).collect();

    for b in &mut region.blocks {
        let n = b.ops.len();
        let mut idx = 0;
        b.ops.retain(|op| {
            let keep = (idx == n - 1 && dialect::is_terminator(&op.name))
                || op_is_live(op, live_vals, live_names, &val2op);
            idx += 1;
            keep
        });
    }
}

/// Phase 3 — re-state orphaned side effects. After the sweep, any kept,
/// side-effecting value op whose result is no longer referenced (e.g. the `g()`
/// from a dropped `let x = g();`) must be emitted for its effect, so wrap it in an
/// `expression_statement` right after its definition.
fn restate_orphans(region: &mut Region) {
    // Everything referenced by a surviving op (operands) or terminator (successor
    // args) across the whole region.
    let mut referenced: HashSet<ValueId> = HashSet::new();
    for b in &region.blocks {
        for op in &b.ops {
            referenced.extend(op.operands.iter().copied());
            for s in &op.successors {
                referenced.extend(s.args.iter().copied());
            }
            if let Some(u) = dialect::for_update_val(op) {
                referenced.insert(u);
            }
        }
    }

    for b in &mut region.blocks {
        let n = b.ops.len();
        let mut rebuilt: Vec<Op> = Vec::with_capacity(n);
        for (i, op) in b.ops.drain(..).enumerate() {
            let is_term = i == n - 1 && dialect::is_terminator(&op.name);
            let orphan = !is_term
                && is_effectful_expr(&op)
                && op.results.first().is_some_and(|r| !referenced.contains(r));
            let result = op.results.first().copied();
            rebuilt.push(op);
            if orphan {
                if let Some(r) = result {
                    let mut es = Op::new("jsir.expression_statement");
                    es.operands.push(r);
                    rebuilt.push(es);
                }
            }
        }
        b.ops = rebuilt;
    }
}
