//! Flatten expression-level control flow into the CFG.
//!
//! JSHIR keeps short-circuiting operators as value ops with a nested region
//! (`jshir.logical_expression(%left)({ rhs })`). The React passes need this as
//! real control flow, so we split the block: the rhs becomes its own block, the
//! result becomes a merge-block **argument** (a phi), and the short-circuit is a
//! `cond_br`. This is the first use of block arguments.
//!
//! Currently handles `&&` / `||`. Ternary and optional chaining are next.

use jsir_ir::{Block, BlockId, Op, Region, ValueId};

use crate::dialect;

/// Flatten logical expressions in a function-body CFG. `next_block` and
/// `next_value` allocate fresh ids (block ids region-scoped, value ids global).
///
/// A logical in a **loop-edge** block (a loop header, or a block that branches to
/// one — a back-edge, preheader, or latch) is left coarse: splitting it would
/// tangle the loop's back-edge/structure annotations. Such a logical round-trips
/// fine as a nested `logical_expression` op; it just isn't turned into control
/// flow here. Every other position (return/declaration/if-test/if-branch/loop
/// body) is flattened.
pub(crate) fn flatten_logicals(region: &mut Region, next_block: &mut u32, next_value: &mut u32) {
    loop {
        let loop_headers: std::collections::HashSet<u32> = region
            .blocks
            .iter()
            .filter(|b| b.ops.last().map_or(false, dialect::is_loop_header))
            .map(|b| b.id.0)
            .collect();
        let is_loop_edge = |b: &Block| -> bool {
            if b.ops.last().map_or(false, dialect::is_loop_header) {
                return true;
            }
            // A `br` whose target is a loop header (back-edge / preheader / latch).
            b.ops.last().map_or(false, |t| {
                t.name == dialect::BR
                    && t.successors
                        .first()
                        .map_or(false, |s| loop_headers.contains(&s.block.0))
            })
        };

        let Some(bi) = region.blocks.iter().position(|b| {
            !is_loop_edge(b) && b.ops.iter().any(is_flattenable_expr)
        }) else {
            return;
        };
        split_block_at_expr(region, bi, next_block, next_value);
    }
}

fn is_flattenable_expr(op: &Op) -> bool {
    op.name == "jshir.logical_expression" || op.name == "jshir.conditional_expression"
}

fn fresh_block(next_block: &mut u32) -> BlockId {
    let id = BlockId(*next_block);
    *next_block += 1;
    id
}
fn fresh_value(next_value: &mut u32) -> ValueId {
    let v = ValueId(*next_value);
    *next_value += 1;
    v
}

/// The `(<eval ops>, value)` of an expression region `{ ..; expr_region_end(%v) }`.
fn region_eval(region: &Region) -> (Vec<Op>, ValueId) {
    let block = &region.blocks[0];
    let n = block.ops.len();
    (block.ops[..n - 1].to_vec(), block.ops[n - 1].operands[0])
}

/// Split `region.blocks[bi]` at its first flattenable expression (logical/ternary).
fn split_block_at_expr(region: &mut Region, bi: usize, next_block: &mut u32, next_value: &mut u32) {
    let ops = std::mem::take(&mut region.blocks[bi].ops);
    let i = ops.iter().position(is_flattenable_expr).expect("caller checked");
    let expr = ops[i].clone();
    let result = expr.results[0];

    let pre: Vec<Op> = ops[..i].to_vec();
    let term = ops[ops.len() - 1].clone();
    let post: Vec<Op> = ops[i + 1..ops.len() - 1].to_vec();

    let merge_id = fresh_block(next_block);
    let phi = fresh_value(next_value);

    // The expression's result is now the merge block's phi: rewrite all uses.
    let post: Vec<Op> = post.iter().map(|o| subst(o, result, phi)).collect();
    let term = subst(&term, result, phi);

    let mut new_blocks: Vec<Block> = Vec::new();

    let cond_br = if expr.name == "jshir.logical_expression" {
        // One rhs branch + a short-circuit edge straight to merge.
        let left = expr.operands[0];
        let operator = expr
            .attrs
            .iter()
            .find(|(k, _)| k == "operator_")
            .and_then(|(_, v)| match v {
                jsir_ir::Attr::Str(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "&&".to_string());
        let (rhs_eval, right) = region_eval(&expr.regions[0]);
        let rhs_id = fresh_block(next_block);
        let mut rhs_ops = rhs_eval;
        rhs_ops.push(dialect::br_args(merge_id, vec![right]));
        new_blocks.push(Block { id: rhs_id, args: Vec::new(), ops: rhs_ops });
        dialect::cond_br_logical(left, &operator, rhs_id, merge_id)
    } else {
        // Ternary: two full branches. regions are [alternate, consequent].
        let test = expr.operands[0];
        let (alt_eval, alt_val) = region_eval(&expr.regions[0]);
        let (cons_eval, cons_val) = region_eval(&expr.regions[1]);
        let cons_id = fresh_block(next_block);
        let alt_id = fresh_block(next_block);
        let mut cons_ops = cons_eval;
        cons_ops.push(dialect::br_args(merge_id, vec![cons_val]));
        new_blocks.push(Block { id: cons_id, args: Vec::new(), ops: cons_ops });
        let mut alt_ops = alt_eval;
        alt_ops.push(dialect::br_args(merge_id, vec![alt_val]));
        new_blocks.push(Block { id: alt_id, args: Vec::new(), ops: alt_ops });
        dialect::cond_br_ternary(test, cons_id, alt_id, merge_id)
    };

    // current block: pre + cond_br
    let mut cur = pre;
    cur.push(cond_br);
    region.blocks[bi].ops = cur;
    region.blocks.extend(new_blocks);

    // merge block: arg %phi ; post + terminator
    let mut merge_ops = post;
    merge_ops.push(term);
    region.blocks.push(Block { id: merge_id, args: vec![phi], ops: merge_ops });
}

/// Replace every use of `from` with `to` in `op` (operands, successor args, and
/// recursively in nested regions).
fn subst(op: &Op, from: ValueId, to: ValueId) -> Op {
    let mut new = op.clone();
    for v in &mut new.operands {
        if *v == from {
            *v = to;
        }
    }
    for s in &mut new.successors {
        for v in &mut s.args {
            if *v == from {
                *v = to;
            }
        }
    }
    for r in &mut new.regions {
        for b in &mut r.blocks {
            b.ops = b.ops.iter().map(|o| subst(o, from, to)).collect();
        }
    }
    new
}
