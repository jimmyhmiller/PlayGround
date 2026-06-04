//! lift_jslir: rebuild structured JSHIR from a JSLIR CFG.
//!
//! Reconstruction walks the CFG from the entry block, emitting instructions in
//! order and turning `cond_br` (with its structured `merge` annotation) back into
//! `jshir.if_statement`. Because the CFG is currently acyclic (loops not lowered
//! yet) and produced by our own builder, a guided traversal suffices; a general
//! relooper/post-dominator reconstruction comes when passes mutate the CFG.

use std::collections::HashMap;

use jsir_ir::{Block, BlockId, Op, Region, ValueId};

use crate::dialect;
use crate::make_block_statement;

/// Mints globally-unique `ValueId`s for ops the lift synthesizes (e.g. the
/// identifier_ref/declarator a `store_local` expands into). Seeded above every
/// existing id so the per-region printer numbering never conflates two defs.
pub(crate) struct Alloc {
    next: u32,
}

impl Alloc {
    pub(crate) fn new(start: u32) -> Alloc {
        Alloc { next: start }
    }
    fn fresh(&mut self) -> ValueId {
        let v = ValueId(self.next);
        self.next += 1;
        v
    }
}

/// The largest `ValueId` defined anywhere in `op` (op results + block args).
pub(crate) fn max_value_id(op: &Op) -> u32 {
    let mut m = op.results.iter().map(|v| v.0).max().unwrap_or(0);
    for r in &op.regions {
        for b in &r.blocks {
            m = m.max(b.args.iter().map(|v| v.0).max().unwrap_or(0));
            for o in &b.ops {
                m = m.max(max_value_id(o));
            }
        }
    }
    m
}

/// If `cfg` is an arrow expression body (it ends, possibly after flattened
/// logical diamonds, in an `arrow_expr` return), reconstruct the body expression:
/// its value ops and the returned value.
pub(crate) fn as_expr_body(cfg: &Region, alloc: &mut Alloc) -> Option<(Vec<Op>, ValueId)> {
    if !cfg.blocks.iter().any(|b| b.ops.last().map_or(false, dialect::is_expr_return)) {
        return None;
    }
    let map: HashMap<BlockId, &Block> = cfg.blocks.iter().map(|b| (b.id, b)).collect();
    let entry = cfg.blocks.first()?.id;
    reconstruct_expr(&map, entry, None, alloc)
}

/// Rebuild an arrow expression-body region: `{ <ops> ; expr_region_end(%val) }`.
pub(crate) fn expr_body_region(ops: Vec<Op>, val: jsir_ir::ValueId) -> Region {
    expr_region(ops, val)
}

pub(crate) fn lift_cfg(cfg: &Region, alloc: &mut Alloc) -> Option<Vec<Op>> {
    let map: HashMap<BlockId, &Block> = cfg.blocks.iter().map(|b| (b.id, b)).collect();
    let entry = cfg.blocks.first()?.id;
    reconstruct(&map, entry, None, alloc)
}

/// Reconstruct the statement sequence from `start`, stopping (exclusive) at
/// `stop` (a merge block). Follows `br`, expands `cond_br` into an `if`, and ends
/// at `return`.
fn reconstruct(
    map: &HashMap<BlockId, &Block>,
    start: BlockId,
    stop: Option<BlockId>,
    alloc: &mut Alloc,
) -> Option<Vec<Op>> {
    let mut stmts = Vec::new();
    let mut cur = start;
    let mut guard = 0usize;
    loop {
        if Some(cur) == stop {
            break;
        }
        guard += 1;
        if guard > 1_000_000 {
            return None; // cycle guard (no loops yet)
        }
        let block = map.get(&cur)?;

        // while-loop header: rebuild a while.
        if block.ops.last().map_or(false, dialect::is_while_header) {
            let term = block.ops.last()?;
            let cond_ops = block.ops[..block.ops.len() - 1].to_vec();
            let cond_val = *term.operands.first()?;
            let body_succ = term.successors.first()?.block;
            let merge = dialect::merge_of(term)?;
            let body_braced = dialect::attr_bool(term, "body_braced");
            // The body branches back to the header; stop reconstruction there.
            let body_stmts = reconstruct(map, body_succ, Some(cur), alloc)?;
            stmts.push(make_while(cond_ops, cond_val, body_stmts, body_braced));
            cur = merge;
            continue;
        }

        // for-loop preheader: this block ends in `br ^header` where ^header is a
        // for-header naming this block as its preheader. Rebuild the whole `for`.
        if let Some((header_id, header)) = as_for_preheader(map, block) {
            let init_stmts = block.ops[..block.ops.len() - 1].to_vec();
            let test_ops = header.ops[..header.ops.len() - 1].to_vec();
            let cond_br = header.ops.last()?;
            let test_val = *cond_br.operands.first()?;
            let body_succ = cond_br.successors.first()?.block;
            let latch_id = dialect::for_latch(cond_br)?;
            let update_val = dialect::for_update_val(cond_br)?;
            let merge = dialect::merge_of(cond_br)?;
            let body_braced = dialect::attr_bool(cond_br, "body_braced");
            let _ = header_id;

            let body_stmts = reconstruct(map, body_succ, Some(latch_id), alloc)?;
            let latch = map.get(&latch_id)?;
            let update_ops = latch.ops[..latch.ops.len() - 1].to_vec();

            stmts.push(make_for(
                init_stmts, test_ops, test_val, update_ops, update_val, body_stmts, body_braced,
            ));
            cur = merge;
            continue;
        }

        let n = block.ops.len();
        let (instrs, term): (&[Op], Option<&Op>) =
            if n > 0 && dialect::is_terminator(&block.ops[n - 1].name) {
                (&block.ops[..n - 1], Some(&block.ops[n - 1]))
            } else {
                (&block.ops[..], None)
            };
        for op in instrs {
            if dialect::is_store_local(op) {
                stmts.push(make_var_decl(op, alloc)?);
            } else {
                stmts.push(op.clone());
            }
        }
        match term {
            None => break,
            Some(t) => match t.name.as_str() {
                dialect::RETURN => {
                    if dialect::is_explicit_return(t) {
                        let mut rs = Op::new("jsir.return_statement");
                        if let Some(v) = t.operands.first() {
                            rs.operands.push(*v);
                        }
                        stmts.push(rs);
                    }
                    break;
                }
                dialect::BR => {
                    match dialect::loop_jump(t) {
                        Some("break") => {
                            stmts.push(Op::new("jshir.break_statement"));
                            break;
                        }
                        Some("continue") => {
                            stmts.push(Op::new("jshir.continue_statement"));
                            break;
                        }
                        _ => cur = t.successors.first()?.block,
                    }
                }
                dialect::COND_BR if dialect::logexpr_operator(t).is_some() => {
                    // A flattened logical expression: rebuild the value op (result =
                    // the merge phi) and continue at the merge in the same statement.
                    let operator = dialect::logexpr_operator(t)?.to_string();
                    let left = *t.operands.first()?;
                    let merge = dialect::merge_of(t)?;
                    let rhs_block = t.successors.iter().map(|s| s.block).find(|b| *b != merge)?;
                    let phi = *map.get(&merge)?.args.first()?;
                    let (rhs_ops, right) = reconstruct_expr(map, rhs_block, Some(merge), alloc)?;
                    stmts.push(make_logical(left, &operator, rhs_ops, right, phi));
                    cur = merge;
                }
                dialect::COND_BR if dialect::is_ternary(t) => {
                    let test = *t.operands.first()?;
                    let cons_block = t.successors.first()?.block;
                    let alt_block = t.successors.get(1)?.block;
                    let merge = dialect::merge_of(t)?;
                    let phi = *map.get(&merge)?.args.first()?;
                    let (cons_ops, cons_val) = reconstruct_expr(map, cons_block, Some(merge), alloc)?;
                    let (alt_ops, alt_val) = reconstruct_expr(map, alt_block, Some(merge), alloc)?;
                    stmts.push(make_conditional(test, alt_ops, alt_val, cons_ops, cons_val, phi));
                    cur = merge;
                }
                dialect::COND_BR => {
                    let cond = *t.operands.first()?;
                    let then_b = t.successors.first()?.block;
                    let else_b = t.successors.get(1)?.block;
                    let merge = dialect::merge_of(t)?;
                    let then_braced = dialect::attr_bool(t, "then_braced");
                    let else_braced = dialect::attr_bool(t, "else_braced");

                    let then_stmts = reconstruct(map, then_b, Some(merge), alloc)?;
                    let (else_stmts, has_else) = if else_b == merge {
                        (Vec::new(), false)
                    } else {
                        (reconstruct(map, else_b, Some(merge), alloc)?, true)
                    };
                    stmts.push(make_if(
                        cond, then_stmts, then_braced, has_else, else_stmts, else_braced,
                    ));
                    cur = merge;
                }
                _ => return None,
            },
        }
    }
    Some(stmts)
}

fn make_if(
    cond: jsir_ir::ValueId,
    then_stmts: Vec<Op>,
    then_braced: bool,
    has_else: bool,
    else_stmts: Vec<Op>,
    else_braced: bool,
) -> Op {
    let mut op = Op::new("jshir.if_statement");
    op.operands.push(cond);
    op.regions.push(branch_region(then_stmts, then_braced));
    if has_else {
        op.regions.push(branch_region(else_stmts, else_braced));
    } else {
        // No else: a zero-block region, exactly as JSHIR emits it.
        op.regions.push(Region::default());
    }
    op
}

/// Detect a `for` preheader: `block` ends in `br ^H` and `^H` is a for-header
/// whose `preheader` is `block`. Returns (header id, header block).
fn as_for_preheader<'a>(
    map: &HashMap<BlockId, &'a Block>,
    block: &Block,
) -> Option<(BlockId, &'a Block)> {
    let last = block.ops.last()?;
    if last.name != dialect::BR {
        return None;
    }
    let header_id = last.successors.first()?.block;
    let header = *map.get(&header_id)?;
    let cond_br = header.ops.last()?;
    if dialect::is_for_header(cond_br) && dialect::for_preheader(cond_br) == Some(block.id) {
        Some((header_id, header))
    } else {
        None
    }
}

/// Reconstruct a flattened expression region (the rhs of a logical) from `start`
/// up to the `br ^stop(%v)` into the merge block. Handles nested logicals.
/// Returns the value ops and the resulting value `%v`.
fn reconstruct_expr(
    map: &HashMap<BlockId, &Block>,
    start: BlockId,
    stop: Option<BlockId>,
    alloc: &mut Alloc,
) -> Option<(Vec<Op>, ValueId)> {
    let mut ops = Vec::new();
    let mut cur = start;
    let mut guard = 0usize;
    loop {
        guard += 1;
        if guard > 100_000 {
            return None;
        }
        let block = map.get(&cur)?;
        let n = block.ops.len();
        if n == 0 {
            return None;
        }
        for op in &block.ops[..n - 1] {
            if dialect::is_store_local(op) {
                ops.push(make_var_decl(op, alloc)?);
            } else {
                ops.push(op.clone());
            }
        }
        let term = &block.ops[n - 1];
        if term.name == dialect::BR {
            let succ = term.successors.first()?;
            if Some(succ.block) != stop {
                return None;
            }
            return Some((ops, *succ.args.first()?));
        } else if dialect::is_expr_return(term) {
            // Arrow expression body terminator.
            return Some((ops, *term.operands.first()?));
        } else if let Some(operator) = dialect::logexpr_operator(term) {
            // Nested logical: emit its value op (result = its phi), continue at merge.
            let left = *term.operands.first()?;
            let merge = dialect::merge_of(term)?;
            let rhs_block = term.successors.iter().map(|s| s.block).find(|b| *b != merge)?;
            let phi = *map.get(&merge)?.args.first()?;
            let operator = operator.to_string();
            let (rhs_ops, right) = reconstruct_expr(map, rhs_block, Some(merge), alloc)?;
            ops.push(make_logical(left, &operator, rhs_ops, right, phi));
            cur = merge;
        } else if dialect::is_ternary(term) {
            let test = *term.operands.first()?;
            let cons_block = term.successors.first()?.block;
            let alt_block = term.successors.get(1)?.block;
            let merge = dialect::merge_of(term)?;
            let phi = *map.get(&merge)?.args.first()?;
            let (cons_ops, cons_val) = reconstruct_expr(map, cons_block, Some(merge), alloc)?;
            let (alt_ops, alt_val) = reconstruct_expr(map, alt_block, Some(merge), alloc)?;
            ops.push(make_conditional(test, alt_ops, alt_val, cons_ops, cons_val, phi));
            cur = merge;
        } else {
            return None;
        }
    }
}

/// Build `jshir.logical_expression(%left) <{operator_}> ({ rhs ; expr_region_end(%right) })`
/// with result `%result` (the value subsequent ops reference).
fn make_logical(
    left: ValueId,
    operator: &str,
    rhs_ops: Vec<Op>,
    right: ValueId,
    result: ValueId,
) -> Op {
    let mut op = Op::new("jshir.logical_expression");
    op.operands.push(left);
    op.attrs.push(("operator_".into(), jsir_ir::Attr::Str(operator.to_string())));
    op.regions.push(expr_region(rhs_ops, right));
    op.results.push(result);
    op
}

/// Build `jshir.conditional_expression(%test) ({ alternate }, { consequent })`
/// (upstream's region order) with result `%result`.
fn make_conditional(
    test: ValueId,
    alt_ops: Vec<Op>,
    alt_val: ValueId,
    cons_ops: Vec<Op>,
    cons_val: ValueId,
    result: ValueId,
) -> Op {
    let mut op = Op::new("jshir.conditional_expression");
    op.operands.push(test);
    op.regions.push(expr_region(alt_ops, alt_val));
    op.regions.push(expr_region(cons_ops, cons_val));
    op.results.push(result);
    op
}

fn make_while(
    cond_ops: Vec<Op>,
    cond_val: jsir_ir::ValueId,
    body_stmts: Vec<Op>,
    body_braced: bool,
) -> Op {
    let mut op = Op::new("jshir.while_statement");
    op.regions.push(expr_region(cond_ops, cond_val));
    op.regions.push(branch_region(body_stmts, body_braced));
    op
}

#[allow(clippy::too_many_arguments)]
fn make_for(
    init_stmts: Vec<Op>,
    test_ops: Vec<Op>,
    test_val: jsir_ir::ValueId,
    update_ops: Vec<Op>,
    update_val: jsir_ir::ValueId,
    body_stmts: Vec<Op>,
    body_braced: bool,
) -> Op {
    let mut op = Op::new("jshir.for_statement");
    // init region: the statements directly (a variable_declaration).
    op.regions.push(Region::with_block(Block::leaf(init_stmts)));
    op.regions.push(expr_region(test_ops, test_val));
    op.regions.push(expr_region(update_ops, update_val));
    op.regions.push(branch_region(body_stmts, body_braced));
    op
}

/// Expand a `jslir.store_local{name, kind}(%init?)` back into a single-binding
/// `jsir.variable_declaration`. The init value (and its defining ops) live at
/// block level; hir2ast resolves the reference via its global value→def index, so
/// we only rebuild the declaration skeleton (identifier_ref + declarator + end)
/// with fresh ids for the ref and declarator results.
fn make_var_decl(store: &Op, alloc: &mut Alloc) -> Option<Op> {
    let (name, kind) = dialect::store_local_parts(store)?;
    let init_val = store.operands.first().copied();

    let ref_val = alloc.fresh();
    let decl_val = alloc.fresh();

    let mut id_ref = Op::new("jsir.identifier_ref");
    id_ref.attrs.push(("name".into(), jsir_ir::Attr::Str(name.to_string())));
    id_ref.results.push(ref_val);

    let mut declarator = Op::new("jsir.variable_declarator");
    declarator.operands.push(ref_val);
    if let Some(iv) = init_val {
        declarator.operands.push(iv);
    }
    declarator.results.push(decl_val);

    let mut end = Op::new("jsir.exprs_region_end");
    end.operands.push(decl_val);

    let mut vd = Op::new("jsir.variable_declaration");
    vd.attrs.push(("kind".into(), jsir_ir::Attr::Str(kind.to_string())));
    vd.regions.push(Region::with_block(Block::leaf(vec![id_ref, declarator, end])));
    Some(vd)
}

/// An expression region: `{ <ops> ; jsir.expr_region_end(%val) }`.
fn expr_region(mut ops: Vec<Op>, val: jsir_ir::ValueId) -> Region {
    let mut end = Op::new("jsir.expr_region_end");
    end.operands.push(val);
    ops.push(end);
    Region::with_block(Block::leaf(ops))
}

/// A braced branch wraps its statements in a `block_statement`; an unbraced one
/// holds the statement ops directly.
fn branch_region(stmts: Vec<Op>, braced: bool) -> Region {
    if braced {
        Region::with_block(Block::leaf(vec![make_block_statement(stmts)]))
    } else {
        Region::with_block(Block::leaf(stmts))
    }
}
