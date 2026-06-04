//! build_jslir: flatten structured JSHIR control flow into a JSLIR CFG.

use jsir_ir::{Block, BlockId, Op, Region};

use crate::dialect;

/// Signals the current function uses a construct the builder can't lower yet, so
/// the whole function is passed through unchanged.
pub(crate) struct Unsupported;

/// An incremental CFG: a growing list of identified blocks with a "current"
/// insertion point. Mirrors the upstream HIR builder's reserve/terminate model.
pub(crate) struct Cfg {
    blocks: Vec<Block>,
    next_id: u32,
    current: usize,
    /// Stack of enclosing loops, for `break`/`continue` targets.
    loops: Vec<LoopCtx>,
}

/// Where an unlabeled `break`/`continue` jumps in the current loop.
struct LoopCtx {
    continue_target: BlockId,
    exit: BlockId,
}

impl Cfg {
    fn new() -> Cfg {
        Cfg {
            blocks: vec![Block { id: BlockId(0), args: Vec::new(), ops: Vec::new() }],
            next_id: 1,
            current: 0,
            loops: Vec::new(),
        }
    }

    fn fresh_block(&mut self) -> BlockId {
        let id = BlockId(self.next_id);
        self.next_id += 1;
        self.blocks.push(Block { id, args: Vec::new(), ops: Vec::new() });
        id
    }

    fn switch_to(&mut self, id: BlockId) {
        self.current = self.blocks.iter().position(|b| b.id == id).expect("known block");
    }

    fn push(&mut self, op: Op) {
        self.blocks[self.current].ops.push(op);
    }

    fn is_terminated(&self) -> bool {
        self.blocks[self.current]
            .ops
            .last()
            .map_or(false, |o| dialect::is_terminator(&o.name))
    }

    /// Append a terminator unless the current block already ended in one (e.g. a
    /// branch whose body returned needs no trailing `br`).
    fn terminate(&mut self, term: Op) {
        if !self.is_terminated() {
            self.push(term);
        }
    }

    fn into_region(self) -> Region {
        Region { blocks: self.blocks }
    }
}

/// Lower the statement list of a function body into `cfg`, starting at the
/// current block.
fn lower_stmts(cfg: &mut Cfg, stmts: &[Op]) -> Result<(), Unsupported> {
    for stmt in stmts {
        if cfg.is_terminated() {
            // Statements after a terminator (e.g. code after `return`) are dead.
            break;
        }
        match stmt.name.as_str() {
            "jsir.return_statement" => {
                let v = stmt.operands.first().copied();
                cfg.terminate(dialect::ret(v, false));
            }
            "jshir.break_statement" | "jshir.continue_statement" => {
                // Labeled break/continue (a `label` attr) is handled via the
                // labeled_statement passthrough; only plain ones get a CFG edge.
                if stmt.attrs.iter().any(|(k, _)| k == "label") {
                    return Err(Unsupported);
                }
                let ctx = cfg.loops.last().ok_or(Unsupported)?;
                let (target, kind) = if stmt.name == "jshir.break_statement" {
                    (ctx.exit, "break")
                } else {
                    (ctx.continue_target, "continue")
                };
                cfg.terminate(dialect::br_loop_exit(target, kind));
            }
            "jshir.if_statement" => lower_if(cfg, stmt)?,
            "jshir.while_statement" => lower_while(cfg, stmt)?,
            "jshir.for_statement" => lower_for(cfg, stmt)?,
            "jsir.variable_declaration" => {
                // Flatten a single simple binding into block-level value ops + a
                // `store_local`; keep multi-binding / destructuring declarations
                // coarse (they round-trip fine and flatten in a later increment).
                if let Some((value_ops, store)) = flatten_var_decl(stmt) {
                    for op in value_ops {
                        cfg.push(op);
                    }
                    cfg.push(store);
                } else {
                    cfg.push(stmt.clone());
                }
            }
            n if crate::is_control_flow_stmt(n) => return Err(Unsupported),
            _ => cfg.push(stmt.clone()),
        }
    }
    Ok(())
}

/// Flatten `jshir.if_statement(%cond)({consequent}, {alternate})` into a diamond:
/// `cond_br(%cond)[^then, ^else]`, both arms branching to `^cont`.
fn lower_if(cfg: &mut Cfg, stmt: &Op) -> Result<(), Unsupported> {
    let cond = *stmt.operands.first().ok_or(Unsupported)?;
    let cons_region = stmt.regions.first().ok_or(Unsupported)?;
    let alt_region = stmt.regions.get(1);

    let (cons_stmts, then_braced) = region_stmts(cons_region);
    let has_alt = alt_region.map_or(false, |r| !r.blocks.is_empty());

    let then_blk = cfg.fresh_block();
    let else_blk = if has_alt { Some(cfg.fresh_block()) } else { None };
    let cont_blk = cfg.fresh_block();
    let else_target = else_blk.unwrap_or(cont_blk);

    let else_braced = match (has_alt, alt_region) {
        (true, Some(r)) => region_stmts(r).1,
        _ => false,
    };

    cfg.terminate(dialect::cond_br_if(
        cond, then_blk, else_target, cont_blk, then_braced, else_braced,
    ));

    cfg.switch_to(then_blk);
    lower_stmts(cfg, &cons_stmts)?;
    cfg.terminate(dialect::br(cont_blk));

    if let (Some(eb), Some(r)) = (else_blk, alt_region) {
        let (alt_stmts, _) = region_stmts(r);
        cfg.switch_to(eb);
        lower_stmts(cfg, &alt_stmts)?;
        cfg.terminate(dialect::br(cont_blk));
    }

    cfg.switch_to(cont_blk);
    Ok(())
}

/// Flatten `jshir.while_statement({cond}, {body})` into a loop:
/// `br ^header`; `^header: <cond>; cond_br(%c)[^body, ^exit]`; `^body: <body>; br ^header`.
fn lower_while(cfg: &mut Cfg, stmt: &Op) -> Result<(), Unsupported> {
    let cond_region = stmt.regions.first().ok_or(Unsupported)?;
    let body_region = stmt.regions.get(1).ok_or(Unsupported)?;
    let (cond_ops, cond_val) = cond_region_parts(cond_region).ok_or(Unsupported)?;
    let (body_stmts, body_braced) = region_stmts(body_region);

    let header = cfg.fresh_block();
    let body_blk = cfg.fresh_block();
    let exit_blk = cfg.fresh_block();

    cfg.terminate(dialect::br(header));

    cfg.switch_to(header);
    for op in &cond_ops {
        cfg.push(op.clone());
    }
    cfg.terminate(dialect::cond_br_loop(cond_val, body_blk, exit_blk, body_braced));

    cfg.switch_to(body_blk);
    cfg.loops.push(LoopCtx { continue_target: header, exit: exit_blk });
    let r = lower_stmts(cfg, &body_stmts);
    cfg.loops.pop();
    r?;
    cfg.terminate(dialect::br(header)); // back-edge

    cfg.switch_to(exit_blk);
    Ok(())
}

/// Flatten the canonical `for (let i = ..; test; update) body` into a loop with a
/// preheader (init), header (test), body, and latch (update). Only this form is
/// lowered; other `for` shapes (expression/empty init, missing test/update) pass
/// through.
fn lower_for(cfg: &mut Cfg, stmt: &Op) -> Result<(), Unsupported> {
    let init_region = stmt.regions.first().ok_or(Unsupported)?;
    let test_region = stmt.regions.get(1).ok_or(Unsupported)?;
    let update_region = stmt.regions.get(2).ok_or(Unsupported)?;
    let body_region = stmt.regions.get(3).ok_or(Unsupported)?;

    let init_stmts = init_region
        .blocks
        .first()
        .map(|b| b.ops.clone())
        .unwrap_or_default();
    if init_stmts.len() != 1 || init_stmts[0].name != "jsir.variable_declaration" {
        return Err(Unsupported);
    }
    let (test_ops, test_val) = cond_region_parts(test_region).ok_or(Unsupported)?;
    let (update_ops, update_val) = cond_region_parts(update_region).ok_or(Unsupported)?;
    let (body_stmts, body_braced) = region_stmts(body_region);

    let preheader = cfg.fresh_block();
    let header = cfg.fresh_block();
    let body_blk = cfg.fresh_block();
    let latch = cfg.fresh_block();
    let exit_blk = cfg.fresh_block();

    cfg.terminate(dialect::br(preheader));

    cfg.switch_to(preheader);
    for op in &init_stmts {
        cfg.push(op.clone());
    }
    cfg.terminate(dialect::br(header));

    cfg.switch_to(header);
    for op in &test_ops {
        cfg.push(op.clone());
    }
    cfg.terminate(dialect::cond_br_for(
        test_val, body_blk, exit_blk, preheader, latch, update_val, body_braced,
    ));

    cfg.switch_to(body_blk);
    cfg.loops.push(LoopCtx { continue_target: latch, exit: exit_blk });
    let r = lower_stmts(cfg, &body_stmts);
    cfg.loops.pop();
    r?;
    cfg.terminate(dialect::br(latch));

    cfg.switch_to(latch);
    for op in &update_ops {
        cfg.push(op.clone());
    }
    cfg.terminate(dialect::br(header));

    cfg.switch_to(exit_blk);
    Ok(())
}

/// Flatten `jsir.variable_declaration` for a single simple binding into the init
/// value ops (to hoist to block level) + a `jslir.store_local`. The declaration's
/// region is `[ identifier_ref(name), <init ops>, variable_declarator, exprs_region_end ]`.
/// Returns None (keep coarse) for destructuring or multi-binding declarations.
fn flatten_var_decl(stmt: &Op) -> Option<(Vec<Op>, Op)> {
    let kind = stmt.attrs.iter().find(|(k, _)| k == "kind").and_then(|(_, v)| match v {
        jsir_ir::Attr::Str(s) => Some(s.as_str()),
        _ => None,
    })?;
    let region = stmt.regions.first()?;
    let ops = &region.blocks.first()?.ops;
    let n = ops.len();
    if n < 3 {
        return None;
    }
    // Shape check: ref first, then init ops, then exactly one declarator, then end.
    if ops[0].name != "jsir.identifier_ref"
        || ops[n - 2].name != "jsir.variable_declarator"
        || ops[n - 1].name != "jsir.exprs_region_end"
    {
        return None;
    }
    if ops.iter().filter(|o| o.name == "jsir.variable_declarator").count() != 1 {
        return None;
    }
    let name = ops[0].attrs.iter().find(|(k, _)| k == "name").and_then(|(_, v)| match v {
        jsir_ir::Attr::Str(s) => Some(s.as_str()),
        _ => None,
    })?;
    let init_val = ops[n - 2].operands.get(1).copied();
    let value_ops: Vec<Op> = ops[1..n - 2].to_vec();
    Some((value_ops, dialect::store_local(name, kind, init_val)))
}

/// A `while`/`for` condition region is `{ <eval ops> ; jsir.expr_region_end(%cond) }`.
/// Returns (the evaluation ops, the condition value).
fn cond_region_parts(region: &Region) -> Option<(Vec<Op>, jsir_ir::ValueId)> {
    let block = region.blocks.first()?;
    let last = block.ops.last()?;
    if last.name != "jsir.expr_region_end" {
        return None;
    }
    let cond_val = *last.operands.first()?;
    let eval = block.ops[..block.ops.len() - 1].to_vec();
    Some((eval, cond_val))
}

/// Statements of a control-flow branch region, plus whether they were brace-
/// wrapped: a braced branch is `{ block_statement }`, an unbraced one is the
/// statement ops directly.
fn region_stmts(region: &Region) -> (Vec<Op>, bool) {
    let Some(block) = region.blocks.first() else {
        return (Vec::new(), false);
    };
    if block.ops.len() == 1 && block.ops[0].name == "jshir.block_statement" {
        let bs = &block.ops[0];
        let stmts = bs
            .regions
            .first()
            .and_then(|r| r.blocks.first())
            .map(|b| b.ops.clone())
            .unwrap_or_default();
        (stmts, true)
    } else {
        (block.ops.clone(), false)
    }
}

/// Build a CFG region for a function body's statement list. Returns Err if it
/// hits an unsupported construct.
pub(crate) fn build_cfg(stmts: &[Op], next_value: &mut u32) -> Result<Region, Unsupported> {
    let mut cfg = Cfg::new();
    lower_stmts(&mut cfg, stmts)?;
    // Ensure the final block is well-formed (fall off the end → implicit return).
    cfg.terminate(dialect::ret(None, true));
    let next_block = cfg.next_id;
    let mut region = cfg.into_region();
    finalize(&mut region, next_block, next_value);
    Ok(region)
}

/// Post-build flattening of expression-level control flow (logical &&/||).
/// `next_value` is a global ValueId counter (phi ids must be globally unique).
fn finalize(region: &mut Region, next_block: u32, next_value: &mut u32) {
    let mut nb = next_block;
    crate::expr_flatten::flatten_logicals(region, &mut nb, next_value);
}

/// Build a single-block CFG for an arrow **expression** body `x => <expr>`. The
/// body region is `{ <expr ops> ; jsir.expr_region_end(%v) }`; the CFG is those
/// ops + an `arrow_expr` return of `%v`.
pub(crate) fn build_expr_body_cfg(
    body_ops: &[Op],
    next_value: &mut u32,
) -> Result<Region, Unsupported> {
    let last = body_ops.last().ok_or(Unsupported)?;
    if last.name != "jsir.expr_region_end" {
        return Err(Unsupported);
    }
    let result = *last.operands.first().ok_or(Unsupported)?;
    let mut ops: Vec<Op> = body_ops[..body_ops.len() - 1].to_vec();
    ops.push(dialect::ret_expr(result));
    let mut region = Region {
        blocks: vec![Block { id: BlockId(0), args: Vec::new(), ops }],
    };
    finalize(&mut region, 1, next_value);
    Ok(region)
}
