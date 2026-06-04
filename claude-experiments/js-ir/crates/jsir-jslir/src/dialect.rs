//! The JSLIR dialect: op-name conventions + terminator constructors.
//!
//! JSLIR reuses the AST-faithful `jsir.*` value/statement ops verbatim as a basic
//! block's instructions (so all of `source_to_ir`'s expression lowering is reused
//! unchanged). What JSLIR ADDS is control flow as a real CFG: structured
//! `jshir.if_statement`/`while_statement`/... ops are replaced by `jslir.*`
//! terminator ops that end a block and name successor blocks (via the generic
//! `Op.successors` added to jsir-ir).

use jsir_ir::{Attr, BlockId, Op, Successor, ValueId};

/// Unconditional branch: `jslir.br [^target]`.
pub const BR: &str = "jslir.br";
/// Conditional branch on operand 0: `jslir.cond_br(%c)[^then, ^else]`.
pub const COND_BR: &str = "jslir.cond_br";
/// Return: `jslir.return(%v?)`. Ends a block with no successors.
pub const RETURN: &str = "jslir.return";
/// A flattened single-binding declaration: `jslir.store_local <{name, decl_kind}>(%init?)`
/// — the analog of React HIR's `StoreLocal`/`DeclareLocal`. A block-level
/// instruction (not a terminator). `%init` is absent for `let x;`.
pub const STORE_LOCAL: &str = "jslir.store_local";

/// Build a `jslir.store_local` for `<kind> <name> = <init>`.
pub fn store_local(name: &str, kind: &str, init: Option<ValueId>) -> Op {
    let mut op = Op::new(STORE_LOCAL);
    op.attrs.push(("decl_kind".into(), Attr::Str(kind.into())));
    op.attrs.push(("name".into(), Attr::Str(name.into())));
    if let Some(v) = init {
        op.operands.push(v);
    }
    op
}

pub fn is_store_local(op: &Op) -> bool {
    op.name == STORE_LOCAL
}

/// `(name, decl_kind)` of a `jslir.store_local`.
pub fn store_local_parts(op: &Op) -> Option<(&str, &str)> {
    let name = op.attrs.iter().find(|(k, _)| k == "name").and_then(|(_, v)| match v {
        Attr::Str(s) => Some(s.as_str()),
        _ => None,
    })?;
    let kind = op.attrs.iter().find(|(k, _)| k == "decl_kind").and_then(|(_, v)| match v {
        Attr::Str(s) => Some(s.as_str()),
        _ => None,
    })?;
    Some((name, kind))
}

/// Every JSLIR terminator op name. Presence of any of these in a region marks it
/// as a lowered CFG (vs an untransformed JSHIR body) for the lift to detect.
pub const TERMINATORS: &[&str] = &[BR, COND_BR, RETURN];

pub fn is_terminator(name: &str) -> bool {
    TERMINATORS.contains(&name)
}

/// `jslir.return(%v)` (explicit `return v;`) or `jslir.return()` (explicit bare
/// `return;`). `implicit` marks a synthesized fall-off-the-end return that the
/// lift must NOT print.
pub fn ret(value: Option<ValueId>, implicit: bool) -> Op {
    let mut op = Op::new(RETURN);
    if let Some(v) = value {
        op.operands.push(v);
    }
    if implicit {
        op.attrs.push(("implicit".into(), Attr::Bool(true)));
    }
    op
}

/// `jslir.return(%v)` for an arrow **expression** body (`x => v`). Marked
/// `arrow_expr` so the lift restores an expression body, not `{ return v; }`.
pub fn ret_expr(value: ValueId) -> Op {
    let mut op = Op::new(RETURN);
    op.operands.push(value);
    op.attrs.push(("arrow_expr".into(), Attr::Bool(true)));
    op
}

/// Is this a `jslir.return` marking an arrow expression body?
pub fn is_expr_return(op: &Op) -> bool {
    op.name == RETURN && attr_bool(op, "arrow_expr")
}

/// `jslir.br [^target]`.
pub fn br(target: BlockId) -> Op {
    let mut op = Op::new(BR);
    op.successors.push(Successor { block: target, args: Vec::new() });
    op
}

/// `jslir.cond_br(%cond)[^then, ^else]`.
pub fn cond_br(cond: ValueId, then_blk: BlockId, else_blk: BlockId) -> Op {
    let mut op = Op::new(COND_BR);
    op.operands.push(cond);
    op.successors.push(Successor { block: then_blk, args: Vec::new() });
    op.successors.push(Successor { block: else_blk, args: Vec::new() });
    op
}

/// `jslir.br [^target(%args)]` — an unconditional branch passing block arguments.
pub fn br_args(target: BlockId, args: Vec<ValueId>) -> Op {
    let mut op = Op::new(BR);
    op.successors.push(Successor { block: target, args });
    op
}

/// A `br` that came from a `break` (to the loop exit) or `continue` (to the loop
/// continue target). Marked so the lift rebuilds the statement instead of
/// following the edge. `kind` is "break" or "continue".
pub fn br_loop_exit(target: BlockId, kind: &str) -> Op {
    let mut op = br(target);
    op.attrs.push(("loopjump".into(), Attr::Str(kind.into())));
    op
}

/// If this `br` is a `break`/`continue`, its kind.
pub fn loop_jump(op: &Op) -> Option<&str> {
    if op.name != BR {
        return None;
    }
    op.attrs.iter().find(|(k, _)| k == "loopjump").and_then(|(_, v)| match v {
        Attr::Str(s) => Some(s.as_str()),
        _ => None,
    })
}

/// A `cond_br` that flattens a logical expression `left && rhs` / `left || rhs`.
/// The result is the `merge` block's argument (a phi): the short-circuit edge
/// passes `left`, the rhs edge passes the rhs value. For `&&`, a truthy `left`
/// continues to `rhs`; for `||`, a truthy `left` short-circuits to `merge`.
pub fn cond_br_logical(left: ValueId, operator: &str, rhs: BlockId, merge: BlockId) -> Op {
    let mut op = Op::new(COND_BR);
    op.operands.push(left);
    let rhs_succ = Successor { block: rhs, args: Vec::new() };
    let merge_succ = Successor { block: merge, args: vec![left] };
    match operator {
        // `&&`: truthy left -> evaluate rhs (then); falsy -> merge with left (else).
        "&&" => {
            op.successors.push(rhs_succ);
            op.successors.push(merge_succ);
        }
        // `||`: truthy left -> merge with left (then); falsy -> evaluate rhs (else).
        _ => {
            op.successors.push(merge_succ);
            op.successors.push(rhs_succ);
        }
    }
    op.attrs.push(("logexpr".into(), Attr::Str(operator.into())));
    op.attrs.push(("merge".into(), Attr::I64(merge.0 as i64)));
    op
}

/// A `cond_br` that flattens a ternary `test ? cons : alt`. `then` = consequent
/// block, `else` = alternate block; both branch to `merge`, passing their value
/// as the merge phi.
pub fn cond_br_ternary(test: ValueId, cons_blk: BlockId, alt_blk: BlockId, merge: BlockId) -> Op {
    let mut op = cond_br(test, cons_blk, alt_blk);
    op.attrs.push(("condexpr".into(), Attr::Bool(true)));
    op.attrs.push(("merge".into(), Attr::I64(merge.0 as i64)));
    op
}

/// Is this `cond_br` a flattened ternary (conditional expression)?
pub fn is_ternary(op: &Op) -> bool {
    op.name == COND_BR && attr_bool(op, "condexpr")
}

/// Is this `cond_br` a flattened logical expression? Returns its operator.
pub fn logexpr_operator(op: &Op) -> Option<&str> {
    if op.name != COND_BR {
        return None;
    }
    op.attrs.iter().find(|(k, _)| k == "logexpr").and_then(|(_, v)| match v {
        Attr::Str(s) => Some(s.as_str()),
        _ => None,
    })
}

/// A `cond_br` carrying structured-reconstruction metadata for an `if`: the merge
/// (join) block both arms reconverge at, and whether each arm was brace-wrapped in
/// source (so the lift rebuilds `if (c) { .. }` vs `if (c) ..` faithfully). This
/// mirrors SPIR-V's `OpSelectionMerge` — explicit structure carried on the CFG.
pub fn cond_br_if(
    cond: ValueId,
    then_blk: BlockId,
    else_blk: BlockId,
    merge: BlockId,
    then_braced: bool,
    else_braced: bool,
) -> Op {
    let mut op = cond_br(cond, then_blk, else_blk);
    op.attrs.push(("merge".into(), Attr::I64(merge.0 as i64)));
    op.attrs.push(("then_braced".into(), Attr::Bool(then_braced)));
    op.attrs.push(("else_braced".into(), Attr::Bool(else_braced)));
    op
}

/// A `while`-loop header `cond_br`: the header holds the condition evaluation;
/// `then`=body (which branches back to the header), `else`=`merge`=exit. Marked
/// `loop`+`kind="while"`. (SPIR-V `OpLoopMerge` analog.)
pub fn cond_br_loop(
    cond: ValueId,
    body_blk: BlockId,
    exit_blk: BlockId,
    body_braced: bool,
) -> Op {
    let mut op = cond_br(cond, body_blk, exit_blk);
    op.attrs.push(("loop".into(), Attr::Bool(true)));
    op.attrs.push(("kind".into(), Attr::Str("while".into())));
    op.attrs.push(("merge".into(), Attr::I64(exit_blk.0 as i64)));
    op.attrs.push(("body_braced".into(), Attr::Bool(body_braced)));
    op
}

/// A `for`-loop header `cond_br`. Like `while`, but carries the surrounding
/// structure: the `preheader` block (the `for` init runs there), the `latch`
/// block (the `for` update runs there, before the back-edge), and the update
/// expression's value (so the lift rebuilds the update region's expr_region_end).
#[allow(clippy::too_many_arguments)]
pub fn cond_br_for(
    cond: ValueId,
    body_blk: BlockId,
    exit_blk: BlockId,
    preheader: BlockId,
    latch: BlockId,
    update_val: ValueId,
    body_braced: bool,
) -> Op {
    let mut op = cond_br(cond, body_blk, exit_blk);
    op.attrs.push(("loop".into(), Attr::Bool(true)));
    op.attrs.push(("kind".into(), Attr::Str("for".into())));
    op.attrs.push(("merge".into(), Attr::I64(exit_blk.0 as i64)));
    op.attrs.push(("preheader".into(), Attr::I64(preheader.0 as i64)));
    op.attrs.push(("latch".into(), Attr::I64(latch.0 as i64)));
    op.attrs.push(("update_val".into(), Attr::I64(update_val.0 as i64)));
    op.attrs.push(("body_braced".into(), Attr::Bool(body_braced)));
    op
}

fn attr_str<'a>(op: &'a Op, key: &str) -> Option<&'a str> {
    op.attrs.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        Attr::Str(s) => Some(s.as_str()),
        _ => None,
    })
}

fn attr_i64(op: &Op, key: &str) -> Option<i64> {
    op.attrs.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        Attr::I64(n) => Some(*n),
        _ => None,
    })
}

/// Is this terminator any loop header (`while` or `for`)?
pub fn is_loop_header(op: &Op) -> bool {
    op.name == COND_BR && attr_bool(op, "loop")
}

/// Is this a `cond_br` built by [`cond_br_if`] — a flattened `if` statement
/// diamond (successors `[then, else]`, carrying brace metadata)? Distinguished
/// from ternary/logical/loop `cond_br`s, which carry their own markers.
pub fn is_if_header(op: &Op) -> bool {
    op.name == COND_BR
        && !is_ternary(op)
        && logexpr_operator(op).is_none()
        && !is_loop_header(op)
        && op.attrs.iter().any(|(k, _)| k == "then_braced")
}

/// Is this terminator a `while`-loop header (not a `for`)?
pub fn is_while_header(op: &Op) -> bool {
    is_loop_header(op) && attr_str(op, "kind") != Some("for")
}

/// Is this terminator a `for`-loop header?
pub fn is_for_header(op: &Op) -> bool {
    is_loop_header(op) && attr_str(op, "kind") == Some("for")
}

pub fn for_preheader(op: &Op) -> Option<BlockId> {
    attr_i64(op, "preheader").map(|n| BlockId(n as u32))
}
pub fn for_latch(op: &Op) -> Option<BlockId> {
    attr_i64(op, "latch").map(|n| BlockId(n as u32))
}
pub fn for_update_val(op: &Op) -> Option<ValueId> {
    attr_i64(op, "update_val").map(|n| ValueId(n as u32))
}

/// The structured merge (join) block annotated on a `cond_br`, if any.
pub fn merge_of(op: &Op) -> Option<BlockId> {
    op.attrs.iter().find(|(k, _)| k == "merge").and_then(|(_, v)| match v {
        Attr::I64(n) => Some(BlockId(*n as u32)),
        _ => None,
    })
}

/// Read a boolean attr (default false).
pub fn attr_bool(op: &Op, key: &str) -> bool {
    op.attrs
        .iter()
        .find(|(k, _)| k == key)
        .map_or(false, |(_, v)| matches!(v, Attr::Bool(true)))
}

/// Did this op end a block as an explicit return? (the `implicit` attr is absent)
pub fn is_explicit_return(op: &Op) -> bool {
    op.name == RETURN && !op.attrs.iter().any(|(k, _)| k == "implicit")
}

/// True if the region holds any JSLIR terminator (i.e. it is a lowered CFG, not
/// an untransformed JSHIR body that `build_jslir` passed through).
pub fn region_is_cfg(region: &jsir_ir::Region) -> bool {
    region
        .blocks
        .iter()
        .any(|b| b.ops.iter().any(|op| is_terminator(&op.name)))
}
