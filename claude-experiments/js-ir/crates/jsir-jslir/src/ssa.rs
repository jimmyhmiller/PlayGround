//! `enter_ssa` — SSA construction over a JSLIR function-body CFG, as a *side-table
//! analysis*: it computes def-use chains (which definition reaches each variable
//! read) and the phi nodes at merge blocks, WITHOUT mutating the IR. Passes
//! consult [`SsaInfo`]; the round-trip is unaffected.
//!
//! Scope of this version (everything else returns `None`, so the result is always
//! correct for what it covers — never silently wrong):
//! - Acyclic CFGs, and **single** (non-nested) natural loops. A loop header gets
//!   a phi for each variable that is live before the loop and modified in its
//!   body (`φ(preheader, latch)`); the back-edge operand is filled at the latch.
//!   Nested/multiple loops are skipped.
//! - Variables keyed by name; functions that **shadow** a name across scopes are
//!   skipped.
//! - Writes handled: `jslir.store_local` (declarations) and `x = e` assignments.
//!   Compound/update/destructuring writes cause a skip.
//!
//! Reads are `jsir.identifier` ops (their result `ValueId` is the use site);
//! their reaching def value is the written value or a phi.

use std::collections::{HashMap, HashSet};

use jsir_ir::{Block, BlockId, Op, Region, ValueId};

use crate::dialect;

/// SSA result: def-use chains + phis. Pure analysis (no IR mutation).
#[derive(Debug, Default)]
pub struct SsaInfo {
    /// Read-op result `ValueId` → the def value reaching it (a written value or a phi value).
    pub reaching: HashMap<ValueId, ValueId>,
    /// Phis inserted at merge blocks.
    pub phis: Vec<Phi>,
}

#[derive(Debug)]
pub struct Phi {
    pub block: BlockId,
    pub var: String,
    /// The synthetic SSA value this phi defines.
    pub value: ValueId,
    /// One operand per predecessor: (pred block, the def value from that pred).
    pub operands: Vec<(BlockId, ValueId)>,
}

/// The variable a read/write op refers to (by resolved-symbol name).
fn var_of_read(op: &Op) -> Option<String> {
    if op.name != "jsir.identifier" {
        return None;
    }
    op.trivia
        .as_ref()
        .and_then(|t| t.referenced_symbol.as_ref())
        .map(|s| s.name.clone())
}

fn referenced_name(op: &Op) -> Option<(String, Option<i64>)> {
    op.trivia
        .as_ref()
        .and_then(|t| t.referenced_symbol.as_ref())
        .map(|s| (s.name.clone(), s.def_scope_uid))
}

/// Run SSA construction on a function-body CFG. Returns `None` (skip) for any
/// construct outside this version's scope.
pub fn enter_ssa(region: &Region, next_value: &mut u32) -> Option<SsaInfo> {
    let entry = region.blocks.first()?.id;
    let block_map: HashMap<BlockId, &Block> = region.blocks.iter().map(|b| (b.id, b)).collect();

    // Predecessors.
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for b in &region.blocks {
        preds.entry(b.id).or_default();
        if let Some(t) = b.ops.last() {
            for s in &t.successors {
                preds.entry(s.block).or_default().push(b.id);
            }
        }
    }

    // Shadowing check + value→def-op index (to resolve assignment lvalues).
    let mut name_scopes: HashMap<String, HashSet<Option<i64>>> = HashMap::new();
    let mut val2op: HashMap<ValueId, &Op> = HashMap::new();
    for b in &region.blocks {
        for op in &b.ops {
            for r in &op.results {
                val2op.insert(*r, op);
            }
            if let Some((name, scope)) = referenced_name(op) {
                name_scopes.entry(name).or_default().insert(scope);
            }
        }
    }
    if name_scopes.values().any(|s| s.len() > 1) {
        return None; // shadowing — name-based keys would be wrong
    }

    // RPO + back-edge detection.
    let (order, back_edges) = analyze_cfg(&block_map, entry);
    if back_edges.len() > 1 {
        return None; // nested/multiple loops out of scope
    }
    // Single natural loop: (latch -> header) and the variables its body modifies.
    let loop_ctx: Option<(BlockId, BlockId, HashSet<String>)> =
        back_edges.first().map(|&(latch, header)| {
            let body = loop_blocks(&block_map, &preds, header, latch);
            let modified = modified_vars(&block_map, &body, &val2op);
            (latch, header, modified)
        });

    let mut info = SsaInfo::default();
    // Per-block: the def value of each variable at block ENTRY and EXIT.
    let mut exit_defs: HashMap<BlockId, HashMap<String, ValueId>> = HashMap::new();
    // Loop-header phis: var -> index into info.phis (latch operand filled later).
    let mut header_phis: HashMap<String, usize> = HashMap::new();

    for bid in &order {
        let block = block_map.get(bid)?;
        // Forward predecessors (exclude the loop back-edge into the header).
        let bpreds: Vec<BlockId> = preds
            .get(bid)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|p| {
                loop_ctx
                    .as_ref()
                    .map_or(true, |(latch, header, _)| !(bid == header && p == latch))
            })
            .collect();

        // Entry defs = merge of predecessors' exit defs (phi where they disagree).
        let mut cur: HashMap<String, ValueId> = HashMap::new();
        if bpreds.len() == 1 {
            cur = exit_defs.get(&bpreds[0]).cloned().unwrap_or_default();
        } else if bpreds.len() > 1 {
            // Variables defined in EVERY predecessor.
            let mut common: Option<HashSet<String>> = None;
            for p in &bpreds {
                let keys: HashSet<String> =
                    exit_defs.get(p).map(|m| m.keys().cloned().collect()).unwrap_or_default();
                common = Some(match common {
                    None => keys,
                    Some(c) => c.intersection(&keys).cloned().collect(),
                });
            }
            for var in common.unwrap_or_default() {
                let operands: Vec<(BlockId, ValueId)> = bpreds
                    .iter()
                    .map(|p| (*p, exit_defs[p][&var]))
                    .collect();
                let first = operands[0].1;
                if operands.iter().all(|(_, v)| *v == first) {
                    cur.insert(var, first); // all agree: no phi
                } else {
                    let phi_val = ValueId(*next_value);
                    *next_value += 1;
                    cur.insert(var.clone(), phi_val);
                    info.phis.push(Phi { block: *bid, var, value: phi_val, operands });
                }
            }
        }

        // Loop header: pre-create a phi for each variable live before the loop and
        // modified in its body. Body uses the phi; the latch operand is filled below.
        if let Some((_, header, modified)) = &loop_ctx {
            if bid == header {
                let live: Vec<String> =
                    cur.keys().filter(|v| modified.contains(*v)).cloned().collect();
                for var in live {
                    let pre_val = cur[&var];
                    let pre_pred = bpreds.first().copied().unwrap_or(*bid);
                    let phi_val = ValueId(*next_value);
                    *next_value += 1;
                    cur.insert(var.clone(), phi_val);
                    header_phis.insert(var.clone(), info.phis.len());
                    info.phis.push(Phi {
                        block: *bid,
                        var,
                        value: phi_val,
                        operands: vec![(pre_pred, pre_val)],
                    });
                }
            }
        }

        // Process instructions: reads resolve, writes update.
        for op in &block.ops {
            // Read of a tracked variable.
            if let Some(name) = var_of_read(op) {
                if let (Some(&def), Some(&res)) = (cur.get(&name), op.results.first()) {
                    info.reaching.insert(res, def);
                }
                continue;
            }
            // Declaration write.
            if dialect::is_store_local(op) {
                if let Some((name, _)) = dialect::store_local_parts(op).map(|(n, k)| (n.to_string(), k)) {
                    if let Some(&v) = op.operands.first() {
                        // Resolve the init value: if it is a variable read, its SSA
                        // value is the reaching def, not the read op's result.
                        let v = info.reaching.get(&v).copied().unwrap_or(v);
                        cur.insert(name, v);
                    } else {
                        // `let x;` with no initializer: leave undefined (no def).
                        cur.remove(&name);
                    }
                }
                continue;
            }
            // Assignment `x = e`.
            if op.name == "jsir.assignment_expression" {
                let operator = op
                    .attrs
                    .iter()
                    .find(|(k, _)| k == "operator_")
                    .and_then(|(_, v)| match v {
                        jsir_ir::Attr::Str(s) => Some(s.as_str()),
                        _ => None,
                    });
                if operator != Some("=") {
                    return None; // compound assignment: out of scope
                }
                let lref = *op.operands.first()?;
                let rhs = *op.operands.get(1)?;
                // Resolve the rhs (a variable read resolves to its reaching def).
                let rhs = info.reaching.get(&rhs).copied().unwrap_or(rhs);
                let lop = val2op.get(&lref)?;
                if lop.name == "jsir.identifier_ref" {
                    if let Some((name, _)) = referenced_name(lop) {
                        cur.insert(name, rhs);
                    }
                }
                // (member-expression assignment lvalues aren't variable writes.)
                continue;
            }
            // Update `x++` / `--x` on a variable: out of scope (reads+writes).
            if op.name == "jsir.update_expression" {
                let target = *op.operands.first()?;
                if val2op.get(&target).map_or(false, |o| o.name == "jsir.identifier_ref") {
                    return None;
                }
            }
        }
        // Latch: fill the back-edge operand of each loop-header phi.
        if let Some((latch, _, _)) = &loop_ctx {
            if bid == latch {
                for (var, &pidx) in &header_phis {
                    if let Some(&v) = cur.get(var) {
                        info.phis[pidx].operands.push((*latch, v));
                    }
                }
            }
        }

        exit_defs.insert(*bid, cur);
    }

    Some(info)
}

/// Iterative DFS: returns (reverse post-order, back-edges). A back-edge `(s, t)`
/// is an edge to a block currently on the DFS stack (`t` is a loop header).
fn analyze_cfg(
    block_map: &HashMap<BlockId, &Block>,
    entry: BlockId,
) -> (Vec<BlockId>, Vec<(BlockId, BlockId)>) {
    let mut state: HashMap<BlockId, u8> = HashMap::new(); // 0=unseen,1=on-stack,2=done
    let mut post: Vec<BlockId> = Vec::new();
    let mut back: Vec<(BlockId, BlockId)> = Vec::new();
    let mut stack: Vec<(BlockId, usize)> = vec![(entry, 0)];
    state.insert(entry, 1);
    while let Some((bid, idx)) = stack.last().copied() {
        let succs: Vec<BlockId> = block_map
            .get(&bid)
            .and_then(|b| b.ops.last())
            .map(|t| t.successors.iter().map(|s| s.block).collect())
            .unwrap_or_default();
        if idx < succs.len() {
            stack.last_mut().unwrap().1 += 1;
            let next = succs[idx];
            match state.get(&next).copied().unwrap_or(0) {
                0 => {
                    state.insert(next, 1);
                    stack.push((next, 0));
                }
                1 => back.push((bid, next)), // back-edge (latch -> header)
                _ => {}
            }
        } else {
            state.insert(bid, 2);
            post.push(bid);
            stack.pop();
        }
    }
    post.reverse();
    (post, back)
}

/// The blocks of a single natural loop: those reachable from `header` that can
/// also reach `latch` (and thus loop back via the back-edge), plus the header.
fn loop_blocks(
    block_map: &HashMap<BlockId, &Block>,
    preds: &HashMap<BlockId, Vec<BlockId>>,
    header: BlockId,
    latch: BlockId,
) -> HashSet<BlockId> {
    // Forward-reachable from header.
    let mut fwd = HashSet::new();
    let mut work = vec![header];
    while let Some(b) = work.pop() {
        if !fwd.insert(b) {
            continue;
        }
        if let Some(t) = block_map.get(&b).and_then(|bl| bl.ops.last()) {
            for s in &t.successors {
                work.push(s.block);
            }
        }
    }
    // Backward-reachable to latch (can reach latch).
    let mut bwd = HashSet::new();
    let mut work = vec![latch];
    while let Some(b) = work.pop() {
        if !bwd.insert(b) {
            continue;
        }
        if let Some(ps) = preds.get(&b) {
            for p in ps {
                work.push(*p);
            }
        }
    }
    let mut loopset: HashSet<BlockId> = fwd.intersection(&bwd).copied().collect();
    loopset.insert(header);
    loopset
}

/// Variables written (declared or `=`-assigned) anywhere in `body`.
fn modified_vars(
    block_map: &HashMap<BlockId, &Block>,
    body: &HashSet<BlockId>,
    val2op: &HashMap<ValueId, &Op>,
) -> HashSet<String> {
    let mut m = HashSet::new();
    for bid in body {
        let Some(block) = block_map.get(bid) else { continue };
        for op in &block.ops {
            if dialect::is_store_local(op) {
                if let Some((n, _)) = dialect::store_local_parts(op) {
                    m.insert(n.to_string());
                }
            } else if op.name == "jsir.assignment_expression" {
                if let Some(lop) = op.operands.first().and_then(|v| val2op.get(v)) {
                    if lop.name == "jsir.identifier_ref" {
                        if let Some((n, _)) = referenced_name(lop) {
                            m.insert(n);
                        }
                    }
                }
            }
        }
    }
    m
}
