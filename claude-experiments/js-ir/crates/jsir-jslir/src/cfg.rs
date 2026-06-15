//! A backend-neutral CFG interface — the shared spine identified in
//! `docs/HIR_COMPARISON.md`.
//!
//! The comparison's conclusion: the React Compiler HIR and JSLIR disagree on the
//! instruction model, value identity, and the effect/scope substrate, but they
//! *agree* on the control-flow skeleton — a CFG of basic blocks with a stable
//! `BlockId` and structured terminators (React's `Terminal` variants ≈ JSLIR's
//! `jslir.*` terminator + structured attrs). This module captures exactly that
//! shared layer so CFG-level pass logic — dominators, reverse-postorder,
//! reachability, structured-control-flow reconstruction — is written **once**
//! against [`Cfg`]/[`TerminalView`] and runs on either backend.
//!
//! Only the JSLIR backend ([`JslirCfg`]) is implemented here, since the React HIR
//! is not in this repo; a React-side `impl Cfg for HIRFunction` would map each
//! `Terminal` variant onto the same [`TerminalView`] (`IfTerminal` → `If`,
//! `WhileTerminal` → `While`, `GotoTerminal{kind:Break}` → `Goto{Break}`, …).
//! The generic algorithms below would then run unchanged on both.

use std::collections::{HashMap, HashSet};

use jsir_ir::{BlockId, Op, Region, ValueId};

use crate::dialect;

/// How an unconditional jump relates to source structure. React encodes this on
/// `GotoTerminal.variant`; JSLIR on a `jslir.br`'s `loopjump` attr.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GotoKind {
    /// A plain edge (fall-through / join).
    Plain,
    /// A `break` to a loop/label exit.
    Break,
    /// A `continue` to a loop's continue target.
    Continue,
}

/// A backend-neutral view of a block's terminator. The variants are the union of
/// what both IRs need to reconstruct source-level control flow; generic graph
/// algorithms ignore the structure and use [`Cfg::successors`], while
/// structured passes match on these.
#[derive(Debug, Clone, PartialEq)]
pub enum TerminalView {
    /// `return v?;` — no successors.
    Return { value: Option<ValueId> },
    /// Unconditional jump (`jslir.br` / React `GotoTerminal`).
    Goto { target: BlockId, kind: GotoKind },
    /// A structured `if` diamond. `merge` is the join block both arms reconverge
    /// at (React's `fallthrough`; JSLIR's `merge` attr).
    If { cond: ValueId, then_blk: BlockId, else_blk: BlockId, merge: Option<BlockId> },
    /// A `while` loop header. `body` branches back to this header; `exit` is the
    /// loop-merge block.
    While { cond: ValueId, body: BlockId, exit: BlockId },
    /// A `for` loop header, carrying the surrounding structure (`preheader` runs
    /// the init, `latch` runs the update before the back-edge).
    For {
        cond: ValueId,
        body: BlockId,
        exit: BlockId,
        preheader: Option<BlockId>,
        latch: Option<BlockId>,
    },
    /// A flattened ternary `test ? cons : alt`, both arms feeding `merge` (a phi).
    Ternary { cond: ValueId, cons: BlockId, alt: BlockId, merge: Option<BlockId> },
    /// A flattened logical `left && rhs` / `left || rhs`.
    Logical { left: ValueId, operator: String, rhs: BlockId, merge: Option<BlockId> },
    /// A two-way branch with no recognized source structure (defensive; a raw
    /// `cond_br` carrying none of the structured markers).
    Branch { cond: ValueId, then_blk: BlockId, else_blk: BlockId },
    /// No terminator op present — an open/un-terminated block (malformed CFG).
    Open,
}

/// The control-flow skeleton shared by React HIR and JSLIR. An implementor
/// exposes its blocks, entry, raw successor edges, and a structured view of each
/// terminator. Everything in [`analysis`] is written against just this.
pub trait Cfg {
    /// The entry block (`HIR.entry` / the region's first block).
    fn entry(&self) -> BlockId;
    /// All block ids, in no particular order.
    fn block_ids(&self) -> Vec<BlockId>;
    /// Raw successor edges of `b` (the terminator's target blocks). The basis for
    /// every generic graph algorithm; independent of the structured view.
    fn successors(&self, b: BlockId) -> Vec<BlockId>;
    /// The structured interpretation of `b`'s terminator.
    fn terminal(&self, b: BlockId) -> TerminalView;
}

// ---------------------------------------------------------------------------
// JSLIR backend
// ---------------------------------------------------------------------------

/// [`Cfg`] over a JSLIR function-body [`Region`] (a CFG produced by `build_jslir`).
pub struct JslirCfg<'a> {
    region: &'a Region,
    by_id: HashMap<BlockId, usize>,
}

impl<'a> JslirCfg<'a> {
    /// Wrap a lowered region. (Behavior is only meaningful for a region that is a
    /// CFG — see [`dialect::region_is_cfg`].)
    pub fn new(region: &'a Region) -> Self {
        let by_id = region.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
        JslirCfg { region, by_id }
    }

    fn terminator(&self, b: BlockId) -> Option<&'a Op> {
        let idx = *self.by_id.get(&b)?;
        self.region.blocks[idx].ops.last().filter(|op| dialect::is_terminator(&op.name))
    }
}

impl<'a> Cfg for JslirCfg<'a> {
    fn entry(&self) -> BlockId {
        self.region.blocks.first().map(|b| b.id).unwrap_or_default()
    }

    fn block_ids(&self) -> Vec<BlockId> {
        self.region.blocks.iter().map(|b| b.id).collect()
    }

    fn successors(&self, b: BlockId) -> Vec<BlockId> {
        match self.terminator(b) {
            Some(op) => op.successors.iter().map(|s| s.block).collect(),
            None => Vec::new(),
        }
    }

    fn terminal(&self, b: BlockId) -> TerminalView {
        let Some(op) = self.terminator(b) else { return TerminalView::Open };
        let succ = |i: usize| op.successors.get(i).map(|s| s.block);

        if op.name == dialect::RETURN {
            return TerminalView::Return { value: op.operands.first().copied() };
        }
        if op.name == dialect::BR {
            let kind = match dialect::loop_jump(op) {
                Some("break") => GotoKind::Break,
                Some("continue") => GotoKind::Continue,
                _ => GotoKind::Plain,
            };
            return TerminalView::Goto { target: succ(0).unwrap_or_default(), kind };
        }
        // From here, a cond_br: disambiguate the structured shapes by the same
        // markers `dialect.rs` writes (and `lift` reads).
        let cond = op.operands.first().copied().unwrap_or(ValueId(0));
        if dialect::is_for_header(op) {
            return TerminalView::For {
                cond,
                body: succ(0).unwrap_or_default(),
                exit: succ(1).unwrap_or_default(),
                preheader: dialect::for_preheader(op),
                latch: dialect::for_latch(op),
            };
        }
        if dialect::is_while_header(op) {
            return TerminalView::While {
                cond,
                body: succ(0).unwrap_or_default(),
                exit: succ(1).unwrap_or_default(),
            };
        }
        if dialect::is_ternary(op) {
            return TerminalView::Ternary {
                cond,
                cons: succ(0).unwrap_or_default(),
                alt: succ(1).unwrap_or_default(),
                merge: dialect::merge_of(op),
            };
        }
        if let Some(operator) = dialect::logexpr_operator(op) {
            // For `&&`: successors are [rhs, merge]; for `||`: [merge, rhs]. The
            // rhs is the block that evaluates the right operand.
            let rhs = if operator == "&&" { succ(0) } else { succ(1) };
            return TerminalView::Logical {
                left: cond,
                operator: operator.to_string(),
                rhs: rhs.unwrap_or_default(),
                merge: dialect::merge_of(op),
            };
        }
        if dialect::is_if_header(op) {
            return TerminalView::If {
                cond,
                then_blk: succ(0).unwrap_or_default(),
                else_blk: succ(1).unwrap_or_default(),
                merge: dialect::merge_of(op),
            };
        }
        TerminalView::Branch {
            cond,
            then_blk: succ(0).unwrap_or_default(),
            else_blk: succ(1).unwrap_or_default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Generic algorithms — written ONCE against `Cfg`, run on any backend.
// ---------------------------------------------------------------------------

/// CFG-level analyses that depend on *nothing* but [`Cfg`]. This is the concrete
/// payoff of the shared interface: every function here would run unchanged on a
/// React-HIR `impl Cfg` and on [`JslirCfg`].
pub mod analysis {
    use super::*;

    /// Predecessor map: `block → blocks that branch to it`. React stores this on
    /// `BasicBlock.preds`; here it is derived from successors, so the two backends
    /// present an identical view.
    pub fn predecessors<C: Cfg>(cfg: &C) -> HashMap<BlockId, Vec<BlockId>> {
        let mut preds: HashMap<BlockId, Vec<BlockId>> =
            cfg.block_ids().into_iter().map(|b| (b, Vec::new())).collect();
        for b in cfg.block_ids() {
            for s in cfg.successors(b) {
                preds.entry(s).or_default().push(b);
            }
        }
        preds
    }

    /// Reverse postorder from the entry: barring cycles, every predecessor
    /// precedes its successors — the order forward dataflow wants. (React's
    /// `HIR.blocks` map is *stored* in this order; JSLIR derives it.)
    pub fn reverse_postorder<C: Cfg>(cfg: &C) -> Vec<BlockId> {
        let mut postorder = Vec::new();
        let mut visited = HashSet::new();
        // Iterative DFS carrying a "children pushed?" flag, to emit in postorder.
        let mut stack = vec![(cfg.entry(), false)];
        while let Some((b, expanded)) = stack.pop() {
            if expanded {
                postorder.push(b);
                continue;
            }
            if !visited.insert(b) {
                continue;
            }
            stack.push((b, true));
            for s in cfg.successors(b) {
                if !visited.contains(&s) {
                    stack.push((s, false));
                }
            }
        }
        postorder.reverse();
        postorder
    }

    /// Blocks reachable from the entry. (The basis for reachability-driven DCE,
    /// which both compilers run.)
    pub fn reachable<C: Cfg>(cfg: &C) -> HashSet<BlockId> {
        let mut seen = HashSet::new();
        let mut work = vec![cfg.entry()];
        while let Some(b) = work.pop() {
            if !seen.insert(b) {
                continue;
            }
            work.extend(cfg.successors(b));
        }
        seen
    }

    /// Immediate dominators via Cooper–Harvey–Kennedy ("A Simple, Fast Dominance
    /// Algorithm"), keyed by block. The entry maps to itself. Unreachable blocks
    /// are absent. This is the workhorse both React's and JSLIR's structured
    /// passes need; written once here.
    pub fn immediate_dominators<C: Cfg>(cfg: &C) -> HashMap<BlockId, BlockId> {
        let rpo = reverse_postorder(cfg);
        let rpo_index: HashMap<BlockId, usize> =
            rpo.iter().enumerate().map(|(i, b)| (*b, i)).collect();
        let preds = predecessors(cfg);
        let entry = cfg.entry();

        let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
        idom.insert(entry, entry);

        let intersect = |mut a: BlockId,
                         mut b: BlockId,
                         idom: &HashMap<BlockId, BlockId>|
         -> BlockId {
            while a != b {
                while rpo_index[&a] > rpo_index[&b] {
                    a = idom[&a];
                }
                while rpo_index[&b] > rpo_index[&a] {
                    b = idom[&b];
                }
            }
            a
        };

        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo.iter() {
                if b == entry {
                    continue;
                }
                let Some(bpreds) = preds.get(&b) else { continue };
                // First processed predecessor.
                let mut new_idom = match bpreds.iter().find(|p| idom.contains_key(p)) {
                    Some(&p) => p,
                    None => continue, // unreachable so far
                };
                for &p in bpreds {
                    if p != new_idom && idom.contains_key(&p) {
                        new_idom = intersect(p, new_idom, &idom);
                    }
                }
                if idom.get(&b) != Some(&new_idom) {
                    idom.insert(b, new_idom);
                    changed = true;
                }
            }
        }
        idom
    }
}
