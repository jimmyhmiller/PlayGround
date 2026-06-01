//! The generic online partial-evaluation / driving engine.
//!
//! This module knows nothing about any particular source language. It walks a
//! client-provided operational semantics over a partially-static state, reduces
//! static control flow away, residualizes dynamic operations into basic blocks,
//! and memoizes specialization contexts so that runtime loops in the source
//! become loops in the residual.
//!
//! It is, structurally, abstract interpretation whose abstract domain is
//! "partially-static state" and whose output is a residual program rather than
//! a table of facts. Two clients drive it: `bf` (finite key space, no
//! generalization) and `imp` (a partially-static store that needs the whistle).

use std::collections::HashMap;
use std::hash::Hash;

use crate::residual::{Block, BlockId, Program, Terminator};

/// Backstop against unbounded *static* unrolling (e.g. `while 1 { i := i+1 }`).
/// If a single program point is specialized more than this many times, the
/// engine forces generalization. Generous enough that real static loops are
/// unaffected; small enough to guarantee termination.
const UNROLL_BOUND: usize = 5_000;

/// A client supplies the *meaning* of its language; the engine supplies the
/// fixpoint, memoization, termination, and CFG-reconstruction plumbing.
pub trait Client {
    /// Partially-static machine state (what is known so far at a program point).
    type State: Clone;
    /// Full memoization key. Two states with the same key share a residual block.
    type Key: Eq + Hash + Clone;
    /// Coarse program point, used to group states for the generalization whistle.
    /// Many distinct `Key`s (different static stores) map to one `Point`.
    type Point: Eq + Hash + Clone;
    /// A straight-line residual instruction (opaque to the engine).
    type Op;
    /// A residual branch condition (opaque to the engine).
    type Cond;

    fn key(&self, s: &Self::State) -> Self::Key;
    fn point(&self, s: &Self::State) -> Self::Point;

    /// Advance the state by one source construct.
    ///
    /// * `at_entry` is true only for the very first step of a fresh block, so a
    ///   client can force a block boundary (end the block, restart fresh) when
    ///   straight-line execution falls into a join point / loop header.
    /// * Push onto `out` to residualize (the binding-time "dynamic" decision).
    /// * Returning `Continue` without pushing is the "static" decision.
    fn step(&self, s: &mut Self::State, out: &mut Vec<Self::Op>, at_entry: bool) -> Step<Self>;

    /// Termination whistle: should re-entering a point with `cand`, having
    /// already specialized it with `seen`, trigger generalization? Default:
    /// never (sound only when the key space is finite, as for BF).
    fn whistle(&self, _seen: &Self::State, _cand: &Self::State) -> bool {
        false
    }

    /// Produce a state more general than `from` (typically by turning some
    /// static values dynamic), pushing into `out` whatever residual ops bridge
    /// the runtime state from `from` to the generalized state (materializing the
    /// now-dynamic values). Default: identity (no generalization).
    fn generalize(
        &self,
        _seen: &Self::State,
        from: &Self::State,
        _out: &mut Vec<Self::Op>,
    ) -> Self::State {
        from.clone()
    }
}

pub enum Step<C: Client + ?Sized> {
    /// Static reduction (or coalescing into pending state): stay in this block.
    Continue,
    /// Unconditional transfer. Subject to the whistle: this is where loop
    /// back-edges are tied off (and, if needed, generalized).
    Jump(C::State),
    /// Two-way dynamic branch. Each arm is specialized independently.
    Branch {
        cond: C::Cond,
        t: C::State,
        f: C::State,
    },
    /// End of computation along this path.
    Halt,
}

/// All the mutable bookkeeping for one specialization run.
struct Ctx<C: Client> {
    blocks: Vec<Block<C::Op, C::Cond>>,
    memo: HashMap<C::Key, BlockId>,
    /// States already specialized, grouped by point, for the whistle.
    seen: HashMap<C::Point, Vec<C::State>>,
    work: Vec<(BlockId, C::State)>,
}

pub fn specialize<C: Client>(client: &C, init: C::State) -> Program<C::Op, C::Cond> {
    let mut cx = Ctx {
        blocks: Vec::new(),
        memo: HashMap::new(),
        seen: HashMap::new(),
        work: Vec::new(),
    };

    let entry = create_or_get(client, init, &mut cx);

    while let Some((bid, state)) = cx.work.pop() {
        let mut s = state;
        let mut ops: Vec<C::Op> = Vec::new();
        let mut at_entry = true;

        let term = loop {
            match client.step(&mut s, &mut ops, at_entry) {
                Step::Continue => at_entry = false,
                Step::Halt => break Terminator::Halt,
                Step::Jump(next) => {
                    // Back-edges run through the whistle and may append bridge
                    // ops (materializations) to this block before transferring.
                    let succ = resolve_jump(client, next, &mut ops, &mut cx);
                    break Terminator::Br(succ);
                }
                Step::Branch { cond, t, f } => {
                    let bt = create_or_get(client, t, &mut cx);
                    let bf = create_or_get(client, f, &mut cx);
                    break Terminator::Cond { cond, t: bt, f: bf };
                }
            }
        };

        cx.blocks[bid.0].ops = ops;
        cx.blocks[bid.0].term = term;
    }

    Program {
        blocks: cx.blocks,
        entry,
    }
}

/// Look up a specialization context, or allocate a fresh block for it. This
/// single memo lookup is what turns source loops into residual loops and makes
/// specialization terminate.
fn create_or_get<C: Client>(client: &C, st: C::State, cx: &mut Ctx<C>) -> BlockId {
    let k = client.key(&st);
    if let Some(&b) = cx.memo.get(&k) {
        return b;
    }
    let id = BlockId(cx.blocks.len());
    cx.blocks.push(Block {
        ops: Vec::new(),
        term: Terminator::Unset,
    });
    cx.memo.insert(k, id);
    cx.seen.entry(client.point(&st)).or_default().push(st.clone());
    cx.work.push((id, st));
    id
}

/// Resolve a back-edge target, applying the generalization whistle. If re-entry
/// risks non-termination (the whistle fires, or the unroll backstop trips), the
/// target is generalized and bridge ops are appended to `ops` before the jump.
fn resolve_jump<C: Client>(
    client: &C,
    mut target: C::State,
    ops: &mut Vec<C::Op>,
    cx: &mut Ctx<C>,
) -> BlockId {
    loop {
        let k = client.key(&target);
        if let Some(&b) = cx.memo.get(&k) {
            return b;
        }

        let p = client.point(&target);
        let force = cx.seen.get(&p).map_or(0, |v| v.len()) >= UNROLL_BOUND;
        let candidate = cx.seen.get(&p).and_then(|v| {
            v.iter()
                .rev()
                .find(|s| force || client.whistle(s, &target))
                .cloned()
        });

        match candidate {
            Some(seen_state) => {
                let general = client.generalize(&seen_state, &target, ops);
                // Generalization must make progress (a strictly more general
                // key), otherwise we would loop forever resolving the same key.
                if client.key(&general) == k {
                    return create_or_get(client, target, cx);
                }
                target = general;
            }
            None => return create_or_get(client, target, cx),
        }
    }
}
