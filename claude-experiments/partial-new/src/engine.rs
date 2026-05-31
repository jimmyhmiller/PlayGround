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
//! a table of facts.

use std::collections::HashMap;
use std::hash::Hash;

use crate::residual::{Block, BlockId, Program, Terminator};

/// A client supplies the *meaning* of its language; the engine supplies the
/// fixpoint, memoization, termination, and CFG-reconstruction plumbing.
pub trait Client {
    /// Partially-static machine state (what is known so far at a program point).
    type State: Clone;
    /// The memoization key derived from a state. Two states with the same key
    /// are considered the same specialization context and share a residual
    /// block. The key space must be finite for specialization to terminate
    /// without generalization (see the note on `get_or_create`).
    type Key: Eq + Hash + Clone;
    /// A straight-line residual instruction (opaque to the engine).
    type Op;
    /// A residual branch condition (opaque to the engine).
    type Cond;

    fn key(&self, s: &Self::State) -> Self::Key;

    /// Advance the state by one source construct.
    ///
    /// * `at_entry` is true only for the very first step of a fresh block, so a
    ///   client can force a block boundary (end the current block and restart)
    ///   when straight-line execution would otherwise fall into a join point.
    /// * Push onto `out` to residualize (the binding-time "dynamic" decision).
    /// * Returning `Continue` without pushing is the "static" decision.
    fn step(&self, s: &mut Self::State, out: &mut Vec<Self::Op>, at_entry: bool) -> Step<Self>;
}

pub enum Step<C: Client + ?Sized> {
    /// Static reduction (or coalescing into pending state): stay in this block.
    Continue,
    /// Unconditional transfer. If the target context is already memoized this
    /// becomes a back-edge, which is how a runtime loop is tied off.
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

pub fn specialize<C: Client>(client: &C, init: C::State) -> Program<C::Op, C::Cond> {
    let mut blocks: Vec<Block<C::Op, C::Cond>> = Vec::new();
    let mut memo: HashMap<C::Key, BlockId> = HashMap::new();
    let mut work: Vec<(BlockId, C::State)> = Vec::new();

    let entry = get_or_create(client, init, &mut blocks, &mut memo, &mut work);

    while let Some((bid, state)) = work.pop() {
        let mut s = state;
        let mut ops: Vec<C::Op> = Vec::new();
        let mut at_entry = true;

        let term = loop {
            match client.step(&mut s, &mut ops, at_entry) {
                Step::Continue => {
                    at_entry = false;
                }
                Step::Halt => break Terminator::Halt,
                Step::Jump(next) => {
                    let succ = get_or_create(client, next, &mut blocks, &mut memo, &mut work);
                    break Terminator::Br(succ);
                }
                Step::Branch { cond, t, f } => {
                    let bt = get_or_create(client, t, &mut blocks, &mut memo, &mut work);
                    let bf = get_or_create(client, f, &mut blocks, &mut memo, &mut work);
                    break Terminator::Cond { cond, t: bt, f: bf };
                }
            }
        };

        blocks[bid.0].ops = ops;
        blocks[bid.0].term = term;
    }

    Program { blocks, entry }
}

/// Look up the specialization context, or allocate a fresh block for it and
/// queue it for processing.
///
/// This single memo lookup is what makes specialization terminate and what
/// turns source-level loops into residual loops. If the client's key space were
/// *infinite* (e.g. an ever-growing static counter), a generalization step
/// would belong right here: detect that some already-seen key homeomorphically
/// embeds the new state, widen the offending part to dynamic, and retry. BF's
/// key space is finite (one key per program counter), so no widening is needed.
fn get_or_create<C: Client>(
    client: &C,
    st: C::State,
    blocks: &mut Vec<Block<C::Op, C::Cond>>,
    memo: &mut HashMap<C::Key, BlockId>,
    work: &mut Vec<(BlockId, C::State)>,
) -> BlockId {
    let k = client.key(&st);
    if let Some(&b) = memo.get(&k) {
        return b;
    }
    let id = BlockId(blocks.len());
    blocks.push(Block {
        ops: Vec::new(),
        term: Terminator::Unset,
    });
    memo.insert(k, id);
    work.push((id, st));
    id
}
