//! The residual program produced by partial evaluation.
//!
//! This is generic: the engine owns the *control-flow skeleton* (basic blocks,
//! block ids, edges, the loop-tying), while the client owns the payloads `Op`
//! (a straight-line residual instruction) and `Cond` (a branch condition). The
//! engine never inspects either.

use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockId(pub usize);

#[derive(Debug)]
pub enum Terminator<Cond> {
    /// Placeholder while a block is under construction. Reaching this at run
    /// time is a hard bug, not a silent fallthrough.
    Unset,
    Halt,
    Br(BlockId),
    Cond { cond: Cond, t: BlockId, f: BlockId },
}

#[derive(Debug)]
pub struct Block<Op, Cond> {
    pub ops: Vec<Op>,
    pub term: Terminator<Cond>,
}

#[derive(Debug)]
pub struct Program<Op, Cond> {
    pub blocks: Vec<Block<Op, Cond>>,
    pub entry: BlockId,
}

impl<Op, Cond> Program<Op, Cond> {
    pub fn op_count(&self) -> usize {
        self.blocks.iter().map(|b| b.ops.len()).sum()
    }
}

/// A generic post-pass: thread jumps through empty forwarding blocks and drop
/// unreachable blocks, then renumber. Online specialization of static loops
/// leaves chains of empty `br` blocks (one per unrolled iteration); this
/// collapses them. It only inspects `ops.is_empty()` and terminator targets, so
/// it is fully generic over the client payloads.
pub fn simplify<Op, Cond>(prog: &mut Program<Op, Cond>) {
    let n = prog.blocks.len();

    // A block is a pure forwarder if it has no ops and an unconditional Br.
    let immediate: Vec<Option<BlockId>> = prog
        .blocks
        .iter()
        .map(|b| match (&b.term, b.ops.is_empty()) {
            (Terminator::Br(t), true) => Some(*t),
            _ => None,
        })
        .collect();

    // Resolve each block to its ultimate non-forwarding target, guarding the
    // empty-self-loop case (`while 1 {}`) so we don't spin.
    let resolve = |start: BlockId| -> BlockId {
        let mut cur = start;
        let mut seen = vec![false; n];
        while let Some(next) = immediate[cur.0] {
            if next == cur || seen[cur.0] {
                break;
            }
            seen[cur.0] = true;
            cur = next;
        }
        cur
    };

    for b in &mut prog.blocks {
        match &mut b.term {
            Terminator::Br(t) => *t = resolve(*t),
            Terminator::Cond { t, f, .. } => {
                *t = resolve(*t);
                *f = resolve(*f);
            }
            _ => {}
        }
    }
    prog.entry = resolve(prog.entry);

    // Mark reachable blocks from the (resolved) entry.
    let mut reachable = vec![false; n];
    let mut stack = vec![prog.entry];
    while let Some(b) = stack.pop() {
        if reachable[b.0] {
            continue;
        }
        reachable[b.0] = true;
        match &prog.blocks[b.0].term {
            Terminator::Br(t) => stack.push(*t),
            Terminator::Cond { t, f, .. } => {
                stack.push(*t);
                stack.push(*f);
            }
            _ => {}
        }
    }

    // Renumber, keeping only reachable blocks.
    let mut remap = vec![BlockId(0); n];
    let mut next = 0usize;
    for (i, r) in reachable.iter().enumerate() {
        if *r {
            remap[i] = BlockId(next);
            next += 1;
        }
    }
    let mut old = std::mem::take(&mut prog.blocks);
    let mut kept = Vec::with_capacity(next);
    for (i, b) in old.drain(..).enumerate() {
        if reachable[i] {
            kept.push(b);
        }
    }
    for b in &mut kept {
        match &mut b.term {
            Terminator::Br(t) => *t = remap[t.0],
            Terminator::Cond { t, f, .. } => {
                *t = remap[t.0];
                *f = remap[f.0];
            }
            _ => {}
        }
    }
    prog.entry = remap[prog.entry.0];
    prog.blocks = kept;
}

impl<Op: fmt::Display, Cond: fmt::Display> fmt::Display for Program<Op, Cond> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, b) in self.blocks.iter().enumerate() {
            let lead = if BlockId(i) == self.entry { "(entry)" } else { "" };
            writeln!(f, "b{i}: {lead}")?;
            for op in &b.ops {
                writeln!(f, "    {op}")?;
            }
            match &b.term {
                Terminator::Unset => writeln!(f, "    <unset>")?,
                Terminator::Halt => writeln!(f, "    halt")?,
                Terminator::Br(t) => writeln!(f, "    br b{}", t.0)?,
                Terminator::Cond { cond, t, f: fb } => {
                    writeln!(f, "    if {cond} -> b{} else b{}", t.0, fb.0)?
                }
            }
        }
        Ok(())
    }
}
