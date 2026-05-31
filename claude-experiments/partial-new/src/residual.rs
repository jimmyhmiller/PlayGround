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
