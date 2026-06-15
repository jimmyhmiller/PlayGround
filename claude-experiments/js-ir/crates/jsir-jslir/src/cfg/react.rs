//! A [`Cfg`] backend over the **real** React Compiler HIR — the vendored Rust
//! port `react_compiler_hir` (`vendor/react-compiler-rust`). Nothing here is a
//! mock: it wraps an actual `react_compiler_hir::HIR` and reads its real
//! `Terminal`s, `BasicBlock`s, and the upstream `each_terminal_successor` edge
//! semantics (replicated here, since that helper lives in the crate's `visitors`
//! module).
//!
//! This is the second backend that makes the "write a CFG pass once, run it on
//! both IRs" claim from `docs/HIR_COMPARISON.md` concrete and checkable
//! (`tests/cross_backend.rs`). The generic [`super::analysis`] algorithms run on
//! a value of this type with no changes.
//!
//! Mapping notes (and where the IRs honestly diverge):
//! - **Block id** is a `u32` newtype on both sides; React's `BlockId(u32)` maps to
//!   `jsir_ir::BlockId(u32)` by its inner value.
//! - **Value identity** is React's `IdentifierId` here (vs JSLIR's `ValueId`),
//!   exposed as [`Cfg::Value`].
//! - **`If`/`Branch`** carry a real `test: Place` → mapped to
//!   [`TerminalView::If`]/[`TerminalView::Branch`] with `cond = test.identifier`.
//!   `If`'s `fallthrough` is the merge block (≙ JSLIR's `merge` attr).
//! - **Loops, `switch`, `try`/`throw`, `for-of`/`for-in`, `optional`, `scope`,
//!   …** are encoded differently than JSLIR's current dialect (e.g. React `While`
//!   points at a separate `test` *block*, not an inline condition), so they
//!   surface as [`TerminalView::Other`]. The [`super::analysis`] passes are
//!   unaffected — they only use [`Cfg::successors`], which is exact for every
//!   terminal.

use jsir_ir::BlockId as JBlockId;
use react_compiler_hir::{BlockId as RBlockId, IdentifierId, Terminal, HIR};

use super::{Cfg, GotoKind, TerminalView};

/// A [`Cfg`] view over a borrowed React Compiler HIR control-flow graph.
pub struct ReactHir<'a> {
    hir: &'a HIR,
}

impl<'a> ReactHir<'a> {
    pub fn new(hir: &'a HIR) -> Self {
        ReactHir { hir }
    }
}

fn jb(b: RBlockId) -> JBlockId {
    JBlockId(b.0)
}
fn rb(b: JBlockId) -> RBlockId {
    RBlockId(b.0)
}

/// The authoritative successor edges of a React terminal — a faithful port of
/// `react_compiler_hir::visitors::each_terminal_successor` (kept in lockstep with
/// upstream: structured `fallthrough`/`test`/`loop_block` blocks are NOT edges;
/// only the listed targets are).
fn each_successor(terminal: &Terminal) -> Vec<RBlockId> {
    let mut out = Vec::new();
    match terminal {
        Terminal::Goto { block, .. } => out.push(*block),
        Terminal::If { consequent, alternate, .. }
        | Terminal::Branch { consequent, alternate, .. } => {
            out.push(*consequent);
            out.push(*alternate);
        }
        Terminal::Switch { cases, .. } => {
            for c in cases {
                out.push(c.block);
            }
        }
        Terminal::Optional { test, .. }
        | Terminal::Ternary { test, .. }
        | Terminal::Logical { test, .. }
        | Terminal::While { test, .. } => out.push(*test),
        Terminal::DoWhile { loop_block, .. } => out.push(*loop_block),
        Terminal::For { init, .. }
        | Terminal::ForOf { init, .. }
        | Terminal::ForIn { init, .. } => out.push(*init),
        Terminal::Label { block, .. }
        | Terminal::Sequence { block, .. }
        | Terminal::Try { block, .. }
        | Terminal::Scope { block, .. }
        | Terminal::PrunedScope { block, .. } => out.push(*block),
        Terminal::MaybeThrow { continuation, handler, .. } => {
            out.push(*continuation);
            if let Some(h) = handler {
                out.push(*h);
            }
        }
        Terminal::Return { .. }
        | Terminal::Throw { .. }
        | Terminal::Unreachable { .. }
        | Terminal::Unsupported { .. } => {}
    }
    out
}

/// The backend `name` for terminals that land in [`TerminalView::Other`].
fn terminal_name(t: &Terminal) -> &'static str {
    match t {
        Terminal::Switch { .. } => "switch",
        Terminal::DoWhile { .. } => "do-while",
        Terminal::While { .. } => "while",
        Terminal::For { .. } => "for",
        Terminal::ForOf { .. } => "for-of",
        Terminal::ForIn { .. } => "for-in",
        Terminal::Logical { .. } => "logical",
        Terminal::Ternary { .. } => "ternary",
        Terminal::Optional { .. } => "optional",
        Terminal::Label { .. } => "label",
        Terminal::Sequence { .. } => "sequence",
        Terminal::MaybeThrow { .. } => "maybe-throw",
        Terminal::Try { .. } => "try",
        Terminal::Scope { .. } => "scope",
        Terminal::PrunedScope { .. } => "pruned-scope",
        Terminal::Throw { .. } => "throw",
        Terminal::Unreachable { .. } => "unreachable",
        Terminal::Unsupported { .. } => "unsupported",
        // Mapped precisely elsewhere; listed for exhaustiveness.
        Terminal::Return { .. } => "return",
        Terminal::Goto { .. } => "goto",
        Terminal::If { .. } => "if",
        Terminal::Branch { .. } => "branch",
    }
}

impl<'a> Cfg for ReactHir<'a> {
    type Value = IdentifierId;

    fn entry(&self) -> JBlockId {
        jb(self.hir.entry)
    }

    fn block_ids(&self) -> Vec<JBlockId> {
        self.hir.blocks.keys().map(|b| jb(*b)).collect()
    }

    fn successors(&self, b: JBlockId) -> Vec<JBlockId> {
        match self.hir.blocks.get(&rb(b)) {
            Some(block) => each_successor(&block.terminal).into_iter().map(jb).collect(),
            None => Vec::new(),
        }
    }

    fn terminal(&self, b: JBlockId) -> TerminalView<IdentifierId> {
        let Some(block) = self.hir.blocks.get(&rb(b)) else {
            return TerminalView::Open;
        };
        match &block.terminal {
            Terminal::Return { value, .. } => {
                TerminalView::Return { value: Some(value.identifier) }
            }
            Terminal::Goto { block, variant, .. } => {
                use react_compiler_hir::GotoVariant;
                let kind = match variant {
                    GotoVariant::Break => GotoKind::Break,
                    GotoVariant::Continue => GotoKind::Continue,
                    GotoVariant::Try => GotoKind::Plain,
                };
                TerminalView::Goto { target: jb(*block), kind }
            }
            Terminal::If { test, consequent, alternate, fallthrough, .. } => TerminalView::If {
                cond: test.identifier,
                then_blk: jb(*consequent),
                else_blk: jb(*alternate),
                merge: Some(jb(*fallthrough)),
            },
            Terminal::Branch { test, consequent, alternate, .. } => TerminalView::Branch {
                cond: test.identifier,
                then_blk: jb(*consequent),
                else_blk: jb(*alternate),
            },
            other => TerminalView::Other {
                name: terminal_name(other),
                successors: each_successor(other).into_iter().map(jb).collect(),
            },
        }
    }
}
