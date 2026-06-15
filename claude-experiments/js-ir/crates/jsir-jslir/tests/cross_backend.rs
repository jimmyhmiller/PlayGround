//! The payoff of `docs/HIR_COMPARISON.md`: the **same** CFG pass code runs on the
//! real React Compiler HIR (the vendored Rust port) and on JSLIR, through one
//! `Cfg` trait. Requires `--features react-hir`.
//!
//! Both halves build the canonical `if (c) {} else {}` diamond
//! (`entry → {then, else} → merge`) and run the identical generic algorithms
//! (`reverse_postorder`, `reachable`, `immediate_dominators`, `predecessors`),
//! asserting the same structural facts. The algorithm calls are byte-identical
//! source across the two backends — only the `Cfg` value differs.
#![cfg(feature = "react-hir")]

use jsir_ir::{BlockId, Op, Region};
use jsir_jslir::cfg::{analysis, Cfg, TerminalView};

/// The generic, backend-agnostic check. This function is written once and called
/// with both a React-HIR `Cfg` and a JSLIR `Cfg`. It is the literal demonstration
/// that the pass code is shared.
fn assert_diamond<C: Cfg>(cfg: &C, entry: BlockId, then_b: BlockId, else_b: BlockId, merge: BlockId) {
    // Reverse postorder: entry first, all four blocks once.
    let rpo = analysis::reverse_postorder(cfg);
    assert_eq!(rpo.first(), Some(&entry), "entry leads RPO");
    assert_eq!(rpo.len(), 4, "four blocks in RPO");

    // Everything reachable from entry.
    assert_eq!(analysis::reachable(cfg).len(), 4);

    // Predecessors: the merge has both arms as predecessors.
    let preds = analysis::predecessors(cfg);
    let mut merge_preds = preds[&merge].clone();
    merge_preds.sort();
    let mut expect = vec![then_b, else_b];
    expect.sort();
    assert_eq!(merge_preds, expect, "merge joins both arms");

    // Dominators: entry dominates itself; the merge is dominated by entry and by
    // NEITHER arm (the classic diamond property a real pass relies on).
    let idom = analysis::immediate_dominators(cfg);
    assert_eq!(idom[&entry], entry);
    assert_eq!(idom[&merge], entry, "merge's idom is the entry, not an arm");
    assert_eq!(idom[&then_b], entry);
    assert_eq!(idom[&else_b], entry);
}

// ---------------------------------------------------------------------------
// Backend 1: the real React Compiler HIR (vendored Rust port).
// ---------------------------------------------------------------------------

mod react_backend {
    use super::*;
    use indexmap::IndexMap;
    use jsir_jslir::cfg::react::ReactHir;
    use react_compiler_hir::{
        BasicBlock, BlockId as RBlockId, BlockKind, Effect, EvaluationOrder, GotoVariant,
        IdentifierId, Place, Terminal, HIR,
    };

    fn place(id: u32) -> Place {
        Place { identifier: IdentifierId(id), effect: Effect::Read, reactive: false, loc: None }
    }
    fn eo(n: u32) -> EvaluationOrder {
        EvaluationOrder(n)
    }
    fn goto(target: u32) -> Terminal {
        Terminal::Goto {
            block: RBlockId(target),
            variant: GotoVariant::Break,
            id: eo(0),
            loc: None,
        }
    }
    fn block(id: u32, kind: BlockKind, terminal: Terminal) -> BasicBlock {
        BasicBlock {
            kind,
            id: RBlockId(id),
            instructions: Vec::new(),
            terminal,
            preds: Default::default(),
            phis: Vec::new(),
        }
    }

    pub fn run() {
        // entry: if (c) -> [^1 then, ^2 else], fallthrough ^3 merge
        // ^1: goto ^3 ; ^2: goto ^3 ; ^3: return
        let mut blocks: IndexMap<RBlockId, BasicBlock> = IndexMap::new();
        blocks.insert(
            RBlockId(0),
            block(
                0,
                BlockKind::Block,
                Terminal::If {
                    test: place(100),
                    consequent: RBlockId(1),
                    alternate: RBlockId(2),
                    fallthrough: RBlockId(3),
                    id: eo(1),
                    loc: None,
                },
            ),
        );
        blocks.insert(RBlockId(1), block(1, BlockKind::Block, goto(3)));
        blocks.insert(RBlockId(2), block(2, BlockKind::Block, goto(3)));
        blocks.insert(
            RBlockId(3),
            block(
                3,
                BlockKind::Block,
                Terminal::Return {
                    value: place(101),
                    return_variant: react_compiler_hir::ReturnVariant::Implicit,
                    id: eo(2),
                    loc: None,
                    effects: None,
                },
            ),
        );
        let hir = HIR { entry: RBlockId(0), blocks };

        let cfg = ReactHir::new(&hir);

        // Sanity: the `if` is recognized structurally with the right merge.
        assert!(matches!(
            cfg.terminal(BlockId(0)),
            TerminalView::If { merge: Some(BlockId(3)), .. }
        ));

        assert_diamond(&cfg, BlockId(0), BlockId(1), BlockId(2), BlockId(3));
    }
}

// ---------------------------------------------------------------------------
// Backend 2: JSLIR, lowered from real source — the SAME assert_diamond runs.
// ---------------------------------------------------------------------------

fn jslir_body(src: &str) -> Region {
    let file = jsir_swc::source_to_ir(src).unwrap();
    let (jslir, _) = jsir_jslir::build_jslir(&file);
    find_body(&jslir).expect("a lowered function body")
}
fn find_body(op: &Op) -> Option<Region> {
    if op.name == "jsir.function_declaration" {
        if let Some(b) = op.regions.get(1) {
            if jsir_jslir::dialect::region_is_cfg(b) {
                return Some(b.clone());
            }
        }
    }
    for r in &op.regions {
        for b in &r.blocks {
            for o in &b.ops {
                if let Some(f) = find_body(o) {
                    return Some(f);
                }
            }
        }
    }
    None
}

#[test]
fn same_passes_run_on_both_backends() {
    // React HIR backend: hand-built diamond.
    react_backend::run();

    // JSLIR backend: the same diamond, lowered from source. Find the `if` header
    // and read its arms/merge through the shared interface, then run the same
    // generic checks.
    let region = jslir_body("function f(c) { let x = 0; if (c) { x = 1; } else { x = 2; } return x; }");
    let cfg = jsir_jslir::cfg::JslirCfg::new(&region);
    let entry = cfg.entry();
    let (then_b, else_b, merge) = cfg
        .block_ids()
        .into_iter()
        .find_map(|b| match cfg.terminal(b) {
            TerminalView::If { then_blk, else_blk, merge: Some(m), .. } => {
                Some((then_blk, else_blk, m))
            }
            _ => None,
        })
        .expect("the if header");
    // In this lowering the if is the entry block; assert and reuse the same check.
    assert_eq!(entry, BlockId(0));
    assert_diamond(&cfg, entry, then_b, else_b, merge);
}
