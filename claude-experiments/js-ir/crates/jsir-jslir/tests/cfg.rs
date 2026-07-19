//! The shared `Cfg`/`TerminalView` interface (`docs/HIR_COMPARISON.md`), exercised
//! on real lowered JSLIR function bodies. Every assertion goes through the
//! backend-neutral interface — none of it touches `jslir.*` op names — which is
//! the point: this same code would run against a React-HIR `impl Cfg`.

use jsir_ir::{Op, Region};
use jsir_jslir::cfg::{analysis, Cfg, GotoKind, JslirCfg, TerminalView};

fn body(src: &str) -> Region {
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
fn straight_line_has_single_return() {
    let region = body("function f(a) { let x = a; return x; }");
    let cfg = JslirCfg::new(&region);
    // One block, ending in a Return, no successors.
    assert_eq!(cfg.block_ids().len(), 1);
    let entry = cfg.entry();
    assert!(cfg.successors(entry).is_empty());
    assert!(matches!(cfg.terminal(entry), TerminalView::Return { .. }));
}

#[test]
fn if_terminal_is_structured_with_merge() {
    let region = body("function f(c) { let x = 0; if (c) { x = 1; } else { x = 2; } return x; }");
    let cfg = JslirCfg::new(&region);

    // Exactly one block reads as an `if`, and its merge block exists in the CFG.
    let ifs: Vec<_> = cfg
        .block_ids()
        .into_iter()
        .filter_map(|b| match cfg.terminal(b) {
            TerminalView::If { then_blk, else_blk, merge, .. } => Some((then_blk, else_blk, merge)),
            _ => None,
        })
        .collect();
    assert_eq!(ifs.len(), 1, "one structured if");
    let (then_blk, else_blk, merge) = ifs[0];
    assert_ne!(then_blk, else_blk);
    let merge = merge.expect("if carries a merge block");
    assert!(cfg.block_ids().contains(&merge), "merge block is in the CFG");

    // Both arms reconverge at the merge (a real diamond).
    assert_eq!(cfg.successors(then_blk), vec![merge]);
    assert_eq!(cfg.successors(else_blk), vec![merge]);
}

#[test]
fn while_loop_has_backedge_and_exit() {
    let region = body("function f(n) { let i = 0; while (i < n) { i = i + 1; } return i; }");
    let cfg = JslirCfg::new(&region);

    let header = cfg
        .block_ids()
        .into_iter()
        .find(|&b| matches!(cfg.terminal(b), TerminalView::While { .. }))
        .expect("a while header");
    let TerminalView::While { body, exit, .. } = cfg.terminal(header) else { unreachable!() };

    // The body eventually branches back to the header (a back-edge), and the exit
    // is reachable but distinct.
    assert_ne!(body, exit);
    let preds = analysis::predecessors(&cfg);
    assert!(
        preds[&header].iter().any(|&p| p != cfg.entry()),
        "the header has a back-edge predecessor from inside the loop"
    );
}

#[test]
fn break_is_visible_as_goto_break() {
    let region = body("function f(n) { let i = 0; while (i < n) { if (i) { break; } i = i + 1; } return i; }");
    let cfg = JslirCfg::new(&region);
    let has_break = cfg
        .block_ids()
        .into_iter()
        .any(|b| matches!(cfg.terminal(b), TerminalView::Goto { kind: GotoKind::Break, .. }));
    assert!(has_break, "the break shows up as a Goto{{Break}} through the neutral view");
}

#[test]
fn generic_algorithms_run_through_the_trait() {
    let region = body("function f(c) { let x = 0; if (c) { x = 1; } else { x = 2; } return x; }");
    let cfg = JslirCfg::new(&region);

    // reverse postorder: entry first, every block appears once, predecessors
    // precede successors (no cycles here).
    let rpo = analysis::reverse_postorder(&cfg);
    assert_eq!(rpo.first(), Some(&cfg.entry()));
    assert_eq!(rpo.len(), cfg.block_ids().len());

    // Everything is reachable from entry in this acyclic diamond.
    assert_eq!(analysis::reachable(&cfg).len(), cfg.block_ids().len());

    // Dominators: entry dominates itself; the merge block is dominated by entry,
    // and NOT by either arm (classic diamond property).
    let idom = analysis::immediate_dominators(&cfg);
    assert_eq!(idom[&cfg.entry()], cfg.entry());
    let TerminalView::If { then_blk, merge, .. } = cfg
        .block_ids()
        .into_iter()
        .map(|b| cfg.terminal(b))
        .find(|t| matches!(t, TerminalView::If { .. }))
        .unwrap()
    else {
        unreachable!()
    };
    let merge = merge.unwrap();
    // The arm does not dominate the merge (the other arm also reaches it).
    assert_ne!(idom[&merge], then_blk, "neither arm dominates the merge");
}
