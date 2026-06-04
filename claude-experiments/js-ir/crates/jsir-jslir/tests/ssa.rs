//! Structural tests for `enter_ssa` (def-use + phis as a side-table analysis).

use jsir_ir::{Op, Region};
use jsir_jslir::ssa::{enter_ssa, SsaInfo};

fn body_cfg(src: &str) -> (Region, u32) {
    let file = jsir_swc::source_to_ir(src).unwrap();
    let (jslir, _) = jsir_jslir::build_jslir(&file);
    let next = max_value(&jslir) + 1;
    (find_body(&jslir).expect("a lowered function body"), next)
}
fn run(src: &str) -> Option<SsaInfo> {
    let (cfg, mut next) = body_cfg(src);
    enter_ssa(&cfg, &mut next)
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
fn max_value(op: &Op) -> u32 {
    let mut m = op.results.iter().map(|v| v.0).max().unwrap_or(0);
    for r in &op.regions {
        for b in &r.blocks {
            m = m.max(b.args.iter().map(|v| v.0).max().unwrap_or(0));
            for o in &b.ops {
                m = m.max(max_value(o));
            }
        }
    }
    m
}

#[test]
fn straight_line_def_use() {
    // `let y = x;` reads x; `return y;` reads y. No control flow → no phis.
    let info = run("function f(a) { let x = a; let y = x; return y; }").expect("ssa");
    assert!(info.phis.is_empty(), "no phis expected: {:?}", info.phis);
    // Every read of a local resolves to some def.
    assert!(!info.reaching.is_empty(), "expected def-use edges");
}

#[test]
fn if_merge_inserts_phi() {
    // x is reassigned in both arms → a phi at the merge; the final read sees it.
    let info =
        run("function f(c) { let x = 0; if (c) { x = 1; } else { x = 2; } return x; }").expect("ssa");
    let x_phis: Vec<_> = info.phis.iter().filter(|p| p.var == "x").collect();
    assert_eq!(x_phis.len(), 1, "exactly one phi for x: {:?}", info.phis);
    let phi = x_phis[0];
    assert_eq!(phi.operands.len(), 2, "phi has both arms");
    // The post-if read of x resolves to the phi value.
    assert!(
        info.reaching.values().any(|v| *v == phi.value),
        "the read of x after the if should resolve to the phi"
    );
}

#[test]
fn if_without_reassignment_no_phi() {
    // x not reassigned → no phi.
    let info = run("function f(c) { let x = 1; if (c) { g(); } return x; }").expect("ssa");
    assert!(info.phis.iter().all(|p| p.var != "x"), "no phi for x: {:?}", info.phis);
}

#[test]
fn while_loop_gets_header_phi() {
    // `x` is live before the loop and modified inside → a header phi
    // φ(preheader: 0, latch: x+1).
    let info = run("function f(n) { let x = 0; while (n) { x = x + 1; } return x; }").expect("ssa");
    let x_phis: Vec<_> = info.phis.iter().filter(|p| p.var == "x").collect();
    assert_eq!(x_phis.len(), 1, "one header phi for x: {:?}", info.phis);
    assert_eq!(x_phis[0].operands.len(), 2, "φ(preheader, latch)");
}

#[test]
fn while_unmodified_var_no_phi() {
    // `y` is read in the loop but never modified → no phi.
    let info =
        run("function f(n) { let y = 1; let x = 0; while (n) { x = y; } return x; }").expect("ssa");
    assert!(info.phis.iter().all(|p| p.var != "y"), "no phi for unmodified y: {:?}", info.phis);
}

#[test]
fn nested_loops_are_skipped() {
    assert!(run(
        "function f(n, m) { while (n) { while (m) { g(); } } }"
    )
    .is_none());
}

#[test]
fn shadowing_is_skipped() {
    // `x` declared in two different scopes → name-based keys unsafe → skip.
    assert!(run("function f(c) { let x = 1; if (c) { let x = 2; g(x); } return x; }").is_none());
}

#[test]
fn nested_if_phis() {
    let info = run(
        "function f(a, b) { let x = 0; if (a) { if (b) { x = 1; } else { x = 2; } } else { x = 3; } return x; }",
    )
    .expect("ssa");
    // Inner if merges x (1 phi), outer if merges that with 3 (1 phi).
    let x_phis = info.phis.iter().filter(|p| p.var == "x").count();
    assert_eq!(x_phis, 2, "two phis for x (inner + outer merge): {:?}", info.phis);
}
