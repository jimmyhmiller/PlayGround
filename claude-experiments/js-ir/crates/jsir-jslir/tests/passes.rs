//! Equivalence tests for ported passes (vs upstream React Compiler behavior).

use jsir_ir::{Op, Region};
use jsir_jslir::passes::eliminate_redundant_phi;
use jsir_jslir::ssa::{enter_ssa, SsaInfo};

fn ssa_of(src: &str) -> SsaInfo {
    let file = jsir_swc::source_to_ir(src).unwrap();
    let (jslir, _) = jsir_jslir::build_jslir(&file);
    let mut next = max_value(&jslir) + 1;
    let cfg = find_body(&jslir).expect("a lowered function body");
    enter_ssa(&cfg, &mut next).expect("ssa")
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
fn self_referential_loop_phi_is_eliminated() {
    // `x = x` in the loop makes the header phi φ(pre, self) trivial.
    let mut info = ssa_of("function f(c) { let x = 0; while (c) { x = x; } return x; }");
    let before = info.phis.iter().filter(|p| p.var == "x").count();
    assert_eq!(before, 1, "enter_ssa created the header phi");
    let removed = eliminate_redundant_phi(&mut info);
    assert_eq!(removed, 1, "the trivial phi is eliminated");
    assert!(info.phis.iter().all(|p| p.var != "x"), "no x phi remains");
    // The eliminated phi's value must not be referenced by any def-use edge.
    assert!(!info.reaching.is_empty());
}

#[test]
fn real_loop_phi_survives() {
    // `x = x + 1` genuinely changes x each iteration → the phi is NOT trivial.
    let mut info = ssa_of("function f(c) { let x = 0; while (c) { x = x + 1; } return x; }");
    let removed = eliminate_redundant_phi(&mut info);
    assert_eq!(removed, 0, "a real loop-carried phi is kept");
    assert_eq!(info.phis.iter().filter(|p| p.var == "x").count(), 1);
}

#[test]
fn if_merge_phi_survives() {
    // A genuine merge phi (distinct arm values) is not redundant.
    let mut info = ssa_of("function f(c) { let x = 0; if (c) { x = 1; } else { x = 2; } return x; }");
    let removed = eliminate_redundant_phi(&mut info);
    assert_eq!(removed, 0);
    assert_eq!(info.phis.iter().filter(|p| p.var == "x").count(), 1);
}

#[test]
fn idempotent() {
    let mut info = ssa_of("function f(c) { let x = 0; while (c) { x = x; } return x; }");
    eliminate_redundant_phi(&mut info);
    assert_eq!(eliminate_redundant_phi(&mut info), 0, "second run removes nothing");
}
