//! Round-trip + structural tests for logical-expression flattening (`&&`/`||`):
//! `a && b` becomes a CFG diamond with a merge-block argument (phi), and lifts
//! back to a `jshir.logical_expression`.

use jsir_ir::{Op, Region};
use jsir_jslir::dialect;

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
fn direct_js(src: &str) -> String {
    let op = jsir_swc::source_to_ir(src).unwrap();
    jsir_swc::ir_to_source(&op).unwrap()
}
fn jslir_js(src: &str) -> String {
    let op = jsir_swc::source_to_ir(src).unwrap();
    let (lifted, _) = jsir_jslir::roundtrip(&op);
    jsir_swc::ir_to_source(&lifted).unwrap()
}
fn assert_roundtrips(src: &str) {
    assert_eq!(norm(&direct_js(src)), norm(&jslir_js(src)), "diverged:\n{src}");
}
fn lowered(src: &str) -> usize {
    let op = jsir_swc::source_to_ir(src).unwrap();
    jsir_jslir::build_jslir(&op).1.lowered
}

fn body_cfg(src: &str) -> Region {
    let file = jsir_swc::source_to_ir(src).unwrap();
    let (jslir, _) = jsir_jslir::build_jslir(&file);
    find_body(&jslir).unwrap()
}
fn find_body(op: &Op) -> Option<Region> {
    if op.name == "jsir.function_declaration" {
        if let Some(b) = op.regions.get(1) {
            if dialect::region_is_cfg(b) {
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
fn and_roundtrips() {
    assert_roundtrips("function f(a, b) { return a && b; }");
    assert_eq!(lowered("function f(a, b) { return a && b; }"), 1);
}

#[test]
fn or_roundtrips() {
    assert_roundtrips("function g(a, b) { let x = a || b; return x; }");
}

#[test]
fn and_produces_cfg_diamond_with_phi() {
    // The logical expression must become real control flow: a cond_br, a merge
    // block with one argument (the phi), and no leftover logical_expression op.
    let cfg = body_cfg("function f(a, b) { return a && b; }");
    assert!(cfg.blocks.len() >= 3, "expected a diamond: {} blocks", cfg.blocks.len());
    // A merge block carries exactly one block argument (the phi).
    assert!(cfg.blocks.iter().any(|b| b.args.len() == 1), "no phi block-arg");
    // A logical cond_br exists.
    assert!(
        cfg.blocks
            .iter()
            .flat_map(|b| &b.ops)
            .any(|o| dialect::logexpr_operator(o).is_some()),
        "no logical cond_br"
    );
    // No structured logical_expression op survives in the CFG.
    assert!(
        !cfg.blocks
            .iter()
            .flat_map(|b| &b.ops)
            .any(|o| o.name == "jshir.logical_expression"),
        "logical_expression not flattened"
    );
}

#[test]
fn nested_and() {
    assert_roundtrips("function f(a, b, c) { return a && b && c; }");
}

#[test]
fn mixed_and_or() {
    assert_roundtrips("function f(a, b, c) { return a && (b || c); }");
}

#[test]
fn logical_in_declaration() {
    assert_roundtrips("function f(a, b) { const x = a && b; return x; }");
}

#[test]
fn logical_in_if_test() {
    assert_roundtrips("function f(a, b) { if (a && b) { return 1; } return 2; }");
}

#[test]
fn logical_in_arrow() {
    assert_roundtrips("const f = (a, b) => a && b;");
}

#[test]
fn ternary_roundtrips() {
    assert_roundtrips("function f(c, a, b) { return c ? a : b; }");
    assert_eq!(lowered("function f(c, a, b) { return c ? a : b; }"), 1);
}

#[test]
fn ternary_produces_diamond_with_phi() {
    let cfg = body_cfg("function f(c, a, b) { return c ? a : b; }");
    assert!(cfg.blocks.len() >= 4, "expected a full diamond: {} blocks", cfg.blocks.len());
    assert!(cfg.blocks.iter().any(|b| b.args.len() == 1), "no phi block-arg");
    assert!(
        cfg.blocks.iter().flat_map(|b| &b.ops).any(dialect::is_ternary),
        "no ternary cond_br"
    );
    assert!(
        !cfg.blocks
            .iter()
            .flat_map(|b| &b.ops)
            .any(|o| o.name == "jshir.conditional_expression"),
        "conditional_expression not flattened"
    );
}

#[test]
fn ternary_in_declaration() {
    assert_roundtrips("function f(c) { const x = c ? 1 : 2; return x; }");
}

#[test]
fn nested_ternary() {
    assert_roundtrips("function f(a, b, c) { return a ? b : c ? 1 : 2; }");
}

#[test]
fn ternary_and_logical_mixed() {
    assert_roundtrips("function f(a, b, c) { return a && b ? c : 0; }");
}

#[test]
fn ternary_in_arrow() {
    assert_roundtrips("const f = (c, a, b) => c ? a : b;");
}
