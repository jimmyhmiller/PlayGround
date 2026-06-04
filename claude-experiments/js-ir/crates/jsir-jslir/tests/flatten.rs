//! Structural tests for within-block flattening: a single-binding `let/const x =
//! e` becomes block-level value ops + a `jslir.store_local`, with no nested
//! `variable_declaration` region (the React-HIR-like flat shape).

use jsir_ir::{Op, Region};
use jsir_jslir::dialect;

fn body_cfg(src: &str) -> Region {
    let file = jsir_swc::source_to_ir(src).unwrap();
    let (jslir, _) = jsir_jslir::build_jslir(&file);
    find_first_function_body(&jslir).expect("a lowered function body")
}

fn find_first_function_body(op: &Op) -> Option<Region> {
    if op.name == "jsir.function_declaration" {
        if let Some(body) = op.regions.get(1) {
            if dialect::region_is_cfg(body) {
                return Some(body.clone());
            }
        }
    }
    for r in &op.regions {
        for b in &r.blocks {
            for o in &b.ops {
                if let Some(found) = find_first_function_body(o) {
                    return Some(found);
                }
            }
        }
    }
    None
}

fn entry_ops(cfg: &Region) -> Vec<String> {
    cfg.blocks[0].ops.iter().map(|o| o.name.clone()).collect()
}

#[test]
fn single_binding_flattens_to_store_local() {
    let cfg = body_cfg("function f() { const x = 1 + 2; return x; }");
    let names = entry_ops(&cfg);
    // The declaration's value ops are hoisted to block level + a store_local; no
    // nested variable_declaration remains.
    assert!(names.iter().any(|n| n == dialect::STORE_LOCAL), "no store_local: {names:?}");
    assert!(!names.iter().any(|n| n == "jsir.variable_declaration"), "decl not flattened: {names:?}");
    assert!(names.iter().any(|n| n == "jsir.binary_expression"), "init ops not hoisted: {names:?}");
}

#[test]
fn store_local_carries_name_and_kind() {
    let cfg = body_cfg("function f() { const total = g(); return total; }");
    let store = cfg.blocks[0]
        .ops
        .iter()
        .find(|o| dialect::is_store_local(o))
        .expect("a store_local");
    assert_eq!(dialect::store_local_parts(store), Some(("total", "const")));
    // It references the call's value as its init operand.
    assert_eq!(store.operands.len(), 1);
}

#[test]
fn declaration_without_init() {
    let cfg = body_cfg("function f() { let x; x = 1; return x; }");
    let store = cfg.blocks[0]
        .ops
        .iter()
        .find(|o| dialect::is_store_local(o))
        .expect("a store_local");
    assert_eq!(dialect::store_local_parts(store), Some(("x", "let")));
    assert_eq!(store.operands.len(), 0, "no init operand for `let x;`");
}

#[test]
fn destructuring_stays_coarse() {
    // Destructuring is not flattened yet; the variable_declaration is kept whole.
    let cfg = body_cfg("function f(o) { const { a, b } = o; return a + b; }");
    let names = entry_ops(&cfg);
    assert!(names.iter().any(|n| n == "jsir.variable_declaration"), "{names:?}");
    assert!(!names.iter().any(|n| n == dialect::STORE_LOCAL), "{names:?}");
}
