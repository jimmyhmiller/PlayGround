//! Equivalence tests for the ported `DeadCodeElimination` pass (vs upstream
//! React Compiler `deadCodeElimination`): unused pure instructions are removed,
//! side effects are always preserved.

use jsir_ir::Op;
use jsir_jslir::constprop::{constant_lattice, fold_constants};
use jsir_jslir::dce::eliminate_dead_code;
use jsir_jslir::ssa::enter_ssa;

/// Build → (optionally fold) → DCE → lift → emit JS, plus the #removed count.
fn dce_emit(src: &str, fold: bool) -> (usize, String) {
    let file = jsir_swc::source_to_ir(src).unwrap();
    let (mut jslir, _) = jsir_jslir::build_jslir(&file);
    let base = max_value(&jslir) + 1;
    let removed = run_first_fn(&mut jslir, base, fold);
    let lifted = jsir_jslir::lift_jslir(&jslir);
    (removed, jsir_swc::ir_to_source(&lifted).unwrap())
}

fn run_first_fn(op: &mut Op, base: u32, fold: bool) -> usize {
    if op.name == "jsir.function_declaration" {
        if let Some(b) = op.regions.get_mut(1) {
            if jsir_jslir::dialect::region_is_cfg(b) {
                let mut next = base;
                if let Some(info) = enter_ssa(b, &mut next) {
                    if fold {
                        let lattice = constant_lattice(b, &info);
                        fold_constants(b, &lattice);
                    }
                }
                return eliminate_dead_code(b);
            }
        }
    }
    for r in &mut op.regions {
        for b in &mut r.blocks {
            for o in &mut b.ops {
                let n = run_first_fn(o, base, fold);
                if n > 0 {
                    return n;
                }
            }
        }
    }
    0
}

#[test]
fn removes_unused_declarations() {
    // `b` is never read → its declaration is dead; `a` is returned → kept.
    let (n, js) = dce_emit("function f() { let a = 1; let b = 2; return a; }", false);
    assert_eq!(n, 2, "the dead `store_local b` and its `2` literal op are both removed");
    assert!(js.contains("let a = 1;") && !js.contains("let b"), "got: {js}");
}

#[test]
fn pairs_with_constant_folding_to_remove_dead_reads() {
    // After folding `x + 1` → `5`, the read of `x` and `let x = 4` are both dead.
    let (_, js) = dce_emit("function f() { let x = 4; let y = x + 1; return y; }", true);
    assert!(js.contains("let y = 5;"), "got: {js}");
    assert!(!js.contains("let x"), "x should be eliminated, got: {js}");
}

#[test]
fn preserves_side_effect_when_binding_is_dead() {
    // `x` is unused, but `g()` must still run — upstream emits a bare `g();`.
    let (_, js) = dce_emit("function f(g) { let x = g(); return 1; }", false);
    assert!(js.contains("g();"), "the call's side effect must survive: {js}");
    assert!(!js.contains("let x"), "the dead binding goes: {js}");
}

#[test]
fn keeps_existing_expression_statements_with_effects() {
    // A bare `g();` statement is a side effect and is never pruned.
    let (n, js) = dce_emit("function f(g) { g(); let z = 5; return z; }", false);
    assert_eq!(n, 0);
    assert!(js.contains("g();") && js.contains("let z = 5;"), "got: {js}");
}

#[test]
fn does_not_touch_a_function_with_no_dead_code() {
    let (n, js) = dce_emit("function f(a, b) { let c = a + b; return c; }", false);
    assert_eq!(n, 0);
    assert!(js.contains("let c = a + b;") && js.contains("return c;"), "got: {js}");
}

#[test]
fn transitively_removes_a_chain_of_dead_pure_defs() {
    // d→c→b→a chain, none reaches the return → all dead.
    let (n, js) = dce_emit(
        "function f() { let a = 1; let b = a + 1; let c = b + 1; let d = c + 1; return 0; }",
        false,
    );
    assert!(n >= 4, "the whole dead chain is removed, removed={n}: {js}");
    assert!(js.contains("return 0;") && !js.contains("let "), "got: {js}");
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
