//! Equivalence tests for the ported `ConstantPropagation` lattice (vs upstream
//! React Compiler `evaluate_instruction`/`evaluate_binary_op`/`evaluate_phi`).
//!
//! Each test feeds a function whose `return` value is a foldable expression and
//! asserts the constant the analysis proves for it — exactly the value upstream
//! would replace the instruction with.

use jsir_ir::{Op, Region, ValueId};
use jsir_jslir::constprop::{constant_lattice, fold_constants, Constant};
use jsir_jslir::ssa::enter_ssa;

/// The constant the analysis proves for a function's `return <expr>` value, if any.
fn const_of_return(src: &str) -> Option<Constant> {
    let file = jsir_swc::source_to_ir(src).unwrap();
    let (jslir, _) = jsir_jslir::build_jslir(&file);
    let mut next = max_value(&jslir) + 1;
    let cfg = find_body(&jslir).expect("a lowered function body");
    let info = enter_ssa(&cfg, &mut next).expect("ssa");
    let lattice = constant_lattice(&cfg, &info);
    let ret_operand = return_operand(&cfg).expect("a `return <value>`");
    lattice.get(&ret_operand).cloned()
}

fn return_operand(cfg: &Region) -> Option<ValueId> {
    for b in &cfg.blocks {
        if let Some(t) = b.ops.last() {
            if t.name == "jslir.return" {
                return t.operands.first().copied();
            }
        }
    }
    None
}

fn assert_num(src: &str, expected: f64) {
    match const_of_return(src) {
        Some(Constant::Number(n)) => assert_eq!(n, expected, "for `{src}`"),
        other => panic!("expected Number({expected}) for `{src}`, got {other:?}"),
    }
}
fn assert_bool(src: &str, expected: bool) {
    match const_of_return(src) {
        Some(Constant::Bool(b)) => assert_eq!(b, expected, "for `{src}`"),
        other => panic!("expected Bool({expected}) for `{src}`, got {other:?}"),
    }
}
fn assert_str(src: &str, expected: &str) {
    match const_of_return(src) {
        Some(Constant::Str(s)) => assert_eq!(s, expected, "for `{src}`"),
        other => panic!("expected Str({expected:?}) for `{src}`, got {other:?}"),
    }
}
fn wrap(body: &str) -> String {
    format!("function f(c) {{ {body} }}")
}

#[test]
fn folds_arithmetic_through_ssa() {
    // `let x = 4; let y = x + 1; return y;` — x propagates, y folds to 5.
    assert_num(&wrap("let x = 4; let y = x + 1; return y;"), 5.0);
    assert_num(&wrap("return 2 * 3 + 4;"), 10.0);
    assert_num(&wrap("return 10 % 3;"), 1.0);
    assert_num(&wrap("return 2 ** 8;"), 256.0);
    assert_num(&wrap("return 7 / 2;"), 3.5);
}

#[test]
fn folds_bitwise_with_js_int32_semantics() {
    assert_num(&wrap("return 5 | 2;"), 7.0);
    assert_num(&wrap("return 6 & 3;"), 2.0);
    assert_num(&wrap("return 5 ^ 1;"), 4.0);
    assert_num(&wrap("return 1 << 4;"), 16.0);
    assert_num(&wrap("return 256 >> 2;"), 64.0);
    // ToInt32 wrapping: 2**32 truncates to 0, so (2**32) | 0 === 0.
    assert_num(&wrap("return 4294967296 | 0;"), 0.0);
}

#[test]
fn folds_string_concat() {
    assert_str(&wrap(r#"return "x" + "y";"#), "xy");
    assert_str(&wrap(r#"let a = "foo"; return a + "bar";"#), "foobar");
}

#[test]
fn folds_comparisons_and_equality() {
    assert_bool(&wrap("return 3 < 4;"), true);
    assert_bool(&wrap("return 4 <= 4;"), true);
    assert_bool(&wrap("return 5 > 9;"), false);
    // Strict vs abstract equality follow JS coercion rules.
    assert_bool(&wrap(r#"return 1 === 1;"#), true);
    assert_bool(&wrap(r#"return 1 == "1";"#), true);
    assert_bool(&wrap(r#"return 1 === "1";"#), false);
    assert_bool(&wrap(r#"return 0 == false;"#), true);
}

#[test]
fn undefined_global_is_not_foldable() {
    // `undefined` lowers to a global load, not a primitive, so upstream's
    // `evaluate_binary_op` (primitives only) does NOT fold these — neither do we.
    assert!(const_of_return(&wrap("return null == undefined;")).is_none());
    assert!(const_of_return(&wrap("return null === undefined;")).is_none());
}

#[test]
fn nan_is_never_strictly_equal_to_itself() {
    // 0/0 → NaN; NaN === NaN is false even though it's the same SSA value.
    assert_bool(&wrap("let a = 0 / 0; return a === a;"), false);
}

#[test]
fn folds_unary() {
    assert_bool(&wrap("return !true;"), false);
    assert_bool(&wrap("return !0;"), true); // 0 is falsy
    assert_bool(&wrap(r#"return !"";"#), true); // "" is falsy
    assert_num(&wrap("let x = 5; return -x;"), -5.0);
}

#[test]
fn phi_with_equal_arms_is_constant() {
    // Both arms assign the same value → the merge phi folds to that constant.
    assert_num(&wrap("let x; if (c) { x = 7; } else { x = 7; } return x;"), 7.0);
}

#[test]
fn phi_with_distinct_arms_is_not_constant() {
    // Arms disagree → not provably constant.
    let r = const_of_return(&wrap("let x; if (c) { x = 1; } else { x = 2; } return x;"));
    assert!(r.is_none(), "distinct merge arms are not a constant, got {r:?}");
}

#[test]
fn non_constant_input_is_not_folded() {
    // `c` is a parameter — unknown — so nothing folds.
    let r = const_of_return(&wrap("return c + 1;"));
    assert!(r.is_none(), "expression over an unknown is not constant, got {r:?}");
}

// ---------------------------------------------------------------------------
// Folding transform: lattice → rewritten IR → JS. Mirrors upstream replacing a
// computed instruction with `Primitive { .. }`.
// ---------------------------------------------------------------------------

/// Build → SSA → lattice → fold → lift → emit JS, returning (#folded, source).
fn fold_and_emit(src: &str) -> (usize, String) {
    let file = jsir_swc::source_to_ir(src).unwrap();
    let (mut jslir, _) = jsir_jslir::build_jslir(&file);
    let next = max_value(&jslir) + 1;
    let folded = fold_first_fn(&mut jslir, next);
    let lifted = jsir_jslir::lift_jslir(&jslir);
    (folded, jsir_swc::ir_to_source(&lifted).unwrap())
}

/// Fold the first lowered `function_declaration` body found; return #folded.
fn fold_first_fn(op: &mut Op, base: u32) -> usize {
    if op.name == "jsir.function_declaration" {
        if let Some(b) = op.regions.get_mut(1) {
            if jsir_jslir::dialect::region_is_cfg(b) {
                let mut next = base;
                if let Some(info) = enter_ssa(b, &mut next) {
                    let lattice = constant_lattice(b, &info);
                    return fold_constants(b, &lattice);
                }
            }
        }
    }
    for r in &mut op.regions {
        for b in &mut r.blocks {
            for o in &mut b.ops {
                let n = fold_first_fn(o, base);
                if n > 0 {
                    return n;
                }
            }
        }
    }
    0
}

#[test]
fn folds_computed_ops_into_literals() {
    // `x + 1` → `5` (one binary folded; the dead `x` read is left for DCE).
    let (n, js) = fold_and_emit(&wrap("let x = 4; let y = x + 1; return y;"));
    assert_eq!(n, 1);
    assert!(js.contains("let y = 5;"), "got: {js}");

    // A whole arithmetic chain collapses to a single literal.
    let (n, js) = fold_and_emit(&wrap("return 2 * 3 + 4;"));
    assert_eq!(n, 2, "both the `*` and `+` fold");
    assert!(js.contains("return 10;"), "got: {js}");
}

#[test]
fn folds_string_comparison_and_unary_to_literals() {
    let (_, js) = fold_and_emit(&wrap(r#"return "x" + "y";"#));
    assert!(js.contains(r#"return "xy";"#), "got: {js}");

    let (_, js) = fold_and_emit(&wrap("return 3 < 4;"));
    assert!(js.contains("return true;"), "got: {js}");

    let (_, js) = fold_and_emit(&wrap("let x = 5; return -x;"));
    assert!(js.contains("return -5;"), "got: {js}");
}

#[test]
fn leaves_non_constant_expressions_untouched() {
    // `c` is an unknown parameter — nothing folds, output is unchanged in shape.
    let (n, js) = fold_and_emit(&wrap("return c + 1;"));
    assert_eq!(n, 0);
    assert!(js.contains("return c + 1;"), "got: {js}");
}

// --- boilerplate: find the lowered body CFG + max ValueId (shared with passes.rs) ---

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
