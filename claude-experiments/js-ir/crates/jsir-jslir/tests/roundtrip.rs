//! Round-trip tests: source → JSHIR → JSLIR → JSHIR → JS, compared (after a
//! light whitespace normalization) to source → JSHIR → JS. For functions the
//! builder lowers, the CFG round-trip must reproduce the structured output.

use jsir_jslir::{build_jslir, roundtrip};

/// Normalize away formatting differences swc may introduce; we only care that the
/// round-trip is semantically/structurally identical to the direct lowering.
fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// JS produced by the plain JSHIR path (the reference).
fn direct_js(src: &str) -> String {
    let op = jsir_swc::source_to_ir(src).expect("source_to_ir");
    jsir_swc::ir_to_source(&op).expect("ir_to_source")
}

/// JS produced through the JSLIR round-trip.
fn jslir_js(src: &str) -> String {
    let op = jsir_swc::source_to_ir(src).expect("source_to_ir");
    let (lifted, _stats) = roundtrip(&op);
    jsir_swc::ir_to_source(&lifted).expect("ir_to_source")
}

fn assert_roundtrips(src: &str) {
    assert_eq!(norm(&direct_js(src)), norm(&jslir_js(src)), "round-trip diverged for:\n{src}");
}

fn lowered_count(src: &str) -> usize {
    let op = jsir_swc::source_to_ir(src).expect("source_to_ir");
    build_jslir(&op).1.lowered
}

#[test]
fn straight_line_with_return() {
    let src = "function f() {\n  let x = 1 + 2;\n  return x;\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1, "the function should have been lowered to a CFG");
}

#[test]
fn straight_line_no_return() {
    let src = "function f() {\n  let x = 1;\n  g(x);\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}

#[test]
fn multiple_statements_and_exprs() {
    let src = "function f(a, b) {\n  let x = { a, b };\n  let y = a.c + b.d;\n  return [x, y];\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}

#[test]
fn nested_function_lowered_too() {
    let src = "function outer() {\n  let g = function () {\n    return 1;\n  };\n  return g;\n}\n";
    assert_roundtrips(src);
    // both outer and the inner function expression are straight-line
    assert_eq!(lowered_count(src), 2);
}

#[test]
fn if_braced_no_else() {
    let src = "function f(x) {\n  if (x) {\n    return 1;\n  }\n  return 2;\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1, "if should now lower to a CFG");
}

#[test]
fn if_unbraced() {
    let src = "function f(x) {\n  if (x) return 1;\n  return 2;\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}

#[test]
fn if_else_braced() {
    let src = "function f(x) {\n  if (x) {\n    return 1;\n  } else {\n    return 2;\n  }\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}

#[test]
fn if_with_following_code() {
    let src = "function f(x) {\n  let y = 0;\n  if (x) {\n    y = 1;\n  } else {\n    y = 2;\n  }\n  return y;\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}

#[test]
fn nested_if() {
    let src = "function f(x, y) {\n  if (x) {\n    if (y) {\n      return 1;\n    }\n    return 2;\n  }\n  return 3;\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}

#[test]
fn while_simple() {
    let src = "function f(x) {\n  while (x) {\n    x;\n  }\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1, "while should now lower to a CFG");
}

#[test]
fn while_with_body_and_after() {
    let src = "function f(n) {\n  let i = 0;\n  while (i < n) {\n    g(i);\n    i = i + 1;\n  }\n  return i;\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}

#[test]
fn while_with_if_inside() {
    let src = "function f(n) {\n  while (n) {\n    if (n) {\n      g();\n    }\n    n = n - 1;\n  }\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}

#[test]
fn if_with_while_inside() {
    let src = "function f(x) {\n  if (x) {\n    while (x) {\n      x = x - 1;\n    }\n  }\n  return x;\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}

#[test]
fn arrow_expression_body() {
    let src = "const f = (x) => x + 1;\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1, "arrow expr body should lower");
}

#[test]
fn arrow_block_body() {
    let src = "const f = (x) => {\n  let y = x + 1;\n  return y;\n};\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1, "block-body arrow should lower");
}

#[test]
fn arrow_expr_body_with_call() {
    let src = "const f = (a, b) => g(a, b);\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}

#[test]
fn arrow_with_if_in_block() {
    let src = "const f = (x) => {\n  if (x) {\n    return 1;\n  }\n  return 2;\n};\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}

#[test]
fn nested_arrows() {
    let src = "const f = (x) => (y) => x + y;\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 2, "both arrows lower");
}

#[test]
fn for_simple() {
    let src = "function f(n) {\n  for (let i = 0; i < n; i = i + 1) {\n    g(i);\n  }\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1, "canonical for should lower to a CFG");
}

#[test]
fn for_with_after_and_const() {
    let src = "function f(n) {\n  let total = 0;\n  for (let i = 0; i < n; i = i + 1) {\n    total = total + i;\n  }\n  return total;\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}

#[test]
fn for_with_if_inside() {
    let src = "function f(n) {\n  for (let i = 0; i < n; i = i + 1) {\n    if (i) {\n      g(i);\n    }\n  }\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}

#[test]
fn for_expression_init_passes_through() {
    // `for (i = 0; ...)` (expression init) is not the canonical form → passthrough.
    let src = "function f(n) {\n  let i;\n  for (i = 0; i < n; i = i + 1) {\n    g(i);\n  }\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 0);
}

#[test]
fn break_inside_loop_lowers() {
    let src = "function f(x) {\n  while (x) {\n    break;\n  }\n}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1, "break now lowers to a real CFG edge");
}

#[test]
fn empty_function() {
    let src = "function f() {}\n";
    assert_roundtrips(src);
    assert_eq!(lowered_count(src), 1);
}
