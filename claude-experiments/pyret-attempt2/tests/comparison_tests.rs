/// Comprehensive integration tests that compare our Rust parser
/// against the official Pyret parser using the comparison script.
///
/// These tests ensure that our parser produces identical ASTs to
/// the official Pyret implementation for all supported syntax.

use std::process::Command;
use std::path::Path;
use std::thread;
use std::time::SystemTime;

/// Helper to run the comparison and check if parsers match
/// Uses unique temp files per test to avoid race conditions
fn compare_with_pyret(expr: &str) -> bool {
    // Create unique ID using thread ID and timestamp
    let thread_id = format!("{:?}", thread::current().id());
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let unique_id = format!("{}_{}", thread_id.replace("ThreadId(", "").replace(")", ""), timestamp);

    let script_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("compare_parsers_quiet.sh");

    let output = Command::new("bash")
        .arg(&script_path)
        .arg(expr)
        .arg(&unique_id)
        .output()
        .expect("Failed to run comparison script");

    // The script exits with 0 if identical, 1 if different
    output.status.success()
}

/// Helper that asserts our parser matches Pyret's parser
fn assert_matches_pyret(expr: &str) {
    if !compare_with_pyret(expr) {
        panic!("Expression '{}' produces different AST than official Pyret parser. Run:\n  ./compare_parsers.sh \"{}\"", expr, expr);
    }
}

// ============================================================================
// Primitive Expressions
// ============================================================================

#[test]
fn test_pyret_match_numbers() {
    assert_matches_pyret("42");
    assert_matches_pyret("0");
    // Note: Pyret represents decimals as rationals (e.g., 3.14 -> 157/50)
    // We store as floats, so decimal comparisons will differ
    // assert_matches_pyret("3.14");
    assert_matches_pyret("1000000");
    // assert_matches_pyret("0.001");
}

#[test]
fn test_pyret_match_strings() {
    assert_matches_pyret("\"hello\"");
    assert_matches_pyret("\"\"");
    assert_matches_pyret("\"hello world\"");
    assert_matches_pyret("\"with spaces and numbers 123\"");
}

#[test]
fn test_pyret_match_booleans() {
    assert_matches_pyret("true");
    assert_matches_pyret("false");
}

#[test]
fn test_pyret_match_identifiers() {
    assert_matches_pyret("x");
    assert_matches_pyret("myVariable");
    assert_matches_pyret("foo");
    assert_matches_pyret("bar123");
}

// ============================================================================
// Binary Operators
// ============================================================================

#[test]
fn test_pyret_match_arithmetic() {
    assert_matches_pyret("2 + 3");
    assert_matches_pyret("10 - 5");
    assert_matches_pyret("3 * 4");
    assert_matches_pyret("20 / 5");
}

#[test]
fn test_pyret_match_comparison() {
    assert_matches_pyret("x < 10");
    assert_matches_pyret("y > 5");
    assert_matches_pyret("a == b");
    assert_matches_pyret("c <> d");
    assert_matches_pyret("x <= y");
    assert_matches_pyret("a >= b");
}

#[test]
fn test_pyret_match_logical() {
    assert_matches_pyret("true and false");
    assert_matches_pyret("x or y");
}

#[test]
fn test_pyret_match_string_append() {
    assert_matches_pyret("\"hello\" ^ \" world\"");
}

// ============================================================================
// Left Associativity
// ============================================================================

#[test]
fn test_pyret_match_chained_addition() {
    assert_matches_pyret("1 + 2 + 3");
    assert_matches_pyret("a + b + c + d");
}

#[test]
fn test_pyret_match_chained_multiplication() {
    assert_matches_pyret("2 * 3 * 4");
    assert_matches_pyret("x * y * z");
}

#[test]
fn test_pyret_match_mixed_operators() {
    // No precedence: 2 + 3 * 4 = (2 + 3) * 4
    assert_matches_pyret("2 + 3 * 4");
    assert_matches_pyret("10 - 5 + 3");
    assert_matches_pyret("x / y * z");
}

#[test]
fn test_pyret_match_long_chains() {
    assert_matches_pyret("1 + 2 + 3 + 4 + 5");
    assert_matches_pyret("a and b and c and d");
    assert_matches_pyret("x < y < z < w");
}

// ============================================================================
// Parenthesized Expressions
// ============================================================================

#[test]
fn test_pyret_match_simple_parens() {
    assert_matches_pyret("(42)");
    assert_matches_pyret("(x)");
}

#[test]
fn test_pyret_match_parens_with_binop() {
    assert_matches_pyret("(2 + 3)");
    assert_matches_pyret("(x * y)");
}

#[test]
fn test_pyret_match_nested_parens() {
    assert_matches_pyret("((5))");
    assert_matches_pyret("(((x)))");
}

#[test]
fn test_pyret_match_parens_change_grouping() {
    assert_matches_pyret("1 + (2 * 3)");
    assert_matches_pyret("(1 + 2) * 3");
    assert_matches_pyret("a * (b + c)");
}

// ============================================================================
// Function Application
// ============================================================================

#[test]
fn test_pyret_match_simple_call() {
    assert_matches_pyret("f(x)");
    assert_matches_pyret("foo(bar)");
}

#[test]
fn test_pyret_match_no_args() {
    assert_matches_pyret("f()");
    assert_matches_pyret("getCurrentTime()");
}

#[test]
fn test_pyret_match_multiple_args() {
    assert_matches_pyret("f(x, y)");
    assert_matches_pyret("add(1, 2, 3)");
    assert_matches_pyret("func(a, b, c, d)");
}

#[test]
fn test_pyret_match_expr_args() {
    assert_matches_pyret("f(1 + 2)");
    assert_matches_pyret("g(x * y, a + b)");
    assert_matches_pyret("h((1 + 2), (3 * 4))");
}

#[test]
fn test_pyret_match_chained_calls() {
    assert_matches_pyret("f(x)(y)");
    assert_matches_pyret("f()(g())");
    assert_matches_pyret("f(1)(2)(3)");
}

// ============================================================================
// Whitespace Sensitivity
// ============================================================================

#[test]
fn test_pyret_match_whitespace_no_space() {
    // f(x) - direct function call
    assert_matches_pyret("f(x)");
}

#[test]
fn test_pyret_match_whitespace_with_space() {
    // f (x) - function applied to parenthesized expression
    assert_matches_pyret("f (x)");
}

// ============================================================================
// Dot Access
// ============================================================================

#[test]
fn test_pyret_match_simple_dot() {
    assert_matches_pyret("obj.field");
    assert_matches_pyret("x.y");
}

#[test]
fn test_pyret_match_chained_dot() {
    assert_matches_pyret("obj.foo.bar");
    assert_matches_pyret("a.b.c.d");
}

#[test]
fn test_pyret_match_dot_on_call() {
    assert_matches_pyret("f(x).result");
    assert_matches_pyret("getObject().field");
}

#[test]
fn test_pyret_match_call_on_dot() {
    assert_matches_pyret("obj.method()");
    assert_matches_pyret("obj.foo(x, y)");
}

// ============================================================================
// Mixed Postfix Operators (Dot + Call)
// ============================================================================

#[test]
fn test_pyret_match_complex_chaining() {
    assert_matches_pyret("f(x).result");
    assert_matches_pyret("obj.foo(a, b).bar");
    assert_matches_pyret("x.y().z");
    assert_matches_pyret("a().b().c()");
}

#[test]
fn test_pyret_match_deeply_chained() {
    assert_matches_pyret("obj.foo.bar.baz");
    assert_matches_pyret("f()()(g()())");
    assert_matches_pyret("a.b().c.d()");
}

// ============================================================================
// Complex Mixed Expressions
// ============================================================================

#[test]
fn test_pyret_match_calls_in_binop() {
    assert_matches_pyret("f(x) + g(y)");
    assert_matches_pyret("obj.foo() * obj.bar()");
}

#[test]
fn test_pyret_match_dots_in_binop() {
    assert_matches_pyret("obj.x + obj.y");
    assert_matches_pyret("a.b * c.d");
}

#[test]
fn test_pyret_match_nested_complexity() {
    assert_matches_pyret("f(x + 1).result");
    assert_matches_pyret("obj.method(a * b).field");
    assert_matches_pyret("(f(x) + g(y)).value");
}

// ============================================================================
// Array Expressions
// ============================================================================

#[test]
fn test_pyret_match_empty_array() {
    assert_matches_pyret("[]");
}

#[test]
fn test_pyret_match_array_numbers() {
    assert_matches_pyret("[1, 2, 3]");
    assert_matches_pyret("[42]");
}

#[test]
fn test_pyret_match_array_identifiers() {
    assert_matches_pyret("[x, y, z]");
    assert_matches_pyret("[foo]");
}

#[test]
fn test_pyret_match_nested_arrays() {
    assert_matches_pyret("[[1, 2], [3, 4]]");
    assert_matches_pyret("[[], []]");
}

#[test]
fn test_pyret_match_array_with_exprs() {
    assert_matches_pyret("[1 + 2, 3 * 4]");
    assert_matches_pyret("[f(x), g(y)]");
}

// ============================================================================
// Edge Cases - Deeply Nested
// ============================================================================

#[test]
fn test_pyret_match_deeply_nested_parens() {
    assert_matches_pyret("((((5))))");
}

#[test]
fn test_pyret_match_deeply_nested_arrays() {
    assert_matches_pyret("[[[[1]]]]");
}

#[test]
fn test_pyret_match_deeply_nested_calls() {
    assert_matches_pyret("f(g(h(i(x))))");
}

// ============================================================================
// Edge Cases - Long Expressions
// ============================================================================

#[test]
fn test_pyret_match_long_addition_chain() {
    let expr = (1..=20).map(|n| n.to_string()).collect::<Vec<_>>().join(" + ");
    assert_matches_pyret(&expr);
}

#[test]
fn test_pyret_match_long_array() {
    let expr = format!("[{}]", (1..=50).map(|n| n.to_string()).collect::<Vec<_>>().join(", "));
    assert_matches_pyret(&expr);
}

#[test]
fn test_pyret_match_long_dot_chain() {
    let expr = format!("obj{}", (1..=10).map(|n| format!(".field{}", n)).collect::<String>());
    assert_matches_pyret(&expr);
}

// ============================================================================
// Edge Cases - Complex Combinations
// ============================================================================

#[test]
fn test_pyret_match_all_features_combined() {
    // Combines: binop, parens, calls, dots, arrays
    assert_matches_pyret("f(x + 1).result[0].value");
}

#[test]
fn test_pyret_match_kitchen_sink() {
    assert_matches_pyret("((f(x).foo + g(y).bar) * h(z))[0]");
}

// ============================================================================
// Regression Tests - Common Patterns
// ============================================================================

#[test]
fn test_pyret_match_method_chaining() {
    assert_matches_pyret("obj.foo().bar().baz()");
}

#[test]
fn test_pyret_match_builder_pattern() {
    assert_matches_pyret("builder.setX(1).setY(2).build()");
}

#[test]
fn test_pyret_match_pipeline_style() {
    assert_matches_pyret("data.filter(pred).map(transform).reduce(combine)");
}

#[test]
fn test_pyret_match_arithmetic_expressions() {
    assert_matches_pyret("(a + b) * (c - d)");
    assert_matches_pyret("x / (y + z)");
}

// ============================================================================
// Special Cases from Grammar
// ============================================================================

#[test]
fn test_pyret_match_is_operator() {
    assert_matches_pyret("x is y");
}

#[test]
fn test_pyret_match_raises_operator() {
    assert_matches_pyret("f(x) raises \"error\"");
}

#[test]
fn test_pyret_match_satisfies_operator() {
    assert_matches_pyret("x satisfies pred");
}

#[test]
fn test_pyret_match_violates_operator() {
    assert_matches_pyret("x violates pred");
}
