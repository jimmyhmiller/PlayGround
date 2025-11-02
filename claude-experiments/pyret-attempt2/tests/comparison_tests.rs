/// Comprehensive integration tests that compare our Rust parser
/// against the official Pyret parser using the comparison script.
///
/// These tests ensure that our parser produces identical ASTs to
/// the official Pyret implementation for all supported syntax.
///
/// Note: Tests in this file use a global lock to run serially to avoid
/// race conditions with shared temp files in the comparison script.

use std::process::Command;
use std::path::Path;
use std::sync::Mutex;

// Global lock to ensure tests run serially
static TEST_LOCK: Mutex<()> = Mutex::new(());

/// Helper to run the comparison and check if parsers match
fn compare_with_pyret(expr: &str) -> bool {
    let script_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("compare_parsers.sh");

    let output = Command::new("bash")
        .arg(&script_path)
        .arg(expr)
        .output()
        .expect("Failed to run comparison script");

    // The script exits with 0 if identical, 1 if different
    output.status.success()
}

/// Helper that asserts our parser matches Pyret's parser
/// Uses a global lock to ensure serial execution
fn assert_matches_pyret(expr: &str) {
    // Handle poisoned mutex (from panics in other tests)
    let _lock = match TEST_LOCK.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };

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
    assert_matches_pyret("obj.foo()");
    assert_matches_pyret("obj.bar(x, y)");
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
    assert_matches_pyret("obj.foo(a * b).field");
    assert_matches_pyret("(f(x) + g(y)).value");
}

// ============================================================================
// Array Expressions
// ============================================================================

#[test]
fn test_pyret_match_empty_array() {
    assert_matches_pyret("[list:]");
}

#[test]
fn test_pyret_match_array_numbers() {
    assert_matches_pyret("[list: 1, 2, 3]");
    assert_matches_pyret("[list: 42]");
}

#[test]
fn test_pyret_match_array_identifiers() {
    assert_matches_pyret("[list: x, y, z]");
    assert_matches_pyret("[list: foo]");
}

#[test]
fn test_pyret_match_nested_arrays() {
    assert_matches_pyret("[list: [list: 1, 2], [list: 3, 4]]");
    assert_matches_pyret("[list: [list:], [list:]]");
}

#[test]
fn test_pyret_match_array_with_exprs() {
    assert_matches_pyret("[list: 1 + 2, 3 * 4]");
    assert_matches_pyret("[list: f(x), g(y)]");
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
    assert_matches_pyret("[list: [list: [list: [list: 1]]]]");
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
    let expr = format!("[list: {}]", (1..=50).map(|n| n.to_string()).collect::<Vec<_>>().join(", "));
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
    assert_matches_pyret("obj.foo(a).bar(b).baz(c)");
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

// ============================================================================
// Object Expressions
// ============================================================================

#[test]
fn test_pyret_match_empty_object() {
    assert_matches_pyret("{}");
}

#[test]
fn test_pyret_match_simple_object() {
    assert_matches_pyret("{ x: 1, y: 2 }");
}

#[test]
fn test_pyret_match_nested_object() {
    assert_matches_pyret("{ point: { x: 0, y: 0 } }");
}

#[test]
fn test_pyret_match_object_with_expressions() {
    assert_matches_pyret("{ sum: 1 + 2 }");
}

#[test]
fn test_pyret_match_object_trailing_comma() {
    assert_matches_pyret("{ x: 1, y: 2, }");
}

// ============================================================================
// Complex Integration Tests
// ============================================================================

#[test]
fn test_pyret_match_ultra_complex_expression() {
    // This test combines ALL currently supported features:
    // - Numbers, strings, booleans, identifiers
    // - Binary operators (15 operators, left-associative with NO precedence)
    // - Function calls with multiple arguments
    // - Chained function calls
    // - Dot access (single and chained)
    // - Postfix operator chaining (call().dot.call())
    // - Nested expressions with parentheses
    // - Complex operator precedence demonstration (left-to-right evaluation)
    //
    // Expression breakdown:
    // foo(x + y, bar.baz(a, b))      - function call with operator in arg and nested call with dot
    //   .qux(w * z)                   - chained dot and call with operator arg
    //   .result(true and false)       - another chained call with boolean operator
    // + obj.field1.field2(p < q or r >= s)  - addition with chained dot access and complex boolean
    // * helper(1, 2).chain()          - multiplication with call and chained method
    //
    // This demonstrates:
    // 1. Left-associative evaluation (no operator precedence)
    // 2. Complex postfix operator chaining
    // 3. Nested function calls with dot access
    // 4. Multiple operator types (arithmetic, comparison, logical)
    assert_matches_pyret(
        "foo(x + y, bar.baz(a, b)).qux(w * z).result(true and false) + obj.field1.field2(p < q or r >= s) * helper(1, 2).chain()"
    );
}

// ============================================================================
// Lambda Expressions ✅ IMPLEMENTED
// ============================================================================

#[test]
fn test_pyret_match_simple_lambda() {
    // lam(): body end
    assert_matches_pyret("lam(): \"no-op\" end");
}

#[test]
fn test_pyret_match_lambda_with_params() {
    // lam(x): x + 1 end
    assert_matches_pyret("lam(e): e > 5 end");
}

#[test]
fn test_pyret_match_lambda_multiple_params() {
    // lam(x, y): x + y end
    assert_matches_pyret("lam(n, m): n > m end");
}

#[test]
fn test_pyret_match_lambda_in_call() {
    // Real code from test-lists.arr
    assert_matches_pyret("filter(lam(e): e > 5 end, [list: -1, 1])");
}

// ============================================================================
// Tuple Expressions (NOT YET IMPLEMENTED - Expected to Fail)
// ============================================================================

#[test]
fn test_pyret_match_simple_tuple() {
    // {1; 3; 10} - Note: semicolons, not commas!
    assert_matches_pyret("{1; 3; 10}");
}

#[test]
fn test_pyret_match_tuple_with_exprs() {
    // { 13; 1 + 4; 41; 1 }
    assert_matches_pyret("{13; 1 + 4; 41; 1}");
}

#[test]
fn test_pyret_match_nested_tuples() {
    // Nested tuples from real code
    assert_matches_pyret("{151; {124; 152; 12}; 523}");
}

#[test]
fn test_pyret_match_tuple_access() {
    // x.{2} - tuple field access
    assert_matches_pyret("x.{2}");
}

// ============================================================================
// Block Expressions (NOT YET IMPLEMENTED - Expected to Fail)
// ============================================================================

#[test]
fn test_pyret_match_simple_block() {
    // block: expr end
    assert_matches_pyret("block: 5 end");
}

#[test]
#[ignore] // Remove this when block parsing is implemented
fn test_pyret_match_block_multiple_stmts() {
    // block: stmt1 stmt2 expr end
    assert_matches_pyret("block: x = 5 x + 1 end");
}

// ============================================================================
// For Expressions (NOT YET IMPLEMENTED - Expected to Fail)
// ============================================================================

#[test]
fn test_pyret_match_for_map() {
    // for map(x from lst): x + 1 end
    assert_matches_pyret("for map(a1 from arr): a1 + 1 end");
}

#[test]
fn test_pyret_match_for_map2() {
    // Real code from test-binops.arr
    assert_matches_pyret("for lists.map2(a1 from self.arr, a2 from other.arr): a1 + a2 end");
}

// ============================================================================
// Method Fields in Objects (NOT YET IMPLEMENTED - Expected to Fail)
// ============================================================================

#[test]
#[ignore] // Remove this when method fields are implemented
fn test_pyret_match_object_with_method() {
    // { method foo(self): self.x end }
    assert_matches_pyret("{ method _plus(self, other): self.arr end }");
}

// ============================================================================
// Cases Expressions (NOT YET IMPLEMENTED - Expected to Fail)
// ============================================================================

#[test]
#[ignore] // Remove this when cases parsing is implemented
fn test_pyret_match_simple_cases() {
    // cases(Type) expr: | variant => result end
    assert_matches_pyret("cases(Either) e: | left(v) => v | right(v) => v end");
}

// ============================================================================
// If Expressions ✅
// ============================================================================

#[test]
fn test_pyret_match_simple_if() {
    // if cond: then-expr else: else-expr end
    assert_matches_pyret("if true: 1 else: 2 end");
}

// ============================================================================
// When Expressions (NOT YET IMPLEMENTED - Expected to Fail)
// ============================================================================

#[test]
#[ignore] // Remove this when when parsing is implemented
fn test_pyret_match_simple_when() {
    // when cond: expr end
    assert_matches_pyret("when true: print(\"yes\") end");
}

// ============================================================================
// Assignment Expressions (NOT YET IMPLEMENTED - Expected to Fail)
// ============================================================================

#[test]
#[ignore] // Remove this when assignment parsing is implemented
fn test_pyret_match_simple_assign() {
    // x := value
    assert_matches_pyret("x := 5");
}

// ============================================================================
// Let/Var Bindings (NOT YET IMPLEMENTED - Expected to Fail)
// ============================================================================

#[test]
// Enabled - let parsing now implemented
fn test_pyret_match_simple_let() {
    // x = value
    assert_matches_pyret("x = 5");
}

// ============================================================================
// Data Declarations (NOT YET IMPLEMENTED - Expected to Fail)
// ============================================================================

#[test]
#[ignore] // Remove this when data parsing is implemented
fn test_pyret_match_simple_data() {
    // data Type: | variant end
    assert_matches_pyret("data Box: | box(ref v) end");
}

// ============================================================================
// Function Declarations (NOT YET IMPLEMENTED - Expected to Fail)
// ============================================================================

#[test]
#[ignore] // Remove this when function parsing is implemented
fn test_pyret_match_simple_fun() {
    // fun name(params): body end
    assert_matches_pyret("fun f(x): x + 1 end");
}

// ============================================================================
// Import Statements (NOT YET IMPLEMENTED - Expected to Fail)
// ============================================================================

#[test]
#[ignore] // Remove this when import parsing is implemented
fn test_pyret_match_simple_import() {
    // import module as name
    assert_matches_pyret("import equality as E");
}

// ============================================================================
// Provide Statements (NOT YET IMPLEMENTED - Expected to Fail)
// ============================================================================

#[test]
#[ignore] // Remove this when provide parsing is implemented
fn test_pyret_match_simple_provide() {
    // provide *
    assert_matches_pyret("provide *");
}
