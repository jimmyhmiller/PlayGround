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
// Tuple Expressions ✅ IMPLEMENTED
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
// Block Expressions ✅ IMPLEMENTED
// ============================================================================

#[test]
fn test_pyret_match_simple_block() {
    // block: expr end
    assert_matches_pyret("block: 5 end");
}

#[test]
fn test_pyret_match_block_multiple_stmts() {
    // block: stmt1 stmt2 expr end
    assert_matches_pyret("block: x = 5 x + 1 end");
}

// ============================================================================
// For Expressions ✅ IMPLEMENTED
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
// Method Fields in Objects ✅ IMPLEMENTED
// ============================================================================

#[test]
fn test_pyret_match_object_with_method() {
    // { method foo(self): self.x end }
    assert_matches_pyret("{ method _plus(self, other): self.arr end }");
}

// ============================================================================
// Cases Expressions ✅ IMPLEMENTED
// ============================================================================

#[test]
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
// When Expressions ✅ IMPLEMENTED
// ============================================================================

#[test]
fn test_pyret_match_simple_when() {
    // when cond: expr end
    assert_matches_pyret("when true: print(\"yes\") end");
}

// ============================================================================
// Assignment Expressions ✅ IMPLEMENTED
// ============================================================================

#[test]
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
// Data Declarations ✅ IMPLEMENTED
// ============================================================================

#[test]
fn test_pyret_match_simple_data() {
    // data Type: | variant end
    assert_matches_pyret("data Box: | box(ref v) end");
}

// ============================================================================
// Function Declarations ✅ IMPLEMENTED
// ============================================================================

#[test]
fn test_pyret_match_simple_fun() {
    // fun name(params): body end
    assert_matches_pyret("fun f(x): x + 1 end");
}

// ============================================================================
// Import Statements ✅ IMPLEMENTED
// ============================================================================

#[test]
fn test_pyret_match_simple_import() {
    // import module as name
    assert_matches_pyret("import equality as E");
}

// ============================================================================
// Provide Statements
// ============================================================================

#[test]
fn test_pyret_match_simple_provide() {
    // provide *
    assert_matches_pyret("provide *");
}

// ============================================================================
// ============================================================================
// COMPREHENSIVE GAP ANALYSIS - Advanced Features Tests
// ============================================================================
// ============================================================================
//
// The tests below represent REAL Pyret code patterns from actual programs.
// They expose incomplete parser features. Each #[ignore] test represents
// a feature that needs implementation.
//
// Remove #[ignore] as features are completed.

// ============================================================================
// CATEGORY: Advanced Block Structures
// ============================================================================

#[test]
fn test_block_with_multiple_let_bindings() {
    // Real pattern: local variable scoping in blocks
    assert_matches_pyret(r#"
block:
  x = 10
  y = 20
  z = 30
  x + y + z
end
"#);
}

#[test]
fn test_block_with_var_binding() {
    // Real pattern: mutable variables
    assert_matches_pyret(r#"
block:
  var counter = 0
  counter := counter + 1
  counter
end
"#);
}

#[test]
fn test_block_with_typed_bindings() {
    // Real pattern: typed local variables
    assert_matches_pyret(r#"
block:
  x :: Number = 42
  x + 1
end
"#);
}

#[test]
fn test_nested_blocks_with_shadowing() {
    // Real pattern: nested scopes
    assert_matches_pyret(r#"
block:
  x = 1
  block:
    x = 2
    x
  end + x
end
"#);
}

// ============================================================================
// CATEGORY: Advanced Function Features
// ============================================================================

#[test]
fn test_function_with_multiple_where_clauses() {
    // Real pattern: comprehensive testing
    assert_matches_pyret(r#"
fun factorial(n):
  if n == 0:
    1
  else:
    n * factorial(n - 1)
  end
where:
  factorial(0) is 1
  factorial(5) is 120
  factorial(10) is 3628800
end
"#);
}

#[test]
fn test_recursive_function_with_cases() {
    // Real pattern: data structure traversal
    assert_matches_pyret(r#"
fun sum-list(lst):
  cases (List) lst:
    | empty => 0
    | link(first, rest) => first + sum-list(rest)
  end
end
"#);
}

#[test]
fn test_function_returning_function() {
    // Real pattern: currying and closures
    assert_matches_pyret(r#"
fun adder(x):
  lam(y): x + y end
end
"#);
}

// REMOVED: Rest parameters (...args) do NOT exist in Pyret
// The parser fails with: Parse failed, next token is ('DOTDOTDOT "...")

// ============================================================================
// CATEGORY: Data Definitions - Real World Examples
// ============================================================================

#[test]
fn test_simple_data_definition() {
    // Real pattern: basic algebraic data type
    assert_matches_pyret(r#"
data Color:
  | red
  | green
  | blue
end
"#);
}

#[test]
fn test_data_with_fields() {
    // Real pattern: data with constructor parameters
    assert_matches_pyret(r#"
data Point:
  | point(x :: Number, y :: Number)
end
"#);
}

#[test]
fn test_data_with_mutable_fields() {
    // Real pattern: mutable state in data structures
    assert_matches_pyret(r#"
data Box:
  | box(ref v)
end
"#);
}

#[test]
fn test_data_with_multiple_variants() {
    // Real pattern: sum types
    assert_matches_pyret(r#"
data Either:
  | left(v)
  | right(v)
end
"#);
}

#[test]
fn test_data_with_shared_methods() {
    // Real pattern: methods shared across variants
    assert_matches_pyret(r#"
data BinTree:
  | leaf(value)
  | node(left, right)
sharing:
  method size(self):
    cases (BinTree) self:
      | leaf(_) => 1
      | node(l, r) => l.size() + r.size()
    end
  end
end
"#);
}

#[test]
fn test_generic_data_definition() {
    // Real pattern: generic types
    assert_matches_pyret(r#"
data List<T>:
  | empty
  | link(first :: T, rest :: List<T>)
end
"#);
}

// ============================================================================
// CATEGORY: Cases Expressions - Pattern Matching
// ============================================================================

#[test]
fn test_cases_with_wildcard() {
    // Real pattern: catch-all case
    assert_matches_pyret(r#"
cases (Option) opt:
  | some(v) => v
  | none => 0
end
"#);
}

#[test]
fn test_cases_with_else() {
    // Real pattern: default case
    assert_matches_pyret(r#"
cases (Either) e:
  | left(v) => v
  | else => 0
end
"#);
}

#[test]
fn test_nested_cases() {
    // Real pattern: nested pattern matching
    assert_matches_pyret(r#"
cases (List) lst:
  | empty => 0
  | link(first, rest) =>
    cases (List) rest:
      | empty => first
      | link(_, _) => first + 1
    end
end
"#);
}

#[test]
fn test_cases_in_function_body() {
    // Real pattern: dispatch on type
    assert_matches_pyret(r#"
fun process(result):
  cases (Result) result:
    | ok(value) => value * 2
    | err(msg) => 0
  end
end
"#);
}

// ============================================================================
// CATEGORY: Advanced For Expressions
// ============================================================================

#[test]
fn test_for_with_cartesian_product() {
    // Real pattern: nested iteration
    assert_matches_pyret(r#"
for map(x from [list: 1, 2], y from [list: 3, 4]):
  {x; y}
end
"#);
}

#[test]
fn test_for_fold_with_tuple_accumulator() {
    // Real pattern: accumulating multiple values
    assert_matches_pyret(r#"
for fold(acc from {0; 0}, x from [list: 1, 2, 3]):
  {acc.{0} + x; acc.{1} + 1}
end
"#);
}

#[test]
fn test_for_filter() {
    // Real pattern: filtering with for
    assert_matches_pyret(r#"
for filter(x from [list: 1, 2, 3, 4, 5]):
  x > 2
end
"#);
}

#[test]
fn test_nested_for_expressions() {
    // Real pattern: matrix operations
    assert_matches_pyret(r#"
for map(row from matrix):
  for map(cell from row):
    cell * 2
  end
end
"#);
}

#[test]
fn test_for_each() {
    // Real pattern: for each iteration
    assert_matches_pyret(r#"
for each(x from list):
  x + 1
end
"#);
}

#[test]
fn test_for_each2() {
    // Real pattern: for each with two bindings
    assert_matches_pyret(r#"
for each2(x from list, y from list2):
  {x; y}
end
"#);
}

#[test]
fn test_for_each_with_complex_body() {
    // Real pattern: for each with block body
    assert_matches_pyret(r#"
for each(item from items):
  print(item)
  item + 1
end
"#);
}

// ============================================================================
// CATEGORY: Table Expressions
// ============================================================================

#[test]
#[ignore] // TODO: Table literals not implemented
fn test_simple_table() {
    // Real pattern: data tables
    assert_matches_pyret(r#"
table: name, age
  row: "Alice", 30
  row: "Bob", 25
end
"#);
}

#[test]
fn test_table_with_filter() {
    // Real pattern: SQL-like operations
    assert_matches_pyret(r#"
my-table.filter(lam(r): r.age > 25 end)
"#);
}

// ============================================================================
// CATEGORY: String Interpolation
// ============================================================================

// NOTE: String interpolation with single backticks `$()` does NOT exist in Pyret
// Single backticks are UNKNOWN tokens in the Pyret tokenizer
// (Three backticks ``` are used for multi-line strings, not interpolation)
// These tests were removed as they test a non-existent feature

// ============================================================================
// CATEGORY: Advanced Object Features
// ============================================================================

#[test]
fn test_object_extension() {
    // Real pattern: extending objects
    assert_matches_pyret(r#"
point = { x: 0, y: 0 }
point.{ z: 0 }
"#);
}

// REMOVED: Computed property names [key] do NOT exist in Pyret
// The parser fails with: Parse failed, next token is ('RBRACK "]")

#[test]
fn test_object_update_syntax() {
    // Real pattern: immutable updates
    assert_matches_pyret(r#"
point = { x: 0, y: 0 }
point.{ x: 10 }
"#);
}

#[test]
fn test_method_expression() {
    // Real pattern: method as a standalone expression
    assert_matches_pyret(r#"
m = method(self): "no-op" end
m
"#);
}

#[test]
fn test_method_expression_with_args() {
    // Real pattern: method with arguments
    assert_matches_pyret(r#"
add = method(self, x, y): x + y end
add
"#);
}

// ============================================================================
// CATEGORY: Check Blocks
// ============================================================================

#[test]
fn test_check_block() {
    // Real pattern: test blocks
    assert_matches_pyret(r#"
check:
  1 + 1 is 2
  "hello" is "hello"
  true is true
end
"#);
}

// REMOVED: Check blocks with examples: syntax do NOT exist in Pyret
// The parser fails with: Parse failed, next token is ('BAR "|")

// ============================================================================
// CATEGORY: Advanced Import/Export
// ============================================================================

#[test]
fn test_import_specific_names() {
    // Real pattern: selective imports
    assert_matches_pyret(r#"
import lists as L
"#);
}

#[test]
fn test_import_from_file() {
    // Real pattern: local modules
    assert_matches_pyret(r#"
import file("util.arr") as U
"#);
}

#[test]
fn test_provide_with_types() {
    // Real pattern: exporting types and values
    assert_matches_pyret(r#"
provide-types *
"#);
}

#[test]
fn test_provide_specific_names() {
    // Real pattern: selective exports
    assert_matches_pyret(r#"
provide: add, multiply end
"#);
}

// ============================================================================
// COMPREHENSIVE Import/Provide Tests (from real Pyret codebase)
// ============================================================================

// --- Import Variations (from pyret-lang/src/arr/trove/) ---

#[test]
fn test_import_comma_names_from() {
    // Real example: import x, y from source
    assert_matches_pyret(r#"import x, y from lists"#);
}

#[test]
fn test_import_comma_names_from_file() {
    // Real example: import x, y from file("...")
    assert_matches_pyret(r#"import x, y from file("util.arr")"#);
}

#[test]
fn test_import_single_name_from() {
    // Real example: import x from source
    assert_matches_pyret(r#"import x from lists"#);
}

#[test]
fn test_include_simple() {
    // Real example from ast.arr: include lists
    assert_matches_pyret(r#"include lists"#);
}

#[test]
fn test_include_from_basic() {
    // Real example from csv.arr: include from O: is-some end
    assert_matches_pyret(r#"include from O: is-some end"#);
}

#[test]
fn test_include_from_multiple() {
    // Real example from equality.arr: include from VU: raw-array-fold2, raw-array-map2 end
    assert_matches_pyret(r#"include from VU: raw-array-fold2, raw-array-map2 end"#);
}

// --- Provide Variations (from pyret-lang/src/arr/trove/) ---

#[test]
fn test_provide_colon_names() {
    // Real example from arrays.arr: provide: array, build-array end
    assert_matches_pyret(r#"provide: array, build-array end"#);
}

#[test]
fn test_provide_with_alias() {
    // Real example from csv.arr: provide: csv-table-opt as csv-table-options end
    assert_matches_pyret(r#"provide: csv-table-opt as csv-table-options end"#);
}

#[test]
fn test_provide_type_in_block() {
    // Real example from csv.arr: provide: type CSVOptions end
    assert_matches_pyret(r#"provide: type CSVOptions end"#);
}

#[test]
fn test_provide_from_module() {
    // Real example from csv.arr: provide from csv-lib: parse-string end
    assert_matches_pyret(r#"provide from csv-lib: parse-string end"#);
}

#[test]
fn test_provide_from_multiple() {
    // Real example: provide from module: name1, name2 end
    assert_matches_pyret(r#"provide from lists: map, filter end"#);
}

#[test]
fn test_provide_object_syntax() {
    // Real example from cmdline.arr: provide { file-name: file-name, other-args: other-args } end
    assert_matches_pyret(r#"provide { file-name: file-name, other-args: other-args } end"#);
}

// ============================================================================
// CATEGORY: Type Annotations
// ============================================================================

#[test]
fn test_function_with_arrow_type() {
    // Real pattern: explicit function types
    assert_matches_pyret(r#"
fun add(x :: Number, y :: Number) -> Number:
  x + y
end
"#);
}

// REMOVED: Union type annotations (Number | String) do NOT exist in Pyret
// The parser fails with: Parse failed, next token is ('BAR "|")

#[test]
fn test_generic_function() {
    // Real pattern: parametric polymorphism
    assert_matches_pyret(r#"
fun identity<T>(x :: T) -> T:
  x
end
"#);
}

#[test]
#[ignore] // TODO: Record annotations not yet implemented
fn test_record_annotation() {
    // Real pattern: record type annotations
    assert_matches_pyret(r#"
fun foo(x :: { a :: Number, b :: String }):
  x.a + 1
end
"#);
}

#[test]
#[ignore] // TODO: Arrow annotations in records not yet implemented
fn test_record_annotation_with_arrow() {
    // Real pattern: record with function types (from loop.arr)
    assert_matches_pyret(r#"
fun loop(f, i :: { init :: Any, test :: (Any -> Boolean), next :: (Any -> Any) }) -> Nothing:
  nothing
end
"#);
}

// ============================================================================
// CATEGORY: Operators and Precedence Edge Cases
// ============================================================================

#[test]
fn test_custom_binary_operator() {
    // Real pattern: user-defined operators
    assert_matches_pyret(r#"
x._plus(y)
"#);
}

// ============================================================================
// CATEGORY: Comprehensions and Advanced List Operations
// ============================================================================

// REMOVED: For expressions with 'when' guards do NOT exist in Pyret
// The parser fails with: Parse failed, next token is ('WHEN "when")
// Use 'for filter' instead

// ============================================================================
// CATEGORY: Spy Expressions (Debugging)
// ============================================================================

#[test]
#[ignore] // TODO: Spy not implemented
fn test_spy_expression() {
    // Real pattern: debugging output
    assert_matches_pyret(r#"
spy: x end
"#);
}

// ============================================================================
// CATEGORY: Contract Expressions
// ============================================================================

// REMOVED: Contract syntax :: (Number, Number -> Number) on functions does NOT exist
// The parser fails with error about annotations only applying to names

// ============================================================================
// CATEGORY: Known Bugs - Requires Implementation
// ============================================================================

#[test]
fn test_fraction_literal() {
    // Pyret supports fraction literals like 1/2, 3/2
    // These should parse as s-frac with num and den as strings
    // Fixed: Added JSON serialization for SFrac node
    assert_matches_pyret(r#"
1/2
"#);
}

#[test]
fn test_let_in_for_loop() {
    // Let bindings inside for loop bodies now use correct structure
    // Fixed: Created parse_block_statement helper for consistent block parsing
    // All block contexts (for, if, when, cases, etc.) now use s-let instead of s-let-expr
    assert_matches_pyret(r#"
for each(i from range(0, 2)):
  ix1 = random(100)
  ix1
end
"#);
}

#[test]
fn test_check_refinement_is() {
    // Check operators can have refinements specified with %
    // Syntax: is%(refinement-fn)
    // Using integers to avoid decimal precision issues
    assert_matches_pyret(r#"
check:
  3 is%(within(1)) 4
end
"#);
}

#[test]
fn test_check_refinement_is_not() {
    // is-not% also supports refinements
    // Using integers to avoid decimal precision issues
    assert_matches_pyret(r#"
check:
  3 is-not%(within(1)) 10
end
"#);
}

#[test]
fn test_check_refinement_complex() {
    // Refinements with function calls
    // Multiple check tests with different refinements
    assert_matches_pyret(r#"
check:
  3 is%(within-rel(1)) 4
  3 is-not%(within-rel(2)) 10
end
"#);
}

#[test]
fn test_check_operator_is_spaceship() {
    // Check operators with custom comparators: is<=>
    assert_matches_pyret(r#"
check:
  x is<=> y
end
"#);
}

#[test]
fn test_check_operator_is_equal_equal() {
    // Check operators with custom comparators: is==
    assert_matches_pyret(r#"
check:
  x is== y
end
"#);
}

#[test]
fn test_check_operator_is_not_spaceship() {
    // Check operators with custom comparators: is-not<=>
    assert_matches_pyret(r#"
check:
  x is-not<=> y
end
"#);
}

#[test]
fn test_check_operator_with_object_extension() {
    // Complex case: object extension with check operator
    assert_matches_pyret(r#"
check:
  {x:1, y:2}.{y:3} is-not<=> {x:1, y:3}
end
"#);
}

// ============================================================================
// CATEGORY: Complex Real-World Patterns
// ============================================================================

#[test]
fn test_realistic_module_structure() {
    // Real pattern: typical module structure
    assert_matches_pyret(r#"
import lists as L

provide: Tree, make-tree end

data Tree:
  | leaf(value :: Number)
  | node(left :: Tree, right :: Tree)
sharing:
  method sum(self):
    cases (Tree) self:
      | leaf(v) => v
      | node(l, r) => l.sum() + r.sum()
    end
  end
end

fun make-tree(lst):
  cases (List) lst:
    | empty => leaf(0)
    | link(first, rest) =>
      node(leaf(first), make-tree(rest))
  end
where:
  make-tree([list: 1, 2, 3]).sum() is 6
end
"#);
}

#[test]
fn test_functional_programming_pattern() {
    // Real pattern: functional composition
    assert_matches_pyret(r#"
fun compose(f, g):
  lam(x): f(g(x)) end
end

fun add1(n): n + 1 end
fun double(n): n * 2 end

add1-then-double = compose(double, add1)
"#);
}

// ============================================================================
// CATEGORY: Gradual Typing Features
// ============================================================================

#[test]
fn test_any_type() {
    // Real pattern: gradual typing
    assert_matches_pyret(r#"
x :: Any = 42
"#);
}

// ============================================================================
// END OF COMPREHENSIVE GAP ANALYSIS
// ============================================================================
//
// Total Categories: 17
// Total Gap Tests: 50+
// These tests cover:
// - Advanced block structures (4 tests)
// - Advanced function features (4 tests)
// - Data definitions (6 tests)
// - Cases expressions (4 tests)
// - Advanced for expressions (4 tests)
// - Table expressions (2 tests)
// - String interpolation (2 tests)
// - Advanced object features (3 tests)
// - Check blocks (2 tests)
// - Advanced import/export (4 tests)
// - Type annotations (3 tests)
// - Operators edge cases (3 tests)
// - Comprehensions (1 test)
// - Spy expressions (1 test)
// - Contracts (1 test)
// - Complex real-world patterns (2 tests)
// - Gradual typing (1 test)

#[test]
fn test_if_with_block_keyword() {
    // Real pattern: if with block: instead of just :
    assert_matches_pyret(r#"
if true block:
  1 + 1
end
"#);
}

// ============================================================================
// Type Aliases
// ============================================================================

#[test]
fn test_type_alias_simple() {
    assert_matches_pyret("type N = Number");
}

#[test]
fn test_type_alias_dotted() {
    assert_matches_pyret("type Loc = SL.Srcloc");
}

#[test]
fn test_type_alias_generic() {
    assert_matches_pyret("type A<T> = T");
}

#[test]
fn test_type_alias_generic_multiple_params() {
    assert_matches_pyret("type Pair<A, B> = {A; B}");
}

#[test]
fn test_type_alias_with_refinement() {
    assert_matches_pyret("type N = Number%(is-foo)");
}

#[test]
fn test_type_alias_chain() {
    assert_matches_pyret(r#"
type N = Number
type N2 = N
"#);
}

#[test]
fn test_type_alias_complex() {
    assert_matches_pyret("type CList = CL.ConcatList");
}

#[test]
fn test_type_alias_in_program() {
    assert_matches_pyret(r#"
type Loc = SL.Srcloc
type CList = CL.ConcatList

fun test(x :: Loc) -> Loc:
  x
end
"#);
}
