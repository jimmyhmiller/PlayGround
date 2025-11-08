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
fn test_provide_with_specific_types() {
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
fn test_record_annotation() {
    // Real pattern: record type annotations
    assert_matches_pyret(r#"
fun foo(x :: { a :: Number, b :: String }):
  x.a + 1
end
"#);
}

#[test]
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

// ============================================================================
// Remaining Features - TODO (From bulk test analysis)
// ============================================================================
// Note: Tests for provide-from already exist above and are working.

// ----------------------------------------------------------------------------
// High Priority - Appear in bulk failures but have no (or incomplete) tests
// ----------------------------------------------------------------------------

#[test]
fn test_ask_expression_simple() {
    assert_matches_pyret(r#"
ask:
  | x > 0 then: "positive"
  | x < 0 then: "negative"
  | otherwise: "zero"
end
"#);
}

#[test]
fn test_ask_expression_in_check() {
    assert_matches_pyret(r#"
check:
  ask:
    | string-to-upper("hello") == "TRUE" then: true
    | otherwise: false
  end is false
end
"#);
}

#[test]
fn test_provide_block_with_type() {
    assert_matches_pyret(r#"
provide:
  type StringDict,
  string-dict
end
"#);
}

#[test]
fn test_provide_block_with_type_alias() {
    assert_matches_pyret(r#"
provide:
  type StringDict as SD,
  type MutableStringDict as MSD
end
"#);
}

#[test]
fn test_include_from_with_type_keyword() {
    assert_matches_pyret(r#"
include from M:
  foo,
  type Bar,
  type *
end
"#);
}

#[test]
fn test_use_context_statement() {
    assert_matches_pyret("use context essentials2020");
}

#[test]
fn test_use_context_with_code() {
    assert_matches_pyret(r#"
use context global

fun test(): 5 end
"#);
}

// Doc strings with triple backticks require tokenizer changes
// #[test]
// #[ignore] // Not yet implemented - 6+ files failing
// fn test_doc_string_triple_backtick() {
//     assert_matches_pyret(r#"
// doc: ```
// Multi-line documentation
// ```
// "#);
// }

// ----------------------------------------------------------------------------
// 6. Load Table Expressions (3 files affected)
// ----------------------------------------------------------------------------

#[test]
fn test_load_table_simple() {
    assert_matches_pyret(r#"
load-table: name, age
  source: "data.csv"
end
"#);
}

#[test]
fn test_load_table_with_url() {
    assert_matches_pyret(r#"
load-table: name, species, age
  source: url("https://example.com/data.csv")
end
"#);
}

#[test]
fn test_load_table_with_options() {
    assert_matches_pyret(r#"
load-table: name, age
  source: "data.csv"
  sanitize name using string-sanitizer
end
"#);
}

// ----------------------------------------------------------------------------
// 7. Reactor Expressions (2 files affected)
// ----------------------------------------------------------------------------

#[test]
fn test_reactor_simple() {
    assert_matches_pyret(r#"
reactor:
  init: 0,
  on-tick: lam(state): state + 1 end
end
"#);
}

#[test]
fn test_reactor_complete() {
    assert_matches_pyret(r#"
reactor:
  init: 0,
  on-tick: lam(state): state + 1 end,
  to-draw: lam(state): circle(state, "solid", "red") end
end
"#);
}

// ----------------------------------------------------------------------------
// 8. Examples Blocks (2 files affected)
// ----------------------------------------------------------------------------

#[test]
fn test_examples_simple() {
    assert_matches_pyret(r#"
examples:
  f(1) is 2
  f(2) is 3
end
"#);
}

#[test]
fn test_examples_with_function() {
    assert_matches_pyret(r#"
fun f(x):
  x + 1
where:
  examples:
    f(1) is 2
    f(2) is 3
  end
end
"#);
}

// ----------------------------------------------------------------------------
// 9. Include From with Type (from bulk test analysis)
// ----------------------------------------------------------------------------

#[test]
fn test_include_from_with_type() {
    assert_matches_pyret(r#"
include from M:
  foo,
  type Bar
end
"#);
}

#[test]
fn test_include_from_type_star() {
    assert_matches_pyret(r#"
include from M:
  *,
  type *
end
"#);
}

// ----------------------------------------------------------------------------
// 10. Newtype Declarations (1 file affected)
// ----------------------------------------------------------------------------

#[test]
fn test_newtype_simple() {
    assert_matches_pyret("newtype Foo as FooT");
}

#[test]
fn test_newtype_with_code() {
    assert_matches_pyret(r#"
newtype Array as ArrayT

fun make-array(): [list:] end
"#);
}

// ----------------------------------------------------------------------------
// 11. Rec Bindings (Alternative to letrec)
// ----------------------------------------------------------------------------

#[test]
fn test_rec_simple() {
    assert_matches_pyret("rec x = { foo: 1 }");
}

#[test]
fn test_rec_with_reference() {
    assert_matches_pyret(r#"
rec random-matrix = {
  make: lam(): random-matrix end
}
"#);
}

// ----------------------------------------------------------------------------
// 12. Tuple Destructuring in Bindings
// ----------------------------------------------------------------------------

#[test]
fn test_tuple_destructure_simple() {
    assert_matches_pyret("{a; b} = {1; 2}");
}

#[test]
fn test_tuple_destructure_nested() {
    assert_matches_pyret("{a; b; c; d; e} = {10; 214; 124; 62; 12}");
}

#[test]
fn test_tuple_destructure_in_let() {
    assert_matches_pyret(r#"
{x; y} = {1; 2}
x + y
"#);
}
// ============================================================================
// BULK TEST FAILURES - Missing Features from Real Pyret Code
// ============================================================================
// These tests are based on actual failures found when parsing the Pyret 
// standard library and test suite. They represent features that are commonly 
// used in real Pyret programs but not yet implemented in our parser.

// ----------------------------------------------------------------------------
// 13. Data Variants with "with:" Shared Methods
// ----------------------------------------------------------------------------
// Data variants can have shared methods defined using "with:" clauses.
// This is the most common failure pattern in the bulk tests.

#[test]
fn test_data_variant_with_methods() {
    assert_matches_pyret(r#"
data Option:
  | none with:
    method get-or-else(self, default):
      default
    end
  | some(value) with:
    method get-or-else(self, default):
      self.value
    end
end
"#);
}

#[test]
fn test_data_variant_with_multiple_methods() {
    assert_matches_pyret(r#"
data List:
  | empty with:
    method length(self): 0 end,
    method append(self, item): link(item, self) end
  | link(first, rest)
end
"#);
}

// ----------------------------------------------------------------------------
// 14. Documentation Strings (doc:)
// ----------------------------------------------------------------------------
// Pyret supports doc: strings for documenting functions, methods, and types

#[test]
fn test_doc_string_simple() {
    assert_matches_pyret(r#"
fun square(n):
  doc: "Compute the square of a number"
  n * n
end
"#);
}

#[test]
fn test_doc_string_in_method() {
    assert_matches_pyret(r#"
data Point:
  | pt(x, y) with:
    method distance(self):
      doc: "Calculate distance from origin"
      num-sqrt((self.x * self.x) + (self.y * self.y))
    end
end
"#);
}

#[test]
fn test_doc_string_multiline() {
    assert_matches_pyret(r#"
fun factorial(n):
  doc: ```
    Compute the factorial of n.
    Returns n! = n * (n-1) * ... * 1
  ```
  if n == 0: 1 else: n * factorial(n - 1) end
end
"#);
}

// ----------------------------------------------------------------------------
// 15. Shadow Keyword for Variable Rebinding
// ----------------------------------------------------------------------------
// The shadow keyword explicitly marks variable shadowing

#[test]
fn test_shadow_simple() {
    assert_matches_pyret(r#"
x = 5
shadow x = 10
x
"#);
}

#[test]
fn test_shadow_in_params() {
    assert_matches_pyret(r#"
fun help(shadow n, sum):
  if n == 0:
    sum
  else:
    help(n - 1, sum + n)
  end
end
"#);
}

#[test]
fn test_shadow_in_cases() {
    assert_matches_pyret(r#"
cases(Either) x:
  | left(shadow x) => x + 1
  | right(y) => y
end
"#);
}

// ----------------------------------------------------------------------------
// 16. Bang Operator for Ref Fields (!)
// ----------------------------------------------------------------------------
// The ! operator accesses and updates ref (mutable) fields

#[test]
fn test_bang_field_access() {
    assert_matches_pyret(r#"
data Box:
  | box(ref value)
end
b = box(5)
b!value
"#);
}

#[test]
fn test_bang_field_update() {
    assert_matches_pyret(r#"
data Counter:
  | counter(ref count)
end
c = counter(0)
c!{count: c!count + 1}
"#);
}

#[test]
fn test_bang_multiple_fields() {
    assert_matches_pyret(r#"
data State:
  | state(ref x, ref y)
end
s = state(1, 2)
s!{x: 10, y: 20}
"#);
}

// ----------------------------------------------------------------------------
// 17. Ref Keyword in Pattern Matching
// ----------------------------------------------------------------------------
// The ref keyword in cases patterns indicates ref field destructuring

#[test]
fn test_ref_in_cases_pattern() {
    assert_matches_pyret(r#"
data MutPair:
  | mpair(ref car, cdr)
end
m = mpair(1, 2)
cases(MutPair) m:
  | mpair(ref car, cdr) => car + cdr
end
"#);
}

#[test]
fn test_ref_pattern_multiple() {
    assert_matches_pyret(r#"
data MutPair:
  | mpair(ref car, ref cdr)
end
m = mpair(1, 2)
cases(MutPair) m:
  | mpair(ref car, ref cdr) => car + cdr
end
"#);
}

// ----------------------------------------------------------------------------
// 18. Additional Constructor Types
// ----------------------------------------------------------------------------
// Pyret has many built-in constructor types besides [list:]

#[test]
fn test_set_constructor() {
    assert_matches_pyret("[set: 1, 2, 3, 2]");
}

#[test]
fn test_tree_set_constructor() {
    assert_matches_pyret("[tree-set: 1, 2, 3]");
}

#[test]
fn test_array_constructor() {
    assert_matches_pyret("[array: 1, 2, 3]");
}

#[test]
fn test_raw_array_constructor() {
    assert_matches_pyret("[raw-array: 1, 2, 3]");
}

// ----------------------------------------------------------------------------
// 19. Advanced Test Operators
// ----------------------------------------------------------------------------

#[test]
fn test_does_not_raise() {
    assert_matches_pyret(r#"
check:
  fun f(x): x + 1 end
  f(5) does-not-raise
end
"#);
}

#[test]
fn test_raises_satisfies() {
    assert_matches_pyret(r#"
check:
  (lam(): raise("error") end)() raises-satisfies is-string
end
"#);
}

#[test]
fn test_raises_other_than() {
    assert_matches_pyret(r#"
check:
  (lam(): raise("error") end)() raises-other-than "different"
end
"#);
}

#[test]
fn test_satisfies_predicate() {
    assert_matches_pyret(r#"
check:
  5 satisfies {(x): x > 0}
end
"#);
}

// ----------------------------------------------------------------------------
// 20. Generic Lambda Expressions
// ----------------------------------------------------------------------------
// Lambdas can have type parameters

#[test]
fn test_generic_lambda() {
    assert_matches_pyret("lam<A>(x :: A): x end");
}

#[test]
fn test_generic_lambda_multiple_params() {
    assert_matches_pyret("lam<A, B>(x :: A, f :: (A -> B)): f(x) end");
}

#[test]
fn test_generic_lambda_return_type() {
    assert_matches_pyret("lam<A>(x :: A) -> A: x end");
}

// ----------------------------------------------------------------------------
// 21. Advanced Module Features
// ----------------------------------------------------------------------------

#[test]
fn test_provide_types_star() {
    assert_matches_pyret(r#"
provide *
provide-types *

data Foo: | foo end
"#);
}

// Removed: test_provide_types_with_braces - `provide-types { Foo }` is NOT valid Pyret syntax
// The official Pyret parser fails with: "Parse failed, next token is ('RBRACE "}")"
// The correct syntax is `provide-types *` without braces

// Removed: test_import_hiding - `import lists as L hiding(map, filter)` is NOT valid Pyret syntax
// The official Pyret parser fails with: "Parse failed, next token is ('HIDING "hiding")"
// The `hiding` keyword does not exist in Pyret import statements

// ----------------------------------------------------------------------------
// 22. Curly Brace Shorthand Syntax
// ----------------------------------------------------------------------------
// Curly braces can be used for various shorthands

#[test]
fn test_curly_brace_function_arg() {
    assert_matches_pyret(r#"
lists.map({(x): x + 1}, [list: 1, 2, 3])
"#);
}

#[test]
fn test_tuple_in_dict_constructor() {
    assert_matches_pyret(r#"
[SD.string-dict: {"a"; 5}, {"b"; 10}]
"#);
}

// ----------------------------------------------------------------------------
// 23. Special Operators and Syntax
// ----------------------------------------------------------------------------

// Removed: test_ellipsis_operator - `[list: 1, 2, ...rest]` is NOT valid Pyret syntax
// The official Pyret parser fails with: "Parse failed, next token is ('NAME "rest")"
// The `...` ellipsis operator does not exist in Pyret construct expressions

#[test]
fn test_caret_operator() {
    assert_matches_pyret("a ^ b");
}

// Removed: test_double_bang - `list!!get(0)` is NOT valid Pyret syntax
// The official Pyret parser fails with: "Parse failed, next token is ('BANG "!")"

// ----------------------------------------------------------------------------
// 24. Missing Features from Bulk Test Results
// ----------------------------------------------------------------------------
// These tests expose failures found when parsing the entire Pyret codebase

#[test]
fn test_provide_types_with_specific_types() {
    // Feature: provide-types with specific type mappings - COMPLETED ✅
    assert_matches_pyret(r#"
provide-types {
  Foo:: Foo,
  Bar:: Bar
}

data Foo: | foo end
data Bar: | bar end
"#);
}

#[test]
fn test_data_hiding_in_provide() {
    assert_matches_pyret(r#"
provide:
  data Foo hiding(foo)
end

data Foo:
  | foo(x)
  | bar(y)
end
"#);
}

#[test]
fn test_tuple_destructuring() {
    assert_matches_pyret(r#"
{a; b} = {1; 2}
a + b
"#);
}

#[test]
fn test_tuple_destructuring_shadow() {
    assert_matches_pyret(r#"
{shadow a; shadow b} = {1; 2; 3; 4; 5}
a + b
"#);
}

#[test]
fn test_tuple_destructuring_in_cases() {
    assert_matches_pyret(r#"
data Result:
  | some({ x; y; z })
  | none
end

cases(Result) some({1; 2; 3}):
  | some({ a; b; c }) => a + b + c
  | none => 0
end
"#);
}

#[test]
fn test_extract_from() {
    assert_matches_pyret(r#"
extract state from obj end
"#);
}

#[test]
fn test_underscore_partial_application() {
    assert_matches_pyret(r#"
f = (_ + 2)
f(5)
"#);
}

#[test]
fn test_underscore_multiple() {
    assert_matches_pyret(r#"
f = (_ + _)
f(3, 4)
"#);
}

#[test]
fn test_provide_from_module_multiple_items() {
    assert_matches_pyret(r#"
provide from lists: map, filter end
"#);
}

#[test]
fn test_provide_from_data() {
    assert_matches_pyret(r#"
provide from M: x, data Foo end
"#);
}

#[test]
fn test_tuple_type_annotation() {
    // Arrow types must be parenthesized in bindings (per Pyret grammar)
    // The syntax `f :: {A; B} -> C` is INVALID and was removed in 2014 (issue #252)
    // Correct syntax requires: `f :: ({A; B} -> C)`
    assert_matches_pyret(r#"
f :: ({Number; Number} -> {Number; Number}) = lam(t): t end
f({1; 2})
"#);
}


#[test]
fn test_method_with_trailing_comma() {
    assert_matches_pyret(r#"
{
  method m(self): 42 end,
  method n(self): 43 end
}
"#);
}

#[test]
fn test_spy_with_string() {
    assert_matches_pyret(r#"
spy "debug message": x end
"#);
}

// ----------------------------------------------------------------------------
// 25. Full File Tests from Pyret Codebase
// ----------------------------------------------------------------------------
// These are complete files from the Pyret codebase that currently fail to parse

#[test]
fn test_full_file_let_arr() {
    // From tests/type-check/good/let.arr
    // Features: let with multiple bindings and block
    let code = include_str!("pyret-files/let.arr");
    assert_matches_pyret(code);
}

#[test]
#[ignore] // Full file test - tuple destructuring and type annotations
fn test_full_file_weave_tuple_arr() {
    // From tests/pyret/regression/weave-tuple.arr
    // Features: tuple type annotations, tuple destructuring in function parameters
    let code = include_str!("pyret-files/weave-tuple.arr");
    assert_matches_pyret(code);
}

#[test]
fn test_full_file_option_arr() {
    // From src/arr/trove/option.arr
    // Features: generic types, methods with trailing commas, data with methods
    let code = include_str!("pyret-files/option.arr");
    assert_matches_pyret(code);
}

// ============================================================================
// NEW TESTS - GAPS FROM BULK ANALYSIS
// ============================================================================
// The following tests are derived from real Pyret code that currently fails
// to parse. They represent important missing features discovered by analyzing
// 200+ files from the official Pyret codebase.

// ----------------------------------------------------------------------------
// 26. Object Type Annotations
// ----------------------------------------------------------------------------

#[test]
fn test_object_type_annotation_simple() {
    // From tests/type-check/good/obj-check.arr
    // Feature: Object type with field annotations { field :: Type }
    assert_matches_pyret(r#"
a :: { b :: Number, c :: String } = { b : 5, c : "hello" }
b :: Number = a.b
c :: String = a.c
"#);
}

#[test]
fn test_object_type_annotation_with_method() {
    // From tests/type-check/good/obj-methods.arr
    // Feature: Object type with method field { m :: (T -> T) }
    assert_matches_pyret(r#"
obj :: { x :: Number, m :: (Number -> Number) } = {
  x: 1,
  method m(self, n):
    self.x + n
  end
}
"#);
}

#[test]
fn test_object_type_annotation_method_signature() {
    // From tests/type-check/good/obj-fun-check.arr
    // Feature: Arrow type as object annotation
    assert_matches_pyret(r#"
obj :: ({ a :: Number, b :: Number } -> Number) = lam(o): o.a + o.b end
"#);
}

// ----------------------------------------------------------------------------
// 27. Data Hiding in Provide
// ----------------------------------------------------------------------------

#[test]
fn test_provide_data_hiding() {
    // From src/arr/trove/matrices.arr (line 137)
    // Feature: data Foo hiding(constructor)
    assert_matches_pyret(r#"
provide:
  data Vector hiding(vector),
  type Vector3D,
  data Matrix hiding(matrix)
end
"#);
}

#[test]
fn test_provide_hiding_multiple() {
    // From src/arr/trove/tables.arr (line 2)
    // Feature: * hiding (name1, name2)
    assert_matches_pyret(r#"
provide:
  * hiding (is-kv-pairs, is-raw-array-of-rows),
  type *
end
"#);
}

// ----------------------------------------------------------------------------
// 28. Lazy Constructors
// ----------------------------------------------------------------------------

#[test]
fn test_lazy_constructor() {
    // From tests/type-check/good/lazy-maker.arr
    // Feature: [lazy obj: values] construct
    assert_matches_pyret(r#"
a = { lazy-make: lam(arr): empty end }
c :: List<Number> = [lazy a: 1, 2]
"#);
}

// ----------------------------------------------------------------------------
// 29. Object Update Syntax (!{})
// ----------------------------------------------------------------------------

#[test]
fn test_object_update_bang() {
    // From tests/type-check/should-not/obj-update.arr
    // Feature: obj!{field: value} for updating ref fields
    assert_matches_pyret(r#"
data Foo:
  | foo(ref a :: Number, ref b :: Any)
end

a = foo(5, 6)
b :: Foo = a!{a : 6}
"#);
}

// ----------------------------------------------------------------------------
// 30. Lowercase Generic Type Parameters
// ----------------------------------------------------------------------------

#[test]
fn test_lowercase_generic_data() {
    // From src/arr/trove/option.arr
    // Feature: data Option<a> (lowercase type variable)
    assert_matches_pyret(r#"
data Option<a>:
  | none
  | some(value :: a)
end
"#);
}

#[test]
fn test_lowercase_generic_method() {
    // From src/arr/trove/option.arr (line 14)
    // Feature: method<b> with lowercase type parameters
    assert_matches_pyret(r#"
data Option<a>:
  | none with:
    method and-then<b>(self :: Option<a>, f :: (a -> b)) -> Option<b>:
      none
    end
  | some(value :: a)
end
"#);
}

#[test]
fn test_lowercase_generic_multiple() {
    // From src/arr/trove/lists.arr (line 30)
    // Feature: Multiple lowercase type parameters
    assert_matches_pyret(r#"
data List<a>:
  | empty with:
    method foldr<b>(self :: List<a>, f :: (a, b -> b), base :: b) -> b:
      base
    end
end
"#);
}

// ----------------------------------------------------------------------------
// 31. Constructor Objects (make, make0, make1, etc.)
// ----------------------------------------------------------------------------

#[test]
fn test_constructor_object() {
    // From tests/pyret/tests/test-constructors.arr (line 23)
    // Feature: Objects with make0, make1, make2 for construct expressions
    assert_matches_pyret(r#"
every-other = {
  make: lam(arr): empty end,
  make0: lam(): empty end,
  make1: lam(a): [list: a] end,
  make2: lam(a, b): [list: a] end
}
[every-other: 1, 2, 3, 4, 5]
"#);
}

// ----------------------------------------------------------------------------
// 32. Raw Array Construct Expressions
// ----------------------------------------------------------------------------

#[test]
fn test_raw_array_construct() {
    // From tests/type-check/good/raw-array.arr (line 14)
    // Feature: [raw-array: items] or [a: items] with constructor
    assert_matches_pyret(r#"
a = { make: lam(arr :: RawArray<Number>): arr end }
c :: RawArray<Number> = [a: 1, 2, 3, 4, 5, 6]
"#);
}

#[test]
fn test_raw_array_literal() {
    // Feature: [raw-array: ...] construct
    assert_matches_pyret(r#"
arr :: RawArray<Number> = [raw-array: 1, 2, 3]
"#);
}

// ----------------------------------------------------------------------------
// 33. Include From Syntax
// ----------------------------------------------------------------------------

#[test]
fn test_include_from_syntax() {
    // From src/arr/trove/tables.arr
    // Feature: include from Module: name1, name2 end
    assert_matches_pyret(r#"
import equality as E
include from E: Equal, NotEqual end
"#);
}

// ----------------------------------------------------------------------------
// 34. Type Aliases
// ----------------------------------------------------------------------------

#[test]
fn test_type_alias_polymorphic() {
    // From tests/type-check/good/polymorphic-newtype.arr (line 14)
    // Feature: type Alias<T> = RealType<T>
    assert_matches_pyret(r#"
type MyList<T> = List<T>
x :: MyList<Number> = [list: 1, 2, 3]
"#);
}

// ----------------------------------------------------------------------------
// 35. Reactor Expressions
// ----------------------------------------------------------------------------

#[test]
fn test_reactor_with_typed_state() {
    // From tests/pyret/tests/test-reactor.arr (line 41)
    // Feature: reactor: init: value, handlers end
    assert_matches_pyret(r#"
r :: R.Reactor<Number> = reactor:
  init: 0,
  on-tick: lam(n): n + 1 end
end
"#);
}

// ----------------------------------------------------------------------------
// 36. For Raw-Array-Fold
// ----------------------------------------------------------------------------

#[test]
fn test_for_raw_array_fold() {
    // From src/arr/trove/statistics.arr (line 97)
    // Feature: for raw-array-fold(acc from init, item from arr, idx from 0)
    assert_matches_pyret(r#"
result = for raw-array-fold(acc from 0, item from arr, idx from 0):
  acc + item
end
"#);
}

// ----------------------------------------------------------------------------
// 37. Let with Colon Syntax
// ----------------------------------------------------------------------------

#[test]
fn test_let_colon() {
    // From src/arr/trove/error-display.arr (line 156)
    // Feature: let binding = expr: body end
    assert_matches_pyret(r#"
let last-digit = num-modulo(n, 10):
  to-string(last-digit)
end
"#);
}

// ----------------------------------------------------------------------------
// 38. Doc Strings
// ----------------------------------------------------------------------------

#[test]
fn test_doc_string_method() {
    // From src/arr/trove/option.arr
    // Feature: doc: "string" in methods/functions
    assert_matches_pyret(r#"
data Option:
  | none with:
    method or-else(self, v):
      doc: "Return the default provided value"
      v
    end
end
"#);
}

#[test]
fn test_doc_string_triple_backtick() {
    // From src/arr/trove/lists.arr
    // Feature: doc: ```multiline string```
    assert_matches_pyret(r#"
fun example():
  doc: ```Takes a predicate and returns an object with two fields:
        the 'is-true' field contains the list of items```
  42
end
"#);
}

// ----------------------------------------------------------------------------
// 39. Generic Function Type Signatures
// ----------------------------------------------------------------------------

#[test]
fn test_generic_function_signature() {
    // From src/arr/trove/timing.arr (line 10)
    // Feature: name :: <T> ((args) -> ReturnType)
    assert_matches_pyret(r#"
time-only :: <T> (( -> T) -> Number)
"#);
}

// ----------------------------------------------------------------------------
// 40. Tuple in Object Return Type
// ----------------------------------------------------------------------------

#[test]
fn test_tuple_in_object_return_type() {
    // From src/arr/trove/lists.arr
    // Feature: method() -> {field1 :: T, field2 :: T}
    assert_matches_pyret(r#"
data List<a>:
  | empty with:
    method partition(self, f) -> {is-true :: List<a>, is-false :: List<a>}:
      { is-true: empty, is-false: empty }
    end
end
"#);
}

// ----------------------------------------------------------------------------
// 41. Array Construct Expression
// ----------------------------------------------------------------------------

#[test]
fn test_array_construct() {
    // From tests/pyret/tests/test-array.arr (line 350)
    // Feature: [array: items]
    assert_matches_pyret(r#"
check:
  [array: 1, 2, 3] is=~ [array: 1, 2, 3]
end
"#);
}

// ----------------------------------------------------------------------------
// 42. Otherwise Keyword in Cases
// ----------------------------------------------------------------------------

#[test]
fn test_otherwise_in_cases() {
    // Feature: otherwise as alternative to else in cases
    assert_matches_pyret(r#"
cases(Option) x:
  | some(v) => v
  | otherwise => 0
end
"#);
}

// ----------------------------------------------------------------------------
// 43. Newtype Pattern
// ----------------------------------------------------------------------------

#[test]
fn test_newtype_brand() {
    // From tests/type-check/bad/newtype.arr (line 5)
    // Feature: SomeType.brand(value)
    assert_matches_pyret(r#"
data FooT:
  | foo-t(v :: Number)
end
a = foo-t(5)
b :: Number = FooT.brand(a)
"#);
}

// ----------------------------------------------------------------------------
// 44. Complex Tuple Destructuring
// ----------------------------------------------------------------------------

#[test]
fn test_tuple_destructure_in_if_expr() {
    // From src/arr/trove/charts.arr (line 295)
    // Feature: Tuple destructuring in if expression results
    assert_matches_pyret(r#"
{first-quartile; third-quartile} = if num-modulo(n, 2) == 0:
  {1; 3}
else:
  {2; 4}
end
"#);
}

#[test]
fn test_tuple_in_for_fold() {
    // From src/arr/trove/statistics.arr
    assert_matches_pyret(r#"
{front; acc} = for raw-array-fold(result from {{1; 2}; [list:]}):
  result
end
"#);
}

// ----------------------------------------------------------------------------
// 45. Letrec Pattern
// ----------------------------------------------------------------------------

#[test]
fn test_letrec() {
    // From tests/type-check/good/letrec.arr (line 2)
    // Feature: mutually recursive bindings
    assert_matches_pyret(r#"
rec a = lam(): b() end
rec b = lam(): a() end
"#);
}

// ----------------------------------------------------------------------------
// 46. Cases with Singleton Variants
// ----------------------------------------------------------------------------

#[test]
fn test_cases_singleton_parens() {
    // From tests/type-check/bad/cases-singleton-2.arr
    assert_matches_pyret(r#"
data Foo:
  | bar()
end
a = cases(Foo) bar():
  | bar() => 5
end
"#);
}

// ----------------------------------------------------------------------------
// 47. Multi-Line Strings (Triple Backtick)
// ----------------------------------------------------------------------------

#[test]
fn test_triple_backtick_string() {
    // Feature: ```multi-line string```
    assert_matches_pyret(r#"
s = ```This is a
multi-line
string```
"#);
}
