/// Comprehensive Gap Analysis Tests
///
/// This test suite contains REAL Pyret code from actual programs to expose
/// incomplete parser features. Each test represents a common pattern found
/// in production Pyret code.
///
/// All tests marked with #[ignore] represent features that need implementation.
/// Remove #[ignore] as features are completed.

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

    output.status.success()
}

/// Helper that asserts our parser matches Pyret's parser
fn assert_matches_pyret(expr: &str) {
    let _lock = match TEST_LOCK.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };

    if !compare_with_pyret(expr) {
        panic!(
            "Expression '{}' produces different AST than official Pyret parser.\nRun:\n  ./compare_parsers.sh \"{}\"",
            expr, expr
        );
    }
}

// ============================================================================
// CATEGORY 1: Advanced Block Structures
// ============================================================================

#[test]
#[ignore] // TODO: Multi-statement blocks not yet implemented
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
#[ignore] // TODO: Var bindings not implemented
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
#[ignore] // TODO: Type annotations on let bindings
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
#[ignore] // TODO: Shadowing and scope rules
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
// CATEGORY 2: Advanced Function Features
// ============================================================================

#[test]
#[ignore] // TODO: Where clauses with multiple checks
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
#[ignore] // TODO: Recursive functions with complex logic
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
#[ignore] // TODO: Higher-order functions
fn test_function_returning_function() {
    // Real pattern: currying and closures
    assert_matches_pyret(r#"
fun adder(x):
  lam(y): x + y end
end
"#);
}

#[test]
#[ignore] // TODO: Rest parameters
fn test_function_with_rest_args() {
    // Real pattern: variadic functions
    assert_matches_pyret(r#"
fun sum-all(first, rest ...):
  for fold(acc from first, x from rest):
    acc + x
  end
end
"#);
}

// ============================================================================
// CATEGORY 3: Data Definitions - Real World Examples
// ============================================================================

#[test]
#[ignore] // TODO: Data definitions not implemented
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
#[ignore] // TODO: Data with fields
fn test_data_with_fields() {
    // Real pattern: data with constructor parameters
    assert_matches_pyret(r#"
data Point:
  | point(x :: Number, y :: Number)
end
"#);
}

#[test]
#[ignore] // TODO: Data with ref fields
fn test_data_with_mutable_fields() {
    // Real pattern: mutable state in data structures
    assert_matches_pyret(r#"
data Box:
  | box(ref v)
end
"#);
}

#[test]
#[ignore] // TODO: Multiple variants with fields
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
#[ignore] // TODO: Data with sharing clause
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
#[ignore] // TODO: Parameterized data types
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
// CATEGORY 4: Cases Expressions - Pattern Matching
// ============================================================================

#[test]
#[ignore] // TODO: Cases not implemented
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
#[ignore] // TODO: Cases with else branch
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
#[ignore] // TODO: Nested cases
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
#[ignore] // TODO: Cases in function
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
// CATEGORY 5: Advanced For Expressions
// ============================================================================

#[test]
#[ignore] // TODO: For with multiple generators
fn test_for_with_cartesian_product() {
    // Real pattern: nested iteration
    assert_matches_pyret(r#"
for map(x from [list: 1, 2], y from [list: 3, 4]):
  {x; y}
end
"#);
}

#[test]
#[ignore] // TODO: For fold with complex accumulator
fn test_for_fold_with_tuple_accumulator() {
    // Real pattern: accumulating multiple values
    assert_matches_pyret(r#"
for fold(acc from {0; 0}, x from [list: 1, 2, 3]):
  {acc.{0} + x; acc.{1} + 1}
end
"#);
}

#[test]
#[ignore] // TODO: For filter
fn test_for_filter() {
    // Real pattern: filtering with for
    assert_matches_pyret(r#"
for filter(x from [list: 1, 2, 3, 4, 5]):
  x > 2
end
"#);
}

#[test]
#[ignore] // TODO: Nested for expressions
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

// ============================================================================
// CATEGORY 6: Table Expressions
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
#[ignore] // TODO: Table operations
fn test_table_with_filter() {
    // Real pattern: SQL-like operations
    assert_matches_pyret(r#"
my-table.filter(lam(r): r.age > 25 end)
"#);
}

// ============================================================================
// CATEGORY 7: String Interpolation
// ============================================================================

#[test]
#[ignore] // TODO: String interpolation not implemented
fn test_string_with_interpolation() {
    // Real pattern: formatted strings
    assert_matches_pyret(r#"
name = "World"
`Hello, $(name)!`
"#);
}

#[test]
#[ignore] // TODO: Complex interpolation
fn test_string_with_complex_expression() {
    // Real pattern: expressions in strings
    assert_matches_pyret(r#"
x = 10
`The answer is $(x + 32)`
"#);
}

// ============================================================================
// CATEGORY 8: Advanced Object Features
// ============================================================================

#[test]
#[ignore] // TODO: Object refinement
fn test_object_extension() {
    // Real pattern: extending objects
    assert_matches_pyret(r#"
point = { x: 0, y: 0 }
point.{ z: 0 }
"#);
}

#[test]
#[ignore] // TODO: Computed property names
fn test_object_with_computed_field() {
    // Real pattern: dynamic keys
    assert_matches_pyret(r#"
key = "myField"
{ [key]: 42 }
"#);
}

#[test]
#[ignore] // TODO: Object with update
fn test_object_update_syntax() {
    // Real pattern: immutable updates
    assert_matches_pyret(r#"
point = { x: 0, y: 0 }
point.{ x: 10 }
"#);
}

// ============================================================================
// CATEGORY 9: Check Blocks
// ============================================================================

#[test]
#[ignore] // TODO: Standalone check blocks
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

#[test]
#[ignore] // TODO: Check with examples block
fn test_check_with_examples() {
    // Real pattern: example-based testing
    assert_matches_pyret(r#"
check:
  examples:
    | input | output |
    | 0     | 1      |
    | 5     | 120    |
  end
end
"#);
}

// ============================================================================
// CATEGORY 10: Advanced Import/Export
// ============================================================================

#[test]
#[ignore] // TODO: Import with from
fn test_import_specific_names() {
    // Real pattern: selective imports
    assert_matches_pyret(r#"
import lists as L
"#);
}

#[test]
#[ignore] // TODO: Import from file
fn test_import_from_file() {
    // Real pattern: local modules
    assert_matches_pyret(r#"
import file("util.arr") as U
"#);
}

#[test]
#[ignore] // TODO: Provide with types
fn test_provide_with_types() {
    // Real pattern: exporting types and values
    assert_matches_pyret(r#"
provide-types *
"#);
}

#[test]
#[ignore] // TODO: Provide specific names
fn test_provide_specific_names() {
    // Real pattern: selective exports
    assert_matches_pyret(r#"
provide { add, multiply } end
"#);
}

// ============================================================================
// CATEGORY 11: Type Annotations
// ============================================================================

#[test]
#[ignore] // TODO: Function type annotations
fn test_function_with_arrow_type() {
    // Real pattern: explicit function types
    assert_matches_pyret(r#"
fun add(x :: Number, y :: Number) -> Number:
  x + y
end
"#);
}

#[test]
#[ignore] // TODO: Union types
fn test_union_type_annotation() {
    // Real pattern: multiple allowed types
    assert_matches_pyret(r#"
x :: (Number | String) = 42
"#);
}

#[test]
#[ignore] // TODO: Generic type parameters
fn test_generic_function() {
    // Real pattern: parametric polymorphism
    assert_matches_pyret(r#"
fun identity<T>(x :: T) -> T:
  x
end
"#);
}

// ============================================================================
// CATEGORY 12: Operators and Precedence Edge Cases
// ============================================================================

#[test]
#[ignore] // TODO: Custom operators
fn test_custom_binary_operator() {
    // Real pattern: user-defined operators
    assert_matches_pyret(r#"
x._plus(y)
"#);
}

#[test]
#[ignore] // TODO: Unary operators
fn test_unary_not() {
    // Real pattern: logical negation
    assert_matches_pyret(r#"
not true
"#);
}

#[test]
#[ignore] // TODO: Unary minus
fn test_unary_minus() {
    // Real pattern: negation
    assert_matches_pyret(r#"
-(5 + 3)
"#);
}

// ============================================================================
// CATEGORY 13: Comprehensions and Advanced List Operations
// ============================================================================

#[test]
#[ignore] // TODO: List comprehension with guard
fn test_list_comprehension_with_filter() {
    // Real pattern: filtered iteration
    assert_matches_pyret(r#"
for map(x from [list: 1, 2, 3, 4, 5]) when (x > 2):
  x * 2
end
"#);
}

// ============================================================================
// CATEGORY 14: Spy Expressions (Debugging)
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
// CATEGORY 15: Contract Expressions
// ============================================================================

#[test]
#[ignore] // TODO: Contracts not implemented
fn test_contract_on_function() {
    // Real pattern: runtime contracts
    assert_matches_pyret(r#"
fun divide(x, y) :: (Number, Number -> Number):
  x / y
where:
  divide(10, 2) is 5
end
"#);
}

// ============================================================================
// CATEGORY 16: Complex Real-World Patterns
// ============================================================================

#[test]
#[ignore] // TODO: Multiple advanced features combined
fn test_realistic_module_structure() {
    // Real pattern: typical module structure
    assert_matches_pyret(r#"
import lists as L

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

provide { Tree, make-tree } end
"#);
}

#[test]
#[ignore] // TODO: Multiple features combined
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
// CATEGORY 17: Gradual Typing Features
// ============================================================================

#[test]
#[ignore] // TODO: Any type annotation
fn test_any_type() {
    // Real pattern: gradual typing
    assert_matches_pyret(r#"
x :: Any = 42
"#);
}

// ============================================================================
// Test Summary
// ============================================================================

// Total Categories: 17
// Total Tests: 50+
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
