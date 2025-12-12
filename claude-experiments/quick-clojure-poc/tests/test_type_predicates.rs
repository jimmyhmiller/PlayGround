// Test suite for type predicates: nil?, number?, string?, fn?, identical?
//
// Test Coverage:
// ✅ nil? - checks if value is nil
// ✅ number? - checks if value is integer or float
// ✅ string? - checks if value is a string
// ✅ fn? - checks if value is a function or closure
// ✅ identical? - raw pointer/value comparison

use quick_clojure_poc::*;
use std::sync::Arc;
use std::cell::UnsafeCell;

/// Helper function to run a test case and return the raw tagged result
fn run_and_get_tagged(code: &str) -> i64 {
    let val = reader::read(code).expect(&format!("Failed to read: {}", code));
    let ast = clojure_ast::analyze(&val).expect(&format!("Failed to analyze: {}", code));

    let runtime = Arc::new(UnsafeCell::new(gc_runtime::GCRuntime::new()));
    trampoline::set_runtime(runtime.clone());
    let mut compiler = compiler::Compiler::new(runtime.clone());
    let result_val = compiler.compile(&ast).expect(&format!("Compiler failed for: {}", code));
    let result_reg = compiler.ensure_register(result_val);
    let instructions = compiler.take_instructions();

    let mut codegen = arm_codegen::Arm64CodeGen::new();
    codegen.compile(&instructions, &result_reg, 0).expect(&format!("Codegen failed for: {}", code));
    codegen.execute().expect(&format!("Execute failed for: {}", code))
}

/// Helper function to run a test case expecting a boolean true result
fn run_test_true(code: &str) {
    let tagged_result = run_and_get_tagged(code);
    // true is 11 (0b1011)
    assert_eq!(tagged_result, 11, "Expected true (11), got {} for: {}", tagged_result, code);
}

/// Helper function to run a test case expecting a boolean false result
fn run_test_false(code: &str) {
    let tagged_result = run_and_get_tagged(code);
    // false is 3 (0b0011)
    assert_eq!(tagged_result, 3, "Expected false (3), got {} for: {}", tagged_result, code);
}

// =============================================================================
// nil? Tests
// =============================================================================

#[test]
fn test_nil_pred_with_nil() {
    run_test_true("(nil? nil)");
}

#[test]
fn test_nil_pred_with_false() {
    // false is NOT nil
    run_test_false("(nil? false)");
}

#[test]
fn test_nil_pred_with_true() {
    run_test_false("(nil? true)");
}

#[test]
fn test_nil_pred_with_zero() {
    run_test_false("(nil? 0)");
}

#[test]
fn test_nil_pred_with_positive_int() {
    run_test_false("(nil? 42)");
}

#[test]
fn test_nil_pred_with_negative_int() {
    run_test_false("(nil? -1)");
}

#[test]
fn test_nil_pred_with_float() {
    run_test_false("(nil? 3.14)");
}

#[test]
fn test_nil_pred_with_string() {
    run_test_false("(nil? \"hello\")");
}

#[test]
fn test_nil_pred_with_empty_string() {
    run_test_false("(nil? \"\")");
}

#[test]
fn test_nil_pred_with_keyword() {
    run_test_false("(nil? :foo)");
}

#[test]
fn test_nil_pred_with_function() {
    run_test_false("(nil? (fn [x] x))");
}

// =============================================================================
// number? Tests
// =============================================================================

#[test]
fn test_number_pred_with_zero() {
    run_test_true("(number? 0)");
}

#[test]
fn test_number_pred_with_positive_int() {
    run_test_true("(number? 42)");
}

#[test]
fn test_number_pred_with_negative_int() {
    run_test_true("(number? -100)");
}

#[test]
fn test_number_pred_with_large_int() {
    run_test_true("(number? 1000000)");
}

#[test]
fn test_number_pred_with_float() {
    run_test_true("(number? 3.14)");
}

#[test]
fn test_number_pred_with_zero_float() {
    run_test_true("(number? 0.0)");
}

#[test]
fn test_number_pred_with_negative_float() {
    run_test_true("(number? -2.5)");
}

#[test]
fn test_number_pred_with_nil() {
    run_test_false("(number? nil)");
}

#[test]
fn test_number_pred_with_true() {
    run_test_false("(number? true)");
}

#[test]
fn test_number_pred_with_false() {
    run_test_false("(number? false)");
}

#[test]
fn test_number_pred_with_string() {
    run_test_false("(number? \"42\")");
}

#[test]
fn test_number_pred_with_keyword() {
    run_test_false("(number? :number)");
}

#[test]
fn test_number_pred_with_function() {
    run_test_false("(number? (fn [x] x))");
}

// =============================================================================
// string? Tests
// =============================================================================

#[test]
fn test_string_pred_with_string() {
    run_test_true("(string? \"hello\")");
}

#[test]
fn test_string_pred_with_empty_string() {
    run_test_true("(string? \"\")");
}

#[test]
fn test_string_pred_with_string_with_spaces() {
    run_test_true("(string? \"hello world\")");
}

#[test]
fn test_string_pred_with_numeric_string() {
    run_test_true("(string? \"42\")");
}

#[test]
fn test_string_pred_with_nil() {
    run_test_false("(string? nil)");
}

#[test]
fn test_string_pred_with_int() {
    run_test_false("(string? 42)");
}

#[test]
fn test_string_pred_with_float() {
    run_test_false("(string? 3.14)");
}

#[test]
fn test_string_pred_with_true() {
    run_test_false("(string? true)");
}

#[test]
fn test_string_pred_with_false() {
    run_test_false("(string? false)");
}

#[test]
fn test_string_pred_with_keyword() {
    run_test_false("(string? :hello)");
}

#[test]
fn test_string_pred_with_function() {
    run_test_false("(string? (fn [x] x))");
}

// =============================================================================
// fn? Tests
// =============================================================================

#[test]
fn test_fn_pred_with_lambda() {
    run_test_true("(fn? (fn [x] x))");
}

#[test]
fn test_fn_pred_with_lambda_no_args() {
    run_test_true("(fn? (fn [] 42))");
}

#[test]
fn test_fn_pred_with_lambda_multiple_args() {
    run_test_true("(fn? (fn [a b c] (+ (+ a b) c)))");
}

#[test]
fn test_fn_pred_with_closure() {
    // This creates a closure that captures 'x'
    run_test_true("(let [x 10] (fn? (fn [y] (+ x y))))");
}

#[test]
fn test_fn_pred_with_nil() {
    run_test_false("(fn? nil)");
}

#[test]
fn test_fn_pred_with_int() {
    run_test_false("(fn? 42)");
}

#[test]
fn test_fn_pred_with_float() {
    run_test_false("(fn? 3.14)");
}

#[test]
fn test_fn_pred_with_string() {
    run_test_false("(fn? \"hello\")");
}

#[test]
fn test_fn_pred_with_true() {
    run_test_false("(fn? true)");
}

#[test]
fn test_fn_pred_with_false() {
    run_test_false("(fn? false)");
}

#[test]
fn test_fn_pred_with_keyword() {
    run_test_false("(fn? :fn)");
}

// =============================================================================
// identical? Tests
// =============================================================================

#[test]
fn test_identical_nil_nil() {
    run_test_true("(identical? nil nil)");
}

#[test]
fn test_identical_true_true() {
    run_test_true("(identical? true true)");
}

#[test]
fn test_identical_false_false() {
    run_test_true("(identical? false false)");
}

#[test]
fn test_identical_same_int() {
    run_test_true("(identical? 42 42)");
}

#[test]
fn test_identical_same_zero() {
    run_test_true("(identical? 0 0)");
}

#[test]
fn test_identical_same_negative_int() {
    run_test_true("(identical? -5 -5)");
}

#[test]
fn test_identical_different_ints() {
    run_test_false("(identical? 1 2)");
}

#[test]
fn test_identical_zero_and_one() {
    run_test_false("(identical? 0 1)");
}

#[test]
fn test_identical_same_keyword() {
    // Keywords are interned, so same keyword should be identical
    run_test_true("(identical? :foo :foo)");
}

#[test]
fn test_identical_different_keywords() {
    run_test_false("(identical? :foo :bar)");
}

#[test]
fn test_identical_nil_and_false() {
    // nil and false are different values
    run_test_false("(identical? nil false)");
}

#[test]
fn test_identical_nil_and_zero() {
    run_test_false("(identical? nil 0)");
}

#[test]
fn test_identical_true_and_false() {
    run_test_false("(identical? true false)");
}

#[test]
fn test_identical_int_and_float() {
    // 1 (integer) and 1.0 (float) have different representations
    run_test_false("(identical? 1 1.0)");
}

#[test]
fn test_identical_string_literals() {
    // String literals may or may not be interned - depends on implementation
    // For safety, we test that two string literals compile without error
    let code = "(identical? \"hello\" \"hello\")";
    let _ = run_and_get_tagged(code); // Just verify it runs
}

#[test]
fn test_identical_different_strings() {
    run_test_false("(identical? \"hello\" \"world\")");
}

// =============================================================================
// Combined/Edge Case Tests
// =============================================================================

#[test]
fn test_nil_pred_in_if() {
    // Test that nil? works correctly in conditional context
    let result = run_and_get_tagged("(if (nil? nil) 1 2)");
    assert_eq!(result >> 3, 1, "nil? nil should be truthy");
}

#[test]
fn test_nil_pred_false_in_if() {
    let result = run_and_get_tagged("(if (nil? 0) 1 2)");
    assert_eq!(result >> 3, 2, "nil? 0 should be falsy");
}

#[test]
fn test_number_pred_in_if() {
    let result = run_and_get_tagged("(if (number? 42) 1 2)");
    assert_eq!(result >> 3, 1, "number? 42 should be truthy");
}

#[test]
fn test_string_pred_in_if() {
    let result = run_and_get_tagged("(if (string? \"hi\") 1 2)");
    assert_eq!(result >> 3, 1, "string? \"hi\" should be truthy");
}

#[test]
fn test_fn_pred_in_if() {
    let result = run_and_get_tagged("(if (fn? (fn [] 1)) 1 2)");
    assert_eq!(result >> 3, 1, "fn? (fn [] 1) should be truthy");
}

#[test]
fn test_identical_in_if() {
    let result = run_and_get_tagged("(if (identical? :a :a) 1 2)");
    assert_eq!(result >> 3, 1, "identical? :a :a should be truthy");
}

#[test]
fn test_chained_predicates() {
    // Test combining predicates in expressions
    let result = run_and_get_tagged("(if (nil? nil) (if (number? 1) 10 20) 30)");
    assert_eq!(result >> 3, 10);
}

#[test]
fn test_predicate_with_let_binding() {
    run_test_true("(let [x nil] (nil? x))");
    run_test_true("(let [x 42] (number? x))");
    run_test_true("(let [x \"hi\"] (string? x))");
}

#[test]
fn test_predicate_with_arithmetic_result() {
    run_test_true("(number? (+ 1 2))");
    run_test_true("(number? (* 3 4))");
    run_test_true("(number? (+ 1.0 2.0))");
}

#[test]
fn test_identical_with_let_binding() {
    run_test_true("(let [x 42] (identical? x 42))");
    run_test_true("(let [x nil] (identical? x nil))");
    run_test_true("(let [x :foo] (identical? x :foo))");
}

#[test]
fn test_identical_same_variable() {
    run_test_true("(let [x 42] (identical? x x))");
    run_test_true("(let [f (fn [x] x)] (identical? f f))");
}
