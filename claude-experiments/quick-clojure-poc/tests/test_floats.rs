// Test suite for float support
//
// Test Coverage:
// ✅ Float literals
// ✅ Float arithmetic (+, -, *, /)
// ✅ Mixed int/float operations with auto-promotion
// ✅ Integer operations still work correctly
// ✅ Special float values (very small, very large)

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
    let result_reg = compiler.compile(&ast).expect(&format!("Compiler failed for: {}", code));
    let instructions = compiler.take_instructions();

    let mut codegen = arm_codegen::Arm64CodeGen::new();
    codegen.compile(&instructions, &result_reg, 0).expect(&format!("Codegen failed for: {}", code));
    codegen.execute().expect(&format!("Execute failed for: {}", code))
}

/// Helper to get the tag from a tagged value
fn get_tag(tagged: i64) -> i64 {
    tagged & 0b111
}

/// Helper to read a float from a tagged float pointer
fn read_float_from_tagged(tagged: i64, runtime: &gc_runtime::GCRuntime) -> f64 {
    runtime.read_float(tagged as usize)
}

/// Helper function to run a test case expecting an integer result
fn run_test_int(code: &str, expected: i64) {
    let tagged_result = run_and_get_tagged(code);
    let tag = get_tag(tagged_result);
    assert_eq!(tag, 0, "Expected integer (tag 0), got tag {} for: {}", tag, code);
    let result = tagged_result >> 3;
    assert_eq!(result, expected, "Integer test failed for: {}", code);
}

/// Helper function to run a test case expecting a float result
fn run_test_float(code: &str, expected: f64) {
    let val = reader::read(code).expect(&format!("Failed to read: {}", code));
    let ast = clojure_ast::analyze(&val).expect(&format!("Failed to analyze: {}", code));

    let runtime = Arc::new(UnsafeCell::new(gc_runtime::GCRuntime::new()));
    trampoline::set_runtime(runtime.clone());
    let mut compiler = compiler::Compiler::new(runtime.clone());
    let result_reg = compiler.compile(&ast).expect(&format!("Compiler failed for: {}", code));
    let instructions = compiler.take_instructions();

    let mut codegen = arm_codegen::Arm64CodeGen::new();
    codegen.compile(&instructions, &result_reg, 0).expect(&format!("Codegen failed for: {}", code));
    let tagged_result = codegen.execute().expect(&format!("Execute failed for: {}", code));

    let tag = get_tag(tagged_result);
    assert_eq!(tag, 1, "Expected float (tag 1), got tag {} for: {}", tag, code);

    // Read the float from the heap
    let rt = unsafe { &*runtime.get() };
    let result = read_float_from_tagged(tagged_result, rt);

    // Use approximate comparison for floats
    let diff = (result - expected).abs();
    let tolerance = 1e-10;
    assert!(diff < tolerance, "Float test failed for: {} - expected {}, got {} (diff: {})",
            code, expected, result, diff);
}

/// Helper function to run a test case expecting a float result with custom tolerance
fn run_test_float_approx(code: &str, expected: f64, tolerance: f64) {
    let val = reader::read(code).expect(&format!("Failed to read: {}", code));
    let ast = clojure_ast::analyze(&val).expect(&format!("Failed to analyze: {}", code));

    let runtime = Arc::new(UnsafeCell::new(gc_runtime::GCRuntime::new()));
    trampoline::set_runtime(runtime.clone());
    let mut compiler = compiler::Compiler::new(runtime.clone());
    let result_reg = compiler.compile(&ast).expect(&format!("Compiler failed for: {}", code));
    let instructions = compiler.take_instructions();

    let mut codegen = arm_codegen::Arm64CodeGen::new();
    codegen.compile(&instructions, &result_reg, 0).expect(&format!("Codegen failed for: {}", code));
    let tagged_result = codegen.execute().expect(&format!("Execute failed for: {}", code));

    let tag = get_tag(tagged_result);
    assert_eq!(tag, 1, "Expected float (tag 1), got tag {} for: {}", tag, code);

    let rt = unsafe { &*runtime.get() };
    let result = read_float_from_tagged(tagged_result, rt);

    let diff = (result - expected).abs();
    assert!(diff < tolerance, "Float test failed for: {} - expected {}, got {} (diff: {}, tolerance: {})",
            code, expected, result, diff, tolerance);
}

// =============================================================================
// Float Literal Tests
// =============================================================================

#[test]
fn test_float_literals() {
    run_test_float("3.14", 3.14);
    run_test_float("0.5", 0.5);
    run_test_float("2.718", 2.718);
    run_test_float("0.0", 0.0);
    run_test_float("123.456", 123.456);
}

#[test]
fn test_float_literals_small() {
    run_test_float("0.001", 0.001);
    run_test_float("0.0001", 0.0001);
    run_test_float_approx("0.00001", 0.00001, 1e-15);
}

#[test]
fn test_float_literals_large() {
    run_test_float("1000.0", 1000.0);
    run_test_float("999999.999", 999999.999);
    run_test_float_approx("1234567.89", 1234567.89, 1e-5);
}

// =============================================================================
// Float Arithmetic Tests
// =============================================================================

#[test]
fn test_float_addition() {
    run_test_float("(+ 1.0 2.0)", 3.0);
    run_test_float("(+ 0.1 0.2)", 0.30000000000000004); // Classic floating point
    run_test_float("(+ 3.14 2.86)", 6.0);
    run_test_float("(+ 0.5 0.5)", 1.0);
}

#[test]
fn test_float_subtraction() {
    run_test_float("(- 5.0 3.0)", 2.0);
    run_test_float("(- 3.14 1.14)", 2.0);
    run_test_float("(- 10.5 0.5)", 10.0);
    run_test_float("(- 1.0 1.0)", 0.0);
}

#[test]
fn test_float_multiplication() {
    run_test_float("(* 2.0 3.0)", 6.0);
    run_test_float("(* 2.5 4.0)", 10.0);
    run_test_float("(* 0.5 0.5)", 0.25);
    run_test_float("(* 3.14 2.0)", 6.28);
}

#[test]
fn test_float_division() {
    run_test_float("(/ 6.0 2.0)", 3.0);
    run_test_float("(/ 5.0 2.0)", 2.5);
    run_test_float("(/ 10.0 4.0)", 2.5);
    run_test_float("(/ 1.0 3.0)", 0.3333333333333333);
}

#[test]
fn test_float_nested_arithmetic() {
    run_test_float("(+ (* 2.0 3.0) 4.0)", 10.0);
    run_test_float("(* (+ 1.0 2.0) (+ 3.0 4.0))", 21.0);
    run_test_float("(- (* 10.0 5.0) 20.0)", 30.0);
    run_test_float("(/ (+ 10.0 10.0) 4.0)", 5.0);
}

// =============================================================================
// Mixed Int/Float Operations (Auto-promotion)
// =============================================================================

#[test]
fn test_mixed_addition_int_first() {
    run_test_float("(+ 1 2.5)", 3.5);
    run_test_float("(+ 10 0.5)", 10.5);
    run_test_float("(+ 0 3.14)", 3.14);
}

#[test]
fn test_mixed_addition_float_first() {
    run_test_float("(+ 2.5 1)", 3.5);
    run_test_float("(+ 0.5 10)", 10.5);
    run_test_float("(+ 3.14 0)", 3.14);
}

#[test]
fn test_mixed_subtraction() {
    run_test_float("(- 5 2.5)", 2.5);
    run_test_float("(- 10.5 5)", 5.5);
    run_test_float("(- 3.14 3)", 0.14000000000000012); // Floating point imprecision
}

#[test]
fn test_mixed_multiplication() {
    run_test_float("(* 3 2.5)", 7.5);
    run_test_float("(* 2.5 3)", 7.5);
    run_test_float("(* 10 0.1)", 1.0);
}

#[test]
fn test_mixed_division() {
    run_test_float("(/ 5.0 2)", 2.5);
    run_test_float("(/ 10 4.0)", 2.5);
    run_test_float("(/ 7 2.0)", 3.5);
}

#[test]
fn test_mixed_nested() {
    run_test_float("(+ (* 2 3.0) 4)", 10.0);
    run_test_float("(* (+ 1 2.0) (+ 3 4.0))", 21.0);
    run_test_float("(/ (+ 10 10.0) 4)", 5.0);
}

// =============================================================================
// Integer Operations Still Work
// =============================================================================

#[test]
fn test_int_operations_unchanged() {
    run_test_int("42", 42);
    run_test_int("(+ 1 2)", 3);
    run_test_int("(- 10 3)", 7);
    run_test_int("(* 6 7)", 42);
    run_test_int("(/ 10 3)", 3); // Integer division
}

#[test]
fn test_int_nested_unchanged() {
    run_test_int("(+ (* 2 3) 4)", 10);
    run_test_int("(* (+ 1 2) (+ 3 4))", 21);
    run_test_int("(- (* 10 5) 20)", 30);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_float_zero_operations() {
    run_test_float("(+ 0.0 0.0)", 0.0);
    run_test_float("(* 0.0 100.0)", 0.0);
    run_test_float("(- 0.0 0.0)", 0.0);
}

#[test]
fn test_float_identity_operations() {
    run_test_float("(+ 3.14 0.0)", 3.14);
    run_test_float("(* 3.14 1.0)", 3.14);
    run_test_float("(- 3.14 0.0)", 3.14);
    run_test_float("(/ 3.14 1.0)", 3.14);
}

#[test]
fn test_float_precision() {
    // Test that we maintain f64 precision
    run_test_float_approx("(+ 1.0000000001 0.0)", 1.0000000001, 1e-9);
    run_test_float_approx("(* 1.23456789 1.0)", 1.23456789, 1e-9);
}

// =============================================================================
// Negative Float Tests
// =============================================================================

#[test]
fn test_negative_float_literals() {
    run_test_float("-3.14", -3.14);
    run_test_float("-0.5", -0.5);
    run_test_float("-1000.0", -1000.0);
}

#[test]
fn test_negative_float_arithmetic() {
    run_test_float("(+ -1.0 -2.0)", -3.0);
    run_test_float("(- -5.0 -3.0)", -2.0);
    run_test_float("(* -2.0 3.0)", -6.0);
    run_test_float("(* -2.0 -3.0)", 6.0);
    run_test_float("(/ -6.0 2.0)", -3.0);
    run_test_float("(/ 6.0 -2.0)", -3.0);
    run_test_float("(/ -6.0 -2.0)", 3.0);
}

#[test]
fn test_mixed_negative_operations() {
    run_test_float("(+ -1 2.5)", 1.5);
    run_test_float("(+ 1.5 -2)", -0.5);
    run_test_float("(* -3 2.0)", -6.0);
    run_test_float("(/ -10 2.0)", -5.0);
}

// =============================================================================
// Type Safety Tests (Non-numeric operands should not crash)
// =============================================================================

/// Helper to test that non-numeric operands don't crash
/// (they currently return 0 as a safe fallback)
fn run_test_no_crash(code: &str) {
    let val = reader::read(code).expect(&format!("Failed to read: {}", code));
    let ast = clojure_ast::analyze(&val).expect(&format!("Failed to analyze: {}", code));

    let runtime = Arc::new(UnsafeCell::new(gc_runtime::GCRuntime::new()));
    trampoline::set_runtime(runtime.clone());
    let mut compiler = compiler::Compiler::new(runtime.clone());
    let result_reg = compiler.compile(&ast).expect(&format!("Compiler failed for: {}", code));
    let instructions = compiler.take_instructions();

    let mut codegen = arm_codegen::Arm64CodeGen::new();
    codegen.compile(&instructions, &result_reg, 0).expect(&format!("Codegen failed for: {}", code));
    // Just verify it doesn't crash - any result is acceptable
    let _ = codegen.execute().expect(&format!("Execute should not crash for: {}", code));
}

#[test]
fn test_bool_in_arithmetic_no_crash() {
    // These should not crash (currently return safe fallback values)
    run_test_no_crash("(+ true 1)");
    run_test_no_crash("(+ 1 true)");
    run_test_no_crash("(+ true false)");
    run_test_no_crash("(* true 2)");
}

#[test]
fn test_nil_in_arithmetic_no_crash() {
    // These should not crash (currently return safe fallback values)
    run_test_no_crash("(+ nil 1)");
    run_test_no_crash("(+ 1 nil)");
    run_test_no_crash("(* nil 2)");
}

#[test]
fn test_string_in_arithmetic_no_crash() {
    // These should not crash (currently return safe fallback values)
    run_test_no_crash("(+ \"hello\" 1)");
    run_test_no_crash("(+ 1 \"world\")");
    run_test_no_crash("(* \"test\" 2)");
}
