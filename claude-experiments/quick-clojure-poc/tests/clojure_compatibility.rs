/// Integration tests that verify our implementation matches Clojure's behavior
///
/// These tests run expressions through both our JIT compiler and Clojure,
/// then compare the outputs to ensure compatibility.

use std::process::Command;
use std::sync::OnceLock;
use std::path::PathBuf;

static BINARY_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Ensure the release binary is built exactly once before any tests run
fn get_binary_path() -> &'static PathBuf {
    BINARY_PATH.get_or_init(|| {
        // Build release binary once
        let status = Command::new("cargo")
            .args(&["build", "--release", "--quiet"])
            .status()
            .expect("Failed to build release binary");

        assert!(status.success(), "Failed to build release binary");

        // Get the path to the binary
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .unwrap_or_else(|_| ".".to_string());
        PathBuf::from(manifest_dir).join("target/release/quick-clojure-poc")
    })
}

/// Helper to run an expression through our implementation
fn run_our_impl(expr: &str) -> String {
    let binary_path = get_binary_path();

    let output = Command::new(&binary_path)
        .args(&["-e", expr])
        .output()
        .expect("Failed to execute our implementation");

    // Only get stdout, ignore stderr (which has DEBUG output)
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

/// Helper to run an expression through Clojure
fn run_clojure(expr: &str) -> String {
    // Create a clean directory to run clj from (avoid picking up src/clojure/core.clj)
    let clean_dir = std::path::Path::new("/tmp/clojure-running");
    // Remove and recreate to ensure it's clean
    let _ = std::fs::remove_dir_all(clean_dir);
    std::fs::create_dir_all(clean_dir).expect("Failed to create clean directory");

    let output = Command::new("clj")
        .args(&["-e", expr])
        .current_dir(clean_dir)
        .output()
        .expect("Failed to execute Clojure - is it installed?");

    let result = String::from_utf8_lossy(&output.stdout).trim().to_string();

    // Clojure prints nothing for nil, we represent that as "nil" for comparison
    if result.is_empty() {
        "nil".to_string()
    } else {
        result
    }
}

/// Helper to compare outputs
fn assert_matches_clojure(expr: &str) {
    let our_output = run_our_impl(expr);
    let clojure_output = run_clojure(expr);

    // Treat empty output as "nil"
    let our_normalized = if our_output.is_empty() { "nil" } else { &our_output };

    assert_eq!(
        our_normalized, clojure_output,
        "\nExpression: {}\nOur output: {:?}\nClojure output: {:?}",
        expr, our_output, clojure_output
    );
}

// ============================================================================
// Literal Tests
// ============================================================================

#[test]
fn test_nil_literal() {
    assert_matches_clojure("nil");
}

#[test]
fn test_true_literal() {
    assert_matches_clojure("true");
}

#[test]
fn test_false_literal() {
    assert_matches_clojure("false");
}

#[test]
fn test_integer_zero() {
    assert_matches_clojure("0");
}

#[test]
fn test_positive_integer() {
    assert_matches_clojure("42");
}

#[test]
fn test_negative_integer() {
    assert_matches_clojure("-5");
}

// ============================================================================
// Equality Tests
// ============================================================================

#[test]
fn test_nil_not_equal_to_zero() {
    assert_matches_clojure("(= nil 0)");
}

#[test]
fn test_nil_not_equal_to_false() {
    assert_matches_clojure("(= nil false)");
}

#[test]
fn test_false_not_equal_to_zero() {
    assert_matches_clojure("(= false 0)");
}

#[test]
fn test_true_not_equal_to_false() {
    assert_matches_clojure("(= true false)");
}

#[test]
fn test_integers_equal() {
    assert_matches_clojure("(= 5 5)");
}

#[test]
fn test_integers_not_equal() {
    assert_matches_clojure("(= 5 3)");
}

// ============================================================================
// Comparison Tests
// ============================================================================

#[test]
fn test_less_than_true() {
    assert_matches_clojure("(< 1 2)");
}

#[test]
fn test_less_than_false() {
    assert_matches_clojure("(< 2 1)");
}

#[test]
fn test_greater_than_true() {
    assert_matches_clojure("(> 2 1)");
}

#[test]
fn test_greater_than_false() {
    assert_matches_clojure("(> 1 2)");
}

// ============================================================================
// Arithmetic Tests
// ============================================================================

#[test]
fn test_addition() {
    assert_matches_clojure("(+ 1 2)");
}

#[test]
fn test_addition_larger() {
    assert_matches_clojure("(+ 10 20)");
}

#[test]
fn test_multiplication() {
    assert_matches_clojure("(* 2 3)");
}

#[test]
fn test_multiplication_larger() {
    assert_matches_clojure("(* 7 8)");
}

#[test]
fn test_subtraction() {
    assert_matches_clojure("(- 5 3)");
}

// ============================================================================
// Let Expression Tests
// ============================================================================

#[test]
fn test_empty_let_body() {
    assert_matches_clojure("(let [x 2])");
}

#[test]
fn test_let_with_body() {
    assert_matches_clojure("(let [x 5] x)");
}

#[test]
fn test_let_with_arithmetic() {
    assert_matches_clojure("(let [x 2 y 3] (+ x y))");
}

#[test]
fn test_let_nested() {
    assert_matches_clojure("(let [x 1] (let [y 2] (+ x y)))");
}

#[test]
fn test_let_shadowing() {
    assert_matches_clojure("(let [x 1] (let [x 2] x))");
}

// ============================================================================
// Boolean Logic Tests (TODO: Not yet implemented)
// ============================================================================

// #[test]
// fn test_and_true_true() {
//     assert_matches_clojure("(and true true)");
// }

// #[test]
// fn test_and_true_false() {
//     assert_matches_clojure("(and true false)");
// }

// #[test]
// fn test_or_false_false() {
//     assert_matches_clojure("(or false false)");
// }

// #[test]
// fn test_or_false_true() {
//     assert_matches_clojure("(or false true)");
// }

// ============================================================================
// If Expression Tests (TODO: Not yet implemented)
// ============================================================================

// #[test]
// fn test_if_true_branch() {
//     assert_matches_clojure("(if true 1 2)");
// }

// #[test]
// fn test_if_false_branch() {
//     assert_matches_clojure("(if false 1 2)");
// }

// #[test]
// fn test_if_nil_is_falsey() {
//     assert_matches_clojure("(if nil 1 2)");
// }

// #[test]
// fn test_if_zero_is_truthy() {
//     assert_matches_clojure("(if 0 1 2)");
// }

// ============================================================================
// Function/Closure Tests
// ============================================================================

#[test]
fn test_simple_fn_call() {
    assert_matches_clojure("((fn [x] x) 5)");
}

#[test]
fn test_fn_with_arithmetic() {
    assert_matches_clojure("((fn [x] (+ x 1)) 5)");
}

#[test]
fn test_fn_two_args() {
    assert_matches_clojure("((fn [x y] (+ x y)) 3 4)");
}

#[test]
fn test_fn_in_let() {
    assert_matches_clojure("(let [f (fn [x] (+ x 1))] (f 5))");
}

#[test]
fn test_multiple_fn_calls() {
    assert_matches_clojure("(let [add1 (fn [x] (+ x 1)) mul2 (fn [x] (* x 2))] (+ (add1 5) (mul2 3)))");
}

#[test]
fn test_closure_captures_variable() {
    assert_matches_clojure("(let [x 10] ((fn [y] (+ x y)) 5))");
}

#[test]
fn test_closure_captures_multiple() {
    assert_matches_clojure("(let [a 1 b 2] ((fn [c] (+ a (+ b c))) 3))");
}

#[test]
fn test_nested_fn_calls() {
    assert_matches_clojure("((fn [x] ((fn [y] (+ x y)) 2)) 3)");
}

#[test]
fn test_fn_returning_fn() {
    assert_matches_clojure("(((fn [x] (fn [y] (+ x y))) 10) 5)");
}

// ============================================================================
// Deep Nesting Tests - Stress test register allocation
// ============================================================================

#[test]
fn test_deeply_nested_addition() {
    // (+ (+ (+ (+ (+ 1 2) 3) 4) 5) 6) = 21
    assert_matches_clojure("(+ (+ (+ (+ (+ 1 2) 3) 4) 5) 6)");
}

#[test]
fn test_deeply_nested_let() {
    // Nested lets with many variables in scope
    assert_matches_clojure("(let [a 1] (let [b 2] (let [c 3] (let [d 4] (let [e 5] (+ a (+ b (+ c (+ d e)))))))))");
}

#[test]
fn test_many_variables_in_let() {
    // Many bindings in single let
    assert_matches_clojure("(let [a 1 b 2 c 3 d 4 e 5 f 6 g 7 h 8] (+ a (+ b (+ c (+ d (+ e (+ f (+ g h))))))))");
}

#[test]
fn test_nested_fn_many_closures() {
    // Multiple levels of closures capturing variables
    assert_matches_clojure("(let [a 1] ((fn [b] (let [c 2] ((fn [d] (+ a (+ b (+ c d)))) 4))) 3))");
}

#[test]
fn test_deeply_nested_arithmetic() {
    // Complex nested arithmetic: ((1 + 2) * 3) + ((4 - 1) * 2) = 9 + 6 = 15
    assert_matches_clojure("(+ (* (+ 1 2) 3) (* (- 4 1) 2))");
}

#[test]
fn test_ten_variable_let() {
    // 10 variables in scope simultaneously
    assert_matches_clojure("(let [v1 1 v2 2 v3 3 v4 4 v5 5 v6 6 v7 7 v8 8 v9 9 v10 10] (+ v1 (+ v2 (+ v3 (+ v4 (+ v5 (+ v6 (+ v7 (+ v8 (+ v9 v10))))))))))");
}

#[test]
fn test_nested_closures_deep() {
    // 5-level deep closure nesting
    assert_matches_clojure("((((((fn [a] (fn [b] (fn [c] (fn [d] (fn [e] (+ a (+ b (+ c (+ d e))))))))) 1) 2) 3) 4) 5)");
}

#[test]
fn test_mixed_nested_operations() {
    // Mix of let, fn, and arithmetic deeply nested
    assert_matches_clojure("(let [x 10] ((fn [y] (let [z 3] (+ x (+ y (* z 2))))) 5))");
}

#[test]
fn test_many_fn_args() {
    // Function with many arguments
    assert_matches_clojure("((fn [a b c d e f] (+ a (+ b (+ c (+ d (+ e f)))))) 1 2 3 4 5 6)");
}

#[test]
fn test_chained_closures() {
    // Chain of closures each capturing previous value
    assert_matches_clojure("(let [f1 (fn [x] (+ x 1)) f2 (fn [x] (+ (f1 x) 2)) f3 (fn [x] (+ (f2 x) 3))] (f3 10))")
}
