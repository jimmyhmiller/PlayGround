/// Integration tests that verify our implementation matches Clojure's behavior
///
/// These tests run expressions through both our JIT compiler and Clojure,
/// then compare the outputs to ensure compatibility.

use std::process::Command;
use std::fs;
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

    // Use tempfile to get a unique path, but persist the file
    // This ensures the file exists during the entire execution
    let temp_dir = tempfile::tempdir()
        .expect("Failed to create temp dir");

    let temp_path = temp_dir.path().join("test.clj");

    fs::write(&temp_path, expr)
        .expect("Failed to write temp file");

    let output = Command::new(&binary_path)
        .arg(temp_path.to_str().unwrap())
        .output()
        .expect("Failed to execute our implementation");

    // temp_dir (and its contents) are automatically deleted when it goes out of scope

    // Only get stdout, ignore stderr (which has DEBUG output)
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

/// Helper to run an expression through Clojure
fn run_clojure(expr: &str) -> String {
    let output = Command::new("clj")
        .args(&["-e", expr])
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
