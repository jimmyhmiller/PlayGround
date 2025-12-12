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

// ============================================================================
// If Expression Tests
// ============================================================================

#[test]
fn test_if_true_branch() {
    assert_matches_clojure("(if true 1 2)");
}

#[test]
fn test_if_false_branch() {
    assert_matches_clojure("(if false 1 2)");
}

#[test]
fn test_if_nil_is_falsey() {
    assert_matches_clojure("(if nil 1 2)");
}

#[test]
fn test_if_zero_is_truthy() {
    assert_matches_clojure("(if 0 1 2)");
}

#[test]
fn test_if_empty_string_is_truthy() {
    // In Clojure, empty string is truthy (unlike JavaScript)
    assert_matches_clojure("(if \"\" 1 2)");
}

#[test]
fn test_if_without_else() {
    // (if condition then) - returns nil when condition is false
    assert_matches_clojure("(if false 1)");
}

#[test]
fn test_if_nested() {
    assert_matches_clojure("(if true (if false 1 2) 3)");
}

// ============================================================================
// Boolean Logic Tests (TODO: and/or/not not yet implemented)
// ============================================================================

#[test]
#[ignore = "and macro not implemented"]
fn test_and_true_true() {
    assert_matches_clojure("(and true true)");
}

#[test]
#[ignore = "and macro not implemented"]
fn test_and_true_false() {
    assert_matches_clojure("(and true false)");
}

#[test]
#[ignore = "and macro not implemented"]
fn test_and_false_short_circuits() {
    // and returns the first falsey value
    assert_matches_clojure("(and false 1)");
}

#[test]
#[ignore = "and macro not implemented"]
fn test_and_returns_last_truthy() {
    // and returns the last truthy value when all are truthy
    assert_matches_clojure("(and 1 2 3)");
}

#[test]
#[ignore = "and macro not implemented"]
fn test_and_nil_short_circuits() {
    assert_matches_clojure("(and nil 1)");
}

#[test]
#[ignore = "or macro not implemented"]
fn test_or_false_false() {
    assert_matches_clojure("(or false false)");
}

#[test]
#[ignore = "or macro not implemented"]
fn test_or_false_true() {
    assert_matches_clojure("(or false true)");
}

#[test]
#[ignore = "or macro not implemented"]
fn test_or_returns_first_truthy() {
    // or returns the first truthy value
    assert_matches_clojure("(or nil false 3 4)");
}

#[test]
#[ignore = "or macro not implemented"]
fn test_or_returns_last_when_all_falsey() {
    // or returns the last value when all are falsey
    assert_matches_clojure("(or nil false)");
}

#[test]
#[ignore = "not function not implemented"]
fn test_not_true() {
    assert_matches_clojure("(not true)");
}

#[test]
#[ignore = "not function not implemented"]
fn test_not_false() {
    assert_matches_clojure("(not false)");
}

#[test]
#[ignore = "not function not implemented"]
fn test_not_nil() {
    assert_matches_clojure("(not nil)");
}

#[test]
#[ignore = "not function not implemented"]
fn test_not_zero() {
    // 0 is truthy in Clojure, so (not 0) = false
    assert_matches_clojure("(not 0)");
}

// ============================================================================
// Loop/Recur Tests
// ============================================================================

#[test]
fn test_simple_loop() {
    // Simple countdown
    assert_matches_clojure("(loop [x 5] (if (= x 0) x (recur (- x 1))))");
}

#[test]
fn test_loop_accumulator() {
    // Sum 1 to 5 = 15
    assert_matches_clojure("(loop [i 1 acc 0] (if (> i 5) acc (recur (+ i 1) (+ acc i))))");
}

#[test]
fn test_loop_factorial() {
    // 5! = 120
    assert_matches_clojure("(loop [n 5 acc 1] (if (= n 0) acc (recur (- n 1) (* acc n))))");
}

#[test]
fn test_loop_fibonacci() {
    // 10th fibonacci (0-indexed): fib(10) = 55
    assert_matches_clojure("(loop [n 10 a 0 b 1] (if (= n 0) a (recur (- n 1) b (+ a b))))");
}

#[test]
fn test_loop_with_let() {
    // Loop with let inside
    assert_matches_clojure("(loop [i 0 acc 0] (if (>= i 5) acc (let [sq (* i i)] (recur (+ i 1) (+ acc sq)))))");
}

#[test]
fn test_nested_loop() {
    // Simple nested loop: outer 0-2, inner 0-2, count = 9
    assert_matches_clojure("(loop [outer 0 count 0] (if (>= outer 3) count (recur (+ outer 1) (loop [inner 0 c count] (if (>= inner 3) c (recur (+ inner 1) (+ c 1)))))))");
}

// ============================================================================
// do Expression Tests
// ============================================================================

#[test]
fn test_do_returns_last() {
    assert_matches_clojure("(do 1 2 3)");
}

#[test]
fn test_do_single_value() {
    assert_matches_clojure("(do 42)");
}

#[test]
fn test_do_with_side_effects() {
    // println returns nil, so do returns nil
    assert_matches_clojure("(do (+ 1 2) (+ 3 4))");
}

// ============================================================================
// Integer Division Tests (TODO: quot/rem/mod not yet implemented)
// ============================================================================

#[test]
#[ignore = "quot not implemented"]
fn test_quot_positive() {
    assert_matches_clojure("(quot 10 3)");
}

#[test]
#[ignore = "quot not implemented"]
fn test_quot_negative() {
    assert_matches_clojure("(quot -10 3)");
}

#[test]
#[ignore = "rem not implemented"]
fn test_rem_positive() {
    assert_matches_clojure("(rem 10 3)");
}

#[test]
#[ignore = "rem not implemented"]
fn test_rem_negative() {
    assert_matches_clojure("(rem -10 3)");
}

#[test]
#[ignore = "mod not implemented"]
fn test_mod_positive() {
    assert_matches_clojure("(mod 10 3)");
}

#[test]
#[ignore = "mod not implemented"]
fn test_mod_negative() {
    // Note: mod and rem differ for negative numbers
    assert_matches_clojure("(mod -10 3)");
}

// ============================================================================
// Bit Operation Tests
// ============================================================================

#[test]
fn test_bit_and() {
    assert_matches_clojure("(bit-and 5 3)");
}

#[test]
fn test_bit_or() {
    assert_matches_clojure("(bit-or 5 3)");
}

#[test]
fn test_bit_xor() {
    assert_matches_clojure("(bit-xor 5 3)");
}

#[test]
fn test_bit_not() {
    assert_matches_clojure("(bit-not 5)");
}

#[test]
fn test_bit_shift_left() {
    assert_matches_clojure("(bit-shift-left 1 4)");
}

#[test]
fn test_bit_shift_right() {
    assert_matches_clojure("(bit-shift-right 16 2)");
}

// ============================================================================
// Type Predicate Tests
// ============================================================================

#[test]
fn test_nil_question() {
    assert_matches_clojure("(nil? nil)");
}

#[test]
fn test_nil_question_false() {
    assert_matches_clojure("(nil? 0)");
}

#[test]
#[ignore = "some? not implemented"]
fn test_some_question() {
    assert_matches_clojure("(some? nil)");
}

#[test]
#[ignore = "some? not implemented"]
fn test_some_question_true() {
    assert_matches_clojure("(some? 0)");
}

#[test]
#[ignore = "true? not implemented"]
fn test_true_question() {
    assert_matches_clojure("(true? true)");
}

#[test]
#[ignore = "true? not implemented"]
fn test_true_question_false() {
    assert_matches_clojure("(true? 1)");
}

#[test]
#[ignore = "false? not implemented"]
fn test_false_question() {
    assert_matches_clojure("(false? false)");
}

#[test]
#[ignore = "false? not implemented"]
fn test_false_question_not_nil() {
    assert_matches_clojure("(false? nil)");
}

#[test]
fn test_number_question() {
    assert_matches_clojure("(number? 42)");
}

#[test]
fn test_number_question_negative() {
    assert_matches_clojure("(number? -5)");
}

#[test]
fn test_number_question_false() {
    assert_matches_clojure("(number? nil)");
}

#[test]
#[ignore = "integer? not implemented"]
fn test_integer_question() {
    assert_matches_clojure("(integer? 42)");
}

#[test]
fn test_fn_question() {
    assert_matches_clojure("(fn? (fn [x] x))");
}

#[test]
fn test_fn_question_false() {
    assert_matches_clojure("(fn? 42)");
}

// ============================================================================
// inc/dec Tests (TODO: inc/dec not yet implemented)
// ============================================================================

#[test]
#[ignore = "inc not implemented"]
fn test_inc() {
    assert_matches_clojure("(inc 5)");
}

#[test]
#[ignore = "inc not implemented"]
fn test_inc_zero() {
    assert_matches_clojure("(inc 0)");
}

#[test]
#[ignore = "inc not implemented"]
fn test_inc_negative() {
    assert_matches_clojure("(inc -1)");
}

#[test]
#[ignore = "dec not implemented"]
fn test_dec() {
    assert_matches_clojure("(dec 5)");
}

#[test]
#[ignore = "dec not implemented"]
fn test_dec_zero() {
    assert_matches_clojure("(dec 0)");
}

#[test]
#[ignore = "dec not implemented"]
fn test_dec_one() {
    assert_matches_clojure("(dec 1)");
}

// ============================================================================
// identity Tests (TODO: identity not yet implemented)
// ============================================================================

#[test]
#[ignore = "identity not implemented"]
fn test_identity_number() {
    assert_matches_clojure("(identity 42)");
}

#[test]
#[ignore = "identity not implemented"]
fn test_identity_nil() {
    assert_matches_clojure("(identity nil)");
}

#[test]
#[ignore = "identity not implemented"]
fn test_identity_true() {
    assert_matches_clojure("(identity true)");
}
