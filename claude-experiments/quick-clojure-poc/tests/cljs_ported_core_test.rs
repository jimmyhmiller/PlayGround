/// Tests ported from ClojureScript core_test.cljs
/// Source: https://github.com/clojure/clojurescript/blob/master/src/test/cljs/cljs/core_test.cljs
///
/// Each test is annotated with the original CLJS test name where applicable.
/// Tests that require features not yet implemented are marked #[ignore].
///
/// Known limitations of this implementation:
/// - No lazy sequences (iterate/repeat with 1 arg, cycle will hang)
/// - No destructuring in fn params or loop bindings
/// - No named fn self-reference
/// - No sorted-set, sorted-map
/// - No ex-info/ex-message/ex-data
/// - No regex
/// - No letfn
/// - No delay/deref
/// - No defonce (with re-def protection)
/// - No meta/with-meta
/// - No extend-type on nil

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

static BINARY_PATH: OnceLock<PathBuf> = OnceLock::new();

fn get_binary_path() -> &'static PathBuf {
    BINARY_PATH.get_or_init(|| {
        let status = Command::new("cargo")
            .args(&["build", "--release", "--quiet"])
            .status()
            .expect("Failed to build release binary");
        assert!(status.success(), "Failed to build release binary");
        let manifest_dir =
            std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(manifest_dir).join("target/release/quick-clojure-poc")
    })
}

/// Run code from a temp file, return (stdout, stderr, success)
fn run_code(code: &str) -> (String, String, bool) {
    let binary_path = get_binary_path();
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let temp_path = temp_dir.path().join("test.clj");
    fs::write(&temp_path, code).expect("Failed to write temp file");

    let output = Command::new(binary_path.as_os_str())
        .arg(temp_path.to_str().unwrap())
        .output()
        .expect("Failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    (stdout, stderr, output.status.success())
}

/// Evaluate expression via -e flag (single expression, returns result)
fn eval_expr(expr: &str) -> String {
    let binary_path = get_binary_path();
    let output = Command::new(binary_path.as_os_str())
        .arg("-e")
        .arg(expr)
        .output()
        .expect("Failed to execute");
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    assert!(
        output.status.success(),
        "\nExpression failed: {}\nStderr: {}",
        expr, stderr
    );
    stdout
}

/// Assert println output matches expected
fn assert_output(code: &str, expected: &str) {
    let (stdout, stderr, success) = run_code(code);
    assert!(
        success,
        "\nCode failed: {}\nStderr: {}",
        code, stderr
    );
    assert_eq!(
        stdout, expected,
        "\nCode: {}\nStderr: {}",
        code, stderr
    );
}

// ============================================================================
// ARITHMETIC
// ============================================================================

#[test]
fn cljs_arithmetic_addition_basic() {
    assert_eq!(eval_expr("(+ 1 2)"), "3");
}

#[test]
fn cljs_arithmetic_addition_variadic() {
    assert_eq!(eval_expr("(+ 1 2 3)"), "6");
    assert_eq!(eval_expr("(+ 1 2 3 4 5)"), "15");
}

#[test]
fn cljs_arithmetic_addition_many() {
    assert_eq!(eval_expr("(+ 1 2 3 4 5 6 7 8 9 10)"), "55");
}

#[test]
fn cljs_arithmetic_addition_zero_arity() {
    assert_eq!(eval_expr("(+)"), "0");
}

#[test]
fn cljs_arithmetic_addition_unary() {
    assert_eq!(eval_expr("(+ 5)"), "5");
}

#[test]
fn cljs_arithmetic_addition_float() {
    assert_eq!(eval_expr("(+ 1.5 2.5)"), "4");
}

#[test]
fn cljs_arithmetic_subtraction_basic() {
    assert_eq!(eval_expr("(- 10 3)"), "7");
}

#[test]
fn cljs_arithmetic_subtraction_variadic() {
    assert_eq!(eval_expr("(- 10 3 2)"), "5");
}

#[test]
fn cljs_arithmetic_subtraction_unary() {
    assert_eq!(eval_expr("(- 5)"), "-5");
}

#[test]
fn cljs_arithmetic_multiplication_basic() {
    assert_eq!(eval_expr("(* 2 3)"), "6");
}

#[test]
fn cljs_arithmetic_multiplication_variadic() {
    assert_eq!(eval_expr("(* 2 3 4)"), "24");
}

#[test]
fn cljs_arithmetic_multiplication_many() {
    assert_eq!(eval_expr("(* 1 2 3 4 5 6 7 8 9 10)"), "3628800");
}

#[test]
fn cljs_arithmetic_multiplication_zero_arity() {
    assert_eq!(eval_expr("(*)"), "1");
}

#[test]
fn cljs_arithmetic_multiplication_unary() {
    assert_eq!(eval_expr("(* 5)"), "5");
}

#[test]
fn cljs_arithmetic_division_basic() {
    assert_eq!(eval_expr("(/ 10 2)"), "5");
}

#[test]
fn cljs_arithmetic_division_integer() {
    assert_eq!(eval_expr("(/ 7 2)"), "3");
}

#[test]
fn cljs_arithmetic_division_float() {
    assert_eq!(eval_expr("(/ 10.0 3)"), "3.3333333333333335");
}

#[test]
fn cljs_arithmetic_quot() {
    assert_eq!(eval_expr("(quot 10 3)"), "3");
    assert_eq!(eval_expr("(quot 7 2)"), "3");
}

#[test]
fn cljs_arithmetic_rem() {
    assert_eq!(eval_expr("(rem 10 3)"), "1");
    assert_eq!(eval_expr("(rem 7 2)"), "1");
    assert_eq!(eval_expr("(rem -7 2)"), "-1");
}

#[test]
fn cljs_arithmetic_mod() {
    assert_eq!(eval_expr("(mod 10 3)"), "1");
    assert_eq!(eval_expr("(mod 7 2)"), "1");
    assert_eq!(eval_expr("(mod -7 2)"), "1");
}

#[test]
fn cljs_arithmetic_inc() {
    assert_eq!(eval_expr("(inc 0)"), "1");
    assert_eq!(eval_expr("(inc 5)"), "6");
    assert_eq!(eval_expr("(inc -1)"), "0");
}

#[test]
fn cljs_arithmetic_dec() {
    assert_eq!(eval_expr("(dec 1)"), "0");
    assert_eq!(eval_expr("(dec 5)"), "4");
    assert_eq!(eval_expr("(dec 0)"), "-1");
}

#[test]
fn cljs_arithmetic_max() {
    assert_eq!(eval_expr("(max 1 2 3)"), "3");
    assert_eq!(eval_expr("(max 5 3 7 1 9)"), "9");
}

#[test]
fn cljs_arithmetic_min() {
    assert_eq!(eval_expr("(min 1 2 3)"), "1");
    assert_eq!(eval_expr("(min 5 3 7 1 9)"), "1");
}

#[test]
fn cljs_arithmetic_abs() {
    assert_eq!(eval_expr("(abs -5)"), "5");
    assert_eq!(eval_expr("(abs 5)"), "5");
    assert_eq!(eval_expr("(abs 0)"), "0");
}

#[test]
fn cljs_arithmetic_large_numbers() {
    assert_eq!(eval_expr("(+ 1000000000 2000000000)"), "3000000000");
    assert_eq!(eval_expr("(* 100000 100000)"), "10000000000");
}

// ============================================================================
// NUMERIC PREDICATES
// ============================================================================

#[test]
fn cljs_numeric_pred_zero() {
    assert_eq!(eval_expr("(zero? 0)"), "true");
    assert_eq!(eval_expr("(zero? 1)"), "false");
}

#[test]
fn cljs_numeric_pred_pos() {
    assert_eq!(eval_expr("(pos? 1)"), "true");
    assert_eq!(eval_expr("(pos? 0)"), "false");
    assert_eq!(eval_expr("(pos? -1)"), "false");
}

#[test]
fn cljs_numeric_pred_neg() {
    assert_eq!(eval_expr("(neg? -1)"), "true");
    assert_eq!(eval_expr("(neg? 0)"), "false");
    assert_eq!(eval_expr("(neg? 1)"), "false");
}

#[test]
fn cljs_numeric_pred_even() {
    assert_eq!(eval_expr("(even? 0)"), "true");
    assert_eq!(eval_expr("(even? 2)"), "true");
    assert_eq!(eval_expr("(even? 3)"), "false");
}

#[test]
fn cljs_numeric_pred_odd() {
    assert_eq!(eval_expr("(odd? 1)"), "true");
    assert_eq!(eval_expr("(odd? 3)"), "true");
    assert_eq!(eval_expr("(odd? 2)"), "false");
}

#[test]
fn cljs_numeric_pred_number() {
    assert_eq!(eval_expr("(number? 42)"), "true");
    assert_eq!(eval_expr("(number? 3.14)"), "true");
    assert_eq!(eval_expr("(number? :a)"), "false");
    assert_eq!(eval_expr("(number? \"hello\")"), "false");
}

#[test]
fn cljs_numeric_pred_integer() {
    assert_eq!(eval_expr("(integer? 1)"), "true");
    assert_eq!(eval_expr("(integer? 1.0)"), "false");
}

#[test]
fn cljs_numeric_pred_float() {
    assert_eq!(eval_expr("(float? 1.0)"), "true");
    assert_eq!(eval_expr("(float? 1)"), "false");
}

// ============================================================================
// COMPARISON OPERATORS
// ============================================================================

#[test]
fn cljs_comparison_less_than() {
    assert_eq!(eval_expr("(< 1 2)"), "true");
    assert_eq!(eval_expr("(< 2 1)"), "false");
    assert_eq!(eval_expr("(< 1 1)"), "false");
}

#[test]
fn cljs_comparison_less_than_variadic() {
    assert_eq!(eval_expr("(< 1 2 3)"), "true");
    assert_eq!(eval_expr("(< 1 2 3 4 5)"), "true");
    assert_eq!(eval_expr("(< 1 3 2)"), "false");
}

#[test]
fn cljs_comparison_less_equal() {
    assert_eq!(eval_expr("(<= 1 1)"), "true");
    assert_eq!(eval_expr("(<= 1 2)"), "true");
    assert_eq!(eval_expr("(<= 2 1)"), "false");
    assert_eq!(eval_expr("(<= 1 1 2 2 3)"), "true");
}

#[test]
fn cljs_comparison_greater_than() {
    assert_eq!(eval_expr("(> 2 1)"), "true");
    assert_eq!(eval_expr("(> 1 2)"), "false");
    assert_eq!(eval_expr("(> 5 4 3 2 1)"), "true");
}

#[test]
fn cljs_comparison_greater_equal() {
    assert_eq!(eval_expr("(>= 2 2)"), "true");
    assert_eq!(eval_expr("(>= 2 1)"), "true");
    assert_eq!(eval_expr("(>= 1 2)"), "false");
    assert_eq!(eval_expr("(>= 3 3 2 1)"), "true");
}

#[test]
fn cljs_comparison_equality_primitives() {
    assert_eq!(eval_expr("(= 1 1)"), "true");
    assert_eq!(eval_expr("(= 1 2)"), "false");
    assert_eq!(eval_expr("(= :a :a)"), "true");
    assert_eq!(eval_expr("(= :a :b)"), "false");
    assert_eq!(eval_expr("(= nil nil)"), "true");
    assert_eq!(eval_expr("(= true true)"), "true");
    assert_eq!(eval_expr("(= false false)"), "true");
    assert_eq!(eval_expr("(= true false)"), "false");
}

#[test]
fn cljs_comparison_equality_variadic() {
    assert_eq!(eval_expr("(= 1 1 1 1)"), "true");
    assert_eq!(eval_expr("(= 1 1 2 1)"), "false");
}

#[test]
fn cljs_comparison_not_equal() {
    assert_eq!(eval_expr("(not= 1 2)"), "true");
    assert_eq!(eval_expr("(not= 1 1)"), "false");
}

#[test]
fn cljs_comparison_double_equals() {
    assert_eq!(eval_expr("(== 1 1)"), "true");
}

// ============================================================================
// BIT OPERATIONS
// ============================================================================

#[test]
fn cljs_bit_and() {
    assert_eq!(eval_expr("(bit-and 255 15)"), "15");
}

#[test]
fn cljs_bit_or() {
    assert_eq!(eval_expr("(bit-or 15 240)"), "255");
}

#[test]
fn cljs_bit_xor() {
    assert_eq!(eval_expr("(bit-xor 255 15)"), "240");
}

#[test]
fn cljs_bit_not() {
    assert_eq!(eval_expr("(bit-not 0)"), "-1");
}

#[test]
fn cljs_bit_shift_left() {
    assert_eq!(eval_expr("(bit-shift-left 1 4)"), "16");
}

#[test]
fn cljs_bit_shift_right() {
    assert_eq!(eval_expr("(bit-shift-right 16 4)"), "1");
}

// ============================================================================
// BOOLEAN AND LOGIC
// ============================================================================

#[test]
fn cljs_logic_not() {
    assert_eq!(eval_expr("(not true)"), "false");
    assert_eq!(eval_expr("(not false)"), "true");
    assert_eq!(eval_expr("(not nil)"), "true");
    assert_eq!(eval_expr("(not 1)"), "false");
    assert_eq!(eval_expr("(not 0)"), "false");
}

#[test]
fn cljs_logic_and() {
    assert_eq!(eval_expr("(and)"), "true");
    assert_eq!(eval_expr("(and true true)"), "true");
    assert_eq!(eval_expr("(and true false)"), "false");
    assert_eq!(eval_expr("(and false true 0)"), "false");
    assert_eq!(eval_expr("(and true true 0)"), "0");
}

#[test]
fn cljs_logic_or() {
    assert_eq!(eval_expr("(or)"), "nil");
    assert_eq!(eval_expr("(or false false 1)"), "1");
    assert_eq!(eval_expr("(or false false false)"), "false");
    assert_eq!(eval_expr("(or nil nil true)"), "true");
}

#[test]
fn cljs_logic_or_many_nils() {
    assert_eq!(
        eval_expr("(or nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil true)"),
        "true"
    );
}

// ============================================================================
// TYPE PREDICATES
// ============================================================================

#[test]
fn cljs_pred_nil() {
    assert_eq!(eval_expr("(nil? nil)"), "true");
    assert_eq!(eval_expr("(nil? 1)"), "false");
    assert_eq!(eval_expr("(nil? false)"), "false");
}

#[test]
fn cljs_pred_some() {
    assert_eq!(eval_expr("(some? nil)"), "false");
    assert_eq!(eval_expr("(some? 1)"), "true");
    assert_eq!(eval_expr("(some? false)"), "true");
}

#[test]
fn cljs_pred_true() {
    assert_eq!(eval_expr("(true? true)"), "true");
    assert_eq!(eval_expr("(true? 1)"), "false");
    assert_eq!(eval_expr("(true? false)"), "false");
}

#[test]
fn cljs_pred_false() {
    assert_eq!(eval_expr("(false? false)"), "true");
    assert_eq!(eval_expr("(false? nil)"), "false");
    assert_eq!(eval_expr("(false? true)"), "false");
}

#[test]
fn cljs_pred_string() {
    assert_eq!(eval_expr("(string? \"hi\")"), "true");
    assert_eq!(eval_expr("(string? 1)"), "false");
    assert_eq!(eval_expr("(string? :a)"), "false");
}

#[test]
fn cljs_pred_keyword() {
    assert_eq!(eval_expr("(keyword? :a)"), "true");
    assert_eq!(eval_expr("(keyword? \"a\")"), "false");
    assert_eq!(eval_expr("(keyword? 1)"), "false");
}

#[test]
fn cljs_pred_symbol() {
    assert_eq!(eval_expr("(symbol? 'a)"), "true");
    assert_eq!(eval_expr("(symbol? :a)"), "false");
    assert_eq!(eval_expr("(symbol? 1)"), "false");
}

#[test]
fn cljs_pred_fn() {
    assert_eq!(eval_expr("(fn? inc)"), "true");
    assert_eq!(eval_expr("(fn? +)"), "true");
    assert_eq!(eval_expr("(fn? 1)"), "false");
}

#[test]
fn cljs_pred_vector() {
    assert_eq!(eval_expr("(vector? [1 2])"), "true");
    assert_eq!(eval_expr("(vector? (list 1))"), "false");
    assert_eq!(eval_expr("(vector? #{1})"), "false");
}

#[test]
fn cljs_pred_map() {
    assert_eq!(eval_expr("(map? {:a 1})"), "true");
    assert_eq!(eval_expr("(map? {})"), "true");
    assert_eq!(eval_expr("(map? [1])"), "false");
}

#[test]
fn cljs_pred_set() {
    assert_eq!(eval_expr("(set? #{1 2})"), "true");
    assert_eq!(eval_expr("(set? #{})"), "true");
    assert_eq!(eval_expr("(set? [1])"), "false");
}

#[test]
fn cljs_pred_list() {
    assert_eq!(eval_expr("(list? (list 1))"), "true");
    assert_eq!(eval_expr("(list? [1])"), "false");
}

#[test]
fn cljs_pred_coll() {
    assert_eq!(eval_expr("(coll? [1])"), "true");
    assert_eq!(eval_expr("(coll? {:a 1})"), "true");
    assert_eq!(eval_expr("(coll? #{1})"), "true");
    assert_eq!(eval_expr("(coll? (list 1))"), "true");
    assert_eq!(eval_expr("(coll? 1)"), "false");
}

#[test]
fn cljs_pred_sequential() {
    assert_eq!(eval_expr("(sequential? [1])"), "true");
    assert_eq!(eval_expr("(sequential? (list 1))"), "true");
    assert_eq!(eval_expr("(sequential? {:a 1})"), "false");
    assert_eq!(eval_expr("(sequential? #{1})"), "false");
}

#[test]
fn cljs_pred_associative() {
    assert_eq!(eval_expr("(associative? {:a 1})"), "true");
    assert_eq!(eval_expr("(associative? [1])"), "true");
}

// ============================================================================
// STRING OPERATIONS
// ============================================================================

#[test]
fn cljs_str_basic() {
    assert_eq!(eval_expr("(str)"), "\"\"");
    assert_eq!(eval_expr("(str nil)"), "\"\"");
    assert_eq!(eval_expr("(str \"a\")"), "\"a\"");
    assert_eq!(eval_expr("(str 1)"), "\"1\"");
}

#[test]
fn cljs_str_concatenation() {
    assert_eq!(eval_expr("(str \"hello\" \" \" \"world\")"), "\"hello world\"");
    assert_eq!(eval_expr("(str \"a\" \"b\")"), "\"ab\"");
}

#[test]
fn cljs_str_mixed_types() {
    assert_eq!(eval_expr("(str 1 2 3)"), "\"123\"");
    assert_eq!(eval_expr("(str \"a\" 1 \"b\" 2 \"c\" 3)"), "\"a1b2c3\"");
    assert_eq!(eval_expr("(str \"a\" nil \"b\")"), "\"ab\"");
}

/// CLJS test-cljs-2864
#[test]
fn cljs_str_xyzzy() {
    assert_eq!(eval_expr("(str \"x\" \"y\" \"z\" \"z\" \"y\")"), "\"xyzzy\"");
}

/// CLJS test-801: str with mixed types
#[test]
fn cljs_str_mixed_all_types() {
    assert_eq!(
        eval_expr("(str 0 \"a\" true nil :key/word)"),
        "\"0atrue:key/word\""
    );
}

#[test]
fn cljs_str_on_collections() {
    assert_eq!(eval_expr("(str [1 2 3])"), "\"[1 2 3]\"");
    assert_eq!(eval_expr("(str (list 1 2 3))"), "\"(1 2 3)\"");
    assert_eq!(eval_expr("(str :hello)"), "\":hello\"");
    assert_eq!(eval_expr("(str true)"), "\"true\"");
    assert_eq!(eval_expr("(str false)"), "\"false\"");
}

#[test]
fn cljs_str_empty_collections() {
    assert_eq!(eval_expr("(str [])"), "\"[]\"");
    assert_eq!(eval_expr("(str (list))"), "\"()\"");
    assert_eq!(eval_expr("(str #{})"), "\"#{}\"");
    assert_eq!(eval_expr("(str {})"), "\"{}\"");
}

#[test]
fn cljs_str_count() {
    assert_eq!(eval_expr("(count \"hello\")"), "5");
    assert_eq!(eval_expr("(count \"\")"), "0");
    assert_eq!(eval_expr("(count \"a\")"), "1");
}

#[test]
fn cljs_str_name() {
    assert_eq!(eval_expr("(name :foo)"), "\"foo\"");
    assert_eq!(eval_expr("(name 'bar)"), "\"bar\"");
    assert_eq!(eval_expr("(name \"already-a-string\")"), "\"already-a-string\"");
}

#[test]
fn cljs_str_apply_str() {
    assert_eq!(eval_expr("(apply str [\"a\" \"b\" \"c\"])"), "\"abc\"");
    assert_eq!(eval_expr("(apply str [])"), "\"\"");
}

#[test]
fn cljs_str_reduce_str() {
    assert_eq!(eval_expr("(reduce str [\"a\" \"b\" \"c\"])"), "\"abc\"");
    assert_eq!(eval_expr("(reduce str \"\" [\"a\" \"b\" \"c\"])"), "\"abc\"");
}

#[test]
fn cljs_str_apply_str_range() {
    assert_eq!(eval_expr("(apply str (range 5))"), "\"01234\"");
}

// ============================================================================
// SYMBOL OPERATIONS
// ============================================================================

#[test]
fn cljs_symbol_from_string() {
    assert_eq!(eval_expr("(symbol \"hello\")"), "hello");
    assert_eq!(eval_expr("(symbol \"ns\" \"name\")"), "ns/name");
}

#[test]
fn cljs_symbol_pred() {
    assert_eq!(eval_expr("(symbol? 'foo)"), "true");
    assert_eq!(eval_expr("(symbol? :foo)"), "false");
}

// ============================================================================
// KEYWORD OPERATIONS
// ============================================================================

#[test]
fn cljs_keyword_as_fn() {
    assert_eq!(eval_expr("(:a {:a 42 :b 99})"), "42");
    assert_eq!(eval_expr("(:c {:a 1})"), "nil");
    assert_eq!(eval_expr("(:c {:a 1} :default)"), ":default");
    assert_eq!(eval_expr("(:b {:a 1} :not-found)"), ":not-found");
}

#[test]
fn cljs_keyword_namespaced() {
    assert_eq!(eval_expr(":foo/bar"), ":foo/bar");
    assert_eq!(eval_expr("(str :hello/world)"), "\":hello/world\"");
}

// ============================================================================
// VECTORS
// ============================================================================

#[test]
fn cljs_vector_creation() {
    assert_eq!(eval_expr("(vector 1 2 3)"), "[1 2 3]");
    assert_eq!(eval_expr("(vec (list 1 2 3))"), "[1 2 3]");
    assert_eq!(eval_expr("[1 2 3]"), "[1 2 3]");
}

#[test]
fn cljs_vector_count() {
    assert_eq!(eval_expr("(count [1 2 3])"), "3");
    assert_eq!(eval_expr("(count [])"), "0");
}

#[test]
fn cljs_vector_nth() {
    assert_eq!(eval_expr("(nth [10 20 30] 0)"), "10");
    assert_eq!(eval_expr("(nth [10 20 30] 1)"), "20");
    assert_eq!(eval_expr("(nth [10 20 30] 2)"), "30");
}

#[test]
fn cljs_vector_nth_with_default() {
    assert_eq!(eval_expr("(nth [1 2 3] 5 :not-found)"), ":not-found");
}

#[test]
fn cljs_vector_get() {
    assert_eq!(eval_expr("(get [10 20 30] 0)"), "10");
    assert_eq!(eval_expr("(get [10 20 30] 1)"), "20");
}

#[test]
fn cljs_vector_first_second_last() {
    assert_eq!(eval_expr("(first [1 2 3])"), "1");
    assert_eq!(eval_expr("(second [1 2 3])"), "2");
    assert_eq!(eval_expr("(last [1 2 3])"), "3");
}

#[test]
fn cljs_vector_conj() {
    assert_eq!(eval_expr("(conj [1 2] 3)"), "[1 2 3]");
    assert_eq!(eval_expr("(vector? (conj [1] 2))"), "true");
}

#[test]
fn cljs_vector_assoc() {
    assert_eq!(eval_expr("(assoc [1 2 3] 0 :a)"), "[:a 2 3]");
    assert_eq!(eval_expr("(assoc [1 2 3] 1 :b)"), "[1 :b 3]");
}

#[test]
fn cljs_vector_update() {
    assert_eq!(eval_expr("(str (update [1 2 3] 0 inc))"), "\"[2 2 3]\"");
}

#[test]
fn cljs_vector_as_fn() {
    assert_eq!(eval_expr("([10 20 30] 0)"), "10");
    assert_eq!(eval_expr("([10 20 30] 1)"), "20");
    assert_eq!(eval_expr("([10 20 30] 2)"), "30");
}

#[test]
fn cljs_vector_contains() {
    assert_eq!(eval_expr("(contains? [5 6 7] 0)"), "true");
    assert_eq!(eval_expr("(contains? [5 6 7] 1)"), "true");
    assert_eq!(eval_expr("(contains? [5 6 7] 2)"), "true");
    assert_eq!(eval_expr("(contains? [5 6 7] 3)"), "false");
}

#[test]
fn cljs_vector_empty() {
    assert_eq!(eval_expr("(empty? [])"), "true");
    assert_eq!(eval_expr("(empty? [1])"), "false");
}

#[test]
fn cljs_vector_seq() {
    assert_eq!(eval_expr("(seq [])"), "nil");
    assert_eq!(eval_expr("(seq [1])"), "(1)");
}

#[test]
fn cljs_vector_empty_fn() {
    assert_eq!(eval_expr("(str (empty [1 2 3]))"), "\"[]\"");
}

#[test]
fn cljs_vector_into() {
    assert_eq!(eval_expr("(str (into [] (list 1 2 3)))"), "\"[1 2 3]\"");
    assert_eq!(eval_expr("(str (into [] (range 5)))"), "\"[0 1 2 3 4]\"");
}

// ============================================================================
// LISTS
// ============================================================================

#[test]
fn cljs_list_creation() {
    assert_eq!(eval_expr("(list 1 2 3)"), "(1 2 3)");
    assert_eq!(eval_expr("(str (list 1 2 3))"), "\"(1 2 3)\"");
}

#[test]
fn cljs_list_predicates() {
    assert_eq!(eval_expr("(list? (list 1))"), "true");
    assert_eq!(eval_expr("(list? [1])"), "false");
}

#[test]
fn cljs_list_conj() {
    assert_eq!(eval_expr("(conj (list 1 2) 3)"), "(3 1 2)");
    assert_eq!(eval_expr("(list? (conj (list 1) 2))"), "true");
}

#[test]
fn cljs_list_first_rest() {
    assert_eq!(eval_expr("(first (list 1 2 3))"), "1");
    assert_eq!(eval_expr("(str (rest (list 1 2 3)))"), "\"(2 3)\"");
}

#[test]
fn cljs_list_count() {
    assert_eq!(eval_expr("(count (list 1 2 3))"), "3");
    assert_eq!(eval_expr("(count (list))"), "0");
}

#[test]
fn cljs_list_into() {
    assert_eq!(eval_expr("(str (into (list) [1 2 3]))"), "\"(3 2 1)\"");
}

#[test]
fn cljs_list_empty_fn() {
    assert_eq!(eval_expr("(str (empty (list 1 2 3)))"), "\"()\"");
}

// ============================================================================
// MAPS
// ============================================================================

#[test]
fn cljs_map_get() {
    assert_eq!(eval_expr("(get {:a 1 :b 2} :a)"), "1");
    assert_eq!(eval_expr("(get {:a 1} :b)"), "nil");
    assert_eq!(eval_expr("(get {:a 1} :b :default)"), ":default");
}

#[test]
fn cljs_map_contains() {
    assert_eq!(eval_expr("(contains? {:a 1 :b 2} :a)"), "true");
    assert_eq!(eval_expr("(contains? {:a 1 :b 2} :z)"), "false");
    assert_eq!(eval_expr("(contains? nil 42)"), "false");
}

#[test]
fn cljs_map_count() {
    assert_eq!(eval_expr("(count {:a 1 :b 2})"), "2");
    assert_eq!(eval_expr("(count {})"), "0");
    assert_eq!(eval_expr("(count {:a 1 :b 2 :c 3 :d 4 :e 5})"), "5");
}

#[test]
fn cljs_map_assoc() {
    assert_eq!(eval_expr("(get (assoc {:a 1} :b 2) :b)"), "2");
    assert_eq!(eval_expr("(count (assoc {:a 1} :b 2))"), "2");
}

#[test]
fn cljs_map_assoc_multiple() {
    assert_eq!(eval_expr("(count (assoc {} :a 1 :b 2 :c 3))"), "3");
    assert_eq!(eval_expr("(get (assoc {} :a 1 :b 2 :c 3) :b)"), "2");
}

#[test]
fn cljs_map_dissoc() {
    assert_eq!(eval_expr("(contains? (dissoc {:a 1 :b 2} :b) :b)"), "false");
    assert_eq!(eval_expr("(count (dissoc {:a 1 :b 2} :b))"), "1");
}

#[test]
fn cljs_map_dissoc_multiple() {
    assert_eq!(eval_expr("(count (dissoc {:a 1 :b 2 :c 3} :a :c))"), "1");
    assert_eq!(eval_expr("(get (dissoc {:a 1 :b 2 :c 3} :a :c) :b)"), "2");
}

#[test]
fn cljs_map_merge() {
    assert_eq!(eval_expr("(count (merge {:a 1} {:b 2} {:c 3}))"), "3");
    assert_eq!(eval_expr("(get (merge {:a 1} {:b 2} {:c 3}) :c)"), "3");
    assert_eq!(eval_expr("(count (merge {:a 1} {:b 2} {:c 3} {:d 4}))"), "4");
}

#[test]
fn cljs_map_merge_overwrite() {
    assert_eq!(eval_expr("(get (merge {:a 1} {:a 2}) :a)"), "2");
    assert_eq!(eval_expr("(get (merge {:a 1 :b 2} {:b 3 :c 4}) :b)"), "3");
}

#[test]
fn cljs_map_update() {
    assert_eq!(eval_expr("(get (update {:a 1} :a inc) :a)"), "2");
    assert_eq!(eval_expr("(get (update {:a 1} :a + 10) :a)"), "11");
    assert_eq!(eval_expr("(get (update {:a 1} :a + 10 20) :a)"), "31");
}

#[test]
fn cljs_map_as_fn() {
    assert_eq!(eval_expr("({:a 1} :a)"), "1");
    assert_eq!(eval_expr("({:a 1} :b)"), "nil");
    assert_eq!(eval_expr("({:a 1} :b 99)"), "99");
    assert_eq!(eval_expr("({:a 1} 2 3)"), "3");
}

#[test]
fn cljs_map_hash_map() {
    assert_eq!(eval_expr("((hash-map :a 1) :a 3)"), "1");
    assert_eq!(eval_expr("(count (apply hash-map [:a 1 :b 2]))"), "2");
}

#[test]
fn cljs_map_keys_vals() {
    assert_eq!(eval_expr("(count (keys {:a 1 :b 2}))"), "2");
    assert_eq!(eval_expr("(count (vals {:a 1 :b 2}))"), "2");
    // Map iteration order is not guaranteed, so check via sort
    assert_eq!(eval_expr("(str (sort (keys {:a 1 :b 2})))"), "\"(:a :b)\"");
    assert_eq!(eval_expr("(str (sort (vals {:a 1 :b 2})))"), "\"(1 2)\"");
}

#[test]
fn cljs_map_key_val() {
    assert_eq!(eval_expr("(key (first {:a 1}))"), ":a");
    assert_eq!(eval_expr("(val (first {:a 1}))"), "1");
}

#[test]
fn cljs_map_select_keys() {
    assert_eq!(eval_expr("(count (select-keys {:a 1 :b 2 :c 3} [:a :c]))"), "2");
    assert_eq!(eval_expr("(get (select-keys {:a 1 :b 2 :c 3} [:a :c]) :a)"), "1");
    assert_eq!(eval_expr("(count (select-keys {:a 1 :b 2 :c 3 :d 4} [:a :c]))"), "2");
}

#[test]
fn cljs_map_zipmap() {
    assert_eq!(eval_expr("(get (zipmap [:a :b :c] [1 2 3]) :b)"), "2");
    assert_eq!(eval_expr("(count (zipmap [:a :b :c] [1 2 3]))"), "3");
}

#[test]
fn cljs_map_frequencies() {
    assert_eq!(eval_expr("(get (frequencies [:a :b :a :c :b :a]) :a)"), "3");
    assert_eq!(eval_expr("(get (frequencies [:a :b :a :c :b :a]) :b)"), "2");
    assert_eq!(eval_expr("(get (frequencies [:a :b :a :c :b :a]) :c)"), "1");
    assert_eq!(eval_expr("(get (frequencies [1 1 2 3 3 3]) 3)"), "3");
    assert_eq!(eval_expr("(get (frequencies [1 1 2 3 3 3]) 1)"), "2");
}

#[test]
fn cljs_map_group_by() {
    assert_eq!(eval_expr("(count (get (group-by odd? [1 2 3 4 5]) true))"), "3");
    assert_eq!(eval_expr("(count (get (group-by odd? [1 2 3 4 5]) false))"), "2");
    assert_eq!(eval_expr("(str (get (group-by odd? [1 2 3 4 5]) true))"), "\"[1 3 5]\"");
}

#[test]
fn cljs_map_empty_fn() {
    assert_eq!(eval_expr("(str (empty {:a 1}))"), "\"{}\"");
}

// ============================================================================
// GET-IN, ASSOC-IN, UPDATE-IN
// ============================================================================

/// CLJS test-in-operations
#[test]
fn cljs_get_in_basic() {
    assert_eq!(eval_expr("(get-in {:foo 1 :bar 2} [:foo])"), "1");
    assert_eq!(eval_expr("(get-in {:foo {:bar 2}} [:foo :bar])"), "2");
    assert_eq!(eval_expr("(get-in {:a {:b {:c 42}}} [:a :b :c])"), "42");
}

#[test]
fn cljs_get_in_vector_index() {
    assert_eq!(eval_expr("(get-in [{:a 1} {:a 2}] [0 :a])"), "1");
    assert_eq!(eval_expr("(get-in [{:a 1} {:a 2}] [1 :a])"), "2");
}

#[test]
fn cljs_get_in_nested_vector() {
    assert_eq!(eval_expr("(get-in [[1 2] [3 4] [5 6]] [1 0])"), "3");
    assert_eq!(eval_expr("(get-in [[1 2] [3 4] [5 6]] [2 1])"), "6");
}

#[test]
fn cljs_get_in_deep() {
    assert_eq!(eval_expr("(get-in {:a {:b {:c {:d 42}}}} [:a :b :c :d])"), "42");
    assert_eq!(
        eval_expr("(get-in [{:foo 1 :bar [{:baz 1} {:buzz 2}]} {:foo 3 :bar [{:baz 3} {:buzz 4}]}] [1 :bar 1 :buzz])"),
        "4"
    );
}

#[test]
fn cljs_get_in_not_found() {
    assert_eq!(eval_expr("(get-in {:a 1} [:b])"), "nil");
}

/// CLJS test-in-operations: update-in
#[test]
fn cljs_update_in_basic() {
    assert_eq!(eval_expr("(get-in (update-in {:a {:b 1}} [:a :b] inc) [:a :b])"), "2");
    assert_eq!(eval_expr("(get-in (update-in {:a {:b [1 2 3]}} [:a :b] count) [:a :b])"), "3");
}

/// CLJS test-in-operations: assoc-in
#[test]
fn cljs_assoc_in_basic() {
    assert_eq!(eval_expr("(get-in (assoc-in {} [:a :b :c] 42) [:a :b :c])"), "42");
    assert_eq!(eval_expr("(get-in (assoc-in {:a {:b 1}} [:a :b] 99) [:a :b])"), "99");
}

// ============================================================================
// SETS
// ============================================================================

/// CLJS test-582
#[test]
fn cljs_set_creation() {
    assert_eq!(eval_expr("(count (set [1 2 2]))"), "2");
    assert_eq!(eval_expr("(count (hash-set 1 2 2))"), "2");
    assert_eq!(eval_expr("(count (apply hash-set [1 2 2]))"), "2");
}

#[test]
fn cljs_set_contains() {
    assert_eq!(eval_expr("(contains? #{1 2 3} 2)"), "true");
    assert_eq!(eval_expr("(contains? #{1 2 3} 4)"), "false");
}

#[test]
fn cljs_set_count() {
    assert_eq!(eval_expr("(count #{1 2 3})"), "3");
    assert_eq!(eval_expr("(count #{})"), "0");
}

#[test]
fn cljs_set_as_fn() {
    assert_eq!(eval_expr("(#{:a :b :c} :a)"), ":a");
    assert_eq!(eval_expr("(#{:a :b :c} :d)"), "nil");
    assert_eq!(eval_expr("(#{1 2 3} 2)"), "2");
    assert_eq!(eval_expr("(#{1 2 3} 4)"), "nil");
}

#[test]
fn cljs_set_disj() {
    assert_eq!(eval_expr("(count (disj #{1 2 3} 2))"), "2");
    assert_eq!(eval_expr("(contains? (disj #{1 2 3} 2) 2)"), "false");
}

#[test]
fn cljs_set_disj_multiple() {
    assert_eq!(eval_expr("(count (disj #{1 2 3 4} 1 3))"), "2");
    assert_eq!(eval_expr("(contains? (disj #{1 2 3 4} 1 3) 2)"), "true");
    assert_eq!(eval_expr("(contains? (disj #{1 2 3 4} 1 3) 1)"), "false");
}

#[test]
fn cljs_set_conj() {
    assert_eq!(eval_expr("(count (conj #{1 2} 3))"), "3");
}

/// CLJS test-3454-conj
#[test]
fn cljs_set_conj_existing() {
    // conj of existing element should not change count
    assert_eq!(eval_expr("(count (conj #{1 2} 2))"), "2");
}

#[test]
fn cljs_set_from_vector() {
    assert_eq!(eval_expr("(count (set [1 2 2 3 3 3]))"), "3");
    assert_eq!(eval_expr("(contains? (set [1 2 3]) 2)"), "true");
}

#[test]
fn cljs_set_empty_fn() {
    assert_eq!(eval_expr("(str (empty #{1 2}))"), "\"#{}\"");
}

#[test]
fn cljs_set_into() {
    assert_eq!(eval_expr("(count (into #{} [1 2 2 3]))"), "3");
}

// ============================================================================
// SEQUENCE OPERATIONS
// ============================================================================

#[test]
fn cljs_seq_first() {
    assert_eq!(eval_expr("(first [1 2 3])"), "1");
    assert_eq!(eval_expr("(first (list 1 2 3))"), "1");
    assert_eq!(eval_expr("(first nil)"), "nil");
    assert_eq!(eval_expr("(first [])"), "nil");
}

#[test]
fn cljs_seq_second() {
    assert_eq!(eval_expr("(second [10 20 30])"), "20");
}

#[test]
fn cljs_seq_rest() {
    assert_eq!(eval_expr("(str (rest [1 2 3]))"), "\"(2 3)\"");
    assert_eq!(eval_expr("(str (rest [1]))"), "\"()\"");
}

#[test]
fn cljs_seq_next() {
    assert_eq!(eval_expr("(str (next [1 2 3]))"), "\"(2 3)\"");
    assert_eq!(eval_expr("(next [1])"), "nil");
    assert_eq!(eval_expr("(next nil)"), "nil");
}

#[test]
fn cljs_seq_last() {
    assert_eq!(eval_expr("(last [1 2 3])"), "3");
    assert_eq!(eval_expr("(last nil)"), "nil");
    assert_eq!(eval_expr("(last [])"), "nil");
}

#[test]
fn cljs_seq_count() {
    assert_eq!(eval_expr("(count [1 2 3])"), "3");
    assert_eq!(eval_expr("(count nil)"), "0");
    assert_eq!(eval_expr("(count [])"), "0");
    assert_eq!(eval_expr("(count {:a 1 :b 2})"), "2");
}

#[test]
fn cljs_seq_cons() {
    assert_output(
        "(println (str (cons 0 [1 2 3])))",
        "(0 1 2 3)",
    );
}

#[test]
fn cljs_seq_conj_nil() {
    assert_eq!(eval_expr("(str (conj nil 1))"), "\"(1)\"");
}

#[test]
fn cljs_seq_concat() {
    assert_output(
        "(println (str (concat [1 2] [3 4] [5])))",
        "(1 2 3 4 5)",
    );
}

/// CLJS test-604
#[test]
fn cljs_seq_concat_empty() {
    assert_eq!(eval_expr("(str (concat nil []))"), "\"\"");
    assert_eq!(eval_expr("(str (concat [] []))"), "\"\"");
}

#[test]
fn cljs_seq_into_vector() {
    assert_output("(println (str (into [] '(1 2 3))))", "[1 2 3]");
}

#[test]
fn cljs_seq_into_map() {
    assert_eq!(eval_expr("(get (into {} [[:a 1] [:b 2]]) :a)"), "1");
}

#[test]
fn cljs_seq_into_set() {
    assert_eq!(eval_expr("(count (into #{} [1 2 2 3]))"), "3");
}

/// CLJS test-3054
#[test]
fn cljs_seq_into_nil() {
    assert_eq!(eval_expr("(str (into nil [1 2 3]))"), "\"(3 2 1)\"");
}

#[test]
fn cljs_seq_reverse() {
    assert_output("(println (str (reverse [1 2 3])))", "(3 2 1)");
}

#[test]
fn cljs_seq_sort() {
    assert_output("(println (str (sort [3 1 2])))", "(1 2 3)");
}

#[test]
fn cljs_seq_sort_with_comparator() {
    assert_eq!(eval_expr("(str (sort > [3 1 2]))"), "\"(3 2 1)\"");
}

#[test]
fn cljs_seq_distinct() {
    assert_output(
        "(println (str (distinct [1 2 1 3 2 4])))",
        "(1 2 3 4)",
    );
}

#[test]
fn cljs_seq_interleave() {
    assert_output(
        "(println (str (interleave [1 2 3] [:a :b :c])))",
        "(1 :a 2 :b 3 :c)",
    );
}

#[test]
fn cljs_seq_partition() {
    assert_output(
        "(println (str (partition 2 [1 2 3 4 5])))",
        "((1 2) (3 4))",
    );
}

#[test]
fn cljs_seq_butlast() {
    assert_output("(println (str (butlast [1 2 3])))", "(1 2)");
}

#[test]
fn cljs_seq_flatten() {
    assert_output(
        "(println (str (flatten [1 [2 [3 4]] 5])))",
        "(1 2 3 4 5)",
    );
    assert_eq!(eval_expr("(str (flatten [1 [2] [[3]]]))"), "\"(1 2 3)\"");
}

#[test]
fn cljs_seq_vec() {
    assert_output("(println (str (vec '(1 2 3))))", "[1 2 3]");
}

#[test]
fn cljs_seq_empty() {
    assert_eq!(eval_expr("(empty? [])"), "true");
    assert_eq!(eval_expr("(empty? nil)"), "true");
    assert_eq!(eval_expr("(empty? [1])"), "false");
}

#[test]
fn cljs_seq_on_empty() {
    assert_eq!(eval_expr("(seq [])"), "nil");
    assert_eq!(eval_expr("(seq nil)"), "nil");
}

#[test]
fn cljs_seq_range() {
    assert_output("(println (str (range 5)))", "(0 1 2 3 4)");
    assert_output("(println (str (range 2 5)))", "(2 3 4)");
    assert_output("(println (str (range 0 10 3)))", "(0 3 6 9)");
}

#[test]
fn cljs_seq_repeat_bounded() {
    assert_output("(println (str (repeat 3 :a)))", "(:a :a :a)");
}

#[test]
fn cljs_seq_take() {
    assert_output("(println (str (take 3 [1 2 3 4 5])))", "(1 2 3)");
    assert_eq!(eval_expr("(str (take 0 [1 2 3]))"), "\"\"");
    assert_eq!(eval_expr("(str (take 10 [1 2 3]))"), "\"(1 2 3)\"");
}

#[test]
fn cljs_seq_drop() {
    assert_output("(println (str (drop 3 [1 2 3 4 5])))", "(4 5)");
    assert_eq!(eval_expr("(str (drop 0 [1 2 3]))"), "\"(1 2 3)\"");
    assert_eq!(eval_expr("(str (drop 10 [1 2 3]))"), "\"\"");
}

#[test]
fn cljs_seq_take_while() {
    assert_output(
        "(println (str (take-while #(< % 4) [1 2 3 4 5])))",
        "(1 2 3)",
    );
}

#[test]
fn cljs_seq_drop_while() {
    assert_output(
        "(println (str (drop-while #(< % 4) [1 2 3 4 5])))",
        "(4 5)",
    );
}

/// CLJS test-725
#[test]
fn cljs_seq_drop_while_partial() {
    assert_eq!(
        eval_expr("(str (apply vector (drop-while (partial = 1) [1 2 3])))"),
        "\"[2 3]\""
    );
}

/// CLJS test-724
#[test]
fn cljs_seq_rest_rest_rest_range() {
    assert_eq!(eval_expr("(first (rest (rest (rest (range 3)))))"), "nil");
}

// ============================================================================
// HIGHER-ORDER FUNCTIONS
// ============================================================================

#[test]
fn cljs_hof_map_basic() {
    assert_output("(println (str (map inc [0 1 2])))", "(1 2 3)");
}

#[test]
fn cljs_hof_map_with_fn_literal() {
    assert_output("(println (str (map #(+ 1 %) [0 1 2])))", "(1 2 3)");
}

#[test]
fn cljs_hof_map_multiple_colls() {
    assert_output(
        "(println (str (map + [1 2 3] [10 20 30])))",
        "(11 22 33)",
    );
    assert_eq!(eval_expr("(str (map + [1 2 3] [4 5 6] [7 8 9]))"), "\"(12 15 18)\"");
}

#[test]
fn cljs_hof_map_with_keywords() {
    assert_eq!(eval_expr("(str (map :a [{:a 1} {:a 2} {:a 3}]))"), "\"(1 2 3)\"");
    assert_eq!(eval_expr("(str (map :name [{:name \"a\"} {:name \"b\"} {:name \"c\"}]))"), "\"(\"a\" \"b\" \"c\")\"");
}

#[test]
fn cljs_hof_map_with_vector() {
    assert_eq!(eval_expr("(str (map vector [:a :b :c] [1 2 3]))"), "\"([:a 1] [:b 2] [:c 3])\"");
}

#[test]
fn cljs_hof_map_inc_range() {
    assert_eq!(eval_expr("(str (map inc (range 5)))"), "\"(1 2 3 4 5)\"");
}

#[test]
fn cljs_hof_map_squared() {
    assert_eq!(eval_expr("(str (map #(* % %) [1 2 3 4 5]))"), "\"(1 4 9 16 25)\"");
}

#[test]
fn cljs_hof_map_constantly() {
    assert_eq!(eval_expr("(str (map (constantly 42) [1 2 3]))"), "\"(42 42 42)\"");
}

#[test]
fn cljs_hof_map_name() {
    assert_eq!(eval_expr("(str (map name [:a :b :c]))"), "\"(\"a\" \"b\" \"c\")\"");
}

#[test]
fn cljs_hof_map_str() {
    assert_eq!(eval_expr("(str (map str [1 2 3]))"), "\"(\"1\" \"2\" \"3\")\"");
}

#[test]
fn cljs_hof_map_count() {
    assert_eq!(eval_expr("(str (map count [[1] [1 2] [1 2 3]]))"), "\"(1 2 3)\"");
}

#[test]
fn cljs_hof_map_first() {
    assert_eq!(eval_expr("(str (map first [[1 2] [3 4] [5 6]]))"), "\"(1 3 5)\"");
}

#[test]
fn cljs_hof_map_rest() {
    assert_eq!(eval_expr("(str (map rest [[1 2 3] [4 5 6]]))"), "\"((2 3) (5 6))\"");
}

#[test]
fn cljs_hof_map_nested() {
    assert_eq!(eval_expr("(str (map #(map inc %) [[1 2] [3 4]]))"), "\"((2 3) (4 5))\"");
}

#[test]
fn cljs_hof_filter() {
    assert_output(
        "(println (str (filter even? [1 2 3 4 5 6])))",
        "(2 4 6)",
    );
}

#[test]
fn cljs_hof_filter_odd() {
    assert_eq!(eval_expr("(str (filter odd? [1 2 3]))"), "\"(1 3)\"");
}

#[test]
fn cljs_hof_filter_empty() {
    assert_eq!(eval_expr("(str (filter (fn [x] false) [1 2 3]))"), "\"\"");
}

#[test]
fn cljs_hof_remove() {
    assert_output(
        "(println (str (remove even? [1 2 3 4 5 6])))",
        "(1 3 5)",
    );
}

#[test]
fn cljs_hof_keep() {
    assert_output("(println (str (keep odd? [0 1 2])))", "(false true false)");
    assert_eq!(eval_expr("(str (keep identity [1 nil 2 nil 3]))"), "\"(1 2 3)\"");
}

#[test]
fn cljs_hof_map_indexed() {
    assert_output(
        "(println (str (map-indexed #(do [%1 %2]) [1 2 3])))",
        "([0 1] [1 2] [2 3])",
    );
    assert_eq!(eval_expr("(str (map-indexed vector [:a :b :c]))"), "\"([0 :a] [1 :b] [2 :c])\"");
}

#[test]
fn cljs_hof_mapcat() {
    assert_output(
        "(println (str (mapcat #(vector % (* % %)) [1 2 3])))",
        "(1 1 2 4 3 9)",
    );
    assert_eq!(
        eval_expr("(str (mapcat reverse [[3 2 1] [6 5 4]]))"),
        "\"(1 2 3 4 5 6)\""
    );
}

#[test]
fn cljs_hof_reduce_basic() {
    assert_eq!(eval_expr("(reduce + [1 2 3 4 5])"), "15");
    assert_eq!(eval_expr("(reduce + 10 [1 2 3])"), "16");
    assert_eq!(eval_expr("(reduce + 0 [])"), "0");
    assert_eq!(eval_expr("(reduce * 1 [])"), "1");
}

#[test]
fn cljs_hof_reduce_with_range() {
    assert_eq!(eval_expr("(reduce + 0 (range 10))"), "45");
    assert_eq!(eval_expr("(reduce + (range 1 6))"), "15");
}

#[test]
fn cljs_hof_reduce_with_map_filter() {
    assert_eq!(
        eval_expr("(reduce + 0 (filter odd? (map inc [0 1 2 3 4])))"),
        "9"
    );
}

#[test]
fn cljs_hof_reduce_conj() {
    assert_eq!(eval_expr("(str (reduce conj [] [1 2 3]))"), "\"[1 2 3]\"");
    assert_eq!(
        eval_expr("(str (reduce (fn [acc x] (conj acc (* x x))) [] [1 2 3 4]))"),
        "\"[1 4 9 16]\""
    );
}

#[test]
fn cljs_hof_reduce_kv_map() {
    assert_eq!(eval_expr("(reduce-kv (fn [acc k v] (+ acc v)) 0 {:a 1 :b 2 :c 3})"), "6");
}

#[test]
fn cljs_hof_reduce_kv_vector() {
    assert_eq!(eval_expr("(reduce-kv (fn [acc k v] (+ acc v)) 0 [10 20 30])"), "60");
}

#[test]
fn cljs_hof_apply() {
    assert_eq!(eval_expr("(apply + [1 2 3])"), "6");
    assert_eq!(eval_expr("(apply + 1 2 [3 4])"), "10");
    assert_eq!(eval_expr("(apply + 1 2 3 [4 5])"), "15");
    assert_eq!(eval_expr("(apply + [])"), "0");
}

#[test]
fn cljs_hof_apply_str() {
    assert_eq!(eval_expr("(apply str [\"a\" \"b\" \"c\"])"), "\"abc\"");
}

#[test]
fn cljs_hof_apply_vector() {
    assert_eq!(eval_expr("(apply vector 1 2 [3 4])"), "[1 2 3 4]");
}

#[test]
fn cljs_hof_apply_list() {
    assert_eq!(eval_expr("(apply list 1 2 [3 4])"), "(1 2 3 4)");
}

#[test]
fn cljs_hof_apply_max_min() {
    assert_eq!(eval_expr("(apply max [1 2 3])"), "3");
    assert_eq!(eval_expr("(apply min [5 3 7])"), "3");
}

#[test]
fn cljs_hof_some() {
    assert_eq!(eval_expr("(some even? [1 3 5 6])"), "true");
    assert_eq!(eval_expr("(some even? [1 3 5])"), "nil");
}

#[test]
fn cljs_hof_some_identity() {
    assert_eq!(eval_expr("(some identity [nil nil 3 nil])"), "3");
    assert_eq!(eval_expr("(some identity [nil nil nil])"), "nil");
}

#[test]
fn cljs_hof_some_set_as_pred() {
    assert_eq!(eval_expr("(some #{1 2} [3 4 1 5])"), "1");
    assert_eq!(eval_expr("(some #{1 2} [3 4 5])"), "nil");
    assert_eq!(eval_expr("(some #{:a} [:b :c :a :d])"), ":a");
}

#[test]
fn cljs_hof_some_keyword_as_pred() {
    assert_eq!(eval_expr("(some :a [{:b 1} {:a 2} {:c 3}])"), "2");
}

#[test]
fn cljs_hof_every() {
    assert_eq!(eval_expr("(every? even? [2 4 6])"), "true");
    assert_eq!(eval_expr("(every? even? [2 3 6])"), "false");
    assert_eq!(eval_expr("(every? :a [{:a 1} {:a 2} {:a 3}])"), "true");
}

#[test]
fn cljs_hof_not_every() {
    assert_eq!(eval_expr("(not-every? even? [2 4 6])"), "false");
    assert_eq!(eval_expr("(not-every? even? [2 3 6])"), "true");
}

#[test]
fn cljs_hof_not_any() {
    assert_eq!(eval_expr("(not-any? even? [1 3 5])"), "true");
    assert_eq!(eval_expr("(not-any? even? [1 2 5])"), "false");
}

// ============================================================================
// IDENTITY, CONSTANTLY, COMP, PARTIAL, COMPLEMENT, JUXT
// ============================================================================

#[test]
fn cljs_hof_identity() {
    assert_eq!(eval_expr("(identity 42)"), "42");
    assert_eq!(eval_expr("(identity nil)"), "nil");
    assert_eq!(eval_expr("(identity :a)"), ":a");
}

#[test]
fn cljs_hof_constantly() {
    assert_eq!(eval_expr("((constantly 5) 1 2 3)"), "5");
    assert_eq!(eval_expr("((constantly nil) 1)"), "nil");
}

#[test]
fn cljs_hof_comp() {
    assert_eq!(eval_expr("((comp inc inc inc) 0)"), "3");
    assert_eq!(eval_expr("((comp str inc) 1)"), "\"2\"");
    assert_eq!(eval_expr("((comp str inc inc) 0)"), "\"2\"");
    assert_eq!(eval_expr("((comp) 42)"), "42");
}

#[test]
fn cljs_hof_partial() {
    assert_eq!(eval_expr("((partial + 10) 5)"), "15");
    assert_eq!(eval_expr("((partial + 1 2 3) 4)"), "10");
    assert_eq!(eval_expr("(apply (partial + 1 2) [3 4])"), "10");
}

#[test]
fn cljs_hof_complement() {
    assert_eq!(eval_expr("((complement nil?) 1)"), "true");
    assert_eq!(eval_expr("((complement nil?) nil)"), "false");
    assert_eq!(eval_expr("((complement even?) 3)"), "true");
    assert_eq!(eval_expr("((complement even?) 4)"), "false");
}

#[test]
fn cljs_hof_juxt() {
    assert_eq!(eval_expr("(nth ((juxt inc dec) 1) 0)"), "2");
    assert_eq!(eval_expr("(nth ((juxt inc dec) 1) 1)"), "0");
    assert_eq!(eval_expr("(str ((juxt inc dec) 1))"), "\"[2 0]\"");
    assert_eq!(eval_expr("(str ((juxt + - *) 3 4))"), "\"[7 -1 12]\"");
    assert_eq!(eval_expr("(str ((juxt first last) [1 2 3]))"), "\"[1 3]\"");
}

// ============================================================================
// CONTROL FLOW
// ============================================================================

#[test]
fn cljs_control_if_basic() {
    assert_eq!(eval_expr("(if true 10 20)"), "10");
    assert_eq!(eval_expr("(if false 10 20)"), "20");
    assert_eq!(eval_expr("(if nil 10 20)"), "20");
}

#[test]
fn cljs_control_if_truthy_values() {
    // 0 and "" are truthy in Clojure
    assert_eq!(eval_expr("(if 0 \"yes\" \"no\")"), "\"yes\"");
    assert_eq!(eval_expr("(if \"\" \"yes\" \"no\")"), "\"yes\"");
    assert_eq!(eval_expr("(if (> 1 0) \"yes\" \"no\")"), "\"yes\"");
}

#[test]
fn cljs_control_if_no_else() {
    assert_eq!(eval_expr("(if true 1)"), "1");
    assert_eq!(eval_expr("(if false 1)"), "nil");
}

#[test]
fn cljs_control_when() {
    assert_eq!(eval_expr("(when true 0 1 2)"), "2");
    assert_eq!(eval_expr("(when false 1)"), "nil");
}

#[test]
fn cljs_control_cond() {
    assert_eq!(eval_expr("(cond true 1 :else 2)"), "1");
    assert_eq!(eval_expr("(cond false 1 :else 2)"), "2");
    assert_eq!(eval_expr("(cond (= 1 2) :a (= 2 3) :b (= 3 3) :c :else :d)"), ":c");
}

#[test]
fn cljs_control_cond_type_check() {
    assert_eq!(eval_expr("(let [x 2] (cond (string? x) 1 (integer? x) 2))"), "2");
}

#[test]
fn cljs_control_if_let() {
    assert_eq!(eval_expr("(if-let [x 42] x 0)"), "42");
    assert_eq!(eval_expr("(if-let [x nil] 1 2)"), "2");
    assert_eq!(eval_expr("(if-let [x false] 1 2)"), "2");
    assert_eq!(eval_expr("(if-let [x (seq [1 2 3])] (first x) :empty)"), "1");
    assert_eq!(eval_expr("(if-let [x (seq [])] (first x) :empty)"), ":empty");
}

#[test]
fn cljs_control_when_let() {
    assert_eq!(eval_expr("(when-let [x 42] x)"), "42");
    assert_eq!(eval_expr("(when-let [x nil] 1)"), "nil");
    assert_eq!(eval_expr("(when-let [x false] 1)"), "nil");
    assert_eq!(eval_expr("(when-let [x (first [42])] x)"), "42");
}

#[test]
fn cljs_control_do() {
    assert_eq!(eval_expr("(do)"), "nil");
    assert_eq!(eval_expr("(do 1 2 3)"), "3");
    assert_eq!(eval_expr("(do 1 2 nil)"), "nil");
}

#[test]
fn cljs_control_comment() {
    assert_eq!(eval_expr("(comment \"anything\")"), "nil");
    assert_eq!(eval_expr("(comment 1)"), "nil");
    assert_eq!(eval_expr("(comment (+ 1 2 (* 3 4)))"), "nil");
}

// ============================================================================
// THREADING MACROS
// ============================================================================

#[test]
fn cljs_thread_first_basic() {
    assert_eq!(eval_expr("(-> 1 inc inc inc)"), "4");
    assert_eq!(eval_expr("(-> 3 inc inc inc)"), "6");
}

#[test]
fn cljs_thread_first_with_collections() {
    assert_eq!(eval_expr("(-> {:a 1} (assoc :b 2) (assoc :c 3) count)"), "3");
    assert_eq!(eval_expr("(-> {:a 1} (assoc :b 2) :b)"), "2");
    assert_eq!(eval_expr("(-> {:a 1 :b 2} (assoc :c 3) (dissoc :a) count)"), "2");
}

#[test]
fn cljs_thread_first_str() {
    assert_eq!(eval_expr("(-> \"hello\" count)"), "5");
}

#[test]
fn cljs_thread_first_vector() {
    assert_eq!(eval_expr("(-> [1 2 3] (conj 4) count)"), "4");
}

#[test]
fn cljs_thread_last_basic() {
    assert_output(
        r#"(println (->> ["foo" "baaar" "baaaaaz"] (map count) (apply max)))"#,
        "7",
    );
}

#[test]
fn cljs_thread_last_pipeline() {
    assert_eq!(
        eval_expr("(->> [1 2 3 4 5] (filter odd?) (map inc) (reduce +))"),
        "12"
    );
    assert_eq!(
        eval_expr("(->> (range 10) (filter even?) (map #(* % %)) (reduce +))"),
        "120"
    );
    assert_eq!(
        eval_expr("(->> (range 1 11) (reduce +))"),
        "55"
    );
}

#[test]
fn cljs_thread_last_complex() {
    assert_eq!(
        eval_expr("(->> (range 20) (filter even?) (map #(* % %)) (take 5) (reduce +))"),
        "120"
    );
    assert_eq!(
        eval_expr("(->> [1 2 3 4 5 6 7 8 9 10] (filter even?) (map #(* % %)) (reduce +))"),
        "220"
    );
}

// ============================================================================
// LET BINDING
// ============================================================================

#[test]
fn cljs_let_basic() {
    assert_eq!(eval_expr("(let [x 1] x)"), "1");
    assert_eq!(eval_expr("(let [x 1 y 2] (+ x y))"), "3");
    assert_eq!(eval_expr("(let [x 1 y (+ x x)] y)"), "2");
}

#[test]
fn cljs_let_many_bindings() {
    assert_eq!(eval_expr("(let [a 1 b 2 c 3 d 4 e 5] (+ a b c d e))"), "15");
}

#[test]
fn cljs_let_nested() {
    assert_eq!(eval_expr("(let [x 1] (let [y 2] (+ x y)))"), "3");
}

#[test]
fn cljs_let_multiple_body() {
    assert_eq!(eval_expr("(let [x 2] 1 2 3 x)"), "2");
}

#[test]
fn cljs_let_shadow() {
    assert_output("(let [x 1] (println (let [x 2] x)) (println x))", "2\n1");
}

#[test]
fn cljs_let_with_map() {
    assert_eq!(eval_expr("(let [m {:a 1 :b 2 :c 3}] (+ (:a m) (:b m) (:c m)))"), "6");
}

// ============================================================================
// FN AND CLOSURES
// ============================================================================

#[test]
fn cljs_fn_literal_basic() {
    assert_eq!(eval_expr("(#(+ 1 %) 1)"), "2");
    assert_eq!(eval_expr("(#(* % %) 5)"), "25");
    assert_eq!(eval_expr("(#(+ %1 %2) 3 4)"), "7");
}

#[test]
fn cljs_fn_literal_three_args() {
    assert_eq!(eval_expr("(str (#(vector %1 %2 %3) 1 2 3))"), "\"[1 2 3]\"");
}

#[test]
fn cljs_fn_literal_rest() {
    assert_eq!(eval_expr("(str (#(do %&) 1 2 3))"), "\"(1 2 3)\"");
}

#[test]
fn cljs_fn_rest_params() {
    assert_eq!(eval_expr("(str ((fn [x & xs] xs) 1 2 3))"), "\"(2 3)\"");
}

#[test]
fn cljs_fn_multi_arity() {
    assert_eq!(eval_expr("((fn ([x] x) ([x y] y)) 1)"), "1");
    assert_eq!(eval_expr("((fn ([x] x) ([x y] y)) 1 2)"), "2");
}

#[test]
fn cljs_fn_variadic_vs_fixed() {
    assert_eq!(
        eval_expr("((fn ([x & xs] \"variadic\") ([x] \"otherwise\")) 1)"),
        "\"otherwise\""
    );
    assert_eq!(
        eval_expr("((fn ([x] \"otherwise\") ([x & xs] \"variadic\")) 1 2)"),
        "\"variadic\""
    );
}

#[test]
fn cljs_fn_apply_with_rest() {
    assert_output(
        "(println (str (apply (fn [x & xs] xs) 1 2 [3 4])))",
        "(2 3 4)",
    );
}

#[test]
fn cljs_fn_closures() {
    assert_eq!(eval_expr("(let [x 10 f (fn [y] (+ x y))] (f 5))"), "15");
    assert_eq!(eval_expr("((let [x 10] (fn [y] (+ x y))) 5)"), "15");
}

#[test]
fn cljs_fn_closure_nested() {
    assert_eq!(
        eval_expr("(let [x 1 y 2] ((fn [] (let [g (fn [] y)] (+ x (g))))))"),
        "3"
    );
}

#[test]
fn cljs_fn_deeply_nested() {
    assert_eq!(
        eval_expr("(let [f (fn [x] (fn [y] (fn [z] (+ x y z))))] (((f 1) 2) 3))"),
        "6"
    );
}

/// CLJS test-3386
#[test]
fn cljs_fn_multi_arity_nil_rest() {
    assert_eq!(
        eval_expr("(do (defn tf ([x] x) ([_ _ & zs] zs)) (nil? (tf 1 2)))"),
        "true"
    );
    assert_eq!(
        eval_expr("(do (defn tf ([x] x) ([_ _ & zs] zs)) (str (tf 1 2 3 4)))"),
        "\"(3 4)\""
    );
}

// ============================================================================
// DEFN
// ============================================================================

#[test]
fn cljs_defn_basic() {
    assert_output(
        r#"(defn foo "increment" [x] (inc x)) (println (foo 1))"#,
        "2",
    );
}

#[test]
fn cljs_defn_multi_arity() {
    assert_eq!(
        eval_expr("(do (defn f ([x] x) ([x y] (+ x y))) (+ (f 1) (f 2 3)))"),
        "6"
    );
}

#[test]
fn cljs_defn_variadic() {
    assert_eq!(eval_expr("(do (defn vari [x & xs] (count xs)) (vari 1 2 3 4))"), "3");
    assert_eq!(eval_expr("(do (defn vari [x & xs] (str xs)) (vari 1 2 3))"), "\"(2 3)\"");
}

#[test]
fn cljs_defn_all_arities() {
    assert_eq!(
        eval_expr("(do (defn f ([] 0) ([x] x) ([x y] (+ x y)) ([x y & more] (apply + x y more))) (+ (f) (f 1) (f 2 3) (f 4 5 6 7)))"),
        "28"
    );
}

#[test]
fn cljs_defn_private() {
    assert_output("(defn- foo [] 1) (println (foo))", "1");
}

#[test]
fn cljs_defn_square() {
    assert_eq!(eval_expr("(do (defn square [x] (* x x)) (square 7))"), "49");
}

#[test]
fn cljs_defn_compose() {
    assert_eq!(
        eval_expr("(do (defn double [x] (* 2 x)) (defn add1 [x] (+ 1 x)) (double (add1 3)))"),
        "8"
    );
}

// ============================================================================
// DEF
// ============================================================================

#[test]
fn cljs_def_basic() {
    assert_output(r#"(def foo "nice val") (println foo)"#, "nice val");
}

#[test]
fn cljs_def_with_docstring() {
    assert_output(r#"(def foo) (def foo "docstring" 2) (println foo)"#, "2");
}

#[test]
fn cljs_def_multiple() {
    assert_eq!(eval_expr("(do (def x 1) (def y 2) (+ x y))"), "3");
}

// ============================================================================
// RECUR AND LOOP
// ============================================================================

#[test]
fn cljs_recur_basic() {
    assert_output(
        "(defn hello [x] (if (< x 10000) (recur (inc x)) x)) (println (hello 0))",
        "10000",
    );
}

#[test]
fn cljs_recur_countdown() {
    assert_eq!(
        eval_expr("(do (defn countdown [n] (if (zero? n) \"done\" (recur (dec n)))) (countdown 100))"),
        "\"done\""
    );
}

#[test]
fn cljs_recur_sum() {
    assert_eq!(
        eval_expr("(do (defn sum [n acc] (if (zero? n) acc (recur (dec n) (+ acc n)))) (sum 100 0))"),
        "5050"
    );
}

#[test]
fn cljs_recur_fibonacci() {
    assert_eq!(
        eval_expr("(do (defn fib [n a b] (if (zero? n) a (recur (dec n) b (+ a b)))) (fib 10 0 1))"),
        "55"
    );
}

#[test]
fn cljs_recur_variadic() {
    assert_output(
        "(println (str ((fn [& args] (if-let [x (next args)] (recur x) args)) 1 2 3 4)))",
        "(4)",
    );
}

#[test]
fn cljs_recur_variadic_with_fixed() {
    assert_output(
        "(println (str ((fn [x & args] (if-let [x (next args)] (recur x x) x)) nil 2 3 4)))",
        "(4)",
    );
}

#[test]
fn cljs_loop_basic() {
    assert_eq!(
        eval_expr("(loop [x 0] (if (< x 10) (recur (inc x)) x))"),
        "10"
    );
}

#[test]
fn cljs_loop_accumulator() {
    assert_eq!(
        eval_expr("(loop [i 0 acc 0] (if (> i 10) acc (recur (inc i) (+ acc i))))"),
        "55"
    );
}

#[test]
fn cljs_loop_factorial() {
    assert_eq!(
        eval_expr("(loop [x 5 acc 1] (if (zero? x) acc (recur (dec x) (* acc x))))"),
        "120"
    );
}

#[test]
fn cljs_loop_conj_list() {
    assert_output(
        "(println (str (loop [l (list 2 1) c (count l)] (if (> c 4) l (recur (conj l (inc c)) (inc c))))))",
        "(5 4 3 2 1)",
    );
}

#[test]
fn cljs_loop_let_shadow() {
    assert_eq!(eval_expr("(let [x 1] (loop [x (inc x)] x))"), "2");
}

#[test]
fn cljs_loop_nested() {
    assert_eq!(
        eval_expr("(loop [i 0 total 0] (if (> i 3) total (recur (inc i) (+ total (loop [j 0 s 0] (if (> j 3) s (recur (inc j) (+ s j))))))))"),
        "24"
    );
}

// ============================================================================
// TRY/CATCH/THROW
// ============================================================================

#[test]
fn cljs_try_returns_body() {
    assert_eq!(eval_expr("(try 1 2 3)"), "3");
}

#[test]
fn cljs_try_returns_quoted() {
    assert_eq!(eval_expr("(try 'hello)"), "hello");
}

#[test]
fn cljs_try_nil_in_body() {
    assert_eq!(eval_expr("(try 1 2 nil)"), "nil");
}

#[test]
fn cljs_try_catch_thrown_string() {
    assert_eq!(eval_expr("(try (throw \"err\") (catch Exception e \"caught\"))"), "\"caught\"");
}

#[test]
fn cljs_try_no_exception() {
    assert_eq!(eval_expr("(try (+ 1 2))"), "3");
    assert_eq!(eval_expr("(try 1 (catch Exception e \"caught\"))"), "1");
}

#[test]
fn cljs_try_finally() {
    assert_eq!(eval_expr("(try 42 (finally nil))"), "42");
}

// ============================================================================
// PROTOCOLS AND DEFTYPE
// ============================================================================

#[test]
fn cljs_protocol_basic() {
    assert_eq!(
        eval_expr("(do (defprotocol IGreet (greet [this])) (deftype Greeter [] IGreet (greet [this] \"hello\")) (greet (Greeter.)))"),
        "\"hello\""
    );
}

#[test]
fn cljs_protocol_with_field() {
    assert_eq!(
        eval_expr("(do (defprotocol ISpeak (speak [this])) (deftype Dog [name] ISpeak (speak [this] (str \"Woof, I am \" name))) (speak (Dog. \"Rex\")))"),
        "\"Woof, I am Rex\""
    );
}

#[test]
fn cljs_protocol_with_computation() {
    assert_eq!(
        eval_expr("(do (defprotocol IArea (area [this])) (deftype Circle [r] IArea (area [this] (* 3.14 r r))) (area (Circle. 5)))"),
        "78.5"
    );
}

#[test]
fn cljs_protocol_multiple_methods() {
    assert_eq!(
        eval_expr("(do (defprotocol ICalc (add [this x]) (mul [this x])) (deftype Num [n] ICalc (add [this x] (+ n x)) (mul [this x] (* n x))) (let [c (Num. 10)] (+ (add c 5) (mul c 3))))"),
        "45"
    );
}

#[test]
fn cljs_protocol_satisfies() {
    assert_eq!(
        eval_expr("(do (defprotocol IFoo (foo [this])) (deftype Bar [] IFoo (foo [this] 42)) (satisfies? IFoo (Bar.)))"),
        "true"
    );
}

#[test]
fn cljs_deftype_wrapper() {
    assert_eq!(
        eval_expr("(do (defprotocol ILen (my-len [this])) (deftype Wrapper [items] ILen (my-len [this] (count items))) (my-len (Wrapper. [1 2 3])))"),
        "3"
    );
}

#[test]
fn cljs_deftype_point() {
    assert_eq!(
        eval_expr("(do (defprotocol IShow (show [this])) (deftype Point [x y] IShow (show [this] (str \"(\" x \",\" y \")\"))) (show (Point. 3 4)))"),
        "\"(3,4)\""
    );
}

// ============================================================================
// IFn - COLLECTIONS AS FUNCTIONS
// ============================================================================

/// CLJS test calling IFns
#[test]
fn cljs_ifn_map_as_fn_found() {
    assert_eq!(eval_expr("({:a 1} :a 3)"), "1");
}

#[test]
fn cljs_ifn_map_as_fn_default() {
    assert_eq!(eval_expr("({:a 1} 2 3)"), "3");
}

#[test]
fn cljs_ifn_hashmap_as_fn() {
    assert_eq!(eval_expr("((hash-map :a 1) :a 3)"), "1");
}

#[test]
fn cljs_ifn_set_as_fn() {
    assert_eq!(eval_expr("(#{:a :b :c} :a)"), ":a");
    assert_eq!(eval_expr("(#{:a :b :c} :d)"), "nil");
}

#[test]
fn cljs_ifn_vector_as_fn() {
    assert_eq!(eval_expr("([10 20 30] 1)"), "20");
}

#[test]
fn cljs_ifn_fn_from_map() {
    assert_eq!(eval_expr("((get {:foo identity} :foo) 1)"), "1");
}

#[test]
fn cljs_ifn_keyword_as_fn() {
    assert_eq!(eval_expr("(:a {:a 42 :b 99})"), "42");
    assert_eq!(eval_expr("(:c {:a 1})"), "nil");
    assert_eq!(eval_expr("(:c {:a 1} :default)"), ":default");
}

// ============================================================================
// PRINTLN OUTPUT TESTS
// ============================================================================

#[test]
fn cljs_println_basic() {
    assert_output("(println 42)", "42");
    assert_output("(println \"hello\")", "hello");
    assert_output("(println nil)", "nil");
    assert_output("(println true)", "true");
    assert_output("(println false)", "false");
    assert_output("(println :kw)", ":kw");
}

#[test]
fn cljs_println_collections() {
    assert_output("(println [1 2 3])", "[1 2 3]");
    assert_output("(println (list 1 2))", "(1 2)");
}

#[test]
fn cljs_println_multiple_args() {
    assert_output("(println \"a\" \"b\" \"c\")", "a b c");
}

#[test]
fn cljs_println_returns_nil() {
    assert_eq!(eval_expr("(nil? (println \"hi\"))"), "hi\ntrue");
}

#[test]
fn cljs_do_side_effects() {
    assert_output("(do (println 1) (println 2))", "1\n2");
}

// ============================================================================
// CONTAINS? (CLJS test-contains?)
// ============================================================================

#[test]
fn cljs_contains_map() {
    assert_eq!(eval_expr("(contains? {:a 1 :b 2} :a)"), "true");
    assert_eq!(eval_expr("(contains? {:a 1 :b 2} :z)"), "false");
}

#[test]
fn cljs_contains_vector() {
    assert_eq!(eval_expr("(contains? [5 6 7] 1)"), "true");
    assert_eq!(eval_expr("(contains? [5 6 7] 2)"), "true");
    assert_eq!(eval_expr("(contains? [5 6 7] 3)"), "false");
}

#[test]
fn cljs_contains_nil() {
    assert_eq!(eval_expr("(contains? nil 42)"), "false");
}

#[test]
fn cljs_contains_set() {
    assert_eq!(eval_expr("(contains? #{1 2 3} 2)"), "true");
    assert_eq!(eval_expr("(contains? #{1 2 3} 4)"), "false");
}

// ============================================================================
// DOTIMES
// ============================================================================

#[test]
fn cljs_dotimes_basic() {
    assert_output("(dotimes [i 3] (println i))", "0\n1\n2");
}

#[test]
fn cljs_dotimes_returns_nil() {
    assert_eq!(eval_expr("(dotimes [i 3] i)"), "nil");
}

// ============================================================================
// GENSYM
// ============================================================================

#[test]
fn cljs_gensym() {
    // Just verify it returns something
    let result = eval_expr("(gensym)");
    assert!(result.starts_with("G__"), "gensym should start with G__, got: {}", result);
}

// ============================================================================
// IDENTICAL?
// ============================================================================

#[test]
fn cljs_identical() {
    assert_eq!(eval_expr("(identical? nil nil)"), "true");
    assert_eq!(eval_expr("(identical? 1 1)"), "true");
}

// ============================================================================
// HASH
// ============================================================================

#[test]
fn cljs_hash_number() {
    assert_eq!(eval_expr("(hash 42)"), "42");
}

// ============================================================================
// ASSOC ON NIL
// ============================================================================

#[test]
fn cljs_assoc_nil() {
    assert_eq!(eval_expr("(count (assoc nil :a 1))"), "1");
}

// ============================================================================
// VARIABLE NAMES MATCHING MACROS
// ============================================================================

#[test]
fn cljs_var_named_merge() {
    assert_output("(defn foo [merge] merge) (println (foo true))", "true");
}

#[test]
fn cljs_var_named_comment() {
    assert_output("(defn foo [comment] comment) (println (foo true))", "true");
}

// ============================================================================
// EMPTY FUNCTION
// ============================================================================

#[test]
fn cljs_empty_vector() {
    assert_eq!(eval_expr("(str (empty [1 2 3]))"), "\"[]\"");
}

#[test]
fn cljs_empty_list() {
    assert_eq!(eval_expr("(str (empty (list 1 2 3)))"), "\"()\"");
}

#[test]
fn cljs_empty_map() {
    assert_eq!(eval_expr("(str (empty {:a 1}))"), "\"{}\"");
}

#[test]
fn cljs_empty_set() {
    assert_eq!(eval_expr("(str (empty #{1 2}))"), "\"#{}\"");
}

// ============================================================================
// FN LITERAL MAP/REDUCE
// ============================================================================

#[test]
fn cljs_fn_literal_with_map() {
    assert_output("(println (str (map #(+ 1 %) [0 1 2])))", "(1 2 3)");
}

#[test]
fn cljs_fn_literal_identity() {
    assert_output("(println (str (map #(do %) [1 2 3])))", "(1 2 3)");
}

#[test]
fn cljs_fn_literal_map_indexed() {
    assert_output(
        "(println (str (map-indexed #(do [%1 %2]) [1 2 3])))",
        "([0 1] [1 2] [2 3])",
    );
}

#[test]
fn cljs_fn_literal_rest_args() {
    assert_output(
        "(println (str (apply #(do %&) [1 2 3])))",
        "(1 2 3)",
    );
}

// ============================================================================
// UNSUPPORTED FEATURES (marked #[ignore])
// ============================================================================

#[test]
fn cljs_some_thread_first() {
    assert_eq!(eval_expr("(some-> nil)"), "nil");
    assert_eq!(eval_expr("(some-> 0)"), "0");
    assert_eq!(eval_expr("(some-> 1 (- 2))"), "-1");
}

#[test]
fn cljs_some_thread_last() {
    assert_eq!(eval_expr("(some->> nil)"), "nil");
    assert_eq!(eval_expr("(some->> 0)"), "0");
    assert_eq!(eval_expr("(some->> 1 (- 2))"), "1");
}

#[test]
fn cljs_cond_thread_first() {
    assert_eq!(eval_expr("(cond-> 0)"), "0");
    assert_eq!(eval_expr("(cond-> 0 true inc true (- 2))"), "-1");
    assert_eq!(eval_expr("(cond-> 0 false inc)"), "0");
}

#[test]
fn cljs_cond_thread_last() {
    assert_eq!(eval_expr("(cond->> 0)"), "0");
    assert_eq!(eval_expr("(cond->> 0 true inc true (- 2))"), "1");
    assert_eq!(eval_expr("(cond->> 0 false inc)"), "0");
}

#[test]
fn cljs_as_thread() {
    assert_eq!(eval_expr("(as-> 0 x)"), "0");
    assert_eq!(eval_expr("(as-> 0 x (inc x))"), "1");
}

#[test]
fn cljs_case_basic() {
    assert_eq!(eval_expr("(case 1, 1 :one, 2 :two, :default)"), ":one");
    assert_eq!(eval_expr("(case 3, 1 :one, 2 :two, :default)"), ":default");
}

#[test]
fn cljs_for_basic() {
    assert_eq!(eval_expr("(str (for [x [1 2 3]] (* x x)))"), "\"(1 4 9)\"");
}

#[test]
fn cljs_atom_basic() {
    assert_eq!(eval_expr("(let [a (atom 0)] (swap! a inc) @a)"), "1");
}

#[test]
fn cljs_compare_basic() {
    assert_eq!(eval_expr("(compare 1 2)"), "-1");
    assert_eq!(eval_expr("(compare 2 1)"), "1");
    assert_eq!(eval_expr("(compare 1 1)"), "0");
}

#[test]
fn cljs_boolean_fn() {
    assert_eq!(eval_expr("(boolean true)"), "true");
    assert_eq!(eval_expr("(boolean false)"), "false");
    assert_eq!(eval_expr("(boolean nil)"), "false");
    assert_eq!(eval_expr("(boolean 1)"), "true");
}

#[test]
fn cljs_subs() {
    assert_eq!(eval_expr("(subs \"hello\" 1)"), "\"ello\"");
    assert_eq!(eval_expr("(subs \"hello\" 1 3)"), "\"el\"");
}

#[test]
fn cljs_keyword_constructor() {
    assert_eq!(eval_expr("(keyword \"foo\")"), ":foo");
    assert_eq!(eval_expr("(keyword \"ns\" \"name\")"), ":ns/name");
}

#[test]
#[ignore] // sorted-set not implemented
fn cljs_sorted_set() {
    assert_eq!(eval_expr("(str (sorted-set 3 1 2))"), "\"#{1 2 3}\"");
}

#[test]
#[ignore] // sorted-map not implemented
fn cljs_sorted_map() {
    assert_eq!(eval_expr("(str (sorted-map :b 2 :a 1))"), "\"{:a 1, :b 2}\"");
}

#[test]
fn cljs_interpose() {
    assert_eq!(eval_expr("(str (interpose :x [1 2 3]))"), "\"(1 :x 2 :x 3)\"");
}

#[test]
fn cljs_partition_all() {
    assert_eq!(eval_expr("(str (partition-all 3 [1 2 3 4 5]))"), "\"((1 2 3) (4 5))\"");
}

#[test]
fn cljs_partition_by() {
    assert_eq!(eval_expr("(str (partition-by even? [1 1 2 2 3 3]))"), "\"((1 1) (2 2) (3 3))\"");
}

#[test]
fn cljs_take_nth() {
    assert_eq!(eval_expr("(str (take-nth 2 (range 10)))"), "\"(0 2 4 6 8)\"");
}

#[test]
fn cljs_dedupe() {
    assert_eq!(eval_expr("(str (dedupe [1 1 2 2 3 3]))"), "\"(1 2 3)\"");
}

#[test]
fn cljs_reductions() {
    assert_eq!(eval_expr("(str (reductions + [1 2 3 4]))"), "\"(1 3 6 10)\"");
}

#[test]
#[ignore] // iterate not implemented (lazy infinite)
fn cljs_iterate() {
    assert_eq!(eval_expr("(str (take 5 (iterate inc 0)))"), "\"(0 1 2 3 4)\"");
}

#[test]
#[ignore] // cycle not implemented (lazy infinite)
fn cljs_cycle() {
    assert_eq!(eval_expr("(str (take 6 (cycle [1 2 3])))"), "\"(1 2 3 1 2 3)\"");
}

#[test]
#[ignore] // repeat with 1 arg (infinite) not supported
fn cljs_repeat_infinite() {
    assert_eq!(eval_expr("(str (take 3 (repeat 5)))"), "\"(5 5 5)\"");
}

#[test]
fn cljs_sort_by() {
    assert_eq!(eval_expr("(str (sort-by count [\"aaa\" \"bb\" \"c\"]))"), "\"(\"c\" \"bb\" \"aaa\")\"");
}

#[test]
fn cljs_merge_with() {
    assert_eq!(eval_expr("(get (merge-with + {:a 1} {:a 2}) :a)"), "3");
}

#[test]
fn cljs_mapv() {
    assert_eq!(eval_expr("(str (mapv inc [1 2 3]))"), "\"[2 3 4]\"");
}

#[test]
fn cljs_filterv() {
    assert_eq!(eval_expr("(str (filterv even? [1 2 3 4]))"), "\"[2 4]\"");
}

#[test]
fn cljs_subvec() {
    assert_eq!(eval_expr("(str (subvec [0 1 2 3 4] 2 4))"), "\"[2 3]\"");
}

#[test]
fn cljs_peek() {
    assert_eq!(eval_expr("(peek [1 2 3])"), "3");
    assert_eq!(eval_expr("(peek (list 1 2 3))"), "1");
}

#[test]
fn cljs_pop() {
    assert_eq!(eval_expr("(str (pop [1 2 3]))"), "\"[1 2]\"");
    assert_eq!(eval_expr("(str (pop (list 1 2 3)))"), "\"(2 3)\"");
}

#[test]
fn cljs_rseq() {
    assert_eq!(eval_expr("(str (rseq [1 2 3]))"), "\"(3 2 1)\"");
}

#[test]
fn cljs_keep_indexed() {
    assert_eq!(
        eval_expr("(str (keep-indexed #(when (odd? %1) %2) [:a :b :c :d :e]))"),
        "\"(:b :d)\""
    );
}

#[test]
fn cljs_seq_pred() {
    assert_eq!(eval_expr("(seq? (list 1 2))"), "true");
    assert_eq!(eval_expr("(seq? [1 2])"), "false");
}

#[test]
fn cljs_not_empty() {
    assert_eq!(eval_expr("(not-empty [1 2 3])"), "[1 2 3]");
    assert_eq!(eval_expr("(not-empty [])"), "nil");
}

#[test]
fn cljs_fnil() {
    assert_eq!(eval_expr("((fnil + 0) nil 3)"), "3");
    assert_eq!(eval_expr("((fnil + 0) 5 3)"), "8");
}

#[test]
fn cljs_ffirst() {
    assert_eq!(eval_expr("(ffirst [[1 2] [3 4]])"), "1");
}

#[test]
fn cljs_while() {
    assert_eq!(eval_expr("(let [x 1] (while false 1) x)"), "1");
}

#[test]
fn cljs_doseq() {
    assert_eq!(eval_expr("(doseq [x [1 2 3]] x)"), "nil");
}

#[test]
fn cljs_run() {
    assert_eq!(eval_expr("(run! identity [1 2 3])"), "nil");
}

#[test]
#[ignore] // into with transducer (3-arg) not fully implemented
fn cljs_into_with_transducer() {
    assert_eq!(eval_expr("(str (into [] (map inc) [1 2 3]))"), "\"[2 3 4]\"");
}

#[test]
fn cljs_reduce_reduced() {
    assert_eq!(
        eval_expr("(reduce (fn [acc x] (if (= x 3) (reduced acc) (+ acc x))) 0 [1 2 3 4 5])"),
        "3"
    );
}

#[test]
fn cljs_namespace() {
    assert_eq!(eval_expr("(namespace :foo/bar)"), "\"foo\"");
    assert_eq!(eval_expr("(namespace :foo)"), "nil");
}

#[test]
fn cljs_ex_info() {
    assert_eq!(
        eval_expr("(try (throw (ex-info \"test\" {:a 1})) (catch Exception e (ex-message e)))"),
        "\"test\""
    );
}

#[test]
fn cljs_pr_str() {
    assert_eq!(eval_expr("(pr-str [1 2 3])"), "\"[1 2 3]\"");
}

#[test]
#[ignore] // meta not implemented (returns nil/crashes)
fn cljs_meta() {
    assert_eq!(eval_expr("(meta (with-meta [1 2 3] {:a 1}))"), "{:a 1}");
}

#[test]
#[ignore] // extend-type on nil not supported
fn cljs_extend_type_nil() {
    assert_eq!(
        eval_expr("(do (defprotocol IStringable (to-string [this])) (extend-type nil IStringable (to-string [this] \"nil!\")) (to-string nil))"),
        "\"nil!\""
    );
}

#[test]
fn cljs_fn_named_recursive() {
    assert_eq!(
        eval_expr("((fn foo [x] (if (< x 3) (foo (inc x)) x)) 0)"),
        "3"
    );
}

#[test]
#[ignore] // destructuring in fn params not supported
fn cljs_fn_destructure() {
    assert_eq!(
        eval_expr("((fn [[x & xs]] (str x)) [1 2 3])"),
        "\"1\""
    );
}

#[test]
#[ignore] // destructuring in loop not supported
fn cljs_loop_destructure() {
    assert_eq!(
        eval_expr("(loop [[x y] [1 2]] (if (= x 3) y (recur [(inc x) y])))"),
        "2"
    );
}

#[test]
fn cljs_for_while_when() {
    assert_output(
        "(println (str (for [i [1 2 3] :while (< i 2) j [4 5 6] :when (even? j)] [i j])))",
        "([1 4] [1 6])",
    );
}

#[test]
#[ignore] // letfn not implemented
fn cljs_letfn() {
    assert_eq!(
        eval_expr("(letfn [(f ([x] (f x 1)) ([x y] (+ x y)))] (f 1))"),
        "2"
    );
}

#[test]
fn cljs_defonce() {
    assert_output("(defonce x 1) (defonce x 2) (println x)", "1");
}

#[test]
fn cljs_condp() {
    assert_eq!(eval_expr("(condp = 1 1 \"one\")"), "\"one\"");
}

#[test]
fn cljs_if_some() {
    assert_eq!(eval_expr("(if-some [foo nil] 1 2)"), "2");
    assert_eq!(eval_expr("(if-some [foo false] 1 2)"), "1");
}

#[test]
fn cljs_when_some() {
    assert_eq!(eval_expr("(when-some [foo nil] 1)"), "nil");
}

#[test]
fn cljs_trampoline() {
    assert_output(
        "(defn hello [x] (if (< x 10000) #(hello (inc x)) x)) (println (trampoline hello 0))",
        "10000",
    );
}

#[test]
fn cljs_delay() {
    assert_eq!(eval_expr("@(delay 1)"), "1");
}

#[test]
fn cljs_equality_vectors() {
    assert_eq!(eval_expr("(= [1 2 3] [1 2 3])"), "true");
}

#[test]
fn cljs_equality_maps() {
    assert_eq!(eval_expr("(= {:a 1 :b 2} {:b 2 :a 1})"), "true");
}

#[test]
fn cljs_equality_sets() {
    assert_eq!(eval_expr("(= #{1 2} #{2 1})"), "true");
}

#[test]
fn cljs_equality_strings() {
    assert_eq!(eval_expr("(= \"hello\" \"hello\")"), "true");
}

#[test]
fn cljs_equality_lists() {
    assert_eq!(eval_expr("(= (list 1 2 3) (list 1 2 3))"), "true");
}

#[test]
fn cljs_conj_multiple_values() {
    assert_eq!(eval_expr("(str (conj [1] 2 3))"), "\"[1 2 3]\"");
}

#[test]
fn cljs_reduce_reduced_unwrap() {
    assert_eq!(
        eval_expr("(str (reduce (fn [acc x] (if (> x 3) (reduced acc) (conj acc x))) [] [1 2 3 4 5]))"),
        "\"[1 2 3]\""
    );
}
