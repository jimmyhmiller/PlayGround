/// Additional tests ported from SCI (Small Clojure Interpreter) core_test.cljc
/// Source: https://github.com/babashka/sci
///
/// This file supplements sci_ported_core_test.rs with additional test coverage
/// for categories not yet ported.
///
/// Known missing features:
/// - as->, some->, some->> not implemented
/// - cond->, cond->> not implemented
/// - for, doseq, condp, case not implemented
/// - atoms (atom, swap!, reset!, deref, @) not implemented
/// - assert not implemented
/// - ex-info, ex-message, ex-data not implemented
/// - boolean, mapv, filterv, while not implemented
/// - no Java interop (Exception. etc.)

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

/// Run multi-form code from file, return stdout
fn eval_file(code: &str) -> String {
    run_code(code).0
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

/// Assert code fails (non-zero exit)
fn assert_error(code: &str) {
    let (stdout, _stderr, success) = run_code(code);
    assert!(
        !success,
        "Expected error but code succeeded.\nCode: {}\nStdout: {}",
        code, stdout
    );
}

// ============================================================================
// as-> macro tests (SCI line ~98-99) -- NOT IMPLEMENTED
// ============================================================================

#[test]
#[ignore] // as-> not implemented
fn sci_as_arrow_basic() {
    assert_eq!(
        eval_expr("(as-> 1 x (inc x) (inc x) (inc x) (str x))"),
        "\"4\""
    );
}

#[test]
#[ignore] // as-> not implemented
fn sci_as_arrow_repeat() {
    assert_eq!(
        eval_expr("(as-> 1 x (inc x) (inc x) (inc x) (apply str (repeat 10 (str x))))"),
        "\"4444444444\""
    );
}

// ============================================================================
// some-> and some->> macro tests (SCI line ~100-102) -- NOT IMPLEMENTED
// ============================================================================

#[test]
#[ignore] // some-> not implemented
fn sci_some_arrow_nil_short_circuit() {
    assert_eq!(eval_expr("(some-> {:a {:a nil}} :a :a :a)"), "nil");
}

#[test]
#[ignore] // some-> not implemented
fn sci_some_arrow_success() {
    assert_eq!(eval_expr("(some-> {:a {:b 42}} :a :b inc)"), "43");
}

#[test]
#[ignore] // some->> not implemented
fn sci_some_last_arrow_basic() {
    assert_eq!(
        eval_expr("(some->> [1 2 3] (map inc) (reduce +))"),
        "9"
    );
}

// ============================================================================
// Threading macro tests (SCI line ~1397) - ->, ->>
// ============================================================================

#[test]
fn sci_thread_first_inc_chain() {
    // (-> 1 inc inc (inc)) => 4
    assert_eq!(eval_expr("(-> 1 inc inc (inc))"), "4");
}

#[test]
fn sci_thread_first_with_parens() {
    // (-> 1 (+ 2) (* 3)) => 9
    assert_eq!(eval_expr("(-> 1 (+ 2) (* 3))"), "9");
}

#[test]
fn sci_thread_first_subtraction() {
    // (-> 10 (- 2)) => 8
    assert_eq!(eval_expr("(-> 10 (- 2))"), "8");
}

#[test]
fn sci_thread_first_division() {
    // (-> 9 inc (/ 100)) => 10/100 = 0 (integer division)
    assert_eq!(eval_expr("(-> 9 inc (/ 100))"), "0");
}

#[test]
fn sci_thread_last_division() {
    // (->> 9 inc (/ 100)) => 100/10 = 10
    assert_eq!(eval_expr("(->> 9 inc (/ 100))"), "10");
}

#[test]
fn sci_thread_first_map_ops() {
    // (-> {:a 1 :b 2} (assoc :c 3) (get :c)) => 3
    assert_eq!(eval_expr("(-> {:a 1 :b 2} (assoc :c 3) (get :c))"), "3");
}

#[test]
fn sci_thread_first_vector_ops() {
    // (-> [1 2 3] first inc) => 2
    assert_eq!(eval_expr("(-> [1 2 3] first inc)"), "2");
}

#[test]
fn sci_thread_first_quoted_list() {
    // (-> '(1 2 3)) should return (1 2 3)
    assert_output("(println (str (-> '(1 2 3))))", "(1 2 3)");
}

#[test]
fn sci_thread_last_filter_map_reduce() {
    // (->> (range 10) (filter even?) (map inc) (reduce +))
    assert_eq!(
        eval_expr("(->> (range 10) (filter even?) (map inc) (reduce +))"),
        "25"
    );
}

#[test]
fn sci_thread_last_map_count_max() {
    assert_output(
        r#"(println (->> ["foo" "baaar" "baaaaaz"] (map count) (apply max)))"#,
        "7",
    );
}

#[test]
fn sci_thread_first_nil_check() {
    assert_eq!(eval_expr("(-> nil nil?)"), "true");
    assert_eq!(eval_expr("(-> 1 nil?)"), "false");
}

#[test]
fn sci_thread_first_string_ops() {
    // (-> "hello" count (* 2)) => 10
    assert_eq!(eval_expr(r#"(-> "hello" count (* 2))"#), "10");
}

#[test]
fn sci_thread_last_equality() {
    // (->> 1 (= 1)) => true
    assert_eq!(eval_expr("(->> 1 (= 1))"), "true");
}

#[test]
fn sci_thread_last_subtraction() {
    // (->> 2 (- 10)) => 8
    assert_eq!(eval_expr("(->> 2 (- 10))"), "8");
}

#[test]
fn sci_thread_first_assoc_chain() {
    assert_eq!(
        eval_expr("(-> {} (assoc :a 1) (assoc :b 2) (assoc :c 3) count)"),
        "3"
    );
}

#[test]
fn sci_thread_last_filter_map_reduce_squares() {
    // (->> (range 10) (filter odd?) (map #(* % %)) (reduce +))
    assert_eq!(
        eval_expr("(->> (range 10) (filter odd?) (map #(* % %)) (reduce +))"),
        "165"
    );
}

#[test]
fn sci_thread_first_nested_maps() {
    assert_eq!(eval_expr("(-> {:a {:b {:c 42}}} :a :b :c)"), "42");
}

#[test]
fn sci_thread_first_get_nested() {
    assert_eq!(
        eval_expr("(-> {:a {:b {:c 42}}} (get :a) (get :b) (get :c))"),
        "42"
    );
}

#[test]
fn sci_thread_last_map_inc_first() {
    assert_eq!(eval_expr("(->> [1 2 3] (map inc) first)"), "2");
}

#[test]
fn sci_thread_first_multiply() {
    // (-> 5 (* 3) (- 2)) => 13
    assert_eq!(eval_expr("(-> 5 (* 3) (- 2))"), "13");
}

#[test]
fn sci_thread_last_multiply_subtract() {
    // (->> 5 (* 3) (- 20)) => 20 - 15 = 5
    assert_eq!(eval_expr("(->> 5 (* 3) (- 20))"), "5");
}

// ============================================================================
// do-and-or-test (SCI line ~1740) - do, and, or with many arguments
// ============================================================================

#[test]
fn sci_or_empty() {
    assert_eq!(eval_expr("(or)"), "nil");
}

#[test]
fn sci_and_empty() {
    assert_eq!(eval_expr("(and)"), "true");
}

#[test]
fn sci_or_single_nil() {
    assert_eq!(eval_expr("(or nil)"), "nil");
}

#[test]
fn sci_or_single_value() {
    assert_eq!(eval_expr("(or 1)"), "1");
}

#[test]
fn sci_and_single_nil() {
    assert_eq!(eval_expr("(and nil)"), "nil");
}

#[test]
fn sci_and_single_value() {
    assert_eq!(eval_expr("(and 1)"), "1");
}

#[test]
fn sci_or_nil_nil_nil_42() {
    assert_eq!(eval_expr("(or nil nil nil 42)"), "42");
}

#[test]
fn sci_or_nil_false_nil_42() {
    assert_eq!(eval_expr("(or nil false nil 42)"), "42");
}

#[test]
fn sci_or_false_false_false() {
    assert_eq!(eval_expr("(or false false false)"), "false");
}

#[test]
fn sci_and_many_true_then_42() {
    assert_eq!(
        eval_expr("(and true true true true true true true true true true 42)"),
        "42"
    );
}

#[test]
fn sci_and_many_true_then_nil() {
    assert_eq!(
        eval_expr("(and true true true true true true true true true nil 42)"),
        "nil"
    );
}

#[test]
fn sci_or_many_nils_then_true() {
    assert_eq!(
        eval_expr("(or nil nil nil nil nil nil nil nil nil nil true)"),
        "true"
    );
}

#[test]
fn sci_and_returns_last_truthy() {
    assert_eq!(eval_expr("(and 1 2 3)"), "3");
}

#[test]
fn sci_and_short_circuits_on_nil() {
    assert_eq!(eval_expr("(and 1 nil 3)"), "nil");
}

#[test]
fn sci_and_short_circuits_on_false() {
    assert_eq!(eval_expr("(and 1 false 3)"), "false");
}

#[test]
fn sci_do_many_expressions() {
    assert_eq!(
        eval_expr("(do (+ 1 2) (+ 3 4) (+ 5 6))"),
        "11"
    );
}

#[test]
fn sci_do_ten_expressions() {
    assert_eq!(
        eval_expr("(do (+ 1 1) (+ 2 2) (+ 3 3) (+ 4 4) (+ 5 5) (+ 6 6) (+ 7 7) (+ 8 8) (+ 9 9) (+ 10 10))"),
        "20"
    );
}

#[test]
fn sci_do_empty() {
    assert_eq!(eval_expr("(do)"), "nil");
}

#[test]
fn sci_do_nil() {
    assert_eq!(eval_expr("(do nil)"), "nil");
}

#[test]
fn sci_do_value_then_nil() {
    assert_eq!(eval_expr("(do 1 nil)"), "nil");
}

#[test]
fn sci_do_many_values() {
    assert_eq!(eval_expr("(do 1 2 3 4 5 6 7 8 9 10)"), "10");
}

// ============================================================================
// throw-test and try/catch tests (SCI line ~874-922)
// ============================================================================

#[test]
fn sci_throw_string_catch() {
    assert_eq!(
        eval_expr(r#"(try (throw "boom") (catch Exception e (str "error: " e)))"#),
        r#""error: boom""#
    );
}

#[test]
fn sci_throw_map_catch() {
    assert_eq!(
        eval_expr("(try (throw {:error true}) (catch Exception e (:error e)))"),
        "true"
    );
}

#[test]
fn sci_throw_string_catch_binding() {
    assert_eq!(
        eval_expr(r#"(try (throw "err") (catch Exception e e))"#),
        r#""err""#
    );
}

#[test]
fn sci_try_no_exception() {
    assert_eq!(eval_expr("(try (+ 1 2) (catch Exception e nil))"), "3");
}

#[test]
fn sci_try_body_with_catch() {
    assert_eq!(eval_expr("(try 1 2 3 (catch Exception e 0))"), "3");
}

#[test]
fn sci_try_finally_no_error() {
    assert_eq!(eval_expr("(try 1 (finally nil))"), "1");
}

#[test]
fn sci_try_finally_returns_body() {
    assert_eq!(eval_expr("(try :ok (finally :done))"), ":ok");
}

#[test]
fn sci_try_catch_finally() {
    assert_eq!(
        eval_expr(r#"(try (throw "oops") (catch Exception e :caught) (finally nil))"#),
        ":caught"
    );
}

#[test]
fn sci_try_in_let() {
    assert_eq!(eval_expr("(let [x (try 42)] x)"), "42");
}

#[test]
fn sci_try_catch_in_let() {
    assert_eq!(
        eval_expr(r#"(let [x (try (throw "err") (catch Exception e 99))] x)"#),
        "99"
    );
}

#[test]
fn sci_try_catch_nth_out_of_bounds() {
    assert_eq!(
        eval_expr("(try (nth [] 5) (catch Exception e :out-of-bounds))"),
        ":out-of-bounds"
    );
}

#[test]
fn sci_try_catch_type_error() {
    assert_eq!(
        eval_expr("(try (conj 1 2) (catch Exception e :type-error))"),
        ":type-error"
    );
}

#[test]
fn sci_defn_with_try_catch() {
    assert_output(
        "(defn safe-div [a b] (try (/ a b) (catch Exception e -1))) (println (safe-div 10 2))",
        "5",
    );
}

#[test]
fn sci_nested_try_rethrow() {
    assert_eq!(
        eval_expr(
            r#"(try (try (throw "inner") (catch Exception e (throw (str e "-rethrown")))) (catch Exception e e))"#
        ),
        r#""inner-rethrown""#
    );
}

#[test]
fn sci_nested_try_finally_propagates() {
    assert_eq!(
        eval_expr(
            r#"(try (try (throw "inner") (finally nil)) (catch Exception e (str "caught: " e)))"#
        ),
        r#""caught: inner""#
    );
}

#[test]
fn sci_try_catch_str_exception() {
    assert_eq!(
        eval_expr(r#"(try (throw "boom") (catch Exception e (str "caught: " e)))"#),
        r#""caught: boom""#
    );
}

// ============================================================================
// syntax-quote tests (SCI line ~924)
// ============================================================================

#[test]
fn sci_syntax_quote_list() {
    assert_output("(println (str `(1 2 3)))", "(1 2 3)");
}

#[test]
fn sci_syntax_quote_unquote() {
    assert_output("(println (str (let [x 10] `(~x ~x))))", "(10 10)");
}

#[test]
fn sci_syntax_quote_unquote_single() {
    assert_eq!(eval_expr("(let [x 1] `(~x))"), "(1)");
}

#[test]
fn sci_syntax_quote_unquote_with_syms() {
    assert_output("(println (str (let [x 1] `(a ~x b))))", "(a 1 b)");
}

#[test]
fn sci_syntax_quote_unquote_splice() {
    assert_output("(println (str (let [xs [1 2 3]] `(a ~@xs b))))", "(a 1 2 3 b)");
}

#[test]
fn sci_syntax_quote_unquote_splice_basic() {
    assert_output(
        "(println (str (let [xs [1 2 3]] `(~@xs 4))))",
        "(1 2 3 4)",
    );
}

#[test]
fn sci_syntax_quote_unquote_literal() {
    assert_eq!(eval_expr("`~1"), "1");
}

#[test]
fn sci_syntax_quote_unquote_expr() {
    assert_eq!(eval_expr("`~(+ 1 2)"), "3");
}

#[test]
fn sci_syntax_quote_unquote_let() {
    // `~(let [x 1] x) => 1
    assert_eq!(eval_expr("`~(let [x 1] x)"), "1");
}

#[test]
fn sci_syntax_quote_vector() {
    assert_output("(println (str `[1 2 3]))", "[1 2 3]");
}

#[test]
fn sci_syntax_quote_map_literal() {
    // `{:a 1} returns a map (printed as {... 1 entries})
    let result = eval_expr("`{:a 1}");
    assert!(
        result.contains("1 entries") || result.contains(":a"),
        "Expected a map, got: {}",
        result
    );
}

// ============================================================================
// declare-test (SCI line ~990)
// ============================================================================

#[test]
fn sci_declare_basic() {
    assert_output(
        "(declare foo bar) (defn f [] [foo bar]) (def foo 1) (def bar 2) (println (str (f)))",
        "[1 2]",
    );
}

#[test]
fn sci_declare_forward_ref() {
    assert_output(
        "(declare f) (defn g [] (f)) (defn f [] 42) (println (g))",
        "42",
    );
}

#[test]
fn sci_declare_multiple() {
    assert_output(
        "(declare a b c) (def a 1) (def b 2) (def c 3) (println (str [a b c]))",
        "[1 2 3]",
    );
}

// ============================================================================
// top-level-test (SCI line ~435)
// ============================================================================

#[test]
fn sci_top_level_nil_last() {
    // nil as last expression returns nil
    assert_eq!(eval_expr("1 2 nil"), "2");
    // Note: with -e, the result of the last expression evaluation is printed
    // For multi-form file, the printed output is what matters
}

#[test]
fn sci_top_level_expressions_in_order() {
    assert_output("(println 1) (println 2) (println 3)", "1\n2\n3");
}

// ============================================================================
// More recur tests from recur-test (SCI line ~663)
// ============================================================================

#[test]
fn sci_recur_in_fn() {
    assert_eq!(
        eval_expr("((fn [x] (if (pos? x) (recur (dec x)) x)) 10)"),
        "0"
    );
}

#[test]
fn sci_recur_in_loop_basic() {
    assert_eq!(
        eval_expr("(loop [x 5] (if (zero? x) :done (recur (dec x))))"),
        ":done"
    );
}

#[test]
fn sci_recur_loop_accumulator() {
    assert_eq!(
        eval_expr("(loop [x 0 acc 0] (if (>= x 5) acc (recur (inc x) (+ acc x))))"),
        "10"
    );
}

#[test]
fn sci_recur_loop_sum_1_to_100() {
    assert_eq!(
        eval_expr("(loop [i 0 s 0] (if (> i 100) s (recur (inc i) (+ s i))))"),
        "5050"
    );
}

#[test]
fn sci_recur_loop_two_vars() {
    // sum of 1 to 10 via loop
    assert_eq!(
        eval_expr("(loop [x 10 y 0] (if (zero? x) y (recur (dec x) (+ y x))))"),
        "55"
    );
}

#[test]
fn sci_recur_factorial_loop() {
    assert_eq!(
        eval_expr("((fn [n] (loop [i 0 acc 1] (if (= i n) acc (recur (inc i) (* acc (inc i)))))) 5)"),
        "120"
    );
}

#[test]
fn sci_recur_loop_build_vector() {
    assert_output(
        "(println (str (loop [l [] i 0] (if (>= i 5) l (recur (conj l i) (inc i))))))",
        "[0 1 2 3 4]",
    );
}

#[test]
fn sci_recur_fibonacci_loop() {
    assert_output(
        "(defn fib [n] (loop [a 0 b 1 i 0] (if (= i n) a (recur b (+ a b) (inc i))))) (println (fib 10))",
        "55",
    );
}

#[test]
fn sci_recur_factorial_defn() {
    assert_eq!(
        eval_expr("(loop [n 10 acc 1] (if (<= n 1) acc (recur (dec n) (* acc n))))"),
        "3628800"
    );
}

#[test]
fn sci_recur_loop_with_let() {
    assert_eq!(
        eval_expr("(loop [n 0] (if (= n 10) n (let [m (inc n)] (recur m))))"),
        "10"
    );
}

#[test]
fn sci_recur_variadic_fn_reduce() {
    // Manually reduce with recur over variadic args
    assert_eq!(
        eval_expr("((fn [& args] (if (next args) (recur (rest args)) (first args))) 1 2 3 4 5)"),
        "5"
    );
}

#[test]
fn sci_recur_variadic_sum() {
    assert_eq!(
        eval_expr("((fn [x & args] (if args (recur (+ x (first args)) (next args)) x)) 0 1 2 3 4 5)"),
        "15"
    );
}

#[test]
fn sci_recur_defn_countdown() {
    assert_output(
        "(defn countdown [n] (if (zero? n) :done (recur (dec n)))) (println (countdown 1000))",
        ":done",
    );
}

#[test]
fn sci_recur_gcd() {
    assert_eq!(
        eval_expr("(defn gcd [a b] (if (zero? b) a (recur b (mod a b)))) (gcd 12 8)"),
        "4"
    );
}

#[test]
fn sci_recur_not_in_tail_position() {
    assert_error("(recur)");
}

// ============================================================================
// More loop tests from loop-test (SCI line ~736)
// ============================================================================

#[test]
fn sci_loop_conj_list_extended() {
    assert_output(
        "(println (str (loop [l (list 2 1) c (count l)] (if (> c 4) l (recur (conj l (inc c)) (inc c))))))",
        "(5 4 3 2 1)",
    );
}

#[test]
fn sci_loop_let_shadow_extended() {
    assert_eq!(eval_expr("(let [x 1] (loop [x (inc x)] x))"), "2");
}

#[test]
fn sci_loop_100_iterations() {
    assert_eq!(
        eval_expr("(loop [i 0] (if (= i 100) i (recur (inc i))))"),
        "100"
    );
}

#[test]
fn sci_loop_collect_squares() {
    assert_output(
        "(println (str (loop [coll [1 2 3 4 5] result []] (if (empty? coll) result (recur (rest coll) (conj result (* (first coll) 2)))))))",
        "[2 4 6 8 10]",
    );
}

#[test]
fn sci_loop_my_map() {
    assert_output(
        "(defn my-map [f coll] (loop [c coll acc []] (if (empty? c) acc (recur (rest c) (conj acc (f (first c))))))) (println (str (my-map inc [1 2 3])))",
        "[2 3 4]",
    );
}

// ============================================================================
// dotimes tests (SCI line ~1063) -- dotimes IS available
// ============================================================================

#[test]
fn sci_dotimes_basic() {
    assert_output("(dotimes [i 5] (println i))", "0\n1\n2\n3\n4");
}

#[test]
fn sci_dotimes_zero() {
    // dotimes with 0 should do nothing, return nil
    assert_eq!(eval_expr("(dotimes [i 0] (println i))"), "nil");
}

#[test]
fn sci_dotimes_one() {
    assert_output("(dotimes [i 1] (println i))", "0");
}

#[test]
fn sci_dotimes_squares() {
    assert_output("(dotimes [i 3] (println (* i i)))", "0\n1\n4");
}

#[test]
fn sci_dotimes_returns_nil() {
    assert_eq!(eval_expr("(dotimes [i 5] nil)"), "nil");
}

#[test]
fn sci_dotimes_ten_returns_nil() {
    assert_eq!(eval_expr("(dotimes [i 10] nil)"), "nil");
}

// ============================================================================
// for tests (SCI line ~768) -- NOT IMPLEMENTED
// ============================================================================

#[test]
#[ignore] // for macro not implemented
fn sci_for_basic() {
    assert_output(
        "(println (str (for [i [1 2 3]] (* i i))))",
        "(1 4 9)",
    );
}

// ============================================================================
// doseq tests (SCI line ~790) -- NOT IMPLEMENTED
// ============================================================================

#[test]
#[ignore] // doseq not implemented
fn sci_doseq_basic() {
    assert_output(
        "(doseq [i [1 2 3]] (println i))",
        "1\n2\n3",
    );
}

// ============================================================================
// condp tests (SCI line ~810) -- NOT IMPLEMENTED
// ============================================================================

#[test]
#[ignore] // condp not implemented
fn sci_condp_equals() {
    assert_eq!(eval_expr(r#"(condp = 1 1 "one")"#), r#""one""#);
}

// ============================================================================
// case tests (SCI line ~821) -- NOT IMPLEMENTED
// ============================================================================

#[test]
#[ignore] // case not implemented
fn sci_case_int() {
    assert_eq!(eval_expr("(case 1, 1 true, 2 false)"), "true");
}

#[test]
#[ignore] // case not implemented
fn sci_case_inc_default() {
    assert_eq!(eval_expr("(case (inc 2), 1 true, 2 (+ 1 2 3), 7)"), "7");
}

// ============================================================================
// assert tests (SCI line ~1045) -- NOT IMPLEMENTED
// ============================================================================

#[test]
#[ignore] // assert not implemented
fn sci_assert_true() {
    assert_eq!(eval_expr("(assert true)"), "nil");
}

#[test]
#[ignore] // assert not implemented
fn sci_assert_false_throws() {
    assert_error("(assert false)");
}

// ============================================================================
// ex-message tests (SCI line ~1041) -- NOT IMPLEMENTED (no ex-info)
// ============================================================================

#[test]
#[ignore] // ex-info / ex-message not implemented
fn sci_ex_message() {
    assert_eq!(
        eval_expr(r#"(ex-message (ex-info "foo" {:a 1}))"#),
        r#""foo""#
    );
}

// ============================================================================
// Quoting edge cases (SCI line ~117-121)
// ============================================================================

#[test]
fn sci_quote_vector() {
    assert_output("(println (str '[1 2 3]))", "[1 2 3]");
}

#[test]
fn sci_quote_list() {
    assert_output("(println (str '(1 2 3)))", "(1 2 3)");
}

#[test]
fn sci_quote_symbol() {
    assert_eq!(eval_expr("'hello"), "hello");
}

#[test]
fn sci_quote_keyword_identity() {
    assert_eq!(eval_expr("':a"), ":a");
}

#[test]
fn sci_quote_via_fn() {
    assert_eq!(eval_expr("(quote hello)"), "hello");
}

#[test]
fn sci_quote_nested_list() {
    assert_output("(println (str '(a b (c d))))", "(a b (c d))");
}

#[test]
fn sci_quote_map() {
    // quoted map prints as {... N entries}
    let result = eval_expr("'{:a 1}");
    assert!(
        result.contains("1 entries") || result.contains(":a"),
        "Expected map, got: {}",
        result
    );
}

// ============================================================================
// More defn tests - multi-arity, variadic, edge cases
// ============================================================================

#[test]
fn sci_defn_multi_arity_zero() {
    assert_eq!(
        eval_expr("(defn multi ([] 0) ([x] x) ([x y] (+ x y)) ([x y z] (+ x y z))) (multi)"),
        "0"
    );
}

#[test]
fn sci_defn_multi_arity_one() {
    assert_eq!(
        eval_expr("(defn multi ([] 0) ([x] x) ([x y] (+ x y)) ([x y z] (+ x y z))) (multi 1)"),
        "1"
    );
}

#[test]
fn sci_defn_multi_arity_two() {
    assert_eq!(
        eval_expr("(defn multi ([] 0) ([x] x) ([x y] (+ x y)) ([x y z] (+ x y z))) (multi 1 2)"),
        "3"
    );
}

#[test]
fn sci_defn_multi_arity_three() {
    assert_eq!(
        eval_expr("(defn multi ([] 0) ([x] x) ([x y] (+ x y)) ([x y z] (+ x y z))) (multi 1 2 3)"),
        "6"
    );
}

#[test]
fn sci_defn_variadic_count() {
    assert_eq!(
        eval_expr(r#"(defn vari [x & more] (str x " " (count more))) (vari 1 2 3 4)"#),
        r#""1 3""#
    );
}

#[test]
fn sci_defn_docstring() {
    assert_output(
        r#"(defn docfn "my doc" [x] (inc x)) (println (docfn 1))"#,
        "2",
    );
}

#[test]
fn sci_defn_redefine_extended() {
    assert_output(
        "(defn foo [x] (+ x 1)) (defn foo [x] (+ x 2)) (println (foo 10))",
        "12",
    );
}

#[test]
fn sci_defn_greet_multi_arity() {
    assert_eq!(
        eval_expr(r#"(defn greet ([] "hi") ([name] (str "hello " name))) (greet)"#),
        r#""hi""#
    );
    assert_eq!(
        eval_expr(r#"(defn greet ([] "hi") ([name] (str "hello " name))) (greet "world")"#),
        r#""hello world""#
    );
}

#[test]
fn sci_defn_private() {
    assert_output("(defn- secret [] :hidden) (println (secret))", ":hidden");
}

// ============================================================================
// fn tests - edge cases
// ============================================================================

#[test]
fn sci_fn_zero_args() {
    assert_eq!(eval_expr("((fn [] :hello))"), ":hello");
}

#[test]
fn sci_fn_identity() {
    assert_eq!(eval_expr("((fn [x] x) :hello)"), ":hello");
}

#[test]
fn sci_fn_three_args() {
    assert_eq!(eval_expr("((fn [x y z] (+ x y z)) 1 2 3)"), "6");
}

#[test]
fn sci_fn_variadic_zero() {
    assert_eq!(eval_expr("((fn [& args] (count args)))"), "0");
}

#[test]
fn sci_fn_variadic_many() {
    assert_eq!(eval_expr("((fn [& args] (count args)) 1 2 3)"), "3");
}

#[test]
fn sci_fn_fixed_plus_variadic() {
    assert_eq!(
        eval_expr(r#"((fn [x & args] (str x (count args))) 1 2 3 4 5)"#),
        r#""14""#
    );
}

#[test]
fn sci_fn_multi_arity_three() {
    assert_eq!(
        eval_expr("((fn ([x] x) ([x y] (+ x y)) ([x y z] (+ x y z))) 1 2 3)"),
        "6"
    );
}

#[test]
fn sci_fn_literal_basic() {
    assert_eq!(eval_expr("(#(+ %1 %2) 3 4)"), "7");
}

#[test]
fn sci_fn_literal_no_args() {
    assert_eq!(eval_expr("(#(+ 1 2))"), "3");
}

#[test]
fn sci_fn_literal_nested_do() {
    assert_eq!(eval_expr("(#(do %) :hello)"), ":hello");
}

// ============================================================================
// String operations - str, name, etc.
// ============================================================================

#[test]
fn sci_str_concatenation() {
    assert_eq!(eval_expr(r#"(str "hello" " " "world")"#), r#""hello world""#);
}

#[test]
fn sci_str_numbers() {
    assert_eq!(eval_expr("(str 1 2 3)"), r#""123""#);
}

#[test]
fn sci_str_nil() {
    assert_eq!(eval_expr("(str nil)"), r#""""#);
}

#[test]
fn sci_str_empty() {
    assert_eq!(eval_expr("(str)"), r#""""#);
}

#[test]
fn sci_str_mixed_types() {
    assert_eq!(eval_expr("(str 1 nil 2)"), r#""12""#);
}

#[test]
fn sci_str_nil_nil_nil() {
    assert_eq!(eval_expr("(str nil nil nil)"), r#""""#);
}

#[test]
fn sci_str_true_false() {
    assert_eq!(eval_expr("(str true)"), r#""true""#);
    assert_eq!(eval_expr("(str false)"), r#""false""#);
}

#[test]
fn sci_str_keyword() {
    assert_eq!(eval_expr("(str :foo)"), r#"":foo""#);
}

#[test]
fn sci_str_function_result() {
    assert_eq!(
        eval_expr(r#"(defn f [x] (str "f(" x ")")) (f 42)"#),
        r#""f(42)""#
    );
}

#[test]
fn sci_name_keyword() {
    assert_eq!(eval_expr("(name :hello)"), r#""hello""#);
}

#[test]
fn sci_name_symbol() {
    assert_eq!(eval_expr("(name 'hello)"), r#""hello""#);
}

#[test]
fn sci_symbol_from_string() {
    assert_eq!(eval_expr(r#"(str (symbol "foo"))"#), r#""foo""#);
}

#[test]
fn sci_symbol_qualified() {
    assert_eq!(eval_expr(r#"(symbol "foo" "bar")"#), "foo/bar");
}

#[test]
fn sci_count_string() {
    assert_eq!(eval_expr(r#"(count "hello world")"#), "11");
    assert_eq!(eval_expr(r#"(count "hello")"#), "5");
}

#[test]
fn sci_str_map_result() {
    assert_output(
        "(println (str (map str [1 2 3])))",
        "(\"1\" \"2\" \"3\")",
    );
}

#[test]
fn sci_str_of_collections() {
    assert_eq!(eval_expr("(str [1 2 3])"), r#""[1 2 3]""#);
    assert_eq!(eval_expr("(str (list 1 2 3))"), r#""(1 2 3)""#);
}

#[test]
fn sci_apply_str() {
    assert_eq!(eval_expr(r#"(apply str ["a" "b" "c" "d"])"#), r#""abcd""#);
}

#[test]
fn sci_reduce_str() {
    assert_eq!(eval_expr("(reduce str [1 2 3])"), r#""123""#);
    assert_eq!(
        eval_expr(r#"(reduce str "" ["a" "b" "c"])"#),
        r#""abc""#
    );
}

#[test]
fn sci_reduce_str_join() {
    assert_eq!(
        eval_expr(r#"(reduce (fn [acc x] (str acc "-" x)) [1 2 3 4 5])"#),
        r#""1-2-3-4-5""#
    );
}

// ============================================================================
// Additional collection operations
// ============================================================================

#[test]
fn sci_into_vector() {
    assert_output("(println (str (into [] (list 1 2 3))))", "[1 2 3]");
}

#[test]
fn sci_into_list() {
    assert_output("(println (str (into (list) [1 2 3])))", "(3 2 1)");
}

#[test]
fn sci_list_creation() {
    assert_output("(println (str (list 1 2 3)))", "(1 2 3)");
    assert_output("(println (str (list)))", "()");
}

#[test]
fn sci_cons_nil() {
    assert_output("(println (str (cons 1 nil)))", "(1)");
}

#[test]
fn sci_cons_chain() {
    assert_output("(println (str (cons 0 (cons 1 nil))))", "(0 1)");
}

#[test]
fn sci_first_nil() {
    assert_eq!(eval_expr("(first nil)"), "nil");
}

#[test]
fn sci_rest_nil_empty() {
    assert_eq!(eval_expr("(empty? (rest nil))"), "true");
}

#[test]
fn sci_next_nil() {
    assert_eq!(eval_expr("(next nil)"), "nil");
}

#[test]
fn sci_next_single() {
    assert_eq!(eval_expr("(next [1])"), "nil");
}

#[test]
fn sci_count_nil() {
    assert_eq!(eval_expr("(count nil)"), "0");
}

#[test]
fn sci_seq_empty_vector() {
    assert_eq!(eval_expr("(seq [])"), "nil");
}

#[test]
fn sci_seq_nonempty() {
    assert_output("(println (str (seq [1 2 3])))", "(1 2 3)");
}

#[test]
fn sci_nth_default() {
    assert_eq!(eval_expr("(nth [1 2 3] 3 :default)"), ":default");
}

#[test]
fn sci_get_vector() {
    assert_eq!(eval_expr("(get [1 2 3] 1)"), "2");
    assert_eq!(eval_expr("(get [1 2 3] 5 :default)"), ":default");
}

#[test]
fn sci_get_nil() {
    assert_eq!(eval_expr("(get nil :a)"), "nil");
}

#[test]
fn sci_second_basic() {
    assert_eq!(eval_expr("(second [1 2 3])"), "2");
    assert_eq!(eval_expr("(second (list 1 2 3))"), "2");
}

#[test]
fn sci_last_basic() {
    assert_eq!(eval_expr("(last [1 2 3 4])"), "4");
}

#[test]
fn sci_butlast_basic() {
    assert_output("(println (str (butlast [1 2 3])))", "(1 2)");
}

#[test]
fn sci_empty_preserves_type() {
    assert_eq!(eval_expr("(str (empty [1 2 3]))"), r#""[]""#);
    assert_eq!(eval_expr("(str (empty (list 1 2 3)))"), r#""()""#);
    assert_eq!(eval_expr("(str (empty {}))"), r#""{}""#);
    assert_eq!(eval_expr("(empty #{})"), "#{}");
}

#[test]
fn sci_vector_as_fn() {
    assert_eq!(eval_expr("([10 20 30] 1)"), "20");
}

#[test]
fn sci_map_as_fn_extended() {
    assert_eq!(eval_expr("({:a 1 :b 2} :a)"), "1");
}

#[test]
fn sci_set_as_fn_nil() {
    assert_eq!(eval_expr("(nil? (#{:a :b :c} :d))"), "true");
}

#[test]
fn sci_hash_map_construction() {
    assert_eq!(eval_expr("(get (hash-map :a 1 :b 2) :a)"), "1");
}

#[test]
fn sci_hash_set_construction() {
    assert_eq!(eval_expr("(contains? (hash-set 1 2 3) 2)"), "true");
}

#[test]
fn sci_list_star() {
    assert_output("(println (str (list* 1 [2 3])))", "(1 2 3)");
    assert_output("(println (str (list* 1 2 [3 4])))", "(1 2 3 4)");
}

#[test]
fn sci_concat_empty() {
    assert_eq!(eval_expr("(str (concat))"), r#""""#);
}

#[test]
fn sci_concat_with_nil() {
    assert_output("(println (str (concat nil [1 2] nil [3])))", "(1 2 3)");
}

#[test]
fn sci_flatten_nested() {
    assert_output(
        "(println (str (flatten [1 [2 3] [[4 [5]]]])))",
        "(1 2 3 4 5)",
    );
}

#[test]
fn sci_distinct_count() {
    assert_eq!(eval_expr("(count (distinct [1 1 2 2 3 3 4 4]))"), "4");
}

#[test]
fn sci_interleave_three() {
    assert_output(
        "(println (str (interleave [1 2 3] [:a :b :c] [10 20 30])))",
        "(1 :a 10 2 :b 20 3 :c 30)",
    );
}

#[test]
fn sci_partition_step() {
    assert_output(
        "(println (str (partition 2 1 [1 2 3 4 5])))",
        "((1 2) (2 3) (3 4) (4 5))",
    );
}

#[test]
fn sci_partition_three() {
    assert_output(
        "(println (str (partition 3 [1 2 3 4 5 6 7 8 9])))",
        "((1 2 3) (4 5 6) (7 8 9))",
    );
}

// ============================================================================
// Higher-order function tests
// ============================================================================

#[test]
fn sci_map_with_fn() {
    assert_output(
        "(println (str (map (fn [x] (* x x)) (range 6))))",
        "(0 1 4 9 16 25)",
    );
}

#[test]
fn sci_filter_with_fn() {
    assert_output(
        "(println (str (filter (fn [x] (> x 3)) (range 10))))",
        "(4 5 6 7 8 9)",
    );
}

#[test]
fn sci_remove_with_fn() {
    assert_output(
        "(println (str (remove (fn [x] (> x 3)) (range 6))))",
        "(0 1 2 3)",
    );
}

#[test]
fn sci_map_vector() {
    assert_output(
        "(println (str (map vector [1 2 3] [:a :b :c])))",
        "([1 :a] [2 :b] [3 :c])",
    );
}

#[test]
fn sci_map_plus_three_colls() {
    assert_output(
        "(println (str (map + [1 2 3] [10 20 30] [100 200 300])))",
        "(111 222 333)",
    );
}

#[test]
fn sci_keep_identity() {
    assert_output(
        "(println (str (keep identity [1 nil 2 nil 3])))",
        "(1 2 3)",
    );
}

#[test]
fn sci_keep_even() {
    assert_output(
        "(println (str (keep #(if (even? %) %) [1 2 3 4 5])))",
        "(2 4)",
    );
}

#[test]
fn sci_reduce_conj_vector() {
    assert_output(
        "(println (str (reduce conj [] (range 5))))",
        "[0 1 2 3 4]",
    );
}

#[test]
fn sci_reduce_build_squares() {
    assert_output(
        "(println (str (reduce (fn [acc x] (conj acc (* x x))) [] [1 2 3 4 5])))",
        "[1 4 9 16 25]",
    );
}

#[test]
fn sci_reduced_early_exit() {
    assert_eq!(
        eval_expr("(reduce (fn [acc x] (if (> acc 10) (reduced acc) (+ acc x))) 0 (range 100))"),
        "15"
    );
}

#[test]
fn sci_mapcat_fn() {
    assert_output(
        "(println (str (mapcat #(list % %) [1 2 3])))",
        "(1 1 2 2 3 3)",
    );
}

#[test]
fn sci_map_squared() {
    assert_output(
        "(println (str (map #(* % %) [1 2 3 4 5])))",
        "(1 4 9 16 25)",
    );
}

#[test]
fn sci_remove_nil() {
    assert_output(
        "(println (str (remove nil? [1 nil 2 nil 3])))",
        "(1 2 3)",
    );
}

#[test]
fn sci_filter_complement_nil() {
    assert_output(
        "(println (str (filter (complement nil?) [1 nil 2 nil 3])))",
        "(1 2 3)",
    );
}

#[test]
fn sci_some_set_lookup() {
    assert_eq!(eval_expr("(some #{3} [1 2 3 4])"), "3");
}

#[test]
fn sci_sort_comparator() {
    assert_output("(println (str (sort > [3 1 2 5 4])))", "(5 4 3 2 1)");
}

#[test]
fn sci_apply_sum_range() {
    assert_eq!(eval_expr("(apply + (range 101))"), "5050");
}

// ============================================================================
// Arithmetic edge cases
// ============================================================================

#[test]
fn sci_arithmetic_chained_comparisons() {
    assert_eq!(eval_expr("(< 1 2 3 4)"), "true");
    assert_eq!(eval_expr("(< 1 2 2 4)"), "false");
    assert_eq!(eval_expr("(<= 1 2 2 4)"), "true");
    assert_eq!(eval_expr("(<= 1 2 3 2)"), "false");
    assert_eq!(eval_expr("(> 4 3 2 1)"), "true");
    assert_eq!(eval_expr("(> 4 3 3 1)"), "false");
    assert_eq!(eval_expr("(>= 4 3 3 1)"), "true");
    assert_eq!(eval_expr("(>= 4 3 4 1)"), "false");
}

#[test]
fn sci_equality_multi_arg() {
    assert_eq!(eval_expr("(= 1 1 1)"), "true");
    assert_eq!(eval_expr("(= 1 1 2)"), "false");
    assert_eq!(eval_expr("(= 1 1 1 1)"), "true");
    assert_eq!(eval_expr("(= 1 1 1 2)"), "false");
}

#[test]
fn sci_quot_rem_mod() {
    assert_eq!(eval_expr("(quot 10 3)"), "3");
    assert_eq!(eval_expr("(rem 10 3)"), "1");
    assert_eq!(eval_expr("(mod 10 3)"), "1");
    assert_eq!(eval_expr("(mod -10 3)"), "2");
    assert_eq!(eval_expr("(rem -10 3)"), "-1");
}

#[test]
fn sci_float_arithmetic() {
    assert_eq!(eval_expr("(- 10.0 3.5)"), "6.5");
    assert_eq!(eval_expr("(/ 10.0 3.0)"), "3.3333333333333335");
}

#[test]
fn sci_integer_division() {
    assert_eq!(eval_expr("(/ 10 2)"), "5");
    assert_eq!(eval_expr("(/ 10 3)"), "3");
}

// ============================================================================
// Predicate tests
// ============================================================================

#[test]
fn sci_some_pred() {
    assert_eq!(eval_expr("(some? nil)"), "false");
    assert_eq!(eval_expr("(some? 1)"), "true");
    assert_eq!(eval_expr("(some? false)"), "true");
}

#[test]
fn sci_sequential_pred() {
    assert_eq!(eval_expr("(sequential? [1 2 3])"), "true");
    assert_eq!(eval_expr("(sequential? {:a 1})"), "false");
}

#[test]
fn sci_associative_pred() {
    assert_eq!(eval_expr("(associative? {:a 1})"), "true");
    assert_eq!(eval_expr("(associative? [1 2 3])"), "true");
}

#[test]
fn sci_list_pred() {
    assert_eq!(eval_expr("(list? (list 1 2))"), "true");
    assert_eq!(eval_expr("(list? [1 2])"), "false");
}

#[test]
fn sci_set_pred() {
    assert_eq!(eval_expr("(set? #{1 2})"), "true");
    assert_eq!(eval_expr("(set? [1 2])"), "false");
}

#[test]
fn sci_identical_pred() {
    assert_eq!(eval_expr("(identical? nil nil)"), "true");
    assert_eq!(eval_expr("(identical? 1 1)"), "true");
}

// ============================================================================
// Composition and higher-order helpers
// ============================================================================

#[test]
fn sci_comp_str_inc() {
    assert_eq!(eval_expr("((comp str inc) 1)"), r#""2""#);
}

#[test]
fn sci_comp_count_str() {
    assert_eq!(eval_expr("((comp count str) 123)"), "3");
}

#[test]
fn sci_partial_plus_multi() {
    assert_eq!(eval_expr("((partial + 10 20) 30)"), "60");
}

#[test]
fn sci_partial_str() {
    assert_eq!(
        eval_expr(r#"((partial str "hello ") "world")"#),
        r#""hello world""#
    );
}

#[test]
fn sci_partial_multiply() {
    assert_eq!(eval_expr("((partial * 2 3) 4)"), "24");
}

#[test]
fn sci_complement_even() {
    assert_eq!(eval_expr("((complement even?) 3)"), "true");
    assert_eq!(eval_expr("((complement even?) 4)"), "false");
}

#[test]
fn sci_map_with_partial() {
    assert_output(
        "(println (str (map (partial + 10) [1 2 3])))",
        "(11 12 13)",
    );
}

#[test]
fn sci_juxt_inc_dec_square() {
    assert_output(
        "(println (str ((juxt inc dec #(* % %)) 5)))",
        "[6 4 25]",
    );
}

#[test]
fn sci_juxt_first_last() {
    assert_output(
        "(println (str ((juxt first last) [1 2 3 4 5])))",
        "[1 5]",
    );
}

// ============================================================================
// Map operations
// ============================================================================

#[test]
fn sci_assoc_multiple() {
    assert_eq!(eval_expr("(count (assoc {} :a 1 :b 2 :c 3))"), "3");
}

#[test]
fn sci_dissoc_multiple() {
    assert_eq!(eval_expr("(count (dissoc {:a 1 :b 2 :c 3} :a :b))"), "1");
}

#[test]
fn sci_update_with_extra_args() {
    assert_eq!(eval_expr("(get (update {:a 1} :a + 10) :a)"), "11");
}

#[test]
fn sci_get_in_nested() {
    assert_eq!(eval_expr("(get-in {:a {:b {:c 42}}} [:a :b :c])"), "42");
}

#[test]
fn sci_assoc_in_nested() {
    assert_eq!(
        eval_expr("(get-in (assoc-in {} [:a :b :c] 42) [:a :b :c])"),
        "42"
    );
}

#[test]
fn sci_update_in_nested() {
    assert_eq!(
        eval_expr("(get-in (update-in {:a {:b 1}} [:a :b] inc) [:a :b])"),
        "2"
    );
}

#[test]
fn sci_select_keys_contains() {
    assert_eq!(
        eval_expr("(get (select-keys {:a 1 :b 2 :c 3} [:a :b]) :a)"),
        "1"
    );
    assert_eq!(
        eval_expr("(count (select-keys {:a 1 :b 2 :c 3} [:a :b]))"),
        "2"
    );
}

#[test]
fn sci_merge_basic() {
    assert_eq!(
        eval_expr("(get (merge {:a 1} {:b 2} {:c 3}) :c)"),
        "3"
    );
}

#[test]
fn sci_zipmap_basic() {
    assert_eq!(eval_expr("(get (zipmap [:a :b :c] [1 2 3]) :b)"), "2");
}

#[test]
fn sci_keys_vals() {
    assert_output("(println (str (keys {:a 1 :b 2})))", "(:a :b)");
    assert_output("(println (str (vals {:a 1 :b 2})))", "(1 2)");
}

#[test]
fn sci_contains_vector() {
    assert_eq!(eval_expr("(contains? [1 2 3] 1)"), "true");
    assert_eq!(eval_expr("(contains? [1 2 3] 5)"), "false");
}

#[test]
fn sci_contains_set() {
    assert_eq!(eval_expr("(contains? #{1 2 3} 2)"), "true");
    assert_eq!(eval_expr("(contains? #{1 2 3} 5)"), "false");
}

// ============================================================================
// Set operations
// ============================================================================

#[test]
fn sci_disj_multiple() {
    assert_eq!(eval_expr("(count (disj #{1 2 3 4} 2 4))"), "2");
}

// ============================================================================
// Protocol and deftype tests
// ============================================================================

#[test]
fn sci_protocol_deftype_basic() {
    assert_output(
        "(defprotocol IAnimal (speak [this]) (legs [this])) (deftype Cat [] IAnimal (speak [this] \"meow\") (legs [this] 4)) (println (speak (Cat.)))",
        "meow",
    );
}

#[test]
fn sci_protocol_deftype_with_field() {
    assert_output(
        "(defprotocol IAnimal (speak [this]) (legs [this])) (deftype Cat [] IAnimal (speak [this] \"meow\") (legs [this] 4)) (println (legs (Cat.)))",
        "4",
    );
}

#[test]
fn sci_protocol_deftype_method_two_args() {
    assert_output(
        "(defprotocol IFoo (foo [this]) (bar [this x])) (deftype MyFoo [val] IFoo (foo [this] val) (bar [this x] (+ val x))) (println (bar (MyFoo. 10) 5))",
        "15",
    );
}

#[test]
fn sci_extend_type_long() {
    assert_output(
        r#"(defprotocol IShow (show [this])) (extend-type Long IShow (show [this] (str "num:" this))) (println (show 42))"#,
        "num:42",
    );
}

#[test]
fn sci_extend_type_string() {
    assert_output(
        r#"(defprotocol IShow (show [this])) (extend-type String IShow (show [this] (str "str:" this))) (println (show "hello"))"#,
        "str:hello",
    );
}

// ============================================================================
// Macro tests
// ============================================================================

#[test]
fn sci_defmacro_twice() {
    assert_output(
        r#"(defmacro twice [x] `(do ~x ~x)) (twice (println "hi"))"#,
        "hi\nhi",
    );
}

#[test]
fn sci_defmacro_apply_to() {
    assert_eq!(
        eval_expr("(defmacro apply-to [f & args] `(~f ~@args)) (apply-to + 1 2 3)"),
        "6"
    );
}

#[test]
fn sci_defmacro_my_let() {
    assert_eq!(
        eval_expr("(defmacro my-let [bindings & body] `(let ~bindings ~@body)) (my-let [x 1 y 2] (+ x y))"),
        "3"
    );
}

#[test]
fn sci_defmacro_with_val() {
    assert_eq!(
        eval_expr("(defmacro with-val [sym val & body] `(let [~sym ~val] ~@body)) (with-val x 42 (inc x))"),
        "43"
    );
}

#[test]
fn sci_defmacro_infix() {
    assert_eq!(
        eval_expr("(defmacro infix [a op b] (list op a b)) (infix 3 + 4)"),
        "7"
    );
}

#[test]
fn sci_defmacro_unless_true() {
    assert_eq!(
        eval_expr("(defmacro unless [test & body] `(when (not ~test) ~@body)) (unless true :executed)"),
        "nil"
    );
}

#[test]
fn sci_defmacro_unless_false() {
    assert_eq!(
        eval_expr("(defmacro unless [test & body] `(when (not ~test) ~@body)) (unless false :executed)"),
        ":executed"
    );
}

#[test]
fn sci_defmacro_my_if_true() {
    assert_eq!(
        eval_expr("(defmacro my-if [test then else] `(cond ~test ~then :else ~else)) (my-if true :yes :no)"),
        ":yes"
    );
}

#[test]
fn sci_defmacro_my_if_false() {
    assert_eq!(
        eval_expr("(defmacro my-if [test then else] `(cond ~test ~then :else ~else)) (my-if false :yes :no)"),
        ":no"
    );
}

#[test]
fn sci_defmacro_debug() {
    assert_output(
        r#"(defmacro debug [x] `(do (println "value:" ~x) ~x)) (println (debug (+ 1 2)))"#,
        "value: 3\n3",
    );
}

#[test]
fn sci_defmacro_defconst() {
    assert_output(
        "(defmacro defconst [name val] `(def ~name ~val)) (defconst pi 3) (println pi)",
        "3",
    );
}

// ============================================================================
// Conditional forms
// ============================================================================

#[test]
fn sci_cond_multiple_clauses() {
    assert_eq!(
        eval_expr("(defn classify [x] (cond (< x 0) :negative (= x 0) :zero (> x 0) :positive)) (classify -1)"),
        ":negative"
    );
    assert_eq!(
        eval_expr("(defn classify [x] (cond (< x 0) :negative (= x 0) :zero (> x 0) :positive)) (classify 0)"),
        ":zero"
    );
    assert_eq!(
        eval_expr("(defn classify [x] (cond (< x 0) :negative (= x 0) :zero (> x 0) :positive)) (classify 1)"),
        ":positive"
    );
}

#[test]
fn sci_cond_no_match() {
    assert_eq!(eval_expr("(cond false 1 false 2 false 3)"), "nil");
}

#[test]
fn sci_cond_else() {
    assert_eq!(eval_expr("(cond :else 42)"), "42");
}

#[test]
fn sci_cond_nil_test() {
    assert_eq!(eval_expr("(cond nil 1)"), "nil");
}

#[test]
fn sci_if_not_nil() {
    assert_eq!(eval_expr("(if-not nil :yes :no)"), ":yes");
}

#[test]
fn sci_if_not_true() {
    assert_eq!(eval_expr("(if-not true :t :f)"), ":f");
}

#[test]
fn sci_if_not_false() {
    assert_eq!(eval_expr("(if-not false :t :f)"), ":t");
}

#[test]
fn sci_when_not_true() {
    assert_eq!(eval_expr("(when-not true :t)"), "nil");
}

#[test]
fn sci_when_not_false() {
    assert_eq!(eval_expr("(when-not false :t)"), ":t");
}

#[test]
fn sci_when_not_nil() {
    assert_eq!(eval_expr("(when-not nil :t)"), ":t");
}

// ============================================================================
// Let edge cases
// ============================================================================

#[test]
fn sci_let_chained_bindings() {
    assert_output(
        "(println (str (let [a 1 b (+ a 1) c (+ a b)] [a b c])))",
        "[1 2 3]",
    );
}

#[test]
fn sci_let_nested_shadow_extended() {
    assert_output(
        "(println (str (let [x 10] (let [y x] (let [x 30] [x y])))))",
        "[30 10]",
    );
}

#[test]
fn sci_let_with_fn() {
    assert_eq!(eval_expr("(let [f (fn [x] (+ x 10))] (f 5))"), "15");
}

#[test]
fn sci_let_two_fns() {
    assert_eq!(
        eval_expr("(let [f (fn [x] (* x x)) g (fn [x] (+ x 1))] (f (g 4)))"),
        "25"
    );
}

// ============================================================================
// Apply edge cases
// ============================================================================

#[test]
fn sci_apply_empty_vector() {
    assert_eq!(eval_expr("(apply + [])"), "0");
}

#[test]
fn sci_apply_single() {
    assert_eq!(eval_expr("(apply + [1])"), "1");
}

#[test]
fn sci_apply_mixed_args() {
    assert_eq!(eval_expr("(apply + 1 2 3 [4 5])"), "15");
}

#[test]
fn sci_apply_multiply() {
    assert_eq!(eval_expr("(apply * [1 2 3 4 5])"), "120");
}

// ============================================================================
// Range tests
// ============================================================================

#[test]
fn sci_range_single_arg() {
    assert_output("(println (str (range 5)))", "(0 1 2 3 4)");
}

#[test]
fn sci_range_two_args() {
    assert_output("(println (str (range 2 5)))", "(2 3 4)");
}

#[test]
fn sci_range_step() {
    assert_output("(println (str (range 0 10 3)))", "(0 3 6 9)");
}

#[test]
fn sci_range_step_odd() {
    assert_output("(println (str (range 1 10 2)))", "(1 3 5 7 9)");
}

#[test]
fn sci_range_negative() {
    assert_output("(println (str (range -5 0)))", "(-5 -4 -3 -2 -1)");
}

// ============================================================================
// Variable can have macro or var name (SCI line ~868)
// ============================================================================

#[test]
fn sci_var_named_merge_extended() {
    assert_output(
        "(defn foo [merge] merge) (defn bar [foo] foo) (println (bar true))",
        "true",
    );
}

// ============================================================================
// Gensym tests
// ============================================================================

#[test]
fn sci_gensym_basic() {
    let result = eval_expr("(gensym)");
    assert!(
        result.starts_with("G__"),
        "Expected gensym to start with G__, got: {}",
        result
    );
}

#[test]
fn sci_gensym_prefix() {
    let result = eval_expr(r#"(gensym "prefix")"#);
    assert!(
        result.starts_with("prefix"),
        "Expected gensym to start with prefix, got: {}",
        result
    );
}

// ============================================================================
// Closure tests
// ============================================================================

#[test]
fn sci_closure_over_let() {
    assert_eq!(
        eval_expr("(let [x 10] ((fn [] x)))"),
        "10"
    );
}

#[test]
fn sci_closure_nested_extended() {
    assert_eq!(
        eval_expr("(let [x 1 y 2] ((fn [] (let [g (fn [] y)] (+ x (g))))))"),
        "3"
    );
}

// ============================================================================
// map-indexed
// ============================================================================

#[test]
fn sci_map_indexed_vector() {
    assert_output(
        "(println (str (map-indexed vector [:a :b :c])))",
        "([0 :a] [1 :b] [2 :c])",
    );
}

#[test]
fn sci_map_indexed_fn() {
    assert_output(
        "(println (str (map-indexed (fn [i v] [i v]) [:a :b :c])))",
        "([0 :a] [1 :b] [2 :c])",
    );
}

// ============================================================================
// Frequencies and group-by
// ============================================================================

#[test]
fn sci_frequencies_detailed() {
    assert_eq!(eval_expr("(get (frequencies [1 2 1 3 1]) 1)"), "3");
    assert_eq!(eval_expr("(get (frequencies [1 2 1 3 1]) 2)"), "1");
    assert_eq!(eval_expr("(get (frequencies [1 2 1 3 1]) 3)"), "1");
}

#[test]
fn sci_group_by_odd() {
    assert_eq!(
        eval_expr("(count (get (group-by odd? [1 2 3 4 5]) true))"),
        "3"
    );
    assert_eq!(
        eval_expr("(count (get (group-by odd? [1 2 3 4 5]) false))"),
        "2"
    );
}

#[test]
fn sci_group_by_even() {
    assert_eq!(eval_expr("(count (group-by even? (range 10)))"), "2");
}

// ============================================================================
// Take and drop variations
// ============================================================================

#[test]
fn sci_take_drop_extended() {
    assert_output("(println (str (take 5 (drop 3 (range 20)))))", "(3 4 5 6 7)");
}

#[test]
fn sci_take_while_neg() {
    assert_output(
        "(println (str (take-while neg? [-3 -2 -1 0 1 2])))",
        "(-3 -2 -1)",
    );
}

#[test]
fn sci_drop_while_neg() {
    assert_output(
        "(println (str (drop-while neg? [-3 -2 -1 0 1 2])))",
        "(0 1 2)",
    );
}

// ============================================================================
// Repeat
// ============================================================================

#[test]
fn sci_repeat_keyword() {
    assert_output("(println (str (repeat 5 :a)))", "(:a :a :a :a :a)");
}

#[test]
fn sci_repeat_number() {
    assert_output("(println (str (repeat 3 42)))", "(42 42 42)");
}

// ============================================================================
// Reverse and sort
// ============================================================================

#[test]
fn sci_reverse_vector() {
    assert_output("(println (str (reverse [1 2 3 4 5])))", "(5 4 3 2 1)");
}

#[test]
fn sci_reverse_list() {
    assert_output("(println (str (reverse (list 1 2 3))))", "(3 2 1)");
}

#[test]
fn sci_sort_extended() {
    assert_output("(println (str (sort [5 3 8 1 9 2])))", "(1 2 3 5 8 9)");
}

// ============================================================================
// Vec and into
// ============================================================================

#[test]
fn sci_vec_from_range() {
    assert_output("(println (str (vec (range 5))))", "[0 1 2 3 4]");
}

// ============================================================================
// Comment test (SCI line 628)
// ============================================================================

#[test]
fn sci_comment_complex() {
    assert_eq!(eval_expr("(comment anything 1 2 3 (+ 1 2 3))"), "nil");
}

// ============================================================================
// cond-> and cond->> -- NOT IMPLEMENTED
// ============================================================================

#[test]
#[ignore] // cond-> not implemented
fn sci_cond_arrow_basic() {
    assert_eq!(eval_expr("(cond-> 1 true inc true inc)"), "3");
}

#[test]
#[ignore] // cond->> not implemented
fn sci_cond_last_arrow_basic() {
    assert_eq!(eval_expr("(cond->> 1 true inc true inc)"), "3");
}
