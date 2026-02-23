/// Tests ported from SCI (Small Clojure Interpreter) core_test.cljc
/// Source: https://github.com/babashka/sci
///
/// Each test is annotated with the original SCI test name and line number.
/// Tests that require features not yet implemented are marked #[ignore].
///
/// Known limitations of current implementation:
/// - pr-str / prn not available
/// - Maps print as "{... N entries}", sets as "#{... N elements}"
/// - Map destructuring ({:keys [a]}) not supported in let/fn
/// - Named fn self-reference (fn foo [x] (foo x)) doesn't work
/// - case not implemented
/// - for macro not implemented
/// - condp not implemented
/// - clojure.string not available
/// - many core functions missing

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
// core-test: do (SCI line 48)
// ============================================================================

#[test]
fn sci_core_do_returns_nil() {
    // (is (= [nil] (eval* "[(do 1 2 nil)]")))
    assert_eq!(eval_expr("(do 1 2 nil)"), "nil");
}

#[test]
fn sci_core_do_returns_last() {
    assert_eq!(eval_expr("(do 1 2 3)"), "3");
}

// ============================================================================
// core-test: if and when (SCI line 58)
// ============================================================================

#[test]
fn sci_core_if_true() {
    assert_eq!(eval_expr("(if true 10 20)"), "10");
}

#[test]
fn sci_core_if_false() {
    assert_eq!(eval_expr("(if false 10 20)"), "20");
}

#[test]
fn sci_core_when_true() {
    assert_eq!(eval_expr("(when true 0 1 2)"), "2");
}

#[test]
fn sci_core_when_false_is_nil() {
    assert_eq!(eval_expr("(when false 1)"), "nil");
}

// ============================================================================
// core-test: and and or (SCI line 80)
// ============================================================================

#[test]
fn sci_core_and_short_circuit() {
    assert_eq!(eval_expr("(and false true 0)"), "false");
}

#[test]
fn sci_core_and_returns_last() {
    assert_eq!(eval_expr("(and true true 0)"), "0");
}

#[test]
fn sci_core_or_returns_first_truthy() {
    assert_eq!(eval_expr("(or false false 1)"), "1");
}

#[test]
fn sci_core_or_all_false() {
    assert_eq!(eval_expr("(or false false false)"), "false");
}

#[test]
fn sci_core_or_many_nils_then_true() {
    assert_eq!(
        eval_expr("(or nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil nil true)"),
        "true"
    );
}

// ============================================================================
// core-test: fn literals (SCI line 87)
// ============================================================================

#[test]
fn sci_core_fn_literal_basic() {
    assert_eq!(eval_expr("(#(+ 1 %) 1)"), "2");
}

#[test]
fn sci_core_fn_literal_with_map() {
    assert_output("(println (str (map #(+ 1 %) [0 1 2])))", "(1 2 3)");
}

// ============================================================================
// core-test: map, keep (SCI line 93)
// ============================================================================

#[test]
fn sci_core_map_inc() {
    assert_output("(println (str (map inc [0 1 2])))", "(1 2 3)");
}

#[test]
fn sci_core_keep() {
    assert_output("(println (str (keep odd? [0 1 2])))", "(false true false)");
}

// ============================================================================
// core-test: calling IFns (SCI line 122)
// ============================================================================

#[test]
fn sci_core_map_as_fn_default() {
    // ({:a 1} 2 3) => 3 (key not found, return default)
    assert_eq!(eval_expr("({:a 1} 2 3)"), "3");
}

#[test]
fn sci_core_map_as_fn_found() {
    assert_eq!(eval_expr("({:a 1} :a 3)"), "1");
}

#[test]
fn sci_core_hashmap_as_fn() {
    assert_eq!(eval_expr("((hash-map :a 1) :a 3)"), "1");
}

#[test]
fn sci_core_set_as_fn() {
    assert_eq!(eval_expr("(#{:a :b :c} :a)"), ":a");
}

#[test]
fn sci_core_eval_fn_from_map() {
    assert_eq!(eval_expr("((get {:foo identity} :foo) 1)"), "1");
}

// ============================================================================
// destructure-test (SCI line 131)
// ============================================================================

#[test]
#[ignore] // map destructuring in let not supported
fn sci_destructure_keys() {
    assert_output("(println (let [{:keys [a]} {:a 1}] a))", "1");
}

#[test]
#[ignore] // map destructuring in fn not supported
fn sci_destructure_fn_keys() {
    assert_output("(println ((fn [{:keys [a]}] a) {:a 1}))", "1");
}

#[test]
#[ignore] // :or destructuring not supported
fn sci_destructure_default_false() {
    assert_output(
        "(println (let [{:keys [:a] :or {a false}} {:b 1}] a))",
        "false",
    );
}

// ============================================================================
// let-test (SCI line 162)
// ============================================================================

#[test]
fn sci_let_basic() {
    assert_eq!(eval_expr("(let [x 1 y (+ x x)] (str \"[\" x \" \" y \"]\"))"), "\"[1 2]\"");
}

#[test]
fn sci_let_basic_values() {
    assert_eq!(eval_expr("(let [x 1 y (+ x x)] y)"), "2");
}

#[test]
fn sci_let_multiple_body() {
    assert_eq!(eval_expr("(let [x 2] 1 2 3 x)"), "2");
}

#[test]
fn sci_let_nested_shadow() {
    // (let [x 1] [(let [x 2] x) x]) => [2 1]
    assert_output(
        "(let [x 1] (println (let [x 2] x)) (println x))",
        "2\n1",
    );
}

// ============================================================================
// closure-test (SCI line 185)
// ============================================================================

#[test]
#[ignore] // closure across top-level forms broken
fn sci_closure_defn_in_let() {
    assert_output("(let [x 1] (defn foo [] x)) (println (foo))", "1");
}

#[test]
fn sci_closure_nested() {
    assert_eq!(
        eval_expr("(let [x 1 y 2] ((fn [] (let [g (fn [] y)] (+ x (g))))))"),
        "3"
    );
}

// ============================================================================
// fn-literal-test (SCI line 191)
// ============================================================================

#[test]
fn sci_fn_literal_identity() {
    assert_output("(println (str (map #(do %) [1 2 3])))", "(1 2 3)");
}

#[test]
fn sci_fn_literal_map_indexed() {
    // map-indexed #([%1 %2]) [1 2 3] => ([0 1] [1 2] [2 3])
    assert_output(
        "(println (str (map-indexed #(do [%1 %2]) [1 2 3])))",
        "([0 1] [1 2] [2 3])",
    );
}

#[test]
fn sci_fn_literal_rest_args() {
    assert_output(
        "(println (str (apply #(do %&) [1 2 3])))",
        "(1 2 3)",
    );
}

// ============================================================================
// fn-test (SCI line 199)
// ============================================================================

#[test]
#[ignore] // named fn self-reference doesn't work
fn sci_fn_named_recursive() {
    assert_eq!(
        eval_expr("((fn foo [x] (if (< x 3) (foo (inc x)) x)) 0)"),
        "3"
    );
}

#[test]
#[ignore] // sequential destructuring in fn params not supported
fn sci_fn_seq_destructure_rest() {
    // ((fn [[x & xs]] xs) [1 2 3]) => (2 3)
    assert_output(
        "(println (str ((fn [[x & xs]] xs) [1 2 3])))",
        "(2 3)",
    );
}

#[test]
fn sci_fn_rest_params() {
    assert_output(
        "(println (str ((fn [x & xs] xs) 1 2 3)))",
        "(2 3)",
    );
}

#[test]
#[ignore] // destructuring in rest params not supported
fn sci_fn_rest_destructure_first() {
    assert_eq!(eval_expr("((fn [x & [y]] y) 1 2 3)"), "2");
}

#[test]
fn sci_fn_multi_arity_single() {
    assert_eq!(eval_expr("((fn ([x] x) ([x y] y)) 1)"), "1");
}

#[test]
fn sci_fn_multi_arity_double() {
    assert_eq!(eval_expr("((fn ([x] x) ([x y] y)) 1 2)"), "2");
}

#[test]
fn sci_fn_variadic_vs_fixed_1arg() {
    assert_eq!(
        eval_expr("((fn ([x & xs] \"variadic\") ([x] \"otherwise\")) 1)"),
        "\"otherwise\""
    );
}

#[test]
fn sci_fn_variadic_vs_fixed_2arg() {
    assert_eq!(
        eval_expr("((fn ([x] \"otherwise\") ([x & xs] \"variadic\")) 1 2)"),
        "\"variadic\""
    );
}

#[test]
fn sci_fn_apply_with_rest() {
    assert_output(
        "(println (str (apply (fn [x & xs] xs) 1 2 [3 4])))",
        "(2 3 4)",
    );
}

// ============================================================================
// def-test (SCI line 229)
// ============================================================================

#[test]
fn sci_def_basic() {
    assert_output(r#"(def foo "nice val") (println foo)"#, "nice val");
}

#[test]
fn sci_def_with_docstring() {
    assert_output(r#"(def foo) (def foo "docstring" 2) (println foo)"#, "2");
}

#[test]
fn sci_def_in_try() {
    assert_output("(println (try (def x 1) x))", "1");
}

#[test]
fn sci_defn_in_try() {
    assert_output("(println (try (defn x [] 1) (x)))", "1");
}

// ============================================================================
// defn-test (SCI line 260)
// ============================================================================

#[test]
fn sci_defn_basic() {
    assert_output(
        r#"(defn foo "increment" [x] (inc x)) (println (foo 1))"#,
        "2",
    );
}

#[test]
fn sci_defn_multi_arity() {
    assert_output(
        "(defn foo ([x] (inc x)) ([x y] (+ x y))) (println (foo 1 2))",
        "3",
    );
}

#[test]
fn sci_defn_redefine() {
    assert_output(
        r#"(defn foo [x] (inc x)) (defn foo "dec" [x] (dec x)) (println (foo 1))"#,
        "0",
    );
}

// ============================================================================
// threading macros (SCI line 448, 1397)
// ============================================================================

#[test]
fn sci_thread_first_basic() {
    assert_eq!(eval_expr("(-> 3 inc inc inc)"), "6");
}

#[test]
fn sci_thread_first_complex() {
    assert_eq!(eval_expr("(-> 1 inc inc inc)"), "4");
}

#[test]
fn sci_thread_last_basic() {
    assert_output(
        r#"(println (->> ["foo" "baaar" "baaaaaz"] (map count) (apply max)))"#,
        "7",
    );
}

// ============================================================================
// comment-test (SCI line 628)
// ============================================================================

#[test]
fn sci_comment_returns_nil() {
    assert_eq!(eval_expr("(comment \"anything\")"), "nil");
    assert_eq!(eval_expr("(comment anything)"), "nil");
    assert_eq!(eval_expr("(comment 1)"), "nil");
    assert_eq!(eval_expr("(comment (+ 1 2 (* 3 4)))"), "nil");
}

// ============================================================================
// recur-test (SCI line 663)
// ============================================================================

#[test]
fn sci_recur_basic() {
    assert_output(
        "(defn hello [x] (if (< x 10000) (recur (inc x)) x)) (println (hello 0))",
        "10000",
    );
}

#[test]
fn sci_recur_variadic() {
    assert_output(
        "(println (str ((fn [& args] (if-let [x (next args)] (recur x) args)) 1 2 3 4)))",
        "(4)",
    );
}

#[test]
fn sci_recur_variadic_with_fixed() {
    assert_output(
        "(println (str ((fn [x & args] (if-let [x (next args)] (recur x x) x)) nil 2 3 4)))",
        "(4)",
    );
}

#[test]
fn sci_recur_defn_with_apply() {
    assert_output(
        "(defn foo [x & xs] (if (pos? x) (recur (dec x) (rest xs)) xs)) (println (str (apply foo 10 (range 11))))",
        "(10)",
    );
}

#[test]
#[ignore] // named fn self-reference not supported
fn sci_recursion_depth() {
    assert_output(
        "(println ((fn foo [x] (if (= 72 x) x (foo (inc x)))) 0))",
        "72",
    );
}

// ============================================================================
// loop-test (SCI line 736)
// ============================================================================

#[test]
#[ignore] // destructuring in loop bindings not supported
fn sci_loop_destructure() {
    assert_eq!(
        eval_expr("(loop [[x y] [1 2]] (if (= x 3) y (recur [(inc x) y])))"),
        "2"
    );
}

#[test]
fn sci_loop_conj_list() {
    assert_output(
        "(println (str (loop [l (list 2 1) c (count l)] (if (> c 4) l (recur (conj l (inc c)) (inc c))))))",
        "(5 4 3 2 1)",
    );
}

#[test]
fn sci_loop_let_shadow() {
    assert_eq!(eval_expr("(let [x 1] (loop [x (inc x)] x))"), "2");
}

// ============================================================================
// for-test (SCI line 768)
// ============================================================================

#[test]
#[ignore] // for macro not implemented
fn sci_for_while_when() {
    assert_output(
        "(println (str (for [i [1 2 3] :while (< i 2) j [4 5 6] :when (even? j)] [i j])))",
        "([1 4] [1 6])",
    );
}

#[test]
#[ignore] // for macro not implemented
fn sci_for_nested_destructure() {
    assert_output(
        "(println (str (for [[_ counts] [[1 [1 2 3]] [3 [1 2 3]]] c counts] c)))",
        "(1 2 3 1 2 3)",
    );
}

// ============================================================================
// cond-test (SCI line 800)
// ============================================================================

#[test]
fn sci_cond_match_int() {
    assert_eq!(eval_expr("(let [x 2] (cond (string? x) 1 (int? x) 2))"), "2");
}

#[test]
fn sci_cond_else() {
    assert_eq!(eval_expr("(let [x 2] (cond (string? x) 1 :else 2))"), "2");
}

// ============================================================================
// condp-test (SCI line 810)
// ============================================================================

#[test]
#[ignore] // condp not implemented
fn sci_condp_basic() {
    assert_eq!(eval_expr("(condp = 1 1 \"one\")"), "\"one\"");
}

// ============================================================================
// case-test (SCI line 821)
// ============================================================================

#[test]
#[ignore] // case not implemented
fn sci_case_match() {
    assert_eq!(eval_expr("(case 1, 1 true, 2 (+ 1 2 3), 6)"), "true");
}

#[test]
#[ignore] // case not implemented
fn sci_case_default() {
    assert_eq!(eval_expr("(case (inc 2), 1 true, 2 (+ 1 2 3), 7)"), "7");
}

// ============================================================================
// variable-can-have-macro-or-var-name (SCI line 868)
// ============================================================================

#[test]
fn sci_var_named_merge() {
    assert_output("(defn foo [merge] merge) (println (foo true))", "true");
}

#[test]
fn sci_var_named_comment() {
    assert_output("(defn foo [comment] comment) (println (foo true))", "true");
}

#[test]
#[ignore] // fn as param name shadows special form
fn sci_var_named_fn() {
    assert_output("(defn foo [fn] (fn 1)) (println (foo inc))", "2");
}

// ============================================================================
// try-catch (SCI line 879)
// ============================================================================

#[test]
fn sci_try_returns_body() {
    assert_eq!(eval_expr("(try 1 2 3)"), "3");
}

#[test]
fn sci_try_returns_quoted() {
    assert_eq!(eval_expr("(try 'hello)"), "hello");
}

#[test]
fn sci_try_nil_in_body() {
    assert_eq!(eval_expr("(try 1 2 nil)"), "nil");
}

#[test]
fn sci_try_nil_then_value() {
    assert_eq!(eval_expr("(try 1 2 nil 1)"), "1");
}

// ============================================================================
// letfn-test (SCI line 1073)
// ============================================================================

#[test]
#[ignore] // letfn not implemented
fn sci_letfn_multi_arity() {
    assert_eq!(
        eval_expr("(letfn [(f ([x] (f x 1)) ([x y] (+ x y)))] (f 1))"),
        "2"
    );
}

#[test]
#[ignore] // letfn not implemented
fn sci_letfn_mutual_recursion() {
    assert_eq!(
        eval_expr("(letfn [(f [x] (g x)) (g [x] (inc x))] (f 10))"),
        "11"
    );
}

// ============================================================================
// defn--test (SCI line 1089)
// ============================================================================

#[test]
fn sci_defn_private() {
    assert_output("(defn- foo [] 1) (println (foo))", "1");
}

// ============================================================================
// defonce-test (SCI line 1105)
// ============================================================================

#[test]
#[ignore] // defonce not implemented
fn sci_defonce() {
    assert_output("(defonce x 1) (defonce x 2) (println x)", "1");
}

// ============================================================================
// ifs-test (SCI line 1250)
// ============================================================================

#[test]
fn sci_if_let_nil() {
    assert_eq!(eval_expr("(if-let [foo nil] 1 2)"), "2");
}

#[test]
fn sci_if_let_false() {
    assert_eq!(eval_expr("(if-let [foo false] 1 2)"), "2");
}

#[test]
fn sci_if_let_truthy() {
    assert_eq!(eval_expr("(if-let [foo 42] foo 0)"), "42");
}

#[test]
#[ignore] // if-some not implemented
fn sci_if_some_nil() {
    assert_eq!(eval_expr("(if-some [foo nil] 1 2)"), "2");
}

#[test]
#[ignore] // if-some not implemented
fn sci_if_some_false() {
    assert_eq!(eval_expr("(if-some [foo false] 1 2)"), "1");
}

// ============================================================================
// whens-test (SCI line 1256)
// ============================================================================

#[test]
fn sci_when_let_nil() {
    assert_eq!(eval_expr("(when-let [foo nil] 1)"), "nil");
}

#[test]
fn sci_when_let_false() {
    assert_eq!(eval_expr("(when-let [foo false] 1)"), "nil");
}

#[test]
fn sci_when_let_truthy() {
    assert_eq!(eval_expr("(when-let [foo 42] foo)"), "42");
}

#[test]
#[ignore] // when-some not implemented
fn sci_when_some_nil() {
    assert_eq!(eval_expr("(when-some [foo nil] 1)"), "nil");
}

// ============================================================================
// self-ref-test (SCI line 1507)
// ============================================================================

#[test]
#[ignore] // named fn self-reference broken
fn sci_self_ref_fn_equality() {
    assert_output("(def f (fn foo [] foo)) (println (= f (f)))", "true");
}

#[test]
#[ignore] // named fn self-reference broken
fn sci_self_ref_closure() {
    assert_output(
        r#"(defn foof [x] (let [f (fn f ([] (f nil)) ([_] x))] f))
(def f1 (foof :a))
(println (f1))"#,
        ":a",
    );
}

// ============================================================================
// Arithmetic operations
// ============================================================================

#[test]
fn sci_add_variadic() {
    assert_eq!(eval_expr("(+ 1 2 3)"), "6");
    assert_eq!(eval_expr("(+ 1 2 3 4 5)"), "15");
}

#[test]
fn sci_mul_variadic() {
    assert_eq!(eval_expr("(* 1 2 3 4)"), "24");
}

#[test]
fn sci_sub_variadic() {
    assert_eq!(eval_expr("(- 10 3 2)"), "5");
}

#[test]
fn sci_unary_ops() {
    assert_eq!(eval_expr("(+ 5)"), "5");
    assert_eq!(eval_expr("(- 5)"), "-5");
    assert_eq!(eval_expr("(* 5)"), "5");
}

#[test]
fn sci_zero_arity() {
    assert_eq!(eval_expr("(+)"), "0");
    assert_eq!(eval_expr("(*)"), "1");
}

#[test]
fn sci_comparisons() {
    assert_eq!(eval_expr("(< 1 2)"), "true");
    assert_eq!(eval_expr("(< 2 1)"), "false");
    assert_eq!(eval_expr("(<= 1 1)"), "true");
    assert_eq!(eval_expr("(> 2 1)"), "true");
    assert_eq!(eval_expr("(>= 2 2)"), "true");
    assert_eq!(eval_expr("(= 1 1)"), "true");
    assert_eq!(eval_expr("(not= 1 2)"), "true");
}

#[test]
fn sci_min_max() {
    assert_eq!(eval_expr("(min 1 2 3)"), "1");
    assert_eq!(eval_expr("(max 1 2 3)"), "3");
}

#[test]
fn sci_inc_dec() {
    assert_eq!(eval_expr("(inc 0)"), "1");
    assert_eq!(eval_expr("(dec 1)"), "0");
}

#[test]
fn sci_numeric_predicates() {
    assert_eq!(eval_expr("(zero? 0)"), "true");
    assert_eq!(eval_expr("(pos? 1)"), "true");
    assert_eq!(eval_expr("(neg? -1)"), "true");
    assert_eq!(eval_expr("(even? 2)"), "true");
    assert_eq!(eval_expr("(odd? 3)"), "true");
}

#[test]
fn sci_mod_rem() {
    assert_eq!(eval_expr("(mod 10 3)"), "1");
    assert_eq!(eval_expr("(rem 10 3)"), "1");
}

#[test]
fn sci_abs() {
    assert_eq!(eval_expr("(abs -5)"), "5");
    assert_eq!(eval_expr("(abs 5)"), "5");
}

// ============================================================================
// Bit operations
// ============================================================================

#[test]
fn sci_bit_ops() {
    assert_eq!(eval_expr("(bit-and 255 15)"), "15");
    assert_eq!(eval_expr("(bit-or 15 240)"), "255");
    assert_eq!(eval_expr("(bit-xor 255 15)"), "240");
    assert_eq!(eval_expr("(bit-not 0)"), "-1");
    assert_eq!(eval_expr("(bit-shift-left 1 4)"), "16");
    assert_eq!(eval_expr("(bit-shift-right 16 4)"), "1");
}

// ============================================================================
// Logic
// ============================================================================

#[test]
fn sci_not() {
    assert_eq!(eval_expr("(not true)"), "false");
    assert_eq!(eval_expr("(not false)"), "true");
    assert_eq!(eval_expr("(not nil)"), "true");
    assert_eq!(eval_expr("(not 1)"), "false");
}

// ============================================================================
// Predicates
// ============================================================================

#[test]
fn sci_nil_pred() {
    assert_eq!(eval_expr("(nil? nil)"), "true");
    assert_eq!(eval_expr("(nil? 1)"), "false");
}

#[test]
fn sci_bool_preds() {
    assert_eq!(eval_expr("(true? true)"), "true");
    assert_eq!(eval_expr("(true? 1)"), "false");
    assert_eq!(eval_expr("(false? false)"), "true");
    assert_eq!(eval_expr("(false? nil)"), "false");
}

#[test]
fn sci_type_predicates() {
    assert_eq!(eval_expr("(number? 42)"), "true");
    assert_eq!(eval_expr("(number? :a)"), "false");
    assert_eq!(eval_expr("(string? \"hi\")"), "true");
    assert_eq!(eval_expr("(string? 1)"), "false");
    assert_eq!(eval_expr("(keyword? :a)"), "true");
    assert_eq!(eval_expr("(symbol? 'a)"), "true");
    assert_eq!(eval_expr("(vector? [1 2])"), "true");
    assert_eq!(eval_expr("(map? {:a 1})"), "true");
    assert_eq!(eval_expr("(fn? inc)"), "true");
}

#[test]
fn sci_integer_float_preds() {
    assert_eq!(eval_expr("(integer? 1)"), "true");
    assert_eq!(eval_expr("(integer? 1.0)"), "false");
    assert_eq!(eval_expr("(float? 1.0)"), "true");
    assert_eq!(eval_expr("(float? 1)"), "false");
}

#[test]
fn sci_coll_pred() {
    assert_eq!(eval_expr("(coll? [1])"), "true");
    assert_eq!(eval_expr("(coll? {:a 1})"), "true");
    assert_eq!(eval_expr("(coll? 1)"), "false");
}

// ============================================================================
// Sequence operations
// ============================================================================

#[test]
fn sci_first_rest_next() {
    assert_eq!(eval_expr("(first [1 2 3])"), "1");
    assert_output("(println (str (rest [1 2 3])))", "(2 3)");
    assert_output("(println (str (next [1 2 3])))", "(2 3)");
    assert_eq!(eval_expr("(next [1])"), "nil");
}

#[test]
fn sci_nth() {
    assert_eq!(eval_expr("(nth [10 20 30] 0)"), "10");
    assert_eq!(eval_expr("(nth [10 20 30] 2)"), "30");
}

#[test]
fn sci_count() {
    assert_eq!(eval_expr("(count [1 2 3])"), "3");
    assert_eq!(eval_expr("(count [])"), "0");
    assert_eq!(eval_expr("(count {:a 1 :b 2})"), "2");
}

#[test]
fn sci_conj_vector() {
    assert_output("(println (str (conj [1 2] 3)))", "[1 2 3]");
}

#[test]
fn sci_conj_list() {
    assert_output("(println (str (conj '(1 2) 3)))", "(3 1 2)");
}

#[test]
fn sci_cons() {
    assert_output("(println (str (cons 0 [1 2 3])))", "(0 1 2 3)");
}

#[test]
fn sci_concat() {
    assert_output(
        "(println (str (concat [1 2] [3 4] [5])))",
        "(1 2 3 4 5)",
    );
}

#[test]
fn sci_into_vector() {
    assert_output("(println (str (into [] '(1 2 3))))", "[1 2 3]");
}

#[test]
fn sci_reverse() {
    assert_output("(println (str (reverse [1 2 3])))", "(3 2 1)");
}

#[test]
fn sci_sort() {
    assert_output("(println (str (sort [3 1 2])))", "(1 2 3)");
}

#[test]
fn sci_distinct() {
    assert_output(
        "(println (str (distinct [1 2 1 3 2 4])))",
        "(1 2 3 4)",
    );
}

#[test]
fn sci_interleave() {
    assert_output(
        "(println (str (interleave [1 2 3] [:a :b :c])))",
        "(1 :a 2 :b 3 :c)",
    );
}

#[test]
fn sci_partition() {
    assert_output(
        "(println (str (partition 2 [1 2 3 4 5])))",
        "((1 2) (3 4))",
    );
}

#[test]
fn sci_last() {
    assert_eq!(eval_expr("(last [1 2 3])"), "3");
}

#[test]
fn sci_butlast() {
    assert_output("(println (str (butlast [1 2 3])))", "(1 2)");
}

#[test]
fn sci_flatten() {
    assert_output(
        "(println (str (flatten [1 [2 [3 4]] 5])))",
        "(1 2 3 4 5)",
    );
}

#[test]
fn sci_vec_from_list() {
    assert_output("(println (str (vec '(1 2 3))))", "[1 2 3]");
}

// ============================================================================
// Higher-order functions
// ============================================================================

#[test]
fn sci_map_basic() {
    assert_output("(println (str (map inc [0 1 2])))", "(1 2 3)");
}

#[test]
fn sci_map_multiple_colls() {
    assert_output(
        "(println (str (map + [1 2 3] [10 20 30])))",
        "(11 22 33)",
    );
}

#[test]
fn sci_filter() {
    assert_output(
        "(println (str (filter even? [1 2 3 4 5 6])))",
        "(2 4 6)",
    );
}

#[test]
fn sci_remove() {
    assert_output(
        "(println (str (remove even? [1 2 3 4 5 6])))",
        "(1 3 5)",
    );
}

#[test]
fn sci_reduce() {
    assert_eq!(eval_expr("(reduce + [1 2 3 4 5])"), "15");
    assert_eq!(eval_expr("(reduce + 10 [1 2 3])"), "16");
}

#[test]
fn sci_reduce_with_map_filter() {
    assert_eq!(
        eval_expr("(reduce + 0 (filter odd? (map inc [0 1 2 3 4])))"),
        "9"
    );
}

#[test]
fn sci_apply() {
    assert_eq!(eval_expr("(apply + [1 2 3])"), "6");
    assert_eq!(eval_expr("(apply + 1 2 [3 4])"), "10");
    assert_eq!(eval_expr("(apply str [\"a\" \"b\" \"c\"])"), "\"abc\"");
}

#[test]
fn sci_some() {
    assert_eq!(eval_expr("(some even? [1 3 5 6])"), "true");
    assert_eq!(eval_expr("(some even? [1 3 5])"), "nil");
}

#[test]
fn sci_every() {
    assert_eq!(eval_expr("(every? even? [2 4 6])"), "true");
    assert_eq!(eval_expr("(every? even? [2 3 6])"), "false");
}

#[test]
fn sci_not_every() {
    assert_eq!(eval_expr("(not-every? even? [2 4 6])"), "false");
    assert_eq!(eval_expr("(not-every? even? [2 3 6])"), "true");
}

#[test]
fn sci_not_any() {
    assert_eq!(eval_expr("(not-any? even? [1 3 5])"), "true");
    assert_eq!(eval_expr("(not-any? even? [1 2 5])"), "false");
}

#[test]
fn sci_mapcat() {
    assert_output(
        "(println (str (mapcat #(vector % (* % %)) [1 2 3])))",
        "(1 1 2 4 3 9)",
    );
}

#[test]
fn sci_take_drop() {
    assert_output("(println (str (take 3 [1 2 3 4 5])))", "(1 2 3)");
    assert_output("(println (str (drop 3 [1 2 3 4 5])))", "(4 5)");
}

#[test]
fn sci_take_while() {
    assert_output(
        "(println (str (take-while #(< % 4) [1 2 3 4 5])))",
        "(1 2 3)",
    );
}

#[test]
fn sci_drop_while() {
    assert_output(
        "(println (str (drop-while #(< % 4) [1 2 3 4 5])))",
        "(4 5)",
    );
}

#[test]
#[ignore] // infinite lazy sequences not supported (no laziness)
fn sci_repeat_infinite() {
    assert_output("(println (str (take 3 (repeat 5))))", "(5 5 5)");
}

#[test]
fn sci_repeat_bounded() {
    assert_output("(println (str (repeat 3 :a)))", "(:a :a :a)");
}

#[test]
#[ignore] // infinite lazy sequences not supported (no laziness)
fn sci_iterate() {
    assert_output(
        "(println (str (take 5 (iterate inc 0))))",
        "(0 1 2 3 4)",
    );
}

#[test]
fn sci_range() {
    assert_output("(println (str (range 5)))", "(0 1 2 3 4)");
    assert_output("(println (str (range 2 5)))", "(2 3 4)");
    assert_output("(println (str (range 0 10 3)))", "(0 3 6 9)");
}

// ============================================================================
// Collection operations (maps, sets)
// ============================================================================

#[test]
fn sci_get() {
    assert_eq!(eval_expr("(get {:a 1 :b 2} :a)"), "1");
    assert_eq!(eval_expr("(get {:a 1} :b)"), "nil");
    assert_eq!(eval_expr("(get {:a 1} :b :default)"), ":default");
}

#[test]
fn sci_contains() {
    assert_eq!(eval_expr("(contains? {:a 1} :a)"), "true");
    assert_eq!(eval_expr("(contains? {:a 1} :b)"), "false");
}

#[test]
fn sci_assoc() {
    assert_eq!(eval_expr("(get (assoc {:a 1} :b 2) :b)"), "2");
    assert_eq!(eval_expr("(count (assoc {:a 1} :b 2))"), "2");
}

#[test]
fn sci_dissoc() {
    assert_eq!(eval_expr("(contains? (dissoc {:a 1 :b 2} :b) :b)"), "false");
    assert_eq!(eval_expr("(count (dissoc {:a 1 :b 2} :b))"), "1");
}

#[test]
fn sci_merge() {
    assert_eq!(eval_expr("(get (merge {:a 1} {:b 2} {:c 3}) :c)"), "3");
    assert_eq!(eval_expr("(count (merge {:a 1} {:b 2} {:c 3}))"), "3");
}

#[test]
fn sci_update() {
    assert_eq!(eval_expr("(get (update {:a 1} :a inc) :a)"), "2");
}

#[test]
fn sci_get_in() {
    assert_eq!(eval_expr("(get-in {:a {:b {:c 42}}} [:a :b :c])"), "42");
}

#[test]
fn sci_assoc_in() {
    assert_eq!(eval_expr("(get-in (assoc-in {} [:a :b :c] 42) [:a :b :c])"), "42");
}

#[test]
fn sci_update_in() {
    assert_eq!(
        eval_expr("(get-in (update-in {:a {:b 1}} [:a :b] inc) [:a :b])"),
        "2"
    );
}

#[test]
fn sci_select_keys() {
    assert_eq!(
        eval_expr("(count (select-keys {:a 1 :b 2 :c 3} [:a :c]))"),
        "2"
    );
    assert_eq!(
        eval_expr("(get (select-keys {:a 1 :b 2 :c 3} [:a :c]) :a)"),
        "1"
    );
}

#[test]
fn sci_keys_vals() {
    assert_eq!(eval_expr("(count (keys {:a 1 :b 2}))"), "2");
    assert_eq!(eval_expr("(count (vals {:a 1 :b 2}))"), "2");
}

#[test]
fn sci_zipmap() {
    assert_eq!(eval_expr("(get (zipmap [:a :b :c] [1 2 3]) :b)"), "2");
    assert_eq!(eval_expr("(count (zipmap [:a :b :c] [1 2 3]))"), "3");
}

#[test]
fn sci_set_from_vector() {
    assert_eq!(eval_expr("(count (set [1 2 2 3 3 3]))"), "3");
    assert_eq!(eval_expr("(contains? (set [1 2 3]) 2)"), "true");
}

#[test]
fn sci_disj() {
    assert_eq!(eval_expr("(count (disj #{1 2 3} 2))"), "2");
    assert_eq!(eval_expr("(contains? (disj #{1 2 3} 2) 2)"), "false");
}

// ============================================================================
// Keyword operations
// ============================================================================

#[test]
fn sci_keyword_as_fn() {
    assert_eq!(eval_expr("(:a {:a 42 :b 99})"), "42");
    assert_eq!(eval_expr("(:c {:a 1})"), "nil");
    assert_eq!(eval_expr("(:c {:a 1} :default)"), ":default");
}

// ============================================================================
// String operations
// ============================================================================

#[test]
fn sci_str() {
    assert_eq!(eval_expr("(str \"hello\" \" \" \"world\")"), "\"hello world\"");
    assert_eq!(eval_expr("(str 1 2 3)"), "\"123\"");
    assert_eq!(eval_expr("(str nil)"), "\"\"");
}

#[test]
fn sci_name() {
    assert_eq!(eval_expr("(name :foo)"), "\"foo\"");
    assert_eq!(eval_expr("(name 'bar)"), "\"bar\"");
}

#[test]
#[ignore] // clojure.string not available
fn sci_str_upper_lower() {
    assert_output("(println (clojure.string/upper-case \"hello\"))", "HELLO");
    assert_output("(println (clojure.string/lower-case \"HELLO\"))", "hello");
}

#[test]
#[ignore] // clojure.string not available
fn sci_str_join() {
    assert_output("(println (clojure.string/join \", \" [1 2 3]))", "1, 2, 3");
}

#[test]
#[ignore] // clojure.string not available
fn sci_str_includes() {
    assert_eq!(
        eval_expr("(clojure.string/includes? \"hello world\" \"world\")"),
        "true"
    );
}

// ============================================================================
// Empty / seq
// ============================================================================

#[test]
fn sci_empty() {
    assert_eq!(eval_expr("(empty? [])"), "true");
    assert_eq!(eval_expr("(empty? [1])"), "false");
    assert_eq!(eval_expr("(empty? nil)"), "true");
}

#[test]
fn sci_seq_on_empty() {
    assert_eq!(eval_expr("(seq [])"), "nil");
    assert_eq!(eval_expr("(seq nil)"), "nil");
}

// ============================================================================
// Identity, constantly, comp, partial, complement, juxt
// ============================================================================

#[test]
fn sci_identity() {
    assert_eq!(eval_expr("(identity 42)"), "42");
}

#[test]
fn sci_constantly() {
    assert_eq!(eval_expr("((constantly 5) 1 2 3)"), "5");
}

#[test]
fn sci_comp() {
    assert_eq!(eval_expr("((comp inc inc inc) 0)"), "3");
}

#[test]
fn sci_partial() {
    assert_eq!(eval_expr("((partial + 10) 5)"), "15");
}

#[test]
fn sci_complement() {
    assert_eq!(eval_expr("((complement nil?) 1)"), "true");
    assert_eq!(eval_expr("((complement nil?) nil)"), "false");
}

#[test]
fn sci_juxt() {
    // Can't check full output since it would be a vector
    // Check individual results via nth
    assert_eq!(eval_expr("(nth ((juxt inc dec) 1) 0)"), "2");
    assert_eq!(eval_expr("(nth ((juxt inc dec) 1) 1)"), "0");
}

// ============================================================================
// Do form
// ============================================================================

#[test]
fn sci_do_returns_last() {
    assert_eq!(eval_expr("(do 1 2 3)"), "3");
}

#[test]
fn sci_do_side_effects() {
    assert_output("(do (println 1) (println 2))", "1\n2");
}

// ============================================================================
// Frequencies, group-by
// ============================================================================

#[test]
fn sci_frequencies() {
    // Can't print map, check individual counts
    assert_eq!(eval_expr("(get (frequencies [:a :b :a :c :b :a]) :a)"), "3");
    assert_eq!(eval_expr("(get (frequencies [:a :b :a :c :b :a]) :b)"), "2");
    assert_eq!(eval_expr("(get (frequencies [:a :b :a :c :b :a]) :c)"), "1");
}

#[test]
fn sci_group_by() {
    assert_eq!(
        eval_expr("(count (get (group-by odd? [1 2 3 4 5]) true))"),
        "3"
    );
    assert_eq!(
        eval_expr("(count (get (group-by odd? [1 2 3 4 5]) false))"),
        "2"
    );
}

// ============================================================================
// While, atom (SCI line 1273)
// ============================================================================

#[test]
#[ignore] // atom/swap!/deref not implemented
fn sci_while_atom() {
    assert_output(
        "(def a (atom 0)) (while (< @a 10) (swap! a inc)) (println @a)",
        "10",
    );
}

// ============================================================================
// Delay (SCI line 1086)
// ============================================================================

#[test]
#[ignore] // delay/deref not implemented
fn sci_delay() {
    assert_eq!(eval_expr("@(delay 1)"), "1");
}

// ============================================================================
// Trampoline (SCI line 645)
// ============================================================================

#[test]
#[ignore] // trampoline not implemented
fn sci_trampoline() {
    assert_output(
        "(defn hello [x] (if (< x 10000) #(hello (inc x)) x)) (println (trampoline hello 0))",
        "10000",
    );
}
