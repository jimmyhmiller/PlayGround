//! Variadic params: `(fn [a b & rs] …)` and `(defn f [& args] …)`.

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn variadic_only_rest() {
    // (defn list [& xs] xs) — the bootstrap definition of `list`.
    assert_eq!(
        eval_str("(def list (fn [& xs] xs)) (list 1 2 3)"),
        "(1 2 3)"
    );
}

#[test]
fn variadic_only_rest_empty() {
    // Empty rest is represented as `nil` (the same bit pattern as
    // `()`) in this implementation. core.clj will distinguish
    // EmptyList from nil at the user-visible layer; the runtime
    // primitive treats them as equivalent.
    assert_eq!(eval_str("(def list (fn [& xs] xs)) (list)"), "nil");
}

#[test]
fn variadic_with_fixed() {
    assert_eq!(
        eval_str("(def f (fn [a & rs] rs)) (f 10 20 30 40)"),
        "(20 30 40)"
    );
}

#[test]
fn variadic_with_fixed_only_min() {
    // Calling with exactly min_arity → rest is the empty list,
    // represented as nil (see comment on variadic_only_rest_empty).
    assert_eq!(eval_str("(def f (fn [a & rs] rs)) (f 99)"), "nil");
}

#[test]
fn variadic_fixed_value_returned() {
    // First fixed arg returned correctly.
    assert_eq!(eval_str("(def f (fn [a & rs] a)) (f 7 8 9)"), "7");
}

#[test]
fn variadic_two_fixed() {
    assert_eq!(
        eval_str("(def f (fn [a b & rs] (+ a b))) (f 10 32 100 200)"),
        "42"
    );
}

#[test]
fn variadic_closure_via_let() {
    // (fn [& xs] xs) as an expression — captured by a let binding,
    // then called indirectly. Indirect path packs args into a list
    // and the body extracts them via the same prologue as the
    // static-call path.
    assert_eq!(
        eval_str("(let [collect (fn [& xs] xs)] (collect 1 2 3 4 5))"),
        "(1 2 3 4 5)"
    );
}

#[test]
fn variadic_closure_with_capture() {
    // The closure captures `n` from the outer let; the variadic
    // body extracts a sum from the list.
    assert_eq!(
        eval_str(
            "(let [n 100] \
                (let [add-all (fn [a b & rs] \
                                 (+ a (+ b (+ (__reader_list_count rs) n))))] \
                  (add-all 1 2 :x :y :z)))"
        ),
        // a=1, b=2, count(rs)=3, n=100 → 106
        "106"
    );
}

#[test]
fn variadic_closure_recur() {
    // recur inside a variadic closure body must repack the user
    // args into a fresh list before re-entering the loop_header.
    let src = "\
        (def cdr-n (fn [n & xs] \
            (if (= n 0) xs (recur (- n 1) (__reader_list_first (__reader_list_rest xs)))))) \
        (cdr-n 0 :a :b :c)";
    assert_eq!(eval_str(src), "(:a :b :c)");
}

// NOTE: an arity-mismatch test would `panic::catch_unwind` the
// eval, but extern "C" panics across the JIT boundary cannot
// currently unwind cleanly (Rust 2024 turns them into SIGABRT). The
// runtime DOES detect arity violations — `__check_args_list` panics
// with a clear ArityException message — but the panic aborts the
// process. Add the catchable version once the runtime moves to
// `extern "C-unwind"`.
