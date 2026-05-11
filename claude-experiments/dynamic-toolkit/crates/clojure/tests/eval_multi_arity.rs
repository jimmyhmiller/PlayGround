//! Multi-arity fn definitions: `(fn ([x] body1) ([x y] body2))`.

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn defn_two_arities_pick_correct() {
    let src = "\
        (def f (fn ([x] x) ([x y] (+ x y)))) \
        (+ (f 7) (f 10 20))";
    assert_eq!(eval_str(src), "37"); // 7 + 30
}

#[test]
fn defn_arity_2_branch() {
    let src = "\
        (def f (fn ([x] :one) ([x y] :two))) \
        (f 1 2)";
    assert_eq!(eval_str(src), ":two");
}

#[test]
fn defn_arity_1_branch() {
    let src = "\
        (def f (fn ([x] :one) ([x y] :two))) \
        (f 1)";
    assert_eq!(eval_str(src), ":one");
}

#[test]
fn defn_three_arities() {
    let src = "\
        (def f (fn ([] 0) ([a] a) ([a b] (+ a b)))) \
        (+ (f) (+ (f 5) (f 10 20)))";
    assert_eq!(eval_str(src), "35"); // 0 + 5 + 30
}

#[test]
fn defn_with_variadic_clause() {
    // Clause selection: 1-arg goes to fixed, 2+ goes to variadic.
    let src = "\
        (def f (fn ([x] :one) ([x y & rs] :many))) \
        (f 100)";
    assert_eq!(eval_str(src), ":one");
}

#[test]
fn defn_with_variadic_clause_takes_variadic() {
    let src = "\
        (def f (fn ([x] :one) ([x & rs] rs))) \
        (f 100 200 300)";
    assert_eq!(eval_str(src), "(200 300)");
}
