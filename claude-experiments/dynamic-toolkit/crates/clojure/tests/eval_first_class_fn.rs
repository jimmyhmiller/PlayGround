//! First-class fn values + higher-order calls (no captures yet).

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn fn_expression_returns_callable() {
    // The fn expression evaluates to an Fn heap object. Calling it
    // through an indirection returns the body's result.
    assert_eq!(eval_str("((fn [x] (* x x)) 7)"), "49");
}

#[test]
fn higher_order_via_let() {
    assert_eq!(eval_str("(let [f (fn [x] (* x 2))] (f 21))"), "42");
}

#[test]
fn fn_zero_args() {
    assert_eq!(eval_str("((fn [] 42))"), "42");
}

#[test]
fn nested_higher_order() {
    // f returns g, g is invoked. No closure capture yet — g is
    // self-contained.
    assert_eq!(
        eval_str("(let [f (fn [] (fn [x] (+ x 1)))] ((f) 41))"),
        "42"
    );
}

#[test]
fn fn_three_args() {
    assert_eq!(
        eval_str("(let [g (fn [a b c] (+ a (+ b c)))] (g 1 2 3))"),
        "6"
    );
}

#[test]
fn defined_fn_still_works() {
    // Static dispatch through the existing `def NAME (fn ...)` path.
    assert_eq!(eval_str("(def square (fn [x] (* x x))) (square 9)"), "81");
}

#[test]
fn closure_captures_outer_let() {
    assert_eq!(eval_str("(let [x 10] ((fn [y] (+ x y)) 32))"), "42");
}

#[test]
fn closure_captures_two_outer_lets() {
    assert_eq!(
        eval_str("(let [x 10 y 20] ((fn [z] (+ (+ x y) z)) 12))"),
        "42"
    );
}

#[test]
fn returned_closure_keeps_capture() {
    assert_eq!(eval_str("(let [n 7] ((fn [] (* n n))))"), "49");
}

#[test]
fn make_adder() {
    assert_eq!(
        eval_str("(def make-adder (fn [n] (fn [x] (+ x n)))) ((make-adder 10) 32)"),
        "42"
    );
}

#[test]
fn higher_order_arity_10() {
    // No fixed cap on indirect-call arity. This was previously
    // capped at 6 by the per-arity __invoke_N machinery.
    assert_eq!(
        eval_str(
            "(let [f (fn [a b c d e g h i j k] \
                       (+ (+ (+ (+ a b) (+ c d)) (+ (+ e g) (+ h i))) (+ j k)))] \
                (f 1 2 3 4 5 6 7 8 9 10))"
        ),
        "55"
    );
}

#[test]
fn higher_order_arity_15() {
    let mut src = String::from(
        "(let [f (fn [a b c d e g h i j k l m n o p] (+ (+ (+ (+ a b) (+ c d)) (+ (+ e g) (+ h i))) (+ (+ (+ j k) (+ l m)) (+ n (+ o p)))))] (f 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15))",
    );
    let _ = src.len(); // silence
    assert_eq!(eval_str(&src), "120");
}

// NOTE: a runtime arity-mismatch test would `panic::catch_unwind`
// the eval, but `extern "C"` panics across the JIT boundary cannot
// currently unwind cleanly (Rust 2024 turns them into SIGABRT). Once
// we move to `extern "C-unwind"` for the runtime panics, add the
// arity-exception test alongside the multi-arity work (#20).

#[test]
fn make_adder_curried_twice() {
    assert_eq!(
        eval_str(
            "(def make-adder (fn [n] (fn [x] (+ x n)))) \
             (let [add5 (make-adder 5) add10 (make-adder 10)] \
                (+ (add5 1) (add10 2)))"
        ),
        "18"
    );
}
