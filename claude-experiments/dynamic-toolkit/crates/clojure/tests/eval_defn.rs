//! Checkpoint 3: `def`, `fn`, `if`, `let`, `do`, recursion.

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let mut e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn def_simple_fn_and_call() {
    assert_eq!(eval_str("(def square (fn [x] (* x x))) (square 5)"), "25");
}

#[test]
fn fn_with_two_args() {
    assert_eq!(eval_str("(def add (fn [a b] (+ a b))) (add 3 4)"), "7");
}

#[test]
fn if_picks_then() {
    assert_eq!(eval_str("(if true 1 2)"), "1");
}

#[test]
fn if_picks_else() {
    assert_eq!(eval_str("(if false 1 2)"), "2");
}

#[test]
fn if_nil_is_falsey() {
    assert_eq!(eval_str("(if nil 1 2)"), "2");
}

#[test]
fn if_zero_is_truthy() {
    // Clojure semantics: only nil and false are falsey.
    assert_eq!(eval_str("(if 0 1 2)"), "1");
}

#[test]
fn let_binding() {
    assert_eq!(eval_str("(let [x 10] x)"), "10");
}

#[test]
fn let_multiple_bindings() {
    assert_eq!(eval_str("(let [x 10 y 20] (+ x y))"), "30");
}

#[test]
fn let_sequential() {
    // Each binding sees the previous.
    assert_eq!(eval_str("(let [x 10 y (+ x 1)] y)"), "11");
}

#[test]
fn do_returns_last() {
    assert_eq!(eval_str("(do 1 2 3)"), "3");
}

#[test]
fn fib_recursion() {
    let src = "(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))) \
               (fib 10)";
    assert_eq!(eval_str(src), "55");
}

#[test]
fn fib_15() {
    let src = "(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))) \
               (fib 15)";
    assert_eq!(eval_str(src), "610");
}

#[test]
fn fact_recursion() {
    let src = "(def fact (fn [n] (if (= n 0) 1 (* n (fact (- n 1)))))) \
               (fact 6)";
    assert_eq!(eval_str(src), "720");
}
