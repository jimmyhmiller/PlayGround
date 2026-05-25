//! Checkpoint 2: list forms call primitive externs.

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let mut e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn add_two_ints() {
    assert_eq!(eval_str("(+ 1 2)"), "3");
}

#[test]
fn nested_call() {
    assert_eq!(eval_str("(+ (* 2 3) 4)"), "10");
}

#[test]
fn subtract() {
    assert_eq!(eval_str("(- 10 3)"), "7");
}

#[test]
fn divide() {
    assert_eq!(eval_str("(/ 10 4)"), "2.5");
}

#[test]
fn lt_returns_bool() {
    assert_eq!(eval_str("(< 1 2)"), "true");
    assert_eq!(eval_str("(< 2 1)"), "false");
}

#[test]
fn eq_on_numbers() {
    assert_eq!(eval_str("(= 3 3)"), "true");
    assert_eq!(eval_str("(= 3 4)"), "false");
}

#[test]
fn nil_predicate() {
    assert_eq!(eval_str("(nil? nil)"), "true");
    assert_eq!(eval_str("(nil? 0)"), "false");
}

#[test]
fn not_truthiness() {
    // Only nil and false are falsey.
    assert_eq!(eval_str("(not nil)"), "true");
    assert_eq!(eval_str("(not false)"), "true");
    assert_eq!(eval_str("(not 0)"), "false");
    assert_eq!(eval_str("(not true)"), "false");
}

#[test]
fn deep_nesting() {
    // (((1 + 2) * 3) + ((10 - 4) / 2)) = 9 + 3 = 12
    assert_eq!(eval_str("(+ (* (+ 1 2) 3) (/ (- 10 4) 2))"), "12");
}
