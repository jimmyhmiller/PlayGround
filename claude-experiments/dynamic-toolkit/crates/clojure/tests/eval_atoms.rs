//! Checkpoint 1: literal expressions evaluate end-to-end through the
//! reader, compiler, and JIT.

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let mut e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn integer_42() {
    assert_eq!(eval_str("42"), "42");
}

#[test]
fn integer_zero() {
    assert_eq!(eval_str("0"), "0");
}

#[test]
fn negative_integer() {
    assert_eq!(eval_str("-7"), "-7");
}

#[test]
fn float_pi() {
    // f64 round-trip; printer's int-fold won't apply.
    assert_eq!(eval_str("3.14"), "3.14");
}

#[test]
fn nil_literal() {
    assert_eq!(eval_str("nil"), "nil");
}

#[test]
fn true_literal() {
    assert_eq!(eval_str("true"), "true");
}

#[test]
fn false_literal() {
    assert_eq!(eval_str("false"), "false");
}

#[test]
fn last_form_wins() {
    // Two forms, only the last result is returned.
    assert_eq!(eval_str("1 2 3"), "3");
}
