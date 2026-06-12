//! Numeric semantics and their fault edges: overflow faults (not wrapping),
//! division/modulo by zero (including floats), and int/float promotion.

use funct::{Funct, Value};

fn eval(src: &str) -> Value {
    let mut vm = Funct::new();
    vm.eval(src)
        .unwrap_or_else(|e| panic!("eval failed: {}\nsource: {}", e, src))
}

fn eval_err(src: &str) -> String {
    let mut vm = Funct::new();
    vm.eval(src).expect_err("expected a fault").to_string()
}

#[test]
fn multiplication_overflow_faults() {
    assert!(eval_err("9223372036854775807 * 2").contains("overflow"));
}

#[test]
fn power_overflow_faults() {
    assert!(eval_err("2 ** 100").contains("overflow"));
}

#[test]
fn modulo_by_zero_faults() {
    assert!(eval_err("10 % 0").contains("by zero"));
}

#[test]
fn float_division_by_zero_faults() {
    // funct chooses to fault rather than produce inf/NaN.
    assert!(eval_err("1.0 / 0.0").contains("by zero"));
    assert!(eval_err("0.0 / 0.0").contains("by zero"));
}

#[test]
fn int_ops_stay_int() {
    assert_eq!(eval("7 / 2"), Value::Int(3));
    assert_eq!(eval("7 % 2"), Value::Int(1));
    assert_eq!(eval("2 ** 10"), Value::Int(1024));
}

#[test]
fn int_float_mix_promotes_to_float() {
    assert_eq!(eval("1 + 2.0"), Value::Float(3.0));
    assert_eq!(eval("3.0 * 2"), Value::Float(6.0));
}

#[test]
fn float_division_is_exact_for_powers_of_two() {
    assert_eq!(eval("1.0 / 4.0"), Value::Float(0.25));
}

#[test]
fn i64_min_literal_is_a_clean_error_not_a_panic() {
    // `-9223...808` lexes the magnitude first (which overflows i64) before the
    // unary minus, so the literal can't be written directly. The contract that
    // matters: it's a reported error, never a panic.
    let mut vm = Funct::new();
    let err = vm
        .eval("-9223372036854775808")
        .expect_err("should not parse");
    assert!(
        err.to_string().contains("too large") || err.to_string().contains("bad int"),
        "{}",
        err
    );
}
