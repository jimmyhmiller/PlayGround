//! Floating point: f32/f64, arithmetic, comparison, and int<->float conversions.

mod common;
use common::build_and_run;

#[test]
fn float_arithmetic_and_truncation() {
    assert_eq!(build_and_run("(defn main [] (-> i64) (cast i64 (fmul 3.0 14.0)))"), 42);
    assert_eq!(build_and_run("(defn main [] (-> i64) (cast i64 (fdiv 85.5 2.0)))"), 42); // 42.75 -> 42
}

#[test]
fn float_comparison() {
    assert_eq!(build_and_run("(defn main [] (-> i64) (if (fcmp-lt 1.5 2.5) 42 0))"), 42);
    assert_eq!(build_and_run("(defn main [] (-> i64) (if (fcmp-ge 1.0 2.0) 0 42))"), 42);
}

#[test]
fn int_and_float_conversions() {
    // int -> float -> int round trip, and float->int truncation.
    assert_eq!(build_and_run("(defn main [] (-> i64) (cast i64 (fadd (cast f64 40) 2.5)))"), 42);
}

#[test]
fn f32_and_f64_widths() {
    // f32 truncates precision; fpext/fptrunc between widths.
    assert_eq!(build_and_run("(defn main [] (-> i64) (cast i64 (cast f32 42.9)))"), 42);
    assert_eq!(build_and_run("(defn main [] (-> i64) (cast i64 (cast f64 (cast f32 42.0))))"), 42);
}

#[test]
fn float_literal_adopts_f32_from_context() {
    // 1.0 adopts f32 from the parameter type (no explicit (cast f32 1.0)).
    let src = r#"
        (defn add1 [(x f32)] (-> f32) (fadd x 1.0))
        (defn main [] (-> i64) (cast i64 (add1 (cast f32 41.0))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn float_struct_fields() {
    let src = r#"
        (defstruct V2 [(x f64) (y f64)])
        (defn main [] (-> i64)
          (let [(mut v) (zeroed V2)]
            (store! (field v x) 40.0)
            (store! (field v y) 2.0)
            (cast i64 (fadd (load (field v x)) (load (field v y))))))
    "#;
    assert_eq!(build_and_run(src), 42);
}
