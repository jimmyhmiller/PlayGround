//! Arithmetic + ordering operators as prelude traits: `+ - * / %` (Add/Sub/Mul/
//! Div/Rem) and `< <= > >=` (Ord), dispatched by type, usable in generics bounded
//! `(T Add)` / `(T Ord)`, and at comptime. Lower to the core ops in the impl
//! (which O3 inlines). `i64` and `f64` impls ship in the prelude.

mod common;
use common::build_and_run;

#[test]
fn integer_operators() {
    // fact + fib written with + - * <= <, no imul/icmp in sight.
    let code = build_and_run(
        "(module app)\n\
         (defn fact [(n i64)] (-> i64) (if (<= n 0) 1 (* n (fact (- n 1)))))\n\
         (defn fib  [(n i64)] (-> i64) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))\n\
         (defn main [] (-> i64) (+ (fact 5) (fib 10)))", // 120 + 55 = 175
    );
    assert_eq!(code, 175);
}

#[test]
fn ordering_operators() {
    let code = build_and_run(
        "(module app)\n\
         (defn main [] (-> i64)\n\
           (+ (if (< 1 2) 1 0) (+ (if (>= 5 5) 10 0)\n\
              (+ (if (> 2 9) 1000 0) (if (<= 7 7) 100 0)))))", // 1 + 10 + 0 + 100 = 111
    );
    assert_eq!(code, 111);
}

#[test]
fn generic_bounded_by_add() {
    let code = build_and_run(
        "(module app)\n\
         (defn sum3 [(T Add)] [(a T) (b T) (c T)] (-> T) (+ (+ a b) c))\n\
         (defn main [] (-> i64) (sum3 10 20 12))", // 42
    );
    assert_eq!(code, 42);
}

#[test]
fn operators_at_comptime() {
    let code = build_and_run(
        "(module app)\n\
         (defn fact [(n i64)] (-> i64) (if (<= n 0) 1 (* n (fact (- n 1)))))\n\
         (defn main [] (-> i64) (comptime (fact 5)))", // 120
    );
    assert_eq!(code, 120);
}

#[test]
fn rem_and_div() {
    let code = build_and_run(
        "(module app)\n(defn main [] (-> i64) (+ (* (/ 100 7) 7) (% 100 7)))", // (14*7)+2 = 100
    );
    assert_eq!(code, 100);
}
