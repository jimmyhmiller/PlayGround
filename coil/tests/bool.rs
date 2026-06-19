//! A real boolean: comparisons return `bool`, with `true`/`false` and the
//! short-circuiting logical operators `and`/`or`/`not`.

mod common;
use common::build_and_run;

#[test]
fn comparisons_return_bool() {
    let src = r#"
        (defn even [(n i64)] (-> bool) (icmp-eq (irem n 2) 0))
        (defn main [] (-> i64) (if (even 10) 42 0))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn logical_operators() {
    let src = r#"
        (defn main [] (-> i64)
          (let [a (icmp-lt 1 2)      ; true
                b (icmp-lt 5 3)]     ; false
            (if (and a (not b)) (if (or b a) 42 0) 0)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn bool_literals_and_storage() {
    let src = r#"
        (defstruct Flags [(on bool) (off bool)])
        (defn main [] (-> i64)
          (let [(mut f) (zeroed Flags)]
            (store! (field f on) true)
            (store! (field f off) false)
            (if (and (load (field f on)) (not (load (field f off)))) 42 0)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn short_circuit_avoids_evaluating_second() {
    // (or true X) must not evaluate X — here X would divide by zero.
    let src = r#"
        (defn main [] (-> i64)
          (if (or (icmp-eq 1 1) (icmp-eq (idiv 1 0) 0)) 42 0))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn bool_in_static_assert() {
    let src = r#"
        (static-assert (and (icmp-eq 1 1) (not (icmp-lt 2 1))) "logic")
        (defn main [] (-> i64) 42)
    "#;
    assert_eq!(build_and_run(src), 42);
}
