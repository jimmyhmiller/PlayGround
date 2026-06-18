//! Sum types (tagged unions) and `match`, including generic ones.

mod common;
use common::build_and_run;

#[test]
fn concrete_sum_and_match() {
    let src = r#"
        (defsum Shape
          (Circle [(r :i64)])
          (Rect   [(w :i64) (h :i64)]))
        (defn area [(s Shape)] (-> :i64)
          (match s
            (Circle [r] (imul 3 (imul r r)))
            (Rect [w h] (imul w h))))
        (defn main [] (-> :i64)
          (iadd (area (Rect 5 6)) (area (Circle 2))))   ; 30 + 12 = 42
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn generic_option_unwrap_or() {
    let src = r#"
        (defsum Option [T] (None) (Some [(val T)]))
        (defn unwrap-or [T] [(o (Option T)) (default T)] (-> T)
          (match o
            (None [] default)
            (Some [v] v)))
        (defn main [] (-> :i64)
          (iadd (unwrap-or [i64] (Some [i64] 40) 0)
                (unwrap-or [i64] (None [i64]) 2)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn generic_result() {
    let src = r#"
        (defsum Result [T E] (Ok [(val T)]) (Err [(err E)]))
        (defn or-zero [(r (Result i64 i64))] (-> :i64)
          (match r (Ok [v] v) (Err [e] 0)))
        (defn main [] (-> :i64)
          (iadd (or-zero (Ok [i64 i64] 42)) (or-zero (Err [i64 i64] 99))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn match_returns_a_pointer() {
    // arms can produce any (matching) type, not just integers.
    let src = r#"
        (defsum Opt [T] (Nil) (Has [(p T)]))
        (defn main [] (-> :i64)
          (let [slot (alloc-stack i64)]
            (store! slot 42)
            (let [o (Has [(ptr i64)] slot)]
              (load (match o
                      (Nil [] slot)
                      (Has [q] q))))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn rejects_non_exhaustive_match() {
    let src = r#"
        (defsum Opt (A) (B) (C))
        (defn main [] (-> :i64)
          (match (A) (A [] 1) (B [] 2)))
    "#;
    assert!(coil::check_source(src).unwrap_err().contains("non-exhaustive"));
}

#[test]
fn rejects_wrong_bind_count() {
    let src = r#"
        (defsum P (Pt [(x :i64) (y :i64)]))
        (defn main [] (-> :i64)
          (match (Pt 1 2) (Pt [x] x)))
    "#;
    assert!(coil::check_source(src).unwrap_err().contains("binds"));
}
