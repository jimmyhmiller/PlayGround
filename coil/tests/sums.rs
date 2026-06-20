//! Sum types (tagged unions) and `match`, including generic ones.

mod common;
use common::build_and_run;

#[test]
fn match_arm_runs_all_forms_not_just_the_first() {
    // Regression: the parser once kept ONLY the first form of a match arm and
    // silently dropped the rest. An arm with multiple forms must run them in
    // order and yield the LAST (here a side effect via the first two, value from
    // the third) — otherwise a multi-statement arm silently mis-behaves.
    let src = r#"
        (defsum Box (Full [(v :i64)]) (Empty))
        (defn take [(b Box)] (-> :i64)
          (let [(mut acc) 0]
            (match b
              (Full [v]
                (store! acc v)
                (store! acc (iadd (load acc) v))
                (load acc))
              (Empty [] 0))))
        (defn main [] (-> :i64) (take (Full 21)))   ; 21 + 21 = 42 (not 21)
    "#;
    assert_eq!(build_and_run(src), 42);
}

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

// --- checking-mode (bidirectional) constructor inference --------------------

#[test]
fn none_and_some_infer_from_return_type() {
    // No [i64] on the constructors: the expected return type pins Option's T.
    let src = r#"
        (defsum Option [T] (None) (Some [(val T)]))
        (defn pick [(b :i64)] (-> (Option i64))
          (if b (Some 42) (None)))
        (defn main [] (-> :i64)
          (match (pick 1) (None [] 0) (Some [v] v)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn result_error_type_inferred_from_context() {
    // Result's `E` appears in no field of `Ok`, so it can't be inferred from
    // arguments — only from the expected type. Checking mode supplies it.
    let src = r#"
        (defsum Result [T E] (Ok [(val T)]) (Err [(err E)]))
        (defsum Fail (Bad))
        (defn go [(n :i64)] (-> (Result i64 Fail))
          (if (icmp-lt n 0) (Err (Bad)) (Ok n)))
        (defn main [] (-> :i64)
          (match (go 42) (Ok [v] v) (Err [e] 0)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn constructor_infers_as_call_argument() {
    // A bare constructor passed to a function adopts the parameter's type.
    let src = r#"
        (defsum Option [T] (None) (Some [(val T)]))
        (defn unwrap [(o (Option i64))] (-> :i64)
          (match o (None [] 0) (Some [v] v)))
        (defn main [] (-> :i64)
          (iadd (unwrap (Some 42)) (unwrap (None))))   ; 42 + 0
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn nested_constructor_infers_through_field() {
    // (Some (None)) with expected (Option (Option i64)): the outer pins the
    // inner's type parameter through the field, in checking mode.
    let src = r#"
        (defsum Option [T] (None) (Some [(val T)]))
        (defn nested [] (-> (Option (Option i64)))
          (Some (None)))
        (defn main [] (-> :i64)
          (match (nested)
            (None [] 1)
            (Some [inner] (match inner (None [] 42) (Some [v] v)))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn bare_constructor_without_context_still_needs_type_args() {
    // With no expected type (a plain let binding), an un-inferable constructor
    // is still a compile error — checking mode doesn't paper over real ambiguity.
    let src = r#"
        (defsum Option [T] (None) (Some [(val T)]))
        (defn main [] (-> :i64) (let [x (None)] 0))
    "#;
    assert!(coil::check_source(src).unwrap_err().contains("cannot infer"));
}
