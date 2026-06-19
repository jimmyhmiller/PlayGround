//! The reference tier: opt-in `mut` mutability + `let` stack places, erased to
//! the `ptr` core before codegen. `ptr` itself is unchanged (the metal tier).

mod common;
use common::build_and_run;

#[test]
fn references_example_runs() {
    assert_eq!(build_and_run(include_str!("../examples/references.coil")), 42);
}

#[test]
fn mut_local_and_mutable_reference() {
    let src = r#"
        (defstruct Point [(x i64) (y i64)])
        (defn move-by [(p (mut Point)) (dx i64) (dy i64)] (-> i64)
          (store! (field p x) (iadd (load (field p x)) dx))
          (store! (field p y) (iadd (load (field p y)) dy))
          0)
        (defn main [] (-> i64)
          (let [(mut p) (zeroed Point)]
            (store! (field p x) 10)
            (store! (field p y) 20)
            (move-by (mut p) 5 7)
            (iadd (load (field p x)) (load (field p y)))))   ; 15 + 27 = 42
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn immutable_reference_can_be_read() {
    let src = r#"
        (defstruct Point [(x i64) (y i64)])
        (defn px [(p Point)] (-> i64) (load (field p x)))   ; bare = immutable ref
        (defn main [] (-> i64)
          (let [(mut p) (zeroed Point)]
            (store! (field p x) 42)
            (px p)))                                          ; pass immutably
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn zeroed_initializes_a_local() {
    let src = r#"
        (defstruct V3 [(a i64) (b i64) (c i64)])
        (defn main [] (-> i64)
          (let [(mut v) (zeroed V3)]
            (store! (field v b) 42)
            (iadd (load (field v a)) (iadd (load (field v b)) (load (field v c))))))
    "#;
    assert_eq!(build_and_run(src), 42); // 0 + 42 + 0
}

#[test]
fn rejects_store_through_immutable_reference() {
    let src = r#"
        (defstruct Point [(x i64) (y i64)])
        (defn bad [(p Point)] (-> i64) (store! (field p x) 1) 0)
        (defn main [] (-> i64) 0)
    "#;
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("immutable reference"), "got: {err}");
}

#[test]
fn rejects_bare_place_to_mut_parameter() {
    // A (mut T) parameter must be passed with (mut x) at the call site.
    let src = r#"
        (defstruct Point [(x i64) (y i64)])
        (defn set-x [(p (mut Point))] (-> i64) (store! (field p x) 1) 0)
        (defn main [] (-> i64)
          (let [(mut p) (zeroed Point)] (set-x p)))
    "#;
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("pass it mutably"), "got: {err}");
}

#[test]
fn rejects_mutable_borrow_of_immutable_local() {
    let src = r#"
        (defstruct Point [(x i64) (y i64)])
        (defn set-x [(p (mut Point))] (-> i64) (store! (field p x) 1) 0)
        (defn main [] (-> i64)
          (let [p (zeroed Point)] (set-x (mut p))))   ; p is not a mut local
    "#;
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("mutable borrow of immutable"), "got: {err}");
}

#[test]
fn ptr_tier_still_works_alongside() {
    // Raw pointers (the metal tier) are unchanged and interoperate.
    let src = r#"
        (defstruct Point [(x i64) (y i64)])
        (defn set-x [(p (mut Point)) (v i64)] (-> i64) (store! (field p x) v) 0)
        (defn main [] (-> i64)
          (let [q (alloc-stack Point)]          ; q : (ptr Point)
            (set-x q 42)                          ; a raw ptr satisfies a (mut) param
            (load (field q x))))
    "#;
    assert_eq!(build_and_run(src), 42);
}
