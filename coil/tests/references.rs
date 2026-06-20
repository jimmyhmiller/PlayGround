//! The reference tier: opt-in `mut` mutability + `let` stack places, erased to
//! the `ptr` core before codegen. `ptr` itself is unchanged (the metal tier).

mod common;
use common::build_and_run;

#[test]
fn struct_value_param_can_be_copied_via_store() {
    // Regression: a by-value struct param (passed as an immutable reference) must
    // be usable AS A VALUE — e.g. copied out via store! — just like a by-value sum
    // param. Previously store! of a struct param errored ("(ref S) vs S") while a
    // sum param worked: an inconsistency. (Reading a ref as its value is a read;
    // const-correctness — storing THROUGH an immutable ref — is unaffected.)
    let code = build_and_run(
        r#"(defstruct S [(a :i64) (b :i64)])
           (defn put [(p (ptr S)) (x S)] (-> :i64) (store! p x) 0)
           (defn main [] (-> :i64)
             (let [s (alloc-stack S) (mut src) (zeroed S)]
               (store! (field src a) 7) (store! (field src b) 5)
               (put s src)
               (iadd (load (field s a)) (load (field s b)))))"#,
    );
    assert_eq!(code, 12);
}

#[test]
fn storing_through_an_immutable_struct_ref_is_still_an_error() {
    // The fix must NOT loosen const-correctness: mutating a field THROUGH an
    // immutable (by-value) struct param remains rejected.
    let err = coil::check_source(
        "(defstruct S [(a :i64)])
         (defn bad [(r S)] (-> :i64) (store! (field r a) 5) 0)
         (defn main [] (-> :i64) 0)",
    )
    .unwrap_err();
    assert!(err.contains("immutable reference"), "got: {err}");
}

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

#[test]
fn rvalue_spill_struct_arg_to_byvalue_param() {
    // A struct RVALUE (a function-call result) passed to a by-value aggregate
    // param — which is an immutable reference — is spilled to a stack slot and
    // borrowed. Previously this errored ("expects a reference to S, got S").
    let src = r#"
        (defstruct Pt [(x i64) (y i64)])
        (defn mk [(x i64) (y i64)] (-> Pt)
          (let [(mut p) (zeroed Pt)]
            (store! (field p x) x) (store! (field p y) y) (load p)))
        (defn sum-pt [(p Pt)] (-> i64)
          (iadd (load (field p x)) (load (field p y))))
        (defn main [] (-> i64) (sum-pt (mk 17 25)))   ; 42
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn rvalue_spill_does_not_apply_to_mut_params() {
    // The spill is for IMMUTABLE references only: a temporary has no stable place
    // to mutate, so a `(mut)` param still rejects an rvalue argument.
    let src = r#"
        (defstruct Pt [(x i64) (y i64)])
        (defn mk [] (-> Pt) (zeroed Pt))
        (defn bump [(p (mut Pt))] (-> i64) (store! (field p x) 1) 0)
        (defn main [] (-> i64) (bump (mk)))
    "#;
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("mut") && err.contains("place"), "got: {err}");
}
