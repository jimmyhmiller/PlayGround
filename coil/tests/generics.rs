//! Generics via monomorphization (explicit type arguments).

mod common;
use common::build_and_run;

#[test]
fn generic_identity_function() {
    let src = r#"
        (defn id [T] [(x T)] (-> T) x)
        (defn main [] (-> :i64) (id [i64] 42))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn generic_struct() {
    let src = r#"
        (defstruct Pair [A B] [(fst A) (snd B)])
        (defn main [] (-> :i64)
          (let [p (alloc-stack (Pair i64 i64))]
            (store! (field p fst) 40)
            (store! (field p snd) 2)
            (iadd (load (field p fst)) (load (field p snd)))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn distinct_type_parameters() {
    // Pair<i32, i64> — the two parameters get different concrete types/widths.
    let src = r#"
        (defstruct Pair [A B] [(fst A) (snd B)])
        (defn main [] (-> :i64)
          (let [p (alloc-stack (Pair i32 i64))]
            (store! (field p fst) (cast :i32 40))
            (store! (field p snd) 2)
            (iadd (cast :i64 (load (field p fst))) (load (field p snd)))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn generic_function_over_generic_struct() {
    let src = r#"
        (defstruct Pair [A B] [(fst A) (snd B)])
        (defn pair-sum [T] [(p (ptr (Pair T T)))] (-> T)
          (iadd (load (field p fst)) (load (field p snd))))
        (defn main [] (-> :i64)
          (let [p (alloc-stack (Pair i64 i64))]
            (store! (field p fst) 40) (store! (field p snd) 2)
            (pair-sum [i64] p)))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn two_instantiations_of_one_generic() {
    // id is used at i64 and i32; each gets its own monomorphic copy, deduped.
    let src = r#"
        (defn id [T] [(x T)] (-> T) x)
        (defn main [] (-> :i64)
          (iadd (id [i64] 40) (cast :i64 (id [i32] (cast :i32 2)))))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn rejects_generic_call_without_type_args() {
    let src = r#"
        (defn id [T] [(x T)] (-> T) x)
        (defn main [] (-> :i64) (id 5))
    "#;
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("explicit type arguments"), "got: {err}");
}

#[test]
fn rejects_wrong_type_arg_count() {
    let src = r#"
        (defstruct Pair [A B] [(fst A) (snd B)])
        (defn main [] (-> :i64)
          (let [p (alloc-stack (Pair i64))] 0))
    "#;
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("expects 2 type arguments"), "got: {err}");
}
