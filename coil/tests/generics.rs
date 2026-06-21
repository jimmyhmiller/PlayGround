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
fn infers_type_args_from_value_args() {
    // No explicit [i64]: T is inferred from the argument's type. The literal 5
    // is i64 by default, so id is instantiated at i64.
    let src = r#"
        (defn id [T] [(x T)] (-> T) x)
        (defn main [] (-> :i64) (id 42))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn infers_type_arg_from_struct_pointer() {
    // T is recovered by unifying (ptr (Pair T T)) against the actual arg type.
    let src = r#"
        (defstruct Pair [A B] [(fst A) (snd B)])
        (defn pair-sum [T] [(p (ptr (Pair T T)))] (-> T)
          (iadd (load (field p fst)) (load (field p snd))))
        (defn main [] (-> :i64)
          (let [p (alloc-stack (Pair i64 i64))]
            (store! (field p fst) 40) (store! (field p snd) 2)
            (pair-sum p)))            ; no [i64] — inferred from p
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn infers_variant_type_args() {
    // (Some 42) infers Option's T = i64 from the field argument; no [i64].
    let src = r#"
        (defsum Option [T] (None) (Some [(val T)]))
        (defn unwrap-or [T] [(o (Option T)) (default T)] (-> T)
          (match o (None [] default) (Some [v] v)))
        (defn main [] (-> :i64) (unwrap-or (Some 42) 0))
    "#;
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn rejects_uninferable_type_arg() {
    // T appears in no parameter, so it cannot be inferred and must be explicit.
    let src = r#"
        (defn phantom [T] [(x :i64)] (-> :i64) x)
        (defn main [] (-> :i64) (phantom 5))
    "#;
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("cannot infer type argument 'T'"), "got: {err}");
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

#[test]
fn infers_through_nested_generic_call_arg() {
    // E12: a generic-call argument that can't infer its own T in isolation is
    // deferred and re-inferred once a sibling argument fixes T. Here `pick`'s T
    // is i64 (from `42`), which must flow into the `(empty-slice)` argument.
    let src = r#"
        (import "lib/slice.coil" :use *)
        (defn empty-slice [T] [] (-> (slice T)) (slice-new (cast (ptr T) 0) 0))
        (defn pick [T] [(s (slice T)) (m T)] (-> T) m)
        (defn main [] (-> :i64) (pick (empty-slice) 42))
    "#;
    assert_eq!(build_and_run(&format!("(module app)\n{src}")), 42);
}

#[test]
fn nested_generic_call_still_rejects_genuinely_ambiguous() {
    // The deferral must NOT over-infer: when nothing constrains the element type,
    // it remains a hard error (no silent wrong default).
    let src = r#"(module app)
        (import "lib/slice.coil" :use *)
        (defn empty-slice [T] [] (-> (slice T)) (slice-new (cast (ptr T) 0) 0))
        (defn main [] (-> :i64) (slice-len (subslice (empty-slice) 0 0)))"#;
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("cannot infer type argument"), "got: {err}");
}

#[test]
fn generic_passes_a_struct_type_arg_by_value_correctly() {
    // Regression: a generic function instantiated with a struct type-arg passes
    // the struct BY VALUE (param_ref_type leaves a type param unwrapped, so unlike
    // a non-generic struct param it isn't by-ref). Codegen bound that by-value
    // struct param as a pointer-typed-as-struct, silently corrupting every use;
    // it must reconstruct + bind the actual struct value.
    let code = build_and_run(
        r#"(defstruct Point [(x i64) (y i64)])
           (defn store-read [K] [(slot (ptr K)) (k K)] (-> i64) (store! slot k) 0)
           (defn main [] (-> i64)
             (let [(mut p) (zeroed Point) dst (alloc-stack Point)]
               (store! (field p x) 17) (store! (field p y) 25)
               (store-read [Point] dst (load p))
               (iadd (load (field dst x)) (load (field dst y)))))"#,
    );
    assert_eq!(code, 42);
}
