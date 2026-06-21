//! Contract tests for the comptime type-reflection builtins as GENERAL primitives
//! (independent of derive / match-else, which consume them): struct-fields,
//! sum-variants, type-kind, type-params, variant-sum, to-vector. Each is exercised
//! by a tiny macro that emits a checkable integer.

mod common;
use common::build_and_run;

const TYPES: &str = "(module app)\n\
    (defstruct P [(x i64) (y i64) (z i64)])\n\
    (defstruct Pair [T] [(a T) (b T)])\n\
    (defsum Shape (Circle [(r i64)]) (Rect [(w i64) (h i64)]) (Dot []))\n";

#[test]
fn struct_fields_and_sum_variants_count() {
    let code = build_and_run(&format!(
        "{TYPES}\
         (defmacro nf [t] (count (struct-fields t)))\n\
         (defmacro nv [t] (count (sum-variants t)))\n\
         (defn main [] (-> i64) (iadd (imul (nf P) 10) (nv Shape)))" // 3*10 + 3 = 33
    ));
    assert_eq!(code, 33);
}

#[test]
fn type_kind_distinguishes_struct_sum_scalar_unknown() {
    let code = build_and_run(&format!(
        "{TYPES}\
         (defmacro k [t] (if (= (type-kind t) :struct) 1 \
                          (if (= (type-kind t) :sum) 2 \
                           (if (= (type-kind t) :scalar) 3 4))))\n\
         (defn main [] (-> i64) \
           (iadd (k P) (iadd (imul (k Shape) 3) (iadd (imul (k i64) 9) (imul (k Nope) 27)))))"
        // struct=1, sum=2, scalar=3, unknown=4, weighted 1/3/9/27 -> 1+6+27+108 = 142
    ));
    assert_eq!(code, 142);
}

#[test]
fn type_params_counts_generic_parameters() {
    let code = build_and_run(&format!(
        "{TYPES}\
         (defmacro np [t] (count (type-params t)))\n\
         (defn main [] (-> i64) (iadd (imul (np Pair) 10) (np P)))" // generic=1*10, mono=0 -> 10
    ));
    assert_eq!(code, 10);
}

#[test]
fn variant_sum_resolves_a_variant_to_its_sum() {
    // variant-sum(Rect) -> Shape, then sum-variants(Shape) has 3 variants.
    let code = build_and_run(&format!(
        "{TYPES}\
         (defmacro vsn [v] (count (sum-variants (variant-sum v))))\n\
         (defn main [] (-> i64) (iadd (vsn Rect) (vsn Circle)))" // 3 + 3 = 6
    ));
    assert_eq!(code, 6);
}

#[test]
fn variant_sum_hard_errors_on_unknown_variant() {
    let err = coil::check_source(&format!(
        "{TYPES}(defmacro vsbad [v] (count (sum-variants (variant-sum v))))\n\
         (defn main [] (-> i64) (vsbad Nope))"
    ))
    .unwrap_err();
    assert!(err.contains("no sum has a variant"), "got: {err}");
}

#[test]
fn to_vector_makes_a_seq_of_the_listed_length() {
    let code = build_and_run(&format!(
        "{TYPES}\
         (defmacro tvlen [] (count (to-vector (list 1 2 3 4 5))))\n\
         (defn main [] (-> i64) (tvlen))" // 5
    ));
    assert_eq!(code, 5);
}
