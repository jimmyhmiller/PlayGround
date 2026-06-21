//! lib/derive.coil — `derive-eq` / `derive-hash` / `derive-keyops` as a PURE
//! LIBRARY over the comptime type-reflection builtins (struct-fields/type-kind).
//! This is the prime directive made true for type-directed code: no compiler
//! `derive` keyword, no hand-written per-type boilerplate — just macros.

mod common;
use common::build_and_run;

const IMPORT: &str = "(module app)\n\
    (import \"lib/derive.coil\" :use *)\n\
    (import \"lib/hashmap.coil\" :use *)\n\
    (import \"lib/alloc.coil\" :use *)\n\
    (import \"lib/result.coil\" :use *)\n";

fn run_with(body: &str) -> i32 {
    build_and_run(&format!("{IMPORT}{body}"))
}

#[test]
fn derive_eq_compares_by_field() {
    let code = run_with(
        r#"(defstruct P [(x i64) (y i64)])
           (derive-eq P)
           (defn mk [(x i64) (y i64)] (-> P)
             (let [(mut p) (zeroed P)] (store! (field p x) x) (store! (field p y) y) (load p)))
           (defn main [] (-> i64)
             (iadd (if (P-eq (mk 3 4) (mk 3 4)) 40 0)      ; equal
                   (if (P-eq (mk 3 4) (mk 3 9)) 0 2)))"#,   // differ
    );
    assert_eq!(code, 42);
}

#[test]
fn derive_eq_and_hash_recurse_into_nested_structs() {
    let code = run_with(
        r#"(defstruct Inner [(v i64)])
           (derive-eq Inner) (derive-hash Inner)
           (defstruct Outer [(a i64) (inner Inner)])
           (derive-eq Outer) (derive-hash Outer)
           (defn main [] (-> i64)
             (let [(mut p) (zeroed Outer) (mut q) (zeroed Outer)]
               (store! (field p a) 1) (store! (field (field p inner) v) 2)
               (store! (field q a) 1) (store! (field (field q inner) v) 2)
               (iadd (if (Outer-eq p q) 40 0)                              ; nested eq
                     (if (icmp-eq (Outer-hash p) (Outer-hash q)) 2 0))))"#, // nested hash
    );
    assert_eq!(code, 42);
}

#[test]
fn derive_keyops_makes_a_struct_a_hashmap_key() {
    // The D10 payoff: a struct key via derived eq+hash, content-keyed (an update
    // by an equal-content key lands on the same entry).
    let code = run_with(
        r#"(defstruct P [(x i64) (y i64)])
           (derive-eq P) (derive-hash P) (derive-keyops P)
           (defn mk [(x i64) (y i64)] (-> P)
             (let [(mut p) (zeroed P)] (store! (field p x) x) (store! (field p y) y) (load p)))
           (defn main [] (-> i64)
             (let [a (malloc-allocator) (mut m) (hm-new [P i64] a (P-keyops))]
               (hm-put! (mut m) (mk 3 4) 10)
               (hm-put! (mut m) (mk 3 4) 100)   ; same key by content -> update
               (hm-put! (mut m) (mk 5 6) 20)
               (iadd (match (hm-get [P i64] m (mk 3 4)) (None [] -1) (Some [v] v))
                     (match (hm-get [P i64] m (mk 5 6)) (None [] -1) (Some [v] v)))))"#,
    );
    assert_eq!(code, 120); // 100 + 20
}

#[test]
fn derive_reflects_a_macro_generated_struct() {
    // Macro-generated types are reflectable (incremental, like the def table): a
    // macro emits a defstruct, and a *later* derive reflects it.
    let code = run_with(
        r#"(defmacro defpair [nm] `(defstruct ~nm [(a i64) (b i64)]))
           (defpair Pair)
           (derive-eq Pair)          ; reflects the macro-generated Pair
           (defn mk [(a i64) (b i64)] (-> Pair)
             (let [(mut p) (zeroed Pair)] (store! (field p a) a) (store! (field p b) b) (load p)))
           (defn main [] (-> i64) (if (Pair-eq (mk 1 2) (mk 1 2)) 42 0))"#,
    );
    assert_eq!(code, 42);
}

#[test]
fn derive_eq_hard_errors_on_a_sum() {
    let err = coil::check_source(&format!(
        "{IMPORT}(defsum S (A []) (B []))\n(derive-eq S)\n(defn main [] (-> i64) 0)"
    ))
    .unwrap_err();
    assert!(err.contains("not a struct") || err.contains("sum"), "got: {err}");
}

#[test]
fn derive_hash_hard_errors_on_a_float_field() {
    let err = coil::check_source(&format!(
        "{IMPORT}(defstruct P [(x f64)])\n(derive-hash P)\n(defn main [] (-> i64) 0)"
    ))
    .unwrap_err();
    assert!(err.contains("float fields unsupported"), "got: {err}");
}
