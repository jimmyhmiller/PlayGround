//! Parameterized traits + generic impls, and the coil.core collection
//! vocabulary built on them: `(len xs)` / `(get xs k)` / `(set! (mut xs) k v)` /
//! `(push! (mut xs) v)` / `(pop! (mut xs))` / `(empty? xs)` / `(for-in [x (in xs)] …)`
//! across ArrayList, slices (strings), HashMap, and StrBuf.

mod common;
use common::build_and_run;

const LIST: &str = "(module app)\n\
                    (import \"lib/arraylist.coil\" :use *)\n\
                    (import \"lib/alloc.coil\" :use *)\n";

#[test]
fn user_parameterized_trait_with_generic_impl() {
    // A user trait with an extra Self-determined type param, implemented for a
    // user generic type — concrete dispatch instantiates the lowered generic fn.
    let src = "(module app)\n\
        (defstruct Box2 [T] [(a T) (b T)])\n\
        (deftrait First [Self E]\n\
          (first [(xs Self)] (-> E)))\n\
        (impl [T] First (Box2 T)\n\
          (first [(xs (Box2 T))] (-> T) (load (field xs a))))\n\
        (defn main [] (-> i64)\n\
          (let [p (alloc-stack (Box2 i64))]\n\
            (store! (field p a) 42)\n\
            (store! (field p b) 1)\n\
            (first (load p))))";
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn arraylist_full_vocabulary() {
    let src = format!(
        "{LIST}(defn main [] (-> i64)\n\
           (let [(mut xs) (al-new [i64] (malloc-allocator))]\n\
             (push! (mut xs) 5)\n\
             (push! (mut xs) 10)\n\
             (push! (mut xs) 20)\n\
             (set! (mut xs) 0 4)\n\
             (let [(mut total) (iadd (len xs) (get xs 2))]      ; 3 + 20\n\
               (match (pop! (mut xs))\n\
                 (Some [v] (store! total (iadd (load total) v)))  ; +20\n\
                 (None [] 0))\n\
               (if (empty? xs) 0 (load total)))))"               // 43
    );
    assert_eq!(build_and_run(&src), 43);
}

#[test]
fn strings_are_slices_with_len_and_get() {
    let src = "(module app)\n\
        (import \"lib/slice.coil\" :use *)\n\
        (defn main [] (-> i64)\n\
          ; len \"hello\" = 5; (get \"hello\" 1) = 'e' = 101; 5 + 101 - 64 = 42\n\
          (isub (iadd (len \"hello\") (cast :i64 (get \"hello\" 1))) 64))";
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn hashmap_get_returns_option_and_set_inserts() {
    let src = "(module app)\n\
        (import \"lib/hashmap.coil\" :use *)\n\
        (import \"lib/alloc.coil\" :use *)\n\
        (defn main [] (-> i64)\n\
          (let [(mut m) (hm-new-scalar [i64 i64] (malloc-allocator))]\n\
            (set! (mut m) 7 40)\n\
            (set! (mut m) 8 2)\n\
            (match (get m 7)\n\
              (Some [v] (iadd v (iadd (len m) \n\
                (match (get m 99) (Some [_] 100) (None [] 0)))))\n\
              (None [] -1))))"; // 40 + 2 + 0 = 42
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn strbuf_len_get_push() {
    let src = "(module app)\n\
        (import \"lib/str.coil\" :use *)\n\
        (import \"lib/alloc.coil\" :use *)\n\
        (defn main [] (-> i64)\n\
          (let [(mut sb) (sb-new (malloc-allocator))]\n\
            (push! (mut sb) (cast :u8 40))\n\
            (push! (mut sb) (cast :u8 100))\n\
            ; len 2 + first byte 40 = 42\n\
            (iadd (len sb) (cast :i64 (get sb 0)))))";
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn for_in_over_any_len_get_collection() {
    let src = format!(
        "{LIST}(defn main [] (-> i64)\n\
           (let [(mut xs) (al-new [i64] (malloc-allocator))]\n\
             (push! (mut xs) 10) (push! (mut xs) 12)\n\
             (let [(mut total) 0]\n\
               (for-in [v (in xs)] (store! total (iadd (load total) v)))     ; 22\n\
               (for-in [c (in \"abc\")]                                       ; not counted\n\
                 (if (icmp-eq (cast :i64 c) 98) (store! total (iadd (load total) 20)) 0)) ; +20 for 'b'\n\
               (load total))))"
    );
    assert_eq!(build_and_run(&src), 42);
}

#[test]
fn bounded_generic_over_a_generic_impl() {
    // The deferred TraitCall path: `Len` (single-param) bound on T, resolved at
    // mono against the GENERIC (ArrayList T) impl — the mangled receiver's base
    // and args are recovered and the lowered generic method instantiated.
    let src = format!(
        "{LIST}(defn total-len [(C Len)] [(a C) (b C)] (-> i64)\n\
           (iadd (len a) (len b)))\n\
         (defn main [] (-> i64)\n\
           (let [(mut xs) (al-new [i64] (malloc-allocator))]\n\
             (push! (mut xs) 1) (push! (mut xs) 2)\n\
             (iadd (total-len xs xs) 38)))" // 2+2+38
    );
    assert_eq!(build_and_run(&src), 42);
}

#[test]
fn bits_accessors_renamed_bit_get_bit_set() {
    // The old `get`/`set!` spellings now belong to the collection traits; the
    // :bits struct accessors are `bit-get` / `bit-set!`.
    let src = "(module app)\n\
        (defstruct Flags :layout bits :backing :i16\n\
          [(lo :bits 4) (mid :bits 8) (hi :bits 4)])\n\
        (defn main [] (-> i64)\n\
          (let [p (alloc-stack Flags)]\n\
            (bit-set! p lo 5)\n\
            (bit-set! p mid 33)\n\
            (iadd (cast :i64 (bit-get p lo)) (iadd (cast :i64 (bit-get p mid)) 4))))";
    assert_eq!(build_and_run(src), 42);
}

// ---- the definition-time contracts ----------------------------------------

#[test]
fn bounds_over_parameterized_traits_are_rejected_clearly() {
    let src = "(module app)\n\
        (import \"lib/arraylist.coil\" :use *)\n\
        (defn first-of [(C Get)] [(xs C)] (-> i64) (get xs 0))\n\
        (defn main [] (-> i64) 0)";
    let err = coil::check_source(src).unwrap_err();
    assert!(
        err.contains("takes type parameters") && err.contains("aren't supported yet"),
        "got: {err}"
    );
}

#[test]
fn impl_type_param_must_appear_in_the_implementing_type() {
    let src = "(module app)\n\
        (deftrait T1 [Self] (m1 [(x Self)] (-> i64)))\n\
        (defstruct S [(v i64)])\n\
        (impl [Q] T1 S (m1 [(x S)] (-> i64) 0))\n\
        (defn main [] (-> i64) 0)";
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("must appear in the implementing type"), "got: {err}");
}

#[test]
fn inconsistent_extra_param_binding_is_a_conformance_error() {
    // E must resolve to ONE type across all of an impl's methods.
    let src = "(module app)\n\
        (deftrait Both [Self E]\n\
          (m-in  [(x Self) (v E)] (-> i64))\n\
          (m-out [(x Self)] (-> E)))\n\
        (defstruct S [(v i64)])\n\
        (impl Both S\n\
          (m-in  [(x S) (v i64)] (-> i64) 0)\n\
          (m-out [(x S)] (-> bool) true))\n\
        (defn main [] (-> i64) 0)";
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("doesn't match the trait"), "got: {err}");
}

#[test]
fn one_impl_per_trait_and_base_still_holds() {
    let src = "(module app)\n\
        (deftrait T2 [Self] (m2 [(x Self)] (-> i64)))\n\
        (impl [T] T2 (slice T) (m2 [(x (slice T))] (-> i64) 1))\n\
        (impl T2 i64 (m2 [(x i64)] (-> i64) 2))\n\
        (impl [Q] T2 (slice Q) (m2 [(x (slice Q))] (-> i64) 3))\n\
        (defn main [] (-> i64) 0)";
    let err = coil::check_source(src).unwrap_err();
    assert!(err.contains("duplicate impl"), "got: {err}");
}
