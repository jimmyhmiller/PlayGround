//! Small dogfood-driven library macros: `box` (heap-init), `hm-for` (HashMap
//! iteration), and variadic `all`/`any` — all macros/library over the core
//! (the prime-directive-correct way), surfaced by the JSON dogfood.

mod common;
use common::build_and_run;

#[test]
fn box_heap_inits_a_value() {
    let code = build_and_run(
        r#"(module app)
           (import "lib/alloc.coil" :use *)
           (import "lib/result.coil" :use *)
           (defstruct B [(v i64)])
           (defn main [] (-> i64)
             (let [a (malloc-allocator)
                   p (box a B (let [(mut b) (zeroed B)] (store! (field b v) 42) (load b)))]
               (load (field p v))))"#,
    );
    assert_eq!(code, 42);
}

#[test]
fn hm_for_iterates_occupied_entries() {
    // Sums values via hm-for; an update keeps one entry (not iterated twice).
    let code = build_and_run(
        r#"(module app)
           (import "lib/alloc.coil" :use *)
           (import "lib/result.coil" :use *)
           (import "lib/control.coil" :use *)
           (import "lib/hashmap.coil" :use *)
           (defn main [] (-> i64)
             (let [a (malloc-allocator) (mut m) (hm-new-scalar [i64 i64] a) (mut s) 0]
               (hm-put! (mut m) 1 10) (hm-put! (mut m) 2 14) (hm-put! (mut m) 3 18)
               (hm-put! (mut m) 2 12)   ; update key2 -> 12 (still one entry, not two)
               (hm-for [k v (mut m)] (store! s (iadd (load s) v)))
               (load s)))"#, // 10 + 12 + 18 = 40
    );
    assert_eq!(code, 40);
}

#[test]
fn variadic_all_and_any() {
    let code = build_and_run(
        r#"(module app)
           (import "lib/control.coil" :use *)
           (defn main [] (-> i64)
             (iadd (if (all (icmp-gt 5 1) (icmp-gt 9 2) (icmp-lt 3 9)) 40 0)   ; all true
                   (if (any (icmp-eq 1 2) (icmp-eq 2 2) (icmp-eq 3 4)) 2 0)))"#, // any true
    );
    assert_eq!(code, 42);
}

#[test]
fn all_any_short_circuit_and_edge_counts() {
    let code = build_and_run(
        r#"(module app)
           (import "lib/control.coil" :use *)
           (defn main [] (-> i64)
             (iadd (if (all) 40 0)                               ; (all) = true
                   (if (any (icmp-eq 1 1)) 2 0)))"#,             // single arg returned
    );
    assert_eq!(code, 42);
}

#[test]
fn case_defaults_to_eq_trait_on_ints() {
    // `case` compares with the prelude `Eq` trait (Eq i64 is provided), so int
    // keys work with no equality argument. Flat `key body` pairs, lone default.
    let code = build_and_run(
        r#"(module app)
           (import "lib/control.coil" :use *)
           (defn classify [(c i64)] (-> i64)
             (case c 1 100 2 200 3 300 999))
           (defn main [] (-> i64) (iadd (classify 2) (classify 7)))"#, // 200 + 999 = 1199; %256 = 175
    );
    assert_eq!(code, 175);
}

#[test]
fn case_by_with_str_eq() {
    // For keys whose type isn't Eq (slices/strings), `case-by` takes an explicit
    // 2-arg equality — here str-eq.
    let code = build_and_run(
        r#"(module app)
           (import "lib/slice.coil" :use *)
           (import "lib/str.coil" :use *)
           (import "lib/control.coil" :use *)
           (defn op [(s (slice u8))] (-> i64)
             (case-by s str-eq "add" 1 "sub" 2 "mul" 3 0))
           (defn main [] (-> i64) (iadd (imul (op "mul") 10) (op "nope")))"#, // 3*10 + 0 = 30
    );
    assert_eq!(code, 30);
}
