//! `HashMap[K V]` — open-addressing hash map as a generic-struct LIBRARY over the
//! allocator + lib/mem. Hashing/equality are EXPLICIT capability values (a KeyOps
//! vtable), so it's correct for every key type with no silent byte-wise footgun:
//! `hm-new-scalar` supplies safe byte-wise ops for scalar/pointer keys, while
//! struct/string keys supply their own KeyOps.

mod common;
use common::build_and_run;

const H: &str = concat!(
    "(module app)\n",
    "(import \"lib/hashmap.coil\" :use *)\n",
    "(import \"lib/alloc.coil\" :use *)\n",
    "(import \"lib/result.coil\" :use *)\n",
    "(import \"lib/control.coil\" :use *)\n",
);

fn run(body: &str) -> i32 {
    build_and_run(&format!("{H}{body}"))
}

#[test]
fn put_get_survives_growth() {
    // Insert 0..30 (several resizes), look one up after rehashing.
    let code = run(
        r#"(defn main [] (-> :i64)
             (let [a (malloc-allocator) (mut m) (hm-new-scalar [i64 i64] a)]
               (for [i 0 30] (hm-put! (mut m) i i))
               (let [r (match (hm-get [i64 i64] m 23) (None [] -1) (Some [v] v))]
                 (hm-free! (mut m))
                 r)))"#,
    );
    assert_eq!(code, 23);
}

#[test]
fn update_overwrites_value_and_keeps_len() {
    let code = run(
        r#"(defn main [] (-> :i64)
             (let [a (malloc-allocator) (mut m) (hm-new-scalar [i64 i64] a)]
               (hm-put! (mut m) 7 1)
               (hm-put! (mut m) 7 42)            ; update, not insert
               (iadd (match (hm-get [i64 i64] m 7) (None [] -1) (Some [v] v))
                     (hm-len m))))"#, // 42 + len 1 = 43
    );
    assert_eq!(code, 43);
}

#[test]
fn remove_tombstones_and_absent_is_none() {
    let code = run(
        r#"(defn main [] (-> :i64)
             (let [a (malloc-allocator) (mut m) (hm-new-scalar [i64 i64] a)]
               (hm-put! (mut m) 1 10) (hm-put! (mut m) 2 20) (hm-put! (mut m) 3 30)
               (hm-remove! (mut m) 2)
               (iadd (if (hm-contains? m 1) 1 0)            ; 1 present
                     (iadd (if (hm-contains? m 2) 100 0)    ; 0 removed
                           (iadd (if (hm-contains? m 99) 100 0)  ; 0 absent
                                 (hm-len m))))))"#, // 1 + 0 + 0 + len 2 = 3
    );
    assert_eq!(code, 3);
}

#[test]
fn get_absent_on_empty_map_is_none() {
    let code = run(
        r#"(defn main [] (-> :i64)
             (let [a (malloc-allocator) (mut m) (hm-new-scalar [i64 i64] a)]
               (match (hm-get [i64 i64] m 5) (None [] 77) (Some [_] 0))))"#,
    );
    assert_eq!(code, 77);
}

#[test]
fn reinsert_over_tombstoned_home_slot_does_not_duplicate() {
    // Regression: re-inserting a key whose HOME slot is tombstoned must update the
    // existing entry (found further down the probe chain), not fill the tombstone
    // and create a DUPLICATE. A constant hash forces all keys into one chain so the
    // collision is deterministic. Bug symptoms: len over-counts and a removed key
    // resurrects. Correct result encodes len==1, value==131, and absent-after-remove.
    let src = r#"(module app)
(import "lib/hashmap.coil" :use *)
(import "lib/alloc.coil" :use *)
(import "lib/result.coil" :use *)
(defn zero-hash [(p (ptr i8)) (n :i64)] (-> :i64) 0)
(defn ks-ops [] (-> (ptr KeyOps))
  (let [o (alloc-static KeyOps)]
    (store! (field o hash) (fnptr-of zero-hash))
    (store! (field o eq) (fnptr-of bytewise-eq))
    (store! (field o size) 8)
    o))
(defn main [] (-> :i64)
  (let [a (malloc-allocator) (mut m) (hm-new [i64 i64] a (ks-ops))]
    (hm-put! (mut m) 5 50)
    (hm-put! (mut m) 13 130)    ; collides with 5
    (hm-remove! (mut m) 5)      ; tombstone 5's home slot
    (hm-put! (mut m) 13 131)    ; update 13 -- must not duplicate
    (let [l1 (hm-len m)
          g  (match (hm-get [i64 i64] m 13) (None [] -1) (Some [v] v))]
      (hm-remove! (mut m) 13)
      (let [g2 (match (hm-get [i64 i64] m 13) (None [] 0) (Some [v] 999))]
        (iadd (imul l1 100) (iadd (isub g 131) g2))))))"#; // 1*100 + 0 + 0 = 100
    assert_eq!(build_and_run(src), 100);
}

#[test]
fn churn_does_not_grow_unbounded() {
    // Many delete+insert cycles at low occupancy must not grow the table forever
    // (tombstones are cleared by same-size rehash). Just must terminate + stay correct.
    let code = run(
        r#"(defn main [] (-> :i64)
             (let [a (malloc-allocator) (mut m) (hm-new-scalar [i64 i64] a)]
               (for [i 0 200]
                 (hm-put! (mut m) i i)
                 (hm-remove! (mut m) i))
               (hm-put! (mut m) 42 42)
               (iadd (hm-len m)                                ; 1
                     (match (hm-get [i64 i64] m 42) (None [] -1) (Some [v] v)))))"#, // + 42 = 43
    );
    assert_eq!(code, 43);
}

#[test]
fn arraylist_and_hashmap_compose() {
    // Regression: both libs once declared their own `abort` extern, which collided
    // when imported together (Coil doesn't dedup externs). OOM now routes through
    // alloc's shared `oom` defn, so the two collections are co-importable.
    let src = r#"(module app)
(import "lib/arraylist.coil" :use *)
(import "lib/hashmap.coil" :use *)
(import "lib/alloc.coil" :use *)
(import "lib/result.coil" :use *)
(defn main [] (-> :i64)
  (let [a (malloc-allocator)
        (mut xs) (al-new [i64] a)
        (mut m) (hm-new-scalar [i64 i64] a)]
    (al-push! (mut xs) 5)
    (hm-put! (mut m) 1 37)
    (iadd (al-get xs 0)
          (match (hm-get [i64 i64] m 1) (None [] 0) (Some [v] v)))))"#; // 5 + 37 = 42
    assert_eq!(build_and_run(src), 42);
}

#[test]
fn string_keys_via_explicit_content_keyops() {
    // The explicit-capability design handles string (content) keys today: supply a
    // KeyOps whose hash/eq deref the stored char* and hash/compare its bytes.
    let src = r#"(module app)
(import "lib/hashmap.coil" :use *)
(import "lib/alloc.coil" :use *)
(import "lib/result.coil" :use *)
(extern strlen :cc c [(ptr i8)] (-> :i64))
(extern strcmp :cc c [(ptr i8) (ptr i8)] (-> :i64))
(defn sh [(slot (ptr i8)) (n :i64)] (-> :i64)
  (let [s (load (cast (ptr (ptr i8)) slot))] (bytewise-hash s (strlen s))))
(defn se [(a (ptr i8)) (b (ptr i8)) (n :i64)] (-> bool)
  (icmp-eq (strcmp (load (cast (ptr (ptr i8)) a)) (load (cast (ptr (ptr i8)) b))) 0))
(defn str-ops [] (-> (ptr KeyOps))
  (let [o (alloc-static KeyOps)]
    (store! (field o hash) (fnptr-of sh))
    (store! (field o eq) (fnptr-of se))
    (store! (field o size) 8)
    o))
(defn main [] (-> :i64)
  (let [a (malloc-allocator) (mut m) (hm-new [(ptr i8) i64] a (str-ops))]
    (hm-put! (mut m) "alpha" 10)
    (hm-put! (mut m) "beta" 20)
    (hm-put! (mut m) "alpha" 100)   ; update by content
    (iadd (match (hm-get [(ptr i8) i64] m "alpha") (None [] -1) (Some [v] v))
          (match (hm-get [(ptr i8) i64] m "beta")  (None [] -1) (Some [v] v)))))"#;
    assert_eq!(build_and_run(src), 120); // 100 + 20
}
