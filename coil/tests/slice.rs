//! `(Slice T)` — a fat-pointer `{data, len}` view over contiguous elements, as an
//! ordinary generic-struct LIBRARY (lib/slice.coil), not a compiler type. Covers
//! construction, indexing, length, subslicing (shared storage), write-through
//! aliasing, and `slice-for` iteration (a macro over `for`/loop).

mod common;
use common::build_and_run;

const IMPORT: &str = "(module app)\n(import \"lib/slice.coil\" :use *)\n";

fn run_with(body: &str) -> i32 {
    build_and_run(&format!("{IMPORT}{body}"))
}

const FILL5: &str = r#"(let [arr (alloc-stack (array i64 5))]
  (store! (index arr 0) 1) (store! (index arr 1) 2) (store! (index arr 2) 3)
  (store! (index arr 3) 4) (store! (index arr 4) 5)"#;

#[test]
fn slice_for_sums_elements() {
    let code = run_with(&format!(
        r#"(defn main [] (-> :i64)
             {FILL5}
               (let [s (slice-of arr 5) (mut acc) 0]
                 (slice-for [x s] (store! acc (iadd (load acc) x)))
                 (load acc))))"#
    ));
    assert_eq!(code, 15); // 1+2+3+4+5
}

#[test]
fn slice_get_and_len() {
    let code = run_with(&format!(
        r#"(defn main [] (-> :i64)
             {FILL5}
               (let [s (slice-of arr 5)]
                 (iadd (imul (slice-get s 2) 10) (slice-len s)))))"#
    ));
    assert_eq!(code, 35); // 3*10 + 5
}

#[test]
fn subslice_is_a_subrange() {
    // Bind the base slice first: generic inference doesn't yet flow T through a
    // nested generic call like `(subslice (slice-of …) …)` — bind-first is the idiom.
    let code = run_with(&format!(
        r#"(defn main [] (-> :i64)
             {FILL5}
               (let [s (slice-of arr 5) mid (subslice s 1 4) (mut acc) 0]
                 (slice-for [x mid] (store! acc (iadd (load acc) x)))
                 (iadd (load acc) (slice-len mid)))))"#
    ));
    assert_eq!(code, 12); // (2+3+4) + len 3
}

#[test]
fn slice_set_writes_through_to_backing_storage() {
    // A slice aliases its array; a subslice aliases the same memory.
    let code = run_with(&format!(
        r#"(defn main [] (-> :i64)
             {FILL5}
               (let [s (slice-of arr 5) mid (subslice s 1 4)]
                 (slice-set! mid 0 20)            ; arr[1] = 20, visible through s
                 (slice-get s 1))))"#
    ));
    assert_eq!(code, 20);
}

#[test]
fn slice_empty_predicate() {
    let code = run_with(&format!(
        r#"(defn main [] (-> :i64)
             {FILL5}
               (let [s (slice-of arr 5) empty (subslice s 2 2)]
                 (if (slice-empty? empty) 7 0))))"#
    ));
    assert_eq!(code, 7);
}

#[test]
fn slice_for_supports_break() {
    let code = run_with(&format!(
        r#"(defn main [] (-> :i64)
             {FILL5}
               (let [s (slice-of arr 5) (mut acc) 0]
                 (slice-for [x s]
                   (if (icmp-gt x 3) (break) 0)
                   (store! acc (iadd (load acc) x)))
                 (load acc))))"#
    ));
    assert_eq!(code, 6); // 1+2+3, then x=4 breaks
}

#[test]
fn slice_for_bad_binding_hard_errors() {
    let err = coil::check_source(&format!(
        "{IMPORT}(defn main [] (-> :i64) (slice-for [x] 1))"
    ))
    .unwrap_err();
    assert!(err.contains("binding must be [x slice]"), "got: {err}");
}
