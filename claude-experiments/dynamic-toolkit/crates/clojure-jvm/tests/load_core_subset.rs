//! Load `tests/fixtures/core_subset.clj` through a real Session and
//! verify each ported clojure.core function works.
//!
//! Goal: this file grows as we add features, with the fixture converging
//! on upstream `clojure/core.clj`. Each fn here is exercised end-to-end
//! through the JIT.

use clojure_jvm::lang::compiler::Session;

const CORE_SUBSET: &str = include_str!("fixtures/core_subset.clj");

/// Build a fresh Session, switch to a temp namespace (so the fixture's
/// `(ns clojure.core)` doesn't fight clojure.core's global state across
/// test runs), and load the subset.
fn load_subset() -> Session {
    let mut sess = Session::new();
    sess.eval_str(CORE_SUBSET);
    sess
}

const TAG_NIL_BITS: u64 = 0x7FFC_0000_0000_0000;

fn nb_to_f64(bits: u64) -> f64 {
    f64::from_bits(bits)
}

fn nb_to_bool(bits: u64) -> bool {
    const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
    const TAG_MASK: u64 = 0x0003_0000_0000_0000;
    const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
    assert_eq!(bits & FULL_MASK, 0x7FFC_0000_0000_0000, "not a NanBox tag");
    let tag = (bits & TAG_MASK) >> 48;
    assert_eq!(tag, 1, "expected TAG_BOOL=1, got {tag}");
    (bits & PAYLOAD_MASK) != 0
}

#[test]
fn loads_without_panicking() {
    let _ = load_subset();
}

#[test]
fn list_returns_a_list() {
    let mut sess = load_subset();
    // (first (list 10 20 30)) → 10
    let v = sess.eval_str("(first (list 10 20 30))");
    assert_eq!(nb_to_f64(v), 10.0);
}

#[test]
fn cons_basic() {
    let mut sess = load_subset();
    let v = sess.eval_str("(first (cons 7 nil))");
    assert_eq!(nb_to_f64(v), 7.0);
}

#[test]
fn second_walks_one() {
    let mut sess = load_subset();
    let v = sess.eval_str("(second (list 1 2 3))");
    assert_eq!(nb_to_f64(v), 2.0);
}

#[test]
fn next_returns_tail() {
    let mut sess = load_subset();
    let v = sess.eval_str("(first (next (list 10 20 30)))");
    assert_eq!(nb_to_f64(v), 20.0);
}

#[test]
fn rest_returns_tail() {
    let mut sess = load_subset();
    let v = sess.eval_str("(first (rest (list 10 20 30)))");
    assert_eq!(nb_to_f64(v), 20.0);
}

#[test]
fn ffirst_first_first() {
    let mut sess = load_subset();
    let v = sess.eval_str("(ffirst (list (list 7 8) 9))");
    assert_eq!(nb_to_f64(v), 7.0);
}

#[test]
fn fnext_skip_one_take_one() {
    let mut sess = load_subset();
    let v = sess.eval_str("(fnext (list 100 200 300))");
    assert_eq!(nb_to_f64(v), 200.0);
}

#[test]
fn nnext_skip_two() {
    let mut sess = load_subset();
    let v = sess.eval_str("(first (nnext (list 1 2 3 4)))");
    assert_eq!(nb_to_f64(v), 3.0);
}

#[test]
fn next_on_one_elem_returns_nil() {
    let mut sess = load_subset();
    let v = sess.eval_str("(next (list 42))");
    assert_eq!(v, TAG_NIL_BITS, "expected nil");
}

/// Marker test so failed runs identify the helper file is correctly
/// included. If this fires, the include_str! path is wrong.
#[test]
fn fixture_is_non_empty() {
    assert!(!CORE_SUBSET.is_empty(), "fixture string must be non-empty");
    assert!(CORE_SUBSET.contains("(ns "), "expected `(ns ` in fixture");
}

#[test]
fn conj_two_arg_prepends() {
    let mut sess = load_subset();
    let v = sess.eval_str("(first (conj nil 1))");
    assert_eq!(nb_to_f64(v), 1.0);
}

#[test]
fn conj_three_arg_via_variadic_recur() {
    let mut sess = load_subset();
    // (conj nil 1 2 3) →  via recur, conses each in turn:
    //   ((conj nil 1) 2 3) → ((1) 2 3) → (recur (cons 2 (1)) 3 nil) → ((2 1) 3) → ...
    // The end result is a 3-element list. We check `first` and a walk.
    let v = sess.eval_str("(first (conj nil 1 2 3))");
    // After all the recur'd cons calls, the last conj of 3 wins as `first`.
    assert_eq!(nb_to_f64(v), 3.0);
}

#[test]
fn nil_q_true_for_nil() {
    let mut sess = load_subset();
    assert!(nb_to_bool(sess.eval_str("(nil? nil)")));
    assert!(!nb_to_bool(sess.eval_str("(nil? 0)")));
}

#[test]
fn true_q_and_false_q() {
    let mut sess = load_subset();
    assert!(nb_to_bool(sess.eval_str("(true? true)")));
    assert!(!nb_to_bool(sess.eval_str("(true? false)")));
    assert!(!nb_to_bool(sess.eval_str("(true? nil)")));

    assert!(nb_to_bool(sess.eval_str("(false? false)")));
    assert!(!nb_to_bool(sess.eval_str("(false? true)")));
}

#[test]
fn not_inverts_truthiness() {
    let mut sess = load_subset();
    assert!(nb_to_bool(sess.eval_str("(not false)")));
    assert!(nb_to_bool(sess.eval_str("(not nil)")));
    assert!(!nb_to_bool(sess.eval_str("(not 0)")));
    assert!(!nb_to_bool(sess.eval_str("(not true)")));
}

#[test]
fn last_walks_to_end() {
    let mut sess = load_subset();
    let v = sess.eval_str("(last (list 1 2 3 4))");
    assert_eq!(nb_to_f64(v), 4.0);
}

#[test]
fn last_of_single_elem() {
    let mut sess = load_subset();
    let v = sess.eval_str("(last (list 99))");
    assert_eq!(nb_to_f64(v), 99.0);
}

#[test]
fn identical_q_basic() {
    let mut sess = load_subset();
    assert!(nb_to_bool(sess.eval_str("(identical? nil nil)")));
    assert!(nb_to_bool(sess.eval_str("(identical? true true)")));
    assert!(!nb_to_bool(sess.eval_str("(identical? true false)")));
    assert!(nb_to_bool(sess.eval_str("(identical? 42 42)")));
}

#[test]
fn inc_dec_basic() {
    let mut sess = load_subset();
    assert_eq!(nb_to_f64(sess.eval_str("(inc 5)")), 6.0);
    assert_eq!(nb_to_f64(sess.eval_str("(dec 5)")), 4.0);
    assert_eq!(nb_to_f64(sess.eval_str("(inc (inc 0))")), 2.0);
}

#[test]
fn pos_neg_zero_predicates() {
    let mut sess = load_subset();
    assert!(nb_to_bool(sess.eval_str("(pos? 5)")));
    assert!(!nb_to_bool(sess.eval_str("(pos? -1)")));
    assert!(!nb_to_bool(sess.eval_str("(pos? 0)")));

    assert!(nb_to_bool(sess.eval_str("(neg? -1)")));
    assert!(!nb_to_bool(sess.eval_str("(neg? 5)")));

    assert!(nb_to_bool(sess.eval_str("(zero? 0)")));
    assert!(!nb_to_bool(sess.eval_str("(zero? 1)")));
}

#[test]
fn max_two_args() {
    let mut sess = load_subset();
    assert_eq!(nb_to_f64(sess.eval_str("(max 3 7)")), 7.0);
    assert_eq!(nb_to_f64(sess.eval_str("(max 9 4)")), 9.0);
}

#[test]
fn min_two_args() {
    let mut sess = load_subset();
    assert_eq!(nb_to_f64(sess.eval_str("(min 3 7)")), 3.0);
    assert_eq!(nb_to_f64(sess.eval_str("(min 9 4)")), 4.0);
}

#[test]
fn max_min_variadic() {
    let mut sess = load_subset();
    assert_eq!(nb_to_f64(sess.eval_str("(max 1 9 3 7 2)")), 9.0);
    assert_eq!(nb_to_f64(sess.eval_str("(min 5 2 8 1 4)")), 1.0);
}

#[test]
fn if_not_macro_swaps_branches() {
    let mut sess = load_subset();
    // (if-not false 1 2) → (if false 2 1) → 1
    assert_eq!(nb_to_f64(sess.eval_str("(if-not false 1 2)")), 1.0);
    assert_eq!(nb_to_f64(sess.eval_str("(if-not true 1 2)")), 2.0);
}

#[test]
fn when_macro_emits_when_truthy() {
    let mut sess = load_subset();
    assert_eq!(nb_to_f64(sess.eval_str("(when true 42)")), 42.0);
    // When test is falsy → nil.
    assert_eq!(sess.eval_str("(when false 42)"), TAG_NIL_BITS);
    assert_eq!(sess.eval_str("(when nil 42)"), TAG_NIL_BITS);
}

#[test]
fn when_not_macro_emits_when_falsy() {
    let mut sess = load_subset();
    assert_eq!(nb_to_f64(sess.eval_str("(when-not false 99)")), 99.0);
    assert_eq!(sess.eval_str("(when-not true 99)"), TAG_NIL_BITS);
}
