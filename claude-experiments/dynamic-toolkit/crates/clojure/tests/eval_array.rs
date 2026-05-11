//! Native mutable Array — primitive backing for core.clj's persistent
//! collections. Exposes make-array / aget / aset / alength / aclone.

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn make_array_starts_nil_filled() {
    assert_eq!(eval_str("(make-array 3)"), "#array[nil nil nil]");
}

#[test]
fn make_array_zero_size() {
    assert_eq!(eval_str("(make-array 0)"), "#array[]");
}

#[test]
fn alength_returns_count() {
    assert_eq!(eval_str("(alength (make-array 5))"), "5");
}

#[test]
fn aset_then_aget() {
    assert_eq!(
        eval_str(
            "(let [a (make-array 3)] \
                (aset a 0 10) \
                (aset a 1 20) \
                (aset a 2 30) \
                (aget a 1))"
        ),
        "20"
    );
}

#[test]
fn aset_returns_value() {
    // aset returns the stored value, mirroring core.clj's expectation.
    assert_eq!(eval_str("(aset (make-array 1) 0 42)"), "42");
}

#[test]
fn aset_mutation_visible_via_aget_in_loop() {
    // The same array referenced after each store sees the new value.
    assert_eq!(
        eval_str(
            "(let [a (make-array 4)] \
                (aset a 0 1) \
                (aset a 1 2) \
                (aset a 2 3) \
                (aset a 3 4) \
                (+ (+ (aget a 0) (aget a 1)) (+ (aget a 2) (aget a 3))))"
        ),
        "10"
    );
}

#[test]
fn aclone_produces_independent_copy() {
    // Mutating the clone must not affect the original.
    assert_eq!(
        eval_str(
            "(let [a (make-array 2)] \
                (aset a 0 99) \
                (aset a 1 100) \
                (let [b (aclone a)] \
                  (aset b 0 1) \
                  (aget a 0)))"
        ),
        "99"
    );
}

#[test]
fn arrays_can_hold_mixed_values() {
    assert_eq!(
        eval_str(
            "(let [a (make-array 3)] \
                (aset a 0 :keyword) \
                (aset a 1 \"str\") \
                (aset a 2 42) \
                a)"
        ),
        "#array[:keyword \"str\" 42]"
    );
}
