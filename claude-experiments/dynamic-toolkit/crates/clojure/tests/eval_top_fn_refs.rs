//! Bare references to `def`d top-level fns should resolve to the
//! `Fn` heap value via `Var.root`, so they can be passed around as
//! first-class callables.

use clojure::Engine;

fn eval_str(src: &str) -> String {
    let e = Engine::new();
    let v = e.eval(src);
    e.print(v)
}

#[test]
fn ref_def_fn_as_value() {
    // `inc-impl` is def'd; `inc-ref` binds the bare reference (which
    // resolves to the Fn obj). Calling `inc-ref` indirectly should
    // invoke `inc-impl`.
    let src = "\
        (def inc-impl (fn [x] (+ x 1))) \
        (let [inc-ref inc-impl] (inc-ref 41))";
    assert_eq!(eval_str(src), "42");
}

#[test]
fn pass_def_fn_as_argument() {
    // `apply1` takes a fn and a value, applies fn to value.
    let src = "\
        (def double-it (fn [x] (* x 2))) \
        (def apply1 (fn [f x] (f x))) \
        (apply1 double-it 21)";
    assert_eq!(eval_str(src), "42");
}

#[test]
fn def_fn_via_let_self_reference() {
    // `f` rebinds to `f` itself in a let — the bare reference picks
    // up the def'd Fn obj.
    let src = "\
        (def square (fn [x] (* x x))) \
        (let [g square] (g 7))";
    assert_eq!(eval_str(src), "49");
}

#[test]
fn def_fn_followed_by_redef_uses_new_root() {
    // Two def's of the same name. The second overwrites Var.root.
    // A subsequent reference reads the CURRENT root, so it picks up
    // the new fn — Vars are mutable cells.
    let src = "\
        (def inc-impl (fn [x] (+ x 1))) \
        (def inc-impl (fn [x] (* x 100))) \
        (let [g inc-impl] (g 5))";
    assert_eq!(eval_str(src), "500");
}
