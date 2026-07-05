//! Locks the two axes the sketch is about, plus backend composition.

use microlang::{
    ClosureComp, CodeSpace, LowBitModel, NanBoxModel, Runtime, Traced, TreeWalk, Val, ValueModel,
};

fn eval1<M: ValueModel>(src: &str) -> String {
    let mut rt = Runtime::<M>::new();
    let cs = TreeWalk;
    let r = rt.eval_str(&cs, src);
    rt.print(r)
}

/// Same source, same engine, correct answer under both value models.
#[test]
fn arithmetic_is_model_independent() {
    assert_eq!(eval1::<LowBitModel>("(+ (* 2 3) (* 4 5))"), "26");
    assert_eq!(eval1::<NanBoxModel>("(+ (* 2 3) (* 4 5))"), "26");
    assert_eq!(eval1::<LowBitModel>("(+ 1.5 2.5)"), "4.0");
    assert_eq!(eval1::<NanBoxModel>("(+ 1.5 2.5)"), "4.0");
}

/// The value axis, measured: the immediate category decides who boxes.
#[test]
fn immediacy_decides_allocation() {
    fn allocs_for<M: ValueModel>(src: &str) -> u64 {
        let mut rt = Runtime::<M>::new();
        let cs = TreeWalk;
        let forms = rt.read_all(src);
        let before = rt.allocs;
        for f in forms {
            rt.eval_top(&cs, f);
        }
        rt.allocs - before
    }
    assert_eq!(allocs_for::<LowBitModel>("(+ (* 2 3) (* 4 5))"), 0);
    assert!(allocs_for::<NanBoxModel>("(+ (* 2 3) (* 4 5))") > 0);
    assert!(allocs_for::<LowBitModel>("(+ (* 2.0 3.0) (* 4.0 5.0))") > 0);
    assert_eq!(allocs_for::<NanBoxModel>("(+ (* 2.0 3.0) (* 4.0 5.0))"), 0);
}

/// The eval axis: macros expand, and a macro may run compiled code.
#[test]
fn macros_and_reentrancy() {
    let mut rt = Runtime::<LowBitModel>::new();
    let cs = TreeWalk;
    rt.eval_str(
        &cs,
        r#"
        (defmacro unless (c a b) (list 'if c b a))
        (def inc (fn (n) (+ n 1)))
        (defmacro add2 (x) (list '+ x (inc 1)))
        "#,
    );
    let f = rt.read_all("(unless false 1 2)");
    let ex = rt.macroexpand(&cs, f[0]);
    assert_eq!(rt.print(ex), "(if false 2 1)");
    let g = rt.read_all("(add2 40)");
    let ex2 = rt.macroexpand(&cs, g[0]);
    assert_eq!(rt.print(ex2), "(+ 40 2)");
    let r = rt.eval_str(&cs, "(add2 40)");
    assert_eq!(rt.print(r), "42");
}

#[test]
fn recursion_and_higher_order() {
    assert_eq!(
        eval1::<LowBitModel>("(def f (fn (n) (if (< n 2) 1 (* n (f (- n 1)))))) (f 5)"),
        "120"
    );
    assert_eq!(
        eval1::<LowBitModel>(
            "(def m (fn (g xs) (if (nil? xs) xs (cons (g (first xs)) (m g (rest xs))))))
             (def inc (fn (n) (+ n 1)))
             (m inc (list 10 20 30))"
        ),
        "(11 21 31)"
    );
}

#[test]
fn equality_is_structural() {
    assert_eq!(eval1::<LowBitModel>("(= (list 1 2 3) (list 1 2 3))"), "true");
    assert_eq!(eval1::<NanBoxModel>("(= (list 1 2 3) (list 1 2 3))"), "true");
    assert_eq!(eval1::<LowBitModel>("(= (list 1 2) (list 1 2 3))"), "false");
}

/// Design-tension #1, resolved: the backend is a value, so backends compose,
/// AND open recursion makes the composition total. A naive wrapper (recursing
/// through `self`) would observe only the ONE call the runtime initiates;
/// threading `top` makes it observe all five recursive `fact` calls.
#[test]
fn backends_compose_with_open_recursion() {
    let traced = Traced::new(TreeWalk);
    let mut rt = Runtime::<LowBitModel>::new();
    let r = rt.eval_str(
        &traced,
        "(def fact (fn (n) (if (< n 2) 1 (* n (fact (- n 1)))))) (fact 5)",
    );
    assert_eq!(rt.print(r), "120");
    // fact invoked at n = 5,4,3,2,1 — every depth flows through the wrapper.
    assert_eq!(traced.invoke_count(), 5);
}

// ── second execution tier: ClosureComp ──────────────────────

fn eval_with<M: ValueModel>(cs: &dyn CodeSpace<M>, src: &str) -> String {
    let mut rt = Runtime::<M>::new();
    let r = rt.eval_str(cs, src);
    rt.print(r)
}

/// The two tiers agree on every program — same `Ir`, same contract, different
/// strategy. This is the whole point of decoupling meaning from execution.
#[test]
fn tiers_agree() {
    let progs = [
        "(+ (* 2 3) (* 4 5))",
        "(def f (fn (n) (if (< n 2) 1 (* n (f (- n 1)))))) (f 6)",
        "(defmacro unless (c a b) (list 'if c b a)) (unless false 1 2)",
        "(def m (fn (g xs) (if (nil? xs) xs (cons (g (first xs)) (m g (rest xs))))))
         (def inc (fn (n) (+ n 1)))
         (m inc (list 10 20 30))",
    ];
    for p in progs {
        let tw = eval_with::<LowBitModel>(&TreeWalk, p);
        let cc = eval_with::<LowBitModel>(&ClosureComp::<LowBitModel>::new(), p);
        assert_eq!(tw, cc, "tiers disagreed on: {p}");
    }
}

/// Compile-once: a function called at many depths compiles a single time.
#[test]
fn compiles_bodies_once() {
    let cs = ClosureComp::<LowBitModel>::new();
    let mut rt = Runtime::<LowBitModel>::new();
    rt.eval_str(
        &cs,
        "(def fact (fn (n) (if (< n 2) 1 (* n (fact (- n 1)))))) (fact 6)",
    );
    // exactly one function body (fact) was compiled, despite 6 recursive calls
    assert_eq!(cs.compiled_bodies(), 1);
}

/// Late binding: a compiled function calls one defined AFTER it. Mutual
/// recursion with a forward reference, on the compiling tier.
#[test]
fn late_binding_forward_reference() {
    let cs = ClosureComp::<LowBitModel>::new();
    let mut rt = Runtime::<LowBitModel>::new();
    let r = rt.eval_str(
        &cs,
        r#"
        (def even? (fn (n) (if (= n 0) true  (odd?  (- n 1)))))
        (def odd?  (fn (n) (if (= n 0) false (even? (- n 1)))))
        (even? 10)
        "#,
    );
    // even?'s body was compiled before odd? existed; resolution is at call time
    assert_eq!(rt.print(r), "true");
}

/// Composition works across tiers too: wrap the compiling backend.
#[test]
fn traced_wraps_compiler() {
    let traced = Traced::new(ClosureComp::<LowBitModel>::new());
    let mut rt = Runtime::<LowBitModel>::new();
    let r = rt.eval_str(
        &traced,
        "(def fact (fn (n) (if (< n 2) 1 (* n (fact (- n 1)))))) (fact 5)",
    );
    assert_eq!(rt.print(r), "120");
    assert_eq!(traced.invoke_count(), 5);
}

// ── slot resolution (compile-time lexical addressing) ───────

/// Run on BOTH tiers and assert they agree with the expected result. Slot
/// resolution is shared (it happens in `analyze`), so this checks the `Ir`/env
/// cut is right for both consumers.
fn both(src: &str, expected: &str) {
    assert_eq!(
        eval_with::<LowBitModel>(&TreeWalk, src),
        expected,
        "TreeWalk: {src}"
    );
    assert_eq!(
        eval_with::<LowBitModel>(&ClosureComp::<LowBitModel>::new(), src),
        expected,
        "ClosureComp: {src}"
    );
}

/// Closures capturing across several frames: `x` at up:1, and a 3-deep case
/// (param, let, inner-fn param) that only works if every `(up, idx)` is exact.
#[test]
fn slot_resolution_deep_capture() {
    both("(def add (fn (x) (fn (y) (+ x y)))) ((add 10) 5)", "15");
    both(
        "(def f (fn (a) (let (b (+ a 1) c (+ b 1)) (fn (d) (+ (+ a b) (+ c d))))))
         ((f 10) 100)",
        "133",
    );
}

/// Shadowing: a local shadows a global; an inner `let` shadows a param, while
/// the init still sees the outer binding (`let*` order).
#[test]
fn slot_resolution_shadowing() {
    both("(def x 1) (let (x 10) x)", "10");
    both("((fn (x) (let (x (+ x 1)) x)) 5)", "6");
}

#[test]
fn nil_encodes() {
    let mut rt = Runtime::<LowBitModel>::new();
    let n = rt.encode(Val::Nil);
    assert_eq!(rt.print(n), "nil");
}
