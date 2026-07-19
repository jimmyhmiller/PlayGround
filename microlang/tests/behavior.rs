//! Locks the two axes the sketch is about, plus backend composition.

use microlang::{
    AlwaysMonomorphic, BlacklistAfter, BytecodeVm, ClosureComp, CodeSpace, Dispatch, DispatchStats,
    HighBitModel, LowBitModel, Megamorphic, ModelEmit, MonomorphicIc, NanBoxModel, NeverSpeculate,
    PolymorphicIc, Runtime, SpecStats, Speculative, SpeculationPolicy, Traced, TreeWalk, Val,
    ValueModel,
};

fn eval1<M: ValueModel>(src: &str) -> String {
    let mut rt = Runtime::<M>::new();
    let cs = TreeWalk;
    let r = microlang::sexpr::eval_str(&mut rt, &cs, src);
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
        let mut sx = microlang::sexpr::Sexpr::new(&mut rt);
        let forms = microlang::sexpr::read_all(&mut rt, src);
        let before = rt.allocs();
        for f in forms {
            sx.eval_top(&mut rt, &cs, f);
        }
        rt.allocs() - before
    }
    assert_eq!(allocs_for::<LowBitModel>("(+ (* 2 3) (* 4 5))"), 0);
    assert!(allocs_for::<NanBoxModel>("(+ (* 2 3) (* 4 5))") > 0);
    assert!(allocs_for::<LowBitModel>("(+ (* 2.0 3.0) (* 4.0 5.0))") > 0);
    assert_eq!(allocs_for::<NanBoxModel>("(+ (* 2.0 3.0) (* 4.0 5.0))"), 0);
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
    let r = microlang::sexpr::eval_str(&mut rt, 
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
    let r = microlang::sexpr::eval_str(&mut rt, cs, src);
    rt.print(r)
}

/// The two tiers agree on every program — same `Ir`, same contract, different
/// strategy. This is the whole point of decoupling meaning from execution.
#[test]
fn tiers_agree() {
    let progs = [
        "(+ (* 2 3) (* 4 5))",
        "(def f (fn (n) (if (< n 2) 1 (* n (f (- n 1)))))) (f 6)",
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
    microlang::sexpr::eval_str(&mut rt, 
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
    let r = microlang::sexpr::eval_str(&mut rt, 
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
    let r = microlang::sexpr::eval_str(&mut rt, 
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

// ── moving GC + the handle discipline ───────────────────────

/// The relocation mechanism: a rooted object survives a collection but MOVES;
/// the handle re-reads the new address; the stale bare pointer is a loud
/// use-after-move. This is form-609's essence in miniature.
#[test]
fn relocation_moves_and_handle_rereads() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let mut rt = Runtime::<LowBitModel>::new();
    let one = rt.encode(Val::Int(1));
    let nil = rt.encode(Val::Nil);
    let list = rt.cons(one, nil);
    let stale = list; // bare pointer the GC cannot see
    let h = rt.root(list); // published to the shadow stack
    for _ in 0..5 {
        rt.cons(one, one); // garbage to make movement obvious
    }
    rt.collect(&None);
    let moved = h.get(&rt);
    assert!(rt.relocated() > 0, "the collector should relocate the survivor");
    assert_ne!(stale, moved, "the object moved to a new address");
    assert_eq!(rt.print(moved), "(1)"); // handle re-read is correct
    rt.pop_root();
    let bad = catch_unwind(AssertUnwindSafe(|| rt.as_cons(stale)));
    assert!(bad.is_err(), "the stale pointer must be a loud use-after-move");
}

/// Globals and the lexical env stay sound across a move: keep is reachable via a
/// global, its list survives and relocates, and `(first keep)` still reads 1
/// through the auto-updated cells.
#[test]
fn globals_and_env_sound_across_move() {
    let mut rt = Runtime::<LowBitModel>::new();
    let cs = TreeWalk;
    let r = microlang::sexpr::eval_str(&mut rt, 
        &cs,
        r#"
        (def keep (list 1 2 3))
        (list 9 9 9)              ; garbage
        (gc)
        (first keep)
        "#,
    );
    assert_eq!(rt.print(r), "1");
    assert!(rt.relocated() > 0);
    assert_eq!(rt.root_depth(), 0, "roots balanced after evaluation");
}

// ── does the value axis actually hold on the HARD paths? ────
//
// The GC/slot tests above ran on LowBit only. Everything model-independent
// (recursion, moving GC, closures capturing across frames, structural equality)
// must give identical results under any conforming representation. Run the
// whole hard suite generically and instantiate it for each model.

fn hard_suite<M: ValueModel>() {
    let cs = TreeWalk;

    // recursion + arithmetic
    {
        let mut rt = Runtime::<M>::new();
        let r = microlang::sexpr::eval_str(&mut rt, &cs, "(def f (fn (n) (if (< n 2) 1 (* n (f (- n 1)))))) (f 6)");
        assert_eq!(rt.print(r), "720");
    }
    // closures capturing across frames (slot resolution)
    {
        let mut rt = Runtime::<M>::new();
        let r = microlang::sexpr::eval_str(&mut rt, 
            &cs,
            "(def f (fn (a) (let (b (+ a 1) c (+ b 1)) (fn (d) (+ (+ a b) (+ c d)))))) ((f 10) 100)",
        );
        assert_eq!(rt.print(r), "133");
    }
    // structural equality over heap lists
    {
        let mut rt = Runtime::<M>::new();
        let r = microlang::sexpr::eval_str(&mut rt, &cs, "(= (list 1 2 3) (list 1 2 3))");
        assert_eq!(rt.print(r), "true");
    }
    // MOVING GC relocates a survivor; handle re-reads; stale pointer dies
    {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let mut rt = Runtime::<M>::new();
        let one = rt.encode(Val::Int(1));
        let nil = rt.encode(Val::Nil);
        let list = rt.cons(one, nil);
        let stale = list;
        let h = rt.root(list);
        for _ in 0..5 {
            rt.cons(one, one);
        }
        rt.collect(&None);
        assert!(rt.relocated() > 0);
        assert_eq!(rt.print(h.get(&rt)), "(1)");
        rt.pop_root();
        assert!(catch_unwind(AssertUnwindSafe(|| rt.as_cons(stale))).is_err());
    }
}

#[test]
fn hard_suite_lowbit() {
    hard_suite::<LowBitModel>();
}

#[test]
fn hard_suite_nanbox() {
    hard_suite::<NanBoxModel>();
}

/// A deliberately different THIRD representation to test "any conforming Repr":
/// high-bit tagging (tag in the TOP 3 bits, payload in the low 61), the mirror
/// image of `LowBit`. Integers immediate, floats boxed. If the axis is real,
/// the whole hard suite passes here with zero changes anywhere else.
#[test]
fn hard_suite_high_bit() {
    hard_suite::<microlang::HighBitModel>();
}

// ── dispatch axis: swappable strategies ─────────────────────

const SHAPES: &str = r#"
    (defmethod area Circle (fn (s) (* (field s 0) (field s 0))))
    (defmethod area Square (fn (s) (+ (field s 0) (field s 0))))
    (def total (fn (xs) (if (nil? xs) 0 (+ (area (first xs)) (total (rest xs))))))
    (total (list (record 'Circle 3) (record 'Square 4) (record 'Circle 5) (record 'Square 6)))
"#;

fn run_dispatch(d: Box<dyn Dispatch>) -> (String, DispatchStats) {
    let mut rt = Runtime::<LowBitModel>::new();
    rt.set_dispatch(d);
    let cs = TreeWalk;
    let r = microlang::sexpr::eval_str(&mut rt, &cs, SHAPES);
    (rt.print(r), rt.dispatch_stats())
}

/// Every strategy computes the same answer (correct dispatch: Circle area is
/// r*r, Square is s+s), but the caches behave differently at the one call site,
/// which is hit four times over an alternating-type list.
#[test]
fn dispatch_strategies_agree_and_cache_differently() {
    let (r_mega, _s_mega) = run_dispatch(Box::new(Megamorphic::new()));
    let (r_mono, s_mono) = run_dispatch(Box::new(MonomorphicIc::new()));
    let (r_poly, s_poly) = run_dispatch(Box::new(PolymorphicIc::new(4)));

    // 3*3 + (4+4) + 5*5 + (6+6) = 9 + 8 + 25 + 12 = 54
    assert_eq!(r_mega, "54");
    assert_eq!(r_mono, "54");
    assert_eq!(r_poly, "54");

    // The site sees Circle, Square, Circle, Square. The mono IC caches one type,
    // so it misses on every element (thrash); the poly IC caches both types and
    // hits after the first of each.
    assert_eq!(s_mono.hits, 0, "mono IC thrashes on alternating types");
    assert_eq!(s_poly.hits, 2, "poly IC hits Circle and Square after warmup");
    assert!(s_poly.hits > s_mono.hits);
}

/// The dispatch⟺GC coupling: a moving collection relocates the receiver AND the
/// method impl, and invalidates the inline cache. Method impls are roots (the
/// registry is forwarded), so dispatch still resolves after the move.
#[test]
fn dispatch_survives_moving_gc() {
    let mut rt = Runtime::<LowBitModel>::new();
    rt.set_dispatch(Box::new(MonomorphicIc::new()));
    let cs = TreeWalk;
    let r = microlang::sexpr::eval_str(&mut rt, 
        &cs,
        r#"
        (defmethod area Circle (fn (s) (* (field s 0) (field s 0))))
        (def c (record 'Circle 3))
        (area c)       ; fills the inline cache
        (gc)           ; relocates c + impl, clears the cache
        (area c)       ; refills; impl survived as a root
        "#,
    );
    assert_eq!(rt.print(r), "9");
    assert!(rt.relocated() > 0);
}

// ── speculation + deopt axis: swappable policies ────────────

const SPEC_DEFS: &str = r#"
    (defmethod area Circle (fn (s) (* (field s 0) (field s 0))))
    (defmethod area Square (fn (s) (+ (field s 0) (field s 0))))
    (def total (fn (xs) (if (nil? xs) 0 (+ (area (first xs)) (total (rest xs))))))
"#;
// alternating types -> a polymorphic call site
const SPEC_POLY: &str =
    "(total (list (record 'Circle 3) (record 'Square 4) (record 'Circle 5) (record 'Square 6)))";
// one type -> a monomorphic call site
const SPEC_MONO: &str =
    "(total (list (record 'Circle 3) (record 'Circle 4) (record 'Circle 5) (record 'Circle 6)))";

fn run_spec(policy: impl SpeculationPolicy + 'static, tail: &str) -> (String, SpecStats) {
    let mut rt = Runtime::<LowBitModel>::new();
    // Speculation is a dispatch strategy wrapping a fallback dispatch.
    let spec = Speculative::new(Megamorphic::new(), policy);
    let counters = spec.counters();
    rt.set_dispatch(Box::new(spec));
    let cs = TreeWalk;
    let prog = format!("{SPEC_DEFS}{tail}");
    let r = microlang::sexpr::eval_str(&mut rt, &cs, &prog);
    (rt.print(r), counters.snapshot())
}

/// The invariant that makes deopt correct: speculation NEVER changes results.
/// Every policy computes the same answer on both inputs, because each guard
/// failure reconciles with the real receiver type.
#[test]
fn speculation_never_changes_results() {
    for tail in [SPEC_POLY, SPEC_MONO] {
        let expected = if tail == SPEC_POLY { "54" } else { "86" };
        assert_eq!(run_spec(NeverSpeculate, tail).0, expected);
        assert_eq!(run_spec(AlwaysMonomorphic, tail).0, expected);
        assert_eq!(run_spec(BlacklistAfter(2), tail).0, expected);
    }
}

/// Same program, three policies, different speculation behavior at the site.
#[test]
fn speculation_policies_behave_differently() {
    // Polymorphic site: never = pure fallback; always = deopt-thrash; blacklist
    // = deopt twice then give up.
    let (_, never) = run_spec(NeverSpeculate, SPEC_POLY);
    assert_eq!(never, SpecStats { spec_hits: 0, deopts: 0, fallbacks: 4 });

    let (_, always) = run_spec(AlwaysMonomorphic, SPEC_POLY);
    assert_eq!(always, SpecStats { spec_hits: 0, deopts: 3, fallbacks: 1 });

    let (_, black) = run_spec(BlacklistAfter(2), SPEC_POLY);
    assert_eq!(black, SpecStats { spec_hits: 0, deopts: 2, fallbacks: 2 });

    // Monomorphic site: speculate once, then all guard hits, zero deopts.
    let (_, mono) = run_spec(AlwaysMonomorphic, SPEC_MONO);
    assert_eq!(mono, SpecStats { spec_hits: 3, deopts: 0, fallbacks: 1 });
}

/// Because speculation is a dispatch strategy (a runtime hook), it composes with
/// BOTH execution tiers — including the closure-compiler, which inlines the
/// dispatch node and would have escaped a node-level wrapper.
#[test]
fn speculation_composes_with_closurecomp() {
    let mut rt = Runtime::<LowBitModel>::new();
    let spec = Speculative::new(Megamorphic::new(), AlwaysMonomorphic);
    let counters = spec.counters();
    rt.set_dispatch(Box::new(spec));
    let cs = ClosureComp::<LowBitModel>::new();
    let prog = format!("{SPEC_DEFS}{SPEC_MONO}");
    let r = microlang::sexpr::eval_str(&mut rt, &cs, &prog);
    assert_eq!(rt.print(r), "86");
    assert!(counters.snapshot().spec_hits > 0);
}

// ── the emit tier: bytecode VM + value-model emit ───────────

fn run_bc<M: ModelEmit>(src: &str) -> String {
    let mut rt = Runtime::<M>::new();
    let vm = BytecodeVm::<M>::new();
    let r = microlang::sexpr::eval_str(&mut rt, &vm, src);
    rt.print(r)
}

fn disasm<M: ModelEmit>(src: &str) -> Vec<String> {
    let mut rt = Runtime::<M>::new();
    let vm = BytecodeVm::<M>::new();
    let forms = microlang::sexpr::read_all(&mut rt, src);
    let ir = microlang::sexpr::analyze(&mut rt, &vm, forms[0]);
    BytecodeVm::<M>::disassemble(&ir)
}

/// The bytecode tier runs real programs — arithmetic and recursion — and every
/// value representation computes the same answer.
#[test]
fn bytecode_runs_and_agrees_across_models() {
    for (src, expected) in [
        ("(+ (* 2 3) (* 4 5))", "26"),
        ("(def f (fn (n) (if (< n 2) 1 (* n (f (- n 1)))))) (f 6)", "720"),
    ] {
        assert_eq!(run_bc::<LowBitModel>(src), expected);
        assert_eq!(run_bc::<HighBitModel>(src), expected);
        assert_eq!(run_bc::<NanBoxModel>(src), expected);
    }
}

/// The emit half: the SAME source produces DIFFERENT bytecode per representation.
/// LowBit shifts to untag before multiply; HighBit needs no shift; NanBox boxes
/// integers so arithmetic is a slow-path call. The compiler is unchanged; only
/// the model's `emit_*` differs.
#[test]
fn bytecode_is_model_specific() {
    let lb = disasm::<LowBitModel>("(* 2 3)");
    assert!(lb.iter().any(|s| s == "Sar 3"));
    assert!(lb.iter().any(|s| s == "MulRaw"));

    let hb = disasm::<HighBitModel>("(* 2 3)");
    assert!(hb.iter().any(|s| s == "MulRaw"));
    assert!(!hb.iter().any(|s| s.starts_with("Sar")), "HighBit needs no shift");

    let nb = disasm::<NanBoxModel>("(* 2 3)");
    assert!(nb.iter().any(|s| s.starts_with("Slow(Mul")), "NanBox boxes ints");
    assert!(!nb.iter().any(|s| s == "MulRaw"), "NanBox emits no raw int multiply");
}

// ── the grand matrix: every combination agrees ─────────────
//
// Prove the axes are orthogonal. A feature-rich program (records + methods +
// dispatch + recursion + a mid-program moving GC) runs across the cross product
// of {3 value representations} × {2 general tiers} × {6 dispatch strategies},
// and an arithmetic program runs across {3 representations} × {3 tiers}
// (including the emit tier). Every single combination computes the same answer.

fn grand_matrix_for<M: ModelEmit>() -> usize {
    let mut count = 0;

    // Arithmetic + recursion: all three execution tiers, including bytecode.
    let prog_b = "(def f (fn (n) (if (< n 2) 1 (* n (f (- n 1)))))) (f 6)";
    for tier_idx in 0..3 {
        let tier: Box<dyn CodeSpace<M>> = match tier_idx {
            0 => Box::new(TreeWalk),
            1 => Box::new(ClosureComp::<M>::new()),
            _ => Box::new(BytecodeVm::<M>::new()),
        };
        let mut rt = Runtime::<M>::new();
        let r = microlang::sexpr::eval_str(&mut rt, tier.as_ref(), prog_b);
        assert_eq!(rt.print(r), "720", "arith combo {tier_idx}");
        count += 1;
    }

    // Records + dispatch + a mid-program moving GC: tiers that support dispatch
    // (tree-walk, closure-compile) × every dispatch strategy.
    let prog_a = concat!(
        "(defmethod area Circle (fn (s) (* (field s 0) (field s 0))))",
        "(defmethod area Square (fn (s) (+ (field s 0) (field s 0))))",
        "(def total (fn (xs) (if (nil? xs) 0 (+ (area (first xs)) (total (rest xs))))))",
        "(def shapes (list (record 'Circle 3) (record 'Square 4) (record 'Circle 5) (record 'Square 6)))",
        "(gc)",            // relocate everything; caches invalidate; impls survive as roots
        "(total shapes)"
    );
    let dispatches: Vec<fn() -> Box<dyn Dispatch>> = vec![
        || Box::new(Megamorphic::new()),
        || Box::new(MonomorphicIc::new()),
        || Box::new(PolymorphicIc::new(4)),
        || Box::new(Speculative::new(Megamorphic::new(), NeverSpeculate)),
        || Box::new(Speculative::new(Megamorphic::new(), AlwaysMonomorphic)),
        || Box::new(Speculative::new(Megamorphic::new(), BlacklistAfter(2))),
    ];
    for tier_idx in 0..2 {
        for make_d in &dispatches {
            let tier: Box<dyn CodeSpace<M>> = if tier_idx == 0 {
                Box::new(TreeWalk)
            } else {
                Box::new(ClosureComp::<M>::new())
            };
            let mut rt = Runtime::<M>::new();
            rt.set_dispatch(make_d());
            let r = microlang::sexpr::eval_str(&mut rt, tier.as_ref(), prog_a);
            assert_eq!(rt.print(r), "54", "dispatch combo tier {tier_idx}");
            count += 1;
        }
    }
    count
}

#[test]
fn grand_matrix_all_combinations_agree() {
    let total = grand_matrix_for::<LowBitModel>()
        + grand_matrix_for::<NanBoxModel>()
        + grand_matrix_for::<HighBitModel>();
    // per model: 3 (arith × tiers) + 12 (2 tiers × 6 dispatch) = 15; × 3 models.
    assert_eq!(total, 45);
}

#[test]
fn nil_encodes() {
    let mut rt = Runtime::<LowBitModel>::new();
    let n = rt.encode(Val::Nil);
    assert_eq!(rt.print(n), "nil");
}
