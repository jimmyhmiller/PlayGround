//! The nanopass optimizer (`Optimized`) must be BEHAVIOR-PRESERVING: wrapping any
//! backend changes speed, never answers. These lock it to the tree-walker (the
//! reference) across value models, and check the fixnum specialization actually
//! fires and stays idempotent.
#![cfg(feature = "jit")]

use microlang::ir::{Ir, Prim};
use microlang::optimize::{
    eliminate_dead_lets, fold_const, inline, propagate_copies, simplify, specialize_fixnums,
};
use microlang::{
    HighBitModel, JitCranelift, LowBitModel, ModelArithJit, NanBoxModel, Optimized, Runtime,
    TreeWalk, ValueModel,
};

fn walk<M: ValueModel>(src: &str) -> String {
    let mut rt = Runtime::<M>::new();
    let r = microlang::sexpr::eval_str(&mut rt, &TreeWalk, src);
    rt.print(r)
}

/// Analyze one core-surface form to `Ir` (no scheme desugar needed for the
/// `fn`/`def`/`if`/`+`/`<` forms used here).
fn to_ir<M: ValueModel>(rt: &mut Runtime<M>, src: &str) -> Ir {
    let forms = microlang::sexpr::read_all(rt, src);
    microlang::sexpr::analyze(rt, &TreeWalk, forms[0])
}

/// Optimized JIT: the pass pipeline in front of the native tier.
fn opt_jit<M: ModelArithJit>(src: &str) -> String {
    let mut rt = Runtime::<M>::new();
    let cs = Optimized::new(JitCranelift::<M>::new());
    let r = microlang::sexpr::eval_str(&mut rt, &cs, src);
    rt.print(r)
}

/// Optimizing must not change any answer, on any value model — including across
/// the full numeric tower (negatives, fixnum→bignum overflow), which exercises
/// the specialized fast path's overflow fall-back.
#[test]
fn optimized_matches_treewalk_across_models() {
    for src in [
        "(def fib (fn (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))) (fib 15)",
        "(def tak (fn (x y z) (if (< y x) (tak (tak (- x 1) y z) (tak (- y 1) z x) (tak (- z 1) x y)) z))) (tak 12 8 4)",
        "(def fact (fn (n) (if (< n 2) 1 (* n (fact (- n 1)))))) (fact 8)",
        "(def go (fn (n acc) (if (= n 0) acc (go (- n 1) (+ acc n))))) (go 20000 0)",
        "(- 3 10)",                               // negative
        "(def f (fn (n) (* n n))) (f 100000000000)", // fixnum*fixnum overflow -> bignum
        "(+ 4611686018427387904 4611686018427387904)", // crosses the fixnum edge
        "(if (< 1 2) 100 200)",
    ] {
        assert_eq!(opt_jit::<LowBitModel>(src), walk::<LowBitModel>(src), "LowBit: {src}");
        assert_eq!(opt_jit::<HighBitModel>(src), walk::<HighBitModel>(src), "HighBit: {src}");
        assert_eq!(opt_jit::<NanBoxModel>(src), walk::<NanBoxModel>(src), "NanBox: {src}");
    }
}

/// Overflow still promotes under the specialized path: `(fact 25)` overflows i64
/// mid-way, and the guarded fast op must fall back to the runtime's bignum.
#[test]
fn optimized_keeps_the_numeric_tower() {
    let src = "(def fact (fn (n) (if (< n 2) 1 (* n (fact (- n 1)))))) (fact 25)";
    assert_eq!(opt_jit::<LowBitModel>(src), walk::<LowBitModel>(src));
    assert_eq!(opt_jit::<LowBitModel>(src), "15511210043330985984000000");
}

fn count(ir: &Ir, pred: &impl Fn(&Ir) -> bool) -> usize {
    let here = pred(ir) as usize;
    let kids: usize = match ir {
        Ir::SetLocal { val, .. } | Ir::SetGlobal { val, .. } => count(val, pred),
        Ir::If(c, t, e) => count(c, pred) + count(t, pred) + count(e, pred),
        Ir::Do(xs) | Ir::Prim(_, xs) | Ir::Dispatch { args: xs, .. } => {
            xs.iter().map(|x| count(x, pred)).sum()
        }
        Ir::Def { init, .. } => count(init, pred),
        Ir::Let(inits, body) => inits.iter().map(|x| count(x, pred)).sum::<usize>() + count(body, pred),
        Ir::Lambda { body, .. } => count(body, pred),
        Ir::Call(f, args) => count(f, pred) + args.iter().map(|x| count(x, pred)).sum::<usize>(),
        Ir::DefMethod { imp, .. } => count(imp, pred),
        _ => 0,
    };
    here + kids
}

fn is_fx(ir: &Ir) -> bool {
    matches!(ir, Ir::Prim(Prim::FxAdd | Prim::FxSub | Prim::FxMul | Prim::FxLt | Prim::FxEq, _))
}
fn is_guard(ir: &Ir) -> bool {
    matches!(ir, Ir::Prim(Prim::AllFixnum, _))
}

/// The specializer fires on a PURE SELF-TAIL LOOP (the profitable shape): a
/// guard at entry plus `Fx*` ops for the params' arithmetic — and the transform
/// is IDEMPOTENT (running it twice adds nothing).
#[test]
fn specialization_fires_and_is_idempotent() {
    let mut rt = Runtime::<LowBitModel>::new();
    // `(fn (n acc) (if (= n 0) acc (go (- n 1) (+ acc 1))))` — tail self-call,
    // no non-tail calls, so it is specialized.
    let ir = { let s = "(fn (n acc) (if (= n 0) acc (go (- n 1) (+ acc 1))))"; to_ir(&mut rt, s) };

    let once = specialize_fixnums::<LowBitModel>(&rt, &ir);
    // `= n 0`, `- n 1`, `+ acc 1` all specialize (guarded param + fixnum literal);
    // the `go` tail call is not arithmetic -> 3 Fx ops, one guard.
    assert_eq!(count(&once, &is_fx), 3, "expected 3 Fx ops");
    assert_eq!(count(&once, &is_guard), 1, "expected 1 AllFixnum guard");

    let twice = specialize_fixnums::<LowBitModel>(&rt, &once);
    assert_eq!(count(&twice, &is_fx), 3, "idempotent: still 3 Fx ops");
    assert_eq!(count(&twice, &is_guard), 1, "idempotent: still 1 guard");
}

/// Call-bound tree recursion (fib) is deliberately NOT specialized: the guard
/// would be per-call overhead for no benefit (Cranelift already lowers the
/// per-op tag checks nearly free). The profitability heuristic leaves it alone.
#[test]
fn tree_recursion_is_left_alone() {
    let mut rt = Runtime::<LowBitModel>::new();
    let ir = { let s = "(fn (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))"; to_ir(&mut rt, s) };
    let s = specialize_fixnums::<LowBitModel>(&rt, &ir);
    assert_eq!(count(&s, &is_fx), 0, "fib has non-tail calls -> not specialized");
    assert_eq!(count(&s, &is_guard), 0);
}

fn is_arith(ir: &Ir) -> bool {
    matches!(ir, Ir::Prim(Prim::Add | Prim::Sub | Prim::Mul | Prim::Lt | Prim::Eq, _))
}

/// `fold_const` evaluates a nested pure-arithmetic expression down to one
/// constant — and matches what the runtime would compute.
#[test]
fn fold_const_collapses_arithmetic() {
    let mut rt = Runtime::<LowBitModel>::new();
    let ir = { let s = "(+ (* 2 3) (* 4 5))"; to_ir(&mut rt, s) };
    let f = fold_const::<LowBitModel>(&mut rt, &ir);
    assert!(matches!(f, Ir::Const(_)), "should fold to a single constant");
    assert_eq!(count(&f, &is_arith), 0, "no arithmetic prims should remain");
    // 2*3 + 4*5 = 26
    let id = if let Ir::Const(id) = f { id } else { unreachable!() };
    assert_eq!(rt.print(rt.get_const(id)), "26");
}

/// Folding preserves answers (and the numeric tower) across models, end to end.
#[test]
fn fold_const_is_behavior_preserving() {
    for src in [
        "(+ (* 2 3) (* 4 5))",
        "(if (< (+ 1 1) 3) 'yes 'no)",              // fold exposes a constant `if`
        "(* (* 1000000 1000000) 1000000)",          // folds through bignum overflow
    ] {
        assert_eq!(opt_jit::<LowBitModel>(src), walk::<LowBitModel>(src), "LowBit: {src}");
        assert_eq!(opt_jit::<NanBoxModel>(src), walk::<NanBoxModel>(src), "NanBox: {src}");
    }
}

fn is_call(ir: &Ir) -> bool {
    matches!(ir, Ir::Call(..))
}
fn is_lambda(ir: &Ir) -> bool {
    matches!(ir, Ir::Lambda { .. })
}

/// `inline` beta-reduces a directly-applied lambda: the call and the lambda both
/// disappear. With ATOM args it substitutes fully — no `let`, no frame; with a
/// COMPLEX arg it falls back to a `let` so the arg is evaluated once.
#[test]
fn inline_beta_reduces_direct_application() {
    let mut rt = Runtime::<LowBitModel>::new();

    // atom arg -> full substitution, frame eliminated
    let ir = { let s = "((fn (x) (+ x x)) 5)"; to_ir(&mut rt, s) };
    let n = inline(&ir);
    assert_eq!(count(&n, &is_call), 0, "the call is gone");
    assert_eq!(count(&n, &is_lambda), 0, "the lambda is gone");
    assert!(!matches!(n, Ir::Let(..)), "atom arg substituted, no let frame");

    // complex (non-atom) arg -> let-based beta (a slot so it evaluates once)
    let ir2 = { let s = "((fn (x) (+ x x)) (first (list 9)))"; to_ir(&mut rt, s) };
    let n2 = inline(&ir2);
    assert_eq!(count(&n2, &is_lambda), 0, "the applied lambda is gone");
    assert!(matches!(n2, Ir::Let(..)), "complex arg kept in a let, not substituted");
}

/// Inlining is behavior-preserving, INCLUDING the capture-avoiding argument
/// shift: here the argument `k` names the call-site scope and must still resolve
/// to it after being lifted one frame into the `let`.
#[test]
fn inline_is_behavior_preserving_with_capture() {
    for src in [
        "((fn (x) (+ x x)) 5)",                                  // -> 10
        "((fn (x y) (- x y)) 10 3)",                             // -> 7
        "(def f (fn (k) ((fn (x) (+ x k)) k))) (f 21)",         // arg references call-site -> 42
        "(def g (fn (k) ((fn (x) (fn (y) (+ (+ x y) k))) 3))) ((g 100) 4)", // nested capture -> 107
        "(((fn (x) (fn (y) (+ x y))) 3) 4)",                     // returned closure -> 7
    ] {
        assert_eq!(opt_jit::<LowBitModel>(src), walk::<LowBitModel>(src), "LowBit: {src}");
        assert_eq!(opt_jit::<HighBitModel>(src), walk::<HighBitModel>(src), "HighBit: {src}");
        assert_eq!(opt_jit::<NanBoxModel>(src), walk::<NanBoxModel>(src), "NanBox: {src}");
    }
}

fn let_binding_count(ir: &Ir) -> usize {
    match ir {
        Ir::Let(inits, _) => inits.len(),
        _ => 0,
    }
}

/// `eliminate_dead_lets` drops a pure, unused binding and renumbers the rest —
/// and keeps an unused binding whose init has a side effect.
#[test]
fn dce_drops_pure_unused_bindings() {
    let mut rt = Runtime::<LowBitModel>::new();

    // `y` unused + pure -> dropped; only `x` remains (referenced).
    let ir = { let s = "(let (x 1 y 2) x)"; to_ir(&mut rt, s) };
    let n = eliminate_dead_lets(&ir);
    assert_eq!(let_binding_count(&n), 1, "one binding should remain");

    // `x` unused + pure, `y` used -> `x` dropped and `y` RENUMBERED slot 1->0.
    let ir2 = { let s = "(let (x 1 y 2) y)"; to_ir(&mut rt, s) };
    let n2 = eliminate_dead_lets(&ir2);
    assert_eq!(let_binding_count(&n2), 1);

    // sequential use: `y`'s init reads `x`, so `x` is live -> keep both.
    let ir3 = { let s = "(let (x 1 y (+ x 1)) y)"; to_ir(&mut rt, s) };
    let n3 = eliminate_dead_lets(&ir3);
    assert_eq!(let_binding_count(&n3), 2, "x is used by y's init");

    // unused but IMPURE init (a call) -> must be kept for its effect.
    let ir4 = { let s = "(let (x (foo) y 2) y)"; to_ir(&mut rt, s) };
    assert_eq!(let_binding_count(&eliminate_dead_lets(&ir4)), 2, "impure init kept");
}

/// DCE (with renumbering) is behavior-preserving end to end, across models.
#[test]
fn dce_is_behavior_preserving() {
    for src in [
        "(let (x 1 y 2) y)",                    // drop+renumber -> 2
        "(let (a 10 b 20 c 30) (+ a c))",       // drop middle `b`, renumber `c` -> 40
        "(let (x 5 y (* x x)) y)",              // keep both -> 25
    ] {
        assert_eq!(opt_jit::<LowBitModel>(src), walk::<LowBitModel>(src), "LowBit: {src}");
        assert_eq!(opt_jit::<HighBitModel>(src), walk::<HighBitModel>(src), "HighBit: {src}");
        assert_eq!(opt_jit::<NanBoxModel>(src), walk::<NanBoxModel>(src), "NanBox: {src}");
    }
}

/// Substituting a VARIABLE argument is unsound if the body mutates it (binding
/// captures the old value; substitution would re-read the mutated one). The
/// inliner must fall back to a `let` when the body assigns. Regression for a real
/// bug this exact case exposed.
#[test]
fn inline_respects_mutation() {
    let src = "(def go (fn () (let (y 1) (do ((fn (x) (do (set! y 99) x)) y))))) (go)";
    assert_eq!(opt_jit::<LowBitModel>(src), walk::<LowBitModel>(src), "must bind, not substitute");
    assert_eq!(opt_jit::<LowBitModel>(src), "1");
}

fn is_local(ir: &Ir) -> bool {
    matches!(ir, Ir::Local { .. })
}

/// `propagate_copies` substitutes a `let`-bound literal into the body: the
/// references become the constant, so no `Local` reads the slot afterward.
#[test]
fn propagate_replaces_literal_bindings() {
    let mut rt = Runtime::<LowBitModel>::new();
    let ir = { let s = "(let (x 3) (+ x x))"; to_ir(&mut rt, s) };
    let n = propagate_copies(&ir);
    assert_eq!(count(&n, &is_local), 0, "x replaced by the constant everywhere");
}

/// The fixpoint compounds the passes: inline exposes a constant arg, folding
/// evaluates it, propagation pushes it in, folding finishes — an opaque
/// expression collapses to its value, and it matches the interpreter.
#[test]
fn passes_compound_to_a_constant() {
    for (src, expect) in [
        ("((fn (x) (+ x x)) (+ 1 2))", "6"),
        ("(let (a 5) (let (b (* a a)) (+ a b)))", "30"),
        ("((fn (n) (* n (+ n 1))) 6)", "42"),
    ] {
        assert_eq!(opt_jit::<LowBitModel>(src), walk::<LowBitModel>(src), "{src}");
        assert_eq!(opt_jit::<LowBitModel>(src), expect, "{src}");
    }
}

/// `simplify` folds a literal `if` down to the taken branch.
#[test]
fn simplify_folds_constant_if() {
    let mut rt = Runtime::<LowBitModel>::new();
    let ir = { let s = "(if true 1 2)"; to_ir(&mut rt, s) };
    let s = simplify::<LowBitModel>(&rt, &ir);
    assert!(matches!(s, Ir::Const(_)), "constant if should fold to a constant");
    assert!(!matches!(s, Ir::If(..)));
}

/// Under NaN-boxing (no immediate fixnums) the specializer produces NO fixnum
/// ops — and the answer is still right.
#[test]
fn nanbox_gets_no_fixnum_ops() {
    let mut rt = Runtime::<NanBoxModel>::new();
    let ir = { let s = "(fn (n) (< n 2))"; to_ir(&mut rt, s) };
    let s = specialize_fixnums::<NanBoxModel>(&rt, &ir);
    assert_eq!(count(&s, &is_fx), 0);
    assert_eq!(count(&s, &is_guard), 0);
}
