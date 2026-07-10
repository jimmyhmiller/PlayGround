//! The native (Cranelift) emit tier locked to the same behavior as the
//! interpreter tiers. Compiled to nothing unless `--features jit` is on.
#![cfg(feature = "jit")]

use microlang::{
    BytecodeVm, CodeSpace, HighBitModel, JitCranelift, LowBitModel, ModelArithJit, NanBoxModel, Runtime,
    Traced, TreeWalk, ValueModel,
};

fn jit<M: ModelArithJit>(src: &str) -> String {
    let mut rt = Runtime::<M>::new();
    let cs = JitCranelift::<M>::new();
    let r = microlang::sexpr::eval_str(&mut rt, &cs, src);
    rt.print(r)
}

fn walk<M: ValueModel>(src: &str) -> String {
    let mut rt = Runtime::<M>::new();
    let r = microlang::sexpr::eval_str(&mut rt, &TreeWalk, src);
    rt.print(r)
}

/// Evaluate on some backend, capturing a panic as `"PANIC"` so two backends can
/// be compared even where a shared emit recipe is known to be lossy.
fn try_eval<M: ModelArithJit>(make: impl Fn() -> Box<dyn CodeSpace<M>>, src: &str) -> String {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut rt = Runtime::<M>::new();
        let v = microlang::sexpr::eval_str(&mut rt, make().as_ref(), src);
        rt.print(v)
    }))
    .unwrap_or_else(|_| "PANIC".into());
    std::panic::set_hook(prev);
    r
}

/// The native tier agrees with the INTERPRETER (the correct reference) under
/// every value model — now across the full numeric tower, not just the fixnum
/// fast path: negatives, overflow→bignum, and huge products all match, because
/// the JIT's guarded arithmetic falls back to the runtime's promoting `prim`.
#[test]
fn jit_matches_treewalk_across_models() {
    for src in [
        "(+ (* 2 3) (* 4 5))",
        "(def fact (fn (n) (if (< n 2) 1 (* n (fact (- n 1)))))) (fact 6)",
        "(def fib (fn (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))) (fib 10)",
        "(if (< 1 2) 100 200)",
        "(if false 1 (if true 7 8))",
        "(- 3 10)",                              // negative result
        "(* 100000000000 100000000000)",         // overflows i64 -> BigInt
        "(+ 4611686018427387904 4611686018427387904)", // crosses the fixnum edge
    ] {
        assert_eq!(jit::<LowBitModel>(src), walk::<LowBitModel>(src), "LowBit: {src}");
        assert_eq!(jit::<HighBitModel>(src), walk::<HighBitModel>(src), "HighBit: {src}");
        assert_eq!(jit::<NanBoxModel>(src), walk::<NanBoxModel>(src), "NanBox: {src}");
    }
}

/// The guarded fast path makes the JIT strictly MORE correct than its sibling
/// bytecode emit tier. Both share the fixnum fast path, but the bytecode recipe
/// wraps / corrupts on the boundary while the JIT range-checks and promotes:
///   * `(* 10^11 10^11)` overflows i64 — bytecode wraps to garbage, the JIT
///     promotes to the exact BigInt the tree-walker computes;
///   * HighBit `(- 3 10)` is negative — the raw bytecode recipe fills the top
///     tag bits and PANICS, while the JIT's retag re-masks and yields `-7`.
/// So the JIT matches the interpreter exactly where the bytecode tier cannot.
#[test]
fn jit_guarded_arith_beats_the_bytecode_recipe() {
    // overflow → bignum: JIT == tree-walker, bytecode wraps to something else
    let big = "(* 100000000000 100000000000)";
    assert_eq!(jit::<LowBitModel>(big), walk::<LowBitModel>(big));
    assert_eq!(jit::<LowBitModel>(big), "10000000000000000000000");
    let bc_big = try_eval::<LowBitModel>(|| Box::new(BytecodeVm::<LowBitModel>::new()), big);
    assert_ne!(bc_big, "10000000000000000000000"); // the wrapping recipe is wrong here

    // HighBit negative: JIT now matches the interpreter; the bytecode recipe panics
    let neg = "(- 3 10)";
    assert_eq!(jit::<HighBitModel>(neg), walk::<HighBitModel>(neg));
    assert_eq!(jit::<HighBitModel>(neg), "-7");
    let bc_neg = try_eval::<HighBitModel>(|| Box::new(BytecodeVm::<HighBitModel>::new()), neg);
    assert_eq!(bc_neg, "PANIC");
}

/// The value axis is genuinely free for the JIT: three representations, one
/// source, one answer.
#[test]
fn arithmetic_is_model_independent_on_jit() {
    let src = "(def f (fn (n) (if (< n 2) 1 (* n (f (- n 1)))))) (f 8)";
    assert_eq!(jit::<LowBitModel>(src), "40320");
    assert_eq!(jit::<HighBitModel>(src), "40320");
    assert_eq!(jit::<NanBoxModel>(src), "40320");
}

/// Higher-order functions: a closure passed and applied, compiled to native code.
#[test]
fn jit_closures_and_higher_order() {
    let src = "(def apply2 (fn (f x) (f (f x))))
               (def inc (fn (n) (+ n 1)))
               (apply2 inc 10)";
    assert_eq!(jit::<LowBitModel>(src), "12");
}

/// Non-arithmetic prims escape to the runtime (the native `Slow` path) and still
/// produce correct heap values.
#[test]
fn jit_runtime_prim_escape() {
    assert_eq!(jit::<LowBitModel>("(first (rest (list 1 2 3)))"), "2");
    assert_eq!(jit::<LowBitModel>("(cons 1 (cons 2 nil))"), "(1 2)");
    assert_eq!(jit::<LowBitModel>("(nil? (rest (list 1)))"), "true");
}

/// `def` evaluates to the defined symbol, matching the tree-walker.
#[test]
fn jit_def_returns_symbol() {
    assert_eq!(jit::<LowBitModel>("(def x 5)"), "x");
    assert_eq!(walk::<LowBitModel>("(def x 5)"), "x");
}

/// Proper tail calls: a self-tail-recursive loop of a million iterations runs in
/// O(1) native stack (the trampoline reuses the frame). Without TCO this would
/// overflow the stack — this is what lets ALL of Scheme's iteration run natively.
#[test]
fn jit_proper_tail_calls_dont_overflow() {
    let src = "(def loop (fn (n) (if (= n 0) 42 (loop (- n 1))))) (loop 1000000)";
    assert_eq!(jit::<LowBitModel>(src), "42");
    // an accumulator loop, likewise tail-recursive
    let sum = "(def go (fn (n acc) (if (= n 0) acc (go (- n 1) (+ acc n))))) (go 100000 0)";
    assert_eq!(jit::<LowBitModel>(sum), walk::<LowBitModel>(sum));
}

/// Frame-pool soundness: a captured frame (a live closure's environment) must
/// NEVER be recycled. We make closures that close over their frame, then run
/// heavy recursion that pools and reuses many frames, then re-invoke the captured
/// closures — they must still see their original captured values. If the pool
/// recycled a captured frame, `add5`/`add10` would return corrupted results.
#[test]
fn jit_frame_pool_never_recycles_a_captured_frame() {
    let src = "
        (def make-adder (fn (n) (fn (m) (+ n m))))
        (def add5 (make-adder 5))
        (def add10 (make-adder 10))
        (def fib (fn (k) (if (< k 2) k (+ (fib (- k 1)) (fib (- k 2))))))
        (fib 18)
        (+ (add5 100) (add10 200))";
    // add5 closes over n=5, add10 over n=10; (add5 100)=105, (add10 200)=210 -> 315.
    // The `n` reference inside the returned lambda is an `up==1` local read of the
    // captured frame — proving both the pool guard and the parent-chain path.
    assert_eq!(jit::<LowBitModel>(src), "315");
    assert_eq!(jit::<LowBitModel>(src), walk::<LowBitModel>(src));
}

/// Native self-tail recursion: a self-tail-call loops in place (no stack growth),
/// and the inner non-tail calls compile to direct native calls. tak/ack shaped.
#[test]
fn jit_self_tail_and_nested_calls_native() {
    // tak: outer call is a self-tail (loops), the three inner calls are non-tail.
    let tak = "(def tak (fn (x y z)
                 (if (< y x)
                     (tak (tak (- x 1) y z) (tak (- y 1) z x) (tak (- z 1) x y))
                     z)))
               (tak 18 12 6)";
    assert_eq!(jit::<LowBitModel>(tak), walk::<LowBitModel>(tak));
    // a big self-tail loop stays O(1) stack.
    let loop_ = "(def go (fn (n acc) (if (= n 0) acc (go (- n 1) (+ acc 1))))) (go 5000000 0)";
    assert_eq!(jit::<LowBitModel>(loop_), "5000000");
}

/// Non-self (mutual) tail recursion between two non-escaping functions must still
/// be O(1) stack: a non-self tail flows through `top` to the trampoline, which
/// loops. This proves the native fast path did NOT regress mutual TCO.
#[test]
fn jit_mutual_tail_recursion_is_o1_stack() {
    let src = "(def a (fn (n) (if (= n 0) 100 (b (- n 1)))))
               (def b (fn (n) (if (= n 0) 200 (a (- n 1)))))
               (a 2000000)";
    // a(2000000): even steps down -> ends at a(0) = 100.
    assert_eq!(jit::<LowBitModel>(src), "100");
    assert_eq!(jit::<LowBitModel>(src), walk::<LowBitModel>(src));
}

/// Composition holds: wrapping the JIT in `Traced` observes EVERY call, because
/// the native `shim_call` recurses through `top` (open recursion), not `self`.
/// `(fact 5)` performs 5 invocations — the same count the tree-walker produces.
#[test]
fn jit_composes_under_traced() {
    let mut rt = Runtime::<LowBitModel>::new();
    let traced = Traced::new(JitCranelift::<LowBitModel>::new());
    let r = microlang::sexpr::eval_str(&mut rt, &traced, "(def fact (fn (n) (if (< n 2) 1 (* n (fact (- n 1)))))) (fact 5)");
    assert_eq!(rt.print(r), "120");
    assert_eq!(traced.invoke_count(), 5);
}

/// `let` (sequential, core semantics) and local/global `set!` compile natively.
#[test]
fn jit_let_and_set() {
    // sequential let: later inits see earlier bindings
    assert_eq!(jit::<LowBitModel>("(let (x 2 y (* x 3)) (+ x y))"), "8");
    // set! on a local mutates the binding
    assert_eq!(jit::<LowBitModel>("(let (x 1) (set! x 10) x)"), "10");
    // letrec-style mutual recursion (what Scheme desugars to) via let + set!
    let src = "(def go (fn ()
                 (let (even? nil odd? nil)
                   (set! even? (fn (n) (if (= n 0) true (odd? (- n 1)))))
                   (set! odd? (fn (n) (if (= n 0) false (even? (- n 1)))))
                   (even? 10))))
               (go)";
    assert_eq!(jit::<LowBitModel>(src), "true");
    // set! on a global
    assert_eq!(jit::<LowBitModel>("(def g 1) (set! g 42) g"), "42");
}

/// Constructs the JIT genuinely cannot compile (continuations) still fail loudly
/// with a clear message pointing at the CEK tier — never a silent wrong answer.
#[test]
#[should_panic(expected = "JIT tier")]
fn jit_rejects_callcc_clearly() {
    let mut rt = Runtime::<LowBitModel>::new();
    let cs = JitCranelift::<LowBitModel>::new();
    microlang::sexpr::eval_str(&mut rt, &cs, "(%callcc (fn (k) 1))");
}
