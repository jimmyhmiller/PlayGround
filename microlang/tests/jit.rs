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

/// A program mixing every inline allocation site (cons cells, closures) plus
/// first/rest walks — the D5 fast paths and the empty-list tail normalization.
const ALLOC_HEAVY: &str = "
    (def build (fn (n acc) (if (= n 0) acc (build (- n 1) (cons n acc)))))
    (def sum (fn (xs acc) (if (nil? xs) acc (sum (rest xs) (+ acc (first xs))))))
    (def make-adder (fn (n) (fn (m) (+ n m))))
    (+ (sum (build 100 nil) 0)
       (+ ((make-adder 5) 10)
          (first (cons 7 (list)))))"; // (cons x ()) => (7): tail normalized

/// Inline allocation under GC-STRESS: closing the AllocWindow (`limit = 0`)
/// forces EVERY emitted allocation through the out-of-line shim; results are
/// identical to the open-window run. This pins the emitted `limit == 0` guard
/// (heap.rs's documented gc-stress mode).
#[test]
fn jit_inline_alloc_matches_under_gc_stress() {
    // window open (armed by Runtime::new): the inline fast path runs
    assert_eq!(jit::<LowBitModel>(ALLOC_HEAVY), "5072");
    assert_eq!(jit::<LowBitModel>(ALLOC_HEAVY), walk::<LowBitModel>(ALLOC_HEAVY));
    // window closed: same program, every site falls to the shim
    let mut rt = Runtime::<LowBitModel>::new();
    rt.heap().window.limit.store(0, std::sync::atomic::Ordering::Release);
    let cs = JitCranelift::<LowBitModel>::new();
    let r = microlang::sexpr::eval_str(&mut rt, &cs, ALLOC_HEAVY);
    assert_eq!(rt.print(r), "5072");
}

/// Assert that evaluating `src` on the JIT dies LOUDLY with `expected` in its
/// stderr. A runtime panic crosses an `extern "C"` shim frame, which cannot
/// unwind — the process aborts (that IS the loudness contract) — so the panic
/// must be observed from a subprocess, not `#[should_panic]`. The child is
/// this same test binary re-running the named test with `MICROLANG_JIT_DIE=src`.
fn assert_dies_loudly(test_name: &str, expected: &str) {
    let src = std::env::var("MICROLANG_JIT_DIE").ok();
    if let Some(src) = src {
        jit::<LowBitModel>(&src);
        unreachable!("expected a loud panic evaluating {src}");
    }
    let exe = std::env::current_exe().unwrap();
    let src = match test_name {
        "jit_aget_out_of_bounds_still_panics" => "(let (v (vector 1 2 3)) (vector-ref v 10))",
        "jit_aset_out_of_bounds_still_panics" => "(let (v (vector 1 2 3)) (vector-set! v -1 0))",
        other => panic!("assert_dies_loudly: unknown test {other}"),
    };
    let out = std::process::Command::new(exe)
        .args([test_name, "--exact", "--nocapture", "--test-threads=1"])
        .env("MICROLANG_JIT_DIE", src)
        .output()
        .expect("spawn child test process");
    assert!(!out.status.success(), "child must die on out-of-bounds access");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains(expected), "expected loud '{expected}' panic, got:\n{err}");
}

/// The inline %aget path bounds-checks against the handle's logical length and
/// routes violations to the shim — the loud out-of-range panic is preserved.
#[test]
fn jit_aget_out_of_bounds_still_panics() {
    assert_dies_loudly("jit_aget_out_of_bounds_still_panics", "out of range");
}

/// Same for %aset — and a negative index (which untags to a huge unsigned
/// value in the inline check) also lands on the shim's panic.
#[test]
fn jit_aset_out_of_bounds_still_panics() {
    assert_dies_loudly("jit_aset_out_of_bounds_still_panics", "out of range");
}

/// The emitted dispatch inline cache is epoch-checked against the relocation
/// count: an explicit (gc) MOVES the receiver and the impl, so a cached entry
/// must never be called stale — the site misses, re-resolves through the shim
/// (finding the moved impl via its root), and refills.
#[test]
fn jit_dispatch_ic_survives_moving_gc() {
    let mut rt = Runtime::<LowBitModel>::new();
    let cs = JitCranelift::<LowBitModel>::new();
    let r = microlang::sexpr::eval_str(
        &mut rt,
        &cs,
        r#"
        (defmethod area Circle (fn (s) (* (field s 0) (field s 0))))
        (def c (record 'Circle 3))
        (def go (fn (x n acc) (if (= n 0) acc (go x (- n 1) (+ acc (area x))))))
        (def a (go c 10 0)) ; hot loop: fills + hits the emitted IC
        (gc)                ; relocates c + the impl closure
        (+ a (go c 10 0))   ; must re-resolve, not call the moved-from impl
        "#,
    );
    assert_eq!(rt.print(r), "180");
    assert!(rt.relocated() > 0);
}

/// The IC epoch also folds the dispatch VERSION: redefining a method between
/// hot loops invalidates immediately (no stale impl from the cache).
#[test]
fn jit_dispatch_ic_sees_redefinition() {
    let src = r#"
        (defmethod area Circle (fn (s) 1))
        (def c (record 'Circle 3))
        (def go (fn (x n acc) (if (= n 0) acc (go x (- n 1) (+ acc (area x))))))
        (def a (go c 5 0))
        (defmethod area Circle (fn (s) 2))
        (+ a (go c 5 0))
    "#;
    assert_eq!(jit::<LowBitModel>(src), "15");
    assert_eq!(jit::<LowBitModel>(src), walk::<LowBitModel>(src));
}

// ── Stage E: allocation-driven GC with real stack maps ──────────────

/// Allocation PRESSURE collects mid-JIT-loop with NO explicit (gc): a live
/// SSA value (`x`, held across the churning call) and a live CAPTURE (`c`)
/// must survive the moving collections — the stack maps + frame walker found
/// their spill slots and the closure's self bits, or (verify heap, on in
/// debug) this dies loudly with use-after-move.
#[test]
fn jit_pressure_gc_collects_mid_loop_with_values_intact() {
    let mut rt = Runtime::<LowBitModel>::new();
    rt.heap().set_trigger_bytes(16 * 1024);
    let cs = JitCranelift::<LowBitModel>::new();
    let r = microlang::sexpr::eval_str(
        &mut rt,
        &cs,
        "(def churn (fn (n) (if (= n 0) nil (do (cons 1 2) (churn (- n 1))))))
         (def probe (fn (x) (do (churn 3000) (first x))))          ; x live across the call
         (def mk (fn (c) (fn () (do (churn 3000) (first c)))))     ; c captured
         (+ (probe (cons 42 nil)) ((mk (cons 58 nil))))",
    );
    assert_eq!(rt.print(r), "100");
    let collections = rt.heap().collections.load(std::sync::atomic::Ordering::Relaxed);
    assert!(collections > 0, "pressure never collected: the test proved nothing");
}

/// Capture reads AFTER a move: a self-tail-looping CLOSURE reads its captures
/// on every iteration while pressure collections relocate the closure object
/// out from under it (back-edge polls fire mid-loop). Every read re-derives
/// the capture base from the stack-mapped self bits — the caps_base staleness
/// class Stage E retired. Wrong plumbing = use-after-move panic (verify heap)
/// or a wrong sum.
#[test]
fn jit_capture_reads_survive_moves_mid_body() {
    let mut rt = Runtime::<LowBitModel>::new();
    rt.heap().set_trigger_bytes(16 * 1024);
    let cs = JitCranelift::<LowBitModel>::new();
    let r = microlang::sexpr::eval_str(
        &mut rt,
        &cs,
        "(def mk (fn (a b)
           (let (go nil)
             (set! go (fn (n acc)
               (if (= n 0) acc
                   (go (- n 1) (do (cons 1 2) (+ acc (+ a b)))))))
             go)))
         ((mk 3 4) 4000 0)",
    );
    assert_eq!(rt.print(r), "28000");
    let collections = rt.heap().collections.load(std::sync::atomic::Ordering::Relaxed);
    assert!(collections > 0, "no collection fired mid-loop");
}

/// THE concurrency test the pre-Stage-E model could not pass: one thread spins
/// in a pure-arithmetic NATIVE self-loop (no shims inside) while the main
/// thread runs explicit (gc)s. The back-edge polls bring the looping thread to
/// a safepoint in bounded time — the program must terminate with the right
/// answer and a moved heap. Before the polls, the collector would wait on the
/// looping thread forever.
#[test]
fn jit_native_loop_parks_for_concurrent_gc() {
    use microlang::{Repr, Val};
    let mut rt = Runtime::<LowBitModel>::new();
    let cs = JitCranelift::<LowBitModel>::new();
    microlang::sexpr::eval_str(
        &mut rt,
        &cs,
        "(def spin (fn (n acc) (if (= n 0) acc (spin (- n 1) (+ acc 1)))))",
    );
    let spin = rt.global(rt.intern("spin")).expect("spin defined");
    let mut child = rt.thread_handle();
    // Root the closure bits in the child's shadow: no collection can COMPLETE
    // before the worker parks, and parking rewrites this slot.
    let slot = child.push_root(spin);
    let worker = std::thread::spawn(move || {
        let wcs = JitCranelift::<LowBitModel>::new();
        let mut crt = child;
        let f = crt.root_get(slot);
        let n = <LowBitModel as microlang::ValueModel>::R::enc_int(30_000_000);
        let z = <LowBitModel as microlang::ValueModel>::R::enc_int(0);
        wcs.invoke(&wcs, &mut crt, f, &[n, z])
    });
    // Let the worker get deep into its native loop, then collect under it.
    std::thread::sleep(std::time::Duration::from_millis(30));
    microlang::sexpr::eval_str(&mut rt, &cs, "(gc)");
    microlang::sexpr::eval_str(&mut rt, &cs, "(gc)");
    let r = worker.join().expect("worker terminated");
    assert_eq!(rt.decode(r), Val::Int(30_000_000));
    assert!(rt.relocated() > 0, "the concurrent collections never ran");
}

/// STAGE I3: the inline `%aset`'s WRITE BARRIER, on the tier that emits it.
///
/// `%aset` is the one emitted store that can put a heap word into an object
/// that is already OLD, so it is the one place emitted code must mark a card.
/// A missing mark is not a slow store, it is a LOST old→young edge: the young
/// cons below is reachable ONLY through the promoted vector's data blob, so if
/// the card is not dirty the next minor never scans that blob, never promotes
/// the cons, and resets the nursery out from under a live reference.
///
/// Two independent things must therefore hold, and both are asserted:
///  - `verify_no_old_to_young` (the missed-barrier walk after every minor) stays
///    silent — this is what fires in debug, naming the object and the slot;
///  - the value read back afterwards is intact. This is the half that does not
///    depend on the walk: with verify off the same missing mark still dies, on
///    the nursery poison ("use-after-move: points into a collected space"),
///    because the slot now names reclaimed nursery. `churn` is what earns this
///    half — it guarantees a minor runs AFTER the last `vector-set!`, while the
///    edge is live. Without it the last store might never be collected under.
///
/// LowBit is the model under test because `INLINE_OBJECTS` gates the inline arm;
/// HighBit/NanBox route `%aset` through the barriered `arr_slice_mut` shim, and
/// the interpreter tiers cover that path in `gc_stress.rs`.
///
/// The explicit `(gc)` prim would prove NOTHING here: it is a MAJOR, which
/// traces the old gen from the roots and so finds the edge whether or not its
/// card was ever marked. Only a MINOR consults the card table, so this drives
/// minors through the ordinary pressure path (a lowered nursery trigger).
///
/// MUTATION-TESTED: deleting the `emit_card_mark` call from the inline `%aset`
/// arm makes this fail. That is also what proves it is not vacuous — it is the
/// evidence that the inline arm, not the shim, is what runs here.
#[test]
fn jit_inline_aset_marks_its_card() {
    use std::sync::atomic::Ordering::Relaxed;
    let mut rt = Runtime::<LowBitModel>::new();
    assert!(
        rt.heap().verify_armed(),
        "this test wants the missed-barrier walk armed: run in debug, or set \
         MICROLANG_GC_VERIFY=1"
    );
    // Small enough that the programs below cross it many times over: each
    // `go` iteration allocates two 24-byte conses, so 400 of them is ~19 KiB
    // against a 4 KiB trigger. (`go`'s depth is what is capped — it recurses
    // non-tail, so 400 is about what the test thread's stack takes; the
    // trigger, not the iteration count, is what buys the collections.) The
    // inline allocation window is re-gated to the trigger, so allocation still
    // runs inline right up to it.
    rt.heap().set_trigger_bytes(4 * 1024);
    let cs = JitCranelift::<LowBitModel>::new();
    // Phase 1 — build the edge. `go` recurses NON-tail on purpose: it is the
    // call boundary that reaches a safepoint. `v` outlives the early minors, so
    // it is promoted, and every `vector-set!` after that promotion is an
    // old→young store into its DATA BLOB (the words live in the blob, not the
    // handle, which is why the blob is the base the mark has to name). The
    // depth is capped at 200: 400 native frames with a collection under them
    // overflow a debug test thread, and the trigger — not the iteration count —
    // is what buys the collections anyway.
    microlang::sexpr::eval_str(
        &mut rt,
        &cs,
        "(def v (vector 0 0 0))
         (def go (fn (n) (if (= n 0) 0
                    (do (vector-set! v 1 (cons n (cons (* n 2) nil)))
                        (+ 0 (go (- n 1)))))))
         (go 200)",
    );
    // v is promoted and the edge v[1] -> young cons now exists, named by nothing
    // else. Asserted, not assumed: with no minor here, `v` would still be young
    // and the barrier would have had nothing to do.
    let after_go = rt.heap().minor_collections.load(Relaxed);
    assert!(
        after_go > 0,
        "no minor ran during `go`, so `v` was never promoted and no store into \
         it was ever old→young — this test would prove nothing"
    );
    // Phase 2 — collect under the edge. `churn` is a self-tail loop (O(1) stack,
    // polling its back-edge), so it can allocate freely; it exists only to force
    // minors AFTER the last `vector-set!`, while the edge is live. That is what
    // makes the value assertion below bite in release, where the missed-barrier
    // walk is off.
    let r = microlang::sexpr::eval_str(
        &mut rt,
        &cs,
        "(def churn (fn (n acc) (if (= n 0) acc (churn (- n 1) (cons n acc)))))
         (churn 4000 nil)
         (+ (first (vector-ref v 1)) (first (rest (vector-ref v 1))))",
    );
    assert!(
        rt.heap().minor_collections.load(Relaxed) > after_go,
        "no minor ran after the last %aset, so nothing ever had to find the \
         young value through the card table — this test proved nothing"
    );
    // `n` counts down, so the surviving store is n=1: (cons 1 (cons 2 nil)).
    assert_eq!(
        rt.print(r),
        "3",
        "the young value stored by the inline %aset did not survive the minor"
    );
}
