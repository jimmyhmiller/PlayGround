//! The mini-Clojure frontend running WHOLLY on the native JIT tier (no fallback).
//! Exercises exactly the machinery the JIT used to bail out on: protocol dispatch,
//! `deftype` methods, `.-field` access, `apply`, and try/catch/finally — plus the
//! ported persistent collections that are built out of all of them.
//!
//! Run with: `cargo test -p clojure-stub --features jit --test jit`.
#![cfg(feature = "jit")]

use microlang::jit_cranelift::JitCranelift;
use microlang::{LowBitModel, Runtime};

/// Run a program on the JIT backend and format the result.
fn jit(src: &str) -> String {
    let mut rt = Runtime::<LowBitModel>::new();
    let backend = JitCranelift::<LowBitModel>::new();
    let r = clojure_stub::run(&mut rt, &backend, src);
    clojure_stub::clj_str(&rt, r)
}

#[test]
fn arithmetic_and_fns() {
    assert_eq!(jit("(defn fact [n] (if (< n 2) 1 (* n (fact (- n 1))))) (fact 10)"), "3628800");
    assert_eq!(jit("(reduce + 0 (range 100))"), "4950");
    assert_eq!(jit("(map inc [1 2 3])"), "(2 3 4)");
}

/// A `try` arm must be Cranelift'd ONCE, not on every execution.
///
/// `shim_try` ran its arms through `top.eval_ir`, which deliberately does not
/// cache (top-level `Ir`s are transient). Try arms are not transient — the emit
/// site keeps them alive as long as the code — so every `try` re-ran the whole
/// compiler, regalloc included: 74.7µs per execution against 24ns for the same
/// expression without the `try`. core.match compiles pattern backtracking into
/// try/catch, which made it ~5900x JVM Clojure and effectively unusable.
///
/// A correctness suite cannot see this: the answers were always right. So the
/// assertion is on `compiled_bodies()` — the compile-once counter — and on the
/// property that actually matters: it must not SCALE with how many times a
/// `try` runs. (Asserting an absolute count would just encode today's prelude;
/// running the same program at two very different iteration counts isolates the
/// per-execution term, which is the bug.)
fn compiles_for(iterations: usize) -> u32 {
    let mut rt = Runtime::<LowBitModel>::new();
    let backend = JitCranelift::<LowBitModel>::new();
    let src = format!(
        "(defn f [x] (try (+ x 1) (catch Exception e 0)))
         (defn g [x] (try (+ x 1) (finally nil)))
         (f 1) (g 1)
         (dotimes [i {iterations}] (f i) (g i))"
    );
    clojure_stub::run(&mut rt, &backend, &src);
    // total_compiles, NOT compiled_bodies: the latter is cache.len(), and the
    // bug was a body recompiled WITHOUT ever being cached — invisible to it.
    // (This test passed against the un-fixed compiler until it counted the
    // right thing.)
    backend.total_compiles()
}

#[test]
fn try_arms_compile_once_not_per_execution() {
    let few = compiles_for(10);
    let many = compiles_for(2000);
    assert_eq!(
        few, many,
        "running try/catch+finally 2000x compiled {} more bodies than running it 10x \
         — the try arms are being recompiled per execution",
        many.saturating_sub(few)
    );
    // and it still means the same thing
    assert_eq!(jit("(try (+ 1 1) (catch Exception e :caught))"), "2");
    assert_eq!(jit("(try (throw \"boom\") (catch Exception e :caught))"), ":caught");
    assert_eq!(jit("(let [a (atom 0)] (try 1 (finally (reset! a 9))) @a)"), "9");
    assert_eq!(
        jit("(try (try (throw \"x\") (finally 1)) (catch Exception e :outer))"),
        ":outer"
    );
}

/// DISTINCT `try` forms must not answer for each other.
///
/// The cache above was keyed by the arm's `Ir` ADDRESS. An `Ir` tree is dropped
/// when its form finishes, and the allocator hands the same address to the next
/// tree — so a later, unrelated `try` hit the earlier one's compiled code. It is
/// a silent wrong ANSWER, which is why the "compile once" test above sailed past
/// it: three sequential `(try N (finally nil))` returned 1, 2, 2, and
/// `(vector (try 1 (finally nil)))` after any other try returned `[nil]`.
/// `Ir::Try` now carries a process-unique `site`, and ids are never recycled.
///
/// Every case here is one program, because the bug needs one try's Ir to be
/// FREED before the next is built — a fresh runtime per expression cannot see it.
#[test]
fn distinct_try_forms_do_not_share_compiled_code() {
    // Each try is its OWN top-level form: that is what lets the first tree be
    // dropped before the next is built, so the addresses collide. Written as one
    // form (a single vector) all three Ir trees are alive at once, get distinct
    // addresses, and the bug hides — this test passed against the unsound key
    // until the tries were split apart like this.
    assert_eq!(
        jit("(def r1 (try 1 (finally nil)))
             (def r2 (try 2 (finally nil)))
             (def r3 (try 3 (finally nil)))
             [r1 r2 r3]"),
        "[1 2 3]"
    );
    // The original repro: a try at top level, then the same shape as an ARGUMENT.
    assert_eq!(
        jit("(def a (try 1 (finally nil)))
             (def b (vector (try 1 (finally nil))))
             [a b]"),
        "[1 [1]]"
    );
    // Distinct catch arms must not swap either.
    assert_eq!(
        jit("(def c1 (try (throw \"x\") (catch Exception e :a)))
             (def c2 (try (throw \"y\") (catch Exception e :b)))
             [c1 c2]"),
        "[:a :b]"
    );
    // `binding` desugars to try/finally, so this is where it surfaced: nested
    // bindings returned [:seq nil :vector] instead of [:seq :vector :seq].
    assert_eq!(
        jit("(def ^:dynamic *v* nil) (defn rd [] *v*) \
             (binding [*v* :seq] [(rd) (binding [*v* :vector] (rd)) (rd)])"),
        "[:seq :vector :seq]"
    );
    // ...and the outer binding must still unwind to the root.
    assert_eq!(
        jit("(def ^:dynamic *w* :root) (defn rd2 [] *w*) \
             (binding [*w* 1] (binding [*w* 2] (rd2))) (rd2)"),
        ":root"
    );
}

/// The INLINED bitwise fast path (`emit_guarded_arith`), which `even?`/`odd?`
/// ride once per element in any filtered pipeline. It untags to raw i64, does
/// the op, and retags with NO range check — sound only because the fixnum range
/// is closed under and/or/xor. These are the cases where an untag/retag or
/// sign-handling slip would hide: negatives (two's complement), both fixnum
/// boundaries, and operands that are NOT fixnums (a promoted bigint), which
/// must fall through the guard to the runtime and still answer correctly.
#[test]
fn bitwise_fast_path_on_jit() {
    // two's complement: the sign bit participates
    assert_eq!(jit("(bit-and -3 1)"), "1");
    assert_eq!(jit("(bit-or -5 3)"), "-5");
    assert_eq!(jit("(bit-xor -7 2)"), "-5");
    assert_eq!(jit("(bit-and -1 -1)"), "-1");
    assert_eq!(jit("(bit-xor -1 0)"), "-1");
    assert_eq!(jit("[(bit-and 12 10) (bit-or 12 10) (bit-xor 12 10)]"), "[8 14 6]");
    // the fixnum boundaries stay in range (no retag overflow)
    assert_eq!(jit("(bit-and 4611686018427387903 -1)"), "4611686018427387903");
    assert_eq!(jit("(bit-and -4611686018427387904 -1)"), "-4611686018427387904");
    // a promoted bigint is not a fixnum: the guard must route it to the runtime
    assert_eq!(jit("(bit-and 10000000000000000000001N 1)"), "1");
    assert_eq!(jit("(even? 10000000000000000000000N)"), "true");
    // even?/odd? are `(zero? (bit-and n 1))`, and agree on negatives
    assert_eq!(jit("[(even? -4) (odd? -3) (even? 0) (odd? 7)]"), "[true true true true]");
    // and a non-integer is a CATCHABLE throw, not a process panic
    assert_eq!(jit("(try (even? 1.5) (catch Exception e :caught))"), ":caught");
}

#[test]
fn persistent_vector_on_jit() {
    assert_eq!(jit("(conj [1 2] 3)"), "[1 2 3]");
    assert_eq!(jit("(assoc [1 2 3] 1 :x)"), "[1 :x 3]");
    assert_eq!(jit("(nth (vec (range 2000)) 1999)"), "1999");
    assert_eq!(jit("(count (vec (range 1000)))"), "1000");
    assert_eq!(jit("(peek (pop [1 2 3]))"), "2");
    assert_eq!(jit("(reduce + (vec (range 100)))"), "4950");
}

#[test]
fn maps_and_sets_on_jit() {
    assert_eq!(jit("(get {:a 1 :b 2} :b)"), "2");
    assert_eq!(jit("(type-of (into {} (map (fn [i] [i i]) (range 20))))"), "PersistentHashMap");
    assert_eq!(jit("(count (into {} (map (fn [i] [i (* i i)]) (range 200))))"), "200");
    assert_eq!(jit("(get (into {} (map (fn [i] [i (* i i)]) (range 200))) 137)"), "18769");
    assert_eq!(jit("(contains? #{1 2 3} 2)"), "true");
    assert_eq!(jit("(count (into #{} [1 1 2 2 3]))"), "3");
    assert_eq!(jit("(count (set (range 100)))"), "100");
}

#[test]
fn protocols_and_deftype_on_jit() {
    // deftype + inline protocol methods + .-field, all JIT-native.
    let pair = "(deftype Pair [a b]) \
                (extend-type Pair ISeqable (-seq [p] (list (field p 0) (field p 1))) \
                                  ICounted (-count [p] 2))";
    assert_eq!(jit(&format!("{pair} (seq (->Pair 10 20))")), "(10 20)");
    assert_eq!(jit(&format!("{pair} (map inc (->Pair 3 4))")), "(4 5)");
    // a user protocol dispatched on the JIT
    assert_eq!(
        jit("(defprotocol Shape (area [s])) (deftype Sq [side]) \
             (extend-type Sq Shape (area [s] (* (field s 0) (field s 0)))) (area (->Sq 5))"),
        "25"
    );
}

#[test]
fn multimethods_on_jit() {
    assert_eq!(
        jit("(defmulti f (fn [x] (mod x 2))) (defmethod f 0 [x] :even) (defmethod f 1 [x] :odd) [(f 4) (f 7)]"),
        "[:even :odd]"
    );
}

#[test]
fn apply_on_jit() {
    assert_eq!(jit("(apply + [1 2 3 4])"), "10");
    assert_eq!(jit("(apply + 1 2 [3 4 5])"), "15");
    assert_eq!(jit("(apply max [3 7 2 9 4])"), "9");
    assert_eq!(jit("(apply str (interpose \",\" [1 2 3]))"), "\"1,2,3\"");
}

#[test]
fn try_catch_on_jit() {
    assert_eq!(jit("(try (throw \"boom\") (catch :default e e))"), "\"boom\"");
    assert_eq!(jit("(try (+ 1 2) (catch :default e :caught))"), "3");
    assert_eq!(jit("(try (nth [1 2] 10) (catch :default e :oob))"), ":oob");
    assert_eq!(
        jit("(let [a (atom 0)] (try (throw \"x\") (catch :default e (reset! a 99)) (finally (swap! a inc))) @a)"),
        "100"
    );
    // a throw raised and caught across a JIT-compiled higher-order call
    assert_eq!(
        jit("(reduce + (map (fn [x] (try (if (even? x) x (throw \"odd\")) (catch :default e 0))) (range 10)))"),
        "20"
    );
}

#[test]
fn dynamic_vars_on_jit() {
    // `^:dynamic` + `binding` compiled: override, dynamic scope through a call,
    // set!, unwind-on-throw, nesting — all via the %dyn-* prims on the JIT.
    assert_eq!(jit("(def ^:dynamic *x* 10) (binding [*x* 20] *x*)"), "20");
    assert_eq!(jit("(def ^:dynamic *x* 1) (defn gx [] *x*) (binding [*x* 99] (gx))"), "99");
    assert_eq!(jit("(def ^:dynamic *x* 0) (binding [*x* 5] (set! *x* 7) *x*)"), "7");
    assert_eq!(jit("(def ^:dynamic *x* 1) (try (binding [*x* 9] (throw \"b\")) (catch :default e *x*))"), "1");
    assert_eq!(jit("(def ^:dynamic *x* 1) [(binding [*x* 2] (binding [*x* 3] *x*)) *x*]"), "[3 1]");
    // thread-local across a JIT worker thread (its binding stack starts empty).
    assert_eq!(jit("(def ^:dynamic *x* 0) (binding [*x* 5] [*x* @(future *x*)])"), "[5 0]");
}

#[test]
fn threads_on_jit() {
    // `future` spawns an OS thread that runs its closure on ITS OWN native JIT,
    // sharing the heap/atoms; `deref`/`@` joins it (parking during the wait).
    assert_eq!(jit("@(future (+ 1 2))"), "3");
    // atomic swap! across JIT worker threads over the shared atom
    assert_eq!(jit("(let [a (atom 0)] (dotimes [_ 5] @(future (swap! a inc))) @a)"), "5");
    // fan out N futures and combine — all workers native
    assert_eq!(jit("(reduce + (map deref (mapv (fn [i] (future (* i i))) (range 8))))"), "140");
    assert_eq!(
        jit("(reduce + (map deref (mapv (fn [i] (future (reduce + (range i)))) (range 10))))"),
        "120"
    );
    // a future whose body itself dispatches through the collection protocols
    assert_eq!(jit("@(future (count (into {} (map (fn [i] [i i]) (range 30)))))"), "30");
}

#[test]
fn repeated_await_inside_let_is_stable() {
    // Regression: `%await`/`(gc)` shims READ `ctx.cur`, but a body containing them
    // was eligible for the caller-built-frame fast path, which leaves `ctx.cur`
    // uninitialized (stack garbage). First/fresh-stack call happened to work;
    // RE-ENTERING a `let` scope that awaits (batch 2+) read a stale heap-id as the
    // frame pointer -> `Arc::clone` fault. Fixed by forcing `needs_cur` for
    // Await/Gc bodies (they take the shim path, so `run_once` inits `cur`).
    // Each of these crashed with SIGSEGV before the fix.
    assert_eq!(
        jit("(defn trial [] (let [c 0] @(future 42) 99)) \
             (loop [i 0 ok 0] (if (= i 5) ok (recur (inc i) (do (trial) (inc ok)))))"),
        "5"
    );
    // heavy: 25 batches of 8 workers CAS-hammering a shared atom, awaited in a let
    assert_eq!(
        jit("(def counter (atom 0)) \
             (defn trial [] (reset! counter 0) \
               (let [fs (mapv (fn [_] (future (dotimes [_ 300] (swap! counter inc)))) (range 8))] \
                 (doseq [f fs] (deref f)) (= (deref counter) 2400))) \
             (loop [i 0 ok 0] (if (= i 25) ok (recur (inc i) (if (trial) (inc ok) ok))))"),
        "25"
    );
}

#[test]
fn lazy_seqs_on_jit() {
    assert_eq!(jit("(take 5 (map (fn [x] (* x x)) (range)))"), "(0 1 4 9 16)");
    assert_eq!(jit("(reduce + (take 10 (iterate inc 1)))"), "55");
    assert_eq!(jit("(for [x (range 4) :when (even? x)] (* x 10))"), "(0 20)");
}

#[test]
fn collection_literals_are_data_on_jit() {
    assert_eq!(
        jit("(defmacro vq [v] (list 'quote (list (vector? v) (type-of v)))) (vq [1 2])"),
        "(true PersistentVector)"
    );
    assert_eq!(jit("(= {:a 1} '{:a 1})"), "true");
    assert_eq!(jit("(read-string \"[1 {:a #{2}}]\")"), "[1 {:a #{2}}]");
}

#[test]
fn real_core_match_on_jit() {
    // The real clojure/core.match, end to end on the native tier (tail-called
    // callable deftype records exercise the trampoline's apply-handler hook).
    let scratch = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/core.match");
    let run = |src: &str| {
        let mut rt = Runtime::<LowBitModel>::new();
        let backend = JitCranelift::<LowBitModel>::new();
        let full = format!("(require '[clojure.core.match :refer [match]])\n{src}");
        let r = clojure_stub::run_with_paths(&mut rt, &backend, &full, vec![scratch.clone()]);
        clojure_stub::clj_str(&rt, r)
    };
    assert_eq!(
        run("[(let [x 1] (match [x] [1] :a :else :b)) \
             (let [v [1 2 3]] (match v [_ _ 2] :a0 [1 1 3] :a1 [1 2 3] :a2 :else :a3)) \
             (let [x {:a 1 :b 1}] (match [x] [{:a _ :b 2}] :a0 [{:a 1 :b 1}] :a1 :else :a2))]"),
        "[:a :a2 :a1]"
    );
}
