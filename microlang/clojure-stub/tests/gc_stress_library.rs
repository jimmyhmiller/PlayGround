//! THE HAMMER, swung at real Clojure library code.
//!
//! `MICROLANG_GC_STRESS=1` keeps the pressure bit permanently up, so EVERY
//! safepoint runs a MINOR (each ending in `Heap::verify_no_old_to_young`, the
//! missed-barrier walk) and every 8th also runs a MAJOR (the semi-space flip +
//! card-table rebuild). With the verify heap armed, a moved object's old address
//! is poisoned — so ANY bare `u64` held across a safepoint dies loudly and
//! immediately, naming itself, instead of corrupting the heap somewhere
//! unrelated much later.
//!
//! WHY THIS SUITE EXISTS. Until the frontend rooting audit, this could not be
//! run at all: `MICROLANG_GC_STRESS=1` could not even BOOT `clojure.core` (it
//! died in `Session::new` with "fn: params must be symbols" — a relocated param
//! list), so `tests/gc_stress.rs`'s battery could only cover the toolkit's own
//! tiny s-expression programs and the Stage I2 barrier suite
//! (`gc_generational.rs`) had to fall back to a lowered nursery trigger with its
//! JIT half `#[ignore]`d. The point of closing that gap was to be able to point
//! the hammer HERE: at the whole bundled standard library.
//!
//! WHAT IT ACTUALLY COVERS — and why "boots at all" is most of the value. Merely
//! constructing a `Session` under gc-stress reads, macro-expands, compiles and
//! RUNS every bundled source: `core.clj` (the seq/HOF/collection library),
//! `cljs_types.clj` (the ported ClojureScript persistent vector/map/set + both
//! transient tiers), `host_jvm.clj`, `host_io.clj`, and the real libraries
//! `clojure.string`, `clojure.set`, `clojure.walk`, `clojure.zip`,
//! `clojure.data.json` and `clojure.test` — thousands of forms, each one a
//! macro expansion (an EVAL, i.e. a safepoint) with the compiler holding form
//! pointers across it. The battery below then exercises those libraries at
//! runtime. Both tiers, because their rooting is independent (`code.rs`'s arms
//! vs `jit_cranelift.rs`'s shims + stack maps).
//!
//! SLOW BY DESIGN (a full collection at every safepoint, over the whole
//! prelude), hence `#[ignore]`d. The gate is:
//!
//!     cargo test -p clojure-stub --features jit --release \
//!         --test gc_stress_library -- --ignored --nocapture
//!
//! `--release` is a strong recommendation, not a correctness requirement: the
//! debug build is roughly an order of magnitude slower here. Verify is armed
//! explicitly below, so it stays armed in release too.
//!
//! This lives in its OWN integration-test binary (= its own process) so the env
//! var cannot leak into the other suites, and everything runs inside ONE test fn
//! (`set_var` must happen before any `Runtime` exists and must not race another
//! thread's env read) — the same shape as `tests/gc_stress.rs`.

use microlang::{LowBitModel, Runtime, TreeWalk};

/// (name, source, expected) — real library code, exercised at runtime.
///
/// Every entry deliberately runs collection-heavy library code whose values flow
/// through lazy seqs, `apply`, transients and protocol dispatch — the paths that
/// carry heap pointers across safepoints. `apply` + a lazy tail is called out
/// because it is the shape that forces user code to run INSIDE `seq_flatten`
/// (`(apply str (map f xs))` — which is literally how `pr-str` is written).
const BATTERY: &[(&str, &str, &str)] = &[
    // clojure.string: split (the in-language regex engine) -> map -> join, i.e.
    // a lazy pipeline through `apply`.
    (
        "string-pipeline",
        r#"(require '[clojure.string :as str])
           (str/join "," (map str/upper-case (str/split "alpha,beta,gamma" #",")))"#,
        r#""ALPHA,BETA,GAMMA""#,
    ),
    (
        "string-replace-trim",
        r#"(require '[clojure.string :as str])
           (str/trim (str/replace "  a-b-c  " "-" "+"))"#,
        r#""a+b+c""#,
    ),
    // clojure.set: every fn here is variadic (`[& sets]`), so each call builds a
    // rest list and goes through `apply`/`reduce` into the transient set path.
    (
        "set-ops",
        "(require '[clojure.set :as set])
         (let [a (set (range 40)) b (set (range 20 60))]
           [(count (set/union a b)) (count (set/intersection a b)) (count (set/difference a b))])",
        "[60 20 20]",
    ),
    (
        "set-relational",
        "(require '[clojure.set :as set])
         (let [rel #{{:a 1 :b 2} {:a 3 :b 4}}]
           [(count (set/project rel [:a])) (set/subset? #{1} #{1 2})])",
        "[2 true]",
    ),
    // clojure.walk: rebuilds a whole nested structure through protocol dispatch,
    // allocating a replacement for every node.
    (
        "walk-postwalk",
        "(require '[clojure.walk :as walk])
         (walk/postwalk (fn [x] (if (number? x) (inc x) x))
                        {:a [1 2 {:b [3 4]}] :c #{5}})",
        "{:a [2 3 {:b [4 5]}], :c #{6}}",
    ),
    // clojure.data.json: a real library — reader + writer, both string-heavy.
    (
        "json-roundtrip",
        r#"(require '[clojure.data.json :as json])
           (json/read-str (json/write-str {"k" [1 2 3] "n" nil "s" "v"}))"#,
        r#"{"k" [1 2 3], "n" nil, "s" "v"}"#,
    ),
    // clojure.zip: a real library built on vectors + metadata-ish nested records.
    (
        "zip-walk",
        "(require '[clojure.zip :as zip])
         (let [z (zip/vector-zip [1 [2 3] 4])]
           (loop [loc z acc []]
             (if (zip/end? loc)
               acc
               (recur (zip/next loc)
                      (if (number? (zip/node loc)) (conj acc (zip/node loc)) acc)))))",
        "[1 2 3 4]",
    ),
    // `apply` with a LAZY tail: user code (`-pr` per element) runs INSIDE
    // `seq_flatten`. This is exactly how core's `pr-str` is defined
    // (`(apply str (interpose " " (map -pr xs)))`) and it is the shape that
    // caught the missing env publication in the Apply arms of both tiers.
    (
        "apply-lazy-tail",
        "(apply str (interpose \"-\" (map (fn [i] (str (* i i))) (range 12))))",
        r#""0-1-4-9-16-25-36-49-64-81-100-121""#,
    ),
    // pr-str over a big nested structure: the same shape, recursively.
    (
        "pr-str-nested",
        "(count (pr-str (vec (map (fn [i] {:i i :xs (vec (range (mod i 5)))}) (range 60)))))",
        "1143",
    ),
    // The collection library under sustained allocation: group-by/sort/frequencies
    // run transients, the HAMT, and comparator dispatch.
    (
        "core-collections",
        "(let [xs (range 300)]
           [(count (group-by (fn [i] (mod i 7)) xs))
            (take 3 (sort > (map (fn [i] (mod i 11)) xs)))
            (count (frequencies (map (fn [i] (mod i 13)) xs)))
            (reduce + (filter even? xs))])",
        "[7 (10 10 10) 13 22350]",
    ),
    // Long lazy pipeline: every stage allocates a thunk realized at a safepoint.
    (
        "lazy-pipeline",
        "(->> (range 400) (map inc) (filter odd?) (mapcat (fn [i] (list i i)))
              (partition 2) (map first) (take 50) (reduce +))",
        "2500",
    ),
    // try/FINALLY, both shapes. `finally` runs arbitrary (allocating) code with
    // two bare heap pointers live across it that no other root names: the body's
    // RESULT, and — when a throw is in flight — the suspended signal's payload,
    // which `take_signal` lifts out of the signal itself. Both were stale before
    // the rooting fix; each of these dies loudly without it.
    (
        "try-finally-result",
        "(try (vec (map (fn [i] {:i i}) (range 5)))
           (finally (dotimes [i 20] (vec (map (fn [j] {:j j}) (range 10))))))",
        "[{:i 0} {:i 1} {:i 2} {:i 3} {:i 4}]",
    ),
    (
        "try-finally-suspended-throw",
        r#"(try
             (try (throw (str "boom-" (apply str (map str (range 5)))))
               (finally (dotimes [i 20] (vec (map (fn [j] {:j j}) (range 10))))))
             (catch Exception e (str e)))"#,
        r#""boom-01234""#,
    ),
    // `apply`'s SHARING path. `apply` hands the applied seq to the callee
    // rather than copying it, so `plan_apply` must resolve the callee's arity
    // BEFORE the frame exists — forcing lazy nodes as it goes. Every force
    // invokes user code = a safepoint that relocates the callee, the leading
    // args and the seq cursor. Writing that passthrough introduced exactly two
    // stale-pointer bugs (a stale `id` across the arity probe, and stale
    // leading args across the split); both were invisible without this stress
    // and named themselves instantly with it. Each entry below reaches a
    // different branch: leading args (consumed and left over), lazy vs
    // realized sources, multi-arity clause selection, and apply-of-an-apply.
    (
        "apply-lazy-and-leading",
        r#"(defn v0 [& xs] (reduce + 0 xs))
           (defn v2 [a b & xs] (+ a b (reduce + 0 xs)))
           (dotimes [i 5]
             (assert (= 45 (apply v0 (map identity (range 10)))))
             (assert (= 45 (apply v2 (map identity (range 10)))))
             (assert (= 55 (apply v0 10 (map identity (range 10)))))
             (assert (= 45 (apply v0 0 1 2 (map identity (range 3 10))))))
           :ok"#,
        ":ok",
    ),
    (
        "apply-arity-select-and-nested",
        r#"(defn multi ([a] :one) ([a b] :two) ([a b & r] (count r)))
           (defn ident [& xs] xs)
           (defn v0 [& xs] (reduce + 0 xs))
           (dotimes [i 5]
             (assert (= :one (apply multi (map identity [1]))))
             (assert (= :two (apply multi (map identity [1 2]))))
             (assert (= 3 (apply multi (map identity [1 2 3 4 5]))))
             (assert (= 3 (apply multi 1 (map identity [2 3 4 5]))))
             (assert (= 0 (apply v0 [])))
             (assert (= nil (apply ident [])))
             ;; the seq an apply passes through, applied again
             (assert (= 45 (apply v0 (apply ident (map identity (range 10))))))
             (assert (= 45 (apply v0 (apply concat [(range 5) (range 5 10)])))))
           :ok"#,
        ":ok",
    ),
    // clojure.test: a real library, and the one that runs assertions through
    // macros + dynamic vars (`binding`) + try/catch.
    (
        "clojure-test",
        "(require '[clojure.test :as t])
         (t/deftest a-test (t/is (= 4 (+ 2 2))) (t/is (= [1 2] (vec (list 1 2)))))
         (let [r (with-out-str (t/run-tests 'user))] (if (re-find #\"0 failures\" r) :pass :fail))",
        ":pass",
    ),
];

fn run_battery(jit: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let tier = if jit { "JIT" } else { "TreeWalk" };
    let (mut total_minors, mut total_majors) = (0u64, 0u64);
    for (name, src, want) in BATTERY {
        let mut rt = Runtime::<LowBitModel>::new();
        assert!(
            rt.heap().stress_mode(),
            "gc-stress-library needs MICROLANG_GC_STRESS=1 set BEFORE the Runtime exists"
        );
        assert!(
            rt.heap().verify_armed(),
            "gc-stress-library needs the verify heap: poisoning the evacuated space is \
             what turns a stale pointer into a loud panic instead of silent corruption"
        );
        let got = {
            #[cfg(feature = "jit")]
            {
                if jit {
                    let backend = microlang::jit_cranelift::JitCranelift::<LowBitModel>::new();
                    let r = clojure_stub::run(&mut rt, &backend, src);
                    clojure_stub::clj_str(&rt, r)
                } else {
                    let r = clojure_stub::run(&mut rt, &TreeWalk, src);
                    clojure_stub::clj_str(&rt, r)
                }
            }
            #[cfg(not(feature = "jit"))]
            {
                let _ = jit;
                let r = clojure_stub::run(&mut rt, &TreeWalk, src);
                clojure_stub::clj_str(&rt, r)
            }
        };
        assert_eq!(got, *want, "gc-stress-library mismatch: {tier} / {name}");
        let (minors, majors) = (
            rt.heap().minor_collections.load(Relaxed),
            rt.heap().major_collections.load(Relaxed),
        );
        // The answer being right proves nothing if nothing collected: these
        // programs exist to be interrupted, and the detector only looks after a
        // minor. Booting the prelude alone is thousands of safepoints, so this
        // bound is loose on purpose — it is a wiring check, not a tuning knob.
        assert!(
            minors > 100 && majors > 0,
            "gc-stress-library {tier} / {name}: {minors} minors / {majors} majors — the \
             hammer is not swinging at both"
        );
        total_minors += minors;
        total_majors += majors;
    }
    eprintln!("gc-stress-library {tier}: {total_minors} minors / {total_majors} majors");
}

/// SLOW by design (a full collection at every safepoint, over the whole
/// prelude — minutes, not seconds). Ignored in the default run; the gate is:
/// `cargo test -p clojure-stub --features jit --release --test gc_stress_library -- --ignored`.
#[test]
#[ignore = "gc-stress hammer over the real standard library: run explicitly via \
            `cargo test -p clojure-stub --features jit --release --test gc_stress_library -- --ignored`"]
fn library_code_survives_gc_stress() {
    // MUST precede every Runtime in this process; this binary's one test is the
    // only thread alive, so the set cannot race an env read.
    std::env::set_var("MICROLANG_GC_STRESS", "1");
    std::env::set_var("MICROLANG_GC_VERIFY", "1"); // armed even in release runs

    run_battery(false);
    #[cfg(feature = "jit")]
    run_battery(true);
}
