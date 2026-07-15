//! Stage E gc-stress: `MICROLANG_GC_STRESS=1` keeps the pressure bit
//! permanently up, so EVERY safepoint runs a full moving collection with the
//! verify heap armed (debug default) — the bug hammer for the rooting
//! discipline of every tier, the JIT stack maps, and the native frame walker.
//!
//! This lives in its OWN integration-test binary (= its own process), so the
//! env var cannot leak into the other suites. Everything runs inside ONE test
//! fn: `set_var` must happen before any `Runtime` exists and must not race
//! other threads' env reads.

use microlang::{CodeSpace, HighBitModel, LowBitModel, NanBoxModel, Runtime, TreeWalk, ValueModel};

/// (name, source, expected) — a battery covering the axes that carry heap
/// pointers across safepoints: closures + captures, lists, vectors, dispatch,
/// arithmetic promotion, try/catch, deep and mutual tail recursion.
const BATTERY: &[(&str, &str, &str)] = &[
    ("arith", "(+ (* 2 3) (* 4 5))", "26"),
    ("bignum", "(* 100000000000 100000000000)", "10000000000000000000000"),
    ("fact", "(def f (fn (n) (if (< n 2) 1 (* n (f (- n 1)))))) (f 10)", "3628800"),
    ("fib", "(def fib (fn (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))) (fib 10)", "55"),
    (
        "hof-capture",
        "(def make-adder (fn (n) (fn (m) (+ n m))))
         (def add5 (make-adder 5))
         (def add10 (make-adder 10))
         (+ (add5 100) (add10 200))",
        "315",
    ),
    (
        "lists",
        "(def build (fn (n acc) (if (= n 0) acc (build (- n 1) (cons n acc)))))
         (def sum (fn (xs acc) (if (nil? xs) acc (sum (rest xs) (+ acc (first xs))))))
         (sum (build 40 nil) 0)",
        "820",
    ),
    (
        "map-hof",
        "(def m (fn (g xs) (if (nil? xs) xs (cons (g (first xs)) (m g (rest xs))))))
         (def inc (fn (n) (+ n 1)))
         (first (rest (m inc (list 10 20 30))))",
        "21",
    ),
    (
        "vectors",
        "(let (v (vector 1 2 3))
           (vector-set! v 1 99)
           (+ (vector-ref v 1) (+ (vector-ref v 2) (vector-length v))))",
        "105",
    ),
    (
        "dispatch",
        "(defmethod area Circle (fn (s) (* (field s 0) (field s 0))))
         (defmethod area Square (fn (s) (+ (field s 0) (field s 0))))
         (def total (fn (xs) (if (nil? xs) 0 (+ (area (first xs)) (total (rest xs))))))
         (total (list (record 'Circle 3) (record 'Square 4) (record 'Circle 5)))",
        "42",
    ),
    (
        "tail-loop",
        "(def go (fn (n acc) (if (= n 0) acc (go (- n 1) (+ acc n))))) (go 100 0)",
        "5050",
    ),
    (
        "mutual-tail",
        "(def a (fn (n) (if (= n 0) 100 (b (- n 1)))))
         (def b (fn (n) (if (= n 0) 200 (a (- n 1)))))
         (a 41)",
        "200",
    ),
    (
        "letrec-closures",
        "(def go (fn ()
           (let (even? nil odd? nil)
             (set! even? (fn (n) (if (= n 0) true (odd? (- n 1)))))
             (set! odd? (fn (n) (if (= n 0) false (even? (- n 1)))))
             (even? 10))))
         (go)",
        "true",
    ),
    (
        "chars",
        "(char->integer (integer->char 65))",
        "65",
    ),
];

fn run_battery<M: ValueModel>(mk: &dyn Fn() -> Box<dyn CodeSpace<M>>, tier: &str, skip: &[&str]) {
    let mut total_collections = 0u64;
    for (name, src, want) in BATTERY {
        if skip.contains(name) {
            continue; // a capability the tier genuinely lacks (it panics loudly)
        }
        let mut rt = Runtime::<M>::new();
        let cs = mk();
        let r = microlang::sexpr::eval_str(&mut rt, cs.as_ref(), src);
        assert_eq!(
            rt.print(r),
            *want,
            "gc-stress mismatch: {tier} / {name}"
        );
        total_collections += rt.heap().collections.load(std::sync::atomic::Ordering::Relaxed);
    }
    // Pure-prim programs have no safepoints, but across the battery the
    // hammer must actually have been swinging — hundreds of collections.
    assert!(
        total_collections > 100,
        "gc-stress ran only {total_collections} collections across the {tier} battery"
    );
}

/// SLOW by design (~2 min: a full moving collection at every safepoint — on
/// the CEK that is every step). Ignored in the default run; the gate is:
/// `cargo test --features jit --test gc_stress -- --ignored`.
#[test]
#[ignore = "gc-stress hammer: run explicitly via `cargo test --features jit --test gc_stress -- --ignored`"]
fn gc_stress_battery_across_tiers_and_models() {
    // MUST precede every Runtime in this process; this binary's one test is
    // the only thread alive, so the set cannot race an env read.
    std::env::set_var("MICROLANG_GC_STRESS", "1");
    std::env::set_var("MICROLANG_GC_VERIFY", "1"); // armed even in release runs

    // The CEK genuinely lacks record dispatch (loud panic; TreeWalk covers it).
    run_battery::<LowBitModel>(&|| Box::new(TreeWalk), "TreeWalk/LowBit", &[]);
    run_battery::<HighBitModel>(&|| Box::new(TreeWalk), "TreeWalk/HighBit", &[]);
    run_battery::<NanBoxModel>(&|| Box::new(TreeWalk), "TreeWalk/NanBox", &[]);
    run_battery::<LowBitModel>(&|| Box::new(microlang::CekMachine), "CEK/LowBit", &["dispatch"]);
    run_battery::<NanBoxModel>(&|| Box::new(microlang::CekMachine), "CEK/NanBox", &["dispatch"]);

    #[cfg(feature = "jit")]
    {
        use microlang::{JitCranelift, Tiered};
        run_battery::<LowBitModel>(&|| Box::new(JitCranelift::<LowBitModel>::new()), "JIT/LowBit", &[]);
        run_battery::<HighBitModel>(&|| Box::new(JitCranelift::<HighBitModel>::new()), "JIT/HighBit", &[]);
        run_battery::<NanBoxModel>(&|| Box::new(JitCranelift::<NanBoxModel>::new()), "JIT/NanBox", &[]);
        run_battery::<LowBitModel>(&|| Box::new(Tiered::<LowBitModel>::new()), "Tiered/LowBit", &[]);
    }
}
