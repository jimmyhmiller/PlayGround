//! Stage E gc-stress: `MICROLANG_GC_STRESS=1` keeps the pressure bit
//! permanently up, so EVERY safepoint collects with the verify heap armed
//! (debug default) — the bug hammer for the rooting discipline of every tier,
//! the JIT stack maps, and the native frame walker.
//!
//! Stage I2 sharpened it into the WRITE BARRIER's hammer as well: every
//! safepoint now runs a MINOR, and every minor ends in the missed-barrier walk
//! (`Heap::verify_no_old_to_young`) — a walk of the whole old gen asserting no
//! old object still points into the nursery a minor just evacuated. So a store
//! that forgot its barrier dies HERE, naming the object and the slot, instead
//! of silently losing an edge and corrupting the heap somewhere unrelated
//! hours later (which is beagle's shipped bug, hunted through a crash in
//! another subsystem entirely). Every 8th collection also runs a major, so the
//! old gen's Cheney, the flip, and the card table's rebuild stay covered.
//!
//! This lives in its OWN integration-test binary (= its own process), so the
//! env var cannot leak into the other suites. Everything runs inside ONE test
//! fn: `set_var` must happen before any `Runtime` exists and must not race
//! other threads' env reads.

use microlang::{
    BytecodeVm, ClosureComp, CodeSpace, HighBitModel, LowBitModel, NanBoxModel, Runtime, TreeWalk,
    ValueModel,
};

/// (name, source, expected) — a battery covering the axes that carry heap
/// pointers across safepoints: closures + captures, lists, vectors, dispatch,
/// arithmetic promotion, try/catch, deep and mutual tail recursion.
///
/// The `oldyoung-*` entries are Stage I2's: each one deliberately builds an
/// OLD→YOUNG edge — a long-lived container that outlives collections (so it is
/// promoted) being re-pointed at a value allocated after that promotion. Those
/// edges are reachable ONLY through the card table, so each entry fails (loudly
/// — the detector, or a wrong answer) if its store's barrier is missing. They
/// recurse non-tail on purpose: a self-tail loop is one interpreter frame with
/// no call boundary, so it reaches no safepoint and would collect nothing.
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
    // BEAGLE'S EXACT CRASH SHAPE: a long-lived atom `swap!`-ed to a freshly
    // allocated value on every iteration. After the first collection the atom
    // is old and the cons is young, and the atom's field is the ONLY reference
    // to it — no root names it. Reading the value back afterwards is the whole
    // point: an unbarriered store leaves the atom pointing at recycled nursery.
    (
        "oldyoung-atom",
        "(def a (%atom-new nil))
         (def go (fn (n) (if (= n 0) 0
                    (do (%atom-set a (cons n (cons (* n 2) nil)))
                        (+ 0 (go (- n 1)))))))
         (go 60)
         (+ (first (%atom-get a)) (first (rest (%atom-get a))))",
        "3",
    ),
    // The same edge through a CAS rather than a store (`swap!` is built on it).
    (
        "oldyoung-atom-cas",
        "(def a (%atom-new nil))
         (def go (fn (n) (if (= n 0) 0
                    (do (%atom-cas a (%atom-get a) (cons n nil))
                        (+ 0 (go (- n 1)))))))
         (go 60)
         (first (%atom-get a))",
        "1",
    ),
    // A long-lived VECTOR (so: a promoted ARRAY handle over a promoted DATA
    // blob) whose elements are re-pointed at young conses. The write lands in
    // the BLOB, not the handle, which is exactly what the barrier has to mark.
    (
        "oldyoung-vector",
        "(def v (vector 0 0 0))
         (def go (fn (n) (if (= n 0) 0
                    (do (vector-set! v 1 (cons n nil))
                        (vector-set! v 2 (cons (* n 2) nil))
                        (+ 0 (go (- n 1)))))))
         (go 60)
         (+ (first (vector-ref v 1)) (first (vector-ref v 2)))",
        "3",
    ),
    // A long-lived vector holding RECORDS built after it was promoted — the
    // young target is a record whose own fields point at more young objects, so
    // the card-scan has to promote TRANSITIVELY out of the dirty card.
    (
        "oldyoung-records",
        "(def v (vector 0 0))
         (def go (fn (n) (if (= n 0) 0
                    (do (vector-set! v 0 (record 'Box (cons n nil)))
                        (+ 0 (go (- n 1)))))))
         (go 60)
         (first (field (vector-ref v 0) 0))",
        "1",
    ),
    // The edge pointing the OTHER way through a chain: a long-lived vector slot
    // re-pointed at a young cons whose tail is the PREVIOUS (already promoted)
    // value. Mixes an old target and a young target under one dirty card.
    (
        "oldyoung-accumulate",
        "(def v (vector nil))
         (def go (fn (n) (if (= n 0) 0
                    (do (vector-set! v 0 (cons n (vector-ref v 0)))
                        (+ 0 (go (- n 1)))))))
         (def len (fn (xs acc) (if (nil? xs) acc (len (rest xs) (+ acc 1)))))
         (go 60)
         (+ (first (vector-ref v 0)) (len (vector-ref v 0) 0))",
        "61",
    ),
];

fn run_battery<M: ValueModel>(mk: &dyn Fn() -> Box<dyn CodeSpace<M>>, tier: &str, skip: &[&str]) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut total_collections = 0u64;
    let mut total_minors = 0u64;
    let mut total_majors = 0u64;
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
        total_collections += rt.heap().collections.load(Relaxed);
        total_minors += rt.heap().minor_collections.load(Relaxed);
        total_majors += rt.heap().major_collections.load(Relaxed);
        // Per program, not just in aggregate: an `oldyoung-*` entry that
        // reached no safepoint would return the right answer having proved
        // nothing at all about the barrier, and would do it silently.
        if name.starts_with("oldyoung-") {
            assert!(
                rt.heap().minor_collections.load(Relaxed) > 0,
                "gc-stress {tier} / {name}: no minor ran, so the missed-barrier \
                 detector never looked — this entry proved nothing"
            );
        }
    }
    // Pure-prim programs have no safepoints, but across the battery the
    // hammer must actually have been swinging — hundreds of collections.
    assert!(
        total_collections > 100,
        "gc-stress ran only {total_collections} collections across the {tier} battery"
    );
    // Both halves of the collector, not just whichever one the policy happened
    // to pick: the minors are what run the missed-barrier walk, the majors are
    // what flip the semi-spaces and rebuild the card table's start index.
    assert!(
        total_minors > 100 && total_majors > 0,
        "gc-stress {tier}: {total_minors} minors / {total_majors} majors — the \
         hammer is not swinging at both"
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

    // The EMIT tier (bytecode VM) and the closure-compiler tier: same rooting
    // hammer. The VM's operand `Vec<u64>` is invisible to the collector, so its
    // live entries must be published across every call safepoint; the compiler
    // builds `argv`/`callee` bare across arg evaluation. Both reach the pressure
    // safepoint at their call boundary now (they polled none before).
    //
    // The bytecode tier genuinely lacks record dispatch and field access
    // (DefMethod/Dispatch/FieldGet panic loudly; TreeWalk/ClosureComp cover
    // them), so every model skips the two dispatch/field entries.
    //
    // `bignum` additionally skips on the LowBit/HighBit models: their
    // model-EMITTED multiply is a raw `MulRaw` on the tagged bits (the whole
    // point of the capstone — arithmetic that differs by representation), which
    // wraps at 64 bits with NO overflow-to-BigInt promotion. That promotion
    // lives only on the `Slow` → `rt.prim` path, which is exactly what the
    // NanBox model emits for arithmetic — so Bytecode/NanBox keeps `bignum`.
    // (bignum still runs on TreeWalk/CEK/JIT/ClosureComp and Bytecode/NanBox.)
    let bc_raw_skip = &["bignum", "dispatch", "oldyoung-records"];
    let bc_nanbox_skip = &["dispatch", "oldyoung-records"];
    run_battery::<LowBitModel>(&|| Box::new(BytecodeVm::<LowBitModel>::new()), "Bytecode/LowBit", bc_raw_skip);
    run_battery::<HighBitModel>(&|| Box::new(BytecodeVm::<HighBitModel>::new()), "Bytecode/HighBit", bc_raw_skip);
    run_battery::<NanBoxModel>(&|| Box::new(BytecodeVm::<NanBoxModel>::new()), "Bytecode/NanBox", bc_nanbox_skip);
    run_battery::<LowBitModel>(&|| Box::new(ClosureComp::<LowBitModel>::new()), "ClosureComp/LowBit", &[]);
    run_battery::<HighBitModel>(&|| Box::new(ClosureComp::<HighBitModel>::new()), "ClosureComp/HighBit", &[]);
    run_battery::<NanBoxModel>(&|| Box::new(ClosureComp::<NanBoxModel>::new()), "ClosureComp/NanBox", &[]);

    #[cfg(feature = "jit")]
    {
        use microlang::{JitCranelift, Tiered};
        run_battery::<LowBitModel>(&|| Box::new(JitCranelift::<LowBitModel>::new()), "JIT/LowBit", &[]);
        run_battery::<HighBitModel>(&|| Box::new(JitCranelift::<HighBitModel>::new()), "JIT/HighBit", &[]);
        run_battery::<NanBoxModel>(&|| Box::new(JitCranelift::<NanBoxModel>::new()), "JIT/NanBox", &[]);
        run_battery::<LowBitModel>(&|| Box::new(Tiered::<LowBitModel>::new()), "Tiered/LowBit", &[]);
    }
}
