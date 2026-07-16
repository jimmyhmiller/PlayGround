//! Stage I2's WRITE BARRIER, hammered with the collection prims that only this
//! frontend can reach: the F3 transients, the HAMT, records, real atoms.
//!
//! Every program here deliberately builds an OLD→YOUNG edge — a container that
//! survives collections (so a minor promotes it) is then re-pointed at a value
//! allocated afterwards. That edge is reachable ONLY through the card table, so
//! a store whose write barrier is missing loses it, and the young object is
//! reclaimed under a live reference. The check is not the answer being right:
//! it is `Heap::verify_no_old_to_young`, the missed-barrier walk that runs
//! after every minor with verify armed (the debug default), which panics naming
//! the object and the slot. Each entry has been mutation-tested — deleting the
//! barrier from `values_mut`/`arr_slice_mut`/`arr_extend` makes it fire.
//!
//! THE TWO GAPS THIS SUITE WAS SHAPED AROUND ARE CLOSED. Both were rooting
//! holes reached from this frontend — bare `u64` values held across a macro
//! expansion / a lazy-seq force, which evaluate code, which reach a safepoint,
//! which collects and relocates them. Neither ever had anything to do with
//! Stage I (both reproduced at Stage I1, 779529155). With them fixed:
//!
//!  1. `MICROLANG_GC_STRESS=1` (collect at EVERY safepoint) now boots
//!     `clojure.core` and runs real library code on BOTH tiers — see
//!     `gc_stress_library.rs`, which is the hammer this suite could not swing.
//!     This suite still drives minors through the ordinary PRESSURE path (a
//!     lowered nursery trigger): deterministic, single-threaded, and every
//!     program below allocates far past the trigger.
//!  2. The JIT half below is no longer `#[ignore]`d, and the map spelling of
//!     the accumulate shape (`(reduce (fn [acc i] (assoc acc i ..)) {} ..)`) —
//!     which used to use-after-move on TreeWalk — is back in the battery.

use microlang::{LowBitModel, Runtime, TreeWalk};

/// Small enough that these programs collect tens of times, large enough that
/// compilation behaves exactly as it does under the default suite.
const NURSERY_TRIGGER: usize = 64 * 1024;

/// (name, source, expected).
const BATTERY: &[(&str, &str, &str)] = &[
    // BEAGLE'S CRASH SHAPE, verbatim: a long-lived atom swap!-ed to a freshly
    // allocated value on every iteration (its game did exactly this to
    // `world_atom`, once a frame). The atom's field is then the ONLY reference
    // to that value — no root names it.
    (
        "atom-swap",
        "(def a (atom nil))
         (dotimes [i 400] (swap! a (fn [_] {:i i :xs (list i (* i 2))})))
         (str (:i @a) \"/\" (count (:xs @a)))",
        "\"399/2\"",
    ),
    // reset! rather than swap!: the unconditional store rather than the CAS.
    (
        "atom-reset",
        "(def a (atom nil))
         (dotimes [i 400] (reset! a (vec (range (mod i 8)))))
         (str (count @a))",
        "\"7\"", // the last iteration is i=399, and (mod 399 8) = 7
    ),
    // Each new value points at BOTH young (this iteration's) and old (already
    // promoted) objects, so the promoted range's Cheney scan has to walk out of
    // a dirty card into both generations.
    (
        "atom-accumulate",
        "(def a (atom []))
         (dotimes [i 400] (swap! a conj {:i i}))
         (str (count @a) \"/\" (:i (last @a)))",
        "\"400/399\"",
    ),
    // TRANSIENTS BUILT ACROSS COLLECTIONS: the transient outlives minors, so it
    // is promoted mid-edit and every later conj! into it is an old→young store.
    // This is the F3 in-place path — it mutates the transient record AND its
    // owned 32-wide tail array in place.
    (
        "transient-vector-conj",
        "(def v (persistent! (reduce (fn [t i] (conj! t {:i i})) (transient []) (range 800))))
         (str (count v) \"/\" (:i (nth v 799)))",
        "\"800/799\"",
    ),
    // assoc! into a promoted transient's trie: the edited nodes are old, the
    // values young.
    (
        "transient-vector-assoc",
        "(def v (persistent!
                  (reduce (fn [t i] (assoc! t i (list i)))
                          (transient (vec (range 600))) (range 600))))
         (str (count v) \"/\" (first (nth v 599)))",
        "\"600/599\"",
    ),
    // pop! shrinks the owned tail in place and re-points the transient's fields.
    (
        "transient-vector-pop",
        "(def v (persistent! (reduce (fn [t _] (pop! t))
                                     (transient (vec (map list (range 600)))) (range 300))))
         (str (count v) \"/\" (first (nth v 299)))",
        "\"300/299\"",
    ),
    // The transient MAP paths: tam_* (array map — in-place pair writes) and,
    // once it outgrows 8 entries, its promotion into thm_* (the HAMT). Both
    // edit nodes in place and take young keys/values into promoted nodes.
    (
        "transient-map-assoc",
        "(def m (persistent! (reduce (fn [t i] (assoc! t (str \"k\" i) (list i i)))
                                     (transient {}) (range 600))))
         (str (count m) \"/\" (first (get m \"k599\")))",
        "\"600/599\"",
    ),
    (
        "transient-map-dissoc",
        "(def m (persistent! (reduce (fn [t i] (dissoc! t (str \"k\" i)))
                                     (transient (into {} (map (fn [i] [(str \"k\" i) (list i)])
                                                              (range 400))))
                                     (range 200))))
         (str (count m) \"/\" (first (get m \"k399\")))",
        "\"200/399\"",
    ),
    // A long-lived PERSISTENT vector taking fresh young values: the accumulator
    // is promoted, each path-copied node is young.
    (
        "persistent-vector-grow",
        "(def v (reduce (fn [acc i] (conj acc (list i))) [] (range 800)))
         (str (count v) \"/\" (first (nth v 799)))",
        "\"800/799\"",
    ),
    // The MAP spelling of the same shape — a promoted HAMT accumulator taking
    // freshly allocated young keys AND values on every step. This is the exact
    // program that use-after-moved on TreeWalk before the rooting fix (gap 2 in
    // this file's header), so it is the regression gate for it.
    (
        "persistent-map-grow",
        "(def m (reduce (fn [acc i] (assoc acc i {:v (list i)})) {} (range 600)))
         (str (count m) \"/\" (first (:v (get m 599))))",
        "\"600/599\"",
    ),
    // RECORDS holding young values, reachable only from a long-lived vector.
    (
        "records-young-fields",
        "(defrecord Box [v])
         (def boxes (reduce (fn [acc i] (conj acc (->Box (list i (* i 2))))) [] (range 600)))
         (str (count boxes) \"/\" (first (:v (nth boxes 599))))",
        "\"600/599\"",
    ),
    // The growable ARRAY handle: `arr_extend`'s in-capacity element write and
    // its grow path, which re-points a promoted handle at a fresh young blob.
    (
        "array-cell-set",
        "(def c (%cell 0))
         (dotimes [i 400] (%cell-set! c 0 (list i)))
         (str (first (%cell-ref c 0)))",
        "\"399\"",
    ),
];

fn run_battery(jit: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let tier = if jit { "JIT" } else { "TreeWalk" };
    for (name, src, want) in BATTERY {
        let mut rt = Runtime::<LowBitModel>::new();
        assert!(
            rt.heap().verify_armed(),
            "gc-generational needs the verify heap: the missed-barrier walk IS the \
             assertion here. Run this suite in debug, or set MICROLANG_GC_VERIFY=1."
        );
        rt.heap().set_trigger_bytes(NURSERY_TRIGGER);
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
        assert_eq!(got, *want, "gc-generational mismatch: {tier} / {name}");
        // The answer being right proves nothing if no minor ran: these programs
        // exist to be interrupted by one, and the walk only looks after a minor.
        assert!(
            rt.heap().minor_collections.load(Relaxed) > 0,
            "gc-generational {tier} / {name}: no minor ran, so the missed-barrier \
             detector never looked — this entry proved nothing"
        );
    }
}

#[test]
fn old_to_young_edges_survive_minors_treewalk() {
    run_battery(false);
}

/// The same battery on the JIT tier. This was `#[ignore]`d for gap 2 in this
/// file's header — the frontend use-after-moved on the JIT for EVERY program
/// here, under a low nursery trigger, all the way back to Stage I1 (779529155).
/// That gap is closed, so the hammer swings: this now runs for real.
#[cfg(feature = "jit")]
#[test]
fn old_to_young_edges_survive_minors_jit() {
    run_battery(true);
}
