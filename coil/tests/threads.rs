//! Concurrency dogfood (examples/threads.coil): N threads × M atomic increments on a
//! shared counter → exactly N·M, race-free. Atomics are a PURE LIBRARY (lib/atomic.coil
//! over the llvm-ir escape hatch); threads are a thin pthread wrapper (lib/thread.coil)
//! using the C-interop callback pattern. Tests the ATOMIC version (deterministic N·M =
//! CI-testable); the non-atomic race is demonstrative-only — a data race is
//! non-deterministic, so asserting "< N·M" would be flaky and is NOT done.

mod common;
use common::build_and_run;

#[test]
fn n_threads_atomic_increment_is_race_free() {
    let src = std::fs::read_to_string("examples/threads.coil").expect("read threads.coil");
    // 0 == the shared counter reached EXACTLY 4 × 100000 (no lost updates) under real
    // thread contention — the atomic library genuinely synchronizes.
    assert_eq!(build_and_run(&src), 0);
}

#[test]
fn generic_pointer_atomics_load_store_cas() {
    // lib/atomic.coil's pointer atomics (generic over T via LLVM opaque pointers) — the
    // atomics lock-free POINTER structures want, with a real-pointer data model (no
    // address-as-i64). Comparison still goes via cast-to-i64 (Coil's icmp is integer-
    // only — a tiny separate ergonomic gap, not a missing capability).
    let src = "(module app)\n\
        (import \"lib/atomic.coil\" :use *)\n\
        (defstruct Node [(val i64) (next (ptr Node))])\n\
        (defn main [] (-> i64)\n\
          (let [cell (alloc-stack (ptr Node)) n (alloc-stack Node)]\n\
            (store! cell (cast (ptr Node) 0))\n\
            (store! (field n val) 7)\n\
            (let [old (atomic-cas-ptr [Node] cell (cast (ptr Node) 0) n)]\n\
              (if (icmp-eq (cast i64 old) 0)                       ; cas succeeded (was null)\n\
                  (load (field (atomic-load-ptr [Node] cell) val)) ; head now n -> 7\n\
                  99))))";
    assert_eq!(build_and_run(src), 7);
}

#[test]
fn lock_free_stack_loses_no_pushes_under_contention() {
    // Stage 2: a genuinely lock-free (Treiber) stack via atomic-cas — NO mutex. 4
    // threads concurrently push 1000 disjoint pre-allocated nodes each (contending on
    // the head via CAS); draining must find all 4000. ABA is sidestepped by
    // construction (nodes never freed/reused; push-only-then-drain) — see the example.
    let src = std::fs::read_to_string("examples/lockfree.coil").expect("read lockfree.coil");
    assert_eq!(build_and_run(&src), 0);
}
