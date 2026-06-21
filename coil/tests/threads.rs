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
