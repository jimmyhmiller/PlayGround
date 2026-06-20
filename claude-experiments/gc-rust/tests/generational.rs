//! Verifies the generational collector is actually engaged: an allocation-heavy
//! program must trigger MINOR collections (the young-generation scavenge), and
//! produce the correct result — proving the nursery + write barrier + promotion
//! path all work end to end. We drive the real `gcr` binary with
//! `GCR_GC_STATS=1` and parse the collection counts.

use std::process::Command;

fn gcr_bin() -> &'static str {
    env!("CARGO_BIN_EXE_gcr")
}

/// Run `example` and return (stdout result line, "M minor + N major" stats).
fn run_with_stats(example: &str) -> (String, usize, usize) {
    let out = Command::new(gcr_bin())
        .args(["run", example])
        .env("GCR_GC_STATS", "1")
        .output()
        .expect("run gcr");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let result = stdout.lines().last().unwrap_or("").trim().to_string();
    // stderr: "gc-rust: <m> minor + <n> major collections"
    let stats = stderr
        .lines()
        .find(|l| l.contains("minor") && l.contains("major"))
        .unwrap_or("");
    let nums: Vec<usize> = stats
        .split_whitespace()
        .filter_map(|w| w.parse::<usize>().ok())
        .collect();
    let (minor, major) = match nums.as_slice() {
        [m, n, ..] => (*m, *n),
        _ => (0, 0),
    };
    (result, minor, major)
}

#[test]
fn binary_trees_triggers_minor_collections() {
    // binary_trees allocates millions of nodes: the nursery fills repeatedly, so
    // minor GCs must fire (and the result must still be correct).
    let (result, minor, major) = run_with_stats("examples/binary_trees.gcr");
    assert_eq!(result, "5242840", "wrong result under generational GC");
    assert!(minor > 0, "expected minor collections, got {} minor / {} major", minor, major);
}

#[test]
fn interior_ref_write_barrier_keeps_young_pointee_alive() {
    // Regression for the interior-reference write barrier. `gc_interior_barrier.gcr`
    // promotes a Holder to tenured, then mutates its flattened value-with-reference
    // field (`FieldLoc::ValueAt`) to point at a freshly-allocated YOUNG Box, then
    // drives minor GCs. The barrier must mark the Holder's card so the minor GC
    // scans the embedded reference; otherwise the young Box — reachable only via the
    // tenured Holder — is reclaimed and the read returns garbage. With the barrier
    // it reads 42; without it (verified by temporarily disabling the barrier) it
    // returned 2796199. The run must also actually collect (minor > 0), else the
    // test would pass vacuously.
    let (result, minor, _major) = run_with_stats("examples/gc_interior_barrier.gcr");
    assert_eq!(result, "42", "young pointee reclaimed — interior write barrier missing");
    assert!(minor > 0, "test must trigger minor GCs to exercise the barrier (got {minor})");
}

#[test]
fn correct_under_generational_collection() {
    // A spread of examples must all produce identical results to the semi-space
    // collector while running on the generational heap.
    for (ex, want) in [
        ("examples/stdlib.gcr", "414"),
        ("examples/strings.gcr", "35"),
        ("examples/vec_prelude.gcr", "285"),
        ("examples/nbody.gcr", "921463"),
    ] {
        let (result, _, _) = run_with_stats(ex);
        assert_eq!(result, want, "{} gave wrong result", ex);
    }
}
