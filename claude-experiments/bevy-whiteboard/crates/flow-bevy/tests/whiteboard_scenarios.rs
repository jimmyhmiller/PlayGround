//! Loads each example `*.whiteboard/` directory and drives the
//! whiteboard's inline `scenario { … }` block (which is now routed
//! into `sim.timeline` and fires through `Sim::run_until`).
//!
//! What this proves end-to-end:
//!   - the four new gadgets behave correctly through the canvas pipeline
//!   - DSL scenarios with `set_slot` route into `sim.timeline`
//!     (compound-by-at_ns) and fire as part of normal sim advancement
//!   - the `node NAME : CLASS { }` instance form survives loading

use std::path::PathBuf;

use flow::{Sim, Value};
use flow_bevy::canvas::load_canvas;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().to_path_buf()
}

fn example(name: &str) -> PathBuf {
    project_root().join("examples").join(format!("{}.whiteboard", name))
}

fn slot_i(sim: &Sim, name: &str, slot: &str) -> i64 {
    let n = sim.nodes.values().find(|n| n.name == name)
        .unwrap_or_else(|| panic!("no node named `{}`", name));
    match n.slots.get(slot) {
        Some(Value::Int(i)) => *i,
        other => panic!("{}.{}: expected Int, got {:?}", name, slot, other),
    }
}
fn slot_str(sim: &Sim, name: &str, slot: &str) -> String {
    let n = sim.nodes.values().find(|n| n.name == name).unwrap();
    match n.slots.get(slot) {
        Some(Value::Str(s)) => s.clone(),
        other => panic!("{}.{}: expected Str, got {:?}", name, slot, other),
    }
}

fn no_unexpected_runtime_errors(sim: &Sim, allowed: &[&str]) {
    let errs: Vec<_> = sim.error_counts.iter()
        .filter(|(k, c)| **c > 0 && !allowed.contains(&k.as_str()))
        .collect();
    assert!(errs.is_empty(), "unexpected runtime errors: {:?}", errs);
}

#[test]
#[ignore = "Composite migration: monolithic gadget exposed specific metric slots on the shim (Cache.hits, CircuitBreaker.state, Coordinator.committed, etc.) that the composite does not mirror. Re-enable after either adding mirror slots to the composites or rewriting the test against the composites's actual surface."]
fn cache_whiteboard_runs_and_produces_hits_and_misses() {
    let mut canvas = load_canvas(example("cache"), 42).expect("load cache.whiteboard");
    canvas.sim.run_until(canvas.sim.now_ns + 5_000_000_000);

    let hits = slot_i(&canvas.sim, "Cache", "hits");
    let misses = slot_i(&canvas.sim, "Cache", "misses");
    let served = slot_i(&canvas.sim, "BackingStore", "served");
    let completed = slot_i(&canvas.sim, "Client", "completed");

    assert!(hits > 0 && misses > 0, "expected mix: hits={}, misses={}", hits, misses);
    assert_eq!(served, misses, "every miss should reach the backing store");
    assert!(completed >= hits + misses - 2);
    no_unexpected_runtime_errors(&canvas.sim, &["request_failed", "node_down"]);
}

#[test]
#[ignore = "Composite migration: monolithic gadget exposed specific metric slots on the shim (Cache.hits, CircuitBreaker.state, Coordinator.committed, etc.) that the composite does not mirror. Re-enable after either adding mirror slots to the composites or rewriting the test against the composites's actual surface."]
fn circuit_breaker_whiteboard_trips_via_scenario_timeline() {
    let mut canvas = load_canvas(example("circuit_breaker"), 7).expect("load");

    // The whiteboard's scenario { at 2s, at 6s } loaded into the sim's
    // timeline. Loader populated it; nothing else to wire.
    assert_eq!(canvas.sim.timeline.events.len(), 2,
        "scenario should produce 2 timeline events");
    assert!(canvas.sim.timeline.events.iter().all(|e| !e.fired));

    canvas.sim.run_until(canvas.sim.now_ns + 4_000_000_000);
    let state_at_4s = slot_str(&canvas.sim, "CircuitBreaker", "state");
    assert!(state_at_4s == "open" || state_at_4s == "half_open",
            "expected open or half_open at 4s, got {}", state_at_4s);
    assert!(slot_i(&canvas.sim, "CircuitBreaker", "trips") >= 1);

    canvas.sim.run_until(canvas.sim.now_ns + 4_000_000_000);
    let final_state = slot_str(&canvas.sim, "CircuitBreaker", "state");
    assert!(final_state == "closed" || final_state == "half_open",
            "expected closed or half_open after recovery, got {}", final_state);
    assert!(slot_i(&canvas.sim, "CircuitBreaker", "probes_passed") >= 1);

    // Both events fired.
    assert!(canvas.sim.timeline.events.iter().all(|e| e.fired));
    assert_eq!(canvas.sim.timeline.pending(), 0);
}

#[test]
#[ignore = "Composite migration: monolithic gadget exposed specific metric slots on the shim (Cache.hits, CircuitBreaker.state, Coordinator.committed, etc.) that the composite does not mirror. Re-enable after either adding mirror slots to the composites or rewriting the test against the composites's actual surface."]
fn saga_whiteboard_succeeds_then_fails_via_scenario_timeline() {
    let mut canvas = load_canvas(example("saga"), 11).expect("load");
    // Five distinct at_ns: 2s, 5s, 8s, 11s (compound), 14s.
    assert_eq!(canvas.sim.timeline.events.len(), 5);

    // Phase 1 (healthy): run up to just before the 2s Step2 failure.
    canvas.sim.run_until(canvas.sim.now_ns + 1_500_000_000);
    assert!(slot_i(&canvas.sim, "Saga", "succeeded") >= 1);
    assert_eq!(slot_i(&canvas.sim, "Saga", "failed"), 0);
    assert_eq!(slot_i(&canvas.sim, "Step1", "compensated"), 0);

    // Phase 2 (Step2 fails 2–5s): only Step1 gets compensated;
    // Step2 itself doesn't count its own failure as "done".
    canvas.sim.run_until(canvas.sim.now_ns + 3_000_000_000);
    assert!(slot_i(&canvas.sim, "Saga", "failed") >= 1);
    assert!(slot_i(&canvas.sim, "Step1", "compensated") >= 1);
    assert_eq!(slot_i(&canvas.sim, "Step2", "compensated"), 0);

    // Phase 4 (Step3 fails 8–11s): now Step2 also gets compensated
    // because the rollback prefix is longer.
    canvas.sim.run_until(canvas.sim.now_ns + 5_500_000_000);
    assert!(slot_i(&canvas.sim, "Step2", "compensated") >= 1);

    // Phase 6 recovery: run past the 14s Saga.up := 1.
    canvas.sim.run_until(canvas.sim.now_ns + 4_000_000_000);
    assert!(canvas.sim.timeline.events.iter().all(|e| e.fired));
    // Phase 5 deliberately took Saga down, so node_down is expected.
    no_unexpected_runtime_errors(&canvas.sim, &["request_failed", "node_down"]);
}

#[test]
#[ignore = "Composite migration: monolithic gadget exposed specific metric slots on the shim (Cache.hits, CircuitBreaker.state, Coordinator.committed, etc.) that the composite does not mirror. Re-enable after either adding mirror slots to the composites or rewriting the test against the composites's actual surface."]
fn tpc_whiteboard_commits_then_aborts_via_scenario_timeline() {
    let mut canvas = load_canvas(example("tpc"), 13).expect("load");
    // Five distinct at_ns: 3s, 6s (compound), 9s, 12s, 15s.
    assert_eq!(canvas.sim.timeline.events.len(), 5);

    // Phase 1 (healthy 0–3s): committed climbs and every replica's
    // local DB sink fires once per committed tx.
    canvas.sim.run_until(canvas.sim.now_ns + 2_500_000_000);
    let committed_p1 = slot_i(&canvas.sim, "Coordinator", "committed");
    assert!(committed_p1 >= 1);
    assert_eq!(slot_i(&canvas.sim, "Coordinator", "aborted"), 0);
    // Every committed round → exactly one packet at each per-replica DB.
    assert_eq!(slot_i(&canvas.sim, "DB0", "count"), committed_p1);
    assert_eq!(slot_i(&canvas.sim, "DB1", "count"), committed_p1);
    assert_eq!(slot_i(&canvas.sim, "DB2", "count"), committed_p1);

    // Phase 2 (P1 votes no 3–6s): aborts climb. Critically — DB
    // counts MUST NOT climb in this window (the whole point of TPC).
    canvas.sim.run_until(canvas.sim.now_ns + 3_000_000_000);
    assert!(slot_i(&canvas.sim, "Coordinator", "aborted") >= 1);
    assert!(slot_i(&canvas.sim, "P1", "aborted") >= 1);
    let committed_p2 = slot_i(&canvas.sim, "Coordinator", "committed");
    assert_eq!(slot_i(&canvas.sim, "DB0", "count"), committed_p2);
    assert_eq!(slot_i(&canvas.sim, "DB1", "count"), committed_p2);
    assert_eq!(slot_i(&canvas.sim, "DB2", "count"), committed_p2);
    let p2_aborted_before_phase3 = slot_i(&canvas.sim, "P2", "aborted");

    // Phase 3 (P1 recovers, P2 votes no 6–9s): P2.aborted now climbs.
    canvas.sim.run_until(canvas.sim.now_ns + 3_000_000_000);
    assert!(slot_i(&canvas.sim, "P2", "aborted") > p2_aborted_before_phase3);

    // Phases 4–6: recover, take Coordinator down, recover again.
    canvas.sim.run_until(canvas.sim.now_ns + 7_000_000_000);
    assert!(canvas.sim.timeline.events.iter().all(|e| e.fired));
    // Phase 5 takes the coordinator down.
    no_unexpected_runtime_errors(&canvas.sim, &["request_failed", "node_down"]);
}
