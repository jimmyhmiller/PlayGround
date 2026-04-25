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
fn saga_whiteboard_succeeds_then_fails_via_scenario_timeline() {
    let mut canvas = load_canvas(example("saga"), 11).expect("load");
    assert_eq!(canvas.sim.timeline.events.len(), 1);

    canvas.sim.run_until(canvas.sim.now_ns + 2_500_000_000);
    let succeeded_before = slot_i(&canvas.sim, "Saga", "succeeded");
    assert!(succeeded_before >= 1);
    assert_eq!(slot_i(&canvas.sim, "Saga", "failed"), 0);
    assert_eq!(slot_i(&canvas.sim, "Step1", "compensated"), 0);

    canvas.sim.run_until(canvas.sim.now_ns + 5_500_000_000);
    assert!(slot_i(&canvas.sim, "Saga", "failed") >= 1);
    assert!(slot_i(&canvas.sim, "Step1", "compensated") >= 1);
    no_unexpected_runtime_errors(&canvas.sim, &["request_failed", "node_down"]);
}

#[test]
fn tpc_whiteboard_commits_then_aborts_via_scenario_timeline() {
    let mut canvas = load_canvas(example("tpc"), 13).expect("load");
    assert_eq!(canvas.sim.timeline.events.len(), 1);

    canvas.sim.run_until(canvas.sim.now_ns + 3_500_000_000);
    assert!(slot_i(&canvas.sim, "Coordinator", "committed") >= 1);
    assert_eq!(slot_i(&canvas.sim, "Coordinator", "aborted"), 0);

    canvas.sim.run_until(canvas.sim.now_ns + 5_500_000_000);
    assert!(slot_i(&canvas.sim, "Coordinator", "aborted") >= 1);
    assert!(slot_i(&canvas.sim, "P1", "aborted") >= 1);
    assert!(slot_i(&canvas.sim, "P0", "aborted") >= 1);
    no_unexpected_runtime_errors(&canvas.sim, &["request_failed", "node_down"]);
}
