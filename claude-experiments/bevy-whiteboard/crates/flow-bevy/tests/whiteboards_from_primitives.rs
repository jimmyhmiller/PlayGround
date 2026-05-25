//! Validates that whiteboards built entirely from primitive gadgets
//! produce behaviorally correct simulations. Each test loads a
//! `*.whiteboard/` directory shipped under `examples/` and asserts on
//! the simulator's observable state after running it.

use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // crates/flow-bevy → workspace root.
    manifest_dir.parent().unwrap().parent().unwrap().to_path_buf()
}

fn node_id(sim: &flow::sim::Sim, name: &str) -> flow::sim::NodeId {
    *sim.nodes
        .iter()
        .find(|(_, n)| n.name == name)
        .map(|(id, _)| id)
        .unwrap_or_else(|| panic!("no node `{}`", name))
}

fn slot_i(sim: &flow::sim::Sim, name: &str, slot: &str) -> i64 {
    let id = node_id(sim, name);
    match sim.nodes[&id].slots.get(slot) {
        Some(flow::value::Value::Int(n)) => *n,
        other => panic!("slot {}.{} = {:?}", name, slot, other),
    }
}

#[test]
fn pipeline_from_primitives_serves_both_colors() {
    let path = workspace_root().join("examples/pipeline_from_primitives.whiteboard");
    let canvas = flow_bevy::canvas::load_canvas(&path, 0)
        .expect("whiteboard loads cleanly");
    let mut sim = canvas.sim;

    // 5 seconds of run: source emits ~50 packets, ~25 red and ~25 blue
    // (alternating), each lane drains via its own puller.
    sim.run_until(5_000_000_000);

    let red  = slot_i(&sim, "RedDone", "count");
    let blue = slot_i(&sim, "BlueDone", "count");

    // Both lanes should make meaningful progress.
    assert!(red >= 5, "RedDone served {} (expected ≥5)", red);
    assert!(blue >= 5, "BlueDone served {} (expected ≥5)", blue);
    assert!(
        sim.error_counts.is_empty(),
        "unexpected runtime errors: {:?}",
        sim.error_counts
    );
    eprintln!(
        "pipeline-from-primitives: RedDone={}, BlueDone={}",
        red, blue
    );
}

#[test]
fn circuit_breaker_from_primitives_loads_and_trips() {
    let path = workspace_root().join("examples/circuit_breaker_from_primitives.whiteboard");
    let canvas = flow_bevy::canvas::load_canvas(&path, 0)
        .expect("whiteboard loads cleanly");
    let mut sim = canvas.sim;

    // Run for 4 seconds — enough to see the breaker trip at least once
    // and observe diverted reqs piling up at ErrSink.
    sim.run_until(4_000_000_000);

    let served = slot_i(&sim, "Server", "served");
    let err    = slot_i(&sim, "ErrSink", "count");

    // ReqSource emits ~20 reqs over 4s (period 200ms). Some get
    // through to Server (failing on each), enough trip the breaker;
    // the rest divert to ErrSink. Recovery flips it back periodically
    // so Server.served will keep growing too.
    assert!(served >= 3, "Server served {} reqs (expected ≥3)", served);
    assert!(err >= 3, "ErrSink absorbed {} diverts (expected ≥3)", err);
    assert!(
        sim.error_counts.is_empty(),
        "unexpected runtime errors: {:?}",
        sim.error_counts
    );
    eprintln!(
        "CB-from-primitives whiteboard: served={}, diverted={}",
        served, err
    );
}
