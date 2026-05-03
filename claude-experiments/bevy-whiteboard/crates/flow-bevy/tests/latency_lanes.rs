//! Smoke test for `examples/latency_lanes.whiteboard`.
//!
//! Five Client → Worker pairs at 1ms, 20ms, 100ms, 500ms, 2s wires.
//! Verify each lane is wired correctly and runs cleanly long enough
//! for at least one round-trip on every lane.
//!
//! Round-trip ≈ 2 × edge_latency + Worker.service_ns (30ms). The
//! glacial lane is 4.03s. Run for 6s so even that lane completes one.
//!
//! Per-lane Client.period_ns is tuned to (round_trip + buffer), so
//! we expect close to (run_time / period) round-trips per lane.

use std::path::PathBuf;

use flow::{Sim, Value};
use flow_bevy::canvas::load_canvas;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().to_path_buf()
}

fn slot_i(sim: &Sim, name: &str, slot: &str) -> i64 {
    let n = sim.nodes.values().find(|n| n.name == name)
        .unwrap_or_else(|| panic!("no node named `{}`", name));
    match n.slots.get(slot) {
        Some(Value::Int(i)) => *i,
        other => panic!("{}.{}: expected Int, got {:?}", name, slot, other),
    }
}

#[test]
fn latency_lanes_loads_and_each_lane_completes() {
    let path = project_root().join("examples/latency_lanes.whiteboard");
    let mut canvas = load_canvas(&path, 11).expect("load latency_lanes.whiteboard");

    canvas.sim.run_until(canvas.sim.now_ns + 6_000_000_000);

    // No unexpected runtime errors.
    let errs: Vec<_> = canvas.sim.error_counts.iter()
        .filter(|(_, c)| **c > 0)
        .collect();
    assert!(errs.is_empty(), "unexpected runtime errors: {:?}", errs);

    // Every lane completed at least one round-trip in 6s.
    for client in [
        "ClientFast", "ClientQuick", "ClientMid", "ClientSlow", "ClientGlacial",
    ] {
        let completed = slot_i(&canvas.sim, client, "completed");
        assert!(completed >= 1,
            "{}.completed should be ≥1 in 6s, got {}", client, completed);
    }

    // Fast lane should clearly out-pace glacial.
    let fast    = slot_i(&canvas.sim, "ClientFast",    "completed");
    let glacial = slot_i(&canvas.sim, "ClientGlacial", "completed");
    assert!(fast > glacial * 5,
        "ClientFast.completed ({}) should massively outpace ClientGlacial.completed ({})",
        fast, glacial);
}
