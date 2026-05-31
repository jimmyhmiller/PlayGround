//! Diagnostic: measure the on-wire animation window (sim_latency × k)
//! for INTERIOR composite edges vs EXTERIOR (user-drawn) edges, under
//! the live replay scale. Interior edges are authored at latency `1`
//! (one nanosecond); exterior edges at 1_000_000 ns (1 ms). The
//! question: does the visual system give interior edges a visible
//! window, or do they animate in ~0 time?

mod common;

use common::{advance_sim_ns, spawn_node, wire};
use flow_bevy::compound::CompoundMembership;
use flow_bevy::edges::VisualTimelineRes;
use flow_bevy::gadgets::Kind;

#[test]
fn measure_interior_vs_exterior_window() {
    let mut app = common::make_app();

    // Live visual scale.
    let k = 410.0;
    app.world_mut()
        .resource_mut::<VisualTimelineRes>()
        .strategy
        .as_replay_mut()
        .k = k;

    let generator = spawn_node(&mut app, Kind::Generator, 0, "Gen_test");
    let queue = spawn_node(&mut app, Kind::Queue, 0, "Queue_test");
    let worker = spawn_node(&mut app, Kind::Worker, 0, "Worker_test");
    let sink = spawn_node(&mut app, Kind::Sink, 0, "Sink_test");
    wire(&mut app, generator, Kind::Generator, queue, Kind::Queue);
    wire(&mut app, worker, Kind::Worker, queue, Kind::Queue);
    wire(&mut app, worker, Kind::Worker, sink, Kind::Sink);

    // Run a few sim seconds so the timeline accumulates packets across
    // both interior (composite) and exterior (user-drawn) edges.
    for _ in 0..200 {
        advance_sim_ns(&mut app, 16_000_000);
        app.update();
    }

    let membership = app.world().resource::<CompoundMembership>().clone();
    eprintln!("membership entries = {}", membership.parent.len());

    // Build a NodeId -> name map straight from the sim so we can classify
    // hops by name (`Gen_test::T`) without depending on membership being
    // populated in this headless harness.
    let names: std::collections::HashMap<flow::NodeId, String> = app
        .world()
        .resource::<flow_bevy::bridge::FlowSim>()
        .nodes
        .iter()
        .map(|(id, n)| (*id, n.name.clone()))
        .collect();
    let is_interior = |from: flow::NodeId, to: flow::NodeId| -> bool {
        let fname = names.get(&from).map(String::as_str).unwrap_or("");
        let tname = names.get(&to).map(String::as_str).unwrap_or("");
        // Interior hop: both endpoints are prefixed children of the SAME
        // compound (`X::a` -> `X::b`).
        match (fname.split_once("::"), tname.split_once("::")) {
            (Some((fc, _)), Some((tc, _))) => fc == tc,
            _ => false,
        }
    };

    let timeline = app.world().resource::<VisualTimelineRes>();
    let replay = timeline.strategy.as_replay();
    eprintln!("total ingested packets = {}", replay.packets.len());

    // Bucket every ingested packet by interior vs exterior hop.
    let mut interior: Vec<f64> = Vec::new();
    let mut exterior: Vec<f64> = Vec::new();
    for p in &replay.packets {
        let window = p.arrive_real - p.emit_real;
        if is_interior(p.from, p.to) {
            interior.push(window);
        } else {
            exterior.push(window);
        }
    }

    let stats = |v: &[f64]| -> (usize, f64, f64) {
        if v.is_empty() {
            return (0, 0.0, 0.0);
        }
        let max = v.iter().cloned().fold(f64::MIN, f64::max);
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        (v.len(), mean, max)
    };
    let (ni, mi, xi) = stats(&interior);
    let (ne, me, xe) = stats(&exterior);

    // A frame at 60fps is ~0.0167s. A packet is only catchable if its
    // window is an appreciable fraction of that.
    eprintln!("k={k}");
    eprintln!("INTERIOR (composite, lat=1ns): n={ni} mean_window={mi:.9}s max_window={xi:.9}s");
    eprintln!("EXTERIOR (user-drawn, lat=1ms): n={ne} mean_window={me:.9}s max_window={xe:.9}s");
    eprintln!("frame budget @60fps = 0.016700000s");

    // The diagnosis: interior packets DO exist and DO get ingested, but
    // their on-wire window is sim_latency(1ns) × k — three orders of
    // magnitude below a single frame, so they are never sampled in
    // flight. There is no minimum-window floor in the replay strategy.
    assert!(ni > 0, "expected interior packets to be ingested");
    assert!(
        xi < 0.001,
        "interior max window {xi:.9}s is sub-millisecond (the bug)"
    );
    assert!(
        xe > 0.05,
        "exterior windows {xe:.9}s should be clearly visible (control)"
    );
}
