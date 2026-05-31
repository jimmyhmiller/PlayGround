//! Cross-lane color leakage check for ThreeLaneFanout.
//!
//! Loads the example, advances the sim, and verifies that every sink
//! only counts packets whose slot matches its own color. A red packet
//! ending up in the blue sink (or being visually painted red on the
//! green lane) would mean the Filter primitive isn't actually filtering
//! — the bug the user reported as "green is being routed where red is
//! supposed to go then converted to red."

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow::Value;
use flow_bevy::bridge::FlowSim;
use flow_bevy::examples::{Example, LoadExample};
use flow_bevy::gadgets::Kind;

fn load(app: &mut App, ex: Example) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<LoadExample>>()
        .write(LoadExample(ex));
    app.update();
    app.update();
}

/// Read the compound shim's effective "color" by peeking at the inner
/// Filter's `match` slot — the SinkComposite/QueueComposite/etc. recipe
/// threads the compound's `color` param into that primitive at
/// expansion time.
fn shim_color(app: &App, nid: flow::NodeId) -> i64 {
    let sim = app.world().resource::<FlowSim>();
    let Some(n) = sim.nodes.get(&nid) else { return -1 };
    let prefix = format!("{}::", n.name);
    for inner in sim.nodes.values() {
        if inner.name.starts_with(&prefix) {
            if let Some(Value::Int(i)) = inner.slots.get("match") {
                return *i;
            }
        }
    }
    -1
}

fn sinks_by_color(app: &App) -> Vec<flow::NodeId> {
    let sim = app.world().resource::<FlowSim>();
    let prefix = format!("{}_", Kind::Sink.label());
    let mut out: Vec<_> = sim
        .nodes
        .iter()
        .filter(|(_, n)| n.name.starts_with(&prefix) && !n.name.contains("::"))
        .map(|(id, _)| *id)
        .collect();
    out.sort_by_key(|nid| shim_color(app, *nid));
    out
}

#[test]
fn three_lane_sinks_count_only_own_color() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);
    advance_sim_ns(&mut app, 5_000_000_000);

    let sinks = sinks_by_color(&app);
    let sim = app.world().resource::<FlowSim>();
    let counts: Vec<i64> = sinks
        .iter()
        .map(|nid| match sim.read_slot_resolved(*nid, "count") {
            Some(Value::Int(i)) => *i,
            _ => -1,
        })
        .collect();
    eprintln!("sink counts by color (0,1,2): {:?}", counts);
    for (i, c) in counts.iter().enumerate() {
        assert!(*c > 0, "sink color {} got no work in 5s: {:?}", i, counts);
    }
}

/// Verify the visual color resolution: every event delivered to the
/// sink shim should carry a payload whose slot tag matches the sink's
/// own color. If the Router or any forwarder is mangling payloads, we
/// catch that here.
#[test]
fn three_lane_event_payloads_match_sink_color() {
    use flow::Sim;
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);
    advance_sim_ns(&mut app, 5_000_000_000);

    let sinks = sinks_by_color(&app);
    let sink_colors: Vec<(flow::NodeId, i64)> = sinks
        .iter()
        .map(|nid| (*nid, shim_color(&app, *nid)))
        .collect();
    let sim: &Sim = app.world().resource::<FlowSim>();

    let mut mismatches: Vec<(flow::NodeId, i64, i64)> = Vec::new();
    for ev in sim.log.iter() {
        let flow::Event::PacketEmitted { to, payload, .. } = ev else { continue };
        let Some((target_id, target_color)) = sink_colors.iter().find(|(id, _)| *id == *to) else {
            continue;
        };
        let Some(slot) = packet_slot(payload) else { continue };
        if slot as i64 != *target_color {
            mismatches.push((*target_id, *target_color, slot as i64));
        }
    }
    assert!(
        mismatches.is_empty(),
        "events arrived at sinks with mismatched color: {:?}",
        mismatches.iter().take(10).collect::<Vec<_>>(),
    );
}

fn packet_slot(v: &Value) -> Option<usize> {
    let (tag, inner) = v.as_variant()?;
    if tag != "packet" && tag != "req" {
        return None;
    }
    match inner {
        Value::Int(i) if *i >= 0 => Some(*i as usize),
        _ => None,
    }
}
