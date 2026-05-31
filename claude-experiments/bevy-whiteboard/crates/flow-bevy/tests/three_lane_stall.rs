//! Reproduce the live-app "slows down after ~20s and stops, 0 absorbed".
//! Runs ThreeLaneFanout for a long sim time and checks for unbounded
//! accumulation (node inboxes / in-flight queue) that would make the
//! engine do ever-more work per tick — the signature of a stall — and
//! that sinks keep absorbing throughout.

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

fn sinks(app: &App) -> Vec<flow::NodeId> {
    let sim = app.world().resource::<FlowSim>();
    let prefix = format!("{}_", Kind::Sink.label());
    sim.nodes
        .iter()
        .filter(|(_, n)| n.name.starts_with(&prefix) && !n.name.contains("::"))
        .map(|(id, _)| *id)
        .collect()
}

fn total_absorbed(app: &App) -> i64 {
    let sim = app.world().resource::<FlowSim>();
    sinks(app)
        .iter()
        .filter_map(|n| match sim.read_slot_resolved(*n, "count") {
            Some(Value::Int(i)) => Some(*i),
            _ => None,
        })
        .sum()
}

/// Largest single-node inbox + the in-flight scheduled count. If either
/// grows without bound across the run, the engine's per-tick cost climbs
/// and the live app "lags out and stops".
fn backlog(app: &App) -> (usize, usize) {
    let sim = app.world().resource::<FlowSim>();
    let max_inbox = sim.nodes.values().map(|n| n.inbox.len()).max().unwrap_or(0);
    (max_inbox, sim.in_flight.len())
}

fn top_inboxes(app: &App, k: usize) -> Vec<(String, usize)> {
    let sim = app.world().resource::<FlowSim>();
    let mut v: Vec<(String, usize)> = sim
        .nodes
        .values()
        .filter(|n| !n.inbox.is_empty())
        .map(|n| (n.name.clone(), n.inbox.len()))
        .collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v.truncate(k);
    v
}

#[test]
fn three_lane_does_not_accumulate_unbounded_backlog() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);

    let mut prev_absorbed = 0i64;
    let mut samples: Vec<(u64, i64, usize, usize)> = Vec::new();
    for t_s in [5u64, 15, 30, 60, 90, 120] {
        // advance to t_s
        let target_ns = t_s * 1_000_000_000;
        let now = samples.last().map(|s| s.0 * 1_000_000_000).unwrap_or(0);
        advance_sim_ns(&mut app, target_ns - now);

        let absorbed = total_absorbed(&app);
        let (max_inbox, in_flight) = backlog(&app);
        eprintln!(
            "t={:>3}s absorbed={:>6} max_inbox={:>6} in_flight={:>6}  top={:?}",
            t_s, absorbed, max_inbox, in_flight, top_inboxes(&app, 5)
        );
        samples.push((t_s, absorbed, max_inbox, in_flight));

        assert!(
            absorbed > prev_absorbed,
            "absorption stalled between previous sample and {}s ({} -> {})",
            t_s, prev_absorbed, absorbed
        );
        prev_absorbed = absorbed;
    }

    // Backlog must not grow without bound. Compare the late-run max inbox
    // against the early-run one: a steadily climbing backlog is the stall.
    let early_inbox = samples[1].2; // @15s
    let late_inbox = samples.last().unwrap().2; // @120s
    assert!(
        late_inbox <= early_inbox.max(64) * 4,
        "node inbox backlog grew unbounded: {} @15s -> {} @120s (engine will lag and stall)",
        early_inbox, late_inbox
    );
}
