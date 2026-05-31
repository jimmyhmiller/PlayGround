//! Long-running sim test: verify data keeps flowing for 30 sim-seconds.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
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

fn slot_int(app: &App, nid: flow::NodeId, slot: &str) -> i64 {
    match app.world().resource::<FlowSim>().read_slot_resolved(nid, slot) {
        Some(flow::Value::Int(i)) => *i,
        other => panic!("slot `{}` not Int: {:?}", slot, other),
    }
}

fn first_of_kind(app: &App, kind: Kind) -> flow::NodeId {
    let sim = app.world().resource::<FlowSim>();
    let prefix = format!("{}_", kind.label());
    sim.nodes
        .iter()
        .find(|(_, n)| n.name.starts_with(&prefix) && !n.name.contains("::"))
        .map(|(id, _)| *id)
        .unwrap_or_else(|| panic!("no node of kind {:?}", kind))
}

fn all_of_kind(app: &App, kind: Kind) -> Vec<flow::NodeId> {
    let sim = app.world().resource::<FlowSim>();
    let prefix = format!("{}_", kind.label());
    sim.nodes
        .iter()
        .filter(|(_, n)| n.name.starts_with(&prefix) && !n.name.contains("::"))
        .map(|(id, _)| *id)
        .collect()
}

#[test]
fn client_queue_worker_still_flowing_at_60s() {
    let mut app = make_app();
    load(&mut app, Example::ClientQueueWorker);
    let sink = first_of_kind(&app, Kind::Sink);
    let mut samples: Vec<(u64, i64)> = Vec::new();
    let mut prev = 0u64;
    for t_s in [3u64, 10, 20, 40, 60] {
        let t_ns = t_s * 1_000_000_000;
        advance_sim_ns(&mut app, t_ns - prev);
        let c = slot_int(&app, sink, "count");
        eprintln!("CQW: sink count @ {}s = {}", t_s, c);
        samples.push((t_s, c));
        prev = t_ns;
    }
    // Each adjacent pair must show growth — no halts mid-run.
    for w in samples.windows(2) {
        let (t_a, c_a) = w[0];
        let (t_b, c_b) = w[1];
        assert!(c_b > c_a, "no new sink work between {}s and {}s ({} → {})", t_a, t_b, c_a, c_b);
    }
}

#[test]
fn three_lane_fanout_still_flowing_at_60s() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);
    let sinks = all_of_kind(&app, Kind::Sink);
    let mut prev = 0u64;
    let mut history: Vec<Vec<i64>> = Vec::new();
    for t_s in [3u64, 10, 20, 40, 60] {
        let t_ns = t_s * 1_000_000_000;
        advance_sim_ns(&mut app, t_ns - prev);
        let counts: Vec<i64> = sinks.iter().map(|s| slot_int(&app, *s, "count")).collect();
        eprintln!("TLF @ {}s = {:?}", t_s, counts);
        history.push(counts);
        prev = t_ns;
    }
    for (lane, _) in sinks.iter().enumerate() {
        for w_idx in 0..history.len() - 1 {
            let a = history[w_idx][lane];
            let b = history[w_idx + 1][lane];
            assert!(b > a, "lane {}: no new work between snapshot {} ({}) and {} ({})",
                lane, w_idx, a, w_idx + 1, b);
        }
    }
}

#[test]
fn client_worker_still_flowing_at_15s() {
    let mut app = make_app();
    load(&mut app, Example::ClientWorker);
    advance_sim_ns(&mut app, 3_000_000_000);
    let client = first_of_kind(&app, Kind::Client);
    let completed_3s = slot_int(&app, client, "completed");
    advance_sim_ns(&mut app, 12_000_000_000);
    let completed_15s = slot_int(&app, client, "completed");
    let delta = completed_15s - completed_3s;
    eprintln!("CW: completed @ 3s = {}, @ 15s = {}, delta = {}", completed_3s, completed_15s, delta);
    assert!(delta > 0, "no new responses received between 3s and 15s");
}
