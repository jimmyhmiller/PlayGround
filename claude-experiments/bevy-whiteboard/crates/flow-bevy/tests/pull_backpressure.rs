//! Isolate the Service-inbox stall: does the pull protocol apply
//! backpressure (excess stays bounded in the queue buffer) or does it
//! let the worker's Service inbox grow without bound?
//!
//! Two cases:
//!  - single worker pulling from a queue, generator oversaturating it
//!  - two workers sharing one queue (the ThreeLaneFanout yellow lane)

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app, spawn_node, wire};
use flow::Value;
use flow_bevy::bridge::FlowSim;
use flow_bevy::gadgets::Kind;

fn set_slot(app: &mut App, nid: flow::NodeId, slot: &str, v: i64) {
    let slot = slot.to_string();
    app.world_mut()
        .resource_mut::<FlowSim>()
        .0
        .with_sim_mut(move |sim| {
            sim.write_slot_resolved(nid, &slot, Value::Int(v));
        });
}

fn inner_inbox(app: &App, shim: flow::NodeId, suffix: &str) -> usize {
    let sim = app.world().resource::<FlowSim>();
    let name = sim.nodes.get(&shim).map(|n| format!("{}::{}", n.name, suffix));
    let Some(name) = name else { return 0 };
    sim.nodes
        .values()
        .find(|n| n.name == name)
        .map(|n| n.inbox.len())
        .unwrap_or(0)
}

#[test]
fn single_worker_pull_keeps_service_bounded() {
    let mut app = make_app();
    let g = spawn_node(&mut app, Kind::Generator, 0, "Gen");
    let queue = spawn_node(&mut app, Kind::Queue, 0, "Q");
    let worker = spawn_node(&mut app, Kind::Worker, 0, "W");
    let sink = spawn_node(&mut app, Kind::Sink, 0, "Snk");
    wire(&mut app, g, Kind::Generator, queue, Kind::Queue);
    wire(&mut app, worker, Kind::Worker, queue, Kind::Queue);
    wire(&mut app, worker, Kind::Worker, sink, Kind::Sink);

    // Oversaturate: gen 30/s, worker 20/s.
    set_slot(&mut app, g, "period_ns", 33_000_000);
    set_slot(&mut app, worker, "service_ns", 50_000_000);

    let mut prev = 0u64;
    let mut sv_history = Vec::new();
    for t_s in [5u64, 15, 30] {
        let target = t_s * 1_000_000_000;
        advance_sim_ns(&mut app, target - prev);
        prev = target;
        let sv = inner_inbox(&app, worker, "Sv");
        eprintln!("single: t={}s  Sv inbox={}", t_s, sv);
        sv_history.push(sv);
    }
    // The queue buffer (cap) should absorb the excess and drop overflow;
    // the worker's Service inbox must stay small and bounded.
    let last = *sv_history.last().unwrap();
    assert!(
        last < 64,
        "single-worker Service inbox grew unbounded: {:?} (pull backpressure broken)",
        sv_history
    );
}

#[test]
fn two_workers_sharing_queue_keep_service_bounded() {
    let mut app = make_app();
    let g = spawn_node(&mut app, Kind::Generator, 0, "Gen");
    let queue = spawn_node(&mut app, Kind::Queue, 0, "Q");
    let w1 = spawn_node(&mut app, Kind::Worker, 0, "W1");
    let w2 = spawn_node(&mut app, Kind::Worker, 0, "W2");
    let sink = spawn_node(&mut app, Kind::Sink, 0, "Snk");
    wire(&mut app, g, Kind::Generator, queue, Kind::Queue);
    wire(&mut app, w1, Kind::Worker, queue, Kind::Queue);
    wire(&mut app, w2, Kind::Worker, queue, Kind::Queue);
    wire(&mut app, w1, Kind::Worker, sink, Kind::Sink);
    wire(&mut app, w2, Kind::Worker, sink, Kind::Sink);

    // gen 30/s; two workers @ 20/s each = 40/s capacity → NOT saturated.
    set_slot(&mut app, g, "period_ns", 33_000_000);
    set_slot(&mut app, w1, "service_ns", 50_000_000);
    set_slot(&mut app, w2, "service_ns", 50_000_000);

    let mut prev = 0u64;
    let mut hist = Vec::new();
    for t_s in [5u64, 15, 30] {
        let target = t_s * 1_000_000_000;
        advance_sim_ns(&mut app, target - prev);
        prev = target;
        let s1 = inner_inbox(&app, w1, "Sv");
        let s2 = inner_inbox(&app, w2, "Sv");
        eprintln!("two: t={}s  W1.Sv={} W2.Sv={}", t_s, s1, s2);
        hist.push((s1, s2));
    }
    let (l1, l2) = *hist.last().unwrap();
    assert!(
        l1 < 64 && l2 < 64,
        "two-worker Service inboxes grew unbounded: {:?} (queue broadcasts each item to all workers)",
        hist
    );
}
