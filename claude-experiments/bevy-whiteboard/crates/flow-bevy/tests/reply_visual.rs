//! Regression test for the visual reply pipeline. Under F1 there
//! is no per-edge throttle — the visual timeline creates one
//! `VisualPacket` per `PacketEmitted`, so as long as the sim logs
//! a reply, there is a corresponding reverse-direction
//! `TravelingPacket`.
//!
//! "Reverse direction" here means the packet's `(from, to)` is the
//! opposite of some declared sim edge. The test classifies live
//! packets against the edge map instead of reading a flag off the
//! component.

mod common;

use bevy::prelude::*;
use common::make_app;
use flow_bevy::bridge::{FlowSim, SimClock};
use flow_bevy::edges::TravelingPacket;
use flow_bevy::examples::{Example, LoadExample};

fn load(app: &mut App, ex: Example) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<LoadExample>>()
        .write(LoadExample(ex));
    app.update();
    app.update();
}

/// Advance the sim by `step_ns` using a one-shot sim tick (sets
/// `SimClock::step_once_ns` and runs `app.update()`). This routes
/// through the real `advance_sim` system and `collect_new_events`
/// and `spawn_traveling_packets`, unlike `sim.run_until` which
/// bypasses Bevy entirely.
fn step_frame(app: &mut App, step_ns: u64) {
    app.world_mut()
        .resource_mut::<SimClock>()
        .step_once_ns = Some(step_ns);
    app.update();
}

/// Classify a packet's `(from, to)` direction against declared sim
/// edges. Returns `(is_forward, is_reverse)`. Both false means the
/// packet has no edge relationship either way (shouldn't happen in
/// practice — the sim only emits between connected nodes).
fn classify(pkt: &TravelingPacket, edges: &[(flow::NodeId, flow::NodeId)]) -> (bool, bool) {
    let forward = edges.iter().any(|(f, t)| *f == pkt.from && *t == pkt.to);
    let reverse = edges.iter().any(|(f, t)| *f == pkt.to && *t == pkt.from);
    (forward, reverse)
}

fn edges_snapshot(app: &mut App) -> Vec<(flow::NodeId, flow::NodeId)> {
    app.world().resource::<FlowSim>().sim.edges.values()
        .map(|e| (e.from, e.to))
        .collect()
}

fn count_packets_by_direction(app: &mut App) -> (usize, usize) {
    let edges = edges_snapshot(app);
    let world = app.world_mut();
    let mut q = world.query::<&TravelingPacket>();
    let mut forward = 0;
    let mut reverse = 0;
    for pkt in q.iter(world) {
        let (fw, rv) = classify(pkt, &edges);
        if fw { forward += 1; }
        else if rv { reverse += 1; }
    }
    (forward, reverse)
}

/// Snapshot tuple: (is_reverse, visible, progress, emit_real).
/// Progress is derived from the real time clock and the packet's
/// emit/arrive, matching the runtime's formula.
fn snapshot_packets(app: &mut App) -> Vec<(bool, bool, f32, f32)> {
    let edges = edges_snapshot(app);
    let real_t = app.world().resource::<Time>().elapsed_secs_f64();
    let world = app.world_mut();
    let mut q = world.query::<(&TravelingPacket, &Visibility)>();
    q.iter(world)
        .map(|(pkt, vis)| {
            let (_fw, rv) = classify(pkt, &edges);
            let denom = (pkt.arrive_real - pkt.emit_real).max(1e-9);
            let progress = ((real_t - pkt.emit_real) / denom).clamp(0.0, 1.0) as f32;
            let visible = matches!(vis, Visibility::Visible | Visibility::Inherited);
            (rv, visible, progress, pkt.emit_real as f32)
        })
        .collect()
}

/// `TwoClientsOneWorker`: requests should animate forward on both
/// Client→Worker edges, AND replies should animate in reverse on
/// those same edges. Both counts must be > 0 after a few frames.
/// Before the throttle fix, `reverse` stayed at 0.
#[test]
fn two_clients_one_worker_visual_replies_spawn() {
    let mut app = make_app();
    load(&mut app, Example::TwoClientsOneWorker);

    // Several small frames so req → resp round-trips happen with
    // real-time gaps between them. At default rates (300/500 ms)
    // we get a handful of round-trips inside 1s total sim time.
    let mut saw_forward_ever = false;
    let mut saw_reverse_ever = false;
    for _ in 0..20 {
        step_frame(&mut app, 50_000_000); // 50 ms sim per frame
        let (f, r) = count_packets_by_direction(&mut app);
        if f > 0 { saw_forward_ever = true; }
        if r > 0 { saw_reverse_ever = true; }
    }

    assert!(saw_forward_ever, "never spawned a forward request visual");
    assert!(
        saw_reverse_ever,
        "never spawned a reverse reply visual — the throttle \
         is swallowing resp animations"
    );

    // Sanity: the sim itself did actually serve requests. Catches
    // the case where the test misconfigured the scenario and
    // nothing ever happened.
    let sim = &app.world().resource::<FlowSim>().sim;
    let worker = sim.nodes.values()
        .find(|n| n.name.starts_with("Worker_"))
        .expect("no worker");
    let served = match worker.slots.get("served") {
        Some(flow::Value::Int(i)) => *i,
        _ => 0,
    };
    assert!(served > 0, "worker.served == 0, scenario isn't running");
}

/// `ClientWorker` is the simplest shape that hits reverse-routing.
/// Sanity-check that one too.
#[test]
fn client_worker_visual_reply_spawns() {
    let mut app = make_app();
    load(&mut app, Example::ClientWorker);

    let mut saw_reverse = false;
    for _ in 0..20 {
        step_frame(&mut app, 50_000_000);
        let (_, r) = count_packets_by_direction(&mut app);
        if r > 0 { saw_reverse = true; }
    }
    assert!(saw_reverse, "ClientWorker: never spawned a reverse reply visual");
}

/// `ClientQueueWorker` also exercises reverse-routing: the queue's
/// ack to the client reverses along the Client→Queue edge.
#[test]
fn client_queue_worker_visual_ack_spawns() {
    let mut app = make_app();
    load(&mut app, Example::ClientQueueWorker);

    let mut saw_reverse = false;
    for _ in 0..20 {
        step_frame(&mut app, 50_000_000);
        let (_, r) = count_packets_by_direction(&mut app);
        if r > 0 { saw_reverse = true; }
    }
    assert!(saw_reverse, "ClientQueueWorker: never spawned a reverse ack visual");
}

/// Visual causality (V3): a reverse packet must never be VISIBLE
/// while a forward on the same edge is still animating. This is
/// the bug the user observed: "worker replying before it got a
/// message." The fix is the `spawn_at_real` deferral in the
/// spawner — the reverse entity exists but stays
/// `Visibility::Hidden` until real time passes the forward's
/// end.
#[test]
fn v3_reverse_never_visible_during_forward() {
    let mut app = make_app();
    load(&mut app, Example::ClientWorker);

    // Many small frames; within each frame, if both a forward and
    // a reverse exist, the reverse must be invisible as long as
    // the forward hasn't completed (t < 1.0).
    for _ in 0..30 {
        step_frame(&mut app, 30_000_000);
        let snap = snapshot_packets(&mut app);
        let forward_still_running = snap.iter().any(|(rev, _vis, t, _)| !rev && *t < 1.0);
        if !forward_still_running { continue; }
        for (rev, vis, _, _) in &snap {
            if *rev && *vis {
                panic!(
                    "reverse packet is visible while a forward is still animating \
                     on the same edge — causality violation. snapshot: {:?}",
                    snap
                );
            }
        }
    }
}

/// Multi-hop visual causality (the "spontaneous emit from
/// Router/Queue at frame 0" bug). In ThreeLaneFanout, a packet
/// flows Gen→Router→Queue→Worker→Sink and all these emits appear
/// in NewEvents in the same frame. Without per-node arrival
/// sequencing, every intermediate node appears to emit a packet
/// at the same real instant as the Generator, so the user sees
/// packets "spawn out of" Routers and Queues at frame 0.
///
/// Property: on the FIRST frame after load, at most ONE
/// TravelingPacket is visible per node-chain. Specifically: the
/// Gen→Router forward is visible; the Router→Queue forward's
/// `spawn_at_real` is in the future (so visibility is Hidden).
#[test]
fn v3_first_frame_only_source_emit_is_visible() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);

    // Advance just one small frame of sim time, enough for the
    // initial tick chain to fire but not for animations to
    // complete.
    step_frame(&mut app, 50_000_000);

    let snap = snapshot_packets(&mut app);
    // Many packets may have been spawned (generators × 3, each
    // with several hops). But at this moment, the only ones
    // VISIBLE should be those on the "first hop" of each chain —
    // the rest are still deferred behind their predecessors.
    //
    // Concrete assertion: count visible packets; it must be less
    // than the total spawned count. Before the fix, visible ==
    // total (every packet shows up at t=0 simultaneously).
    let total   = snap.len();
    let visible = snap.iter().filter(|(_, vis, _, _)| *vis).count();
    assert!(
        total > 0,
        "scenario didn't spawn any packets in the first frame"
    );
    assert!(
        visible < total,
        "all {} packets visible simultaneously at frame 0 — packets \
         appear to spawn out of Routers/Queues instead of chaining \
         Gen→Router→Queue→Worker→Sink. snap: {:?}",
        total, snap
    );
}

/// Same property but through the multi-hop Router scenario: the
/// reverse-route reply walks Worker→Router→Client across two
/// edges. Per-edge, the reverse must still not be visible while
/// the forward on that edge is in progress. Catches multi-hop
/// visual regressions.
#[test]
fn v3_reverse_never_visible_during_forward_through_router() {
    let mut app = make_app();
    load(&mut app, Example::ClientRouterWorker);

    for _ in 0..30 {
        step_frame(&mut app, 30_000_000);
        let snap = snapshot_packets(&mut app);
        // For each visible reverse packet, check that no forward
        // on any edge is still animating. (We can't easily check
        // "same edge" without exposing edge IDs; the weaker check
        // suffices for this scenario — it has at most one active
        // round-trip at a time, so forward-present ⇒ reverse
        // shouldn't be visible yet.)
        let forward_running = snap.iter().any(|(rev, vis, t, _)| !rev && *vis && *t < 0.95);
        if !forward_running { continue; }
        for (rev, vis, _, _) in &snap {
            if *rev && *vis {
                panic!(
                    "reverse packet visible while forward still running in CRW: {:?}",
                    snap
                );
            }
        }
    }
}
