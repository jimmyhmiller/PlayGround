//! End-to-end: spin up the headless app with a counter gadget,
//! advance, rewind, verify state matches the rewind target plus the
//! visual layer was reset.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow_bevy::bridge::FlowSim;
use flow_bevy::edges::{RewindEpochSeen, VisualTimelineRes};
use flow_bevy::gadgets::Kind;
use flow_bevy::visual::VisualStrategy;

fn now_ns(app: &App) -> u64 {
    app.world().resource::<FlowSim>().now_ns
}

#[test]
fn rewind_to_zero_resets_sim_and_visuals() {
    let mut app = make_app();
    // Use a Generator: it self-loops, so plenty of events flow that
    // the visual layer will pick up. Period is whatever the default
    // generator params provide.
    let _gen = common::spawn_node(&mut app, Kind::Generator, 0, "Gen_rwd");

    // Advance a few seconds so the ring has accumulated multiple
    // captures past the t=0 anchor.
    advance_sim_ns(&mut app, 2_000_000_000);
    let advanced_now = now_ns(&app);
    assert!(advanced_now >= 2_000_000_000, "sim should have advanced past 2s");

    // Rewind to t=0. The anchor is sticky so this must succeed even
    // after eviction would otherwise have rolled t=0 off the ring.
    {
        let world = app.world_mut();
        let mut driver = world.resource_mut::<FlowSim>();
        let landed = driver.0.rewind(0);
        assert_eq!(landed, 0, "rewind to 0 should land at exactly t=0");
    }

    // After republish, the snapshot reflects t=0. Rewind itself does
    // *not* bump `rewind_epoch` — that's reserved for topology resets
    // (LoadExample/canvas-load) where the visual timeline is
    // discarded. Pure rewinds preserve the visual records and just
    // shift `visual_now`.
    app.update();
    let world = app.world();
    let snap = &world.resource::<flow_bevy::sim_driver::SimSnapshotRes>().0;
    assert_eq!(snap.now_ns, 0, "snapshot now_ns should be back at 0");
    assert!(
        !snap.rewind_markers_ns.is_empty(),
        "marker times should include at least the anchor"
    );
    assert_eq!(snap.rewind_markers_ns[0], 0, "anchor marker is at t=0");

    // We still observe the visual layer's last-seen epoch matches
    // the snapshot's — they should track in lockstep.
    let seen = app.world().resource::<RewindEpochSeen>().0;
    assert_eq!(seen, snap.rewind_epoch, "visual layer should track epoch");
}

#[test]
fn rewind_after_load_keeps_loaded_topology() {
    // Repro for: user loads an example, lets it run, hits rewind to
    // 0, expects the loaded topology — not the empty pre-load
    // canvas. The fix is `reset_history` after `*sim = new_sim`.
    let mut app = make_app();

    // Pretend the user pre-load: empty canvas, sim ticks for a while.
    advance_sim_ns(&mut app, 500_000_000);

    // Now load the ClientWorker example (replaces the sim wholesale).
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<flow_bevy::examples::LoadExample>>()
        .write(flow_bevy::examples::LoadExample(flow_bevy::examples::Example::ClientWorker));
    app.update();
    app.update();

    // Sanity: the loaded example has nodes.
    let after_load_nodes = app.world().resource::<FlowSim>().nodes.len();
    assert!(
        after_load_nodes > 0,
        "ClientWorker example should populate nodes; got {}",
        after_load_nodes,
    );

    // Run a bit, then rewind to 0. The rewind must land in the
    // post-load topology, not the empty pre-load one.
    advance_sim_ns(&mut app, 200_000_000);
    {
        let world = app.world_mut();
        let mut driver = world.resource_mut::<FlowSim>();
        driver.0.rewind(0);
    }
    app.update();

    let nodes_after_rewind = app.world().resource::<FlowSim>().nodes.len();
    assert_eq!(
        nodes_after_rewind, after_load_nodes,
        "rewind to 0 dropped the loaded topology (had {} pre-rewind, {} post-rewind)",
        after_load_nodes, nodes_after_rewind,
    );
}

#[test]
fn rewind_repopulates_visuals_with_in_flight_packets() {
    // Two distinct nodes connected by an edge with a long latency
    // (200ms in sim time). Inject a packet that takes the whole
    // window to traverse, advance to halfway through, rewind, and
    // verify the visual layer re-materializes that packet partway
    // through its animation rather than waiting for a fresh emit.
    let mut app = make_app();
    {
        let mut tl = app
            .world_mut()
            .resource_mut::<VisualTimelineRes>();
        tl.set_k(200.0);
    }

    // Use a Generator → Sink topology with a long edge so a packet
    // is reliably in flight at any sample point. The default
    // generator emits on its own period; we just need cross-edge
    // traffic.
    use flow_bevy::gadgets::Kind;
    let gen_id = common::spawn_node(&mut app, Kind::Generator, 0, "Gen_rwd_v");
    let sink = common::spawn_node(&mut app, Kind::Sink, 0, "Sink_rwd_v");
    common::wire(&mut app, gen_id, Kind::Generator, sink, Kind::Sink);

    // Stretch the Gen→Sink edge to 200ms in sim time so a packet
    // emitted at e.g. t=1.4s is still in flight at t=1.5s.
    {
        let world = app.world_mut();
        let mut driver = world.resource_mut::<FlowSim>();
        driver.0.with_sim_mut(move |sim| {
            for edge in sim.edges.values_mut() {
                if edge.from == gen_id && edge.to == sink {
                    edge.latency_ns = flow::Expr::int(200_000_000);
                }
            }
        });
    }

    // Run long enough that several packets have been emitted on
    // the long edge.
    advance_sim_ns(&mut app, 2_000_000_000);
    app.update();
    let mid = now_ns(&app);

    advance_sim_ns(&mut app, 1_000_000_000);
    app.update();

    // Rewind to mid. At that moment the sim had packets traveling
    // between Client and Worker — the visual layer should
    // re-materialize them.
    {
        let world = app.world_mut();
        let mut driver = world.resource_mut::<FlowSim>();
        driver.0.rewind(mid);
    }
    app.update();
    app.update();

    let in_flight_count = app.world().resource::<FlowSim>().in_flight.len();
    let timeline_packet_count = app.world()
        .resource::<VisualTimelineRes>()
        .0
        .as_replay()
        .packets
        .len();
    let visual_now = app.world().resource::<flow_bevy::bridge::SimClock>().visual_now;
    let visible = {
        let tl = app.world().resource::<VisualTimelineRes>();
        tl.0.visible_at(visual_now).count()
    };
    assert!(
        visible > 0,
        "after rewind to a moment with in-flight traffic, the visual layer \
         should preserve the originally-ingested visual packets and just \
         shift visual_now back; \
         visible={} in_flight={} packets_recorded={} visual_now={}",
        visible, in_flight_count, timeline_packet_count, visual_now,
    );
}

#[test]
fn reverse_play_walks_sim_time_backwards() {
    // Run forward, flip on reverse_play_rate, drive a few frames,
    // verify sim_now decreased monotonically.
    let mut app = make_app();
    let _gen = common::spawn_node(&mut app, flow_bevy::gadgets::Kind::Generator, 0, "Gen_rev");

    advance_sim_ns(&mut app, 2_000_000_000);
    app.update();
    let start = now_ns(&app);
    assert!(start > 0, "expected forward play to advance sim");

    app.world_mut()
        .resource_mut::<flow_bevy::bridge::SimClock>()
        .reverse_play_rate = 1.0;
    let mut prev = start;
    for _ in 0..10 {
        app.update();
        let now = now_ns(&app);
        assert!(
            now <= prev,
            "sim_now must not advance forward while reverse-play is active: {} -> {}",
            prev, now
        );
        prev = now;
    }
    assert!(
        prev < start,
        "reverse play should walk sim_now backward: start={} after 10 frames={}",
        start, prev
    );
}

#[test]
fn rewind_to_midpoint_lands_close() {
    let mut app = make_app();
    let _gen = common::spawn_node(&mut app, Kind::Generator, 0, "Gen_rwd_mid");

    // Run forward 1s, capture an intermediate "now", then run another
    // 1s and rewind back. We can't promise we land exactly on the
    // intermediate point (it depends on cadence), but we should land
    // at or before the requested time and not at zero.
    advance_sim_ns(&mut app, 1_000_000_000);
    let target = now_ns(&app);
    advance_sim_ns(&mut app, 1_000_000_000);

    let landed = {
        let world = app.world_mut();
        let mut driver = world.resource_mut::<FlowSim>();
        driver.0.rewind(target)
    };

    assert_eq!(
        landed, target,
        "rewind runs forward to target after restoring the snapshot"
    );
    app.update();
    let snap = &app.world().resource::<flow_bevy::sim_driver::SimSnapshotRes>().0;
    assert_eq!(snap.now_ns, target);
}
