//! End-to-end: spin up the headless app with a counter gadget,
//! advance, rewind, verify state matches the rewind target plus the
//! visual layer was reset.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow_bevy::bridge::FlowSim;
use flow_bevy::edges::{RewindEpochSeen, VisualTimelineRes};
use flow_bevy::gadgets::Kind;

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
        .strategy
        .as_replay()
        .packets
        .len();
    let visual_now = app.world().resource::<flow_bevy::bridge::SimClock>().visual_now;
    let visible = {
        let tl = app.world().resource::<VisualTimelineRes>();
        tl.visible.len()
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

// ─────────────────────────────────────────────────────────────────
// Performance tests
//
// These exist because the user reported (a) `«` not smooth, (b)
// dragging the rewind slider not smooth, (c) the whole UI locking
// up when rewinding from a long session. The shape is: a single
// rewind whose snap is much earlier than the target turns into a
// long `run_until` on the worker, blocking everything. The tests
// below pin down acceptable wall-time budgets so regressions in
// the rewind-strategy layer fail loudly.
//
// Budgets are deliberately loose — we're catching seconds-long
// lockups, not micro-optimizations.
// ─────────────────────────────────────────────────────────────────

use std::time::Instant;

/// Build a topology that mirrors the live app's load. 50
/// client→worker pairs at fast firing rates produce enough
/// event density that run-forward cost during rewind is
/// measurable.
fn spawn_dense_topology(app: &mut App) {
    use flow::Value;
    for i in 0..50 {
        let client = common::spawn_node(&mut *app, Kind::Client, i % 3, &format!("Cli_{}", i));
        let worker = common::spawn_node(&mut *app, Kind::Worker, i % 3, &format!("Wkr_{}", i));
        {
            let world = app.world_mut();
            let mut driver = world.resource_mut::<FlowSim>();
            driver.0.with_sim_mut(move |sim| {
                if let Some(n) = sim.nodes.get_mut(&client) {
                    n.slots.insert("period_ns".into(), Value::Int(20_000_000));
                }
                if let Some(n) = sim.nodes.get_mut(&worker) {
                    n.slots.insert("service_ns".into(), Value::Int(5_000_000));
                }
            });
        }
        common::wire(&mut *app, client, Kind::Client, worker, Kind::Worker);
    }
}

/// Single rewind to a target outside the snapshot ring's
/// coverage. With default cap 64 × 250ms cadence the ring covers
/// only the most recent ~16s, so a rewind to e.g. 30s in a 60s
/// session has to fall through to the t=0 anchor and run forward
/// 30 sim-seconds. We assert a wall-time budget; the user
/// reported multi-second lockups in this scenario.
#[test]
fn rewind_from_long_session_completes_quickly() {
    let mut app = make_app();
    spawn_dense_topology(&mut app);
    // 60s session. Advance in 250ms increments so the snapshot
    // ring gets populated naturally; a single huge advance only
    // produces one ring entry at the tail and doesn't model the
    // live app where the worker captures continuously.
    for _ in 0..240 {
        advance_sim_ns(&mut app, 250_000_000);
    }
    app.update();
    app.update(); // warmup

    // Target 30s — outside the ring's coverage of [44s, 60s], so we
    // fall through to the t=0 anchor and must run forward 30
    // sim-seconds during do_rewind.
    let t0 = Instant::now();
    let landed = {
        let world = app.world_mut();
        let mut driver = world.resource_mut::<FlowSim>();
        driver.0.rewind(30_000_000_000)
    };
    app.update();
    let elapsed = t0.elapsed();

    assert!(
        elapsed.as_millis() < 100,
        "rewind from 60s to 30s took {}ms — UI would lock up",
        elapsed.as_millis(),
    );
    assert_eq!(landed, 30_000_000_000, "rewind didn't land at target");
}

/// Scrub-back simulation: many rapid rewinds, like the user
/// dragging the slider backward. Each rewind triggers a
/// run-forward; if those add up linearly the experience janks.
#[test]
fn many_rewinds_in_a_row_stay_bounded() {
    let mut app = make_app();
    spawn_dense_topology(&mut app);
    for _ in 0..120 {
        advance_sim_ns(&mut app, 250_000_000);
    }
    app.update();

    // 30 rewinds stepping from 25s back to ~10s.
    let start = Instant::now();
    for i in 0..30 {
        let target_ns = 25_000_000_000_u64 - (i as u64) * 500_000_000;
        let world = app.world_mut();
        let mut driver = world.resource_mut::<FlowSim>();
        driver.0.rewind(target_ns);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 2_000,
        "30 rewinds took {}ms — scrubbing won't be smooth",
        elapsed.as_millis(),
    );
}

/// Repro for the « bug: when there was a sparse marker spacing
/// (a 1.5s gap between t=0 and the first auto_capture, caused by
/// a stale `last_tick` after `ResetHistory`/`CmdReply`), pressing
/// « at sim_now < 1.5s would jump straight to 0 instead of
/// stepping by ~250ms. We verify markers are densely packed all
/// the way down to t=0 — no big gap.
#[test]
fn snapshot_markers_are_densely_packed_after_load() {
    let mut app = make_app();
    spawn_dense_topology(&mut app);
    // Run for 5 seconds in 250ms chunks. With the fix, every
    // 250ms-long chunk produces an auto_capture at sim time
    // ~k*250ms. Without it, the first chunk's last_tick was
    // stale-from-spawn-phase and sim jumped past 250ms in one
    // step, producing a single coarse marker at e.g. 1500ms.
    for _ in 0..20 {
        advance_sim_ns(&mut app, 250_000_000);
    }
    app.update();

    let markers = app.world()
        .resource::<flow_bevy::sim_driver::SimSnapshotRes>().0
        .rewind_markers_ns
        .clone();
    let max_gap_ns = markers
        .windows(2)
        .map(|w| w[1] - w[0])
        .max()
        .unwrap_or(0);

    assert!(
        max_gap_ns < 600_000_000,
        "max marker gap {}ns ({}ms) exceeds 600ms — `«` would jump multiple steps near sparse regions; markers={:?}",
        max_gap_ns, max_gap_ns / 1_000_000, markers,
    );
}

/// Repro for: after rewind in worker mode, the next wall-delta
/// advance raced the `paused` atomic and advanced sim past the
/// target. Mirrors the HUD's actual button flow (rewind + pause).
/// In Direct mode this also catches the case where Bevy's
/// per-frame `dt_real` advance system would push sim past target.
#[test]
fn rewind_target_lands_exactly_no_overshoot() {
    let mut app = make_app();
    spawn_dense_topology(&mut app);
    for _ in 0..40 {
        advance_sim_ns(&mut app, 250_000_000);
    }
    let before = now_ns(&app);
    assert!(before >= 10_000_000_000);

    let target = 5_000_000_000;
    {
        // Pause first — that's what the HUD's rewind buttons do
        // before issuing the rewind. Without this the per-frame
        // advance system runs in the next app.update and pushes
        // sim past target by up to `dt_real * multiplier`.
        let world = app.world_mut();
        world.resource_mut::<flow_bevy::bridge::SimClock>().paused = true;
        let mut driver = world.resource_mut::<FlowSim>();
        driver.0.rewind(target);
    }
    app.update();
    let after = now_ns(&app);
    assert_eq!(
        after, target,
        "rewind to {}ns landed at {}ns — overshoot of {}ns",
        target, after, after.saturating_sub(target),
    );
}

/// «-button repeated stepping. Each click rewinds by one capture
/// marker (~250ms by default). 50 clicks shouldn't add up to a
/// noticeable hang.
#[test]
fn repeated_step_back_stays_bounded() {
    let mut app = make_app();
    spawn_dense_topology(&mut app);
    for _ in 0..120 {
        advance_sim_ns(&mut app, 250_000_000);
    }
    app.update();

    let start = Instant::now();
    let mut anchor = now_ns(&app);
    for _ in 0..50 {
        // Find previous marker (mirrors HUD logic).
        let snap = app.world().resource::<flow_bevy::sim_driver::SimSnapshotRes>().0.clone();
        let Some(prev) = snap.rewind_markers_ns.iter().copied().filter(|m| *m < anchor).max()
        else { break };
        let world = app.world_mut();
        let mut driver = world.resource_mut::<FlowSim>();
        driver.0.rewind(prev);
        drop(driver);
        app.update();
        anchor = prev;
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 2_000,
        "50 « clicks took {}ms — would feel like a freeze",
        elapsed.as_millis(),
    );
}
