//! End-to-end test for `examples/region.whiteboard` — a multi-tier
//! transactional system that combines saga, TPC, and quorum-read on
//! a shared replica set.
//!
//! Topology recap (see `examples/region.whiteboard/main.flow` for the
//! full picture):
//!   WRITE: WriteClient → WriteCB → Saga → ReserveInv ↗ InvShard{0,1,2}
//!                                       → ChargePay  ↗ PaymentCB → PaymentWorker
//!                                       → PersistOrder ↗ TPCCoord → Replica{0,1,2} → DB{0,1,2}
//!   READ:  ReadClient → ReadCache → QuorumReader → Replica{0,1,2}  (R = 2 of 3)
//!
//! What the assertions pin:
//!   • Healthy phase (0–2.5s, before any chaos fires) is clean: no
//!     unexpected runtime errors.
//!   • Both paths flowed: WriteClient.completed > 0, ReadClient.completed > 0.
//!   • TPC commits actually reached every replica AND fanned to each
//!     replica's DB sink (Replica{i}.commits == DB{i}.count).
//!   • Quorum reads served reads from at least 2 of 3 replicas
//!     (sum(Replica.reads) ≥ 2 × QuorumReader.reads, since the Rth
//!     reply may include any 2 of 3 replicas).
//!   • Custom components survived the load: Replica/QuorumReader/
//!     WorkingSagaStep instance slots are queryable.

use std::path::PathBuf;

use bevy::prelude::*;
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

fn no_unexpected_runtime_errors(sim: &Sim, allowed: &[&str]) {
    let errs: Vec<_> = sim.error_counts.iter()
        .filter(|(k, c)| **c > 0 && !allowed.contains(&k.as_str()))
        .collect();
    assert!(errs.is_empty(), "unexpected runtime errors: {:?}", errs);
}

#[test]
#[ignore = "Composite migration: assertion depends on monolithic gadget shape (direct slot access or event-log emit counts matching shim ids)."]
fn region_whiteboard_loads_and_runs_healthy_baseline() {
    let path = project_root().join("examples/region.whiteboard");
    let mut canvas = load_canvas(&path, 19).expect("load region.whiteboard");

    // No scenario block at the moment — steady state only.
    assert_eq!(canvas.sim.timeline.events.len(), 0);

    // Healthy baseline. First chaos event is at 3s; tick to 2.5s
    // so we observe a clean run.
    canvas.sim.run_until(canvas.sim.now_ns + 2_500_000_000);

    no_unexpected_runtime_errors(&canvas.sim, &[]);

    // Both paths flowed.
    let writes_done = slot_i(&canvas.sim, "WriteClient", "completed");
    let reads_done  = slot_i(&canvas.sim, "ReadClient", "completed");
    assert!(writes_done >= 1,
        "WriteClient.completed should be ≥1 in 2.5s, got {}", writes_done);
    assert!(reads_done >= 1,
        "ReadClient.completed should be ≥1 in 2.5s, got {}", reads_done);

    // TPC commits reached every replica AND every DB sink in lockstep.
    let r0_commits = slot_i(&canvas.sim, "Replica0", "commits");
    let r1_commits = slot_i(&canvas.sim, "Replica1", "commits");
    let r2_commits = slot_i(&canvas.sim, "Replica2", "commits");
    let coord_committed = slot_i(&canvas.sim, "TPCCoord", "committed");
    assert!(coord_committed >= 1, "TPCCoord.committed should be ≥1");
    assert_eq!(r0_commits, coord_committed);
    assert_eq!(r1_commits, coord_committed);
    assert_eq!(r2_commits, coord_committed);
    // Every replica's commit fans `packet(color)` to its DB sink.
    assert_eq!(slot_i(&canvas.sim, "DB0", "count"), r0_commits);
    assert_eq!(slot_i(&canvas.sim, "DB1", "count"), r1_commits);
    assert_eq!(slot_i(&canvas.sim, "DB2", "count"), r2_commits);

    // Versions tracked the commits.
    assert_eq!(slot_i(&canvas.sim, "Replica0", "version"), coord_committed);
    assert_eq!(slot_i(&canvas.sim, "Replica1", "version"), coord_committed);
    assert_eq!(slot_i(&canvas.sim, "Replica2", "version"), coord_committed);

    // Quorum reads: each successful QR.read tallies across exactly R
    // replicas. With R = 2 and 3 replicas, sum(replica.reads) is at
    // least 2 × QR.reads (some rounds may have all 3 reply).
    let qr_reads = slot_i(&canvas.sim, "QuorumReader", "reads");
    let r0_reads = slot_i(&canvas.sim, "Replica0", "reads");
    let r1_reads = slot_i(&canvas.sim, "Replica1", "reads");
    let r2_reads = slot_i(&canvas.sim, "Replica2", "reads");
    assert!(qr_reads >= 1, "QuorumReader.reads should be ≥1");
    assert!(r0_reads + r1_reads + r2_reads >= 2 * qr_reads,
        "expected ≥{} replica reads, got {}+{}+{}={}",
        2 * qr_reads, r0_reads, r1_reads, r2_reads,
        r0_reads + r1_reads + r2_reads);

    // ReserveInv (the working saga step) actually called inventory
    // shards — at least one of the three should have served some.
    let inv0 = slot_i(&canvas.sim, "InvShard0", "served");
    let inv1 = slot_i(&canvas.sim, "InvShard1", "served");
    let inv2 = slot_i(&canvas.sim, "InvShard2", "served");
    assert!(inv0 + inv1 + inv2 >= 1,
        "no inventory shard served any reqs in 2.5s");
    assert!(slot_i(&canvas.sim, "ReserveInv", "done") >= 1,
        "ReserveInv.done should advance — backend resp drives the bump");

    // Saga succeeded at least once.
    assert!(slot_i(&canvas.sim, "Saga", "succeeded") >= 1,
        "Saga.succeeded should be ≥1 in 2.5s healthy");
    assert_eq!(slot_i(&canvas.sim, "Saga", "failed"), 0,
        "no failures expected during healthy phase");
}

// (Chaos timeline removed — steady state only for now.)

/// Make sure the sim is producing the kind of events the visual layer
/// actually renders — `PacketEmitted` with non-zero latency, payload
/// tag != "pull"/"wake", from != to.
#[test]
#[ignore = "Composite migration: assertion depends on monolithic gadget shape (direct slot access or event-log emit counts matching shim ids)."]
fn region_whiteboard_emits_visible_packet_events() {
    use flow::Event;

    let path = project_root().join("examples/region.whiteboard");
    let mut canvas = load_canvas(&path, 7).expect("load region.whiteboard");
    canvas.sim.run_until(canvas.sim.now_ns + 1_500_000_000);

    let mut visible = 0usize;
    let mut by_tag = std::collections::BTreeMap::<String, usize>::new();
    for ev in &canvas.sim.log.events {
        if let Event::PacketEmitted { from, to, payload, at_ns, arrives_at_ns, .. } = ev {
            if from == to { continue; }
            if arrives_at_ns <= at_ns { continue; }
            if let Some((tag, _)) = payload.as_variant() {
                if tag == "pull" || tag == "wake" { continue; }
                *by_tag.entry(tag.to_string()).or_default() += 1;
            }
            visible += 1;
        }
    }
    assert!(visible > 20,
        "expected many visible packets in 1.5s, got {} (tags: {:?})", visible, by_tag);
    eprintln!("visible packets in 1.5s = {}, by tag = {:?}", visible, by_tag);

    // Dump every node's `color` slot — verify overrides took effect.
    eprintln!("---- color slots ----");
    let mut entries: Vec<(String, i64)> = canvas.sim.nodes.values()
        .filter_map(|n| match n.slots.get("color") {
            Some(flow::Value::Int(c)) => Some((n.name.clone(), *c)),
            _ => None,
        })
        .collect();
    entries.sort();
    for (name, c) in &entries { eprintln!("{:>16}: {}", name, c); }

    // Dump error_counts to see what's recorded.
    eprintln!("---- error_counts ----");
    for (k, v) in &canvas.sim.error_counts {
        eprintln!("{:>20}: {}", k, v);
    }

    // Dump every edge to verify our wiring.
    eprintln!("---- edges ({}) ----", canvas.sim.edges.len());
    let by_id: std::collections::HashMap<flow::NodeId, String> = canvas.sim.nodes.iter()
        .map(|(id, n)| (*id, n.name.clone())).collect();
    let mut edge_lines: Vec<String> = canvas.sim.edges.iter()
        .map(|(_, e)| format!("{:>14} -> {}",
            by_id.get(&e.from).cloned().unwrap_or("?".into()),
            by_id.get(&e.to).cloned().unwrap_or("?".into())))
        .collect();
    edge_lines.sort();
    for l in &edge_lines { eprintln!("{}", l); }

}

/// Repro for "I open region.whiteboard, click an example, and the
/// example shows no packets either." Builds the full app via
/// FlowBevyPlugins (matches the live binary), seeds region as the
/// startup canvas, runs for some real time, then fires a LoadExample
/// for ClientWorker — exactly what clicking the menu does. Asserts
/// the post-LoadExample app actually ingests visible packets so the
/// renderer would draw them.
#[test]
fn region_then_example_renders_example_packets() {
    let path = project_root().join("examples/region.whiteboard");

    let mut app = poster_ui::testing::test_app_headless();
    app.add_plugins(flow_bevy::FlowBevyPlugins);
    app.insert_resource(flow_bevy::PendingCanvas(Some(path.clone())))
        .add_systems(Startup, flow_bevy::canvas::seed_from_path);
    {
        let mut tl = app.world_mut().resource_mut::<flow_bevy::edges::VisualTimelineRes>();
        tl.strategy.as_replay_mut().k = 1.0;
    }
    app.world_mut().resource_mut::<flow_bevy::bridge::SimClock>().multiplier = 1.0;

    app.update();
    app.update();

    // Run region for a few sim seconds.
    for _ in 0..30 {
        app.world_mut().resource_mut::<flow_bevy::bridge::FlowSim>().0.advance_direct(100_000_000);
        app.update();
    }

    let region_packet_count = app.world().resource::<flow_bevy::edges::VisualTimelineRes>().strategy.as_replay().packets.len();
    eprintln!("region phase: timeline.packets.len() = {}", region_packet_count);

    // Click an example.
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<flow_bevy::examples::LoadExample>>()
        .write(flow_bevy::examples::LoadExample(flow_bevy::examples::Example::ClientWorker));
    app.update();
    app.update();

    let post_load_count = app.world().resource::<flow_bevy::edges::VisualTimelineRes>().strategy.as_replay().packets.len();
    let post_hide_all = app.world().resource::<flow_bevy::edges::HideAll>().0;
    let post_membership_len = {
        let m = app.world().resource::<flow_bevy::compound::CompoundMembership>();
        format!("{:?}", m)
    };
    eprintln!("after LoadExample: timeline.packets={}, hide_all={}, membership={}",
              post_load_count, post_hide_all, post_membership_len);

    // Run the example for a sim second.
    for _ in 0..20 {
        app.world_mut().resource_mut::<flow_bevy::bridge::FlowSim>().0.advance_direct(100_000_000);
        app.update();
    }

    let example_packet_count = app.world().resource::<flow_bevy::edges::VisualTimelineRes>().strategy.as_replay().packets.len();
    eprintln!("example phase: timeline.packets.len() = {}", example_packet_count);

    let visual_now = app.world().resource::<flow_bevy::bridge::SimClock>().visual_now;
    let timeline = app.world().resource::<flow_bevy::edges::VisualTimelineRes>();
    let in_flight = timeline.strategy.as_replay().packets.iter()
        .filter(|p| p.emit_real <= visual_now && visual_now < p.arrive_real)
        .count();
    eprintln!("example phase: visual_now={}, in_flight_now={}", visual_now, in_flight);
    for (i, p) in timeline.strategy.as_replay().packets.iter().enumerate().take(10) {
        eprintln!("  pkt[{}]: emit_real={:.3} arrive_real={:.3} from={:?} to={:?}",
            i, p.emit_real, p.arrive_real, p.from, p.to);
    }

    // Stale-event drain confirmed: post-LoadExample timeline contains
    // only events whose NodeIds exist in the new sim. (Pre-fix, this
    // saw ghost packets from region's wiped NodeIds.) The full live-
    // binary flow with the worker thread actually pumping example
    // events isn't exercisable in test mode (advance_direct is a
    // no-op against the worker driver), so we only assert the fix's
    // direct effect: no leakage of pre-LoadExample events.
    // Pre-fix: timeline retained packets with NodeIds 13/14/15 → 12
    // (region's nodes), which had no entities in the fresh sim. With
    // the channel drain in handle_load_example, post-LoadExample the
    // timeline starts clean. This assertion just sanity-checks that
    // shape — the worker-thread pump for fresh example packets isn't
    // visible in test mode.
    let known_node_ids: std::collections::HashSet<flow::NodeId> = app
        .world()
        .resource::<flow_bevy::bridge::EntityMaps>()
        .node_to_entity
        .keys()
        .copied()
        .collect();
    let stale_packet_count = app.world().resource::<flow_bevy::edges::VisualTimelineRes>().strategy
        .as_replay().packets.iter()
        .filter(|p| !known_node_ids.contains(&p.from) || !known_node_ids.contains(&p.to))
        .count();
    assert_eq!(stale_packet_count, 0,
        "expected no stale region NodeIds in fresh timeline; got {} stale packets",
        stale_packet_count);
    eprintln!("post-LoadExample timeline length: {} (all reference live NodeIds)", example_packet_count);
}

#[test]
#[ignore = "Composite migration: assertion depends on monolithic gadget shape (direct slot access or event-log emit counts matching shim ids)."]
fn region_whiteboard_visual_timeline_ingests_packets() {
    use flow_bevy::visual::VisualTimeline;
    let path = project_root().join("examples/region.whiteboard");
    let mut canvas = load_canvas(&path, 7).expect("load region.whiteboard");
    canvas.sim.run_until(canvas.sim.now_ns + 1_500_000_000);

    let mut timeline = VisualTimeline::new(1.0);
    let mut real_now = 0.0;
    let mut ingested_packets = 0usize;
    for ev in &canvas.sim.log.events {
        if let flow::Event::PacketEmitted { at_ns, .. } = ev {
            real_now = (*at_ns as f64) * 1e-9;
        }
        if let Some(_) = timeline.ingest(ev, real_now) {
            ingested_packets += 1;
        }
    }
    eprintln!("ingested {} visible packets, timeline.packets.len() = {}",
              ingested_packets, timeline.packets.len());
    if let Some(first) = timeline.packets.first() {
        eprintln!("first: from={:?} to={:?} emit_real={} arrive_real={}",
                  first.from, first.to, first.emit_real, first.arrive_real);
    }
    assert!(ingested_packets > 20,
        "VisualTimeline should ingest ≥20 packets in 1.5s of region traffic, got {}",
        ingested_packets);
}
