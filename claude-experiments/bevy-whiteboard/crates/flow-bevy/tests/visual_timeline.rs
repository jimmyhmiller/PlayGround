//! Visual timeline tests.
//!
//! Per-packet Bevy entities are gone (they were a 200k-entity tax
//! at sim load — the actual rendering is the instanced
//! `packet_cloud` material). The data they carried lives entirely
//! on the `VisualTimelineRes` resource: `(packet_id, from, to,
//! emit_real, arrive_real)` per packet, with visibility derived
//! from `SimClock.visual_now`. These tests sample that resource
//! frame-by-frame and reconstruct the same per-packet snapshots
//! the old per-entity queries produced.
//!
//! Property assertions over the captured timeline:
//!   P1 — No visible packet ever has position == world origin
//!        (unless its actual source or destination is at origin).
//!   P2 — Every visible packet's position lies on the line
//!        between its edge's from-node and to-node (within eps).
//!   P3 — Chain sequencing: in a forward chain
//!        (Gen→Router→Queue→Worker→Sink), the `first-visible
//!        real-time` of each hop is strictly greater than the
//!        prior hop's. No two consecutive hops show simultaneously.

mod common;

use bevy::prelude::*;
use common::make_app;
use flow_bevy::bridge::{EntityMaps, FlowSim, FlowNodeRef, SimClock};
use flow_bevy::edges::VisualTimelineRes;
use flow_bevy::examples::{Example, LoadExample};

// ─────────────────────────────────────────────────────────────
// Frame sampler
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PacketFrame {
    real_t: f32,
    /// Stable packet identifier from the sim's `PacketEmitted` event;
    /// replaces the old per-frame Bevy `Entity` id, with the same
    /// "uniquely identifies one logical packet" semantics. The visual
    /// timeline guarantees one record per `PacketEmitted`, so packet_id
    /// is also the right key for double-spawn / phantom-spawn checks.
    packet_id: flow::PacketId,
    visible: bool,
    from: flow::NodeId,
    to:   flow::NodeId,
    pos: Vec2,
    edge_from_pos: Vec2,
    edge_to_pos:   Vec2,
    /// 0..1 progress along the edge at `real_t`. Computed from
    /// emit_real / arrive_real — same formula the runtime uses.
    progress: f32,
    emit_real: f32,
    arrive_real: f32,
}

#[derive(Debug, Clone, Default)]
struct Timeline {
    /// Parallel vector: one entry per sampled frame. Each entry
    /// is the full snapshot of active packets that frame.
    frames: Vec<Vec<PacketFrame>>,
}

impl Timeline {
    /// All times at which a packet first appears with `visible = true`,
    /// in the order they become visible. Keyed by `packet_id`.
    #[allow(dead_code)]
    fn first_visible_times(&self) -> std::collections::HashMap<flow::PacketId, f32> {
        use std::collections::HashMap;
        let mut out: HashMap<flow::PacketId, f32> = HashMap::new();
        for frame in &self.frames {
            for p in frame {
                if p.visible && !out.contains_key(&p.packet_id) {
                    out.insert(p.packet_id, p.real_t);
                }
            }
        }
        out
    }
}

fn load(app: &mut App, ex: Example) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<LoadExample>>()
        .write(LoadExample(ex));
    app.update();
    app.update();
}

/// Advance both sim time (via SimClock.step_once_ns) AND real time
/// (via `TimeUpdateStrategy::ManualDuration`). In a headless test,
/// wall-clock delta between `app.update()` calls is ~0, so the
/// visual causality logic that reads `time.elapsed_secs()` and
/// `time.delta_secs()` needs explicit help. `ManualDuration` tells
/// Bevy's TimePlugin to advance by a fixed step each frame instead
/// of measuring wall time.
fn step_frame(app: &mut App, step_ns: u64) {
    use bevy::time::TimeUpdateStrategy;
    app.world_mut()
        .resource_mut::<SimClock>()
        .step_once_ns = Some(step_ns);
    // Match real-time advance to sim-time advance at 1× multiplier
    // (the SimClock default). Keeps visual-duration math and
    // spawn_at_real deferral consistent with what a user would see
    // at 1× playback.
    app.insert_resource(TimeUpdateStrategy::ManualDuration(
        std::time::Duration::from_nanos(step_ns),
    ));
    app.update();
}

fn sample_frame(app: &mut App) -> Vec<PacketFrame> {
    // Visual time is what the renderer animates against; SimClock
    // owns it (it's wall-clock with pause / multiplier applied) and
    // it's the single input that determines packet visibility +
    // position via the `is_visible_at` / `progress_at` formulas.
    let visual_now = app.world().resource::<SimClock>().visual_now;
    let real_t = app.world().resource::<Time>().elapsed_secs_f64();

    // Build the NodeId → world-position map by joining EntityMaps
    // with the node Transform query. Used to resolve each packet's
    // (from, to) endpoints, and to derive the per-packet position
    // by linearly interpolating start → end at `progress`.
    let world = app.world_mut();
    let maps = world.resource::<EntityMaps>().node_to_entity.clone();
    let mut q_nodes = world.query_filtered::<(Entity, &Transform), With<FlowNodeRef>>();
    let node_pos: std::collections::HashMap<flow::NodeId, Vec2> = {
        let mut out = std::collections::HashMap::new();
        let entity_to_node: std::collections::HashMap<Entity, flow::NodeId> =
            maps.iter().map(|(nid, e)| (*e, *nid)).collect();
        for (e, tf) in q_nodes.iter(world) {
            if let Some(&nid) = entity_to_node.get(&e) {
                out.insert(nid, tf.translation.truncate());
            }
        }
        out
    };

    let timeline = world.resource::<VisualTimelineRes>();
    timeline
        .strategy
        .as_replay()
        .packets
        .iter()
        .map(|pkt| {
            let src_pos = node_pos.get(&pkt.from).copied().unwrap_or(Vec2::ZERO);
            let dst_pos = node_pos.get(&pkt.to).copied().unwrap_or(Vec2::ZERO);
            let progress = pkt.progress_at(visual_now);
            // Interpolated world position — what the GPU vertex
            // shader computes for visible packets. For
            // hidden-but-not-yet-emitted packets this is just the
            // source position (clamped progress = 0).
            let pos = src_pos.lerp(dst_pos, progress);
            PacketFrame {
                real_t: real_t as f32,
                packet_id: pkt.packet_id,
                visible: pkt.is_visible_at(visual_now),
                from: pkt.from,
                to: pkt.to,
                pos,
                edge_from_pos: src_pos,
                edge_to_pos: dst_pos,
                progress,
                emit_real: pkt.emit_real as f32,
                arrive_real: pkt.arrive_real as f32,
            }
        })
        .collect()
}

fn capture_timeline(app: &mut App, step_ns: u64, n_frames: usize) -> Timeline {
    let mut t = Timeline::default();
    for _ in 0..n_frames {
        step_frame(app, step_ns);
        t.frames.push(sample_frame(app));
    }
    t
}

// ─────────────────────────────────────────────────────────────
// P1 — no visible packet at world origin (unless it's a source)
// ─────────────────────────────────────────────────────────────

#[test]
fn p1_no_visible_packet_at_world_origin() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);
    let tl = capture_timeline(&mut app, 30_000_000, 60);

    for frame in &tl.frames {
        for p in frame {
            if !p.visible { continue; }
            // If either endpoint of this packet's edge is at
            // origin, then a visible-at-origin packet is
            // legitimate. Otherwise it's the bug.
            let on_origin_endpoint =
                p.edge_from_pos.length() < 1.0 || p.edge_to_pos.length() < 1.0;
            if !on_origin_endpoint && p.pos.length() < 1.0 {
                panic!(
                    "visible packet at world origin but neither edge endpoint is: \
                     t_real={:.3}, from={:?}, to={:?}, pos={:?}, \
                     edge_from={:?}, edge_to={:?}, progress={:.3}, emit_real={:.3}, arrive_real={:.3}",
                    p.real_t, p.from, p.to, p.pos,
                    p.edge_from_pos, p.edge_to_pos, p.progress, p.emit_real, p.arrive_real,
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// P1b — STRONGER: no packet (visible or not) ever has position
//       at world origin unless an endpoint is there. Catches
//       the bug-hypothesis where `Visibility::Hidden` fails to
//       suppress a render, leaving a dot at screen center. Even
//       hidden packets should sit at a meaningful spot (their
//       source node), not (0, 0).
// ─────────────────────────────────────────────────────────────

#[test]
fn p1b_no_packet_ever_at_world_origin() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);
    let tl = capture_timeline(&mut app, 30_000_000, 60);

    for frame in &tl.frames {
        for p in frame {
            let on_origin_endpoint =
                p.edge_from_pos.length() < 1.0 || p.edge_to_pos.length() < 1.0;
            if !on_origin_endpoint && p.pos.length() < 1.0 {
                panic!(
                    "packet at world origin regardless of visibility: \
                     t_real={:.3}, visible={}, from={:?}, to={:?}, \
                     pos={:?}, edge_from={:?}, edge_to={:?}, \
                     progress={:.3}, emit_real={:.3}, arrive_real={:.3}",
                    p.real_t, p.visible, p.from, p.to, p.pos,
                    p.edge_from_pos, p.edge_to_pos, p.progress, p.emit_real, p.arrive_real,
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// P2 — visible packet is on its edge's line (with tolerance)
// ─────────────────────────────────────────────────────────────

#[test]
fn p2_visible_packets_on_edge_line() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);
    let tl = capture_timeline(&mut app, 30_000_000, 60);

    for frame in &tl.frames {
        for p in frame {
            if !p.visible { continue; }
            // Point-to-segment distance. The packet should sit
            // ON the segment from edge_from_pos to edge_to_pos.
            // We allow a small eps for floating-point drift.
            let dist = point_to_segment_distance(p.pos, p.edge_from_pos, p.edge_to_pos);
            assert!(
                dist < 5.0,
                "visible packet is {} px off its edge: t_real={:.3}, \
                 from={:?} → to={:?} ({:?} → {:?}), pos={:?}",
                dist, p.real_t, p.from, p.to, p.edge_from_pos, p.edge_to_pos, p.pos,
            );
        }
    }
}

fn point_to_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-6 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    let proj = a + ab * t;
    (p - proj).length()
}

// ─────────────────────────────────────────────────────────────
// P3 — chain sequencing at the per-packet level.
//
// The old P3 checked "first-visible time" per edge layer across
// ALL packets in the scenario, but that's not a well-formed
// invariant under F1: unrelated initial emits (e.g. a worker
// firing a probe packet toward its sink at t≈0 before the
// request/response cycle ever starts) legitimately make a later
// layer's first-packet appear before an earlier layer's.
//
// What's actually invariant is per-packet causality: for any
// sim event B causally depending on event A
// (`B.at_ns >= A.arrives_at_ns`), the visual obeys
// `B.emit_real >= A.arrive_real`. That's a property of
// `VisualTimeline::ingest` and is exhaustively proven in
// `visual_properties::causality_preserved_under_scaling`. No
// need to re-check it through Bevy here.
#[test]
#[ignore = "Replaced by visual_properties::causality_preserved_under_scaling"]
fn p3_chain_hops_not_simultaneously_visible() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);

    // Capture enough frames for multiple hop layers to become
    // visible. With per-node sequencing + 3 concurrent streams
    // per layer, each layer's first-visible is ~(dur+dwell)*3 =
    // ~2s after the previous layer, so 5s covers 3 layers
    // comfortably.
    let tl = capture_timeline(&mut app, 50_000_000, 100); // 50 ms × 100 = 5 s

    // Group packets by (from, to). For each pair, record when any
    // packet on it first becomes visible.
    use std::collections::HashMap;
    let mut first_visible_per_pair: HashMap<(flow::NodeId, flow::NodeId), f32> = HashMap::new();
    for frame in &tl.frames {
        for p in frame {
            if !p.visible { continue; }
            first_visible_per_pair.entry((p.from, p.to))
                .and_modify(|t| if p.real_t < *t { *t = p.real_t })
                .or_insert(p.real_t);
        }
    }

    // Classify (from, to) pairs into hop layers.
    let hop_layers = {
        let sim = &app.world().resource::<FlowSim>();
        let mut gens = Vec::new();
        let mut router = None;
        let mut queues = Vec::new();
        let mut workers = Vec::new();
        let mut sinks = Vec::new();
        for (id, n) in sim.nodes.iter() {
            if      n.name.starts_with("Gen_")    { gens.push(*id); }
            else if n.name.starts_with("Router_") { router = Some(*id); }
            else if n.name.starts_with("Queue_")  { queues.push(*id); }
            else if n.name.starts_with("Worker_") { workers.push(*id); }
            else if n.name.starts_with("Sink_")   { sinks.push(*id); }
        }
        let router = router.expect("no router");
        let mut gen_to_router:    Vec<(flow::NodeId, flow::NodeId)> = Vec::new();
        let mut router_to_queue:  Vec<(flow::NodeId, flow::NodeId)> = Vec::new();
        let mut queue_to_worker:  Vec<(flow::NodeId, flow::NodeId)> = Vec::new();
        let mut worker_to_sink:   Vec<(flow::NodeId, flow::NodeId)> = Vec::new();
        for g in &gens { gen_to_router.push((*g, router)); }
        for q in &queues { router_to_queue.push((router, *q)); }
        for q in &queues { for w in &workers { queue_to_worker.push((*q, *w)); } }
        for w in &workers { for s in &sinks { worker_to_sink.push((*w, *s)); } }
        vec![
            ("gen→router", gen_to_router),
            ("router→queue", router_to_queue),
            ("queue→worker", queue_to_worker),
            ("worker→sink", worker_to_sink),
        ]
    };

    let layer_first_visible: Vec<(&str, Option<f32>, Option<(flow::NodeId, flow::NodeId)>)> = hop_layers.iter()
        .map(|(name, pairs)| {
            let mut best: Option<((flow::NodeId, flow::NodeId), f32)> = None;
            for pair in pairs {
                if let Some(&t) = first_visible_per_pair.get(pair) {
                    if best.map_or(true, |(_, bt)| t < bt) {
                        best = Some((*pair, t));
                    }
                }
            }
            (*name, best.map(|(_, t)| t), best.map(|(p, _)| p))
        })
        .collect();

    // Strict ordering: each non-None layer must have a first-
    // visible time STRICTLY LATER than the previous non-None
    // layer's. Under F1 this falls out of sim causality — the
    // sim emits the outgoing packet at or after the incoming's
    // arrival, and we map sim time → real time monotonically.
    let mut prev_name: Option<&str> = None;
    let mut prev_t: Option<f32> = None;
    for (name, t, pair) in &layer_first_visible {
        let Some(t) = t else { continue; };
        if let (Some(pn), Some(pt)) = (prev_name, prev_t) {
            assert!(
                *t > pt,
                "layer `{}` on {:?} became visible at {:.3}, but prior layer `{}` was at {:.3}. Full layers: {:?}",
                name, pair, t, pn, pt, layer_first_visible,
            );
        }
        prev_name = Some(*name);
        prev_t = Some(*t);
    }

    let showed = layer_first_visible.iter().filter(|(_, t, _)| t.is_some()).count();
    assert!(
        showed >= 2,
        "expected at least 2 hop layers to become visible, got {}: {:?}",
        showed, layer_first_visible,
    );
}

// ─────────────────────────────────────────────────────────────
// P6 — liveness: as long as the sim is emitting packets,
//      visuals keep flowing. The canvas can't go silent for
//      long stretches while the sim is still active.
// ─────────────────────────────────────────────────────────────

#[test]
fn p6_visuals_keep_flowing_while_sim_emits() {
    use flow::event::Event;
    use flow_bevy::edges::VisualTimelineRes;

    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);

    // Run 10 real seconds.
    for _ in 0..200 {
        step_frame(&mut app, 50_000_000);
    }

    // Liveness at the formalism level: every sim data emit
    // maps to exactly one `VisualPacket`, so the timeline's
    // packets vector (before GC) mirrors the sim log. Because
    // `gc_timeline` trims history older than 2s, we don't
    // check equality — we check that the TOTAL emits over the
    // run correspond to packets that the timeline has either
    // already gc'd (seen + removed) or still holds.
    //
    // Concretely: the live app NEVER produces silent gaps if
    // the sim is producing emits, because ingestion is 1:1
    // and happens every frame. So the assertion is: "did the
    // timeline receive at least as many packets as we expect
    // from the sim's data-emit stream in the last 2s window"
    // (the GC retention window).
    let sim = &app.world().resource::<FlowSim>();
    let now_ns = sim.now_ns;
    let window_ns: u64 = 2_000_000_000; // 2s
    let window_start_ns = now_ns.saturating_sub(window_ns);

    let recent_data_emits: usize = sim.log.iter().filter(|ev| match ev {
        Event::PacketEmitted { from, to, payload, at_ns, .. } => {
            *at_ns >= window_start_ns
                && from != to
                && !matches!(payload,
                    flow::Value::Variant { tag, .. } if tag == "pull" || tag == "wake")
        }
        _ => false,
    }).count();

    let total_data_emits: usize = sim.log.iter().filter(|ev| matches!(ev,
        Event::PacketEmitted { from, to, payload, .. }
        if from != to && !matches!(payload,
            flow::Value::Variant { tag, .. } if tag == "pull" || tag == "wake"
        )
    )).count();
    assert!(total_data_emits > 50,
        "scenario produced only {} data emits over 10s — test misconfigured",
        total_data_emits);

    let timeline_len = app.world().resource::<VisualTimelineRes>().strategy.as_replay().packets.len();
    assert!(
        timeline_len >= recent_data_emits,
        "visual timeline holds {} packets but sim emitted {} data packets in \
         the last 2s — visual layer is dropping events.",
        timeline_len, recent_data_emits,
    );
}

// ─────────────────────────────────────────────────────────────
// P4s — STRICT: exactly ONE VisualPacket per PacketEmitted,
//       correlated by `packet_id`. Catches double-ingest bugs
//       and phantom packets.
//
// Used to check this through `Query<&TravelingPacket>` — one
// entity per packet — and assert no packet_id appeared on two
// entities ever. Now reads `VisualTimelineRes` directly:
// `timeline.packets` is the source of truth (was the source of
// truth all along; the entities were a stale mirror), so the
// invariant becomes "no packet_id appears twice in
// `timeline.packets`, and every packet_id corresponds to a
// real sim emit."
// ─────────────────────────────────────────────────────────────

#[test]
fn p4s_one_visual_per_emit_packet_id() {
    use flow::event::Event;
    use std::collections::{HashMap, HashSet};

    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);

    // Per-frame: each packet_id must appear at most ONCE in
    // `timeline.packets`. Also accumulate across frames — even if
    // a packet_id only appears once per frame, it must NEVER be
    // re-introduced after being GC'd (the spawn-despawn-respawn
    // double-ingest bug).
    let mut ever_seen_packet_ids: HashSet<flow::PacketId> = HashSet::new();
    let mut prev_seen_packet_ids: HashSet<flow::PacketId> = HashSet::new();

    for _frame in 0..60 {
        step_frame(&mut app, 30_000_000);

        // Per-frame count by packet_id.
        let timeline = app.world().resource::<VisualTimelineRes>().strategy.as_replay();
        let mut per_id: HashMap<flow::PacketId, usize> = HashMap::new();
        for p in &timeline.packets {
            *per_id.entry(p.packet_id).or_insert(0) += 1;
        }
        for (id, n) in &per_id {
            assert!(
                *n <= 1,
                "at this frame, packet_id {:?} appears {} times in \
                 the visual timeline — a duplicate was ingested",
                id,
                n,
            );
        }
        let this_frame: HashSet<flow::PacketId> = per_id.keys().copied().collect();
        // Detect re-ingest: a packet_id present this frame, absent the
        // previous frame, AND already seen in some earlier frame.
        for id in this_frame.difference(&prev_seen_packet_ids) {
            assert!(
                !ever_seen_packet_ids.contains(id),
                "packet_id {:?} was re-ingested into the visual timeline \
                 after having been previously GC'd — each sim emit should \
                 produce at most one visual ever",
                id,
            );
            ever_seen_packet_ids.insert(*id);
        }
        prev_seen_packet_ids = this_frame;
    }

    // Every packet_id we ever saw corresponds to a real, non-filtered
    // sim emit. Catches phantom ingests.
    let sim = &app.world().resource::<FlowSim>();
    let emit_ids: HashSet<flow::PacketId> = sim.log.iter()
        .filter_map(|ev| if let Event::PacketEmitted { packet, from, to, payload, .. } = ev {
            if from == to { return None; }
            if let flow::Value::Variant { tag, .. } = payload {
                if tag == "pull" || tag == "wake" { return None; }
            }
            Some(*packet)
        } else { None })
        .collect();
    for id in &ever_seen_packet_ids {
        assert!(
            emit_ids.contains(id),
            "visual timeline ingested packet_id {:?} but no non-filtered \
             sim PacketEmitted event has that id — phantom packet",
            id
        );
    }
}

// ─────────────────────────────────────────────────────────────
// P5 — DELETED. The original assertion ("at most one visible
// packet per destination per instant") over-constrained the
// faithful visualization. If the sim has three generators
// emitting concurrent packets into the same router, F1 shows
// three dots arriving together — and that's correct, it's what
// the sim is doing. The user explicitly accepts concurrent
// arrivals at a node from distinct sources.
//
// The invariant that SHOULD hold is the one V2 proves in the
// pure tests: one `VisualPacket` per `PacketEmitted`, no
// spontaneous 1-in-N-out duplication. That's the behavior the
// user flagged ("queue CREATED OUT OF THIN AIR multiple
// packets"), and it's covered.
// ─────────────────────────────────────────────────────────────

#[test]
#[ignore = "Replaced by visual_properties::v2_each_packet_has_one_matching_emit"]
fn p5_at_most_one_incoming_per_node_per_instant() {
    use std::collections::HashMap;

    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);
    let tl = capture_timeline(&mut app, 30_000_000, 120); // 120 × 30ms = 3.6 s

    // Destination node of a packet: edge_to_pos in forward mode,
    // edge_from_pos in reverse mode. In snapshot terms, it's
    // `edge_to_pos` because I oriented `(edge_from_pos,
    // edge_to_pos)` by reversed in sample_frame.
    //
    // Group visible packets per frame by destination; any frame
    // with >1 visible packet sharing a destination is a violation.
    for (frame_i, frame) in tl.frames.iter().enumerate() {
        let mut per_dst: HashMap<(i32, i32), Vec<&PacketFrame>> = HashMap::new();
        for p in frame {
            if !p.visible { continue; }
            let key = (p.edge_to_pos.x as i32, p.edge_to_pos.y as i32);
            per_dst.entry(key).or_default().push(p);
        }
        for (dst, pkts) in &per_dst {
            if pkts.len() > 1 {
                let details: Vec<String> = pkts.iter().map(|p| format!(
                    "from={:?}→to={:?} progress={:.3} emit={:.3} arrive={:.3} pos={:?}",
                    p.from, p.to, p.progress, p.emit_real, p.arrive_real, p.pos,
                )).collect();
                panic!(
                    "frame {} at t_real≈{:.3}s: {} visible packets heading to dst {:?}:\n  {}",
                    frame_i,
                    frame.first().map(|p| p.real_t).unwrap_or(0.0),
                    pkts.len(), dst,
                    details.join("\n  "),
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// P4 — (loose) visual-timeline packet count reasonable vs emits
// ─────────────────────────────────────────────────────────────

#[test]
fn p4_packet_entity_count_tracks_emissions() {
    use flow::event::Event;

    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);

    // Run the sim long enough for a few round-trips.
    for _ in 0..30 {
        step_frame(&mut app, 30_000_000);
    }

    // Count emitted data packets (exclude pull/wake tick/wake).
    let sim = &app.world().resource::<FlowSim>();
    let mut emit_count = 0usize;
    for ev in sim.log.iter() {
        if let Event::PacketEmitted { from, to, payload, .. } = ev {
            // Skip self-loops (tick, wake, done-to-self).
            if from == to { continue; }
            if let flow::Value::Variant { tag, .. } = payload {
                if tag == "pull" || tag == "wake" { continue; }
            }
            emit_count += 1;
        }
    }

    // Live = records currently in the visual timeline. The timeline
    // GC trims arrived packets older than its retention window
    // (`gc_before`), so this snapshot is bounded by emit_count + a
    // small slop for in-flight + just-arrived. We assert no spurious
    // extra ingests.
    let live = app.world().resource::<VisualTimelineRes>().strategy.as_replay().packets.len();
    assert!(
        live <= emit_count + 5,
        "visual timeline holds {} packets but the sim emitted only {} \
         data packets — something is ingesting extras",
        live, emit_count,
    );
}
