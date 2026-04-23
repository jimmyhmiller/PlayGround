//! Visual timeline tests.
//!
//! Previous tests have checked the `Visibility` component on
//! `TravelingPacket` entities at a moment in time — but that's
//! too narrow. Bugs the user reported (*"packets spawn out of
//! Router/Queue at frame 0"*, *"extra green appearing from thin
//! air"*) live in the intersection of component state and
//! rendered position: a packet hidden at world-origin still
//! renders a dot at screen-center if the visibility propagation
//! misbehaves, and a visible packet at (0,0) would look like it
//! "materialized in the middle of the canvas."
//!
//! The tests here sample frame-by-frame:
//!   - how many `TravelingPacket` entities exist,
//!   - which are currently Visible,
//!   - their world-space position,
//!   - what position they *should* be at given edge geometry.
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
use flow_bevy::bridge::{EntityMaps, FlowSim, SimClock};
use flow_bevy::edges::TravelingPacket;
use flow_bevy::examples::{Example, LoadExample};

// ─────────────────────────────────────────────────────────────
// Frame sampler
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PacketFrame {
    real_t: f32,
    entity: Entity,
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
    /// All times at which the entity first appears with
    /// `visible = true`, in the order they become visible.
    #[allow(dead_code)]
    fn first_visible_times(&self) -> std::collections::HashMap<Entity, f32> {
        use std::collections::HashMap;
        let mut out: HashMap<Entity, f32> = HashMap::new();
        for frame in &self.frames {
            for p in frame {
                if p.visible && !out.contains_key(&p.entity) {
                    out.insert(p.entity, p.real_t);
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
    let real_t = app.world().resource::<Time>().elapsed_secs_f64();

    let world = app.world_mut();
    let maps = world.resource::<EntityMaps>().node_to_entity.clone();
    let node_pos: std::collections::HashMap<flow::NodeId, Vec2> = {
        let mut out = std::collections::HashMap::new();
        let mut q = world.query::<(Entity, &Transform)>();
        for (e, tf) in q.iter(world) {
            for (nid, ent) in maps.iter() {
                if *ent == e {
                    out.insert(*nid, tf.translation.truncate());
                }
            }
        }
        out
    };

    let mut out = Vec::new();
    let mut q = world.query::<(Entity, &TravelingPacket, &Transform, &Visibility)>();
    for (e, pkt, tf, vis) in q.iter(world) {
        let src_pos = node_pos.get(&pkt.from).copied().unwrap_or(Vec2::ZERO);
        let dst_pos = node_pos.get(&pkt.to).copied().unwrap_or(Vec2::ZERO);
        let denom = (pkt.arrive_real - pkt.emit_real).max(1e-9);
        let progress = ((real_t - pkt.emit_real) / denom).clamp(0.0, 1.0) as f32;
        out.push(PacketFrame {
            real_t: real_t as f32,
            entity: e,
            visible: matches!(vis, Visibility::Visible | Visibility::Inherited),
            from: pkt.from,
            to: pkt.to,
            pos: tf.translation.truncate(),
            edge_from_pos: src_pos,
            edge_to_pos:   dst_pos,
            progress,
            emit_real: pkt.emit_real as f32,
            arrive_real: pkt.arrive_real as f32,
        });
    }
    out
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
        let sim = &app.world().resource::<FlowSim>().sim;
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
    let sim = &app.world().resource::<FlowSim>().sim;
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

    let timeline_len = app.world().resource::<VisualTimelineRes>().0.packets.len();
    assert!(
        timeline_len >= recent_data_emits,
        "visual timeline holds {} packets but sim emitted {} data packets in \
         the last 2s — visual layer is dropping events.",
        timeline_len, recent_data_emits,
    );
}

// ─────────────────────────────────────────────────────────────
// P4s — STRICT: exactly ONE TravelingPacket per PacketEmitted
//       (correlated by packet_id). Catches double-spawn bugs
//       that the loose count-based test below misses.
//
// Why the weaker test passes while this can fail: `p4` allows
// `live_count <= emit_count + 5`. If *one* extra packet spawns
// per emit on a particular edge, `live` roughly doubles —
// still well within slop when traffic is light. And `live`
// is a snapshot: despawned packets aren't counted. This test
// catches ANY extra spawn by tracking `packet_id` across the
// whole timeline.
// ─────────────────────────────────────────────────────────────

#[test]
fn p4s_one_visual_per_emit_packet_id() {
    use flow::event::Event;
    use std::collections::{HashMap, HashSet};

    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);

    // Per-frame: at any single instant, each packet_id must map
    // to at most ONE live TravelingPacket entity. Also track the
    // CUMULATIVE set of (packet_id, entity) pairs ever observed
    // — any packet_id seen on two distinct entities EVER is a
    // double-spawn, even if one has already despawned before the
    // test's end state snapshot would reveal it.
    let mut ever_seen_entities_per_id: HashMap<flow::PacketId, HashSet<Entity>> =
        HashMap::new();

    for _frame in 0..60 {
        step_frame(&mut app, 30_000_000);

        // Current live set.
        let live_per_id: HashMap<flow::PacketId, Vec<Entity>> = {
            let world = app.world_mut();
            let mut q = world.query::<(Entity, &TravelingPacket)>();
            let mut m: HashMap<flow::PacketId, Vec<Entity>> = HashMap::new();
            for (e, p) in q.iter(world) {
                m.entry(p.packet_id).or_default().push(e);
            }
            m
        };

        for (id, ents) in &live_per_id {
            assert!(
                ents.len() <= 1,
                "at this frame, packet_id {:?} is held by {} simultaneous \
                 live TravelingPacket entities — a duplicate was spawned",
                id, ents.len(),
            );
            // Accumulate across frames.
            let seen = ever_seen_entities_per_id.entry(*id).or_default();
            for e in ents { seen.insert(*e); }
        }
    }

    // Over the entire run, each packet_id should only EVER have
    // been held by one entity. Catches the spawn-then-despawn-
    // then-respawn-with-same-id double-spawn scenario.
    for (id, ents) in &ever_seen_entities_per_id {
        assert_eq!(
            ents.len(), 1,
            "over the run, packet_id {:?} was held by {} distinct \
             entities (times it was visually spawned). Each sim emit \
             should produce at most one visual.",
            id, ents.len(),
        );
    }

    // Every live packet_id corresponds to a real sim emit.
    let sim = &app.world().resource::<FlowSim>().sim;
    let emit_ids: HashSet<flow::PacketId> = sim.log.iter()
        .filter_map(|ev| if let Event::PacketEmitted { packet, from, to, payload, .. } = ev {
            if from == to { return None; }
            if let flow::Value::Variant { tag, .. } = payload {
                if tag == "pull" || tag == "wake" { return None; }
            }
            Some(*packet)
        } else { None })
        .collect();
    for id in ever_seen_entities_per_id.keys() {
        assert!(
            emit_ids.contains(id),
            "TravelingPacket was spawned with packet_id {:?} but no \
             non-filtered sim PacketEmitted event has that id — \
             phantom packet",
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
// P4 — (loose) live TravelingPacket count reasonable vs emits
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
    let sim = &app.world().resource::<FlowSim>().sim;
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

    // Count total TravelingPacket entities ever spawned. Since
    // some may have been despawned already (t>=1), we rely on
    // the throttle: spawns are coalesced when MIN_SPAWN_INTERVAL
    // < gap. So entities-ever-spawned ≤ emit_count. We assert the
    // weaker "no spurious ghost spawns" — live packets count is
    // reasonable (not wildly > emit count).
    let world = app.world_mut();
    let mut q = world.query::<&TravelingPacket>();
    let live = q.iter(world).count();
    assert!(
        live <= emit_count + 5,
        "live TravelingPacket count ({}) wildly exceeds emitted data-packet count ({}) \
         — something is spawning extras",
        live, emit_count,
    );
}
