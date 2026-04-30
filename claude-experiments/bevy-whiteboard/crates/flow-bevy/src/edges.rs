//! Edges: connect mode, arrow rendering, and the Bevy shell around the
//! pure `visual::VisualTimeline`.
//!
//! The packet pipeline here is deliberately thin. All the "when is a
//! packet visible, where is it on the edge" logic lives in
//! [`crate::visual`] as a pure data type so it can be property-tested
//! without a Bevy world. This file only does three things with packets:
//!
//!   1. `ingest_new_events` — every frame, feed `NewEvents` into the
//!      timeline resource and spawn a Bevy entity for each new
//!      `VisualPacket`.
//!   2. `sync_packet_transforms` — each frame, read the timeline's
//!      answer for every live entity (visible? progress?) and write
//!      that into the entity's `Transform` / `Visibility`.
//!   3. `despawn_arrived_packets` — once real time passes a packet's
//!      `arrive_real`, remove the entity.
//!
//! No sequencing, FIFO, dwell, backlog cap, or throttle. Causality is
//! inherited from the sim: if the sim emits an outgoing packet at or
//! after an incoming packet's delivery, the visuals will too (because
//! we just scale sim time to real time).

use bevy::asset::RenderAssetUsages;
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;
use bevy::camera::visibility::NoFrustumCulling;
use flow::EdgeId;

use crate::bridge::{EntityMaps, FlowEdgeRef, NewEvents};
use crate::sim_driver::{SimDriverRes, SimSnapshotRes};
use crate::camera::{MainCamera, cursor_to_world};
use crate::nodes::NodeKind;
use crate::theme::Theme;
use crate::tool::{ActiveTool, Tool};
use crate::visual::{Strategy, StrategyKind, VisualStrategy, VisualTimeline};

pub struct EdgesPlugin;
impl Plugin for EdgesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConnectState>()
            .init_resource::<HiddenEdges>()
            .init_resource::<HideAll>()
            .init_resource::<RewindEpochSeen>()
            .insert_resource(VisualTimelineRes(initial_strategy()))
            .add_systems(Startup, spawn_edge_mesh)
            .add_systems(Update, (
                handle_connect_click,
                toggle_hide_all,
                draw_edges,
                apply_rewind_reset,
                ingest_new_events,
                gc_timeline,
            ).chain());
    }
}

/// Last `RenderSnapshot::rewind_epoch` the visual layer has consumed.
/// Compared each frame against the live snapshot so the visual
/// timeline can detect rewinds and clear derived state.
#[derive(Resource, Default)]
pub struct RewindEpochSeen(pub u64);

/// When `rewind_epoch` bumps — either a pure rewind or a topology
/// reset — recompute the visual layer's state from the sim's event
/// log. The visual timeline is *derived* from the event sequence:
/// given the same events ingested in sim-time order with the right
/// `real_now` for each, the strategy produces identical visual
/// records to the live ingestion. We rely on this rather than
/// stashing visual records across rewinds, because the strategies'
/// internal state (rate-sampling timestamps, causal clamps, etc.)
/// can't be partially-rewound coherently — a fresh derivation is
/// the only thing that produces pixel-identical output.
///
/// The time mapping is:
///
/// ```text
/// synth_real_now = visual_now + (at_ns - sim_now_ns) * 1e-9
/// ```
///
/// `visual_now` and `sim_now` advance in lockstep (both scaled by
/// `multiplier × wall_dt`), so an event with `at_ns = sim_now_ns`
/// lands at exactly the current `visual_now`; events further in the
/// past land at the wall-clock time they would have had when the
/// sim first reached them. **No `k` factor** — `k` only stretches
/// per-packet animation duration via `arrive_real - emit_real`, not
/// the temporal anchor at which each event is ingested.
///
/// On topology resets `replay_events` is empty (the new sim hasn't
/// emitted anything yet), so the result is just "reset". Also
/// rebases `visual_offset` so `visual_now` doesn't snap backwards
/// when `sim.now_ns` jumps to 0 on a reset.
fn apply_rewind_reset(
    snapshot: Res<SimSnapshotRes>,
    mut clock: ResMut<crate::bridge::SimClock>,
    mut seen: ResMut<RewindEpochSeen>,
    mut events: ResMut<NewEvents>,
    mut timeline: ResMut<VisualTimelineRes>,
) {
    let cur = snapshot.0.rewind_epoch;
    if cur == seen.0 {
        return;
    }
    seen.0 = cur;
    events.0.clear();
    timeline.reset();
    // Re-anchor visual_now to whatever it was just before the
    // reset, so `sync_visual_now_from_sim` produces a consistent
    // value next frame even if `sim.now_ns` jumped (e.g. a
    // topology reset to 0 or a rewind from 30s back to 5s).
    clock.visual_offset = clock.visual_now - snapshot.0.now_ns as f64 * 1e-9;

    let visual_now = clock.visual_now;
    let sim_now_ns = snapshot.0.now_ns;
    for ev in snapshot.0.replay_events.iter() {
        let at_ns = match ev {
            flow::Event::PacketEmitted { at_ns, .. } => *at_ns,
            _ => continue,
        };
        // Map sim time → wall clock with the lockstep relationship
        // (no `k`). For at_ns == sim_now_ns this gives `visual_now`
        // exactly; for older events it walks backwards in real time
        // by the same number of seconds the event is older in sim.
        let synth_real_now = visual_now + (at_ns as i64 - sim_now_ns as i64) as f64 * 1e-9;
        timeline.ingest(ev, synth_real_now);
    }
}

/// Global "hide everything visual" toggle. When on, `draw_edges`
/// skips all edge arrows and the packet-cloud renderer zeroes its
/// active count.
///
/// Toggled by the `H` keyboard hotkey. Designed for stress-test
/// canvases (Game of Life and similar) where thousands of arrows
/// drown out the actual state.
#[derive(Resource, Default)]
pub struct HideAll(pub bool);

fn toggle_hide_all(keys: Res<ButtonInput<KeyCode>>, mut hide: ResMut<HideAll>) {
    if keys.just_pressed(KeyCode::KeyH) {
        hide.0 = !hide.0;
    }
}

/// Build the initial visual strategy. `FLOW_BEVY_VISUAL_STRATEGY`
/// (case-insensitive `replay` / `rate-sampled` / `rate_sampled`)
/// picks the variant for one-off A/B comparisons without
/// recompiling. Default: Replay (the historical behavior).
fn initial_strategy() -> Strategy {
    let kind = std::env::var("FLOW_BEVY_VISUAL_STRATEGY")
        .ok()
        .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "replay" => Some(StrategyKind::Replay),
            "rate-sampled" | "rate_sampled" | "ratesampled" => Some(StrategyKind::RateSampled),
            "drop-orphans" | "drop_orphans" | "droporphans" => Some(StrategyKind::DropOrphans),
            "causal-rate" | "causal_rate" | "causalrate" |
            "causal-rate-sampled" | "causal_rate_sampled" => Some(StrategyKind::CausalRateSampled),
            "bundle" | "bundle-summarized" | "bundlesummarized" => Some(StrategyKind::BundleSummarized),
            _ => None,
        })
        .unwrap_or(StrategyKind::Replay);
    Strategy::new_of_kind(kind, VisualTimeline::K_DEFAULT)
}

/// Bevy wrapper around the visual `Strategy` so it can live as a
/// `Resource`. The underlying `Strategy` (and the strategies it
/// dispatches to) deliberately has no Bevy dependency.
///
/// Deref still targets the trait so existing call sites can write
/// `timeline.visible_at(now)` etc. without caring which concrete
/// strategy is in play. Tests that need Replay-specific fields
/// (`packets`, `node_arrivals`) reach in via `.0.as_replay()`.
#[derive(Resource)]
pub struct VisualTimelineRes(pub Strategy);

impl std::ops::Deref for VisualTimelineRes {
    type Target = dyn VisualStrategy;
    fn deref(&self) -> &(dyn VisualStrategy + 'static) { &self.0 }
}
impl std::ops::DerefMut for VisualTimelineRes {
    fn deref_mut(&mut self) -> &mut (dyn VisualStrategy + 'static) { &mut self.0 }
}

/// Sim edges that exist solely for routing — not drawn, not visualized.
/// When the user establishes a pull relationship, we create a forward
/// "pull signal" edge *and* a reverse "data" edge under the hood. Only
/// one of them represents what the user drew; the other is hidden here
/// so it doesn't clutter the canvas with a mirror arrow.
///
/// Visual packets don't consult this set — they interpolate between
/// `from`/`to` node positions directly, which coincides spatially
/// whether the edge is rendered or hidden.
#[derive(Resource, Default)]
pub struct HiddenEdges {
    pub set: std::collections::HashSet<EdgeId>,
}

/// First click in Connect mode stashes the source node entity; second
/// click picks the target and creates an edge.
#[derive(Resource, Default)]
pub struct ConnectState {
    pub source: Option<Entity>,
}

/// If the payload is `packet(Int(slot))` or `req(Int(slot))`, extract
/// the slot index. Returns `None` for nil payloads or other shapes —
/// caller falls back to the emitter's colour.
pub fn payload_slot(payload: &flow::Value) -> Option<usize> {
    let inner = match payload {
        flow::Value::Variant { tag, payload } if tag == "packet" || tag == "req" => payload,
        _ => return None,
    };
    match inner.as_ref() {
        flow::Value::Int(i) if *i >= 0 => Some(*i as usize),
        _ => None,
    }
}

/// Compute the render colour for a traveling packet.
///
/// Resolution order:
///   1. Payload colour: if the payload is `packet(Int(slot))` or
///      `req(Int(slot))`, use `palette[slot]`. This lets a data
///      packet's colour persist end-to-end through routers, queues,
///      and workers — what the sink receives is the colour of
///      whoever originated the stream, not the last hop.
///   2. Emitter colour: fall back to the emitter's `NodeColors` entry.
///      Used for payloads with no slot tag (e.g. `pull(NodeRef)`,
///      which is a control signal, not data).
///   3. Accent: final fallback for fully untyped payloads.
///
/// Out-of-range slot indices clamp to the last palette entry rather
/// than panicking — palette size can change with theme swaps.
pub fn packet_color(
    payload: &flow::Value,
    emitter_color: Option<Color>,
    palette: &[Color],
    accent: Color,
) -> Color {
    if let Some(slot) = payload_slot(payload) {
        if !palette.is_empty() {
            return palette[slot.min(palette.len() - 1)];
        }
    }
    emitter_color.unwrap_or(accent)
}

// ---------------- connect mode ----------------

fn handle_connect_click(
    buttons: Res<ButtonInput<MouseButton>>,
    active: Res<ActiveTool>,
    mut connect: ResMut<ConnectState>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    nodes: Query<(Entity, &Transform, &NodeKind, &crate::nodes::BodyShape), With<crate::bridge::FlowNodeRef>>,
    mut maps: ResMut<EntityMaps>,
    mut driver: ResMut<SimDriverRes>,
    mut hidden: ResMut<HiddenEdges>,
    mut commands: Commands,
    ui: Query<&Interaction>,
) {
    if !matches!(active.0, Tool::Connect) { return; }
    if !buttons.just_pressed(MouseButton::Left) { return; }
    if poster_ui::pointer_over_ui(&ui) { return; }
    let Some(world) = cursor_to_world(&windows, &cams) else { return; };

    use crate::nodes::hit_size;
    let hit = nodes.iter().find(|(_, tf, _, shape)| {
        let half = hit_size(shape) * 0.5;
        let p = tf.translation.truncate();
        (world - p).abs().cmple(half).all()
    }).map(|(e, _, _, _)| e);
    let Some(entity) = hit else { return; };

    if let Some(src) = connect.source.take() {
        let Some(&from_nid) = maps.entity_to_node.get(&src) else { return; };
        let Some(&to_nid) = maps.entity_to_node.get(&entity) else { return; };
        let from_kind = nodes.get(src).map(|(_, _, k, _)| k.0).ok();
        let to_kind = nodes.get(entity).map(|(_, _, k, _)| k.0).ok();
        wire_flow_edge(
            &mut driver,
            &mut maps,
            &mut hidden,
            &mut commands,
            from_nid, from_kind,
            to_nid, to_kind,
        );
    } else {
        connect.source = Some(entity);
    }
}

/// Create the sim edges + state bookkeeping for a user-drawn connection.
///
/// # Queue ← Worker pull semantics
///
/// Pull is one-directional: the arrow has to start at the worker and
/// point at the queue (**Worker → Queue**). When drawn that way:
///
///  * Visible edge: Worker → Queue (what the user drew).
///  * Hidden edge:  Queue → Worker — the data return path.
///  * Kickoff pull injected into the queue.
///
/// The sim work runs through `driver.with_sim_mut` so we get back the
/// freshly-allocated EdgeIds and can register their entities here in
/// the same call. In Worker mode this blocks for at most one tick
/// (rare on a click).
pub fn wire_flow_edge(
    driver: &mut SimDriverRes,
    maps: &mut EntityMaps,
    hidden: &mut HiddenEdges,
    commands: &mut Commands,
    from_nid: flow::NodeId,
    from_kind: Option<crate::gadgets::Kind>,
    to_nid: flow::NodeId,
    to_kind: Option<crate::gadgets::Kind>,
) {
    use crate::gadgets::Kind;

    if matches!(from_kind, Some(Kind::Worker)) && matches!(to_kind, Some(Kind::Queue)) {
        let worker_id = from_nid;
        let queue_id  = to_nid;

        let (signal_eid, data_eid) = driver.0.with_sim_mut(move |sim| {
            let signal_eid = sim.add_edge(worker_id, queue_id, flow::Expr::int(1_000_000));
            let data_eid   = sim.add_edge(queue_id,  worker_id, flow::Expr::int(1_000_000));
            if let Some(n) = sim.nodes.get_mut(&worker_id) {
                n.slots.insert("upstream".into(), flow::Value::NodeRef(queue_id));
            }
            sim.inject(
                queue_id,
                flow::Value::variant("pull", flow::Value::NodeRef(worker_id)),
            );
            (signal_eid, data_eid)
        });

        let sig_ent = commands.spawn((FlowEdgeRef(signal_eid),)).id();
        let dat_ent = commands.spawn((FlowEdgeRef(data_eid),)).id();
        maps.edge_to_entity.insert(signal_eid, sig_ent);
        maps.entity_to_edge.insert(sig_ent, signal_eid);
        maps.edge_to_entity.insert(data_eid, dat_ent);
        maps.entity_to_edge.insert(dat_ent, data_eid);
        hidden.set.insert(data_eid);
        return;
    }

    let from_kind_local = from_kind;
    let eid = driver.0.with_sim_mut(move |sim| {
        let eid = sim.add_edge(from_nid, to_nid, flow::Expr::int(1_000_000));
        if matches!(from_kind_local, Some(Kind::Worker)) {
            if let Some(n) = sim.nodes.get_mut(&from_nid) {
                n.slots.insert("downstream".into(), flow::Value::NodeRef(to_nid));
            }
        }
        eid
    });
    let ent = commands.spawn((FlowEdgeRef(eid),)).id();
    maps.edge_to_entity.insert(eid, ent);
    maps.entity_to_edge.insert(ent, eid);
}

// ---------------- arrow rendering ----------------
//
// Arrows are rendered as a single `Mesh2d` LineList rebuilt every frame
// rather than via `Gizmos`. Reason: Bevy 2D gizmos render in a pass
// after the 2D opaque/transparent passes, ignoring world z — so they
// always paint on top of mesh-rendered content like the packet cloud.
// A real Mesh2d at z=2.0 sits cleanly between nodes (z=1.0) and
// packets (z=3.0) the way the visual order is supposed to read:
// nodes < arrows < packets.

#[derive(Component)]
struct EdgeMesh;

fn spawn_edge_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    theme: Res<Theme>,
) {
    let mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
    let material = materials.add(ColorMaterial::from(theme.ink_soft));
    commands.spawn((
        Mesh2d(meshes.add(mesh)),
        MeshMaterial2d(material),
        Transform::from_xyz(0.0, 0.0, 2.0),
        Visibility::Visible,
        // Vertex positions change every frame; the static AABB Bevy
        // computes once would frustum-cull us as soon as the camera
        // moves off origin.
        NoFrustumCulling,
        EdgeMesh,
    ));
}

fn draw_edges(
    snapshot: Res<SimSnapshotRes>,
    nodes: Query<(&Transform, &crate::nodes::BodyShape), With<crate::bridge::FlowNodeRef>>,
    maps: Res<EntityMaps>,
    theme: Res<Theme>,
    connect: Res<ConnectState>,
    hidden: Res<HiddenEdges>,
    hide_all: Res<HideAll>,
    edge_mesh: Query<&Mesh2d, With<EdgeMesh>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    edge_mat: Query<&MeshMaterial2d<ColorMaterial>, With<EdgeMesh>>,
    membership: Res<crate::compound::CompoundMembership>,
    current_scope: Res<crate::compound::CurrentScope>,
    mut gizmos: Gizmos,
    mut perf: ResMut<crate::perf::PhaseTimings>,
) {
    crate::time_phase!(perf, "edges.draw_edges", {
    let Ok(handle) = edge_mesh.single() else { return; };
    let Some(mesh) = meshes.get_mut(&handle.0) else { return; };

    // Keep the colour in sync with theme swaps.
    if let Ok(mat_handle) = edge_mat.single() {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            mat.color = theme.ink_soft;
        }
    }

    let mut positions: Vec<[f32; 3]> = Vec::new();

    if !hide_all.0 {
        for (eid, edge) in snapshot.0.edges.iter() {
            if hidden.set.contains(eid) { continue; }
            // Scope filter — same single rule as `Scoped` /
            // `sync_scoped_visibility`. The arrow mesh is one big
            // line-list (no per-edge entity to flag) so we apply the
            // membership/scope check inline at draw time.
            let owner = crate::compound::canonical_edge_owner(edge.from, edge.to, &membership);
            if membership.parent_of(owner) != current_scope.0 { continue; }
            let Some(&ent_from) = maps.node_to_entity.get(&edge.from) else { continue; };
            let Some(&ent_to)   = maps.node_to_entity.get(&edge.to)   else { continue; };
            let Ok((tf_from, shape_from)) = nodes.get(ent_from) else { continue; };
            let Ok((tf_to,   shape_to))   = nodes.get(ent_to)   else { continue; };

            // Self-loops are sim plumbing (tick / done / period), not user edges.
            if edge.from == edge.to { continue; }

            let from_center = tf_from.translation.truncate();
            let to_center = tf_to.translation.truncate();
            let dir = (to_center - from_center).normalize_or_zero();
            if dir.length_squared() == 0.0 { continue; }

            use crate::nodes::hit_size;
            let from_half = hit_size(shape_from) * 0.5;
            let to_half = hit_size(shape_to) * 0.5;
            let from_exit = from_center + dir * border_exit(from_half, dir);
            let to_entry = to_center - dir * (border_exit(to_half, -dir) + 4.0);

            push_arrow_segments(&mut positions, from_exit, to_entry);
        }
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);

    // Connect-source preview circle stays a gizmo — it's transient
    // (only visible while actively wiring) and doesn't fight packets
    // for layering.
    if let Some(src) = connect.source {
        if let Ok((tf, _)) = nodes.get(src) {
            gizmos.circle_2d(tf.translation.truncate(), 40.0, theme.accent);
        }
    }
    });
}

/// Ray-into-box exit distance.
fn border_exit(half: Vec2, dir: Vec2) -> f32 {
    let tx = if dir.x.abs() > 1e-4 { half.x / dir.x.abs() } else { f32::INFINITY };
    let ty = if dir.y.abs() > 1e-4 { half.y / dir.y.abs() } else { f32::INFINITY };
    tx.min(ty)
}

/// Append the line segments for one arrow (shaft + filled triangular
/// head) to a flat positions buffer for a `LineList` mesh. Matches the
/// shape of the previous gizmo `draw_arrow` exactly.
fn push_arrow_segments(positions: &mut Vec<[f32; 3]>, a: Vec2, b: Vec2) {
    let delta = b - a;
    let dist = delta.length();
    if dist < 1.0 { return; }
    let dir = delta / dist;

    // Shaft.
    positions.push([a.x, a.y, 0.0]);
    positions.push([b.x, b.y, 0.0]);

    let tip = b;
    let head_perp = Vec2::new(-dir.y, dir.x);
    let head_len = 11.0_f32;
    let head_w = 6.5_f32;
    let back = tip - dir * head_len;
    let left = back + head_perp * head_w;
    let right = back - head_perp * head_w;

    const FILL_STEPS: usize = 6;
    for i in 0..=FILL_STEPS {
        let t = i as f32 / FILL_STEPS as f32;
        let base_pt = left.lerp(right, t);
        positions.push([tip.x, tip.y, 0.0]);
        positions.push([base_pt.x, base_pt.y, 0.0]);
    }
    positions.push([tip.x, tip.y, 0.0]);
    positions.push([left.x, left.y, 0.0]);
    positions.push([tip.x, tip.y, 0.0]);
    positions.push([right.x, right.y, 0.0]);
    positions.push([left.x, left.y, 0.0]);
    positions.push([right.x, right.y, 0.0]);
}

// ---------------- packet pipeline (F1) ----------------

/// Feed sim events into the pure timeline and spawn a Bevy entity
/// for each new visible packet.
///
/// F5: the anchor is NOT re-set per frame. It's set once at
/// scenario load (see `examples::handle_load_example`) or when the
/// user changes `k` (see `apply_visual_scale_change`). Events are
/// ingested through a fixed sim↔real mapping, so visuals play at a
/// rate that's independent of sim speed.
fn ingest_new_events(
    evs: Res<NewEvents>,
    clock: Res<crate::bridge::SimClock>,
    mut timeline: ResMut<VisualTimelineRes>,
    mut perf: ResMut<crate::perf::PhaseTimings>,
) {
    let real_now = clock.visual_now;
    crate::time_phase!(perf, "edges.ingest_new_events", {
    for ev in &evs.0 {
        // State-change boundaries (`TimelineEventFired`, `UserSlotEdit`)
        // used to wipe the future-queued visual backlog here. We
        // dropped that on purpose: it made Replay's clamp visibly
        // "reset" on every UI interaction, and the other strategies
        // don't future-queue, so they had nothing to drop anyway.
        // Now boundary events fall through to `ingest` like any
        // other; the strategies' `parse_emit` filter rejects every
        // non-`PacketEmitted` variant, so this is a no-op for them.
        // In-flight visuals play out naturally; new sim events feed
        // the strategy from the moment they arrive.
        timeline.ingest(ev, real_now);
    }
    });
}

/// Trim the timeline's arrived-history so long sessions don't grow
/// unbounded. The keep window matches the snapshot ring's
/// scrub-back horizon so the user can rewind anywhere within that
/// horizon and still see the visuals that were on screen at the
/// rewound moment. (A 2s window — the original value — clipped
/// scrub-back to exactly 2 seconds of past visuals because anything
/// older had already been GC'd before the rewind happened.)
///
/// While the sim is paused, GC is skipped entirely. Pause is a
/// strong signal that the user is scrubbing through history; we
/// don't want a pause-during-scrub to silently delete past visuals
/// the user is about to navigate to.
const VISUAL_GC_KEEP_PAST_S: f64 = 30.0;

fn gc_timeline(
    clock: Res<crate::bridge::SimClock>,
    mut timeline: ResMut<VisualTimelineRes>,
    mut perf: ResMut<crate::perf::PhaseTimings>,
) {
    if clock.paused { return; }
    let now = clock.visual_now;
    crate::time_phase!(perf, "edges.gc_timeline", {
        timeline.gc_before(now, VISUAL_GC_KEEP_PAST_S);
    });
}

#[cfg(test)]
mod color_tests {
    //! Unit tests for the packet-colour resolution chain.

    use super::*;
    use bevy::prelude::Color;
    use flow::Value;

    fn v_packet(slot: i64) -> Value {
        Value::variant("packet", Value::Int(slot))
    }
    fn v_req(slot: i64) -> Value {
        Value::variant("req", Value::Int(slot))
    }

    const RED:    Color = Color::srgb(1.0, 0.0, 0.0);
    const YELLOW: Color = Color::srgb(1.0, 1.0, 0.0);
    const BLUE:   Color = Color::srgb(0.0, 0.0, 1.0);
    const ACCENT: Color = Color::srgb(0.5, 0.5, 0.5);
    const EMITTER: Color = Color::srgb(0.1, 0.2, 0.3);
    const PAL: [Color; 3] = [RED, YELLOW, BLUE];

    #[test]
    fn payload_slot_reads_packet_variant() {
        assert_eq!(payload_slot(&v_packet(0)), Some(0));
        assert_eq!(payload_slot(&v_packet(1)), Some(1));
        assert_eq!(payload_slot(&v_packet(42)), Some(42));
    }

    #[test]
    fn payload_slot_reads_req_variant() {
        assert_eq!(payload_slot(&v_req(0)), Some(0));
        assert_eq!(payload_slot(&v_req(2)), Some(2));
    }

    #[test]
    fn payload_slot_ignores_other_tags() {
        assert_eq!(payload_slot(&Value::variant("pull", Value::Nil)), None);
        assert_eq!(payload_slot(&Value::variant("resp", Value::Int(0))), None);
        assert_eq!(payload_slot(&Value::variant("done", Value::Int(0))), None);
        assert_eq!(payload_slot(&Value::variant("wake", Value::Nil)), None);
        assert_eq!(payload_slot(&Value::variant("tick", Value::Nil)), None);
    }

    #[test]
    fn payload_slot_ignores_non_int_payloads() {
        assert_eq!(payload_slot(&Value::variant("packet", Value::Nil)), None);
        assert_eq!(
            payload_slot(&Value::variant("packet", Value::Str("x".into()))),
            None
        );
    }

    #[test]
    fn payload_slot_rejects_negative() {
        assert_eq!(payload_slot(&v_packet(-1)), None);
    }

    #[test]
    fn packet_color_prefers_payload_over_emitter() {
        let c = packet_color(&v_packet(1), Some(RED), &PAL, ACCENT);
        assert_eq!(c, YELLOW);
    }

    #[test]
    fn packet_color_falls_back_to_emitter_for_control_payloads() {
        let c = packet_color(
            &Value::variant("pull", Value::Nil),
            Some(BLUE),
            &PAL,
            ACCENT,
        );
        assert_eq!(c, BLUE);
    }

    #[test]
    fn packet_color_falls_back_to_accent_when_both_missing() {
        let c = packet_color(&Value::variant("wake", Value::Nil), None, &PAL, ACCENT);
        assert_eq!(c, ACCENT);
    }

    #[test]
    fn packet_color_clamps_oversized_slot() {
        let c = packet_color(&v_packet(99), Some(EMITTER), &PAL, ACCENT);
        assert_eq!(c, BLUE);
    }

    #[test]
    fn packet_color_handles_empty_palette() {
        let c = packet_color(&v_packet(0), Some(EMITTER), &[], ACCENT);
        assert_eq!(c, EMITTER);
        let c = packet_color(&v_packet(0), None, &[], ACCENT);
        assert_eq!(c, ACCENT);
    }

    #[test]
    fn packet_color_resolves_each_stream_to_its_slot() {
        let forwarder_color = Some(RED);
        for slot in 0..3 {
            let c = packet_color(&v_packet(slot as i64), forwarder_color, &PAL, ACCENT);
            assert_eq!(c, PAL[slot], "slot {} should resolve to palette[{}]", slot, slot);
        }
    }
}
