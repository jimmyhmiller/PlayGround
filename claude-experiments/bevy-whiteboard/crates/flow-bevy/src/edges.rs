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

use bevy::prelude::*;
use flow::{EdgeId, NodeId, PacketId};

use crate::bridge::{EntityMaps, FlowEdgeRef, FlowSim, NewEvents};
use crate::camera::{MainCamera, cursor_to_world};
use crate::nodes::NodeKind;
use crate::theme::Theme;
use crate::tool::{ActiveTool, Tool};
use crate::visual::{VisualPacket, VisualTimeline};

pub struct EdgesPlugin;
impl Plugin for EdgesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConnectState>()
            .init_resource::<HiddenEdges>()
            .init_resource::<DebugMode>()
            .insert_resource(VisualTimelineRes(VisualTimeline::default()))
            .add_systems(Update, (
                handle_connect_click,
                draw_edges,
                ingest_new_events,
                sync_packet_transforms,
                sync_packet_id_labels,
                despawn_arrived_packets,
                gc_timeline,
            ).chain());
    }
}

/// Bevy wrapper around the pure timeline so it can live as a
/// `Resource`. The underlying `VisualTimeline` deliberately has no
/// Bevy dependency.
#[derive(Resource)]
pub struct VisualTimelineRes(pub VisualTimeline);

impl std::ops::Deref for VisualTimelineRes {
    type Target = VisualTimeline;
    fn deref(&self) -> &VisualTimeline { &self.0 }
}
impl std::ops::DerefMut for VisualTimelineRes {
    fn deref_mut(&mut self) -> &mut VisualTimeline { &mut self.0 }
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

/// A live visual packet. Fields are a read-only copy of the matching
/// `VisualPacket` in the timeline resource — both are set at spawn
/// and never mutated. Keeping the values here avoids a timeline
/// lookup in the per-frame sync system and makes tests trivial
/// (`app.world().query::<&TravelingPacket>()`).
#[derive(Component, Clone, Debug)]
pub struct TravelingPacket {
    pub packet_id: PacketId,
    pub from: NodeId,
    pub to: NodeId,
    pub emit_real: f64,
    pub arrive_real: f64,
}

/// Marker component on the text child entity that shows a packet's
/// id in debug mode. Updated to match visibility state of the parent
/// packet each frame.
#[derive(Component)]
pub struct PacketIdLabel;

/// When on, each `TravelingPacket` renders its `packet_id` as a
/// small label on the dot. Toggled by hotkey `d` (see palette).
#[derive(Resource, Default)]
pub struct DebugMode {
    pub on: bool,
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
    nodes: Query<(Entity, &Transform, &NodeKind), With<crate::bridge::FlowNodeRef>>,
    mut maps: ResMut<EntityMaps>,
    mut flow: ResMut<FlowSim>,
    mut hidden: ResMut<HiddenEdges>,
    mut commands: Commands,
    ui: Query<&Interaction>,
) {
    if !matches!(active.0, Tool::Connect) { return; }
    if !buttons.just_pressed(MouseButton::Left) { return; }
    if poster_ui::pointer_over_ui(&ui) { return; }
    let Some(world) = cursor_to_world(&windows, &cams) else { return; };

    use crate::nodes::{body_shape, hit_size};
    let hit = nodes.iter().find(|(_, tf, kind)| {
        let half = hit_size(&body_shape(kind.0)) * 0.5;
        let p = tf.translation.truncate();
        (world - p).abs().cmple(half).all()
    }).map(|(e, _, _)| e);
    let Some(entity) = hit else { return; };

    if let Some(src) = connect.source.take() {
        let Some(&from_nid) = maps.entity_to_node.get(&src) else { return; };
        let Some(&to_nid) = maps.entity_to_node.get(&entity) else { return; };
        let from_kind = nodes.get(src).map(|(_, _, k)| k.0).ok();
        let to_kind = nodes.get(entity).map(|(_, _, k)| k.0).ok();
        wire_flow_edge(
            &mut flow.sim,
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
pub fn wire_flow_edge(
    sim: &mut flow::Sim,
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

        let signal_eid = sim.add_edge(worker_id, queue_id,  flow::Expr::int(1_000_000));
        let data_eid   = sim.add_edge(queue_id,  worker_id, flow::Expr::int(1_000_000));

        let sig_ent = commands.spawn((FlowEdgeRef(signal_eid),)).id();
        let dat_ent = commands.spawn((FlowEdgeRef(data_eid),)).id();
        maps.edge_to_entity.insert(signal_eid, sig_ent);
        maps.entity_to_edge.insert(sig_ent, signal_eid);
        maps.edge_to_entity.insert(data_eid, dat_ent);
        maps.entity_to_edge.insert(dat_ent, data_eid);
        hidden.set.insert(data_eid);

        if let Some(n) = sim.nodes.get_mut(&worker_id) {
            n.slots.insert("upstream".into(), flow::Value::NodeRef(queue_id));
        }

        sim.inject(
            queue_id,
            flow::Value::variant("pull", flow::Value::NodeRef(worker_id)),
        );
        return;
    }

    let eid = sim.add_edge(from_nid, to_nid, flow::Expr::int(1_000_000));
    let ent = commands.spawn((FlowEdgeRef(eid),)).id();
    maps.edge_to_entity.insert(eid, ent);
    maps.entity_to_edge.insert(ent, eid);

    if matches!(from_kind, Some(Kind::Worker)) {
        if let Some(n) = sim.nodes.get_mut(&from_nid) {
            n.slots.insert("downstream".into(), flow::Value::NodeRef(to_nid));
        }
    }
}

// ---------------- arrow rendering ----------------

fn draw_edges(
    mut gizmos: Gizmos,
    flow: Res<FlowSim>,
    nodes: Query<(&Transform, &crate::nodes::NodeKind), With<crate::bridge::FlowNodeRef>>,
    maps: Res<EntityMaps>,
    theme: Res<Theme>,
    connect: Res<ConnectState>,
    hidden: Res<HiddenEdges>,
) {
    for (eid, edge) in flow.sim.edges.iter() {
        if hidden.set.contains(eid) { continue; }
        let Some(&ent_from) = maps.node_to_entity.get(&edge.from) else { continue; };
        let Some(&ent_to)   = maps.node_to_entity.get(&edge.to)   else { continue; };
        let Ok((tf_from, kind_from)) = nodes.get(ent_from) else { continue; };
        let Ok((tf_to,   kind_to))   = nodes.get(ent_to)   else { continue; };

        // Self-loops are sim plumbing (tick / done / period), not user edges.
        if edge.from == edge.to { continue; }

        let from_center = tf_from.translation.truncate();
        let to_center = tf_to.translation.truncate();
        let dir = (to_center - from_center).normalize_or_zero();
        if dir.length_squared() == 0.0 { continue; }

        use crate::nodes::{body_shape, hit_size};
        let from_half = hit_size(&body_shape(kind_from.0)) * 0.5;
        let to_half = hit_size(&body_shape(kind_to.0)) * 0.5;
        let from_exit = from_center + dir * border_exit(from_half, dir);
        let to_entry = to_center - dir * (border_exit(to_half, -dir) + 4.0);

        draw_arrow(&mut gizmos, from_exit, to_entry, theme.ink_soft);
    }

    if let Some(src) = connect.source {
        if let Ok((tf, _)) = nodes.get(src) {
            gizmos.circle_2d(tf.translation.truncate(), 40.0, theme.accent);
        }
    }
}

/// Ray-into-box exit distance.
fn border_exit(half: Vec2, dir: Vec2) -> f32 {
    let tx = if dir.x.abs() > 1e-4 { half.x / dir.x.abs() } else { f32::INFINITY };
    let ty = if dir.y.abs() > 1e-4 { half.y / dir.y.abs() } else { f32::INFINITY };
    tx.min(ty)
}

fn draw_arrow(gizmos: &mut Gizmos, a: Vec2, b: Vec2, color: Color) {
    let delta = b - a;
    let dist = delta.length();
    if dist < 1.0 { return; }
    let dir = delta / dist;
    gizmos.line_2d(a, b, color);

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
        gizmos.line_2d(tip, base_pt, color);
    }
    gizmos.line_2d(tip, left, color);
    gizmos.line_2d(tip, right, color);
    gizmos.line_2d(left, right, color);
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
    mut commands: Commands,
    evs: Res<NewEvents>,
    clock: Res<crate::bridge::SimClock>,
    maps: Res<EntityMaps>,
    theme: Res<Theme>,
    node_colors: Res<crate::tool::NodeColors>,
    mut timeline: ResMut<VisualTimelineRes>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    node_transforms: Query<&Transform, With<crate::bridge::FlowNodeRef>>,
    existing_packets: Query<(Entity, &TravelingPacket)>,
) {
    let real_now = clock.visual_now;
    for ev in &evs.0 {
        // State-change boundary: either a scheduled timeline event
        // fired, or the user manually edited a slot from the
        // inspector. Drop only the FUTURE-QUEUED backlog (packets
        // whose emit_real hasn't happened yet at `real_now`) — the
        // visual layer's causal clamp had been pushing each new
        // packet farther into the future, and we want that queue
        // zapped so the canvas reflects post-change sim state.
        //
        // Currently-animating packets (`emit_real <= real_now`) are
        // kept untouched: they're real recent past, the user is
        // watching them mid-flight, killing them mid-animation
        // looks like glitches.
        if matches!(ev,
            flow::Event::TimelineEventFired { .. } |
            flow::Event::UserSlotEdit { .. }
        ) {
            let dropped_ids = timeline.0.drop_pending_after(real_now);
            for (e, pkt) in existing_packets.iter() {
                if dropped_ids.contains(&pkt.packet_id) {
                    commands.entity(e).despawn();
                }
            }
            continue;
        }
        let Some(idx) = timeline.ingest(ev, real_now) else { continue; };
        let vp = timeline.packets[idx].clone();
        spawn_packet_entity(
            &mut commands,
            &mut meshes,
            &mut materials,
            &maps,
            &theme,
            &node_colors,
            &node_transforms,
            &vp,
        );
    }
}

fn spawn_packet_entity(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    maps: &EntityMaps,
    theme: &Theme,
    node_colors: &crate::tool::NodeColors,
    node_transforms: &Query<&Transform, With<crate::bridge::FlowNodeRef>>,
    vp: &VisualPacket,
) {
    let pkt_color = packet_color(
        &vp.payload,
        node_colors.0.get(&vp.from).copied(),
        &theme.data,
        theme.accent,
    );

    // Initial position = source node's world pos. If Visibility has a
    // one-frame render glitch, the dot flashes AT its source — which
    // reads as "about to leave" — not at world origin.
    let initial_pos = maps.node_to_entity.get(&vp.from)
        .and_then(|e| node_transforms.get(*e).ok())
        .map(|t| t.translation.truncate())
        .unwrap_or(Vec2::ZERO);

    let packet_entity = commands.spawn((
        Mesh2d(meshes.add(Circle::new(6.0))),
        MeshMaterial2d(materials.add(ColorMaterial::from(pkt_color))),
        Transform::from_xyz(initial_pos.x, initial_pos.y, 3.0),
        Visibility::Hidden,
        TravelingPacket {
            packet_id: vp.packet_id,
            from: vp.from,
            to: vp.to,
            emit_real: vp.emit_real,
            arrive_real: vp.arrive_real,
        },
    )).id();

    commands.entity(packet_entity).with_children(|p| {
        p.spawn((
            Text2d::new(format!("{}", vp.packet_id.0)),
            TextFont { font_size: 10.0, ..default() },
            TextColor(theme.ink),
            Transform::from_xyz(0.0, 12.0, 0.1),
            Visibility::Hidden,
            PacketIdLabel,
        ));
    });
}

/// Per frame: snap each packet's transform + visibility to the pure
/// formalism. `emit_real <= now < arrive_real` ⇒ visible, position
/// interpolated `from → to`. Outside that window ⇒ hidden (either
/// not-yet-alive or already-arrived; `despawn_arrived_packets`
/// deletes the latter).
fn sync_packet_transforms(
    clock: Res<crate::bridge::SimClock>,
    maps: Res<EntityMaps>,
    nodes: Query<&Transform, (With<crate::bridge::FlowNodeRef>, Without<TravelingPacket>)>,
    mut pkts: Query<(&mut Transform, &TravelingPacket, &mut Visibility)>,
) {
    let now = clock.visual_now;
    for (mut tf, pkt, mut vis) in pkts.iter_mut() {
        if now < pkt.emit_real || now >= pkt.arrive_real {
            *vis = Visibility::Hidden;
            continue;
        }
        *vis = Visibility::Visible;
        let denom = pkt.arrive_real - pkt.emit_real;
        let prog = ((now - pkt.emit_real) / denom).clamp(0.0, 1.0) as f32;
        let Some(&ent_from) = maps.node_to_entity.get(&pkt.from) else { continue; };
        let Some(&ent_to)   = maps.node_to_entity.get(&pkt.to)   else { continue; };
        let Ok(a) = nodes.get(ent_from) else { continue; };
        let Ok(b) = nodes.get(ent_to) else { continue; };
        let p = a.translation.truncate().lerp(b.translation.truncate(), prog);
        tf.translation.x = p.x;
        tf.translation.y = p.y;
    }
}

/// Flip each `PacketIdLabel`'s visibility to match `DebugMode.on`.
fn sync_packet_id_labels(
    debug: Res<DebugMode>,
    mut labels: Query<&mut Visibility, With<PacketIdLabel>>,
) {
    let want = if debug.on { Visibility::Inherited } else { Visibility::Hidden };
    for mut v in labels.iter_mut() {
        if *v != want { *v = want; }
    }
}

/// Despawn entities whose packets have arrived.
fn despawn_arrived_packets(
    mut commands: Commands,
    clock: Res<crate::bridge::SimClock>,
    q: Query<(Entity, &TravelingPacket)>,
) {
    let now = clock.visual_now;
    for (e, pkt) in q.iter() {
        if now >= pkt.arrive_real {
            commands.entity(e).despawn();
        }
    }
}

/// Trim the timeline's arrived-history so long sessions don't grow
/// unbounded. Keeps a ~2s window past arrival for debugging and
/// test introspection.
///
/// `gc_before` retains packets whose `arrive_real >= now - keep`,
/// so already-despawned entities get pruned without affecting any
/// live `TravelingPacket` (live ones have `arrive_real >= now`).
fn gc_timeline(
    clock: Res<crate::bridge::SimClock>,
    mut timeline: ResMut<VisualTimelineRes>,
) {
    let now = clock.visual_now;
    timeline.gc_before(now, 2.0);
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
