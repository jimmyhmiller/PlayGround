//! Edges: connect mode, rendering, animated packets from sim events.

use bevy::prelude::*;
use flow::{Event, EdgeId};

use crate::bridge::{EntityMaps, FlowEdgeRef, FlowSim, NewEvents};
use crate::camera::{MainCamera, cursor_to_world};
use crate::nodes::NodeKind;
use crate::theme::Theme;
use crate::tool::{ActiveTool, Tool};

pub struct EdgesPlugin;
impl Plugin for EdgesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ConnectState>()
            .init_resource::<EdgeVisualState>()
            .init_resource::<HiddenEdges>()
            .add_systems(Update, (
                handle_connect_click,
                draw_edges,
                spawn_traveling_packets,
                animate_packets,
                despawn_finished_packets,
            ).chain());
    }
}

/// Per-edge last-spawn real-time, for coalescing burst traffic.
#[derive(Resource, Default)]
pub struct EdgeVisualState {
    pub last_spawn: std::collections::HashMap<EdgeId, f32>,
}

/// Sim edges that exist solely for routing — not drawn, not visualized.
/// When the user establishes a pull relationship, we create a forward
/// "pull signal" edge *and* a reverse "data" edge under the hood. Only
/// one of them represents what the user drew; the other is hidden here
/// so it doesn't clutter the canvas with a mirror arrow and a second set
/// of moving dots.
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

/// A visible packet. Animates over a fixed real-time duration
/// regardless of the edge's sim-time latency — the visual is a cue
/// ("an event happened on this edge"), not a faithful rendering of
/// sim timing. This keeps the canvas watchable whether the edge is
/// 1μs or 1s in sim time.
#[derive(Component)]
pub struct TravelingPacket {
    pub edge: EdgeId,
    pub t: f32,                  // 0..1, animated real-time
    pub duration_real_s: f32,
}

/// Visual traversal time is proportional to the packet's actual sim
/// latency (so faster edges produce visibly faster packets), scaled
/// to real time and clamped for readability.
///
/// Base scale: a 1 ms sim edge takes ~0.6 real seconds to traverse.
const VIS_REAL_S_PER_SIM_NS: f32 = 6e-7;
const VIS_MIN_REAL_S:        f32 = 0.15;
const VIS_MAX_REAL_S:        f32 = 3.0;

/// Minimum real-time gap between two visual packets on the same edge;
/// coalesces bursty emissions so we don't flood the canvas.
pub const MIN_SPAWN_INTERVAL: f32 = 0.05;

fn visual_duration_for(sim_latency_ns: u64) -> f32 {
    let raw = (sim_latency_ns as f32) * VIS_REAL_S_PER_SIM_NS;
    raw.clamp(VIS_MIN_REAL_S, VIS_MAX_REAL_S)
}

/// If the packet's payload is `packet(Int(slot))` or `req(Int(slot))`,
/// extract the slot index. Returns `None` for nil payloads or other
/// shapes — caller falls back to the emitter's colour.
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
///      packet's colour persist end-to-end through routers, queues, and
///      workers — what the sink receives is the colour of whoever
///      originated the stream, not the last hop.
///   2. Emitter colour: fall back to the emitter's `NodeColors` entry.
///      Used for payloads with no slot tag (e.g. `pull(NodeRef)`, which
///      is a control signal, not data).
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

    // Pick node under cursor. Uses the same hit-size as node dragging so
    // edge-connect and node-pick behave consistently.
    use crate::nodes::{body_shape, hit_size};
    let hit = nodes.iter().find(|(_, tf, kind)| {
        let half = hit_size(&body_shape(kind.0)) * 0.5;
        let p = tf.translation.truncate();
        (world - p).abs().cmple(half).all()
    }).map(|(e, _, _)| e);
    let Some(entity) = hit else { return; };

    if let Some(src) = connect.source.take() {
        // Second click: create edge + wire pull semantics if applicable.
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
/// point at the queue (**Worker → Queue**), matching the natural reading
/// "worker pulls from queue". When that exact pairing is drawn:
///
///  * Visible edge: Worker → Queue (what the user drew), arrowhead at
///    the queue — the pull arrow.
///  * Hidden edge:  Queue → Worker — the data return path, created
///    under the hood so `Respond`-equivalent routing has somewhere to
///    travel. Added to [`HiddenEdges`] so it doesn't clutter the canvas.
///  * Kickoff pull injected into the queue.
///
/// Drawing **Queue → Worker** does NOT establish a pull relationship.
/// The queue won't push on its own (no self-drain), so nothing flows —
/// this is on purpose: the user must draw the arrow in the pull
/// direction to wire up a working chain.
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

    // Pull requires Worker → Queue. The other direction is a no-op chain.
    if matches!(from_kind, Some(Kind::Worker)) && matches!(to_kind, Some(Kind::Queue)) {
        let worker_id = from_nid;
        let queue_id  = to_nid;

        // The visible edge is the one the user drew: Worker → Queue.
        // The hidden data-return edge is Queue → Worker.
        let signal_eid = sim.add_edge(worker_id, queue_id,  flow::Expr::int(1_000_000));
        let data_eid   = sim.add_edge(queue_id,  worker_id, flow::Expr::int(1_000_000));

        let sig_ent = commands.spawn((FlowEdgeRef(signal_eid),)).id();
        let dat_ent = commands.spawn((FlowEdgeRef(data_eid),)).id();
        maps.edge_to_entity.insert(signal_eid, sig_ent);
        maps.entity_to_edge.insert(sig_ent, signal_eid);
        maps.edge_to_entity.insert(data_eid, dat_ent);
        maps.entity_to_edge.insert(dat_ent, data_eid);
        hidden.set.insert(data_eid);

        // Worker remembers its upstream so `done` can aim pull signals.
        if let Some(n) = sim.nodes.get_mut(&worker_id) {
            n.slots.insert("upstream".into(), flow::Value::NodeRef(queue_id));
        }

        // Kick off the cycle. Pull payload carries the worker's NodeRef
        // so the queue can route the response back.
        sim.inject(
            queue_id,
            flow::Value::variant("pull", flow::Value::NodeRef(worker_id)),
            Some(worker_id),
        );
        return;
    }

    // Default case: one ordinary edge.
    let eid = sim.add_edge(from_nid, to_nid, flow::Expr::int(1_000_000));
    let ent = commands.spawn((FlowEdgeRef(eid),)).id();
    maps.edge_to_entity.insert(eid, ent);
    maps.entity_to_edge.insert(ent, eid);

    // Worker → downstream (anything non-Queue) records downstream.
    if matches!(from_kind, Some(Kind::Worker)) {
        if let Some(n) = sim.nodes.get_mut(&from_nid) {
            n.slots.insert("downstream".into(), flow::Value::NodeRef(to_nid));
        }
    }

    // Client → Worker is request/response: also create the hidden
    // response edge Worker → Client so the worker's `Respond` can route
    // back. Without this the worker silently drops responses (engine
    // does this gracefully now, but the user wouldn't see anything
    // happen). Hide the response edge so the user sees one arrow per
    // user-drawn connection — same as the pull case.
    if matches!(from_kind, Some(Kind::Client)) && matches!(to_kind, Some(Kind::Worker)) {
        let resp_eid = sim.add_edge(to_nid, from_nid, flow::Expr::int(1_000_000));
        let resp_ent = commands.spawn((FlowEdgeRef(resp_eid),)).id();
        maps.edge_to_entity.insert(resp_eid, resp_ent);
        maps.entity_to_edge.insert(resp_ent, resp_eid);
        hidden.set.insert(resp_eid);
    }
}

// ---------------- rendering ----------------

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

        // Self-loops are a sim implementation detail — Generator / Client /
        // Queue use them to drive their own tick rules. The user doesn't
        // need to see that plumbing, so we skip rendering them. Packet
        // visualizer already filters self-loop packets out too
        // (see spawn_traveling_packets).
        if edge.from == edge.to {
            continue;
        }

        let from_center = tf_from.translation.truncate();
        let to_center = tf_to.translation.truncate();
        let dir = (to_center - from_center).normalize_or_zero();
        if dir.length_squared() == 0.0 { continue; }

        // Trim the line to the node borders so the arrow doesn't overlap
        // the body meshes. Circle bodies use radius, rect bodies use
        // ray-into-box; hit_size() gives a consistent bounding extent.
        use crate::nodes::{body_shape, hit_size};
        let from_half = hit_size(&body_shape(kind_from.0)) * 0.5;
        let to_half = hit_size(&body_shape(kind_to.0)) * 0.5;
        let from_exit = from_center + dir * border_exit(from_half, dir);
        let to_entry = to_center - dir * (border_exit(to_half, -dir) + 4.0);

        draw_arrow(&mut gizmos, from_exit, to_entry, theme.ink_soft);
    }

    // Show the "source" node highlight during connect mode.
    if let Some(src) = connect.source {
        if let Ok((tf, _)) = nodes.get(src) {
            gizmos.circle_2d(tf.translation.truncate(), 40.0, theme.accent);
        }
    }
}

/// Ray-into-box exit distance: how far along `dir` from the centre until we
/// hit the box border. For circular bodies whose hit_size is square, this
/// is an outer-square approximation — slightly outside the actual circle
/// radius at diagonals, which reads fine.
fn border_exit(half: Vec2, dir: Vec2) -> f32 {
    let tx = if dir.x.abs() > 1e-4 { half.x / dir.x.abs() } else { f32::INFINITY };
    let ty = if dir.y.abs() > 1e-4 { half.y / dir.y.abs() } else { f32::INFINITY };
    tx.min(ty)
}

/// Draw a straight edge from `a` to `b` with a filled-wedge arrowhead at
/// `b`. Gizmos don't render filled primitives, so the head is faked by a
/// fan of strokes from the tip to interpolated points along the base.
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

    // Fan-fill the wedge so it reads as a solid arrowhead rather than a
    // three-stroke wireframe.
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

// ---------------- packets ----------------

fn spawn_traveling_packets(
    mut commands: Commands,
    evs: Res<NewEvents>,
    flow: Res<FlowSim>,
    maps: Res<EntityMaps>,
    theme: Res<Theme>,
    node_colors: Res<crate::tool::NodeColors>,
    clock: Res<crate::bridge::SimClock>,
    time: Res<Time>,
    hidden: Res<HiddenEdges>,
    mut vis: ResMut<EdgeVisualState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let now_real = time.elapsed_secs();
    let multiplier = clock.multiplier.max(0.01) as f32;
    for ev in &evs.0 {
        let Event::PacketEmitted { from, to, at_ns, arrives_at_ns, payload, .. } = ev else { continue; };

        // Skip internal plumbing emissions: pull signals and queue wake
        // ticks. Users only care about data moving.
        if let flow::Value::Variant { tag, .. } = payload {
            if tag == "pull" || tag == "wake" { continue; }
        }

        let edge_id = flow.sim.edges.iter()
            .find(|(_, e)| e.from == *from && e.to == *to)
            .map(|(eid, _)| *eid);
        let Some(edge_id) = edge_id else { continue; };
        if flow.sim.edges[&edge_id].from == flow.sim.edges[&edge_id].to {
            continue;
        }
        // NOTE: we deliberately DON'T skip hidden edges here. In a pull
        // relationship the data edge (Queue→Worker) is often the hidden
        // one, but data packets traveling along it are the whole point
        // of the visual. The visible edge's line coincides spatially
        // (same two nodes), so the animation lands on the right line.
        let _ = &hidden; // touched to prove we considered it
        let last = vis.last_spawn.get(&edge_id).copied().unwrap_or(f32::NEG_INFINITY);
        if now_real - last < MIN_SPAWN_INTERVAL { continue; }
        vis.last_spawn.insert(edge_id, now_real);

        // Visual duration scales with the actual sim latency of THIS
        // packet, so faster edges visibly produce faster-moving dots.
        // Then divide by the playback multiplier so cranking to 4× makes
        // packets travel 4× faster across the edge — we're going for
        // "impression of flow", not cycle-accurate animation.
        let sim_latency = arrives_at_ns.saturating_sub(*at_ns);
        let dur = visual_duration_for(sim_latency) / multiplier;

        let pkt_color = packet_color(
            payload,
            node_colors.0.get(from).copied(),
            &theme.data,
            theme.accent,
        );

        let _ = maps;
        commands.spawn((
            Mesh2d(meshes.add(Circle::new(6.0))),
            MeshMaterial2d(materials.add(ColorMaterial::from(pkt_color))),
            Transform::from_xyz(0.0, 0.0, 3.0),
            TravelingPacket {
                edge: edge_id,
                t: 0.0,
                duration_real_s: dur,
            },
        ));
    }
}

fn animate_packets(
    time: Res<Time>,
    flow: Res<FlowSim>,
    maps: Res<EntityMaps>,
    nodes: Query<&Transform, (With<crate::bridge::FlowNodeRef>, Without<TravelingPacket>)>,
    mut pkts: Query<(&mut Transform, &mut TravelingPacket)>,
) {
    let dt = time.delta_secs();
    for (mut tf, mut pkt) in pkts.iter_mut() {
        pkt.t = (pkt.t + dt / pkt.duration_real_s).min(1.0);
        let Some(edge) = flow.sim.edges.get(&pkt.edge) else { continue; };
        let Some(&ent_from) = maps.node_to_entity.get(&edge.from) else { continue; };
        let Some(&ent_to)   = maps.node_to_entity.get(&edge.to)   else { continue; };
        let Ok(a) = nodes.get(ent_from) else { continue; };
        let Ok(b) = nodes.get(ent_to) else { continue; };
        let p0 = a.translation.truncate();
        let p1 = b.translation.truncate();
        let p = p0.lerp(p1, pkt.t);
        tf.translation.x = p.x;
        tf.translation.y = p.y;
    }
}

fn despawn_finished_packets(
    mut commands: Commands,
    q: Query<(Entity, &TravelingPacket)>,
) {
    for (e, pkt) in q.iter() {
        if pkt.t >= 1.0 {
            commands.entity(e).despawn();
        }
    }
}

#[cfg(test)]
mod color_tests {
    //! Unit tests for the packet-colour resolution chain. These cover
    //! the pure function so every downstream visual test can rely on
    //! the same rules: payload colour beats emitter colour beats accent.

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
        // A yellow (slot 1) packet emitted by a red-coloured node still
        // reads yellow on the wire. This is the invariant that lets a
        // packet's origin colour persist through routers/queues/workers.
        let c = packet_color(&v_packet(1), Some(RED), &PAL, ACCENT);
        assert_eq!(c, YELLOW);
    }

    #[test]
    fn packet_color_falls_back_to_emitter_for_control_payloads() {
        // `pull(self)` has no slot tag — the emitter's colour wins. This
        // is how a worker's pull-signal back to a queue still renders in
        // the worker's colour.
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
        // Slot index beyond palette length clamps to last entry. Prevents
        // panics if theme palette shrinks while in-flight packets still
        // carry their original slot.
        let c = packet_color(&v_packet(99), Some(EMITTER), &PAL, ACCENT);
        assert_eq!(c, BLUE); // last palette entry
    }

    #[test]
    fn packet_color_handles_empty_palette() {
        // Edge case: no palette at all. Fall through to emitter, then accent.
        let c = packet_color(&v_packet(0), Some(EMITTER), &[], ACCENT);
        assert_eq!(c, EMITTER);
        let c = packet_color(&v_packet(0), None, &[], ACCENT);
        assert_eq!(c, ACCENT);
    }

    #[test]
    fn packet_color_resolves_each_stream_to_its_slot() {
        // Three parallel streams, each tagged with its slot, all emitted
        // by the same forwarder (emitter colour = red). Each packet
        // should render in its stream's colour — proves routers can
        // multiplex without losing colour identity.
        let forwarder_color = Some(RED);
        for slot in 0..3 {
            let c = packet_color(&v_packet(slot as i64), forwarder_color, &PAL, ACCENT);
            assert_eq!(c, PAL[slot], "slot {} should resolve to palette[{}]", slot, slot);
        }
    }
}
