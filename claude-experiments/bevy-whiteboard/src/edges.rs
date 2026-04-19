use crate::bridge::{EntityMaps, SimEdgeRef, SimNodeRef, SimResource, register_edge};
use crate::camera::MainCamera;
use crate::nodes::SimNode;
use crate::palette::pointer_over_ui;
use crate::sim::Node as SimNodeState;
use crate::theme::Theme;
use crate::tool::{ActiveTool, Tool};
use bevy::prelude::*;

pub struct EdgesPlugin;

impl Plugin for EdgesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EdgeDrawState>()
            .init_resource::<AnalyticalRates>()
            .add_systems(
                Update,
                (
                    pick_nodes_for_edge,
                    place_probe_on_click,
                    recompute_analytical_rates,
                    draw_edges_gizmos,
                    update_probe_visuals,
                )
                    .chain(),
            );
    }
}

/// Analytical steady-state rate for every edge, recomputed from node
/// parameters each frame. Swapped in for the old event-counting probe so
/// readouts don't jitter, stay correct while paused, and never lie.
#[derive(Resource, Default)]
pub struct AnalyticalRates {
    pub per_edge: std::collections::HashMap<crate::sim::EdgeId, f64>,
}

fn recompute_analytical_rates(
    sim_res: Res<SimResource>,
    mut rates: ResMut<AnalyticalRates>,
) {
    // The sim's topology + node parameters are the only inputs here, so even
    // when paused this returns the "what would flow" numbers. Cheap enough to
    // run every frame for the whiteboard-scale topologies we expect.
    rates.per_edge = sim_res.0.analytical_edge_rates();
}

/// An edge connects two Bevy node entities. The sim-side EdgeId lives on a
/// sibling `SimEdgeRef` component.
#[derive(Component, Clone, Copy)]
pub struct Edge {
    pub from: Entity,
    pub to: Entity,
}

#[derive(Resource, Default)]
pub struct EdgeDrawState {
    pub pending_from: Option<Entity>,
}

/// A probe targets either a specific edge (and measures flow rate of packets
/// crossing it) or a node (and reads cumulative stats straight from the sim).
#[derive(Clone, Copy)]
pub enum ProbeTarget {
    Edge(Entity),
    Node(Entity),
}

#[derive(Component)]
pub struct Probe {
    pub target: ProbeTarget,
}

fn pick_nodes_for_edge(
    mouse: Res<ButtonInput<MouseButton>>,
    active_tool: Res<ActiveTool>,
    ui: Query<&Interaction>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    nodes: Query<(Entity, &Transform, &SimNode)>,
    existing_edges: Query<&Edge>,
    mut draw_state: ResMut<EdgeDrawState>,
    mut commands: Commands,
    mut sim_res: ResMut<SimResource>,
    mut maps: ResMut<EntityMaps>,
) {
    if active_tool.0 != Tool::Edge {
        if draw_state.pending_from.is_some() {
            draw_state.pending_from = None;
        }
        return;
    }
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    if pointer_over_ui(&ui) {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    let Ok((cam, cam_tf)) = cams.single() else { return };
    let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) else { return };

    let hit = nodes.iter().find(|(_, tf, sn)| {
        let half = sn.kind.size() / 2.0;
        let center = tf.translation.truncate();
        (world.x - center.x).abs() <= half.x && (world.y - center.y).abs() <= half.y
    });

    let Some((entity, _, _)) = hit else {
        draw_state.pending_from = None;
        return;
    };

    match draw_state.pending_from {
        None => {
            draw_state.pending_from = Some(entity);
        }
        Some(from) if from == entity => {
            draw_state.pending_from = None;
        }
        Some(from) => {
            // Interpret the gesture. If the user drew from a
            // pull-capable consumer (Worker etc.) toward a data
            // source (Queue etc.), that's the natural gesture for
            // "this one pulls from that one" — swap the sim
            // endpoints so the sim edge represents data flow, and
            // flag the edge as Pull mode so the sim's pull executor
            // drives it.
            let (sim_from, sim_to, is_pull) =
                resolve_edge_kind(from, entity, &maps, &sim_res);
            let dup = existing_edges
                .iter()
                .any(|e| e.from == sim_from && e.to == sim_to);
            if !dup {
                let edge_entity = commands
                    .spawn(Edge {
                        from: sim_from,
                        to: sim_to,
                    })
                    .id();
                if let Some(eid) =
                    register_edge(&mut sim_res, &mut maps, edge_entity, sim_from, sim_to)
                {
                    if is_pull {
                        sim_res.0.set_edge_mode(eid, crate::sim::EdgeMode::Pull);
                    }
                    commands.entity(edge_entity).insert(SimEdgeRef(eid));
                }
            }
            draw_state.pending_from = None;
        }
    }
}

/// Interpret the user's edge-drawing gesture. The raw gesture goes
/// from clicked-first (`a`) to clicked-second (`b`); in most cases the
/// sim edge is just (a → b). BUT if `a` is a pull-only consumer
/// (Worker etc. — has `PullInbound`, no `AcceptInbound`) and `b` is a
/// data source (Buffer or ForwardOut in its steps), we swap: the sim
/// edge is (b → a), reading as "Worker pulls from Queue." This
/// matches the natural gesture of pointing *from* the consumer
/// *toward* its data source.
/// Decide `(sim_from, sim_to, is_pull)` from a draw gesture of (a → b).
/// Default: straight push from a to b. If `a` is a pull-capable
/// consumer and `b` is a source-y node, swap endpoints and mark Pull.
fn resolve_edge_kind(
    a: Entity,
    b: Entity,
    maps: &EntityMaps,
    sim_res: &SimResource,
) -> (Entity, Entity, bool) {
    use crate::sim::Step;
    let Some(&a_id) = maps.entity_to_node.get(&a) else { return (a, b, false) };
    let Some(&b_id) = maps.entity_to_node.get(&b) else { return (a, b, false) };
    let Some(a_node) = sim_res.0.nodes.get(&a_id) else { return (a, b, false) };
    let Some(b_node) = sim_res.0.nodes.get(&b_id) else { return (a, b, false) };
    let a_is_pull = a_node.is_pulling() && !a_node.accepts_push();
    let b_is_source = b_node
        .steps
        .iter()
        .any(|s| matches!(s, Step::Buffer { .. } | Step::ForwardOut { .. } | Step::EmitAtRate { .. }));
    if a_is_pull && b_is_source {
        (b, a, true)
    } else {
        (a, b, false)
    }
}

#[allow(clippy::too_many_arguments)]
fn place_probe_on_click(
    mouse: Res<ButtonInput<MouseButton>>,
    active_tool: Res<ActiveTool>,
    ui: Query<&Interaction>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    edges: Query<(Entity, &Edge)>,
    nodes_tf: Query<(Entity, &Transform, &SimNode)>,
    nodes_only_tf: Query<&Transform, With<SimNode>>,
    existing_probes: Query<(Entity, &Probe)>,
    mut commands: Commands,
) {
    if active_tool.0 != Tool::Probe {
        return;
    }
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    if pointer_over_ui(&ui) {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    let Ok((cam, cam_tf)) = cams.single() else { return };
    let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) else { return };

    let node_hit = nodes_tf.iter().find(|(_, tf, sn)| {
        let half = sn.kind.size() / 2.0;
        let c = tf.translation.truncate();
        (world.x - c.x).abs() <= half.x && (world.y - c.y).abs() <= half.y
    });

    let target = if let Some((nid, _, _)) = node_hit {
        ProbeTarget::Node(nid)
    } else {
        let mut best: Option<(Entity, f32)> = None;
        for (eid, edge) in edges.iter() {
            let (Ok(a), Ok(b)) = (nodes_only_tf.get(edge.from), nodes_only_tf.get(edge.to)) else {
                continue;
            };
            let d = dist_to_segment(world, a.translation.truncate(), b.translation.truncate());
            if d < 18.0 && best.map(|(_, bd)| d < bd).unwrap_or(true) {
                best = Some((eid, d));
            }
        }
        let Some((eid, _)) = best else { return };
        ProbeTarget::Edge(eid)
    };

    // Clicking the Probe tool on something already probed removes that probe —
    // the tool toggles, so the user doesn't need a separate delete action.
    let existing = existing_probes.iter().find(|(_, p)| match (p.target, target) {
        (ProbeTarget::Edge(a), ProbeTarget::Edge(b)) => a == b,
        (ProbeTarget::Node(a), ProbeTarget::Node(b)) => a == b,
        _ => false,
    });
    if let Some((probe_entity, _)) = existing {
        commands.entity(probe_entity).despawn();
        return;
    }

    commands.spawn((
        Probe { target },
        Text2d::new("0.0/s"),
        TextFont {
            font_size: 14.0,
            ..default()
        },
        TextColor(Color::srgb(0.1, 0.1, 0.15)),
        Transform::from_xyz(0.0, 0.0, 1.0),
    ));
}

fn dist_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
    let proj = a + ab * t;
    (p - proj).length()
}


fn draw_edges_gizmos(
    mut gizmos: Gizmos,
    theme: Res<Theme>,
    edges: Query<(&Edge, Option<&SimEdgeRef>)>,
    nodes: Query<(&Transform, &SimNode, Option<&SimNodeRef>)>,
    nodes_tf: Query<&Transform, With<SimNode>>,
    draw_state: Res<EdgeDrawState>,
    sim_res: Res<SimResource>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
) {
    let ink = theme.ink;
    let preview = theme.accent;

    for (edge, edge_ref) in edges.iter() {
        let (Ok((a_tf, a_sn, _)), Ok((b_tf, b_sn, _))) =
            (nodes.get(edge.from), nodes.get(edge.to))
        else {
            continue;
        };
        let is_pull = edge_ref
            .and_then(|r| sim_res.0.edges.get(&r.0))
            .map(|e| e.mode == crate::sim::EdgeMode::Pull)
            .unwrap_or(false);

        // Sim edge always stores (from=data-source, to=consumer).
        // Push edges draw in that direction (data flow). Pull edges
        // draw the arrow in the OPPOSITE direction so the arrowhead
        // lands on the source — reading as "this consumer pulls
        // from that producer."
        let (head_tf, head_sn, tail_tf, tail_sn) = if is_pull {
            (a_tf, a_sn, b_tf, b_sn)
        } else {
            (b_tf, b_sn, a_tf, a_sn)
        };
        let tail_center = tail_tf.translation.truncate();
        let head_center = head_tf.translation.truncate();
        let tail_half = tail_sn.kind.size() / 2.0;
        let head_half = head_sn.kind.size() / 2.0;

        let dir = (head_center - tail_center).normalize_or_zero();
        if dir.length_squared() == 0.0 {
            continue;
        }
        let tail_exit = tail_center + dir * border_exit(tail_half, dir);
        let head_entry = head_center - dir * (border_exit(head_half, -dir) + 4.0);

        if is_pull {
            draw_dashed_arrow(&mut gizmos, tail_exit, head_entry, ink);
        } else {
            draw_curved_arrow(&mut gizmos, tail_exit, head_entry, ink);
        }
    }

    if let Some(from) = draw_state.pending_from {
        let Ok(from_tf) = nodes_tf.get(from) else { return };
        let Ok(win) = windows.single() else { return };
        let Some(cursor) = win.cursor_position() else { return };
        let Ok((cam, cam_tf)) = cams.single() else { return };
        let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) else { return };
        let a = from_tf.translation.truncate();
        draw_curved_arrow(&mut gizmos, a, world, preview);
    }
}

/// Ray-into-box exit distance: how far along `dir` from the centre of an
/// axis-aligned box until we hit its border. Used to trim edges at node
/// boundaries without drawing through them.
fn border_exit(half: Vec2, dir: Vec2) -> f32 {
    let tx = if dir.x.abs() > 1e-4 { half.x / dir.x.abs() } else { f32::INFINITY };
    let ty = if dir.y.abs() > 1e-4 { half.y / dir.y.abs() } else { f32::INFINITY };
    tx.min(ty)
}

/// Draw a dashed line from `a` to `b` with a hollow arrowhead at `b`.
/// Used for pull edges: the downstream is actively pulling from upstream,
/// so the edge reads as a demand channel rather than a push flow. The
/// dashing makes it visually distinct at a glance.
fn draw_dashed_arrow(gizmos: &mut Gizmos, a: Vec2, b: Vec2, color: Color) {
    let delta = b - a;
    let dist = delta.length();
    if dist < 1.0 {
        return;
    }
    let dir = delta / dist;
    const DASH: f32 = 6.0;
    const GAP: f32 = 4.0;
    let mut t = 0.0;
    while t < dist {
        let end = (t + DASH).min(dist);
        gizmos.line_2d(a + dir * t, a + dir * end, color);
        t = end + GAP;
    }

    // Hollow arrowhead: three outline strokes only (no fan fill).
    let tip = b;
    let head_perp = Vec2::new(-dir.y, dir.x);
    let head_len = 11.0_f32;
    let head_w = 6.5_f32;
    let back = tip - dir * head_len;
    let left = back + head_perp * head_w;
    let right = back - head_perp * head_w;
    gizmos.line_2d(tip, left, color);
    gizmos.line_2d(tip, right, color);
    gizmos.line_2d(left, right, color);
}

/// Draw a straight edge from `a` to `b` with a filled arrowhead at `b`.
/// Kept deliberately un-curved — the straight routing reads as diagrammatic
/// rather than noodly, and the filled wedge is the only visual flourish the
/// arrow needs.
fn draw_curved_arrow(gizmos: &mut Gizmos, a: Vec2, b: Vec2, color: Color) {
    let delta = b - a;
    let dist = delta.length();
    if dist < 1.0 {
        return;
    }
    let dir = delta / dist;
    gizmos.line_2d(a, b, color);

    // Arrowhead wedge. Base sits `head_len` back from the tip, `head_w`
    // wide on each side.
    let tip = b;
    let head_perp = Vec2::new(-dir.y, dir.x);
    let head_len = 11.0_f32;
    let head_w = 6.5_f32;
    let back = tip - dir * head_len;
    let left = back + head_perp * head_w;
    let right = back - head_perp * head_w;

    // Fake-fill the triangle by drawing a fan of line segments from `tip`
    // to points interpolated along the base. Gizmos don't render filled
    // primitives, and 6 strokes at this size read as a solid wedge without
    // the wireframe look the old three-line arrow had.
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

/// Probe visuals. For edge probes: rate label at the midpoint. For node
/// probes: read cumulative stats straight out of the sim via SimNodeRef.
fn update_probe_visuals(
    mut gizmos: Gizmos,
    mut probes: Query<(&Probe, &mut Transform, &mut Text2d)>,
    edges: Query<&Edge>,
    edge_refs: Query<&SimEdgeRef>,
    nodes: Query<(&Transform, &SimNode, Option<&SimNodeRef>), Without<Probe>>,
    sim_res: Res<SimResource>,
    rates: Res<AnalyticalRates>,
    maps: Res<EntityMaps>,
) {
    let marker_color = Color::srgb(0.1, 0.1, 0.15);
    for (probe, mut tf, mut text) in probes.iter_mut() {
        match probe.target {
            ProbeTarget::Edge(eid) => {
                let Ok(edge) = edges.get(eid) else { continue };
                let (Ok((a_tf, _, _)), Ok((b_tf, _, _))) =
                    (nodes.get(edge.from), nodes.get(edge.to))
                else {
                    continue;
                };
                let a = a_tf.translation.truncate();
                let b = b_tf.translation.truncate();
                let mid = (a + b) / 2.0;
                let dir = (b - a).normalize_or_zero();
                let perp = Vec2::new(-dir.y, dir.x);
                let label_pos = mid + perp * 16.0;
                tf.translation.x = label_pos.x;
                tf.translation.y = label_pos.y;
                // Rate on this edge = analytical steady-state flow.
                let rate = edge_refs
                    .get(eid)
                    .ok()
                    .and_then(|r| rates.per_edge.get(&r.0).copied())
                    .unwrap_or(0.0);
                **text = format_probe_rate(rate);

                let s = 5.0;
                gizmos.line_2d(mid + Vec2::new(0.0, s), mid + Vec2::new(s, 0.0), marker_color);
                gizmos.line_2d(mid + Vec2::new(s, 0.0), mid + Vec2::new(0.0, -s), marker_color);
                gizmos.line_2d(mid + Vec2::new(0.0, -s), mid + Vec2::new(-s, 0.0), marker_color);
                gizmos.line_2d(mid + Vec2::new(-s, 0.0), mid + Vec2::new(0.0, s), marker_color);
            }
            ProbeTarget::Node(nid) => {
                let Ok((node_tf, sn, nref_opt)) = nodes.get(nid) else { continue };
                let c = node_tf.translation.truncate();
                let half = sn.kind.size() / 2.0;
                let label_pos = c + Vec2::new(0.0, -half.y - 22.0);
                tf.translation.x = label_pos.x;
                tf.translation.y = label_pos.y;

                let stats_line = nref_opt
                    .and_then(|r| sim_res.0.nodes.get(&r.0))
                    .map(|n| format_node_stats(n))
                    .unwrap_or_default();
                // Node probe rate = total analytical flow arriving at the node
                // (sum of inbound edge rates). For generators, it's outbound.
                let node_rate = nref_opt
                    .map(|r| node_rate(&sim_res.0, &rates, r.0))
                    .unwrap_or(0.0);
                let _ = &maps; // quiet unused
                **text = format!("{}  {}", format_probe_rate(node_rate), stats_line);

                gizmos.rect_2d(c, (half + Vec2::splat(10.0)) * 2.0, marker_color);
            }
        }
    }
}

/// Arrival rate at a node = sum of analytical rates on its inbound edges.
/// For pure sources (generators, which have no inbound) fall back to the
/// sum of *outbound* rates so the probe still shows a useful number.
fn node_rate(
    sim: &crate::sim::Sim,
    rates: &AnalyticalRates,
    id: crate::sim::NodeId,
) -> f64 {
    let inbound: f64 = sim
        .edges
        .iter()
        .filter(|(_, e)| e.to == id)
        .map(|(eid, _)| rates.per_edge.get(eid).copied().unwrap_or(0.0))
        .sum();
    if inbound > 0.0 {
        return inbound;
    }
    sim.edges
        .iter()
        .filter(|(_, e)| e.from == id)
        .map(|(eid, _)| rates.per_edge.get(eid).copied().unwrap_or(0.0))
        .sum()
}

/// Format a probe's rate (pkts/simulated-second). Scales the unit so the
/// number stays in [1, 1000) — the board can mix nanosecond and second
/// pipelines on the same canvas.
fn format_probe_rate(rate: f64) -> String {
    if rate <= 0.0 {
        "0/s".to_string()
    } else if rate >= 1.0e9 {
        format!("{:.2} Gpps", rate / 1.0e9)
    } else if rate >= 1.0e6 {
        format!("{:.2} Mpps", rate / 1.0e6)
    } else if rate >= 1.0e3 {
        format!("{:.2} kpps", rate / 1.0e3)
    } else {
        format!("{:.1}/s", rate)
    }
}

fn format_node_stats(node: &SimNodeState) -> String {
    use crate::sim::NodeKind;
    match node.kind {
        NodeKind::Sink => {
            let mut line = format!("Σ{}  drop:{}", node.sink_total, node.dropped);
            let mut pairs: Vec<_> = node.sink_per_color.iter().collect();
            pairs.sort_by(|a, b| b.1.cmp(a.1));
            for (k, v) in pairs.iter().take(4) {
                line.push_str(&format!("  #{:06x}:{}", k.0, v));
            }
            line
        }
        NodeKind::Worker => {
            format!("proc:{}  drop:{}", node.processed, node.dropped)
        }
        NodeKind::Queue => {
            format!(
                "in:{} out:{} depth:{} lost:{}",
                node.total_in,
                node.total_out,
                node.buffer.len(),
                node.lost
            )
        }
        NodeKind::Generator => {
            format!("emit:{} drop:{}", node.emitted, node.dropped)
        }
        NodeKind::Client => {
            let rtt = if node.rtt_count > 0 {
                node.rtt_sum_ns / node.rtt_count as u64
            } else {
                0
            };
            format!(
                "sent:{} recv:{} drop:{} rtt:{}ns",
                node.sent, node.received, node.dropped, rtt
            )
        }
        NodeKind::Router | NodeKind::Custom => String::new(),
    }
}
