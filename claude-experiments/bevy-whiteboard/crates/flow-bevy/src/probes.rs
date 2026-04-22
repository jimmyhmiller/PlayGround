//! Edge probes: pick an edge → float a live rate readout (pkts/sec) at its
//! midpoint. Rate is computed from `PacketEmitted` events recorded on that
//! edge over a sliding window. The probe is a Bevy-side annotation — the
//! sim doesn't know about it.

use bevy::prelude::*;
use flow::{EdgeId, Event};
use poster_ui::{Bold, Mono, Theme};

use crate::bridge::{EntityMaps, FlowSim, NewEvents};
use crate::camera::{MainCamera, cursor_to_world};
use crate::tool::{ActiveTool, Tool};

pub struct ProbesPlugin;
impl Plugin for ProbesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProbeSamples>()
            .add_systems(
                Update,
                (
                    // Must run after bridge's `collect_new_events` populates
                    // `NewEvents` — otherwise the probe reads an empty bucket
                    // and no samples land in the rolling window.
                    collect_probe_samples.after(crate::bridge::collect_new_events),
                    handle_probe_click,
                    update_probe_positions,
                    update_probe_labels,
                )
                    .chain(),
            );
    }
}

/// Rolling window of recent emission timestamps per edge, in **sim
/// nanoseconds**. Sim-time (not real-time) so the probe's reading stays
/// true regardless of how fast the user has cranked playback — a 10/s
/// generator reads "10/s" at 1× and at 4× alike.
#[derive(Resource, Default)]
pub struct ProbeSamples {
    pub per_edge: std::collections::HashMap<EdgeId, std::collections::VecDeque<u64>>,
}

/// Rolling window length in sim time. A 2-sim-second window averages out
/// short-period jitter without lagging too far behind live rate changes.
const WINDOW_NS: u64 = 2_000_000_000;

/// Marker on a probe entity. Observes either an edge (showing packet
/// rate) or a named reading on a single node. Node probes are domain-
/// agnostic: the reader `fn` is supplied by the gadget module and the
/// probe system just runs it every frame.
#[derive(Component, Clone, Copy)]
pub struct Probe {
    pub target: ProbeTarget,
}

#[derive(Debug, Clone, Copy)]
pub enum ProbeTarget {
    Edge(EdgeId),
    Node {
        node: flow::NodeId,
        label: &'static str,
        /// Reads the formatted value off a sim node. Fn pointers are
        /// `Copy + 'static`, so they sit directly inside the component.
        reader: fn(&flow::Node) -> String,
        /// Stacking index — probes for the same node stack vertically
        /// above it, in the order the gadget declared them.
        slot_index: usize,
    },
}

impl Probe {
    pub fn edge(id: EdgeId) -> Self {
        Self { target: ProbeTarget::Edge(id) }
    }
    pub fn node(node: flow::NodeId, spec: crate::gadgets::ProbeSpec, slot_index: usize) -> Self {
        Self {
            target: ProbeTarget::Node {
                node,
                label: spec.label,
                reader: spec.read,
                slot_index,
            },
        }
    }
}

/// Marker on the text child inside the probe that the label-sync writes to.
#[derive(Component)]
struct ProbeLabel;

// ──────────────────────────────────────────────────────────────
// Sample collection
// ──────────────────────────────────────────────────────────────

fn collect_probe_samples(
    evs: Res<NewEvents>,
    flow: Res<FlowSim>,
    mut samples: ResMut<ProbeSamples>,
) {
    for ev in &evs.0 {
        let Event::PacketEmitted { from, to, at_ns, .. } = ev else { continue };
        // Match event to an edge id.
        let Some(eid) = flow
            .sim
            .edges
            .iter()
            .find(|(_, e)| e.from == *from && e.to == *to)
            .map(|(id, _)| *id)
        else { continue };
        samples.per_edge.entry(eid).or_default().push_back(*at_ns);
    }
    // Evict samples older than the window — a running cleanup amortised
    // across frames keeps the deque small even on unused edges.
    let now = flow.sim.now_ns;
    let cutoff = now.saturating_sub(WINDOW_NS);
    for q in samples.per_edge.values_mut() {
        while q.front().copied().map_or(false, |t| t < cutoff) {
            q.pop_front();
        }
    }
}

// ──────────────────────────────────────────────────────────────
// Placement
// ──────────────────────────────────────────────────────────────

fn handle_probe_click(
    mut commands: Commands,
    buttons: Res<ButtonInput<MouseButton>>,
    mut active: ResMut<ActiveTool>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    flow: Res<FlowSim>,
    maps: Res<EntityMaps>,
    nodes: Query<(Entity, &Transform, &crate::nodes::NodeKind), With<crate::bridge::FlowNodeRef>>,
    ui: Query<&Interaction>,
    theme: Res<Theme>,
) {
    if !matches!(active.0, Tool::Probe) { return; }
    if !buttons.just_pressed(MouseButton::Left) { return; }
    if poster_ui::pointer_over_ui(&ui) { return; }
    let Some(world) = cursor_to_world(&windows, &cams) else { return; };

    // Node hit wins over edge proximity — the body's bbox is what the
    // user is most likely aiming at if the cursor is close to a node.
    use crate::nodes::{body_shape, hit_size};
    let node_hit = nodes.iter().find(|(_, tf, kind)| {
        let half = hit_size(&body_shape(kind.0)) * 0.5;
        let p = tf.translation.truncate();
        (world - p).abs().cmple(half).all()
    });
    if let Some((entity, tf, kind)) = node_hit {
        let Some(&nid) = maps.entity_to_node.get(&entity) else { return; };
        let specs = crate::gadgets::probes_for_kind(kind.0);
        if specs.is_empty() {
            // Routers (or any kind with no declared probes) silently do
            // nothing. Keep the tool active so the click lands elsewhere.
            return;
        }
        for (i, spec) in specs.iter().enumerate() {
            let pos = tf.translation.truncate()
                + Vec2::new(0.0, node_probe_offset_y(kind.0, i));
            spawn_probe_entity(&mut commands, &theme, pos, Probe::node(nid, *spec, i));
        }
        active.0 = Tool::Select;
        return;
    }

    // Otherwise, find the nearest rendered (non-self-loop) edge within a
    // hit radius and probe that.
    const HIT_RADIUS: f32 = 20.0;
    let mut best: Option<(EdgeId, f32, Vec2)> = None;
    for (eid, edge) in flow.sim.edges.iter() {
        if edge.from == edge.to { continue; }
        let Some(&from_ent) = maps.node_to_entity.get(&edge.from) else { continue };
        let Some(&to_ent) = maps.node_to_entity.get(&edge.to) else { continue };
        let Ok((_, tf_from, _)) = nodes.get(from_ent) else { continue };
        let Ok((_, tf_to, _)) = nodes.get(to_ent) else { continue };
        let a = tf_from.translation.truncate();
        let b = tf_to.translation.truncate();
        let (dist, mid) = segment_distance_and_midpoint(world, a, b);
        if dist <= HIT_RADIUS && best.map_or(true, |(_, d, _)| dist < d) {
            best = Some((*eid, dist, mid));
        }
    }
    let Some((eid, _, mid)) = best else { return; };
    spawn_probe_entity(&mut commands, &theme, mid, Probe::edge(eid));
    active.0 = Tool::Select;
}

fn spawn_probe_entity(
    commands: &mut Commands,
    theme: &Theme,
    world_pos: Vec2,
    probe: Probe,
) {
    commands
        .spawn((
            Transform::from_translation(world_pos.extend(4.0)),
            GlobalTransform::default(),
            Visibility::Inherited,
            probe,
        ))
        .with_children(|p| {
            p.spawn((
                Text2d::new("—"),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.ink),
                Bold,
                Mono,
                Transform::from_xyz(0.0, 0.0, 0.1),
                ProbeLabel,
            ));
        });
}

/// Vertical offset above a node for its Nth probe. Base height clears the
/// node body and its always-on state label; each extra spec stacks 18 px
/// higher so labels don't collide.
fn node_probe_offset_y(kind: crate::gadgets::Kind, slot_index: usize) -> f32 {
    use crate::nodes::{body_shape, hit_size};
    let half_h = hit_size(&body_shape(kind)).y * 0.5;
    half_h + 30.0 + slot_index as f32 * 18.0
}

fn segment_distance_and_midpoint(p: Vec2, a: Vec2, b: Vec2) -> (f32, Vec2) {
    let ab = b - a;
    let len_sq = ab.length_squared().max(1e-6);
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    let proj = a + ab * t;
    (proj.distance(p), (a + b) * 0.5)
}

// ──────────────────────────────────────────────────────────────
// Live updates
// ──────────────────────────────────────────────────────────────

fn update_probe_positions(
    flow: Res<FlowSim>,
    maps: Res<EntityMaps>,
    nodes: Query<(&Transform, &crate::nodes::NodeKind), (With<crate::bridge::FlowNodeRef>, Without<Probe>)>,
    mut probes: Query<(&Probe, &mut Transform)>,
) {
    for (probe, mut tf) in probes.iter_mut() {
        match probe.target {
            ProbeTarget::Edge(eid) => {
                let Some(edge) = flow.sim.edges.get(&eid) else { continue };
                let Some(&from_ent) = maps.node_to_entity.get(&edge.from) else { continue };
                let Some(&to_ent) = maps.node_to_entity.get(&edge.to) else { continue };
                let Ok((a_tf, _)) = nodes.get(from_ent) else { continue };
                let Ok((b_tf, _)) = nodes.get(to_ent) else { continue };
                let mid = (a_tf.translation.truncate() + b_tf.translation.truncate()) * 0.5;
                tf.translation.x = mid.x;
                tf.translation.y = mid.y;
            }
            ProbeTarget::Node { node, slot_index, .. } => {
                let Some(&ent) = maps.node_to_entity.get(&node) else { continue };
                let Ok((ntf, kind)) = nodes.get(ent) else { continue };
                tf.translation.x = ntf.translation.x;
                tf.translation.y = ntf.translation.y + node_probe_offset_y(kind.0, slot_index);
            }
        }
    }
}

/// Current rate on `edge` in packets-per-**sim**-second over the last
/// [`WINDOW_NS`] window. Sim-time means the rate a probe reports is
/// independent of the user's playback speed: cranking to 4× shows the
/// same rate as 1×, it just arrives there faster.
pub fn rate_for_edge(samples: &ProbeSamples, edge: EdgeId) -> f32 {
    let q = match samples.per_edge.get(&edge) {
        Some(q) => q,
        None => return 0.0,
    };
    let window_s = WINDOW_NS as f32 / 1_000_000_000.0;
    q.len() as f32 / window_s
}

fn update_probe_labels(
    samples: Res<ProbeSamples>,
    flow: Res<FlowSim>,
    probes: Query<(&Probe, &Children)>,
    mut labels: Query<&mut Text2d, With<ProbeLabel>>,
) {
    for (probe, kids) in probes.iter() {
        let label = match probe.target {
            ProbeTarget::Edge(eid) => {
                let rate = rate_for_edge(&samples, eid);
                format_rate(rate)
            }
            ProbeTarget::Node { node, label, reader, .. } => {
                match flow.sim.nodes.get(&node) {
                    Some(n) => format!("{} {}", label, reader(n)),
                    None => "—".into(),
                }
            }
        };
        for kid in kids.iter() {
            if let Ok(mut t) = labels.get_mut(kid) {
                if t.0 != label { t.0 = label.clone(); }
            }
        }
    }
}

fn format_rate(rate: f32) -> String {
    if rate <= 0.0 { "—".into() }
    else if rate >= 10.0 { format!("{:.0}/s", rate) }
    else { format!("{:.1}/s", rate) }
}
