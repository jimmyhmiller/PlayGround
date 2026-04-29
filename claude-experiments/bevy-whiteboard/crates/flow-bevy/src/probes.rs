//! Node probes: pick a node → float live readouts above it for every
//! probe declared on its class. Rate-on-edge probes were removed —
//! the sample collector cost was O(events × edges) per frame and
//! dominated frame time on dense canvases (~15 ms/frame on Life
//! 30×30). If we ever re-introduce edge rates, drive them off the
//! sim instead of resampling events at the visualization layer.

use bevy::prelude::*;
use poster_ui::{Bold, Mono, Theme};

use crate::bridge::EntityMaps;
use crate::camera::{MainCamera, cursor_to_world};
use crate::sim_driver::SimSnapshotRes;
use crate::tool::{ActiveTool, Tool};

pub struct ProbesPlugin;
impl Plugin for ProbesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                handle_probe_click,
                update_probe_positions,
                update_probe_labels,
            )
                .chain(),
        );
    }
}

/// Marker on a probe entity. Pinned to a node, reads a class-declared
/// probe by label.
#[derive(Component, Clone)]
pub struct Probe {
    pub node: flow::NodeId,
    /// Probe label as declared in the class's `probes { }` block.
    /// Looked up on the snapshot at display time.
    pub label: String,
    /// Stacking index — probes for the same node stack vertically
    /// above it, in declaration order.
    pub slot_index: usize,
}

impl Probe {
    pub fn node(node: flow::NodeId, label: String, slot_index: usize) -> Self {
        Self { node, label, slot_index }
    }
}

/// Marker on the text child inside the probe that the label-sync writes to.
#[derive(Component)]
struct ProbeLabel;

// ──────────────────────────────────────────────────────────────
// Placement
// ──────────────────────────────────────────────────────────────

fn handle_probe_click(
    mut commands: Commands,
    buttons: Res<ButtonInput<MouseButton>>,
    mut active: ResMut<ActiveTool>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    snapshot: Res<SimSnapshotRes>,
    maps: Res<EntityMaps>,
    nodes: Query<(Entity, &Transform, &crate::nodes::BodyShape), With<crate::bridge::FlowNodeRef>>,
    ui: Query<&Interaction>,
    theme: Res<Theme>,
) {
    if !matches!(active.0, Tool::Probe) { return; }
    if !buttons.just_pressed(MouseButton::Left) { return; }
    if poster_ui::pointer_over_ui(&ui) { return; }
    let Some(world) = cursor_to_world(&windows, &cams) else { return; };

    use crate::nodes::hit_size;
    let node_hit = nodes.iter().find(|(_, tf, shape)| {
        let half = hit_size(shape) * 0.5;
        let p = tf.translation.truncate();
        (world - p).abs().cmple(half).all()
    });
    let Some((entity, tf, shape)) = node_hit else { return };
    let Some(&nid) = maps.entity_to_node.get(&entity) else { return; };
    let labels = snapshot.0.nodes.get(&nid)
        .map(|n| n.probe_labels.clone())
        .unwrap_or_default();
    if labels.is_empty() {
        // Routers (or any class with no declared probes) silently do
        // nothing. Keep the tool active so the click lands elsewhere.
        return;
    }
    for (i, label) in labels.into_iter().enumerate() {
        let pos = tf.translation.truncate()
            + Vec2::new(0.0, node_probe_offset_y(shape, i));
        spawn_probe_entity(&mut commands, &theme, pos, Probe::node(nid, label, i));
    }
    active.0 = Tool::Select;
}

fn spawn_probe_entity(
    commands: &mut Commands,
    theme: &Theme,
    world_pos: Vec2,
    probe: Probe,
) {
    let host = probe.node;
    commands
        .spawn((
            Transform::from_translation(world_pos.extend(4.0)),
            GlobalTransform::default(),
            Visibility::Inherited,
            // Probes inherit the scope of the node they attach to. The
            // central visibility system handles the rest — at top
            // level a probe on a Life cell hides; drill in and it
            // appears alongside the cell.
            crate::compound::Scoped(host),
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
fn node_probe_offset_y(shape: &crate::nodes::BodyShape, slot_index: usize) -> f32 {
    use crate::nodes::hit_size;
    let half_h = hit_size(shape).y * 0.5;
    half_h + 30.0 + slot_index as f32 * 18.0
}

// ──────────────────────────────────────────────────────────────
// Live updates
// ──────────────────────────────────────────────────────────────

fn update_probe_positions(
    maps: Res<EntityMaps>,
    nodes: Query<(&Transform, &crate::nodes::BodyShape), (With<crate::bridge::FlowNodeRef>, Without<Probe>)>,
    mut probes: Query<(&Probe, &mut Transform)>,
) {
    for (probe, mut tf) in probes.iter_mut() {
        let Some(&ent) = maps.node_to_entity.get(&probe.node) else { continue };
        let Ok((ntf, shape)) = nodes.get(ent) else { continue };
        tf.translation.x = ntf.translation.x;
        tf.translation.y = ntf.translation.y + node_probe_offset_y(shape, probe.slot_index);
    }
}

fn update_probe_labels(
    snapshot: Res<SimSnapshotRes>,
    probes: Query<(&Probe, &Children)>,
    mut labels: Query<&mut Text2d, With<ProbeLabel>>,
) {
    for (probe, kids) in probes.iter() {
        let reading = snapshot.0.nodes.get(&probe.node)
            .and_then(|n| n.probe_readings.iter()
                .find(|(l, _)| l == &probe.label)
                .map(|(_, v)| v.clone()));
        let label = match reading {
            Some(value) => format!("{} {}", probe.label, value),
            None => "—".into(),
        };
        for kid in kids.iter() {
            if let Ok(mut t) = labels.get_mut(kid) {
                if t.0 != label { t.0 = label.clone(); }
            }
        }
    }
}
