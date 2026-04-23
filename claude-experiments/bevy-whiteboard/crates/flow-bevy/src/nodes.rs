//! Node entities: spawning, rendering, dragging, selection.
//!
//! Each node is a parent entity whose `Transform` is the authoritative world
//! position. Children (shadow, border, body, glyph, label) carry the visual
//! layering via local-z offsets. Dragging moves the parent and all children
//! follow for free.

use bevy::prelude::*;
use flow::NodeId;
use poster_ui::{Bold, caps_spaced};

use crate::bridge::{EntityMaps, FlowNodeRef, FlowSim};
use crate::camera::{MainCamera, cursor_to_world};
use crate::gadgets::{Kind, spawn as spawn_gadget};
use crate::theme::Theme;
use crate::tool::{ActiveSlot, ActiveTool, NodeColors, Tool};

pub struct NodesPlugin;
impl Plugin for NodesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DragState>()
            .init_resource::<Selection>()
            .init_resource::<NodeCounter>()
            .add_systems(Update, (
                handle_drop,
                begin_drag,
                drag_selected,
                end_drag,
                draw_selection_outline,
                sync_node_labels,
                sync_data_dot_colors,
                sync_node_state_labels,
                delete_selected,
            ).chain());
    }
}

// DemoSeedPlugin moved to lib.rs — the seed now dispatches through the
// ExamplesPlugin's LoadExample message instead of a bespoke Startup
// system, so there's one code path for "set up this scenario."

/// Kind sticks to the Bevy entity so rendering knows what shape to use.
#[derive(Component, Clone, Copy)]
pub struct NodeKind(pub Kind);

/// Auto-incrementing suffix for names like "Worker_3".
#[derive(Resource, Default)]
pub struct NodeCounter(pub u32);

#[derive(Resource, Default)]
pub struct DragState {
    pub entity: Option<Entity>,
    pub offset: Vec2,
}

#[derive(Resource, Default)]
pub struct Selection {
    pub entity: Option<Entity>,
}

// ---------------- spawn ----------------

/// Which 2d primitive to use for a given kind. Circular bodies for emitters /
/// sinks (they read as "portals" at the ends of a flow), rectangles for
/// processors (they read as "boxes" doing work on a packet).
pub enum BodyShape {
    Circle(f32),
    Rect(Vec2),
}

pub fn body_shape(kind: Kind) -> BodyShape {
    match kind {
        Kind::Generator | Kind::Client | Kind::Sink => BodyShape::Circle(34.0),
        Kind::Worker | Kind::Router => BodyShape::Rect(Vec2::new(80.0, 56.0)),
        Kind::Queue => BodyShape::Rect(Vec2::new(100.0, 46.0)),
    }
}

fn body_mesh(shape: &BodyShape, meshes: &mut Assets<Mesh>) -> Handle<Mesh> {
    match shape {
        BodyShape::Circle(r) => meshes.add(Circle::new(*r)),
        BodyShape::Rect(size) => meshes.add(Rectangle::new(size.x, size.y)),
    }
}

/// Size for hit-testing / selection-outline. Circles get a square bounding
/// box; rectangles use their actual extents. Kept slightly inside the border
/// so dragging feels snappy.
pub fn hit_size(shape: &BodyShape) -> Vec2 {
    match shape {
        BodyShape::Circle(r) => Vec2::splat(r * 2.0),
        BodyShape::Rect(size) => *size,
    }
}

/// Glyph font size — larger for circular emitters (they're the hero shapes),
/// a bit smaller for rectangular processors.
fn glyph_size_for(kind: Kind) -> f32 {
    match kind {
        Kind::Generator | Kind::Client | Kind::Sink => 26.0,
        _ => 20.0,
    }
}

/// Materialize a Bevy entity for a freshly-created Flow node. The parent
/// entity owns the body mesh; children stack shadow / border behind and
/// glyph / label on top via local-z offsets.
pub fn spawn_node_entity(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    maps: &mut EntityMaps,
    theme: &Theme,
    flow_id: NodeId,
    kind: Kind,
    name: String,
    pos: Vec2,
) -> Entity {
    let shape = body_shape(kind);
    let hsize = hit_size(&shape);

    // Body — parent. Filled with paper so the interior reads as "empty",
    // not a coloured swatch. Kind is conveyed by shape + glyph + label, not
    // fill colour (colour-coded bodies turned out to confuse, since the
    // same hues appear on edge packets and data-palette swatches).
    let entity = commands.spawn((
        Mesh2d(body_mesh(&shape, meshes)),
        MeshMaterial2d(materials.add(ColorMaterial::from(theme.paper))),
        Transform::from_translation(pos.extend(1.0)),
        FlowNodeRef(flow_id),
        NodeKind(kind),
    )).id();

    commands.entity(entity).with_children(|parent| {
        // Ink outline — a slightly-larger companion mesh rendered just
        // behind the body. Bevy 0.18 Mesh2d doesn't stroke natively; a
        // darker silhouette underneath reads as a clean outline.
        let (border_mesh, _) = scaled_mesh(&shape, 3.0, meshes);
        parent.spawn((
            Mesh2d(border_mesh),
            MeshMaterial2d(materials.add(ColorMaterial::from(theme.ink))),
            Transform::from_xyz(0.0, 0.0, -0.1),
        ));

        // Offset "print misregistration" shadow — a further-behind, offset
        // copy in the soft-ink colour. Reads as an iso50 screenprint detail.
        let (shadow_mesh, _) = scaled_mesh(&shape, 3.0, meshes);
        parent.spawn((
            Mesh2d(shadow_mesh),
            MeshMaterial2d(materials.add(ColorMaterial::from(theme.ink_soft))),
            Transform::from_xyz(6.0, -6.0, -0.2),
        ));

        // Internal glyph — the kind icon. Centered in the body. Always
        // ink-on-paper now that the body is uniform.
        parent.spawn((
            Text2d::new(kind.glyph().to_string()),
            TextColor(theme.ink),
            TextFont { font_size: glyph_size_for(kind), ..default() },
            Transform::from_xyz(0.0, 0.0, 0.1),
        ));

        // Data-colour indicator — small coloured dot in the body's top-right
        // corner. Placeholder fills with ink here; the real colour is
        // stamped each frame by `sync_data_dot_colors` from `NodeColors`
        // (which the drop path populates at spawn time). Skipped for
        // Routers: they're neutral forwarders, not tied to any data type.
        if !matches!(kind, Kind::Router) {
            let (dx, dy) = data_dot_offset(&shape);
            parent.spawn((
                Mesh2d(meshes.add(Circle::new(5.0))),
                MeshMaterial2d(materials.add(ColorMaterial::from(theme.ink))),
                Transform::from_xyz(dx, dy, 0.2),
                NodeColorDot(flow_id),
            ));
        }

        // Label below — tracked caps, bold.
        parent.spawn((
            Text2d::new(caps_spaced(&name)),
            TextColor(theme.ink),
            TextFont { font_size: 11.0, ..default() },
            Transform::from_xyz(0.0, -hsize.y * 0.5 - 12.0, 0.1),
            Bold,
            NodeNameText,
        ));

        // Live state label ABOVE the body. Content is kind-specific —
        // rate for generators, fill for queues, service time for
        // workers, absorbed count for sinks. `sync_node_state_labels`
        // resolves the value each frame from the sim slots.
        parent.spawn((
            Text2d::new(""),
            TextColor(theme.ink_soft),
            TextFont { font_size: 10.0, ..default() },
            Transform::from_xyz(0.0, hsize.y * 0.5 + 14.0, 0.1),
            poster_ui::Mono,
            NodeStateLabel(flow_id),
        ));
    });

    maps.node_to_entity.insert(flow_id, entity);
    maps.entity_to_node.insert(entity, flow_id);
    entity
}

/// Build a larger copy of the node's body shape for use as a border / shadow
/// layer. `pad` is the extra size in world units on each side.
fn scaled_mesh(
    shape: &BodyShape,
    pad: f32,
    meshes: &mut Assets<Mesh>,
) -> (Handle<Mesh>, BodyShape) {
    match shape {
        BodyShape::Circle(r) => {
            let new = BodyShape::Circle(r + pad);
            let handle = body_mesh(&new, meshes);
            (handle, new)
        }
        BodyShape::Rect(size) => {
            let new = BodyShape::Rect(Vec2::new(size.x + pad * 2.0, size.y + pad * 2.0));
            let handle = body_mesh(&new, meshes);
            (handle, new)
        }
    }
}

/// Marker on the `Text2d` child that carries the node's name — the label
/// sync system only touches entities with this component (so it doesn't
/// clobber the internal glyph, which is also a `Text2d` child).
#[derive(Component)]
pub struct NodeNameText;

/// Marker on the small coloured dot in each node's top-right corner. The
/// inner `NodeId` lets `sync_data_dot_colors` look the right entry up in
/// `NodeColors` — the body's parent entity carries `FlowNodeRef` too, but
/// reading it requires traversing the hierarchy; keeping the id on the dot
/// itself makes the sync system a single Query.
#[derive(Component)]
pub struct NodeColorDot(pub NodeId);

/// Marker on the state-readout `Text2d` that floats above every node.
/// Carries the node's `NodeId` so the live-sync system can look up its
/// current stats in the sim without walking the hierarchy.
#[derive(Component)]
pub struct NodeStateLabel(pub NodeId);

/// Where the data-colour dot sits relative to the body's centre. Top-right
/// corner with a small inset so it doesn't hug the border.
fn data_dot_offset(shape: &BodyShape) -> (f32, f32) {
    match shape {
        BodyShape::Circle(r) => {
            // Place on the upper-right quadrant at ~45°, inside the circle.
            let k = r / std::f32::consts::SQRT_2 - 3.0;
            (k, k)
        }
        BodyShape::Rect(size) => {
            (size.x * 0.5 - 8.0, size.y * 0.5 - 8.0)
        }
    }
}

// ---------------- drop from palette ----------------

fn handle_drop(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut flow: ResMut<FlowSim>,
    mut maps: ResMut<EntityMaps>,
    mut node_colors: ResMut<NodeColors>,
    theme: Res<Theme>,
    active_slot: Res<ActiveSlot>,
    mut active: ResMut<ActiveTool>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut counter: ResMut<NodeCounter>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    ui: Query<&Interaction>,
) {
    let Tool::Drop(kind) = active.0 else { return; };
    if !buttons.just_pressed(MouseButton::Left) { return; }
    // Clicking a palette button also arrives as a left-press — without this
    // guard, selecting a tool would immediately drop a node at wherever the
    // cursor happened to be (and flip back to Select).
    if poster_ui::pointer_over_ui(&ui) { return; }
    let Some(pos) = cursor_to_world(&windows, &cams) else { return; };

    counter.0 += 1;
    let name = format!("{}_{}", kind.label(), counter.0);
    let slot = active_slot.0.min(theme.data.len() - 1);
    // Router ignores the slot (passes 0 as a placeholder) — it doesn't
    // store a colour anyway. Every other kind stores `color = Int(slot)`
    // in its sim slots so the router's `slot_of(n, "color")` filter can
    // pick the matching downstream branch.
    let id = spawn_gadget(&mut flow.sim, kind, &name, slot);
    spawn_node_entity(&mut commands, &mut meshes, &mut materials, &mut maps, &theme, id, kind, name, pos);
    // Snapshot the active data-slot colour onto this node so its emitted
    // packets render in that hue. Theme swaps after drop won't recolour
    // it. Routers are untyped — they forward whatever arrives, so they
    // don't get a colour tag.
    if !matches!(kind, Kind::Router) {
        node_colors.0.insert(id, theme.data[slot]);
    }

    // Return to Select after a single drop. Hold-to-keep-dropping TBD.
    active.0 = Tool::Select;
}

// ---------------- drag ----------------

fn begin_drag(
    buttons: Res<ButtonInput<MouseButton>>,
    mut drag: ResMut<DragState>,
    mut selection: ResMut<Selection>,
    active: Res<ActiveTool>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    nodes: Query<(Entity, &Transform, &NodeKind), With<FlowNodeRef>>,
    ui: Query<&Interaction>,
) {
    if !matches!(active.0, Tool::Select) { return; }
    if !buttons.just_pressed(MouseButton::Left) { return; }
    if poster_ui::pointer_over_ui(&ui) { return; }
    let Some(world) = cursor_to_world(&windows, &cams) else { return; };
    // Pick topmost node under cursor.
    let mut picked: Option<(Entity, Vec2)> = None;
    for (e, tf, kind) in nodes.iter() {
        let size = hit_size(&body_shape(kind.0));
        let min = tf.translation.truncate() - size * 0.5;
        let max = tf.translation.truncate() + size * 0.5;
        if world.x >= min.x && world.x <= max.x && world.y >= min.y && world.y <= max.y {
            picked = Some((e, tf.translation.truncate()));
        }
    }
    if let Some((e, p)) = picked {
        drag.entity = Some(e);
        drag.offset = world - p;
        selection.entity = Some(e);
    } else {
        selection.entity = None;
    }
}

fn drag_selected(
    drag: Res<DragState>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    mut q: Query<&mut Transform, With<FlowNodeRef>>,
) {
    if !buttons.pressed(MouseButton::Left) { return; }
    let Some(e) = drag.entity else { return; };
    let Some(world) = cursor_to_world(&windows, &cams) else { return; };
    if let Ok(mut tf) = q.get_mut(e) {
        let new_pos = world - drag.offset;
        tf.translation.x = new_pos.x;
        tf.translation.y = new_pos.y;
    }
}

fn end_drag(buttons: Res<ButtonInput<MouseButton>>, mut drag: ResMut<DragState>) {
    if buttons.just_released(MouseButton::Left) {
        drag.entity = None;
    }
}

// ---------------- selection outline ----------------

#[derive(Component)]
struct SelectionOutline;

fn draw_selection_outline(
    mut commands: Commands,
    selection: Res<Selection>,
    outlines: Query<Entity, With<SelectionOutline>>,
    mut gizmos: Gizmos,
    nodes: Query<(&Transform, &NodeKind), With<FlowNodeRef>>,
    theme: Res<Theme>,
) {
    let _ = (&mut commands, outlines);
    let Some(e) = selection.entity else { return; };
    let Ok((tf, kind)) = nodes.get(e) else { return; };
    let size = hit_size(&body_shape(kind.0)) + Vec2::splat(14.0);
    gizmos.rect_2d(tf.translation.truncate(), size, theme.accent);
}

// ---------------- label sync ----------------

/// Paint the always-on state label above each node. Format is
/// kind-specific:
///
///   - Generator / Client — emission rate derived from `period_ns`.
///   - Queue              — "N queued" where N is the buffer length.
///   - Worker             — "Xms" service time.
///   - Sink               — "N absorbed" from the `count` slot.
///   - Router             — no label (neutral forwarder with no state).
///
/// Runs every frame; only writes when the formatted string actually
/// changes so we don't churn the text system.
fn sync_node_state_labels(
    flow: Res<FlowSim>,
    mut labels: Query<(&NodeStateLabel, &ChildOf, &mut Text2d)>,
    parent_kind_q: Query<&NodeKind>,
) {
    for (label, child_of, mut text) in labels.iter_mut() {
        let Ok(kind) = parent_kind_q.get(child_of.parent()) else { continue };
        let Some(node) = flow.sim.nodes.get(&label.0) else { continue };
        let formatted = format_node_state(kind.0, node);
        if text.0 != formatted { text.0 = formatted; }
    }
}

fn format_node_state(kind: Kind, node: &flow::Node) -> String {
    match kind {
        Kind::Generator | Kind::Client => {
            let period_ns = match node.slots.get("period_ns") {
                Some(flow::Value::Int(i)) => *i,
                _ => 0,
            };
            if period_ns <= 0 { String::new() }
            else {
                let rate = 1_000_000_000.0 / period_ns as f64;
                if rate >= 10.0 { format!("{:.0}/s", rate) } else { format!("{:.1}/s", rate) }
            }
        }
        Kind::Queue => {
            let len = match node.slots.get("len") {
                Some(flow::Value::Int(i)) => *i,
                _ => 0,
            };
            format!("{} queued", len)
        }
        Kind::Worker => {
            let ns = match node.slots.get("service_ns") {
                Some(flow::Value::Int(i)) => *i,
                _ => 0,
            };
            let ms = ns / 1_000_000;
            format!("{}ms", ms)
        }
        Kind::Sink => {
            let count = match node.slots.get("count") {
                Some(flow::Value::Int(i)) => *i,
                _ => 0,
            };
            format!("{} absorbed", count)
        }
        Kind::Router => String::new(),
    }
}

/// Keep each node's data-colour dot painted with whatever colour
/// `NodeColors` currently holds for that node. Cheap: only touches
/// materials whose colour disagrees with the expected value.
fn sync_data_dot_colors(
    node_colors: Res<NodeColors>,
    theme: Res<Theme>,
    dots: Query<(&NodeColorDot, &MeshMaterial2d<ColorMaterial>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for (dot, mat_handle) in dots.iter() {
        let want = node_colors.0.get(&dot.0).copied().unwrap_or(theme.accent);
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            if mat.color != want {
                mat.color = want;
            }
        }
    }
}

fn sync_node_labels(
    flow: Res<FlowSim>,
    q: Query<(&FlowNodeRef, &Children)>,
    mut texts: Query<&mut Text2d, With<NodeNameText>>,
) {
    for (node_ref, kids) in q.iter() {
        let name = match flow.sim.nodes.get(&node_ref.0) {
            Some(n) => caps_spaced(&n.name),
            None => continue,
        };
        for kid in kids.iter() {
            if let Ok(mut t) = texts.get_mut(kid) {
                if t.0 != name { t.0 = name.clone(); }
            }
        }
    }
}

// ---------------- delete ----------------

fn delete_selected(
    mut commands: Commands,
    mut flow: ResMut<FlowSim>,
    mut maps: ResMut<EntityMaps>,
    mut selection: ResMut<Selection>,
    keys: Res<ButtonInput<KeyCode>>,
    q: Query<&FlowNodeRef>,
) {
    if !(keys.just_pressed(KeyCode::KeyX) || keys.just_pressed(KeyCode::Delete) || keys.just_pressed(KeyCode::Backspace)) {
        return;
    }
    let Some(e) = selection.entity.take() else { return; };
    let Ok(node_ref) = q.get(e) else { return; };
    let nid = node_ref.0;
    flow.sim.despawn_node(nid);
    commands.entity(e).despawn();
    maps.node_to_entity.remove(&nid);
    maps.entity_to_node.remove(&e);
}
