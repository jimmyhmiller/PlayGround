//! Node entities: spawning, rendering, dragging, selection.
//!
//! Each node is a parent entity whose `Transform` is the authoritative world
//! position. Children (shadow, border, body, glyph, label) carry the visual
//! layering via local-z offsets. Dragging moves the parent and all children
//! follow for free.

use std::collections::HashMap;

use bevy::prelude::*;
use flow::NodeId;

use crate::bitmap_label::{
    AtlasMetrics, BitmapLabel, TextAlign, spawn_label_chars,
};
use crate::bridge::{EntityMaps, FlowNodeRef};
use crate::camera::{MainCamera, cursor_to_world};
use crate::gadgets::{Kind, spawn as spawn_gadget};
use crate::sim_driver::{NodeView, SimCommand, SimDriverRes, SimSnapshotRes};
use crate::theme::Theme;
use crate::tool::{ActiveSlot, ActiveTool, NodeColors, Tool};

pub struct NodesPlugin;
impl Plugin for NodesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DragState>()
            .init_resource::<Selection>()
            .init_resource::<NodeCounter>()
            .init_resource::<NodeAssetCache>()
            .add_systems(Update, (
                handle_drop,
                begin_drag,
                drag_selected,
                end_drag,
                clear_stale_selection,
                draw_selection_outline,
                sync_node_labels,
                sync_data_dot_colors,
                sync_node_state_labels,
                sync_binary_slot_paint,
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

/// Which 2d primitive to use for a node body. Circular bodies for emitters /
/// sinks (they read as "portals" at the ends of a flow), rectangles for
/// processors (they read as "boxes" doing work on a packet).
///
/// Stored as a `Component` on each node entity so the canvas-driven shape
/// (from `visual.json`'s `classes` block) and the palette-driven shape
/// (from `Kind`) feed through the same render path. Callers query
/// `&BodyShape` from their node query rather than re-deriving from `Kind`.
#[derive(Component, Clone, Copy, Debug)]
pub enum BodyShape {
    Circle(f32),
    Rect(Vec2),
}

/// Default shape for a built-in palette `Kind`. Used when no per-entity
/// override is supplied (i.e. nodes dropped from the palette, or nodes
/// loaded from a canvas whose `visual.json` doesn't declare a shape for
/// the class).
pub fn body_shape(kind: Kind) -> BodyShape {
    match kind {
        Kind::Generator | Kind::Client | Kind::BackoffClient | Kind::Sink => BodyShape::Circle(34.0),
        Kind::Worker | Kind::Router => BodyShape::Rect(Vec2::new(80.0, 56.0)),
        Kind::Queue => BodyShape::Rect(Vec2::new(100.0, 46.0)),
    }
}

/// Cache of `Handle<Mesh>` and `Handle<ColorMaterial>` deduplicated by
/// shape and color. Without this, every spawned node creates 3+ unique
/// meshes and 4+ unique materials — for the Life canvases with ~900
/// cells, that's thousands of redundant assets feeding the renderer
/// every frame. With it, every same-shape body shares one vertex
/// buffer and every same-color material is one handle.
#[derive(Resource, Default)]
pub struct NodeAssetCache {
    bodies: HashMap<BodyShapeKey, Handle<Mesh>>,
    materials: HashMap<u32, Handle<ColorMaterial>>,
    dot_mesh: Option<Handle<Mesh>>,
}

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
enum BodyShapeKey {
    Circle(u32),
    Rect(u32, u32),
}

impl BodyShapeKey {
    fn from_shape(s: &BodyShape) -> Self {
        match s {
            BodyShape::Circle(r) => BodyShapeKey::Circle(r.to_bits()),
            BodyShape::Rect(s) => BodyShapeKey::Rect(s.x.to_bits(), s.y.to_bits()),
        }
    }
}

fn color_key(c: Color) -> u32 {
    let s = c.to_srgba();
    let q = |f: f32| (f.clamp(0.0, 1.0) * 255.0) as u8;
    u32::from_le_bytes([q(s.red), q(s.green), q(s.blue), q(s.alpha)])
}

impl NodeAssetCache {
    pub fn body_mesh(&mut self, shape: &BodyShape, meshes: &mut Assets<Mesh>) -> Handle<Mesh> {
        let key = BodyShapeKey::from_shape(shape);
        self.bodies
            .entry(key)
            .or_insert_with(|| match shape {
                BodyShape::Circle(r) => meshes.add(Circle::new(*r)),
                BodyShape::Rect(s) => meshes.add(Rectangle::new(s.x, s.y)),
            })
            .clone()
    }

    pub fn material(
        &mut self,
        color: Color,
        materials: &mut Assets<ColorMaterial>,
    ) -> Handle<ColorMaterial> {
        let key = color_key(color);
        self.materials
            .entry(key)
            .or_insert_with(|| materials.add(ColorMaterial::from(color)))
            .clone()
    }

    pub fn dot_mesh(&mut self, meshes: &mut Assets<Mesh>) -> Handle<Mesh> {
        self.dot_mesh
            .get_or_insert_with(|| meshes.add(Circle::new(5.0)))
            .clone()
    }
}

fn padded_shape(shape: &BodyShape, pad: f32) -> BodyShape {
    match shape {
        BodyShape::Circle(r) => BodyShape::Circle(r + pad),
        BodyShape::Rect(s) => BodyShape::Rect(Vec2::new(s.x + pad * 2.0, s.y + pad * 2.0)),
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
        Kind::Generator | Kind::Client | Kind::BackoffClient | Kind::Sink => 26.0,
        _ => 20.0,
    }
}

/// Materialize a Bevy entity for a freshly-created Flow node. The parent
/// entity owns the body mesh; children stack shadow / border behind and
/// glyph / label on top via local-z offsets.
///
/// `shape_override` lets the canvas pipeline supply a class-driven shape
/// from `visual.json`. When `None`, falls back to the built-in
/// [`body_shape`] mapping for the supplied `Kind`. The resulting
/// [`BodyShape`] is attached as a component on the parent so all
/// downstream queries (hit-testing, arrow rendering, selection outline)
/// read it directly without re-deriving from `Kind`.
///
/// `self_painted = true` means the class declares its own visual (a
/// `paint` block in `visual.json` — e.g. LifeCell flipping black/white
/// on `alive`). Suppresses every kind-driven decoration: the data
/// colour dot, the kind glyph, the drop shadow, and the name + state
/// labels. The body fill becomes the entire visual, which is what
/// makes 100×100 grids of cells legible.
///
/// `has_color = true` means the class declares a `color` slot (the
/// gadget participates in data-palette tagging). Drives whether the
/// data-colour dot is drawn — gadgets without a `color` slot (Router,
/// LifeCell, anything else opted out) get no dot, regardless of
/// `Kind`.
pub fn spawn_node_entity(
    commands: &mut Commands,
    cache: &mut NodeAssetCache,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    maps: &mut EntityMaps,
    theme: &Theme,
    metrics: &AtlasMetrics,
    flow_id: NodeId,
    kind: Kind,
    shape_override: Option<BodyShape>,
    self_painted: bool,
    has_color: bool,
    name: String,
    pos: Vec2,
) -> Entity {
    let shape = shape_override.unwrap_or_else(|| body_shape(kind));
    let hsize = hit_size(&shape);

    let body_handle = cache.body_mesh(&shape, meshes);
    let border_shape = padded_shape(&shape, 3.0);
    let border_handle = cache.body_mesh(&border_shape, meshes);
    let dot_handle = cache.dot_mesh(meshes);
    let paper_mat = cache.material(theme.paper, materials);
    let ink_mat = cache.material(theme.ink, materials);
    let ink_soft_mat = cache.material(theme.ink_soft, materials);

    let entity = commands.spawn((
        Mesh2d(body_handle),
        MeshMaterial2d(paper_mat),
        Transform::from_translation(pos.extend(1.0)),
        FlowNodeRef(flow_id),
        NodeKind(kind),
        shape,
    )).id();

    commands.entity(entity).with_children(|parent| {
        // Ink outline.
        parent.spawn((
            Mesh2d(border_handle.clone()),
            MeshMaterial2d(ink_mat.clone()),
            Transform::from_xyz(0.0, 0.0, -0.1),
        ));
        // Drop shadow — kind-driven decoration; self-painted gadgets
        // skip it so dense grids of cells don't bleed shadows into
        // each other.
        if !self_painted {
            parent.spawn((
                Mesh2d(border_handle),
                MeshMaterial2d(ink_soft_mat),
                Transform::from_xyz(6.0, -6.0, -0.2),
            ));
        }

        if !self_painted {
            // Internal glyph — kind icon (unicode geometric chars; the
            // atlas's CoreText cascade fills them in from system fonts
            // when SF Mono lacks them).
            let glyph_size = glyph_size_for(kind);
            let glyph_scale = glyph_size / 11.0; // matches DEFAULT_FONT_SIZE
            let glyph_cell_w = metrics.cell_w * glyph_scale;
            let glyph_cell_h = metrics.cell_h * glyph_scale;
            let glyph_label = parent.spawn((
                BitmapLabel {
                    text: kind.glyph().to_string(),
                    color: theme.ink,
                    align: TextAlign::Center,
                    capacity: 2,
                    cell_w: glyph_cell_w,
                    cell_h: glyph_cell_h,
                },
                Transform::from_xyz(0.0, 0.0, 0.1),
                Visibility::Inherited,
            )).id();
            parent.commands().entity(glyph_label).with_children(|cp| {
                spawn_label_chars(cp, metrics, 2, theme.ink, glyph_cell_w, glyph_cell_h);
            });

            // Data-colour indicator dot. Drawn only for gadgets that
            // declared a `color` slot — Router (matches *by* colour),
            // LifeCell (self-painted), and any other gadget that opted
            // out of palette tagging skip it.
            if has_color {
                let (dx, dy) = data_dot_offset(&shape);
                parent.spawn((
                    Mesh2d(dot_handle),
                    MeshMaterial2d(ink_mat),
                    Transform::from_xyz(dx, dy, 0.2),
                    NodeColorDot(flow_id),
                ));
            }

            // Name label below the body — set once, almost never changes.
            // (`caps_spaced` from poster_ui inserts U+2009 thin-space
            // between every char, which Text2d rendered as letter-spacing
            // but the bitmap atlas renders as literal extra cells. Plain
            // uppercase reads correctly because the atlas is already
            // monospaced.)
            let name_text = name.to_uppercase();
            let name_capacity = name_text.chars().count().max(8);
            let name_label = parent.spawn((
                BitmapLabel {
                    text: name_text,
                    color: theme.ink,
                    align: TextAlign::Center,
                    capacity: name_capacity,
                    cell_w: metrics.cell_w,
                    cell_h: metrics.cell_h,
                },
                Transform::from_xyz(0.0, -hsize.y * 0.5 - 12.0, 0.1),
                Visibility::Inherited,
                NodeNameText,
            )).id();
            parent.commands().entity(name_label).with_children(|cp| {
                spawn_label_chars(cp, metrics, name_capacity, theme.ink, metrics.cell_w, metrics.cell_h);
            });

            // Live state label ABOVE the body. Empty initially.
            let state_capacity = 16;
            let state_label = parent.spawn((
                BitmapLabel {
                    text: String::new(),
                    color: theme.ink_soft,
                    align: TextAlign::Center,
                    capacity: state_capacity,
                    cell_w: metrics.cell_w,
                    cell_h: metrics.cell_h,
                },
                Transform::from_xyz(0.0, hsize.y * 0.5 + 14.0, 0.1),
                Visibility::Inherited,
                NodeStateLabel(flow_id),
            )).id();
            parent.commands().entity(state_label).with_children(|cp| {
                spawn_label_chars(cp, metrics, state_capacity, theme.ink_soft, metrics.cell_w, metrics.cell_h);
            });
        }
    });

    maps.node_to_entity.insert(flow_id, entity);
    maps.entity_to_node.insert(entity, flow_id);
    entity
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

/// Marks a node body whose fill color tracks a binary slot value (0 or
/// 1) in the sim. Driven by the `paint` field on a class entry in
/// `visual.json` — flow-bevy itself has no notion of which classes
/// receive this treatment.
#[derive(Component)]
pub struct BinarySlotPaint {
    pub node: NodeId,
    pub slot: String,
    pub on: Color,
    pub off: Color,
}

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
    mut cache: ResMut<NodeAssetCache>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut driver: ResMut<SimDriverRes>,
    mut maps: ResMut<EntityMaps>,
    mut node_colors: ResMut<NodeColors>,
    theme: Res<Theme>,
    metrics: Res<AtlasMetrics>,
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
    // Synchronous spawn-and-get-id: we need the NodeId now to map it
    // to the entity we're about to spawn. `with_sim_mut` blocks
    // briefly on the worker; acceptable for an interactive click.
    // Router ignores the slot — see comment in `spawn_gadget`.
    let name_for_sim = name.clone();
    let (id, has_color) = driver.0.with_sim_mut(move |sim| {
        let id = spawn_gadget(sim, kind, &name_for_sim, slot);
        let has_color = crate::gadgets::has_color_slot(sim, id);
        (id, has_color)
    });
    spawn_node_entity(&mut commands, &mut cache, &mut meshes, &mut materials, &mut maps, &theme, &metrics, id, kind, None, false, has_color, name, pos);
    // Snapshot the active data-slot colour onto this node so its emitted
    // packets render in that hue. Theme swaps after drop won't recolour
    // it. Gadgets that don't declare a `color` slot in their DSL stay
    // untagged — Routers (which match *by* colour) and self-painted
    // gadgets like LifeCell.
    if has_color {
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
    nodes: Query<(Entity, &Transform, &BodyShape), With<FlowNodeRef>>,
    ui: Query<&Interaction>,
) {
    if !matches!(active.0, Tool::Select) { return; }
    if !buttons.just_pressed(MouseButton::Left) { return; }
    if poster_ui::pointer_over_ui(&ui) { return; }
    let Some(world) = cursor_to_world(&windows, &cams) else { return; };
    // Pick topmost node under cursor.
    let mut picked: Option<(Entity, Vec2)> = None;
    for (e, tf, shape) in nodes.iter() {
        let size = hit_size(shape);
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

/// Drop the current selection whenever the user changes tool or the
/// scope (drill-in / drill-out) changes. Without this, the previous
/// selection's gizmo outline lingers on an entity that's now hidden
/// (compound drill-in) or in a context where selection has no meaning
/// (any non-Select tool). Runs after `begin_drag` in the chain so that
/// when a double-click both selects the compound and changes scope, the
/// scope-change clear wins.
fn clear_stale_selection(
    active: Res<ActiveTool>,
    scope: Res<crate::compound::CurrentScope>,
    mut selection: ResMut<Selection>,
) {
    if active.is_changed() || scope.is_changed() {
        if selection.entity.is_some() {
            selection.entity = None;
        }
    }
}

#[derive(Component)]
struct SelectionOutline;

fn draw_selection_outline(
    mut commands: Commands,
    selection: Res<Selection>,
    outlines: Query<Entity, With<SelectionOutline>>,
    mut gizmos: Gizmos,
    nodes: Query<(&Transform, &BodyShape), With<FlowNodeRef>>,
    theme: Res<Theme>,
) {
    let _ = (&mut commands, outlines);
    let Some(e) = selection.entity else { return; };
    let Ok((tf, shape)) = nodes.get(e) else { return; };
    let size = hit_size(shape) + Vec2::splat(14.0);
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
    snapshot: Res<SimSnapshotRes>,
    mut labels: Query<(&NodeStateLabel, &ChildOf, &mut BitmapLabel)>,
    parent_kind_q: Query<&NodeKind>,
    mut perf: ResMut<crate::perf::PhaseTimings>,
) {
    crate::time_phase!(perf, "nodes.sync_node_state_labels", {
    for (label, child_of, mut bitmap) in labels.iter_mut() {
        let Ok(kind) = parent_kind_q.get(child_of.parent()) else { continue };
        let Some(node) = snapshot.0.nodes.get(&label.0) else { continue };
        let formatted = format_node_state(kind.0, node);
        // Guard with `!=` so Bevy's change-detection only fires on
        // actual content changes — `sync_bitmap_labels` then skips
        // unchanged labels entirely.
        if bitmap.text != formatted { bitmap.text = formatted; }
    }
    });
}

fn format_node_state(kind: Kind, node: &NodeView) -> String {
    match kind {
        Kind::Generator | Kind::Client | Kind::BackoffClient => {
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
/// `NodeColors` currently holds for that node.
///
/// Implementation note: rather than mutating the dot's `ColorMaterial`
/// asset (which forces every dot to own a unique material), we look up
/// or insert a shared cached material for the desired colour and swap
/// the entity's `MeshMaterial2d` handle to it. With ~900 cells in the
/// Life canvas, this collapses ~900 unique dot materials into one per
/// distinct colour in the data palette.
fn sync_data_dot_colors(
    node_colors: Res<NodeColors>,
    theme: Res<Theme>,
    mut cache: ResMut<NodeAssetCache>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut dots: Query<(&NodeColorDot, &mut MeshMaterial2d<ColorMaterial>)>,
    mut perf: ResMut<crate::perf::PhaseTimings>,
) {
    crate::time_phase!(perf, "nodes.sync_data_dot_colors", {
    for (dot, mut mat_handle) in dots.iter_mut() {
        let want = node_colors.0.get(&dot.0).copied().unwrap_or(theme.accent);
        let want_handle = cache.material(want, &mut materials);
        if mat_handle.0.id() != want_handle.id() {
            mat_handle.0 = want_handle;
        }
    }
    });
}

/// Repaint each `BinarySlotPaint`-tagged body each frame from the sim
/// slot, swapping between two cached material handles (`on` and `off`)
/// instead of mutating per-entity materials. With ~900 BinarySlotPaint
/// cells flipping each tick, this is the difference between 900
/// material-asset writes per frame and 900 component-handle writes,
/// while also collapsing all 900 unique materials into two shared ones.
fn sync_binary_slot_paint(
    snapshot: Res<SimSnapshotRes>,
    mut cache: ResMut<NodeAssetCache>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut paints: Query<(&BinarySlotPaint, &mut MeshMaterial2d<ColorMaterial>)>,
    mut perf: ResMut<crate::perf::PhaseTimings>,
) {
    crate::time_phase!(perf, "nodes.sync_binary_slot_paint", {
    for (paint, mut mat_handle) in paints.iter_mut() {
        let Some(node) = snapshot.0.nodes.get(&paint.node) else { continue };
        let on = match node.slots.get(&paint.slot) {
            Some(flow::Value::Int(i)) => *i != 0,
            Some(flow::Value::Bool(b)) => *b,
            _ => false,
        };
        let want = if on { paint.on } else { paint.off };
        let want_handle = cache.material(want, &mut materials);
        if mat_handle.0.id() != want_handle.id() {
            mat_handle.0 = want_handle;
        }
    }
    });
}

fn sync_node_labels(
    snapshot: Res<SimSnapshotRes>,
    q: Query<(&FlowNodeRef, &Children)>,
    mut labels: Query<&mut BitmapLabel, With<NodeNameText>>,
) {
    for (node_ref, kids) in q.iter() {
        let name = match snapshot.0.nodes.get(&node_ref.0) {
            Some(n) => n.name.to_uppercase(),
            None => continue,
        };
        for kid in kids.iter() {
            if let Ok(mut bitmap) = labels.get_mut(kid) {
                if bitmap.text != name { bitmap.text = name.clone(); }
            }
        }
    }
}

// ---------------- delete ----------------

fn delete_selected(
    mut commands: Commands,
    mut driver: ResMut<SimDriverRes>,
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
    driver.0.send_command(SimCommand::new(move |sim| sim.despawn_node(nid)));
    commands.entity(e).despawn();
    maps.node_to_entity.remove(&nid);
    maps.entity_to_node.remove(&e);
}
