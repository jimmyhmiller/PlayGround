use crate::bridge::{EntityMaps, SimNodeRef, SimResource, bind_existing_edge, bind_existing_node, register_node};
use crate::camera::MainCamera;
use crate::edges::{Probe, ProbeTarget};
use crate::palette::pointer_over_ui;
use crate::sim::{
    NS_PER_MS, NS_PER_S, NS_PER_US, NodeId, parse_duration_ns, parse_rate_pps, period_ns_to_rate,
    rate_to_period_ns,
};
use crate::theme::Theme;
use crate::tool::{ActiveColor, ActiveTool, Tool};
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

pub struct NodesPlugin;

impl Plugin for NodesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DragState>()
            .init_resource::<EditState>()
            .add_systems(
                Update,
                (
                    place_node_on_click,
                    select_and_begin_drag,
                    drag_selected_node,
                    end_drag,
                    sync_node_positions,
                    update_selection_outline,
                    start_edit_on_key,
                    handle_edit_keys,
                    delete_selected,
                    adjust_selected_rate,
                    re_skin_node_meshes,
                    update_canvas_labels,
                    toggle_worker_down,
                    update_down_visual,
                    group_selected_into_composite,
                    draw_composite_boundaries,
                )
                    .chain(),
            );
    }
}

/// Kind of sim node — duplicated from `sim` because Bevy rendering code needs
/// size/color/letter metadata that doesn't belong in a pure sim module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Generator,
    Client,
    Worker,
    Sink,
    Router,
    Queue,
    /// Composite node built via `Sim::group_into_composite` — holds an
    /// inner sim. Renders as a double-bordered block to read as
    /// "container of things" at a glance.
    Custom,
}

impl NodeKind {
    pub fn size(self) -> Vec2 {
        match self {
            NodeKind::Generator => Vec2::new(60.0, 60.0),
            NodeKind::Client => Vec2::new(60.0, 60.0),
            NodeKind::Worker => Vec2::new(70.0, 50.0),
            NodeKind::Sink => Vec2::new(60.0, 60.0),
            NodeKind::Router => Vec2::new(70.0, 50.0),
            NodeKind::Queue => Vec2::new(90.0, 40.0),
            NodeKind::Custom => Vec2::new(100.0, 60.0),
        }
    }

    /// Node body colour. Deliberately neutral (== `theme.paper`, matching
    /// the canvas) so the coloured data-dot on each node is the only hint
    /// of which packet colour it cares about. Kind is communicated by the
    /// ink glyph and label, not by a fill tint.
    pub fn body_color(self, theme: &crate::theme::Theme) -> Color {
        let _ = self;
        theme.paper
    }
}

/// Rendering metadata for a node entity. Behavioral state lives in `Sim`.
#[derive(Component)]
pub struct SimNode {
    pub kind: NodeKind,
    pub color: Color,
}

#[derive(Component)]
pub struct Selected;

/// Bevy-side marker for "user has taken this worker down". We mirror this
/// into the sim's down flag so sim routing respects it.
#[derive(Component)]
pub struct Down;

/// Marker on the per-node border rectangle child so the theme re-skinner can
/// find border materials and repaint them to `theme.ink` on swap.
#[derive(Component)]
pub struct NodeBorderChild;

/// Marker on the kind-letter Text2d child (the "G"/"W"/"C"/...) so the
/// re-skinner can repaint its color when the theme changes.
#[derive(Component)]
pub struct NodeGlyphText;

/// Marker on the single-line canvas label below each node (e.g. "2/s",
/// "500ms", "3"). Text is updated every frame from the sim.
#[derive(Component)]
pub struct NodeCanvasLabel;

/// Cached node positions for anything that wants (x, y) by Entity without
/// a Transform query.
#[derive(Resource, Default)]
pub struct NodeRegistry {
    pub positions: HashMap<Entity, Vec2>,
}

#[derive(Resource, Default)]
pub struct DragState {
    pub entity: Option<Entity>,
    pub offset: Vec2,
}

// ---- Node placement ------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn place_node_on_click(
    mouse: Res<ButtonInput<MouseButton>>,
    active_tool: Res<ActiveTool>,
    active_color: Res<ActiveColor>,
    theme: Res<Theme>,
    ui: Query<&Interaction>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    existing: Query<(&Transform, &SimNode)>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut registry: ResMut<NodeRegistry>,
    mut sim_res: ResMut<SimResource>,
    mut maps: ResMut<EntityMaps>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let kind = match active_tool.0 {
        Tool::Generator => NodeKind::Generator,
        Tool::Client => NodeKind::Client,
        Tool::Worker => NodeKind::Worker,
        Tool::Sink => NodeKind::Sink,
        Tool::Router => NodeKind::Router,
        Tool::Queue => NodeKind::Queue,
        _ => return,
    };
    if pointer_over_ui(&ui) {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    let Ok((cam, cam_tf)) = cams.single() else { return };
    let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) else { return };

    for (tf, sn) in existing.iter() {
        let half = sn.kind.size() / 2.0;
        let c = tf.translation.truncate();
        if (world.x - c.x).abs() <= half.x && (world.y - c.y).abs() <= half.y {
            return;
        }
    }

    spawn_node(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut registry,
        &mut sim_res,
        &mut maps,
        &theme,
        kind,
        active_color.0,
        world,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_node(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    registry: &mut NodeRegistry,
    sim_res: &mut SimResource,
    maps: &mut EntityMaps,
    theme: &Theme,
    kind: NodeKind,
    color: Color,
    pos: Vec2,
) -> Entity {
    let size = kind.size();

    let entity = commands
        .spawn((
            SimNode { kind, color },
            Mesh2d(meshes.add(Rectangle::new(size.x, size.y))),
            MeshMaterial2d(materials.add(kind.body_color(theme))),
            Transform::from_xyz(pos.x, pos.y, 0.0),
            Visibility::default(),
        ))
        .with_children(|p| {
            // Small data-color dot in the top-right of colour-aware nodes.
            // Sinks/workers/queues/generators/clients all have a
            // single-colour identity; router passes everything through and
            // is left neutral so it doesn't visually claim one colour.
            if kind != NodeKind::Router {
                p.spawn((
                    Mesh2d(meshes.add(Circle::new(5.0))),
                    MeshMaterial2d(materials.add(color)),
                    Transform::from_xyz(size.x / 2.0 - 9.0, size.y / 2.0 - 9.0, 0.2),
                ));
            }
            p.spawn((
                Mesh2d(meshes.add(Rectangle::new(size.x + 3.0, size.y + 3.0))),
                MeshMaterial2d(materials.add(theme.ink)),
                Transform::from_xyz(0.0, 0.0, -0.1),
                NodeBorderChild,
            ));
            p.spawn((
                Text2d::new(kind_letter(kind)),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(theme.ink),
                Transform::from_xyz(0.0, 0.0, 0.3),
                NodeGlyphText,
            ));

            // Canvas label: one-line summary under the node. Populated and
            // kept up to date by `update_node_label` below. Empty string at
            // spawn so the first frame of rendering isn't a "0" or similar.
            p.spawn((
                Text2d::new(""),
                TextFont {
                    font_size: 11.0,
                    ..default()
                },
                TextColor(theme.ink_soft),
                Transform::from_xyz(0.0, -size.y / 2.0 - 12.0, 0.3),
                NodeCanvasLabel,
            ));
        })
        .id();

    let nid = register_node(sim_res, maps, entity, kind, color);
    commands.entity(entity).insert(SimNodeRef(nid));

    registry.positions.insert(entity, pos);
    entity
}

fn kind_letter(k: NodeKind) -> &'static str {
    match k {
        NodeKind::Generator => "G",
        NodeKind::Client => "C",
        NodeKind::Worker => "W",
        NodeKind::Sink => "S",
        NodeKind::Router => "R",
        NodeKind::Queue => "Q",
        NodeKind::Custom => "⊞",
    }
}

// ---- Selection + drag (visual only — sim doesn't know about position) ----

fn hit_test_node(
    world: Vec2,
    nodes: &Query<(Entity, &Transform, &SimNode)>,
) -> Option<Entity> {
    nodes
        .iter()
        .find(|(_, tf, sn)| {
            let half = sn.kind.size() / 2.0;
            let c = tf.translation.truncate();
            (world.x - c.x).abs() <= half.x && (world.y - c.y).abs() <= half.y
        })
        .map(|(e, _, _)| e)
}

/// Hit-test composite boundary rectangles. Returns the composite
/// entity if the cursor lands inside its padded bounding box. Use as
/// a fallback after `hit_test_node` so that clicking a member picks
/// the member, but clicking the padding picks the composite.
fn hit_test_composite(
    world: Vec2,
    sim: &SimResource,
    composites: &Query<(Entity, &SimNodeRef), With<CompositeTag>>,
    members: &Query<(&SimNodeRef, &Transform, &SimNode), Without<CompositeTag>>,
) -> Option<Entity> {
    let pad = 16.0;
    for (entity, nref) in composites.iter() {
        let Some(node) = sim.0.nodes.get(&nref.0) else { continue };
        if node.contains.is_empty() {
            continue;
        }
        let contains: HashSet<crate::sim::NodeId> = node.contains.iter().copied().collect();
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for (m_ref, tf, sn) in members.iter() {
            if !contains.contains(&m_ref.0) {
                continue;
            }
            let half = sn.kind.size() / 2.0;
            let c = tf.translation.truncate();
            min = min.min(c - half);
            max = max.max(c + half);
        }
        if min.x.is_infinite() {
            continue;
        }
        min -= Vec2::splat(pad);
        max += Vec2::splat(pad);
        if world.x >= min.x && world.x <= max.x && world.y >= min.y && world.y <= max.y {
            return Some(entity);
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn select_and_begin_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    active_tool: Res<ActiveTool>,
    ui: Query<&Interaction>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    nodes: Query<(Entity, &Transform, &SimNode)>,
    composites: Query<(Entity, &SimNodeRef), With<CompositeTag>>,
    members: Query<(&SimNodeRef, &Transform, &SimNode), Without<CompositeTag>>,
    sim_res: Res<SimResource>,
    selected: Query<Entity, With<Selected>>,
    mut drag: ResMut<DragState>,
    mut commands: Commands,
) {
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

    // Prefer member hit; fall back to composite boundary.
    let hit = hit_test_node(world, &nodes)
        .or_else(|| hit_test_composite(world, &sim_res, &composites, &members));
    let additive = keys.pressed(KeyCode::ShiftLeft)
        || keys.pressed(KeyCode::ShiftRight)
        || keys.pressed(KeyCode::SuperLeft)
        || keys.pressed(KeyCode::SuperRight);

    if !additive {
        for e in selected.iter() {
            commands.entity(e).remove::<Selected>();
        }
    }
    if let Some(hit) = hit {
        if additive && selected.iter().any(|e| e == hit) {
            commands.entity(hit).remove::<Selected>();
        } else {
            commands.entity(hit).insert(Selected);
            if active_tool.0 == Tool::Select && !additive {
                let node_pos = nodes
                    .get(hit)
                    .map(|(_, tf, _)| tf.translation.truncate())
                    .unwrap_or(world);
                drag.entity = Some(hit);
                drag.offset = node_pos - world;
            }
        }
    }
}

fn drag_selected_node(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    drag: Res<DragState>,
    composites: Query<&SimNodeRef, With<CompositeTag>>,
    node_refs: Query<(Entity, &SimNodeRef), With<SimNode>>,
    sim_res: Res<SimResource>,
    mut tfs: Query<&mut Transform, With<SimNode>>,
) {
    if !mouse.pressed(MouseButton::Left) {
        return;
    }
    let Some(entity) = drag.entity else { return };
    let Ok(win) = windows.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    let Ok((cam, cam_tf)) = cams.single() else { return };
    let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) else { return };

    let target = world + drag.offset;

    // If the dragged entity is a composite, translate every member
    // entity by the same delta so the group moves as a unit.
    if let Ok(nref) = composites.get(entity) {
        let Some(node) = sim_res.0.nodes.get(&nref.0) else { return };
        let current = match tfs.get(entity) {
            Ok(t) => t.translation.truncate(),
            Err(_) => return,
        };
        let delta = target - current;
        if delta.length_squared() < 1e-6 {
            return;
        }
        let member_set: HashSet<crate::sim::NodeId> = node.contains.iter().copied().collect();
        let member_entities: Vec<Entity> = node_refs
            .iter()
            .filter_map(|(e, r)| member_set.contains(&r.0).then_some(e))
            .collect();
        if let Ok(mut tf) = tfs.get_mut(entity) {
            tf.translation.x = target.x;
            tf.translation.y = target.y;
        }
        for e in member_entities {
            if let Ok(mut tf) = tfs.get_mut(e) {
                tf.translation.x += delta.x;
                tf.translation.y += delta.y;
            }
        }
        return;
    }

    if let Ok(mut tf) = tfs.get_mut(entity) {
        tf.translation.x = target.x;
        tf.translation.y = target.y;
    }
}

fn end_drag(mouse: Res<ButtonInput<MouseButton>>, mut drag: ResMut<DragState>) {
    if mouse.just_released(MouseButton::Left) {
        drag.entity = None;
    }
}

fn sync_node_positions(
    q: Query<(Entity, &Transform), (With<SimNode>, Changed<Transform>)>,
    mut registry: ResMut<NodeRegistry>,
) {
    for (e, tf) in q.iter() {
        registry.positions.insert(e, tf.translation.truncate());
    }
}

fn update_selection_outline(
    mut gizmos: Gizmos,
    q: Query<(&Transform, &SimNode), With<Selected>>,
) {
    for (tf, sn) in q.iter() {
        let half = sn.kind.size() / 2.0 + Vec2::splat(6.0);
        let c = tf.translation.truncate();
        let color = Color::srgb(0.25, 0.55, 0.90);
        gizmos.rect_2d(c, half * 2.0, color);
    }
}

// ---- Rate / processing-time adjustment (writes into Sim) ----------------

/// Keyboard adjustment of the selected node. Fine steps by default; hold
/// Shift for coarse steps.
///
// ---- Text-edit mode for node rate/processing-time labels ----------------

/// What the user is currently editing via the on-canvas text prompt. The
/// editor captures raw characters into `buffer`, renders them into the
/// target node's normal label, and commits on Enter by parsing the buffer
/// with the sim's `parse_rate_pps` / `parse_duration_ns` helpers.
#[derive(Clone, Debug)]
pub enum EditTarget {
    GeneratorRate(NodeId),
    ClientRate(NodeId),
    WorkerTime(NodeId),
}

#[derive(Resource, Default)]
pub struct EditState {
    pub target: Option<EditTarget>,
    pub buffer: String,
    /// Set to `true` when the latest commit attempt failed to parse, so the
    /// label can render in an error color until the user edits further.
    pub parse_error: bool,
}

impl EditState {
    pub fn is_editing(&self) -> bool {
        self.target.is_some()
    }

    pub fn editing_node(&self) -> Option<NodeId> {
        match self.target {
            Some(EditTarget::GeneratorRate(id))
            | Some(EditTarget::ClientRate(id))
            | Some(EditTarget::WorkerTime(id)) => Some(id),
            None => None,
        }
    }
}

/// `E` on a selected generator/worker starts editing its primary field.
fn start_edit_on_key(
    keys: Res<ButtonInput<KeyCode>>,
    selected: Query<(&SimNodeRef, &SimNode), With<Selected>>,
    sim_res: Res<SimResource>,
    mut edit: ResMut<EditState>,
    mut key_events: MessageReader<KeyboardInput>,
) {
    if edit.is_editing() {
        return;
    }
    if !keys.just_pressed(KeyCode::KeyE) {
        return;
    }
    for (nref, sn) in selected.iter() {
        let Some(node) = sim_res.0.nodes.get(&nref.0) else { continue };
        let target = match sn.kind {
            NodeKind::Generator => {
                edit.buffer = format_rate_for_edit(period_ns_to_rate(node.emit_period_ns()));
                Some(EditTarget::GeneratorRate(nref.0))
            }
            NodeKind::Client => {
                edit.buffer = format_rate_for_edit(period_ns_to_rate(node.emit_period_ns()));
                Some(EditTarget::ClientRate(nref.0))
            }
            NodeKind::Worker => {
                edit.buffer = format_duration_for_edit(node.processing_ns());
                Some(EditTarget::WorkerTime(nref.0))
            }
            _ => None,
        };
        if let Some(t) = target {
            edit.target = Some(t);
            edit.parse_error = false;
            // Drop the KeyboardInput event for the very `E` that triggered us,
            // otherwise `handle_edit_keys` (chained after) sees it and appends
            // "e" to the buffer on the same frame.
            key_events.clear();
            return;
        }
    }
}

/// Character collection + Enter/Escape/Backspace while editing.
fn handle_edit_keys(
    mut key_events: MessageReader<KeyboardInput>,
    mut edit: ResMut<EditState>,
    mut sim_res: ResMut<SimResource>,
) {
    if !edit.is_editing() {
        key_events.clear();
        return;
    }
    for ev in key_events.read() {
        if !ev.state.is_pressed() {
            continue;
        }
        match &ev.logical_key {
            Key::Enter => {
                let ok = commit_edit(&edit, &mut sim_res);
                if ok {
                    edit.target = None;
                    edit.buffer.clear();
                    edit.parse_error = false;
                } else {
                    edit.parse_error = true;
                }
            }
            Key::Escape => {
                edit.target = None;
                edit.buffer.clear();
                edit.parse_error = false;
            }
            Key::Backspace => {
                edit.buffer.pop();
                edit.parse_error = false;
            }
            Key::Character(s) => {
                // Ignore non-printable control chars.
                for c in s.chars() {
                    if !c.is_control() {
                        edit.buffer.push(c);
                    }
                }
                edit.parse_error = false;
            }
            _ => {}
        }
    }
}

fn commit_edit(edit: &EditState, sim_res: &mut ResMut<SimResource>) -> bool {
    match edit.target.as_ref() {
        Some(EditTarget::GeneratorRate(id)) => {
            let Some(rate) = parse_rate_pps(&edit.buffer) else {
                return false;
            };
            let period = rate_to_period_ns(rate);
            sim_res.0.set_generator_period_ns(*id, period);
            true
        }
        Some(EditTarget::ClientRate(id)) => {
            let Some(rate) = parse_rate_pps(&edit.buffer) else {
                return false;
            };
            let period = rate_to_period_ns(rate);
            sim_res.0.set_client_period_ns(*id, period);
            true
        }
        Some(EditTarget::WorkerTime(id)) => {
            let Some(ns) = parse_duration_ns(&edit.buffer) else {
                return false;
            };
            sim_res.0.set_worker_processing_ns(*id, ns.max(1));
            true
        }
        None => true,
    }
}

/// Render a rate as a round-trip-editable string (e.g. "1.2Mpps", "10/s").
fn format_rate_for_edit(rate: f64) -> String {
    if rate <= 0.0 {
        return "0/s".to_string();
    }
    if rate >= 1.0e9 {
        format!("{}G/s", trim_trailing(rate / 1.0e9))
    } else if rate >= 1.0e6 {
        format!("{}M/s", trim_trailing(rate / 1.0e6))
    } else if rate >= 1.0e3 {
        format!("{}k/s", trim_trailing(rate / 1.0e3))
    } else {
        format!("{}/s", trim_trailing(rate))
    }
}

fn format_duration_for_edit(ns: u64) -> String {
    if ns >= NS_PER_S {
        format!("{}s", trim_trailing(ns as f64 / NS_PER_S as f64))
    } else if ns >= NS_PER_MS {
        format!("{}ms", trim_trailing(ns as f64 / NS_PER_MS as f64))
    } else if ns >= NS_PER_US {
        format!("{}us", trim_trailing(ns as f64 / NS_PER_US as f64))
    } else {
        format!("{}ns", ns)
    }
}

fn trim_trailing(n: f64) -> String {
    // Avoid "1.000000" when "1" will do; also cap at 3 decimals.
    let s = format!("{:.3}", n);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    s.to_string()
}

// ---- Delete selected (disabled while editing) --------------------------

/// Delete or Backspace on a selected node removes it from the sim and
/// cascade-despawns every edge touching it plus any probe that targeted the
/// deleted node or its edges. Gated by `EditState` so Backspace during label
/// edit still deletes a character instead of the node.
fn delete_selected(
    keys: Res<ButtonInput<KeyCode>>,
    edit: Res<EditState>,
    selected: Query<(Entity, &SimNodeRef), With<Selected>>,
    edges_q: Query<(Entity, &crate::edges::Edge)>,
    probes_q: Query<(Entity, &crate::edges::Probe)>,
    mut sim_res: ResMut<SimResource>,
    mut maps: ResMut<EntityMaps>,
    mut commands: Commands,
) {
    if edit.is_editing() {
        return;
    }
    if !(keys.just_pressed(KeyCode::Delete) || keys.just_pressed(KeyCode::Backspace)) {
        return;
    }
    // Collect the set of (node entity, sim id) pairs to delete. Cloning into
    // a Vec keeps the query borrow independent of the later iter_mut work.
    let targets: Vec<(Entity, crate::sim::NodeId)> =
        selected.iter().map(|(e, r)| (e, r.0)).collect();
    if targets.is_empty() {
        return;
    }

    // Gather every edge entity that touches one of the deleted nodes — we
    // need to despawn them regardless of whether they appear as a from or to.
    let deleted_node_entities: std::collections::HashSet<Entity> =
        targets.iter().map(|(e, _)| *e).collect();
    let edges_to_despawn: Vec<(Entity, crate::edges::Edge)> = edges_q
        .iter()
        .filter(|(_, edge)| {
            deleted_node_entities.contains(&edge.from)
                || deleted_node_entities.contains(&edge.to)
        })
        .map(|(e, edge)| (e, *edge))
        .collect();

    // Probes that point at any deleted node or edge must go too, else they
    // become zombies that read stale entity ids.
    let deleted_edge_entities: std::collections::HashSet<Entity> =
        edges_to_despawn.iter().map(|(e, _)| *e).collect();
    for (probe_entity, probe) in probes_q.iter() {
        let dangling = match probe.target {
            crate::edges::ProbeTarget::Node(n) => deleted_node_entities.contains(&n),
            crate::edges::ProbeTarget::Edge(e) => deleted_edge_entities.contains(&e),
        };
        if dangling {
            commands.entity(probe_entity).despawn();
        }
    }

    for (entity, nid) in targets {
        // Removing the node returns its incident edge ids — drop those from
        // the entity map too so we don't leak stale lookups.
        for eid in sim_res.0.remove_node(nid) {
            if let Some(edge_entity) = maps.edge_to_entity.remove(&eid) {
                maps.entity_to_edge.remove(&edge_entity);
            }
        }
        maps.node_to_entity.remove(&nid);
        maps.entity_to_node.remove(&entity);
        commands.entity(entity).despawn();
    }
    for (edge_entity, _) in edges_to_despawn {
        commands.entity(edge_entity).despawn();
    }
}

// ---- Keyboard +/- adjust (disabled while editing) -----------------------

/// - Generator rate: fine 0.1/s, coarse 1.0/s. No upper bound.
/// - Worker processing time: fine 0.01s, coarse 0.1s. Lower bound 1ns.
fn adjust_selected_rate(
    keys: Res<ButtonInput<KeyCode>>,
    selected: Query<(&SimNodeRef, &SimNode), With<Selected>>,
    mut sim_res: ResMut<SimResource>,
    edit: Res<EditState>,
) {
    if edit.is_editing() {
        return;
    }
    let up = keys.just_pressed(KeyCode::Equal)
        || keys.just_pressed(KeyCode::NumpadAdd)
        || keys.just_pressed(KeyCode::ArrowUp);
    let down = keys.just_pressed(KeyCode::Minus)
        || keys.just_pressed(KeyCode::NumpadSubtract)
        || keys.just_pressed(KeyCode::ArrowDown);
    if !(up || down) {
        return;
    }
    let dir: f64 = if up { 1.0 } else { -1.0 };
    let coarse =
        keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    for (nref, sn) in selected.iter() {
        let Some(node) = sim_res.0.nodes.get(&nref.0) else { continue };
        match sn.kind {
            NodeKind::Generator => {
                let rate = period_ns_to_rate(node.emit_period_ns());
                let step = if coarse { 1.0 } else { 0.1 };
                let new_rate = (rate + dir * step).max(0.0);
                let new_period = rate_to_period_ns(new_rate);
                sim_res.0.set_generator_period_ns(nref.0, new_period);
            }
            NodeKind::Client => {
                let rate = period_ns_to_rate(node.emit_period_ns());
                let step = if coarse { 1.0 } else { 0.1 };
                let new_rate = (rate + dir * step).max(0.0);
                let new_period = rate_to_period_ns(new_rate);
                sim_res.0.set_client_period_ns(nref.0, new_period);
            }
            NodeKind::Worker => {
                let current = node.processing_ns();
                // + is faster → shorter processing time.
                let step_ns: i64 = if coarse { NS_PER_MS as i64 * 100 } else { NS_PER_MS as i64 * 10 };
                let delta: i64 = (-dir as i64) * step_ns;
                let new = (current as i64 + delta).max(1) as u64;
                sim_res.0.set_worker_processing_ns(nref.0, new);
            }
            _ => {}
        }
    }
}

// ---- Worker down toggle (writes into Sim) -------------------------------

fn toggle_worker_down(
    keys: Res<ButtonInput<KeyCode>>,
    selected: Query<(Entity, &SimNodeRef, &SimNode, Option<&Down>), With<Selected>>,
    mut commands: Commands,
    mut sim_res: ResMut<SimResource>,
) {
    if !keys.just_pressed(KeyCode::KeyD) {
        return;
    }
    for (e, nref, sn, down) in selected.iter() {
        if sn.kind != NodeKind::Worker {
            continue;
        }
        let new_down = down.is_none();
        if let Some(w) = sim_res.0.nodes.get_mut(&nref.0) {
            w.down = new_down;
        }
        if new_down {
            commands.entity(e).insert(Down);
        } else {
            commands.entity(e).remove::<Down>();
        }
    }
}

/// Repaint baked node materials when the Theme resource changes. We don't
/// despawn/respawn the node entities (that would lose user state like drag
/// position and selection) — instead we mutate the existing
/// `Assets<ColorMaterial>` entries that the body and border meshes already
/// hold handles to.
fn re_skin_node_meshes(
    theme: Res<Theme>,
    nodes: Query<(&SimNode, &MeshMaterial2d<ColorMaterial>)>,
    children_q: Query<&Children>,
    border_q: Query<&MeshMaterial2d<ColorMaterial>, With<NodeBorderChild>>,
    mut glyph_q: Query<&mut TextColor, (With<NodeGlyphText>, Without<NodeCanvasLabel>)>,
    mut label_q: Query<&mut TextColor, (With<NodeCanvasLabel>, Without<NodeGlyphText>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    if !theme.is_changed() {
        return;
    }
    for (sn, mat_handle) in nodes.iter() {
        if let Some(mat) = materials.get_mut(&mat_handle.0) {
            mat.color = sn.kind.body_color(&theme);
        }
    }
    // Borders + text live on child entities. Walk all children trees and
    // match by marker so we don't need a parallel parent→child join.
    for entity in children_q.iter() {
        for child in entity.iter() {
            if let Ok(mat_handle) = border_q.get(child) {
                if let Some(mat) = materials.get_mut(&mat_handle.0) {
                    mat.color = theme.ink;
                }
            }
            if let Ok(mut tc) = glyph_q.get_mut(child) {
                tc.0 = theme.ink;
            }
            if let Ok(mut tc) = label_q.get_mut(child) {
                tc.0 = theme.ink_soft;
            }
        }
    }
}

/// One-line stat label below each node. Intentionally minimal: just the
/// single number that most characterises the node at a glance — the
/// inspector carries everything else. No warning badges here; drops/losses
/// live in the inspector so the canvas stays calm.
fn update_canvas_labels(
    sim_res: Res<SimResource>,
    nodes: Query<(Entity, &SimNodeRef, &SimNode, &Children)>,
    probes: Query<&Probe>,
    mut labels: Query<&mut Text2d, With<NodeCanvasLabel>>,
) {
    // A node whose label is being taken over by a probe stays empty — the
    // probe draws its own readout just below. Anything with drops gets a
    // "drop:N" tacked on so the canvas surfaces failure even without a probe.
    let probed: HashSet<Entity> = probes
        .iter()
        .filter_map(|p| match p.target {
            ProbeTarget::Node(e) => Some(e),
            ProbeTarget::Edge(_) => None,
        })
        .collect();

    for (entity, nref, sn, children) in nodes.iter() {
        let text = if probed.contains(&entity) {
            String::new()
        } else {
            let base = match sim_res.0.nodes.get(&nref.0) {
                Some(node) => match sn.kind {
                    NodeKind::Generator | NodeKind::Client => {
                        fmt_rate(period_ns_to_rate(node.emit_period_ns()))
                    }
                    NodeKind::Worker => fmt_duration_short(node.processing_ns()),
                    NodeKind::Queue => format!("{}", node.buffer.len()),
                    NodeKind::Sink => format!("{}", node.sink_total),
                    NodeKind::Router => String::new(),
                    NodeKind::Custom => format!("{} nodes", node.contains.len()),
                },
                None => String::new(),
            };
            let drops = node_drop_count(&sim_res.0, nref.0);
            match (base.is_empty(), drops) {
                (_, 0) => base,
                (true, d) => format!("drop:{}", d),
                (false, d) => format!("{}  drop:{}", base, d),
            }
        };
        for child in children.iter() {
            if let Ok(mut label) = labels.get_mut(child) {
                if label.0 != text {
                    label.0 = text.clone();
                }
            }
        }
    }
}

fn node_drop_count(sim: &crate::sim::Sim, id: NodeId) -> u32 {
    match sim.nodes.get(&id) {
        Some(n) => match n.kind {
            crate::sim::NodeKind::Queue => n.lost,
            _ => n.dropped,
        },
        None => 0,
    }
}

fn fmt_rate(rate: f64) -> String {
    if rate <= 0.0 {
        return "paused".into();
    }
    if rate >= 1.0e3 {
        format!("{:.1}k/s", rate / 1.0e3)
    } else if rate >= 10.0 {
        format!("{:.0}/s", rate)
    } else {
        format!("{:.1}/s", rate)
    }
}

fn fmt_duration_short(ns: u64) -> String {
    if ns >= NS_PER_S {
        let s = ns as f64 / NS_PER_S as f64;
        if s >= 10.0 { format!("{:.0}s", s) } else { format!("{:.1}s", s) }
    } else if ns >= NS_PER_MS {
        let ms = ns as f64 / NS_PER_MS as f64;
        if ms >= 10.0 { format!("{:.0}ms", ms) } else { format!("{:.1}ms", ms) }
    } else if ns >= NS_PER_US {
        format!("{:.0}µs", ns as f64 / NS_PER_US as f64)
    } else {
        format!("{}ns", ns)
    }
}

#[allow(clippy::too_many_arguments)]
fn group_selected_into_composite(
    keys: Res<ButtonInput<KeyCode>>,
    selected_q: Query<(Entity, &Transform, &SimNodeRef, &SimNode), With<Selected>>,
    edge_q: Query<(Entity, &crate::edges::Edge, &crate::bridge::SimEdgeRef)>,
    theme: Res<Theme>,
    edit: Res<EditState>,
    mut sim_res: ResMut<SimResource>,
    mut maps: ResMut<EntityMaps>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut registry: ResMut<NodeRegistry>,
) {
    if edit.is_editing() {
        return;
    }
    let chord = keys.pressed(KeyCode::SuperLeft)
        || keys.pressed(KeyCode::SuperRight)
        || keys.pressed(KeyCode::ControlLeft)
        || keys.pressed(KeyCode::ControlRight);
    if !chord || !keys.just_pressed(KeyCode::KeyG) {
        return;
    }

    let selected: Vec<(Entity, Vec2, crate::sim::NodeId, Color)> = selected_q
        .iter()
        .map(|(e, tf, nref, sn)| (e, tf.translation.truncate(), nref.0, sn.color))
        .collect();
    if selected.len() < 2 {
        return;
    }

    let centroid = selected.iter().map(|(_, p, _, _)| *p).sum::<Vec2>() / selected.len() as f32;
    let color = selected[0].3;
    let selected_ids: HashSet<crate::sim::NodeId> =
        selected.iter().map(|(_, _, id, _)| *id).collect();

    let sim_color = crate::bridge::bevy_to_sim_color(color);
    let Some(result) = sim_res.0.group_into_composite(&selected_ids, "Group", sim_color) else {
        return;
    };

    // Members stay as Bevy entities — the composite is a boundary
    // drawn AROUND them, not a replacement for them. Only external
    // edges that the sim-side rewired need new edge entities; the
    // removed outer edges need their Bevy entities despawned.
    let removed_edges: HashSet<crate::sim::EdgeId> =
        result.removed_outer_edges.iter().copied().collect();
    for (edge_entity, _, eref) in edge_q.iter() {
        if removed_edges.contains(&eref.0) {
            maps.entity_to_edge.remove(&edge_entity);
            maps.edge_to_entity.remove(&eref.0);
            commands.entity(edge_entity).despawn();
        }
    }
    // Also deselect the absorbed members so the next thing the user
    // does doesn't operate on 5 selected nodes.
    for (entity, _, nref, _) in selected_q.iter() {
        if result.absorbed_nodes.contains(&nref.0) {
            commands.entity(entity).remove::<Selected>();
        }
    }

    let composite_entity = spawn_composite_entity(
        &mut commands,
        &mut meshes,
        &mut materials,
        &theme,
        centroid,
        color,
    );
    commands
        .entity(composite_entity)
        .insert(SimNodeRef(result.composite));
    bind_existing_node(&mut maps, composite_entity, result.composite);
    registry.positions.insert(composite_entity, centroid);

    for eid in &result.new_outer_edges {
        let Some(edge) = sim_res.0.edges.get(eid).cloned() else { continue };
        let from_entity = if edge.from == result.composite {
            composite_entity
        } else {
            match maps.node_to_entity.get(&edge.from) {
                Some(e) => *e,
                None => continue,
            }
        };
        let to_entity = if edge.to == result.composite {
            composite_entity
        } else {
            match maps.node_to_entity.get(&edge.to) {
                Some(e) => *e,
                None => continue,
            }
        };
        let edge_entity = commands
            .spawn((
                crate::edges::Edge {
                    from: from_entity,
                    to: to_entity,
                },
                crate::bridge::SimEdgeRef(*eid),
            ))
            .id();
        bind_existing_edge(&mut maps, edge_entity, *eid);
    }
}

/// Marker on a composite's lightweight entity. The composite doesn't
/// own a body mesh — it's drawn as a boundary box by
/// `draw_composite_boundaries` each frame, computed from the bounding
/// box of its member nodes' current positions.
#[derive(Component)]
pub struct CompositeTag;

fn spawn_composite_entity(
    commands: &mut Commands,
    _meshes: &mut Assets<Mesh>,
    _materials: &mut Assets<ColorMaterial>,
    _theme: &Theme,
    pos: Vec2,
    color: Color,
) -> Entity {
    // The composite is not rendered as a block — it's just a sim
    // routing node with a label. The boundary around its members is
    // drawn by `draw_composite_boundaries` from live transforms.
    commands
        .spawn((
            SimNode {
                kind: NodeKind::Custom,
                color,
            },
            CompositeTag,
            Transform::from_xyz(pos.x, pos.y, 0.0),
            Visibility::default(),
        ))
        .id()
}

/// Draw a cream-filled rounded rect around every composite's member
/// nodes, with the composite's label at the top-left corner. Runs
/// every frame so dragging a member updates the box instantly.
fn draw_composite_boundaries(
    mut gizmos: Gizmos,
    theme: Res<Theme>,
    composites: Query<(&SimNodeRef, &SimNode), With<CompositeTag>>,
    member_tfs: Query<(&SimNodeRef, &Transform, &SimNode), Without<CompositeTag>>,
    sim_res: Res<SimResource>,
) {
    for (nref, _sn) in composites.iter() {
        let Some(node) = sim_res.0.nodes.get(&nref.0) else { continue };
        let contains: HashSet<crate::sim::NodeId> = node.contains.iter().copied().collect();
        if contains.is_empty() {
            continue;
        }
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for (m_ref, tf, m_sn) in member_tfs.iter() {
            if !contains.contains(&m_ref.0) {
                continue;
            }
            let half = m_sn.kind.size() / 2.0;
            let c = tf.translation.truncate();
            min = min.min(c - half);
            max = max.max(c + half);
        }
        if min.x.is_infinite() {
            continue;
        }
        let pad = 16.0;
        min -= Vec2::splat(pad);
        max += Vec2::splat(pad);
        let corners = [
            Vec2::new(min.x, min.y),
            Vec2::new(max.x, min.y),
            Vec2::new(max.x, max.y),
            Vec2::new(min.x, max.y),
        ];
        for i in 0..4 {
            gizmos.line_2d(corners[i], corners[(i + 1) % 4], theme.ink);
        }
    }
}

fn update_down_visual(
    mut gizmos: Gizmos,
    q: Query<(&Transform, &SimNode), With<Down>>,
) {
    for (tf, sn) in q.iter() {
        let c = tf.translation.truncate();
        let half = sn.kind.size() / 2.0;
        let red = Color::srgb(0.85, 0.2, 0.2);
        gizmos.line_2d(c - half, c + half, red);
        gizmos.line_2d(
            Vec2::new(c.x - half.x, c.y + half.y),
            Vec2::new(c.x + half.x, c.y - half.y),
            red,
        );
    }
}
