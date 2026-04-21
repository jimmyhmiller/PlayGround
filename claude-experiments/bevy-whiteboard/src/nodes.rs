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
            .init_resource::<StepsRowActivity>()
            .init_resource::<RowDragState>()
            .init_resource::<SelectedStep>()
            .add_systems(
                Update,
                (
                    start_row_drag,
                    end_row_drag,
                    sync_step_row_transforms,
                    draw_selected_step_outline,
                    delete_selected_step,
                    clear_selected_step_on_canvas_click,
                )
                    .chain(),
            )
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
            )
            .add_systems(
                Update,
                (
                    rebuild_node_rendering,
                    record_steps_row_activity,
                    update_steps_row_highlight,
                    draw_steps_loop_arc,
                    spawn_steps_loop_dots,
                    animate_steps_loop_dots,
                    toggle_opened_on_double_click,
                )
                    .chain(),
            );
    }
}

/// Reconcile every node's visual with its current (opened, row_count)
/// state. Steps containers are effectively always "opened"; every
/// other kind renders as the small kind-glyph box unless it has the
/// [`Opened`] marker. On state change, despawn children, resize the
/// body mesh, and respawn children in the appropriate style.
#[allow(clippy::too_many_arguments)]
fn rebuild_node_rendering(
    sim_res: Res<crate::bridge::SimResource>,
    theme: Res<Theme>,
    mut nodes: Query<(
        Entity,
        &mut SimNode,
        &crate::bridge::SimNodeRef,
        Option<&Children>,
        Option<&Opened>,
        Option<&RenderedAs>,
    )>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for (entity, mut sn, nref, children, opened, rendered) in nodes.iter_mut() {
        let Some(node) = sim_res.0.nodes.get(&nref.0) else { continue };
        // Every Steps container is always opened; any other kind
        // opens on explicit double-click via the [`Opened`] marker.
        let is_open = sn.kind == NodeKind::Steps || opened.is_some();
        let row_count = if is_open { node.program.len() } else { 0 };
        let want = (is_open, row_count);
        let have = rendered.map(|r| (r.opened, r.row_count));
        if have == Some(want) {
            continue;
        }
        // Tear down all children and rebuild in the new style.
        if let Some(children) = children {
            for child in children.iter() {
                commands.entity(child).despawn();
            }
        }
        let new_size = if is_open {
            steps_container_size(row_count.max(1))
        } else {
            sn.kind.size()
        };
        sn.size = new_size;
        commands
            .entity(entity)
            .insert(Mesh2d(meshes.add(Rectangle::new(new_size.x, new_size.y))))
            .insert(MeshMaterial2d(materials.add(sn.kind.body_color(&theme))))
            .insert(RenderedAs {
                opened: is_open,
                row_count,
            });
        if is_open {
            let rows_snapshot: Vec<(usize, String)> = node
                .program
                .iter()
                .enumerate()
                .map(|(i, r)| (i, format_step_row(r)))
                .collect();
            let color = sn.color;
            let header = if sn.kind == NodeKind::Steps {
                "STEPS".to_string()
            } else {
                format!("{} · PROGRAM", kind_name(sn.kind).to_uppercase())
            };
            spawn_opened_children(
                &mut commands,
                entity,
                &mut meshes,
                &mut materials,
                &theme,
                color,
                new_size,
                &rows_snapshot,
                nref.0,
                &header,
            );
        } else {
            spawn_closed_children(
                &mut commands,
                entity,
                &mut meshes,
                &mut materials,
                &theme,
                sn.kind,
                sn.color,
                new_size,
            );
        }
    }
}

/// On mouse-down over a row of an opened node, pick up that row
/// for potential reordering. Run BEFORE select-and-drag so a row
/// pickup suppresses the usual node-drag gesture.
#[allow(clippy::too_many_arguments)]
fn start_row_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    active_tool: Res<ActiveTool>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    ui: Query<&Interaction>,
    nodes_q: Query<(Entity, &Transform, &SimNode, &crate::bridge::SimNodeRef, Option<&Opened>)>,
    sim_res: Res<crate::bridge::SimResource>,
    mut drag: ResMut<RowDragState>,
    mut selected: ResMut<SelectedStep>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    if active_tool.0 != Tool::Select {
        return;
    }
    if pointer_over_ui(&ui) {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    let Ok((cam, cam_tf)) = cams.single() else { return };
    let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) else { return };
    for (entity, tf, sn, nref, opened) in nodes_q.iter() {
        let center = tf.translation.truncate();
        let half = sn.size / 2.0;
        if (world.x - center.x).abs() > half.x || (world.y - center.y).abs() > half.y {
            continue;
        }
        let is_open = sn.kind == NodeKind::Steps || opened.is_some();
        if !is_open {
            return;
        }
        let row_count = sim_res
            .0
            .nodes
            .get(&nref.0)
            .map(|n| n.program.len())
            .unwrap_or(0);
        if let Some(row) = steps_row_at(world, center, row_count) {
            drag.source_entity = Some(entity);
            drag.source_node = nref.0;
            drag.source_row = row;
            // Selecting a row and picking it up are the same
            // gesture — set the selection immediately so the
            // outline appears on press, not only on release.
            selected.entity = Some(entity);
            selected.node = nref.0;
            selected.row = row;
        } else {
            // Clicked the node header/padding but not a row —
            // clear row selection, the node-drag will take over.
            selected.entity = None;
        }
        return;
    }
}

/// Per-frame reconciliation of step-row local transforms. Writes
/// each `StepsRow` child's Y to `steps_row_center_y(row_count, i)`,
/// then — if a row is being dragged — overrides the dragged row's
/// Y to track the cursor. This is what makes reorder *feel*
/// reordery: the row you grabbed follows your finger; everything
/// else snaps to its natural slot on drop.
#[allow(clippy::too_many_arguments)]
fn sync_step_row_transforms(
    sim_res: Res<crate::bridge::SimResource>,
    nodes_q: Query<(Entity, &Transform, &crate::bridge::SimNodeRef, &Children)>,
    mut rows: Query<(&StepsRow, &mut Transform), Without<crate::bridge::SimNodeRef>>,
    drag: Res<RowDragState>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
) {
    let cursor_world = windows
        .single()
        .ok()
        .and_then(|w| w.cursor_position())
        .and_then(|c| {
            cams.single()
                .ok()
                .and_then(|(cam, cam_tf)| cam.viewport_to_world_2d(cam_tf, c).ok())
        });
    for (entity, node_tf, nref, children) in nodes_q.iter() {
        let row_count = sim_res
            .0
            .nodes
            .get(&nref.0)
            .map(|n| n.program.len().max(1))
            .unwrap_or(1);
        let node_center = node_tf.translation.truncate();
        let dragging_this =
            drag.source_entity == Some(entity) && cursor_world.is_some();
        // Hover target: which row the cursor is currently over. If
        // the user hasn't moved off the source row, hover = None
        // and nothing previews.
        let hover_target = if dragging_this {
            let cursor = cursor_world.unwrap();
            steps_row_at(cursor, node_center, row_count)
                .filter(|t| *t != drag.source_row)
        } else {
            None
        };
        for child in children.iter() {
            let Ok((row, mut tf)) = rows.get_mut(child) else { continue };
            if dragging_this && row.index == drag.source_row {
                // Dragged row follows the cursor Y within the
                // container. Clamped + z-lifted so it draws above
                // the siblings.
                let cursor_y = cursor_world.unwrap().y;
                let local_y = cursor_y - node_center.y;
                let max = steps_row_center_y(row_count, 0).abs();
                tf.translation.y = local_y.clamp(-max, max);
                tf.translation.z = 0.5;
            } else if let Some(target) = hover_target {
                // Preview the swap: the row sitting at `target`
                // would move to `source_row`'s slot after drop.
                // Everyone else keeps their index.
                let displayed_index = if row.index == target {
                    drag.source_row
                } else {
                    row.index
                };
                tf.translation.y = steps_row_center_y(row_count, displayed_index);
                tf.translation.z = 0.15;
            } else {
                tf.translation.y = steps_row_center_y(row_count, row.index);
                tf.translation.z = 0.15;
            }
        }
    }
}

/// Each frame, if a step is selected, draw a gizmo outline around
/// its row rectangle. Tracks the node's current transform + size so
/// the outline follows when the user drags the container around.
fn draw_selected_step_outline(
    mut gizmos: Gizmos,
    theme: Res<Theme>,
    selected: Res<SelectedStep>,
    sim_res: Res<crate::bridge::SimResource>,
    nodes_q: Query<(Entity, &Transform, &SimNode, &crate::bridge::SimNodeRef)>,
) {
    let Some(entity) = selected.entity else { return };
    let Ok((_e, tf, sn, nref)) = nodes_q.get(entity) else { return };
    if sn.kind != NodeKind::Steps {
        // Non-Steps nodes can only be selected if opened; still OK.
    }
    let row_count = sim_res
        .0
        .nodes
        .get(&nref.0)
        .map(|n| n.program.len().max(1))
        .unwrap_or(1);
    if selected.row >= row_count {
        return;
    }
    let center = tf.translation.truncate();
    let cy = center.y + steps_row_center_y(row_count, selected.row);
    let w = sn.size.x - 10.0;
    let h = STEPS_ROW_HEIGHT + 2.0;
    let half = Vec2::new(w / 2.0, h / 2.0);
    let corners = [
        Vec2::new(center.x - half.x, cy - half.y),
        Vec2::new(center.x + half.x, cy - half.y),
        Vec2::new(center.x + half.x, cy + half.y),
        Vec2::new(center.x - half.x, cy + half.y),
    ];
    let color = theme.accent;
    for i in 0..4 {
        gizmos.line_2d(corners[i], corners[(i + 1) % 4], color);
    }
}

/// Delete/Backspace removes the selected step row if there is one.
/// Otherwise the key falls through to `delete_selected` (node delete).
fn delete_selected_step(
    keys: Res<ButtonInput<KeyCode>>,
    edit: Res<EditState>,
    mut selected: ResMut<SelectedStep>,
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
    let Some(_entity) = selected.entity else { return };
    let removed = sim_res.0.remove_step_row(selected.node, selected.row);
    for eid in removed {
        if let Some(edge_entity) = maps.edge_to_entity.remove(&eid) {
            commands.entity(edge_entity).despawn();
            maps.entity_to_edge.remove(&edge_entity);
        }
    }
    selected.entity = None;
}

/// If the user clicks blank canvas (no node, no row), clear the
/// step selection. Mirrors how clicking blank canvas clears node
/// selection in `select_and_begin_drag`.
fn clear_selected_step_on_canvas_click(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    nodes_q: Query<(&Transform, &SimNode)>,
    ui: Query<&Interaction>,
    mut selected: ResMut<SelectedStep>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    if pointer_over_ui(&ui) {
        return;
    }
    if selected.entity.is_none() {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    let Ok((cam, cam_tf)) = cams.single() else { return };
    let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) else { return };
    let any_hit = nodes_q.iter().any(|(tf, sn)| {
        let half = sn.size / 2.0;
        let c = tf.translation.truncate();
        (world.x - c.x).abs() <= half.x && (world.y - c.y).abs() <= half.y
    });
    if !any_hit {
        selected.entity = None;
    }
}

/// On mouse release, if a row-drag is active, compute the target
/// row under the cursor and swap if different. Clears drag state
/// either way.
#[allow(clippy::too_many_arguments)]
fn end_row_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    nodes_q: Query<(Entity, &Transform, &SimNode, &crate::bridge::SimNodeRef)>,
    mut sim_res: ResMut<crate::bridge::SimResource>,
    mut drag: ResMut<RowDragState>,
    mut selected: ResMut<SelectedStep>,
) {
    if !mouse.just_released(MouseButton::Left) {
        return;
    }
    let Some(source_entity) = drag.source_entity.take() else { return };
    let source_row = drag.source_row;
    let source_node = drag.source_node;
    let Ok(win) = windows.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    let Ok((cam, cam_tf)) = cams.single() else { return };
    let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) else { return };
    for (entity, tf, sn, nref) in nodes_q.iter() {
        if entity != source_entity {
            continue;
        }
        let center = tf.translation.truncate();
        let half = sn.size / 2.0;
        if (world.x - center.x).abs() > half.x || (world.y - center.y).abs() > half.y {
            return;
        }
        let row_count = sim_res
            .0
            .nodes
            .get(&nref.0)
            .map(|n| n.program.len())
            .unwrap_or(0);
        if let Some(target_row) = steps_row_at(world, center, row_count) {
            if target_row != source_row {
                sim_res.0.swap_step_rows(source_node, source_row, target_row);
                // The row the user picked up moved to `target_row`.
                // Keep the selection on the same logical step.
                if selected.entity == Some(entity) && selected.row == source_row {
                    selected.row = target_row;
                }
            }
        }
        return;
    }
}

/// Detect a left-click double-click on a node body and toggle its
/// `Opened` marker. Selection happens via the existing
/// `select_and_begin_drag`; this just layers the open-toggle onto
/// the second click. Steps containers are always opened and ignore
/// this gesture.
fn toggle_opened_on_double_click(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cams: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    nodes_q: Query<(Entity, &Transform, &SimNode, Option<&Opened>)>,
    ui: Query<&Interaction>,
    mut last_click: Local<Option<(f64, Vec2)>>,
    time: Res<Time>,
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

    let now = time.elapsed_secs_f64();
    let is_double = matches!(*last_click, Some((t, p)) if now - t < 0.4 && (p - world).length() < 8.0);
    *last_click = Some((now, world));
    if !is_double {
        return;
    }

    for (entity, tf, sn, opened) in nodes_q.iter() {
        if sn.kind == NodeKind::Steps {
            continue;
        }
        let center = tf.translation.truncate();
        let half = sn.size / 2.0;
        if (world.x - center.x).abs() > half.x || (world.y - center.y).abs() > half.y {
            continue;
        }
        if opened.is_some() {
            commands.entity(entity).remove::<Opened>();
        } else {
            commands.entity(entity).insert(Opened);
        }
        break;
    }
}

/// Friendly name for a kind used in the opened-mode header.
fn kind_name(k: NodeKind) -> &'static str {
    match k {
        NodeKind::Generator => "Generator",
        NodeKind::Client => "Client",
        NodeKind::Worker => "Worker",
        NodeKind::Sink => "Sink",
        NodeKind::Router => "Router",
        NodeKind::Queue => "Queue",
        NodeKind::Custom => "Group",
        NodeKind::Steps => "Steps",
    }
}

/// Spawn the visuals for a closed (non-opened, non-Steps) node:
/// the data-color dot, ink border, kind-letter glyph, and canvas
/// label below. Mirrors the inline spawn in `spawn_node` so both
/// paths produce the same visuals.
#[allow(clippy::too_many_arguments)]
fn spawn_closed_children(
    commands: &mut Commands,
    parent: Entity,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    theme: &Theme,
    kind: NodeKind,
    color: Color,
    size: Vec2,
) {
    commands.entity(parent).with_children(|p| {
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
            TextFont { font_size: 18.0, ..default() },
            TextColor(theme.ink),
            Transform::from_xyz(0.0, 0.0, 0.3),
            NodeGlyphText,
        ));
        p.spawn((
            Text2d::new(""),
            TextFont { font_size: 11.0, ..default() },
            TextColor(theme.ink_soft),
            Transform::from_xyz(0.0, -size.y / 2.0 - 12.0, 0.3),
            NodeCanvasLabel,
            crate::bridge::Mono,
        ));
    });
}

/// Spawn the child entities that make an *opened* node render as
/// a vertical stack of step rows. Used by Steps containers (always
/// opened) and by any other kind when the user has double-clicked
/// it to crack it open. `header` is the banner text shown at the
/// top of the container.
#[allow(clippy::too_many_arguments)]
fn spawn_opened_children(
    commands: &mut Commands,
    parent: Entity,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
    theme: &Theme,
    color: Color,
    size: Vec2,
    rows: &[(usize, String)],
    node_id: crate::sim::NodeId,
    header: &str,
) {
    commands.entity(parent).with_children(|p| {
        p.spawn((
            Mesh2d(meshes.add(Circle::new(5.0))),
            MeshMaterial2d(materials.add(color)),
            Transform::from_xyz(size.x / 2.0 - 9.0, size.y / 2.0 - 9.0, 0.2),
        ));
        p.spawn((
            Mesh2d(meshes.add(Rectangle::new(size.x + 3.0, size.y + 3.0))),
            MeshMaterial2d(materials.add(theme.ink)),
            Transform::from_xyz(0.0, 0.0, -0.1),
            NodeBorderChild,
        ));
        let header_y = size.y / 2.0 - STEPS_HEADER_H / 2.0 - 2.0;
        p.spawn((
            Text2d::new(header.to_string()),
            TextFont {
                font_size: 10.0,
                ..default()
            },
            TextColor(theme.ink_soft),
            Transform::from_xyz(0.0, header_y, 0.3),
        ));
        let row_count = rows.len().max(1);
        let row_w = size.x - 14.0;
        let row_h = STEPS_ROW_HEIGHT - 4.0;
        for (i, _label) in rows {
            let cy = steps_row_center_y(row_count, *i);
            let row_index = *i;
            p.spawn((
                StepsRow { index: row_index },
                Transform::from_xyz(0.0, cy, 0.15),
                Visibility::default(),
            ))
            .with_children(|r| {
                r.spawn((
                    Mesh2d(meshes.add(Rectangle::new(row_w, row_h))),
                    MeshMaterial2d(materials.add(theme.paper_alt)),
                    Transform::from_xyz(0.0, 0.0, 0.0),
                    StepsRowBody,
                ));
                r.spawn((
                    Mesh2d(meshes.add(Rectangle::new(row_w + 1.5, row_h + 1.5))),
                    MeshMaterial2d(materials.add(theme.rule)),
                    Transform::from_xyz(0.0, 0.0, -0.05),
                ));
                // Empty initial text + LiveText binding — the
                // ui::sync_live_text_canvas system fills it every
                // frame, so a slider change to a Worker row's
                // duration reflects here without a structural
                // rebuild.
                r.spawn((
                    Text2d::new(String::new()),
                    TextFont {
                        font_size: 11.0,
                        ..default()
                    },
                    TextColor(theme.ink),
                    Transform::from_xyz(0.0, 0.0, 0.1),
                    crate::ui::LiveText {
                        node: node_id,
                        field: crate::ui::LiveField::StepRowLabel(row_index),
                    },
                ));
            });
        }
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
    });
}

/// Control points for the loop-back arc on the left side of a Steps
/// container. Endpoints sit just outside the left border at the
/// vertical centres of the first and last rows. Control point pulls
/// the curve further left so the arc reads as an obvious return
/// path. Returns `None` for containers with <1 row (nothing to loop).
fn steps_arc_points(center: Vec2, size: Vec2, row_count: usize) -> Option<(Vec2, Vec2, Vec2)> {
    if row_count == 0 {
        return None;
    }
    let first_y = center.y + steps_row_center_y(row_count, 0);
    let last_y = center.y + steps_row_center_y(row_count, row_count.saturating_sub(1));
    let left_x = center.x - size.x / 2.0 - 4.0;
    let start = Vec2::new(left_x, last_y);
    let end = Vec2::new(left_x, first_y);
    let bulge = (first_y - last_y).abs().max(24.0) * 0.6;
    let ctrl = Vec2::new(left_x - bulge, (first_y + last_y) / 2.0);
    Some((start, ctrl, end))
}

fn quad_bezier(a: Vec2, b: Vec2, c: Vec2, t: f32) -> Vec2 {
    let u = 1.0 - t;
    a * (u * u) + b * (2.0 * u * t) + c * (t * t)
}

/// Draw a curved return line on the left of every Steps container,
/// connecting the last row back to the first. Purely cosmetic — the
/// token loop actually happens instantaneously in the sim.
fn draw_steps_loop_arc(
    mut gizmos: Gizmos,
    theme: Res<Theme>,
    sim_res: Res<crate::bridge::SimResource>,
    nodes: Query<(&Transform, &SimNode, &crate::bridge::SimNodeRef)>,
) {
    for (tf, sn, nref) in nodes.iter() {
        if sn.kind != NodeKind::Steps {
            continue;
        }
        let Some(node) = sim_res.0.nodes.get(&nref.0) else { continue };
        let row_count = node.program.len();
        // Need at least 2 rows to have an actual loop-back (a single
        // row visibly jumps from itself to itself — skip to avoid
        // drawing a zero-length arc).
        if row_count < 2 {
            continue;
        }
        let center = tf.translation.truncate();
        let Some((a, b, c)) = steps_arc_points(center, sn.size, row_count) else {
            continue;
        };
        const SEGMENTS: usize = 20;
        let mut prev = a;
        for i in 1..=SEGMENTS {
            let t = i as f32 / SEGMENTS as f32;
            let p = quad_bezier(a, b, c, t);
            gizmos.line_2d(prev, p, theme.rule);
            prev = p;
        }
    }
}

/// A short-lived dot that animates along a Steps container's
/// loop-back arc, spawned each time the sim emits
/// `SimEvent::StepsLooped`. Purely visual; despawned on completion.
#[derive(Component)]
pub struct StepsLoopDot {
    pub node: Entity,
    pub t: f32,
    pub duration: f32,
}

/// React to `SimEvent::StepsLooped` events from the latest tick by
/// spawning a traveling dot entity for each. Only fires when the
/// container has ≥2 rows — matches the arc-drawing threshold.
fn spawn_steps_loop_dots(
    events: Res<crate::bridge::TickEvents>,
    maps: Res<crate::bridge::EntityMaps>,
    sim_res: Res<crate::bridge::SimResource>,
    nodes: Query<(&Transform, &SimNode)>,
    theme: Res<Theme>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for ev in &events.0 {
        let crate::sim::SimEvent::StepsLooped { node: nid } = ev else {
            continue;
        };
        let Some(&entity) = maps.node_to_entity.get(nid) else { continue };
        let Ok((_tf, _sn)) = nodes.get(entity) else { continue };
        let row_count = sim_res
            .0
            .nodes
            .get(nid)
            .map(|n| n.program.len())
            .unwrap_or(0);
        if row_count < 2 {
            continue;
        }
        commands.spawn((
            StepsLoopDot {
                node: entity,
                t: 0.0,
                duration: 0.35,
            },
            Mesh2d(meshes.add(Circle::new(4.0))),
            MeshMaterial2d(materials.add(theme.accent)),
            Transform::from_xyz(0.0, 0.0, 0.6),
            Visibility::default(),
        ));
    }
}

fn animate_steps_loop_dots(
    time: Res<Time>,
    sim_res: Res<crate::bridge::SimResource>,
    nodes: Query<(&Transform, &SimNode, &crate::bridge::SimNodeRef), Without<StepsLoopDot>>,
    mut dots: Query<(Entity, &mut StepsLoopDot, &mut Transform)>,
    mut commands: Commands,
) {
    let dt = time.delta_secs();
    for (entity, mut dot, mut tf) in dots.iter_mut() {
        dot.t += dt / dot.duration.max(0.01);
        if dot.t >= 1.0 {
            commands.entity(entity).despawn();
            continue;
        }
        let Ok((node_tf, node_sn, nref)) = nodes.get(dot.node) else {
            commands.entity(entity).despawn();
            continue;
        };
        let row_count = sim_res
            .0
            .nodes
            .get(&nref.0)
            .map(|n| n.program.len())
            .unwrap_or(0);
        let Some((a, b, c)) = steps_arc_points(
            node_tf.translation.truncate(),
            node_sn.size,
            row_count,
        ) else {
            continue;
        };
        let p = quad_bezier(a, b, c, dot.t.clamp(0.0, 1.0));
        tf.translation.x = p.x;
        tf.translation.y = p.y;
    }
}

/// Real-time duration of the post-entry flash on a step row.
/// 250ms is long enough that a row which was only current for 0 sim
/// time (instant Client roundtrip) is still clearly visible.
const STEPS_ROW_FLASH_SECS: f32 = 0.25;

/// Per-step-row "last entered at" timestamps (real-time seconds from
/// `Time::elapsed_secs`). Updated from `SimEvent::StepsRowEntered`
/// and from the currently-active row each frame so even non-event
/// rows (e.g. initial row 0 at spawn) get a lit indicator.
#[derive(Resource, Default)]
pub struct StepsRowActivity {
    pub last_entered: HashMap<(crate::sim::NodeId, usize), f32>,
}

/// Update the activity tracker from sim events (caught by the
/// bridge's `TickEvents`) and from the current row (so the active
/// row's entry holds indefinitely while it's executing — the
/// "flash" only decays *after* the row advances past).
fn record_steps_row_activity(
    time: Res<Time>,
    events: Res<crate::bridge::TickEvents>,
    sim_res: Res<crate::bridge::SimResource>,
    nodes: Query<(&crate::bridge::SimNodeRef, &SimNode)>,
    mut activity: ResMut<StepsRowActivity>,
) {
    let now = time.elapsed_secs();
    for ev in &events.0 {
        if let crate::sim::SimEvent::StepsRowEntered { node, row } = ev {
            activity.last_entered.insert((*node, *row), now);
        }
    }
    // Also refresh the currently-executing row each frame so its
    // highlight doesn't fade while it's still doing work (awaiting
    // response, holding a Worker dwell, etc.).
    for (nref, sn) in nodes.iter() {
        if sn.kind != NodeKind::Steps {
            continue;
        }
        if let Some(cur) = sim_res
            .0
            .nodes
            .get(&nref.0)
            .and_then(|n| n.cursor.as_ref())
            .and_then(|p| p.first().copied())
        {
            activity.last_entered.insert((nref.0, cur), now);
        }
    }
}

/// Paint each row body with a mix between accent (just-active) and
/// paper_alt (dormant) based on how recently it was active.
fn update_steps_row_highlight(
    time: Res<Time>,
    theme: Res<Theme>,
    activity: Res<StepsRowActivity>,
    nodes: Query<(&crate::bridge::SimNodeRef, &SimNode, &Children)>,
    row_q: Query<(&StepsRow, &Children)>,
    body_q: Query<&MeshMaterial2d<ColorMaterial>, With<StepsRowBody>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let now = time.elapsed_secs();
    for (nref, sn, children) in nodes.iter() {
        if sn.kind != NodeKind::Steps {
            continue;
        }
        for child in children.iter() {
            let Ok((row, row_children)) = row_q.get(child) else { continue };
            let since = activity
                .last_entered
                .get(&(nref.0, row.index))
                .map(|t| now - t)
                .unwrap_or(f32::INFINITY);
            let intensity =
                (1.0 - (since / STEPS_ROW_FLASH_SECS)).clamp(0.0, 1.0);
            let target = lerp_color(theme.paper_alt, theme.accent, intensity);
            for inner in row_children.iter() {
                if let Ok(mat) = body_q.get(inner) {
                    if let Some(m) = materials.get_mut(&mat.0) {
                        m.color = target;
                    }
                }
            }
        }
    }
}

fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    let a = a.to_srgba();
    let b = b.to_srgba();
    Color::srgba(
        a.red + (b.red - a.red) * t,
        a.green + (b.green - a.green) * t,
        a.blue + (b.blue - a.blue) * t,
        a.alpha + (b.alpha - a.alpha) * t,
    )
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
    /// Scripted process — rendered as a stack of step rows. See the
    /// sim-side [`crate::sim::NodeKind::Steps`] and
    /// [`crate::sim::StepRow`].
    Steps,
}

impl NodeKind {
    /// Default visual size for a fixed-size kind. Steps nodes ignore
    /// this — their size depends on the current row count (see
    /// [`steps_container_size`]) and is stored on the per-entity
    /// `SimNode::size` so hit-tests read the live dimension without
    /// cross-referencing the sim.
    pub fn size(self) -> Vec2 {
        match self {
            NodeKind::Generator => Vec2::new(60.0, 60.0),
            NodeKind::Client => Vec2::new(60.0, 60.0),
            NodeKind::Worker => Vec2::new(70.0, 50.0),
            NodeKind::Sink => Vec2::new(60.0, 60.0),
            NodeKind::Router => Vec2::new(70.0, 50.0),
            NodeKind::Queue => Vec2::new(90.0, 40.0),
            NodeKind::Custom => Vec2::new(100.0, 60.0),
            NodeKind::Steps => steps_container_size(1),
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

/// Pixel height of a single step row (excluding vertical padding
/// between the row and the container border).
pub const STEPS_ROW_HEIGHT: f32 = 28.0;
/// Pixel width of the Steps container box.
pub const STEPS_WIDTH: f32 = 170.0;
/// Inner vertical padding at the top (for the "Steps" header) and
/// bottom of the container.
pub const STEPS_HEADER_H: f32 = 18.0;
pub const STEPS_PAD_Y: f32 = 8.0;

/// Total outer size of a Steps container given its row count.
pub fn steps_container_size(row_count: usize) -> Vec2 {
    let rows = row_count.max(1) as f32;
    Vec2::new(
        STEPS_WIDTH,
        STEPS_HEADER_H + STEPS_PAD_Y + rows * STEPS_ROW_HEIGHT + STEPS_PAD_Y,
    )
}

/// Y-offset of row `i`'s centre relative to the container centre
/// (positive Y = up). Used by edge anchoring and row hit-testing.
pub fn steps_row_center_y(row_count: usize, i: usize) -> f32 {
    let total = steps_container_size(row_count).y;
    let top_y = total / 2.0;
    // First row starts below header + pad.
    let row_top = top_y - STEPS_HEADER_H - STEPS_PAD_Y;
    let center = row_top - STEPS_ROW_HEIGHT * (i as f32 + 0.5);
    center
}

/// Which row (if any) a click at `world` falls on, given a Steps
/// container centred at `node_center` with `row_count` rows. Returns
/// `None` if the click was in the header/padding or outside the rows.
pub fn steps_row_at(
    world: Vec2,
    node_center: Vec2,
    row_count: usize,
) -> Option<usize> {
    if row_count == 0 {
        return None;
    }
    let dy = world.y - node_center.y;
    for i in 0..row_count {
        let cy = steps_row_center_y(row_count, i);
        if (dy - cy).abs() <= STEPS_ROW_HEIGHT / 2.0 {
            return Some(i);
        }
    }
    None
}

/// Rendering metadata for a node entity. Behavioral state lives in `Sim`.
#[derive(Component)]
pub struct SimNode {
    pub kind: NodeKind,
    pub color: Color,
    /// Current visual outer size in world pixels. For fixed-size kinds
    /// this mirrors `kind.size()`; for Steps it's recomputed whenever
    /// the row count changes.
    pub size: Vec2,
}

/// Child-entity marker on a row rectangle inside a [`NodeKind::Steps`]
/// container. Carries the row index so the highlight system can map
/// the sim's `current_row` onto the right visual.
#[derive(Component)]
pub struct StepsRow {
    pub index: usize,
}

/// Marker on a node that's been "cracked open" on the canvas —
/// the node renders its program as a vertical stack of step rows
/// instead of the small kind-glyph box. Added by double-click,
/// removed by double-click again (or Escape). Works on any kind;
/// Steps containers are effectively always opened.
#[derive(Component)]
pub struct Opened;

/// Cache of the last-rendered shape per node so
/// `rebuild_node_rendering` can detect mismatches and rebuild only
/// when visual state actually changed.
#[derive(Component)]
pub struct RenderedAs {
    pub opened: bool,
    pub row_count: usize,
}

/// Marker on the per-row body mesh whose colour the highlight system
/// paints (accent while active, paper otherwise).
#[derive(Component)]
pub struct StepsRowBody;

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

/// Active row-drag: the user pressed on a specific row of an
/// opened node and is (presumably) sliding it up or down. Released
/// on mouseup — target row index decides whether `swap_step_rows`
/// fires or we noop. `None` = no row drag in progress.
#[derive(Resource, Default)]
pub struct RowDragState {
    pub source_entity: Option<Entity>,
    pub source_node: crate::sim::NodeId,
    pub source_row: usize,
}

/// A step row is currently selected on the canvas. Mirrors the
/// node-level `Selected` marker, but targets a specific row within
/// an opened node. Delete/Backspace removes the selected row;
/// clicking elsewhere clears the selection.
#[derive(Resource, Default)]
pub struct SelectedStep {
    pub entity: Option<Entity>,
    pub node: crate::sim::NodeId,
    pub row: usize,
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
    existing: Query<(&Transform, &SimNode, &SimNodeRef, Option<&Opened>)>,
    library: Res<crate::ui::PresetLibrary>,
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
    if pointer_over_ui(&ui) {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    let Ok((cam, cam_tf)) = cams.single() else { return };
    let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) else { return };
    let sc = crate::bridge::bevy_to_sim_color(active_color.0);

    // First: did the click land on an existing node? If so, the
    // tool might want to mutate it (append a Client row, drop a
    // primitive into an opened node, etc.) rather than spawn a
    // fresh node. This runs BEFORE the node-placement kind lookup
    // because tools like Primitive/UserPreset aren't placement
    // tools at all — they only act on existing targets.
    for (tf, sn, nref, opened) in existing.iter() {
        let half = sn.size / 2.0;
        let c = tf.translation.truncate();
        if (world.x - c.x).abs() <= half.x && (world.y - c.y).abs() <= half.y {
            let is_openable = sn.kind == NodeKind::Steps || opened.is_some();
            let instr = match active_tool.0 {
                Tool::Client if sn.kind == NodeKind::Steps => {
                    Some(crate::sim::client_step(sc))
                }
                Tool::Worker if sn.kind == NodeKind::Steps => {
                    Some(crate::sim::worker_step(500 * crate::sim::NS_PER_MS, sc))
                }
                Tool::UserPreset(i) if sn.kind == NodeKind::Steps => library
                    .user
                    .get(i)
                    .map(|p| crate::sim::Instruction::Sequence {
                        label: p.label.clone(),
                        body: p.body.clone(),
                    }),
                Tool::Primitive(kind) if is_openable => {
                    Some(kind.default_instruction(sc))
                }
                _ => None,
            };
            if let Some(instr) = instr {
                sim_res.0.push_instruction(nref.0, instr);
            }
            // The click hit an existing node: never spawn a new
            // one at this position, even if the tool didn't apply.
            return;
        }
    }

    // Click was on empty canvas. If it's a node-placement tool,
    // spawn. Non-placement tools (Primitive, UserPreset, Select,
    // Edge, Probe) fall through and do nothing.
    let kind = match active_tool.0 {
        Tool::Generator => NodeKind::Generator,
        Tool::Client => NodeKind::Client,
        Tool::Worker => NodeKind::Worker,
        Tool::Sink => NodeKind::Sink,
        Tool::Router => NodeKind::Router,
        Tool::Queue => NodeKind::Queue,
        Tool::Steps => NodeKind::Steps,
        _ => return,
    };

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
    // Steps containers need their sim node to exist before visuals
    // so the initial row count is known for sizing. The
    // placeholder dance lets us reuse `register_node` cleanly.
    let nid = {
        let placeholder = commands.spawn_empty().id();
        let id = register_node(sim_res, maps, placeholder, kind, color);
        commands.entity(placeholder).despawn();
        maps.entity_to_node.remove(&placeholder);
        id
    };
    let row_count = sim_res.0.nodes.get(&nid).map(|n| n.program.len()).unwrap_or(0);
    let size = if kind == NodeKind::Steps {
        steps_container_size(row_count.max(1))
    } else {
        kind.size()
    };

    // Minimal entity — body mesh for the size, core components. The
    // actual children (data dot, border, glyph or stacked rows) are
    // populated by `rebuild_node_rendering` on the first frame.
    // Keeping spawn_node thin means one code path for rendering,
    // avoiding the split-brain we had before.
    let entity = commands
        .spawn((
            SimNode { kind, color, size },
            SimNodeRef(nid),
            Mesh2d(meshes.add(Rectangle::new(size.x, size.y))),
            MeshMaterial2d(materials.add(kind.body_color(theme))),
            Transform::from_xyz(pos.x, pos.y, 0.0),
            Visibility::default(),
        ))
        .id();
    // Rebind entity→node (the placeholder mapping was for
    // `register_node`; we've since despawned it).
    maps.node_to_entity.insert(nid, entity);
    maps.entity_to_node.insert(entity, nid);

    registry.positions.insert(entity, pos);
    let _ = theme;
    entity
}

/// Render a top-level program instruction as the short label shown
/// inside its canvas-row rectangle. Named `Sequence`s use their
/// label (with extra detail where useful — e.g. a "Worker" sequence
/// surfaces its dwell time). Bare primitives render as their
/// instruction name so the user sees the machinery.
pub fn format_step_row(instr: &crate::sim::Instruction) -> String {
    use crate::sim::Instruction;
    match instr {
        Instruction::Sequence { label, body } => {
            if label == "Worker" {
                if let Some(Instruction::Hold { duration_ns }) = body.first() {
                    return format!("Worker {}", fmt_duration_ns(*duration_ns));
                }
            }
            label.clone()
        }
        Instruction::Emit { one_way: false, .. } => "Emit request".to_string(),
        Instruction::Emit { one_way: true, .. } => "Emit".to_string(),
        Instruction::Hold { duration_ns } => format!("Hold {}", fmt_duration_ns(*duration_ns)),
        Instruction::AwaitResponse => "Await".to_string(),
        Instruction::Process { duration_ns } => {
            format!("Process {}", fmt_duration_ns(*duration_ns))
        }
        Instruction::Buffer { .. } => "Buffer".to_string(),
        Instruction::MatchColor { .. } => "Match color".to_string(),
        Instruction::Respond => "Respond".to_string(),
        Instruction::Consume => "Consume".to_string(),
        Instruction::EmitAtRate { .. } => "Emit at rate".to_string(),
        Instruction::AcceptInbound => "Accept".to_string(),
        Instruction::PullInbound => "Pull".to_string(),
        Instruction::Filter { pred } => format!("Filter · {}", format_port_pred(*pred)),
        Instruction::Sort { key } => format!("Sort · {}", format_port_key(*key)),
        Instruction::Take { n } => format!("Take {}", n),
        Instruction::Send => "Send".to_string(),
        Instruction::Require { .. } => "Require".to_string(),
    }
}

fn format_port_pred(p: crate::sim::PortPredicate) -> &'static str {
    use crate::sim::PortPredicate;
    match p {
        PortPredicate::Ready => "ready",
        PortPredicate::ColorMatches => "color match",
    }
}

fn format_port_key(k: crate::sim::PortKey) -> &'static str {
    use crate::sim::PortKey;
    match k {
        PortKey::LastSentAt => "last sent",
        PortKey::QueueDepth => "queue depth",
        PortKey::EdgeOrder => "edge order",
        PortKey::Random => "random",
    }
}

fn fmt_duration_ns(ns: u64) -> String {
    if ns >= NS_PER_S {
        format!("{:.1}s", ns as f64 / NS_PER_S as f64)
    } else if ns >= NS_PER_MS {
        format!("{}ms", ns / NS_PER_MS)
    } else if ns >= NS_PER_US {
        format!("{}us", ns / NS_PER_US)
    } else {
        format!("{}ns", ns)
    }
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
        // Steps container uses row children for its visible content,
        // so the body glyph would collide with the header text the row
        // renderer draws at the top. Keep it empty.
        NodeKind::Steps => "",
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
            let half = sn.size / 2.0;
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
            let half = sn.size / 2.0;
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
    row_drag: Res<RowDragState>,
    mut drag: ResMut<DragState>,
    mut commands: Commands,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    // If the press started a row-drag, don't start a node-drag in
    // addition — the user is reordering, not moving the node.
    if row_drag.source_entity.is_some() {
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
        let half = sn.size / 2.0 + Vec2::splat(6.0);
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
    selected_step: Res<SelectedStep>,
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
    // A step is currently selected on the canvas. Delete belongs
    // to the step; `delete_selected_step` handled it — don't also
    // delete the containing node.
    if selected_step.entity.is_some() {
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
                    // Row info is drawn directly on the Steps body by
                    // the row renderer; keep the shared canvas label
                    // empty to avoid double-labeling.
                    NodeKind::Steps => String::new(),
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
                size: NodeKind::Custom.size(),
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
            let half = m_sn.size / 2.0;
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
        let half = sn.size / 2.0;
        let red = Color::srgb(0.85, 0.2, 0.2);
        gizmos.line_2d(c - half, c + half, red);
        gizmos.line_2d(
            Vec2::new(c.x - half.x, c.y + half.y),
            Vec2::new(c.x + half.x, c.y - half.y),
            red,
        );
    }
}
