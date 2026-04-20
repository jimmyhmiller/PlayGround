//! Selection inspector — slots into the right palette panel under the Data
//! Palette section. Shows readouts and editable controls for the currently
//! selected node, mirroring the design's `InspectorBody`.
//!
//! Layout decisions:
//! - The `InspectorRoot` UI node lives at the bottom of the palette body. Its
//!   children are despawned and respawned every time the selection changes,
//!   which is fine: selection only changes on user click.
//! - Sliders are native — a track UI node with a coloured fill child whose
//!   width tracks the slider value. Click and drag inside the track to set
//!   the value; the slider writes back to the sim through a `SliderTarget`.
//!
//! No DOM inspector wrappers (egui etc.) — kept lean so future themes can
//! re-skin via the same `Theme` resource pattern as the rest of the app.

use crate::bridge::{bevy_to_sim_color, Bold, SimResource};
use crate::nodes::{NodeKind, SimNode};
use crate::palette::caps_spaced;
use crate::sim::{NodeId, NS_PER_MS, NS_PER_S, period_ns_to_rate, rate_to_period_ns};
use crate::theme::{Theme, DATA_SLOT_COUNT};
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::prelude::*;
use bevy::ui::UiGlobalTransform;

pub struct InspectorPlugin;

impl Plugin for InspectorPlugin {
    fn build(&self, app: &mut App) {
        // The InspectorRoot mount is spawned by `palette::spawn_palette` so
        // it sits inside the panel body in the right place from frame one.
        // We just rebuild its children whenever selection changes; sliders
        // refresh from sim state every frame so keyboard adjustments stay
        // visually in sync.
        app.init_resource::<InspectorState>()
            .init_resource::<SliderDrag>()
            .add_systems(
                Update,
                (
                    rebuild_inspector_on_selection_change,
                    start_slider_drag,
                    continue_slider_drag,
                    sync_slider_visuals,
                    handle_delete_button,
                    handle_color_picker,
                )
                    .chain(),
            );
    }
}

/// While the user is dragging a slider's thumb, we pin the drag to that
/// entity so the value keeps tracking the cursor even if it strays off the
/// track's hitbox. Cleared when the left mouse button is released.
#[derive(Resource, Default)]
struct SliderDrag {
    entity: Option<Entity>,
}

/// Marker for the empty container we slot inspector contents into. Spawned
/// by `palette::spawn_palette` as the last child of the panel body.
#[derive(Component)]
pub struct InspectorRoot;

#[derive(Resource, Default)]
pub struct InspectorState {
    /// `None` means "show nothing"; `Some(id)` means show this node's body.
    /// Tracked separately from the live `Selected` query so we only rebuild
    /// on actual selection changes — except when the rebuild cache is busted
    /// externally (e.g. the color picker writes a new colour and resets
    /// this to `None` so the next frame rebuilds with the new state).
    pub last_selected: Option<NodeId>,
}

/// Slider control. Carries everything we need to update the sim — `target`
/// names what to write, `min`/`max` clamp the user's drag, `value` mirrors
/// the current sim value (refreshed each frame so the slider visual matches
/// keyboard `+`/`-` adjustments).
#[derive(Component)]
struct Slider {
    pub min: f32,
    pub max: f32,
    pub value: f32,
    pub target: SliderTarget,
}

/// Marker on the rectangular fill child of a `Slider`. Its width is set by
/// `sync_slider_visuals` to `value/(max-min)` of its parent's width.
#[derive(Component)]
struct SliderFill;

#[derive(Clone, Copy)]
enum SliderTarget {
    GeneratorRate(NodeId),
    ClientRate(NodeId),
    WorkerProcessingMs(NodeId),
    /// Worker-row duration inside a Steps container. Writes back
    /// through `Sim::set_steps_worker_duration_ns`.
    StepsWorkerMs { node: NodeId, row: usize },
}

#[derive(Component, Clone, Copy)]
struct DeleteSelectedButton;

/// One swatch in the inspector's color picker. Holds the data-slot index
/// (0..DATA_SLOT_COUNT) and the sim NodeId it should write to when pressed.
#[derive(Component, Clone, Copy)]
struct ColorPickerSwatch {
    slot: usize,
    target: NodeId,
}

// ──────────────────────────────────────────────────────────────────
// Setup
// ──────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────
// Selection-driven rebuild
// ──────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn rebuild_inspector_on_selection_change(
    mut commands: Commands,
    theme: Res<Theme>,
    sim: Res<SimResource>,
    selected_q: Query<(&crate::bridge::SimNodeRef, &SimNode), With<crate::nodes::Selected>>,
    inspector_q: Query<Entity, With<InspectorRoot>>,
    children_q: Query<&Children>,
    mut inspector_node_q: Query<&mut Node, With<InspectorRoot>>,
    mut state: ResMut<InspectorState>,
) {
    let current = selected_q
        .iter()
        .next()
        .map(|(nref, sn)| (nref.0, sn.kind));

    // Always re-render when nothing was selected before but something is now,
    // when the selected node changes, or when selection just cleared. We
    // include the kind in the comparison so re-selecting a node of a
    // different kind triggers a rebuild even if the id sequence happens to
    // collide with a stale entry.
    let current_id = current.map(|(id, _)| id);
    if current_id == state.last_selected {
        return;
    }
    state.last_selected = current_id;

    let Ok(root) = inspector_q.single() else { return };
    // Children may not exist yet if we've never populated the inspector;
    // that's fine — nothing to despawn.
    if let Ok(children) = children_q.get(root) {
        for child in children.iter() {
            commands.entity(child).despawn();
        }
    }

    // Hide if nothing selected.
    let Some((node_id, kind)) = current else {
        if let Ok(mut n) = inspector_node_q.single_mut() {
            n.display = Display::None;
        }
        return;
    };

    if let Ok(mut n) = inspector_node_q.single_mut() {
        n.display = Display::Flex;
    }

    commands.entity(root).with_children(|inspector| {
        section_header(inspector, "Inspector", &theme);

        kv_row(inspector, "Type", kind_label(kind), &theme, false);
        kv_row(inspector, "ID", &format!("{}", node_id), &theme, true);

        if let Some(node) = sim.0.nodes.get(&node_id) {
            match kind {
                NodeKind::Generator => {
                    spawn_slider_row(
                        inspector,
                        "Rate",
                        period_ns_to_rate(node.emit_period_ns()) as f32,
                        0.0,
                        10.0,
                        "/s",
                        SliderTarget::GeneratorRate(node_id),
                        &theme,
                    );
                    spawn_color_picker(inspector, node_id, node.color, &theme);
                }
                NodeKind::Client => {
                    spawn_slider_row(
                        inspector,
                        "Rate",
                        period_ns_to_rate(node.emit_period_ns()) as f32,
                        0.0,
                        10.0,
                        "/s",
                        SliderTarget::ClientRate(node_id),
                        &theme,
                    );
                    spawn_color_picker(inspector, node_id, node.color, &theme);
                    kv_row(inspector, "Sent", &format!("{}", node.sent), &theme, true);
                    kv_row(inspector, "Recv", &format!("{}", node.received), &theme, true);
                }
                NodeKind::Worker => {
                    spawn_slider_row(
                        inspector,
                        "Service",
                        (node.processing_ns() / NS_PER_MS) as f32,
                        1.0,
                        2000.0,
                        "ms",
                        SliderTarget::WorkerProcessingMs(node_id),
                        &theme,
                    );
                    kv_row(inspector, "Processed", &format!("{}", node.processed), &theme, true);
                }
                NodeKind::Queue => {
                    kv_row(inspector, "Buffered", &format!("{}", node.buffer.len()), &theme, true);
                    kv_row(inspector, "In", &format!("{}", node.total_in), &theme, true);
                    kv_row(inspector, "Out", &format!("{}", node.total_out), &theme, true);
                }
                NodeKind::Sink => {
                    kv_row(inspector, "Total", &format!("{}", node.sink_total), &theme, true);
                    kv_row(inspector, "Dropped", &format!("{}", node.dropped), &theme, true);
                }
                NodeKind::Router => {
                    kv_row(inspector, "Policy", "Round-robin", &theme, false);
                }
                NodeKind::Custom => {
                    kv_row(
                        inspector,
                        "Contains",
                        &format!("{} nodes", node.contains.len()),
                        &theme,
                        true,
                    );
                }
                NodeKind::Steps => {
                    kv_row(
                        inspector,
                        "Current",
                        &node
                            .current_row
                            .map(|i| i.to_string())
                            .unwrap_or_else(|| "-".into()),
                        &theme,
                        true,
                    );
                    kv_row(inspector, "Sent", &format!("{}", node.sent), &theme, true);
                    kv_row(inspector, "Recv", &format!("{}", node.received), &theme, true);

                    if !node.step_rows.is_empty() {
                        section_header(inspector, "Rows", &theme);
                        for (i, row) in node.step_rows.iter().enumerate() {
                            match row {
                                crate::sim::StepRow::Client { .. } => {
                                    kv_row(
                                        inspector,
                                        &format!("{}", i),
                                        "Client",
                                        &theme,
                                        false,
                                    );
                                }
                                crate::sim::StepRow::Worker { duration_ns, .. } => {
                                    spawn_slider_row(
                                        inspector,
                                        &format!("{} · Worker", i),
                                        (*duration_ns / NS_PER_MS) as f32,
                                        1.0,
                                        5000.0,
                                        "ms",
                                        SliderTarget::StepsWorkerMs {
                                            node: node_id,
                                            row: i,
                                        },
                                        &theme,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        spawn_delete_button(inspector, &theme);
    });
}

// ──────────────────────────────────────────────────────────────────
// Row helpers
// ──────────────────────────────────────────────────────────────────

fn section_header(parent: &mut ChildSpawnerCommands, label: &str, theme: &Theme) {
    parent
        .spawn((
            Node {
                width: Val::Percent(100.0),
                padding: UiRect {
                    top: Val::Px(12.0),
                    bottom: Val::Px(6.0),
                    left: Val::Px(4.0),
                    right: Val::Px(4.0),
                },
                margin: UiRect::bottom(Val::Px(6.0)),
                border: UiRect::bottom(Val::Px(1.0)),
                ..default()
            },
            BorderColor::all(theme.rule),
        ))
        .with_children(|p| {
            p.spawn((
                Text::new(caps_spaced(label)),
                TextFont { font_size: 10.0, ..default() },
                TextColor(theme.ink_soft),
                Bold,
            ));
        });
}

fn kv_row(parent: &mut ChildSpawnerCommands, label: &str, value: &str, theme: &Theme, _mono: bool) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            padding: UiRect::vertical(Val::Px(5.0)),
            justify_content: JustifyContent::SpaceBetween,
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|r| {
            r.spawn((
                Text::new(caps_spaced(label)),
                TextFont { font_size: 9.0, ..default() },
                TextColor(theme.ink_soft),
                Bold,
            ));
            r.spawn((
                Text::new(value.to_string()),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.ink),
                Bold,
            ));
        });
}

fn spawn_slider_row(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    value: f32,
    min: f32,
    max: f32,
    unit: &'static str,
    target: SliderTarget,
    theme: &Theme,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            padding: UiRect::vertical(Val::Px(6.0)),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(4.0),
            ..default()
        })
        .with_children(|row| {
            // Top: label + value readout.
            row.spawn(Node {
                width: Val::Percent(100.0),
                justify_content: JustifyContent::SpaceBetween,
                ..default()
            })
            .with_children(|head| {
                head.spawn((
                    Text::new(caps_spaced(label)),
                    TextFont { font_size: 9.0, ..default() },
                    TextColor(theme.ink_soft),
                    Bold,
                ));
                head.spawn((
                    Text::new(format_slider_value(value, unit)),
                    TextFont { font_size: 11.0, ..default() },
                    TextColor(theme.ink),
                    Bold,
                    SliderValueText(target_id(target)),
                ));
            });

            // Track: clickable bar with a fill child.
            row.spawn((
                Button,
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(14.0),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    overflow: Overflow::clip(),
                    ..default()
                },
                BackgroundColor(theme.paper),
                BorderColor::all(theme.ink),
                Slider { min, max, value, target },
            ))
            .with_children(|t| {
                t.spawn((
                    Node {
                        width: Val::Percent(slider_fill_pct(value, min, max)),
                        height: Val::Percent(100.0),
                        ..default()
                    },
                    BackgroundColor(theme.accent),
                    SliderFill,
                ));
            });
        });
}

/// Marker on the slider's value-readout `Text` so we can update it without
/// rebuilding the row. Holds the matching `target_id` so multi-slider
/// inspectors stay disambiguated.
#[derive(Component)]
struct SliderValueText(u64);

fn target_id(t: SliderTarget) -> u64 {
    match t {
        SliderTarget::GeneratorRate(id) => id,
        SliderTarget::ClientRate(id) => id,
        SliderTarget::WorkerProcessingMs(id) => id,
        // Mix node id + row index so multiple Worker-row sliders on
        // the same Steps node don't collide. High 16 bits = row,
        // low 48 bits = node id — plenty of room for either.
        SliderTarget::StepsWorkerMs { node, row } => {
            ((row as u64) << 48) | (node & 0xFFFF_FFFF_FFFF)
        }
    }
}

fn slider_fill_pct(value: f32, min: f32, max: f32) -> f32 {
    if max <= min { return 0.0; }
    ((value - min) / (max - min)).clamp(0.0, 1.0) * 100.0
}

fn format_slider_value(v: f32, unit: &str) -> String {
    if unit == "/s" {
        format!("{:.1}{}", v, unit)
    } else if unit == "ms" {
        if v >= 1000.0 {
            format!("{:.2}s", v / 1000.0)
        } else {
            format!("{:.0}{}", v, unit)
        }
    } else {
        format!("{:.0}{}", v, unit)
    }
}

fn spawn_color_picker(
    parent: &mut ChildSpawnerCommands,
    target: NodeId,
    current: crate::sim::Color,
    theme: &Theme,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            padding: UiRect::vertical(Val::Px(6.0)),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(4.0),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(caps_spaced("Color")),
                TextFont { font_size: 9.0, ..default() },
                TextColor(theme.ink_soft),
                Bold,
            ));
            row.spawn(Node {
                width: Val::Percent(100.0),
                column_gap: Val::Px(4.0),
                ..default()
            })
            .with_children(|swatches| {
                for slot in 0..DATA_SLOT_COUNT {
                    let theme_color = theme.data[slot];
                    let is_active = bevy_to_sim_color(theme_color) == current;
                    swatches.spawn((
                        Button,
                        Node {
                            flex_grow: 1.0,
                            height: Val::Px(22.0),
                            border: UiRect::all(Val::Px(if is_active { 2.0 } else { 1.0 })),
                            border_radius: BorderRadius::all(Val::Px(4.0)),
                            ..default()
                        },
                        BackgroundColor(theme_color),
                        BorderColor::all(if is_active { theme.ink } else { theme.rule }),
                        ColorPickerSwatch { slot, target },
                    ));
                }
            });
        });
}

fn spawn_delete_button(parent: &mut ChildSpawnerCommands, theme: &Theme) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                margin: UiRect::top(Val::Px(10.0)),
                padding: UiRect::vertical(Val::Px(8.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::NONE),
            BorderColor::all(theme.accent),
            DeleteSelectedButton,
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(caps_spaced("Delete · ⌫")),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.accent),
                Bold,
            ));
        });
}

fn kind_label(k: NodeKind) -> &'static str {
    match k {
        NodeKind::Generator => "Generator",
        NodeKind::Client => "Client",
        NodeKind::Worker => "Worker",
        NodeKind::Router => "Router",
        NodeKind::Queue => "Queue",
        NodeKind::Sink => "Sink",
        NodeKind::Custom => "Group",
        NodeKind::Steps => "Steps",
    }
}

// ──────────────────────────────────────────────────────────────────
// Slider drag + sync
// ──────────────────────────────────────────────────────────────────

/// On any slider's first `Pressed` frame, pin it as the active drag target.
/// We track it across frames in `SliderDrag` because Bevy's `Interaction`
/// goes back to `None` the moment the cursor leaves the track — which would
/// make drag-past-the-edge stop updating. This way the drag sticks to the
/// slider until the mouse is released.
fn start_slider_drag(
    sliders: Query<(Entity, &Interaction), (Changed<Interaction>, With<Slider>)>,
    mut drag: ResMut<SliderDrag>,
) {
    for (entity, interaction) in sliders.iter() {
        if *interaction == Interaction::Pressed {
            drag.entity = Some(entity);
        }
    }
}

/// While the left mouse button is held and we have a pinned slider, read
/// the cursor each frame, compute its X fraction along the track in node-
/// local space, and write that back through the sim setter. Release clears
/// the pin.
fn continue_slider_drag(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    mut drag: ResMut<SliderDrag>,
    mut sliders: Query<(&ComputedNode, &UiGlobalTransform, &mut Slider)>,
    mut sim: ResMut<SimResource>,
) {
    if !mouse.pressed(MouseButton::Left) {
        drag.entity = None;
        return;
    }
    let Some(entity) = drag.entity else { return };
    let Ok(win) = windows.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    let Ok((computed, xform, mut slider)) = sliders.get_mut(entity) else {
        drag.entity = None;
        return;
    };

    // Compute the track's logical-pixel rect from UiGlobalTransform's
    // translation (= node centre) and ComputedNode.size. Bail if layout
    // hasn't produced a valid size yet — writing a value based on a zero-
    // width track would snap the slider to 0 every frame.
    let size = computed.size;
    if size.x <= 0.0 {
        return;
    }
    // Bevy 0.18's `ComputedNode.size` and `UiGlobalTransform.translation`
    // are in PHYSICAL pixels, but `cursor_position()` returns LOGICAL
    // pixels. On a Retina display these differ by `scale_factor` (2×),
    // which was silently pinning every drag to frac=0. Scale cursor up to
    // physical before doing the math.
    let scale = win.scale_factor();
    let cursor_px = cursor.x * scale;
    let center_x = xform.translation.x;
    let left = center_x - size.x * 0.5;
    let frac = ((cursor_px - left) / size.x).clamp(0.0, 1.0);
    let new_val = slider.min + frac * (slider.max - slider.min);
    if (new_val - slider.value).abs() < 1e-4 {
        return;
    }
    slider.value = new_val;
    write_slider_to_sim(&mut sim, slider.target, new_val);
}

fn write_slider_to_sim(sim: &mut SimResource, target: SliderTarget, value: f32) {
    match target {
        SliderTarget::GeneratorRate(id) => {
            let period = rate_to_period_ns(value as f64);
            sim.0.set_generator_period_ns(id, period);
        }
        SliderTarget::ClientRate(id) => {
            let period = rate_to_period_ns(value as f64);
            sim.0.set_client_period_ns(id, period);
        }
        SliderTarget::WorkerProcessingMs(id) => {
            let ns = (value as u64).max(1) * NS_PER_MS;
            sim.0.set_worker_processing_ns(id, ns);
        }
        SliderTarget::StepsWorkerMs { node, row } => {
            let ns = (value as u64).max(1) * NS_PER_MS;
            sim.0.set_steps_worker_duration_ns(node, row, ns);
        }
    }
}

/// Per-frame: refresh the slider's `value` from sim state, repaint the fill
/// width, and update the readout text. This means keyboard `+`/`-` or the
/// `E` editor change the slider without us listening to those input paths.
fn sync_slider_visuals(
    sim: Res<SimResource>,
    mut sliders: Query<(&mut Slider, &Children)>,
    mut fills: Query<&mut Node, With<SliderFill>>,
    mut texts: Query<(&mut Text, &SliderValueText)>,
) {
    for (mut slider, children) in sliders.iter_mut() {
        let live = match slider.target {
            SliderTarget::GeneratorRate(id) | SliderTarget::ClientRate(id) => sim
                .0
                .nodes
                .get(&id)
                .map(|n| period_ns_to_rate(n.emit_period_ns()) as f32),
            SliderTarget::WorkerProcessingMs(id) => sim
                .0
                .nodes
                .get(&id)
                .map(|n| (n.processing_ns() / NS_PER_MS) as f32),
            SliderTarget::StepsWorkerMs { node, row } => sim
                .0
                .nodes
                .get(&node)
                .and_then(|n| n.step_rows.get(row))
                .and_then(|r| match r {
                    crate::sim::StepRow::Worker { duration_ns, .. } => {
                        Some((*duration_ns / NS_PER_MS) as f32)
                    }
                    _ => None,
                }),
        };
        if let Some(live) = live {
            slider.value = live;
        }
        let pct = slider_fill_pct(slider.value, slider.min, slider.max);
        for child in children.iter() {
            if let Ok(mut n) = fills.get_mut(child) {
                n.width = Val::Percent(pct);
            }
        }
        let id = target_id(slider.target);
        let unit = slider_unit(slider.target);
        for (mut text, tag) in texts.iter_mut() {
            if tag.0 == id {
                text.0 = format_slider_value(slider.value, unit);
            }
        }
    }
    let _ = NS_PER_S;
}

fn slider_unit(t: SliderTarget) -> &'static str {
    match t {
        SliderTarget::GeneratorRate(_) | SliderTarget::ClientRate(_) => "/s",
        SliderTarget::WorkerProcessingMs(_) | SliderTarget::StepsWorkerMs { .. } => "ms",
    }
}

/// Click on a color-picker swatch to retag the selected emitter's data
/// colour. Forces a rebuild on the next frame so the active-state outline
/// snaps to the chosen swatch (the rebuild also re-renders the data colour
/// pip on the node body, which reads from sim).
fn handle_color_picker(
    swatches: Query<(&Interaction, &ColorPickerSwatch), Changed<Interaction>>,
    theme: Res<Theme>,
    mut sim: ResMut<SimResource>,
    mut state: ResMut<InspectorState>,
) {
    for (interaction, swatch) in swatches.iter() {
        if *interaction == Interaction::Pressed {
            let theme_color = theme.data[swatch.slot];
            sim.0.set_emitter_color(swatch.target, bevy_to_sim_color(theme_color));
            // Bust the rebuild cache so the inspector picks up the change.
            state.last_selected = None;
        }
    }
}

/// "Delete · ⌫" button at the bottom of the inspector. Despawns the selected
/// node entity; the rebuild system will then hide the inspector body.
fn handle_delete_button(
    mut commands: Commands,
    btn_q: Query<&Interaction, (Changed<Interaction>, With<DeleteSelectedButton>)>,
    selected_q: Query<(Entity, &crate::bridge::SimNodeRef), With<crate::nodes::Selected>>,
    probes_q: Query<(Entity, &crate::edges::Probe)>,
    mut sim: ResMut<SimResource>,
    mut maps: ResMut<crate::bridge::EntityMaps>,
) {
    let pressed = btn_q.iter().any(|i| *i == Interaction::Pressed);
    if !pressed {
        return;
    }
    let Some((entity, nref)) = selected_q.iter().next() else { return };
    let removed_edges = sim.0.remove_node(nref.0);
    let mut despawned_edges: std::collections::HashSet<Entity> = Default::default();
    for eid in removed_edges {
        if let Some(edge_entity) = maps.edge_to_entity.remove(&eid) {
            commands.entity(edge_entity).despawn();
            despawned_edges.insert(edge_entity);
        }
        maps.entity_to_edge.retain(|_, v| *v != eid);
    }
    for (probe_entity, probe) in probes_q.iter() {
        let dangling = match probe.target {
            crate::edges::ProbeTarget::Node(n) => n == entity,
            crate::edges::ProbeTarget::Edge(e) => despawned_edges.contains(&e),
        };
        if dangling {
            commands.entity(probe_entity).despawn();
        }
    }
    maps.entity_to_node.remove(&entity);
    maps.node_to_entity.remove(&nref.0);
    commands.entity(entity).despawn();
}
