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
            .init_resource::<InspectorFocus>()
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
                    handle_step_row_buttons,
                    handle_cycle_enum_buttons,
                    handle_breadcrumb_clicks,
                    handle_focus_unwind,
                    handle_canvas_drill_click,
                    handle_save_preset_button,
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
    /// `None` = show nothing; `Some(id)` = show this node's body.
    /// Distinct from the live `Selected` query so we only rebuild
    /// on actual selection changes.
    pub last_selected: Option<NodeId>,
    /// Mirror of the current `InspectorFocus.path` so we rebuild when
    /// the user drills into a Sequence or unwinds out.
    pub last_focus_depth: usize,
    /// External "please rebuild next frame" flag. Set by edit
    /// handlers (color picker, row buttons, drill click, etc.)
    /// that mutated state the rebuild needs to pick up. Consumed
    /// by the rebuild system. Separate from `last_selected` so
    /// force-rebuild doesn't mimic a selection change and clobber
    /// focus.
    pub dirty: bool,
}

/// Drill-in state. `open = false` shows the kind's summary view
/// (per-kind sliders, stats). `open = true` shows the raw program
/// for the selected node — a list of `Instruction`s. `path`
/// extends further when the user drills into a nested Sequence.
#[derive(Resource, Default)]
pub struct InspectorFocus {
    pub open: bool,
    pub path: Vec<usize>,
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
    mut focus: ResMut<InspectorFocus>,
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

    let current_id = current.map(|(id, _)| id);
    // Only genuinely moving between two different nodes (both
    // Some, different ids) clears drill-in state. Deselecting
    // doesn't — we keep the focus around so the next selection of
    // the same node restores the drill-in view.
    if let (Some(cur), Some(last)) = (current_id, state.last_selected) {
        if cur != last {
            focus.path.clear();
            focus.open = false;
        }
    }

    // Rebuild trigger: selection change, focus depth change, or an
    // explicit dirty flag set by edit handlers.
    let focus_depth = if focus.open { focus.path.len() + 1 } else { 0 };
    let dirty = state.dirty;
    if !dirty
        && current_id == state.last_selected
        && focus_depth == state.last_focus_depth
    {
        return;
    }
    state.last_selected = current_id;
    state.last_focus_depth = focus_depth;
    state.dirty = false;

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
            // "Opened" view — drill-in shows the node's program as
            // rows. Triggered by double-click on any node. For
            // Steps containers we always show the row list
            // regardless of `open` because it IS their primary UI.
            let program_view = focus.open || kind == NodeKind::Steps;
            if program_view {
                spawn_program_view(
                    inspector,
                    &theme,
                    node_id,
                    kind,
                    node,
                    &focus.path,
                );
            } else {
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
                    NodeKind::Steps => unreachable!("handled by program_view above"),
                }
            }
        }

        spawn_delete_button(inspector, &theme);
    });
}

/// The "program view" — shown when a node is cracked open
/// (double-clicked) or it's a Steps container. Renders a
/// breadcrumb, each top-level program entry as a row (editable for
/// Steps at top-level; read-only otherwise), and a Save-as-preset
/// button when drilled into a Sequence.
fn spawn_program_view(
    inspector: &mut ChildSpawnerCommands,
    theme: &Theme,
    node_id: crate::sim::NodeId,
    kind: NodeKind,
    node: &crate::sim::Node,
    path: &[usize],
) {
    use crate::ui::{LiveField, LiveText, RowControl, row};

    // Steps top-level readouts: container stats. Other kinds
    // cracked open don't have analogous stats at this level.
    if kind == NodeKind::Steps && path.is_empty() {
        row(
            inspector,
            theme,
            "Current",
            RowControl::Readout {
                text: "",
                live: Some(LiveText { node: node_id, field: LiveField::CurrentRow }),
            },
        );
        row(
            inspector,
            theme,
            "Sent",
            RowControl::Readout {
                text: "",
                live: Some(LiveText { node: node_id, field: LiveField::Sent }),
            },
        );
        row(
            inspector,
            theme,
            "Recv",
            RowControl::Readout {
                text: "",
                live: Some(LiveText { node: node_id, field: LiveField::Received }),
            },
        );
    }

    // Breadcrumb: kind name as root, then each drilled-in Sequence label.
    spawn_breadcrumb(inspector, theme, kind, &node.program, path);

    let scope = crate::sim::body_at(&node.program, path)
        .unwrap_or(node.program.as_slice());

    // Drilled into a Sequence → offer Save-as-preset.
    if !path.is_empty() {
        spawn_save_preset_button(inspector, theme, node_id, path.to_vec());
    }

    if !scope.is_empty() {
        section_header(
            inspector,
            if path.is_empty() { "Program" } else { "Body" },
            theme,
        );
        let row_count = scope.len();
        // Editing (slider, reorder, delete) is only valid for
        // Steps at top-level: sim setters and `from_row` edge
        // anchors both key off top-level indices there. Every
        // other case shows read-only rows.
        let editable = kind == NodeKind::Steps && path.is_empty();
        for (i, instr) in scope.iter().enumerate() {
            spawn_step_row_editor(
                inspector,
                theme,
                node_id,
                i,
                instr,
                row_count,
                editable,
            );
        }
    }
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
                    crate::bridge::Mono,
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

/// Render a breadcrumb trail like "Steps › Client" showing the
/// current focus depth. Each segment is a button tagged with the
/// depth it navigates to (0 = top-level, 1 = first Sequence body,
/// etc.) so `handle_breadcrumb_clicks` can pop the focus stack.
fn spawn_breadcrumb(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    kind: NodeKind,
    program: &[crate::sim::Instruction],
    path: &[usize],
) {
    use crate::sim::Instruction;
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            padding: UiRect::vertical(Val::Px(4.0)),
            column_gap: Val::Px(4.0),
            align_items: AlignItems::Center,
            flex_wrap: FlexWrap::Wrap,
            ..default()
        })
        .with_children(|bc| {
            spawn_breadcrumb_segment(bc, theme, kind_label(kind), 0);

            let mut scope: &[Instruction] = program;
            for (depth_plus_one, &idx) in path.iter().enumerate() {
                let sep = bc.spawn(Node::default());
                drop(sep);
                bc.spawn((
                    Text::new("›".to_string()),
                    TextFont { font_size: 11.0, ..default() },
                    TextColor(theme.ink_soft),
                ));
                let label = match scope.get(idx) {
                    Some(Instruction::Sequence { label, body }) => {
                        scope = body;
                        label.clone()
                    }
                    _ => format!("[{}]", idx),
                };
                spawn_breadcrumb_segment(bc, theme, &label, depth_plus_one + 1);
            }
        });
}

fn spawn_breadcrumb_segment(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    label: &str,
    target_depth: usize,
) {
    parent
        .spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(6.0), Val::Px(2.0)),
                border: UiRect::all(Val::Px(0.0)),
                ..default()
            },
            BackgroundColor(Color::NONE),
            BreadcrumbSegment { target_depth },
        ))
        .with_children(|s| {
            s.spawn((
                Text::new(label.to_string()),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.ink),
                crate::bridge::Bold,
            ));
        });
}

/// Marker component on a breadcrumb button. `target_depth` is the
/// `focus.path.len()` we should truncate to when pressed (so
/// clicking "Steps" truncates to 0 = back to the top, clicking the
/// first drill-in level truncates to 1, etc.).
#[derive(Component, Clone, Copy)]
struct BreadcrumbSegment {
    target_depth: usize,
}

/// One inspector row for editing a single step row of a Steps node.
/// Layout: `idx · kind · [control]? · [↑ ↓ ×]`. Client rows show
/// just a readout; Worker rows show a duration slider. Both get the
/// reorder + delete trailing actions.
fn spawn_step_row_editor(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    node_id: crate::sim::NodeId,
    index: usize,
    instr: &crate::sim::Instruction,
    row_count: usize,
    editable: bool,
) {
    use crate::sim::Instruction;
    use crate::ui::{RowControl, row, row_with_actions};
    let duration = match instr {
        Instruction::Process { duration_ns } => Some(*duration_ns),
        Instruction::Hold { duration_ns } => Some(*duration_ns),
        _ => None,
    };
    // Is this an "enum-cycling" primitive? Sort/Filter/Take/Require
    // all have a finite variant set and can be advanced in place
    // with a click.
    let cyclable = matches!(
        instr,
        Instruction::Sort { .. }
            | Instruction::Filter { .. }
            | Instruction::Take { .. }
            | Instruction::Require { .. }
    );
    match instr {
        Instruction::Sequence { label, body } if label == "Worker" && editable => {
            spawn_worker_row_editor(parent, theme, node_id, index, body, row_count);
        }
        _ if duration.is_some() && editable => {
            spawn_duration_row_editor(
                parent,
                theme,
                node_id,
                index,
                duration.unwrap(),
                crate::nodes::format_step_row(instr),
                row_count,
            );
        }
        _ if cyclable && editable => {
            spawn_cycling_row_editor(
                parent,
                theme,
                node_id,
                index,
                crate::nodes::format_step_row(instr),
                row_count,
            );
        }
        Instruction::Sequence { label, .. } if editable => {
            let label_text = label.clone();
            row_with_actions(
                parent,
                theme,
                &format!("{}", index),
                RowControl::Readout { text: &label_text, live: None },
                |actions| spawn_row_action_buttons(actions, theme, node_id, index, row_count),
            );
        }
        other if editable => {
            let name = crate::nodes::format_step_row(other);
            row_with_actions(
                parent,
                theme,
                &format!("{}", index),
                RowControl::Readout { text: &name, live: None },
                |actions| spawn_row_action_buttons(actions, theme, node_id, index, row_count),
            );
        }
        other => {
            // Read-only (drilled-in scope).
            let name = crate::nodes::format_step_row(other);
            row(
                parent,
                theme,
                &format!("{}", index),
                RowControl::Readout { text: &name, live: None },
            );
        }
    }
}

/// Row with a clickable value that cycles to the next variant of
/// the row's underlying enum (Sort's key, Filter's pred, etc.).
/// Trailing reorder/delete actions are kept.
fn spawn_cycling_row_editor(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    node_id: crate::sim::NodeId,
    index: usize,
    label_text: String,
    row_count: usize,
) {
    use crate::ui::{RowControl, row_with_actions};
    row_with_actions(
        parent,
        theme,
        &format!("{}", index),
        RowControl::Readout { text: "", live: None },
        |actions| {
            actions
                .spawn((
                    Button,
                    Node {
                        padding: UiRect::axes(Val::Px(6.0), Val::Px(3.0)),
                        border: UiRect::all(Val::Px(1.0)),
                        border_radius: BorderRadius::all(Val::Px(4.0)),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        ..default()
                    },
                    BackgroundColor(Color::NONE),
                    BorderColor::all(theme.rule),
                    CycleEnumButton { node: node_id, row: index },
                ))
                .with_children(|b| {
                    b.spawn((
                        Text::new(label_text),
                        TextFont { font_size: 11.0, ..default() },
                        TextColor(theme.ink),
                        crate::bridge::Bold,
                    ));
                });
            spawn_row_action_buttons(actions, theme, node_id, index, row_count);
        },
    );
}

#[derive(Component, Clone, Copy)]
struct CycleEnumButton {
    node: crate::sim::NodeId,
    row: usize,
}

fn handle_cycle_enum_buttons(
    q: Query<(&Interaction, &CycleEnumButton), Changed<Interaction>>,
    mut sim_res: ResMut<SimResource>,
    mut state: ResMut<InspectorState>,
) {
    for (i, btn) in q.iter() {
        if *i == Interaction::Pressed {
            sim_res.0.cycle_steps_row_enum(btn.node, btn.row);
            state.dirty = true;
        }
    }
}

/// Row with a duration slider plus trailing reorder/delete
/// actions. Used for bare `Process` / `Hold` rows.
fn spawn_duration_row_editor(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    node_id: crate::sim::NodeId,
    index: usize,
    duration_ns: u64,
    label: String,
    row_count: usize,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(2.0),
            ..default()
        })
        .with_children(|col| {
            spawn_slider_row(
                col,
                &format!("{} · {}", index, label),
                (duration_ns / NS_PER_MS) as f32,
                1.0,
                5000.0,
                "ms",
                SliderTarget::StepsWorkerMs { node: node_id, row: index },
                theme,
            );
            col.spawn(Node {
                width: Val::Percent(100.0),
                column_gap: Val::Px(3.0),
                justify_content: JustifyContent::FlexEnd,
                ..default()
            })
            .with_children(|actions| {
                spawn_row_action_buttons(actions, theme, node_id, index, row_count);
            });
        });
}

fn spawn_row_action_buttons(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    node_id: crate::sim::NodeId,
    index: usize,
    row_count: usize,
) {
    use crate::ui::icon_button;
    let can_up = index > 0;
    let can_down = index + 1 < row_count;
    parent
        .spawn(Node {
            column_gap: Val::Px(3.0),
            ..default()
        })
        .with_children(|b| {
            if can_up {
                icon_button(
                    b,
                    theme,
                    "↑",
                    StepRowMoveUpButton { node: node_id, row: index },
                );
            }
            if can_down {
                icon_button(
                    b,
                    theme,
                    "↓",
                    StepRowMoveDownButton { node: node_id, row: index },
                );
            }
            icon_button(
                b,
                theme,
                "×",
                StepRowDeleteButton { node: node_id, row: index },
            );
        });
}

fn spawn_worker_row_editor(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    node_id: crate::sim::NodeId,
    index: usize,
    body: &[crate::sim::Instruction],
    row_count: usize,
) {
    // A "Worker" sequence's body is `[Hold { duration_ns }]` — pull
    // the duration back out for the slider.
    let duration_ns = body
        .iter()
        .find_map(|i| match i {
            crate::sim::Instruction::Hold { duration_ns } => Some(*duration_ns),
            _ => None,
        })
        .unwrap_or(0);
    // Outer container for this row — vertical stack with slider on
    // top and action-button row below it.
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(2.0),
            ..default()
        })
        .with_children(|col| {
            spawn_slider_row(
                col,
                &format!("{} · Worker", index),
                (duration_ns / NS_PER_MS) as f32,
                1.0,
                5000.0,
                "ms",
                SliderTarget::StepsWorkerMs { node: node_id, row: index },
                theme,
            );
            col.spawn(Node {
                width: Val::Percent(100.0),
                column_gap: Val::Px(3.0),
                justify_content: JustifyContent::FlexEnd,
                ..default()
            })
            .with_children(|actions| {
                spawn_row_action_buttons(actions, theme, node_id, index, row_count);
            });
        });
}

#[derive(Component, Clone, Copy)]
struct StepRowMoveUpButton {
    node: crate::sim::NodeId,
    row: usize,
}
#[derive(Component, Clone, Copy)]
struct StepRowMoveDownButton {
    node: crate::sim::NodeId,
    row: usize,
}
#[derive(Component, Clone, Copy)]
struct StepRowDeleteButton {
    node: crate::sim::NodeId,
    row: usize,
}

/// "Save as preset" button — visible only when the user has
/// drilled into a Sequence. On press `handle_save_preset_button`
/// captures the current Sequence into `PresetLibrary.user`.
fn spawn_save_preset_button(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    node_id: crate::sim::NodeId,
    path: Vec<usize>,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Percent(100.0),
                margin: UiRect::vertical(Val::Px(6.0)),
                padding: UiRect::vertical(Val::Px(6.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::NONE),
            BorderColor::all(theme.ink),
            SavePresetButton { node: node_id, path },
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(caps_spaced("Save as preset")),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.ink),
                crate::bridge::Bold,
            ));
        });
}

#[derive(Component, Clone)]
struct SavePresetButton {
    node: crate::sim::NodeId,
    path: Vec<usize>,
}

fn handle_save_preset_button(
    q: Query<(&Interaction, &SavePresetButton), Changed<Interaction>>,
    sim: Res<SimResource>,
    mut library: ResMut<crate::ui::PresetLibrary>,
) {
    for (i, btn) in q.iter() {
        if *i != Interaction::Pressed {
            continue;
        }
        let Some(node) = sim.0.nodes.get(&btn.node) else { continue };
        let Some(instr) = crate::sim::instr_at(&node.program, &btn.path) else { continue };
        let (label, body) = match instr {
            crate::sim::Instruction::Sequence { label, body } => {
                (label.clone(), body.clone())
            }
            _ => continue,
        };
        // Dedupe by label: overwrite if one exists with the same
        // label rather than stacking duplicates.
        if let Some(existing) = library.user.iter_mut().find(|p| p.label == label) {
            existing.body = body;
        } else {
            library.user.push(crate::ui::Preset { label, body });
        }
    }
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
                .get_steps_row_duration_ns(node, row)
                .map(|ns| (ns / NS_PER_MS) as f32),
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
            state.dirty = true;
        }
    }
}

/// Click a breadcrumb segment to jump focus up the path. Clicking
/// the root segment (depth 0) when the path is already empty
/// closes the program view entirely — the inverse of double-click.
fn handle_breadcrumb_clicks(
    q: Query<(&Interaction, &BreadcrumbSegment), Changed<Interaction>>,
    mut focus: ResMut<InspectorFocus>,
    mut state: ResMut<InspectorState>,
) {
    for (i, seg) in q.iter() {
        if *i != Interaction::Pressed {
            continue;
        }
        if seg.target_depth < focus.path.len() {
            focus.path.truncate(seg.target_depth);
            state.dirty = true;
        } else if seg.target_depth == 0 && focus.path.is_empty() && focus.open {
            focus.open = false;
            state.dirty = true;
        }
    }
}

/// Escape pops one level: first unwinds through drilled-in
/// Sequence bodies, then (at depth 0) closes the program view
/// back to the kind's summary.
fn handle_focus_unwind(
    keys: Res<ButtonInput<KeyCode>>,
    edit: Res<crate::nodes::EditState>,
    mut focus: ResMut<InspectorFocus>,
    mut state: ResMut<InspectorState>,
) {
    if edit.is_editing() {
        return;
    }
    if !keys.just_pressed(KeyCode::Escape) {
        return;
    }
    if !focus.path.is_empty() {
        focus.path.pop();
        state.dirty = true;
    } else if focus.open {
        focus.open = false;
        state.dirty = true;
    }
}

/// Listen for a double-click on a canvas step-row entity. If the
/// clicked row is a `Sequence`, push its index onto `focus.path`
/// so the inspector re-renders at that deeper level.
fn handle_canvas_drill_click(
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    cams: Query<
        (&Camera, &GlobalTransform),
        With<crate::camera::MainCamera>,
    >,
    nodes_q: Query<
        (
            &Transform,
            &crate::nodes::SimNode,
            &crate::bridge::SimNodeRef,
        ),
    >,
    sim_res: Res<SimResource>,
    mut last_click: Local<Option<(f64, Vec2)>>,
    time: Res<Time>,
    mut focus: ResMut<InspectorFocus>,
    mut state: ResMut<InspectorState>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Some(cursor) = win.cursor_position() else { return };
    let Ok((cam, cam_tf)) = cams.single() else { return };
    let Ok(world) = cam.viewport_to_world_2d(cam_tf, cursor) else { return };

    let now = time.elapsed_secs_f64();
    let is_double = if let Some((t, p)) = *last_click {
        now - t < 0.4 && (p - world).length() < 8.0
    } else {
        false
    };
    *last_click = Some((now, world));
    if !is_double {
        return;
    }

    for (tf, sn, nref) in nodes_q.iter() {
        let center = tf.translation.truncate();
        let half = sn.size / 2.0;
        if (world.x - center.x).abs() > half.x || (world.y - center.y).abs() > half.y {
            continue;
        }
        // Any node double-click opens its program view. Additional
        // behavior for Steps containers: if the cursor was on a
        // specific row AND that row is a Sequence, also drill into
        // its body (extend the focus path).
        focus.open = true;
        if sn.kind == crate::nodes::NodeKind::Steps {
            let row_count = sim_res
                .0
                .nodes
                .get(&nref.0)
                .map(|n| n.program.len())
                .unwrap_or(0);
            if let Some(row) = crate::nodes::steps_row_at(world, center, row_count) {
                let is_sequence = sim_res
                    .0
                    .nodes
                    .get(&nref.0)
                    .and_then(|n| n.program.get(row))
                    .map(|i| matches!(i, crate::sim::Instruction::Sequence { .. }))
                    .unwrap_or(false);
                if is_sequence {
                    focus.path.push(row);
                }
            }
        }
        state.dirty = true;
        break;
    }
}

/// Handle clicks on per-step-row action buttons: ↑ / ↓ / ×. Busts
/// the inspector rebuild cache after any edit so the panel
/// regenerates with the new row list next frame. Edges anchored to
/// deleted rows are despawned here too (via the sim's returned
/// EdgeIds).
fn handle_step_row_buttons(
    mut commands: Commands,
    up_q: Query<(&Interaction, &StepRowMoveUpButton), Changed<Interaction>>,
    down_q: Query<(&Interaction, &StepRowMoveDownButton), Changed<Interaction>>,
    del_q: Query<(&Interaction, &StepRowDeleteButton), Changed<Interaction>>,
    mut sim_res: ResMut<SimResource>,
    mut maps: ResMut<crate::bridge::EntityMaps>,
    mut state: ResMut<InspectorState>,
) {
    let mut changed = false;
    for (i, btn) in up_q.iter() {
        if *i == Interaction::Pressed && btn.row > 0 {
            sim_res.0.swap_step_rows(btn.node, btn.row, btn.row - 1);
            changed = true;
        }
    }
    for (i, btn) in down_q.iter() {
        if *i == Interaction::Pressed {
            sim_res.0.swap_step_rows(btn.node, btn.row, btn.row + 1);
            changed = true;
        }
    }
    for (i, btn) in del_q.iter() {
        if *i == Interaction::Pressed {
            let removed = sim_res.0.remove_step_row(btn.node, btn.row);
            for eid in removed {
                if let Some(edge_entity) = maps.edge_to_entity.remove(&eid) {
                    commands.entity(edge_entity).despawn();
                    maps.entity_to_edge.remove(&edge_entity);
                }
            }
            changed = true;
        }
    }
    if changed {
        // Force the inspector to rebuild from the new row list.
        state.dirty = true;
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
