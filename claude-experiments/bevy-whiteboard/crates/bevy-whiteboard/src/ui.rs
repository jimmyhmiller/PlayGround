//! Shared UI primitives. Two conventions live here:
//!
//! 1. **Live value bindings** — widgets declare what sim value they're
//!    showing via a [`LiveText`] marker. A per-frame system resolves
//!    the binding against `SimResource` and writes the string into
//!    either a `Text` (panel) or `Text2d` (canvas) component. Structure
//!    rebuilds only on topology changes (new row, selection change);
//!    numbers sync cheaply every frame.
//!
//! 2. **Row primitive** — one builder that lays out `label | control |
//!    actions` rows with consistent spacing and styling. Inspector kv
//!    rows, slider rows, and color pickers all come from this (Stage
//!    A4's reorder/delete handles will hang off the `actions` slot).

use crate::bridge::{Bold, SimResource};
use crate::palette::caps_spaced;
use crate::sim::{NS_PER_MS, NS_PER_S, NS_PER_US, NodeId};
use crate::theme::Theme;
use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::prelude::*;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PresetLibrary>()
            .add_systems(
                Update,
                (
                    sync_live_text_panel,
                    sync_live_text_canvas,
                    scroll_panes_on_wheel,
                ),
            );
    }
}

/// A reusable step: an `Instruction::Sequence` captured with a
/// name the user chose. Saved from the inspector's "Save as
/// preset" action when drilled into a Sequence.
#[derive(Clone)]
pub struct Preset {
    pub label: String,
    pub body: Vec<crate::sim::Instruction>,
}

/// User-defined presets that appear as extra palette buttons.
/// Built-in kinds (Client, Worker) are separate — those stay
/// hard-coded as `Tool::Client` / `Tool::Worker` so they can
/// parameterise over active-color at placement time. User presets
/// are literal Instructions with their captured config.
#[derive(Resource, Default)]
pub struct PresetLibrary {
    pub user: Vec<Preset>,
}

/// Marker on a UI node that should vertically scroll on mousewheel.
/// Apply alongside `Interaction::None` and `ScrollPosition::default()`
/// on a node with `Overflow::scroll_y()` for a complete scroll setup.
#[derive(Component)]
pub struct ScrollPane;

/// Route mousewheel events to whichever `ScrollPane` the cursor is
/// physically over. We don't use `Interaction` here: it only flags
/// the topmost hit, so hovering a button or row *inside* the pane
/// would leave the pane itself `Interaction::None` and the scroll
/// would be lost. Cursor-in-bounds is consistent.
fn scroll_panes_on_wheel(
    mut wheel: MessageReader<bevy::input::mouse::MouseWheel>,
    windows: Query<&Window>,
    mut panes: Query<
        (
            &bevy::ui::ComputedNode,
            &bevy::ui::UiGlobalTransform,
            &mut ScrollPosition,
        ),
        With<ScrollPane>,
    >,
) {
    let mut dy = 0.0f32;
    for ev in wheel.read() {
        // Lines vs. pixels: normalise to a consistent "logical px".
        // One wheel-line ≈ 20 logical pixels — feels right on both
        // trackpads and mice in practice.
        let scale = match ev.unit {
            bevy::input::mouse::MouseScrollUnit::Line => 20.0,
            bevy::input::mouse::MouseScrollUnit::Pixel => 1.0,
        };
        dy -= ev.y * scale;
    }
    if dy.abs() < 1e-3 {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Some(cursor_logical) = win.cursor_position() else { return };
    let scale_f = win.scale_factor();
    // ComputedNode / UiGlobalTransform report physical pixels; the
    // window cursor is logical. Convert to a common space before
    // bounding-box testing.
    let cursor_px = cursor_logical * scale_f;
    for (computed, xform, mut pos) in panes.iter_mut() {
        let half = computed.size * 0.5;
        let center = xform.translation;
        let min = center - half;
        let max = center + half;
        if cursor_px.x >= min.x && cursor_px.x <= max.x
            && cursor_px.y >= min.y && cursor_px.y <= max.y
        {
            pos.y = (pos.y + dy).max(0.0);
            return;
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Live value bindings
// ──────────────────────────────────────────────────────────────────

/// Marks a text widget as bound to a sim value. Every frame, the
/// sync systems resolve the binding and write the current value into
/// the widget's string. Strip + skip on missing-node so a deleted
/// node's widgets go blank gracefully until they themselves are
/// despawned by the owning panel's structural rebuild.
#[derive(Component, Clone)]
pub struct LiveText {
    pub node: NodeId,
    pub field: LiveField,
}

/// Which field of a [`crate::sim::Node`] to read. Add variants as
/// new live readouts are needed; keep the arm order of the match in
/// `resolve` mirroring this enum so it's obvious what's covered.
#[derive(Clone, Copy)]
pub enum LiveField {
    /// Short label for step row `i` (e.g. `"Worker 500ms"`).
    StepRowLabel(usize),
    /// Number of requests this component has sent (Clients + Steps).
    Sent,
    /// Responses this component has received.
    Received,
    /// Packets consumed (sinks).
    SinkTotal,
    /// Packets processed (workers).
    Processed,
    /// Current size of the internal buffer (queues).
    BufferLen,
    /// Emitted packet count (generators).
    Emitted,
    /// Dropped packet count (any node with drops).
    Dropped,
    /// Index of the currently-executing step row, or `-` if dormant.
    CurrentRow,
}

fn resolve_live(sim: &SimResource, lt: &LiveText) -> String {
    let Some(node) = sim.0.nodes.get(&lt.node) else {
        return String::new();
    };
    match lt.field {
        LiveField::StepRowLabel(i) => node
            .program
            .get(i)
            .map(crate::nodes::format_step_row)
            .unwrap_or_default(),
        LiveField::Sent => node.sent.to_string(),
        LiveField::Received => node.received.to_string(),
        LiveField::SinkTotal => node.sink_total.to_string(),
        LiveField::Processed => node.processed.to_string(),
        LiveField::BufferLen => node.buffer.len().to_string(),
        LiveField::Emitted => node.emitted.to_string(),
        LiveField::Dropped => node.dropped.to_string(),
        LiveField::CurrentRow => node
            .cursor
            .as_ref()
            .and_then(|p| p.first())
            .map(|i| i.to_string())
            .unwrap_or_else(|| "-".into()),
    }
}

fn sync_live_text_panel(
    sim: Res<SimResource>,
    mut q: Query<(&LiveText, &mut Text)>,
) {
    for (lt, mut text) in q.iter_mut() {
        let new = resolve_live(&sim, lt);
        if text.0 != new {
            text.0 = new;
        }
    }
}

fn sync_live_text_canvas(
    sim: Res<SimResource>,
    mut q: Query<(&LiveText, &mut Text2d)>,
) {
    for (lt, mut text) in q.iter_mut() {
        let new = resolve_live(&sim, lt);
        if text.0 != new {
            text.0 = new;
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Row primitive
// ──────────────────────────────────────────────────────────────────

/// What goes in the "control" slot of a [`row`]. The builder owns the
/// visual layout; callers pick the kind of control and its params.
/// Extend with new variants as needed rather than growing the builder
/// argument list.
pub enum RowControl<'a> {
    /// Just a value readout. The string may be static or — more
    /// commonly — empty with a [`LiveText`] binding that a sync
    /// system will fill in every frame.
    Readout {
        text: &'a str,
        live: Option<LiveText>,
    },
}

/// Spawn one row (`label · control`) into `parent`. Consistent
/// padding, font sizing, and theme colours for every kind of row.
pub fn row(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    label: &str,
    control: RowControl<'_>,
) {
    row_with_actions(parent, theme, label, control, |_| {});
}

/// Same as [`row`] but lets the caller spawn trailing action widgets
/// (move-up, delete ×, etc.) into the row after the control. The
/// closure receives a ChildSpawnerCommands for the row itself, so it
/// can append any number of action entities with whatever custom
/// marker components the caller needs for interaction handling.
pub fn row_with_actions<F>(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    label: &str,
    control: RowControl<'_>,
    actions: F,
) where
    F: FnOnce(&mut ChildSpawnerCommands),
{
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            padding: UiRect::vertical(Val::Px(5.0)),
            column_gap: Val::Px(6.0),
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
            match control {
                RowControl::Readout { text, live } => {
                    let mut e = r.spawn((
                        Text::new(text.to_string()),
                        TextFont { font_size: 11.0, ..default() },
                        TextColor(theme.ink),
                        Bold,
                        // Live values almost always contain digits;
                        // monospace keeps their width stable as the
                        // numbers tick over.
                        crate::bridge::Mono,
                    ));
                    if let Some(binding) = live {
                        e.insert(binding);
                    }
                }
            }
            actions(r);
        });
}

/// Small square-ish icon button used for row actions. Caller passes
/// a glyph and any marker component(s) needed to route the
/// interaction. Styled subtly so a row of them reads as actions, not
/// primary CTAs.
pub fn icon_button(
    parent: &mut ChildSpawnerCommands,
    theme: &Theme,
    glyph: &str,
    extra: impl Bundle,
) {
    parent
        .spawn((
            Button,
            Node {
                width: Val::Px(20.0),
                height: Val::Px(20.0),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                ..default()
            },
            BackgroundColor(Color::NONE),
            BorderColor::all(theme.rule),
            extra,
        ))
        .with_children(|b| {
            b.spawn((
                Text::new(glyph.to_string()),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.ink),
                Bold,
            ));
        });
}

/// Format a duration short-label (ms-centric). Shared across row
/// controls and canvas labels so "Worker 500ms" reads the same in
/// both places.
pub fn fmt_duration(ns: u64) -> String {
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
