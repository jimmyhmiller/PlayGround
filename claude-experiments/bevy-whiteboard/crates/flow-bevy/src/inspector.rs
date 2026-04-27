//! Selection-driven inspector pinned below the palette body.
//!
//! When a node is selected on the canvas, we repopulate an `InspectorMount`
//! container with rows describing the node. For now, the useful controls
//! are:
//!
//!  * **Generator / Client**: rate (emissions per second) with `½×` / `2×`
//!    buttons that halve or double `period_ns`.
//!  * **Queue**: read-only fill + dropped counters.
//!  * **Sink**: read-only absorbed-count.
//!
//! We rebuild the inspector only when `Selection` changes structurally —
//! numbers within a selected node tick over via `LiveText`-style syncs each
//! frame. Structural rebuild despawns children; each syncing system queries
//! back against the current selection.

use bevy::prelude::*;
use flow::{NodeId, Value};
use poster_ui::{Bold, Mono, Slider, Theme, caps_spaced, spawn_slider};

use crate::errors::NodeErrorStats;
use crate::gadgets::Kind;
use crate::nodes::{NodeKind, Selection};
use crate::sim_driver::{NodeView, SimCommand, SimDriverRes, SimSnapshotRes};

/// Text labels on the up/down toggle. Small constants so the spawn
/// path, the click handler, and the live-sync system all agree.
const UP_LABEL: &str = "Up";
const DOWN_LABEL: &str = "Down";

pub struct InspectorPlugin;

impl Plugin for InspectorPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                rebuild_inspector_on_selection_change,
                push_rate_slider_to_sim,
                handle_up_toggle_clicks,
                sync_up_toggle_visual,
                handle_router_mode_clicks,
                sync_router_mode_visual,
                sync_queue_fill,
                sync_sink_count,
                sync_error_breakdown,
                // Order matters: push (slider→slot) must run before
                // sync (slot→slider) within a frame. If sync runs
                // first, it sees the stale slot value and overwrites
                // the user's in-progress drag back to that stale value
                // before push gets a chance to fire.
                (
                    push_generic_float_slider_to_sim,
                    sync_generic_float_slider_from_slot,
                ).chain(),
                sync_generic_readout,
                handle_generic_bool_toggle_clicks,
                sync_generic_bool_toggle_visual,
            ),
        );
    }
}

/// Marker on the container the palette spawns. The rebuild system despawns
/// this entity's children and re-spawns rows for the currently selected
/// node.
#[derive(Component)]
pub struct InspectorMount;

/// Marker on every inspector-spawned child so the rebuild pass can despawn
/// them selectively. We rebuild by despawning all descendants, so this
/// isn't strictly necessary, but keeping a marker makes diagnostics easier.
#[derive(Component)]
struct InspectorChild;

// ──────────────────────────────────────────────────────────────
// Markers
// ──────────────────────────────────────────────────────────────

/// Marker on a rate slider. The `kind` drives how the slider's [0, max]
/// value maps to a sim slot value — generators' sliders are in pkts/s and
/// write `period_ns = 1s / rate`, worker sliders are in milliseconds and
/// write `service_ns = ms * 1_000_000`. Tests can query `Query<(&Slider,
/// &RateSlider)>` to read the live value.
#[derive(Component, Debug, Clone, Copy)]
pub struct RateSlider {
    pub node: NodeId,
    pub kind: RateSliderKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateSliderKind {
    /// Generator / Client — value is emissions/sec; writes `period_ns`.
    EmitPerSecond,
    /// Worker — value is service time in ms; writes `service_ns`.
    ServiceMs,
}

/// Live queue fill count.
#[derive(Component)]
struct QueueFillReadout {
    node: NodeId,
}

/// Live sink absorbed count.
#[derive(Component)]
struct SinkCountReadout {
    node: NodeId,
}

/// Clickable toggle for a node's `up` slot. Flips 1 ↔ 0 on press; the
/// sync system paints the label + border to match the current state.
#[derive(Component, Debug, Clone, Copy)]
struct UpToggle {
    node: NodeId,
}

/// Clickable toggle for a Router's `mode` slot. Flips FanOut (0) ↔
/// RoundRobin (1) on press; the sync system paints the label.
#[derive(Component, Debug, Clone, Copy)]
struct RouterModeToggle {
    node: NodeId,
}

// ──────────────────────────────────────────────────────────────
// Rebuild
// ──────────────────────────────────────────────────────────────

fn rebuild_inspector_on_selection_change(
    selection: Res<Selection>,
    theme: Res<Theme>,
    snapshot: Res<SimSnapshotRes>,
    mount_q: Query<Entity, With<InspectorMount>>,
    kind_q: Query<(&NodeKind, &crate::bridge::FlowNodeRef)>,
    mut commands: Commands,
) {
    // Only rebuild when the selected entity actually changes. We still
    // rebuild on theme changes so freshly-spawned rows paint with the new
    // theme colours.
    if !(selection.is_changed() || theme.is_changed()) {
        return;
    }
    let Ok(mount) = mount_q.single() else { return };

    // Despawn any prior inspector rows.
    commands.entity(mount).despawn_related::<Children>();

    let Some(entity) = selection.entity else { return };
    let Ok((kind, node_ref)) = kind_q.get(entity) else { return };
    let nid = node_ref.0;
    let Some(node) = snapshot.0.nodes.get(&nid) else { return };

    commands.entity(mount).with_children(|body| {
        // Section heading with the node name.
        inspector_heading(body, &theme, &node.name);

        match kind.0 {
            Kind::Generator | Kind::Client | Kind::BackoffClient => {
                let rate = period_ns_to_rate(read_int_slot(node, "period_ns"));
                spawn_slider(
                    body, &theme,
                    "Rate",
                    /*min*/ 0.5, /*max*/ 20.0,
                    rate as f32, "/s",
                    RateSlider { node: nid, kind: RateSliderKind::EmitPerSecond },
                );
            }
            Kind::Worker => {
                let ms = (read_int_slot(node, "service_ns") / 1_000_000) as f32;
                spawn_slider(
                    body, &theme,
                    "Service",
                    /*min*/ 1.0, /*max*/ 2000.0,
                    ms, "ms",
                    RateSlider { node: nid, kind: RateSliderKind::ServiceMs },
                );
            }
            Kind::Queue => {
                queue_rows(body, &theme, nid);
            }
            Kind::Sink => {
                sink_row(body, &theme, nid);
            }
            Kind::Router => {
                let mode = read_int_slot(node, "mode");
                spawn_router_mode_toggle(body, &theme, nid, mode);
            }
        }

        // Every gadget carries an `up` slot; surface a toggle for it
        // after the kind-specific rows. The initial label matches the
        // slot's current state so the button doesn't flash on rebuild.
        if node.slots.contains_key("up") {
            let up = read_int_slot(node, "up") != 0;
            spawn_up_toggle(body, &theme, nid, up);
        }

        // Generic per-slot state section. Anything not already covered
        // by a bespoke widget above shows up here as a slider / toggle /
        // readout depending on type. Hidden if the node has no
        // generic-eligible slots.
        spawn_generic_state_section(body, &theme, nid, node);

        // Per-node error breakdown. Always spawned (even when the
        // node has no errors yet) so `sync_error_breakdown` can fill
        // it in live without triggering a structural rebuild that
        // would drop slider state mid-drag. Hidden when count == 0.
        spawn_error_breakdown(body, &theme, nid);
    });
}

/// Per-node error list. Single multi-line Text that the sync system
/// fills in each frame. Hidden entirely when the node has no errors.
#[derive(Component)]
struct ErrorBreakdown {
    node: NodeId,
}

fn spawn_error_breakdown(
    parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands,
    theme: &Theme,
    nid: NodeId,
) {
    parent
        .spawn((
            Node {
                width: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                margin: UiRect::top(Val::Px(8.0)),
                padding: UiRect::all(Val::Px(6.0)),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(4.0)),
                row_gap: Val::Px(2.0),
                ..default()
            },
            BackgroundColor(theme.paper_alt),
            BorderColor::all(theme.accent),
            Visibility::Hidden,
        ))
        .with_children(|section| {
            section.spawn((
                Text::new("Errors"),
                TextFont { font_size: 10.0, ..default() },
                TextColor(theme.accent),
                Mono,
                Bold,
            ));
            section.spawn((
                Text::new(""),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.ink),
                Mono,
                ErrorBreakdown { node: nid },
            ));
        });
}

/// Fill the selected node's error breakdown every frame. We can't
/// put this inside the rebuild pass because rebuild is gated on
/// `Selection` / `Theme` changes (by design — rebuilding on every
/// error tick would reset a slider mid-drag). Instead we look up
/// the breakdown entity by its marker and patch its text + the
/// container's visibility in-place.
fn sync_error_breakdown(
    stats: Res<NodeErrorStats>,
    mut breakdowns: Query<(&ErrorBreakdown, &mut Text, &ChildOf)>,
    mut containers: Query<&mut Visibility>,
) {
    for (bd, mut text, parent) in breakdowns.iter_mut() {
        let kinds = stats.per_node.get(&bd.node);
        let new_text = match kinds {
            Some(map) if !map.is_empty() => {
                let mut entries: Vec<(&String, u64)> =
                    map.iter().map(|(k, v)| (k, *v)).collect();
                entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
                let max_w = entries.iter().map(|(_, v)| v.to_string().len()).max().unwrap_or(1);
                entries
                    .iter()
                    .map(|(k, v)| format!("{:>w$}  {}", v, k, w = max_w))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            _ => String::new(),
        };
        if text.0 != new_text {
            text.0 = new_text.clone();
        }
        // Toggle the container's visibility — parent is the section
        // row, which is the `ChildOf` target here.
        if let Ok(mut vis) = containers.get_mut(parent.parent()) {
            let want = if new_text.is_empty() {
                Visibility::Hidden
            } else {
                Visibility::Inherited
            };
            if *vis != want {
                *vis = want;
            }
        }
    }
}

fn spawn_up_toggle(
    parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands,
    theme: &Theme,
    nid: NodeId,
    up: bool,
) {
    // Label + button row. The button takes the full remaining width so
    // it's an easy target; the label sits to its left matching the
    // existing slider rows' visual rhythm.
    parent
        .spawn((
            Node {
                width: Val::Percent(100.0),
                padding: UiRect::vertical(Val::Px(6.0)),
                column_gap: Val::Px(8.0),
                align_items: AlignItems::Center,
                ..default()
            },
            InspectorChild,
        ))
        .with_children(|row| {
            row.spawn((
                Text::new(caps_spaced("State")),
                TextFont { font_size: 9.0, ..default() },
                TextColor(theme.ink_soft),
                poster_ui::Bold,
            ));
            let (border, label) = up_toggle_visuals(theme, up);
            row.spawn((
                Button,
                Node {
                    flex_grow: 1.0,
                    padding: UiRect::vertical(Val::Px(6.0)),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(6.0)),
                    ..default()
                },
                BackgroundColor(Color::NONE),
                BorderColor::all(border),
                UpToggle { node: nid },
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new(caps_spaced(label)),
                    TextFont { font_size: 11.0, ..default() },
                    TextColor(border),
                    poster_ui::Bold,
                ));
            });
        });
}

/// (border/text colour, label) for the toggle in the given state.
/// `accent` for up — matches the "everything is wired" feel; `muted`
/// for down — a deliberate off/grey so a crashed node reads as inert
/// without shouting red at the user.
fn up_toggle_visuals(theme: &Theme, up: bool) -> (Color, &'static str) {
    if up {
        (theme.accent, UP_LABEL)
    } else {
        (theme.muted, DOWN_LABEL)
    }
}

fn spawn_router_mode_toggle(
    parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands,
    theme: &Theme,
    nid: NodeId,
    mode: i64,
) {
    parent
        .spawn((
            Node {
                width: Val::Percent(100.0),
                padding: UiRect::vertical(Val::Px(6.0)),
                column_gap: Val::Px(8.0),
                align_items: AlignItems::Center,
                ..default()
            },
            InspectorChild,
        ))
        .with_children(|row| {
            row.spawn((
                Text::new(caps_spaced("Mode")),
                TextFont { font_size: 9.0, ..default() },
                TextColor(theme.ink_soft),
                poster_ui::Bold,
            ));
            let (border, label) = router_mode_visuals(theme, mode);
            row.spawn((
                Button,
                Node {
                    flex_grow: 1.0,
                    padding: UiRect::vertical(Val::Px(6.0)),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(6.0)),
                    ..default()
                },
                BackgroundColor(Color::NONE),
                BorderColor::all(border),
                RouterModeToggle { node: nid },
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new(caps_spaced(label)),
                    TextFont { font_size: 11.0, ..default() },
                    TextColor(border),
                    poster_ui::Bold,
                ));
            });
        });
}

/// Router mode label/colour. Accent for RoundRobin (the "active load
/// balancing" feel); ink_soft for FanOut (the neutral broadcast
/// default). Avoid using `muted` here since that reads as "disabled"
/// — both modes are valid on-states.
fn router_mode_visuals(theme: &Theme, mode: i64) -> (Color, &'static str) {
    if mode == 1 {
        (theme.accent, "Round Robin")
    } else {
        (theme.ink_soft, "Fan Out")
    }
}

fn inspector_heading(parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands, theme: &Theme, name: &str) {
    parent
        .spawn((
            Node {
                width: Val::Percent(100.0),
                padding: UiRect { top: Val::Px(12.0), bottom: Val::Px(6.0), left: Val::Px(4.0), right: Val::Px(4.0) },
                margin: UiRect::bottom(Val::Px(6.0)),
                border: UiRect::bottom(Val::Px(1.0)),
                ..default()
            },
            BorderColor::all(theme.rule),
            InspectorChild,
        ))
        .with_children(|h| {
            h.spawn((
                Text::new(caps_spaced(name)),
                TextFont { font_size: 10.0, ..default() },
                TextColor(theme.ink_soft),
                poster_ui::Bold,
            ));
        });
}

/// Read an integer sim slot, returning 0 on missing/wrong-type (sliders
/// then sit at their min and the user can still drag them).
fn read_int_slot(node: &NodeView, slot: &str) -> i64 {
    node.slots
        .get(slot)
        .and_then(|v| match v { Value::Int(i) => Some(*i), _ => None })
        .unwrap_or(0)
}

fn period_ns_to_rate(period_ns: i64) -> f64 {
    if period_ns <= 0 { 0.0 } else { 1_000_000_000.0 / period_ns as f64 }
}

fn queue_rows(parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands, theme: &Theme, nid: NodeId) {
    parent
        .spawn((
            Node {
                width: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            InspectorChild,
        ))
        .with_children(|col| {
            simple_readout(col, theme, "Fill", "0", QueueFillReadout { node: nid });
        });
}

fn sink_row(parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands, theme: &Theme, nid: NodeId) {
    parent
        .spawn((
            Node {
                width: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            InspectorChild,
        ))
        .with_children(|col| {
            simple_readout(col, theme, "Absorbed", "0", SinkCountReadout { node: nid });
        });
}

fn simple_readout(
    parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands,
    theme: &Theme,
    label: &str,
    initial: &str,
    marker: impl Bundle,
) {
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
                poster_ui::Bold,
            ));
            r.spawn((
                Text::new(initial.to_string()),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.ink),
                poster_ui::Bold,
                poster_ui::Mono,
                marker,
            ));
        });
}

// ──────────────────────────────────────────────────────────────
// Live sync systems
// ──────────────────────────────────────────────────────────────

/// Translate the slider's live value back into the node's period / service
/// slot, every frame the user drags. The slider owns the authoritative
/// value — when it changes, the sim follows.
fn push_rate_slider_to_sim(
    q: Query<(&Slider, &RateSlider), Changed<Slider>>,
    snapshot: Res<SimSnapshotRes>,
    mut driver: ResMut<SimDriverRes>,
) {
    for (slider, rs) in q.iter() {
        if !snapshot.0.nodes.contains_key(&rs.node) { continue; }
        let nid = rs.node;
        match rs.kind {
            RateSliderKind::EmitPerSecond => {
                let rate = slider.value.max(0.01) as f64;
                let period_ns = (1_000_000_000.0 / rate) as i64;
                driver.0.send_command(SimCommand::new(move |sim| {
                    sim.user_edit_slot(nid, "period_ns", Value::Int(period_ns));
                }));
            }
            RateSliderKind::ServiceMs => {
                let ms = slider.value.max(0.5) as i64;
                driver.0.send_command(SimCommand::new(move |sim| {
                    sim.user_edit_slot(nid, "service_ns", Value::Int(ms * 1_000_000));
                }));
            }
        }
    }
}

/// Flip the node's `up` slot on every press. We gate on `Pressed`
/// (the same edge the palette buttons use) so hover / release don't
/// re-toggle the state.
///
/// On the 0→1 transition we inject a `resume(nil)` packet so stateful
/// gadgets can kick their loops back to life. Worker's pull-loop is
/// the motivating case: its `done` rule (which emits the next pull)
/// was consumed by `done_crashed` while down, so without a resume
/// kick the worker sits idle forever even though `up == 1`.
fn handle_up_toggle_clicks(
    q: Query<(&Interaction, &UpToggle), (Changed<Interaction>, With<Button>)>,
    snapshot: Res<SimSnapshotRes>,
    mut driver: ResMut<SimDriverRes>,
) {
    for (interaction, toggle) in q.iter() {
        if *interaction != Interaction::Pressed { continue; }
        let Some(node) = snapshot.0.nodes.get(&toggle.node) else { continue };
        let cur = node.slots.get("up")
            .and_then(|v| match v { Value::Int(i) => Some(*i), _ => None })
            .unwrap_or(1);
        let next = if cur == 0 { 1 } else { 0 };
        let nid = toggle.node;
        driver.0.send_command(SimCommand::new(move |sim| {
            sim.user_edit_slot(nid, "up", Value::Int(next));
            if cur == 0 && next == 1 {
                sim.inject(nid, Value::variant("resume", Value::Nil));
            }
        }));
    }
}

fn handle_router_mode_clicks(
    q: Query<(&Interaction, &RouterModeToggle), (Changed<Interaction>, With<Button>)>,
    snapshot: Res<SimSnapshotRes>,
    mut driver: ResMut<SimDriverRes>,
) {
    for (interaction, toggle) in q.iter() {
        if *interaction != Interaction::Pressed { continue; }
        let Some(node) = snapshot.0.nodes.get(&toggle.node) else { continue };
        let cur = node.slots.get("mode")
            .and_then(|v| match v { Value::Int(i) => Some(*i), _ => None })
            .unwrap_or(0);
        let next = if cur == 0 { 1 } else { 0 };
        let nid = toggle.node;
        driver.0.send_command(SimCommand::new(move |sim| {
            sim.user_edit_slot(nid, "mode", Value::Int(next));
        }));
    }
}

fn sync_router_mode_visual(
    theme: Res<Theme>,
    snapshot: Res<SimSnapshotRes>,
    mut buttons: Query<(&RouterModeToggle, &mut BorderColor, &Children)>,
    mut texts: Query<(&mut Text, &mut TextColor)>,
) {
    for (toggle, mut border, children) in buttons.iter_mut() {
        let mode = snapshot.0.nodes.get(&toggle.node)
            .and_then(|n| n.slots.get("mode"))
            .and_then(|v| match v { Value::Int(i) => Some(*i), _ => None })
            .unwrap_or(0);
        let (color, label) = router_mode_visuals(&theme, mode);
        *border = BorderColor::all(color);
        for child in children.iter() {
            if let Ok((mut text, mut tc)) = texts.get_mut(child) {
                let spaced = caps_spaced(label);
                if text.0 != spaced { text.0 = spaced; }
                *tc = TextColor(color);
            }
        }
    }
}

/// Repaint the toggle to match the slot every frame. Cheap — the
/// query is scoped to entities carrying `UpToggle` (at most one per
/// inspector rebuild). Also covers slot edits from any other source
/// (scripts, tests, future DSL hooks).
fn sync_up_toggle_visual(
    theme: Res<Theme>,
    snapshot: Res<SimSnapshotRes>,
    mut buttons: Query<(&UpToggle, &mut BorderColor, &Children)>,
    mut texts: Query<(&mut Text, &mut TextColor)>,
) {
    for (toggle, mut border, children) in buttons.iter_mut() {
        let up = snapshot.0.nodes.get(&toggle.node)
            .and_then(|n| n.slots.get("up"))
            .and_then(|v| match v { Value::Int(i) => Some(*i != 0), _ => None })
            .unwrap_or(true);
        let (color, label) = up_toggle_visuals(&theme, up);
        *border = BorderColor::all(color);
        // The button has exactly one text child (spawned above).
        for child in children.iter() {
            if let Ok((mut text, mut tc)) = texts.get_mut(child) {
                let spaced = caps_spaced(label);
                if text.0 != spaced { text.0 = spaced; }
                *tc = TextColor(color);
            }
        }
    }
}

fn sync_queue_fill(
    snapshot: Res<SimSnapshotRes>,
    mut q: Query<(&QueueFillReadout, &mut Text)>,
) {
    for (r, mut text) in q.iter_mut() {
        let len = snapshot.0
            .nodes
            .get(&r.node)
            .and_then(|n| n.slots.get("len"))
            .and_then(|v| match v { Value::Int(i) => Some(*i as usize), _ => None })
            .unwrap_or(0);
        let label = format!("{}", len);
        if text.0 != label { text.0 = label; }
    }
}

fn sync_sink_count(
    snapshot: Res<SimSnapshotRes>,
    mut q: Query<(&SinkCountReadout, &mut Text)>,
) {
    for (r, mut text) in q.iter_mut() {
        let count = snapshot.0
            .nodes
            .get(&r.node)
            .and_then(|n| n.slots.get("count"))
            .and_then(|v| match v { Value::Int(i) => Some(*i), _ => None })
            .unwrap_or(0);
        let label = format!("{}", count);
        if text.0 != label { text.0 = label; }
    }
}

// ──────────────────────────────────────────────────────────────
// Generic per-slot state section
// ──────────────────────────────────────────────────────────────
//
// Iterates every slot on the selected node and surfaces it as the
// best-fit widget for its type. Bespoke widgets above (Rate slider,
// Service slider, Up/Down toggle, Router mode) cover the curated
// cases — this fallback exposes everything else: gadget knobs like
// `hit_rate`, `fail_prob`, `vote_yes_prob`, `threshold`, etc.
//
// Layout discipline matches the bespoke rows: section heading first,
// then one row per slot. Hidden entirely when the node has nothing
// generic-eligible to show, so plain Generators / Sinks aren't
// cluttered with empty sections.

/// Slots already handled by curated widgets. Skipped in the generic
/// section to avoid double-rendering.
const HIDDEN_GENERIC_SLOTS: &[&str] = &["period_ns", "service_ns", "mode", "up"];

/// Slider that writes to a Float slot. The slider's value is f32 in
/// `[0, 1]` — mapped 1:1 to `Value::Float`. Picking a [0, 1] range
/// matches the dominant Float-slot use case (probabilities); raw Float
/// slots outside that range get clamped on edit, which is fine for
/// gadgets where `hit_rate` etc. are already bounded by their meaning.
#[derive(Component, Debug, Clone)]
pub struct GenericFloatSlider {
    pub node: NodeId,
    pub slot: String,
}

/// Live readout showing a slot's current scalar value. Used for Int /
/// String slots that aren't yet edit-capable.
#[derive(Component, Debug, Clone)]
struct GenericReadout {
    node: NodeId,
    slot: String,
}

/// Toggle for a Bool slot (or an Int slot used as 0/1). Click flips the
/// value; sync system paints to match the live state.
#[derive(Component, Debug, Clone)]
struct GenericBoolToggle {
    node: NodeId,
    slot: String,
}

fn spawn_generic_state_section(
    parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands,
    theme: &Theme,
    nid: NodeId,
    node: &NodeView,
) {
    // Collect the eligible slots up front so we can decide whether to
    // emit a section header at all. BTreeMap iteration is already
    // alphabetical, which gives stable visual order across rebuilds.
    let mut rows: Vec<(&String, &Value)> = Vec::new();
    for (name, value) in node.slots.iter() {
        if HIDDEN_GENERIC_SLOTS.contains(&name.as_str()) { continue; }
        if matches!(value, Value::Samples(_) | Value::Nil | Value::List(_) | Value::Variant { .. } | Value::Record(_) | Value::NodeRef(_)) {
            continue;
        }
        rows.push((name, value));
    }
    if rows.is_empty() { return; }

    parent
        .spawn((
            Node {
                width: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                margin: UiRect::top(Val::Px(10.0)),
                padding: UiRect::all(Val::Px(6.0)),
                row_gap: Val::Px(2.0),
                ..default()
            },
            BackgroundColor(theme.paper_alt),
            InspectorChild,
        ))
        .with_children(|section| {
            section.spawn((
                Text::new(caps_spaced("State")),
                TextFont { font_size: 9.0, ..default() },
                TextColor(theme.ink_soft),
                poster_ui::Bold,
            ));
            for (name, value) in rows {
                spawn_generic_slot_row(section, theme, nid, name, value);
            }
        });
}

fn spawn_generic_slot_row(
    parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands,
    theme: &Theme,
    nid: NodeId,
    name: &str,
    value: &Value,
) {
    match value {
        Value::Float(f) => {
            poster_ui::spawn_slider(
                parent, theme,
                name,
                /*min*/ 0.0, /*max*/ 1.0,
                (*f as f32).clamp(0.0, 1.0),
                "",
                GenericFloatSlider { node: nid, slot: name.to_string() },
            );
        }
        Value::Bool(b) => {
            spawn_generic_bool_toggle_row(parent, theme, nid, name, *b);
        }
        Value::Int(i) => {
            let label = if name.ends_with("_ns") {
                format!("{} ms", *i / 1_000_000)
            } else {
                format!("{}", i)
            };
            simple_readout(parent, theme, name, &label, GenericReadout { node: nid, slot: name.to_string() });
        }
        Value::Str(s) => {
            simple_readout(parent, theme, name, s, GenericReadout { node: nid, slot: name.to_string() });
        }
        _ => {}
    }
}

fn spawn_generic_bool_toggle_row(
    parent: &mut bevy::ecs::hierarchy::ChildSpawnerCommands,
    theme: &Theme,
    nid: NodeId,
    slot: &str,
    on: bool,
) {
    parent
        .spawn(Node {
            width: Val::Percent(100.0),
            padding: UiRect::vertical(Val::Px(5.0)),
            column_gap: Val::Px(8.0),
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(caps_spaced(slot)),
                TextFont { font_size: 9.0, ..default() },
                TextColor(theme.ink_soft),
                poster_ui::Bold,
            ));
            let (border, label) = bool_toggle_visuals(theme, on);
            row.spawn((
                Button,
                Node {
                    flex_grow: 1.0,
                    padding: UiRect::vertical(Val::Px(5.0)),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(6.0)),
                    ..default()
                },
                BackgroundColor(Color::NONE),
                BorderColor::all(border),
                GenericBoolToggle { node: nid, slot: slot.to_string() },
            ))
            .with_children(|b| {
                b.spawn((
                    Text::new(caps_spaced(label)),
                    TextFont { font_size: 11.0, ..default() },
                    TextColor(border),
                    poster_ui::Bold,
                ));
            });
        });
}

fn bool_toggle_visuals(theme: &Theme, on: bool) -> (Color, &'static str) {
    if on { (theme.accent, "On") } else { (theme.muted, "Off") }
}

/// Slider drag → slot write. Mirrors `push_rate_slider_to_sim` but
/// generic over slot name. Writes Float to whatever slot is named.
fn push_generic_float_slider_to_sim(
    q: Query<(&Slider, &GenericFloatSlider), Changed<Slider>>,
    snapshot: Res<SimSnapshotRes>,
    mut driver: ResMut<SimDriverRes>,
) {
    for (slider, marker) in q.iter() {
        let Some(node) = snapshot.0.nodes.get(&marker.node) else { continue };
        if !matches!(node.slots.get(&marker.slot), Some(Value::Float(_))) { continue; }
        let nid = marker.node;
        let slot = marker.slot.clone();
        let value = slider.value as f64;
        driver.0.send_command(SimCommand::new(move |sim| {
            sim.user_edit_slot(nid, slot, Value::Float(value));
        }));
    }
}

/// Slot write → slider value. Catches scenario / timeline / external
/// writes so the slider tracks the real state. Without this, an
/// external slot change wouldn't be reflected on the slider until the
/// inspector rebuilds (next selection change).
///
/// Skips when the slider's already at the right value to avoid
/// retriggering the `Changed<Slider>` filter and looping with the
/// push system above.
fn sync_generic_float_slider_from_slot(
    snapshot: Res<SimSnapshotRes>,
    mut q: Query<(&mut Slider, &GenericFloatSlider)>,
) {
    for (mut slider, marker) in q.iter_mut() {
        let Some(node) = snapshot.0.nodes.get(&marker.node) else { continue };
        let Some(Value::Float(f)) = node.slots.get(&marker.slot) else { continue };
        let want = (*f as f32).clamp(slider.min, slider.max);
        if (slider.value - want).abs() > 1e-6 {
            slider.value = want;
        }
    }
}

/// Live-tick readout text for Int/String slots.
fn sync_generic_readout(
    snapshot: Res<SimSnapshotRes>,
    mut q: Query<(&GenericReadout, &mut Text)>,
) {
    for (r, mut text) in q.iter_mut() {
        let Some(node) = snapshot.0.nodes.get(&r.node) else { continue };
        let Some(value) = node.slots.get(&r.slot) else { continue };
        let label = match value {
            Value::Int(i) => {
                if r.slot.ends_with("_ns") { format!("{} ms", *i / 1_000_000) }
                else { format!("{}", i) }
            }
            Value::Str(s) => s.clone(),
            _ => continue,
        };
        if text.0 != label { text.0 = label; }
    }
}

fn handle_generic_bool_toggle_clicks(
    q: Query<(&Interaction, &GenericBoolToggle), (Changed<Interaction>, With<Button>)>,
    snapshot: Res<SimSnapshotRes>,
    mut driver: ResMut<SimDriverRes>,
) {
    for (interaction, toggle) in q.iter() {
        if *interaction != Interaction::Pressed { continue; }
        let Some(node) = snapshot.0.nodes.get(&toggle.node) else { continue };
        let cur = match node.slots.get(&toggle.slot) {
            Some(Value::Bool(b)) => *b,
            _ => continue,
        };
        let nid = toggle.node;
        let slot = toggle.slot.clone();
        driver.0.send_command(SimCommand::new(move |sim| {
            sim.user_edit_slot(nid, slot, Value::Bool(!cur));
        }));
    }
}

fn sync_generic_bool_toggle_visual(
    theme: Res<Theme>,
    snapshot: Res<SimSnapshotRes>,
    mut buttons: Query<(&GenericBoolToggle, &mut BorderColor, &Children)>,
    mut texts: Query<(&mut Text, &mut TextColor)>,
) {
    for (toggle, mut border, children) in buttons.iter_mut() {
        let Some(node) = snapshot.0.nodes.get(&toggle.node) else { continue };
        let on = match node.slots.get(&toggle.slot) {
            Some(Value::Bool(b)) => *b,
            _ => continue,
        };
        let (color, label) = bool_toggle_visuals(&theme, on);
        *border = BorderColor::all(color);
        for child in children.iter() {
            if let Ok((mut text, mut tc)) = texts.get_mut(child) {
                let spaced = caps_spaced(label);
                if text.0 != spaced { text.0 = spaced; }
                *tc = TextColor(color);
            }
        }
    }
}

