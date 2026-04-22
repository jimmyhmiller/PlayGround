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
use poster_ui::{Slider, Theme, caps_spaced, spawn_slider};

use crate::bridge::FlowSim;
use crate::gadgets::Kind;
use crate::nodes::{NodeKind, Selection};

pub struct InspectorPlugin;

impl Plugin for InspectorPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                rebuild_inspector_on_selection_change,
                push_rate_slider_to_sim,
                sync_queue_fill,
                sync_sink_count,
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

// ──────────────────────────────────────────────────────────────
// Rebuild
// ──────────────────────────────────────────────────────────────

fn rebuild_inspector_on_selection_change(
    selection: Res<Selection>,
    theme: Res<Theme>,
    sim: Res<FlowSim>,
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
    let Some(node) = sim.sim.nodes.get(&nid) else { return };

    commands.entity(mount).with_children(|body| {
        // Section heading with the node name.
        inspector_heading(body, &theme, &node.name);

        match kind.0 {
            Kind::Generator | Kind::Client => {
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
                // Router has no rate — it forwards whatever arrives.
            }
        }
    });
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
fn read_int_slot(node: &flow::Node, slot: &str) -> i64 {
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
    mut sim: ResMut<FlowSim>,
) {
    for (slider, rs) in q.iter() {
        let Some(node) = sim.sim.nodes.get_mut(&rs.node) else { continue };
        match rs.kind {
            RateSliderKind::EmitPerSecond => {
                // rate ≤ 0 would divide by zero; clamp to a minimum so the
                // generator still ticks at something sensible.
                let rate = slider.value.max(0.01) as f64;
                let period_ns = (1_000_000_000.0 / rate) as i64;
                node.slots.insert("period_ns".into(), Value::Int(period_ns));
            }
            RateSliderKind::ServiceMs => {
                let ms = slider.value.max(0.5) as i64;
                node.slots.insert("service_ns".into(), Value::Int(ms * 1_000_000));
            }
        }
    }
}

fn sync_queue_fill(
    sim: Res<FlowSim>,
    mut q: Query<(&QueueFillReadout, &mut Text)>,
) {
    for (r, mut text) in q.iter_mut() {
        let len = sim
            .sim
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
    sim: Res<FlowSim>,
    mut q: Query<(&SinkCountReadout, &mut Text)>,
) {
    for (r, mut text) in q.iter_mut() {
        let count = sim
            .sim
            .nodes
            .get(&r.node)
            .and_then(|n| n.slots.get("count"))
            .and_then(|v| match v { Value::Int(i) => Some(*i), _ => None })
            .unwrap_or(0);
        let label = format!("{}", count);
        if text.0 != label { text.0 = label; }
    }
}

