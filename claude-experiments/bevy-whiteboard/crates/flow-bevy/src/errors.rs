//! Error surfacing — both the global and per-node views.
//!
//! The sim records `Event::RuntimeError { kind, node, ... }` for every
//! recoverable failure (busy rejection, color mismatch, node down,
//! etc). That raw stream is consumed here into two indices:
//!
//!   * `NodeErrorStats.by_kind` — global count per kind. Fed to a
//!     compact panel above the HUD so the user can see at a glance
//!     which classes of failure are happening and how often.
//!   * `NodeErrorStats.per_node[_total]` — per-node, per-kind and
//!     total counts. Fed to a small red badge floating above each
//!     node (count only) and to the inspector (full per-kind list
//!     when a node is selected).
//!
//! Stats reset on `LoadExample` because each scenario wipes the sim
//! and re-creates nodes; carrying counts across would double-count
//! when the same node id is re-used.

use std::collections::{BTreeMap, HashMap};

use bevy::prelude::*;
use bevy::sprite::Anchor;
use flow::{Event, NodeId};
use poster_ui::{Bold, Mono, Theme};

use crate::bridge::{FlowNodeRef, NewEvents};
use crate::examples::LoadExample;
use crate::nodes::NodeKind;

pub struct ErrorsPlugin;
impl Plugin for ErrorsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NodeErrorStats>()
            .add_systems(Startup, spawn_global_panel)
            .add_systems(
                Update,
                (
                    reset_on_load_example,
                    ingest_errors,
                    sync_global_panel,
                    spawn_node_badges,
                    sync_node_badges,
                )
                    .chain(),
            );
    }
}

// ───────────────────────────────────────────────────────────────
// Stats resource
// ───────────────────────────────────────────────────────────────

#[derive(Resource, Default)]
pub struct NodeErrorStats {
    /// Cumulative count per error kind across the whole sim.
    pub by_kind: BTreeMap<String, u64>,
    /// Per-node, per-kind counts. Empty map means no errors have hit
    /// that node yet.
    pub per_node: HashMap<NodeId, BTreeMap<String, u64>>,
    /// Per-node sum across kinds. Duplicates `per_node` roll-up so the
    /// badge sync doesn't have to re-sum every frame.
    pub per_node_total: HashMap<NodeId, u64>,
}

impl NodeErrorStats {
    fn clear(&mut self) {
        self.by_kind.clear();
        self.per_node.clear();
        self.per_node_total.clear();
    }
}

fn ingest_errors(new_events: Res<NewEvents>, mut stats: ResMut<NodeErrorStats>) {
    for ev in &new_events.0 {
        let Event::RuntimeError { kind, node, .. } = ev else { continue };
        *stats.by_kind.entry(kind.clone()).or_insert(0) += 1;
        if let Some(nid) = node {
            *stats
                .per_node
                .entry(*nid)
                .or_default()
                .entry(kind.clone())
                .or_insert(0) += 1;
            *stats.per_node_total.entry(*nid).or_insert(0) += 1;
        }
    }
}

fn reset_on_load_example(
    mut events: MessageReader<LoadExample>,
    mut stats: ResMut<NodeErrorStats>,
) {
    if events.read().next().is_some() {
        // Drain the rest too so a later reader doesn't double-reset.
        for _ in events.read() {}
        stats.clear();
    }
}

// ───────────────────────────────────────────────────────────────
// Global panel — floats just above the HUD, lists kinds by count
// ───────────────────────────────────────────────────────────────

#[derive(Component)]
struct GlobalErrorPanel;

#[derive(Component)]
struct GlobalErrorText;

fn spawn_global_panel(mut commands: Commands, theme: Res<Theme>) {
    // Anchor to the bottom-left stack above the HUD (HUD itself is at
    // bottom 20 with ~40px height; start this panel a little higher
    // so there's visible air between them).
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(20.0),
                bottom: Val::Px(72.0),
                padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                border: UiRect::all(Val::Px(1.5)),
                border_radius: BorderRadius::all(Val::Px(8.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(2.0),
                ..default()
            },
            BackgroundColor(theme.paper_alt),
            BorderColor::all(theme.accent),
            Visibility::Hidden,
            GlobalErrorPanel,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new(""),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.accent),
                Mono,
                Bold,
                GlobalErrorText,
            ));
        });
}

fn sync_global_panel(
    stats: Res<NodeErrorStats>,
    mut panel_q: Query<&mut Visibility, (With<GlobalErrorPanel>, Without<GlobalErrorText>)>,
    mut text_q: Query<&mut Text, With<GlobalErrorText>>,
) {
    if !stats.is_changed() {
        return;
    }
    let Ok(mut vis) = panel_q.single_mut() else { return };
    let Ok(mut text) = text_q.single_mut() else { return };

    if stats.by_kind.is_empty() {
        if *vis != Visibility::Hidden {
            *vis = Visibility::Hidden;
        }
        return;
    }
    if *vis != Visibility::Visible {
        *vis = Visibility::Visible;
    }

    // Sort by count desc, kind asc for stable tie-break.
    let mut entries: Vec<(&String, u64)> =
        stats.by_kind.iter().map(|(k, v)| (k, *v)).collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    // One line per kind: "<count>  <kind>", padded so the counts
    // line up in a mono font.
    let max_count_width = entries
        .iter()
        .map(|(_, v)| v.to_string().len())
        .max()
        .unwrap_or(1);
    let formatted = entries
        .iter()
        .map(|(k, v)| format!("{:>w$}  {}", v, k, w = max_count_width))
        .collect::<Vec<_>>()
        .join("\n");
    if text.0 != formatted {
        text.0 = formatted;
    }
}

// ───────────────────────────────────────────────────────────────
// Per-node badge — small red "!N" above each erroring node
// ───────────────────────────────────────────────────────────────

#[derive(Component)]
struct NodeErrorBadge {
    node: NodeId,
    /// Last rendered count; avoids re-writing the `Text2d` every frame.
    last_count: u64,
}

/// One badge per node, spawned as a child so it follows the node's
/// transform automatically. Pinned to the top-right corner — the
/// vertical "above the node" slot is already taken by the
/// always-on state label (rate / service / fill) and by the probe
/// stack. A corner badge sits out of their way and reads as a
/// conventional notification marker.
fn spawn_node_badges(
    mut commands: Commands,
    theme: Res<Theme>,
    newly_added: Query<(Entity, &FlowNodeRef, &NodeKind), Added<FlowNodeRef>>,
) {
    use crate::nodes::{body_shape, hit_size};
    for (entity, fid, kind) in &newly_added {
        let hsize = hit_size(&body_shape(kind.0));
        // Slightly above & outside the top-right corner so the text
        // reads clearly against the canvas rather than the node fill.
        let x = hsize.x * 0.5 + 4.0;
        let y = hsize.y * 0.5 + 4.0;
        commands.entity(entity).with_children(|p| {
            p.spawn((
                Text2d::new(""),
                TextFont { font_size: 11.0, ..default() },
                TextColor(theme.accent),
                Bold,
                Mono,
                Anchor::BOTTOM_LEFT,
                Transform::from_xyz(x, y, 5.0),
                Visibility::Hidden,
                NodeErrorBadge { node: fid.0, last_count: 0 },
            ));
        });
    }
}

fn sync_node_badges(
    stats: Res<NodeErrorStats>,
    mut badges: Query<(&mut NodeErrorBadge, &mut Text2d, &mut Visibility)>,
) {
    if !stats.is_changed() {
        return;
    }
    for (mut badge, mut text, mut vis) in badges.iter_mut() {
        let count = stats.per_node_total.get(&badge.node).copied().unwrap_or(0);
        if count == badge.last_count {
            continue;
        }
        badge.last_count = count;
        if count == 0 {
            if *vis != Visibility::Hidden {
                *vis = Visibility::Hidden;
            }
        } else {
            if *vis != Visibility::Visible {
                *vis = Visibility::Visible;
            }
            text.0 = format!("!{}", count);
        }
    }
}
