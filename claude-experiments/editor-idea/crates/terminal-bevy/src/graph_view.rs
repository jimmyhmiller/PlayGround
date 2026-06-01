//! Generic node/edge graph component for panes.
//!
//! Domain-agnostic: it knows about nodes (with a layer/slot position and
//! a status), edges between them, and how to lay them out and render
//! them as Bevy sprites + text under a pane's `content_root`. It does
//! NOT know about workflows, agents, or files. Adapters (see
//! `workflow_graph.rs`) build a [`GraphModel`] and hand it here.
//!
//! Layout is a layered top-to-bottom flow: `layer` (field `col`) is the
//! flow depth, drawn as a horizontal band stacked downward; `slot`
//! (field `row`) is the position within that band, spread left-to-right
//! and centered. Vertical flow keeps the graph's width bounded by the
//! widest parallel batch rather than by the number of stages, which is
//! the right shape for pipelines that fan out into parallel substeps.
//! Edges run from the bottom-center of a source node to the top-center
//! of its target.
//!
//! The whole graph is uniformly scaled to fit the pane's content area
//! (down to a readability floor), so small graphs render full-size and
//! deep ones shrink to stay fully visible. Rendering is rebuild-on-
//! change; node/edge counts are small (tens), so this is cheap.

use std::collections::HashMap;

use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::{LineHeight, TextBounds};

// ---------- Model ----------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeStatus {
    Pending,
    Running,
    Done,
    Failed,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GNode {
    /// Stable identity (used only for edge wiring; not displayed).
    pub id: String,
    pub label: String,
    pub sublabel: String,
    pub status: NodeStatus,
    /// Flow depth / layer, drawn top-to-bottom (kept named `col` so
    /// adapters that think in "wave index" map straight onto it).
    pub col: u32,
    /// Slot within the layer, spread left-to-right.
    pub row: u32,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GEdge {
    pub from: String,
    pub to: String,
}

#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct GraphModel {
    pub nodes: Vec<GNode>,
    pub edges: Vec<GEdge>,
    /// Optional one-line caption shown at the top of the content area.
    pub caption: String,
}

// ---------- Palette ----------

#[derive(Clone)]
pub struct GraphPalette {
    pub node_bg: Color,
    pub label: Color,
    pub sublabel: Color,
    pub edge: Color,
    pub caption: Color,
    pub pending: Color,
    pub running: Color,
    pub done: Color,
    pub failed: Color,
}

impl GraphPalette {
    fn status_color(&self, s: NodeStatus) -> Color {
        match s {
            NodeStatus::Pending => self.pending,
            NodeStatus::Running => self.running,
            NodeStatus::Done => self.done,
            NodeStatus::Failed => self.failed,
        }
    }
}

// ---------- Render state ----------

/// Tracks the entities spawned for the current model so the next render
/// can despawn them before rebuilding. Lives on the pane entity.
#[derive(Component, Default)]
pub struct GraphView {
    spawned: Vec<Entity>,
    drawn: Option<GraphModel>,
}

impl GraphView {
    pub fn needs_render(&self, model: &GraphModel) -> bool {
        self.drawn.as_ref() != Some(model)
    }
}

// ---------- Layout constants (unscaled) ----------

const NODE_W: f32 = 178.0;
const NODE_H: f32 = 52.0;
const SLOT_GAP: f32 = 34.0;
const LAYER_GAP: f32 = 30.0;
const PAD: f32 = 16.0;
const CAPTION_H: f32 = 24.0;
const STRIPE_H: f32 = 4.0;
const TEXT_PAD: f32 = 9.0;
const EDGE_THICKNESS: f32 = 2.0;
const LABEL_SIZE: f32 = 13.0;
const SUBLABEL_SIZE: f32 = 10.5;
const CAPTION_SIZE: f32 = 12.0;
/// Don't shrink below this or text becomes unreadable; past here the
/// graph is allowed to overflow (and clip) instead.
const MIN_SCALE: f32 = 0.5;

/// Per-layer slot counts, indexed by layer.
fn layer_slot_counts(nodes: &[GNode]) -> Vec<u32> {
    let max_layer = nodes.iter().map(|n| n.col).max().unwrap_or(0);
    let mut counts = vec![0u32; (max_layer + 1) as usize];
    for n in nodes {
        let slot = &mut counts[n.col as usize];
        *slot = (*slot).max(n.row + 1);
    }
    counts
}

/// Unscaled top-left of a node, in content "screen" coords (x right, y
/// down), with each layer centered horizontally against the widest one.
fn node_top_left(node: &GNode, slot_counts: &[u32], widest_w: f32) -> Vec2 {
    let slots = slot_counts.get(node.col as usize).copied().unwrap_or(1).max(1);
    let layer_w = slots as f32 * NODE_W + (slots.saturating_sub(1)) as f32 * SLOT_GAP;
    let x0 = (widest_w - layer_w) * 0.5;
    let x = x0 + node.row as f32 * (NODE_W + SLOT_GAP);
    let y = CAPTION_H + node.col as f32 * (NODE_H + LAYER_GAP);
    Vec2::new(x, y)
}

// ---------- Rendering ----------

/// Despawn the previous render and rebuild from `model`. Gate on
/// [`GraphView::needs_render`] to avoid pointless rebuilds.
pub fn render(
    commands: &mut Commands,
    content_root: Entity,
    font: &Handle<Font>,
    palette: &GraphPalette,
    content_size: Vec2,
    model: &GraphModel,
    view: &mut GraphView,
) {
    for e in view.spawned.drain(..) {
        if let Ok(mut ec) = commands.get_entity(e) {
            ec.despawn();
        }
    }

    let slot_counts = layer_slot_counts(&model.nodes);
    let n_layers = slot_counts.len().max(1) as f32;
    let widest_slots = slot_counts.iter().copied().max().unwrap_or(1).max(1) as f32;

    // Unscaled graph extent.
    let raw_w = widest_slots * NODE_W + (widest_slots - 1.0) * SLOT_GAP;
    let raw_h = CAPTION_H + n_layers * NODE_H + (n_layers - 1.0) * LAYER_GAP;

    // Fit uniformly into the content area, floored for readability.
    let avail_w = (content_size.x - 2.0 * PAD).max(1.0);
    let avail_h = (content_size.y - 2.0 * PAD).max(1.0);
    let scale = (avail_w / raw_w)
        .min(avail_h / raw_h)
        .min(1.0)
        .max(MIN_SCALE);

    // Center the scaled graph within the content area.
    let scaled_w = raw_w * scale;
    let scaled_h = raw_h * scale;
    let origin = Vec2::new(
        PAD + ((avail_w - scaled_w) * 0.5).max(0.0),
        PAD + ((avail_h - scaled_h) * 0.5).max(0.0),
    );

    // Content "screen" point (y down, unscaled) → Transform translation
    // (y up) relative to content_root, applying scale + centering.
    let place = |p: Vec2, z: f32| {
        let x = origin.x + p.x * scale;
        let y = origin.y + p.y * scale;
        Vec3::new(x, -y, z)
    };

    let nw = NODE_W * scale;
    let nh = NODE_H * scale;
    let label_size = (LABEL_SIZE * scale).max(7.0);
    let sub_size = (SUBLABEL_SIZE * scale).max(6.0);

    let mut out: Vec<Entity> = Vec::new();

    // Caption.
    if !model.caption.is_empty() {
        out.push(
            commands
                .spawn((
                    ChildOf(content_root),
                    Text2d::new(model.caption.clone()),
                    TextFont {
                        font: font.clone(),
                        font_size: CAPTION_SIZE,
                        ..default()
                    },
                    LineHeight::Px(CAPTION_SIZE * 1.4),
                    TextColor(palette.caption),
                    Anchor::TOP_LEFT,
                    bevy::text::TextLayout::new_with_no_wrap(),
                    Transform::from_translation(Vec3::new(PAD, -PAD, 0.3)),
                ))
                .id(),
        );
    }

    // Node anchor points for edge routing (bottom-center / top-center).
    let mut top_mid: HashMap<&str, Vec2> = HashMap::new();
    let mut bot_mid: HashMap<&str, Vec2> = HashMap::new();
    for n in &model.nodes {
        let tl = node_top_left(n, &slot_counts, raw_w);
        top_mid.insert(n.id.as_str(), Vec2::new(tl.x + NODE_W * 0.5, tl.y));
        bot_mid.insert(n.id.as_str(), Vec2::new(tl.x + NODE_W * 0.5, tl.y + NODE_H));
    }

    // Edges (under nodes).
    for edge in &model.edges {
        let (Some(from), Some(to)) =
            (bot_mid.get(edge.from.as_str()), top_mid.get(edge.to.as_str()))
        else {
            continue;
        };
        let a = place(*from, 0.05).truncate();
        let b = place(*to, 0.05).truncate();
        let delta = b - a;
        let len = delta.length();
        if len < 1.0 {
            continue;
        }
        let angle = delta.y.atan2(delta.x);
        out.push(
            commands
                .spawn((
                    ChildOf(content_root),
                    Sprite {
                        color: palette.edge,
                        custom_size: Some(Vec2::new(len, EDGE_THICKNESS)),
                        ..default()
                    },
                    Anchor::CENTER_LEFT,
                    Transform::from_translation(Vec3::new(a.x, a.y, 0.05))
                        .with_rotation(Quat::from_rotation_z(angle)),
                ))
                .id(),
        );
    }

    // Nodes.
    for n in &model.nodes {
        let tl = node_top_left(n, &slot_counts, raw_w);
        let status = palette.status_color(n.status);

        // Card background.
        out.push(
            commands
                .spawn((
                    ChildOf(content_root),
                    Sprite {
                        color: palette.node_bg,
                        custom_size: Some(Vec2::new(nw, nh)),
                        ..default()
                    },
                    Anchor::TOP_LEFT,
                    Transform::from_translation(place(tl, 0.1)),
                ))
                .id(),
        );

        // Status stripe across the top edge.
        out.push(
            commands
                .spawn((
                    ChildOf(content_root),
                    Sprite {
                        color: status,
                        custom_size: Some(Vec2::new(nw, STRIPE_H * scale)),
                        ..default()
                    },
                    Anchor::TOP_LEFT,
                    Transform::from_translation(place(tl, 0.15)),
                ))
                .id(),
        );

        let inner_w = (nw - 2.0 * TEXT_PAD * scale).max(16.0);
        let text_x = tl.x + TEXT_PAD;

        // Title.
        out.push(
            commands
                .spawn((
                    ChildOf(content_root),
                    Text2d::new(n.label.clone()),
                    TextFont {
                        font: font.clone(),
                        font_size: label_size,
                        ..default()
                    },
                    LineHeight::Px(label_size * 1.3),
                    TextColor(palette.label),
                    Anchor::TOP_LEFT,
                    bevy::text::TextLayout::new_with_no_wrap(),
                    TextBounds {
                        width: Some(inner_w),
                        height: Some(label_size * 1.5),
                    },
                    Transform::from_translation(place(
                        Vec2::new(text_x, tl.y + STRIPE_H + 6.0),
                        0.2,
                    )),
                ))
                .id(),
        );

        // Sublabel.
        out.push(
            commands
                .spawn((
                    ChildOf(content_root),
                    Text2d::new(n.sublabel.clone()),
                    TextFont {
                        font: font.clone(),
                        font_size: sub_size,
                        ..default()
                    },
                    LineHeight::Px(sub_size * 1.3),
                    TextColor(palette.sublabel),
                    Anchor::TOP_LEFT,
                    bevy::text::TextLayout::new_with_no_wrap(),
                    TextBounds {
                        width: Some(inner_w),
                        height: Some(sub_size * 1.5),
                    },
                    Transform::from_translation(place(
                        Vec2::new(text_x, tl.y + NODE_H - 17.0),
                        0.2,
                    )),
                ))
                .id(),
        );
    }

    view.spawned = out;
    view.drawn = Some(model.clone());
}
