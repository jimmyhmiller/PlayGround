//! Fleet layout — arranges the workers an
//! [`AutoScalingGroup`](crate::gadgets::Kind::AutoScalingGroup) spawns
//! into a fan-out around the ASG node.
//!
//! The ASG is a top-level node that spawns real gadget instances (its
//! `worker_class`) at the top level and tracks them in its `members`
//! slot. Those workers aren't in `visual.json`, so the reconciler drops
//! each at a default grid slot; here we read the ASG's live `members`
//! list and ease every worker onto a fan-out arc anchored on the ASG,
//! re-flowing as the fleet grows and shrinks.
//!
//! Everything is top-level, so the dispatch edges (ASG → worker) render
//! and animate on the canvas — the fan-out is visible without drilling
//! into anything.

use std::collections::BTreeMap;

use bevy::prelude::*;
use flow::{NodeId, Value};

use crate::bridge::FlowNodeRef;
use crate::sim_driver::SimSnapshotRes;

/// Class name of the autoscaler leaf node.
const ASG_CLASS: &str = "AutoScalingGroup";

/// Exponential-smoothing factor per second for the re-flow ease.
const REFLOW_LERP_PER_SEC: f32 = 12.0;

pub struct FleetLayoutPlugin;
impl Plugin for FleetLayoutPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            layout_fleets.after(crate::compound::sync_canvas_population),
        );
    }
}

fn layout_fleets(
    time: Res<Time>,
    snapshot: Res<SimSnapshotRes>,
    mut q: Query<(&FlowNodeRef, &mut Transform)>,
) {
    // ── Gather ASG anchors (position) + their member lists.
    let mut anchors: BTreeMap<NodeId, Vec2> = BTreeMap::new();
    for (fref, tf) in q.iter() {
        if snapshot.0.nodes.get(&fref.0).map(|v| v.class_name.as_deref())
            == Some(Some(ASG_CLASS))
        {
            anchors.insert(fref.0, tf.translation.truncate());
        }
    }
    if anchors.is_empty() {
        return;
    }

    // ── Compute each member's target slot, anchored on its ASG.
    let mut targets: BTreeMap<NodeId, Vec2> = BTreeMap::new();
    for (asg, anchor) in &anchors {
        let members: Vec<NodeId> = match snapshot.0.resolve_slot(*asg, "members") {
            Some(Value::List(items)) => items
                .iter()
                .filter_map(|v| if let Value::NodeRef(id) = v { Some(*id) } else { None })
                .collect(),
            _ => Vec::new(),
        };
        let n = members.len();
        for (i, m) in members.iter().enumerate() {
            targets.insert(*m, fan_out_slot(*anchor, i, n));
        }
    }
    if targets.is_empty() {
        return;
    }

    // ── Ease each worker toward its slot; snap ones still sitting at the
    // reconciler's far-away default grid origin so they don't fly in.
    let alpha = (REFLOW_LERP_PER_SEC * time.delta_secs()).clamp(0.0, 1.0);
    for (fref, mut tf) in q.iter_mut() {
        let Some(target) = targets.get(&fref.0).copied() else { continue };
        let cur = tf.translation.truncate();
        let next = if cur.distance(target) > 600.0 {
            target
        } else {
            cur.lerp(target, alpha)
        };
        tf.translation.x = next.x;
        tf.translation.y = next.y;
    }
}

/// Even fan on an arc opening to the right of the anchor. The arc widens
/// with fleet size (bounded) and the radius grows so workers don't crowd
/// as the fleet scales up.
fn fan_out_slot(anchor: Vec2, i: usize, n: usize) -> Vec2 {
    const STEP: f32 = 0.40; // ~23° between adjacent workers
    const MAX_HALF: f32 = 1.20; // cap half-spread at ~69°
    let half = ((n as f32 - 1.0) * 0.5 * STEP).min(MAX_HALF);
    let t = if n <= 1 { 0.5 } else { i as f32 / (n as f32 - 1.0) };
    let angle = -half + t * (2.0 * half);
    let radius = 190.0 + 9.0 * n as f32;
    anchor + Vec2::new(radius * angle.cos(), radius * angle.sin())
}
