//! Bevy ↔ Flow bridge.
//!
//! A `FlowSim` resource owns the single authoritative `flow::Sim`.
//! Bevy entities carry `FlowNodeRef(NodeId)` components. Per frame we
//! advance the sim by real-time × multiplier, then drain new events
//! from the log to spawn transient visual packets.

use std::collections::HashMap;

use bevy::prelude::*;
use flow::{Event, NodeId, EdgeId, Sim};

use crate::gadgets;

pub struct FlowBridgePlugin;
impl Plugin for FlowBridgePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FlowSim>()
            .init_resource::<EntityMaps>()
            .init_resource::<NewEvents>()
            .init_resource::<SimClock>()
            .add_systems(Update, (advance_sim, collect_new_events).chain());
    }
}

/// Wraps the live simulation. Gadgets create nodes directly in here.
#[derive(Resource)]
pub struct FlowSim {
    pub sim: Sim,
    /// Event-log index up to which we've already consumed events for
    /// visualization. Events past this need to be "spawned".
    pub consumed_log_index: u64,
}

impl Default for FlowSim {
    fn default() -> Self {
        let mut sim = Sim::new(1);
        gadgets::install_default_params(&mut sim);
        Self { sim, consumed_log_index: 0 }
    }
}

#[derive(Resource, Default)]
pub struct EntityMaps {
    pub node_to_entity: HashMap<NodeId, Entity>,
    pub entity_to_node: HashMap<Entity, NodeId>,
    pub edge_to_entity: HashMap<EdgeId, Entity>,
    pub entity_to_edge: HashMap<Entity, EdgeId>,
}

/// Component attached to every node entity.
#[derive(Component, Clone, Copy)]
pub struct FlowNodeRef(pub NodeId);

/// Component attached to every edge entity.
#[derive(Component, Clone, Copy)]
pub struct FlowEdgeRef(pub EdgeId);

/// Sim playback state. `multiplier` controls ONLY how fast the sim
/// advances — it does NOT affect visual packet duration. The visual
/// scale lives in `VisualTimeline::k` (see `visual.rs`) and is
/// tuned independently with its own hotkeys.
///
/// This split is what the user asked for: fast sim + readable
/// visuals (or slow sim + fast-flashing visuals) are both possible.
///
/// `[` / `]` → sim speed (halve / double `multiplier`)
/// `-` / `=` → visual scale (halve / double `VisualTimeline::k`)
#[derive(Resource)]
pub struct SimClock {
    /// Sim seconds per wall second. 1.0 = sim runs at wall clock.
    pub multiplier: f64,
    pub paused: bool,
    pub step_once_ns: Option<u64>,
}
impl Default for SimClock {
    fn default() -> Self {
        Self { multiplier: 1.0, paused: false, step_once_ns: None }
    }
}

/// Events from the most recent advance, for downstream systems.
#[derive(Resource, Default)]
pub struct NewEvents(pub Vec<Event>);

fn advance_sim(
    time: Res<Time>,
    mut flow: ResMut<FlowSim>,
    mut clock: ResMut<SimClock>,
) {
    let dt_real = time.delta_secs_f64();
    let dt_sim_ns: u64 = if let Some(step) = clock.step_once_ns.take() {
        step
    } else if clock.paused {
        return;
    } else {
        (dt_real * 1_000_000_000.0 * clock.multiplier).max(0.0) as u64
    };
    if dt_sim_ns == 0 { return; }
    let target = flow.sim.now_ns.saturating_add(dt_sim_ns);
    flow.sim.run_until(target);
}

/// Grab events recorded since last frame into the `NewEvents` bucket.
pub fn collect_new_events(mut flow: ResMut<FlowSim>, mut bucket: ResMut<NewEvents>) {
    bucket.0.clear();
    let seen = flow.consumed_log_index;
    let total = flow.sim.log.total_recorded;
    if total <= seen { return; }
    // The event log is a ring — if too many events happened we lose
    // some. Skip the missed window.
    let ring_len = flow.sim.log.events.len() as u64;
    let first_resident = total.saturating_sub(ring_len);
    let skip_to = seen.max(first_resident);
    let start = (skip_to - first_resident) as usize;
    // Copy new events; we only read, but they may be cloned cheaply.
    for ev in flow.sim.log.events.iter().skip(start) {
        bucket.0.push(ev.clone());
    }
    flow.consumed_log_index = total;
}
