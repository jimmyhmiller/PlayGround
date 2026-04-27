//! Bevy ↔ Flow bridge.
//!
//! The sim runs on a worker thread (or inline in `Direct` mode for
//! tests) — see `sim_driver`. Bevy systems read state from the
//! per-frame `SimSnapshotRes` and send mutations through
//! `SimDriverRes` as commands.

use std::collections::HashMap;
use std::sync::Mutex;

use bevy::prelude::*;
use flow::{Event, NodeId, EdgeId, Sim};

use crate::gadgets;
use crate::sim_driver::{
    SimDriver, SimDriverRes, SimEventRx, SimSnapshotRes, SnapshotReady, sync_snapshot_system,
};

/// Back-compat alias for tests that imported `bridge::FlowSim` and
/// did `world.resource_mut::<FlowSim>().sim.X`. `SimDriverRes` impls
/// `Deref<Target=Sim>` (Direct mode) so `flow_res.sim_mut().X`
/// or just `flow_res.X` works the same.
pub use crate::sim_driver::SimDriverRes as FlowSim;

pub struct FlowBridgePlugin;
impl Plugin for FlowBridgePlugin {
    fn build(&self, app: &mut App) {
        // Build the sim. Default driver is `Worker` (background thread,
        // decoupled from the render frame); set `FLOW_BEVY_SYNC_SIM=1`
        // to fall back to the old `Direct` (in-process synchronous)
        // behavior for A/B benching.
        let mut sim = Sim::new(1);
        gadgets::install_default_params(&mut sim);
        let use_direct = std::env::var("FLOW_BEVY_SYNC_SIM")
            .ok().filter(|s| !s.is_empty()).is_some();
        let (driver, events_rx) = if use_direct {
            bevy::log::info!("flow-bevy: using Direct (synchronous) sim driver");
            SimDriver::direct(sim, 1.0)
        } else {
            SimDriver::worker(sim, 1.0)
        };
        app.insert_resource(SimDriverRes(driver))
            .insert_resource(SimEventRx(Mutex::new(events_rx)))
            .init_resource::<SimSnapshotRes>()
            .init_resource::<EntityMaps>()
            .init_resource::<NewEvents>()
            .init_resource::<SimClock>()
            // Snapshot publishes in `PreUpdate` so every `Update`
            // reader (inspector, hud, edges, probes, …) sees this
            // frame's view, not last frame's. Before this lived in
            // `Update` we hit a race where Bevy could schedule a
            // reader ahead of the publisher.
            .add_systems(PreUpdate, sync_snapshot_system.in_set(SnapshotReady))
            .add_systems(
                Update,
                (advance_visual_clock, push_clock_to_driver, collect_new_events).chain(),
            );
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
///
/// Mutations to this resource are mirrored to the driver's
/// `SimControl` atomics by `push_clock_to_driver` (runs every frame
/// near the start of `Update`), so the worker reacts within one
/// yield without us having to push a command for each tweak.
#[derive(Resource)]
pub struct SimClock {
    /// Sim seconds per wall second. 1.0 = sim runs at wall clock.
    pub multiplier: f64,
    pub paused: bool,
    pub step_once_ns: Option<u64>,
    /// Pause-aware wall clock used by every visual system that needs
    /// "now" in real seconds. Advanced by `time.delta_secs_f64()`
    /// each frame, EXCEPT while `paused == true` — so packet
    /// animations freeze in place when the user pauses, and resume
    /// from where they left off when the user unpauses.
    /// Bevy's `time.elapsed_secs_f64()` keeps ticking through pause;
    /// using it directly was the bug that let visuals keep
    /// animating while the sim sat still.
    pub visual_now: f64,
}
impl Default for SimClock {
    fn default() -> Self {
        Self { multiplier: 1.0, paused: false, step_once_ns: None, visual_now: 0.0 }
    }
}

/// Events from the most recent advance, for downstream systems.
#[derive(Resource, Default)]
pub struct NewEvents(pub Vec<Event>);

/// Advance the pause-aware visual clock. Runs first in the update
/// chain so every visual system that reads `clock.visual_now` sees
/// today's value, not yesterday's. Stepping (`step_once_ns`)
/// advances visual time by the same delta the sim takes, so a
/// single-step is visible: packets that emit/arrive within the step
/// flash in their lane.
fn advance_visual_clock(
    time: Res<Time>,
    mut clock: ResMut<SimClock>,
) {
    let dt_real = time.delta_secs_f64();
    if clock.paused && clock.step_once_ns.is_none() {
        return;
    }
    clock.visual_now += dt_real * clock.multiplier;
}

/// Mirror SimClock state into the driver's atomics each frame, plus
/// (Direct mode only) drive a per-frame sim advance using the
/// existing wall-clock × multiplier model — that's what tests rely
/// on. In Worker mode the worker self-paces and this only pushes
/// the atomics through.
fn push_clock_to_driver(
    time: Res<Time>,
    mut clock: ResMut<SimClock>,
    mut driver: ResMut<SimDriverRes>,
    mut perf: ResMut<crate::perf::PhaseTimings>,
) {
    let control = driver.0.control().clone();
    control.set_multiplier(clock.multiplier);
    control.set_paused(clock.paused);

    if driver.0.is_worker() {
        if let Some(step) = clock.step_once_ns.take() {
            control.request_step(step);
        }
        return;
    }

    // Direct mode (tests): tick the sim inline like the old bridge did.
    let dt_real = time.delta_secs_f64();
    let dt_sim_ns: u64 = if let Some(step) = clock.step_once_ns.take() {
        step
    } else if clock.paused {
        return;
    } else {
        (dt_real * 1_000_000_000.0 * clock.multiplier).max(0.0) as u64
    };
    if dt_sim_ns == 0 { return; }
    crate::time_phase!(perf, "bridge.advance_sim", {
        driver.0.advance_direct(dt_sim_ns);
    });
}

/// Drain events the sim worker has produced since last frame into
/// the `NewEvents` bucket. Drops the perf samples carried on the
/// freshly-loaded snapshot into `WorkerPerf` so the bench can show
/// "what is the worker thread doing" — separately from
/// `PhaseTimings`, which is reserved for *main-thread* per-frame
/// phases that actually contribute to frame time.
pub fn collect_new_events(
    snapshot: Res<SimSnapshotRes>,
    rx: Res<SimEventRx>,
    mut bucket: ResMut<NewEvents>,
    mut worker_perf: ResMut<crate::perf::WorkerPerf>,
) {
    bucket.0.clear();
    {
        let rx = rx.0.lock().expect("event channel mutex poisoned");
        while let Ok(ev) = rx.try_recv() {
            bucket.0.push(ev);
        }
    }
    for (phase, us) in &snapshot.0.perf_samples {
        worker_perf.0.record_us(*phase, *us);
    }
}
