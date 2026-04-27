#![allow(dead_code)]
//! Shared test harness for flow-bevy integration tests. Spins up a headless
//! Bevy app with every flow-bevy plugin loaded **without** the demo seed,
//! so tests start with an empty canvas and build their own topology.
//!
//! The headless app uses `poster_ui::testing::test_app_headless` under the
//! hood — full `DefaultPlugins` (so UI layout + interaction work), no winit
//! event loop, no GPU backend.

use bevy::prelude::*;
use flow_bevy::FlowBevyPlugins;
use flow_bevy::bridge::{EntityMaps, FlowSim};
use flow_bevy::edges::{HiddenEdges, wire_flow_edge};
use flow_bevy::gadgets::{Kind, spawn as spawn_gadget};
use flow_bevy::nodes::NodeCounter;
use flow_bevy::sim_driver::{SimDriver, SimDriverRes, SimEventRx};
use flow_bevy::tool::NodeColors;
use flow_bevy::theme::Theme;

/// Build a test app with flow-bevy plugins loaded and `Startup` run.
/// The canvas is empty — no demo seed. Each test is responsible for
/// spawning whatever nodes it needs via [`spawn_node`] + [`wire`] or
/// through the palette click helpers.
///
/// Switches the sim driver to **Direct** mode so tests get
/// deterministic, single-threaded sim execution. The default driver
/// installed by `FlowBridgePlugin` is `Worker` (good for the live
/// app, bad for tests because ticks happen on a background thread
/// out of test control).
pub fn make_app() -> App {
    let mut app = poster_ui::testing::test_app_headless();
    app.add_plugins(FlowBevyPlugins);
    // Test-only knob settings: sim at 1× wall clock, and visual
    // scale `k = 1.0` so sim nanoseconds map 1:1 to real seconds.
    // The live app uses `k = 200` so gadget edges (~1 ms) are
    // clearly visible (~200 ms); tests prefer the identity mapping
    // so assertions like "a packet emitted at at_ns=T becomes
    // visible at real_t ≈ T * 1e-9" hold without unit conversion.
    app.world_mut().resource_mut::<flow_bevy::bridge::SimClock>().multiplier = 1.0;
    {
        let mut tl = app.world_mut().resource_mut::<flow_bevy::edges::VisualTimelineRes>();
        tl.0.k = 1.0;
    }
    // Replace the worker-mode driver the bridge installs with a
    // Direct one. Default-fresh `Sim` carries the gadget params the
    // bridge installed.
    {
        let world = app.world_mut();
        let mut sim = flow::Sim::new(1);
        flow_bevy::gadgets::install_default_params(&mut sim);
        let (driver, events_rx) = SimDriver::direct(sim, 1.0);
        world.insert_resource(SimDriverRes(driver));
        world.insert_resource(SimEventRx(std::sync::Mutex::new(events_rx)));
    }
    app.update();
    app.update();
    app
}

/// Spawn a single gadget and register it with the UI state the test
/// harness needs (NodeColors for packet rendering, the counter for
/// unique names). Returns the `NodeId` for wiring.
///
/// This bypasses palette clicks — use it when a test just needs the
/// nodes present and doesn't care about exercising the drop path. For
/// tests that *do* care about drop/connect interactions, use the
/// palette click helpers in `poster_ui::testing`.
pub fn spawn_node(app: &mut App, kind: Kind, slot: usize, name: &str) -> flow::NodeId {
    let world = app.world_mut();
    let data_color = world.resource::<Theme>().data[slot];
    let name_owned = name.to_string();
    let id = world.resource_mut::<FlowSim>().0.with_sim_mut(move |sim| {
        spawn_gadget(sim, kind, &name_owned, slot)
    });
    world.resource_mut::<NodeCounter>().0 += 1;
    if !matches!(kind, Kind::Router) {
        world.resource_mut::<NodeColors>().0.insert(id, data_color);
    }
    id
}

/// Wire two sim nodes together the same way the Connect tool does.
/// This goes through the shared `wire_flow_edge` helper so pull-semantics
/// (Worker→Queue reverse edges, upstream/downstream slot wiring) are set
/// up identically to user interactions. Commands are applied before
/// returning so the resulting edges are queryable in the same frame.
pub fn wire(app: &mut App, from: flow::NodeId, from_kind: Kind, to: flow::NodeId, to_kind: Kind) {
    let world = app.world_mut();
    let mut sys_state: bevy::ecs::system::SystemState<(
        Commands,
        ResMut<FlowSim>,
        ResMut<EntityMaps>,
        ResMut<HiddenEdges>,
    )> = bevy::ecs::system::SystemState::new(world);
    {
        let (mut commands, mut driver, mut maps, mut hidden) = sys_state.get_mut(world);
        wire_flow_edge(
            &mut driver,
            &mut maps,
            &mut hidden,
            &mut commands,
            from, Some(from_kind),
            to,   Some(to_kind),
        );
        sys_state.apply(world);
    }
}

/// Advance the sim deterministically by `duration_ns` without relying on
/// `Time::delta_secs` (which is 0 in a headless test). Directly drives
/// the Direct-mode driver, which republishes a snapshot afterward so
/// downstream reads see the new state.
pub fn advance_sim_ns(app: &mut App, duration_ns: u64) {
    let world = app.world_mut();
    world.resource_mut::<FlowSim>().0.advance_direct(duration_ns);
}

/// Convenience builder for the canonical pull chain:
/// `Gen → Queue ← Worker → Sink`. Returns each id so the test can assert
/// on specific slots. Everything is slot 0 (red) — single-colour stream.
pub fn build_pull_chain(app: &mut App) -> PullChain {
    let generator = spawn_node(app, Kind::Generator, 0, "Gen_test");
    let queue = spawn_node(app, Kind::Queue, 0, "Queue_test");
    let worker = spawn_node(app, Kind::Worker, 0, "Worker_test");
    let sink = spawn_node(app, Kind::Sink, 0, "Sink_test");
    wire(app, generator, Kind::Generator, queue, Kind::Queue);
    wire(app, worker, Kind::Worker, queue, Kind::Queue);
    wire(app, worker, Kind::Worker, sink, Kind::Sink);
    PullChain { generator, queue, worker, sink }
}

pub struct PullChain {
    pub generator: flow::NodeId,
    pub queue: flow::NodeId,
    pub worker: flow::NodeId,
    pub sink: flow::NodeId,
}

/// Count of sim nodes / edges currently in the app. Tests use this as
/// a before/after baseline when asserting relative drop/connect deltas.
pub fn node_count(app: &App) -> usize {
    app.world().resource::<FlowSim>().nodes.len()
}

pub fn edge_count(app: &App) -> usize {
    app.world().resource::<FlowSim>().edges.len()
}
