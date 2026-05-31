//! Reproduce the panic when double-clicking a compound (drill-in).

mod common;

use bevy::prelude::*;
use common::make_app;
use flow_bevy::bridge::FlowSim;
use flow_bevy::compound::{CurrentScope, CompoundBodyMarker};
use flow_bevy::examples::{Example, LoadExample};
use flow_bevy::gadgets::Kind;

fn load(app: &mut App, ex: Example) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<LoadExample>>()
        .write(LoadExample(ex));
    app.update();
    app.update();
}

fn first_compound_shim_of(app: &App, kind: Kind) -> flow::NodeId {
    let sim = app.world().resource::<FlowSim>();
    let prefix = format!("{}_", kind.label());
    sim.nodes.iter()
        .find(|(_, n)| n.is_compound() && n.name.starts_with(&prefix) && !n.name.contains("::"))
        .map(|(id, _)| *id)
        .unwrap_or_else(|| panic!("no compound shim of kind {:?}", kind))
}

#[test]
fn drill_into_each_kind_no_panic() {
    let mut app = make_app();
    for ex in [
        Example::ClientWorker,
        Example::ClientQueueWorker,
        Example::ClientRouterWorker,
        Example::ThreeLaneFanout,
    ] {
        load(&mut app, ex);
        app.update();
        app.update();
        let worker = first_compound_shim_of(&app, Kind::Worker);
        // Verify the shim has the drill-in marker first.
        let maps = app.world().resource::<flow_bevy::bridge::EntityMaps>();
        let entity = *maps.node_to_entity.get(&worker).expect("worker shim entity");
        assert!(app.world().entity(entity).get::<CompoundBodyMarker>().is_some(),
            "{:?}: worker shim missing CompoundBodyMarker", ex);

        // Simulate double-click: set CurrentScope to the worker shim.
        let mut scope = app.world_mut().resource_mut::<CurrentScope>();
        scope.0 = Some(worker);
        // Pump the scheduler a few times so sync_canvas_population can
        // despawn out-of-scope entities and spawn inner ones, and any
        // queued commands flush.
        for _ in 0..6 { app.update(); }

        // Drill back out.
        let mut scope = app.world_mut().resource_mut::<CurrentScope>();
        scope.0 = None;
        for _ in 0..6 { app.update(); }
    }
}
