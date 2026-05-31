//! The canvas "N absorbed" label and inspector sink readout must show
//! the real absorbed count. Both read from the render snapshot via
//! `RenderSnapshot::resolve_slot`, which must resolve a gadget shim's
//! `count` into the inner Counter — otherwise the shim has no `count`
//! slot and every sink reads 0 even while it's absorbing.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow::Value;
use flow_bevy::bridge::FlowSim;
use flow_bevy::examples::{Example, LoadExample};
use flow_bevy::gadgets::Kind;
use flow_bevy::sim_driver::{make_snapshot, SimDriverRes};

fn load(app: &mut App, ex: Example) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<LoadExample>>()
        .write(LoadExample(ex));
    app.update();
    app.update();
}

fn sink_shims(app: &App) -> Vec<flow::NodeId> {
    let sim = app.world().resource::<FlowSim>();
    let prefix = format!("{}_", Kind::Sink.label());
    sim.nodes
        .iter()
        .filter(|(_, n)| n.name.starts_with(&prefix) && !n.name.contains("::"))
        .map(|(id, _)| *id)
        .collect()
}

/// `resolve_slot(sink_shim, "count")` on a fresh render snapshot must
/// equal the inner Counter's count, and be > 0 after the sim runs.
#[test]
fn snapshot_resolves_sink_absorbed_count() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);
    advance_sim_ns(&mut app, 5_000_000_000);

    let shims = sink_shims(&app);
    assert_eq!(shims.len(), 3, "expected three sink shims");

    // Build a snapshot exactly like the live render path does.
    let snap = app
        .world_mut()
        .resource_mut::<SimDriverRes>()
        .0
        .with_sim_mut(|sim| make_snapshot(sim));

    for shim in &shims {
        // The shim itself must NOT carry a `count` slot (it's a compound
        // port-shim) — proving the readout can't just read it directly.
        let shim_direct = snap.nodes.get(shim).and_then(|n| n.slots.get("count"));
        assert!(
            shim_direct.is_none(),
            "sink shim unexpectedly has a direct `count` slot; test premise stale"
        );

        // Resolved value must be the real, growing absorbed count.
        let resolved = match snap.resolve_slot(*shim, "count") {
            Some(Value::Int(i)) => *i,
            other => panic!("resolve_slot(sink, count) = {:?}, expected Int", other),
        };
        assert!(
            resolved > 0,
            "sink {:?} resolved absorbed count is 0 — readout would show '0 absorbed'",
            shim
        );
    }
}
