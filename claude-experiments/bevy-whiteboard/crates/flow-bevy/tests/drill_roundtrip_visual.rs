//! After drilling INTO a compound and back OUT, the compound's visual
//! must match what it looked like originally — same `NodeKind`, same
//! `BodyShape`. The bug: respawn went through `spawn_compound_body_entity`
//! (LabeledBox), losing the gadget glyph/shape that the examples builder
//! had originally produced via `spawn_node_entity`.

mod common;

use bevy::prelude::*;
use common::make_app;
use flow_bevy::bridge::{EntityMaps, FlowSim};
use flow_bevy::compound::{CompoundBodyMarker, CurrentScope};
use flow_bevy::examples::{Example, LoadExample};
use flow_bevy::gadgets::Kind;
use flow_bevy::nodes::{BodyShape, NodeKind};

fn load(app: &mut App, ex: Example) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<LoadExample>>()
        .write(LoadExample(ex));
    app.update();
    app.update();
    app.update();
}

fn first_shim_of(app: &App, kind: Kind) -> flow::NodeId {
    let sim = app.world().resource::<FlowSim>();
    let prefix = format!("{}_", kind.label());
    sim.nodes
        .iter()
        .find(|(_, n)| n.is_compound() && n.name.starts_with(&prefix) && !n.name.contains("::"))
        .map(|(id, _)| *id)
        .unwrap_or_else(|| panic!("no compound shim of kind {:?}", kind))
}

fn entity_of(app: &App, nid: flow::NodeId) -> Entity {
    let maps = app.world().resource::<EntityMaps>();
    *maps
        .node_to_entity
        .get(&nid)
        .unwrap_or_else(|| panic!("no entity for node {:?}", nid))
}

fn snapshot_compound_visual(app: &App, nid: flow::NodeId) -> CompoundVisualSnap {
    let entity = entity_of(app, nid);
    let world = app.world();
    let e = world.entity(entity);
    CompoundVisualSnap {
        node_kind: e.get::<NodeKind>().map(|nk| nk.0),
        body_shape_debug: e.get::<BodyShape>().map(|s| format!("{:?}", s)),
        has_marker: e.get::<CompoundBodyMarker>().is_some(),
    }
}

#[derive(Debug, PartialEq)]
struct CompoundVisualSnap {
    node_kind: Option<Kind>,
    body_shape_debug: Option<String>,
    has_marker: bool,
}

/// Drill into a generator and back out — the gadget shape (circle with
/// generator glyph), node-kind marker, and drill-in marker must all
/// survive. The bug rebuilt it as a generic LabeledBox compound visual.
#[test]
fn drill_in_then_out_preserves_gadget_visual() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);

    let gen_id = first_shim_of(&app, Kind::Generator);
    let before = snapshot_compound_visual(&app, gen_id);

    // Sanity: the initial spawn must give us the gadget visual.
    assert_eq!(
        before.node_kind,
        Some(Kind::Generator),
        "initial generator spawn missing NodeKind: {:?}",
        before,
    );
    assert!(before.has_marker, "initial generator missing CompoundBodyMarker");

    // Drill in.
    app.world_mut().resource_mut::<CurrentScope>().0 = Some(gen_id);
    for _ in 0..6 {
        app.update();
    }
    // Drill back out.
    app.world_mut().resource_mut::<CurrentScope>().0 = None;
    for _ in 0..6 {
        app.update();
    }

    let after = snapshot_compound_visual(&app, gen_id);
    assert_eq!(
        before, after,
        "compound visual changed after drill in/out roundtrip:\n  before = {:?}\n  after  = {:?}",
        before, after,
    );
}

/// Same as above but for every kind used in ThreeLaneFanout — generator,
/// router, queue, worker, sink. All five must keep their gadget visuals.
#[test]
fn drill_roundtrip_three_lane_all_kinds() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);

    let kinds = [
        Kind::Generator,
        Kind::Router,
        Kind::Queue,
        Kind::Worker,
        Kind::Sink,
    ];

    let befores: Vec<(Kind, flow::NodeId, CompoundVisualSnap)> = kinds
        .iter()
        .map(|k| {
            let id = first_shim_of(&app, *k);
            (*k, id, snapshot_compound_visual(&app, id))
        })
        .collect();

    for (_, id, _) in &befores {
        app.world_mut().resource_mut::<CurrentScope>().0 = Some(*id);
        for _ in 0..6 {
            app.update();
        }
        app.world_mut().resource_mut::<CurrentScope>().0 = None;
        for _ in 0..6 {
            app.update();
        }
    }

    for (kind, id, before) in &befores {
        let after = snapshot_compound_visual(&app, *id);
        assert_eq!(
            before, &after,
            "{:?} compound visual changed after drill in/out:\n  before = {:?}\n  after  = {:?}",
            kind, before, after,
        );
    }
}
