//! Drilling into a compound and back out must NOT scramble the canvas
//! layout: every top-level node returns to the same position it had
//! before. Regression: the respawn fell back to a default grid because
//! example positions weren't persisted across despawn.

mod common;

use bevy::prelude::*;
use common::make_app;
use flow_bevy::bridge::EntityMaps;
use flow_bevy::bridge::FlowSim;
use flow_bevy::compound::CurrentScope;
use flow_bevy::examples::{Example, LoadExample};
use flow_bevy::gadgets::Kind;

fn load(app: &mut App, ex: Example) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<LoadExample>>()
        .write(LoadExample(ex));
    app.update();
    app.update();
    app.update();
}

fn positions(app: &App) -> std::collections::BTreeMap<flow::NodeId, Vec2> {
    let maps = app.world().resource::<EntityMaps>();
    let mut out = std::collections::BTreeMap::new();
    for (nid, e) in maps.node_to_entity.iter() {
        if let Some(tf) = app.world().entity(*e).get::<Transform>() {
            out.insert(*nid, tf.translation.truncate());
        }
    }
    out
}

fn first_shim_of(app: &App, kind: Kind) -> flow::NodeId {
    let sim = app.world().resource::<FlowSim>();
    let prefix = format!("{}_", kind.label());
    sim.nodes
        .iter()
        .find(|(_, n)| n.is_compound() && n.name.starts_with(&prefix) && !n.name.contains("::"))
        .map(|(id, _)| *id)
        .unwrap_or_else(|| panic!("no shim of {:?}", kind))
}

#[test]
fn drill_roundtrip_restores_top_level_layout() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);

    let before = positions(&app);
    assert!(before.len() >= 6, "expected several top-level nodes, got {}", before.len());

    // Drill into the generator, then back out.
    let gen_id = first_shim_of(&app, Kind::Generator);
    app.world_mut().resource_mut::<CurrentScope>().0 = Some(gen_id);
    for _ in 0..6 { app.update(); }
    app.world_mut().resource_mut::<CurrentScope>().0 = None;
    for _ in 0..6 { app.update(); }

    let after = positions(&app);

    // Every node present before must be present after, at the same spot.
    let mut moved: Vec<(flow::NodeId, Vec2, Vec2)> = Vec::new();
    for (nid, p0) in &before {
        match after.get(nid) {
            Some(p1) if p0.distance(*p1) < 0.5 => {}
            Some(p1) => moved.push((*nid, *p0, *p1)),
            None => moved.push((*nid, *p0, Vec2::splat(f32::NAN))),
        }
    }
    assert!(
        moved.is_empty(),
        "drill round-trip moved {} top-level nodes: {:?}",
        moved.len(),
        moved.iter().take(8).collect::<Vec<_>>()
    );
}
