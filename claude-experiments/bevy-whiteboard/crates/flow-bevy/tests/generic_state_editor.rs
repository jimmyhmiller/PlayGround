//! Generic per-slot state editor — surfaces every Float slot as a
//! `[0, 1]` slider and every Bool slot as a toggle, regardless of the
//! gadget's `Kind`. Validates the cache's `hit_rate` round-trips
//! through the slider.

mod common;

use std::path::PathBuf;

use bevy::prelude::*;
use common::make_app;
use flow::Value;
use flow_bevy::bridge::{EntityMaps, FlowSim};
use flow_bevy::canvas::load_canvas;
use flow_bevy::inspector::GenericFloatSlider;
use flow_bevy::nodes::Selection;
use poster_ui::Slider;

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().to_path_buf()
}

/// Load the cache.whiteboard into a freshly-built test app, replacing
/// its blank FlowSim with the loaded canvas's sim. The canvas loader
/// returns a fully-populated `Sim`, but it doesn't spawn Bevy node
/// entities — that's normally done by `seed_from_path`. We replicate
/// the entity-spawning bit minimally so `Selection` has something to
/// point at.
fn load_cache_canvas(app: &mut App) -> flow::NodeId {
    let canvas = load_canvas(project_root().join("examples/cache.whiteboard"), 1)
        .expect("load cache.whiteboard");
    {
        let mut flow = app.world_mut().resource_mut::<FlowSim>();
        flow.sim = canvas.sim;
        flow.consumed_log_index = flow.sim.log.total_recorded;
    }
    // Spawn a minimal Bevy entity per sim node so the inspector's
    // selection-driven query can resolve the node kind.
    let node_ids: Vec<(flow::NodeId, String)> = {
        let flow = app.world().resource::<FlowSim>();
        flow.sim.nodes.iter().map(|(id, n)| (*id, n.name.clone())).collect()
    };
    let mut cache_nid = None;
    for (nid, name) in node_ids {
        let kind = flow_bevy::canvas::class_to_kind(
            app.world().resource::<FlowSim>().sim.nodes[&nid]
                .class.as_deref().unwrap_or("Worker")
        );
        let ent = app.world_mut().spawn((
            flow_bevy::nodes::NodeKind(kind),
            flow_bevy::bridge::FlowNodeRef(nid),
        )).id();
        app.world_mut().resource_mut::<EntityMaps>().node_to_entity.insert(nid, ent);
        if name == "Cache" {
            cache_nid = Some((nid, ent));
        }
    }
    let (cache_nid, cache_ent) = cache_nid.expect("Cache node missing");
    app.world_mut().resource_mut::<Selection>().entity = Some(cache_ent);
    app.update();
    app.update();
    cache_nid
}

#[test]
fn float_slot_gets_a_slider_in_generic_section() {
    let mut app = make_app();
    let cache = load_cache_canvas(&mut app);

    // The cache.whiteboard sets hit_rate=0.5, so the slider should
    // exist with that initial value.
    let world = app.world_mut();
    let mut q = world.query::<(&Slider, &GenericFloatSlider)>();
    let (slider, marker) = q.iter(world)
        .find(|(_, m)| m.node == cache && m.slot == "hit_rate")
        .expect("no GenericFloatSlider for Cache.hit_rate");

    assert!((slider.value - 0.5).abs() < 1e-3,
        "expected slider initial 0.5, got {}", slider.value);
    let _ = marker;
}

#[test]
fn dragging_slider_updates_float_slot() {
    let mut app = make_app();
    let cache = load_cache_canvas(&mut app);

    // Find the slider entity and set its value as if the user dragged.
    let target = {
        let world = app.world_mut();
        let mut q = world.query::<(Entity, &GenericFloatSlider)>();
        q.iter(world)
            .find(|(_, m)| m.node == cache && m.slot == "hit_rate")
            .map(|(e, _)| e).expect("slider missing")
    };
    {
        let world = app.world_mut();
        let mut q = world.query::<&mut Slider>();
        let mut s = q.get_mut(world, target).unwrap();
        s.value = 0.85;
    }
    app.update();

    let v = match &app.world().resource::<FlowSim>().sim.nodes[&cache].slots["hit_rate"] {
        Value::Float(f) => *f, other => panic!("hit_rate not Float: {:?}", other),
    };
    assert!((v - 0.85).abs() < 1e-3, "slot didn't follow slider, got {}", v);
}

#[test]
fn external_slot_write_updates_slider() {
    let mut app = make_app();
    let cache = load_cache_canvas(&mut app);

    // Simulate a scenario / timeline writing the slot directly.
    {
        let mut sim = app.world_mut().resource_mut::<FlowSim>();
        sim.sim.nodes.get_mut(&cache).unwrap()
            .slots.insert("hit_rate".into(), Value::Float(0.2));
    }
    app.update();

    let world = app.world_mut();
    let mut q = world.query::<(&Slider, &GenericFloatSlider)>();
    let (slider, _) = q.iter(world)
        .find(|(_, m)| m.node == cache && m.slot == "hit_rate")
        .unwrap();
    assert!((slider.value - 0.2).abs() < 1e-3,
        "slider should track external write, got {}", slider.value);
}
