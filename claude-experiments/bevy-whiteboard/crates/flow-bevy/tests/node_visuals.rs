//! Visual-metadata tests: the data-colour indicator dot on each node carries
//! the expected colour, and a newly-dropped node spawns a dot matching the
//! active swatch.
//!
//! Arrows on edges are drawn via `Gizmos` (immediate-mode lines), which leave
//! no persistent component we can inspect — verifying them requires either a
//! screenshot or tapping into the gizmos buffer mid-frame. Out of scope for
//! this pass.

mod common;

use bevy::prelude::*;
use common::make_app;
use flow_bevy::bridge::{EntityMaps, FlowSim};
use flow_bevy::gadgets::Kind;
use flow_bevy::nodes::NodeColorDot;
use flow_bevy::palette::{ColorSwatch, ToolBtn};
use flow_bevy::tool::{NodeColors, Tool};
use poster_ui::Theme;
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

fn latest_of_kind(app: &App, kind: Kind) -> Option<flow::NodeId> {
    let sim = &app.world().resource::<FlowSim>().sim;
    let prefix = format!("{}_", kind.label());
    sim.nodes
        .iter()
        .filter_map(|(id, n)| {
            n.name
                .strip_prefix(&prefix)
                .and_then(|s| s.parse::<u32>().ok())
                .map(|num| (*id, num))
        })
        .max_by_key(|(_, num)| *num)
        .map(|(id, _)| id)
}

/// Returns the `Color` of the data-colour dot for the entity that owns the
/// given NodeId — reads the dot's `ColorMaterial` handle.
fn dot_color_for(app: &mut App, nid: flow::NodeId) -> Color {
    // Find the parent entity via EntityMaps, then look at its children for
    // a `NodeColorDot` child, then read that dot's material.
    let parent = *app
        .world()
        .resource::<EntityMaps>()
        .node_to_entity
        .get(&nid)
        .expect("no Bevy entity for that flow node id");

    let dot_entity = {
        let world = app.world();
        let children = world.get::<Children>(parent).expect("parent has no Children");
        children
            .iter()
            .find(|c| world.get::<NodeColorDot>(*c).is_some())
            .expect("no NodeColorDot child under parent")
    };

    let mat_handle = app
        .world()
        .get::<MeshMaterial2d<ColorMaterial>>(dot_entity)
        .expect("dot has no material")
        .0
        .clone();

    let materials = app.world().resource::<Assets<ColorMaterial>>();
    materials
        .get(&mat_handle)
        .expect("material asset missing")
        .color
}

#[test]
fn every_non_router_node_has_a_color_dot_child() {
    let app = make_app();
    let sim = &app.world().resource::<FlowSim>().sim;
    let maps = app.world().resource::<EntityMaps>();
    for (nid, node) in sim.nodes.iter() {
        let entity = *maps
            .node_to_entity
            .get(nid)
            .expect("sim node not mapped to an entity");
        let children = app
            .world()
            .get::<Children>(entity)
            .expect("node parent missing Children");
        let has_dot = children
            .iter()
            .any(|c| app.world().get::<NodeColorDot>(c).is_some());
        // Routers are neutral forwarders — no data type, no dot. Every
        // other kind must have one.
        let is_router = node.name.starts_with("Router_");
        assert_eq!(
            has_dot, !is_router,
            "node {:?}: has_dot={}, but is_router={}",
            node.name, has_dot, is_router
        );
    }
}

#[test]
fn dropped_node_dot_matches_active_swatch() {
    let mut app = make_app();

    // Select slot 3 (dusty teal in iso50), drop a Worker at (0, 250).
    // (Router wouldn't work — routers don't get a data-colour dot.)
    click_by_marker::<ColorSwatch, _>(&mut app, |s| s.0 == 3);
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Worker));
    simulate_canvas_click(&mut app, Vec2::new(0.0, 250.0));

    let nid = latest_of_kind(&app, Kind::Worker).expect("worker wasn't dropped");

    let expected = app.world().resource::<Theme>().data[3];
    let recorded = *app
        .world()
        .resource::<NodeColors>()
        .0
        .get(&nid)
        .expect("NodeColors missing entry");
    assert_eq!(recorded, expected, "NodeColors entry wrong slot");

    // Dot material should have caught up after the drop (needs one
    // app.update() for sync_data_dot_colors to run after spawn).
    app.update();
    let painted = dot_color_for(&mut app, nid);
    assert_eq!(
        painted, expected,
        "NodeColorDot for worker paints slot-3 colour: got {:?}, want {:?}",
        painted, expected
    );
}

#[test]
fn routers_have_no_dot_and_no_color_entry() {
    let mut app = make_app();

    click_by_marker::<ColorSwatch, _>(&mut app, |s| s.0 == 2);
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Router));
    simulate_canvas_click(&mut app, Vec2::new(0.0, 300.0));

    let nid = latest_of_kind(&app, Kind::Router).expect("router wasn't dropped");

    // No NodeColors entry.
    assert!(
        !app.world().resource::<NodeColors>().0.contains_key(&nid),
        "Routers shouldn't get a NodeColors entry — they're untyped forwarders"
    );

    // No NodeColorDot child.
    let parent = *app
        .world()
        .resource::<EntityMaps>()
        .node_to_entity
        .get(&nid)
        .unwrap();
    let children = app.world().get::<Children>(parent).expect("no children");
    let has_dot = children
        .iter()
        .any(|c| app.world().get::<NodeColorDot>(c).is_some());
    assert!(!has_dot, "Router spawned a data-colour dot it shouldn't have");
}

#[test]
fn two_dots_differ_when_slots_differ() {
    let mut app = make_app();

    click_by_marker::<ColorSwatch, _>(&mut app, |s| s.0 == 0);
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, 250.0));
    let a = latest_of_kind(&app, Kind::Generator).unwrap();

    click_by_marker::<ColorSwatch, _>(&mut app, |s| s.0 == 2);
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(300.0, 250.0));
    let b = latest_of_kind(&app, Kind::Generator).unwrap();

    app.update();
    let ca = dot_color_for(&mut app, a);
    let cb = dot_color_for(&mut app, b);
    assert_ne!(ca, cb, "dots for two different slot-tagged nodes must differ");
}
