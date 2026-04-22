//! Exercise the Connect tool: drop two nodes, enter Connect mode, click
//! source then target, verify the sim gained an edge between the two.

mod common;

use bevy::prelude::*;
use common::{edge_count, make_app};
use flow_bevy::bridge::{EntityMaps, FlowNodeRef, FlowSim};
use flow_bevy::gadgets::Kind;
use flow_bevy::palette::ToolBtn;
use flow_bevy::tool::Tool;
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

fn world_pos_of_node(app: &App, node_name: &str) -> Vec2 {
    // Find the sim node by name, then the Bevy entity it maps to, then its Transform.
    let world = app.world();
    let sim = &world.resource::<FlowSim>().sim;
    let (nid, _) = sim
        .nodes
        .iter()
        .find(|(_, n)| n.name == node_name)
        .expect("sim has no node with that name");
    let maps = world.resource::<EntityMaps>();
    let entity = *maps
        .node_to_entity
        .get(nid)
        .expect("node not mapped to an entity");
    let tf = world
        .get::<Transform>(entity)
        .expect("entity missing Transform");
    tf.translation.truncate()
}

#[test]
fn connect_two_newly_dropped_nodes() {
    let mut app = make_app();

    // Drop a Generator and a Sink offset above the seeded demo chain so
    // the hit-test in Connect mode picks our freshly-placed nodes rather
    // than colliding with Queue_2 (which sits at (-200, 0) — right where
    // we'd otherwise drop the Generator).
    let gen_xy = Vec2::new(-200.0, 200.0);
    let sink_xy = Vec2::new(200.0, 200.0);
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, gen_xy);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Sink));
    simulate_canvas_click(&mut app, sink_xy);

    // Both are in the sim now. Grab their latest names (suffixed by the
    // NodeCounter) and their world positions.
    let (gen_nm, sink_name) = {
        let sim = &app.world().resource::<FlowSim>().sim;
        let gen_n = sim.nodes.values().filter(|n| n.name.starts_with("Gen_")).last().unwrap().name.clone();
        let sink_n = sim.nodes.values().filter(|n| n.name.starts_with("Sink_")).last().unwrap().name.clone();
        (gen_n, sink_n)
    };
    let gen_pos = world_pos_of_node(&app, &gen_nm);
    let sink_pos = world_pos_of_node(&app, &sink_name);

    let before_edges = edge_count(&app);

    // Enter Connect mode via the palette button, then two canvas clicks.
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Connect);
    simulate_canvas_click(&mut app, gen_pos);
    simulate_canvas_click(&mut app, sink_pos);

    let after_edges = edge_count(&app);
    assert_eq!(
        after_edges,
        before_edges + 1,
        "expected 1 new edge after Connect, got delta {}",
        after_edges as i64 - before_edges as i64
    );

    // Verify the edge actually runs from generator to sink.
    let gen_id = {
        let world = app.world();
        let sim = &world.resource::<FlowSim>().sim;
        *sim.nodes.iter().find(|(_, n)| n.name == gen_nm).unwrap().0
    };
    let sink_id = {
        let world = app.world();
        let sim = &world.resource::<FlowSim>().sim;
        *sim.nodes.iter().find(|(_, n)| n.name == sink_name).unwrap().0
    };
    let sim = &app.world().resource::<FlowSim>().sim;
    let matched = sim
        .edges
        .values()
        .any(|e| e.from == gen_id && e.to == sink_id);
    assert!(matched, "no sim edge found from {} → {}", gen_nm, sink_name);

    // All newly-spawned edge entities should map to edge ids too.
    let maps = app.world().resource::<EntityMaps>();
    assert!(
        maps.entity_to_edge.values().count() >= 1,
        "EntityMaps didn't track the new edge"
    );
    let _ = FlowNodeRef(gen_id); // keep FlowNodeRef import alive
}
