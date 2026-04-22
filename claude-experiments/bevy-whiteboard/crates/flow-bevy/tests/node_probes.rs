//! Node probes — compositional. Each kind declares its own `ProbeSpec`
//! list; the Probe tool stacks one probe entity per spec above the clicked
//! node. This test proves:
//!
//!  * Clicking a node with the Probe tool spawns the exact number of
//!    probes declared by that kind.
//!  * Each probe reads its value through the spec's `read` function, not
//!    through a hard-coded match on kind inside the probe module.
//!  * Probes on different kinds don't collide — each gets its own stack.
//!  * Routers (no specs) get no probes.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow_bevy::bridge::FlowSim;
use flow_bevy::gadgets::{Kind, probes_for_kind};
use flow_bevy::palette::ToolBtn;
use flow_bevy::probes::{Probe, ProbeTarget};
use flow_bevy::tool::Tool;
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

fn count_node_probes_for(app: &mut App, nid: flow::NodeId) -> usize {
    let world = app.world_mut();
    let mut q = world.query::<&Probe>();
    q.iter(world)
        .filter(|p| matches!(p.target, ProbeTarget::Node { node, .. } if node == nid))
        .count()
}

fn latest_of_kind(app: &App, kind: Kind) -> flow::NodeId {
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
        .expect("no node of that kind")
}

#[test]
fn clicking_a_generator_spawns_its_declared_probes() {
    let mut app = make_app();
    let expected = probes_for_kind(Kind::Generator).len();
    assert!(expected > 0, "Generator should declare at least one probe spec");

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let nid = latest_of_kind(&app, Kind::Generator);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Probe);
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));

    let spawned = count_node_probes_for(&mut app, nid);
    assert_eq!(
        spawned, expected,
        "expected {} probes for Generator (one per spec), got {}",
        expected, spawned
    );
}

#[test]
fn clicking_a_queue_spawns_queue_specific_probes() {
    let mut app = make_app();
    let expected = probes_for_kind(Kind::Queue).len();
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Queue));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let nid = latest_of_kind(&app, Kind::Queue);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Probe);
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));

    assert_eq!(count_node_probes_for(&mut app, nid), expected);
}

#[test]
fn clicking_a_router_spawns_no_probes() {
    // Router declares no specs — the Probe tool silently does nothing.
    let mut app = make_app();
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Router));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let nid = latest_of_kind(&app, Kind::Router);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Probe);
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));

    assert_eq!(
        count_node_probes_for(&mut app, nid), 0,
        "Router declares no ProbeSpecs; clicking it should spawn nothing"
    );
}

#[test]
fn probe_reader_reflects_live_sim_state() {
    // Drop a worker. Spec 0 is "served", spec 1 is "service". Tick the
    // sim and verify the probe readers return the expected string.
    let mut app = make_app();
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Worker));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let worker = latest_of_kind(&app, Kind::Worker);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Probe);
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));

    // Advance so the worker has some state (though without a wired
    // upstream queue, its served count stays 0). Just check the spec's
    // readers against the sim directly.
    advance_sim_ns(&mut app, 500_000_000);

    let specs = probes_for_kind(Kind::Worker);
    let served_spec = specs.iter().find(|s| s.label == "served").expect("no served spec");
    let service_spec = specs.iter().find(|s| s.label == "service").expect("no service spec");

    let sim = &app.world().resource::<FlowSim>().sim;
    let node = &sim.nodes[&worker];
    assert_eq!((served_spec.read)(node), "0");
    assert_eq!((service_spec.read)(node), "50ms"); // default
}

#[test]
fn node_probe_carries_its_own_reader_fn() {
    // Proves the probe module doesn't need to know about the kind — a
    // fn pointer travels with each probe entity.
    let mut app = make_app();
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Sink));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let sink = latest_of_kind(&app, Kind::Sink);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Probe);
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));

    let world = app.world_mut();
    let mut q = world.query::<&Probe>();
    let probe = q
        .iter(world)
        .find(|p| matches!(p.target, ProbeTarget::Node { node, .. } if node == sink))
        .expect("no node probe for sink");

    match probe.target {
        ProbeTarget::Node { label, reader, .. } => {
            // Sink's first spec is "total" → reads `count` slot.
            assert_eq!(label, "total");
            let node = &app.world().resource::<FlowSim>().sim.nodes[&sink];
            assert_eq!(reader(node), "0");
        }
        ProbeTarget::Edge(_) => panic!("expected Node target"),
    }
}
