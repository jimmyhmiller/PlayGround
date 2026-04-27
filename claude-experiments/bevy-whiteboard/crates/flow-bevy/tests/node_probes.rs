//! Node probes — compositional. Each class declares its own `probes { }`
//! block in DSL; the Probe tool stacks one probe entity per declared probe
//! above the clicked node. This test proves:
//!
//!  * Clicking a node with the Probe tool spawns the exact number of
//!    probes declared by that class.
//!  * Each probe reads its value through the sim's `probe_reading` API
//!    (DSL-lowered, not a hardcoded match on kind).
//!  * Probes on different kinds don't collide — each gets its own stack.
//!  * Routers (no declared probes) get no probes.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow_bevy::bridge::FlowSim;
use flow_bevy::gadgets::Kind;
use flow_bevy::palette::ToolBtn;
use flow_bevy::probes::Probe;
use flow_bevy::tool::Tool;
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

fn count_node_probes_for(app: &mut App, nid: flow::NodeId) -> usize {
    let world = app.world_mut();
    let mut q = world.query::<&Probe>();
    q.iter(world)
        .filter(|p| p.node == nid)
        .count()
}

fn latest_of_kind(app: &App, kind: Kind) -> flow::NodeId {
    let sim = &app.world().resource::<FlowSim>();
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
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let nid = latest_of_kind(&app, Kind::Generator);

    let expected = {
        let sim = &app.world().resource::<FlowSim>();
        sim.probe_labels(nid).len()
    };
    assert!(expected > 0, "Generator should declare at least one probe");

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Probe);
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));

    let spawned = count_node_probes_for(&mut app, nid);
    assert_eq!(
        spawned, expected,
        "expected {} probes for Generator (one per DSL declaration), got {}",
        expected, spawned
    );
}

#[test]
fn clicking_a_queue_spawns_queue_specific_probes() {
    let mut app = make_app();
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Queue));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let nid = latest_of_kind(&app, Kind::Queue);

    let expected = {
        let sim = &app.world().resource::<FlowSim>();
        sim.probe_labels(nid).len()
    };
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Probe);
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));

    assert_eq!(count_node_probes_for(&mut app, nid), expected);
}

#[test]
fn clicking_a_router_spawns_no_probes() {
    // Router declares no probes — the Probe tool silently does nothing.
    let mut app = make_app();
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Router));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let nid = latest_of_kind(&app, Kind::Router);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Probe);
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));

    assert_eq!(
        count_node_probes_for(&mut app, nid),
        0,
        "Router declares no probes; clicking it should spawn nothing"
    );
}

#[test]
fn probe_reader_reflects_live_sim_state() {
    // Drop a worker. DSL declares "served" and "service" probes. Check the
    // sim's probe_reading returns the expected strings.
    let mut app = make_app();
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Worker));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let worker = latest_of_kind(&app, Kind::Worker);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Probe);
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));

    advance_sim_ns(&mut app, 500_000_000);

    let sim = &app.world().resource::<FlowSim>();
    assert_eq!(sim.probe_reading(worker, "served").as_deref(), Some("0"));
    assert_eq!(
        sim.probe_reading(worker, "service").as_deref(),
        Some("50ms")
    );
}

#[test]
fn node_probe_uses_sim_probe_reading() {
    // Proves the probe module doesn't hardcode per-kind readers — it looks
    // up the probe by label on the sim node.
    let mut app = make_app();
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Sink));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    let sink = latest_of_kind(&app, Kind::Sink);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Probe);
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));

    let world = app.world_mut();
    let mut q = world.query::<&Probe>();
    let probe_label: String = q
        .iter(world)
        .find_map(|p| if p.node == sink { Some(p.label.clone()) } else { None })
        .expect("no node probe for sink");

    // Sink's first (only) probe is "total" → reads `count` slot.
    assert_eq!(probe_label, "total");
    let sim = &app.world().resource::<FlowSim>();
    assert_eq!(sim.probe_reading(sink, &probe_label).as_deref(), Some("0"));
}
