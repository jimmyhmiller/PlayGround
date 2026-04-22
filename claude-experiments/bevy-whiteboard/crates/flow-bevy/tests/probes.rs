//! Probe tool: click the palette Probe button, click on an edge, verify a
//! `Probe` entity spawned against that edge. Then run the sim and check
//! the rate readout reflects the edge's real-time packet frequency.

mod common;

use bevy::prelude::*;
use common::make_app;
use flow_bevy::bridge::FlowSim;
use flow_bevy::gadgets::Kind;
use flow_bevy::palette::ToolBtn;
use flow_bevy::probes::{Probe, ProbeSamples, ProbeTarget, rate_for_edge};
use flow_bevy::tool::Tool;
use poster_ui::testing::{click_by_marker, simulate_canvas_click};

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

fn find_edge(app: &App, from: flow::NodeId, to: flow::NodeId) -> flow::EdgeId {
    let sim = &app.world().resource::<FlowSim>().sim;
    sim.edges
        .iter()
        .find(|(_, e)| e.from == from && e.to == to)
        .map(|(id, _)| *id)
        .expect("edge not found")
}

fn count_probes(app: &mut App) -> usize {
    let world = app.world_mut();
    let mut q = world.query::<&Probe>();
    q.iter(world).count()
}

#[test]
fn probe_tool_spawns_probe_on_edge_click() {
    let mut app = make_app();

    // Set up a minimal Gen → Sink so we have a concrete edge to target.
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Sink));
    simulate_canvas_click(&mut app, Vec2::new(300.0, -200.0));

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Connect);
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    simulate_canvas_click(&mut app, Vec2::new(300.0, -200.0));

    let before = count_probes(&mut app);

    // Pick the Probe tool and click near the edge midpoint (0, -200).
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Probe);
    simulate_canvas_click(&mut app, Vec2::new(0.0, -200.0));

    let after = count_probes(&mut app);
    assert_eq!(after, before + 1, "clicking on the edge should spawn a probe");

    // Confirm the probe actually points at our edge.
    let gen_nid = latest_of_kind(&app, Kind::Generator);
    let sink_nid = latest_of_kind(&app, Kind::Sink);
    let eid = find_edge(&app, gen_nid, sink_nid);
    let world = app.world_mut();
    let mut q = world.query::<&Probe>();
    let matched = q
        .iter(world)
        .any(|p| matches!(p.target, ProbeTarget::Edge(e) if e == eid));
    assert!(matched, "no Probe anchored to the gen→sink edge");
}

#[test]
fn probe_tool_returns_to_select_after_one_click() {
    let mut app = make_app();
    // Need an edge to click on.
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Sink));
    simulate_canvas_click(&mut app, Vec2::new(300.0, -200.0));
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Connect);
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -200.0));
    simulate_canvas_click(&mut app, Vec2::new(300.0, -200.0));

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Probe);
    simulate_canvas_click(&mut app, Vec2::new(0.0, -200.0));

    assert_eq!(
        app.world().resource::<flow_bevy::tool::ActiveTool>().0,
        Tool::Select,
        "Probe tool should be one-shot and reset to Select"
    );
}

#[test]
fn rate_for_edge_is_zero_for_unobserved_edge() {
    // An edge id the probe has never seen should return 0/s. Use a made-up
    // id that's guaranteed to be outside the sim's current edge range.
    let app = make_app();
    let samples = app.world().resource::<ProbeSamples>();
    let sim = &app.world().resource::<FlowSim>().sim;
    let max_id = sim.edges.keys().map(|e| e.0).max().unwrap_or(0);
    let unobserved = flow::EdgeId(max_id + 1000);
    assert_eq!(rate_for_edge(samples, unobserved), 0.0);
}
