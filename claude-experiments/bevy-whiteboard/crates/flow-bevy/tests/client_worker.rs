//! Connecting a Client to a Worker should "just work" — the user draws
//! a single arrow Client→Worker, and the worker's `Respond` packets reach
//! the client without crashing the sim.
//!
//! Regression test for the panic the user hit:
//!   "Respond: no outbound edge from responder back to requester"

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow::Value;
use flow_bevy::bridge::FlowSim;
use flow_bevy::gadgets::Kind;
use flow_bevy::palette::ToolBtn;
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

#[test]
fn client_to_worker_does_not_panic() {
    // Reproduces the user's report. Drop a client + a worker, connect
    // client → worker, run sim. Should not panic; client should observe
    // some completed responses.
    let mut app = make_app();

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Client));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -300.0));
    let client = latest_of_kind(&app, Kind::Client);
    let client_xy = Vec2::new(-300.0, -300.0);

    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Worker));
    simulate_canvas_click(&mut app, Vec2::new(0.0, -300.0));
    let worker_xy = Vec2::new(0.0, -300.0);

    // Connect client → worker. The auto-wire creates a hidden response
    // edge worker → client so the worker's `Respond` has somewhere to
    // route the resp packet.
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Connect);
    simulate_canvas_click(&mut app, client_xy);
    simulate_canvas_click(&mut app, worker_xy);

    // Run the sim; this used to panic. Now it should advance cleanly.
    advance_sim_ns(&mut app, 3_000_000_000);

    // Client tracks `completed` responses. With auto-wired return path
    // it should be > 0.
    let completed = match &app.world().resource::<FlowSim>().sim.nodes[&client].slots["completed"] {
        Value::Int(i) => *i,
        _ => 0,
    };
    assert!(
        completed > 0,
        "client should have received some response packets, got {}",
        completed
    );
}

#[test]
fn client_only_no_worker_does_not_panic() {
    // Even without a worker on the other end (so no Respond happens),
    // the sim shouldn't crash. The client just sees no responses.
    let mut app = make_app();
    click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Client));
    simulate_canvas_click(&mut app, Vec2::new(-300.0, -300.0));

    // No connection drawn — client emits requests into the void.
    advance_sim_ns(&mut app, 1_000_000_000);
    // Reaching here without panicking is the assertion.
}
