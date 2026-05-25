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
    // it should be > 0. Use read_slot_resolved so this works whether
    // `client` is a monolithic node or a compound shim (the slot lives
    // on `client::Endpoint` for the composite version).
    let completed = match app.world().resource::<FlowSim>().read_slot_resolved(client, "completed") {
        Some(Value::Int(i)) => *i,
        _ => 0,
    };
    assert!(
        completed > 0,
        "client should have received some response packets, got {}",
        completed
    );
}

#[test]
fn worker_waits_service_ns_before_responding() {
    // The Worker must take `service_ns` to produce a resp — not reply
    // instantly. Regression for a bug where `serve` emitted resp
    // directly on req, so RTT was effectively 2 * edge_latency. The
    // fix routes through a `done_req` self-emit so the self-edge
    // latency (= service_ns) gates the reply.
    use common::{spawn_node, wire};
    use flow_bevy::gadgets::Kind;

    let mut app = make_app();
    let client = spawn_node(&mut app, Kind::Client, 0, "Client_test");
    let worker = spawn_node(&mut app, Kind::Worker, 0, "Worker_test");
    wire(&mut app, client, Kind::Client, worker, Kind::Worker);

    // Slow worker (200ms service) vs fast client (10ms period). With
    // the bug, a resp lands in a few ms. With the fix, the first resp
    // can't land before ~200ms (service) + a couple of 1ms edge hops.
    // write_slot_resolved walks into composite children so this works
    // whether worker/client are monolithic or compound shims.
    {
        let world = app.world_mut();
        let mut flow = world.resource_mut::<FlowSim>();
        flow.write_slot_resolved(worker, "service_ns", flow::Value::Int(200_000_000));
        flow.write_slot_resolved(client, "period_ns", flow::Value::Int(10_000_000));
    }

    // t = 50ms: well before service_ns, so nothing should have
    // completed yet.
    advance_sim_ns(&mut app, 50_000_000);
    let completed_early = match app.world().resource::<FlowSim>().read_slot_resolved(client, "completed") {
        Some(flow::Value::Int(i)) => *i,
        _ => -1,
    };
    assert_eq!(
        completed_early, 0,
        "no resp should have landed yet at t=50ms (service_ns=200ms), got completed={}",
        completed_early
    );

    // t = 500ms: comfortably past service_ns, at least one resp should
    // have made it back.
    advance_sim_ns(&mut app, 450_000_000);
    let completed_late = match app.world().resource::<FlowSim>().read_slot_resolved(client, "completed") {
        Some(flow::Value::Int(i)) => *i,
        _ => -1,
    };
    assert!(
        completed_late > 0,
        "expected completed > 0 by t=500ms, got {}",
        completed_late
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
