//! BackoffClient — every `resp_error` extends the next-tick delay by
//! doubling `backoff_ns` (seeded from `period_ns`, capped at
//! `max_backoff_ns`). A successful `resp` resets it to 0.
//!
//! The demo this crate is working toward: many BackoffClients hit a
//! shared down Worker, come back into sync at 1s, 2s, 4s, 8s, and
//! thunder the worker the moment it comes back up. These tests pin
//! the single-client mechanism that thundering herd is built on.

mod common;

use common::{advance_sim_ns, make_app, spawn_node, wire};
use flow::Value;
use flow_bevy::bridge::FlowSim;
use flow_bevy::gadgets::Kind;

fn slot_int(app: &bevy::prelude::App, nid: flow::NodeId, key: &str) -> i64 {
    match app.world().resource::<FlowSim>().nodes[&nid].slots.get(key) {
        Some(Value::Int(i)) => *i,
        _ => 0,
    }
}

fn set_slot_int(app: &mut bevy::prelude::App, nid: flow::NodeId, key: &str, v: i64) {
    app.world_mut()
        .resource_mut::<FlowSim>()
        .nodes
        .get_mut(&nid)
        .unwrap()
        .slots
        .insert(key.into(), Value::Int(v));
}

#[test]
fn backoff_ns_doubles_on_successive_resp_errors() {
    let mut app = make_app();
    let client = spawn_node(&mut app, Kind::BackoffClient, 0, "BackoffClient_test");
    let worker = spawn_node(&mut app, Kind::Worker, 0, "Worker_test");
    wire(&mut app, client, Kind::BackoffClient, worker, Kind::Worker);

    // Take the worker down so it replies with resp_error to every req.
    set_slot_int(&mut app, worker, "up", 0);

    let period = slot_int(&app, client, "period_ns");
    let max_backoff = slot_int(&app, client, "max_backoff_ns");
    assert!(period > 0 && max_backoff > 0);

    // Precondition: no failures yet.
    assert_eq!(slot_int(&app, client, "backoff_ns"), 0);

    // Drive the sim long enough for many ticks even as backoff grows.
    // Budget > max_backoff so we definitely hit the cap.
    advance_sim_ns(&mut app, (max_backoff as u64) * 4 + 2_000_000_000);

    let failed = slot_int(&app, client, "failed");
    let backoff = slot_int(&app, client, "backoff_ns");

    // At least a few failures have accumulated — enough to have doubled
    // past the cap and clamped.
    assert!(failed >= 4, "expected several failures, got {}", failed);

    // Backoff must equal the cap (we ran long enough to saturate), or
    // a power-of-two multiple of period that's <= max. Either way, it
    // should not be zero (no success ever landed).
    assert!(backoff > 0, "backoff_ns should be > 0 after failures");
    assert!(
        backoff <= max_backoff,
        "backoff_ns {} must not exceed cap {}",
        backoff,
        max_backoff
    );
    assert_eq!(
        backoff, max_backoff,
        "with enough failures backoff should saturate at the cap"
    );
}

#[test]
fn backoff_client_rtt_honours_worker_service_ns() {
    // The sim-level guarantee: when a BackoffClient talks to a healthy
    // Worker, the reply must not arrive earlier than service_ns after
    // the request. If this ever passes trivially (completed > 0 at
    // t < service_ns) the `serve` → `done_req` → `finish_req` chain
    // has collapsed back to an instant reply.
    let mut app = make_app();
    let client = spawn_node(&mut app, Kind::BackoffClient, 0, "BackoffClient_test");
    let worker = spawn_node(&mut app, Kind::Worker, 0, "Worker_test");
    wire(&mut app, client, Kind::BackoffClient, worker, Kind::Worker);

    // Slow worker (500ms), fast client (20ms period). The first req is
    // injected at sim-t=0 by the gadget constructor, so the earliest a
    // resp could legally land is service_ns + tiny edge hops ≈ 502ms.
    set_slot_int(&mut app, worker, "service_ns", 500_000_000);
    set_slot_int(&mut app, client, "period_ns", 20_000_000);

    // Probe several points inside the service window — completed must
    // stay at 0 the whole time.
    for t_ns in [100_000_000u64, 250_000_000, 490_000_000] {
        // Each iteration advances to an absolute sim time. Compute the
        // delta from now so `advance_sim_ns` stays monotonic.
        let now = app.world().resource::<FlowSim>().now_ns;
        if t_ns > now {
            advance_sim_ns(&mut app, t_ns - now);
        }
        let completed = slot_int(&app, client, "completed");
        assert_eq!(
            completed, 0,
            "at sim-t={}ns (< 500ms service), completed must be 0, got {}",
            t_ns, completed
        );
    }

    // Past the service window — at least one resp should have landed.
    advance_sim_ns(&mut app, 200_000_000);
    let completed = slot_int(&app, client, "completed");
    assert!(
        completed > 0,
        "expected completed > 0 past the service window, got {}",
        completed
    );
}

#[test]
fn backoff_resets_on_successful_resp() {
    let mut app = make_app();
    let client = spawn_node(&mut app, Kind::BackoffClient, 0, "BackoffClient_test");
    let worker = spawn_node(&mut app, Kind::Worker, 0, "Worker_test");
    wire(&mut app, client, Kind::BackoffClient, worker, Kind::Worker);

    // Start with worker down: accumulate some backoff.
    set_slot_int(&mut app, worker, "up", 0);
    advance_sim_ns(&mut app, 3_000_000_000);
    let backoff_after_failures = slot_int(&app, client, "backoff_ns");
    assert!(
        backoff_after_failures > 0,
        "expected backoff_ns > 0 after failures, got {}",
        backoff_after_failures
    );

    // Bring the worker back up. The next tick's resp will reset backoff.
    set_slot_int(&mut app, worker, "up", 1);
    // One full backoff window plus service time + a safety margin
    // guarantees at least one successful round trip.
    advance_sim_ns(
        &mut app,
        (backoff_after_failures as u64) + 3_000_000_000,
    );

    assert_eq!(
        slot_int(&app, client, "backoff_ns"),
        0,
        "backoff_ns should reset to 0 after a successful resp"
    );
    assert!(
        slot_int(&app, client, "completed") > 0,
        "expected at least one completed resp after worker back up"
    );
}
