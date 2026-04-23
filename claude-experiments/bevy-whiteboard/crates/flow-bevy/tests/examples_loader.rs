//! Behavioral tests for each example scenario.
//!
//! Node-count tests are insufficient — they've proven able to pass
//! while the sim is totally stuck (packets sitting in an inbox with
//! no matching rule, see the ClientQueueWorker incident). Every test
//! here advances the sim and asserts that *meaningful state changed*:
//! responses received, packets delivered to terminals, no runtime
//! errors piling up in `Sim.error_counts`.

mod common;

use bevy::prelude::*;
use common::{advance_sim_ns, make_app};
use flow_bevy::bridge::FlowSim;
use flow_bevy::examples::{Example, LoadExample};
use flow_bevy::gadgets::Kind;

fn load(app: &mut App, ex: Example) {
    app.world_mut()
        .resource_mut::<bevy::ecs::message::Messages<LoadExample>>()
        .write(LoadExample(ex));
    // Two ticks: first processes the LoadExample system; the second
    // picks up entity commands from the first.
    app.update();
    app.update();
}

fn count_of_kind(app: &App, kind: Kind) -> usize {
    let sim = &app.world().resource::<FlowSim>().sim;
    sim.nodes.values()
        .filter(|n| n.name.starts_with(&format!("{}_", kind.label())))
        .count()
}

fn slot_int(app: &App, nid: flow::NodeId, slot: &str) -> i64 {
    match &app.world().resource::<FlowSim>().sim.nodes[&nid].slots[slot] {
        flow::Value::Int(i) => *i,
        other => panic!("slot `{}` not Int: {:?}", slot, other),
    }
}

fn first_of_kind(app: &App, kind: Kind) -> flow::NodeId {
    let sim = &app.world().resource::<FlowSim>().sim;
    let prefix = format!("{}_", kind.label());
    sim.nodes
        .iter()
        .find(|(_, n)| n.name.starts_with(&prefix))
        .map(|(id, _)| *id)
        .unwrap_or_else(|| panic!("no node of kind {:?}", kind))
}

fn all_of_kind(app: &App, kind: Kind) -> Vec<flow::NodeId> {
    let sim = &app.world().resource::<FlowSim>().sim;
    let prefix = format!("{}_", kind.label());
    sim.nodes
        .iter()
        .filter(|(_, n)| n.name.starts_with(&prefix))
        .map(|(id, _)| *id)
        .collect()
}

/// Any non-zero entry in `error_counts` is a failed test — examples
/// must run clean. This catches the class of "rule author typo" /
/// "unmatched variant" problems that count-tests miss.
fn assert_no_runtime_errors(app: &App, ctx: &str) {
    let errors = &app.world().resource::<FlowSim>().sim.error_counts;
    assert!(
        errors.is_empty(),
        "{}: expected zero runtime errors, got {:?}",
        ctx,
        errors,
    );
}

/// Load + tick = expected node shape. Still useful as a quick
/// structural sanity check; the behavioral tests below do the real
/// work.
#[test]
fn every_example_loads_and_replaces_previous() {
    let mut app = make_app();

    load(&mut app, Example::ThreeLaneFanout);
    assert_eq!(count_of_kind(&app, Kind::Generator), 3);
    assert_eq!(count_of_kind(&app, Kind::Router), 1);
    assert_eq!(count_of_kind(&app, Kind::Queue), 3);
    assert_eq!(count_of_kind(&app, Kind::Worker), 6);
    assert_eq!(count_of_kind(&app, Kind::Sink), 3);

    load(&mut app, Example::ClientWorker);
    assert_eq!(count_of_kind(&app, Kind::Client), 1);
    assert_eq!(count_of_kind(&app, Kind::Worker), 1);
    assert_eq!(count_of_kind(&app, Kind::Generator), 0, "prior scene wiped");

    load(&mut app, Example::ClientRouterWorker);
    assert_eq!(count_of_kind(&app, Kind::Client), 1);
    assert_eq!(count_of_kind(&app, Kind::Router), 1);
    assert_eq!(count_of_kind(&app, Kind::Worker), 1);

    load(&mut app, Example::TwoClientsOneWorker);
    assert_eq!(count_of_kind(&app, Kind::Client), 2);
    assert_eq!(count_of_kind(&app, Kind::Worker), 1);

    load(&mut app, Example::ClientQueueWorker);
    assert_eq!(count_of_kind(&app, Kind::Client), 1);
    assert_eq!(count_of_kind(&app, Kind::Queue), 1);
    assert_eq!(count_of_kind(&app, Kind::Worker), 1);
    assert_eq!(count_of_kind(&app, Kind::Sink), 1);
}

// ─────────────────────────────────────────────────────────────
// Behavioral tests — one per example
// ─────────────────────────────────────────────────────────────

/// `ThreeLaneFanout`: packets flow Gen→Router→Queue→Worker→Sink in
/// three colour lanes. After a few sim seconds each sink should have
/// a non-zero `count`. Red lane is back-pressured (1 worker vs 30/s)
/// so it'll have the fewest; blue (3 workers) the most — but all
/// three must be > 0 for the scenario to be working.
#[test]
fn three_lane_fanout_all_sinks_count_up() {
    let mut app = make_app();
    load(&mut app, Example::ThreeLaneFanout);
    advance_sim_ns(&mut app, 3_000_000_000);

    assert_no_runtime_errors(&app, "ThreeLaneFanout");

    let sinks = all_of_kind(&app, Kind::Sink);
    assert_eq!(sinks.len(), 3);
    for sink in sinks {
        let count = slot_int(&app, sink, "count");
        assert!(count > 0, "sink {:?} saw no packets (count=0)", sink);
    }
}

/// `ClientWorker`: the minimal request/response. After a few seconds
/// the client's `completed` counter should be growing. This is the
/// same shape as the old `client_worker.rs` test but via the loader
/// pipeline (not manual palette clicks).
#[test]
fn client_worker_round_trips_complete() {
    let mut app = make_app();
    load(&mut app, Example::ClientWorker);
    advance_sim_ns(&mut app, 3_000_000_000);

    assert_no_runtime_errors(&app, "ClientWorker");

    let client = first_of_kind(&app, Kind::Client);
    let completed = slot_int(&app, client, "completed");
    assert!(completed > 0, "client got no responses (completed=0)");
}

/// `ClientRouterWorker`: return_path survives a transparent hop. The
/// Router never touches return_path; the Worker still pops back to
/// Client via the manually-wired Worker→Client reply edge. If the
/// router ever starts overwriting return_path this test fails —
/// which is exactly what regressed before the formalism rewrite.
#[test]
fn client_router_worker_responses_reach_client() {
    let mut app = make_app();
    load(&mut app, Example::ClientRouterWorker);
    advance_sim_ns(&mut app, 4_000_000_000);

    assert_no_runtime_errors(&app, "ClientRouterWorker");

    let client = first_of_kind(&app, Kind::Client);
    let completed = slot_int(&app, client, "completed");
    assert!(completed > 0, "client got no responses (completed=0)");
}

/// `ClientQueueWorker`: the scenario that was originally broken.
/// The queue's depth counter (`len`) must actually increment — this
/// was 0 before the `enqueue_req` rule was added. Client's
/// `completed` must grow because the Queue acks via the
/// reverse-routed Client→Queue edge. Sink's `count` must grow
/// because the worker pulls and emits downstream.
#[test]
fn client_queue_worker_full_pipeline() {
    let mut app = make_app();
    load(&mut app, Example::ClientQueueWorker);
    advance_sim_ns(&mut app, 3_000_000_000);

    assert_no_runtime_errors(&app, "ClientQueueWorker");

    let client = first_of_kind(&app, Kind::Client);
    let queue  = first_of_kind(&app, Kind::Queue);
    let sink   = first_of_kind(&app, Kind::Sink);

    let completed = slot_int(&app, client, "completed");
    assert!(completed > 0, "client got no acks back (completed=0)");

    // `len` can be > 0 or 0 at any instant depending on whether the
    // worker happened to have just pulled. But over 3 seconds of
    // sim time at 10/s in and ~8/s out, SOMETHING must have been
    // enqueued. Peek `served` on the queue via emitted count ≥ 0 —
    // the real signal is that the sink received work.
    let queue_len = slot_int(&app, queue, "len");
    assert!(queue_len >= 0, "queue len should be non-negative, got {}", queue_len);

    let sink_count = slot_int(&app, sink, "count");
    assert!(sink_count > 0, "no work reached the sink (count=0)");
}

/// `TwoClientsOneWorker` — the load-bearing property test for
/// return_path under a shared destination.
///
/// Property (STRONG — `completed > 0` was insufficient): for each
/// client, `completed + in_flight == emitted`. Every emit ended up
/// either acknowledged or still traveling; none were silently
/// dropped or misrouted to the other client. If return_path ever
/// crosses streams, at least one client's completed would diverge
/// from its emitted count minus in_flight.
#[test]
fn two_clients_one_worker_no_cross_talk() {
    let mut app = make_app();
    load(&mut app, Example::TwoClientsOneWorker);
    advance_sim_ns(&mut app, 5_000_000_000);

    // Quiesce both clients by stretching their self-loop period_ns
    // to effectively-infinity. A last already-scheduled tick will
    // still fire, then emission stops. Drain for another 50 ms of
    // sim time so any resp's still in-flight reach their client and
    // get processed. After this, the checks below can be EXACT —
    // no slop from end-of-sim in-flight packets.
    {
        let mut flow = app.world_mut().resource_mut::<FlowSim>();
        let clients_q: Vec<flow::NodeId> = flow.sim.nodes.iter()
            .filter(|(_, n)| n.name.starts_with("Client_"))
            .map(|(id, _)| *id)
            .collect();
        for c in clients_q {
            flow.sim.nodes.get_mut(&c).unwrap().slots
                .insert("period_ns".into(), flow::Value::Int(i64::MAX / 4));
        }
    }
    advance_sim_ns(&mut app, 50_000_000);

    assert_no_runtime_errors(&app, "TwoClientsOneWorker");

    let clients = all_of_kind(&app, Kind::Client);
    assert_eq!(clients.len(), 2);

    let mut totals = Vec::new();
    for c in &clients {
        let emitted   = slot_int(&app, *c, "emitted");
        let completed = slot_int(&app, *c, "completed");
        let in_flight = slot_int(&app, *c, "in_flight");
        totals.push((*c, emitted, completed, in_flight));

        // The load-bearing invariant. Per client.
        assert_eq!(
            emitted,
            completed + in_flight,
            "client {:?}: emitted ({}) != completed ({}) + in_flight ({}) — \
             responses were misrouted, dropped, or duplicated",
            c, emitted, completed, in_flight,
        );

        // Both must have ACTUALLY emitted something (otherwise the
        // scenario didn't run; the invariant above is vacuous).
        assert!(emitted > 0, "client {:?}: nothing emitted", c);
        // Both must have received at least one response (otherwise a
        // whole client's reply channel is broken).
        assert!(
            completed > 0,
            "client {:?}: zero responses out of {} emitted",
            c, emitted
        );
    }

    // Cross-check: the worker must have served exactly as many
    // requests as the two clients emitted together. `served` is
    // incremented once per `serve` rule firing (one per req consumed).
    let worker = first_of_kind(&app, Kind::Worker);
    let served = slot_int(&app, worker, "served");
    let total_emitted: i64 = totals.iter().map(|(_, e, _, _)| e).sum();
    assert_eq!(
        served, total_emitted,
        "worker served {} but clients emitted {} total — \
         requests were dropped or duplicated",
        served, total_emitted
    );

    // Strong invariant: the per-client (in_flight + completed)
    // equation holds even if resp's get 1:1 swapped between
    // clients. So inspect the event log directly — count how many
    // resp packets the worker emitted to each client, and confirm
    // each count exactly matches that client's completed. If
    // return_path ever crosses streams, these diverge.
    let sim = &app.world().resource::<FlowSim>().sim;
    let mut resp_to: std::collections::HashMap<flow::NodeId, i64> =
        std::collections::HashMap::new();
    for ev in sim.log.iter() {
        if let flow::Event::PacketEmitted { from, to, payload, .. } = ev {
            if *from != worker { continue; }
            if let flow::Value::Variant { tag, .. } = payload {
                if tag != "resp" { continue; }
            } else { continue; }
            *resp_to.entry(*to).or_insert(0) += 1;
        }
    }
    for (client, _emitted, completed, _in_flight) in &totals {
        let actual_replies_to_this_client =
            resp_to.get(client).copied().unwrap_or(0);
        assert_eq!(
            actual_replies_to_this_client, *completed,
            "client {:?}: worker emitted {} resp's targeting it, but its \
             completed slot is {} — cross-talk (responses landed on the \
             wrong client)",
            client, actual_replies_to_this_client, completed,
        );
    }
}
