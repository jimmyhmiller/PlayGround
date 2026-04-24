//! Round-robin router coverage. Two questions worth answering:
//!
//!  1. With `mode = 1`, N packets spread evenly over M colour-matching
//!     workers — not all onto the first one, not skewed by same-tick
//!     emissions. This is the "does argmin + edge_last_sent actually
//!     rotate?" test.
//!
//!  2. When traffic mixes colours through the same Router, each colour
//!     lane round-robins *independently*. This is the test that proves
//!     we didn't need a per-colour abstraction — edge-scoped state
//!     already partitions correctly because edges are colour-typed by
//!     the Worker they wire to.
//!
//! Both tests inject packets directly at the Router so the Generator's
//! scheduling cadence isn't under test here. Worker service_ns is left
//! at the template default; we assert on `PacketEmitted` edges leaving
//! the Router, not on work completion downstream.

mod common;

use common::{advance_sim_ns, make_app, spawn_node, wire};
use flow::Event;
use flow::Value;
use flow_bevy::bridge::FlowSim;
use flow_bevy::gadgets::Kind;

/// Count `PacketEmitted` events whose `from` is the router and `to` is
/// each given worker. This is the per-edge delivery count out of the
/// router, which is exactly what round-robin controls.
fn router_emits_by_worker(app: &bevy::prelude::App, router: flow::NodeId, workers: &[flow::NodeId]) -> Vec<usize> {
    let sim = &app.world().resource::<FlowSim>().sim;
    workers.iter().map(|w| {
        sim.log.iter().filter(|ev| matches!(ev,
            Event::PacketEmitted { from, to, .. } if *from == router && *to == *w
        )).count()
    }).collect()
}

fn set_slot_int(app: &mut bevy::prelude::App, nid: flow::NodeId, slot: &str, v: i64) {
    let mut flow_res = app.world_mut().resource_mut::<FlowSim>();
    flow_res.sim.nodes.get_mut(&nid).expect("node").slots.insert(slot.into(), Value::Int(v));
}

#[test]
fn round_robin_distributes_evenly_across_same_color_workers() {
    let mut app = make_app();

    // Three workers on the same colour lane (slot 0 = red). Wiring order
    // determines edge-id order, which is also the argmin tie-break on
    // the first few emissions.
    let router = spawn_node(&mut app, Kind::Router, 0, "Router_rr");
    let w0 = spawn_node(&mut app, Kind::Worker, 0, "Worker_r0");
    let w1 = spawn_node(&mut app, Kind::Worker, 0, "Worker_r1");
    let w2 = spawn_node(&mut app, Kind::Worker, 0, "Worker_r2");
    wire(&mut app, router, Kind::Router, w0, Kind::Worker);
    wire(&mut app, router, Kind::Router, w1, Kind::Worker);
    wire(&mut app, router, Kind::Router, w2, Kind::Worker);

    set_slot_int(&mut app, router, "mode", 1);

    // Inject 9 red packets. With 3 workers that's exactly 3 per worker
    // if round-robin is working. If it degenerates to "always first"
    // (the same-sim-tick tie bug), w0 gets all 9 and the others get 0.
    {
        let mut flow_res = app.world_mut().resource_mut::<FlowSim>();
        for _ in 0..9 {
            flow_res.sim.inject(router, Value::variant("packet", Value::Int(0)));
        }
    }
    advance_sim_ns(&mut app, 10_000_000); // 10 ms — plenty for latency + rule fires.

    let counts = router_emits_by_worker(&app, router, &[w0, w1, w2]);
    assert_eq!(counts, vec![3, 3, 3],
        "round-robin should distribute 9 packets evenly; got {:?}", counts);
}

#[test]
fn round_robin_per_color_lanes_dont_interfere() {
    let mut app = make_app();

    // Two red workers + two blue workers. The router sees both colours
    // but each worker is mono-colour, so the router's out-edges are
    // naturally partitioned by colour.
    let router = spawn_node(&mut app, Kind::Router, 0, "Router_mix");
    let r0 = spawn_node(&mut app, Kind::Worker, 0, "Worker_r0");
    let r1 = spawn_node(&mut app, Kind::Worker, 0, "Worker_r1");
    let b0 = spawn_node(&mut app, Kind::Worker, 1, "Worker_b0");
    let b1 = spawn_node(&mut app, Kind::Worker, 1, "Worker_b1");
    wire(&mut app, router, Kind::Router, r0, Kind::Worker);
    wire(&mut app, router, Kind::Router, r1, Kind::Worker);
    wire(&mut app, router, Kind::Router, b0, Kind::Worker);
    wire(&mut app, router, Kind::Router, b1, Kind::Worker);

    set_slot_int(&mut app, router, "mode", 1);

    // Interleave red/blue packets. The key property: a red packet bumps
    // edge_last_sent on a red-wired edge only — blue edges' counters
    // are unaffected. So the next red packet still sees its OWN lane
    // in the same state ("other red worker is older") and alternates
    // properly, not getting knocked out of rotation by the blue in
    // between.
    //
    // 6 reds + 6 blues = 3 per worker if per-colour rotation works.
    // If colour lanes interfered (e.g. a shared counter), the
    // distribution skews — some worker gets 0 or 6.
    {
        let mut flow_res = app.world_mut().resource_mut::<FlowSim>();
        for _ in 0..6 {
            flow_res.sim.inject(router, Value::variant("packet", Value::Int(0)));
            flow_res.sim.inject(router, Value::variant("packet", Value::Int(1)));
        }
    }
    advance_sim_ns(&mut app, 10_000_000);

    let counts = router_emits_by_worker(&app, router, &[r0, r1, b0, b1]);
    assert_eq!(counts, vec![3, 3, 3, 3],
        "red and blue lanes should each distribute 6 packets evenly; got {:?}", counts);
}

/// Sanity: mode=0 (default) is still fan-out — every colour-match
/// worker gets a copy of every matching packet. Guards against the
/// round-robin rules accidentally firing in fan-out mode, and
/// documents that the two modes coexist as expected.
#[test]
fn fanout_mode_still_broadcasts_to_all_color_matches() {
    let mut app = make_app();

    let router = spawn_node(&mut app, Kind::Router, 0, "Router_fo");
    let w0 = spawn_node(&mut app, Kind::Worker, 0, "Worker_a");
    let w1 = spawn_node(&mut app, Kind::Worker, 0, "Worker_b");
    let w2 = spawn_node(&mut app, Kind::Worker, 0, "Worker_c");
    wire(&mut app, router, Kind::Router, w0, Kind::Worker);
    wire(&mut app, router, Kind::Router, w1, Kind::Worker);
    wire(&mut app, router, Kind::Router, w2, Kind::Worker);

    // mode defaults to 0 — don't touch it.

    {
        let mut flow_res = app.world_mut().resource_mut::<FlowSim>();
        for _ in 0..4 {
            flow_res.sim.inject(router, Value::variant("packet", Value::Int(0)));
        }
    }
    advance_sim_ns(&mut app, 10_000_000);

    let counts = router_emits_by_worker(&app, router, &[w0, w1, w2]);
    assert_eq!(counts, vec![4, 4, 4],
        "fan-out mode should broadcast each of 4 packets to all 3 workers; got {:?}", counts);
}
