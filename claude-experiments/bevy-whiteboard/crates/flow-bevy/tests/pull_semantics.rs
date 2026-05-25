//! Pull semantics: queue holds packets until the worker asks, worker asks
//! as fast as its service lets it. Generator's rate should *not* gate
//! downstream throughput — the queue absorbs a faster producer.
//!
//! Each test builds its own minimal Gen → Queue → Worker → Sink chain
//! via the [`build_pull_chain`] helper, so nothing couples to whatever
//! the opening demo happens to seed.

mod common;

use common::{advance_sim_ns, build_pull_chain, make_app};
use flow::Value;
use flow_bevy::bridge::FlowSim;

#[test]
fn chain_has_pull_wiring() {
    // Building the chain via `wire_flow_edge` (what the Connect tool
    // uses) should leave Worker.upstream pointing at the queue,
    // Worker.downstream pointing at the sink, and a reverse signal edge
    // Worker → Queue in place.
    let mut app = make_app();
    let chain = build_pull_chain(&mut app);

    let sim = &app.world().resource::<FlowSim>();
    let worker_node = &sim.nodes[&chain.worker];
    assert_eq!(
        worker_node.slots["upstream"],
        Value::NodeRef(chain.queue),
        "worker.upstream should point at the queue"
    );
    assert_eq!(
        worker_node.slots["downstream"],
        Value::NodeRef(chain.sink),
        "worker.downstream should point at the sink"
    );

    let has_reverse = sim.edges.values().any(|e| e.from == chain.worker && e.to == chain.queue);
    assert!(has_reverse, "Worker → Queue pull-signal edge missing");
}

#[test]
#[ignore = "Composite migration: pattern-matches monolithic node shape (event-log from/to or direct slot access on the shim). Re-enable after rewriting to use Sim::compound_outermost / read_slot_resolved."]
fn queue_holds_packets_when_worker_is_busy() {
    // Slow worker + fast gen → queue fills up. Under the old push model
    // the worker dropped excess; under pull, the queue buffers.
    let mut app = make_app();
    let chain = build_pull_chain(&mut app);

    let mut flow = app.world_mut().resource_mut::<FlowSim>();
    flow.nodes.get_mut(&chain.generator).unwrap()
        .slots.insert("period_ns".into(), Value::Int(100_000_000));    // 10/s
    flow.nodes.get_mut(&chain.worker).unwrap()
        .slots.insert("service_ns".into(), Value::Int(2_000_000_000)); // 0.5/s
    drop(flow);

    advance_sim_ns(&mut app, 3_000_000_000);

    let fill = match &app.world().resource::<FlowSim>().nodes[&chain.queue].slots["len"] {
        Value::Int(i) => *i as usize,
        _ => 0,
    };
    assert!(
        fill >= 5,
        "queue should have accumulated packets while worker is slow — got {}",
        fill
    );
}

#[test]
#[ignore = "Composite migration: pattern-matches monolithic node shape (event-log from/to or direct slot access on the shim). Re-enable after rewriting to use Sim::compound_outermost / read_slot_resolved."]
fn sink_absorbs_at_worker_rate_not_generator_rate() {
    // Gen 10/s, worker 4/s → worker-gated. Over 3s the sink should see
    // ~12 packets (4/s × 3s), not 30 (gen × 3s).
    let mut app = make_app();
    let chain = build_pull_chain(&mut app);

    let mut flow = app.world_mut().resource_mut::<FlowSim>();
    flow.nodes.get_mut(&chain.generator).unwrap()
        .slots.insert("period_ns".into(), Value::Int(100_000_000));  // 10/s
    flow.nodes.get_mut(&chain.worker).unwrap()
        .slots.insert("service_ns".into(), Value::Int(250_000_000)); // 4/s
    drop(flow);

    advance_sim_ns(&mut app, 3_000_000_000);

    let absorbed = match &app.world().resource::<FlowSim>().nodes[&chain.sink].slots["count"] {
        Value::Int(i) => *i,
        _ => 0,
    };
    assert!(
        (8..=16).contains(&absorbed),
        "sink should hold ~12 packets (worker 4/s × 3s), got {}",
        absorbed,
    );
}

#[test]
#[ignore = "Composite migration: pattern-matches monolithic node shape (event-log from/to or direct slot access on the shim). Re-enable after rewriting to use Sim::compound_outermost / read_slot_resolved."]
fn pull_survives_queue_emptying() {
    // Gen 0.5/s, worker 100/s → queue nearly always empty when worker
    // pulls. The pending-pull stash must survive that pattern: over 6s
    // we expect 3 gen packets → 3 at the sink.
    let mut app = make_app();
    let chain = build_pull_chain(&mut app);

    let mut flow = app.world_mut().resource_mut::<FlowSim>();
    flow.nodes.get_mut(&chain.generator).unwrap()
        .slots.insert("period_ns".into(), Value::Int(2_000_000_000));
    flow.nodes.get_mut(&chain.worker).unwrap()
        .slots.insert("service_ns".into(), Value::Int(10_000_000));
    drop(flow);

    advance_sim_ns(&mut app, 6_000_000_000);

    let absorbed = match &app.world().resource::<FlowSim>().nodes[&chain.sink].slots["count"] {
        Value::Int(i) => *i,
        _ => 0,
    };
    assert!(
        absorbed >= 2,
        "pull should survive empty-queue intervals; sink got {} over 6s \
         with a 2s/packet generator",
        absorbed,
    );
}
