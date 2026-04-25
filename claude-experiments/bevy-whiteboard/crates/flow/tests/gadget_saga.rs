//! Saga gadget — return_path as transaction log.
//!
//! Topology: Client → Saga → Step1 → Step2 → Step3
//! Every forward edge has a reverse edge so resp and compensate
//! can walk back along the same path the req took.
//!
//! Validates:
//! - all-success: every step bumps `done`, no `compensated`, Saga.succeeded == 1.
//! - mid-chain fail: upstream steps get compensated, downstream steps don't run.
//! - last-step fail: all prior steps get compensated.
//! - no runtime-error counter increments anywhere.

use std::collections::BTreeMap;

use flow::sim::Sim;
use flow::value::{Pattern, Value};
use flow::rule::{Effect, Rule, When};
use flow::expr::Expr;
use flow::NodeId;

const SAGA_DSL: &str = include_str!("../../flow-bevy/src/gadgets/saga.flow");

struct Topology {
    client: NodeId,
    saga: NodeId,
    s1: NodeId,
    s2: NodeId,
    s3: NodeId,
}

/// Build: Client → Saga → Step1 → Step2 → Step3 with full reverse edges.
/// Client is a plain node (not DSL-authored) with two rules that count
/// resp / resp_error arrivals. That sidesteps the need to wire a
/// second DSL class just for the test harness.
fn build_topology(sim: &mut Sim, step_fail_probs: [f64; 3]) -> Topology {
    flow::dsl::register_classes(sim, SAGA_DSL).unwrap();

    let saga = sim.instantiate("Saga", "saga").unwrap();
    let s1 = sim.instantiate("SagaStep", "s1").unwrap();
    let s2 = sim.instantiate("SagaStep", "s2").unwrap();
    let s3 = sim.instantiate("SagaStep", "s3").unwrap();

    // Apply fail_prob on each step.
    for (nid, p) in [(s1, step_fail_probs[0]), (s2, step_fail_probs[1]), (s3, step_fail_probs[2])] {
        sim.nodes.get_mut(&nid).unwrap().slots.insert("fail_prob".into(), Value::Float(p));
    }

    let client = sim.add_node(
        "client",
        BTreeMap::from([
            ("ok_count".to_string(), Value::Int(0)),
            ("err_count".to_string(), Value::Int(0)),
        ]),
        vec![
            Rule::new("on_ok")
                .when(When::input(Pattern::variant("resp", Pattern::wild())))
                .do_(Effect::SetSlot {
                    slot: "ok_count".into(),
                    value: Expr::add(Expr::slot("ok_count"), Expr::int(1)),
                }),
            Rule::new("on_err")
                .when(When::input(Pattern::variant("resp_error", Pattern::wild())))
                .do_(Effect::SetSlot {
                    slot: "err_count".into(),
                    value: Expr::add(Expr::slot("err_count"), Expr::int(1)),
                }),
        ],
    );

    // Forward edges
    sim.add_edge(client, saga, Expr::int(1_000_000));
    sim.add_edge(saga, s1, Expr::int(1_000_000));
    sim.add_edge(s1, s2, Expr::int(1_000_000));
    sim.add_edge(s2, s3, Expr::int(1_000_000));
    // Reverse edges — needed for resp / compensate to walk back.
    sim.add_edge(saga, client, Expr::int(1_000_000));
    sim.add_edge(s1, saga, Expr::int(1_000_000));
    sim.add_edge(s2, s1, Expr::int(1_000_000));
    sim.add_edge(s3, s2, Expr::int(1_000_000));

    Topology { client, saga, s1, s2, s3 }
}

fn slot_i(sim: &Sim, id: NodeId, key: &str) -> i64 {
    match sim.nodes[&id].slots.get(key) {
        Some(Value::Int(n)) => *n,
        other => panic!("slot `{}` on {:?} = {:?}, expected Int", key, id, other),
    }
}

/// Inject one req at Saga with return_path = [client], then drive the
/// sim for 200ms — far longer than the whole chain (~10 * 1ms edges).
fn run_one_tx(sim: &mut Sim, topo: &Topology) {
    sim.inject_with(
        topo.saga,
        Value::variant("req", Value::Int(0)),
        BTreeMap::new(),
        vec![topo.client],
    );
    sim.run_until(sim.now_ns + 200_000_000);
}

fn assert_no_runtime_errors(sim: &Sim) {
    let errs: Vec<_> = sim.error_counts.iter().filter(|(_, c)| **c > 0).collect();
    assert!(errs.is_empty(), "unexpected runtime errors: {:?}", errs);
}

#[test]
fn all_success_every_step_done() {
    let mut sim = Sim::new(1);
    let topo = build_topology(&mut sim, [0.0, 0.0, 0.0]);
    run_one_tx(&mut sim, &topo);

    assert_eq!(slot_i(&sim, topo.s1, "done"), 1, "s1 done");
    assert_eq!(slot_i(&sim, topo.s2, "done"), 1, "s2 done");
    assert_eq!(slot_i(&sim, topo.s3, "done"), 1, "s3 done");
    assert_eq!(slot_i(&sim, topo.s1, "compensated"), 0);
    assert_eq!(slot_i(&sim, topo.s2, "compensated"), 0);
    assert_eq!(slot_i(&sim, topo.s3, "compensated"), 0);

    assert_eq!(slot_i(&sim, topo.saga, "succeeded"), 1);
    assert_eq!(slot_i(&sim, topo.saga, "failed"), 0);
    assert_eq!(slot_i(&sim, topo.saga, "in_flight"), 0);

    assert_eq!(slot_i(&sim, topo.client, "ok_count"), 1);
    assert_eq!(slot_i(&sim, topo.client, "err_count"), 0);

    assert_no_runtime_errors(&sim);
}

#[test]
fn mid_chain_fail_rolls_back_upstream_only() {
    let mut sim = Sim::new(2);
    let topo = build_topology(&mut sim, [0.0, 1.0, 0.0]);
    run_one_tx(&mut sim, &topo);

    // s1 forwarded, s2 failed before counting as done, s3 never ran.
    assert_eq!(slot_i(&sim, topo.s1, "done"), 1, "s1 did its forward");
    assert_eq!(slot_i(&sim, topo.s2, "done"), 0, "s2 failed before done++");
    assert_eq!(slot_i(&sim, topo.s3, "done"), 0, "s3 never reached");

    // Compensation walked back: s1 rolled back; s2/s3 didn't self-compensate.
    assert_eq!(slot_i(&sim, topo.s1, "compensated"), 1);
    assert_eq!(slot_i(&sim, topo.s2, "compensated"), 0);
    assert_eq!(slot_i(&sim, topo.s3, "compensated"), 0);

    assert_eq!(slot_i(&sim, topo.saga, "succeeded"), 0);
    assert_eq!(slot_i(&sim, topo.saga, "failed"), 1);
    assert_eq!(slot_i(&sim, topo.saga, "in_flight"), 0);

    assert_eq!(slot_i(&sim, topo.client, "ok_count"), 0);
    assert_eq!(slot_i(&sim, topo.client, "err_count"), 1);

    assert_no_runtime_errors(&sim);
}

#[test]
fn last_step_fail_rolls_back_all_prior() {
    let mut sim = Sim::new(3);
    let topo = build_topology(&mut sim, [0.0, 0.0, 1.0]);
    run_one_tx(&mut sim, &topo);

    // s1, s2 did their forward; s3 failed before done++.
    assert_eq!(slot_i(&sim, topo.s1, "done"), 1);
    assert_eq!(slot_i(&sim, topo.s2, "done"), 1);
    assert_eq!(slot_i(&sim, topo.s3, "done"), 0);

    // Compensation walked through s2, then s1.
    assert_eq!(slot_i(&sim, topo.s1, "compensated"), 1);
    assert_eq!(slot_i(&sim, topo.s2, "compensated"), 1);
    assert_eq!(slot_i(&sim, topo.s3, "compensated"), 0);

    assert_eq!(slot_i(&sim, topo.saga, "succeeded"), 0);
    assert_eq!(slot_i(&sim, topo.saga, "failed"), 1);

    assert_eq!(slot_i(&sim, topo.client, "err_count"), 1);

    assert_no_runtime_errors(&sim);
}
