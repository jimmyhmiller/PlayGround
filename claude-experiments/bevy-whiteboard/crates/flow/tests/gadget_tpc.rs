//! Two-phase commit gadgets — stashed-reply + fan-out-and-gather.
//!
//! Topology: Client → Coordinator ↔ 3 Participants
//! The coordinator stashes the client's return_path in `pending_rp`
//! while the prepare round runs, then replies once the vote tally
//! completes.

use std::collections::BTreeMap;

use flow::sim::Sim;
use flow::value::{Pattern, Value};
use flow::rule::{Effect, Rule, When};
use flow::expr::Expr;
use flow::NodeId;

const TPC_DSL: &str = include_str!("../../flow-bevy/src/gadgets/tpc.flow");

struct Topology {
    client: NodeId,
    coord:  NodeId,
    p0:     NodeId,
    p1:     NodeId,
    p2:     NodeId,
}

fn build(sim: &mut Sim, vote_probs: [f64; 3]) -> Topology {
    flow::dsl::register_classes(sim, TPC_DSL).unwrap();

    let coord = sim.instantiate("TPCCoordinator", "coord").unwrap();
    let p0    = sim.instantiate("TPCParticipant", "p0").unwrap();
    let p1    = sim.instantiate("TPCParticipant", "p1").unwrap();
    let p2    = sim.instantiate("TPCParticipant", "p2").unwrap();

    for (nid, p) in [(p0, vote_probs[0]), (p1, vote_probs[1]), (p2, vote_probs[2])] {
        sim.nodes.get_mut(&nid).unwrap().slots.insert("vote_yes_prob".into(), Value::Float(p));
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

    // Forward edges first so `filter(out_neighbors, n != head(rp))`
    // correctly excludes the client.
    sim.add_edge(client, coord, Expr::int(1_000_000));
    for &p in &[p0, p1, p2] {
        sim.add_edge(coord, p,     Expr::int(1_000_000));
        sim.add_edge(p,     coord, Expr::int(1_000_000));
    }
    // Reverse client edge last.
    sim.add_edge(coord, client, Expr::int(1_000_000));

    Topology { client, coord, p0, p1, p2 }
}

fn slot_i(sim: &Sim, id: NodeId, key: &str) -> i64 {
    match sim.nodes[&id].slots.get(key) {
        Some(Value::Int(n)) => *n,
        other => panic!("slot `{}` on {:?} = {:?}", key, id, other),
    }
}
fn slot_str(sim: &Sim, id: NodeId, key: &str) -> String {
    match sim.nodes[&id].slots.get(key) {
        Some(Value::Str(s)) => s.clone(),
        other => panic!("slot `{}` on {:?} = {:?}", key, id, other),
    }
}

fn start_tx(sim: &mut Sim, topo: &Topology) {
    sim.inject_with(
        topo.coord,
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
fn all_yes_commits() {
    let mut sim = Sim::new(1);
    let topo = build(&mut sim, [1.0, 1.0, 1.0]);
    start_tx(&mut sim, &topo);

    assert_eq!(slot_i(&sim, topo.coord, "committed"), 1);
    assert_eq!(slot_i(&sim, topo.coord, "aborted"), 0);
    assert_eq!(slot_str(&sim, topo.coord, "phase"), "idle");
    assert_eq!(slot_i(&sim, topo.coord, "participants"), 3, "should count 3 participants (not 4)");

    for &p in &[topo.p0, topo.p1, topo.p2] {
        assert_eq!(slot_i(&sim, p, "committed"), 1);
        assert_eq!(slot_i(&sim, p, "aborted"), 0);
        assert_eq!(slot_i(&sim, p, "prepared"), 1);
    }

    assert_eq!(slot_i(&sim, topo.client, "ok_count"), 1);
    assert_eq!(slot_i(&sim, topo.client, "err_count"), 0);

    assert_no_runtime_errors(&sim);
}

#[test]
fn one_no_vote_aborts() {
    let mut sim = Sim::new(2);
    // p0 votes no.
    let topo = build(&mut sim, [0.0, 1.0, 1.0]);
    start_tx(&mut sim, &topo);

    assert_eq!(slot_i(&sim, topo.coord, "committed"), 0);
    assert_eq!(slot_i(&sim, topo.coord, "aborted"), 1);
    assert_eq!(slot_str(&sim, topo.coord, "phase"), "idle");

    // Every participant is aborted, including the yes-voters.
    for &p in &[topo.p0, topo.p1, topo.p2] {
        assert_eq!(slot_i(&sim, p, "committed"), 0);
        assert_eq!(slot_i(&sim, p, "aborted"), 1);
    }

    assert_eq!(slot_i(&sim, topo.client, "ok_count"), 0);
    assert_eq!(slot_i(&sim, topo.client, "err_count"), 1);

    assert_no_runtime_errors(&sim);
}

#[test]
fn middle_no_vote_still_aborts_regardless_of_order() {
    // p1 votes no; p0 and p2 vote yes. Tests the "last-vote-yes but
    // earlier-no" branch (vote_final_abort's if/else-in-expr path).
    let mut sim = Sim::new(3);
    let topo = build(&mut sim, [1.0, 0.0, 1.0]);
    start_tx(&mut sim, &topo);

    assert_eq!(slot_i(&sim, topo.coord, "committed"), 0);
    assert_eq!(slot_i(&sim, topo.coord, "aborted"), 1);
    assert_eq!(slot_str(&sim, topo.coord, "phase"), "idle");

    for &p in &[topo.p0, topo.p1, topo.p2] {
        assert_eq!(slot_i(&sim, p, "aborted"), 1);
    }

    assert_eq!(slot_i(&sim, topo.client, "err_count"), 1);
    assert_no_runtime_errors(&sim);
}
