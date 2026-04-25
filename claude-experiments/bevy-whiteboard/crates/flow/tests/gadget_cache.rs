//! Cache gadget — probabilistic hit/miss model.
//!
//! Topology: Client → Cache → BackingStore
//! No keys or values are stored. A "hit" is a `Bernoulli(hit_rate)`
//! draw; misses are forwarded to the backing store with the client's
//! return_path stashed in `pending_rp` so the relayed resp lands at
//! the right client.

use std::collections::BTreeMap;

use flow::sim::Sim;
use flow::value::{Pattern, Value};
use flow::rule::{Effect, Rule, When};
use flow::expr::Expr;
use flow::NodeId;

const CACHE_DSL: &str = include_str!("../../flow-bevy/src/gadgets/cache.flow");

const BACKING_STORE_DSL: &str = r#"
node BackingStore {
    slots {
        served: Int = 0
        color:  Int = 0
    }
    rule serve {
        on req(c)
        when c == color
        do {
            served := served + 1
            emit resp(nil) popping to (head(return_path))
        }
    }
}
"#;

struct Topology {
    client: NodeId,
    cache: NodeId,
    store: NodeId,
}

fn build(sim: &mut Sim, hit_rate: f64) -> Topology {
    let combined = format!("{}\n{}", CACHE_DSL, BACKING_STORE_DSL);
    flow::dsl::register_classes(sim, &combined).unwrap();

    let cache = sim.instantiate("Cache", "cache").unwrap();
    let store = sim.instantiate("BackingStore", "store").unwrap();
    sim.nodes.get_mut(&cache).unwrap().slots.insert("hit_rate".into(), Value::Float(hit_rate));

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

    sim.add_edge(client, cache, Expr::int(1_000_000));
    sim.add_edge(cache, client, Expr::int(1_000_000));
    sim.add_edge(cache, store, Expr::int(1_000_000));
    sim.add_edge(store, cache, Expr::int(1_000_000));

    Topology { client, cache, store }
}

fn slot_i(sim: &Sim, id: NodeId, key: &str) -> i64 {
    match sim.nodes[&id].slots.get(key) {
        Some(Value::Int(n)) => *n,
        other => panic!("slot `{}` on {:?} = {:?}, expected Int", key, id, other),
    }
}

fn inject_reqs(sim: &mut Sim, topo: &Topology, n: usize) {
    for _ in 0..n {
        sim.inject_with(
            topo.cache,
            Value::variant("req", Value::Int(0)),
            BTreeMap::new(),
            vec![topo.client],
        );
    }
}

fn assert_no_runtime_errors(sim: &Sim) {
    let errs: Vec<_> = sim.error_counts.iter().filter(|(_, c)| **c > 0).collect();
    assert!(errs.is_empty(), "unexpected runtime errors: {:?}", errs);
}

#[test]
fn hit_rate_one_every_req_hits() {
    let mut sim = Sim::new(1);
    let topo = build(&mut sim, 1.0);
    inject_reqs(&mut sim, &topo, 10);
    sim.run_until(sim.now_ns + 100_000_000);

    assert_eq!(slot_i(&sim, topo.cache, "hits"), 10);
    assert_eq!(slot_i(&sim, topo.cache, "misses"), 0);
    assert_eq!(slot_i(&sim, topo.store, "served"), 0, "backing store untouched");
    assert_eq!(slot_i(&sim, topo.client, "ok_count"), 10);
    assert_no_runtime_errors(&sim);
}

#[test]
fn hit_rate_zero_every_req_misses() {
    let mut sim = Sim::new(2);
    let topo = build(&mut sim, 0.0);
    inject_reqs(&mut sim, &topo, 10);
    sim.run_until(sim.now_ns + 100_000_000);

    assert_eq!(slot_i(&sim, topo.cache, "hits"), 0);
    assert_eq!(slot_i(&sim, topo.cache, "misses"), 10);
    assert_eq!(slot_i(&sim, topo.store, "served"), 10, "every miss reaches store");
    assert_eq!(slot_i(&sim, topo.client, "ok_count"), 10);
    assert_no_runtime_errors(&sim);
}

#[test]
fn hit_rate_half_produces_mix() {
    let mut sim = Sim::new(42);
    let topo = build(&mut sim, 0.5);
    inject_reqs(&mut sim, &topo, 40);
    sim.run_until(sim.now_ns + 200_000_000);

    let hits = slot_i(&sim, topo.cache, "hits");
    let misses = slot_i(&sim, topo.cache, "misses");
    assert_eq!(hits + misses, 40);
    assert!(hits > 0 && misses > 0, "expected mix: hits={}, misses={}", hits, misses);
    assert_eq!(slot_i(&sim, topo.store, "served"), misses);
    assert_eq!(slot_i(&sim, topo.client, "ok_count"), 40);
    assert_no_runtime_errors(&sim);
}
