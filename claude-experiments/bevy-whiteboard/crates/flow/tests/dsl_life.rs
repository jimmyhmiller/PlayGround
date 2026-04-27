//! End-to-end check that the Life compound — declared as a single DSL
//! file (no `flow-life-gen` involvement) — parses, expands, lowers,
//! and ticks. This is the "Phase E" verification that the new
//! generative DSL constructs (parametric `compound`, `for`,
//! `Cell_{x}_{y}` name interpolation) actually replace the standalone
//! generator for the simple case.
//!
//! We use a 5×5 grid with a single live cell at the centre and run
//! enough sim time for the first generation to apply. After draining,
//! we assert:
//!   - all 25 cells exist (named `Life::Cell_x_y`),
//!   - no runtime errors were recorded,
//!   - the cell state changed in a Conway-consistent way (the lone
//!     centre cell dies; some neighbours ought to remain dead because
//!     they have only 1 live neighbor).

use std::collections::BTreeMap;

use flow::{
    dsl,
    sim::{NodeId, Sim},
    value::Value,
};

const LIFE_SRC: &str = include_str!("../examples/life.flow");

fn name_to_id(sim: &Sim) -> BTreeMap<String, NodeId> {
    sim.nodes.iter().map(|(id, n)| (n.name.clone(), *id)).collect()
}

fn alive_count(sim: &Sim) -> usize {
    sim.nodes
        .values()
        .filter(|n| {
            n.name.starts_with("Life::Cell_")
                && matches!(n.slots.get("alive"), Some(Value::Int(1)))
        })
        .count()
}

#[test]
fn life_5x5_loads_and_ticks() {
    let mut sim = dsl::load(LIFE_SRC, 0).expect("life.flow must load via parse + expand + lower");

    // All 25 cells should exist with the expected qualified names.
    let names = name_to_id(&sim);
    for x in 0..5 {
        for y in 0..5 {
            let n = format!("Life::Cell_{}_{}", x, y);
            assert!(names.contains_key(&n), "missing cell `{}`", n);
        }
    }

    // Plus the compound port-shim node itself.
    assert!(names.contains_key("Life"), "missing Life compound shim");

    // Initial state: exactly one live cell at (2,2).
    assert_eq!(alive_count(&sim), 1, "initial state should have exactly one live cell");
    let centre = names["Life::Cell_2_2"];
    assert!(matches!(sim.nodes[&centre].slots["alive"], Value::Int(1)));

    // Run past the first tick (period = 200ms; cell latency = 1ms).
    // After one full generation the lone live cell — with zero live
    // neighbors — must die per Conway rules. No new births either,
    // since the centre's 8 neighbors each see only one live neighbor
    // (B3 needs exactly 3).
    sim.run_until(250_000_000); // 250 ms

    assert_eq!(
        alive_count(&sim),
        0,
        "after one generation the lone centre cell must die (Conway B3/S23)"
    );

    // No engine-recorded runtime errors — if the wiring or rules were
    // off (e.g. wrong neighbor count, missing report), error_counts
    // would have populated.
    assert!(
        sim.error_counts.is_empty(),
        "unexpected runtime errors: {:?}",
        sim.error_counts
    );
}

#[test]
fn life_compound_port_shim_lowers_to_compound_node() {
    // The legacy port-shim form (empty `items`/`params`, only `in`/`out`
    // maps) should still lower to a Sim compound node. life.flow doesn't
    // declare any in/out ports, but the compound itself still becomes
    // a node in the Sim.
    let sim = dsl::load(LIFE_SRC, 0).expect("life.flow must load");
    let life = sim
        .nodes
        .iter()
        .find(|(_, n)| n.name == "Life")
        .expect("Life compound node must exist");
    assert!(
        life.1.is_compound(),
        "Life node should be a compound, found leaf"
    );
}
