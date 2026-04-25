//! Minimal syntax check: does each in-progress gadget at least parse
//! and lower as a class template? Doesn't run them, just compiles them.

use flow::sim::Sim;

#[test]
fn cache_flow_parses() {
    let src = include_str!("../../flow-bevy/src/gadgets/cache.flow");
    let mut sim = Sim::new(0);
    flow::dsl::register_classes(&mut sim, src).expect("cache.flow failed to register");
}

#[test]
fn circuit_breaker_flow_parses() {
    let src = include_str!("../../flow-bevy/src/gadgets/circuit_breaker.flow");
    let mut sim = Sim::new(0);
    flow::dsl::register_classes(&mut sim, src).expect("circuit_breaker.flow failed to register");
}

#[test]
fn tpc_flow_parses() {
    let src = include_str!("../../flow-bevy/src/gadgets/tpc.flow");
    let mut sim = Sim::new(0);
    flow::dsl::register_classes(&mut sim, src).expect("tpc.flow failed to register");
}

#[test]
fn saga_flow_parses() {
    let src = include_str!("../../flow-bevy/src/gadgets/saga.flow");
    let mut sim = Sim::new(0);
    flow::dsl::register_classes(&mut sim, src).expect("saga.flow failed to register");
}
