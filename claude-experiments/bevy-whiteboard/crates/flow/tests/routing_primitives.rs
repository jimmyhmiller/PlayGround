//! Engine-level tests for the routing-strategy primitives:
//! `OutNeighbors`, `SlotOf`, `EmitToEach`, plus the list operators
//! (`Length`, `Index`, `Filter`, `Map`, `Reduce`).
//!
//! These prove every routing strategy expressible from these
//! primitives is mechanical Expr/Effect composition — no engine code
//! changes per strategy.

use std::collections::BTreeMap;

use flow::{
    expr::Expr,
    rule::{Effect, EmitTo, Rule, When},
    sim::Sim,
    value::{Pattern, Value},
};

fn slots(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

fn count_emissions_to(sim: &flow::sim::Sim, target: flow::sim::NodeId) -> usize {
    sim.log
        .events
        .iter()
        .filter(|ev| matches!(ev, flow::event::Event::PacketEmitted { to, .. } if *to == target))
        .count()
}

#[test]
fn out_neighbors_returns_outbound_node_refs() {
    // src has two outbound + one inbound + one self-loop. OutNeighbors
    // should return only the two real outbounds, in id order, no self.
    let mut sim = Sim::new(0);
    let src = sim.add_node("src", slots(&[]), vec![
        Rule::new("emit_count_to_self")
            .when(When::input(Pattern::variant("go", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "count".into(),
                value: Expr::length(Expr::out_neighbors()),
            }),
    ]);
    sim.nodes.get_mut(&src).unwrap().slots.insert("count".into(), Value::Int(0));
    let a = sim.add_node("a", slots(&[]), vec![]);
    let b = sim.add_node("b", slots(&[]), vec![]);
    let _other = sim.add_node("other", slots(&[]), vec![]);

    sim.add_edge(src, a, Expr::int(1));
    sim.add_edge(src, b, Expr::int(1));
    sim.add_edge(_other, src, Expr::int(1)); // inbound — not counted
    sim.add_edge(src, src, Expr::int(1));    // self-loop — not counted

    sim.inject(src, Value::variant("go", Value::Nil));
    sim.run_until(1);

    // Length expression stored 2 in `count`.
    assert_eq!(sim.nodes[&src].slots["count"], Value::Int(2));
}

#[test]
fn slot_of_reads_neighbors_state() {
    let mut sim = Sim::new(0);
    let neighbour = sim.add_node("nbr", slots(&[("color", Value::Int(7))]), vec![]);
    let src = sim.add_node("src", slots(&[("seen_color", Value::Int(-1))]), vec![
        Rule::new("read")
            .when(When::input(Pattern::variant("go", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "seen_color".into(),
                value: Expr::slot_of(
                    Expr::lit(Value::NodeRef(neighbour)),
                    "color",
                ),
            }),
    ]);
    sim.add_edge(src, neighbour, Expr::int(1));
    sim.inject(src, Value::variant("go", Value::Nil));
    sim.run_until(1);
    assert_eq!(sim.nodes[&src].slots["seen_color"], Value::Int(7));
}

#[test]
fn emit_to_each_broadcasts_to_all_outbounds() {
    let mut sim = Sim::new(0);
    let src = sim.add_node("src", slots(&[]), vec![
        Rule::new("broadcast")
            .when(When::input(Pattern::variant("go", Pattern::wild())))
            .do_(Effect::emit_to_each(
                Expr::variant("packet", Expr::lit(Value::Nil)),
                Expr::out_neighbors(),
            )),
    ]);
    let a = sim.add_node("a", slots(&[]), vec![]);
    let b = sim.add_node("b", slots(&[]), vec![]);
    let c = sim.add_node("c", slots(&[]), vec![]);
    sim.add_edge(src, a, Expr::int(1));
    sim.add_edge(src, b, Expr::int(1));
    sim.add_edge(src, c, Expr::int(1));

    sim.inject(src, Value::variant("go", Value::Nil));
    sim.run_until(2);

    // One emission scheduled per outbound.
    assert_eq!(count_emissions_to(&sim, a), 1);
    assert_eq!(count_emissions_to(&sim, b), 1);
    assert_eq!(count_emissions_to(&sim, c), 1);
}

#[test]
fn round_robin_router_via_index_and_modulo() {
    // Use Index over OutNeighbors with a counter slot — pure data,
    // no engine extension. Send 6 packets, expect each of 3 outbounds
    // to receive 2.
    let mut sim = Sim::new(0);
    let src = sim.add_node(
        "src",
        slots(&[("rr", Value::Int(0))]),
        vec![
            Rule::new("forward")
                .when(When::input(Pattern::variant("packet", Pattern::var("p"))))
                .do_(Effect::SetSlot {
                    slot: "rr".into(),
                    value: Expr::add(Expr::slot("rr"), Expr::int(1)),
                })
                .do_(Effect::emit(
                    Expr::variant("packet", Expr::var("p")),
                    EmitTo::ToTargetExpr(Expr::index(
                        Expr::out_neighbors(),
                        Expr::slot("rr"),
                    )),
                )),
        ],
    );
    let a = sim.add_node("a", slots(&[]), vec![]);
    let b = sim.add_node("b", slots(&[]), vec![]);
    let c = sim.add_node("c", slots(&[]), vec![]);
    sim.add_edge(src, a, Expr::int(1));
    sim.add_edge(src, b, Expr::int(1));
    sim.add_edge(src, c, Expr::int(1));

    for _ in 0..6 {
        sim.inject(src, Value::variant("packet", Value::Nil));
    }
    sim.run_until(10);

    assert_eq!(count_emissions_to(&sim, a), 2);
    assert_eq!(count_emissions_to(&sim, b), 2);
    assert_eq!(count_emissions_to(&sim, c), 2);
}

#[test]
fn least_loaded_router_via_reduce_slot_of() {
    // Pure-data least-loaded routing: the router picks the outbound
    // neighbour whose `load` slot is currently smallest. Two
    // candidates with distinct load values — the lower one wins.
    let mut sim = Sim::new(0);
    let busy = sim.add_node("busy", slots(&[("load", Value::Int(10))]), vec![]);
    let idle = sim.add_node("idle", slots(&[("load", Value::Int(1))]), vec![]);

    let src = sim.add_node(
        "src",
        slots(&[]),
        vec![
            Rule::new("forward")
                .when(When::input(Pattern::variant("packet", Pattern::var("p"))))
                .do_(Effect::emit(
                    Expr::variant("packet", Expr::var("p")),
                    EmitTo::ToTargetExpr(Expr::reduce(
                        Expr::out_neighbors(),
                        "n", "best",
                        Expr::index(Expr::out_neighbors(), Expr::int(0)),
                        Expr::if_(
                            Expr::lt(
                                Expr::slot_of(Expr::var("n"), "load"),
                                Expr::slot_of(Expr::var("best"), "load"),
                            ),
                            Expr::var("n"),
                            Expr::var("best"),
                        ),
                    )),
                )),
        ],
    );
    sim.add_edge(src, busy, Expr::int(1));
    sim.add_edge(src, idle, Expr::int(1));

    sim.inject(src, Value::variant("packet", Value::Nil));
    sim.run_until(2);

    // Idle wins both packets.
    assert_eq!(count_emissions_to(&sim, idle), 1);
    assert_eq!(count_emissions_to(&sim, busy), 0);
}

#[test]
fn filter_then_emit_to_subset() {
    // Keep only neighbours whose `enabled` slot is 1.
    let mut sim = Sim::new(0);
    let on1 = sim.add_node("on1", slots(&[("enabled", Value::Int(1))]), vec![]);
    let off = sim.add_node("off", slots(&[("enabled", Value::Int(0))]), vec![]);
    let on2 = sim.add_node("on2", slots(&[("enabled", Value::Int(1))]), vec![]);
    let src = sim.add_node(
        "src",
        slots(&[]),
        vec![
            Rule::new("emit_to_enabled")
                .when(When::input(Pattern::variant("go", Pattern::wild())))
                .do_(Effect::emit_to_each(
                    Expr::variant("packet", Expr::lit(Value::Nil)),
                    Expr::filter(
                        Expr::out_neighbors(),
                        "n",
                        Expr::eq(
                            Expr::slot_of(Expr::var("n"), "enabled"),
                            Expr::int(1),
                        ),
                    ),
                )),
        ],
    );
    sim.add_edge(src, on1, Expr::int(1));
    sim.add_edge(src, off, Expr::int(1));
    sim.add_edge(src, on2, Expr::int(1));

    sim.inject(src, Value::variant("go", Value::Nil));
    sim.run_until(2);

    assert_eq!(count_emissions_to(&sim, on1), 1);
    assert_eq!(count_emissions_to(&sim, on2), 1);
    assert_eq!(count_emissions_to(&sim, off), 0);
}

#[test]
fn map_can_extract_slot_values_into_a_list() {
    // Sanity for Map: turn a list of NodeRefs into a list of Ints by
    // reading their `color` slots.
    let mut sim = Sim::new(0);
    let a = sim.add_node("a", slots(&[("color", Value::Int(1))]), vec![]);
    let b = sim.add_node("b", slots(&[("color", Value::Int(2))]), vec![]);
    let c = sim.add_node("c", slots(&[("color", Value::Int(3))]), vec![]);
    let src = sim.add_node(
        "src",
        slots(&[("colors_seen", Value::Nil)]),
        vec![
            Rule::new("snapshot_colors")
                .when(When::input(Pattern::variant("go", Pattern::wild())))
                .do_(Effect::SetSlot {
                    slot: "colors_seen".into(),
                    value: Expr::map(
                        Expr::out_neighbors(),
                        "n",
                        Expr::slot_of(Expr::var("n"), "color"),
                    ),
                }),
        ],
    );
    sim.add_edge(src, a, Expr::int(1));
    sim.add_edge(src, b, Expr::int(1));
    sim.add_edge(src, c, Expr::int(1));

    sim.inject(src, Value::variant("go", Value::Nil));
    sim.run_until(1);

    assert_eq!(
        sim.nodes[&src].slots["colors_seen"],
        Value::List(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
    );
}
