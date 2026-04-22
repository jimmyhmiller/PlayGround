//! DSL-level tests for the routing-strategy primitives:
//! `out_neighbors()`, `slot_of(node, "slot")`, `length`, `index`,
//! `filter`, `map`, `reduce`, plus the `emit_each` statement.
//!
//! These prove every routing strategy is expressible from the DSL —
//! no engine code changes per strategy.

use flow::{event::Event, sim::NodeId, value::Value};

fn count_emissions_to(sim: &flow::sim::Sim, target: NodeId) -> usize {
    sim.log
        .events
        .iter()
        .filter(|ev| matches!(ev, Event::PacketEmitted { to, .. } if *to == target))
        .count()
}

fn node_id_by_name(sim: &flow::sim::Sim, name: &str) -> NodeId {
    *sim.nodes
        .iter()
        .find(|(_, n)| n.name == name)
        .map(|(id, _)| id)
        .unwrap_or_else(|| panic!("no node named `{}`", name))
}

#[test]
fn dsl_emit_each_broadcasts_to_all_outbounds() {
    let src = r#"
        node Src {
            slots { dummy: Int = 0 }
            rule broadcast {
                on go(_)
                do {
                    emit_each packet(nil) to out_neighbors()
                }
            }
        }
        node A { slots { x: Int = 0 } }
        node B { slots { x: Int = 0 } }
        node C { slots { x: Int = 0 } }
        edges {
            Src -> A : 1
            Src -> B : 1
            Src -> C : 1
        }
        scenario {
            at 0ns: inject Src <- go()
        }
    "#;
    let mut sim = flow::dsl::load(src, 0).unwrap();
    sim.run_until(2);
    let a = node_id_by_name(&sim, "A");
    let b = node_id_by_name(&sim, "B");
    let c = node_id_by_name(&sim, "C");
    assert_eq!(count_emissions_to(&sim, a), 1);
    assert_eq!(count_emissions_to(&sim, b), 1);
    assert_eq!(count_emissions_to(&sim, c), 1);
}

#[test]
fn dsl_length_of_out_neighbors() {
    let src = r#"
        node Src {
            slots { count: Int = 0 }
            rule snap {
                on go(_)
                do { count := length(out_neighbors()) }
            }
        }
        node A { slots { x: Int = 0 } }
        node B { slots { x: Int = 0 } }
        edges {
            Src -> A : 1
            Src -> B : 1
        }
        scenario { at 0ns: inject Src <- go() }
    "#;
    let mut sim = flow::dsl::load(src, 0).unwrap();
    sim.run_until(1);
    let src_id = node_id_by_name(&sim, "Src");
    assert_eq!(sim.nodes[&src_id].slots["count"], Value::Int(2));
}

#[test]
fn dsl_round_robin_via_index() {
    // Pure DSL composition: index(out_neighbors(), rr) with a counter slot.
    let src = r#"
        node Src {
            slots { rr: Int = 0 }
            rule forward {
                on packet(p)
                do {
                    rr := rr + 1
                    emit packet(p) to (index(out_neighbors(), rr))
                }
            }
        }
        node A { slots { x: Int = 0 } }
        node B { slots { x: Int = 0 } }
        node C { slots { x: Int = 0 } }
        edges {
            Src -> A : 1
            Src -> B : 1
            Src -> C : 1
        }
        scenario {
            at 0ns: inject Src <- packet()
            at 0ns: inject Src <- packet()
            at 0ns: inject Src <- packet()
            at 0ns: inject Src <- packet()
            at 0ns: inject Src <- packet()
            at 0ns: inject Src <- packet()
        }
    "#;
    let mut sim = flow::dsl::load(src, 0).unwrap();
    sim.run_until(10);
    let a = node_id_by_name(&sim, "A");
    let b = node_id_by_name(&sim, "B");
    let c = node_id_by_name(&sim, "C");
    assert_eq!(count_emissions_to(&sim, a), 2);
    assert_eq!(count_emissions_to(&sim, b), 2);
    assert_eq!(count_emissions_to(&sim, c), 2);
}

#[test]
fn dsl_filter_then_emit_each() {
    let src = r#"
        node Src {
            slots { dummy: Int = 0 }
            rule fanout {
                on go(_)
                do {
                    emit_each packet(nil) to filter(out_neighbors(), "n", slot_of(n, "enabled") == 1)
                }
            }
        }
        node On1 { slots { enabled: Int = 1 } }
        node Off { slots { enabled: Int = 0 } }
        node On2 { slots { enabled: Int = 1 } }
        edges {
            Src -> On1 : 1
            Src -> Off : 1
            Src -> On2 : 1
        }
        scenario { at 0ns: inject Src <- go() }
    "#;
    let mut sim = flow::dsl::load(src, 0).unwrap();
    sim.run_until(2);
    let on1 = node_id_by_name(&sim, "On1");
    let off = node_id_by_name(&sim, "Off");
    let on2 = node_id_by_name(&sim, "On2");
    assert_eq!(count_emissions_to(&sim, on1), 1);
    assert_eq!(count_emissions_to(&sim, on2), 1);
    assert_eq!(count_emissions_to(&sim, off), 0);
}

#[test]
fn dsl_least_loaded_via_reduce() {
    // Reduce over out_neighbors picking the lowest-load neighbour —
    // pure DSL, no engine extension.
    let src = r#"
        node Src {
            slots { dummy: Int = 0 }
            rule forward {
                on packet(p)
                do {
                    emit packet(p) to (reduce(out_neighbors(), "n", "best",
                        index(out_neighbors(), 0),
                        if slot_of(n, "load") < slot_of(best, "load") then n else best))
                }
            }
        }
        node Busy { slots { load: Int = 10 } }
        node Idle { slots { load: Int = 1 } }
        edges {
            Src -> Busy : 1
            Src -> Idle : 1
        }
        scenario { at 0ns: inject Src <- packet() }
    "#;
    let mut sim = flow::dsl::load(src, 0).unwrap();
    sim.run_until(2);
    let busy = node_id_by_name(&sim, "Busy");
    let idle = node_id_by_name(&sim, "Idle");
    assert_eq!(count_emissions_to(&sim, idle), 1);
    assert_eq!(count_emissions_to(&sim, busy), 0);
}

#[test]
fn dsl_map_extracts_slot_values() {
    let src = r#"
        node Src {
            slots { colors: Any = nil }
            rule snap {
                on go(_)
                do { colors := map(out_neighbors(), "n", slot_of(n, "color")) }
            }
        }
        node A { slots { color: Int = 1 } }
        node B { slots { color: Int = 2 } }
        node C { slots { color: Int = 3 } }
        edges {
            Src -> A : 1
            Src -> B : 1
            Src -> C : 1
        }
        scenario { at 0ns: inject Src <- go() }
    "#;
    let mut sim = flow::dsl::load(src, 0).unwrap();
    sim.run_until(1);
    let src_id = node_id_by_name(&sim, "Src");
    assert_eq!(
        sim.nodes[&src_id].slots["colors"],
        Value::List(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
    );
}
