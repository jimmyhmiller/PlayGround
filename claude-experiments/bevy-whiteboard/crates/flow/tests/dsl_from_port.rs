//! Tests for `emit X to port NAME` — leaf-node named output ports.
//!
//! Proves a node can have multiple distinct outputs differentiated by
//! `from_port` on outbound edges, without needing the parent-compound
//! `ToOutPort` indirection.

use flow::{event::Event, sim::NodeId};

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
fn emit_to_port_routes_only_to_matching_edges() {
    let src = r#"
        node Switch {
            slots { mode: Int = 0 }
            rule pass   { on go(_) when mode == 0 do { emit msg(nil) to port pass } }
            rule divert { on go(_) when mode == 1 do { emit msg(nil) to port divert } }
        }
        node A { slots { x: Int = 0 } }
        node B { slots { x: Int = 0 } }
        edges {
            Switch.pass   -> A : 1
            Switch.divert -> B : 1
        }
        scenario {
            at 0ns: inject Switch <- go()
        }
    "#;
    let mut sim = flow::dsl::load(src, 0).unwrap();
    sim.run_until(2);
    let a = node_id_by_name(&sim, "A");
    let b = node_id_by_name(&sim, "B");
    // mode == 0 → only A receives, not B
    assert_eq!(count_emissions_to(&sim, a), 1);
    assert_eq!(count_emissions_to(&sim, b), 0);
}

#[test]
fn emit_to_port_other_side() {
    let src = r#"
        node Switch {
            slots { mode: Int = 1 }
            rule pass   { on go(_) when mode == 0 do { emit msg(nil) to port pass } }
            rule divert { on go(_) when mode == 1 do { emit msg(nil) to port divert } }
        }
        node A { slots { x: Int = 0 } }
        node B { slots { x: Int = 0 } }
        edges {
            Switch.pass   -> A : 1
            Switch.divert -> B : 1
        }
        scenario { at 0ns: inject Switch <- go() }
    "#;
    let mut sim = flow::dsl::load(src, 0).unwrap();
    sim.run_until(2);
    let a = node_id_by_name(&sim, "A");
    let b = node_id_by_name(&sim, "B");
    assert_eq!(count_emissions_to(&sim, a), 0);
    assert_eq!(count_emissions_to(&sim, b), 1);
}

#[test]
fn emit_to_port_fans_to_multiple_matching_edges() {
    let src = r#"
        node Src {
            slots { dummy: Int = 0 }
            rule fire { on go(_) do { emit msg(nil) to port broadcast } }
        }
        node A { slots { x: Int = 0 } }
        node B { slots { x: Int = 0 } }
        node C { slots { x: Int = 0 } }
        edges {
            Src.broadcast -> A : 1
            Src.broadcast -> B : 1
            Src.other     -> C : 1
        }
        scenario { at 0ns: inject Src <- go() }
    "#;
    let mut sim = flow::dsl::load(src, 0).unwrap();
    sim.run_until(2);
    let a = node_id_by_name(&sim, "A");
    let b = node_id_by_name(&sim, "B");
    let c = node_id_by_name(&sim, "C");
    assert_eq!(count_emissions_to(&sim, a), 1);
    assert_eq!(count_emissions_to(&sim, b), 1);
    assert_eq!(count_emissions_to(&sim, c), 0);
}

#[test]
fn emit_to_port_unwired_is_silent_drop() {
    let src = r#"
        node Src {
            slots { x: Int = 0 }
            rule fire { on go(_) do { emit msg(nil) to port nowhere } }
        }
        node A { slots { x: Int = 0 } }
        edges {
            Src.elsewhere -> A : 1
        }
        scenario { at 0ns: inject Src <- go() }
    "#;
    let mut sim = flow::dsl::load(src, 0).unwrap();
    sim.run_until(2);
    let a = node_id_by_name(&sim, "A");
    assert_eq!(count_emissions_to(&sim, a), 0);
    // No errors recorded — unwired output is a normal state.
    assert!(sim.error_counts.is_empty(), "unwired port should not record an error");
}
