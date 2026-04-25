//! Circuit breaker gadget — three-state FSM (closed / open / half_open).
//!
//! Topology: Client → CircuitBreaker → Service (with toggleable up).
//! Validates: failures accumulate in a sliding window, the breaker
//! trips at threshold, rejects reqs while open, transitions to half_open
//! after recovery_ns via self-tick, then probes and either closes on
//! success or re-opens on failure.

use std::collections::BTreeMap;

use flow::sim::Sim;
use flow::value::{Pattern, Value};
use flow::rule::{Effect, Rule, When};
use flow::expr::Expr;
use flow::NodeId;

const CB_DSL: &str = include_str!("../../flow-bevy/src/gadgets/circuit_breaker.flow");

const SERVICE_DSL: &str = r#"
node Service {
    slots {
        served: Int = 0
        color:  Int = 0
        up:     Int = 1
    }
    rule serve_up {
        on req(c)
        when c == color && up == 1
        do {
            served := served + 1
            emit resp(nil) popping to (head(return_path))
        }
    }
    rule serve_down {
        on req(_)
        when up == 0
        do {
            emit resp_error(nil) popping to (head(return_path))
        }
    }
}
"#;

struct Topology {
    client: NodeId,
    cb:     NodeId,
    svc:    NodeId,
}

fn build(sim: &mut Sim) -> Topology {
    let combined = format!("{}\n{}", CB_DSL, SERVICE_DSL);
    flow::dsl::register_classes(sim, &combined).unwrap();

    let cb  = sim.instantiate("CircuitBreaker", "cb").unwrap();
    let svc = sim.instantiate("Service", "svc").unwrap();

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

    // Forward edges first so cache-style "to default" and filter tricks
    // both pick the forward direction.
    sim.add_edge(client, cb,  Expr::int(1_000_000));
    sim.add_edge(cb,     svc, Expr::int(1_000_000));
    // Reverse.
    sim.add_edge(svc,    cb,     Expr::int(1_000_000));
    sim.add_edge(cb,     client, Expr::int(1_000_000));

    Topology { client, cb, svc }
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

fn set_slot_int(sim: &mut Sim, id: NodeId, key: &str, v: i64) {
    sim.nodes.get_mut(&id).unwrap().slots.insert(key.into(), Value::Int(v));
}

fn inject_req(sim: &mut Sim, topo: &Topology) {
    sim.inject_with(
        topo.cb,
        Value::variant("req", Value::Int(0)),
        BTreeMap::new(),
        vec![topo.client],
    );
}

#[test]
fn trips_open_after_threshold_failures() {
    let mut sim = Sim::new(1);
    let topo = build(&mut sim);
    // Service down → every req yields resp_error at the breaker.
    set_slot_int(&mut sim, topo.svc, "up", 0);

    assert_eq!(slot_str(&sim, topo.cb, "state"), "closed");

    // Inject threshold+2 reqs (threshold default is 5) with minimal
    // spacing; all within the 1s window.
    for _ in 0..7 {
        inject_req(&mut sim, &topo);
        sim.run_until(sim.now_ns + 10_000_000); // 10ms between injects
    }

    // Drive a bit more for in-flight round trips.
    sim.run_until(sim.now_ns + 100_000_000);

    assert_eq!(slot_str(&sim, topo.cb, "state"), "open", "breaker should have tripped");
    assert!(slot_i(&sim, topo.cb, "trips") >= 1);
    assert!(slot_i(&sim, topo.cb, "opened_at_ns") > 0);
    // Client saw resp_errors.
    assert!(slot_i(&sim, topo.client, "err_count") >= 5);
}

#[test]
fn rejects_while_open_without_contacting_service() {
    let mut sim = Sim::new(2);
    let topo = build(&mut sim);
    set_slot_int(&mut sim, topo.svc, "up", 0);

    // Trip it.
    for _ in 0..7 {
        inject_req(&mut sim, &topo);
        sim.run_until(sim.now_ns + 10_000_000);
    }
    sim.run_until(sim.now_ns + 50_000_000);
    assert_eq!(slot_str(&sim, topo.cb, "state"), "open");

    // Record current service traffic. While open, further reqs must
    // NOT reach the service — rejected by the breaker directly.
    let served_at_open = slot_i(&sim, topo.svc, "served");
    let rejected_before = slot_i(&sim, topo.cb, "rejected");

    for _ in 0..5 {
        inject_req(&mut sim, &topo);
        sim.run_until(sim.now_ns + 10_000_000);
    }
    sim.run_until(sim.now_ns + 50_000_000);

    assert_eq!(slot_i(&sim, topo.svc, "served"), served_at_open,
               "service traffic must not increase while breaker is open");
    assert!(slot_i(&sim, topo.cb, "rejected") >= rejected_before + 5);
}

#[test]
fn recovers_through_half_open_probe() {
    let mut sim = Sim::new(3);
    let topo = build(&mut sim);
    set_slot_int(&mut sim, topo.svc, "up", 0);

    // Trip.
    for _ in 0..7 {
        inject_req(&mut sim, &topo);
        sim.run_until(sim.now_ns + 10_000_000);
    }
    sim.run_until(sim.now_ns + 50_000_000);
    assert_eq!(slot_str(&sim, topo.cb, "state"), "open");

    // Advance past recovery_ns (default 2s). The self-tick fires
    // every 100ms and should flip state to half_open.
    sim.run_until(sim.now_ns + 2_200_000_000);
    assert_eq!(slot_str(&sim, topo.cb, "state"), "half_open",
               "should be half_open after recovery window");

    // Bring service back up, inject one probe req. It admits; on
    // resp it closes the breaker.
    set_slot_int(&mut sim, topo.svc, "up", 1);
    inject_req(&mut sim, &topo);
    sim.run_until(sim.now_ns + 50_000_000);

    assert_eq!(slot_str(&sim, topo.cb, "state"), "closed");
    assert!(slot_i(&sim, topo.cb, "probes_passed") >= 1);

    // Further reqs flow through normally.
    let svc_before = slot_i(&sim, topo.svc, "served");
    for _ in 0..3 {
        inject_req(&mut sim, &topo);
        sim.run_until(sim.now_ns + 10_000_000);
    }
    sim.run_until(sim.now_ns + 50_000_000);
    assert_eq!(slot_i(&sim, topo.svc, "served"), svc_before + 3);
}
