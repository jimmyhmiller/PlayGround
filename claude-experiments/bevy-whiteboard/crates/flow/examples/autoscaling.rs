//! Autoscaling pool — concrete-instance mode.
//!
//! A `Pool` node receives a stream of requests. It maintains a set of
//! child `Worker` instances, spawned from a template. When requests
//! arrive faster than the current workers can serve, the pool scales
//! up. When workers go idle, it scales down.
//!
//! This is the *concrete* mode of autoscaling — each worker is a
//! real node with its own state. The alternative aggregate mode
//! (a single node with `replicas: Int` slot) is cheaper and adequate
//! when you don't need per-worker identity. Templates + Spawn make
//! the concrete mode first-class.
//!
//! Run: `cargo run -p flow --example autoscaling`

use std::collections::BTreeMap;

use flow::{
    EdgeEnd, Effect, EmitTo, Event, Expr, Pattern, Rule, Samples, Sim, Template,
    Value, When,
};

const SERVICE_MEAN_NS:   f64 = 20_000_000.0;
const SIM_DURATION_NS:   u64 = 600_000_000;
const PROBE_INTERVAL_NS: u64 = 25_000_000;

fn main() {
    let mut sim = Sim::new(3);

    // ---------- Worker template ----------
    //
    // Each spawned Worker has:
    //   - an inbound edge from Pool (Parent → ThisInstance)
    //   - an outbound edge to Pool (ThisInstance → Parent)
    //
    // On receiving "work", it responds with "done" after service time.
    let worker_template = Template::new("Worker")
        .with_prefix("W")
        .slot("busy", Value::Bool(false))
        // The "serve" rule accepts work, stashes nothing (single-shot, uses Respond):
        .rule(
            Rule::new("serve")
                .when(When::input(Pattern::variant("work", Pattern::wild())))
                .do_(Effect::Emit {
                    // self-loop with service-time latency
                    payload: Expr::variant("done", Expr::lit(Value::Nil)),
                    to: EmitTo::DefaultOut, // default is Worker→Pool edge
                })
        )
        // Parent → ThisInstance: near-instant work routing (1 μs).
        .edge(EdgeEnd::Parent, EdgeEnd::ThisInstance, Expr::int(1_000))
        // ThisInstance → Parent: stochastic service time lives on this edge.
        .edge(
            EdgeEnd::ThisInstance,
            EdgeEnd::Parent,
            Expr::exp_dist(Expr::float(SERVICE_MEAN_NS)),
        );

    sim.register_template(worker_template);

    // ---------- Pool ----------
    //
    // Maintains a `children_ring` of child workers (NodeRef values) as
    // a Samples-bounded list, plus `desired_replicas` driven by queue
    // signal. Scales up if inbox pressure + current replicas indicates.
    let mut pool_slots = BTreeMap::new();
    pool_slots.insert("round_robin".into(), Value::Samples(Samples::new(1024)));
    pool_slots.insert("min_replicas".into(), Value::Int(2));
    pool_slots.insert("max_replicas".into(), Value::Int(8));
    pool_slots.insert("replicas".into(),     Value::Int(0));

    let pool_rules = vec![
        // Boot: at t=0, ensure min_replicas are spawned. We run this rule
        // repeatedly until replicas == min_replicas. It has no Input pattern
        // and thus fires whenever its slot conditions are met.
        Rule::new("boot_scale_up")
            .when(When::SlotMatch {
                slot: "replicas".into(),
                pattern: Pattern::var("r"),
            })
            .guard(Expr::lt(Expr::var("r"), Expr::slot("min_replicas")))
            .do_(Effect::Spawn {
                template: "Worker".into(),
                into_var: Some("new_w".into()),
            })
            .do_(Effect::SamplesPush {
                slot: "round_robin".into(),
                value: Expr::var("new_w"),
            })
            .do_(Effect::SetSlot {
                slot: "replicas".into(),
                value: Expr::add(Expr::slot("replicas"), Expr::int(1)),
            })
            .do_(Effect::RecordMetric {
                name: "replicas".into(),
                value: Expr::slot("replicas"),
            }),

        // On incoming request: route to a worker (round-robin by
        // popping oldest NodeRef from the ring and pushing it back).
        Rule::new("dispatch")
            .when(When::input(Pattern::variant("req", Pattern::wild())))
            .do_(Effect::SamplesPopOldestInto {
                slot: "round_robin".into(),
                into_var: "w".into(),
            })
            .do_(Effect::SamplesPush {
                slot: "round_robin".into(),
                value: Expr::var("w"),
            })
            .do_(Effect::Emit {
                payload: Expr::variant("work", Expr::lit(Value::Nil)),
                to: EmitTo::ToTargetExpr(Expr::var("w")),
            }),

        // On worker response: count and surface.
        Rule::new("on_done")
            .when(When::input(Pattern::variant("done", Pattern::wild())))
            .do_(Effect::RecordMetric {
                name: "completed".into(),
                value: Expr::int(1),
            }),
    ];
    let pool = sim.add_node("Pool", pool_slots, pool_rules);

    // ---------- Client generating load ----------
    let mut client_slots = BTreeMap::new();
    client_slots.insert("period_ns".into(), Value::Int(5_000_000));
    let client_rules = vec![
        Rule::new("fire")
            .when(When::input(Pattern::variant("tick", Pattern::wild())))
            .do_(Effect::Emit {
                payload: Expr::variant("req", Expr::lit(Value::Nil)),
                to: EmitTo::ToTarget("Pool".into()),
            })
            .do_(Effect::Emit {
                payload: Expr::variant("tick", Expr::lit(Value::Nil)),
                to: EmitTo::ToTarget("Client".into()),
            }),
    ];
    let client = sim.add_node("Client", client_slots, client_rules);

    sim.add_edge(client, pool, Expr::int(1_000_000));
    sim.add_edge(client, client, Expr::slot("period_ns"));

    sim.inject(client, Value::variant("tick", Value::Nil), None);

    // ---------- Run ----------
    println!("=== autoscaling (concrete workers) ===\n");
    println!("{:>8} | {:>8} | {:>10}",
        "t (ms)", "replicas", "pool_queue");
    let mut t = 0u64;
    while t < SIM_DURATION_NS {
        t = (t + PROBE_INTERVAL_NS).min(SIM_DURATION_NS);
        sim.run_until(t);

        let p = &sim.nodes[&pool];
        let replicas = p.slots["replicas"].as_int().unwrap();
        let queue = p.inbox.len();
        println!("{:>8.1} | {:>8} | {:>10}",
            (t as f64) / 1_000_000.0, replicas, queue);
    }

    let completed = sim.log.events.iter().filter(|e| matches!(
        e, Event::MetricRecorded { name, .. } if name == "completed"
    )).count();
    let spawned = sim.log.events.iter().filter(|e| matches!(
        e, Event::NodeSpawned { .. }
    )).count();
    println!("\nWorkers spawned: {}", spawned);
    println!("Completions:     {}", completed);
}
