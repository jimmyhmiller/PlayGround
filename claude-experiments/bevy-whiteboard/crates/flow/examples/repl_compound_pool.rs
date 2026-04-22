//! Drop into the REPL with the compound_pool sim pre-loaded.
//!
//!   cargo run -p flow --example repl_compound_pool
//!
//! Try:
//!   help
//!   inspect all
//!   run 100ms
//!   metrics
//!   set_param service_mean Exp(200ms)   # slow the service way down
//!   run 200ms
//!   snap save before_speedup
//!   set_param service_mean 5ms          # speed it way up
//!   run 400ms
//!   metrics
//!   snap restore before_speedup
//!   run 400ms                           # alternate reality: slower service
//!
//! This is the payoff for params-are-live + snapshot: you can try
//! different parameter values from the same prior history without
//! re-running from scratch.

use std::collections::BTreeMap;

use flow::{Effect, EmitTo, Expr, Pattern, Rule, Samples, Sim, Value, When, repl::Repl};

fn main() {
    let mut sim = Sim::new(11);

    // Live params.
    sim.set_param("network",      Expr::int(1_000_000));
    sim.set_param("service_mean", Expr::int(25_000_000));
    sim.set_param("dispatch_cost", Expr::int(1_000));

    // Router (inner).
    let mut router_slots = BTreeMap::new();
    router_slots.insert("ready".into(), Value::Samples(Samples::new(64)));
    let router_rules = vec![
        Rule::new("worker_ready")
            .when(When::input(Pattern::variant("ready", Pattern::var("w"))))
            .do_(Effect::SamplesPush { slot: "ready".into(), value: Expr::var("w") }),
        Rule::new("dispatch")
            .when(When::input(Pattern::variant("req", Pattern::wild())))
            .guard(Expr::gt(Expr::samples_len("ready"), Expr::int(0)))
            .do_(Effect::SamplesPopOldestInto { slot: "ready".into(), into_var: "w".into() })
            .do_(Effect::Emit {
                payload: Expr::variant("work", Expr::lit(Value::Nil)),
                to: EmitTo::ToTargetExpr(Expr::var("w")),
            }),
        Rule::new("worker_done")
            .when(When::input(Pattern::variant("done", Pattern::var("w"))))
            .do_(Effect::SamplesPush { slot: "ready".into(), value: Expr::var("w") })
            .do_(Effect::Emit {
                payload: Expr::variant("resp", Expr::lit(Value::Nil)),
                to: EmitTo::ToOutPort("resp".into()),
            }),
    ];
    let router = sim.add_node("Router", router_slots, router_rules);

    // Workers (inner).
    let mut worker_ids = Vec::new();
    for i in 0..3 {
        let name = format!("W{}", i);
        let rules = vec![
            Rule::new("accept_work")
                .when(When::input(Pattern::variant("work", Pattern::wild())))
                .do_(Effect::Emit {
                    payload: Expr::variant("tick_done", Expr::lit(Value::Nil)),
                    to: EmitTo::ToTargetExpr(Expr::self_ref()),
                }),
            Rule::new("publish_done")
                .when(When::input(Pattern::variant("tick_done", Pattern::wild())))
                .do_(Effect::Emit {
                    payload: Expr::variant("done", Expr::self_ref()),
                    to: EmitTo::ToTarget("Router".into()),
                }),
        ];
        let w = sim.add_node(&name, BTreeMap::new(), rules);
        worker_ids.push(w);
    }

    // Compound wrapping Router + Workers.
    let mut in_ports = BTreeMap::new();
    in_ports.insert("req".into(), router);
    let mut out_ports = BTreeMap::new();
    out_ports.insert("resp".into(), router);
    let pool = sim.add_compound("Pool", in_ports, out_ports);

    for w in &worker_ids {
        sim.add_edge(router, *w, Expr::param("dispatch_cost"));
        sim.add_edge(*w, router, Expr::param("dispatch_cost"));
        sim.add_edge(*w, *w, Expr::exp_dist(Expr::param("service_mean")));
    }

    // Seed ready list.
    {
        let router_node = sim.nodes.get_mut(&router).unwrap();
        if let Value::Samples(s) = router_node.slots.get_mut("ready").unwrap() {
            for w in &worker_ids { s.push(Value::NodeRef(*w)); }
        }
    }

    // External client.
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
        Rule::new("got_resp")
            .when(When::input(Pattern::variant("resp", Pattern::wild())))
            .do_(Effect::RecordMetric { name: "completed".into(), value: Expr::int(1) }),
    ];
    let client = sim.add_node("Client", BTreeMap::new(), client_rules);

    sim.add_edge_ports(client, None, pool, Some("req".into()), Expr::param("network"));
    sim.add_edge_ports(pool, Some("resp".into()), client, None, Expr::param("network"));
    sim.add_edge(client, client, Expr::int(10_000_000));
    sim.inject(client, Value::variant("tick", Value::Nil), None);

    Repl::new(sim).run();
}
