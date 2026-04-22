//! Compound nodes — the "node contains nodes" story.
//!
//! We build a `LoadBalancedPool` as a COMPOUND NODE whose internals are
//! a Router leaf + 3 Worker leaves wired with internal edges. The
//! compound exposes two ports:
//!
//!   in_port  "req"  → Router
//!   out_port "resp" → one of the Workers (via a shared "ready" list
//!                     inside Router; response comes back through the
//!                     same Router)
//!
//! From OUTSIDE the compound, the Client sees ONE node (the Pool) with
//! two port labels. Client → Pool.req delivers a request; Pool.resp →
//! Client receives the response. The Client doesn't know how many
//! Workers the Pool contains, or anything about routing policy.
//!
//! From INSIDE the compound, Workers don't know the Client exists.
//! They just receive work from the Router and emit `resp` on their
//! compound's out-port — wherever that routes externally is someone
//! else's problem.
//!
//! That's OO-style encapsulation, mechanically enforced: port names
//! are the only shared vocabulary between outer and inner.
//!
//! Run: `cargo run -p flow --example compound_pool`

use std::collections::BTreeMap;

use flow::{Effect, EmitTo, Event, Expr, Pattern, Rule, Samples, Sim, Value, When};

fn main() {
    let mut sim = Sim::new(11);

    // ---------- Params (live-editable) ----------
    //
    // Note how times become a global tuning surface. Edges and rules
    // below read these by name; `sim.set_param(...)` at any moment
    // rebinds them globally.
    sim.set_param("network",      Expr::int(1_000_000));         // 1 ms
    sim.set_param("service_mean", Expr::int(25_000_000));        // 25 ms
    sim.set_param("dispatch_cost", Expr::int(1_000));            // 1 μs

    // ---------- Inner nodes (of the compound) ----------
    //
    // We create them first, *then* wrap them into a compound. Their
    // parent is set automatically by add_compound when referenced in
    // port maps.

    // Router: receives `req`, picks a ready worker, emits `work`.
    //         Receives `done` from a worker, emits `resp` on out-port.
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
        // Worker finished: push their ref back to ready AND forward a
        // `resp` out through the compound's out-port.
        Rule::new("worker_done")
            .when(When::input(Pattern::variant("done", Pattern::var("w"))))
            .do_(Effect::SamplesPush { slot: "ready".into(), value: Expr::var("w") })
            .do_(Effect::Emit {
                payload: Expr::variant("resp", Expr::lit(Value::Nil)),
                to: EmitTo::ToOutPort("resp".into()),
            }),
    ];
    let router = sim.add_node("Router", router_slots, router_rules);

    // Workers.
    let mut worker_ids = Vec::new();
    for i in 0..3 {
        let name = format!("W{}", i);
        let rules = vec![
            Rule::new("accept_work")
                .when(When::input(Pattern::variant("work", Pattern::wild())))
                .do_(Effect::Emit {
                    payload: Expr::variant("tick_done", Expr::lit(Value::Nil)),
                    to: EmitTo::ToTargetExpr(Expr::self_ref()),  // service-time self-loop
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

    // ---------- Compound node ("Pool") ----------
    //
    // in_port "req"  → Router   (external requests land here)
    // out_port "resp" → Router  (Router is what emits `resp` via ToOutPort)
    //
    // Note the authoring rhythm: draw the internals, then wrap them.
    let mut in_ports = BTreeMap::new();
    in_ports.insert("req".into(), router);
    let mut out_ports = BTreeMap::new();
    out_ports.insert("resp".into(), router);
    let pool = sim.add_compound("Pool", in_ports, out_ports);

    // ---------- Internal edges (Router ↔ Workers) ----------
    //
    // These are ordinary edges; inner nodes talk to each other
    // directly. External nodes can't reach them.
    for w in &worker_ids {
        sim.add_edge(router, *w, Expr::param("dispatch_cost"));   // Router → Worker
        sim.add_edge(*w, router, Expr::param("dispatch_cost"));   // Worker → Router
        sim.add_edge(*w, *w, Expr::exp_dist(Expr::param("service_mean"))); // work time
    }

    // Pre-populate Router's ready list with all workers (bootstrap).
    // This is a one-shot setup — a real system would have workers
    // announce themselves on spawn; here we just seed.
    {
        let router_node = sim.nodes.get_mut(&router).unwrap();
        if let Value::Samples(s) = router_node.slots.get_mut("ready").unwrap() {
            for w in &worker_ids { s.push(Value::NodeRef(*w)); }
        }
    }

    // ---------- External: Client ----------
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

    // External edges: Client → Pool.req, Pool.resp → Client, Client self-tick.
    sim.add_edge_ports(client, None, pool, Some("req".into()), Expr::param("network"));
    sim.add_edge_ports(pool, Some("resp".into()), client, None, Expr::param("network"));
    sim.add_edge(client, client, Expr::int(10_000_000));  // 10ms tick period

    sim.inject(client, Value::variant("tick", Value::Nil), None);

    // ---------- Run ----------
    sim.run_until(500_000_000);
    let completed = sim.log.events.iter().filter(|e| matches!(
        e, Event::MetricRecorded { name, .. } if name == "completed"
    )).count();
    println!("=== compound_pool ===");
    println!("Compound Pool wraps Router + 3 Workers behind ports req/resp.");
    println!("Client sees only one node (Pool).");
    println!("Completed in 500ms: {}", completed);
    println!("Nodes in sim: {} ({} compound)", sim.nodes.len(),
        sim.nodes.values().filter(|n| n.is_compound()).count());
}

