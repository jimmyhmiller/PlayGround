//! Pool routing by "available worker" — entirely emergent from
//! existing primitives. No new substrate.
//!
//! This is the correct shape for routing policies: the pool doesn't
//! have a special "least-loaded" operator baked in. It simply tracks
//! a ready-list, and workers publish their readiness via messages.
//! Backpressure, fairness, and availability-sensitivity all fall out.
//!
//! Mechanism:
//!   - Pool slot `ready: Samples<NodeRef>` — workers currently idle
//!   - Worker sends `ready{me}` on startup and after each completion
//!   - Pool's dispatch rule fires only when `ready` is non-empty
//!   - If `ready` is empty, requests pile up in the Pool's inbox
//!     (natural backpressure); when a worker becomes free, it pushes
//!     into `ready` and the next firing of `dispatch` picks it up
//!
//! Run: `cargo run -p flow --example pool_least_busy`

use std::collections::BTreeMap;

use flow::{
    EdgeEnd, Effect, EmitTo, Event, Expr, Pattern, Rule, Samples, Sim,
    Template, Value, When,
};

const NUM_WORKERS: usize = 3;
const SERVICE_MEAN_NS: f64 = 20_000_000.0;  // 20 ms
const REQUEST_PERIOD_NS: i64 = 8_000_000;   // 8 ms
const SIM_DURATION_NS: u64 = 600_000_000;
const PROBE_INTERVAL_NS: u64 = 25_000_000;

fn main() {
    let mut sim = Sim::new(42);

    // ---------- Worker template ----------
    //
    // No `me` slot. No bootstrap rule. No `set_id` packet. The worker
    // uses `Expr::SelfRef` to refer to its own NodeRef wherever needed.
    let worker = Template::new("Worker")
        .with_prefix("W")
        // Take work: dispatch a `done` to ourselves via the self-loop edge
        // (whose latency is the stochastic service time).
        .rule(
            Rule::new("take_work")
                .when(When::input(Pattern::variant("work", Pattern::wild())))
                .do_(Effect::Emit {
                    payload: Expr::variant("done", Expr::lit(Value::Nil)),
                    to: EmitTo::ToTargetExpr(Expr::self_ref()),
                })
        )
        // Work finished: announce readiness to Pool, carrying our own ref.
        .rule(
            Rule::new("finish")
                .when(When::input(Pattern::variant("done", Pattern::wild())))
                .do_(Effect::Emit {
                    payload: Expr::variant("ready", Expr::self_ref()),
                    to: EmitTo::ToTarget("Pool".into()),
                })
        )
        .edge(EdgeEnd::Parent, EdgeEnd::ThisInstance, Expr::int(1_000))
        .edge(EdgeEnd::ThisInstance, EdgeEnd::Parent, Expr::int(1_000))
        .edge(
            EdgeEnd::ThisInstance,
            EdgeEnd::ThisInstance,
            Expr::exp_dist(Expr::float(SERVICE_MEAN_NS)),
        );
    sim.register_template(worker);

    // ---------- Pool ----------
    //
    // Slot `ready` is a Samples ring of NodeRefs. Dispatching
    // pops oldest (FIFO-fair among ready workers); a guard ensures
    // we only dispatch when someone's available.
    let mut pool_slots = BTreeMap::new();
    pool_slots.insert("ready".into(), Value::Samples(Samples::new(128)));
    pool_slots.insert("dispatched".into(), Value::Int(0));

    let pool_rules = vec![
        // Worker announces readiness — enqueue.
        Rule::new("worker_ready")
            .when(When::input(Pattern::variant("ready", Pattern::var("w"))))
            .do_(Effect::SamplesPush {
                slot: "ready".into(),
                value: Expr::var("w"),
            }),

        // Dispatch a request to the oldest ready worker — but only
        // if one exists. If not, `req` stays in inbox until someone
        // publishes readiness, which re-enables this rule.
        Rule::new("dispatch")
            .when(When::input(Pattern::variant("req", Pattern::wild())))
            .guard(Expr::gt(Expr::samples_len("ready"), Expr::int(0)))
            .do_(Effect::SamplesPopOldestInto {
                slot: "ready".into(),
                into_var: "w".into(),
            })
            .do_(Effect::SetSlot {
                slot: "dispatched".into(),
                value: Expr::add(Expr::slot("dispatched"), Expr::int(1)),
            })
            .do_(Effect::Emit {
                payload: Expr::variant("work", Expr::lit(Value::Nil)),
                to: EmitTo::ToTargetExpr(Expr::var("w")),
            }),
    ];
    let pool = sim.add_node("Pool", pool_slots, pool_rules);

    // ---------- Spawn workers at boot ----------
    //
    // A single rule: on `kick`, Spawn a Worker (bound to `w`) and push
    // its NodeRef directly into `ready`. One atomic firing — no
    // round-trip, no bootstrap rule on the worker side.
    let spawn_rule = Rule::new("spawn_and_enroll")
        .when(When::input(Pattern::variant("kick", Pattern::wild())))
        .do_(Effect::Spawn {
            template: "Worker".into(),
            into_var: Some("w".into()),
        })
        .do_(Effect::SamplesPush {
            slot: "ready".into(),
            value: Expr::var("w"),
        });
    sim.nodes.get_mut(&pool).unwrap().rules.push(spawn_rule);

    for _ in 0..NUM_WORKERS {
        sim.inject(pool, Value::variant("kick", Value::Nil), None);
    }

    // ---------- Client generating load ----------
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
    let client = sim.add_node("Client", BTreeMap::new(), client_rules);
    sim.add_edge(client, pool, Expr::int(1_000_000));
    sim.add_edge(client, client, Expr::int(REQUEST_PERIOD_NS));
    sim.inject(client, Value::variant("tick", Value::Nil), None);

    // ---------- Observe ----------
    println!("=== pool_least_busy ===\n");
    println!("{} workers, {} ms service mean, {} ms request period.",
        NUM_WORKERS, (SERVICE_MEAN_NS / 1_000_000.0) as i64, REQUEST_PERIOD_NS / 1_000_000);
    println!("ρ = arrival_rate / service_rate = {:.2}\n",
        (1_000_000_000.0 / REQUEST_PERIOD_NS as f64) /
        ((1_000_000_000.0 / SERVICE_MEAN_NS) * NUM_WORKERS as f64));

    println!("{:>8} | {:>7} | {:>10} | {:>11}",
        "t (ms)", "ready", "pool_queue", "dispatched");

    let mut t = 0u64;
    while t < SIM_DURATION_NS {
        t = (t + PROBE_INTERVAL_NS).min(SIM_DURATION_NS);
        sim.run_until(t);

        let p = &sim.nodes[&pool];
        let ready = match &p.slots["ready"] {
            Value::Samples(s) => s.len(),
            _ => 0,
        };
        let q = p.inbox.len();
        let dispatched = p.slots["dispatched"].as_int().unwrap();

        println!("{:>8.1} | {:>7} | {:>10} | {:>11}",
            (t as f64) / 1_000_000.0, ready, q, dispatched);
    }

    let spawned = sim.log.events.iter().filter(|e| matches!(
        e, Event::NodeSpawned { .. }
    )).count();
    println!("\nWorkers spawned: {}", spawned);
}
