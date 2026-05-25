//! Mutex contention with stochastic service times.
//!
//! One Mutex node, N Workers. Each worker continuously acquires the
//! mutex, "works" for a stochastic duration (modeled as the latency of
//! the grant-reply edge), then releases and re-acquires.
//!
//! Observable: the Mutex's inbox depth over time = number of workers
//! waiting. Under saturation this queues up; under plenty of capacity
//! it stays near zero.
//!
//! Run: `cargo run -p flow --example mutex_contention`

use std::collections::BTreeMap;

use flow::{Effect, EmitTo, Event, Expr, Pattern, Rule, Sim, Value, When};

const NUM_WORKERS: usize = 4;
const SERVICE_MEAN_NS: f64 = 10_000_000.0; // 10 ms avg service time
const SIM_DURATION_NS: u64 = 500_000_000;  // 500 ms of simulated time
const PROBE_INTERVAL_NS: u64 = 20_000_000; // sample queue depth every 20 ms

fn main() {
    let mut sim = Sim::new(42);

    // ---------- Mutex ----------
    let mut mutex_slots = BTreeMap::new();
    mutex_slots.insert("holder".into(), Value::variant("None", Value::Nil));

    let mutex_rules = vec![
        // Grant if free: bind the requester's id from the acquire payload.
        Rule::new("grant")
            .when(When::input(Pattern::variant("acquire", Pattern::var("who"))))
            .when(When::slot("holder", Pattern::variant("None", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "holder".into(),
                value: Expr::variant("Some", Expr::var("who")),
            })
            .do_(Effect::respond(
                Expr::variant("granted", Expr::lit(Value::Nil)),
            )),
        // Release frees the holder slot.
        Rule::new("release")
            .when(When::input(Pattern::variant("release", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "holder".into(),
                value: Expr::variant("None", Expr::lit(Value::Nil)),
            }),
    ];
    let mutex = sim.add_node("Mutex", mutex_slots, mutex_rules);

    // ---------- Workers ----------
    let mut workers = Vec::new();
    for i in 0..NUM_WORKERS {
        let wname = format!("W{}", i);

        let mut slots = BTreeMap::new();
        slots.insert("idle".into(), Value::Bool(true));
        slots.insert("completed".into(), Value::Int(0));

        let rules = vec![
            // Kick off an acquire whenever idle. With no Input pattern, this
            // fires as soon as `idle` is true. Setting idle=false before Emit
            // prevents the runaway re-fire within the same instant.
            Rule::new("request")
                .when(When::slot("idle", Pattern::lit(Value::Bool(true))))
                .do_(Effect::SetSlot {
                    slot: "idle".into(),
                    value: Expr::bool(false),
                })
                .do_(Effect::emit(
                    Expr::variant("acquire", Expr::lit(Value::str(wname.clone()))),
                    EmitTo::ToTarget("Mutex".into()),
                )),
            // On grant: bump counter, release immediately, go back to idle.
            Rule::new("on_granted")
                .when(When::input(Pattern::variant("granted", Pattern::wild())))
                .do_(Effect::SetSlot {
                    slot: "completed".into(),
                    value: Expr::add(Expr::slot("completed"), Expr::int(1)),
                })
                .do_(Effect::RecordMetric {
                    name: "completed".into(),
                    value: Expr::slot("completed"),
                })
                .do_(Effect::emit(
                    Expr::variant("release", Expr::lit(Value::Nil)),
                    EmitTo::ToTarget("Mutex".into()),
                ))
                .do_(Effect::SetSlot {
                    slot: "idle".into(),
                    value: Expr::bool(true),
                }),
        ];

        let w = sim.add_node(&wname, slots, rules);
        workers.push(w);

        // Worker → Mutex: control messages (acquire/release) are near-instant.
        sim.add_edge(w, mutex, Expr::int(1_000)); // 1 μs
        // Mutex → Worker: the grant reply carries the service-time delay.
        // This is where "work" happens — during grant travel the holder slot
        // is already Some(who), so contention shows up in the mutex inbox.
        sim.add_edge(mutex, w, Expr::exp_dist(Expr::float(SERVICE_MEAN_NS)));
    }

    // ---------- Run ----------
    //
    // Sample the mutex inbox depth at regular intervals by stepping in
    // PROBE_INTERVAL_NS chunks and reading the inbox length between runs.
    // (This is a scripted observer — no built-in rule is needed.)
    println!("=== mutex contention ===\n");
    println!("{:>8} | {:>10} | {:>6} | per-worker completed",
        "t (ms)", "holder", "queue");

    let mut t = 0u64;
    while t < SIM_DURATION_NS {
        t = (t + PROBE_INTERVAL_NS).min(SIM_DURATION_NS);
        sim.run_until(t);

        let mnode = &sim.nodes[&mutex];
        let holder_str = match (&mnode.slots["holder"]).as_variant() {
            Some(("Some", Value::Str(s))) => s.clone(),
            Some(("Some", _)) => "?".into(),
            _ => "-".into(),
        };
        let queue = mnode.inbox.len();

        let per_worker: Vec<String> = workers.iter().map(|wid| {
            let w = &sim.nodes[wid];
            match w.slots.get("completed") {
                Some(Value::Int(n)) => format!("{}={}", w.name, n),
                _ => format!("{}=?", w.name),
            }
        }).collect();

        println!("{:>8.1} | {:>10} | {:>6} | {}",
            (t as f64) / 1_000_000.0,
            holder_str,
            queue,
            per_worker.join("  "));
    }

    // Total completions recorded via metric events:
    let total_completed: usize = sim.log.iter().filter(|e| matches!(
        e, Event::MetricRecorded { name, .. } if name == "completed"
    )).count();
    println!("\nTotal completions: {}", total_completed);
    println!("Expected under perfect serialization: {} completions / ({:.0} ns each) ≈ {:.0}",
        total_completed,
        SERVICE_MEAN_NS,
        (SIM_DURATION_NS as f64) / SERVICE_MEAN_NS);
}
