//! Scripted scenario: mid-run network partition.
//!
//! Baseline: Client ticks every 5 ms, sending requests to a Server
//! that responds with 10 ms service time.
//!
//! Scripted: at t=200 ms, the Client→Server edge slows by 100× (1 ms
//! to 100 ms). At t=400 ms it recovers.
//!
//! Note: edge latency is evaluated at *emit time* — so packets emitted
//! during the slow window travel slowly even after healing, draining
//! gradually. This matches real partitions: in-flight requests don't
//! speed up just because the link recovered.
//!
//! Run: `cargo run -p flow --example scenario_partition`

use std::collections::BTreeMap;

use flow::{
    Action, Effect, EmitTo, Expr, Pattern, Rule, Samples, Scenario, Sim,
    Value, When,
};

fn main() {
    let mut sim = Sim::new(11);

    // ---------- Server ----------
    let server_rules = vec![
        Rule::new("respond")
            .when(When::input(Pattern::variant("req", Pattern::wild())))
            .do_(Effect::respond(
                Expr::variant("resp", Expr::lit(Value::Nil)),
            )),
    ];
    let server = sim.add_node("Server", BTreeMap::new(), server_rules);

    // ---------- Client ----------
    let mut client_slots = BTreeMap::new();
    client_slots.insert("in_flight".into(), Value::Int(0));
    client_slots.insert("sent_at".into(),   Value::Samples(Samples::new(10_000)));

    let client_rules = vec![
        Rule::new("send")
            .when(When::input(Pattern::variant("tick", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "in_flight".into(),
                value: Expr::add(Expr::slot("in_flight"), Expr::int(1)),
            })
            .do_(Effect::SamplesPush { slot: "sent_at".into(), value: Expr::now() })
            .do_(Effect::emit(
                Expr::variant("req", Expr::lit(Value::Nil)),
                EmitTo::ToTarget("Server".into()),
            ))
            .do_(Effect::emit(
                Expr::variant("tick", Expr::lit(Value::Nil)),
                EmitTo::ToTarget("Client".into()),
            )),
        Rule::new("recv")
            .when(When::input(Pattern::variant("resp", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "in_flight".into(),
                value: Expr::sub(Expr::slot("in_flight"), Expr::int(1)),
            })
            .do_(Effect::SamplesPopOldestInto {
                slot: "sent_at".into(),
                into_var: "sent".into(),
            }),
    ];
    let client = sim.add_node("Client", client_slots, client_rules);

    // Edges.
    let fwd = sim.add_edge(client, server, Expr::int(1_000_000));   // 1ms forward
    sim.add_edge(server, client, Expr::int(10_000_000));             // 10ms reply
    sim.add_edge(client, client, Expr::int(5_000_000));              // 5ms tick loop

    // Kick off the ticker.
    sim.inject(client, Value::variant("tick", Value::Nil));

    // ---------- Scenario: partition from t=200ms to t=400ms ----------
    let scenario = Scenario::new()
        // Outage begins: forward edge latency jumps 100× to 100 ms.
        .at(200_000_000, Action::SetEdgeLatency {
            edge: fwd,
            latency: Expr::int(100_000_000),
        })
        // Partition heals.
        .at(400_000_000, Action::SetEdgeLatency {
            edge: fwd,
            latency: Expr::int(1_000_000),
        });
    sim.load_scenario(scenario);

    // ---------- Run ----------
    println!("=== scenario_partition ===\n");
    println!("Slowdown:    t = 200 – 400 ms   (forward latency 100 ms)");
    println!("Ticks every 5 ms → 200 reqs/s nominal\n");
    println!("{:>8} | {:>10} | {:>6}", "t (ms)", "in_flight", "note");

    let probes: &[(u64, &str)] = &[
        (100_000_000,  "pre"),
        (150_000_000,  "pre"),
        (200_000_000,  "partition↓"),
        (250_000_000,  "partitioned"),
        (300_000_000,  "partitioned"),
        (350_000_000,  "partitioned"),
        (400_000_000,  "partition↑"),
        (450_000_000,  "recovering"),
        (500_000_000,  "recovering"),
        (600_000_000,  "post"),
        (800_000_000,  "post"),
        (1_000_000_000,"post"),
    ];
    for (t, note) in probes {
        sim.run_until(*t);
        let c = &sim.nodes[&client];
        let in_flight = c.slots["in_flight"].as_int().unwrap();
        println!("{:>8.1} | {:>10} | {:>6}",
            (*t as f64) / 1_000_000.0, in_flight, note);
    }
}
