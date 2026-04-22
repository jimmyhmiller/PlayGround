//! Serialized server — the one with a growing queue.
//!
//! Promise-aggregate showed an *infinite-parallelism* server
//! converging to Little's Law (in_flight ≈ λ·W). This example forces
//! serialization: the Server accepts ONE request at a time, stashes
//! the requester in a slot, works for ~20ms, then emits the response
//! back via *dynamic routing* (`EmitTo::ToTargetExpr`) to whoever
//! that stashed name names.
//!
//! Without dynamic emit, there's no way for a serialized server to
//! know "whom to respond to" after the service-time timer fires,
//! because `Respond` is tied to the packet currently being consumed.
//!
//! Expected: Server's inbox queue grows without bound (request rate
//! 200/s, service rate 50/s → ρ = 4.0). Client's `in_flight` tracks
//! this growth faithfully without ever storing request IDs.
//!
//! Run: `cargo run -p flow --example serialized_server`

use std::collections::BTreeMap;

use flow::{Effect, EmitTo, Event, Expr, Pattern, Rule, Samples, Sim, Value, When};

const REQUEST_PERIOD_NS: i64 = 5_000_000;     // 5 ms between sends
const SERVICE_MEAN_NS:   f64 = 20_000_000.0;  // 20 ms avg service
const SIM_DURATION_NS:   u64 = 600_000_000;   // 600 ms
const PROBE_INTERVAL_NS: u64 = 25_000_000;

fn main() {
    let mut sim = Sim::new(7);

    // ---------- Server (single-threaded) ----------
    let mut server_slots = BTreeMap::new();
    server_slots.insert("busy".into(),           Value::Bool(false));
    server_slots.insert("current_client".into(), Value::Str(String::new()));
    server_slots.insert("total_served".into(),   Value::Int(0));

    let server_rules = vec![
        // Accept a request only when idle. Stash the requester's name.
        // The packet format is Variant("req", Record { who: <name> }).
        Rule::new("accept")
            .when(When::input(Pattern::variant(
                "req",
                Pattern::record([("who", Pattern::var("client_name"))]),
            )))
            .when(When::slot("busy", Pattern::lit(Value::Bool(false))))
            .do_(Effect::SetSlot { slot: "busy".into(), value: Expr::bool(true) })
            .do_(Effect::SetSlot {
                slot: "current_client".into(),
                value: Expr::var("client_name"),
            })
            // Emit work-timer to self. The self-loop edge carries the service
            // time, so `complete` fires after ~service_mean ns.
            .do_(Effect::Emit {
                payload: Expr::variant("work_done", Expr::lit(Value::Nil)),
                to: EmitTo::ToTarget("Server".into()),
            }),

        // Work completed: emit response to whoever's name is stashed,
        // bump counter, clear busy.
        Rule::new("complete")
            .when(When::input(Pattern::variant("work_done", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "total_served".into(),
                value: Expr::add(Expr::slot("total_served"), Expr::int(1)),
            })
            .do_(Effect::Emit {
                payload: Expr::variant("resp", Expr::lit(Value::Nil)),
                to: EmitTo::ToTargetExpr(Expr::slot("current_client")),
            })
            .do_(Effect::SetSlot { slot: "busy".into(), value: Expr::bool(false) }),
    ];
    let server = sim.add_node("Server", server_slots, server_rules);

    // ---------- Client ----------
    let mut client_slots = BTreeMap::new();
    client_slots.insert("in_flight".into(), Value::Int(0));
    client_slots.insert("sent_at".into(),   Value::Samples(Samples::new(100_000)));
    client_slots.insert("peak".into(),      Value::Int(0));

    let client_rules = vec![
        Rule::new("send")
            .when(When::input(Pattern::variant("tick", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "in_flight".into(),
                value: Expr::add(Expr::slot("in_flight"), Expr::int(1)),
            })
            .do_(Effect::SamplesPush {
                slot: "sent_at".into(),
                value: Expr::now(),
            })
            .do_(Effect::SetSlot {
                slot: "peak".into(),
                value: Expr::if_(
                    Expr::gt(Expr::slot("in_flight"), Expr::slot("peak")),
                    Expr::slot("in_flight"),
                    Expr::slot("peak"),
                ),
            })
            // Payload carries the client's own name so the server can stash it.
            .do_(Effect::Emit {
                payload: Expr::variant(
                    "req",
                    Expr::record([("who", Expr::lit(Value::str("Client")))]),
                ),
                to: EmitTo::ToTarget("Server".into()),
            })
            .do_(Effect::Emit {
                payload: Expr::variant("tick", Expr::lit(Value::Nil)),
                to: EmitTo::ToTarget("Client".into()),
            }),
        Rule::new("recv")
            .when(When::input(Pattern::variant("resp", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "in_flight".into(),
                value: Expr::sub(Expr::slot("in_flight"), Expr::int(1)),
            })
            .do_(Effect::SamplesPopOldestInto {
                slot: "sent_at".into(),
                into_var: "sent".into(),
            })
            .do_(Effect::RecordMetric {
                name: "rtt_ns".into(),
                value: Expr::sub(Expr::now(), Expr::var("sent")),
            }),
    ];
    let client = sim.add_node("Client", client_slots, client_rules);

    // ---------- Edges ----------
    // Client → Server request path: 1 ms network.
    sim.add_edge(client, server, Expr::int(1_000_000));
    // Server → Client reply path: 1 ms network. Service time is NOT here —
    // it's on the Server's self-loop.
    sim.add_edge(server, client, Expr::int(1_000_000));
    // Client self-loop for ticking.
    sim.add_edge(client, client, Expr::int(REQUEST_PERIOD_NS));
    // Server self-loop: service time lives here.
    sim.add_edge(server, server, Expr::exp_dist(Expr::float(SERVICE_MEAN_NS)));

    sim.inject(client, Value::variant("tick", Value::Nil), None);

    // ---------- Observe ----------
    println!("=== serialized_server ===\n");
    println!("Request period: {} ms  |  Service mean: {} ms",
        REQUEST_PERIOD_NS / 1_000_000, (SERVICE_MEAN_NS / 1_000_000.0) as i64);
    println!("ρ = 4.0 (saturated) — Client in_flight and Server queue should both grow.\n");
    println!("{:>8} | {:>10} | {:>12} | {:>10}",
        "t (ms)", "in_flight", "server_queue", "rtt_avg_ms");

    let mut t = 0u64;
    while t < SIM_DURATION_NS {
        t = (t + PROBE_INTERVAL_NS).min(SIM_DURATION_NS);
        sim.run_until(t);

        let c = &sim.nodes[&client];
        let s = &sim.nodes[&server];
        let in_flight = c.slots["in_flight"].as_int().unwrap();
        let server_queue = s.inbox.len();

        let recent_rtts: Vec<f64> = sim.log.events.iter().rev()
            .take(200)
            .filter_map(|e| match e {
                Event::MetricRecorded { name, value, .. } if name == "rtt_ns" => value.as_float(),
                _ => None,
            })
            .collect();
        let avg_rtt_ms = if recent_rtts.is_empty() { 0.0 } else {
            recent_rtts.iter().sum::<f64>() / (recent_rtts.len() as f64) / 1_000_000.0
        };

        println!("{:>8.1} | {:>10} | {:>12} | {:>10.2}",
            (t as f64) / 1_000_000.0, in_flight, server_queue, avg_rtt_ms);
    }

    let peak = sim.nodes[&client].slots["peak"].as_int().unwrap();
    let total_served = sim.nodes[&server].slots["total_served"].as_int().unwrap();
    println!("\nPeak in_flight: {}", peak);
    println!("Total served:   {}", total_served);
}
