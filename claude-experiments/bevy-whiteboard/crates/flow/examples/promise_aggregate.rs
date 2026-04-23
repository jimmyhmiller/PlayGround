//! Promise-map growth — *modeled, not implemented*.
//!
//! The Client tracks pending requests with two slots:
//!   - `in_flight: Int`            — scalar count of outstanding
//!   - `sent_at:   Samples<Time>`  — bounded ring of recent send times
//!
//! It does NOT track individual request IDs. "Pretend there are 10M
//! outstanding" would work identically — memory is bounded regardless
//! of scale.
//!
//! Setup: Client fires requests every 5ms. Server responds with a
//! stochastic Exp(20ms) latency on the reply edge — this Server is
//! *infinite-parallelism*, so each response travels independently.
//!
//! Expected behavior (Little's Law):
//!    in_flight_steady = arrival_rate × mean_response_time
//!                     = 200/s × 20 ms = 4
//!
//! For a demo of *serialized* backup (where in_flight grows without
//! bound), a second example `promise_aggregate_serial.rs` could model
//! a single-threaded Server with a busy slot and a work-timer self-loop.
//! (Not included in v1; requires a small extension for dynamic routing.)
//!
//! Run: `cargo run -p flow --example promise_aggregate`

use std::collections::BTreeMap;

use flow::{Effect, EmitTo, Event, Expr, Pattern, Rule, Samples, Sim, Value, When};

const REQUEST_PERIOD_NS: i64 = 5_000_000;    // 5 ms between sends
const SERVICE_MEAN_NS:   f64 = 20_000_000.0; // 20 ms avg service
const SIM_DURATION_NS:   u64 = 600_000_000;  // 600 ms total
const PROBE_INTERVAL_NS: u64 = 25_000_000;   // 25 ms probe cadence

fn main() {
    let mut sim = Sim::new(7);

    // ---------- Server ----------
    // Single-threaded responder. Each received request is responded to
    // immediately, with service time on the reply edge.
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
    client_slots.insert("peak".into(),      Value::Int(0));

    let client_rules = vec![
        // On tick: send a request, update counters.
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
            // Track peak for final reporting.
            .do_(Effect::SetSlot {
                slot: "peak".into(),
                value: Expr::if_(
                    Expr::gt(Expr::slot("in_flight"), Expr::slot("peak")),
                    Expr::slot("in_flight"),
                    Expr::slot("peak"),
                ),
            })
            .do_(Effect::emit(
                Expr::variant("req", Expr::lit(Value::Nil)),
                EmitTo::ToTarget("Server".into()),
            ))
            // Re-tick on self-loop (fires another tick `REQUEST_PERIOD_NS` later).
            .do_(Effect::emit(
                Expr::variant("tick", Expr::lit(Value::Nil)),
                EmitTo::ToTarget("Client".into()),
            )),
        // On response: decrement count, pop oldest sent_at, record RTT.
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
    // Client → Server: fast network (1 ms).
    sim.add_edge(client, server, Expr::int(1_000_000));
    // Server → Client reply: the service time. Slow compared to inter-request period.
    sim.add_edge(server, client, Expr::exp_dist(Expr::float(SERVICE_MEAN_NS)));
    // Client self-loop for ticking.
    sim.add_edge(client, client, Expr::int(REQUEST_PERIOD_NS));

    // ---------- Kick off ----------
    sim.inject(client, Value::variant("tick", Value::Nil));

    // ---------- Observe ----------
    println!("=== promise_aggregate ===\n");
    println!("Request period: {} ms | Service mean: {} ms",
        REQUEST_PERIOD_NS / 1_000_000, (SERVICE_MEAN_NS / 1_000_000.0) as i64);
    println!("Little's Law predicts in_flight ≈ λ·W = 200/s × 20ms = 4\n");
    println!("{:>8} | {:>10} | {:>10} | {:>10}",
        "t (ms)", "in_flight", "sent_at.n", "rtt_avg_ms");

    let mut t = 0u64;
    while t < SIM_DURATION_NS {
        t = (t + PROBE_INTERVAL_NS).min(SIM_DURATION_NS);
        sim.run_until(t);

        let c = &sim.nodes[&client];
        let in_flight = c.slots["in_flight"].as_int().unwrap();
        let sent_at_n = match &c.slots["sent_at"] {
            Value::Samples(s) => s.len(),
            _ => 0,
        };

        // Average RTT over recent completions, from the event log.
        let recent_rtts: Vec<f64> = sim.log.events.iter().rev()
            .take(200)
            .filter_map(|e| match e {
                Event::MetricRecorded { name, value, .. } if name == "rtt_ns" => {
                    value.as_float()
                }
                _ => None,
            })
            .collect();
        let avg_rtt_ms = if recent_rtts.is_empty() { 0.0 } else {
            recent_rtts.iter().sum::<f64>() / (recent_rtts.len() as f64) / 1_000_000.0
        };

        println!("{:>8.1} | {:>10} | {:>10} | {:>10.2}",
            (t as f64) / 1_000_000.0,
            in_flight,
            sent_at_n,
            avg_rtt_ms);
    }

    let peak = sim.nodes[&client].slots["peak"].as_int().unwrap();
    let total_rtts = sim.log.iter().filter(|e| matches!(
        e, Event::MetricRecorded { name, .. } if name == "rtt_ns"
    )).count();
    println!("\nPeak in_flight: {}", peak);
    println!("Total completed responses: {}", total_rtts);
}
