//! Retry with exponential backoff.
//!
//! A Client fires a request. A Server always responds with `err`
//! (for demonstrative purposes — determinism over realism). The
//! Client's Retry node catches the err, waits `100ms * 2^attempts`,
//! and retries, up to `max_attempts`. Then it gives up.
//!
//! What to watch in the output:
//!   - RuleFired events for "retry" at t = 100, 300, 700 ms (cumulative)
//!   - Final "giveup" after `max_attempts` failures.
//!
//! Run: `cargo run -p flow --example retry_backoff`

use std::collections::BTreeMap;

use flow::{
    BinOp, Effect, EmitTo, Event, Expr, Pattern, Rule, Sim, Value, When,
};

fn main() {
    let mut sim = Sim::new(0);

    // --------- Server: always replies with err ---------
    let server_rules = vec![
        Rule::new("reply_err")
            .when(When::input(Pattern::wild()))
            .do_(Effect::Respond {
                payload: Expr::variant("err", Expr::lit(Value::Nil)),
            }),
    ];
    let server = sim.add_node("Server", BTreeMap::new(), server_rules);

    // --------- Client: retry with exponential backoff ---------
    let mut client_slots = BTreeMap::new();
    client_slots.insert("attempts".to_string(), Value::Int(0));
    client_slots.insert("max_attempts".to_string(), Value::Int(4));
    client_slots.insert("base_ns".to_string(), Value::Int(100_000_000)); // 100ms
    client_slots.insert("alive".to_string(), Value::Bool(true));

    let client_rules = vec![
        // On the initial kick ("go" packet), fire the first request.
        Rule::new("initial_fire")
            .when(When::input(Pattern::variant("go", Pattern::wild())))
            .do_(Effect::Emit {
                payload: Expr::variant("req", Expr::lit(Value::Nil)),
                to: EmitTo::ToTarget("Server".to_string()),
            }),

        // Server replied with err, and we're under the retry budget.
        Rule::new("retry")
            .when(When::input(Pattern::variant("err", Pattern::wild())))
            .when(When::SlotMatch {
                slot: "alive".to_string(),
                pattern: Pattern::lit(Value::Bool(true)),
            })
            .guard(Expr::lt(Expr::slot("attempts"), Expr::slot("max_attempts")))
            .do_(Effect::SetSlot {
                slot: "attempts".to_string(),
                value: Expr::add(Expr::slot("attempts"), Expr::int(1)),
            })
            .do_(Effect::RecordMetric {
                name: "retry".to_string(),
                value: Expr::slot("attempts"),
            })
            // Emit an outbound request. Retry delay is modeled by the Client→Server
            // edge's latency expression (see below), which reads `attempts`.
            .do_(Effect::Emit {
                payload: Expr::variant("req", Expr::lit(Value::Nil)),
                to: EmitTo::ToTarget("Server".to_string()),
            }),

        // Budget exhausted: give up.
        Rule::new("giveup")
            .when(When::input(Pattern::variant("err", Pattern::wild())))
            .when(When::SlotMatch {
                slot: "alive".to_string(),
                pattern: Pattern::lit(Value::Bool(true)),
            })
            .guard(Expr::BinOp(
                BinOp::Ge,
                Box::new(Expr::slot("attempts")),
                Box::new(Expr::slot("max_attempts")),
            ))
            .do_(Effect::SetSlot {
                slot: "alive".to_string(),
                value: Expr::bool(false),
            })
            .do_(Effect::RecordMetric {
                name: "gave_up".to_string(),
                value: Expr::slot("attempts"),
            }),
    ];
    let client = sim.add_node("Client", client_slots, client_rules);

    // --------- Edges ---------
    //
    // Client → Server latency: constant 1ms (the network).
    // Server → Client reply latency: 1ms + backoff = base * 2^(attempts-1) when attempts>0.
    //
    // We put the backoff on the *return* path: after an err reply, the
    // retry rule fires and emits a new request — but we want the retry
    // to appear delayed. Simplest: the backoff is on the *outbound*
    // edge, read on Emit. The client's `attempts` slot was just
    // incremented to 1, 2, 3, ... before the Emit fires.
    let forward_latency = {
        // base * 2^(attempts-1), clamped to ≥1 so attempts=0 (initial send) is fast.
        let attempts = Expr::slot("attempts");
        let base = Expr::slot("base_ns");
        // pow_arg = attempts - 1, but minimum 0 (so initial send has no backoff).
        //   if attempts == 0 then base_ns * 0 + 1ms  (the 1ms is the network)
        //   else 1ms + base_ns * 2^(attempts-1)
        let network = Expr::int(1_000_000); // 1ms
        let is_initial = Expr::eq(attempts.clone(), Expr::int(0));
        let backoff = Expr::mul(
            base.clone(),
            Expr::pow(Expr::int(2), Expr::sub(attempts.clone(), Expr::int(1))),
        );
        Expr::if_(is_initial, network.clone(), Expr::add(network, backoff))
    };
    sim.add_edge(client, server, forward_latency);
    sim.add_edge(server, client, Expr::int(1_000_000)); // 1ms return

    // --------- Kick off ---------
    sim.inject(client, Value::variant("go", Value::Nil), None);

    // --------- Run ---------
    // Deadline: enough to cover the final backoff (100 + 200 + 400 + 800 ≈ 1.5s of backoff)
    sim.run_until(5_000_000_000); // 5 seconds

    // --------- Report ---------
    println!("=== retry_backoff trace ===\n");
    for ev in sim.log.iter() {
        match ev {
            Event::RuleFired { node, rule, at_ns } => {
                let name = &sim.nodes[node].name;
                println!("t={:>8.3} ms   {:<12} fires `{}`",
                    (*at_ns as f64) / 1_000_000.0, name, rule);
            }
            Event::MetricRecorded { node, name, value, at_ns } => {
                let nname = &sim.nodes[node].name;
                println!("t={:>8.3} ms   {:<12} METRIC {} = {:?}",
                    (*at_ns as f64) / 1_000_000.0, nname, name, value);
            }
            Event::PacketEmitted { from, to, at_ns, arrives_at_ns, .. } => {
                let f = &sim.nodes[from].name;
                let t = &sim.nodes[to].name;
                println!("t={:>8.3} ms   {:<12} emit → {} (arrives t={:.3} ms)",
                    (*at_ns as f64) / 1_000_000.0, f, t,
                    (*arrives_at_ns as f64) / 1_000_000.0);
            }
            _ => {}
        }
    }

    // Sanity check: expect the final "gave_up" metric at attempts == max_attempts.
    let gave_up = sim.log.iter().any(|e| matches!(
        e, Event::MetricRecorded { name, .. } if name == "gave_up"
    ));
    println!("\ngave_up: {}", gave_up);
    assert!(gave_up, "expected client to give up after max_attempts");
}
