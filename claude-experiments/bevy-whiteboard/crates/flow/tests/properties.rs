//! Property-based tests for engine invariants.
//!
//! The properties are organized by category and documented with
//! the property-list ID used in project discussion. Each test uses
//! `proptest!` to generate random parameters and assert the
//! invariant holds across the space.
//!
//! Categories:
//!   C — causality (temporal ordering of events)
//!   R — return_path behaviour
//!   S — engine soundness (no hidden silent failures)

use std::collections::{BTreeMap, HashMap};

use proptest::prelude::*;

use flow::event::Event;
use flow::expr::Expr;
use flow::rule::{Effect, EmitTo, ReturnPathOp, Rule, When};
use flow::sim::{NodeId, Sim};
use flow::value::{Pattern, Value};

// ─────────────────────────────────────────────────────────────
// Test-only builders
// ─────────────────────────────────────────────────────────────

/// Build a minimal client/worker pair. Client fires `req(color)`
/// pushing self every `client_period` ns; worker's `serve` rule
/// replies `resp(nil)` popping return_path. Run the sim for
/// `duration` ns.
fn simple_client_worker(
    seed: u64,
    client_color: i64,
    client_period: u64,
    duration: u64,
) -> (Sim, NodeId, NodeId) {
    let mut sim = Sim::new(seed);
    let mut client_slots = BTreeMap::new();
    client_slots.insert("color".into(), Value::Int(client_color));
    client_slots.insert("period_ns".into(), Value::Int(client_period as i64));
    client_slots.insert("emitted".into(), Value::Int(0));
    client_slots.insert("in_flight".into(), Value::Int(0));
    client_slots.insert("completed".into(), Value::Int(0));

    let client_rules = vec![
        Rule::new("fire")
            .when(When::input(Pattern::variant("tick", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "emitted".into(),
                value: Expr::BinOp(
                    flow::expr::BinOp::Add,
                    Box::new(Expr::slot("emitted")),
                    Box::new(Expr::int(1)),
                ),
            })
            .do_(Effect::SetSlot {
                slot: "in_flight".into(),
                value: Expr::BinOp(
                    flow::expr::BinOp::Add,
                    Box::new(Expr::slot("in_flight")),
                    Box::new(Expr::int(1)),
                ),
            })
            .do_(Effect::Emit {
                payload: Expr::variant("req", Expr::slot("color")),
                to: EmitTo::DefaultOut,
                meta_ops: Vec::new(),
                return_path_op: ReturnPathOp::Push(Expr::self_ref()),
            })
            .do_(Effect::emit(
                Expr::variant("tick", Expr::lit(Value::Nil)),
                EmitTo::ToTargetExpr(Expr::self_ref()),
            )),
        Rule::new("on_resp")
            .when(When::input(Pattern::variant("resp", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "in_flight".into(),
                value: Expr::BinOp(
                    flow::expr::BinOp::Sub,
                    Box::new(Expr::slot("in_flight")),
                    Box::new(Expr::int(1)),
                ),
            })
            .do_(Effect::SetSlot {
                slot: "completed".into(),
                value: Expr::BinOp(
                    flow::expr::BinOp::Add,
                    Box::new(Expr::slot("completed")),
                    Box::new(Expr::int(1)),
                ),
            }),
    ];

    let client = sim.add_node("client", client_slots, client_rules);
    // Self-loop for the tick, with latency = period_ns slot.
    sim.add_edge(client, client, Expr::slot("period_ns"));

    let mut worker_slots = BTreeMap::new();
    worker_slots.insert("served".into(), Value::Int(0));
    let worker_rules = vec![
        Rule::new("serve")
            .when(When::input(Pattern::variant("req", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "served".into(),
                value: Expr::BinOp(
                    flow::expr::BinOp::Add,
                    Box::new(Expr::slot("served")),
                    Box::new(Expr::int(1)),
                ),
            })
            .do_(Effect::respond(Expr::variant("resp", Expr::lit(Value::Nil)))),
    ];
    let worker = sim.add_node("worker", worker_slots, worker_rules);
    sim.add_edge(client, worker, Expr::int(1_000_000));

    // Kick off with one tick.
    sim.inject(client, Value::variant("tick", Value::Nil));
    sim.run_until(duration);

    (sim, client, worker)
}

// ─────────────────────────────────────────────────────────────
// C1 — PacketEmitted.at_ns ≤ arrives_at_ns
// ─────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 50, ..ProptestConfig::default() })]

    /// Property C1: every emitted packet arrives no earlier than it
    /// was emitted. Negative or zero latencies shouldn't produce
    /// time-traveling packets.
    #[test]
    fn c1_packet_never_arrives_before_emit(
        seed in 0u64..1000,
        period in 50_000_000u64..500_000_000,
        duration in 100_000_000u64..5_000_000_000,
    ) {
        let (sim, _, _) = simple_client_worker(seed, 0, period, duration);
        for ev in sim.log.iter() {
            if let Event::PacketEmitted { at_ns, arrives_at_ns, .. } = ev {
                prop_assert!(
                    arrives_at_ns >= at_ns,
                    "packet arrives at {} before it was emitted at {}",
                    arrives_at_ns, at_ns,
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
// C2 — No resp without a prior req consume
// ─────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 50, ..ProptestConfig::default() })]

    /// Property C2: every resp packet emitted from the worker is
    /// preceded in event order by a req PacketConsumed on that
    /// same worker. Causality: no reply without a received request.
    #[test]
    fn c2_resp_causally_follows_req(
        seed in 0u64..1000,
        period in 50_000_000u64..500_000_000,
        duration in 200_000_000u64..3_000_000_000,
    ) {
        let (sim, _client, worker) = simple_client_worker(seed, 0, period, duration);

        let mut worker_req_consumes: u64 = 0;
        let mut worker_resp_emits:  u64 = 0;

        for ev in sim.log.iter() {
            match ev {
                Event::PacketConsumed { by, rule, .. } if *by == worker && rule == "serve" => {
                    worker_req_consumes += 1;
                    // At this point in the log, the count of resp
                    // emits must not exceed the count of consumes.
                    prop_assert!(
                        worker_resp_emits <= worker_req_consumes,
                        "resp emit got ahead of req consume: consumes={} emits={}",
                        worker_req_consumes, worker_resp_emits,
                    );
                }
                Event::PacketEmitted { from, payload, .. } if *from == worker => {
                    if let Value::Variant { tag, .. } = payload {
                        if tag == "resp" {
                            worker_resp_emits += 1;
                        }
                    }
                }
                _ => {}
            }
        }

        // After the full run, each req consume produced at most
        // one resp emit. (Could be exactly one if serve always
        // emits; could be fewer if pop fails, etc.)
        prop_assert!(
            worker_resp_emits <= worker_req_consumes,
            "total: consumes={} emits={}",
            worker_req_consumes, worker_resp_emits,
        );
    }
}

// ─────────────────────────────────────────────────────────────
// R3 — per-client, resp emits to C == C's completed (post-drain)
// ─────────────────────────────────────────────────────────────

/// Quiesce one or more clients by stretching `period_ns` to
/// effectively-infinity, then iteratively drain the sim until no
/// events are scheduled within the next 1 s. Using a single fixed
/// drain window is fragile: a proptest-picked period and duration
/// can leave a tick in-flight that, when delivered, triggers one
/// more req/resp round-trip whose tail falls past the window.
///
/// The loop runs `run_until(next_event + 1 ms)` repeatedly until
/// the next event is more than 1 s away (i.e., only the
/// huge-period self-ticks remain, everything immediate is done).
/// Bounded by a step cap so a misconfigured test can't spin
/// forever.
fn quiesce_and_drain(sim: &mut Sim, clients: &[NodeId]) {
    for c in clients {
        if let Some(n) = sim.nodes.get_mut(c) {
            n.slots.insert("period_ns".into(), Value::Int(i64::MAX / 4));
        }
    }
    for _ in 0..1000 {
        let Some(next) = sim.next_event_time_ns() else { break; };
        if next > sim.now_ns.saturating_add(1_000_000_000) { break; }
        sim.run_until(next.saturating_add(1_000_000));
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 30, ..ProptestConfig::default() })]

    /// Property R3 (single-client form): after quiescing the client
    /// and draining, the number of resp PacketEmitted events from
    /// the worker exactly equals the client's `completed` slot.
    /// No in-flight slop, no lost resp's.
    #[test]
    fn r3_single_client_resp_count_matches_completed(
        seed in 0u64..1000,
        period in 50_000_000u64..300_000_000,
        duration in 500_000_000u64..2_000_000_000,
    ) {
        let (mut sim, client, worker) = simple_client_worker(seed, 0, period, duration);
        quiesce_and_drain(&mut sim, &[client]);

        let mut resp_to_client: u64 = 0;
        for ev in sim.log.iter() {
            if let Event::PacketEmitted { from, to, payload, .. } = ev {
                if *from != worker || *to != client { continue; }
                if let Value::Variant { tag, .. } = payload {
                    if tag == "resp" { resp_to_client += 1; }
                }
            }
        }

        let completed = match sim.nodes[&client].slots["completed"] {
            Value::Int(i) => i as u64,
            _ => 0,
        };
        prop_assert_eq!(
            resp_to_client, completed,
            "after drain: worker emitted {} resp's to client, but client.completed = {}",
            resp_to_client, completed,
        );
    }
}

// ─────────────────────────────────────────────────────────────
// S — engine soundness: never panics, never silently drops
// a non-intentional case
// ─────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 30, ..ProptestConfig::default() })]

    /// Property S: running this topology over a random parameter
    /// space never panics, and if errors appear in `error_counts`
    /// they're identified by known kind strings (not typos or
    /// unexpected kinds).
    #[test]
    fn s_no_unexpected_error_kinds(
        seed in 0u64..1000,
        period in 50_000_000u64..500_000_000,
        duration in 100_000_000u64..2_000_000_000,
    ) {
        let (sim, _, _) = simple_client_worker(seed, 0, period, duration);
        let known: &[&str] = &[
            "emit_no_edge",
            "emit_unknown_target",
            "emit_target_bad_type",
            "emit_to_each_bad_targets",
            "emit_bad_port",
            "return_path_empty_pop",
            "return_path_push_bad_type",
            "return_path_replace_bad_type",
            "slot_missing",
            "slot_type_mismatch",
            "samples_empty_pop",
            "expr_type_mismatch",
            "edge_latency_bad_type",
            "edge_latency_negative",
            "spawn_failed",
            "color_mismatch",
            "node_down",
            "request_failed",
        ];
        for (kind, _count) in sim.error_counts.iter() {
            prop_assert!(
                known.contains(&kind.as_str()),
                "unexpected error kind `{}` — if this is intentional, add it to the known list",
                kind,
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Two-client cross-talk property (R3 multi-client form)
// ─────────────────────────────────────────────────────────────

fn two_clients_one_worker(
    seed: u64,
    period_a: u64,
    period_b: u64,
    duration: u64,
) -> Sim {
    let mut sim = Sim::new(seed);

    // Shared worker.
    let mut worker_slots = BTreeMap::new();
    worker_slots.insert("served".into(), Value::Int(0));
    let worker_rules = vec![
        Rule::new("serve")
            .when(When::input(Pattern::variant("req", Pattern::wild())))
            .do_(Effect::SetSlot {
                slot: "served".into(),
                value: Expr::BinOp(
                    flow::expr::BinOp::Add,
                    Box::new(Expr::slot("served")),
                    Box::new(Expr::int(1)),
                ),
            })
            .do_(Effect::respond(Expr::variant("resp", Expr::lit(Value::Nil)))),
    ];
    let worker = sim.add_node("worker", worker_slots, worker_rules);

    let make_client = |name: &str, period: u64, sim: &mut Sim| -> NodeId {
        let mut slots = BTreeMap::new();
        slots.insert("color".into(), Value::Int(0));
        slots.insert("period_ns".into(), Value::Int(period as i64));
        slots.insert("emitted".into(), Value::Int(0));
        slots.insert("in_flight".into(), Value::Int(0));
        slots.insert("completed".into(), Value::Int(0));
        let rules = vec![
            Rule::new("fire")
                .when(When::input(Pattern::variant("tick", Pattern::wild())))
                .do_(Effect::SetSlot {
                    slot: "emitted".into(),
                    value: Expr::BinOp(
                        flow::expr::BinOp::Add,
                        Box::new(Expr::slot("emitted")),
                        Box::new(Expr::int(1)),
                    ),
                })
                .do_(Effect::SetSlot {
                    slot: "in_flight".into(),
                    value: Expr::BinOp(
                        flow::expr::BinOp::Add,
                        Box::new(Expr::slot("in_flight")),
                        Box::new(Expr::int(1)),
                    ),
                })
                .do_(Effect::Emit {
                    payload: Expr::variant("req", Expr::slot("color")),
                    to: EmitTo::ToTargetExpr(Expr::lit(Value::Str("worker".into()))),
                    meta_ops: Vec::new(),
                    return_path_op: ReturnPathOp::Push(Expr::self_ref()),
                })
                .do_(Effect::emit(
                    Expr::variant("tick", Expr::lit(Value::Nil)),
                    EmitTo::ToTargetExpr(Expr::self_ref()),
                )),
            Rule::new("on_resp")
                .when(When::input(Pattern::variant("resp", Pattern::wild())))
                .do_(Effect::SetSlot {
                    slot: "in_flight".into(),
                    value: Expr::BinOp(
                        flow::expr::BinOp::Sub,
                        Box::new(Expr::slot("in_flight")),
                        Box::new(Expr::int(1)),
                    ),
                })
                .do_(Effect::SetSlot {
                    slot: "completed".into(),
                    value: Expr::BinOp(
                        flow::expr::BinOp::Add,
                        Box::new(Expr::slot("completed")),
                        Box::new(Expr::int(1)),
                    ),
                }),
        ];
        let id = sim.add_node(name, slots, rules);
        sim.add_edge(id, id, Expr::slot("period_ns"));
        sim.add_edge(id, worker, Expr::int(1_000_000));
        sim.inject(id, Value::variant("tick", Value::Nil));
        id
    };
    let client_a = make_client("client_a", period_a, &mut sim);
    let client_b = make_client("client_b", period_b, &mut sim);

    sim.run_until(duration);
    quiesce_and_drain(&mut sim, &[client_a, client_b]);
    sim
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 20, ..ProptestConfig::default() })]

    /// Property R3 (multi-client): per-client resp count from
    /// worker equals that client's `completed`. No reply ever
    /// lands on the wrong client.
    #[test]
    fn r3_two_clients_no_cross_talk(
        seed in 0u64..1000,
        period_a in 100_000_000u64..500_000_000,
        period_b in 100_000_000u64..500_000_000,
        duration in 500_000_000u64..2_000_000_000,
    ) {
        let sim = two_clients_one_worker(seed, period_a, period_b, duration);

        let client_a = sim.node_by_name("client_a").unwrap();
        let client_b = sim.node_by_name("client_b").unwrap();
        let worker   = sim.node_by_name("worker").unwrap();

        let mut resp_to: HashMap<NodeId, u64> = HashMap::new();
        for ev in sim.log.iter() {
            if let Event::PacketEmitted { from, to, payload, .. } = ev {
                if *from != worker { continue; }
                if let Value::Variant { tag, .. } = payload {
                    if tag == "resp" {
                        *resp_to.entry(*to).or_insert(0) += 1;
                    }
                }
            }
        }
        for c in [client_a, client_b] {
            let completed = match sim.nodes[&c].slots["completed"] {
                Value::Int(i) => i as u64,
                _ => 0,
            };
            let emits = resp_to.get(&c).copied().unwrap_or(0);
            prop_assert_eq!(
                emits, completed,
                "client {:?}: worker emitted {} resp's, client.completed = {}",
                c, emits, completed,
            );
        }
    }
}
