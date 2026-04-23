//! Pins down the packet `metadata` + `return_path` mechanism:
//! default-inheritance through intermediaries, explicit push/pop by
//! rule authors, and the no-panic contract on malformed ops (empty
//! pop, bad-type push — both become RuntimeError events, not crashes).

use std::collections::BTreeMap;

use flow::event::Event;
use flow::expr::Expr;
use flow::rule::{Effect, EmitTo, MetaOp, ReturnPathOp, Rule, When};
use flow::sim::Sim;
use flow::value::{Pattern, Value};

fn emit_with(payload: Expr, to: EmitTo, rp: ReturnPathOp, meta: Vec<MetaOp>) -> Effect {
    Effect::Emit { payload, to, meta_ops: meta, return_path_op: rp }
}

/// Client → Forwarder → Worker. Forwarder does nothing to the packet;
/// Worker pops and replies to head(return_path). Without an edge from
/// Forwarder to Worker that's wired for the response, this would fail
/// — so we wire Worker → Client directly. Verifies:
///   - metadata inherited through Forwarder
///   - return_path inherited through Forwarder
///   - Worker's pop lands response at the right node
#[test]
fn metadata_and_return_path_inherit_through_forwarder() {
    let mut sim = Sim::new(1);

    // Nodes: client, forwarder (transparent), worker, plus edges.
    let client = sim.add_node("client", BTreeMap::new(), Vec::new());
    let forwarder = sim.add_node(
        "forwarder",
        BTreeMap::new(),
        vec![Rule::new("forward")
            .when(When::input(Pattern::variant("req", Pattern::wild())))
            // Default Emit: meta_ops = [], return_path_op = Inherit
            .do_(Effect::emit(
                Expr::variant("req", Expr::lit(Value::Nil)),
                EmitTo::DefaultOut,
            ))],
    );
    // Worker records the metadata value and the length of return_path
    // on receipt, so the test can assert they flowed through.
    let worker = sim.add_node(
        "worker",
        BTreeMap::from([
            ("seen_corr".to_string(), Value::Nil),
            ("rp_len_on_arrival".to_string(), Value::Int(-1)),
        ]),
        vec![Rule::new("serve")
            .when(When::input(Pattern::variant("req", Pattern::wild())))
            .do_(Effect::SetSlot { slot: "seen_corr".into(), value: Expr::meta("corr") })
            .do_(Effect::SetSlot {
                slot: "rp_len_on_arrival".into(),
                value: Expr::length(Expr::return_path()),
            })
            // Reply pops head(return_path) back to client.
            .do_(Effect::respond(Expr::variant("resp", Expr::lit(Value::Nil))))],
    );
    let client_sink = sim.add_node(
        "client_sink",
        BTreeMap::from([("got_response".to_string(), Value::Bool(false))]),
        vec![Rule::new("on_resp")
            .when(When::input(Pattern::variant("resp", Pattern::wild())))
            .do_(Effect::SetSlot { slot: "got_response".into(), value: Expr::bool(true) })],
    );
    // Edges: client→forwarder, forwarder→worker, worker→client_sink
    sim.add_edge(client, forwarder, Expr::int(1));
    sim.add_edge(forwarder, worker, Expr::int(1));
    // Response edge. head(return_path) after client pushes itself is
    // `client_sink` (the test injects with return_path=[client_sink]
    // directly — simpler than modeling a full Client template here).
    sim.add_edge(worker, client_sink, Expr::int(1));

    // Inject a req with metadata {corr: 42} and return_path=[client_sink].
    let mut meta = BTreeMap::new();
    meta.insert("corr".to_string(), Value::Int(42));
    sim.inject_with(
        forwarder,
        Value::variant("req", Value::Nil),
        meta,
        vec![client_sink],
    );

    sim.run_until(1_000_000);

    // Worker saw metadata passed through the forwarder.
    assert_eq!(sim.nodes[&worker].slots["seen_corr"], Value::Int(42));
    // Forwarder didn't push/pop — worker saw return_path of length 1.
    assert_eq!(sim.nodes[&worker].slots["rp_len_on_arrival"], Value::Int(1));
    // Response popped back to client_sink.
    assert_eq!(sim.nodes[&client_sink].slots["got_response"], Value::Bool(true));

    // Silent path: client node never had any role — suppress unused var.
    let _ = client;
}

/// Popping an empty return_path must record an error and drop the
/// emit — NOT panic. The sim keeps running; counter increments.
#[test]
fn empty_pop_records_error_no_panic() {
    let mut sim = Sim::new(1);

    let responder = sim.add_node(
        "responder",
        BTreeMap::new(),
        vec![Rule::new("try_respond")
            .when(When::input(Pattern::variant("ping", Pattern::wild())))
            .do_(emit_with(
                Expr::variant("pong", Expr::lit(Value::Nil)),
                EmitTo::ToTargetExpr(Expr::head(Expr::return_path())),
                ReturnPathOp::Pop,
                Vec::new(),
            ))],
    );

    // Inject without a return_path — the rule's Pop attempt must
    // trigger `return_path_empty_pop`.
    sim.inject(responder, Value::variant("ping", Value::Nil));
    sim.run_until(1_000_000);

    assert_eq!(
        sim.error_counts.get("return_path_empty_pop").copied(),
        Some(1),
        "expected exactly one return_path_empty_pop error"
    );
    let had_error_event = sim.log.iter().any(|e| matches!(
        e,
        Event::RuntimeError { kind, .. } if kind == "return_path_empty_pop"
    ));
    assert!(had_error_event, "RuntimeError event must be emitted");
}

/// `pushing` a non-NodeRef value records a `return_path_push_bad_type`
/// error and drops the emit. The sim doesn't panic.
#[test]
fn push_non_noderef_records_error() {
    let mut sim = Sim::new(1);

    let a = sim.add_node(
        "a",
        BTreeMap::new(),
        vec![Rule::new("misuse_push")
            .when(When::input(Pattern::variant("go", Pattern::wild())))
            .do_(emit_with(
                Expr::variant("next", Expr::lit(Value::Nil)),
                EmitTo::DefaultOut,
                // Int(999) is not a NodeRef — must record an error.
                ReturnPathOp::Push(Expr::int(999)),
                Vec::new(),
            ))],
    );
    let b = sim.add_node("b", BTreeMap::new(), Vec::new());
    sim.add_edge(a, b, Expr::int(1));

    sim.inject(a, Value::variant("go", Value::Nil));
    sim.run_until(1_000_000);

    assert_eq!(
        sim.error_counts.get("return_path_push_bad_type").copied(),
        Some(1)
    );
    // b never received the packet — the emit was dropped.
    assert!(sim.nodes[&b].inbox.is_empty());
}

/// Metadata mutation via `meta { k: v }`: SET / remove inheritance
/// and explicit key removal.
#[test]
fn metadata_ops_set_and_remove() {
    let mut sim = Sim::new(1);

    // Rule-side: receive `req(_)`, set `trace: 7`, forget `auth`.
    // The packet we inject carries {trace: 1, auth: "token"}.
    let proxy = sim.add_node(
        "proxy",
        BTreeMap::new(),
        vec![Rule::new("fwd")
            .when(When::input(Pattern::variant("req", Pattern::wild())))
            .do_(emit_with(
                Expr::variant("req", Expr::lit(Value::Nil)),
                EmitTo::DefaultOut,
                ReturnPathOp::Inherit,
                vec![
                    MetaOp::Set { key: "trace".into(), value: Expr::int(7) },
                    MetaOp::Remove { key: "auth".into() },
                ],
            ))],
    );
    let observer = sim.add_node(
        "observer",
        BTreeMap::from([
            ("trace".to_string(), Value::Nil),
            ("auth".to_string(), Value::Nil),
        ]),
        vec![Rule::new("observe")
            .when(When::input(Pattern::variant("req", Pattern::wild())))
            .do_(Effect::SetSlot { slot: "trace".into(), value: Expr::meta("trace") })
            .do_(Effect::SetSlot { slot: "auth".into(), value: Expr::meta("auth") })],
    );
    sim.add_edge(proxy, observer, Expr::int(1));

    let mut meta = BTreeMap::new();
    meta.insert("trace".to_string(), Value::Int(1));
    meta.insert("auth".to_string(), Value::Str("token".into()));
    sim.inject_with(
        proxy,
        Value::variant("req", Value::Nil),
        meta,
        Vec::new(),
    );
    sim.run_until(1_000_000);

    // Observer saw overridden trace and no auth.
    assert_eq!(sim.nodes[&observer].slots["trace"], Value::Int(7));
    assert_eq!(sim.nodes[&observer].slots["auth"], Value::Nil);
}
