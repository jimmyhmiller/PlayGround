//! Auto-routing test: handler A emits an effect whose `response_event`
//! triggers handler B. After a single `StartPing` event we expect both
//! `pings_sent` and `pongs_received` to be 1.

use serde_json::{json, Value as Json};

use ir::manifest::Manifest;
use runtime::{InboundEvent, MockAdapter, Runtime};

const MANIFEST_JSON: &str = r#"
{
  "name": "ping-pong",
  "version": "0.1.0",
  "schemas": {
    "Empty":       { "kind": "record", "fields": {} },
    "PongPayload": {
      "kind": "record",
      "fields": {
        "emit_id": "u64",
        "outcome": {
          "kind": "record",
          "fields": { "tag": "string" }
        }
      }
    }
  },
  "events": {
    "StartPing":    { "payload": "Empty" },
    "PongReceived": { "payload": "PongPayload" }
  },
  "state": {
    "pings_sent":     { "kind": "atom", "schema": "u64" },
    "pongs_received": { "kind": "atom", "schema": "u64" }
  },
  "effects": {
    "Ping": {
      "request":  "Empty",
      "response": "Empty",
      "response_event": "PongReceived"
    }
  },
  "handlers": [
    {
      "name": "on_start",
      "on":   "StartPing",
      "read":  ["pings_sent"],
      "write": ["pings_sent"],
      "emit":  ["Ping"],
      "body":  { "hash": "sha256:0", "uri": "on_start" }
    },
    {
      "name": "on_pong",
      "on":   "PongReceived",
      "read":  ["pongs_received"],
      "write": ["pongs_received"],
      "emit":  [],
      "body":  { "hash": "sha256:0", "uri": "on_pong" }
    }
  ]
}
"#;

fn load() -> Manifest {
    serde_json::from_str(MANIFEST_JSON).expect("parse")
}

fn n(v: &Json) -> u64 {
    v.as_u64().unwrap_or(0)
}

#[test]
fn auto_routed_response_drives_second_handler() {
    let mut rt = Runtime::new(load()).expect("validates");

    rt.bodies.register("on_start", |ctx| {
        let current = ctx.read_atom("pings_sent")?;
        let next = n(&current) + 1;
        ctx.set_atom("pings_sent", json!(next))?;
        ctx.emit("Ping", json!({}))?;
        Ok(())
    });
    rt.bodies.register("on_pong", |ctx| {
        let current = ctx.read_atom("pongs_received")?;
        let next = n(&current) + 1;
        ctx.set_atom("pongs_received", json!(next))?;
        Ok(())
    });

    let ping_adapter = MockAdapter::with_response(json!({}));
    rt.adapters
        .register("Ping", Box::new(ping_adapter.clone()));

    rt.enqueue(InboundEvent::new("StartPing", json!({})))
        .expect("enqueue");
    rt.run_to_quiescence().expect("run");

    assert_eq!(
        rt.state.get_atom("pings_sent").cloned().unwrap_or(json!(0)),
        json!(1),
        "on_start should have incremented pings_sent"
    );
    assert_eq!(
        rt.state
            .get_atom("pongs_received")
            .cloned()
            .unwrap_or(json!(0)),
        json!(1),
        "on_pong should have run as a result of the auto-routed response"
    );

    // The Ping adapter was called exactly once.
    assert_eq!(ping_adapter.calls().len(), 1);
}

#[test]
fn multiple_pings_in_one_session() {
    let mut rt = Runtime::new(load()).expect("validates");

    rt.bodies.register("on_start", |ctx| {
        let current = ctx.read_atom("pings_sent")?;
        ctx.set_atom("pings_sent", json!(n(&current) + 1))?;
        ctx.emit("Ping", json!({}))?;
        Ok(())
    });
    rt.bodies.register("on_pong", |ctx| {
        let current = ctx.read_atom("pongs_received")?;
        ctx.set_atom("pongs_received", json!(n(&current) + 1))?;
        Ok(())
    });

    let ping_adapter = MockAdapter::with_response(json!({}));
    rt.adapters
        .register("Ping", Box::new(ping_adapter.clone()));

    for _ in 0..5 {
        rt.enqueue(InboundEvent::new("StartPing", json!({})))
            .unwrap();
    }
    rt.run_to_quiescence().unwrap();

    assert_eq!(rt.state.get_atom("pings_sent").cloned(), Some(json!(5)));
    assert_eq!(
        rt.state.get_atom("pongs_received").cloned(),
        Some(json!(5))
    );
    assert_eq!(ping_adapter.calls().len(), 5);
}

#[test]
fn response_payload_carries_emit_id_and_outcome() {
    let mut rt = Runtime::new(load()).expect("validates");

    rt.bodies.register("on_start", |ctx| {
        ctx.set_atom("pings_sent", json!(1))?;
        ctx.emit("Ping", json!({}))?;
        Ok(())
    });

    // on_pong asserts on the synthesized payload shape directly.
    rt.bodies.register("on_pong", |ctx| {
        let payload = ctx.event;
        assert!(
            payload.get("emit_id").and_then(|v| v.as_u64()).is_some(),
            "PongReceived payload should carry emit_id, got: {payload}"
        );
        let outcome = payload.get("outcome").expect("payload has outcome");
        let tag = outcome.get("tag").and_then(|v| v.as_str());
        assert_eq!(tag, Some("ok"), "outcome should be ok");

        ctx.set_atom("pongs_received", json!(1))?;
        Ok(())
    });

    rt.adapters
        .register("Ping", Box::new(MockAdapter::with_response(json!({}))));

    rt.enqueue(InboundEvent::new("StartPing", json!({})))
        .unwrap();
    rt.run_to_quiescence().unwrap();
}
