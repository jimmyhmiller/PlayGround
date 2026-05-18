//! End-to-end test: a real WASM component used as a handler body, loaded
//! via the generic `wasm::load_handler_body` (no hardcoded glue for this
//! component's specific world).

use std::path::PathBuf;

use serde_json::json;

use ir::manifest::Manifest;
use runtime::wasm::load_handler_body;
use runtime::{InboundEvent, MockAdapter, Runtime};

// The bumper component's WIT (see wasm-samples/bumper/wit/world.wit) imports
// `emit-notify: func(text: string) -> u64`, so the manifest declares Notify's
// request as a bare `string` — the generic loader generates the matching
// `emit-notify: func(req: string) -> u64` import from this.
const MANIFEST_JSON: &str = r#"
{
  "name": "bumper-test",
  "version": "0.1.0",
  "events": {
    "Bump": { "payload": "u32" }
  },
  "state": {
    "counter": { "kind": "atom", "schema": "u32" }
  },
  "effects": {
    "Notify": { "request": "string", "response": "string" }
  },
  "handlers": [
    {
      "name": "bump_counter",
      "on":   "Bump",
      "read":  ["counter"],
      "write": ["counter"],
      "emit":  ["Notify"],
      "body":  { "hash": "sha256:0", "uri": "bumper" }
    }
  ]
}
"#;

fn component_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/wasm/bumper.component.wasm")
}

fn make_runtime() -> (Runtime, MockAdapter) {
    let manifest: Manifest = serde_json::from_str(MANIFEST_JSON).expect("parse manifest");
    let mut rt = Runtime::new(manifest.clone()).expect("validates");

    let mut body = load_handler_body(&manifest, "bump_counter", component_path())
        .expect("load bumper component");
    rt.bodies.register("bumper", move |ctx| body(ctx));

    let notify = MockAdapter::with_response(json!(""));
    rt.adapters.register("Notify", Box::new(notify.clone()));

    // Seed counter to 0.
    rt.state.set_atom("counter", json!(0u32));

    (rt, notify)
}

#[test]
fn wasm_body_reads_writes_emits_through_runtime() {
    let (mut rt, notify) = make_runtime();

    rt.enqueue(InboundEvent::new("Bump", json!(5))).unwrap();
    rt.enqueue(InboundEvent::new("Bump", json!(3))).unwrap();
    rt.run_to_quiescence().expect("run");

    // State: 0 + 5 + 3 = 8.
    assert_eq!(rt.state.get_atom("counter").cloned(), Some(json!(8)));

    // Notify was called twice with the expected text. Notify's request is a
    // bare `string` in this manifest (matches the bumper component's WIT).
    let calls = notify.calls();
    assert_eq!(calls.len(), 2, "Notify called once per Bump");
    assert_eq!(calls[0], json!("counter is now 5"));
    assert_eq!(calls[1], json!("counter is now 8"));
}

#[test]
fn wasm_body_event_log_shows_writes_and_emits() {
    let (mut rt, _notify) = make_runtime();

    rt.enqueue(InboundEvent::new("Bump", json!(7))).unwrap();
    rt.run_to_quiescence().unwrap();

    let entries = rt.log.entries();
    // Expected: 1 EventEnqueued + 1 HandlerInvoked + 1 EffectFulfilled
    assert_eq!(entries.len(), 3, "got: {:#?}", entries);

    use runtime::LogEntryKind;
    let invocation = entries
        .iter()
        .find_map(|e| match &e.kind {
            LogEntryKind::HandlerInvoked {
                handler,
                writes,
                emits,
                ..
            } => Some((handler.clone(), writes.len(), emits.len())),
            _ => None,
        })
        .expect("found HandlerInvoked");
    assert_eq!(invocation, ("bump_counter".to_string(), 1, 1));
}

#[test]
fn wasm_body_cannot_violate_footprint() {
    // The bumper body only writes `counter`. The runtime's footprint check
    // applies at the BodyCtx layer — so the body physically cannot call
    // any other write op. (The component's WIT world doesn't even import
    // anything else.) This test verifies state-only writes happen, by
    // checking no other cells appear.
    //
    // In v0.1 the manifest has only one cell, but the property is:
    // BumperLoader's body only ever calls set_atom("counter", ...) on its
    // BodyCtx, never any other cell.
    let (mut rt, _) = make_runtime();
    rt.enqueue(InboundEvent::new("Bump", json!(1))).unwrap();
    rt.run_to_quiescence().unwrap();

    // Sanity: counter advanced; no other state touched.
    assert_eq!(rt.state.get_atom("counter").cloned(), Some(json!(1)));
    assert_eq!(rt.state.atoms().len(), 1);
    assert_eq!(rt.state.maps().len(), 0);
}
