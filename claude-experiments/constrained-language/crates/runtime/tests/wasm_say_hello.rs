//! Hello-world for the generic WASM loader.
//!
//! Exercises:
//!   * record event payload (`Greeting { name: string }`)
//!   * Map<string, u32> state, keyed by `$event.name`
//!   * record effect request (`PrintReq { text: string }`)
//!   * a real adapter (`StdoutPrinter`) that actually prints to stdout.
//!
//! This is the canonical demonstration that handlers compiled to a WASM
//! component can be loaded *as data* — the runtime knows nothing about the
//! say_hello world specifically; it derives the host imports from the
//! handler's declared footprint and links them dynamically.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::json;

use ir::load_manifest_file;
use runtime::effect::{Adapter, AdapterResult};
use runtime::wasm::load_handler_body;
use runtime::{InboundEvent, Runtime, Value};

fn sample_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("wasm-samples/say_hello")
}

fn manifest_path() -> PathBuf {
    sample_dir().join("manifest.toml")
}

fn component_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/wasm/say_hello.component.wasm")
}

/// Real adapter: prints to stdout and records every line for assertions.
#[derive(Default, Clone)]
struct StdoutPrinter {
    inner: Arc<Mutex<Vec<String>>>,
}

impl StdoutPrinter {
    fn lines(&self) -> Vec<String> {
        self.inner.lock().unwrap().clone()
    }
}

impl Adapter for StdoutPrinter {
    fn fulfill(&mut self, request: Value, _emit_id: u64) -> AdapterResult {
        let text = request
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        println!("{text}");
        self.inner.lock().unwrap().push(text.clone());
        AdapterResult::Ok(Value::String("ok".into()))
    }
}

fn make_runtime() -> (Runtime, StdoutPrinter) {
    let manifest = load_manifest_file(manifest_path()).expect("load manifest.toml");
    let mut rt = Runtime::new(manifest.clone()).expect("validates");

    let mut body = load_handler_body(&manifest, "say_hello", component_path())
        .expect("load say_hello component");
    rt.bodies.register("say_hello", move |ctx| body(ctx));

    let printer = StdoutPrinter::default();
    rt.adapters.register("Print", Box::new(printer.clone()));
    (rt, printer)
}

#[test]
fn first_greeting_says_one_time() {
    let (mut rt, printer) = make_runtime();
    rt.enqueue(InboundEvent::new("Greeted", json!({ "name": "Alice" })))
        .unwrap();
    rt.run_to_quiescence().unwrap();

    assert_eq!(printer.lines(), vec!["Hello, Alice! (greeted 1 time)"]);
    // State now has Alice → 1.
    let entries = rt.state.list_map("greet_counts");
    assert_eq!(entries, vec![(json!("Alice"), json!(1u32))]);
}

#[test]
fn repeat_greetings_increment_per_name() {
    let (mut rt, printer) = make_runtime();
    for name in ["Alice", "Bob", "Alice", "Alice", "Bob"] {
        rt.enqueue(InboundEvent::new("Greeted", json!({ "name": name })))
            .unwrap();
    }
    rt.run_to_quiescence().unwrap();

    assert_eq!(
        printer.lines(),
        vec![
            "Hello, Alice! (greeted 1 time)",
            "Hello, Bob! (greeted 1 time)",
            "Hello, Alice! (greeted 2 times)",
            "Hello, Alice! (greeted 3 times)",
            "Hello, Bob! (greeted 2 times)",
        ]
    );
}

#[test]
fn log_records_writes_and_emits_with_correct_shapes() {
    let (mut rt, _) = make_runtime();
    rt.enqueue(InboundEvent::new("Greeted", json!({ "name": "Alice" })))
        .unwrap();
    rt.run_to_quiescence().unwrap();

    use runtime::log::{LogEntryKind, WriteRecord};
    let writes_emits = rt
        .log
        .entries()
        .iter()
        .find_map(|e| match &e.kind {
            LogEntryKind::HandlerInvoked { writes, emits, .. } => {
                Some((writes.clone(), emits.clone()))
            }
            _ => None,
        })
        .expect("HandlerInvoked entry");
    let (writes, emits) = writes_emits;

    assert_eq!(writes.len(), 1);
    match &writes[0] {
        WriteRecord::PutMap { cell, key, value } => {
            assert_eq!(cell, "greet_counts");
            assert_eq!(key, &json!("Alice"));
            assert_eq!(value, &json!(1u32));
        }
        other => panic!("expected PutMap, got {other:?}"),
    }

    assert_eq!(emits.len(), 1);
    assert_eq!(emits[0].effect, "Print");
    assert_eq!(
        emits[0].request,
        json!({ "text": "Hello, Alice! (greeted 1 time)" })
    );
}

#[test]
fn body_cannot_touch_undeclared_state() {
    // The handler only declares `greet_counts[$event.name]`. Even though the
    // body was authored to only touch that, the runtime's footprint check
    // would catch a violation. Sanity: confirm we don't accidentally let
    // unrelated cells leak into state.
    let (mut rt, _) = make_runtime();
    rt.enqueue(InboundEvent::new("Greeted", json!({ "name": "Alice" })))
        .unwrap();
    rt.run_to_quiescence().unwrap();
    assert_eq!(rt.state.atoms().len(), 0);
    assert_eq!(rt.state.maps().len(), 1);
}
