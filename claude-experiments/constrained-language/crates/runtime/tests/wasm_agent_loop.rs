//! End-to-end agent-loop test, with all three handler bodies as real WASM
//! components loaded through the generic loader.
//!
//! This is the WASM port of `agent_loop.rs` (which uses Rust closures). It
//! exercises every non-trivial schema shape the generic loader supports:
//!
//!   - record event payloads (`GoalReceivedPayload`, `*ReturnedPayload`)
//!   - sum/variant outcomes (`LlmOutcome`, `ToolOutcome`)
//!   - nested records (`GoalRecord` containing `list<Message>`)
//!   - `Map<string, GoalRecord>` and `Map<u64, string>` state
//!   - wildcard reads/writes (`goals[*]`, `pending_*[*]`)
//!   - event-bound key reads (`pending_llm[$event.emit_id]`)
//!   - auto-routed `response_event` synthesizing `{tag, value}` outcome
//!
//! The bodies are byte-identical to what would run in production; only the
//! adapters are scripted/mocked for deterministic testing.

use std::path::PathBuf;

use serde_json::{json, Value as Json};

use ir::load_manifest_file;
use runtime::wasm::load_handler_body;
use runtime::{InboundEvent, MockAdapter, Runtime, ScriptedAdapter};

fn sample_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("wasm-samples/agent_loop")
}

fn component_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/wasm")
        .join(format!("{name}.component.wasm"))
}

fn make_runtime() -> (Runtime, ScriptedAdapter, ScriptedAdapter, MockAdapter) {
    let manifest = load_manifest_file(sample_dir().join("manifest.toml"))
        .expect("load agent loop manifest");
    let mut rt = Runtime::new(manifest.clone()).expect("validates");

    for handler in [
        "on_user_message",
        "on_llm_returned",
        "on_tool_returned",
    ] {
        let mut body = load_handler_body(&manifest, handler, component_path(handler))
            .unwrap_or_else(|e| panic!("load {handler}: {e}"));
        rt.bodies.register(handler, move |ctx| body(ctx));
    }

    let llm = ScriptedAdapter::new();
    let tool = ScriptedAdapter::new();
    let notify = MockAdapter::with_response(json!(""));

    rt.adapters.register("CallLlm", Box::new(llm.clone()));
    rt.adapters.register("ExecuteTool", Box::new(tool.clone()));
    rt.adapters.register("NotifyUser", Box::new(notify.clone()));

    (rt, llm, tool, notify)
}

#[test]
fn one_goal_one_tool_call_then_final_answer() {
    let (mut rt, llm, tool, notify) = make_runtime();

    llm.push_ok(json!({
        "kind": "needs_tool",
        "text": "Let me search for that.",
        "tool": "search",
        "args": "what is the meaning of life"
    }));
    llm.push_ok(json!({
        "kind": "final",
        "text": "The answer is 42.",
        "tool": "",
        "args": ""
    }));
    tool.push_ok(json!({ "result": "42" }));

    rt.enqueue(InboundEvent::new(
        "UserMessage",
        json!({ "session_id": "g1", "text": "what is the meaning of life" }),
    ))
    .unwrap();
    rt.run_to_quiescence().unwrap();

    let entries = rt.state.list_map("goals");
    assert_eq!(entries.len(), 1, "exactly one goal in state");
    let goal = &entries[0].1;
    assert_eq!(goal["status"], "complete");
    let messages = goal["messages"].as_array().expect("messages");
    assert_eq!(
        messages.len(),
        4,
        "conversation has 4 turns (user + assistant + tool + assistant)"
    );
    assert_eq!(messages.last().unwrap()["role"], "assistant");
    assert_eq!(messages.last().unwrap()["text"], "The answer is 42.");

    assert_eq!(llm.calls().len(), 2);
    assert_eq!(tool.calls().len(), 1);
    assert_eq!(notify.calls().len(), 1);
    assert_eq!(notify.calls()[0]["text"], "The answer is 42.");

    assert!(rt.state.list_map("pending_llm").is_empty());
    assert!(rt.state.list_map("pending_tool").is_empty());
    assert_eq!(llm.remaining(), 0);
    assert_eq!(tool.remaining(), 0);
}

#[test]
fn two_goals_interleave_through_one_runtime() {
    let (mut rt, llm, tool, notify) = make_runtime();

    for _ in 0..2 {
        llm.push_ok(json!({
            "kind": "needs_tool",
            "text": "let me check",
            "tool": "search",
            "args": "x"
        }));
        llm.push_ok(json!({
            "kind": "final",
            "text": "done",
            "tool": "",
            "args": ""
        }));
        tool.push_ok(json!({ "result": "ok" }));
    }

    for gid in ["g1", "g2"] {
        rt.enqueue(InboundEvent::new(
            "UserMessage",
            json!({ "session_id": gid, "text": "p" }),
        ))
        .unwrap();
    }
    rt.run_to_quiescence().unwrap();

    let goals: Vec<(Json, Json)> = rt.state.list_map("goals").into_iter().collect();
    assert_eq!(goals.len(), 2);
    for (_id, g) in &goals {
        assert_eq!(g["status"], "complete");
    }
    assert_eq!(llm.calls().len(), 4);
    assert_eq!(tool.calls().len(), 2);
    assert_eq!(notify.calls().len(), 2);
}

#[test]
fn user_message_continues_an_existing_session() {
    let (mut rt, llm, _tool, notify) = make_runtime();

    // Turn 1: just a final answer, no tool.
    llm.push_ok(json!({
        "kind": "final",
        "text": "hi",
        "tool": "",
        "args": "",
    }));
    // Turn 2: also a final answer (the user's follow-up).
    llm.push_ok(json!({
        "kind": "final",
        "text": "still here",
        "tool": "",
        "args": "",
    }));

    rt.enqueue(InboundEvent::new(
        "UserMessage",
        json!({ "session_id": "s1", "text": "hello" }),
    ))
    .unwrap();
    rt.run_to_quiescence().unwrap();

    // After turn 1, session s1 has 2 messages: user "hello" + assistant "hi".
    let goal = rt
        .state
        .list_map("goals")
        .into_iter()
        .find_map(|(k, v)| if k == json!("s1") { Some(v) } else { None })
        .expect("session s1 exists");
    let msgs = goal["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["text"], "hello");
    assert_eq!(msgs[1]["role"], "assistant");
    assert_eq!(msgs[1]["text"], "hi");

    // Turn 2: same session, another user message.
    rt.enqueue(InboundEvent::new(
        "UserMessage",
        json!({ "session_id": "s1", "text": "are you there?" }),
    ))
    .unwrap();
    rt.run_to_quiescence().unwrap();

    let goal = rt
        .state
        .list_map("goals")
        .into_iter()
        .find_map(|(k, v)| if k == json!("s1") { Some(v) } else { None })
        .expect("session s1 still exists");
    let msgs = goal["messages"].as_array().unwrap();
    assert_eq!(
        msgs.len(),
        4,
        "conversation should accumulate across user turns"
    );
    assert_eq!(msgs[2]["role"], "user");
    assert_eq!(msgs[2]["text"], "are you there?");
    assert_eq!(msgs[3]["role"], "assistant");
    assert_eq!(msgs[3]["text"], "still here");

    assert_eq!(notify.calls().len(), 2, "notified once per assistant final");
    assert_eq!(notify.calls()[0]["text"], "hi");
    assert_eq!(notify.calls()[1]["text"], "still here");
}

#[test]
fn llm_failure_marks_goal_failed() {
    let (mut rt, llm, _tool, notify) = make_runtime();
    llm.push_failed("rate limited");

    rt.enqueue(InboundEvent::new(
        "UserMessage",
        json!({ "session_id": "g1", "text": "p" }),
    ))
    .unwrap();
    rt.run_to_quiescence().unwrap();

    let entries = rt.state.list_map("goals");
    let goal = &entries[0].1;
    assert_eq!(goal["status"], "failed");
    assert_eq!(notify.calls().len(), 0, "no notification on failure");
}
