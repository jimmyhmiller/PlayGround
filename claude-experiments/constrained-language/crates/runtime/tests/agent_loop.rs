//! Agent-loop end-to-end test.
//!
//! Models a minimal LLM agent: a goal arrives, the runtime calls an LLM,
//! the LLM asks for a tool, the runtime calls the tool, the result is
//! fed back to the LLM, the LLM produces a final answer, the user is
//! notified.
//!
//! Every step in this loop is driven by the runtime's event/effect
//! auto-routing. Nothing in the bodies "waits" — they emit and return,
//! and the response event re-invokes the next handler.

use serde_json::{json, Value as Json};

use ir::manifest::Manifest;
use runtime::{InboundEvent, MockAdapter, Runtime, ScriptedAdapter};

const MANIFEST_JSON: &str = r#"
{
  "name": "agent-loop",
  "version": "0.1.0",
  "schemas": {
    "Empty":   { "kind": "record", "fields": {} },
    "Message": {
      "kind": "record",
      "fields": { "role": "string", "text": "string" }
    },
    "Messages": { "kind": "list", "of": "Message" },

    "Status": {
      "kind": "sum",
      "variants": {
        "thinking":     null,
        "tool_calling": null,
        "complete":     null,
        "failed":       null
      }
    },
    "GoalRecord": {
      "kind": "record",
      "fields": {
        "id":         "string",
        "prompt":     "string",
        "status":     "Status",
        "messages":   "Messages",
        "step_count": "u32"
      }
    },

    "LlmReq": {
      "kind": "record",
      "fields": { "goal_id": "string", "messages": "Messages" }
    },
    "LlmResp": {
      "kind": "record",
      "fields": {
        "kind": "string",
        "text": "string",
        "tool": "string",
        "args": "string"
      }
    },
    "ToolReq": {
      "kind": "record",
      "fields": { "goal_id": "string", "tool": "string", "args": "string" }
    },
    "ToolResp": {
      "kind": "record",
      "fields": { "result": "string" }
    },
    "Notification": {
      "kind": "record",
      "fields": { "goal_id": "string", "text": "string" }
    },

    "ResponsePayload": {
      "kind": "record",
      "fields": {
        "emit_id": "u64",
        "outcome": {
          "kind": "record",
          "fields": { "status": "string" }
        }
      }
    }
  },

  "events": {
    "GoalReceived": {
      "payload": {
        "kind": "record",
        "fields": { "goal_id": "string", "prompt": "string" }
      }
    },
    "LlmReturned":  { "payload": "ResponsePayload" },
    "ToolReturned": { "payload": "ResponsePayload" }
  },

  "state": {
    "goals":        { "kind": "map", "key": "string", "value": "GoalRecord" },
    "pending_llm":  { "kind": "map", "key": "u64",    "value": "string" },
    "pending_tool": { "kind": "map", "key": "u64",    "value": "string" }
  },

  "effects": {
    "CallLlm": {
      "request":  "LlmReq",
      "response": "LlmResp",
      "response_event": "LlmReturned"
    },
    "ExecuteTool": {
      "request":  "ToolReq",
      "response": "ToolResp",
      "response_event": "ToolReturned"
    },
    "NotifyUser": {
      "request":  "Notification",
      "response": "Empty"
    }
  },

  "handlers": [
    {
      "name": "start_goal",
      "on":   "GoalReceived",
      "read":  [],
      "write": ["goals[*]", "pending_llm[*]"],
      "emit":  ["CallLlm"],
      "body":  { "hash": "sha256:0", "uri": "start_goal" }
    },
    {
      "name": "on_llm_returned",
      "on":   "LlmReturned",
      "read":  ["pending_llm[$event.emit_id]", "goals[*]"],
      "write": ["goals[*]", "pending_llm[*]", "pending_tool[*]"],
      "emit":  ["ExecuteTool", "NotifyUser"],
      "body":  { "hash": "sha256:0", "uri": "on_llm_returned" }
    },
    {
      "name": "on_tool_returned",
      "on":   "ToolReturned",
      "read":  ["pending_tool[$event.emit_id]", "goals[*]"],
      "write": ["goals[*]", "pending_tool[*]", "pending_llm[*]"],
      "emit":  ["CallLlm"],
      "body":  { "hash": "sha256:0", "uri": "on_tool_returned" }
    }
  ]
}
"#;

fn load() -> Manifest {
    serde_json::from_str(MANIFEST_JSON).expect("parse manifest")
}

fn append_assistant(messages: &mut Vec<Json>, text: &str) {
    messages.push(json!({ "role": "assistant", "text": text }));
}

fn append_tool(messages: &mut Vec<Json>, text: &str) {
    messages.push(json!({ "role": "tool", "text": text }));
}

fn unwrap_ok_response(outcome: &Json) -> Option<&Json> {
    if outcome.get("tag")? == "ok" {
        outcome.get("value")
    } else {
        None
    }
}

fn make_runtime() -> (Runtime, ScriptedAdapter, ScriptedAdapter, MockAdapter) {
    let mut rt = Runtime::new(load()).expect("validates");

    rt.bodies.register("start_goal", |ctx| {
        let goal_id = ctx
            .event
            .get("goal_id")
            .cloned()
            .expect("event has goal_id");
        let prompt = ctx
            .event
            .get("prompt")
            .cloned()
            .expect("event has prompt");

        let messages = json!([
            { "role": "user", "text": prompt }
        ]);
        let goal = json!({
            "id":         goal_id,
            "prompt":     prompt,
            "status":     { "tag": "thinking" },
            "messages":   messages,
            "step_count": 0u32
        });
        ctx.put_map("goals", goal_id.clone(), goal.clone())?;

        let emit_id = ctx.emit(
            "CallLlm",
            json!({ "goal_id": goal_id, "messages": goal["messages"] }),
        )?;
        ctx.put_map("pending_llm", json!(emit_id), goal_id)?;
        Ok(())
    });

    rt.bodies.register("on_llm_returned", |ctx| {
        let emit_id = ctx
            .event
            .get("emit_id")
            .and_then(Json::as_u64)
            .expect("event has emit_id");
        let goal_id = ctx
            .read_map_entry("pending_llm", &json!(emit_id))?
            .expect("a goal_id is registered for this emit_id");
        ctx.delete_map("pending_llm", json!(emit_id))?;

        let mut goal = ctx
            .read_map_entry("goals", &goal_id)?
            .expect("goal exists");
        let mut messages = goal["messages"].as_array().cloned().unwrap_or_default();

        let outcome = ctx.event.get("outcome").expect("payload has outcome");
        let Some(response) = unwrap_ok_response(outcome) else {
            goal["status"] = json!({ "tag": "failed" });
            ctx.put_map("goals", goal_id, goal)?;
            return Ok(());
        };

        let kind = response.get("kind").and_then(Json::as_str).unwrap_or("");
        let text = response
            .get("text")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string();
        append_assistant(&mut messages, &text);

        match kind {
            "final" => {
                goal["status"] = json!({ "tag": "complete" });
                goal["messages"] = json!(messages);
                ctx.put_map("goals", goal_id.clone(), goal)?;
                ctx.emit(
                    "NotifyUser",
                    json!({ "goal_id": goal_id, "text": text }),
                )?;
            }
            "needs_tool" => {
                let tool = response
                    .get("tool")
                    .and_then(Json::as_str)
                    .unwrap_or("")
                    .to_string();
                let args = response
                    .get("args")
                    .and_then(Json::as_str)
                    .unwrap_or("")
                    .to_string();
                goal["status"] = json!({ "tag": "tool_calling" });
                goal["messages"] = json!(messages);
                ctx.put_map("goals", goal_id.clone(), goal)?;

                let req_id = ctx.emit(
                    "ExecuteTool",
                    json!({ "goal_id": goal_id, "tool": tool, "args": args }),
                )?;
                ctx.put_map("pending_tool", json!(req_id), goal_id)?;
            }
            other => panic!("LLM returned unknown kind: {other}"),
        }
        Ok(())
    });

    rt.bodies.register("on_tool_returned", |ctx| {
        let emit_id = ctx
            .event
            .get("emit_id")
            .and_then(Json::as_u64)
            .expect("event has emit_id");
        let goal_id = ctx
            .read_map_entry("pending_tool", &json!(emit_id))?
            .expect("a goal_id is registered for this emit_id");
        ctx.delete_map("pending_tool", json!(emit_id))?;

        let mut goal = ctx
            .read_map_entry("goals", &goal_id)?
            .expect("goal exists");
        let mut messages = goal["messages"].as_array().cloned().unwrap_or_default();

        let outcome = ctx.event.get("outcome").expect("payload has outcome");
        let response = unwrap_ok_response(outcome).expect("tool ok");
        let result_text = response
            .get("result")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string();
        append_tool(&mut messages, &result_text);

        goal["status"] = json!({ "tag": "thinking" });
        goal["messages"] = json!(messages.clone());
        ctx.put_map("goals", goal_id.clone(), goal)?;

        let req_id = ctx.emit(
            "CallLlm",
            json!({ "goal_id": goal_id, "messages": messages }),
        )?;
        ctx.put_map("pending_llm", json!(req_id), goal_id)?;
        Ok(())
    });

    // Adapters
    let llm = ScriptedAdapter::new();
    let tool = ScriptedAdapter::new();
    let notify = MockAdapter::with_response(json!({}));

    rt.adapters.register("CallLlm", Box::new(llm.clone()));
    rt.adapters.register("ExecuteTool", Box::new(tool.clone()));
    rt.adapters
        .register("NotifyUser", Box::new(notify.clone()));

    (rt, llm, tool, notify)
}

#[test]
fn one_goal_one_tool_call_then_final_answer() {
    let (mut rt, llm, tool, notify) = make_runtime();

    // Script: first LLM call asks for the `search` tool; second LLM call
    // gives the final answer.
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
        "GoalReceived",
        json!({ "goal_id": "g1", "prompt": "what is the meaning of life" }),
    ))
    .unwrap();
    rt.run_to_quiescence().unwrap();

    // Goal exists and is complete.
    let entries = rt.state.list_map("goals");
    assert_eq!(entries.len(), 1, "exactly one goal in state");
    let goal = &entries[0].1;
    assert_eq!(goal["status"], json!({ "tag": "complete" }));
    let messages = goal["messages"].as_array().expect("messages");
    // user prompt + assistant ("needs_tool" message) + tool result + assistant ("final")
    assert_eq!(messages.len(), 4, "conversation has 4 turns");
    let last = messages.last().unwrap();
    assert_eq!(last["role"], "assistant");
    assert_eq!(last["text"], "The answer is 42.");

    // Effect call sequence.
    assert_eq!(llm.calls().len(), 2, "two LLM calls");
    assert_eq!(tool.calls().len(), 1, "one tool call");
    assert_eq!(notify.calls().len(), 1, "user notified once");
    assert_eq!(notify.calls()[0]["text"], "The answer is 42.");

    // No leftover pending entries.
    assert!(rt.state.list_map("pending_llm").is_empty());
    assert!(rt.state.list_map("pending_tool").is_empty());

    // Scripts were exhausted.
    assert_eq!(llm.remaining(), 0);
    assert_eq!(tool.remaining(), 0);
}

#[test]
fn two_goals_interleave_through_one_runtime() {
    let (mut rt, llm, tool, notify) = make_runtime();

    // Script (handlers run in goal-arrival order, so calls alternate):
    // g1: needs_tool → tool → final
    // g2: needs_tool → tool → final
    // Two LLM calls per goal, one tool call per goal.
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
            "GoalReceived",
            json!({ "goal_id": gid, "prompt": "p" }),
        ))
        .unwrap();
    }
    rt.run_to_quiescence().unwrap();

    let goals: Vec<(Json, Json)> = rt.state.list_map("goals").into_iter().collect();
    assert_eq!(goals.len(), 2);
    for (_id, g) in &goals {
        assert_eq!(g["status"], json!({ "tag": "complete" }));
    }
    assert_eq!(llm.calls().len(), 4);
    assert_eq!(tool.calls().len(), 2);
    assert_eq!(notify.calls().len(), 2);
}

#[test]
fn llm_failure_marks_goal_failed() {
    let (mut rt, llm, _tool, notify) = make_runtime();
    llm.push_failed("rate limited");

    rt.enqueue(InboundEvent::new(
        "GoalReceived",
        json!({ "goal_id": "g1", "prompt": "p" }),
    ))
    .unwrap();
    rt.run_to_quiescence().unwrap();

    let entries = rt.state.list_map("goals");
    let goal = &entries[0].1;
    assert_eq!(goal["status"], json!({ "tag": "failed" }));
    assert_eq!(notify.calls().len(), 0, "no notification on failure");
}
