//! Trivial deterministic LLM stand-in for tests and offline runs.
//!
//! Looks at the most recent message in the request: if it's a "tool" message
//! (the runtime's marker for a tool result), produces a `final` answer that
//! echoes the tool result; otherwise asks the agent to call `calc` with the
//! message text as the expression.
//!
//! Paired with `SimpleToolsAdapter`'s `calc` tool, this turns the agent loop
//! into a math evaluator that exercises the full event/tool/notify chain
//! without needing an API key.

use runtime::effect::{Adapter, AdapterResult};
use runtime::Value;

#[derive(Default, Clone)]
pub struct FakeLlmAdapter {}

impl FakeLlmAdapter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Adapter for FakeLlmAdapter {
    fn fulfill(&mut self, request: Value, _emit_id: u64) -> AdapterResult {
        let messages = request
            .get("messages")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let last = messages.last().cloned().unwrap_or_else(|| serde_json::json!({}));
        let role = last.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let text = last.get("text").and_then(|v| v.as_str()).unwrap_or("");

        let resp = match role {
            "tool" => serde_json::json!({
                "kind": "final",
                "text": format!("(fake-llm) result: {text}"),
                "tool": "",
                "args": "",
            }),
            _ => serde_json::json!({
                "kind": "needs_tool",
                "text": "let me compute that",
                "tool": "calc",
                "args": text,
            }),
        };
        AdapterResult::Ok(resp)
    }
}
