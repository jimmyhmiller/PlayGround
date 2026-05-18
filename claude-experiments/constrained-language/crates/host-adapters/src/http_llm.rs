//! Anthropic Messages API adapter (blocking, ureq-based).
//!
//! Translates `LlmReq { goal_id, messages: list<Message> }` into a Messages
//! API call, then parses Claude's response as JSON matching the manifest's
//! `LlmResp { kind, text, tool, args }` shape. The system prompt instructs
//! Claude to output exactly that JSON; the prompt also describes the
//! available tools so Claude knows what to ask for.
//!
//! The API key is read from the `ANTHROPIC_API_KEY` environment variable.

use std::time::Duration;

use serde::Deserialize;
use thiserror::Error;

use runtime::effect::{Adapter, AdapterResult};
use runtime::Value;

const DEFAULT_MODEL: &str = "claude-sonnet-4-6";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";

/// System prompt that constrains Claude's output to the JSON shape our
/// manifest's `LlmResp` declares, and lists the tools available via the
/// `ExecuteTool` effect.
const SYSTEM_PROMPT: &str = r#"You are an agent that may call tools to answer a user's question.

You MUST respond with a single JSON object and nothing else. No prose, no markdown fences. Two valid shapes:

When you need a tool, respond like:
{"kind":"needs_tool","text":"<short thought>","tool":"<tool_name>","args":"<tool_args>"}

When you have the final answer, respond like:
{"kind":"final","text":"<answer>","tool":"","args":""}

Available tools:
- calc: evaluate a math expression. args is the expression, e.g. "2 + 3 * 4".
- time: return the current unix timestamp (UTC seconds). args is ignored.
- echo: return the args verbatim. args is the string to echo.

Use tools only when needed. After receiving a tool result (which appears as a "[tool result: ...]" message), produce a final answer."#;

#[derive(Debug, Error)]
pub enum AnthropicError {
    #[error("ANTHROPIC_API_KEY not set")]
    MissingKey,
    #[error("http: {0}")]
    Http(String),
    #[error("response parse: {0}")]
    Parse(String),
    #[error("model returned non-JSON content: {0}")]
    NonJsonContent(String),
}

#[derive(Clone)]
pub struct AnthropicLlmAdapter {
    api_key: String,
    model: String,
    timeout: Duration,
}

impl AnthropicLlmAdapter {
    /// Build using `ANTHROPIC_API_KEY` from the environment and the default
    /// model. Returns an error if the key is unset.
    pub fn from_env() -> Result<Self, AnthropicError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| AnthropicError::MissingKey)?;
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        Ok(Self {
            api_key,
            model,
            timeout: Duration::from_secs(60),
        })
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
}

impl Adapter for AnthropicLlmAdapter {
    fn fulfill(&mut self, request: Value, _emit_id: u64) -> AdapterResult {
        match call_anthropic(&self.api_key, &self.model, self.timeout, &request) {
            Ok(llm_resp) => AdapterResult::Ok(llm_resp),
            Err(e) => AdapterResult::Failed {
                reason: e.to_string(),
            },
        }
    }
}

fn call_anthropic(
    api_key: &str,
    model: &str,
    timeout: Duration,
    request: &Value,
) -> Result<Value, AnthropicError> {
    let messages = request.get("messages").and_then(|v| v.as_array());

    // Translate our Message[] into Anthropic's role/content shape. Our roles
    // are "user", "assistant", "tool"; Anthropic only knows user/assistant,
    // so tool messages become user messages prefixed with "[tool result:".
    let api_messages: Vec<serde_json::Value> = messages
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    let role = m.get("role").and_then(|v| v.as_str())?;
                    let text = m.get("text").and_then(|v| v.as_str())?;
                    let (out_role, out_text) = match role {
                        "user" | "assistant" => (role.to_string(), text.to_string()),
                        "tool" => ("user".to_string(), format!("[tool result: {text}]")),
                        _ => return None,
                    };
                    Some(serde_json::json!({
                        "role": out_role,
                        "content": out_text,
                    }))
                })
                .collect()
        })
        .unwrap_or_default();

    let body = serde_json::json!({
        "model": model,
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": api_messages,
    });

    let agent = ureq::AgentBuilder::new().timeout(timeout).build();
    let resp = agent
        .post(ANTHROPIC_URL)
        .set("x-api-key", api_key)
        .set("anthropic-version", ANTHROPIC_VERSION)
        .set("content-type", "application/json")
        .send_json(body);

    let raw_resp = match resp {
        Ok(r) => r,
        Err(ureq::Error::Status(code, r)) => {
            let body = r.into_string().unwrap_or_default();
            return Err(AnthropicError::Http(format!("HTTP {code}: {body}")));
        }
        Err(e) => return Err(AnthropicError::Http(e.to_string())),
    };

    let parsed: AnthropicResponse = raw_resp
        .into_json()
        .map_err(|e| AnthropicError::Parse(e.to_string()))?;

    let text = parsed
        .content
        .iter()
        .find(|b| b.kind == "text")
        .map(|b| b.text.clone())
        .ok_or_else(|| AnthropicError::Parse("no text content block".into()))?;

    // The model may wrap its JSON in stray whitespace or accidental markdown;
    // be tolerant.
    let cleaned = strip_code_fences(text.trim());

    let llm_resp: Value = serde_json::from_str(cleaned)
        .map_err(|_| AnthropicError::NonJsonContent(cleaned.to_string()))?;

    // Ensure the response has all four fields the manifest's LlmResp
    // declares (kind, text, tool, args). Missing fields default to "".
    let mut obj = llm_resp
        .as_object()
        .cloned()
        .ok_or_else(|| AnthropicError::NonJsonContent(cleaned.to_string()))?;
    for f in ["kind", "text", "tool", "args"] {
        obj.entry(f.to_string())
            .or_insert(Value::String(String::new()));
    }
    Ok(Value::Object(obj))
}

fn strip_code_fences(s: &str) -> &str {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("```json") {
        rest.trim_start_matches('\n').trim_end_matches("```").trim()
    } else if let Some(rest) = s.strip_prefix("```") {
        rest.trim_start_matches('\n').trim_end_matches("```").trim()
    } else {
        s
    }
}
