//! Tiny OpenAI-compatible chat-completions client.
//!
//! Used for classifier-shaped inferences ("is this directory a good
//! default cwd for this project?"). Blocking HTTP via `ureq`; call
//! from a dedicated thread, not from a Bevy system.
//!
//! Config from env, resolved once per call:
//!
//! - `LLM_BASE_URL`  default `https://api.deepseek.com/v1`
//! - `LLM_API_KEY`   fallback to `DEEPSEEK_KEY` then `OPENAI_API_KEY`
//! - `LLM_MODEL`     default `deepseek-chat`
//!
//! The structured-output story is "ask nicely with `response_format =
//! json_object` and validate". DeepSeek and other OpenAI-compatible
//! servers all support that; only OpenAI proper supports strict
//! `json_schema`. We deserialize the returned JSON into a typed Rust
//! struct via `serde_json::from_str`; on parse failure we surface the
//! raw text so the caller can log it.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum LlmError {
    MissingApiKey,
    Http(String),
    BadResponse(String),
    ParseJson { raw: String, err: String },
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::MissingApiKey => write!(
                f,
                "no LLM_API_KEY / DEEPSEEK_KEY / OPENAI_API_KEY env var set"
            ),
            LlmError::Http(e) => write!(f, "http: {}", e),
            LlmError::BadResponse(s) => write!(f, "bad response shape: {}", s),
            LlmError::ParseJson { raw, err } => {
                write!(f, "parse model output as JSON: {} (raw: {})", err, raw)
            }
        }
    }
}

impl std::error::Error for LlmError {}

#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl LlmConfig {
    /// Resolve config from environment. Returns `MissingApiKey` if no
    /// usable key is found in any of the recognised env vars.
    pub fn from_env() -> Result<Self, LlmError> {
        let base_url = std::env::var("LLM_BASE_URL")
            .unwrap_or_else(|_| "https://api.deepseek.com/v1".into());
        let api_key = std::env::var("LLM_API_KEY")
            .ok()
            .or_else(|| std::env::var("DEEPSEEK_KEY").ok())
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or(LlmError::MissingApiKey)?;
        let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "deepseek-chat".into());
        Ok(Self {
            base_url,
            api_key,
            model,
        })
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    response_format: ResponseFormat,
    temperature: f32,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

/// One-shot classifier call. Returns `T` parsed from the model's JSON
/// response. `system` describes the role; `user` is the case under
/// classification. The schema is communicated to the model purely
/// through the prompt — the OpenAI-compat servers we target don't all
/// support strict-schema mode, so we keep things portable.
pub fn classify<T: DeserializeOwned>(
    cfg: &LlmConfig,
    system: &str,
    user: &str,
) -> Result<T, LlmError> {
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let body = ChatRequest {
        model: &cfg.model,
        messages: vec![
            ChatMessage {
                role: "system",
                content: system,
            },
            ChatMessage {
                role: "user",
                content: user,
            },
        ],
        response_format: ResponseFormat {
            kind: "json_object",
        },
        // Low temperature for classifier-shaped outputs.
        temperature: 0.0,
    };
    let res = ureq::post(&url)
        .set("Authorization", &format!("Bearer {}", cfg.api_key))
        .set("Content-Type", "application/json")
        .send_json(serde_json::to_value(&body).expect("serialize request"));
    let res = res.map_err(|e| LlmError::Http(e.to_string()))?;
    let parsed: ChatResponse = res
        .into_json()
        .map_err(|e| LlmError::BadResponse(e.to_string()))?;
    let raw = parsed
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| LlmError::BadResponse("no choices in response".into()))?
        .message
        .content;
    serde_json::from_str::<T>(&raw).map_err(|e| LlmError::ParseJson {
        raw,
        err: e.to_string(),
    })
}
