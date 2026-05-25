//! Built-in adapter registry. Maps a `kind` string from runner.toml to a
//! concrete `Box<dyn Adapter>`.
//!
//! Adding a new adapter means adding it here AND in `host-adapters`.

use indexmap::IndexMap;
use toml::Value as TomlValue;

use host_adapters::{
    AnthropicLlmAdapter, FakeLlmAdapter, SimpleToolsAdapter, StdoutNotifyAdapter,
};
use runtime::effect::Adapter;

#[derive(Debug, thiserror::Error)]
pub enum AdapterBuildError {
    #[error("unknown adapter kind `{0}` (known: anthropic, fake_llm, simple_tools, stdout)")]
    UnknownKind(String),
    #[error("anthropic adapter: {0}")]
    Anthropic(#[from] host_adapters::AnthropicError),
}

pub fn build(
    kind: &str,
    options: &IndexMap<String, TomlValue>,
) -> Result<Box<dyn Adapter>, AdapterBuildError> {
    match kind {
        "anthropic" => {
            let mut a = AnthropicLlmAdapter::from_env()?;
            if let Some(model) = options.get("model").and_then(|v| v.as_str()) {
                a = a.with_model(model);
            }
            Ok(Box::new(a))
        }
        "fake_llm" => Ok(Box::new(FakeLlmAdapter::new())),
        "simple_tools" => Ok(Box::new(SimpleToolsAdapter::new())),
        "stdout" => Ok(Box::new(StdoutNotifyAdapter::new())),
        other => Err(AdapterBuildError::UnknownKind(other.to_string())),
    }
}
