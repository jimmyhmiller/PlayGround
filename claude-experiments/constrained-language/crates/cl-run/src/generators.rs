//! Built-in generator registry. Maps a `kind` string from the manifest's
//! `[generators.<name>]` to a concrete `Box<dyn Generator>`.

use host_adapters::StdinLinesGenerator;
use runtime::generator::Generator;
use serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub enum GeneratorBuildError {
    #[error("unknown generator kind `{0}` (known: stdin_lines)")]
    UnknownKind(String),
}

pub fn build(kind: &str, payload: Option<Value>) -> Result<Box<dyn Generator>, GeneratorBuildError> {
    match kind {
        "stdin_lines" => Ok(Box::new(StdinLinesGenerator::new(
            payload.unwrap_or(Value::Null),
        ))),
        other => Err(GeneratorBuildError::UnknownKind(other.to_string())),
    }
}
