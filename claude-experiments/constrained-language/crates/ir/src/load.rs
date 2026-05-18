//! File-based manifest loaders.
//!
//! Both JSON and TOML are supported; the canonical authoring format is TOML
//! (it allows comments and is easier to keep readable). JSON remains the
//! interchange format used by `canonical.rs` for content-addressed hashing.

use std::path::Path;

use thiserror::Error;

use crate::manifest::Manifest;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported manifest extension `{0}` (use .toml or .json)")]
    UnknownExtension(String),
    #[error("toml parse: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("json parse: {0}")]
    Json(#[from] serde_json::Error),
}

/// Load a manifest from disk. Format is selected by extension: `.toml` or
/// `.json`. Extension matching is case-insensitive.
pub fn load_manifest_file(path: impl AsRef<Path>) -> Result<Manifest, LoadError> {
    let path = path.as_ref();
    let raw = std::fs::read_to_string(path)?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "toml" => Ok(toml::from_str(&raw)?),
        "json" => Ok(serde_json::from_str(&raw)?),
        other => Err(LoadError::UnknownExtension(other.to_string())),
    }
}

/// Parse a manifest from a TOML string directly (useful for tests).
pub fn parse_toml(raw: &str) -> Result<Manifest, LoadError> {
    Ok(toml::from_str(raw)?)
}

/// Parse a manifest from a JSON string directly (useful for tests).
pub fn parse_json(raw: &str) -> Result<Manifest, LoadError> {
    Ok(serde_json::from_str(raw)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::StateDecl;
    use crate::schema::{SchemaDef, SchemaRef};

    const TOML: &str = r#"
name = "hello-world"
version = "0.1.0"

[schemas.Greeting]
kind = "record"
fields = { name = "string" }

[events.Greeted]
payload = "Greeting"

[state.greet_counts]
kind = "map"
key = "string"
value = "u32"

[effects.Print]
request = "Greeting"
response = "string"

[[handlers]]
name  = "say_hello"
on    = "Greeted"
read  = ["greet_counts[$event.name]"]
write = ["greet_counts[$event.name]"]
emit  = ["Print"]
body  = { hash = "sha256:0", uri = "say_hello" }
"#;

    #[test]
    fn toml_round_trip_with_named_refs() {
        let m = parse_toml(TOML).expect("parse toml");
        assert_eq!(m.name, "hello-world");
        assert_eq!(m.handlers.len(), 1);
        assert_eq!(m.handlers[0].name, "say_hello");

        // Untagged SchemaRef should resolve to Named for bare strings.
        match &m.events.get("Greeted").unwrap().payload {
            SchemaRef::Named(n) => assert_eq!(n, "Greeting"),
            other => panic!("expected named, got {other:?}"),
        }

        // Tagged SchemaDef enum (kind = "map") should round-trip.
        match m.state.get("greet_counts").unwrap() {
            StateDecl::Map { key, value } => {
                assert!(matches!(key, SchemaRef::Named(n) if n == "string"));
                assert!(matches!(value, SchemaRef::Named(n) if n == "u32"));
            }
            other => panic!("expected Map, got {other:?}"),
        }

        // Nested fields in a record schema.
        match m.schemas.get("Greeting").unwrap() {
            SchemaDef::Record { fields } => {
                assert!(matches!(fields.get("name"), Some(SchemaRef::Named(n)) if n == "string"));
            }
            other => panic!("expected Record, got {other:?}"),
        }
    }

    #[test]
    fn toml_inline_schema_ref() {
        // payload as inline table — exercises the Inline variant of the
        // untagged SchemaRef enum.
        let src = r#"
name = "x"
version = "0"

[events.E]
payload = { kind = "record", fields = { x = "u32" } }
"#;
        let m = parse_toml(src).unwrap();
        match &m.events.get("E").unwrap().payload {
            SchemaRef::Inline(def) => match def.as_ref() {
                SchemaDef::Record { fields } => {
                    assert!(matches!(fields.get("x"), Some(SchemaRef::Named(n)) if n == "u32"));
                }
                _ => panic!("expected record"),
            },
            other => panic!("expected inline, got {other:?}"),
        }
    }

    #[test]
    fn validates_after_toml_parse() {
        let m = parse_toml(TOML).unwrap();
        crate::validate::validate(&m).expect("manifest is valid");
    }
}
