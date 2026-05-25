//! Canonical JSON encoding for hashing / content-addressing the manifest.
//!
//! Rules:
//! * Object keys sorted lexicographically.
//! * No insignificant whitespace.
//! * UTF-8 strings with `serde_json`'s standard escaping.
//! * Number representation follows `serde_json` for the value's source type
//!   (we do not introduce normalization beyond key sorting).

use serde::Serialize;
use serde_json::Value;

use crate::manifest::Manifest;

#[derive(Debug, thiserror::Error)]
pub enum CanonicalError {
    #[error("serialization failed: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// Serialize a manifest (or anything `Serialize`) to canonical JSON bytes.
pub fn to_canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, CanonicalError> {
    let v = serde_json::to_value(value)?;
    let sorted = sort_keys(v);
    Ok(serde_json::to_vec(&sorted)?)
}

pub fn to_canonical_string<T: Serialize>(value: &T) -> Result<String, CanonicalError> {
    Ok(String::from_utf8(to_canonical_bytes(value)?).expect("canonical JSON is UTF-8"))
}

pub fn canonical_manifest(manifest: &Manifest) -> Result<Vec<u8>, CanonicalError> {
    to_canonical_bytes(manifest)
}

fn sort_keys(v: Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut entries: Vec<(String, Value)> = map.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let mut out = serde_json::Map::with_capacity(entries.len());
            for (k, v) in entries {
                out.insert(k, sort_keys(v));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sort_keys).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sorts_keys_recursively() {
        let v = json!({
            "b": 1,
            "a": { "y": 2, "x": [ {"d": 4, "c": 3} ] }
        });
        let canon = to_canonical_string(&v).unwrap();
        assert_eq!(canon, r#"{"a":{"x":[{"c":3,"d":4}],"y":2},"b":1}"#);
    }

    #[test]
    fn no_whitespace() {
        let v = json!({"a": [1, 2, 3], "b": "hi"});
        let canon = to_canonical_string(&v).unwrap();
        assert!(!canon.contains(' '));
        assert!(!canon.contains('\n'));
    }
}
