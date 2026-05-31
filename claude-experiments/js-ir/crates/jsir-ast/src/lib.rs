//! JSIR-schema JavaScript AST + byte-exact JSON serialization for the
//! Rust JSIR reimplementation.
//!
//! Milestone 1 builds this out node-by-node. The foundation here is the
//! nlohmann-faithful JSON dumper ([`json::Json`]) plus a corpus-wide proof that
//! the dumper reproduces upstream `ast.json` formatting exactly.

pub mod ast_node;
pub mod json;
pub mod model;
pub mod schema_generated;

pub use ast_node::{AstNode, Field};
pub use json::Json;
pub use model::{ExtraVal, FieldValue, Node};

/// Convert a parsed `serde_json::Value` into our insertion-ordered [`Json`].
///
/// `serde_json` (with the `preserve_order` feature) keeps object key order, so
/// this is a faithful bridge: parsing a golden `ast.json` and re-dumping it
/// through [`Json::dump2`] must reproduce the original bytes. This validates
/// the dumper independently of the AST node structs.
pub fn json_from_serde(v: &serde_json::Value) -> Json {
    match v {
        serde_json::Value::Null => Json::Null,
        serde_json::Value::Bool(b) => Json::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Json::Int(i)
            } else if let Some(u) = n.as_u64() {
                // uids/offsets are small; represent as Int.
                Json::Int(u as i64)
            } else {
                Json::Float(n.as_f64().expect("finite f64"))
            }
        }
        serde_json::Value::String(s) => Json::Str(s.clone()),
        serde_json::Value::Array(items) => {
            Json::Array(items.iter().map(json_from_serde).collect())
        }
        serde_json::Value::Object(map) => Json::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), json_from_serde(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod corpus_tests {
    use super::*;

    /// Parse every golden `ast.json` into the **typed AST** ([`Node`]),
    /// re-serialize, and require byte-for-byte equality. This proves the typed
    /// node layer (schema + interpreter) is lossless across the whole corpus.
    #[test]
    fn typed_ast_round_trips_every_fixture() {
        let mut checked = 0;
        let mut failures = Vec::new();
        for f in jsir_oracle::list_fixtures() {
            let Some(expected) = f.expected_ast_json() else {
                continue;
            };
            let value: serde_json::Value = serde_json::from_str(&expected).expect("valid json");
            let node = match Node::from_json(&value) {
                Ok(n) => n,
                Err(e) => {
                    failures.push(format!("{}: {e}", f.name));
                    continue;
                }
            };
            let actual = node.to_json_string();
            if let Some(diff) = jsir_oracle::byte_diff(&expected, &actual) {
                failures.push(format!("{}:\n{diff}", f.name));
            } else {
                checked += 1;
            }
        }
        assert!(
            failures.is_empty(),
            "typed AST diverged on {} fixtures:\n{}",
            failures.len(),
            failures.join("\n\n")
        );
        assert!(checked >= 40, "expected ~46 fixtures, checked {checked}");
    }

    /// Parse every golden `ast.json`, round-trip it through our dumper, and
    /// require byte-for-byte equality. This proves [`Json::dump2`] matches
    /// `nlohmann::ordered_json::dump(2)` across the entire corpus.
    #[test]
    fn dumper_reproduces_every_fixture_ast_json() {
        let mut checked = 0;
        let mut failures = Vec::new();
        for f in jsir_oracle::list_fixtures() {
            let Some(expected) = f.expected_ast_json() else {
                continue;
            };
            let value: serde_json::Value = match serde_json::from_str(&expected) {
                Ok(v) => v,
                Err(e) => {
                    failures.push(format!("{}: parse error: {e}", f.name));
                    continue;
                }
            };
            let actual = json_from_serde(&value).dump2();
            if let Some(diff) = jsir_oracle::byte_diff(&expected, &actual) {
                failures.push(format!("{}:\n{diff}", f.name));
            } else {
                checked += 1;
            }
        }
        assert!(
            failures.is_empty(),
            "dumper diverged from nlohmann on {} fixtures:\n{}",
            failures.len(),
            failures.join("\n")
        );
        assert!(checked >= 40, "expected to check ~46 fixtures, checked {checked}");
    }
}
