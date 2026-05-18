//! Runtime value type.
//!
//! v0.1 uses `serde_json::Value` directly. We rely on its `PartialEq` for map
//! key equality (linear scan in the state store). When we add typed bindings
//! to WASM components, we'll likely keep this layer but add a typed-value
//! variant alongside.

pub type Value = serde_json::Value;

pub fn unit() -> Value {
    Value::Null
}
