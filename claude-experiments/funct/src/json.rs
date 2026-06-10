//! Conversions between funct `Value` and `serde_json::Value`, for host
//! protocols that cross thread/process boundaries as JSON (render frames,
//! state snapshots, message-bus payloads).
//!
//! Lossy by design in one direction only: JSON has no tuples/ranges/atoms,
//! so converting *to* JSON fails loudly on values JSON cannot represent
//! (closures, atoms, native handles, ranges) — never a silent drop. Tuples
//! become arrays. Converting *from* JSON always succeeds.

use crate::interop::{FromValue, ToValue};
use crate::value::Value;
use crate::vm::Fault;
use std::collections::BTreeMap;

use crate::value::shared::Sh;

impl Value {
    /// Convert to JSON. Fails loudly on values JSON cannot represent
    /// (Closure, NativeFn, Atom, Cell, Native, Range, Variant).
    /// Variants are rejected too — unwrap them first (`unwrap_or`, `match`)
    /// or encode them as records; an implicit tagging scheme would be a
    /// silent format commitment.
    pub fn to_json(&self) -> Result<serde_json::Value, Fault> {
        use serde_json::Value as J;
        Ok(match self {
            Value::Unit => J::Null,
            Value::Bool(b) => J::Bool(*b),
            Value::Int(i) => J::Number((*i).into()),
            Value::Float(f) => serde_json::Number::from_f64(*f)
                .map(J::Number)
                .ok_or_else(|| Fault::new(format!("cannot represent {} in JSON", f)))?,
            Value::Str(s) => J::String(s.to_string()),
            Value::List(items) | Value::Tuple(items) => {
                J::Array(items.iter().map(|v| v.to_json()).collect::<Result<_, _>>()?)
            }
            Value::Record(r) => J::Object(
                r.iter()
                    .map(|(k, v)| Ok((k.clone(), v.to_json()?)))
                    .collect::<Result<_, Fault>>()?,
            ),
            other => {
                return Err(Fault::new(format!(
                    "cannot convert {} to JSON; convert it to records/lists/primitives first",
                    other.type_name()
                )))
            }
        })
    }

    /// Convert from JSON. Always succeeds: null → (), objects → records,
    /// arrays → lists, integral numbers → Int, other numbers → Float.
    pub fn from_json(j: &serde_json::Value) -> Value {
        use serde_json::Value as J;
        match j {
            J::Null => Value::Unit,
            J::Bool(b) => Value::Bool(*b),
            J::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else {
                    // u64 > i64::MAX or fractional
                    Value::Float(n.as_f64().unwrap_or(f64::NAN))
                }
            }
            J::String(s) => Value::str(s.clone()),
            J::Array(items) => Value::List(Sh::new(items.iter().map(Value::from_json).collect())),
            J::Object(fields) => {
                let map: BTreeMap<String, Value> =
                    fields.iter().map(|(k, v)| (k.clone(), Value::from_json(v))).collect();
                Value::Record(Sh::new(map))
            }
        }
    }
}

/// `vm.register1("on_payload", |p: serde_json::Value| ...)` just works.
impl FromValue for serde_json::Value {
    fn from_value(v: Value) -> Result<Self, Fault> {
        v.to_json()
    }
}

/// Rust can hand JSON straight to scripts (e.g. bus payloads).
impl ToValue for serde_json::Value {
    fn to_value(self) -> Value {
        Value::from_json(&self)
    }
}
