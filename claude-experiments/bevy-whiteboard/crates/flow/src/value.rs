use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::samples::Samples;
use crate::sim::NodeId;

/// The primitive data type flowing through the simulation.
///
/// Only scalars and small records are "state-bearing" for modeling
/// purposes. `Samples` is the one bounded-collection slot type we ship
/// — a ring of recent values, from which statistics and representative
/// draws are taken. Collections like `Map<K,V>` are deliberately
/// absent; you're expected to model via counts + distributions, not
/// implement the underlying data structure.
///
/// Tagged "envelopes" (what used to be `Variant { tag, payload }`) are
/// now plain `Record` values with the conventional shape
/// `{ kind: Str("..."), value: ... }`. The `Value::variant(tag, payload)`
/// constructor builds that shape; `as_variant()` reads it back. Treating
/// envelopes as data means rules can pattern-match on the structure
/// directly with `Pattern::Record`, including matching across any kind
/// (just bind the `kind` field as a `Var`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Nil,
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    /// Named fields. Uses `BTreeMap` for deterministic iteration.
    Record(BTreeMap<String, Value>),
    /// Bounded ring of recent samples. See `samples.rs`.
    Samples(Samples),
    /// A reference to a node in the sim. Produced by `Spawn` and by
    /// pool/children queries; consumed by dynamic emit routing and
    /// `Despawn`. Not a user-facing "database" of nodes — just a
    /// live handle.
    NodeRef(NodeId),
    /// Ordered list of values. First-class so routing strategies can
    /// enumerate neighbors, filter, map, fold, etc. Bounded only by
    /// the cost of building it. Operators that take a list argument
    /// (`Length`, `Index`, `Filter`, `Map`, `Reduce`, `EmitToEach`)
    /// panic loudly if they get the wrong shape.
    List(Vec<Value>),
}

/// Field name for the tag-discriminator in the record-encoded "variant"
/// shape. Use these so a future schema change touches one place.
pub const KIND_FIELD: &str = "kind";
pub const VALUE_FIELD: &str = "value";

impl Value {
    pub fn int(n: i64) -> Self { Value::Int(n) }
    pub fn float(f: f64) -> Self { Value::Float(f) }
    pub fn bool(b: bool) -> Self { Value::Bool(b) }
    pub fn nil() -> Self { Value::Nil }
    pub fn str(s: impl Into<String>) -> Self { Value::Str(s.into()) }
    /// Construct the conventional tagged envelope shape used by rules:
    /// `{ kind: Str(tag), value: payload }`. This used to be a separate
    /// `Variant` enum variant; it's now a plain record so pattern
    /// matching is uniform with any other record-shaped data.
    pub fn variant(tag: impl Into<String>, payload: Value) -> Self {
        let mut m = BTreeMap::new();
        m.insert(KIND_FIELD.to_string(), Value::Str(tag.into()));
        m.insert(VALUE_FIELD.to_string(), payload);
        Value::Record(m)
    }
    pub fn record<I, K>(fields: I) -> Self
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<String>,
    {
        Value::Record(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    pub fn list<I: IntoIterator<Item = Value>>(items: I) -> Self {
        Value::List(items.into_iter().collect())
    }
    pub fn as_list(&self) -> Option<&[Value]> {
        if let Value::List(v) = self { Some(v) } else { None }
    }

    pub fn as_int(&self) -> Option<i64> { if let Value::Int(n) = self { Some(*n) } else { None } }
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(n) => Some(*n as f64),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> { if let Value::Bool(b) = self { Some(*b) } else { None } }
    pub fn as_str(&self) -> Option<&str> { if let Value::Str(s) = self { Some(s) } else { None } }

    /// If this is the record-encoded tagged envelope shape
    /// (`{ kind: Str(_), value: _ }`), return `(tag, payload)`.
    /// Reads the new uniform Record encoding (no separate Variant enum
    /// variant exists anymore).
    pub fn as_variant(&self) -> Option<(&str, &Value)> {
        let Value::Record(m) = self else { return None };
        let Value::Str(tag) = m.get(KIND_FIELD)? else { return None };
        let payload = m.get(VALUE_FIELD)?;
        Some((tag.as_str(), payload))
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        use Value::*;
        match (self, other) {
            (Nil, Nil) => true,
            (Int(a), Int(b)) => a == b,
            (Float(a), Float(b)) => a == b,
            (Int(a), Float(b)) | (Float(b), Int(a)) => (*a as f64) == *b,
            (Bool(a), Bool(b)) => a == b,
            (Str(a), Str(b)) => a == b,
            (Record(a), Record(b)) => a == b,
            (NodeRef(a), NodeRef(b)) => a == b,
            (List(a), List(b)) => a == b,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

/// Patterns used to match against packets and slot values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Pattern {
    /// Matches anything, binds nothing.
    Wild,
    /// Matches anything, binds the whole value to `name`.
    Var(String),
    /// Matches if the value equals the literal.
    Lit(Value),
    /// Matches a record containing at least these fields, each matching.
    /// Extra fields in the value are allowed (structural, open).
    /// "Tagged envelope" matching (what used to be `Pattern::Variant`)
    /// is just a `Record` with `kind: Lit(Str(_))` plus a `value` sub-pattern
    /// — built by `Pattern::variant(tag, inner)`.
    Record(BTreeMap<String, Pattern>),
}

impl Pattern {
    pub fn wild() -> Self { Pattern::Wild }
    pub fn var(name: impl Into<String>) -> Self { Pattern::Var(name.into()) }
    pub fn lit(v: Value) -> Self { Pattern::Lit(v) }
    /// Build a record pattern that matches the conventional tagged
    /// envelope shape: `{ kind: Lit(Str(tag)), value: inner }`. Used by
    /// the DSL when lowering `on packet(p)` etc.
    pub fn variant(tag: impl Into<String>, inner: Pattern) -> Self {
        let mut m = BTreeMap::new();
        m.insert(KIND_FIELD.to_string(), Pattern::Lit(Value::Str(tag.into())));
        m.insert(VALUE_FIELD.to_string(), inner);
        Pattern::Record(m)
    }
    pub fn record<I, K>(fields: I) -> Self
    where
        I: IntoIterator<Item = (K, Pattern)>,
        K: Into<String>,
    {
        Pattern::Record(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
}

pub type Bindings = BTreeMap<String, Value>;

/// Try to match `value` against `pattern`, extending `bindings`.
/// Returns true on success; on failure, `bindings` may have been
/// partially mutated and should be considered invalid by the caller.
pub fn match_pattern(value: &Value, pattern: &Pattern, bindings: &mut Bindings) -> bool {
    match (pattern, value) {
        (Pattern::Wild, _) => true,
        (Pattern::Var(name), _) => {
            bindings.insert(name.clone(), value.clone());
            true
        }
        (Pattern::Lit(lit), v) => lit == v,
        (Pattern::Record(pfields), Value::Record(vfields)) => {
            for (k, pat) in pfields {
                let Some(vv) = vfields.get(k) else { return false; };
                if !match_pattern(vv, pat, bindings) { return false; }
            }
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_variant_with_var() {
        let v = Value::variant("Some", Value::Int(42));
        let pat = Pattern::variant("Some", Pattern::var("x"));
        let mut b = Bindings::new();
        assert!(match_pattern(&v, &pat, &mut b));
        assert_eq!(b.get("x"), Some(&Value::Int(42)));
    }

    #[test]
    fn match_record_partial() {
        let v = Value::record([("id", Value::Int(7)), ("ts", Value::Int(100))]);
        let pat = Pattern::record([("id", Pattern::var("i"))]);
        let mut b = Bindings::new();
        assert!(match_pattern(&v, &pat, &mut b));
        assert_eq!(b.get("i"), Some(&Value::Int(7)));
    }

    #[test]
    fn match_fail_wrong_variant() {
        let v = Value::variant("Some", Value::Int(1));
        let pat = Pattern::variant("None", Pattern::wild());
        let mut b = Bindings::new();
        assert!(!match_pattern(&v, &pat, &mut b));
    }
}
