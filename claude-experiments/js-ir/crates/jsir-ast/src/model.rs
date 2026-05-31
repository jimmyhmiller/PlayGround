//! The typed-but-data-driven AST representation and its byte-exact JSON
//! (de)serialization, driven by the generated schema in [`crate::schema_generated`].
//!
//! A [`Node`] is `{ type, fields }` where `fields` runs parallel to
//! `node_schema(type)`. Serialization routes through [`crate::json::Json`] whose
//! `dump2` is proven byte-identical to upstream `nlohmann::ordered_json::dump(2)`.

use crate::json::Json;
use crate::schema_generated::{helper_schema, node_schema};

/// How a field behaves when its value is absent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Presence {
    /// Always present and non-null.
    Required,
    /// Always present; serialized as `null` when absent (`"key": null`).
    MaybeNull,
    /// Omitted entirely from the object when absent.
    MaybeUndef,
}

/// The JSON-level shape of a field's value.
#[derive(Debug, Clone, Copy)]
pub enum Repr {
    Bool,
    Int,
    Float,
    /// A JSON string (covers plain strings and the operator enums).
    Str,
    /// A polymorphic AST node, dispatched on its `"type"` tag.
    Node,
    /// A fixed helper struct (no `"type"` tag), parsed by `helper_schema(name)`.
    Extra(&'static str),
    /// A JSON array of `repr` values.
    List(&'static Repr),
    /// A JSON array of `repr` values where holes are `null` (e.g. `[1, , 3]`).
    ListHoles(&'static Repr),
}

/// A single field's schema entry.
#[derive(Debug, Clone, Copy)]
pub struct FieldSpec {
    pub key: &'static str,
    pub presence: Presence,
    pub repr: Repr,
}

/// A field's runtime value.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    /// MaybeUndef field that is not present (omitted on serialize).
    Absent,
    /// Explicit JSON null (MaybeNull absent, or an array hole).
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Node(Box<Node>),
    Extra(Box<ExtraVal>),
    List(Vec<FieldValue>),
}

/// A polymorphic AST node. `fields` is parallel to `node_schema(&ty)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub ty: String,
    pub fields: Vec<FieldValue>,
}

/// A fixed helper struct value. `fields` is parallel to `helper_schema(&name)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtraVal {
    pub name: String,
    pub fields: Vec<FieldValue>,
}

#[derive(Debug)]
pub struct ParseError(pub String);

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AST parse error: {}", self.0)
    }
}
impl std::error::Error for ParseError {}

type Result<T> = std::result::Result<T, ParseError>;

impl Node {
    /// Parse a node from a `serde_json::Value` object (dispatches on `"type"`).
    pub fn from_json(v: &serde_json::Value) -> Result<Node> {
        let obj = v
            .as_object()
            .ok_or_else(|| ParseError("expected object for node".into()))?;
        let ty = obj
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or_else(|| ParseError("node missing \"type\"".into()))?
            .to_string();
        let schema = node_schema(&ty);
        if schema.is_empty() {
            return Err(ParseError(format!("unknown node type {ty:?}")));
        }
        let fields = parse_fields(schema, obj)?;
        Ok(Node { ty, fields })
    }

    /// Serialize to our ordered [`Json`] tree.
    pub fn to_json(&self) -> Json {
        let schema = node_schema(&self.ty);
        let mut pairs: Vec<(String, Json)> = Vec::with_capacity(self.fields.len() + 1);
        pairs.push(("type".to_string(), Json::Str(self.ty.clone())));
        emit_fields(schema, &self.fields, &mut pairs);
        Json::Object(pairs)
    }

    /// Serialize to a byte-exact `ast.json` string.
    pub fn to_json_string(&self) -> String {
        self.to_json().dump2()
    }

    /// Look up a field value by its JSON key (schema order). Returns `None` if
    /// the key is not in this node's schema.
    pub fn field(&self, key: &str) -> Option<&FieldValue> {
        let schema = node_schema(&self.ty);
        schema
            .iter()
            .position(|s| s.key == key)
            .and_then(|i| self.fields.get(i))
    }
}

impl ExtraVal {
    fn from_json(name: &str, v: &serde_json::Value) -> Result<ExtraVal> {
        let obj = v
            .as_object()
            .ok_or_else(|| ParseError(format!("expected object for helper {name}")))?;
        let schema = helper_schema(name);
        let fields = parse_fields(schema, obj)?;
        Ok(ExtraVal {
            name: name.to_string(),
            fields,
        })
    }

    fn to_json(&self) -> Json {
        let schema = helper_schema(&self.name);
        let mut pairs: Vec<(String, Json)> = Vec::with_capacity(self.fields.len());
        emit_fields(schema, &self.fields, &mut pairs);
        Json::Object(pairs)
    }

    /// Look up a helper field value by its JSON key (schema order).
    pub fn field(&self, key: &str) -> Option<&FieldValue> {
        let schema = helper_schema(&self.name);
        schema
            .iter()
            .position(|s| s.key == key)
            .and_then(|i| self.fields.get(i))
    }
}

/// Parse all fields of an object per `schema`, producing values parallel to it.
fn parse_fields(
    schema: &[FieldSpec],
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<FieldValue>> {
    let mut out = Vec::with_capacity(schema.len());
    for spec in schema {
        let v = obj.get(spec.key);
        let value = match v {
            None => match spec.presence {
                Presence::Required => {
                    return Err(ParseError(format!("missing required field {:?}", spec.key)))
                }
                // Absent maybe-null/undef: omit on re-serialize.
                _ => FieldValue::Absent,
            },
            Some(serde_json::Value::Null) => FieldValue::Null,
            Some(val) => parse_repr(&spec.repr, val)?,
        };
        out.push(value);
    }
    Ok(out)
}

fn parse_repr(repr: &Repr, v: &serde_json::Value) -> Result<FieldValue> {
    Ok(match repr {
        Repr::Bool => FieldValue::Bool(
            v.as_bool()
                .ok_or_else(|| ParseError(format!("expected bool, got {v}")))?,
        ),
        Repr::Int => FieldValue::Int(
            v.as_i64()
                .ok_or_else(|| ParseError(format!("expected int, got {v}")))?,
        ),
        Repr::Float => FieldValue::Float(
            v.as_f64()
                .ok_or_else(|| ParseError(format!("expected float, got {v}")))?,
        ),
        Repr::Str => FieldValue::Str(
            v.as_str()
                .ok_or_else(|| ParseError(format!("expected string, got {v}")))?
                .to_string(),
        ),
        Repr::Node => FieldValue::Node(Box::new(Node::from_json(v)?)),
        Repr::Extra(name) => FieldValue::Extra(Box::new(ExtraVal::from_json(name, v)?)),
        Repr::List(elem) => {
            let arr = v
                .as_array()
                .ok_or_else(|| ParseError("expected array".into()))?;
            let mut items = Vec::with_capacity(arr.len());
            for e in arr {
                items.push(parse_repr(elem, e)?);
            }
            FieldValue::List(items)
        }
        Repr::ListHoles(elem) => {
            let arr = v
                .as_array()
                .ok_or_else(|| ParseError("expected array".into()))?;
            let mut items = Vec::with_capacity(arr.len());
            for e in arr {
                items.push(if e.is_null() {
                    FieldValue::Null
                } else {
                    parse_repr(elem, e)?
                });
            }
            FieldValue::List(items)
        }
    })
}

/// Emit `fields` (parallel to `schema`) into ordered JSON pairs. Absent fields
/// are skipped; everything else (including explicit Null) is emitted.
fn emit_fields(schema: &[FieldSpec], fields: &[FieldValue], pairs: &mut Vec<(String, Json)>) {
    for (spec, val) in schema.iter().zip(fields) {
        if matches!(val, FieldValue::Absent) {
            continue;
        }
        pairs.push((spec.key.to_string(), field_to_json(val)));
    }
}

fn field_to_json(val: &FieldValue) -> Json {
    match val {
        FieldValue::Absent | FieldValue::Null => Json::Null,
        FieldValue::Bool(b) => Json::Bool(*b),
        FieldValue::Int(i) => Json::Int(*i),
        FieldValue::Float(f) => Json::Float(*f),
        FieldValue::Str(s) => Json::Str(s.clone()),
        FieldValue::Node(n) => n.to_json(),
        FieldValue::Extra(e) => e.to_json(),
        FieldValue::List(items) => Json::Array(items.iter().map(field_to_json).collect()),
    }
}
