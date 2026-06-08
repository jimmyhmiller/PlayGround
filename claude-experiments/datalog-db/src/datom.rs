use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub type EntityId = u64;
pub type TxId = u64;

/// Interned string used inside `Value::String`. `Arc<str>` makes
/// `clone()` a refcount bump instead of a heap allocation —
/// critical on join workloads where tuples (and the strings they
/// contain) are cloned thousands of times per query.
pub type Str = Arc<str>;

/// Payload for `Value::Enum`. Boxed inside `Value` so the rare enum
/// variant doesn't bloat every other `Value` in memory — relevant
/// because every tuple slot is `Option<Value>`, and intermediate
/// tuples are clone-hot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnumValue {
    pub variant: String,
    pub fields: HashMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Value {
    #[serde(rename = "string")]
    String(Str),
    #[serde(rename = "i64")]
    I64(i64),
    #[serde(rename = "f64")]
    F64(f64),
    #[serde(rename = "bool")]
    Bool(bool),
    #[serde(rename = "ref")]
    Ref(EntityId),
    #[serde(rename = "bytes")]
    Bytes(Vec<u8>),
    /// High-level enum value — expanded to tag + field datoms during transaction.
    /// Never stored directly in a datom; only exists in the user-facing API.
    /// Boxed to keep `Value` small.
    #[serde(rename = "enum")]
    Enum(Box<EnumValue>),
    /// An ordered list of homogeneous scalar values — the storage form of a
    /// cardinality-many (`[T]`) field. Stored as a single atomic datom value
    /// (last-write-wins like any scalar); membership is queryable via the
    /// `contains` predicate. Elements are never `Enum`, `Null`, or nested
    /// `List` (the schema enforces a scalar element type).
    #[serde(rename = "list")]
    List(Vec<Value>),
    /// Null — represents a missing optional field in query results.
    /// Never stored as a datom.
    #[serde(rename = "null")]
    Null,
}

impl Value {
    /// Byte tag for index encoding. Only valid for storable (non-Enum) values.
    pub fn type_tag(&self) -> u8 {
        match self {
            Value::String(_) => 0x01,
            Value::I64(_) => 0x02,
            Value::F64(_) => 0x03,
            Value::Bool(_) => 0x04,
            Value::Ref(_) => 0x05,
            Value::Bytes(_) => 0x06,
            Value::List(_) => 0x07,
            Value::Enum(_) => panic!("Enum values cannot be stored directly as datoms"),
            Value::Null => panic!("Null values cannot be stored directly as datoms"),
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::I64(n) => write!(f, "{}", n),
            Value::F64(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Ref(id) => write!(f, "ref({})", id),
            Value::Bytes(b) => write!(f, "bytes({})", b.len()),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Value::Enum(e) => {
                if e.fields.is_empty() {
                    write!(f, "{}", e.variant)
                } else {
                    write!(f, "{}{{...}}", e.variant)
                }
            }
            Value::Null => write!(f, "null"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Datom {
    pub entity: EntityId,
    pub attribute: String,
    pub value: Value,
    pub tx: TxId,
    pub added: bool,
}
