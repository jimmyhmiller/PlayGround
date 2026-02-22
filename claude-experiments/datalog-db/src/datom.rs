use std::collections::HashMap;

use serde::{Deserialize, Serialize};

pub type EntityId = u64;
pub type TxId = u64;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Value {
    #[serde(rename = "string")]
    String(String),
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
    #[serde(rename = "enum")]
    Enum {
        variant: String,
        fields: HashMap<String, Value>,
    },
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
            Value::Enum { .. } => panic!("Enum values cannot be stored directly as datoms"),
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
            Value::Enum { variant, fields } => {
                if fields.is_empty() {
                    write!(f, "{}", variant)
                } else {
                    write!(f, "{}{{...}}", variant)
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
