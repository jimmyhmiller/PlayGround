use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// A reference to a schema definition.
///
/// In JSON, a bare string is a named reference; an object is an inline
/// definition. Named references resolve either to a user-defined entry in
/// `manifest.schemas` or to one of the built-in primitive names.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum SchemaRef {
    Named(String),
    Inline(Box<SchemaDef>),
}

/// A structural schema definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SchemaDef {
    Bool,
    U32,
    U64,
    I32,
    I64,
    F32,
    F64,
    String,
    Bytes,
    Timestamp,
    Record {
        fields: IndexMap<String, SchemaRef>,
    },
    Sum {
        /// `null` payload means a unit variant.
        variants: IndexMap<String, Option<SchemaRef>>,
    },
    List {
        of: SchemaRef,
    },
    Map {
        key: SchemaRef,
        value: SchemaRef,
    },
    Option {
        of: SchemaRef,
    },
}

impl SchemaDef {
    /// Return the `SchemaDef` for one of the lowercase primitive names, or `None`.
    pub fn primitive_by_name(name: &str) -> Option<Self> {
        Some(match name {
            "bool" => SchemaDef::Bool,
            "u32" => SchemaDef::U32,
            "u64" => SchemaDef::U64,
            "i32" => SchemaDef::I32,
            "i64" => SchemaDef::I64,
            "f32" => SchemaDef::F32,
            "f64" => SchemaDef::F64,
            "string" => SchemaDef::String,
            "bytes" => SchemaDef::Bytes,
            "timestamp" => SchemaDef::Timestamp,
            _ => return None,
        })
    }

    pub fn is_primitive_name(name: &str) -> bool {
        Self::primitive_by_name(name).is_some()
    }
}
