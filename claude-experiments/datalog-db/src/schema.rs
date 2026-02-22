use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::datom::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    String,
    I64,
    F64,
    Bool,
    Ref(std::string::String),
    Bytes,
    Enum(std::string::String),
}

impl FieldType {
    pub fn matches_value(&self, value: &Value) -> bool {
        matches!(
            (self, value),
            (FieldType::String, Value::String(_))
                | (FieldType::I64, Value::I64(_))
                | (FieldType::F64, Value::F64(_))
                | (FieldType::Bool, Value::Bool(_))
                | (FieldType::Ref(_), Value::Ref(_))
                | (FieldType::Bytes, Value::Bytes(_))
                | (FieldType::Enum(_), Value::Enum { .. })
                | (FieldType::Enum(_), Value::String(_)) // unit variant as string
        )
    }

    pub fn type_name(&self) -> &str {
        match self {
            FieldType::String => "string",
            FieldType::I64 => "i64",
            FieldType::F64 => "f64",
            FieldType::Bool => "bool",
            FieldType::Ref(t) => t.as_str(),
            FieldType::Bytes => "bytes",
            FieldType::Enum(t) => t.as_str(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    pub name: std::string::String,
    #[serde(rename = "type")]
    pub field_type: FieldType,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub unique: bool,
    #[serde(default)]
    pub indexed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityTypeDef {
    pub name: std::string::String,
    pub fields: Vec<FieldDef>,
}

impl EntityTypeDef {
    pub fn get_field(&self, name: &str) -> Option<&FieldDef> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn attribute_name(&self, field: &str) -> std::string::String {
        format!("{}/{}", self.name, field)
    }
}

/// A single variant of an enum type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumVariant {
    pub name: std::string::String,
    #[serde(default)]
    pub fields: Vec<FieldDef>,
}

/// An enum (sum type) definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumTypeDef {
    pub name: std::string::String,
    pub variants: Vec<EnumVariant>,
}

impl EnumTypeDef {
    pub fn get_variant(&self, name: &str) -> Option<&EnumVariant> {
        self.variants.iter().find(|v| v.name == name)
    }
}

#[derive(Debug, Default, Clone)]
pub struct SchemaRegistry {
    types: HashMap<std::string::String, EntityTypeDef>,
    enums: HashMap<std::string::String, EnumTypeDef>,
}

impl SchemaRegistry {
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            enums: HashMap::new(),
        }
    }

    pub fn register(&mut self, type_def: EntityTypeDef) {
        self.types.insert(type_def.name.clone(), type_def);
    }

    pub fn register_enum(&mut self, enum_def: EnumTypeDef) {
        self.enums.insert(enum_def.name.clone(), enum_def);
    }

    pub fn get(&self, name: &str) -> Option<&EntityTypeDef> {
        self.types.get(name)
    }

    pub fn get_enum(&self, name: &str) -> Option<&EnumTypeDef> {
        self.enums.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.types.contains_key(name)
    }

    pub fn contains_enum(&self, name: &str) -> bool {
        self.enums.contains_key(name)
    }
}
