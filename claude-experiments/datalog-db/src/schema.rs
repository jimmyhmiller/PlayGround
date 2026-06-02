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
                | (FieldType::Enum(_), Value::Enum(_))
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

    /// Remove an entity type from the registry, returning its definition
    /// if it was present. Used by drop/purge; the caller is responsible
    /// for any persisted retraction or datom deletion.
    pub fn remove(&mut self, name: &str) -> Option<EntityTypeDef> {
        self.types.remove(name)
    }

    /// Remove an enum type from the registry, returning its definition
    /// if it was present.
    pub fn remove_enum(&mut self, name: &str) -> Option<EnumTypeDef> {
        self.enums.remove(name)
    }

    /// Human-readable descriptions of every *other* live type or enum
    /// whose schema references entity type `name` via a `ref` field.
    /// A self-reference (`Type.field: ref(Type)`) is not reported.
    /// Used to block a hard purge that would leave the schema dangling.
    pub fn referrers_of_type(&self, name: &str) -> Vec<String> {
        let mut out = Vec::new();
        for t in self.types.values() {
            if t.name == name {
                continue;
            }
            for f in &t.fields {
                if matches!(&f.field_type, FieldType::Ref(target) if target == name) {
                    out.push(format!("type {}.{}", t.name, f.name));
                }
            }
        }
        for e in self.enums.values() {
            for v in &e.variants {
                for f in &v.fields {
                    if matches!(&f.field_type, FieldType::Ref(target) if target == name) {
                        out.push(format!("enum {}::{}.{}", e.name, v.name, f.name));
                    }
                }
            }
        }
        out.sort();
        out
    }

    /// Human-readable descriptions of every live type or enum whose schema
    /// references enum `name` via an `enum` field. A self-reference inside
    /// the same enum is not reported.
    pub fn referrers_of_enum(&self, name: &str) -> Vec<String> {
        let mut out = Vec::new();
        for t in self.types.values() {
            for f in &t.fields {
                if matches!(&f.field_type, FieldType::Enum(target) if target == name) {
                    out.push(format!("type {}.{}", t.name, f.name));
                }
            }
        }
        for e in self.enums.values() {
            if e.name == name {
                continue;
            }
            for v in &e.variants {
                for f in &v.fields {
                    if matches!(&f.field_type, FieldType::Enum(target) if target == name) {
                        out.push(format!("enum {}::{}.{}", e.name, v.name, f.name));
                    }
                }
            }
        }
        out.sort();
        out
    }

    pub fn all_types(&self) -> Vec<&EntityTypeDef> {
        let mut types: Vec<_> = self.types.values().collect();
        types.sort_by_key(|t| &t.name);
        types
    }

    pub fn all_enums(&self) -> Vec<&EnumTypeDef> {
        let mut enums: Vec<_> = self.enums.values().collect();
        enums.sort_by_key(|e| &e.name);
        enums
    }
}
