use std::collections::HashMap;

use crate::datom::{Datom, EntityId, TxId, Value};
use crate::index;
use crate::schema::{FieldType, SchemaRegistry};
use crate::storage::{self, StorageBackend};

#[derive(Debug, thiserror::Error)]
pub enum TxError {
    #[error("Unknown entity type: {0}")]
    UnknownType(String),
    #[error("Unknown field '{field}' on type '{entity_type}'")]
    UnknownField { entity_type: String, field: String },
    #[error("Type mismatch for field '{field}': expected {expected}, got {actual}")]
    TypeMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    #[error("Missing required field '{field}' on type '{entity_type}'")]
    MissingRequired { entity_type: String, field: String },
    #[error("Unknown enum type: {0}")]
    UnknownEnum(String),
    #[error("Unknown variant '{variant}' on enum '{enum_name}'")]
    UnknownVariant { enum_name: String, variant: String },
    #[error("Unknown field '{field}' on variant '{enum_name}::{variant}'")]
    UnknownVariantField {
        enum_name: String,
        variant: String,
        field: String,
    },
    #[error("Type mismatch for field '{field}' on variant '{enum_name}::{variant}': expected {expected}, got {actual}")]
    VariantFieldTypeMismatch {
        enum_name: String,
        variant: String,
        field: String,
        expected: String,
        actual: String,
    },
    #[error("Missing required field '{field}' on variant '{enum_name}::{variant}'")]
    MissingVariantField {
        enum_name: String,
        variant: String,
        field: String,
    },
    #[error("Entity {0} not found")]
    EntityNotFound(EntityId),
    #[error("Storage error: {0}")]
    Storage(#[from] storage::StorageError),
}

pub type Result<T> = std::result::Result<T, TxError>;

/// A single operation in a transaction.
#[derive(Debug)]
pub enum TxOp {
    /// Assert a new entity or update fields on an existing one.
    Assert {
        entity_type: String,
        entity: Option<EntityId>,
        data: HashMap<String, Value>,
    },
    /// Retract specific fields from an entity.
    Retract {
        entity_type: String,
        entity: EntityId,
        fields: Vec<String>,
    },
}

impl TxOp {
    pub fn from_json(v: &serde_json::Value) -> std::result::Result<Self, String> {
        let obj = v.as_object().ok_or("tx op must be an object")?;

        if let Some(type_name) = obj.get("assert").and_then(|t| t.as_str()) {
            let entity = obj.get("entity").and_then(|e| e.as_u64());
            let data = obj
                .get("data")
                .and_then(|d| d.as_object())
                .ok_or("assert op requires 'data' object")?;

            let mut fields = HashMap::new();
            for (key, val) in data {
                let value = json_to_value(val)?;
                fields.insert(key.clone(), value);
            }

            Ok(TxOp::Assert {
                entity_type: type_name.to_string(),
                entity,
                data: fields,
            })
        } else if let Some(type_name) = obj.get("retract").and_then(|t| t.as_str()) {
            let entity = obj
                .get("entity")
                .and_then(|e| e.as_u64())
                .ok_or("retract op requires 'entity'")?;
            let fields = obj
                .get("fields")
                .and_then(|f| f.as_array())
                .ok_or("retract op requires 'fields' array")?
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect();

            Ok(TxOp::Retract {
                entity_type: type_name.to_string(),
                entity,
                fields,
            })
        } else {
            Err("tx op must have 'assert' or 'retract' key".to_string())
        }
    }
}

fn json_to_value(v: &serde_json::Value) -> std::result::Result<Value, String> {
    match v {
        serde_json::Value::String(s) => Ok(Value::String(s.clone())),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::I64(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Value::F64(f))
            } else {
                Err(format!("unsupported number: {}", n))
            }
        }
        serde_json::Value::Bool(b) => Ok(Value::Bool(*b)),
        serde_json::Value::Object(obj) => {
            if let Some(id) = obj.get("ref").and_then(|r| r.as_u64()) {
                return Ok(Value::Ref(id));
            }
            if obj.len() == 1 {
                let (variant_name, variant_data) = obj.iter().next().unwrap();
                let mut fields = HashMap::new();
                if let Some(field_obj) = variant_data.as_object() {
                    for (k, v) in field_obj {
                        fields.insert(k.clone(), json_to_value(v)?);
                    }
                } else {
                    return Err(format!(
                        "enum variant '{}' data must be an object",
                        variant_name
                    ));
                }
                return Ok(Value::Enum {
                    variant: variant_name.clone(),
                    fields,
                });
            }
            Err("unsupported object value".to_string())
        }
        _ => Err(format!("unsupported value type: {}", v)),
    }
}

#[derive(Debug)]
pub struct TransactionResult {
    pub tx_id: TxId,
    pub entity_ids: Vec<EntityId>,
    pub datom_count: usize,
}

/// Process a transaction: validate against schema, generate datoms, write to storage.
pub async fn process_transaction(
    storage: &dyn StorageBackend,
    schema: &SchemaRegistry,
    tx_id: TxId,
    entity_counter: &mut EntityId,
    ops: Vec<TxOp>,
) -> Result<TransactionResult> {
    let mut datoms = Vec::new();
    let mut entity_ids = Vec::new();

    for op in &ops {
        match op {
            TxOp::Assert {
                entity_type,
                entity,
                data,
            } => {
                let type_def = schema
                    .get(entity_type)
                    .ok_or_else(|| TxError::UnknownType(entity_type.clone()))?;

                // Validate fields
                for (field_name, value) in data {
                    let field_def = type_def.get_field(field_name).ok_or_else(|| {
                        TxError::UnknownField {
                            entity_type: entity_type.clone(),
                            field: field_name.clone(),
                        }
                    })?;

                    if let FieldType::Enum(enum_name) = &field_def.field_type {
                        validate_enum_value(schema, enum_name, field_name, value)?;
                    } else if !field_def.field_type.matches_value(value) {
                        return Err(TxError::TypeMismatch {
                            field: field_name.clone(),
                            expected: field_def.field_type.type_name().to_string(),
                            actual: format!("{}", value),
                        });
                    }
                }

                let is_new = entity.is_none();
                let eid = match entity {
                    Some(id) => *id,
                    None => {
                        *entity_counter += 1;
                        let eid = *entity_counter;
                        datoms.push(Datom {
                            entity: eid,
                            attribute: "__type".to_string(),
                            value: Value::String(entity_type.clone()),
                            tx: tx_id,
                            added: true,
                        });
                        eid
                    }
                };

                entity_ids.push(eid);

                // Check required fields for new entities
                if is_new {
                    for field_def in &type_def.fields {
                        if field_def.required && !data.contains_key(&field_def.name) {
                            return Err(TxError::MissingRequired {
                                entity_type: entity_type.clone(),
                                field: field_def.name.clone(),
                            });
                        }
                    }
                }

                // Generate datoms for each field
                for (field_name, value) in data {
                    let field_def = type_def.get_field(field_name).unwrap();

                    if let FieldType::Enum(enum_name) = &field_def.field_type {
                        let (variant_name, variant_fields) = extract_enum_parts(value);
                        let enum_def = schema.get_enum(enum_name).unwrap();
                        let _variant_def = enum_def.get_variant(&variant_name).unwrap();
                        let base_attr = type_def.attribute_name(field_name);

                        // On update: retract current enum state (tag + all variant fields)
                        if !is_new {
                            retract_current_enum(
                                storage,
                                &mut datoms,
                                eid,
                                &base_attr,
                                tx_id,
                            )
                            .await?;
                        }

                        // Assert new tag
                        datoms.push(Datom {
                            entity: eid,
                            attribute: format!("{}/__tag", base_attr),
                            value: Value::String(variant_name.clone()),
                            tx: tx_id,
                            added: true,
                        });

                        // Assert variant fields
                        for (vf_name, vf_value) in &variant_fields {
                            datoms.push(Datom {
                                entity: eid,
                                attribute: format!(
                                    "{}.{}/{}",
                                    base_attr, variant_name, vf_name
                                ),
                                value: vf_value.clone(),
                                tx: tx_id,
                                added: true,
                            });
                        }
                    } else {
                        // Scalar field
                        let attr = type_def.attribute_name(field_name);

                        if !is_new {
                            // Retract old value if it exists and differs
                            if let Some(current_val) =
                                get_latest_value(storage, eid, &attr).await?
                            {
                                if current_val == *value {
                                    continue; // Same value — no-op
                                }
                                datoms.push(Datom {
                                    entity: eid,
                                    attribute: attr.clone(),
                                    value: current_val,
                                    tx: tx_id,
                                    added: false,
                                });
                            }
                        }

                        datoms.push(Datom {
                            entity: eid,
                            attribute: attr,
                            value: value.clone(),
                            tx: tx_id,
                            added: true,
                        });
                    }
                }
            }
            TxOp::Retract {
                entity_type,
                entity,
                fields,
            } => {
                let type_def = schema
                    .get(entity_type)
                    .ok_or_else(|| TxError::UnknownType(entity_type.clone()))?;

                entity_ids.push(*entity);

                for field_name in fields {
                    let field_def = type_def.get_field(field_name).ok_or_else(|| {
                        TxError::UnknownField {
                            entity_type: entity_type.clone(),
                            field: field_name.clone(),
                        }
                    })?;

                    if let FieldType::Enum(_) = &field_def.field_type {
                        retract_current_enum(
                            storage,
                            &mut datoms,
                            *entity,
                            &type_def.attribute_name(field_name),
                            tx_id,
                        )
                        .await?;
                    } else {
                        let attr = type_def.attribute_name(field_name);
                        if let Some(current_val) =
                            get_latest_value(storage, *entity, &attr).await?
                        {
                            datoms.push(Datom {
                                entity: *entity,
                                attribute: attr,
                                value: current_val,
                                tx: tx_id,
                                added: false,
                            });
                        }
                    }
                }
            }
        }
    }

    // Encode all datoms into index keys
    let datom_count = datoms.len();
    let mut kv_ops = Vec::new();
    for datom in &datoms {
        kv_ops.extend(index::encode_datom(datom));
    }

    // Write atomically
    storage.batch_write(kv_ops).await?;

    Ok(TransactionResult {
        tx_id,
        entity_ids,
        datom_count,
    })
}

// --- Helpers ---

/// Get the latest live value for a (entity, attribute) pair.
/// Resolves assert/retract history to find the current state.
async fn get_latest_value(
    storage: &dyn StorageBackend,
    entity: EntityId,
    attr: &str,
) -> Result<Option<Value>> {
    let prefix = index::eavt_entity_attr_prefix(entity, attr);
    let end = index::prefix_end(&prefix);
    let existing = storage.scan(&prefix, &end).await?;

    let mut datoms: Vec<Datom> = existing
        .iter()
        .filter_map(|(k, _)| index::decode_datom_from_eavt(k))
        .collect();

    // Sort by tx, retracts before asserts within same tx
    datoms.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));

    let mut current: Option<Value> = None;
    for d in datoms {
        if d.added {
            current = Some(d.value);
        } else if current.as_ref() == Some(&d.value) {
            current = None;
        }
    }

    Ok(current)
}

/// Validate that a value is a valid enum value for the given enum type.
fn validate_enum_value(
    schema: &SchemaRegistry,
    enum_name: &str,
    field_name: &str,
    value: &Value,
) -> Result<()> {
    let enum_def = schema
        .get_enum(enum_name)
        .ok_or_else(|| TxError::UnknownEnum(enum_name.to_string()))?;

    let (variant_name, variant_fields) = match value {
        Value::String(s) => (s.clone(), HashMap::new()),
        Value::Enum { variant, fields } => (variant.clone(), fields.clone()),
        other => {
            return Err(TxError::TypeMismatch {
                field: field_name.to_string(),
                expected: format!("enum({})", enum_name),
                actual: format!("{}", other),
            })
        }
    };

    let variant_def = enum_def.get_variant(&variant_name).ok_or_else(|| {
        TxError::UnknownVariant {
            enum_name: enum_name.to_string(),
            variant: variant_name.clone(),
        }
    })?;

    for (vf_name, vf_value) in &variant_fields {
        let vf_def =
            variant_def
                .fields
                .iter()
                .find(|f| f.name == *vf_name)
                .ok_or_else(|| TxError::UnknownVariantField {
                    enum_name: enum_name.to_string(),
                    variant: variant_name.clone(),
                    field: vf_name.clone(),
                })?;
        if !vf_def.field_type.matches_value(vf_value) {
            return Err(TxError::VariantFieldTypeMismatch {
                enum_name: enum_name.to_string(),
                variant: variant_name.clone(),
                field: vf_name.clone(),
                expected: vf_def.field_type.type_name().to_string(),
                actual: format!("{}", vf_value),
            });
        }
    }

    for vf_def in &variant_def.fields {
        if vf_def.required && !variant_fields.contains_key(&vf_def.name) {
            return Err(TxError::MissingVariantField {
                enum_name: enum_name.to_string(),
                variant: variant_name.clone(),
                field: vf_def.name.clone(),
            });
        }
    }

    Ok(())
}

/// Extract variant name and fields from a Value (String for unit, Enum for data variant).
fn extract_enum_parts(value: &Value) -> (String, HashMap<String, Value>) {
    match value {
        Value::String(s) => (s.clone(), HashMap::new()),
        Value::Enum { variant, fields } => (variant.clone(), fields.clone()),
        _ => panic!("extract_enum_parts called on non-enum value"),
    }
}

/// Retract all current state of an enum field: tag + all variant field datoms.
async fn retract_current_enum(
    storage: &dyn StorageBackend,
    datoms: &mut Vec<Datom>,
    entity: EntityId,
    field_attr_base: &str, // e.g. "Drawing/shape"
    tx_id: TxId,
) -> Result<()> {
    // Retract the tag
    let tag_attr = format!("{}/__tag", field_attr_base);
    if let Some(tag_val) = get_latest_value(storage, entity, &tag_attr).await? {
        datoms.push(Datom {
            entity,
            attribute: tag_attr,
            value: tag_val,
            tx: tx_id,
            added: false,
        });
    }

    // Scan all datoms for this entity and filter for variant field attributes.
    // We can't use eavt_entity_attr_prefix for a partial attribute match because
    // the attribute encoding includes a length prefix, so "Drawing/shape." (len=15)
    // won't match "Drawing/shape.Circle/radius" (len=27) in the key bytes.
    let scan_prefix = index::eavt_entity_prefix(entity);
    let scan_end = index::prefix_end(&scan_prefix);
    let all_entries = storage.scan(&scan_prefix, &scan_end).await?;

    let variant_attr_prefix = format!("{}.", field_attr_base);
    let mut attr_datoms: HashMap<String, Vec<Datom>> = HashMap::new();
    for (key, _) in &all_entries {
        if let Some(datom) = index::decode_datom_from_eavt(key) {
            if datom.attribute.starts_with(&variant_attr_prefix) {
                attr_datoms
                    .entry(datom.attribute.clone())
                    .or_default()
                    .push(datom);
            }
        }
    }

    for (attr, mut attr_history) in attr_datoms {
        // Sort by tx, retracts before asserts within same tx
        attr_history.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));

        let mut current: Option<Value> = None;
        for d in attr_history {
            if d.added {
                current = Some(d.value);
            } else if current.as_ref() == Some(&d.value) {
                current = None;
            }
        }

        if let Some(val) = current {
            datoms.push(Datom {
                entity,
                attribute: attr,
                value: val,
                tx: tx_id,
                added: false,
            });
        }
    }

    Ok(())
}
