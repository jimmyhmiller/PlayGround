use std::collections::HashMap;

use crate::datom::{Datom, EntityId, TxId, Value};
use crate::index;
use crate::schema::{FieldType, SchemaRegistry};
use crate::storage::{self, TxnOps};

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
    #[error("Unique constraint violated: attribute '{attribute}' already has value {value} on entity {existing_entity}")]
    UniqueViolation {
        attribute: String,
        value: String,
        existing_entity: EntityId,
    },
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
    /// Retract an entire entity (soft delete): retract all currently-asserted datoms.
    RetractEntity {
        entity_type: String,
        entity: EntityId,
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
        } else if let Some(type_name) = obj.get("retract_entity").and_then(|t| t.as_str()) {
            let entity = obj
                .get("entity")
                .and_then(|e| e.as_u64())
                .ok_or("retract_entity op requires 'entity'")?;

            Ok(TxOp::RetractEntity {
                entity_type: type_name.to_string(),
                entity,
            })
        } else {
            Err("tx op must have 'assert', 'retract', or 'retract_entity' key".to_string())
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
    pub timestamp_ms: u64,
}

/// Process a transaction: validate against schema, generate datoms, write to storage.
pub fn process_transaction(
    txn: &dyn TxnOps,
    schema: &SchemaRegistry,
    tx_id: TxId,
    entity_counter: &mut EntityId,
    ops: Vec<TxOp>,
) -> Result<TransactionResult> {
    let mut datoms = Vec::new();
    let mut entity_ids = Vec::new();
    let mut pending_unique: HashMap<(String, String), EntityId> = HashMap::new();

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

                for (field_name, value) in data {
                    let field_def = type_def.get_field(field_name).unwrap();

                    if let FieldType::Enum(enum_name) = &field_def.field_type {
                        let (variant_name, variant_fields) = extract_enum_parts(value);
                        let enum_def = schema.get_enum(enum_name).unwrap();
                        let _variant_def = enum_def.get_variant(&variant_name).unwrap();
                        let base_attr = type_def.attribute_name(field_name);

                        if !is_new {
                            retract_current_enum(txn, &mut datoms, eid, &base_attr, tx_id)?;
                        }

                        datoms.push(Datom {
                            entity: eid,
                            attribute: format!("{}/__tag", base_attr),
                            value: Value::String(variant_name.clone()),
                            tx: tx_id,
                            added: true,
                        });

                        for (vf_name, vf_value) in &variant_fields {
                            datoms.push(Datom {
                                entity: eid,
                                attribute: format!("{}.{}/{}", base_attr, variant_name, vf_name),
                                value: vf_value.clone(),
                                tx: tx_id,
                                added: true,
                            });
                        }
                    } else {
                        let attr = type_def.attribute_name(field_name);
                        let field_def = type_def.get_field(field_name).unwrap();

                        if !is_new {
                            if let Some(current_val) = get_latest_value(txn, eid, &attr)? {
                                if current_val == *value {
                                    continue;
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

                        if field_def.unique {
                            check_unique_constraint(txn, eid, &attr, value, &pending_unique)?;
                            pending_unique
                                .insert((attr.clone(), format!("{}", value)), eid);
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
                            txn,
                            &mut datoms,
                            *entity,
                            &type_def.attribute_name(field_name),
                            tx_id,
                        )?;
                    } else {
                        let attr = type_def.attribute_name(field_name);
                        if let Some(current_val) = get_latest_value(txn, *entity, &attr)? {
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
            TxOp::RetractEntity {
                entity_type,
                entity,
            } => {
                // Validate the entity type exists
                let _type_def = schema
                    .get(entity_type)
                    .ok_or_else(|| TxError::UnknownType(entity_type.clone()))?;

                entity_ids.push(*entity);

                // Scan all datoms for this entity
                let scan_prefix = index::eavt_entity_prefix(*entity);
                let scan_end = index::prefix_end(&scan_prefix);
                let all_entries = txn.scan(&scan_prefix, &scan_end)?;

                // Group by attribute
                let mut attr_datoms: HashMap<String, Vec<Datom>> = HashMap::new();
                for (key, _) in &all_entries {
                    if let Some(datom) = index::decode_datom_from_eavt(key) {
                        attr_datoms
                            .entry(datom.attribute.clone())
                            .or_default()
                            .push(datom);
                    }
                }

                // Resolve current values and retract them all
                for (attr, mut history) in attr_datoms {
                    history.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));

                    let mut current: Option<Value> = None;
                    for d in history {
                        if d.added {
                            current = Some(d.value);
                        } else if current.as_ref() == Some(&d.value) {
                            current = None;
                        }
                    }

                    if let Some(val) = current {
                        datoms.push(Datom {
                            entity: *entity,
                            attribute: attr,
                            value: val,
                            tx: tx_id,
                            added: false,
                        });
                    }
                }
            }
        }
    }

    let datom_count = datoms.len();
    for datom in &datoms {
        // Write historical indexes
        for (key, value) in index::encode_datom(datom) {
            txn.put(key, value)?;
        }

        // Maintain current-state indexes
        if datom.added {
            // Assert: look up old value to clean up stale CURRENT_AVET entry
            let (aevt_key, aevt_val) =
                index::encode_current_aevt(&datom.attribute, datom.entity, &datom.value);

            // Check for previous value to delete stale CURRENT_AVET
            if let Some(old_val_bytes) = txn.get(&aevt_key)? {
                if let Some(old_val) = index::decode_current_value(&old_val_bytes) {
                    let old_avet_key = index::encode_current_avet(
                        &datom.attribute,
                        &old_val,
                        datom.entity,
                    );
                    txn.delete(&old_avet_key)?;
                }
            }

            txn.put(aevt_key, aevt_val)?;

            let avet_key =
                index::encode_current_avet(&datom.attribute, &datom.value, datom.entity);
            txn.put(avet_key, vec![])?;

            // Track entity count: new entity type assertion
            if datom.attribute == "__type" {
                if let Value::String(type_name) = &datom.value {
                    increment_type_count(txn, type_name)?;
                }
            }
        } else {
            // Retract: delete current-state entries
            let aevt_prefix = index::current_aevt_attr_prefix(&datom.attribute);
            let mut aevt_key = aevt_prefix;
            aevt_key.extend_from_slice(&datom.entity.to_be_bytes());

            txn.delete(&aevt_key)?;

            let avet_key =
                index::encode_current_avet(&datom.attribute, &datom.value, datom.entity);
            txn.delete(&avet_key)?;

            // Track entity count: entity type retraction
            if datom.attribute == "__type" {
                if let Value::String(type_name) = &datom.value {
                    decrement_type_count(txn, type_name)?;
                }
            }
        }
    }

    Ok(TransactionResult {
        tx_id,
        entity_ids,
        datom_count,
        timestamp_ms: 0, // Filled in by Database::transact()
    })
}

fn get_latest_value(
    txn: &dyn TxnOps,
    entity: EntityId,
    attr: &str,
) -> Result<Option<Value>> {
    let prefix = index::eavt_entity_attr_prefix(entity, attr);
    let end = index::prefix_end(&prefix);
    let existing = txn.scan(&prefix, &end)?;

    let mut datoms: Vec<Datom> = existing
        .iter()
        .filter_map(|(k, _)| index::decode_datom_from_eavt(k))
        .collect();

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

fn check_unique_constraint(
    txn: &dyn TxnOps,
    entity: EntityId,
    attr: &str,
    value: &Value,
    pending_unique: &HashMap<(String, String), EntityId>,
) -> Result<()> {
    let value_str = format!("{}", value);

    let key = (attr.to_string(), value_str.clone());
    if let Some(&existing_eid) = pending_unique.get(&key) {
        if existing_eid != entity {
            return Err(TxError::UniqueViolation {
                attribute: attr.to_string(),
                value: value_str,
                existing_entity: existing_eid,
            });
        }
    }

    let prefix = index::avet_attr_value_prefix(attr, value);
    let end = index::prefix_end(&prefix);
    let entries = txn.scan(&prefix, &end)?;

    let mut entity_datoms: HashMap<EntityId, Vec<Datom>> = HashMap::new();
    for (k, _) in &entries {
        if let Some(datom) = index::decode_datom_from_avet(k) {
            entity_datoms.entry(datom.entity).or_default().push(datom);
        }
    }

    for (eid, mut datoms) in entity_datoms {
        if eid == entity {
            continue;
        }
        datoms.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));
        let mut currently_asserted = false;
        for d in datoms {
            currently_asserted = d.added;
        }
        if currently_asserted {
            return Err(TxError::UniqueViolation {
                attribute: attr.to_string(),
                value: value_str,
                existing_entity: eid,
            });
        }
    }

    Ok(())
}

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

fn extract_enum_parts(value: &Value) -> (String, HashMap<String, Value>) {
    match value {
        Value::String(s) => (s.clone(), HashMap::new()),
        Value::Enum { variant, fields } => (variant.clone(), fields.clone()),
        _ => panic!("extract_enum_parts called on non-enum value"),
    }
}

fn type_count_key(type_name: &str) -> Vec<u8> {
    index::meta_key(&format!("type_count:{}", type_name))
}

fn increment_type_count(txn: &dyn TxnOps, type_name: &str) -> Result<()> {
    let key = type_count_key(type_name);
    let count = match txn.get(&key)? {
        Some(b) if b.len() == 8 => u64::from_be_bytes(b.try_into().unwrap()) + 1,
        _ => 1,
    };
    txn.put(key, count.to_be_bytes().to_vec())?;
    Ok(())
}

fn decrement_type_count(txn: &dyn TxnOps, type_name: &str) -> Result<()> {
    let key = type_count_key(type_name);
    let count = match txn.get(&key)? {
        Some(b) if b.len() == 8 => {
            let c = u64::from_be_bytes(b.try_into().unwrap());
            c.saturating_sub(1)
        }
        _ => 0,
    };
    txn.put(key, count.to_be_bytes().to_vec())?;
    Ok(())
}

fn retract_current_enum(
    txn: &dyn TxnOps,
    datoms: &mut Vec<Datom>,
    entity: EntityId,
    field_attr_base: &str,
    tx_id: TxId,
) -> Result<()> {
    let tag_attr = format!("{}/__tag", field_attr_base);
    if let Some(tag_val) = get_latest_value(txn, entity, &tag_attr)? {
        datoms.push(Datom {
            entity,
            attribute: tag_attr,
            value: tag_val,
            tx: tx_id,
            added: false,
        });
    }

    let scan_prefix = index::eavt_entity_prefix(entity);
    let scan_end = index::prefix_end(&scan_prefix);
    let all_entries = txn.scan(&scan_prefix, &scan_end)?;

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
