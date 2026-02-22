use std::any::Any;
use std::sync::{Arc, RwLock};

use byteorder::{BigEndian, ByteOrder};

use crate::datom::{EntityId, TxId, Value};
use crate::index;
use crate::query::executor::{self, QueryResult};
use crate::query::Query;
use crate::schema::{EntityTypeDef, EnumTypeDef, FieldDef, FieldType, SchemaRegistry};
use crate::storage::{StorageBackend, StorageError};
use crate::tx::{self, TxOp};

const TX_COUNTER_KEY: &str = "tx_counter";
const ENTITY_COUNTER_KEY: &str = "entity_counter";

#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("Storage error: {0}")]
    Storage(#[from] crate::storage::StorageError),
    #[error("Transaction error: {0}")]
    Transaction(#[from] tx::TxError),
    #[error("Query error: {0}")]
    Query(String),
    #[error("Schema error: {0}")]
    Schema(String),
}

pub type Result<T> = std::result::Result<T, DbError>;

pub struct Database {
    storage: Arc<dyn StorageBackend>,
    schema: RwLock<SchemaRegistry>,
}

impl Database {
    /// Open or create a database with the given storage backend.
    pub fn open(storage: Arc<dyn StorageBackend>) -> Result<Self> {
        let db = Self {
            storage,
            schema: RwLock::new(SchemaRegistry::new()),
        };

        // Load schema from storage
        db.load_schema()?;

        Ok(db)
    }

    /// Load schema definitions from stored datoms.
    fn load_schema(&self) -> Result<()> {
        let result = self
            .storage
            .execute_txn(Box::new(|txn| {
                let enum_prefix = index::aevt_attr_prefix("__schema_enum");
                let enum_end = index::prefix_end(&enum_prefix);
                let enum_entries = txn.scan(&enum_prefix, &enum_end)?;

                let type_prefix = index::aevt_attr_prefix("__schema_type");
                let type_end = index::prefix_end(&type_prefix);
                let type_entries = txn.scan(&type_prefix, &type_end)?;

                Ok(Box::new((enum_entries, type_entries))
                    as Box<dyn Any + Send>)
            }))?;

        let (enum_entries, type_entries) = *result
            .downcast::<(Vec<(Vec<u8>, Vec<u8>)>, Vec<(Vec<u8>, Vec<u8>)>)>()
            .expect("wrong type");

        let mut schema = self.schema.write().unwrap();

        for (key, _) in &enum_entries {
            if let Some(datom) = index::decode_datom_from_aevt(key) {
                if !datom.added {
                    continue;
                }
                if let Value::String(json) = &datom.value {
                    if let Ok(enum_def) = serde_json::from_str::<EnumTypeDef>(json) {
                        schema.register_enum(enum_def);
                    }
                }
            }
        }

        for (key, _) in &type_entries {
            if let Some(datom) = index::decode_datom_from_aevt(key) {
                if !datom.added {
                    continue;
                }
                if let Value::String(json) = &datom.value {
                    if let Ok(type_def) = serde_json::from_str::<EntityTypeDef>(json) {
                        schema.register(type_def);
                    }
                }
            }
        }

        Ok(())
    }

    /// Define a new enum (sum) type.
    pub fn define_enum(&self, enum_def: EnumTypeDef) -> Result<TxId> {
        if enum_def.name.is_empty() {
            return Err(DbError::Schema("enum type name cannot be empty".into()));
        }
        if enum_def.name.starts_with("__") {
            return Err(DbError::Schema(
                "enum type name cannot start with '__'".into(),
            ));
        }
        if enum_def.variants.is_empty() {
            return Err(DbError::Schema("enum must have at least one variant".into()));
        }

        // Validate variant field types
        {
            let schema = self.schema.read().unwrap();
            for variant in &enum_def.variants {
                for field in &variant.fields {
                    match &field.field_type {
                        FieldType::Ref(target) => {
                            if !schema.contains(target) {
                                return Err(DbError::Schema(format!(
                                    "variant '{}' field '{}' references unknown type '{}'",
                                    variant.name, field.name, target
                                )));
                            }
                        }
                        FieldType::Enum(target) => {
                            if target != &enum_def.name && !schema.contains_enum(target) {
                                return Err(DbError::Schema(format!(
                                    "variant '{}' field '{}' references unknown enum '{}'",
                                    variant.name, field.name, target
                                )));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        let json =
            serde_json::to_string(&enum_def).map_err(|e| DbError::Schema(e.to_string()))?;

        let result = self
            .storage
            .execute_txn(Box::new(move |txn| {
                // Read+lock tx counter
                let tx_key = index::meta_key(TX_COUNTER_KEY);
                let tx_id: u64 = match txn.get_for_update(&tx_key, true)? {
                    Some(b) if b.len() == 8 => BigEndian::read_u64(&b) + 1,
                    _ => 1,
                };
                txn.put(tx_key, encode_u64(tx_id))?;

                // Read+lock entity counter
                let ec_key = index::meta_key(ENTITY_COUNTER_KEY);
                let entity_id: u64 = match txn.get_for_update(&ec_key, true)? {
                    Some(b) if b.len() == 8 => BigEndian::read_u64(&b) + 1,
                    _ => 1,
                };
                txn.put(ec_key, encode_u64(entity_id))?;

                let datom = crate::datom::Datom {
                    entity: entity_id,
                    attribute: "__schema_enum".to_string(),
                    value: Value::String(json),
                    tx: tx_id,
                    added: true,
                };

                for (key, value) in index::encode_datom(&datom) {
                    txn.put(key, value)?;
                }

                Ok(Box::new(tx_id) as Box<dyn Any + Send>)
            }))?;

        let tx_id = *result.downcast::<TxId>().expect("wrong type");
        self.schema.write().unwrap().register_enum(enum_def);

        Ok(tx_id)
    }

    /// Define a new entity type.
    pub fn define_type(&self, type_def: EntityTypeDef) -> Result<TxId> {
        // Validate the type definition
        if type_def.name.is_empty() {
            return Err(DbError::Schema("entity type name cannot be empty".into()));
        }
        if type_def.name.starts_with("__") {
            return Err(DbError::Schema(
                "entity type name cannot start with '__'".into(),
            ));
        }

        // Validate field types
        {
            let schema = self.schema.read().unwrap();
            for field in &type_def.fields {
                // Reject unique on enum-typed fields (unclear semantics)
                if field.unique {
                    if let FieldType::Enum(_) = &field.field_type {
                        return Err(DbError::Schema(format!(
                            "unique constraint is not supported on enum field '{}'",
                            field.name
                        )));
                    }
                }
                match &field.field_type {
                    FieldType::Ref(target) => {
                        if target != &type_def.name && !schema.contains(target) {
                            return Err(DbError::Schema(format!(
                                "ref field '{}' references unknown type '{}'",
                                field.name, target
                            )));
                        }
                    }
                    FieldType::Enum(target) => {
                        if !schema.contains_enum(target) {
                            return Err(DbError::Schema(format!(
                                "enum field '{}' references unknown enum type '{}'. Define the enum first.",
                                field.name, target
                            )));
                        }
                    }
                    _ => {}
                }
            }
        }

        let json =
            serde_json::to_string(&type_def).map_err(|e| DbError::Schema(e.to_string()))?;

        let result = self
            .storage
            .execute_txn(Box::new(move |txn| {
                // Read+lock tx counter
                let tx_key = index::meta_key(TX_COUNTER_KEY);
                let tx_id: u64 = match txn.get_for_update(&tx_key, true)? {
                    Some(b) if b.len() == 8 => BigEndian::read_u64(&b) + 1,
                    _ => 1,
                };
                txn.put(tx_key, encode_u64(tx_id))?;

                // Read+lock entity counter
                let ec_key = index::meta_key(ENTITY_COUNTER_KEY);
                let entity_id: u64 = match txn.get_for_update(&ec_key, true)? {
                    Some(b) if b.len() == 8 => BigEndian::read_u64(&b) + 1,
                    _ => 1,
                };
                txn.put(ec_key, encode_u64(entity_id))?;

                let datom = crate::datom::Datom {
                    entity: entity_id,
                    attribute: "__schema_type".to_string(),
                    value: Value::String(json),
                    tx: tx_id,
                    added: true,
                };

                for (key, value) in index::encode_datom(&datom) {
                    txn.put(key, value)?;
                }

                Ok(Box::new(tx_id) as Box<dyn Any + Send>)
            }))?;

        let tx_id = *result.downcast::<TxId>().expect("wrong type");

        // Register in memory
        self.schema.write().unwrap().register(type_def);

        Ok(tx_id)
    }

    /// Execute a transaction.
    pub fn transact(&self, ops: Vec<TxOp>) -> Result<tx::TransactionResult> {
        let schema = self.schema.read().unwrap().clone();

        let result = self
            .storage
            .execute_txn(Box::new(move |txn| {
                // Read+lock tx counter
                let tx_key = index::meta_key(TX_COUNTER_KEY);
                let tx_id: u64 = match txn.get_for_update(&tx_key, true)? {
                    Some(b) if b.len() == 8 => BigEndian::read_u64(&b) + 1,
                    _ => 1,
                };
                txn.put(tx_key.clone(), encode_u64(tx_id))?;

                // Read+lock entity counter
                let ec_key = index::meta_key(ENTITY_COUNTER_KEY);
                let mut entity_counter: u64 = match txn.get_for_update(&ec_key, true)? {
                    Some(b) if b.len() == 8 => BigEndian::read_u64(&b),
                    _ => 0,
                };

                // Run transaction logic
                let tx_result =
                    tx::process_transaction(txn, &schema, tx_id, &mut entity_counter, ops)
                        .map_err(|e| StorageError::Backend(e.to_string()))?;

                // Persist updated entity counter
                txn.put(ec_key, encode_u64(entity_counter))?;

                Ok(Box::new(tx_result) as Box<dyn Any + Send>)
            }))?;

        Ok(*result.downcast::<tx::TransactionResult>().expect("wrong type"))
    }

    /// Execute a query.
    pub fn query(&self, query: &Query) -> Result<QueryResult> {
        let schema = self.schema.read().unwrap().clone();
        let query = query.clone();

        let result = self
            .storage
            .execute_txn(Box::new(move |txn| {
                let qr = executor::execute_query(txn, &query, &schema)
                    .map_err(|e| StorageError::Backend(e))?;
                Ok(Box::new(qr) as Box<dyn Any + Send>)
            }))?;

        Ok(*result.downcast::<QueryResult>().expect("wrong type"))
    }

    /// Get all current field values for an entity.
    pub fn get_entity(
        &self,
        entity: EntityId,
    ) -> Result<Option<std::collections::HashMap<String, Value>>> {
        let result = self
            .storage
            .execute_txn(Box::new(move |txn| {
                let prefix = index::eavt_entity_prefix(entity);
                let end = index::prefix_end(&prefix);
                let entries = txn.scan(&prefix, &end)?;

                if entries.is_empty() {
                    return Ok(
                        Box::new(None::<std::collections::HashMap<String, Value>>)
                            as Box<dyn Any + Send>,
                    );
                }

                // Group datoms by attribute
                let mut attr_datoms: std::collections::HashMap<
                    String,
                    Vec<crate::datom::Datom>,
                > = std::collections::HashMap::new();

                for (key, _) in &entries {
                    if let Some(datom) = index::decode_datom_from_eavt(key) {
                        attr_datoms
                            .entry(datom.attribute.clone())
                            .or_default()
                            .push(datom);
                    }
                }

                // Resolve current value per attribute using full history
                let mut fields = std::collections::HashMap::new();
                for (attr, value) in resolve_current_values(attr_datoms, None) {
                    fields.insert(attr, value);
                }

                if fields.is_empty() {
                    Ok(Box::new(None::<std::collections::HashMap<String, Value>>)
                        as Box<dyn Any + Send>)
                } else {
                    Ok(
                        Box::new(Some(fields) as Option<std::collections::HashMap<String, Value>>)
                            as Box<dyn Any + Send>,
                    )
                }
            }))?;

        Ok(*result
            .downcast::<Option<std::collections::HashMap<String, Value>>>()
            .expect("wrong type"))
    }

    /// Return all datoms in EAVT order. For debugging and testing.
    pub fn all_datoms(&self) -> Result<Vec<crate::datom::Datom>> {
        let result = self
            .storage
            .execute_txn(Box::new(|txn| {
                let prefix = vec![index::EAVT_PREFIX];
                let end = index::prefix_end(&prefix);
                let entries = txn.scan(&prefix, &end)?;
                let datoms: Vec<crate::datom::Datom> = entries
                    .iter()
                    .filter_map(|(k, _)| index::decode_datom_from_eavt(k))
                    .collect();
                Ok(Box::new(datoms) as Box<dyn Any + Send>)
            }))?;

        Ok(*result.downcast::<Vec<crate::datom::Datom>>().expect("wrong type"))
    }

    /// Return all datoms for a specific entity in EAVT order.
    pub fn entity_datoms(&self, entity: EntityId) -> Result<Vec<crate::datom::Datom>> {
        let result = self
            .storage
            .execute_txn(Box::new(move |txn| {
                let prefix = index::eavt_entity_prefix(entity);
                let end = index::prefix_end(&prefix);
                let entries = txn.scan(&prefix, &end)?;
                let datoms: Vec<crate::datom::Datom> = entries
                    .iter()
                    .filter_map(|(k, _)| index::decode_datom_from_eavt(k))
                    .collect();
                Ok(Box::new(datoms) as Box<dyn Any + Send>)
            }))?;

        Ok(*result.downcast::<Vec<crate::datom::Datom>>().expect("wrong type"))
    }
}

fn encode_u64(val: u64) -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    BigEndian::write_u64(&mut buf, val);
    buf
}

/// Resolve the current value for each attribute from a full history of datoms.
/// Within the same tx, retracts are ordered before asserts (added=false < added=true).
/// Replays the history to determine which attributes currently have a value.
/// If `as_of` is Some, only datoms with tx <= as_of are considered.
pub fn resolve_current_values(
    attr_datoms: std::collections::HashMap<String, Vec<crate::datom::Datom>>,
    as_of: Option<TxId>,
) -> Vec<(String, Value)> {
    let mut result = Vec::new();

    for (attr, mut datoms) in attr_datoms {
        // Filter by as_of if provided
        if let Some(max_tx) = as_of {
            datoms.retain(|d| d.tx <= max_tx);
        }

        // Sort by tx ascending, retracts before asserts within same tx
        datoms.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));

        let mut current: Option<Value> = None;
        for d in datoms {
            if d.added {
                current = Some(d.value);
            } else {
                // Retract: clear if it matches current value (or unconditionally for simplicity)
                if current.as_ref() == Some(&d.value) || current.is_some() {
                    current = None;
                }
            }
        }

        if let Some(val) = current {
            result.push((attr, val));
        }
    }

    result
}

/// Parse a "define" request (entity type) from JSON.
pub fn parse_define_request(
    v: &serde_json::Value,
) -> std::result::Result<EntityTypeDef, String> {
    let name = v
        .get("entity_type")
        .and_then(|n| n.as_str())
        .ok_or("missing 'entity_type'")?
        .to_string();

    let fields = v
        .get("fields")
        .and_then(|f| f.as_array())
        .ok_or("missing 'fields' array")?
        .iter()
        .map(|f| {
            let field_name = f
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or("field missing 'name'")?
                .to_string();
            let field_type = parse_field_type(
                f.get("type")
                    .and_then(|t| t.as_str())
                    .ok_or("field missing 'type'")?,
            )?;
            let required = f.get("required").and_then(|r| r.as_bool()).unwrap_or(false);
            let unique = f.get("unique").and_then(|u| u.as_bool()).unwrap_or(false);
            let indexed = f.get("indexed").and_then(|i| i.as_bool()).unwrap_or(false);
            Ok(FieldDef {
                name: field_name,
                field_type,
                required,
                unique,
                indexed,
            })
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;

    Ok(EntityTypeDef { name, fields })
}

/// Parse a "define_enum" request from JSON.
pub fn parse_define_enum_request(
    v: &serde_json::Value,
) -> std::result::Result<EnumTypeDef, String> {
    let name = v
        .get("enum_name")
        .and_then(|n| n.as_str())
        .ok_or("missing 'enum_name'")?
        .to_string();

    let variants = v
        .get("variants")
        .and_then(|vs| vs.as_array())
        .ok_or("missing 'variants' array")?
        .iter()
        .map(|variant| {
            let variant_name = variant
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or("variant missing 'name'")?
                .to_string();

            let fields = variant
                .get("fields")
                .and_then(|f| f.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|f| {
                            let field_name = f
                                .get("name")
                                .and_then(|n| n.as_str())
                                .ok_or("variant field missing 'name'")?
                                .to_string();
                            let field_type = parse_field_type(
                                f.get("type")
                                    .and_then(|t| t.as_str())
                                    .ok_or("variant field missing 'type'")?,
                            )?;
                            let required =
                                f.get("required").and_then(|r| r.as_bool()).unwrap_or(false);
                            Ok(FieldDef {
                                name: field_name,
                                field_type,
                                required,
                                unique: false,
                                indexed: false,
                            })
                        })
                        .collect::<std::result::Result<Vec<_>, String>>()
                })
                .transpose()?
                .unwrap_or_default();

            Ok(crate::schema::EnumVariant {
                name: variant_name,
                fields,
            })
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;

    Ok(EnumTypeDef { name, variants })
}

fn parse_field_type(s: &str) -> std::result::Result<FieldType, String> {
    match s {
        "string" => Ok(FieldType::String),
        "i64" => Ok(FieldType::I64),
        "f64" => Ok(FieldType::F64),
        "bool" => Ok(FieldType::Bool),
        "bytes" => Ok(FieldType::Bytes),
        other => {
            if let Some(inner) = other.strip_prefix("ref(").and_then(|s| s.strip_suffix(')')) {
                Ok(FieldType::Ref(inner.to_string()))
            } else if let Some(inner) =
                other.strip_prefix("enum(").and_then(|s| s.strip_suffix(')'))
            {
                Ok(FieldType::Enum(inner.to_string()))
            } else {
                Err(format!("unknown field type: {}", other))
            }
        }
    }
}
