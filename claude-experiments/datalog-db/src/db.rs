use std::any::Any;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;

use byteorder::{BigEndian, ByteOrder};

use crate::datom::{EntityId, TxId, Value};
use crate::index;
use crate::query::executor::{self, QueryResult};
use crate::query::planner;
pub use crate::query::planner::QueryPlan;
use crate::query::{Pattern, Query};
use crate::schema::{EntityTypeDef, EnumTypeDef, FieldDef, FieldType, SchemaRegistry};
use crate::storage::{StorageBackend, StorageError, TxnOps};
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

/// Cache key derived from query shape, with literal values erased.
/// Two queries that differ only in constant values share the same plan.
#[derive(Clone, PartialEq, Eq)]
struct PlanCacheKey {
    hash: u64,
}

impl PlanCacheKey {
    fn from_query(query: &Query) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // Hash find variables
        query.find.hash(&mut hasher);
        // Hash whether as_of is present (not the value)
        query.as_of.is_some().hash(&mut hasher);
        // Hash clause shapes
        for clause in &query.where_clauses {
            clause.entity_type.hash(&mut hasher);
            clause.bind.hash(&mut hasher);
            for (field_name, pattern) in &clause.field_patterns {
                field_name.hash(&mut hasher);
                // Hash pattern shape, not values
                std::mem::discriminant(pattern).hash(&mut hasher);
                match pattern {
                    Pattern::Variable(v) => v.hash(&mut hasher),
                    Pattern::Constant(_) => { /* value erased */ }
                    Pattern::Predicate { op, .. } => {
                        std::mem::discriminant(op).hash(&mut hasher);
                    }
                    Pattern::EnumMatch { variant, field_patterns } => {
                        std::mem::discriminant(variant.as_ref()).hash(&mut hasher);
                        for (name, pat) in field_patterns {
                            name.hash(&mut hasher);
                            std::mem::discriminant(pat).hash(&mut hasher);
                        }
                    }
                }
            }
        }
        PlanCacheKey {
            hash: hasher.finish(),
        }
    }
}

impl Hash for PlanCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

pub struct Database {
    storage: Arc<dyn StorageBackend>,
    schema: RwLock<SchemaRegistry>,
    plan_cache: RwLock<HashMap<PlanCacheKey, Arc<QueryPlan>>>,
}

impl Database {
    /// Open or create a database with the given storage backend.
    pub fn open(storage: Arc<dyn StorageBackend>) -> Result<Self> {
        let db = Self {
            storage,
            schema: RwLock::new(SchemaRegistry::new()),
            plan_cache: RwLock::new(HashMap::new()),
        };

        // Load schema from storage
        db.load_schema()?;

        Ok(db)
    }

    /// Load schema definitions from stored datoms.
    fn load_schema(&self) -> Result<()> {
        let result = self
            .storage
            .execute_read(Box::new(|snap| {
                let enum_prefix = index::aevt_attr_prefix("__schema_enum");
                let enum_end = index::prefix_end(&enum_prefix);
                let enum_entries = snap.scan(&enum_prefix, &enum_end)?;

                let type_prefix = index::aevt_attr_prefix("__schema_type");
                let type_end = index::prefix_end(&type_prefix);
                let type_entries = snap.scan(&type_prefix, &type_end)?;

                Ok(Box::new((enum_entries, type_entries))
                    as Box<dyn Any + Send>)
            }))?;

        let (enum_entries, type_entries) = *result
            .downcast::<(Vec<(Vec<u8>, Vec<u8>)>, Vec<(Vec<u8>, Vec<u8>)>)>()
            .expect("wrong type");

        let mut schema = self.schema.write();

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

        // Hold write lock across validate + persist + register to prevent TOCTOU races
        let mut schema = self.schema.write();

        // Validate variant field types
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

        let json =
            serde_json::to_string(&enum_def).map_err(|e| DbError::Schema(e.to_string()))?;
        let timestamp_ms = now_millis();

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

                // Store wall-clock timestamp
                store_tx_timestamp(txn, tx_id, timestamp_ms)?;

                Ok(Box::new(tx_id) as Box<dyn Any + Send>)
            }))?;

        let tx_id = *result.downcast::<TxId>().expect("wrong type");
        schema.register_enum(enum_def);
        self.plan_cache.write().clear();

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

        // Hold write lock across validate + persist + register to prevent TOCTOU races
        let mut schema = self.schema.write();

        // Validate field types
        for field in &type_def.fields {
            if field.unique && !field.required {
                return Err(DbError::Schema(format!(
                    "unique field '{}' must also be required",
                    field.name
                )));
            }
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

        let json =
            serde_json::to_string(&type_def).map_err(|e| DbError::Schema(e.to_string()))?;
        let timestamp_ms = now_millis();

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

                // Store wall-clock timestamp
                store_tx_timestamp(txn, tx_id, timestamp_ms)?;

                Ok(Box::new(tx_id) as Box<dyn Any + Send>)
            }))?;

        let tx_id = *result.downcast::<TxId>().expect("wrong type");
        schema.register(type_def);
        self.plan_cache.write().clear();

        Ok(tx_id)
    }

    /// Execute a transaction.
    pub fn transact(&self, ops: Vec<TxOp>) -> Result<tx::TransactionResult> {
        let schema = self.schema.read().clone();
        let timestamp_ms = now_millis();

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
                let mut tx_result =
                    tx::process_transaction(txn, &schema, tx_id, &mut entity_counter, ops)
                        .map_err(|e| StorageError::Backend(e.to_string()))?;

                // Persist updated entity counter
                txn.put(ec_key, encode_u64(entity_counter))?;

                // Store wall-clock timestamp
                store_tx_timestamp(txn, tx_id, timestamp_ms)?;
                tx_result.timestamp_ms = timestamp_ms;

                Ok(Box::new(tx_result) as Box<dyn Any + Send>)
            }))?;

        Ok(*result.downcast::<tx::TransactionResult>().expect("wrong type"))
    }

    /// Resolve a wall-clock timestamp to the last tx_id at or before that time.
    pub fn resolve_tx_for_time(&self, target_ms: u64) -> Result<Option<TxId>> {
        let result = self
            .storage
            .execute_read(Box::new(move |snap| {
                let prefix = index::meta_key("tx_ts:");
                let end = index::prefix_end(&prefix);
                let entries = snap.scan(&prefix, &end)?;

                let mut best: Option<TxId> = None;
                for (key, value) in &entries {
                    if value.len() != 8 {
                        continue;
                    }
                    let ts = BigEndian::read_u64(value);
                    if ts <= target_ms {
                        // Extract tx_id from the key: meta key is "tx_ts:{20-digit tx_id}"
                        let key_str = String::from_utf8_lossy(key);
                        if let Some(tx_part) = key_str.rsplit("tx_ts:").next() {
                            if let Ok(tx_id) = tx_part.parse::<u64>() {
                                best = Some(tx_id);
                            }
                        }
                    }
                }

                Ok(Box::new(best) as Box<dyn Any + Send>)
            }))?;

        Ok(*result.downcast::<Option<TxId>>().expect("wrong type"))
    }

    /// Execute a query.
    pub fn query(&self, query: &Query) -> Result<QueryResult> {
        let schema = self.schema.read().clone();
        let mut query = query.clone();

        // Convert as_of_time to as_of if needed
        if query.as_of.is_none() {
            if let Some(target_ms) = query.as_of_time {
                match self.resolve_tx_for_time(target_ms)? {
                    Some(tx_id) => query.as_of = Some(tx_id),
                    // Time is before all transactions — use tx 0 so nothing matches
                    None => query.as_of = Some(0),
                }
            }
        }

        // Check plan cache
        let cache_key = PlanCacheKey::from_query(&query);
        let cached_plan = {
            let cache = self.plan_cache.read();
            cache.get(&cache_key).cloned()
        };

        if let Some(plan) = cached_plan {
            // Re-use cached plan, just update as_of
            let mut plan = (*plan).clone();
            plan.as_of = query.as_of;
            let result = self
                .storage
                .execute_read(Box::new(move |snap| {
                    let qr = executor::execute_plan(snap, &plan, &schema)
                        .map_err(|e| StorageError::Backend(e))?;
                    Ok(Box::new(qr) as Box<dyn Any + Send>)
                }))?;
            return Ok(*result.downcast::<QueryResult>().expect("wrong type"));
        }

        let result = self
            .storage
            .execute_read(Box::new(move |snap| {
                let plan = planner::plan_query(snap, &query, &schema)
                    .map_err(|e| StorageError::Backend(e))?;

                let qr = executor::execute_plan(snap, &plan, &schema)
                    .map_err(|e| StorageError::Backend(e))?;

                Ok(Box::new((plan, qr)) as Box<dyn Any + Send>)
            }))?;

        let (plan, qr) = *result
            .downcast::<(QueryPlan, QueryResult)>()
            .expect("wrong type");

        // Store in cache
        {
            let mut cache = self.plan_cache.write();
            cache.insert(cache_key, Arc::new(plan));
        }

        Ok(qr)
    }

    /// Generate a query plan without executing.
    pub fn explain(&self, query: &Query) -> Result<QueryPlan> {
        let schema = self.schema.read().clone();
        let mut query = query.clone();

        // Convert as_of_time to as_of if needed
        if query.as_of.is_none() {
            if let Some(target_ms) = query.as_of_time {
                match self.resolve_tx_for_time(target_ms)? {
                    Some(tx_id) => query.as_of = Some(tx_id),
                    None => query.as_of = Some(0),
                }
            }
        }

        let result = self
            .storage
            .execute_read(Box::new(move |snap| {
                let plan = planner::plan_query(snap, &query, &schema)
                    .map_err(|e| StorageError::Backend(e))?;
                Ok(Box::new(plan) as Box<dyn Any + Send>)
            }))?;

        Ok(*result.downcast::<QueryPlan>().expect("wrong type"))
    }

    /// Get all current field values for an entity.
    pub fn get_entity(
        &self,
        entity: EntityId,
    ) -> Result<Option<std::collections::HashMap<String, Value>>> {
        let result = self
            .storage
            .execute_read(Box::new(move |snap| {
                let prefix = index::eavt_entity_prefix(entity);
                let end = index::prefix_end(&prefix);
                let entries = snap.scan(&prefix, &end)?;

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
            .execute_read(Box::new(|snap| {
                let prefix = vec![index::EAVT_PREFIX];
                let end = index::prefix_end(&prefix);
                let entries = snap.scan(&prefix, &end)?;
                let datoms: Vec<crate::datom::Datom> = entries
                    .iter()
                    .filter_map(|(k, _)| index::decode_datom_from_eavt(k))
                    .collect();
                Ok(Box::new(datoms) as Box<dyn Any + Send>)
            }))?;

        Ok(*result.downcast::<Vec<crate::datom::Datom>>().expect("wrong type"))
    }

    /// Return the current schema as JSON.
    pub fn schema_json(&self) -> serde_json::Value {
        let schema = self.schema.read();
        serde_json::json!({
            "types": schema.all_types(),
            "enums": schema.all_enums(),
        })
    }

    /// Return all datoms for a specific entity in EAVT order.
    pub fn entity_datoms(&self, entity: EntityId) -> Result<Vec<crate::datom::Datom>> {
        let result = self
            .storage
            .execute_read(Box::new(move |snap| {
                let prefix = index::eavt_entity_prefix(entity);
                let end = index::prefix_end(&prefix);
                let entries = snap.scan(&prefix, &end)?;
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

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_millis() as u64
}

fn tx_timestamp_key(tx_id: TxId) -> Vec<u8> {
    index::meta_key(&format!("tx_ts:{:020}", tx_id))
}

/// Store tx_id -> timestamp mapping as metadata.
fn store_tx_timestamp(txn: &dyn TxnOps, tx_id: TxId, ts: u64) -> std::result::Result<(), StorageError> {
    txn.put(tx_timestamp_key(tx_id), encode_u64(ts))
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
