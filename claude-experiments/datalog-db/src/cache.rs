use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::datom::{EntityId, Value};
use crate::index;
use crate::schema::{FieldType, SchemaRegistry};
use crate::storage::ReadOps;

/// In-memory cache of entity data, organized by type.
/// On first access to a type, all entities of that type are bulk-loaded
/// into memory via efficient AEVT scans. Subsequent reads are pure HashMap lookups.
pub struct QueryCache {
    /// type_name → Arc<entity_id → field_values>
    types: RwLock<HashMap<String, Arc<HashMap<EntityId, HashMap<String, Value>>>>>,
    /// Generation counter incremented on every invalidation.
    /// Used to detect stale loads: if the generation changes between
    /// starting and finishing a load, the loaded data may be stale.
    generation: AtomicU64,
}

impl QueryCache {
    pub fn new() -> Self {
        Self {
            types: RwLock::new(HashMap::new()),
            generation: AtomicU64::new(0),
        }
    }

    /// Ensure all entities of a type are loaded into cache and return the data.
    /// Combines ensure + get into a single operation to reduce lock overhead.
    /// Returns None only if a concurrent invalidation discarded the loaded data.
    pub fn ensure_type_loaded(
        &self,
        txn: &dyn ReadOps,
        type_name: &str,
        schema: &SchemaRegistry,
    ) -> Result<Option<Arc<HashMap<EntityId, HashMap<String, Value>>>>, String> {
        // Fast path: already loaded (single read lock)
        {
            let types = self.types.read();
            if let Some(data) = types.get(type_name) {
                return Ok(Some(data.clone()));
            }
        }

        // Snapshot the generation before loading
        let gen_before = self.generation.load(Ordering::Acquire);

        // Expensive loading without holding any lock
        let type_cache = load_type_data(txn, type_name, schema)?;

        // Insert with write lock, but only if no invalidation occurred during loading
        let mut types = self.types.write();
        let gen_after = self.generation.load(Ordering::Acquire);
        if gen_before == gen_after {
            // No invalidation happened — safe to cache
            let arc = types
                .entry(type_name.to_string())
                .or_insert_with(|| Arc::new(type_cache))
                .clone();
            Ok(Some(arc))
        } else {
            // Invalidation happened — discard stale data, fall through to uncached path
            Ok(None)
        }
    }

    /// Invalidate a specific type's cache entry.
    pub fn invalidate_type(&self, type_name: &str) {
        self.generation.fetch_add(1, Ordering::Release);
        self.types.write().remove(type_name);
    }

    /// Clear all cached data.
    pub fn invalidate_all(&self) {
        self.generation.fetch_add(1, Ordering::Release);
        self.types.write().clear();
    }
}

/// Load all entities and their field values for a given type.
fn load_type_data(
    txn: &dyn ReadOps,
    type_name: &str,
    schema: &SchemaRegistry,
) -> Result<HashMap<EntityId, HashMap<String, Value>>, String> {
    // Find all entity IDs of this type via CURRENT_AVET index
    let type_value = Value::String(type_name.to_string());
    let prefix = index::current_avet_attr_value_prefix("__type", &type_value);
    let end = index::prefix_end(&prefix);
    let mut entity_ids = Vec::new();
    txn.scan_foreach(&prefix, &end, &mut |key, _value| {
        entity_ids.push(index::current_avet_entity_at(key));
        true
    })
    .map_err(|e| e.to_string())?;

    // Initialize cache map with empty field maps for each entity
    let mut type_cache: HashMap<EntityId, HashMap<String, Value>> = HashMap::new();
    for &eid in &entity_ids {
        type_cache.insert(eid, HashMap::new());
    }

    // Load field values based on schema
    if let Some(type_def) = schema.get(type_name) {
        let entity_set: HashSet<EntityId> = entity_ids.iter().copied().collect();

        for field in &type_def.fields {
            match &field.field_type {
                FieldType::Enum(enum_name) => {
                    // Load enum tag: Type/field/__tag
                    load_field_into_cache(
                        txn,
                        type_name,
                        &format!("{}/__tag", field.name),
                        &entity_set,
                        &mut type_cache,
                    )?;

                    // Load variant sub-fields
                    if let Some(enum_def) = schema.get_enum(enum_name) {
                        for variant in &enum_def.variants {
                            for vf in &variant.fields {
                                load_field_into_cache(
                                    txn,
                                    type_name,
                                    &format!("{}.{}/{}", field.name, variant.name, vf.name),
                                    &entity_set,
                                    &mut type_cache,
                                )?;
                            }
                        }
                    }
                }
                _ => {
                    // Simple scalar/ref field
                    load_field_into_cache(
                        txn,
                        type_name,
                        &field.name,
                        &entity_set,
                        &mut type_cache,
                    )?;
                }
            }
        }
    }

    Ok(type_cache)
}

/// Scan CURRENT_AEVT for a single attribute and populate the cache map.
fn load_field_into_cache(
    txn: &dyn ReadOps,
    type_name: &str,
    field_name: &str,
    entity_set: &HashSet<EntityId>,
    type_cache: &mut HashMap<EntityId, HashMap<String, Value>>,
) -> Result<(), String> {
    let attr = format!("{}/{}", type_name, field_name);
    let prefix = index::current_aevt_attr_prefix(&attr);
    let end = index::prefix_end(&prefix);
    let attr_byte_len = attr.as_bytes().len();

    txn.scan_foreach(&prefix, &end, &mut |key, value| {
        let eid = index::current_aevt_entity_at(key, attr_byte_len);
        if entity_set.contains(&eid) {
            if let Some(val) = index::decode_current_value(value) {
                if let Some(fields) = type_cache.get_mut(&eid) {
                    fields.insert(field_name.to_string(), val);
                }
            }
        }
        true
    })
    .map_err(|e| e.to_string())?;

    Ok(())
}
