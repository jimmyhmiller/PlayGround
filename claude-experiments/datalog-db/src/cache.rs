use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::datom::{EntityId, Value};
use crate::index;
use crate::intern::AttrInterner;
use crate::schema::{FieldType, SchemaRegistry};
use crate::storage::ReadOps;

/// Policy controlling how the per-type query cache is populated and
/// evicted.
///
/// The cache trades memory for query latency: under `Unbounded`, every
/// type touched by a query gets fully loaded into memory and stays there
/// until invalidated. Under `Bounded`, the cache keeps at most
/// `max_types` types in LRU order. Under `None`, the cache is disabled
/// entirely and queries read from the storage indexes on every call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// Never cache. Each query reads from storage indexes directly.
    /// Use this for high-cardinality types where caching would blow up
    /// RAM, or as a baseline to measure cache effectiveness.
    None,
    /// Cache up to `max_types` types in LRU order. When inserting a new
    /// type would exceed the limit, the least-recently-used type is
    /// evicted. `max_types == 0` is equivalent to `None`.
    Bounded { max_types: usize },
    /// Cache everything that gets queried, no eviction. Original
    /// behavior. Risks unbounded memory growth for high-cardinality
    /// types — pick `Bounded` or `None` in that case.
    Unbounded,
}

impl Default for CachePolicy {
    fn default() -> Self {
        CachePolicy::Unbounded
    }
}

/// In-memory cache of entity data, organized by type.
///
/// On first access to a type under a caching policy, all entities of
/// that type are bulk-loaded via efficient AEVT scans. Subsequent
/// reads are pure HashMap lookups.
pub struct QueryCache {
    policy: CachePolicy,
    inner: RwLock<CacheInner>,
    /// Generation counter incremented on every invalidation.
    /// Used to detect stale loads: if the generation changes between
    /// starting and finishing a load, the loaded data may be stale.
    generation: AtomicU64,
    /// Monotonic counter used to stamp each cache hit so we can pick
    /// the least-recently-used type to evict under `Bounded`.
    access_counter: AtomicU64,
}

struct CacheInner {
    types: HashMap<String, CachedType>,
}

struct CachedType {
    data: Arc<TypeData>,
    /// Last value read from `QueryCache::access_counter` when this entry
    /// was inserted or hit.
    last_access: AtomicU64,
}

/// Columnar per-type cache.
///
/// Layout:
/// - `entity_ids`: sorted vec of all entity IDs in this type
/// - `columns`: per-attribute Vec parallel to `entity_ids`. The value
///   for entity `entity_ids[idx]` on attribute `attr` is
///   `columns[attr][idx]`, or `None` if missing.
///
/// Replaces the previous `HashMap<EntityId, HashMap<String, Value>>`:
/// - Iterating all entities of a type is a sequential `Vec` walk
///   (cache-friendly) instead of HashMap bucket-chasing.
/// - The per-field attribute-name hash happens **once per query**
///   (when callers ask for a column reference via `column()`), not
///   once per entity per field. For Graph 2-Hop's inner manager
///   lookup loop, that's the dominant cost.
/// Per-attribute column. `Arc<Vec<...>>` lets resolved query patterns
/// hold owned references to columns (via Arc clone) instead of
/// borrowing them — sidesteps the self-referential struct problem
/// when caching resolved patterns inside a `ScanIterator`.
pub type Column = Arc<Vec<Option<Value>>>;

pub struct TypeData {
    pub entity_ids: Vec<EntityId>,
    pub columns: HashMap<String, Column>,
}

impl TypeData {
    pub fn empty() -> Self {
        Self {
            entity_ids: Vec::new(),
            columns: HashMap::new(),
        }
    }

    /// Binary search for an entity's row index. `None` if not present.
    #[inline]
    pub fn index_of(&self, eid: EntityId) -> Option<usize> {
        self.entity_ids.binary_search(&eid).ok()
    }

    /// Borrow a column reference (the inner Arc is shared, not cloned).
    /// Cheaper than `column_arc` when you just need the slice for a
    /// short-lived scan.
    #[inline]
    pub fn column(&self, attr: &str) -> Option<&Column> {
        self.columns.get(attr)
    }

    /// Clone a column Arc. Use this when you need to store the column
    /// outside the type cache's lifetime (e.g., in a resolved-pattern
    /// cache that survives across iterator opens).
    #[inline]
    pub fn column_arc(&self, attr: &str) -> Option<Column> {
        self.columns.get(attr).cloned()
    }

    pub fn len(&self) -> usize {
        self.entity_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entity_ids.is_empty()
    }
}

impl QueryCache {
    /// Construct a cache with the default policy (`Unbounded`).
    pub fn new() -> Self {
        Self::with_policy(CachePolicy::default())
    }

    /// Construct a cache with an explicit policy.
    pub fn with_policy(policy: CachePolicy) -> Self {
        Self {
            policy,
            inner: RwLock::new(CacheInner {
                types: HashMap::new(),
            }),
            generation: AtomicU64::new(0),
            access_counter: AtomicU64::new(0),
        }
    }

    /// Inspect the active policy.
    pub fn policy(&self) -> CachePolicy {
        self.policy
    }

    /// Ensure all entities of a type are loaded into cache and return the data.
    /// Combines ensure + get into a single operation to reduce lock overhead.
    ///
    /// Returns `Ok(None)` when:
    ///   - the policy is `None` (always uncached), OR
    ///   - the policy is `Bounded { max_types: 0 }` (degenerate Bounded), OR
    ///   - a concurrent invalidation discarded the loaded data.
    ///
    /// In all of those cases the caller falls back to the uncached path
    /// that reads from the storage indexes directly.
    pub fn ensure_type_loaded(
        &self,
        txn: &dyn ReadOps,
        interner: &AttrInterner,
        type_name: &str,
        schema: &SchemaRegistry,
    ) -> Result<Option<Arc<TypeData>>, String> {
        // Disabled or degenerate policies skip the load entirely.
        match self.policy {
            CachePolicy::None => return Ok(None),
            CachePolicy::Bounded { max_types: 0 } => return Ok(None),
            _ => {}
        }

        // Fast path: already loaded (single read lock).
        {
            let inner = self.inner.read();
            if let Some(entry) = inner.types.get(type_name) {
                self.touch(entry);
                return Ok(Some(entry.data.clone()));
            }
        }

        // Snapshot the generation before loading — if it changes while
        // we're holding no lock, an invalidation happened and our work
        // must be discarded.
        let gen_before = self.generation.load(Ordering::Acquire);

        // Expensive load without any lock held.
        let type_cache = load_type_data(txn, interner, type_name, schema)?;
        let data = Arc::new(type_cache);

        // Insert with write lock, but only if no invalidation occurred
        // during loading.
        let mut inner = self.inner.write();
        let gen_after = self.generation.load(Ordering::Acquire);
        if gen_before != gen_after {
            // Stale — abandon the freshly loaded copy and let the caller
            // fall through to the uncached path.
            return Ok(None);
        }

        // Another thread may have raced us and inserted the type. If so,
        // return their copy and drop ours.
        if let Some(entry) = inner.types.get(type_name) {
            self.touch(entry);
            return Ok(Some(entry.data.clone()));
        }

        // Enforce `Bounded` capacity by evicting LRU types before
        // inserting.
        if let CachePolicy::Bounded { max_types } = self.policy {
            while inner.types.len() >= max_types {
                let victim = inner
                    .types
                    .iter()
                    .min_by_key(|(_, t)| t.last_access.load(Ordering::Relaxed))
                    .map(|(k, _)| k.clone());
                match victim {
                    Some(k) => {
                        inner.types.remove(&k);
                    }
                    None => break,
                }
            }
        }

        let access = self.access_counter.fetch_add(1, Ordering::Relaxed);
        inner.types.insert(
            type_name.to_string(),
            CachedType {
                data: data.clone(),
                last_access: AtomicU64::new(access),
            },
        );
        Ok(Some(data))
    }

    /// Update the LRU stamp on a cache entry. Cheap: a single atomic add
    /// on the global access counter plus an atomic store on the entry.
    fn touch(&self, entry: &CachedType) {
        let stamp = self.access_counter.fetch_add(1, Ordering::Relaxed);
        entry.last_access.store(stamp, Ordering::Relaxed);
    }

    /// Invalidate a specific type's cache entry.
    pub fn invalidate_type(&self, type_name: &str) {
        self.generation.fetch_add(1, Ordering::Release);
        self.inner.write().types.remove(type_name);
    }

    /// Incrementally extend the cached `TypeData` with one freshly-added
    /// entity instead of throwing the entry away. The new entity_id and
    /// its field values are appended to the existing columns; columns
    /// that don't exist yet are created with `None` backfilled for the
    /// pre-existing rows.
    ///
    /// No-op when the type isn't currently cached — the next read will
    /// load fresh from storage and naturally pick up the new entity.
    ///
    /// Constraints satisfied by the caller:
    /// - `eid` is monotonic (came from the entity_counter), so pushing
    ///   keeps `entity_ids` sorted ascending.
    /// - The entity is brand new (this is the pure-add fast path), so we
    ///   don't have to find and overwrite an existing row.
    ///
    /// Enum values are NOT supported here — they fan out into multiple
    /// `field/__tag` and `field.variant/sub_field` columns. The caller
    /// in `tx.rs` is responsible for falling back to `invalidate_type`
    /// when the transaction contains any enum values.
    pub fn append_entity(
        &self,
        type_name: &str,
        eid: EntityId,
        values: &HashMap<String, Value>,
    ) {
        if matches!(
            self.policy,
            CachePolicy::None | CachePolicy::Bounded { max_types: 0 }
        ) {
            return;
        }

        let mut inner = self.inner.write();
        let Some(entry) = inner.types.get_mut(type_name) else {
            return;
        };

        let old = entry.data.clone();
        let n = old.entity_ids.len();

        let mut entity_ids = old.entity_ids.clone();
        entity_ids.push(eid);

        let mut columns: HashMap<String, Column> =
            HashMap::with_capacity(old.columns.len().max(values.len()));
        for (k, col) in &old.columns {
            let mut v: Vec<Option<Value>> = (**col).clone();
            v.push(values.get(k).cloned());
            columns.insert(k.clone(), Arc::new(v));
        }
        for (k, val) in values {
            if !columns.contains_key(k) {
                let mut v: Vec<Option<Value>> = Vec::with_capacity(n + 1);
                v.resize(n, None);
                v.push(Some(val.clone()));
                columns.insert(k.clone(), Arc::new(v));
            }
        }

        entry.data = Arc::new(TypeData {
            entity_ids,
            columns,
        });
    }

    /// Clear all cached data.
    pub fn invalidate_all(&self) {
        self.generation.fetch_add(1, Ordering::Release);
        self.inner.write().types.clear();
    }

    /// Number of types currently held in the cache. Test/diagnostic helper.
    pub fn cached_type_count(&self) -> usize {
        self.inner.read().types.len()
    }
}

/// Load all entities and their field values for a given type.
fn load_type_data(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    type_name: &str,
    schema: &SchemaRegistry,
) -> Result<TypeData, String> {
    // Find all entity IDs of this type via CURRENT_AVET index. If the
    // `__type` attribute has never been allocated (empty DB), there's
    // nothing to load.
    let type_attr_id = match interner.lookup("__type") {
        Some(id) => id,
        None => return Ok(TypeData::empty()),
    };

    let type_value = Value::String(type_name.into());
    let prefix = index::current_avet_attr_value_prefix(type_attr_id, &type_value);
    let end = index::prefix_end(&prefix);
    let mut entity_ids = Vec::new();
    txn.scan_foreach(&prefix, &end, &mut |key, _value| {
        entity_ids.push(index::current_avet_entity_at(key));
        true
    })
    .map_err(|e| e.to_string())?;

    // CURRENT_AVET scan already returns entities in attr-id then value
    // order; for our point-lookup by entity_id we want entity_id order.
    entity_ids.sort_unstable();
    entity_ids.dedup();

    // Build the entity_id → row_index map once. Used by
    // `load_field_into_cache` below to fill each column.
    let entity_to_idx: HashMap<EntityId, usize> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &eid)| (eid, i))
        .collect();

    let mut columns: HashMap<String, Column> = HashMap::new();

    if let Some(type_def) = schema.get(type_name) {
        for field in &type_def.fields {
            match &field.field_type {
                FieldType::Enum(enum_name) => {
                    // Load enum tag: Type/field/__tag
                    load_field_into_column(
                        txn,
                        interner,
                        type_name,
                        &format!("{}/__tag", field.name),
                        &entity_to_idx,
                        entity_ids.len(),
                        &mut columns,
                    )?;

                    // Load variant sub-fields
                    if let Some(enum_def) = schema.get_enum(enum_name) {
                        for variant in &enum_def.variants {
                            for vf in &variant.fields {
                                load_field_into_column(
                                    txn,
                                    interner,
                                    type_name,
                                    &format!("{}.{}/{}", field.name, variant.name, vf.name),
                                    &entity_to_idx,
                                    entity_ids.len(),
                                    &mut columns,
                                )?;
                            }
                        }
                    }
                }
                _ => {
                    load_field_into_column(
                        txn,
                        interner,
                        type_name,
                        &field.name,
                        &entity_to_idx,
                        entity_ids.len(),
                        &mut columns,
                    )?;
                }
            }
        }
    }

    Ok(TypeData {
        entity_ids,
        columns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{Result as StorageResult, ReadOps};

    /// Stub `ReadOps` that returns no rows. Lets us drive
    /// `ensure_type_loaded` without standing up a real storage backend —
    /// every loaded type ends up with zero entities, which is fine for
    /// exercising the cache's LRU machinery.
    struct EmptyOps;

    impl ReadOps for EmptyOps {
        fn get(&self, _key: &[u8]) -> StorageResult<Option<Vec<u8>>> {
            Ok(None)
        }
        fn scan(&self, _start: &[u8], _end: &[u8]) -> StorageResult<Vec<(Vec<u8>, Vec<u8>)>> {
            Ok(Vec::new())
        }
        fn scan_foreach(
            &self,
            _start: &[u8],
            _end: &[u8],
            _f: &mut dyn FnMut(&[u8], &[u8]) -> bool,
        ) -> StorageResult<()> {
            Ok(())
        }
    }

    fn empty_schema() -> SchemaRegistry {
        SchemaRegistry::new()
    }

    fn empty_interner() -> AttrInterner {
        AttrInterner::new()
    }

    #[test]
    fn disabled_policy_never_inserts() {
        let cache = QueryCache::with_policy(CachePolicy::None);
        let schema = empty_schema();
        let interner = empty_interner();
        let result = cache
            .ensure_type_loaded(&EmptyOps, &interner, "User", &schema)
            .unwrap();
        assert!(result.is_none(), "disabled cache must return None");
        assert_eq!(cache.cached_type_count(), 0);
    }

    #[test]
    fn unbounded_policy_grows_without_eviction() {
        let cache = QueryCache::with_policy(CachePolicy::Unbounded);
        let schema = empty_schema();
        let interner = empty_interner();
        for name in ["A", "B", "C", "D"] {
            cache
                .ensure_type_loaded(&EmptyOps, &interner, name, &schema)
                .unwrap();
        }
        assert_eq!(cache.cached_type_count(), 4);
    }

    #[test]
    fn bounded_evicts_lru_when_full() {
        let cache = QueryCache::with_policy(CachePolicy::Bounded { max_types: 2 });
        let schema = empty_schema();
        let interner = empty_interner();

        cache.ensure_type_loaded(&EmptyOps, &interner, "A", &schema).unwrap();
        cache.ensure_type_loaded(&EmptyOps, &interner, "B", &schema).unwrap();
        assert_eq!(cache.cached_type_count(), 2);

        cache.ensure_type_loaded(&EmptyOps, &interner, "A", &schema).unwrap();
        cache.ensure_type_loaded(&EmptyOps, &interner, "C", &schema).unwrap();
        assert_eq!(cache.cached_type_count(), 2);

        cache.ensure_type_loaded(&EmptyOps, &interner, "B", &schema).unwrap();
        assert_eq!(cache.cached_type_count(), 2);
    }

    #[test]
    fn bounded_zero_returns_none() {
        let cache = QueryCache::with_policy(CachePolicy::Bounded { max_types: 0 });
        let schema = empty_schema();
        let interner = empty_interner();
        let result = cache
            .ensure_type_loaded(&EmptyOps, &interner, "User", &schema)
            .unwrap();
        assert!(result.is_none());
        assert_eq!(cache.cached_type_count(), 0);
    }

    #[test]
    fn invalidate_type_removes_entry() {
        let cache = QueryCache::with_policy(CachePolicy::Unbounded);
        let schema = empty_schema();
        let interner = empty_interner();
        cache
            .ensure_type_loaded(&EmptyOps, &interner, "User", &schema)
            .unwrap();
        assert_eq!(cache.cached_type_count(), 1);
        cache.invalidate_type("User");
        assert_eq!(cache.cached_type_count(), 0);
    }
}

/// Scan CURRENT_AEVT for a single attribute and populate one column of
/// the type's columnar storage. The column is a `Vec<Option<Value>>`
/// of length `num_entities`, parallel to the type's `entity_ids` vec.
/// Missing values stay `None`.
fn load_field_into_column(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    type_name: &str,
    field_name: &str,
    entity_to_idx: &HashMap<EntityId, usize>,
    num_entities: usize,
    columns: &mut HashMap<String, Column>,
) -> Result<(), String> {
    let attr = format!("{}/{}", type_name, field_name);
    let attr_id = match interner.lookup(&attr) {
        Some(id) => id,
        None => return Ok(()),
    };
    let prefix = index::current_aevt_attr_prefix(attr_id);
    let end = index::prefix_end(&prefix);

    let mut col: Vec<Option<Value>> = vec![None; num_entities];
    let mut any = false;
    txn.scan_foreach(&prefix, &end, &mut |key, value| {
        let eid = index::current_aevt_entity_at(key);
        if let Some(&idx) = entity_to_idx.get(&eid) {
            if let Some(val) = index::decode_current_value(value) {
                col[idx] = Some(val);
                any = true;
            }
        }
        true
    })
    .map_err(|e| e.to_string())?;

    // Only insert if at least one entity has the field; saves a HashMap
    // entry (and a Vec allocation) for never-set optional fields.
    if any {
        columns.insert(field_name.to_string(), Arc::new(col));
    }
    Ok(())
}
