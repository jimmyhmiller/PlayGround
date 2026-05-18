use std::any::Any;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::{Mutex, RwLock};

use byteorder::{BigEndian, ByteOrder};

use crate::cache::{CachePolicy, QueryCache};
use crate::datom::{EntityId, TxId, Value};
use crate::index;
use crate::intern::AttrInterner;
use crate::query::executor::{self, QueryResult};
use crate::query::planner;
pub use crate::query::planner::QueryPlan;
use crate::query::{Pattern, Query};
use crate::schema::{EntityTypeDef, EnumTypeDef, FieldDef, FieldType, SchemaRegistry};
use crate::storage::{StorageBackend, StorageError, TxnOps};
use crate::tx::{self, TxOp};

const TX_COUNTER_KEY: &str = "tx_counter";
const ENTITY_COUNTER_KEY: &str = "entity_counter";

// Reserved attribute names used by the engine itself.
const SCHEMA_ENUM_ATTR: &str = "__schema_enum";
const SCHEMA_TYPE_ATTR: &str = "__schema_type";

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

/// Config for the auto-batching writer thread.
#[derive(Debug, Clone, Copy)]
pub struct GroupCommitConfig {
    /// Maximum number of `transact()` calls coalesced into one group.
    /// Larger reduces per-tx overhead but increases tail latency, since
    /// the first request may wait while up to `max_batch_size - 1`
    /// others trickle in.
    pub max_batch_size: usize,
    /// Maximum time the writer thread waits to accumulate more requests
    /// before committing a partial batch. Use `Duration::ZERO` to fire
    /// immediately whenever the channel goes quiet.
    pub max_window: Duration,
}

impl Default for GroupCommitConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_window: Duration::from_millis(1),
        }
    }
}

/// Tunable options for opening a `Database`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DatabaseOptions {
    /// Policy for the in-memory per-type query cache.
    pub cache: CachePolicy,
    /// When `Some`, spawn a background writer thread that auto-batches
    /// concurrent `transact()` calls into one atomic group commit per
    /// group. `transact_many` always goes synchronous and bypasses the
    /// writer thread.
    pub group_commit: Option<GroupCommitConfig>,
    /// EXPERIMENTAL: allow concurrent writers. When true, suppresses the
    /// group-commit writer thread and skips the internal write lock so
    /// multiple threads can call `transact()` in parallel. tx_id and
    /// entity_id allocation moves to atomic `fetch_add`, and RocksDB's
    /// internal WAL coalesces the resulting concurrent batches.
    ///
    /// Tradeoffs: tx_id no longer strictly equals commit order (so an
    /// asOf(N) query right after the in-flight tx that owns N may briefly
    /// not see it), and unique-constraint checks against in-flight
    /// concurrent siblings are best-effort (each writer sees only the
    /// snapshot taken before its own batch began). Sequential and
    /// single-threaded behavior is unchanged.
    pub parallel_writes: bool,
}

pub struct Database {
    storage: Arc<dyn StorageBackend>,
    schema: Arc<RwLock<SchemaRegistry>>,
    plan_cache: Arc<RwLock<HashMap<PlanCacheKey, Arc<QueryPlan>>>>,
    query_cache: Arc<QueryCache>,
    /// In-memory mirror of the persisted attribute-id table. Encoded
    /// into every index key in place of the attribute name. Initialized
    /// from storage at open and grown atomically with each write that
    /// uses a new attribute.
    attr_interner: Arc<AttrInterner>,

    /// Monotonic transaction-id counter. Last committed tx_id. Allocations
    /// are `current + 1` while holding `write_lock`. Initialized from
    /// storage at open and rewritten into every batch so a crash restores
    /// the right starting point.
    tx_counter: Arc<AtomicU64>,
    /// Monotonic entity-id counter. Same lifecycle as `tx_counter`.
    entity_counter: Arc<AtomicU64>,
    /// Serializes all writers. The combination of "hold this, peek
    /// atomics, validate, write batch, advance atomics" preserves
    /// tx_id-equals-commit-order, which is what asOf semantics rely on.
    /// Unused when `parallel_writes` is true.
    write_lock: Arc<Mutex<()>>,

    /// Background writer when group commit is enabled. `None` means
    /// `transact()` runs synchronously on the caller thread.
    writer: Option<WriterHandle>,

    /// When true, callers of `transact()` skip both the writer thread and
    /// the internal `write_lock`. Counter allocation is via atomic
    /// fetch_add. Multiple writers can run their batches in parallel; the
    /// RocksDB WAL is the only remaining serialization point.
    parallel_writes: bool,
}

/// Request submitted to the writer thread.
struct WriteRequest {
    ops: Vec<TxOp>,
    reply: SyncSender<Result<tx::TransactionResult>>,
}

/// Owning handle for the writer thread + its submission channel.
///
/// On drop, the channel is closed first so the writer sees a
/// disconnected receive and exits its loop; then we join the thread.
struct WriterHandle {
    tx: Option<Sender<WriteRequest>>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for WriterHandle {
    fn drop(&mut self) {
        // Closing the sender side signals the writer to shut down on its
        // next blocking recv.
        self.tx.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Database {
    /// Open or create a database with the given storage backend and
    /// default options.
    pub fn open(storage: Arc<dyn StorageBackend>) -> Result<Self> {
        Self::open_with(storage, DatabaseOptions::default())
    }

    /// Open or create a database with explicit options.
    pub fn open_with(
        storage: Arc<dyn StorageBackend>,
        opts: DatabaseOptions,
    ) -> Result<Self> {
        // Initialize the in-memory counters from their persisted values.
        // Every successful batch rewrites these keys so a crash always
        // recovers the correct last-committed values.
        let tx_counter_init = read_persisted_counter(&*storage, TX_COUNTER_KEY)?;
        let entity_counter_init = read_persisted_counter(&*storage, ENTITY_COUNTER_KEY)?;

        // Build the attribute interner and hydrate from the persisted
        // `attr:<name>` mapping table. Any datom read or written below
        // must go through this — it owns the `String → AttrId` and
        // `AttrId → String` mappings for the lifetime of the DB.
        let attr_interner = Arc::new(AttrInterner::new());
        attr_interner.load_from_storage(&*storage)?;

        let mut db = Self {
            storage,
            schema: Arc::new(RwLock::new(SchemaRegistry::new())),
            plan_cache: Arc::new(RwLock::new(HashMap::new())),
            query_cache: Arc::new(QueryCache::with_policy(opts.cache)),
            attr_interner,
            tx_counter: Arc::new(AtomicU64::new(tx_counter_init)),
            entity_counter: Arc::new(AtomicU64::new(entity_counter_init)),
            write_lock: Arc::new(Mutex::new(())),
            writer: None,
            parallel_writes: opts.parallel_writes,
        };

        // Load schema from storage
        db.load_schema()?;

        // Spawn the writer thread once everything else is set up.
        // `transact()` checks `self.writer` per-call, so the synchronous
        // path stays unused while the thread is active. Parallel-writes
        // mode suppresses the writer entirely — the whole point is to let
        // multiple `transact()` callers run their batches concurrently.
        if let Some(config) = opts.group_commit {
            if !opts.parallel_writes {
                db.writer = Some(spawn_writer(
                    config,
                    db.storage.clone(),
                    db.schema.clone(),
                    db.query_cache.clone(),
                    db.attr_interner.clone(),
                    db.tx_counter.clone(),
                    db.entity_counter.clone(),
                    db.write_lock.clone(),
                ));
            }
        }

        Ok(db)
    }

    /// Load schema definitions from stored datoms.
    fn load_schema(&self) -> Result<()> {
        // If neither schema attribute has ever been written, there's
        // nothing to load — the interner will not have assigned IDs for
        // them and the scan would match no data anyway.
        let enum_attr_id = self.attr_interner.lookup(SCHEMA_ENUM_ATTR);
        let type_attr_id = self.attr_interner.lookup(SCHEMA_TYPE_ATTR);
        if enum_attr_id.is_none() && type_attr_id.is_none() {
            return Ok(());
        }

        let result = self
            .storage
            .execute_read(Box::new(move |snap| {
                let enum_entries = if let Some(id) = enum_attr_id {
                    let prefix = index::aevt_attr_prefix(id);
                    let end = index::prefix_end(&prefix);
                    snap.scan(&prefix, &end)?
                } else {
                    Vec::new()
                };
                let type_entries = if let Some(id) = type_attr_id {
                    let prefix = index::aevt_attr_prefix(id);
                    let end = index::prefix_end(&prefix);
                    snap.scan(&prefix, &end)?
                } else {
                    Vec::new()
                };
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

        // Serialize writers; allocate ids while holding the lock so
        // tx_id ordering matches commit ordering.
        let _guard = self.write_lock.lock();
        let tx_id = self.tx_counter.load(Ordering::Acquire) + 1;
        let entity_id = self.entity_counter.load(Ordering::Acquire) + 1;

        let interner = self.attr_interner.clone();
        self.storage
            .execute_batch(Box::new(move |txn| {
                let tx_key = index::meta_key(TX_COUNTER_KEY);
                let ec_key = index::meta_key(ENTITY_COUNTER_KEY);
                txn.put(tx_key, encode_u64(tx_id))?;
                txn.put(ec_key, encode_u64(entity_id))?;

                // Intern the schema attribute. The first call ever
                // writes the `attr:__schema_enum` mapping into this
                // batch, atomically with the datom that uses it.
                let attr_id = interner.intern(txn, SCHEMA_ENUM_ATTR)?;
                let value = Value::String(json.into());

                for (key, value) in index::encode_datom(entity_id, attr_id, &value, tx_id, true) {
                    txn.put(key, value)?;
                }

                store_tx_timestamp(txn, tx_id, timestamp_ms)?;

                Ok(Box::new(()) as Box<dyn Any + Send>)
            }))?;

        // Batch applied — advance the in-memory counters.
        self.tx_counter.store(tx_id, Ordering::Release);
        self.entity_counter.store(entity_id, Ordering::Release);
        drop(_guard);

        schema.register_enum(enum_def);
        self.plan_cache.write().clear();
        self.query_cache.invalidate_all();

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

        let _guard = self.write_lock.lock();
        let tx_id = self.tx_counter.load(Ordering::Acquire) + 1;
        let entity_id = self.entity_counter.load(Ordering::Acquire) + 1;

        let interner = self.attr_interner.clone();
        self.storage
            .execute_batch(Box::new(move |txn| {
                let tx_key = index::meta_key(TX_COUNTER_KEY);
                let ec_key = index::meta_key(ENTITY_COUNTER_KEY);
                txn.put(tx_key, encode_u64(tx_id))?;
                txn.put(ec_key, encode_u64(entity_id))?;

                let attr_id = interner.intern(txn, SCHEMA_TYPE_ATTR)?;
                let value = Value::String(json.into());

                for (key, value) in index::encode_datom(entity_id, attr_id, &value, tx_id, true) {
                    txn.put(key, value)?;
                }

                store_tx_timestamp(txn, tx_id, timestamp_ms)?;

                Ok(Box::new(()) as Box<dyn Any + Send>)
            }))?;

        self.tx_counter.store(tx_id, Ordering::Release);
        self.entity_counter.store(entity_id, Ordering::Release);
        drop(_guard);

        schema.register(type_def);
        self.plan_cache.write().clear();
        self.query_cache.invalidate_all();

        Ok(tx_id)
    }

    /// Execute a transaction.
    ///
    /// When `DatabaseOptions::group_commit` is enabled, the request is
    /// handed to the background writer thread, which auto-batches
    /// concurrent calls into one atomic group commit. Otherwise the
    /// transaction is applied synchronously on the caller thread.
    pub fn transact(&self, ops: Vec<TxOp>) -> Result<tx::TransactionResult> {
        // Auto-batching path: hand off to the writer thread and block on
        // its reply. Other callers' requests may be coalesced with ours.
        if let Some(writer) = &self.writer {
            let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
            let sender = writer
                .tx
                .as_ref()
                .expect("writer handle present but channel taken");
            sender
                .send(WriteRequest { ops, reply: reply_tx })
                .map_err(|_| {
                    DbError::Storage(StorageError::Backend(
                        "group-commit writer thread terminated".to_string(),
                    ))
                })?;
            return reply_rx.recv().map_err(|_| {
                DbError::Storage(StorageError::Backend(
                    "group-commit writer dropped reply channel".to_string(),
                ))
            })?;
        }

        // Synchronous path: 1-element group commit. Goes through exactly
        // the same code as `transact_many` so behavior matches.
        let mut results = do_group_commit(
            &*self.storage,
            &self.schema,
            &self.query_cache,
            &self.attr_interner,
            &self.tx_counter,
            &self.entity_counter,
            &self.write_lock,
            self.parallel_writes,
            vec![ops],
        )?;
        let inner = results.pop().expect("1-element group must yield 1 result");
        inner.map_err(DbError::from)
    }

    /// Execute multiple transactions as one group commit.
    ///
    /// All transactions in the group share a single underlying RocksDB
    /// `WriteBatch` and therefore a single WAL append and (under
    /// `Durability::Sync`) a single fsync. Each transaction is validated
    /// independently against the snapshot plus the running overlay, so
    /// later transactions in the group can see earlier ones' writes
    /// (e.g. for unique-constraint enforcement).
    ///
    /// A failure in one transaction rolls back only its own writes —
    /// siblings still commit. The returned vector has one entry per
    /// input `ops_list`, in order.
    ///
    /// tx_id ordering matches input order: the first ops_list gets the
    /// smallest tx_id in the group, the last gets the largest. This
    /// preserves Datomic-style asOf semantics.
    pub fn transact_many(
        &self,
        ops_lists: Vec<Vec<TxOp>>,
    ) -> Result<Vec<std::result::Result<tx::TransactionResult, tx::TxError>>> {
        do_group_commit(
            &*self.storage,
            &self.schema,
            &self.query_cache,
            &self.attr_interner,
            &self.tx_counter,
            &self.entity_counter,
            &self.write_lock,
            self.parallel_writes,
            ops_lists,
        )
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
            let cache = self.query_cache.clone();
            let interner = self.attr_interner.clone();
            let result = self
                .storage
                .execute_read(Box::new(move |snap| {
                    let qr = executor::execute_plan(snap, &plan, &schema, &cache, &interner)
                        .map_err(|e| StorageError::Backend(e))?;
                    Ok(Box::new(qr) as Box<dyn Any + Send>)
                }))?;
            return Ok(*result.downcast::<QueryResult>().expect("wrong type"));
        }

        let cache = self.query_cache.clone();
        let interner = self.attr_interner.clone();
        let result = self
            .storage
            .execute_read(Box::new(move |snap| {
                let plan = planner::plan_query(snap, &query, &schema, &interner)
                    .map_err(|e| StorageError::Backend(e))?;

                let qr = executor::execute_plan(snap, &plan, &schema, &cache, &interner)
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

        let interner = self.attr_interner.clone();
        let result = self
            .storage
            .execute_read(Box::new(move |snap| {
                let plan = planner::plan_query(snap, &query, &schema, &interner)
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
        let interner = self.attr_interner.clone();
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
                    if let Some(decoded) = index::decode_datom_from_eavt(key) {
                        if let Some(datom) = interner.resolve(decoded) {
                            attr_datoms
                                .entry(datom.attribute.clone())
                                .or_default()
                                .push(datom);
                        }
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
        let interner = self.attr_interner.clone();
        let result = self
            .storage
            .execute_read(Box::new(move |snap| {
                let prefix = vec![index::EAVT_PREFIX];
                let end = index::prefix_end(&prefix);
                let entries = snap.scan(&prefix, &end)?;
                let datoms: Vec<crate::datom::Datom> = entries
                    .iter()
                    .filter_map(|(k, _)| {
                        index::decode_datom_from_eavt(k).and_then(|d| interner.resolve(d))
                    })
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
        let interner = self.attr_interner.clone();
        let result = self
            .storage
            .execute_read(Box::new(move |snap| {
                let prefix = index::eavt_entity_prefix(entity);
                let end = index::prefix_end(&prefix);
                let entries = snap.scan(&prefix, &end)?;
                let datoms: Vec<crate::datom::Datom> = entries
                    .iter()
                    .filter_map(|(k, _)| {
                        index::decode_datom_from_eavt(k).and_then(|d| interner.resolve(d))
                    })
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

/// Shared group-commit kernel used by both `transact_many` and the
/// background writer thread. Holds the write lock, allocates sequential
/// tx_ids, builds one callback per request, runs them through
/// `execute_group_batch`, then advances the in-memory counters and
/// invalidates per-tx caches.
fn do_group_commit(
    storage: &dyn StorageBackend,
    schema_lock: &RwLock<SchemaRegistry>,
    query_cache: &QueryCache,
    attr_interner: &Arc<AttrInterner>,
    tx_counter: &AtomicU64,
    entity_counter: &AtomicU64,
    write_lock: &Mutex<()>,
    parallel_writes: bool,
    ops_lists: Vec<Vec<TxOp>>,
) -> Result<Vec<std::result::Result<tx::TransactionResult, tx::TxError>>> {
    if ops_lists.is_empty() {
        return Ok(Vec::new());
    }

    let schema = schema_lock.read().clone();
    let timestamp_ms = now_millis();

    let mut all_modified_types: Vec<Vec<String>> = Vec::with_capacity(ops_lists.len());
    for ops in &ops_lists {
        let modified: Vec<String> = ops
            .iter()
            .map(|op| match op {
                TxOp::Assert { entity_type, .. } => entity_type.clone(),
                TxOp::Retract { entity_type, .. } => entity_type.clone(),
                TxOp::RetractEntity { entity_type, .. } => entity_type.clone(),
            })
            .collect();
        all_modified_types.push(modified);
    }

    // Counter allocation: under the legacy locked path, we peek atomics,
    // execute, then write them back. Under parallel_writes, allocation is
    // atomic fetch_add up front so concurrent batches get disjoint ranges
    // without ever blocking on each other. We have to pre-count the new
    // entities each batch needs so we can reserve a contiguous range per
    // call to process_transaction (it expects a monotonic counter).
    let _maybe_guard = if parallel_writes {
        None
    } else {
        Some(write_lock.lock())
    };

    let num_batches = ops_lists.len() as u64;
    let (first_tx_id, starting_entity) = if parallel_writes {
        let first_tx = tx_counter.fetch_add(num_batches, Ordering::AcqRel) + 1;
        let entities_needed: u64 = ops_lists.iter().map(|ops| count_new_entities(ops)).sum();
        let start_entity = entity_counter.fetch_add(entities_needed, Ordering::AcqRel);
        (first_tx, start_entity)
    } else {
        let first_tx = tx_counter.load(Ordering::Acquire) + 1;
        let start_entity = entity_counter.load(Ordering::Acquire);
        (first_tx, start_entity)
    };

    let shared_entity_counter = Arc::new(Mutex::new(starting_entity));

    let mut callbacks: Vec<crate::storage::GroupTxnCallback> =
        Vec::with_capacity(ops_lists.len());
    let mut tx_ids: Vec<TxId> = Vec::with_capacity(ops_lists.len());

    let mut next_tx_id = first_tx_id;
    for ops in ops_lists {
        let tx_id = next_tx_id;
        next_tx_id += 1;
        tx_ids.push(tx_id);

        let schema_clone = schema.clone();
        let ec = shared_entity_counter.clone();
        let interner = attr_interner.clone();

        let cb: crate::storage::GroupTxnCallback = Box::new(move |txn| {
            let tx_key = index::meta_key(TX_COUNTER_KEY);
            let ec_key = index::meta_key(ENTITY_COUNTER_KEY);
            txn.put(tx_key, encode_u64(tx_id))?;

            let mut entity_counter_local = *ec.lock();
            let mut tx_result = tx::process_transaction(
                txn,
                &interner,
                &schema_clone,
                tx_id,
                &mut entity_counter_local,
                ops,
            )
            .map_err(|e| StorageError::Backend(e.to_string()))?;
            txn.put(ec_key, encode_u64(entity_counter_local))?;
            *ec.lock() = entity_counter_local;

            store_tx_timestamp(txn, tx_id, timestamp_ms)?;
            tx_result.timestamp_ms = timestamp_ms;

            Ok(Box::new(tx_result) as Box<dyn Any + Send>)
        });
        callbacks.push(cb);
    }

    let raw_results = storage.execute_group_batch(callbacks)?;

    let final_entity = *shared_entity_counter.lock();
    let mut converted: Vec<std::result::Result<tx::TransactionResult, tx::TxError>> =
        Vec::with_capacity(raw_results.len());
    let mut highest_committed_tx = if parallel_writes {
        0
    } else {
        tx_counter.load(Ordering::Acquire)
    };

    for (i, (raw, tx_id)) in raw_results.into_iter().zip(tx_ids.iter().copied()).enumerate() {
        match raw {
            Ok(boxed) => {
                let tx_result = *boxed
                    .downcast::<tx::TransactionResult>()
                    .expect("group callback returned wrong type");
                highest_committed_tx = highest_committed_tx.max(tx_id);
                // If the transaction was a pure-add (all Asserts on
                // fresh entities, no retracts, no value updates),
                // incrementally extend the cached TypeData instead of
                // invalidating it. This is the fast path for the Mixed
                // workload: 80% reads stay cache-warm even while writers
                // are committing new entities every 5th operation.
                //
                // The fallback (invalidate_type) still fires for any
                // transaction that includes a Retract, RetractEntity, or
                // an Assert that updates an existing entity's field —
                // those mutate existing rows and the cache can't be
                // patched in-place safely without a full reload.
                if tx_result.cache_appends.is_empty() {
                    for ty in &all_modified_types[i] {
                        query_cache.invalidate_type(ty);
                    }
                } else {
                    for app in &tx_result.cache_appends {
                        query_cache.append_entity(
                            &app.entity_type,
                            app.entity_id,
                            &app.values,
                        );
                    }
                }
                converted.push(Ok(tx_result));
            }
            Err(e) => {
                converted.push(Err(tx::TxError::Storage(e)));
            }
        }
    }

    // In parallel mode the counters were advanced up front via
    // fetch_add — even if some inner callbacks errored, the ids they
    // reserved are still burnt (they're recorded as failed in the result
    // vector). Skipping the writeback also avoids racing other concurrent
    // writers' fetch_adds.
    if !parallel_writes {
        tx_counter.store(highest_committed_tx, Ordering::Release);
        entity_counter.store(final_entity, Ordering::Release);
    }
    drop(_maybe_guard);

    Ok(converted)
}

/// Count the number of new entities (Assert with `entity = None`) across
/// a batch. Used to reserve a contiguous entity_id range up front under
/// `parallel_writes`.
fn count_new_entities(ops: &[TxOp]) -> u64 {
    let mut count: u64 = 0;
    for op in ops {
        if let TxOp::Assert { entity: None, .. } = op {
            count += 1;
        }
    }
    count
}

/// Spawn the background writer thread for auto-batched group commit.
fn spawn_writer(
    config: GroupCommitConfig,
    storage: Arc<dyn StorageBackend>,
    schema: Arc<RwLock<SchemaRegistry>>,
    query_cache: Arc<QueryCache>,
    attr_interner: Arc<AttrInterner>,
    tx_counter: Arc<AtomicU64>,
    entity_counter: Arc<AtomicU64>,
    write_lock: Arc<Mutex<()>>,
) -> WriterHandle {
    let (tx, rx) = std::sync::mpsc::channel::<WriteRequest>();
    let thread = std::thread::Builder::new()
        .name("datalog-db-writer".to_string())
        .spawn(move || {
            writer_loop(
                rx,
                config,
                storage,
                schema,
                query_cache,
                attr_interner,
                tx_counter,
                entity_counter,
                write_lock,
            );
        })
        .expect("failed to spawn writer thread");
    WriterHandle {
        tx: Some(tx),
        thread: Some(thread),
    }
}

/// Body of the writer thread. Blocks on the first request, then greedily
/// drains additional ones up to `max_batch_size` or `max_window`,
/// whichever comes first. Each accumulated batch becomes one group
/// commit via `do_group_commit`. Per-request replies go out through
/// each request's reply channel.
fn writer_loop(
    rx: Receiver<WriteRequest>,
    config: GroupCommitConfig,
    storage: Arc<dyn StorageBackend>,
    schema: Arc<RwLock<SchemaRegistry>>,
    query_cache: Arc<QueryCache>,
    attr_interner: Arc<AttrInterner>,
    tx_counter: Arc<AtomicU64>,
    entity_counter: Arc<AtomicU64>,
    write_lock: Arc<Mutex<()>>,
) {
    loop {
        // Block until at least one request arrives, or the channel
        // closes (Database dropped → shutdown).
        let first = match rx.recv() {
            Ok(r) => r,
            Err(_) => return,
        };

        let mut batch: Vec<WriteRequest> = vec![first];

        // Greedy drain: pick up anything already queued without
        // waiting. This is the natural batching mechanism for
        // concurrent workloads — while we were committing the previous
        // batch, other writers piled requests up in the channel.
        while batch.len() < config.max_batch_size {
            match rx.try_recv() {
                Ok(r) => batch.push(r),
                Err(_) => break,
            }
        }

        // Optional window: if max_window > 0, wait a bit longer for
        // more requests to arrive. Trades latency for batching when the
        // workload is bursty but the channel happens to be momentarily
        // empty. `max_window = ZERO` skips the wait entirely (default
        // for low-contention single-threaded scenarios).
        if !config.max_window.is_zero() {
            let deadline = Instant::now() + config.max_window;
            while batch.len() < config.max_batch_size {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }
                match rx.recv_timeout(remaining) {
                    Ok(r) => batch.push(r),
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
        }

        // Build the group commit input. We have to take ownership of
        // `ops` (Vec<TxOp> isn't Clone-cheap and we want to move it).
        let mut ops_lists: Vec<Vec<TxOp>> = Vec::with_capacity(batch.len());
        let mut replies: Vec<SyncSender<Result<tx::TransactionResult>>> =
            Vec::with_capacity(batch.len());
        for req in batch {
            ops_lists.push(req.ops);
            replies.push(req.reply);
        }

        let group_result = do_group_commit(
            &*storage,
            &schema,
            &query_cache,
            &attr_interner,
            &tx_counter,
            &entity_counter,
            &write_lock,
            false,
            ops_lists,
        );

        match group_result {
            Ok(per_request) => {
                for (result, reply) in per_request.into_iter().zip(replies) {
                    let mapped = result.map_err(DbError::from);
                    let _ = reply.send(mapped);
                }
            }
            Err(e) => {
                // Whole batch failed at the backend level. Every caller
                // gets the same error so no one is left waiting. DbError
                // doesn't impl Clone, so stringify once and rebuild per
                // reply — this only fires on catastrophic backend
                // failures (`execute_group_batch` returning Err).
                let msg = e.to_string();
                for reply in replies {
                    let _ = reply.send(Err(DbError::Storage(StorageError::Backend(
                        msg.clone(),
                    ))));
                }
            }
        }
    }
}

/// Read the persisted value of a meta-key counter (`tx_counter` or
/// `entity_counter`). Returns 0 if the key doesn't exist yet — first
/// transaction will allocate id 1.
fn read_persisted_counter(
    storage: &dyn StorageBackend,
    key: &'static str,
) -> Result<u64> {
    let result = storage.execute_read(Box::new(move |snap| {
        let k = index::meta_key(key);
        let v = snap.get(&k)?;
        let n = match v {
            Some(b) if b.len() == 8 => BigEndian::read_u64(&b),
            _ => 0,
        };
        Ok(Box::new(n) as Box<dyn Any + Send>)
    }))?;
    Ok(*result.downcast::<u64>().expect("wrong type"))
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
