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
use crate::query::{AggFunc, Clause, FindElem, Pattern, PredOp, Query};
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

/// Cache key derived from query shape AND its literal values.
///
/// Literal values MUST be part of the key: a cached `QueryPlan` bakes the
/// concrete predicate/constant values into its scan strategy and execution
/// re-uses the plan verbatim (only `as_of` is refreshed). If two queries
/// that differ only in a literal (`qty > 10` vs `qty > 200`) shared a key,
/// the second would silently return the first's results.
#[derive(Clone, PartialEq, Eq)]
struct PlanCacheKey {
    hash: u64,
}

/// Feed a `Value` into the hasher in a stable, collision-resistant way.
/// `f64` is hashed by its raw bits so distinct floats key distinct plans.
fn hash_value<H: Hasher>(v: &Value, h: &mut H) {
    std::mem::discriminant(v).hash(h);
    match v {
        Value::String(s) => s.hash(h),
        Value::I64(n) => n.hash(h),
        Value::F64(f) => f.to_bits().hash(h),
        Value::Bool(b) => b.hash(h),
        Value::Ref(id) => id.hash(h),
        Value::Bytes(bytes) => bytes.hash(h),
        Value::List(items) => {
            items.len().hash(h);
            for item in items {
                hash_value(item, h);
            }
        }
        Value::Vector(v) => {
            v.len().hash(h);
            for f in v {
                f.to_bits().hash(h);
            }
        }
        // Enum/Null never appear as query-pattern literals, but hash the
        // discriminant (already done above) so the key stays well-defined.
        Value::Enum(_) | Value::Null => {}
    }
}

/// Feed a pattern's full shape AND values into the hasher.
fn hash_pattern<H: Hasher>(pattern: &Pattern, h: &mut H) {
    std::mem::discriminant(pattern).hash(h);
    match pattern {
        Pattern::Variable(v) => v.hash(h),
        Pattern::Constant(value) => hash_value(value, h),
        Pattern::Predicate { op, value } => {
            std::mem::discriminant(op).hash(h);
            hash_value(value, h);
        }
        Pattern::BoundPredicate { var, op, value } => {
            var.hash(h);
            std::mem::discriminant(op).hash(h);
            hash_value(value, h);
        }
        Pattern::EnumMatch { variant, field_patterns } => {
            hash_pattern(variant, h);
            for (name, pat) in field_patterns {
                name.hash(h);
                hash_pattern(pat, h);
            }
        }
        Pattern::Near { query, k, metric, score_var } => {
            query.len().hash(h);
            for f in query {
                f.to_bits().hash(h);
            }
            k.hash(h);
            std::mem::discriminant(metric).hash(h);
            score_var.hash(h);
        }
        Pattern::Search { query, k, score_var } => {
            query.hash(h);
            k.hash(h);
            score_var.hash(h);
        }
    }
}

/// Hash a where clause tree's shape AND literal values into the plan key.
fn hash_clause<H: Hasher>(clause: &Clause, h: &mut H) {
    std::mem::discriminant(clause).hash(h);
    match clause {
        Clause::Pattern(wc) => {
            wc.entity_type.hash(h);
            wc.bind.hash(h);
            for (field_name, pattern) in &wc.field_patterns {
                field_name.hash(h);
                hash_pattern(pattern, h);
            }
        }
        Clause::And(children) | Clause::Or(children) => {
            children.len().hash(h);
            for c in children {
                hash_clause(c, h);
            }
        }
        Clause::Not(child) => hash_clause(child, h),
    }
}

impl PlanCacheKey {
    fn from_query(query: &Query) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // Hash find variables
        query.find.hash(&mut hasher);
        // Hash whether as_of is present (not the value)
        query.as_of.is_some().hash(&mut hasher);
        // Hash clause shapes AND their literal values
        hash_clause(&query.where_clause, &mut hasher);
        PlanCacheKey {
            hash: hasher.finish(),
        }
    }
}

/// Coerce a numeric literal so its runtime type matches a field's declared
/// numeric type. Returns `Some((op, value))` with the adjusted operator and
/// value, or `None` to leave the literal unchanged.
///
/// The AVET index keys values by their exact runtime type, so a range scan
/// for an `I64` bound never sees `F64`-stored values (and vice versa). Worse,
/// the cross-type post-filter would silently drop everything. We normalize the
/// literal to the field's type up front. For an integer field compared against
/// a fractional float, we pick the operator-preserving integer bound that is
/// exactly equivalent over the integers (e.g. `x > 10.5` ⇔ `x > 10`).
fn coerce_numeric_literal(
    field_type: &FieldType,
    op: Option<&PredOp>,
    value: &Value,
) -> Option<(Option<PredOp>, Value)> {
    match (field_type, value) {
        // Float field, integer literal → exact float.
        (FieldType::F64, Value::I64(n)) => Some((op.cloned(), Value::F64(*n as f64))),

        // Integer field, float literal.
        (FieldType::I64, Value::F64(f)) => {
            let f = *f;
            match op {
                // Equality: only an integral float can equal an integer.
                // A fractional literal is left as-is so the exact scan finds
                // nothing (the correct empty result).
                None => (f.fract() == 0.0).then(|| (None, Value::I64(f as i64))),
                // Range bounds: round toward the operator-preserving integer.
                //   x >  f ⇔ x >  floor(f)      x >= f ⇔ x >= ceil(f)
                //   x <  f ⇔ x <  ceil(f)       x <= f ⇔ x <= floor(f)
                Some(PredOp::Gt) => Some((Some(PredOp::Gt), Value::I64(f.floor() as i64))),
                Some(PredOp::Gte) => Some((Some(PredOp::Gte), Value::I64(f.ceil() as i64))),
                Some(PredOp::Lt) => Some((Some(PredOp::Lt), Value::I64(f.ceil() as i64))),
                Some(PredOp::Lte) => Some((Some(PredOp::Lte), Value::I64(f.floor() as i64))),
                // `!=` against a fractional float is true for every integer;
                // leave it for the cross-type evaluator (which returns true).
                Some(PredOp::Ne) => {
                    (f.fract() == 0.0).then(|| (Some(PredOp::Ne), Value::I64(f as i64)))
                }
                // String-search ops don't apply to a numeric field; no
                // coercion (they're filtered out before reaching an index).
                Some(PredOp::Contains | PredOp::StartsWith | PredOp::EndsWith) => None,
            }
        }
        _ => None,
    }
}

/// Running state for one aggregate within one group.
enum AggState {
    Count(i64),
    /// Distinct values seen so far, keyed by the stable group-key encoding of
    /// a single value (reuses `encode_group_key` so f64/list/etc. hash safely).
    CountDistinct(std::collections::HashSet<String>),
    Sum { i_sum: i64, f_sum: f64, any_float: bool },
    Avg { sum: f64, n: i64 },
    MinMax(Option<Value>),
}

impl AggState {
    fn new(func: AggFunc) -> Self {
        match func {
            AggFunc::Count => AggState::Count(0),
            AggFunc::CountDistinct => AggState::CountDistinct(std::collections::HashSet::new()),
            AggFunc::Sum => AggState::Sum {
                i_sum: 0,
                f_sum: 0.0,
                any_float: false,
            },
            AggFunc::Avg => AggState::Avg { sum: 0.0, n: 0 },
            AggFunc::Min | AggFunc::Max => AggState::MinMax(None),
        }
    }

    /// Fold one row's contribution. `value` is `None` for `count(*)`.
    fn update(&mut self, func: AggFunc, value: Option<&Value>) {
        let numeric = |v: &Value| -> Option<f64> {
            match v {
                Value::I64(n) => Some(*n as f64),
                Value::F64(f) => Some(*f),
                _ => None,
            }
        };
        match self {
            AggState::Count(c) => match value {
                // count(*) counts every row; count(?x) counts non-null x.
                None => *c += 1,
                Some(v) if !matches!(v, Value::Null) => *c += 1,
                _ => {}
            },
            AggState::CountDistinct(set) => {
                // Count distinct non-null values. `distinct *` is rejected at
                // parse time, so `value` is always Some here.
                if let Some(v) = value {
                    if !matches!(v, Value::Null) {
                        set.insert(encode_group_key(std::slice::from_ref(v)));
                    }
                }
            }
            AggState::Sum {
                i_sum,
                f_sum,
                any_float,
            } => {
                if let Some(v) = value {
                    match v {
                        Value::I64(n) => *i_sum += *n,
                        Value::F64(f) => {
                            *f_sum += *f;
                            *any_float = true;
                        }
                        _ => {}
                    }
                }
            }
            AggState::Avg { sum, n } => {
                if let Some(x) = value.and_then(numeric) {
                    *sum += x;
                    *n += 1;
                }
            }
            AggState::MinMax(best) => {
                if let Some(v) = value {
                    if matches!(v, Value::Null) {
                        return;
                    }
                    match best {
                        None => *best = Some(v.clone()),
                        Some(cur) => {
                            let ord = crate::query::compare_values(v, cur);
                            let take = match func {
                                AggFunc::Min => ord == std::cmp::Ordering::Less,
                                _ => ord == std::cmp::Ordering::Greater,
                            };
                            if take {
                                *best = Some(v.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    fn finalize(self) -> Value {
        match self {
            AggState::Count(c) => Value::I64(c),
            AggState::CountDistinct(set) => Value::I64(set.len() as i64),
            AggState::Sum {
                i_sum,
                f_sum,
                any_float,
            } => {
                if any_float {
                    Value::F64(i_sum as f64 + f_sum)
                } else {
                    Value::I64(i_sum)
                }
            }
            AggState::Avg { sum, n } => {
                if n == 0 {
                    Value::Null
                } else {
                    Value::F64(sum / n as f64)
                }
            }
            AggState::MinMax(best) => best.unwrap_or(Value::Null),
        }
    }
}

/// Encode group-key values into a stable, collision-resistant string so rows
/// can be grouped via a `HashMap` (Value isn't `Hash` because of `f64`).
fn encode_group_key(vals: &[Value]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for v in vals {
        match v {
            Value::String(x) => {
                s.push('s');
                s.push_str(x);
            }
            Value::I64(n) => {
                let _ = write!(s, "i{}", n);
            }
            Value::F64(f) => {
                let _ = write!(s, "f{}", f.to_bits());
            }
            Value::Bool(b) => {
                s.push('b');
                s.push(if *b { '1' } else { '0' });
            }
            Value::Ref(r) => {
                let _ = write!(s, "r{}", r);
            }
            Value::Bytes(by) => {
                s.push('y');
                for b in by {
                    let _ = write!(s, "{:02x}", b);
                }
            }
            Value::Enum(_) => s.push('e'),
            Value::List(items) => {
                let _ = write!(s, "l{}", items.len());
                s.push('\u{2}');
                s.push_str(&encode_group_key(items));
            }
            Value::Vector(v) => {
                let _ = write!(s, "v{}", v.len());
                for f in v {
                    let _ = write!(s, ":{:08x}", f.to_bits());
                }
            }
            Value::Null => s.push('n'),
        }
        s.push('\u{1}'); // field separator
    }
    s
}

/// Collapse result rows into aggregate groups when `find` contains any
/// aggregate. Plain find variables form the grouping key (implicit GROUP BY);
/// with no plain variables, the whole result is one group. Output columns are
/// the `find` element labels, in order. No-op when there are no aggregates.
fn apply_aggregation(qr: &mut QueryResult, query: &Query) -> Result<()> {
    let has_agg = query
        .find_elems
        .iter()
        .any(|e| matches!(e, FindElem::Agg { .. }));
    if !has_agg {
        return Ok(());
    }

    let col_of = |var: &str| -> Result<usize> {
        qr.columns
            .iter()
            .position(|c| c == var)
            .ok_or_else(|| DbError::Query(format!("aggregate/group variable '{}' is not bound by the query", var)))
    };

    // Per output element: take the next grouping value or aggregate result.
    enum OutSpec {
        Group,
        Agg,
    }
    let mut specs: Vec<OutSpec> = Vec::with_capacity(query.find_elems.len());
    let mut group_cols: Vec<usize> = Vec::new();
    let mut agg_list: Vec<(AggFunc, Option<usize>)> = Vec::new();
    for e in &query.find_elems {
        match e {
            FindElem::Var(v) => {
                group_cols.push(col_of(v)?);
                specs.push(OutSpec::Group);
            }
            FindElem::Agg { func, var, .. } => {
                let col = if var == "*" { None } else { Some(col_of(var)?) };
                agg_list.push((*func, col));
                specs.push(OutSpec::Agg);
            }
        }
    }

    // Group rows, preserving first-seen group order.
    let mut group_index: HashMap<String, usize> = HashMap::new();
    let mut groups: Vec<(Vec<Value>, Vec<AggState>)> = Vec::new();
    for row in &qr.rows {
        let key_vals: Vec<Value> = group_cols.iter().map(|&c| row[c].clone()).collect();
        let key = encode_group_key(&key_vals);
        let gi = *group_index.entry(key).or_insert_with(|| {
            let states = agg_list.iter().map(|(f, _)| AggState::new(*f)).collect();
            groups.push((key_vals, states));
            groups.len() - 1
        });
        let states = &mut groups[gi].1;
        for (i, (func, col)) in agg_list.iter().enumerate() {
            let v = col.map(|c| &row[c]);
            states[i].update(*func, v);
        }
    }

    // When there are no grouping vars and zero matching rows, aggregates still
    // produce one row (e.g. count(*) over an empty set is 0).
    if group_cols.is_empty() && groups.is_empty() {
        let states = agg_list.iter().map(|(f, _)| AggState::new(*f)).collect();
        groups.push((Vec::new(), states));
    }

    // Build output.
    qr.columns = query.find_elems.iter().map(|e| e.label().to_string()).collect();
    let mut out_rows = Vec::with_capacity(groups.len());
    for (key_vals, states) in groups {
        let mut state_iter = states.into_iter();
        let mut key_iter = key_vals.into_iter();
        let mut row = Vec::with_capacity(specs.len());
        for spec in &specs {
            match spec {
                OutSpec::Group => row.push(key_iter.next().expect("group value")),
                OutSpec::Agg => row.push(state_iter.next().expect("agg state").finalize()),
            }
        }
        out_rows.push(row);
    }
    qr.rows = out_rows;
    Ok(())
}

/// Post-execution shaping: aggregate, then order, then paginate.
fn post_process(qr: &mut QueryResult, query: &Query) -> Result<()> {
    apply_aggregation(qr, query)?;
    apply_ordering_and_paging(qr, query)?;
    Ok(())
}

/// Apply `order_by`, `offset`, and `limit` to a finished result set. Ordering
/// is post-projection, so order keys must be variables present in `find`.
/// The sort is stable, so equal keys preserve the underlying scan order.
fn apply_ordering_and_paging(qr: &mut QueryResult, query: &Query) -> Result<()> {
    use std::cmp::Ordering;

    if !query.order_by.is_empty() {
        // Resolve each order variable to its output column index up front.
        let mut keys: Vec<(usize, bool)> = Vec::with_capacity(query.order_by.len());
        for ok in &query.order_by {
            let col = qr.columns.iter().position(|c| c == &ok.var).ok_or_else(|| {
                DbError::Query(format!(
                    "order_by variable '{}' must be one of the find variables {:?}",
                    ok.var, qr.columns
                ))
            })?;
            keys.push((col, ok.desc));
        }
        qr.rows.sort_by(|a, b| {
            for &(col, desc) in &keys {
                let ord = crate::query::compare_values(&a[col], &b[col]);
                let ord = if desc { ord.reverse() } else { ord };
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            Ordering::Equal
        });
    }

    // Offset then limit, both relative to the (possibly) ordered rows.
    if let Some(offset) = query.offset {
        if offset >= qr.rows.len() {
            qr.rows.clear();
        } else if offset > 0 {
            qr.rows.drain(0..offset);
        }
    }
    if let Some(limit) = query.limit {
        qr.rows.truncate(limit);
    }
    Ok(())
}

/// Reject a query that references an entity type not in the schema. The
/// executor derives index keys from the type+field names in the query, so
/// a dropped (or never-defined) type would otherwise read its leftover
/// datoms straight from the indexes. Validating here is what makes a soft
/// `drop` actually hide a type from reads, and turns a typo'd type name
/// into a clear error instead of a silently-empty result.
fn validate_query_types(query: &Query, schema: &SchemaRegistry) -> Result<()> {
    let mut missing: Option<String> = None;
    query.where_clause.for_each_pattern(&mut |clause| {
        if missing.is_none() && !schema.contains(&clause.entity_type) {
            missing = Some(clause.entity_type.clone());
        }
    });
    match missing {
        Some(name) => Err(DbError::Query(format!(
            "unknown entity type '{}' (not defined, or dropped)",
            name
        ))),
        None => Ok(()),
    }
}

/// Rewrite each clause's scalar field literals to match the declared field
/// type, so the planner and AVET index see correctly-typed values. Runs once
/// per query before planning; enum sub-field patterns are left untouched.
fn normalize_query_literals(query: &mut Query, schema: &SchemaRegistry) {
    query.where_clause.for_each_pattern_mut(&mut |clause| {
        let Some(type_def) = schema.get(&clause.entity_type) else {
            return;
        };
        for (field_name, pattern) in &mut clause.field_patterns {
            let Some(fd) = type_def.get_field(field_name) else {
                continue;
            };
            let ft = fd.field_type.clone();
            match pattern {
                Pattern::Constant(value) => {
                    if let Some((_, nv)) = coerce_numeric_literal(&ft, None, value) {
                        *value = nv;
                    }
                }
                Pattern::Predicate { op, value } => {
                    if let Some((Some(nop), nv)) = coerce_numeric_literal(&ft, Some(&*op), value) {
                        *op = nop;
                        *value = nv;
                    }
                }
                Pattern::BoundPredicate { op, value, .. } => {
                    if let Some((Some(nop), nv)) = coerce_numeric_literal(&ft, Some(&*op), value) {
                        *op = nop;
                        *value = nv;
                    }
                }
                _ => {}
            }
        }
    });
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

        // A schema definition is just a datom: its current state has to be
        // replayed (assert/retract) per schema entity, exactly like any
        // other attribute. A soft drop retracts the definition datom; if we
        // blindly registered every `added` datom we'd resurrect dropped
        // types on every restart. Only definitions that are *currently
        // asserted* get registered. When two live entities carry the same
        // name (a redefinition writes a fresh entity), the higher entity id
        // — i.e. the later definition — wins, matching insert order here.
        for (entity, json) in current_schema_defs(&enum_entries) {
            if let Ok(enum_def) = serde_json::from_str::<EnumTypeDef>(&json) {
                schema.register_enum(enum_def);
            }
            let _ = entity;
        }

        for (entity, json) in current_schema_defs(&type_entries) {
            if let Ok(type_def) = serde_json::from_str::<EntityTypeDef>(&json) {
                schema.register(type_def);
            }
            let _ = entity;
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

        // Reject incompatible redefinition of an existing type. A redefinition
        // may only ADD fields: removing a field would orphan its on-disk data,
        // and changing an existing field's type or modifiers would leave stored
        // data inconsistent with the schema. Re-stating existing fields
        // unchanged (and adding new ones) is allowed.
        if let Some(existing) = schema.get(&type_def.name) {
            for old_field in &existing.fields {
                match type_def.get_field(&old_field.name) {
                    None => {
                        return Err(DbError::Schema(format!(
                            "cannot redefine type '{}': field '{}' is missing from the new \
                             definition. Removing a field would orphan its data; redefinitions \
                             may only add fields.",
                            type_def.name, old_field.name
                        )));
                    }
                    Some(new_field) if !field_redefinition_ok(old_field, new_field) => {
                        return Err(DbError::Schema(format!(
                            "cannot redefine type '{}': field '{}' changed definition. An existing \
                             field's type, cardinality, required and unique cannot change; only \
                             adding new fields, or turning on the additive `indexed`/`fulltext` \
                             modifiers, is allowed. (Existing rows are NOT retroactively indexed — \
                             re-assert them to populate a newly-enabled index.)",
                            type_def.name, old_field.name
                        )));
                    }
                    _ => {}
                }
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

    /// Drop an entity type.
    ///
    /// A **soft** drop (`hard == false`) retracts the type's schema
    /// definition as an ordinary transaction: it disappears from `schema`
    /// and can no longer be queried, but every datom, index entry and
    /// history record is preserved untouched. The drop itself is recorded
    /// in history, and re-`define`-ing the type makes all the old data
    /// visible and queryable again.
    ///
    /// A **hard** purge (`hard == true`) deletes the type's schema
    /// definition AND every datom of every entity of that type from all
    /// indexes (EAVT/AEVT/AVET/VAET plus the current-state mirrors). It is
    /// irreversible and destroys history for those entities. A hard purge
    /// is refused if another live type or enum still references this type
    /// via a `ref` field, so the schema is never left dangling.
    pub fn drop_type(&self, name: &str, hard: bool) -> Result<DropResult> {
        if name.starts_with("__") {
            return Err(DbError::Schema("cannot drop reserved type".into()));
        }
        let mut schema = self.schema.write();
        if !schema.contains(name) {
            return Err(DbError::Schema(format!("unknown entity type '{}'", name)));
        }
        let referrers = schema.referrers_of_type(name);
        self.drop_named(&mut schema, name, "type", SCHEMA_TYPE_ATTR, false, hard, referrers)
    }

    /// Drop an enum type. Soft/hard semantics mirror [`drop_type`]. An enum
    /// owns no entities of its own — its values live inline on the types
    /// that use it — so a hard purge of an enum only deletes the enum's
    /// schema-definition datoms (the inline values are purged when their
    /// owning type is purged). A hard purge is refused while any live type
    /// or enum still has a field of this enum type.
    pub fn drop_enum(&self, name: &str, hard: bool) -> Result<DropResult> {
        if name.starts_with("__") {
            return Err(DbError::Schema("cannot drop reserved enum".into()));
        }
        let mut schema = self.schema.write();
        if !schema.contains_enum(name) {
            return Err(DbError::Schema(format!("unknown enum type '{}'", name)));
        }
        let referrers = schema.referrers_of_enum(name);
        self.drop_named(&mut schema, name, "enum", SCHEMA_ENUM_ATTR, true, hard, referrers)
    }

    /// Shared body for `drop_type` / `drop_enum`. `schema` is the held
    /// write guard; `schema_attr` is the reserved attribute the definition
    /// is stored under; `parse_enum` selects how the stored JSON is parsed
    /// to recover its declared name; `referrers` is the precomputed list of
    /// live schema references to `name`.
    fn drop_named(
        &self,
        schema: &mut SchemaRegistry,
        name: &str,
        kind: &'static str,
        schema_attr: &'static str,
        parse_enum: bool,
        hard: bool,
        referrers: Vec<String>,
    ) -> Result<DropResult> {
        let mut warnings = Vec::new();

        let result = if hard {
            if !referrers.is_empty() {
                return Err(DbError::Schema(format!(
                    "cannot hard-purge {} '{}': still referenced by {}. \
                     Drop or redefine those first.",
                    kind,
                    name,
                    referrers.join(", ")
                )));
            }
            // Resolve fulltext attrs while we still hold the schema guard, so
            // hard_purge needn't re-lock the schema (it would deadlock).
            let fulltext_attrs: Vec<(crate::intern::AttrId, String)> = schema
                .get(name)
                .map(|td| {
                    td.fields
                        .iter()
                        .filter(|f| f.fulltext)
                        .filter_map(|f| {
                            let attr = td.attribute_name(&f.name);
                            self.attr_interner.lookup(&attr).map(|id| (id, attr))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let (entities_purged, datoms_deleted, dangling_refs) =
                self.hard_purge(name, schema_attr, parse_enum, fulltext_attrs)?;
            DropResult {
                name: name.to_string(),
                kind,
                hard: true,
                tx_id: None,
                entities_purged,
                datoms_deleted,
                dangling_refs,
                warnings,
            }
        } else {
            if !referrers.is_empty() {
                warnings.push(format!(
                    "{} '{}' is still referenced by {}; those definitions now point at \
                     a dropped {}. Re-define '{}' to restore.",
                    kind,
                    name,
                    referrers.join(", "),
                    kind,
                    name
                ));
            }
            let tx_id = self.soft_retract_schema_defs(name, schema_attr, parse_enum)?;
            DropResult {
                name: name.to_string(),
                kind,
                hard: false,
                tx_id: Some(tx_id),
                entities_purged: 0,
                datoms_deleted: 0,
                dangling_refs: 0,
                warnings,
            }
        };

        if parse_enum {
            schema.remove_enum(name);
        } else {
            schema.remove(name);
        }
        self.plan_cache.write().clear();
        self.query_cache.invalidate_all();
        Ok(result)
    }

    /// Soft drop: retract every currently-asserted schema-definition datom
    /// whose declared name matches `name`, as one transaction. Returns the
    /// retraction's tx_id. Mirrors the locking/counter discipline of
    /// `define_type`.
    fn soft_retract_schema_defs(
        &self,
        name: &str,
        schema_attr: &'static str,
        parse_enum: bool,
    ) -> Result<TxId> {
        let attr_id = self.attr_interner.lookup(schema_attr).ok_or_else(|| {
            DbError::Schema(format!("no schema definitions exist for '{}'", name))
        })?;
        let timestamp_ms = now_millis();
        let name_owned = name.to_string();

        let _guard = self.write_lock.lock();
        let tx_id = self.tx_counter.load(Ordering::Acquire) + 1;

        self.storage.execute_batch(Box::new(move |txn| {
            // Scan the definition attribute and replay each schema entity to
            // its current value, then retract those whose declared name
            // matches. Retraction value must equal the asserted JSON so the
            // replay on the next load sees it as a clearing retract.
            let prefix = index::aevt_attr_prefix(attr_id);
            let end = index::prefix_end(&prefix);
            let entries = txn.scan(&prefix, &end)?;

            let mut retracted = 0usize;
            for (entity, json) in current_schema_defs(&entries) {
                if schema_def_name(&json, parse_enum).as_deref() != Some(name_owned.as_str()) {
                    continue;
                }
                let value = Value::String(json.into());
                for (key, val) in index::encode_datom(entity, attr_id, &value, tx_id, false) {
                    txn.put(key, val)?;
                }
                retracted += 1;
            }

            if retracted == 0 {
                return Err(StorageError::Backend(format!(
                    "no live schema definition found for '{}'",
                    name_owned
                )));
            }

            txn.put(index::meta_key(TX_COUNTER_KEY), encode_u64(tx_id))?;
            store_tx_timestamp(txn, tx_id, timestamp_ms)?;
            Ok(Box::new(()) as Box<dyn Any + Send>)
        }))?;

        self.tx_counter.store(tx_id, Ordering::Release);
        drop(_guard);
        Ok(tx_id)
    }

    /// Hard purge: delete the schema definition AND all datoms of all
    /// entities of `name` from every index, atomically in one batch.
    /// Returns `(entities_purged, datoms_deleted, dangling_inbound_refs)`.
    fn hard_purge(
        &self,
        name: &str,
        schema_attr: &'static str,
        parse_enum: bool,
        // (attr_id, attr_name) of this type's fulltext fields, resolved by the
        // caller (which holds the schema guard) to avoid re-locking the schema.
        fulltext_attrs: Vec<(crate::intern::AttrId, String)>,
    ) -> Result<(u64, u64, u64)> {
        use std::collections::HashSet;
        use crate::intern::AttrId;

        let type_marker_id = self.attr_interner.lookup("__type");
        let schema_attr_id = self.attr_interner.lookup(schema_attr);
        let name_owned = name.to_string();

        let _guard = self.write_lock.lock();

        let result = self.storage.execute_batch(Box::new(move |txn| {
            let mut data_entities: HashSet<EntityId> = HashSet::new();
            let mut schema_entities: HashSet<EntityId> = HashSet::new();

            // Entities of this type, by their __type marker (history-complete:
            // AVET holds every assert/retract of __type == name).
            if let Some(tid) = type_marker_id {
                let prefix =
                    index::avet_attr_value_prefix(tid, &Value::String(name_owned.as_str().into()));
                let end = index::prefix_end(&prefix);
                for (k, _) in txn.scan(&prefix, &end)? {
                    if let Some(d) = index::decode_datom_from_avet(&k) {
                        data_entities.insert(d.entity);
                    }
                }
            }

            // Schema-definition entities for this name (all of them, live or
            // already retracted — a hard purge wipes history too).
            if let Some(sid) = schema_attr_id {
                let prefix = index::aevt_attr_prefix(sid);
                let end = index::prefix_end(&prefix);
                for (k, _) in txn.scan(&prefix, &end)? {
                    if let Some(d) = index::decode_datom_from_aevt(&k) {
                        if let Value::String(json) = &d.value {
                            if schema_def_name(json, parse_enum).as_deref()
                                == Some(name_owned.as_str())
                            {
                                schema_entities.insert(d.entity);
                            }
                        }
                    }
                }
            }

            let mut purge_set: HashSet<EntityId> = HashSet::new();
            purge_set.extend(&data_entities);
            purge_set.extend(&schema_entities);

            // Delete every datom of every purged entity from all indexes.
            let mut datoms_deleted: u64 = 0;
            for &eid in &purge_set {
                let prefix = index::eavt_entity_prefix(eid);
                let end = index::prefix_end(&prefix);
                let entries = txn.scan(&prefix, &end)?;
                for (k, _) in &entries {
                    let Some(d) = index::decode_datom_from_eavt(k) else {
                        continue;
                    };
                    txn.delete(&index::encode_eavt(d.entity, d.attr_id, &d.value, d.tx, d.added))?;
                    txn.delete(&index::encode_aevt(d.entity, d.attr_id, &d.value, d.tx, d.added))?;
                    txn.delete(&index::encode_avet(d.entity, d.attr_id, &d.value, d.tx, d.added))?;
                    if matches!(d.value, Value::Ref(_)) {
                        txn.delete(&index::encode_vaet(
                            d.entity, d.attr_id, &d.value, d.tx, d.added,
                        ))?;
                    }
                    // Current-state mirrors. CURRENT_AEVT is one key per
                    // (attr,entity); CURRENT_AVET keys the current value —
                    // deleting a stale historical value's key is a harmless
                    // no-op, the live one is hit when its datom is reached.
                    let mut caevt = index::current_aevt_attr_prefix(d.attr_id);
                    caevt.extend_from_slice(&d.entity.to_be_bytes());
                    txn.delete(&caevt)?;
                    txn.delete(&index::encode_current_avet(d.attr_id, &d.value, d.entity))?;
                    datoms_deleted += 1;
                }
            }

            // The per-type entity counter meta key (no-op for enums).
            txn.delete(&index::meta_key(&format!("type_count:{}", name_owned)))?;

            // Tear down the full-text inverted index for this type's fulltext
            // fields: delete all postings + doclen under each attr, and reset
            // the corpus stats. Otherwise stale postings/stats survive the
            // purge and corrupt later BM25 scoring.
            for (attr_id, attr_name) in &fulltext_attrs {
                let mut pp = vec![index::FTS_POSTINGS_PREFIX];
                pp.extend_from_slice(&attr_id.to_be_bytes());
                let pend = index::prefix_end(&pp);
                for (k, _) in txn.scan(&pp, &pend)? {
                    txn.delete(&k)?;
                }
                let mut dp = vec![index::FTS_DOCLEN_PREFIX];
                dp.extend_from_slice(&attr_id.to_be_bytes());
                let dend = index::prefix_end(&dp);
                for (k, _) in txn.scan(&dp, &dend)? {
                    txn.delete(&k)?;
                }
                txn.delete(&index::meta_key(&format!("fts_ndocs:{}", attr_name)))?;
                txn.delete(&index::meta_key(&format!("fts_totlen:{}", attr_name)))?;
            }

            // Report inbound refs from *other* entities that now dangle.
            // Read after the deletes above (the overlay reflects them), so
            // refs internal to the purged set don't show up. Replay each
            // (referrer, attr) to its current target to avoid counting
            // refs that were later retracted or repointed.
            let mut ref_hist: HashMap<(EntityId, AttrId), Vec<(EntityId, TxId, bool)>> =
                HashMap::new();
            for &eid in &data_entities {
                let prefix = index::vaet_value_prefix(&Value::Ref(eid));
                let end = index::prefix_end(&prefix);
                for (k, _) in txn.scan(&prefix, &end)? {
                    if let Some(d) = index::decode_datom_from_vaet(&k) {
                        ref_hist
                            .entry((d.entity, d.attr_id))
                            .or_default()
                            .push((eid, d.tx, d.added));
                    }
                }
            }
            let mut dangling_entities: HashSet<EntityId> = HashSet::new();
            for ((referrer, _attr), mut hist) in ref_hist {
                if purge_set.contains(&referrer) {
                    continue;
                }
                hist.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2)));
                let mut current: Option<EntityId> = None;
                for (target, _tx, added) in hist {
                    if added {
                        current = Some(target);
                    } else if current == Some(target) {
                        current = None;
                    }
                }
                if let Some(t) = current {
                    if data_entities.contains(&t) {
                        dangling_entities.insert(referrer);
                    }
                }
            }

            // Report data entities only; the internal schema-definition
            // entity is an implementation detail (its datoms still count
            // toward datoms_deleted).
            let stats = (
                data_entities.len() as u64,
                datoms_deleted,
                dangling_entities.len() as u64,
            );
            Ok(Box::new(stats) as Box<dyn Any + Send>)
        }))?;

        drop(_guard);
        Ok(*result.downcast::<(u64, u64, u64)>().expect("wrong type"))
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
        validate_query_types(query, &schema)?;
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

        // Align literal types with the schema (e.g. integer literal against
        // an f64 field) before planning, so the index scan and post-filter
        // see correctly-typed values. Must run before the plan-cache key so
        // the key and the cached plan agree on the normalized literals.
        normalize_query_literals(&mut query, &schema);

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
            // Sample the cache generation BEFORE execute_read takes its
            // snapshot, so a write that lands during this read is detected.
            let cache_gen = self.query_cache.generation();
            let result = self
                .storage
                .execute_read(Box::new(move |snap| {
                    let qr =
                        executor::execute_plan(snap, &plan, &schema, &cache, &interner, cache_gen)
                            .map_err(|e| StorageError::Backend(e))?;
                    Ok(Box::new(qr) as Box<dyn Any + Send>)
                }))?;
            let mut qr = *result.downcast::<QueryResult>().expect("wrong type");
            post_process(&mut qr, &query)?;
            return Ok(qr);
        }

        let cache = self.query_cache.clone();
        let interner = self.attr_interner.clone();
        // Clone for the planning closure so the original `query` survives for
        // post-execution ordering/paging below.
        let query_for_plan = query.clone();
        // Sample the cache generation BEFORE execute_read takes its snapshot.
        let cache_gen = self.query_cache.generation();
        let result = self
            .storage
            .execute_read(Box::new(move |snap| {
                let plan = planner::plan_query(snap, &query_for_plan, &schema, &interner)
                    .map_err(|e| StorageError::Backend(e))?;

                let qr =
                    executor::execute_plan(snap, &plan, &schema, &cache, &interner, cache_gen)
                        .map_err(|e| StorageError::Backend(e))?;

                Ok(Box::new((plan, qr)) as Box<dyn Any + Send>)
            }))?;

        let (plan, mut qr) = *result
            .downcast::<(QueryPlan, QueryResult)>()
            .expect("wrong type");

        // Store in cache
        {
            let mut cache = self.plan_cache.write();
            cache.insert(cache_key, Arc::new(plan));
        }

        post_process(&mut qr, &query)?;
        Ok(qr)
    }

    /// Generate a query plan without executing.
    pub fn explain(&self, query: &Query) -> Result<QueryPlan> {
        let schema = self.schema.read().clone();
        validate_query_types(query, &schema)?;
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

    /// Take a point-in-time on-disk checkpoint of the live database at
    /// `path`. The path must not yet exist; RocksDB creates it. Checkpoints
    /// are hard-link based, so `path` must be on the same filesystem as
    /// the live data-dir. Restore is "stop the server and point
    /// `--data-dir` at the checkpoint directory".
    pub fn create_checkpoint(&self, path: &std::path::Path) -> Result<()> {
        self.storage.checkpoint(path).map_err(DbError::from)
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
                        // The column cache is patched in-place above, but a new
                        // entity can carry a vector — invalidate the type's ANN
                        // index so it rebuilds with the new point. Cheap unless
                        // the type actually has a vector field.
                        if app.values.keys().any(|f| {
                            schema
                                .get(&app.entity_type)
                                .and_then(|td| td.get_field(f))
                                .map(|fd| matches!(fd.field_type, crate::schema::FieldType::Vector(_)))
                                .unwrap_or(false)
                        }) {
                            query_cache.invalidate_ann_for_type(&app.entity_type);
                        }
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

/// Given the raw AEVT entries for a schema attribute (`__schema_type` or
/// `__schema_enum`), replay each schema entity's assert/retract history and
/// return the currently-asserted JSON definition for each live entity,
/// ordered by entity id ascending. A soft drop retracts the definition, so
/// a dropped type yields nothing here; a redefinition writes a new entity,
/// so later (higher-id) definitions sort last and win on registration.
fn current_schema_defs(entries: &[(Vec<u8>, Vec<u8>)]) -> Vec<(EntityId, String)> {
    let mut by_entity: HashMap<EntityId, Vec<index::DecodedDatom>> = HashMap::new();
    for (key, _) in entries {
        if let Some(datom) = index::decode_datom_from_aevt(key) {
            by_entity.entry(datom.entity).or_default().push(datom);
        }
    }

    let mut out: Vec<(EntityId, String)> = Vec::new();
    for (entity, mut datoms) in by_entity {
        datoms.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));
        let mut current: Option<String> = None;
        for d in datoms {
            match (&d.value, d.added) {
                (Value::String(json), true) => current = Some(json.to_string()),
                (Value::String(json), false) if current.as_deref() == Some(json.as_ref()) => {
                    current = None
                }
                _ => {}
            }
        }
        if let Some(json) = current {
            out.push((entity, json));
        }
    }
    out.sort_by_key(|(e, _)| *e);
    out
}

/// Recover the declared name from a stored schema-definition JSON blob.
/// `parse_enum` selects the `EnumTypeDef` shape over `EntityTypeDef`.
/// Both carry a top-level `name`, so a cheap untyped parse suffices and
/// stays correct even if the two shapes diverge later.
fn schema_def_name(json: &str, parse_enum: bool) -> Option<String> {
    let _ = parse_enum;
    serde_json::from_str::<serde_json::Value>(json)
        .ok()?
        .get("name")?
        .as_str()
        .map(|s| s.to_string())
}

/// Outcome of a `drop_type` / `drop_enum` call.
#[derive(Debug, Clone)]
pub struct DropResult {
    /// The type or enum name that was dropped.
    pub name: String,
    /// `"type"` or `"enum"`.
    pub kind: &'static str,
    /// True for a hard purge (datoms deleted), false for a soft drop
    /// (definition retracted, history and data preserved).
    pub hard: bool,
    /// Transaction id of the retraction. Present only for a soft drop;
    /// a hard purge deletes datoms outside the transaction log.
    pub tx_id: Option<TxId>,
    /// Hard purge only: number of entities whose datoms were deleted
    /// (data entities of the type plus its schema-definition entities).
    pub entities_purged: u64,
    /// Hard purge only: total index-key-bearing datoms deleted.
    pub datoms_deleted: u64,
    /// Hard purge only: number of *other* entities still holding a ref
    /// to a purged entity (now dangling). Informational; these are left
    /// untouched so unrelated types are never silently mutated.
    pub dangling_refs: u64,
    /// Non-fatal advisories (e.g. soft-dropping a type still referenced
    /// by another live type).
    pub warnings: Vec<String>,
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
            let fulltext = f.get("fulltext").and_then(|x| x.as_bool()).unwrap_or(false);
            let ann = f.get("ann").and_then(|x| x.as_bool()).unwrap_or(false);
            // cardinality: "many" or true → Many; default One.
            let cardinality = match f.get("cardinality") {
                Some(serde_json::Value::String(s)) if s == "many" => {
                    crate::schema::Cardinality::Many
                }
                Some(serde_json::Value::Bool(true)) => crate::schema::Cardinality::Many,
                _ => crate::schema::Cardinality::One,
            };
            // A many field can't also be a List (that would be a list-of-sets);
            // and `many` + `unique` would mean a per-value global unique, which
            // we don't support — reject both.
            if matches!(cardinality, crate::schema::Cardinality::Many) {
                if matches!(field_type, FieldType::List(_)) {
                    return Err(format!(
                        "field '{}' cannot be both a list type and cardinality-many",
                        field_name
                    ));
                }
                if unique {
                    return Err(format!(
                        "field '{}' cannot be both unique and cardinality-many",
                        field_name
                    ));
                }
            }
            // fulltext only makes sense on string fields (incl. many-of-string).
            if fulltext {
                let elem_is_string = match &field_type {
                    FieldType::String => true,
                    FieldType::List(inner) => matches!(**inner, FieldType::String),
                    _ => false,
                };
                if !elem_is_string {
                    return Err(format!(
                        "field '{}' is marked fulltext but is not a string field",
                        field_name
                    ));
                }
            }
            // ann only makes sense on vector fields.
            if ann && !matches!(field_type, FieldType::Vector(_)) {
                return Err(format!(
                    "field '{}' is marked ann but is not a vector field",
                    field_name
                ));
            }
            Ok(FieldDef {
                name: field_name,
                field_type,
                required,
                unique,
                indexed,
                cardinality,
                fulltext,
                ann,
            })
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;

    // Optional composite unique keys: [["doc","idx"], ...].
    let unique_keys: Vec<Vec<String>> = match v.get("unique_keys") {
        None => Vec::new(),
        Some(uk) => {
            let arr = uk
                .as_array()
                .ok_or("'unique_keys' must be an array of field-name arrays")?;
            let mut keys = Vec::with_capacity(arr.len());
            for entry in arr {
                let fields_arr = entry
                    .as_array()
                    .ok_or("each unique key must be an array of field names")?;
                let mut key_fields = Vec::with_capacity(fields_arr.len());
                for fname in fields_arr {
                    let fname = fname
                        .as_str()
                        .ok_or("unique key field name must be a string")?
                        .to_string();
                    // Validate the field exists on this type.
                    if !fields.iter().any(|f| f.name == fname) {
                        return Err(format!(
                            "unique key references unknown field '{}'",
                            fname
                        ));
                    }
                    key_fields.push(fname);
                }
                if key_fields.is_empty() {
                    return Err("a unique key must name at least one field".into());
                }
                keys.push(key_fields);
            }
            keys
        }
    };

    Ok(EntityTypeDef {
        name,
        fields,
        unique_keys,
    })
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
                                cardinality: crate::schema::Cardinality::One,
                                fulltext: false,
                                ann: false,
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
    let s = s.trim();
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
            } else if let Some(inner) =
                other.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
            {
                // `[elem]` — a cardinality-many list of `elem`.
                let elem = parse_field_type(inner)?;
                validate_list_elem(&elem)?;
                Ok(FieldType::List(Box::new(elem)))
            } else if let Some(inner) =
                other.strip_prefix("list(").and_then(|s| s.strip_suffix(')'))
            {
                let elem = parse_field_type(inner)?;
                validate_list_elem(&elem)?;
                Ok(FieldType::List(Box::new(elem)))
            } else if let Some(inner) =
                other.strip_prefix("vector(").and_then(|s| s.strip_suffix(')'))
            {
                let dim: usize = inner
                    .trim()
                    .parse()
                    .map_err(|_| format!("vector dimension must be a positive integer, got '{}'", inner))?;
                if dim == 0 {
                    return Err("vector dimension must be at least 1".into());
                }
                Ok(FieldType::Vector(dim))
            } else {
                Err(format!("unknown field type: {}", other))
            }
        }
    }
}

/// Whether redefining `old` to `new` is a compatible change. Identical fields
/// are always fine. Beyond that, only the *additive* index modifiers
/// (`indexed`, `fulltext`) may turn on (false→true) — they build a new index
/// without invalidating existing data. Everything structural (type,
/// cardinality, required, unique) must match exactly. Turning an index modifier
/// OFF is disallowed (it would orphan index entries).
fn field_redefinition_ok(old: &FieldDef, new: &FieldDef) -> bool {
    if old == new {
        return true;
    }
    if old.field_type != new.field_type
        || old.cardinality != new.cardinality
        || old.required != new.required
        || old.unique != new.unique
    {
        return false;
    }
    // `indexed` / `fulltext` / `ann` may only go false → true (additive index
    // builds; turning one off would orphan its index).
    let indexed_ok = new.indexed || !old.indexed;
    let fulltext_ok = new.fulltext || !old.fulltext;
    let ann_ok = new.ann || !old.ann;
    indexed_ok && fulltext_ok && ann_ok
}

/// A list element must be a scalar type — no nested lists, no enums (enums
/// fan out into sub-attribute datoms and can't live inside a single value).
fn validate_list_elem(elem: &FieldType) -> std::result::Result<(), String> {
    match elem {
        FieldType::List(_) => Err("nested list element types are not supported".into()),
        FieldType::Enum(_) => Err("enum element types inside a list are not supported".into()),
        _ => Ok(()),
    }
}
