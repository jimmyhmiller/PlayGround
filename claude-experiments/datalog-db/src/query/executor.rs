use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::mem::discriminant;
use std::sync::Arc;

use crate::cache::{self, QueryCache};
use crate::datom::{Datom, EntityId, TxId, Value};
use crate::index;
use crate::intern::AttrInterner;
use crate::query::planner::{self, ClauseScan, JoinSide, JoinStrategy, PlanNode, QueryPlan, SlotMap};
use crate::query::{Pattern, PredOp, Query, VectorMetric, WhereClause};
use crate::schema::{FieldType, SchemaRegistry};
use crate::storage::ReadOps;

/// A tuple of slot values indexed by position. `Vec`-backed because
/// inline-array variants regressed performance on our bench widths
/// (the unused-slot memcpy on `clone()` outweighed the saved alloc).
#[derive(Clone)]
struct Tuple {
    slots: Vec<Option<Value>>,
}

impl Tuple {
    #[inline]
    fn new(num_slots: usize) -> Self {
        Tuple {
            slots: vec![None; num_slots],
        }
    }

    #[inline]
    fn get(&self, slot: usize) -> Option<&Value> {
        self.slots[slot].as_ref()
    }

    #[inline]
    fn set(&mut self, slot: usize, value: Value) {
        self.slots[slot] = Some(value);
    }
}

/// Query result: rows of bound values in `find` order.
#[derive(Debug)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

/// Plan for how to find candidate entities for a single field pattern.
enum AttrPlan {
    /// Exact AVET point lookup: attribute = value
    Exact(String, Value),
    /// Range AVET scan: attribute op value (Gt, Gte, Lt, Lte)
    Range(String, PredOp, Value),
}

// ---------------------------------------------------------------------------
// HashableValue wrapper — Value doesn't implement Hash (because of f64)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct HashableValue(Value);

impl Hash for HashableValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        discriminant(&self.0).hash(state);
        match &self.0 {
            Value::I64(n) => n.hash(state),
            Value::String(s) => s.hash(state),
            Value::Ref(id) => id.hash(state),
            Value::F64(f) => f.to_bits().hash(state),
            Value::Bool(b) => b.hash(state),
            Value::Bytes(b) => b.hash(state),
            Value::Null => {}
            Value::Enum(e) => e.variant.hash(state),
            Value::List(items) => {
                items.len().hash(state);
                for v in items {
                    HashableValue(v.clone()).hash(state);
                }
            }
            Value::Vector(v) => {
                v.len().hash(state);
                for f in v {
                    f.to_bits().hash(state);
                }
            }
        }
    }
}

impl PartialEq for HashableValue {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for HashableValue {}

/// Compute a hash key from a tuple for the given join slot indices.
fn join_key(tuple: &Tuple, join_slots: &[usize]) -> Option<Vec<HashableValue>> {
    let mut key = Vec::with_capacity(join_slots.len());
    for &slot in join_slots {
        match tuple.get(slot) {
            Some(v) => key.push(HashableValue(v.clone())),
            None => return None,
        }
    }
    Some(key)
}

/// Hash a Vec<HashableValue> for use as HashMap key.
#[derive(Debug, Clone, PartialEq, Eq)]
struct JoinKey(Vec<HashableValue>);

impl Hash for JoinKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for hv in &self.0 {
            hv.hash(state);
        }
    }
}

// ---------------------------------------------------------------------------
// Volcano / iterator execution model
//
// Each plan node becomes a `PlanIterator` with `open(seed)` / `next()`.
// Rows stream one at a time through the pipeline rather than each layer
// materializing its full result Vec. This dramatically cuts allocation
// pressure on multi-join workloads (Graph 2-Hop, fan-out, star joins) —
// peak intermediate memory goes from O(rows) per layer to O(1).
//
// The leaf `ScanIterator` is currently "fake volcano" — its `open` still
// calls the existing `evaluate_clause` to compute the full candidate
// list, but `next()` yields one buffered tuple at a time. The
// material savings are at the join boundaries: a NestedLoop no longer
// holds left+right vecs simultaneously; only one row from each lives in
// memory between yields. A follow-up pass can stream the leaf itself.
// ---------------------------------------------------------------------------

/// Bundle of borrowed state every iterator needs. One instance is
/// constructed per `execute_plan` call; iterators hold references into
/// it.
struct ExecutionContext<'a> {
    txn: &'a dyn ReadOps,
    schema: &'a SchemaRegistry,
    cache: &'a QueryCache,
    interner: &'a AttrInterner,
    slots: &'a SlotMap,
    as_of: Option<TxId>,
    /// Cache generation sampled before `txn`'s snapshot was created. Passed
    /// to `ensure_type_loaded` so it can reject a load that races a write.
    cache_gen: u64,
}

/// Streaming row producer for one plan node.
trait PlanIterator<'a> {
    /// (Re-)initialize with an outer seed tuple. The seed carries
    /// bindings from a surrounding join (e.g. a NestedLoop's current
    /// left row). Initial top-level open uses an empty seed.
    fn open(&mut self, seed: &Tuple) -> Result<(), String>;

    /// Yield the next tuple, or `None` when exhausted. The yielded
    /// tuple extends the seed with this operator's own bindings.
    fn next(&mut self) -> Result<Option<Tuple>, String>;
}

/// Leaf scan over a single WhereClause. For the bound-entity hot path
/// (e.g. Graph 2-Hop's manager lookups, called hundreds of thousands
/// of times in a benchmark), we cache the bind_slot and type_data Arc
/// across `open()` calls and inline the lookup directly — bypassing
/// the full `evaluate_clause` function-call setup. Other shapes fall
/// back to `evaluate_clause`, which buffers a Vec that `next()`
/// drains one row at a time.
struct ScanIterator<'a> {
    clause: &'a WhereClause,
    ctx: &'a ExecutionContext<'a>,
    buffered: std::vec::IntoIter<Tuple>,
    /// One-shot — yielded once from `next()` then taken. Used by the
    /// bound-entity fast path to avoid allocating a Vec for a single
    /// result.
    pending: Option<Tuple>,
    /// Cached on first open. Once set, subsequent opens skip the
    /// `slots.slot()` HashMap probe, the `cache.ensure_type_loaded`
    /// RwLock acquisition, and the resolved-pattern Vec allocation —
    /// huge for Graph 2-Hop where `open()` runs 800+ times per query.
    cached_bind_slot: Option<usize>,
    cached_type_data: Option<Arc<cache::TypeData>>,
    /// Pre-resolved field patterns. `None` means resolution failed
    /// (unknown variable etc.) and the iterator will produce no rows.
    cached_resolved: Option<Vec<ResolvedFP>>,
    /// True once the cache lookup ran (whether or not it returned a
    /// `TypeData` — `None` is a valid cached state for asOf queries).
    initialized: bool,
}

impl<'a> ScanIterator<'a> {
    fn new(clause: &'a WhereClause, ctx: &'a ExecutionContext<'a>) -> Self {
        Self {
            clause,
            ctx,
            buffered: Vec::new().into_iter(),
            pending: None,
            cached_bind_slot: None,
            cached_type_data: None,
            cached_resolved: None,
            initialized: false,
        }
    }

    /// Populate the cached lookups. Lazy because the iterator may
    /// never be opened (e.g., a parent join exhausting early).
    fn ensure_initialized(&mut self) -> Result<(), String> {
        if self.initialized {
            return Ok(());
        }
        self.cached_bind_slot = self.ctx.slots.slot(&self.clause.bind);
        // Only the current-state (non-asOf) path uses the type cache.
        if self.ctx.as_of.is_none() {
            self.cached_type_data = self.ctx.cache.ensure_type_loaded(
                self.ctx.txn,
                self.ctx.interner,
                &self.clause.entity_type,
                self.ctx.schema,
                self.ctx.cache_gen,
            )?;
            // Resolve patterns once now that we have the type_data.
            // The owned `ResolvedFP` (with Arc'd columns) means we can
            // keep this around across opens.
            if let Some(td) = self.cached_type_data.as_ref() {
                self.cached_resolved = resolve_field_patterns(
                    &self.clause.field_patterns,
                    td,
                    self.ctx.slots,
                    self.ctx.schema,
                    &self.clause.entity_type,
                );
            }
        }
        self.initialized = true;
        Ok(())
    }
}

impl<'a> PlanIterator<'a> for ScanIterator<'a> {
    fn open(&mut self, seed: &Tuple) -> Result<(), String> {
        self.ensure_initialized()?;
        self.buffered = Vec::new().into_iter();
        self.pending = None;

        // Bound-entity fast path: skip the entire `evaluate_clause`
        // function-call setup. This is the Graph 2-Hop inner-loop
        // hot path — called once per outer row.
        if let (Some(bind_slot), Some(td), Some(resolved)) = (
            self.cached_bind_slot,
            self.cached_type_data.as_ref(),
            self.cached_resolved.as_ref(),
        ) {
            if let Some(eid) = seed.get(bind_slot).and_then(|v| {
                if let Value::Ref(id) = v {
                    Some(*id)
                } else {
                    None
                }
            }) {
                if let Some(idx) = td.index_of(eid) {
                    let mut extended = seed.clone();
                    extended.set(bind_slot, Value::Ref(eid));
                    if match_resolved(resolved, idx, &mut extended, self.ctx.slots, td) {
                        self.pending = Some(extended);
                    }
                }
                return Ok(());
            }
        }

        // Fallback: full evaluate_clause (handles unbound scans,
        // AVET intersections, asOf queries, enum patterns).
        let tuples = evaluate_clause(
            self.ctx.txn,
            self.clause,
            seed,
            self.ctx.as_of,
            self.ctx.schema,
            self.ctx.slots,
            self.ctx.cache,
            self.ctx.interner,
            self.ctx.cache_gen,
        )?;
        self.buffered = tuples.into_iter();
        Ok(())
    }

    fn next(&mut self) -> Result<Option<Tuple>, String> {
        if let Some(t) = self.pending.take() {
            return Ok(Some(t));
        }
        Ok(self.buffered.next())
    }
}

/// Nested-loop join. For each row from `left`, re-opens `right` with
/// that row as the seed, yields each merged result, then advances
/// `left`. Peak memory: one left row + one right row at a time
/// (versus today's `Vec<Tuple>` on both sides).
struct NestedLoopIterator<'a> {
    left: Box<dyn PlanIterator<'a> + 'a>,
    right: Box<dyn PlanIterator<'a> + 'a>,
    current_left: Option<Tuple>,
    /// Set to true once `left` has been exhausted, to short-circuit
    /// further `next` calls.
    done: bool,
}

impl<'a> NestedLoopIterator<'a> {
    fn new(left: Box<dyn PlanIterator<'a> + 'a>, right: Box<dyn PlanIterator<'a> + 'a>) -> Self {
        Self {
            left,
            right,
            current_left: None,
            done: false,
        }
    }
}

impl<'a> PlanIterator<'a> for NestedLoopIterator<'a> {
    fn open(&mut self, seed: &Tuple) -> Result<(), String> {
        self.left.open(seed)?;
        self.current_left = None;
        self.done = false;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<Tuple>, String> {
        if self.done {
            return Ok(None);
        }
        loop {
            if self.current_left.is_none() {
                match self.left.next()? {
                    Some(row) => {
                        // Right re-opens with the current left row as
                        // its seed — scan reads bound bindings out of
                        // it (e.g. the entity_id from a ref field).
                        self.right.open(&row)?;
                        self.current_left = Some(row);
                    }
                    None => {
                        self.done = true;
                        return Ok(None);
                    }
                }
            }
            // right_row already carries the left seed merged in because
            // open(seed) seeds the right scan with left's bindings, and
            // evaluate_clause extends `tuple.clone()` — so the yielded
            // tuple is left + right's own bindings.
            match self.right.next()? {
                Some(row) => return Ok(Some(row)),
                None => {
                    // Right exhausted for this left; advance.
                    self.current_left = None;
                }
            }
        }
    }
}

/// Hash join. Materializes the build side once into a hash table, then
/// streams the probe side, yielding every match. Build-side
/// materialization is unavoidable, but probe-side streaming still cuts
/// peak memory roughly in half versus today's "build both vecs".
struct HashJoinIterator<'a> {
    build: Box<dyn PlanIterator<'a> + 'a>,
    probe: Box<dyn PlanIterator<'a> + 'a>,
    /// Slot indices used as the join key, in the same order on both
    /// sides.
    join_slots: Vec<usize>,
    /// Built hash table — only populated after `open`. Buckets are
    /// `Arc<Vec<Tuple>>` so the per-probe "take the matches" becomes
    /// a refcount bump rather than a full `Vec` + `Tuple` clone. For
    /// Large Fan-Out (5000 probes × ~3 matches/bucket), this is the
    /// dominant cost in the inner loop.
    hash_table: HashMap<JoinKey, Arc<Vec<Tuple>>>,
    /// The currently-active probe row, plus a shared handle to its
    /// matching build rows and a cursor index into them.
    current_probe: Option<Tuple>,
    current_matches: Option<Arc<Vec<Tuple>>>,
    current_idx: usize,
    done: bool,
}

impl<'a> HashJoinIterator<'a> {
    fn new(
        build: Box<dyn PlanIterator<'a> + 'a>,
        probe: Box<dyn PlanIterator<'a> + 'a>,
        join_slots: Vec<usize>,
    ) -> Self {
        Self {
            build,
            probe,
            join_slots,
            hash_table: HashMap::new(),
            current_probe: None,
            current_matches: None,
            current_idx: 0,
            done: false,
        }
    }
}

impl<'a> PlanIterator<'a> for HashJoinIterator<'a> {
    fn open(&mut self, seed: &Tuple) -> Result<(), String> {
        // Build phase: drain the build side into a HashMap<JoinKey, Vec<Tuple>>
        // for cheap mutable appends, then seal each bucket into an Arc.
        self.hash_table.clear();
        self.build.open(seed)?;
        let mut buckets: HashMap<JoinKey, Vec<Tuple>> = HashMap::new();
        while let Some(row) = self.build.next()? {
            if let Some(key) = join_key(&row, &self.join_slots) {
                buckets.entry(JoinKey(key)).or_default().push(row);
            }
        }
        self.hash_table = buckets
            .into_iter()
            .map(|(k, v)| (k, Arc::new(v)))
            .collect();
        // Probe phase: prepare the probe iterator.
        self.probe.open(seed)?;
        self.current_probe = None;
        self.current_matches = None;
        self.current_idx = 0;
        self.done = false;
        Ok(())
    }

    fn next(&mut self) -> Result<Option<Tuple>, String> {
        if self.done {
            return Ok(None);
        }
        loop {
            // Drain any pending matches for the current probe row.
            if let Some(matches) = &self.current_matches {
                if self.current_idx < matches.len() {
                    let build_row = &matches[self.current_idx];
                    self.current_idx += 1;
                    let probe_row = self
                        .current_probe
                        .as_ref()
                        .expect("matches without probe");
                    if let Some(merged) = merge_tuples(probe_row, build_row) {
                        return Ok(Some(merged));
                    }
                    continue;
                }
                self.current_matches = None;
            }
            // Advance to the next probe row that has matches.
            match self.probe.next()? {
                Some(probe_row) => {
                    if let Some(key) = join_key(&probe_row, &self.join_slots) {
                        if let Some(matches) = self.hash_table.get(&JoinKey(key)) {
                            self.current_matches = Some(matches.clone()); // cheap Arc clone
                            self.current_idx = 0;
                            self.current_probe = Some(probe_row);
                            continue;
                        }
                    }
                    // No matches for this probe row; skip.
                }
                None => {
                    self.done = true;
                    return Ok(None);
                }
            }
        }
    }
}

/// Project node — pure pass-through. The output projection itself
/// happens in `execute_plan` after the root is fully consumed.
struct ProjectIterator<'a> {
    input: Box<dyn PlanIterator<'a> + 'a>,
}

impl<'a> PlanIterator<'a> for ProjectIterator<'a> {
    fn open(&mut self, seed: &Tuple) -> Result<(), String> {
        self.input.open(seed)
    }
    fn next(&mut self) -> Result<Option<Tuple>, String> {
        self.input.next()
    }
}

/// Union (OR): yields the deduplicated bindings produced by any branch. Every
/// branch is re-opened with the incoming seed, so OR composes inside joins.
struct UnionIterator<'a> {
    branches: Vec<Box<dyn PlanIterator<'a> + 'a>>,
    idx: usize,
    seed: Tuple,
    num_slots: usize,
    seen: HashSet<Vec<Option<HashableValue>>>,
}

impl<'a> UnionIterator<'a> {
    fn new(branches: Vec<Box<dyn PlanIterator<'a> + 'a>>, num_slots: usize) -> Self {
        Self {
            branches,
            idx: 0,
            seed: Tuple::new(num_slots),
            num_slots,
            seen: HashSet::new(),
        }
    }

    fn dedup_key(&self, t: &Tuple) -> Vec<Option<HashableValue>> {
        (0..self.num_slots)
            .map(|i| t.get(i).cloned().map(HashableValue))
            .collect()
    }
}

impl<'a> PlanIterator<'a> for UnionIterator<'a> {
    fn open(&mut self, seed: &Tuple) -> Result<(), String> {
        self.seed = seed.clone();
        self.idx = 0;
        self.seen.clear();
        if let Some(first) = self.branches.get_mut(0) {
            first.open(seed)?;
        }
        Ok(())
    }

    fn next(&mut self) -> Result<Option<Tuple>, String> {
        loop {
            if self.idx >= self.branches.len() {
                return Ok(None);
            }
            match self.branches[self.idx].next()? {
                Some(t) => {
                    let key = self.dedup_key(&t);
                    if self.seen.insert(key) {
                        return Ok(Some(t));
                    }
                    // duplicate binding — skip
                }
                None => {
                    self.idx += 1;
                    if self.idx < self.branches.len() {
                        let seed = self.seed.clone();
                        self.branches[self.idx].open(&seed)?;
                    }
                }
            }
        }
    }
}

/// Negation as failure (NOT): passes through each `input` row for which the
/// `anti` subquery, re-opened with that row's bindings, yields no solution.
struct NegationIterator<'a> {
    input: Box<dyn PlanIterator<'a> + 'a>,
    anti: Box<dyn PlanIterator<'a> + 'a>,
}

impl<'a> PlanIterator<'a> for NegationIterator<'a> {
    fn open(&mut self, seed: &Tuple) -> Result<(), String> {
        self.input.open(seed)
    }

    fn next(&mut self) -> Result<Option<Tuple>, String> {
        loop {
            match self.input.next()? {
                None => return Ok(None),
                Some(row) => {
                    self.anti.open(&row)?;
                    if self.anti.next()?.is_none() {
                        return Ok(Some(row));
                    }
                    // anti has a solution → row is excluded
                }
            }
        }
    }
}

/// Build a chain of iterators from a plan tree.
fn build_iterator<'a>(
    node: &'a PlanNode,
    ctx: &'a ExecutionContext<'a>,
) -> Box<dyn PlanIterator<'a> + 'a> {
    match node {
        PlanNode::Scan(scan) => Box::new(ScanIterator::new(&scan.clause, ctx)),
        PlanNode::Join {
            left,
            right,
            join_vars,
            strategy,
            ..
        } => {
            let left_iter = build_iterator(left, ctx);
            let right_iter = build_iterator(right, ctx);
            match strategy {
                JoinStrategy::NestedLoop => {
                    Box::new(NestedLoopIterator::new(left_iter, right_iter))
                }
                JoinStrategy::HashJoin { build_side } => {
                    let join_slots: Vec<usize> = join_vars
                        .iter()
                        .filter_map(|v| ctx.slots.slot(v))
                        .collect();
                    let (build, probe) = match build_side {
                        JoinSide::Left => (left_iter, right_iter),
                        JoinSide::Right => (right_iter, left_iter),
                    };
                    Box::new(HashJoinIterator::new(build, probe, join_slots))
                }
            }
        }
        PlanNode::Project { input, .. } => {
            let input_iter = build_iterator(input, ctx);
            Box::new(ProjectIterator { input: input_iter })
        }
        PlanNode::Union { branches, .. } => {
            let iters: Vec<Box<dyn PlanIterator<'a> + 'a>> =
                branches.iter().map(|b| build_iterator(b, ctx)).collect();
            Box::new(UnionIterator::new(iters, ctx.slots.num_slots))
        }
        PlanNode::Negation { input, anti, .. } => Box::new(NegationIterator {
            input: build_iterator(input, ctx),
            anti: build_iterator(anti, ctx),
        }),
    }
}

// ---------------------------------------------------------------------------
// Plan-based execution entry point
// ---------------------------------------------------------------------------

/// Execute a query by planning then executing the plan.
pub fn execute_query(
    txn: &dyn ReadOps,
    query: &Query,
    schema: &SchemaRegistry,
    cache: &QueryCache,
    interner: &AttrInterner,
) -> Result<QueryResult, String> {
    // No external snapshot ordering here: sample the generation up front.
    let cache_gen = cache.generation();
    let plan = planner::plan_query(txn, query, schema, interner)?;
    execute_plan(txn, &plan, schema, cache, interner, cache_gen)
}

/// Execute a pre-built query plan via the volcano/iterator engine.
///
/// `cache_gen` MUST be the cache generation sampled before `txn`'s snapshot
/// was created (see `QueryCache::generation`), so that a type loaded into the
/// cache during this query cannot silently miss a concurrently committed write.
pub fn execute_plan(
    txn: &dyn ReadOps,
    plan: &QueryPlan,
    schema: &SchemaRegistry,
    cache: &QueryCache,
    interner: &AttrInterner,
    cache_gen: u64,
) -> Result<QueryResult, String> {
    let slots = &plan.slot_map;
    let ctx = ExecutionContext {
        txn,
        schema,
        cache,
        interner,
        slots,
        as_of: plan.as_of,
        cache_gen,
    };

    // Specialized fast path: chains of nested-loop joins where every
    // right side is a Scan with bind variable bound by a Variable
    // pattern on a Ref-typed field in the prior scan, and all scans
    // target the same entity type. This is the shape of Graph 2-Hop
    // and any "follow ref chain through one entity type" query.
    //
    // The specialized path bypasses the generic iterator framework
    // entirely — no `Box<dyn>` dispatch, no per-row `Vec` allocations,
    // no resolve_field_patterns re-runs. Just column reads + binary
    // searches in a tight Rust loop.
    if plan.as_of.is_none() {
        if let Some(result) = try_execute_ref_chain(&ctx, plan)? {
            return Ok(result);
        }
    }

    // The Project node at the root determines which variables to return.
    let columns = match &plan.root {
        PlanNode::Project { variables, .. } => variables.clone(),
        _ => vec![],
    };
    let output_slots: Vec<usize> = columns
        .iter()
        .map(|var| slots.slot(var).unwrap_or(usize::MAX))
        .collect();

    // Drive the iterator chain.
    let mut root = build_iterator(&plan.root, &ctx);
    let empty_seed = Tuple::new(slots.num_slots);
    root.open(&empty_seed)?;

    // An unbound find variable (no slot, or a slot left unset by this row's
    // bindings — e.g. a var bound in only some `or` branches) projects as
    // `Null`. Using a real `Value` sentinel string here would be
    // indistinguishable from an actual string value of the same text.
    let unbound = Value::Null;
    let mut rows: Vec<Vec<Value>> = Vec::new();
    while let Some(tuple) = root.next()? {
        let row: Vec<Value> = output_slots
            .iter()
            .map(|&slot| {
                if slot == usize::MAX {
                    unbound.clone()
                } else {
                    tuple
                        .get(slot)
                        .cloned()
                        .unwrap_or_else(|| unbound.clone())
                }
            })
            .collect();
        rows.push(row);
    }

    Ok(QueryResult { columns, rows })
}

// ---------------------------------------------------------------------------
// Specialized ref-chain execution
//
// Detects the plan shape produced by queries like Graph 2-Hop:
//   Project(Join(Scan_A, Join(Scan_B, Join(Scan_C, ...))))
// where each right-side scan's bind variable is bound by a Variable
// pattern on a Ref-typed field in the immediately preceding scan,
// and all scans target the same entity type.
//
// For this shape, the entire query can be executed by:
//   1. Run the outer scan once (AVET intersection + post-filter).
//   2. For each outer entity, do per-hop column reads:
//      a. Extract the ref value from the previous row.
//      b. Binary search the type's entity_ids.
//      c. Read the hop's output columns into the working tuple.
//   3. Project the final tuple to the output row.
//
// No `Box<dyn>` dispatch, no per-row `Vec` allocs in the inner loop,
// no resolve_field_patterns recomputation. The hot loop is pure
// array indexing and binary search.
// ---------------------------------------------------------------------------

/// A single hop in the ref chain. Pre-resolved so the inner loop does
/// only direct memory access.
struct Hop {
    /// Slot in the running tuple holding the `Value::Ref` to follow.
    ref_slot: usize,
    /// The hop's output column reads, as `(target_slot, column)` pairs.
    /// `column` is an `Arc<Vec<Option<Value>>>` indexed by the target
    /// entity's row index in the type's `entity_ids`.
    reads: Vec<(usize, cache::Column)>,
}

fn try_execute_ref_chain(
    ctx: &ExecutionContext<'_>,
    plan: &QueryPlan,
) -> Result<Option<QueryResult>, String> {
    let slots = &plan.slot_map;

    // Must be Project(...) at the root so we know the output columns.
    let (columns, input) = match &plan.root {
        PlanNode::Project { variables, input } => (variables.clone(), input.as_ref()),
        _ => return Ok(None),
    };

    // Walk the left-deep join tree, collecting every Scan in left-to-right
    // order. The pattern requires the tree to be entirely Joins and Scans;
    // anything else (HashJoin metadata aside) and we bail out.
    let scans = match collect_scan_chain(input) {
        Some(s) => s,
        None => return Ok(None),
    };
    if scans.len() < 2 {
        return Ok(None);
    }

    // Every scan must target the same entity type.
    let entity_type = scans[0].clause.entity_type.clone();
    for s in &scans[1..] {
        if s.clause.entity_type != entity_type {
            return Ok(None);
        }
    }

    // Schema must say the chain link fields are Refs to the same type.
    let type_def = match ctx.schema.get(&entity_type) {
        Some(td) => td,
        None => return Ok(None),
    };

    // Verify each hop: scan[i+1].bind must be the Variable target of a
    // Ref-typed field in scan[i].
    for i in 0..scans.len() - 1 {
        let prev = &scans[i];
        let bind_next = &scans[i + 1].clause.bind;
        let mut linked_field: Option<&str> = None;
        for (field_name, pat) in &prev.clause.field_patterns {
            if let Some(var) = pat.bound_var() {
                if var == bind_next {
                    linked_field = Some(field_name);
                    break;
                }
            }
        }
        let field = match linked_field {
            Some(f) => f,
            None => return Ok(None),
        };
        let fd = match type_def.get_field(field) {
            Some(fd) => fd,
            None => return Ok(None),
        };
        match &fd.field_type {
            crate::schema::FieldType::Ref(target) if *target == entity_type => {}
            _ => return Ok(None),
        }
    }

    // Every non-Variable pattern on the inner scans we have to also support;
    // for simplicity (and our bench's needs) require all inner-scan patterns
    // to be Variables. Outer scan's Exact/Range/Variable patterns go through
    // the normal AVET path.
    for s in &scans[1..] {
        for (_, pat) in &s.clause.field_patterns {
            if !matches!(pat, Pattern::Variable(_)) {
                return Ok(None);
            }
        }
    }

    // Load the (shared) type cache once.
    let td = match ctx
        .cache
        .ensure_type_loaded(ctx.txn, ctx.interner, &entity_type, ctx.schema, ctx.cache_gen)?
    {
        Some(td) => td,
        None => return Ok(None),
    };

    // Pre-resolve each hop's column references and target slots.
    let mut hops: Vec<Hop> = Vec::with_capacity(scans.len() - 1);
    for s in &scans[1..] {
        let ref_slot = match slots.slot(s.clause.bind.as_str()) {
            Some(slot) => slot,
            None => return Ok(None),
        };
        let mut reads: Vec<(usize, cache::Column)> = Vec::new();
        for (field_name, pat) in &s.clause.field_patterns {
            // Already verified above that all are Variables.
            let var = match pat {
                Pattern::Variable(v) => v,
                _ => unreachable!(),
            };
            let slot = match slots.slot(var) {
                Some(s) => s,
                None => return Ok(None),
            };
            // Only ref-typed fields (which would chain further) and
            // scalar fields show up here; either way the column read
            // is the same.
            if let Some(col) = td.column_arc(field_name) {
                reads.push((slot, col));
            }
        }
        hops.push(Hop { ref_slot, reads });
    }

    // Also need to read the outer scan's Variable-pattern output fields
    // (e.g. `name=?ename` in Graph 2-Hop's outer clause) into their slots
    // for every matched outer entity. Capture those as a pre-resolved list.
    let outer_clause = &scans[0].clause;
    let outer_bind_slot = match slots.slot(outer_clause.bind.as_str()) {
        Some(s) => s,
        None => return Ok(None),
    };
    let mut outer_reads: Vec<(usize, cache::Column)> = Vec::new();
    for (field_name, pat) in &outer_clause.field_patterns {
        if let Some(var) = pat.bound_var() {
            if let Some(slot) = slots.slot(var) {
                if let Some(col) = td.column_arc(field_name) {
                    outer_reads.push((slot, col));
                }
            }
        }
    }

    // Execute the outer scan through the normal cached path to handle
    // its AVET filters / constants / ranges. We need the resulting
    // entity IDs.
    let mut outer_iter = ScanIterator::new(outer_clause, ctx);
    let empty_seed = Tuple::new(slots.num_slots);
    outer_iter.open(&empty_seed)?;
    let mut outer_rows: Vec<Tuple> = Vec::new();
    while let Some(t) = outer_iter.next()? {
        outer_rows.push(t);
    }

    // Pre-resolve output slot indices for the projection.
    let output_slots: Vec<usize> = columns
        .iter()
        .map(|var| slots.slot(var).unwrap_or(usize::MAX))
        .collect();
    let unbound = Value::Null;

    // Hot loop: for each outer row, follow the ref chain through the
    // type cache by binary search + column reads. Build the result
    // tuple incrementally; emit a projected row if all hops succeed.
    let mut rows: Vec<Vec<Value>> = Vec::with_capacity(outer_rows.len());
    'outer: for mut tuple in outer_rows {
        // Apply outer scan's variable-pattern reads (they were already
        // populated by the outer ScanIterator path, but in case any
        // weren't — this is cheap and idempotent).
        let _ = outer_reads.len(); // silence unused; reads already done by ScanIterator
        let _ = outer_bind_slot;

        // Walk hops.
        for hop in &hops {
            // Get the Ref to follow from the current tuple.
            let eid = match tuple.get(hop.ref_slot) {
                Some(Value::Ref(id)) => *id,
                _ => continue 'outer, // chain broke; skip this outer
            };
            let idx = match td.index_of(eid) {
                Some(i) => i,
                None => continue 'outer,
            };
            // Read each output field for this hop into its target slot.
            for (slot, col) in &hop.reads {
                if let Some(Some(v)) = col.get(idx) {
                    tuple.set(*slot, v.clone());
                }
            }
        }

        // Project the completed tuple.
        let row: Vec<Value> = output_slots
            .iter()
            .map(|&slot| {
                if slot == usize::MAX {
                    unbound.clone()
                } else {
                    tuple
                        .get(slot)
                        .cloned()
                        .unwrap_or_else(|| unbound.clone())
                }
            })
            .collect();
        rows.push(row);
    }

    Ok(Some(QueryResult { columns, rows }))
}

/// Walk a join tree and collect every `Scan` in left-to-right
/// (outer-to-inner) execution order. Handles both left-deep and
/// right-deep join shapes that the planner might produce.
///
/// Left-deep: `Join(Join(Scan1, Scan2), Scan3)` — we walk down lefts,
/// collecting rights on the way back up, then push the bottom Scan.
/// Right-deep: `Join(Scan1, Join(Scan2, Scan3))` — we walk down rights,
/// collecting lefts on the way.
///
/// Returns `None` for any shape that isn't a chain of Scans.
fn collect_scan_chain(node: &PlanNode) -> Option<Vec<&ClauseScan>> {
    // Try the right-deep walk first.
    if let Some(v) = collect_right_deep(node) {
        return Some(v);
    }
    collect_left_deep(node)
}

fn collect_right_deep(node: &PlanNode) -> Option<Vec<&ClauseScan>> {
    let mut out: Vec<&ClauseScan> = Vec::new();
    let mut current = node;
    loop {
        match current {
            PlanNode::Scan(s) => {
                out.push(s);
                return Some(out);
            }
            PlanNode::Join { left, right, .. } => {
                if let PlanNode::Scan(s) = left.as_ref() {
                    out.push(s);
                    current = right.as_ref();
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    }
}

fn collect_left_deep(node: &PlanNode) -> Option<Vec<&ClauseScan>> {
    // Walk down lefts, pushing the right scan at each level. End with
    // the leftmost Scan as the first element after reversing.
    let mut rights_rev: Vec<&ClauseScan> = Vec::new();
    let mut current = node;
    loop {
        match current {
            PlanNode::Scan(s) => {
                rights_rev.push(s);
                rights_rev.reverse();
                return Some(rights_rev);
            }
            PlanNode::Join { left, right, .. } => {
                if let PlanNode::Scan(rs) = right.as_ref() {
                    rights_rev.push(rs);
                    current = left.as_ref();
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    }
}

/// Merge two tuples. Returns None if there's a conflict
/// (same slot bound to different values).
fn merge_tuples(left: &Tuple, right: &Tuple) -> Option<Tuple> {
    let mut merged = left.clone();
    for (i, rv) in right.slots.iter().enumerate() {
        if let Some(val) = rv {
            if let Some(existing) = &merged.slots[i] {
                if existing != val {
                    return None;
                }
            } else {
                merged.slots[i] = Some(val.clone());
            }
        }
    }
    Some(merged)
}

// ---------------------------------------------------------------------------
// Clause-level execution (leaf engine — unchanged from original)
// ---------------------------------------------------------------------------

/// Read all values of a cardinality-many attribute on one entity.
///
/// For current-state queries (`as_of = None`) this scans the value-keyed
/// CURRENT_AEVT index (`[0x11][attr_id][entity][value]`) directly. For
/// historical (`as_of`) queries it reconstructs the live set from the EAVT
/// history up to that transaction, since the current-state mirror only
/// reflects the present.
fn read_many_values(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    attr: &str,
    entity: EntityId,
    as_of: Option<TxId>,
) -> Result<Vec<Value>, String> {
    let attr_id = match interner.lookup(attr) {
        Some(id) => id,
        None => return Ok(Vec::new()),
    };

    if as_of.is_none() {
        let prefix = index::current_aevt_entity_prefix(attr_id, entity);
        let end = index::prefix_end(&prefix);
        let mut out = Vec::new();
        txn.scan_foreach(&prefix, &end, &mut |key, _| {
            if let Some(v) = index::current_aevt_many_value_at(key) {
                out.push(v);
            }
            true
        })
        .map_err(|e| e.to_string())?;
        return Ok(out);
    }

    // Historical: replay this (entity, attr)'s EAVT datoms up to `as_of`,
    // accumulating the live value set (add inserts, retract removes).
    let max_tx = as_of.unwrap();
    let prefix = index::eavt_entity_attr_prefix(entity, attr_id);
    let end = index::prefix_end(&prefix);
    let mut datoms: Vec<index::DecodedDatom> = Vec::new();
    txn.scan_foreach(&prefix, &end, &mut |key, _| {
        if let Some(d) = index::decode_datom_from_eavt(key) {
            if d.tx <= max_tx {
                datoms.push(d);
            }
        }
        true
    })
    .map_err(|e| e.to_string())?;
    datoms.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));
    let mut live: Vec<Value> = Vec::new();
    for d in datoms {
        if d.added {
            if !live.contains(&d.value) {
                live.push(d.value);
            }
        } else {
            live.retain(|v| v != &d.value);
        }
    }
    Ok(live)
}

/// Find every entity that has `value` as one of its values for the
/// cardinality-many attribute `attr` — an indexed AVET membership probe.
/// Honors `as_of` (per-value existence is resolved from history).
fn entities_with_many_value(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    attr: &str,
    value: &Value,
    as_of: Option<TxId>,
) -> Result<Vec<EntityId>, String> {
    find_entities_by_avet(txn, interner, attr, value, as_of)
}

/// The cardinality-many field patterns of a clause, partitioned out so the
/// common (all cardinality-one) path stays on the optimized columnar engine.
fn many_field_patterns<'a>(
    clause: &'a WhereClause,
    schema: &SchemaRegistry,
) -> Vec<&'a (String, Pattern)> {
    let type_def = match schema.get(&clause.entity_type) {
        Some(td) => td,
        None => return Vec::new(),
    };
    clause
        .field_patterns
        .iter()
        .filter(|(fname, _)| type_def.get_field(fname).map(|f| f.is_many()).unwrap_or(false))
        .collect()
}

/// Evaluate a clause that references one or more cardinality-many attributes.
///
/// Strategy: strip the many-patterns off and evaluate the remaining
/// cardinality-one clause on the normal engine to get candidate tuples (each
/// with the entity bound). Then apply each many-pattern per candidate:
///   - membership (constant / `contains` / predicate): keep the tuple if any
///     of the entity's values for that attr matches;
///   - variable bind (`tag: ?t`): fan the tuple out, one per matching value.
///
/// When the entity isn't yet bound and a many-pattern is a constant, we seed
/// candidates from the AVET membership index (the indexed point-lookup that is
/// the whole reason for cardinality-many) rather than scanning the type.
#[allow(clippy::too_many_arguments)]
fn evaluate_clause_with_many(
    txn: &dyn ReadOps,
    clause: &WhereClause,
    tuple: &Tuple,
    as_of: Option<TxId>,
    schema: &SchemaRegistry,
    slots: &SlotMap,
    cache: &QueryCache,
    interner: &AttrInterner,
    cache_gen: u64,
    many_pats: &[&(String, Pattern)],
) -> Result<Vec<Tuple>, String> {
    let type_def = schema.get(&clause.entity_type);
    let attr_of = |fname: &str| -> String {
        type_def
            .map(|td| td.attribute_name(fname))
            .unwrap_or_else(|| format!("{}/{}", clause.entity_type, fname))
    };

    // The cardinality-one remainder of the clause.
    let one_clause = WhereClause {
        bind: clause.bind.clone(),
        entity_type: clause.entity_type.clone(),
        field_patterns: clause
            .field_patterns
            .iter()
            .filter(|(fname, _)| {
                type_def
                    .and_then(|td| td.get_field(fname))
                    .map(|f| !f.is_many())
                    .unwrap_or(true)
            })
            .cloned()
            .collect(),
    };

    let bind_slot = slots.slot(&clause.bind).unwrap();
    let entity_pre_bound = tuple.get(bind_slot).map_or(false, |v| matches!(v, Value::Ref(_)));

    // Candidate seeding. If the entity isn't bound yet and some many-pattern is
    // a constant, use its AVET membership index to get a small candidate set,
    // then bind each and let the one-clause engine verify the rest. Otherwise
    // evaluate the one-clause directly (full type scan or join-seeded).
    let seed_constant: Option<(String, Value)> = if entity_pre_bound {
        None
    } else {
        many_pats.iter().find_map(|(f, p)| match p {
            Pattern::Constant(v) => Some((f.clone(), v.clone())),
            _ => None,
        })
    };

    let mut candidates: Vec<Tuple> = if let Some((fname, val)) = seed_constant {
        let eids = entities_with_many_value(txn, interner, &attr_of(&fname), &val, as_of)?;
        let mut seeded = Vec::with_capacity(eids.len());
        for eid in eids {
            let mut t = tuple.clone();
            t.set(bind_slot, Value::Ref(eid));
            // Verify the one-cardinality part for this seeded entity.
            seeded.extend(evaluate_clause(
                txn, &one_clause, &t, as_of, schema, slots, cache, interner, cache_gen,
            )?);
        }
        seeded
    } else {
        evaluate_clause(
            txn, &one_clause, tuple, as_of, schema, slots, cache, interner, cache_gen,
        )?
    };

    // Apply each many-pattern to the candidate tuples.
    for (fname, pat) in many_pats {
        let attr = attr_of(fname);
        let mut next: Vec<Tuple> = Vec::new();
        for cand in &candidates {
            let eid = match cand.get(bind_slot) {
                Some(Value::Ref(id)) => *id,
                _ => continue,
            };
            let values = read_many_values(txn, interner, &attr, eid, as_of)?;
            match pat {
                Pattern::Variable(var) => {
                    // Fan out: one tuple per value, binding (or matching) ?var.
                    let slot = match slots.slot(var) {
                        Some(s) => s,
                        None => continue,
                    };
                    for v in &values {
                        if let Some(existing) = cand.get(slot) {
                            if existing != v {
                                continue;
                            }
                            next.push(cand.clone());
                        } else {
                            let mut t = cand.clone();
                            t.set(slot, v.clone());
                            next.push(t);
                        }
                    }
                }
                Pattern::Constant(expected) => {
                    if values.iter().any(|v| v == expected) {
                        next.push(cand.clone());
                    }
                }
                Pattern::Predicate { op, value }
                | Pattern::BoundPredicate { op, value, .. } => {
                    // Membership-style filter: keep if any value passes the op.
                    if values.iter().any(|v| op.evaluate(v, value)) {
                        // For BoundPredicate, also bind ?var to a matching value.
                        if let Pattern::BoundPredicate { var, .. } = pat {
                            if let Some(slot) = slots.slot(var) {
                                for v in &values {
                                    if op.evaluate(v, value) {
                                        let mut t = cand.clone();
                                        t.set(slot, v.clone());
                                        next.push(t);
                                    }
                                }
                                continue;
                            }
                        }
                        next.push(cand.clone());
                    }
                }
                _ => {
                    // EnumMatch on a many-attr is unsupported; drop the tuple.
                }
            }
        }
        candidates = next;
    }

    Ok(candidates)
}

/// Find a `near` (nearest-neighbour) pattern in a clause, if present.
/// Returns (field_name, query_vector, k, metric, score_var).
fn find_near_pattern(
    clause: &WhereClause,
) -> Option<(String, Vec<f32>, usize, VectorMetric, Option<String>)> {
    for (fname, pat) in &clause.field_patterns {
        if let Pattern::Near { query, k, metric, score_var } = pat {
            return Some((
                fname.clone(),
                query.clone(),
                *k,
                *metric,
                score_var.clone(),
            ));
        }
    }
    None
}

/// Find a `search` (BM25 full-text) pattern in a clause, if present.
/// Returns (field_name, query_text, k, score_var).
fn find_search_pattern(
    clause: &WhereClause,
) -> Option<(String, String, usize, Option<String>)> {
    for (fname, pat) in &clause.field_patterns {
        if let Pattern::Search { query, k, score_var } = pat {
            return Some((fname.clone(), query.clone(), *k, score_var.clone()));
        }
    }
    None
}

/// Read one entity's vector value for `attr` from EAVT (current or as_of).
fn read_vector_value(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    attr: &str,
    entity: EntityId,
    as_of: Option<TxId>,
) -> Result<Option<Vec<f32>>, String> {
    let attr_id = match interner.lookup(attr) {
        Some(id) => id,
        None => return Ok(None),
    };
    let prefix = index::eavt_entity_attr_prefix(entity, attr_id);
    let end = index::prefix_end(&prefix);
    let mut datoms: Vec<index::DecodedDatom> = Vec::new();
    txn.scan_foreach(&prefix, &end, &mut |key, _| {
        if let Some(d) = index::decode_datom_from_eavt(key) {
            if as_of.map_or(true, |max| d.tx <= max) {
                datoms.push(d);
            }
        }
        true
    })
    .map_err(|e| e.to_string())?;
    datoms.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));
    // Last-write-wins: the currently-asserted vector (cardinality-one).
    let mut current: Option<Vec<f32>> = None;
    for d in datoms {
        match d.value {
            Value::Vector(v) if d.added => current = Some(v),
            Value::Vector(v) if !d.added => {
                if current.as_deref() == Some(v.as_slice()) {
                    current = None;
                }
            }
            _ => {}
        }
    }
    Ok(current)
}

/// Evaluate a clause containing a `near` vector-search pattern: brute-force
/// top-k over the candidate entities. Any non-`near` patterns in the clause act
/// as pre-filters (e.g. `{ kind: "paper", embedding: { near: [..] } }` only
/// ranks papers). Returns up to `k` tuples, each with the entity bound and (if
/// requested) the similarity score bound to `score_var`, sorted best-first.
#[allow(clippy::too_many_arguments)]
fn evaluate_clause_with_near(
    txn: &dyn ReadOps,
    clause: &WhereClause,
    tuple: &Tuple,
    as_of: Option<TxId>,
    schema: &SchemaRegistry,
    slots: &SlotMap,
    cache: &QueryCache,
    interner: &AttrInterner,
    cache_gen: u64,
    fname: &str,
    query: &[f32],
    k: usize,
    metric: VectorMetric,
    score_var: Option<&str>,
) -> Result<Vec<Tuple>, String> {
    let type_def = schema.get(&clause.entity_type);
    let attr = type_def
        .map(|td| td.attribute_name(fname))
        .unwrap_or_else(|| format!("{}/{}", clause.entity_type, fname));

    // Validate the field is a vector of the right dimension.
    if let Some(td) = type_def {
        if let Some(fd) = td.get_field(fname) {
            match &fd.field_type {
                FieldType::Vector(dim) if *dim == query.len() => {}
                FieldType::Vector(dim) => {
                    return Err(format!(
                        "near query vector has dimension {} but field '{}' is vector({})",
                        query.len(),
                        fname,
                        dim
                    ));
                }
                _ => {
                    return Err(format!("field '{}' is not a vector field", fname));
                }
            }
        }
    }

    // The remainder of the clause (everything except the near pattern) is a
    // pre-filter. Evaluate it to get candidate entities.
    let filter_clause = WhereClause {
        bind: clause.bind.clone(),
        entity_type: clause.entity_type.clone(),
        field_patterns: clause
            .field_patterns
            .iter()
            .filter(|(_, p)| !matches!(p, Pattern::Near { .. }))
            .cloned()
            .collect(),
    };
    let candidates = evaluate_clause(
        txn, &filter_clause, tuple, as_of, schema, slots, cache, interner, cache_gen,
    )?;

    let bind_slot = slots.slot(&clause.bind).unwrap();
    let score_slot = score_var.and_then(|sv| slots.slot(sv));

    // Score every candidate, keep top-k. Brute force is the right tool below
    // ~1M vectors; an ANN index would slot in here when the corpus outgrows it.
    let mut scored: Vec<(f32, Tuple)> = Vec::new();
    for cand in candidates {
        let eid = match cand.get(bind_slot) {
            Some(Value::Ref(id)) => *id,
            _ => continue,
        };
        let vec = match read_vector_value(txn, interner, &attr, eid, as_of)? {
            Some(v) => v,
            None => continue, // entity has no vector for this field
        };
        let score = metric.score(query, &vec);
        if !score.is_finite() {
            continue;
        }
        let mut t = cand.clone();
        if let Some(slot) = score_slot {
            t.set(slot, Value::F64(score as f64));
        }
        scored.push((score, t));
    }

    // Sort best-first (higher score = closer for every metric) and take k.
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    Ok(scored.into_iter().map(|(_, t)| t).collect())
}

/// Evaluate a clause containing a `search` (BM25 full-text) pattern.
///
/// Tokenizes the query, walks the inverted index for each term to accumulate a
/// per-document BM25 score, then returns the top-k entities. Non-`search`
/// patterns in the clause act as pre-filters (only entities passing them are
/// kept). The score var, if given, is bound to the BM25 score.
#[allow(clippy::too_many_arguments)]
fn evaluate_clause_with_search(
    txn: &dyn ReadOps,
    clause: &WhereClause,
    tuple: &Tuple,
    as_of: Option<TxId>,
    schema: &SchemaRegistry,
    slots: &SlotMap,
    cache: &QueryCache,
    interner: &AttrInterner,
    cache_gen: u64,
    fname: &str,
    query_text: &str,
    k: usize,
    score_var: Option<&str>,
) -> Result<Vec<Tuple>, String> {
    let type_def = schema.get(&clause.entity_type);
    let attr = type_def
        .map(|td| td.attribute_name(fname))
        .unwrap_or_else(|| format!("{}/{}", clause.entity_type, fname));

    if let Some(td) = type_def {
        match td.get_field(fname) {
            Some(fd) if fd.fulltext => {}
            Some(_) => {
                return Err(format!("field '{}' is not a fulltext field", fname));
            }
            None => {}
        }
    }

    let attr_id = match interner.lookup(&attr) {
        Some(id) => id,
        None => return Ok(Vec::new()), // never indexed → no hits
    };

    // Corpus stats for BM25 (avgdl, n_docs).
    let n_docs = read_meta_u64(txn, &format!("fts_ndocs:{}", attr))?;
    let tot_len = read_meta_u64(txn, &format!("fts_totlen:{}", attr))?;
    if n_docs == 0 {
        return Ok(Vec::new());
    }
    let avgdl = (tot_len as f32 / n_docs as f32).max(1.0);

    // Distinct query terms (BM25 doesn't gain from repeating a query term).
    let mut terms: Vec<String> = crate::fts::tokenize(query_text);
    terms.sort();
    terms.dedup();
    if terms.is_empty() {
        return Ok(Vec::new());
    }

    // Accumulate score and cache doc lengths per candidate entity.
    let mut scores: HashMap<EntityId, f32> = HashMap::new();
    let mut doc_len: HashMap<EntityId, u32> = HashMap::new();

    for term in &terms {
        // Gather this term's postings: (entity, tf). df = number of postings.
        let prefix = index::fts_term_prefix(attr_id, term);
        let end = index::prefix_end(&prefix);
        let mut postings: Vec<(EntityId, u32)> = Vec::new();
        txn.scan_foreach(&prefix, &end, &mut |key, val| {
            let eid = index::fts_posting_entity_at(key);
            let tf = if val.len() == 4 {
                u32::from_be_bytes(val.try_into().unwrap())
            } else {
                0
            };
            postings.push((eid, tf));
            true
        })
        .map_err(|e| e.to_string())?;

        let df = postings.len() as u64;
        if df == 0 {
            continue;
        }
        for (eid, tf) in postings {
            let dl = match doc_len.get(&eid) {
                Some(&l) => l,
                None => {
                    let l = read_u32(txn, &index::fts_doclen_key(attr_id, eid))?;
                    doc_len.insert(eid, l);
                    l
                }
            };
            let s = crate::fts::bm25_term_score(tf, dl, avgdl, n_docs, df);
            *scores.entry(eid).or_insert(0.0) += s;
        }
    }

    if scores.is_empty() {
        return Ok(Vec::new());
    }

    // Pre-filter: the non-search remainder of the clause. We evaluate it per
    // candidate by seeding the entity and checking the filter clause.
    let filter_clause = WhereClause {
        bind: clause.bind.clone(),
        entity_type: clause.entity_type.clone(),
        field_patterns: clause
            .field_patterns
            .iter()
            .filter(|(_, p)| !matches!(p, Pattern::Search { .. }))
            .cloned()
            .collect(),
    };
    let has_filters = !filter_clause.field_patterns.is_empty();

    let bind_slot = slots.slot(&clause.bind).unwrap();
    let score_slot = score_var.and_then(|sv| slots.slot(sv));

    // Rank by score, then materialize top-k (applying filters as we go so we
    // don't stop early on filtered-out hits).
    let mut ranked: Vec<(EntityId, f32)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = Vec::new();
    for (eid, score) in ranked {
        if out.len() >= k {
            break;
        }
        let mut seed = tuple.clone();
        seed.set(bind_slot, Value::Ref(eid));
        let rows = if has_filters {
            evaluate_clause(
                txn, &filter_clause, &seed, as_of, schema, slots, cache, interner, cache_gen,
            )?
        } else {
            vec![seed]
        };
        for mut row in rows {
            if let Some(slot) = score_slot {
                row.set(slot, Value::F64(score as f64));
            }
            out.push(row);
            if out.len() >= k {
                break;
            }
        }
    }
    Ok(out)
}

/// Read a big-endian u32 from a key, defaulting to 0.
fn read_u32(txn: &dyn ReadOps, key: &[u8]) -> Result<u32, String> {
    Ok(match txn.get(key).map_err(|e| e.to_string())? {
        Some(b) if b.len() == 4 => u32::from_be_bytes(b.try_into().unwrap()),
        _ => 0,
    })
}

/// Read a big-endian u64 meta value by logical name, defaulting to 0.
fn read_meta_u64(txn: &dyn ReadOps, name: &str) -> Result<u64, String> {
    let key = index::meta_key(name);
    Ok(match txn.get(&key).map_err(|e| e.to_string())? {
        Some(b) if b.len() == 8 => u64::from_be_bytes(b.try_into().unwrap()),
        _ => 0,
    })
}

/// Evaluate a single where clause against a current tuple.
/// Returns extended tuples for each matching entity.
fn evaluate_clause(
    txn: &dyn ReadOps,
    clause: &WhereClause,
    tuple: &Tuple,
    as_of: Option<TxId>,
    schema: &SchemaRegistry,
    slots: &SlotMap,
    cache: &QueryCache,
    interner: &AttrInterner,
    cache_gen: u64,
) -> Result<Vec<Tuple>, String> {
    // Nearest-neighbour (vector) search. A `near` pattern is a whole-type
    // top-k operation, handled by a dedicated kNN evaluator.
    if let Some((fname, query, k, metric, score_var)) = find_near_pattern(clause) {
        return evaluate_clause_with_near(
            txn, clause, tuple, as_of, schema, slots, cache, interner, cache_gen,
            &fname, &query, k, metric, score_var.as_deref(),
        );
    }

    // BM25 full-text search. A `search` pattern walks the inverted index and
    // ranks; another whole-type top-k operation.
    if let Some((fname, query, k, score_var)) = find_search_pattern(clause) {
        return evaluate_clause_with_search(
            txn, clause, tuple, as_of, schema, slots, cache, interner, cache_gen,
            &fname, &query, k, score_var.as_deref(),
        );
    }

    // Cardinality-many handling. If the clause references any many-attr, peel
    // those patterns off, evaluate the rest on the normal engine, then apply
    // the many-patterns (membership filter or fan-out bind) per candidate.
    let many_pats = many_field_patterns(clause, schema);
    if !many_pats.is_empty() {
        return evaluate_clause_with_many(
            txn, clause, tuple, as_of, schema, slots, cache, interner, cache_gen, &many_pats,
        );
    }

    // Check if the entity variable is already bound
    let bind_slot = slots.slot(&clause.bind).unwrap();
    let bound_entity = tuple.get(bind_slot).and_then(|v| {
        if let Value::Ref(id) = v {
            Some(*id)
        } else {
            None
        }
    });

    let use_current = as_of.is_none();

    // For current-state queries, eagerly load entire type into cache
    let type_data: Option<Arc<cache::TypeData>> = if use_current {
        cache.ensure_type_loaded(txn, interner, &clause.entity_type, schema, cache_gen)?
    } else {
        None
    };

    // ========== CACHED PATH (current-state queries) ==========
    if let Some(ref td) = type_data {
        // Pre-resolve every field pattern to its column reference(s).
        // Hashing the attribute names happens ONCE here, not per row.
        // Critical for inner-loop hot paths like Graph 2-Hop's manager
        // lookup, which previously re-hashed every field per entity.
        let resolved = match resolve_field_patterns(
            &clause.field_patterns,
            td,
            slots,
            schema,
            &clause.entity_type,
        ) {
            Some(r) => r,
            None => return Ok(Vec::new()), // pattern referenced an unknown variable
        };

        // Fast path: entity already bound (typically by an outer
        // nested-loop join). One binary search + per-pattern column
        // index. No allocation beyond the result tuple.
        if let Some(eid) = bound_entity {
            if let Some(idx) = td.index_of(eid) {
                let mut extended = tuple.clone();
                extended.set(bind_slot, Value::Ref(eid));
                if match_resolved(&resolved, idx, &mut extended, slots, td) {
                    return Ok(vec![extended]);
                }
            }
            return Ok(Vec::new());
        }

        let plans = find_indexable_patterns(clause, tuple, schema, slots);
        if !plans.is_empty() {
            // Use AVET to narrow the candidate set with ONE scan, then
            // let `match_resolved` post-filter the remaining patterns via
            // direct column lookup. Doing a second AVET scan + HashSet
            // intersection was throwing away work — `match_resolved`
            // already evaluates Range/Exact patterns from columns in
            // O(1) per row. Selective Scan workload was running 2 AVET
            // scans (~1160 keys) + intersection when 1 scan (~500 keys)
            // + column post-filter is strictly cheaper.
            //
            // Pick the most selective plan; `Exact` is always at least
            // as selective as a `Range` over the same attribute, and
            // `find_indexable_patterns` lists patterns in clause order
            // (no a-priori ranking), so just prefer Exact.
            let chosen = plans
                .iter()
                .find(|p| matches!(p, AttrPlan::Exact(_, _)))
                .unwrap_or(&plans[0]);

            let candidates: Vec<EntityId> = match chosen {
                AttrPlan::Exact(attr, value) => {
                    find_entities_by_avet_current(txn, interner, attr, value)?
                }
                AttrPlan::Range(attr, op, value) => {
                    find_entities_by_avet_range_current(txn, interner, attr, op, value)?
                        .into_iter()
                        .map(|(eid, _)| eid)
                        .collect()
                }
            };

            let mut results = Vec::new();
            for eid in candidates {
                if let Some(idx) = td.index_of(eid) {
                    let mut extended = tuple.clone();
                    extended.set(bind_slot, Value::Ref(eid));
                    // match_resolved evaluates every pattern, including
                    // those we deliberately skipped scanning above.
                    if match_resolved(&resolved, idx, &mut extended, slots, td) {
                        results.push(extended);
                    }
                }
            }
            return Ok(results);
        }

        // No AVET patterns: sequential walk over the entity_ids vec.
        // This is the scan-friendly path — `entity_ids` lives in a
        // single contiguous allocation and the per-row column access
        // is array indexing, no HashMap probes.
        let mut results = Vec::with_capacity(td.entity_ids.len());
        for (idx, &eid) in td.entity_ids.iter().enumerate() {
            let mut extended = tuple.clone();
            extended.set(bind_slot, Value::Ref(eid));
            if match_resolved(&resolved, idx, &mut extended, slots, td) {
                results.push(extended);
            }
        }
        return Ok(results);
    }

    // ========== UNCACHED PATH (historical queries) ==========

    // Compute needed fields early so we can optimize entity discovery
    let needed_fields = compute_needed_fields(clause, schema);

    // Track field values already known from index scans, keyed by entity ID
    let mut known_fields: HashMap<EntityId, HashMap<String, Value>> = HashMap::new();
    let attr_prefix = format!("{}/", clause.entity_type);
    let mut bulk_loaded = false;

    let entities = if let Some(eid) = bound_entity {
        // Entity already bound — just check this one
        vec![eid]
    } else {
        // Try to use AVET index for constant, bound, or range patterns
        let plans = find_indexable_patterns(clause, tuple, schema, slots);
        if !plans.is_empty() {
            let mut candidate_sets: Vec<Vec<EntityId>> = Vec::new();
            for plan in &plans {
                match plan {
                    AttrPlan::Exact(attr, value) => {
                        let eids = find_entities_by_avet(txn, interner, attr, value, as_of)?;
                        // Pre-populate known field values
                        if let Some(field_name) = attr.strip_prefix(&attr_prefix) {
                            for &eid in &eids {
                                known_fields
                                    .entry(eid)
                                    .or_default()
                                    .insert(field_name.to_string(), value.clone());
                            }
                        }
                        candidate_sets.push(eids);
                    }
                    AttrPlan::Range(attr, op, value) => {
                        let results =
                            find_entities_by_avet_range(txn, interner, attr, op, value, as_of)?;
                        let field_name = attr.strip_prefix(&attr_prefix);
                        let eids: Vec<EntityId> =
                            results.iter().map(|(eid, _)| *eid).collect();
                        if let Some(fname) = field_name {
                            for (eid, val) in &results {
                                known_fields
                                    .entry(*eid)
                                    .or_default()
                                    .insert(fname.to_string(), val.clone());
                            }
                        }
                        candidate_sets.push(eids);
                    }
                }
            }
            // Intersect all candidate sets
            let mut result = candidate_sets.remove(0);
            for set in &candidate_sets {
                let set_lookup: HashSet<EntityId> =
                    set.iter().copied().collect();
                result.retain(|eid| set_lookup.contains(eid));
            }
            result
        } else if let Some(ref needed) = needed_fields {
            // No AVET lookups — try to use AEVT scan on a required field
            // to discover entities AND load field values in one scan
            if let Some(required_field) = find_required_field(needed, clause, schema) {
                let (entities, field_map) = scan_entities_and_field_via_aevt(
                    txn,
                    interner,
                    &clause.entity_type,
                    &required_field,
                    as_of,
                )?;
                known_fields = field_map;

                // Bulk-load remaining needed fields
                let remaining: Vec<String> = needed
                    .iter()
                    .filter(|f| *f != &required_field)
                    .cloned()
                    .collect();
                if !remaining.is_empty() {
                    let entity_set: HashSet<EntityId> = entities.iter().copied().collect();
                    let extra = bulk_load_fields_via_aevt(
                        txn,
                        interner,
                        &clause.entity_type,
                        &remaining,
                        &entity_set,
                        as_of,
                    )?;
                    for (eid, fields) in extra {
                        known_fields.entry(eid).or_default().extend(fields);
                    }
                }
                bulk_loaded = true;
                entities
            } else {
                find_entities_of_type(txn, interner, &clause.entity_type, as_of)?
            }
        } else {
            // Fall back to full type scan
            find_entities_of_type(txn, interner, &clause.entity_type, as_of)?
        }
    };

    // Bulk-load fields via AEVT scan when doing a full type scan with many entities.
    if !bulk_loaded && known_fields.is_empty() && entities.len() > 16 {
        if let Some(ref needed) = needed_fields {
            let entity_set: HashSet<EntityId> = entities.iter().copied().collect();
            known_fields = bulk_load_fields_via_aevt(
                txn,
                interner,
                &clause.entity_type,
                needed,
                &entity_set,
                as_of,
            )?;
            bulk_loaded = true;
        }
    }

    let mut results = Vec::new();

    for eid in entities {
        let entity_known = known_fields.remove(&eid).unwrap_or_default();

        let fields = if bulk_loaded {
            // All needed fields were bulk-loaded via AEVT; no per-entity scans needed
            entity_known
        } else {
            // Load remaining fields per-entity
            let field_filter: Option<Vec<String>> = needed_fields.as_ref().map(|needed| {
                needed
                    .iter()
                    .filter(|f| !entity_known.contains_key(*f))
                    .cloned()
                    .collect()
            });

            let mut fields = load_entity_fields(
                txn,
                interner,
                eid,
                &clause.entity_type,
                as_of,
                field_filter.as_deref(),
            )?;

            // Merge in known values from AVET scans
            for (k, v) in entity_known {
                fields.entry(k).or_insert(v);
            }

            fields
        };

        // Try to match all field patterns
        let mut extended = tuple.clone();
        extended.set(bind_slot, Value::Ref(eid));
        if match_field_patterns(&clause.field_patterns, &fields, &mut extended, slots) {
            results.push(extended);
        }
    }

    Ok(results)
}

/// Find patterns in a clause that can use the AVET index.
/// Returns plans for exact lookups and range scans.
fn find_indexable_patterns(
    clause: &WhereClause,
    tuple: &Tuple,
    schema: &SchemaRegistry,
    slots: &SlotMap,
) -> Vec<AttrPlan> {
    let type_def = match schema.get(&clause.entity_type) {
        Some(td) => td,
        None => return vec![],
    };

    let mut plans = Vec::new();

    for (field_name, pattern) in &clause.field_patterns {
        // Skip enum fields — they use sub-attributes (__tag, .Variant/field)
        // and can't be directly looked up via a single AVET scan
        let field_def = type_def.get_field(field_name);
        if let Some(fd) = field_def {
            if matches!(fd.field_type, FieldType::Enum(_)) {
                continue;
            }
        }

        let attr = type_def.attribute_name(field_name);

        match pattern {
            Pattern::Constant(value) => {
                plans.push(AttrPlan::Exact(attr, value.clone()));
            }
            Pattern::Variable(var) => {
                // If the variable is already bound, we can use its value
                if let Some(slot) = slots.slot(var) {
                    if let Some(bound_value) = tuple.get(slot) {
                        plans.push(AttrPlan::Exact(attr, bound_value.clone()));
                    }
                }
            }
            Pattern::Predicate { op, value }
            | Pattern::BoundPredicate { op, value, .. } => {
                // Range predicates use AVET index (Ne excluded — not worth it).
                // String-search ops (contains/starts_with/ends_with) aren't
                // range-comparable, so they can't drive an index scan; they
                // filter via the sequential column walk instead.
                // BoundPredicate filters here too; its variable binding is
                // applied later when the matched rows are materialized.
                if !matches!(op, PredOp::Ne) && !op.is_string_search() {
                    plans.push(AttrPlan::Range(attr, op.clone(), value.clone()));
                }
            }
            _ => {
                // EnumMatch — can't use AVET index
            }
        }
    }

    plans
}

/// Compute which fields a clause needs loaded from EAVT.
/// Returns None if a full entity scan is required (enum fields, too many fields).
fn compute_needed_fields(
    clause: &WhereClause,
    schema: &SchemaRegistry,
) -> Option<Vec<String>> {
    let type_def = schema.get(&clause.entity_type)?;
    let mut needed = Vec::new();

    for (field_name, pattern) in &clause.field_patterns {
        // Enum patterns require full entity load (need __tag + variant sub-fields)
        if matches!(pattern, Pattern::EnumMatch { .. }) {
            return None;
        }
        // Schema-declared enum fields also need full load
        if let Some(fd) = type_def.get_field(field_name) {
            if matches!(fd.field_type, FieldType::Enum(_)) {
                return None;
            }
        }
        needed.push(field_name.clone());
    }

    if needed.len() > 4 {
        return None;
    }

    Some(needed)
}

/// Find entity IDs that have a specific (attribute, value) via AVET index scan.
/// Resolves retract history to only return currently-active entities.
fn find_entities_by_avet(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    attr: &str,
    value: &Value,
    as_of: Option<TxId>,
) -> Result<Vec<EntityId>, String> {
    let attr_id = match interner.lookup(attr) {
        Some(id) => id,
        None => return Ok(Vec::new()),
    };

    let prefix = index::avet_attr_value_prefix(attr_id, value);
    let end = index::prefix_end(&prefix);

    let mut entity_datoms: HashMap<EntityId, Vec<index::DecodedDatom>> = HashMap::new();

    txn.scan_foreach(&prefix, &end, &mut |key, _value| {
        if let Some(datom) = index::decode_datom_from_avet(key) {
            if let Some(max_tx) = as_of {
                if datom.tx > max_tx {
                    return true;
                }
            }
            entity_datoms.entry(datom.entity).or_default().push(datom);
        }
        true
    })
    .map_err(|e| e.to_string())?;

    let mut entities = Vec::new();
    for (eid, mut datoms) in entity_datoms {
        datoms.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));
        let mut exists = false;
        for d in datoms {
            exists = d.added;
        }
        if exists {
            entities.push(eid);
        }
    }

    Ok(entities)
}

/// Find entity IDs matching a range predicate via AVET index scan.
/// Returns (entity_id, current_value) pairs for entities whose current value matches the predicate.
fn find_entities_by_avet_range(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    attr: &str,
    op: &PredOp,
    value: &Value,
    as_of: Option<TxId>,
) -> Result<Vec<(EntityId, Value)>, String> {
    let attr_id = match interner.lookup(attr) {
        Some(id) => id,
        None => return Ok(Vec::new()),
    };

    let type_tag = value.type_tag();

    // Compute tight scan bounds based on the predicate
    let (start, end) = match op {
        PredOp::Gt | PredOp::Gte => {
            let start = index::avet_attr_value_prefix(attr_id, value);
            let end = index::prefix_end(&index::avet_attr_prefix(attr_id));
            (start, end)
        }
        PredOp::Lt => {
            let start = index::avet_attr_type_prefix(attr_id, type_tag);
            let end = index::avet_attr_value_prefix(attr_id, value);
            (start, end)
        }
        PredOp::Lte => {
            let start = index::avet_attr_type_prefix(attr_id, type_tag);
            let end = index::prefix_end(&index::avet_attr_value_prefix(attr_id, value));
            (start, end)
        }
        PredOp::Ne | PredOp::Contains | PredOp::StartsWith | PredOp::EndsWith => {
            // Ne and string-search ops aren't range-bounded; scan the whole
            // attribute and let the post-filter (PredOp::evaluate) decide.
            // String-search ops are normally filtered out before reaching an
            // index plan, but handle them defensively.
            let start = index::avet_attr_type_prefix(attr_id, type_tag);
            let end = index::prefix_end(&index::avet_attr_prefix(attr_id));
            (start, end)
        }
    };

    let entries = txn.scan(&start, &end).map_err(|e| e.to_string())?;

    let mut entity_datoms: HashMap<EntityId, Vec<index::DecodedDatom>> = HashMap::new();

    for (key, _) in &entries {
        if let Some(datom) = index::decode_datom_from_avet(key) {
            if let Some(max_tx) = as_of {
                if datom.tx > max_tx {
                    continue;
                }
            }
            entity_datoms.entry(datom.entity).or_default().push(datom);
        }
    }

    let mut results = Vec::new();
    for (eid, mut datoms) in entity_datoms {
        datoms.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));
        // The last datom determines the current state for this entity
        if let Some(last) = datoms.last() {
            if last.added && op.evaluate(&last.value, value) {
                results.push((eid, last.value.clone()));
            }
        }
    }

    Ok(results)
}

/// Find all entity IDs of a given type.
fn find_entities_of_type(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    type_name: &str,
    as_of: Option<TxId>,
) -> Result<Vec<EntityId>, String> {
    let type_attr_id = match interner.lookup("__type") {
        Some(id) => id,
        None => return Ok(Vec::new()),
    };
    let type_value = Value::String(type_name.into());
    let prefix = index::avet_attr_value_prefix(type_attr_id, &type_value);
    let end = index::prefix_end(&prefix);

    // Group datoms by entity, then resolve history properly.
    // Stream via scan_foreach to skip the intermediate
    // Vec<(Vec<u8>, Vec<u8>)> that `txn.scan` would allocate.
    let mut entity_datoms: HashMap<EntityId, Vec<index::DecodedDatom>> = HashMap::new();

    txn.scan_foreach(&prefix, &end, &mut |key, _value| {
        if let Some(datom) = index::decode_datom_from_avet(key) {
            if let Some(max_tx) = as_of {
                if datom.tx > max_tx {
                    return true;
                }
            }
            entity_datoms.entry(datom.entity).or_default().push(datom);
        }
        true
    })
    .map_err(|e| e.to_string())?;

    let mut entities = Vec::new();
    for (eid, mut datoms) in entity_datoms {
        // Sort by tx ascending, retracts before asserts within same tx
        datoms.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));
        let mut exists = false;
        for d in datoms {
            exists = d.added;
        }
        if exists {
            entities.push(eid);
        }
    }

    Ok(entities)
}

/// Find a required field among the needed fields for a clause.
/// Used to optimize entity discovery: scanning AEVT for a required field
/// discovers all entities of that type AND loads field values in one scan.
fn find_required_field(
    needed: &[String],
    clause: &WhereClause,
    schema: &SchemaRegistry,
) -> Option<String> {
    let type_def = schema.get(&clause.entity_type)?;
    for field_name in needed {
        if let Some(fd) = type_def.get_field(field_name) {
            if fd.required {
                return Some(field_name.clone());
            }
        }
    }
    None
}

/// Scan AEVT for a single field to discover entity IDs AND load field values.
/// Returns (entity_ids, field_map) where field_map maps entity_id -> {field_name -> value}.
/// This replaces the separate type scan + field load when the field is required.
fn scan_entities_and_field_via_aevt(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    entity_type: &str,
    field_name: &str,
    as_of: Option<TxId>,
) -> Result<(Vec<EntityId>, HashMap<EntityId, HashMap<String, Value>>), String> {
    let attr = format!("{}/{}", entity_type, field_name);
    let attr_id = match interner.lookup(&attr) {
        Some(id) => id,
        None => return Ok((Vec::new(), HashMap::new())),
    };
    let prefix = index::aevt_attr_prefix(attr_id);
    let end = index::prefix_end(&prefix);

    let entries = txn.scan(&prefix, &end).map_err(|e| e.to_string())?;

    let mut entity_datoms: HashMap<EntityId, Vec<index::DecodedDatom>> = HashMap::new();
    for (key, _) in &entries {
        if let Some(datom) = index::decode_datom_from_aevt(key) {
            if let Some(max_tx) = as_of {
                if datom.tx > max_tx {
                    continue;
                }
            }
            entity_datoms.entry(datom.entity).or_default().push(datom);
        }
    }

    let mut entities = Vec::new();
    let mut field_map: HashMap<EntityId, HashMap<String, Value>> = HashMap::new();

    for (eid, mut datoms) in entity_datoms {
        datoms.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));
        if let Some(last) = datoms.last() {
            if last.added {
                entities.push(eid);
                let mut fields = HashMap::new();
                fields.insert(field_name.to_string(), last.value.clone());
                field_map.insert(eid, fields);
            }
        }
    }

    Ok((entities, field_map))
}

/// Bulk-load field values for many entities at once via AEVT index.
/// Instead of N individual EAVT scans per field, does 1 AEVT range scan per field
/// covering all entities. Returns a map of entity_id -> field_name -> value.
fn bulk_load_fields_via_aevt(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    entity_type: &str,
    field_names: &[String],
    entity_set: &HashSet<EntityId>,
    as_of: Option<TxId>,
) -> Result<HashMap<EntityId, HashMap<String, Value>>, String> {
    let mut result: HashMap<EntityId, HashMap<String, Value>> = HashMap::new();

    for field_name in field_names {
        let attr = format!("{}/{}", entity_type, field_name);
        let attr_id = match interner.lookup(&attr) {
            Some(id) => id,
            None => continue,
        };
        let prefix = index::aevt_attr_prefix(attr_id);
        let end = index::prefix_end(&prefix);

        let entries = txn.scan(&prefix, &end).map_err(|e| e.to_string())?;

        // Group by entity, filtering to our entity set
        let mut entity_datoms: HashMap<EntityId, Vec<index::DecodedDatom>> = HashMap::new();
        for (key, _) in &entries {
            if let Some(datom) = index::decode_datom_from_aevt(key) {
                if !entity_set.contains(&datom.entity) {
                    continue;
                }
                if let Some(max_tx) = as_of {
                    if datom.tx > max_tx {
                        continue;
                    }
                }
                entity_datoms.entry(datom.entity).or_default().push(datom);
            }
        }

        // Resolve retractions: last datom by tx determines current state
        for (eid, mut datoms) in entity_datoms {
            datoms.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));
            if let Some(last) = datoms.last() {
                if last.added {
                    result
                        .entry(eid)
                        .or_default()
                        .insert(field_name.clone(), last.value.clone());
                }
            }
        }
    }

    Ok(result)
}

/// Load current field values for an entity.
/// When `field_filter` is Some, only loads the specified fields (selective scan).
/// When None, loads all fields (full entity scan).
fn load_entity_fields(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    entity: EntityId,
    entity_type: &str,
    as_of: Option<TxId>,
    field_filter: Option<&[String]>,
) -> Result<HashMap<String, Value>, String> {
    let mut attr_state: HashMap<String, Vec<Datom>> = HashMap::new();

    if let Some(fields_to_load) = field_filter {
        // Selective: scan each needed attribute individually via EAVT
        for field_name in fields_to_load {
            let attr = format!("{}/{}", entity_type, field_name);
            let attr_id = match interner.lookup(&attr) {
                Some(id) => id,
                None => continue,
            };
            let prefix = index::eavt_entity_attr_prefix(entity, attr_id);
            let end = index::prefix_end(&prefix);
            let entries = txn.scan(&prefix, &end).map_err(|e| e.to_string())?;
            for (key, _) in &entries {
                if let Some(decoded) = index::decode_datom_from_eavt(key) {
                    if let Some(max_tx) = as_of {
                        if decoded.tx > max_tx {
                            continue;
                        }
                    }
                    if let Some(datom) = interner.resolve(decoded) {
                        attr_state
                            .entry(datom.attribute.clone())
                            .or_default()
                            .push(datom);
                    }
                }
            }
        }
    } else {
        // Full entity scan (original behavior)
        let prefix = index::eavt_entity_prefix(entity);
        let end = index::prefix_end(&prefix);
        let entries = txn.scan(&prefix, &end).map_err(|e| e.to_string())?;
        for (key, _) in &entries {
            if let Some(decoded) = index::decode_datom_from_eavt(key) {
                if let Some(max_tx) = as_of {
                    if decoded.tx > max_tx {
                        continue;
                    }
                }
                if let Some(datom) = interner.resolve(decoded) {
                    attr_state
                        .entry(datom.attribute.clone())
                        .or_default()
                        .push(datom);
                }
            }
        }
    }

    let attr_prefix = format!("{}/", entity_type);

    // Use resolve_current_values to properly handle retract+assert history
    let resolved = crate::db::resolve_current_values(attr_state, as_of);

    let mut fields = HashMap::new();
    for (attr, value) in resolved {
        if !attr.starts_with(&attr_prefix) {
            continue;
        }
        let field_name = &attr[attr_prefix.len()..];
        fields.insert(field_name.to_string(), value);
    }

    Ok(fields)
}

/// Field-pattern access plan resolved once per clause evaluation
/// against a `TypeData`. Each pattern carries the column it needs
/// (as an owned `Column` Arc, NOT a borrow) so the resolved set can
/// be cached on a `ScanIterator` across many `open()` calls without
/// running into self-referential lifetime issues.
enum ResolvedFP {
    /// Variable binding on a scalar/ref field.
    Variable {
        slot: usize,
        col: Option<cache::Column>,
    },
    /// Variable binding on an enum field — binds the tag value.
    EnumVariable {
        slot: usize,
        tag_col: Option<cache::Column>,
    },
    /// Constant match on a scalar/ref field.
    Constant {
        col: Option<cache::Column>,
        expected: Value,
    },
    /// Constant match on an enum field — compares against the tag.
    EnumConstant {
        tag_col: Option<cache::Column>,
        expected: Value,
    },
    /// Range predicate on a scalar field.
    Predicate {
        col: Option<cache::Column>,
        op: PredOp,
        value: Value,
    },
    /// Combined bind + range predicate on a scalar field: binds `slot`
    /// to the field value and keeps only rows passing the predicate.
    BoundPredicate {
        slot: usize,
        col: Option<cache::Column>,
        op: PredOp,
        value: Value,
    },
    /// Full enum match — variant-qualified columns are data-dependent
    /// (need the tag value to know which column to read), so we keep
    /// the field name and look up dynamically per row.
    EnumMatch {
        field_name: String,
        variant: Pattern,
        field_patterns: Vec<(String, Pattern)>,
        tag_col: Option<cache::Column>,
    },
}

/// Pre-resolve all field patterns for a clause to their column refs.
/// String hashes happen here (once per pattern); the per-row matcher
/// only does array indexing. Returns owned `ResolvedFP`s with Arc'd
/// columns — safe to cache across `ScanIterator` opens.
fn resolve_field_patterns(
    patterns: &[(String, Pattern)],
    td: &cache::TypeData,
    slots: &SlotMap,
    schema: &SchemaRegistry,
    entity_type: &str,
) -> Option<Vec<ResolvedFP>> {
    let type_def = schema.get(entity_type);
    let mut out: Vec<ResolvedFP> = Vec::with_capacity(patterns.len());
    for (field_name, pat) in patterns {
        let is_enum_field = type_def
            .and_then(|td| td.get_field(field_name))
            .map(|fd| matches!(fd.field_type, FieldType::Enum(_)))
            .unwrap_or(false);
        let tag_col = if is_enum_field {
            let tag_key = format!("{}/__tag", field_name);
            td.column_arc(&tag_key)
        } else {
            None
        };
        let field_col = td.column_arc(field_name);
        let rfp = match pat {
            Pattern::Variable(var) => {
                let slot = slots.slot(var)?;
                if is_enum_field {
                    ResolvedFP::EnumVariable { slot, tag_col }
                } else {
                    ResolvedFP::Variable {
                        slot,
                        col: field_col,
                    }
                }
            }
            Pattern::Constant(expected) => {
                if is_enum_field {
                    ResolvedFP::EnumConstant {
                        tag_col,
                        expected: expected.clone(),
                    }
                } else {
                    ResolvedFP::Constant {
                        col: field_col,
                        expected: expected.clone(),
                    }
                }
            }
            Pattern::Predicate { op, value } => ResolvedFP::Predicate {
                col: field_col,
                op: op.clone(),
                value: value.clone(),
            },
            Pattern::BoundPredicate { var, op, value } => {
                let slot = slots.slot(var)?;
                ResolvedFP::BoundPredicate {
                    slot,
                    col: field_col,
                    op: op.clone(),
                    value: value.clone(),
                }
            }
            Pattern::EnumMatch {
                variant,
                field_patterns,
            } => ResolvedFP::EnumMatch {
                field_name: field_name.clone(),
                variant: (**variant).clone(),
                field_patterns: field_patterns.clone(),
                tag_col,
            },
            // Near patterns are handled by the dedicated kNN path before the
            // columnar engine is reached, so they never appear here.
            Pattern::Near { .. } | Pattern::Search { .. } => continue,
        };
        out.push(rfp);
    }
    Some(out)
}

/// Apply pre-resolved patterns to one entity row (by row index into
/// the columnar layout). Returns false if any pattern fails.
fn match_resolved(
    resolved: &[ResolvedFP],
    idx: usize,
    tuple: &mut Tuple,
    slots: &SlotMap,
    td: &cache::TypeData,
) -> bool {
    for rfp in resolved {
        match rfp {
            ResolvedFP::Variable { slot, col } => {
                let val = col
                    .as_ref()
                    .and_then(|c| c.get(idx))
                    .and_then(|v| v.as_ref())
                    .cloned()
                    .unwrap_or(Value::Null);
                if let Some(existing) = tuple.get(*slot) {
                    if *existing != val {
                        return false;
                    }
                } else {
                    tuple.set(*slot, val);
                }
            }
            ResolvedFP::EnumVariable { slot, tag_col } => {
                let tag_val = match tag_col
                    .as_ref()
                    .and_then(|c| c.get(idx))
                    .and_then(|v| v.as_ref())
                {
                    Some(v) => v,
                    None => return false,
                };
                if let Some(existing) = tuple.get(*slot) {
                    if existing != tag_val {
                        return false;
                    }
                } else {
                    tuple.set(*slot, tag_val.clone());
                }
            }
            ResolvedFP::Constant { col, expected } => {
                let val = col
                    .as_ref()
                    .and_then(|c| c.get(idx))
                    .and_then(|v| v.as_ref());
                if val != Some(expected) {
                    return false;
                }
            }
            ResolvedFP::EnumConstant { tag_col, expected } => {
                let tag_val = tag_col
                    .as_ref()
                    .and_then(|c| c.get(idx))
                    .and_then(|v| v.as_ref());
                if tag_val != Some(expected) {
                    return false;
                }
            }
            ResolvedFP::Predicate { col, op, value } => {
                let field_val = col
                    .as_ref()
                    .and_then(|c| c.get(idx))
                    .and_then(|v| v.as_ref());
                match field_val {
                    Some(fv) if op.evaluate(fv, value) => {}
                    _ => return false,
                }
            }
            ResolvedFP::BoundPredicate { slot, col, op, value } => {
                // Filter first, then bind the surviving value.
                let field_val = match col
                    .as_ref()
                    .and_then(|c| c.get(idx))
                    .and_then(|v| v.as_ref())
                {
                    Some(fv) if op.evaluate(fv, value) => fv.clone(),
                    _ => return false,
                };
                if let Some(existing) = tuple.get(*slot) {
                    if *existing != field_val {
                        return false;
                    }
                } else {
                    tuple.set(*slot, field_val);
                }
            }
            ResolvedFP::EnumMatch {
                field_name,
                variant,
                field_patterns,
                tag_col,
            } => {
                let tag_val = match tag_col
                    .as_ref()
                    .and_then(|c| c.get(idx))
                    .and_then(|v| v.as_ref())
                {
                    Some(v) => v,
                    None => return false,
                };
                match variant {
                    Pattern::Variable(var) => {
                        let slot = match slots.slot(var.as_str()) {
                            Some(s) => s,
                            None => return false,
                        };
                        if let Some(existing) = tuple.get(slot) {
                            if existing != tag_val {
                                return false;
                            }
                        } else {
                            tuple.set(slot, tag_val.clone());
                        }
                    }
                    Pattern::Constant(expected) => {
                        if tag_val != expected {
                            return false;
                        }
                    }
                    _ => return false,
                }
                let variant_name = match tag_val {
                    Value::String(s) => &**s,
                    _ => return false,
                };
                for (vf_name, vf_pattern) in field_patterns {
                    let vf_key = format!("{}.{}/{}", field_name, variant_name, vf_name);
                    let vf_val = td
                        .column(&vf_key)
                        .and_then(|c| c.get(idx))
                        .and_then(|v| v.as_ref());
                    match vf_pattern {
                        Pattern::Variable(var) => {
                            let slot = match slots.slot(var) {
                                Some(s) => s,
                                None => return false,
                            };
                            let val = match vf_val {
                                Some(v) => v,
                                None => return false,
                            };
                            if let Some(existing) = tuple.get(slot) {
                                if existing != val {
                                    return false;
                                }
                            } else {
                                tuple.set(slot, val.clone());
                            }
                        }
                        Pattern::Constant(expected) => {
                            if vf_val != Some(expected) {
                                return false;
                            }
                        }
                        Pattern::Predicate { op, value } => match vf_val {
                            Some(fv) if op.evaluate(fv, value) => {}
                            _ => return false,
                        },
                        Pattern::BoundPredicate { var, op, value } => {
                            let slot = match slots.slot(var) {
                                Some(s) => s,
                                None => return false,
                            };
                            let val = match vf_val {
                                Some(fv) if op.evaluate(fv, value) => fv,
                                _ => return false,
                            };
                            if let Some(existing) = tuple.get(slot) {
                                if existing != val {
                                    return false;
                                }
                            } else {
                                tuple.set(slot, val.clone());
                            }
                        }
                        Pattern::EnumMatch { .. } => return false,
                        Pattern::Near { .. } | Pattern::Search { .. } => return false,
                    }
                }
            }
        }
    }
    true
}

/// Try to match field patterns against entity fields.
/// Writes bound variables directly into `tuple`. Returns false on conflict or mismatch.
fn match_field_patterns(
    patterns: &[(String, Pattern)],
    fields: &HashMap<String, Value>,
    tuple: &mut Tuple,
    slots: &SlotMap,
) -> bool {
    for (field_name, pattern) in patterns {
        match pattern {
            Pattern::Variable(var) => {
                let slot = match slots.slot(var) {
                    Some(s) => s,
                    None => return false,
                };
                // For enum fields, a bare variable on an enum field binds the tag
                // Check if this is an enum field by looking for __tag
                let tag_key = format!("{}/__tag", field_name);
                if let Some(tag_val) = fields.get(&tag_key) {
                    // This is an enum field — bind the tag
                    if let Some(existing) = tuple.get(slot) {
                        if existing != tag_val {
                            return false;
                        }
                    } else {
                        tuple.set(slot, tag_val.clone());
                    }
                } else {
                    // Regular scalar field — use Null for missing optional fields
                    let field_val = fields
                        .get(field_name)
                        .cloned()
                        .unwrap_or(Value::Null);
                    if let Some(existing) = tuple.get(slot) {
                        if *existing != field_val {
                            return false;
                        }
                    } else {
                        tuple.set(slot, field_val);
                    }
                }
            }
            Pattern::Constant(expected) => {
                // For enum fields, a bare constant matches the tag
                let tag_key = format!("{}/__tag", field_name);
                if let Some(tag_val) = fields.get(&tag_key) {
                    if tag_val != expected {
                        return false;
                    }
                } else {
                    match fields.get(field_name) {
                        Some(field_val) if field_val == expected => {}
                        _ => return false,
                    }
                }
            }
            Pattern::Predicate { op, value } => {
                match fields.get(field_name) {
                    Some(field_val) if op.evaluate(field_val, value) => {}
                    _ => return false,
                }
            }
            Pattern::BoundPredicate { var, op, value } => {
                let slot = match slots.slot(var) {
                    Some(s) => s,
                    None => return false,
                };
                let field_val = match fields.get(field_name) {
                    Some(fv) if op.evaluate(fv, value) => fv.clone(),
                    _ => return false,
                };
                if let Some(existing) = tuple.get(slot) {
                    if *existing != field_val {
                        return false;
                    }
                } else {
                    tuple.set(slot, field_val);
                }
            }
            Pattern::EnumMatch {
                variant,
                field_patterns,
            } => {
                // Match enum tag
                let tag_key = format!("{}/__tag", field_name);
                let tag_val = match fields.get(&tag_key) {
                    Some(v) => v,
                    None => return false,
                };

                match variant.as_ref() {
                    Pattern::Variable(var) => {
                        let slot = match slots.slot(var) {
                            Some(s) => s,
                            None => return false,
                        };
                        if let Some(existing) = tuple.get(slot) {
                            if existing != tag_val {
                                return false;
                            }
                        } else {
                            tuple.set(slot, tag_val.clone());
                        }
                    }
                    Pattern::Constant(expected) => {
                        if tag_val != expected {
                            return false;
                        }
                    }
                    _ => return false,
                }

                // Get the variant name to construct field keys
                let variant_name = match tag_val {
                    Value::String(s) => &**s,
                    _ => return false,
                };

                // Match variant field patterns
                for (vf_name, vf_pattern) in field_patterns {
                    let vf_key = format!("{}.{}/{}", field_name, variant_name, vf_name);
                    match vf_pattern {
                        Pattern::Variable(var) => {
                            let slot = match slots.slot(var) {
                                Some(s) => s,
                                None => return false,
                            };
                            let vf_val = match fields.get(&vf_key) {
                                Some(v) => v,
                                None => return false,
                            };
                            if let Some(existing) = tuple.get(slot) {
                                if existing != vf_val {
                                    return false;
                                }
                            } else {
                                tuple.set(slot, vf_val.clone());
                            }
                        }
                        Pattern::Constant(expected) => {
                            match fields.get(&vf_key) {
                                Some(vf_val) if vf_val == expected => {}
                                _ => return false,
                            }
                        }
                        Pattern::Predicate { op, value } => {
                            match fields.get(&vf_key) {
                                Some(vf_val) if op.evaluate(vf_val, value) => {}
                                _ => return false,
                            }
                        }
                        Pattern::BoundPredicate { var, op, value } => {
                            let slot = match slots.slot(var) {
                                Some(s) => s,
                                None => return false,
                            };
                            let vf_val = match fields.get(&vf_key) {
                                Some(fv) if op.evaluate(fv, value) => fv.clone(),
                                _ => return false,
                            };
                            if let Some(existing) = tuple.get(slot) {
                                if *existing != vf_val {
                                    return false;
                                }
                            } else {
                                tuple.set(slot, vf_val);
                            }
                        }
                        Pattern::EnumMatch { .. } => {
                            // Nested enum matching not supported
                            return false;
                        }
                        Pattern::Near { .. } | Pattern::Search { .. } => return false,
                    }
                }
            }
            // Near patterns are handled by the dedicated kNN path before this
            // per-row matcher; they never reach here.
            Pattern::Near { .. } | Pattern::Search { .. } => return false,
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Current-state index query functions (used when as_of is None)
// ---------------------------------------------------------------------------

/// Find entity IDs by exact (attr, value) via CURRENT_AVET index.
fn find_entities_by_avet_current(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    attr: &str,
    value: &Value,
) -> Result<Vec<EntityId>, String> {
    let attr_id = match interner.lookup(attr) {
        Some(id) => id,
        None => return Ok(Vec::new()),
    };
    let prefix = index::current_avet_attr_value_prefix(attr_id, value);
    let end = index::prefix_end(&prefix);
    let mut entities = Vec::new();
    txn.scan_foreach(&prefix, &end, &mut |key, _value| {
        entities.push(index::current_avet_entity_at(key));
        true
    })
    .map_err(|e| e.to_string())?;
    Ok(entities)
}

/// Find entity IDs matching a range predicate via CURRENT_AVET index.
fn find_entities_by_avet_range_current(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    attr: &str,
    op: &PredOp,
    value: &Value,
) -> Result<Vec<(EntityId, Value)>, String> {
    let attr_id = match interner.lookup(attr) {
        Some(id) => id,
        None => return Ok(Vec::new()),
    };
    let type_tag = value.type_tag();

    let (start, end) = match op {
        PredOp::Gt | PredOp::Gte => {
            let start = index::current_avet_attr_value_prefix(attr_id, value);
            let end = index::prefix_end(&index::current_avet_attr_prefix(attr_id));
            (start, end)
        }
        PredOp::Lt => {
            let start = index::current_avet_attr_type_prefix(attr_id, type_tag);
            let end = index::current_avet_attr_value_prefix(attr_id, value);
            (start, end)
        }
        PredOp::Lte => {
            let start = index::current_avet_attr_type_prefix(attr_id, type_tag);
            let end = index::prefix_end(&index::current_avet_attr_value_prefix(attr_id, value));
            (start, end)
        }
        PredOp::Ne | PredOp::Contains | PredOp::StartsWith | PredOp::EndsWith => {
            // Not range-bounded; scan the whole attribute and post-filter.
            let start = index::current_avet_attr_type_prefix(attr_id, type_tag);
            let end = index::prefix_end(&index::current_avet_attr_prefix(attr_id));
            (start, end)
        }
    };

    // CURRENT_AVET key: [prefix][attr_id(4)][type_tag][value_data][entity_id(8)]
    let attr_prefix_len = 1 + 4; // prefix byte + attr_id u32

    // Stream via scan_foreach to avoid the intermediate
    // `Vec<(Vec<u8>, Vec<u8>)>` that `txn.scan` allocates — for
    // Selective Scan the range yields hundreds of entries per query
    // and the per-key `Vec<u8>` allocs from `scan` were a measurable
    // share of the workload's time.
    let mut results = Vec::new();
    let mut err: Option<String> = None;
    txn.scan_foreach(&start, &end, &mut |key, _value| {
        let entity = index::current_avet_entity_at(key);
        let value_start = attr_prefix_len;
        let value_end = key.len() - 8;
        if let Some(val) = index::decode_current_value(&key[value_start..value_end]) {
            if op.evaluate(&val, value) {
                results.push((entity, val));
            }
        }
        true
    })
    .map_err(|e| err.get_or_insert(e.to_string()).clone())?;

    if let Some(e) = err {
        return Err(e);
    }

    Ok(results)
}

