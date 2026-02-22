use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::mem::discriminant;

use crate::datom::{Datom, EntityId, TxId, Value};
use crate::index;
use crate::query::planner::{self, JoinSide, JoinStrategy, PlanNode, QueryPlan};
use crate::query::{Pattern, PredOp, Query, WhereClause};
use crate::schema::{FieldType, SchemaRegistry};
use crate::storage::ReadOps;

/// A set of variable bindings produced during query execution.
pub type Bindings = HashMap<String, Value>;

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
            Value::Enum { variant, .. } => variant.hash(state),
        }
    }
}

impl PartialEq for HashableValue {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for HashableValue {}

/// Compute a hash key from bindings for the given join variables.
fn join_key(bindings: &Bindings, join_vars: &[String]) -> Option<Vec<HashableValue>> {
    let mut key = Vec::with_capacity(join_vars.len());
    for var in join_vars {
        match bindings.get(var) {
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
// Plan-based execution entry point
// ---------------------------------------------------------------------------

/// Execute a query by planning then executing the plan.
pub fn execute_query(
    txn: &dyn ReadOps,
    query: &Query,
    schema: &SchemaRegistry,
) -> Result<QueryResult, String> {
    let plan = planner::plan_query(txn, query, schema)?;
    execute_plan(txn, &plan, schema)
}

/// Execute a pre-built query plan.
pub fn execute_plan(
    txn: &dyn ReadOps,
    plan: &QueryPlan,
    schema: &SchemaRegistry,
) -> Result<QueryResult, String> {
    let binding_sets = execute_node(txn, &plan.root, plan.as_of, schema)?;

    // The Project node determines which variables to return
    let columns = match &plan.root {
        PlanNode::Project { variables, .. } => variables.clone(),
        _ => vec![],
    };

    let rows = binding_sets
        .into_iter()
        .map(|bindings| {
            columns
                .iter()
                .map(|var| {
                    bindings
                        .get(var)
                        .cloned()
                        .unwrap_or(Value::String("<unbound>".to_string()))
                })
                .collect()
        })
        .collect();

    Ok(QueryResult { columns, rows })
}

/// Recursively execute a plan node.
fn execute_node(
    txn: &dyn ReadOps,
    node: &PlanNode,
    as_of: Option<TxId>,
    schema: &SchemaRegistry,
) -> Result<Vec<Bindings>, String> {
    match node {
        PlanNode::Scan(scan) => {
            // Evaluate clause independently with empty bindings
            let empty = Bindings::new();
            evaluate_clause(txn, &scan.clause, &empty, as_of, schema)
        }
        PlanNode::Join {
            left,
            right,
            join_vars,
            strategy,
            ..
        } => match strategy {
            JoinStrategy::NestedLoop => {
                execute_nested_loop(txn, left, right, as_of, schema)
            }
            JoinStrategy::HashJoin { build_side } => {
                execute_hash_join(txn, left, right, join_vars, *build_side, as_of, schema)
            }
        },
        PlanNode::Project { input, .. } => {
            execute_node(txn, input, as_of, schema)
        }
    }
}

/// Nested-loop join: for each left binding, evaluate right clause with left bindings.
fn execute_nested_loop(
    txn: &dyn ReadOps,
    left: &PlanNode,
    right: &PlanNode,
    as_of: Option<TxId>,
    schema: &SchemaRegistry,
) -> Result<Vec<Bindings>, String> {
    let left_bindings = execute_node(txn, left, as_of, schema)?;

    // For nested loop, we need to get the right clause to call evaluate_clause
    // with the left bindings so index lookups can use bound variables.
    let right_clause = match right {
        PlanNode::Scan(scan) => &scan.clause,
        _ => {
            // For non-scan right sides, fall back to executing both independently
            // and merging (effectively a hash join)
            let right_bindings = execute_node(txn, right, as_of, schema)?;
            let mut results = Vec::new();
            for lb in &left_bindings {
                for rb in &right_bindings {
                    if let Some(merged) = merge_bindings(lb, rb) {
                        results.push(merged);
                    }
                }
            }
            return Ok(results);
        }
    };

    let mut results = Vec::new();
    for lb in &left_bindings {
        let matches = evaluate_clause(txn, right_clause, lb, as_of, schema)?;
        results.extend(matches);
    }

    Ok(results)
}

/// Hash join: execute both sides independently, build hash table, probe.
fn execute_hash_join(
    txn: &dyn ReadOps,
    left: &PlanNode,
    right: &PlanNode,
    join_vars: &[String],
    build_side: JoinSide,
    as_of: Option<TxId>,
    schema: &SchemaRegistry,
) -> Result<Vec<Bindings>, String> {
    let left_bindings = execute_node(txn, left, as_of, schema)?;
    let right_bindings = execute_node(txn, right, as_of, schema)?;

    let (build_bindings, probe_bindings) = match build_side {
        JoinSide::Left => (&left_bindings, &right_bindings),
        JoinSide::Right => (&right_bindings, &left_bindings),
    };

    // Build hash table
    let mut hash_table: HashMap<JoinKey, Vec<&Bindings>> = HashMap::new();
    for b in build_bindings {
        if let Some(key) = join_key(b, join_vars) {
            hash_table.entry(JoinKey(key)).or_default().push(b);
        }
    }

    // Probe
    let mut results = Vec::new();
    for pb in probe_bindings {
        if let Some(key) = join_key(pb, join_vars) {
            if let Some(matches) = hash_table.get(&JoinKey(key)) {
                for mb in matches {
                    if let Some(merged) = merge_bindings(pb, mb) {
                        results.push(merged);
                    }
                }
            }
        }
    }

    Ok(results)
}

/// Merge two binding sets. Returns None if there's a conflict
/// (same variable bound to different values).
fn merge_bindings(left: &Bindings, right: &Bindings) -> Option<Bindings> {
    let mut merged = left.clone();
    for (k, v) in right {
        if let Some(existing) = merged.get(k) {
            if existing != v {
                return None;
            }
        } else {
            merged.insert(k.clone(), v.clone());
        }
    }
    Some(merged)
}

// ---------------------------------------------------------------------------
// Clause-level execution (leaf engine — unchanged from original)
// ---------------------------------------------------------------------------

/// Evaluate a single where clause against current bindings.
/// Returns extended binding sets for each matching entity.
pub fn evaluate_clause(
    txn: &dyn ReadOps,
    clause: &WhereClause,
    bindings: &Bindings,
    as_of: Option<TxId>,
    schema: &SchemaRegistry,
) -> Result<Vec<Bindings>, String> {
    // Check if the entity variable is already bound
    let bound_entity = bindings.get(&clause.bind).and_then(|v| {
        if let Value::Ref(id) = v {
            Some(*id)
        } else {
            None
        }
    });

    // Compute needed fields early so we can optimize entity discovery
    let needed_fields = compute_needed_fields(clause, schema);

    // Track field values already known from index scans, keyed by entity ID
    let mut known_fields: HashMap<EntityId, HashMap<String, Value>> = HashMap::new();
    let attr_prefix = format!("{}/", clause.entity_type);
    let mut bulk_loaded = false;

    let use_current = as_of.is_none();

    let entities = if let Some(eid) = bound_entity {
        // Entity already bound — just check this one
        vec![eid]
    } else {
        // Try to use AVET index for constant, bound, or range patterns
        let plans = find_indexable_patterns(clause, bindings, schema);
        if !plans.is_empty() {
            let mut candidate_sets: Vec<Vec<EntityId>> = Vec::new();
            for plan in &plans {
                match plan {
                    AttrPlan::Exact(attr, value) => {
                        let eids = if use_current {
                            find_entities_by_avet_current(txn, attr, value)?
                        } else {
                            find_entities_by_avet(txn, attr, value, as_of)?
                        };
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
                        let results = if use_current {
                            find_entities_by_avet_range_current(txn, attr, op, value)?
                        } else {
                            find_entities_by_avet_range(txn, attr, op, value, as_of)?
                        };
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
                let (entities, field_map) = if use_current {
                    scan_entities_and_field_current(
                        txn,
                        &clause.entity_type,
                        &required_field,
                    )?
                } else {
                    scan_entities_and_field_via_aevt(
                        txn,
                        &clause.entity_type,
                        &required_field,
                        as_of,
                    )?
                };
                known_fields = field_map;

                // Bulk-load remaining needed fields
                let remaining: Vec<String> = needed
                    .iter()
                    .filter(|f| *f != &required_field)
                    .cloned()
                    .collect();
                if !remaining.is_empty() {
                    let entity_set: HashSet<EntityId> = entities.iter().copied().collect();
                    let extra = if use_current {
                        bulk_load_fields_current(
                            txn,
                            &clause.entity_type,
                            &remaining,
                            &entity_set,
                        )?
                    } else {
                        bulk_load_fields_via_aevt(
                            txn,
                            &clause.entity_type,
                            &remaining,
                            &entity_set,
                            as_of,
                        )?
                    };
                    for (eid, fields) in extra {
                        known_fields.entry(eid).or_default().extend(fields);
                    }
                }
                bulk_loaded = true;
                entities
            } else if use_current {
                find_entities_of_type_current(txn, &clause.entity_type)?
            } else {
                find_entities_of_type(txn, &clause.entity_type, as_of)?
            }
        } else {
            // Fall back to full type scan
            if use_current {
                find_entities_of_type_current(txn, &clause.entity_type)?
            } else {
                find_entities_of_type(txn, &clause.entity_type, as_of)?
            }
        }
    };

    // Bulk-load fields via AEVT scan when doing a full type scan with many entities.
    // Instead of N individual EAVT scans per field, we do 1 AEVT scan per field.
    if !bulk_loaded && known_fields.is_empty() && entities.len() > 16 {
        if let Some(ref needed) = needed_fields {
            let entity_set: HashSet<EntityId> = entities.iter().copied().collect();
            known_fields = if use_current {
                bulk_load_fields_current(
                    txn,
                    &clause.entity_type,
                    needed,
                    &entity_set,
                )?
            } else {
                bulk_load_fields_via_aevt(
                    txn,
                    &clause.entity_type,
                    needed,
                    &entity_set,
                    as_of,
                )?
            };
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

            let mut fields = if use_current {
                load_entity_fields_current(
                    txn,
                    eid,
                    &clause.entity_type,
                    field_filter.as_deref(),
                )?
            } else {
                load_entity_fields(
                    txn,
                    eid,
                    &clause.entity_type,
                    as_of,
                    field_filter.as_deref(),
                )?
            };

            // Merge in known values from AVET scans
            for (k, v) in entity_known {
                fields.entry(k).or_insert(v);
            }

            fields
        };

        // Try to match all field patterns
        if let Some(new_bindings) =
            match_field_patterns(&clause.field_patterns, &fields, bindings)
        {
            let mut extended = bindings.clone();
            // Bind the entity variable as a Ref
            extended.insert(clause.bind.clone(), Value::Ref(eid));
            extended.extend(new_bindings);
            results.push(extended);
        }
    }

    Ok(results)
}

/// Find patterns in a clause that can use the AVET index.
/// Returns plans for exact lookups and range scans.
fn find_indexable_patterns(
    clause: &WhereClause,
    bindings: &Bindings,
    schema: &SchemaRegistry,
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
                if let Some(bound_value) = bindings.get(var) {
                    plans.push(AttrPlan::Exact(attr, bound_value.clone()));
                }
            }
            Pattern::Predicate { op, value } => {
                // Range predicates use AVET index (Ne excluded — not worth it)
                if !matches!(op, PredOp::Ne) {
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
    attr: &str,
    value: &Value,
    as_of: Option<TxId>,
) -> Result<Vec<EntityId>, String> {
    let prefix = index::avet_attr_value_prefix(attr, value);
    let end = index::prefix_end(&prefix);

    let entries = txn
        .scan(&prefix, &end)
        .map_err(|e| e.to_string())?;

    let mut entity_datoms: HashMap<EntityId, Vec<Datom>> = HashMap::new();

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
    attr: &str,
    op: &PredOp,
    value: &Value,
    as_of: Option<TxId>,
) -> Result<Vec<(EntityId, Value)>, String> {
    let type_tag = value.type_tag();

    // Compute tight scan bounds based on the predicate
    let (start, end) = match op {
        PredOp::Gt | PredOp::Gte => {
            // Scan from the target value to the end of the attribute
            let start = index::avet_attr_value_prefix(attr, value);
            let end = index::prefix_end(&index::avet_attr_prefix(attr));
            (start, end)
        }
        PredOp::Lt => {
            // Scan from the start of the type to just before the target value
            let start = index::avet_attr_type_prefix(attr, type_tag);
            let end = index::avet_attr_value_prefix(attr, value);
            (start, end)
        }
        PredOp::Lte => {
            // Scan from the start of the type through the target value (inclusive)
            let start = index::avet_attr_type_prefix(attr, type_tag);
            let end = index::prefix_end(&index::avet_attr_value_prefix(attr, value));
            (start, end)
        }
        PredOp::Ne => {
            // Full type scan — skip exact matches in post-filter
            let start = index::avet_attr_type_prefix(attr, type_tag);
            let end = index::prefix_end(&index::avet_attr_prefix(attr));
            (start, end)
        }
    };

    let entries = txn.scan(&start, &end).map_err(|e| e.to_string())?;

    let mut entity_datoms: HashMap<EntityId, Vec<Datom>> = HashMap::new();

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
    type_name: &str,
    as_of: Option<TxId>,
) -> Result<Vec<EntityId>, String> {
    let type_value = Value::String(type_name.to_string());
    let prefix = index::avet_attr_value_prefix("__type", &type_value);
    let end = index::prefix_end(&prefix);

    let entries = txn
        .scan(&prefix, &end)
        .map_err(|e| e.to_string())?;

    // Group datoms by entity, then resolve history properly
    let mut entity_datoms: HashMap<EntityId, Vec<Datom>> = HashMap::new();

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
    entity_type: &str,
    field_name: &str,
    as_of: Option<TxId>,
) -> Result<(Vec<EntityId>, HashMap<EntityId, HashMap<String, Value>>), String> {
    let attr = format!("{}/{}", entity_type, field_name);
    let prefix = index::aevt_attr_prefix(&attr);
    let end = index::prefix_end(&prefix);

    let entries = txn.scan(&prefix, &end).map_err(|e| e.to_string())?;

    let mut entity_datoms: HashMap<EntityId, Vec<Datom>> = HashMap::new();
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
    entity_type: &str,
    field_names: &[String],
    entity_set: &HashSet<EntityId>,
    as_of: Option<TxId>,
) -> Result<HashMap<EntityId, HashMap<String, Value>>, String> {
    let mut result: HashMap<EntityId, HashMap<String, Value>> = HashMap::new();

    for field_name in field_names {
        let attr = format!("{}/{}", entity_type, field_name);
        let prefix = index::aevt_attr_prefix(&attr);
        let end = index::prefix_end(&prefix);

        let entries = txn.scan(&prefix, &end).map_err(|e| e.to_string())?;

        // Group by entity, filtering to our entity set
        let mut entity_datoms: HashMap<EntityId, Vec<Datom>> = HashMap::new();
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
            let prefix = index::eavt_entity_attr_prefix(entity, &attr);
            let end = index::prefix_end(&prefix);
            let entries = txn.scan(&prefix, &end).map_err(|e| e.to_string())?;
            for (key, _) in &entries {
                if let Some(datom) = index::decode_datom_from_eavt(key) {
                    if let Some(max_tx) = as_of {
                        if datom.tx > max_tx {
                            continue;
                        }
                    }
                    attr_state
                        .entry(datom.attribute.clone())
                        .or_default()
                        .push(datom);
                }
            }
        }
    } else {
        // Full entity scan (original behavior)
        let prefix = index::eavt_entity_prefix(entity);
        let end = index::prefix_end(&prefix);
        let entries = txn.scan(&prefix, &end).map_err(|e| e.to_string())?;
        for (key, _) in &entries {
            if let Some(datom) = index::decode_datom_from_eavt(key) {
                if let Some(max_tx) = as_of {
                    if datom.tx > max_tx {
                        continue;
                    }
                }
                attr_state
                    .entry(datom.attribute.clone())
                    .or_default()
                    .push(datom);
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

/// Try to match field patterns against entity fields. Returns new bindings if successful.
fn match_field_patterns(
    patterns: &[(String, Pattern)],
    fields: &HashMap<String, Value>,
    existing_bindings: &Bindings,
) -> Option<Bindings> {
    let mut new_bindings = Bindings::new();

    for (field_name, pattern) in patterns {
        match pattern {
            Pattern::Variable(var) => {
                // For enum fields, a bare variable on an enum field binds the tag
                // Check if this is an enum field by looking for __tag
                let tag_key = format!("{}/__tag", field_name);
                if let Some(tag_val) = fields.get(&tag_key) {
                    // This is an enum field — bind the tag
                    if let Some(bound_value) = existing_bindings
                        .get(var)
                        .or_else(|| new_bindings.get(var))
                    {
                        if bound_value != tag_val {
                            return None;
                        }
                    } else {
                        new_bindings.insert(var.clone(), tag_val.clone());
                    }
                } else {
                    // Regular scalar field — use Null for missing optional fields
                    let field_val = fields
                        .get(field_name)
                        .cloned()
                        .unwrap_or(Value::Null);
                    if let Some(bound_value) = existing_bindings
                        .get(var)
                        .or_else(|| new_bindings.get(var))
                    {
                        if *bound_value != field_val {
                            return None;
                        }
                    } else {
                        new_bindings.insert(var.clone(), field_val);
                    }
                }
            }
            Pattern::Constant(expected) => {
                // For enum fields, a bare constant matches the tag
                let tag_key = format!("{}/__tag", field_name);
                if let Some(tag_val) = fields.get(&tag_key) {
                    if tag_val != expected {
                        return None;
                    }
                } else {
                    let field_val = fields.get(field_name)?;
                    if field_val != expected {
                        return None;
                    }
                }
            }
            Pattern::Predicate { op, value } => {
                let field_val = fields.get(field_name)?;
                if !op.evaluate(field_val, value) {
                    return None;
                }
            }
            Pattern::EnumMatch {
                variant,
                field_patterns,
            } => {
                // Match enum tag
                let tag_key = format!("{}/__tag", field_name);
                let tag_val = fields.get(&tag_key)?;

                match variant.as_ref() {
                    Pattern::Variable(var) => {
                        if let Some(bound_value) = existing_bindings
                            .get(var)
                            .or_else(|| new_bindings.get(var))
                        {
                            if bound_value != tag_val {
                                return None;
                            }
                        } else {
                            new_bindings.insert(var.clone(), tag_val.clone());
                        }
                    }
                    Pattern::Constant(expected) => {
                        if tag_val != expected {
                            return None;
                        }
                    }
                    _ => return None,
                }

                // Get the variant name to construct field keys
                let variant_name = match tag_val {
                    Value::String(s) => s.as_str(),
                    _ => return None,
                };

                // Match variant field patterns
                for (vf_name, vf_pattern) in field_patterns {
                    let vf_key = format!("{}.{}/{}", field_name, variant_name, vf_name);
                    match vf_pattern {
                        Pattern::Variable(var) => {
                            if let Some(bound_value) = existing_bindings
                                .get(var)
                                .or_else(|| new_bindings.get(var))
                            {
                                let vf_val = fields.get(&vf_key)?;
                                if bound_value != vf_val {
                                    return None;
                                }
                            } else {
                                let vf_val = fields.get(&vf_key)?;
                                new_bindings.insert(var.clone(), vf_val.clone());
                            }
                        }
                        Pattern::Constant(expected) => {
                            let vf_val = fields.get(&vf_key)?;
                            if vf_val != expected {
                                return None;
                            }
                        }
                        Pattern::Predicate { op, value } => {
                            let vf_val = fields.get(&vf_key)?;
                            if !op.evaluate(vf_val, value) {
                                return None;
                            }
                        }
                        Pattern::EnumMatch { .. } => {
                            // Nested enum matching not supported
                            return None;
                        }
                    }
                }
            }
        }
    }

    Some(new_bindings)
}

// ---------------------------------------------------------------------------
// Current-state index query functions (used when as_of is None)
// ---------------------------------------------------------------------------

/// Find all entity IDs of a given type via CURRENT_AVET index.
/// No retraction resolution needed — current state index only has live entries.
fn find_entities_of_type_current(
    txn: &dyn ReadOps,
    type_name: &str,
) -> Result<Vec<EntityId>, String> {
    let type_value = Value::String(type_name.to_string());
    let prefix = index::current_avet_attr_value_prefix("__type", &type_value);
    let end = index::prefix_end(&prefix);
    let mut entities = Vec::new();
    txn.scan_foreach(&prefix, &end, &mut |key, _value| {
        entities.push(index::current_avet_entity_at(key));
        true
    })
    .map_err(|e| e.to_string())?;
    Ok(entities)
}

/// Find entity IDs by exact (attr, value) via CURRENT_AVET index.
fn find_entities_by_avet_current(
    txn: &dyn ReadOps,
    attr: &str,
    value: &Value,
) -> Result<Vec<EntityId>, String> {
    let prefix = index::current_avet_attr_value_prefix(attr, value);
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
    attr: &str,
    op: &PredOp,
    value: &Value,
) -> Result<Vec<(EntityId, Value)>, String> {
    let type_tag = value.type_tag();

    let (start, end) = match op {
        PredOp::Gt | PredOp::Gte => {
            let start = index::current_avet_attr_value_prefix(attr, value);
            let end = index::prefix_end(&index::current_avet_attr_prefix(attr));
            (start, end)
        }
        PredOp::Lt => {
            let start = index::current_avet_attr_type_prefix(attr, type_tag);
            let end = index::current_avet_attr_value_prefix(attr, value);
            (start, end)
        }
        PredOp::Lte => {
            let start = index::current_avet_attr_type_prefix(attr, type_tag);
            let end = index::prefix_end(&index::current_avet_attr_value_prefix(attr, value));
            (start, end)
        }
        PredOp::Ne => {
            let start = index::current_avet_attr_type_prefix(attr, type_tag);
            let end = index::prefix_end(&index::current_avet_attr_prefix(attr));
            (start, end)
        }
    };

    // We need to decode the value from the key for range results.
    // CURRENT_AVET key: [prefix][attr_len][attr_bytes][type_tag][value_data][entity_id(8)]
    // We know the attr prefix length, so we can decode value from the key.
    let attr_prefix_len = 1 + 2 + attr.as_bytes().len(); // prefix byte + u16 len + attr bytes

    let entries = txn.scan(&start, &end).map_err(|e| e.to_string())?;
    let mut results = Vec::new();

    for (key, _) in &entries {
        // Decode the full datom from the CURRENT_AVET key to get entity + value
        // Key layout: [0x12][attr_len(u16)][attr_bytes][type_tag][value_data][entity_id(u64)]
        let entity = index::current_avet_entity_at(key);
        // Decode the value portion: starts after attr prefix, ends 8 bytes before key end
        let value_start = attr_prefix_len;
        let value_end = key.len() - 8;
        if let Some(val) = index::decode_current_value(&key[value_start..value_end]) {
            if op.evaluate(&val, value) {
                results.push((entity, val));
            }
        }
    }

    Ok(results)
}

/// Scan CURRENT_AEVT for a single field to discover entity IDs AND load field values.
fn scan_entities_and_field_current(
    txn: &dyn ReadOps,
    entity_type: &str,
    field_name: &str,
) -> Result<(Vec<EntityId>, HashMap<EntityId, HashMap<String, Value>>), String> {
    let attr = format!("{}/{}", entity_type, field_name);
    let prefix = index::current_aevt_attr_prefix(&attr);
    let end = index::prefix_end(&prefix);
    let attr_byte_len = attr.as_bytes().len();

    let mut entities = Vec::new();
    let mut field_map: HashMap<EntityId, HashMap<String, Value>> = HashMap::new();

    txn.scan_foreach(&prefix, &end, &mut |key, value| {
        let eid = index::current_aevt_entity_at(key, attr_byte_len);
        if let Some(val) = index::decode_current_value(value) {
            entities.push(eid);
            let mut fields = HashMap::new();
            fields.insert(field_name.to_string(), val);
            field_map.insert(eid, fields);
        }
        true
    })
    .map_err(|e| e.to_string())?;

    Ok((entities, field_map))
}

/// Bulk-load field values for many entities via CURRENT_AEVT index.
fn bulk_load_fields_current(
    txn: &dyn ReadOps,
    entity_type: &str,
    field_names: &[String],
    entity_set: &HashSet<EntityId>,
) -> Result<HashMap<EntityId, HashMap<String, Value>>, String> {
    let mut result: HashMap<EntityId, HashMap<String, Value>> = HashMap::new();

    for field_name in field_names {
        let attr = format!("{}/{}", entity_type, field_name);
        let prefix = index::current_aevt_attr_prefix(&attr);
        let end = index::prefix_end(&prefix);
        let attr_byte_len = attr.as_bytes().len();

        txn.scan_foreach(&prefix, &end, &mut |key, value| {
            let eid = index::current_aevt_entity_at(key, attr_byte_len);
            if entity_set.contains(&eid) {
                if let Some(val) = index::decode_current_value(value) {
                    result
                        .entry(eid)
                        .or_default()
                        .insert(field_name.clone(), val);
                }
            }
            true
        })
        .map_err(|e| e.to_string())?;
    }

    Ok(result)
}

/// Load current field values for an entity from CURRENT_AEVT (single get per field).
fn load_entity_fields_current(
    txn: &dyn ReadOps,
    entity: EntityId,
    entity_type: &str,
    field_filter: Option<&[String]>,
) -> Result<HashMap<String, Value>, String> {
    let mut fields = HashMap::new();

    if let Some(fields_to_load) = field_filter {
        for field_name in fields_to_load {
            let attr = format!("{}/{}", entity_type, field_name);
            // Key = [0x11][attr_len][attr_bytes][entity_id] — independent of value
            let mut lookup_key = index::current_aevt_attr_prefix(&attr);
            lookup_key.extend_from_slice(&entity.to_be_bytes());

            if let Some(val_bytes) = txn.get(&lookup_key).map_err(|e| e.to_string())? {
                if let Some(val) = index::decode_current_value(&val_bytes) {
                    fields.insert(field_name.clone(), val);
                }
            }
        }
    } else {
        // Full entity scan: scan all attrs for this entity type that start with Type/
        // We need to scan the entity's EAVT prefix and filter. But with current state,
        // we don't have an entity-first current index. Fall back to EAVT historical.
        // This only happens for enum fields or >4 field patterns, which is rare.
        let prefix = index::eavt_entity_prefix(entity);
        let end_key = index::prefix_end(&prefix);
        let entries = txn.scan(&prefix, &end_key).map_err(|e| e.to_string())?;

        let attr_prefix = format!("{}/", entity_type);
        let mut attr_state: HashMap<String, Vec<Datom>> = HashMap::new();
        for (key, _) in &entries {
            if let Some(datom) = index::decode_datom_from_eavt(key) {
                attr_state
                    .entry(datom.attribute.clone())
                    .or_default()
                    .push(datom);
            }
        }

        let resolved = crate::db::resolve_current_values(attr_state, None);
        for (attr, value) in resolved {
            if let Some(field_name) = attr.strip_prefix(&attr_prefix) {
                fields.insert(field_name.to_string(), value);
            }
        }
    }

    Ok(fields)
}
