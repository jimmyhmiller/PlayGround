use std::collections::HashMap;
use std::fmt;

use crate::datom::{TxId, Value};
use crate::intern::AttrInterner;
use crate::query::{Clause, Pattern, PredOp, Query, WhereClause};
use crate::schema::SchemaRegistry;
use crate::storage::ReadOps;

// ---------------------------------------------------------------------------
// Plan node types
// ---------------------------------------------------------------------------

/// How to find candidate entities for a clause.
#[derive(Debug, Clone)]
pub enum ScanStrategy {
    /// Scan AVET for __type = entity_type (full type scan)
    TypeScan,
    /// Exact AVET lookups (one or more attr=value pairs, intersected)
    IndexLookup(Vec<(String, Value)>),
    /// AVET range scan on one attribute
    RangeScan { attr: String, op: PredOp, value: Value },
    /// Executed as a per-tuple index probe seeded by an upstream join.
    /// The plan's `estimate_clause` runs before join variables are bound,
    /// so a clause whose only selective field is a join variable is labelled
    /// `TypeScan` there; this variant records that at execution time the
    /// nested-loop seeds the right side by `field` (the bind entity, or an
    /// indexed/unique field) — an O(log n) probe, not a full scan. Purely
    /// an EXPLAIN-accuracy annotation; the executor's behaviour is unchanged.
    IndexProbe { field: String, var: String },
}

/// Which side of a join to use as the build side for hash join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinSide {
    Left,
    Right,
}

/// How to execute a join between two plan nodes.
#[derive(Debug, Clone)]
pub enum JoinStrategy {
    NestedLoop,
    HashJoin { build_side: JoinSide },
}

/// Cardinality estimate for a plan node.
#[derive(Debug, Clone)]
pub struct CostEstimate {
    pub estimated_rows: usize,
}

/// A single clause scanned independently.
#[derive(Debug, Clone)]
pub struct ClauseScan {
    pub clause: WhereClause,
    pub strategy: ScanStrategy,
    pub estimate: CostEstimate,
}

/// A node in the query plan tree.
#[derive(Debug, Clone)]
pub enum PlanNode {
    Scan(ClauseScan),
    Join {
        left: Box<PlanNode>,
        right: Box<PlanNode>,
        join_vars: Vec<String>,
        strategy: JoinStrategy,
        estimate: CostEstimate,
    },
    Project {
        input: Box<PlanNode>,
        variables: Vec<String>,
    },
    /// Disjunction: the deduplicated union of its branches' bindings.
    Union {
        branches: Vec<PlanNode>,
        estimate: CostEstimate,
    },
    /// Negation as failure: pass through each `input` row for which `anti`
    /// (re-opened with that row's bindings) produces no solution.
    Negation {
        input: Box<PlanNode>,
        anti: Box<PlanNode>,
        estimate: CostEstimate,
    },
}

impl PlanNode {
    pub fn estimate(&self) -> &CostEstimate {
        match self {
            PlanNode::Scan(s) => &s.estimate,
            PlanNode::Join { estimate, .. } => estimate,
            PlanNode::Project { input, .. } => input.estimate(),
            PlanNode::Union { estimate, .. } => estimate,
            PlanNode::Negation { estimate, .. } => estimate,
        }
    }
}

/// Maps variable names to fixed slot indices in a tuple.
#[derive(Debug, Clone)]
pub struct SlotMap {
    name_to_slot: HashMap<String, usize>,
    pub num_slots: usize,
}

impl SlotMap {
    /// Build a SlotMap by collecting all unique variable names from a query.
    pub fn from_query(query: &Query) -> Self {
        let mut name_to_slot = HashMap::new();
        let mut next_slot = 0usize;

        // find variables
        for var in &query.find {
            Self::ensure_slot(var, &mut name_to_slot, &mut next_slot);
        }

        // where clause variables (across the whole And/Or/Not tree)
        query.where_clause.for_each_pattern(&mut |clause| {
            Self::ensure_slot(&clause.bind, &mut name_to_slot, &mut next_slot);
            Self::collect_pattern_vars(&clause.field_patterns, &mut name_to_slot, &mut next_slot);
        });

        SlotMap {
            name_to_slot,
            num_slots: next_slot,
        }
    }

    fn ensure_slot(name: &str, map: &mut HashMap<String, usize>, next: &mut usize) {
        if !map.contains_key(name) {
            map.insert(name.to_string(), *next);
            *next += 1;
        }
    }

    fn collect_pattern_vars(
        patterns: &[(String, Pattern)],
        map: &mut HashMap<String, usize>,
        next: &mut usize,
    ) {
        for (_field_name, pattern) in patterns {
            Self::collect_from_pattern(pattern, map, next);
        }
    }

    fn collect_from_pattern(
        pattern: &Pattern,
        map: &mut HashMap<String, usize>,
        next: &mut usize,
    ) {
        match pattern {
            Pattern::Variable(v) => {
                if !map.contains_key(v) {
                    map.insert(v.clone(), *next);
                    *next += 1;
                }
            }
            Pattern::BoundPredicate { var, .. } => {
                if !map.contains_key(var) {
                    map.insert(var.clone(), *next);
                    *next += 1;
                }
            }
            Pattern::EnumMatch { variant, field_patterns } => {
                Self::collect_from_pattern(variant, map, next);
                Self::collect_pattern_vars(field_patterns, map, next);
            }
            Pattern::Near { score_var: Some(sv), .. }
            | Pattern::Search { score_var: Some(sv), .. } => {
                if !map.contains_key(sv) {
                    map.insert(sv.clone(), *next);
                    *next += 1;
                }
            }
            Pattern::Constant(_)
            | Pattern::Predicate { .. }
            | Pattern::Near { .. }
            | Pattern::Search { .. } => {}
        }
    }

    /// Look up the slot index for a variable name.
    #[inline]
    pub fn slot(&self, name: &str) -> Option<usize> {
        self.name_to_slot.get(name).copied()
    }
}

/// The complete query plan.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub root: PlanNode,
    pub as_of: Option<TxId>,
    pub slot_map: SlotMap,
}

// ---------------------------------------------------------------------------
// Display — Unicode tree format
// ---------------------------------------------------------------------------

impl fmt::Display for QueryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_node(&self.root, f, "", true)
    }
}

fn fmt_node(node: &PlanNode, f: &mut fmt::Formatter<'_>, prefix: &str, is_last: bool) -> fmt::Result {
    let connector = if prefix.is_empty() {
        ""
    } else if is_last {
        "└── "
    } else {
        "├── "
    };

    match node {
        PlanNode::Scan(scan) => {
            let strategy_str = match &scan.strategy {
                ScanStrategy::TypeScan => "TypeScan".to_string(),
                ScanStrategy::IndexLookup(pairs) => {
                    let keys: Vec<_> = pairs.iter().map(|(k, _)| k.clone()).collect();
                    format!("IndexLookup({})", keys.join(", "))
                }
                ScanStrategy::RangeScan { attr, op, value } => {
                    format!("RangeScan({} {:?} {})", attr, op, value)
                }
                ScanStrategy::IndexProbe { field, var } => {
                    format!("IndexProbe({} = {})", field, var)
                }
            };
            writeln!(
                f,
                "{}{}Scan {} ({}, est. {} rows)",
                prefix, connector, scan.clause.entity_type, strategy_str, scan.estimate.estimated_rows
            )?;

            // Print details
            let detail_prefix = if prefix.is_empty() {
                "      ".to_string()
            } else if is_last {
                format!("{}    ", prefix)
            } else {
                format!("{}│   ", prefix)
            };

            writeln!(f, "{}bind: {}", detail_prefix, scan.clause.bind)?;

            let field_names: Vec<_> = scan.clause.field_patterns.iter().map(|(k, _)| k.clone()).collect();
            if !field_names.is_empty() {
                writeln!(f, "{}load: {}", detail_prefix, field_names.join(", "))?;
            }
        }
        PlanNode::Join { left, right, join_vars, strategy, estimate } => {
            let strategy_str = match strategy {
                JoinStrategy::NestedLoop => "NestedLoop".to_string(),
                JoinStrategy::HashJoin { build_side } => {
                    let side = match build_side {
                        JoinSide::Left => "left",
                        JoinSide::Right => "right",
                    };
                    format!("HashJoin, build={}", side)
                }
            };
            let vars_str = join_vars.join(", ");
            writeln!(
                f,
                "{}{}{} on [{}] (est. {} rows)",
                prefix, connector, strategy_str, vars_str, estimate.estimated_rows
            )?;

            let child_prefix = if prefix.is_empty() {
                "".to_string()
            } else if is_last {
                format!("{}    ", prefix)
            } else {
                format!("{}│   ", prefix)
            };

            fmt_node(left, f, &child_prefix, false)?;
            fmt_node(right, f, &child_prefix, true)?;
        }
        PlanNode::Project { input, variables } => {
            writeln!(f, "{}{}Project [{}]", prefix, connector, variables.join(", "))?;

            let child_prefix = if prefix.is_empty() {
                "".to_string()
            } else if is_last {
                format!("{}    ", prefix)
            } else {
                format!("{}│   ", prefix)
            };

            fmt_node(input, f, &child_prefix, true)?;
        }
        PlanNode::Union { branches, estimate } => {
            writeln!(
                f,
                "{}{}Union ({} branches, est. {} rows)",
                prefix, connector, branches.len(), estimate.estimated_rows
            )?;
            let child_prefix = if prefix.is_empty() {
                "".to_string()
            } else if is_last {
                format!("{}    ", prefix)
            } else {
                format!("{}│   ", prefix)
            };
            for (i, b) in branches.iter().enumerate() {
                fmt_node(b, f, &child_prefix, i == branches.len() - 1)?;
            }
        }
        PlanNode::Negation { input, anti, estimate } => {
            writeln!(
                f,
                "{}{}Negation (est. {} rows)",
                prefix, connector, estimate.estimated_rows
            )?;
            let child_prefix = if prefix.is_empty() {
                "".to_string()
            } else if is_last {
                format!("{}    ", prefix)
            } else {
                format!("{}│   ", prefix)
            };
            fmt_node(input, f, &child_prefix, false)?;
            fmt_node(anti, f, &child_prefix, true)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

impl QueryPlan {
    pub fn to_json(&self) -> serde_json::Value {
        node_to_json(&self.root)
    }
}

fn node_to_json(node: &PlanNode) -> serde_json::Value {
    match node {
        PlanNode::Scan(scan) => {
            let strategy = match &scan.strategy {
                ScanStrategy::TypeScan => "TypeScan".to_string(),
                ScanStrategy::IndexLookup(pairs) => {
                    let keys: Vec<_> = pairs.iter().map(|(k, _)| k.clone()).collect();
                    format!("IndexLookup({})", keys.join(", "))
                }
                ScanStrategy::RangeScan { attr, op, .. } => {
                    format!("RangeScan({} {:?})", attr, op)
                }
                ScanStrategy::IndexProbe { field, var } => {
                    format!("IndexProbe({} = {})", field, var)
                }
            };

            let fields: Vec<_> = scan.clause.field_patterns.iter().map(|(k, _)| k.clone()).collect();

            serde_json::json!({
                "node": "Scan",
                "type": scan.clause.entity_type,
                "bind": scan.clause.bind,
                "strategy": strategy,
                "fields": fields,
                "estimated_rows": scan.estimate.estimated_rows,
            })
        }
        PlanNode::Join { left, right, join_vars, strategy, estimate } => {
            let (strategy_name, build_side) = match strategy {
                JoinStrategy::NestedLoop => ("NestedLoop", serde_json::Value::Null),
                JoinStrategy::HashJoin { build_side } => {
                    let side = match build_side {
                        JoinSide::Left => "left",
                        JoinSide::Right => "right",
                    };
                    ("HashJoin", serde_json::json!(side))
                }
            };

            let mut obj = serde_json::json!({
                "node": strategy_name,
                "join_on": join_vars,
                "estimated_rows": estimate.estimated_rows,
                "left": node_to_json(left),
                "right": node_to_json(right),
            });

            if !build_side.is_null() {
                obj["build_side"] = build_side;
            }

            obj
        }
        PlanNode::Project { input, variables } => {
            serde_json::json!({
                "node": "Project",
                "variables": variables,
                "input": node_to_json(input),
            })
        }
        PlanNode::Union { branches, estimate } => {
            serde_json::json!({
                "node": "Union",
                "estimated_rows": estimate.estimated_rows,
                "branches": branches.iter().map(node_to_json).collect::<Vec<_>>(),
            })
        }
        PlanNode::Negation { input, anti, estimate } => {
            serde_json::json!({
                "node": "Negation",
                "estimated_rows": estimate.estimated_rows,
                "input": node_to_json(input),
                "anti": node_to_json(anti),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Planning logic
// ---------------------------------------------------------------------------

/// Main entry point: Query → QueryPlan
pub fn plan_query(
    txn: &dyn ReadOps,
    query: &Query,
    schema: &SchemaRegistry,
    interner: &AttrInterner,
) -> Result<QueryPlan, String> {
    if query.where_clause.is_empty() {
        return Err("query must have at least one where clause".into());
    }

    // Collapse double negation (`not { not { c } }` ⇒ `c`) so the inner
    // negation isn't rejected as a non-producing clause.
    let where_clause = query.where_clause.clone().simplified();

    let mut type_counts: HashMap<String, usize> = HashMap::new();
    let mut root = compile_clause(
        &where_clause,
        txn,
        schema,
        interner,
        query.as_of,
        &mut type_counts,
    )?;

    // EXPLAIN accuracy: relabel TypeScan children of NestedLoop joins that are
    // actually executed as per-tuple index probes (see ScanStrategy::IndexProbe).
    relabel_index_probes(&mut root, schema);

    // Wrap in Project
    let root = PlanNode::Project {
        input: Box::new(root),
        variables: query.find.clone(),
    };

    let slot_map = SlotMap::from_query(query);

    Ok(QueryPlan {
        root,
        as_of: query.as_of,
        slot_map,
    })
}

/// Walk the plan tree and relabel `TypeScan` strategies that are actually
/// executed as per-tuple index probes. A NestedLoop join seeds its right side
/// with the left row's bindings; when a join variable is the right scan's bind
/// entity (direct entity lookup) or matches an indexed/unique field on the
/// right clause, the executor probes the index rather than scanning the type.
/// This only changes the EXPLAIN label, never execution.
fn relabel_index_probes(node: &mut PlanNode, schema: &SchemaRegistry) {
    match node {
        PlanNode::Join { left, right, join_vars, strategy, .. } => {
            relabel_index_probes(left, schema);
            relabel_index_probes(right, schema);
            if matches!(strategy, JoinStrategy::NestedLoop) && !join_vars.is_empty() {
                if let PlanNode::Scan(scan) = right.as_mut() {
                    if matches!(scan.strategy, ScanStrategy::TypeScan) {
                        if let Some((field, var)) = probe_field(&scan.clause, join_vars, schema) {
                            scan.strategy = ScanStrategy::IndexProbe { field, var };
                        }
                    }
                }
            }
        }
        PlanNode::Project { input, .. } => relabel_index_probes(input, schema),
        PlanNode::Negation { input, anti, .. } => {
            relabel_index_probes(input, schema);
            relabel_index_probes(anti, schema);
        }
        PlanNode::Union { branches, .. } => {
            for b in branches {
                relabel_index_probes(b, schema);
            }
        }
        PlanNode::Scan(_) => {}
    }
}

/// If a join variable lets the executor probe an index on `clause`, return the
/// `(field, var)` describing it: the bind variable itself (direct entity lookup,
/// reported as `id`), or an indexed/unique field bound to a join var.
fn probe_field(
    clause: &WhereClause,
    join_vars: &[String],
    schema: &SchemaRegistry,
) -> Option<(String, String)> {
    for jv in join_vars {
        // The join var IS the entity — a direct by-id lookup.
        if jv == &clause.bind {
            return Some(("id".to_string(), jv.clone()));
        }
        // The join var is bound to an indexed/unique field on this clause.
        if let Some(type_def) = schema.get(&clause.entity_type) {
            for (field_name, pattern) in &clause.field_patterns {
                if pattern.bound_var() == Some(jv.as_str()) {
                    if let Some(fd) = type_def.get_field(field_name) {
                        if fd.unique || fd.indexed {
                            return Some((field_name.clone(), jv.clone()));
                        }
                    }
                }
            }
        }
    }
    None
}

/// Compile a where clause tree into a plan node. Flat conjunctions of
/// patterns use the cost-based join planner; trees containing `Or`/`Not`
/// use a general nested-loop compiler.
fn compile_clause(
    clause: &Clause,
    txn: &dyn ReadOps,
    schema: &SchemaRegistry,
    interner: &AttrInterner,
    as_of: Option<TxId>,
    type_counts: &mut HashMap<String, usize>,
) -> Result<PlanNode, String> {
    match clause {
        Clause::Pattern(wc) => {
            let (strategy, estimate) =
                estimate_clause(txn, interner, wc, schema, as_of, type_counts);
            Ok(PlanNode::Scan(ClauseScan {
                clause: wc.clone(),
                strategy,
                estimate,
            }))
        }
        Clause::Or(children) => {
            let mut branches = Vec::with_capacity(children.len());
            let mut est = 0usize;
            for c in children {
                let node = compile_clause(c, txn, schema, interner, as_of, type_counts)?;
                est += node.estimate().estimated_rows;
                branches.push(node);
            }
            Ok(PlanNode::Union {
                branches,
                estimate: CostEstimate { estimated_rows: est.max(1) },
            })
        }
        Clause::Not(_) => {
            Err("negation (not) must appear inside a conjunction with a positive clause".into())
        }
        Clause::And(children) => {
            compile_and(children, txn, schema, interner, as_of, type_counts)
        }
    }
}

/// Compile a conjunction. A flat And of pattern leaves keeps the existing
/// cost-based planner. Otherwise positives are nested-loop joined and each
/// `not` child becomes a `Negation` wrapper.
fn compile_and(
    children: &[Clause],
    txn: &dyn ReadOps,
    schema: &SchemaRegistry,
    interner: &AttrInterner,
    as_of: Option<TxId>,
    type_counts: &mut HashMap<String, usize>,
) -> Result<PlanNode, String> {
    // Fast path: every child is a pattern leaf → cost-based join tree.
    if children.iter().all(|c| matches!(c, Clause::Pattern(_))) {
        let mut clause_estimates: Vec<(WhereClause, ScanStrategy, CostEstimate)> =
            Vec::with_capacity(children.len());
        for c in children {
            if let Clause::Pattern(wc) = c {
                let (strategy, estimate) =
                    estimate_clause(txn, interner, wc, schema, as_of, type_counts);
                clause_estimates.push((wc.clone(), strategy, estimate));
            }
        }
        clause_estimates.sort_by_key(|(_, _, est)| est.estimated_rows);
        return Ok(build_join_tree(&clause_estimates, schema));
    }

    // General path: split positives from negations.
    let mut positives: Vec<&Clause> = Vec::new();
    let mut negatives: Vec<&Clause> = Vec::new();
    for c in children {
        match c {
            Clause::Not(inner) => negatives.push(inner),
            other => positives.push(other),
        }
    }
    if positives.is_empty() {
        return Err("a conjunction of only negations is not allowed; add a positive clause".into());
    }

    // Compile positives, then order by estimate ascending and fold into a
    // left-deep nested-loop chain. NestedLoop joins via seeding (the right
    // subtree filters by the left row's bindings), so it composes with any
    // node type.
    let mut compiled: Vec<PlanNode> = Vec::with_capacity(positives.len());
    for c in positives {
        compiled.push(compile_clause(c, txn, schema, interner, as_of, type_counts)?);
    }
    compiled.sort_by_key(|n| n.estimate().estimated_rows);

    let mut tree = compiled.remove(0);
    for node in compiled {
        let join_vars: Vec<String> = node_bound_vars(&tree)
            .intersection(&node_bound_vars(&node))
            .cloned()
            .collect();
        let left_est = tree.estimate().estimated_rows;
        let right_est = node.estimate().estimated_rows;
        let join_est = if join_vars.is_empty() {
            left_est.saturating_mul(right_est)
        } else {
            left_est.min(right_est)
        };
        tree = PlanNode::Join {
            left: Box::new(tree),
            right: Box::new(node),
            join_vars,
            strategy: JoinStrategy::NestedLoop,
            estimate: CostEstimate { estimated_rows: join_est.max(1) },
        };
    }

    // Wrap with each negation. Compile the anti-subquery as its own clause.
    for inner in negatives {
        let anti = compile_clause(inner, txn, schema, interner, as_of, type_counts)?;
        let est = tree.estimate().estimated_rows;
        tree = PlanNode::Negation {
            input: Box::new(tree),
            anti: Box::new(anti),
            estimate: CostEstimate { estimated_rows: est },
        };
    }

    Ok(tree)
}

/// The set of variables a plan subtree binds. For a `Union`, only variables
/// bound by every branch are guaranteed (intersection). `Negation` binds
/// exactly what its positive input binds (the anti-side adds nothing).
fn node_bound_vars(node: &PlanNode) -> std::collections::HashSet<String> {
    use std::collections::HashSet;
    match node {
        PlanNode::Scan(s) => {
            let mut set = HashSet::new();
            set.insert(s.clause.bind.clone());
            for (_, pat) in &s.clause.field_patterns {
                collect_pattern_bound_vars(pat, &mut set);
            }
            set
        }
        PlanNode::Join { left, right, .. } => {
            let mut set = node_bound_vars(left);
            set.extend(node_bound_vars(right));
            set
        }
        PlanNode::Project { input, .. } => node_bound_vars(input),
        PlanNode::Negation { input, .. } => node_bound_vars(input),
        PlanNode::Union { branches, .. } => {
            let mut iter = branches.iter().map(node_bound_vars);
            let first = iter.next().unwrap_or_default();
            iter.fold(first, |acc, s| acc.intersection(&s).cloned().collect())
        }
    }
}

fn collect_pattern_bound_vars(pat: &Pattern, set: &mut std::collections::HashSet<String>) {
    match pat {
        Pattern::Variable(v) => {
            set.insert(v.clone());
        }
        Pattern::BoundPredicate { var, .. } => {
            set.insert(var.clone());
        }
        Pattern::EnumMatch { variant, field_patterns } => {
            collect_pattern_bound_vars(variant, set);
            for (_, p) in field_patterns {
                collect_pattern_bound_vars(p, set);
            }
        }
        Pattern::Near { score_var: Some(sv), .. }
        | Pattern::Search { score_var: Some(sv), .. } => {
            set.insert(sv.clone());
        }
        Pattern::Constant(_)
        | Pattern::Predicate { .. }
        | Pattern::Near { .. }
        | Pattern::Search { .. } => {}
    }
}

/// Estimate cardinality of a clause.
fn estimate_clause(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    clause: &WhereClause,
    schema: &SchemaRegistry,
    as_of: Option<TxId>,
    type_counts: &mut HashMap<String, usize>,
) -> (ScanStrategy, CostEstimate) {
    let type_def = schema.get(&clause.entity_type);

    // Check for exact constant on unique field → est = 1
    // Check for exact constants on non-unique → est = type_count / 10
    // Check for range → est = type_count / 3
    let mut has_exact = false;
    let mut has_unique_exact = false;
    let mut exact_pairs: Vec<(String, Value)> = Vec::new();
    let mut range_info: Option<(String, PredOp, Value)> = None;

    for (field_name, pattern) in &clause.field_patterns {
        match pattern {
            Pattern::Constant(value) => {
                let attr = type_def
                    .map(|td| td.attribute_name(field_name))
                    .unwrap_or_else(|| format!("{}/{}", clause.entity_type, field_name));
                exact_pairs.push((attr, value.clone()));
                has_exact = true;

                if let Some(td) = type_def {
                    if let Some(fd) = td.get_field(field_name) {
                        if fd.unique {
                            has_unique_exact = true;
                        }
                    }
                }
            }
            Pattern::Predicate { op, value } | Pattern::BoundPredicate { op, value, .. }
                if !matches!(op, PredOp::Ne) && !op.is_string_search() =>
            {
                if range_info.is_none() {
                    let attr = type_def
                        .map(|td| td.attribute_name(field_name))
                        .unwrap_or_else(|| format!("{}/{}", clause.entity_type, field_name));
                    range_info = Some((attr, op.clone(), value.clone()));
                }
            }
            _ => {}
        }
    }

    // Unique exact match: skip type count scan entirely (est = 1)
    if has_unique_exact {
        let strategy = ScanStrategy::IndexLookup(exact_pairs);
        let estimate = CostEstimate { estimated_rows: 1 };
        return (strategy, estimate);
    }

    // Defer type count scan until we actually need it
    let mut type_count = || -> usize {
        *type_counts
            .entry(clause.entity_type.clone())
            .or_insert_with(|| count_entities_of_type(txn, interner, &clause.entity_type, as_of))
    };

    if has_exact {
        let est = (type_count() / 10).max(1);
        let strategy = ScanStrategy::IndexLookup(exact_pairs);
        let estimate = CostEstimate { estimated_rows: est };
        return (strategy, estimate);
    }

    if let Some((attr, op, value)) = range_info {
        let est = (type_count() / 3).max(1);
        let strategy = ScanStrategy::RangeScan { attr, op, value };
        let estimate = CostEstimate { estimated_rows: est };
        return (strategy, estimate);
    }

    // No indexable patterns → full type scan
    (ScanStrategy::TypeScan, CostEstimate { estimated_rows: type_count().max(1) })
}

/// Count entities of a given type.
/// First tries the metadata key (O(1) lookup), falls back to AVET scan.
fn count_entities_of_type(
    txn: &dyn ReadOps,
    interner: &AttrInterner,
    type_name: &str,
    as_of: Option<TxId>,
) -> usize {
    use crate::index;

    // For current-state queries, try the metadata key first
    if as_of.is_none() {
        let meta_key = index::meta_key(&format!("type_count:{}", type_name));
        if let Ok(Some(bytes)) = txn.get(&meta_key) {
            if bytes.len() == 8 {
                return u64::from_be_bytes(bytes.try_into().unwrap()) as usize;
            }
        }
    }

    // Fall back to AVET scan (for as_of queries or if metadata doesn't exist).
    // If `__type` has never been interned, there's nothing to scan.
    let type_attr_id = match interner.lookup("__type") {
        Some(id) => id,
        None => return 0,
    };
    let type_value = Value::String(type_name.into());
    let prefix = index::avet_attr_value_prefix(type_attr_id, &type_value);
    let end = index::prefix_end(&prefix);

    let entries = match txn.scan(&prefix, &end) {
        Ok(entries) => entries,
        Err(_) => return 0,
    };

    // Simple count: resolve retraction history per entity
    let mut entity_state: HashMap<u64, bool> = HashMap::new();

    for (key, _) in &entries {
        if let Some(datom) = index::decode_datom_from_avet(key) {
            if let Some(max_tx) = as_of {
                if datom.tx > max_tx {
                    continue;
                }
            }
            // Last write wins (sorted by tx in index order)
            entity_state.insert(datom.entity, datom.added);
        }
    }

    entity_state.values().filter(|&&added| added).count()
}

/// Find shared join variables between two clauses.
///
/// A variable is a join var if it's the `bind` of one clause and appears
/// in a field pattern of the other, OR if it appears as a variable in
/// field patterns of both clauses.
pub fn find_join_vars(left: &WhereClause, right: &WhereClause) -> Vec<String> {
    let mut vars = Vec::new();

    // left.bind appears in right's field patterns
    for (_, pattern) in &right.field_patterns {
        if let Some(v) = pattern.bound_var() {
            if v == left.bind && !vars.iter().any(|x| x == v) {
                vars.push(v.to_string());
            }
        }
    }

    // right.bind appears in left's field patterns
    for (_, pattern) in &left.field_patterns {
        if let Some(v) = pattern.bound_var() {
            if v == right.bind && !vars.iter().any(|x| x == v) {
                vars.push(v.to_string());
            }
        }
    }

    // Variables appearing in field patterns of both
    let left_vars: Vec<&str> = left.field_patterns.iter()
        .filter_map(|(_, p)| p.bound_var())
        .collect();

    let right_vars: Vec<&str> = right.field_patterns.iter()
        .filter_map(|(_, p)| p.bound_var())
        .collect();

    for lv in &left_vars {
        for rv in &right_vars {
            if lv == rv && !vars.iter().any(|x| x == lv) {
                vars.push((*lv).to_string());
            }
        }
    }

    vars
}

/// Collect all clauses in a plan subtree.
fn collect_clauses(node: &PlanNode) -> Vec<&WhereClause> {
    match node {
        PlanNode::Scan(s) => vec![&s.clause],
        PlanNode::Join { left, right, .. } => {
            let mut v = collect_clauses(left);
            v.extend(collect_clauses(right));
            v
        }
        PlanNode::Project { input, .. } => collect_clauses(input),
        // The positive input contributes joinable clauses; the anti side does not.
        PlanNode::Negation { input, .. } => collect_clauses(input),
        PlanNode::Union { branches, .. } => {
            branches.iter().flat_map(|b| collect_clauses(b)).collect()
        }
    }
}

/// Find join vars between a plan subtree and a clause.
fn find_join_vars_with_tree(tree: &PlanNode, clause: &WhereClause) -> Vec<String> {
    let tree_clauses = collect_clauses(tree);
    let mut all_vars = Vec::new();
    for tc in tree_clauses {
        for v in find_join_vars(tc, clause) {
            if !all_vars.contains(&v) {
                all_vars.push(v);
            }
        }
    }
    all_vars
}

/// Choose join strategy.
fn choose_join_strategy(
    left_est: usize,
    right_est: usize,
    join_vars: &[String],
    right_clause: &WhereClause,
    schema: &SchemaRegistry,
) -> JoinStrategy {
    if !join_vars.is_empty() {
        for jv in join_vars {
            // Case 1: Join var IS the right clause's bind variable.
            // The right entity is directly bound by ID from the left tuple,
            // so evaluate_clause just does a single entity lookup — always fast.
            if jv == &right_clause.bind {
                return JoinStrategy::NestedLoop;
            }

            // Case 2: Join var matches an indexed/unique field on the right side.
            // NestedLoop does index probes per left tuple instead of scanning
            // the entire right side. Better when left < right.
            if let Some(type_def) = schema.get(&right_clause.entity_type) {
                for (field_name, pattern) in &right_clause.field_patterns {
                    if let Some(v) = pattern.bound_var() {
                        if v == jv {
                            if let Some(fd) = type_def.get_field(field_name) {
                                if fd.unique || fd.indexed {
                                    return JoinStrategy::NestedLoop;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Otherwise → HashJoin, build on smaller side
    let build_side = if left_est <= right_est {
        JoinSide::Left
    } else {
        JoinSide::Right
    };

    JoinStrategy::HashJoin { build_side }
}

/// Build a left-deep join tree, greedily picking the next clause that is
/// connected (shares join variables) with the current tree. Among connected
/// candidates, prefer the one with the lowest cardinality estimate.
/// This avoids catastrophic cross products when a smaller but unconnected
/// clause would otherwise be joined first.
fn build_join_tree(
    clauses: &[(WhereClause, ScanStrategy, CostEstimate)],
    schema: &SchemaRegistry,
) -> PlanNode {
    assert!(!clauses.is_empty());

    // Start with the lowest-cardinality clause (already sorted)
    let (clause, strategy, estimate) = &clauses[0];
    let mut tree = PlanNode::Scan(ClauseScan {
        clause: clause.clone(),
        strategy: strategy.clone(),
        estimate: estimate.clone(),
    });

    let mut remaining: Vec<usize> = (1..clauses.len()).collect();

    while !remaining.is_empty() {
        // Find the best next clause: prefer connected (has join vars with tree),
        // break ties by cardinality estimate.
        let mut best_idx = None;
        let mut best_connected = false;
        let mut best_est = usize::MAX;

        for (pos, &clause_idx) in remaining.iter().enumerate() {
            let (clause, _, estimate) = &clauses[clause_idx];
            let join_vars = find_join_vars_with_tree(&tree, clause);
            let connected = !join_vars.is_empty();
            let est = estimate.estimated_rows;

            // Prefer connected over disconnected; among same connectivity, prefer smaller
            if connected && !best_connected {
                best_idx = Some(pos);
                best_connected = true;
                best_est = est;
            } else if connected == best_connected && est < best_est {
                best_idx = Some(pos);
                best_est = est;
            }
        }

        let pos = best_idx.unwrap();
        let clause_idx = remaining.remove(pos);
        let (clause, strategy, estimate) = &clauses[clause_idx];

        let right = PlanNode::Scan(ClauseScan {
            clause: clause.clone(),
            strategy: strategy.clone(),
            estimate: estimate.clone(),
        });

        let join_vars = find_join_vars_with_tree(&tree, clause);

        let left_est = tree.estimate().estimated_rows;
        let right_est = estimate.estimated_rows;

        let join_strategy =
            choose_join_strategy(left_est, right_est, &join_vars, clause, schema);

        let join_est = if join_vars.is_empty() {
            // Cross join: product
            left_est * right_est
        } else {
            // Estimated join selectivity
            left_est.min(right_est)
        };

        tree = PlanNode::Join {
            left: Box::new(tree),
            right: Box::new(right),
            join_vars,
            strategy: join_strategy,
            estimate: CostEstimate { estimated_rows: join_est },
        };
    }

    tree
}
