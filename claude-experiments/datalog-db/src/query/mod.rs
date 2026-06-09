pub mod executor;
pub mod planner;

use serde::{Deserialize, Serialize};

use crate::datom::{TxId, Value};

/// A pattern that can appear in a where clause field position.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Pattern {
    /// A variable binding like "?name"
    Variable(String),
    /// A variable binding that also filters: {"var": "?age", "gt": 25}.
    /// Binds `var` to the field value AND keeps only rows where the
    /// predicate holds. Ordered before `Predicate` so serde's untagged
    /// deserializer prefers it when a `var` key is present.
    BoundPredicate { var: String, op: PredOp, value: Value },
    /// A predicate like {"gt": 21}
    Predicate { op: PredOp, value: Value },
    /// A concrete constant value
    Constant(Value),
    /// Match on an enum field: {"match": "Circle", "radius": "?r"}
    EnumMatch {
        /// Pattern for the variant tag (Variable or Constant string)
        variant: Box<Pattern>,
        /// Patterns for the variant's fields
        field_patterns: Vec<(String, Pattern)>,
    },
    /// Nearest-neighbour search on a `vector(N)` field:
    /// `{"near": [..query..], "k": 20, "score": "?sim", "metric": "cosine"}`.
    /// Produces up to `k` entities ranked by similarity to `query`, binding
    /// `score` (when given) to the similarity. This is a whole-type top-k
    /// operation, not a per-row filter.
    Near {
        query: Vec<f32>,
        k: usize,
        metric: VectorMetric,
        /// Optional variable to bind the similarity score into.
        score_var: Option<String>,
    },
    /// BM25 full-text search on a `fulltext` string field:
    /// `{"search": "recursive functions", "k": 20, "score": "?bm25"}`.
    /// Tokenizes the query, walks the inverted index, ranks by BM25, returns
    /// up to `k` entities binding `score` (when given) to the BM25 score.
    Search {
        query: String,
        k: usize,
        score_var: Option<String>,
    },
}

/// Distance / similarity metric for vector search. `Cosine` and `Dot` are
/// similarities (higher = closer); `L2` is a distance (lower = closer) but is
/// negated into a score so "higher = closer" holds uniformly for ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VectorMetric {
    Cosine,
    Dot,
    L2,
}

impl VectorMetric {
    pub fn from_name(s: &str) -> Option<VectorMetric> {
        match s {
            "cosine" => Some(VectorMetric::Cosine),
            "dot" | "dotproduct" | "inner" => Some(VectorMetric::Dot),
            "l2" | "euclidean" => Some(VectorMetric::L2),
            _ => None,
        }
    }

    /// Score `candidate` against `query` so that a HIGHER score is always more
    /// similar (L2 distance is negated). Mismatched dimensions score -inf.
    pub fn score(&self, query: &[f32], candidate: &[f32]) -> f32 {
        if query.len() != candidate.len() {
            return f32::NEG_INFINITY;
        }
        match self {
            VectorMetric::Dot => dot(query, candidate),
            VectorMetric::Cosine => {
                let d = dot(query, candidate);
                let nq = dot(query, query).sqrt();
                let nc = dot(candidate, candidate).sqrt();
                if nq == 0.0 || nc == 0.0 {
                    0.0
                } else {
                    d / (nq * nc)
                }
            }
            VectorMetric::L2 => {
                let mut s = 0.0f32;
                for (a, b) in query.iter().zip(candidate.iter()) {
                    let d = a - b;
                    s += d * d;
                }
                -s.sqrt()
            }
        }
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

impl Pattern {
    pub fn from_json(v: &serde_json::Value) -> std::result::Result<Self, String> {
        if let Some(s) = v.as_str() {
            if s.starts_with('?') {
                return Ok(Pattern::Variable(s.to_string()));
            }
            return Ok(Pattern::Constant(Value::String(s.into())));
        }
        if let Some(n) = v.as_i64() {
            return Ok(Pattern::Constant(Value::I64(n)));
        }
        if let Some(n) = v.as_f64() {
            return Ok(Pattern::Constant(Value::F64(n)));
        }
        if let Some(b) = v.as_bool() {
            return Ok(Pattern::Constant(Value::Bool(b)));
        }
        if let Some(obj) = v.as_object() {
            // An entity reference ({"ref": N}) is a constant, not a predicate.
            if obj.len() == 1 && obj.contains_key("ref") {
                return Ok(Pattern::Constant(value_from_json(v)));
            }

            // Nearest-neighbour search: {"near": [..], "k": N, "score": "?s"}.
            if let Some(near_val) = obj.get("near") {
                let arr = near_val
                    .as_array()
                    .ok_or("'near' must be an array of numbers (the query vector)")?;
                let mut query = Vec::with_capacity(arr.len());
                for x in arr {
                    query.push(
                        x.as_f64().ok_or("'near' vector elements must be numbers")? as f32,
                    );
                }
                if query.is_empty() {
                    return Err("'near' query vector must not be empty".into());
                }
                let k = obj
                    .get("k")
                    .and_then(|k| k.as_u64())
                    .map(|k| k as usize)
                    .unwrap_or(10);
                if k == 0 {
                    return Err("'k' must be at least 1".into());
                }
                let metric = match obj.get("metric").and_then(|m| m.as_str()) {
                    None => VectorMetric::Cosine,
                    Some(s) => VectorMetric::from_name(s)
                        .ok_or_else(|| format!("unknown vector metric '{}'", s))?,
                };
                let score_var = obj
                    .get("score")
                    .and_then(|s| s.as_str())
                    .map(|s| s.to_string());
                if let Some(ref sv) = score_var {
                    if !sv.starts_with('?') {
                        return Err("'score' must be a variable like \"?sim\"".into());
                    }
                }
                return Ok(Pattern::Near {
                    query,
                    k,
                    metric,
                    score_var,
                });
            }

            // BM25 full-text search: {"search": "terms", "k": N, "score": "?s"}.
            if let Some(search_val) = obj.get("search") {
                let query = search_val
                    .as_str()
                    .ok_or("'search' must be a string (the query text)")?
                    .to_string();
                let k = obj
                    .get("k")
                    .and_then(|k| k.as_u64())
                    .map(|k| k as usize)
                    .unwrap_or(10);
                if k == 0 {
                    return Err("'k' must be at least 1".into());
                }
                let score_var = obj
                    .get("score")
                    .and_then(|s| s.as_str())
                    .map(|s| s.to_string());
                if let Some(ref sv) = score_var {
                    if !sv.starts_with('?') {
                        return Err("'score' must be a variable like \"?bm25\"".into());
                    }
                }
                return Ok(Pattern::Search { query, k, score_var });
            }

            // Check for enum match: {"match": "Circle", ...}
            if let Some(match_val) = obj.get("match") {
                let variant = Box::new(Pattern::from_json(match_val)?);
                let mut field_patterns = Vec::new();
                for (k, val) in obj {
                    if k == "match" {
                        continue;
                    }
                    field_patterns.push((k.clone(), Pattern::from_json(val)?));
                }
                return Ok(Pattern::EnumMatch {
                    variant,
                    field_patterns,
                });
            }

            // Combined bind + predicate: {"var": "?age", "gt": 25}.
            // Binds the variable to the field value and filters by the
            // predicate in the same clause field.
            if let Some(var) = obj.get("var").and_then(|x| x.as_str()) {
                if var.starts_with('?') {
                    for (key, val) in obj {
                        if let Some(op) = PredOp::from_key(key) {
                            return Ok(Pattern::BoundPredicate {
                                var: var.to_string(),
                                op,
                                value: value_from_json(val),
                            });
                        }
                    }
                    // A {"var": ...} object with no recognized operator is a
                    // bare bind — equivalent to "?var".
                    if obj.len() == 1 {
                        return Ok(Pattern::Variable(var.to_string()));
                    }
                    return Err(format!(
                        "field pattern {{\"var\": {:?}, ...}} has no known predicate operator; \
                         expected one of {:?}",
                        var,
                        PredOp::known_keys()
                    ));
                }
            }

            // Check for predicate operators
            for (key, val) in obj {
                if let Some(op) = PredOp::from_key(key) {
                    let value = value_from_json(val);
                    return Ok(Pattern::Predicate { op, value });
                }
            }

            // An object that matched none of the above is almost always a
            // typo'd operator (e.g. {"contians": "x"}). Silently treating it
            // as a constant turns the clause into a no-op filter that matches
            // every row — a footgun. Reject it loudly instead.
            let keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
            return Err(format!(
                "unknown field pattern with keys {:?}; expected a constant, a variable \"?x\", \
                 a predicate {{op: value}} (one of {:?}), an enum {{\"match\": ...}}, \
                 or an entity ref {{\"ref\": N}}",
                keys,
                PredOp::known_keys()
            ));
        }
        // null and any other JSON shape: route through value_from_json.
        Ok(Pattern::Constant(value_from_json(v)))
    }

    /// The variable this pattern binds into the result tuple, if any.
    /// Both bare variables and bind+predicate patterns bind a variable.
    pub fn bound_var(&self) -> Option<&str> {
        match self {
            Pattern::Variable(v) => Some(v.as_str()),
            Pattern::BoundPredicate { var, .. } => Some(var.as_str()),
            _ => None,
        }
    }
}

fn value_from_json(v: &serde_json::Value) -> Value {
    if let Some(s) = v.as_str() {
        Value::String(s.into())
    } else if let Some(n) = v.as_i64() {
        Value::I64(n)
    } else if let Some(n) = v.as_f64() {
        Value::F64(n)
    } else if let Some(b) = v.as_bool() {
        Value::Bool(b)
    } else if let Some(arr) = v.as_array() {
        Value::List(arr.iter().map(value_from_json).collect())
    } else if let Some(r) = v.get("ref").and_then(|x| x.as_u64()) {
        // Entity reference: {"ref": N}. Without this a ref constant in a
        // where clause (e.g. `lead: #11`) would stringify and never match a
        // stored `Ref`.
        Value::Ref(r)
    } else {
        Value::String(format!("{}", v).into())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredOp {
    Gt,
    Lt,
    Gte,
    Lte,
    Ne,
    /// Substring match on string values (case-sensitive).
    Contains,
    /// Prefix match on string values.
    StartsWith,
    /// Suffix match on string values.
    EndsWith,
}

impl PredOp {
    /// Map a JSON predicate key ("gt", "lte", ...) to its operator.
    pub fn from_key(key: &str) -> Option<PredOp> {
        match key {
            "gt" => Some(PredOp::Gt),
            "lt" => Some(PredOp::Lt),
            "gte" => Some(PredOp::Gte),
            "lte" => Some(PredOp::Lte),
            "ne" => Some(PredOp::Ne),
            "contains" => Some(PredOp::Contains),
            "starts_with" => Some(PredOp::StartsWith),
            "ends_with" => Some(PredOp::EndsWith),
            _ => None,
        }
    }

    /// True for the string-search operators (substring/prefix/suffix). These
    /// can't drive a range index scan — they filter via a sequential walk.
    pub fn is_string_search(&self) -> bool {
        matches!(self, PredOp::Contains | PredOp::StartsWith | PredOp::EndsWith)
    }

    /// The set of JSON predicate keys, for error messages.
    pub fn known_keys() -> &'static [&'static str] {
        &["gt", "lt", "gte", "lte", "ne", "contains", "starts_with", "ends_with"]
    }

    pub fn evaluate(&self, actual: &Value, target: &Value) -> bool {
        match (actual, target) {
            (Value::I64(a), Value::I64(b)) => match self {
                PredOp::Gt => a > b,
                PredOp::Lt => a < b,
                PredOp::Gte => a >= b,
                PredOp::Lte => a <= b,
                PredOp::Ne => a != b,
                // String-search ops never apply to numbers.
                PredOp::Contains | PredOp::StartsWith | PredOp::EndsWith => false,
            },
            (Value::F64(a), Value::F64(b)) => match self {
                PredOp::Gt => a > b,
                PredOp::Lt => a < b,
                PredOp::Gte => a >= b,
                PredOp::Lte => a <= b,
                PredOp::Ne => a != b,
                PredOp::Contains | PredOp::StartsWith | PredOp::EndsWith => false,
            },
            // Mixed numeric: compare as f64 so an integer literal and a
            // float field (or vice versa) don't silently fail to match.
            // Coercion normally aligns types before this, but this keeps
            // any remaining cross-type comparison (e.g. enum sub-fields)
            // numerically correct.
            (Value::I64(a), Value::F64(b)) => {
                let a = *a as f64;
                match self {
                    PredOp::Gt => a > *b,
                    PredOp::Lt => a < *b,
                    PredOp::Gte => a >= *b,
                    PredOp::Lte => a <= *b,
                    PredOp::Ne => a != *b,
                    PredOp::Contains | PredOp::StartsWith | PredOp::EndsWith => false,
                }
            }
            (Value::F64(a), Value::I64(b)) => {
                let b = *b as f64;
                match self {
                    PredOp::Gt => *a > b,
                    PredOp::Lt => *a < b,
                    PredOp::Gte => *a >= b,
                    PredOp::Lte => *a <= b,
                    PredOp::Ne => *a != b,
                    PredOp::Contains | PredOp::StartsWith | PredOp::EndsWith => false,
                }
            }
            (Value::String(a), Value::String(b)) => match self {
                PredOp::Gt => a > b,
                PredOp::Lt => a < b,
                PredOp::Gte => a >= b,
                PredOp::Lte => a <= b,
                PredOp::Ne => a != b,
                PredOp::Contains => a.contains(b.as_ref()),
                PredOp::StartsWith => a.starts_with(b.as_ref()),
                PredOp::EndsWith => a.ends_with(b.as_ref()),
            },
            // List membership: `subjects: contains "logic"` is true when the
            // list field holds an element equal to the target. `ne` means "the
            // list does not equal this exact list".
            (Value::List(items), _) => match self {
                PredOp::Contains => items.iter().any(|it| it == target),
                PredOp::Ne => actual != target,
                _ => false,
            },
            _ => match self {
                PredOp::Ne => actual != target,
                // String-search ops only apply to strings/lists; any other
                // value (or a mismatched target) never matches.
                _ => false,
            },
        }
    }
}

/// A where clause binds an entity variable to a typed entity with field patterns.
#[derive(Debug, Clone)]
pub struct WhereClause {
    /// Entity variable, e.g. "?u"
    pub bind: String,
    /// Entity type, e.g. "User"
    pub entity_type: String,
    /// Field patterns: field_name -> pattern
    pub field_patterns: Vec<(String, Pattern)>,
}

/// A node in the where clause tree. The top-level `where` is an `And` of
/// these. `And`/`Or`/`Not` may nest arbitrarily.
#[derive(Debug, Clone)]
pub enum Clause {
    /// Leaf: match a typed entity (the classic where clause).
    Pattern(WhereClause),
    /// Conjunction — all children must hold.
    And(Vec<Clause>),
    /// Disjunction — the union of bindings satisfying any child.
    Or(Vec<Clause>),
    /// Negation as failure — keep bindings for which the child has no solution.
    Not(Box<Clause>),
}

impl Clause {
    /// Parse a clause from JSON. A clause is one of:
    ///   {"bind": "?v", "type": "T", <field patterns>}   -> Pattern
    ///   {"and": [clause, ...]}                            -> And
    ///   {"or":  [clause, ...]}                            -> Or
    ///   {"not": clause | [clause, ...]}                   -> Not (of an And)
    /// An array is shorthand for an `And` of its elements.
    pub fn from_json(v: &serde_json::Value) -> std::result::Result<Clause, String> {
        if let Some(arr) = v.as_array() {
            return Ok(Clause::And(
                arr.iter()
                    .map(Clause::from_json)
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            ));
        }
        let obj = v.as_object().ok_or("clause must be an object or array")?;
        if let Some(children) = obj.get("or") {
            let arr = children.as_array().ok_or("'or' must be an array of clauses")?;
            if arr.len() < 2 {
                return Err("'or' requires at least two branches".into());
            }
            return Ok(Clause::Or(
                arr.iter()
                    .map(Clause::from_json)
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            ));
        }
        if let Some(children) = obj.get("and") {
            let arr = children.as_array().ok_or("'and' must be an array of clauses")?;
            return Ok(Clause::And(
                arr.iter()
                    .map(Clause::from_json)
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            ));
        }
        if let Some(child) = obj.get("not") {
            return Ok(Clause::Not(Box::new(Clause::from_json(child)?)));
        }
        // Otherwise it's a pattern leaf.
        let bind = obj
            .get("bind")
            .and_then(|b| b.as_str())
            .ok_or("clause missing 'bind'")?
            .to_string();
        let entity_type = obj
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or("clause missing 'type'")?
            .to_string();
        let mut field_patterns = Vec::new();
        for (key, val) in obj {
            if key == "bind" || key == "type" {
                continue;
            }
            field_patterns.push((key.clone(), Pattern::from_json(val)?));
        }
        Ok(Clause::Pattern(WhereClause {
            bind,
            entity_type,
            field_patterns,
        }))
    }

    /// True when this is a plain conjunction of pattern leaves (the common
    /// case), in which case the optimized flat planner applies.
    pub fn as_flat_patterns(&self) -> Option<Vec<&WhereClause>> {
        match self {
            Clause::Pattern(wc) => Some(vec![wc]),
            Clause::And(children) => {
                let mut out = Vec::with_capacity(children.len());
                for c in children {
                    if let Clause::Pattern(wc) = c {
                        out.push(wc);
                    } else {
                        return None;
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }

    /// Visit every `Pattern` leaf immutably.
    pub fn for_each_pattern<F: FnMut(&WhereClause)>(&self, f: &mut F) {
        match self {
            Clause::Pattern(wc) => f(wc),
            Clause::And(cs) | Clause::Or(cs) => {
                for c in cs {
                    c.for_each_pattern(f);
                }
            }
            Clause::Not(c) => c.for_each_pattern(f),
        }
    }

    /// Visit every `Pattern` leaf mutably (used to normalize literals).
    pub fn for_each_pattern_mut<F: FnMut(&mut WhereClause)>(&mut self, f: &mut F) {
        match self {
            Clause::Pattern(wc) => f(wc),
            Clause::And(cs) | Clause::Or(cs) => {
                for c in cs {
                    c.for_each_pattern_mut(f);
                }
            }
            Clause::Not(c) => c.for_each_pattern_mut(f),
        }
    }

    /// True if the tree has no pattern leaves at all.
    pub fn is_empty(&self) -> bool {
        let mut any = false;
        self.for_each_pattern(&mut |_| any = true);
        !any
    }

    /// Collapse double negation: `not { not { c } }` ⇒ `c`. Without this,
    /// the inner `not` reaches the planner as a non-producing clause and is
    /// rejected, even though the doubly-negated form is a valid positive.
    pub fn simplified(self) -> Clause {
        match self {
            Clause::Pattern(_) => self,
            Clause::And(cs) => Clause::And(cs.into_iter().map(Clause::simplified).collect()),
            Clause::Or(cs) => Clause::Or(cs.into_iter().map(Clause::simplified).collect()),
            Clause::Not(inner) => match inner.simplified() {
                // not(not(c)) == c
                Clause::Not(c) => *c,
                other => Clause::Not(Box::new(other)),
            },
        }
    }
}

/// One sort key for `order_by`: a find variable and a direction.
#[derive(Debug, Clone)]
pub struct OrderKey {
    pub var: String,
    pub desc: bool,
}

/// Aggregate function in a `find` clause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

impl AggFunc {
    pub fn from_name(name: &str) -> Option<AggFunc> {
        match name {
            "count" => Some(AggFunc::Count),
            "sum" => Some(AggFunc::Sum),
            "avg" => Some(AggFunc::Avg),
            "min" => Some(AggFunc::Min),
            "max" => Some(AggFunc::Max),
            _ => None,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            AggFunc::Count => "count",
            AggFunc::Sum => "sum",
            AggFunc::Avg => "avg",
            AggFunc::Min => "min",
            AggFunc::Max => "max",
        }
    }
}

/// One element of a `find` clause: either a plain variable (which also acts
/// as a grouping key when aggregates are present) or an aggregate over a
/// variable. `label` is the output column name.
#[derive(Debug, Clone)]
pub enum FindElem {
    Var(String),
    Agg {
        func: AggFunc,
        /// Input variable, or "*" for `count(*)`.
        var: String,
        label: String,
    },
}

impl FindElem {
    /// Parse a single find token, e.g. "?name", "count(?u)", "count(*)".
    pub fn parse(token: &str) -> std::result::Result<FindElem, String> {
        let t = token.trim();
        // Aggregate form: func(arg)
        if let Some(open) = t.find('(') {
            if !t.ends_with(')') {
                return Err(format!("malformed aggregate in find: '{}'", token));
            }
            let func_name = t[..open].trim();
            let func = AggFunc::from_name(func_name)
                .ok_or_else(|| format!("unknown aggregate function '{}'", func_name))?;
            let arg = t[open + 1..t.len() - 1].trim();
            if arg != "*" && !arg.starts_with('?') {
                return Err(format!(
                    "aggregate argument must be a variable or '*', got '{}'",
                    arg
                ));
            }
            if func == AggFunc::Count && arg == "*" {
                return Ok(FindElem::Agg {
                    func,
                    var: "*".to_string(),
                    label: "count(*)".to_string(),
                });
            }
            if arg == "*" {
                return Err(format!("'{}' requires a variable argument", func_name));
            }
            return Ok(FindElem::Agg {
                func,
                var: arg.to_string(),
                label: format!("{}({})", func_name, arg),
            });
        }
        if t.starts_with('?') {
            return Ok(FindElem::Var(t.to_string()));
        }
        Err(format!("find element must be a variable or aggregate, got '{}'", token))
    }

    /// The output column label.
    pub fn label(&self) -> &str {
        match self {
            FindElem::Var(v) => v,
            FindElem::Agg { label, .. } => label,
        }
    }
}

/// The top-level query.
#[derive(Debug, Clone)]
pub struct Query {
    /// Effective projection: the variables the executor must bind/output.
    /// Derived from `find_elems` (group vars + aggregate input vars, deduped).
    pub find: Vec<String>,
    /// Output specification: plain vars and/or aggregates, in output order.
    pub find_elems: Vec<FindElem>,
    /// The where clause tree (top level is a conjunction).
    pub where_clause: Clause,
    pub as_of: Option<TxId>,
    pub as_of_time: Option<u64>,
    /// If true, return the query plan instead of executing.
    pub explain: bool,
    /// Sort keys applied to the result rows (each must be a find variable).
    pub order_by: Vec<OrderKey>,
    /// Maximum number of rows to return, applied after ordering.
    pub limit: Option<usize>,
    /// Number of rows to skip, applied after ordering and before `limit`.
    pub offset: Option<usize>,
}

/// Total ordering over `Value` for sorting result rows. Numeric values
/// compare numerically (i64/f64 mixed); other types compare within their
/// kind, and different kinds fall back to a stable kind ranking so the sort
/// is always total. `Null` sorts after every concrete value.
pub fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    // Numeric cross-type comparison first.
    let num = |v: &Value| -> Option<f64> {
        match v {
            Value::I64(n) => Some(*n as f64),
            Value::F64(f) => Some(*f),
            _ => None,
        }
    };
    if let (Some(x), Some(y)) = (num(a), num(b)) {
        // Keep i64 exact when both are integers.
        if let (Value::I64(ai), Value::I64(bi)) = (a, b) {
            return ai.cmp(bi);
        }
        return x.total_cmp(&y);
    }
    match (a, b) {
        (Value::String(x), Value::String(y)) => x.cmp(y),
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        (Value::Ref(x), Value::Ref(y)) => x.cmp(y),
        (Value::Bytes(x), Value::Bytes(y)) => x.cmp(y),
        // Lists compare lexicographically by element.
        (Value::List(x), Value::List(y)) => {
            for (xi, yi) in x.iter().zip(y.iter()) {
                let c = compare_values(xi, yi);
                if c != std::cmp::Ordering::Equal {
                    return c;
                }
            }
            x.len().cmp(&y.len())
        }
        // Mixed / unorderable kinds: order by a stable kind rank.
        _ => value_kind_rank(a).cmp(&value_kind_rank(b)),
    }
}

fn value_kind_rank(v: &Value) -> u8 {
    match v {
        Value::I64(_) | Value::F64(_) => 0,
        Value::String(_) => 1,
        Value::Bool(_) => 2,
        Value::Ref(_) => 3,
        Value::Bytes(_) => 4,
        Value::Enum(_) => 5,
        Value::List(_) => 6,
        Value::Vector(_) => 7,
        Value::Null => 8,
    }
}

impl Query {
    /// Parse a query from JSON.
    pub fn from_json(v: &serde_json::Value) -> std::result::Result<Self, String> {
        // `find` entries are either strings ("?x" / "count(?u)") or objects
        // ({"agg":"sum","var":"?p"}). Parse into output specs, then derive the
        // effective projection (the underlying variables the executor binds).
        let find_arr = v
            .get("find")
            .and_then(|f| f.as_array())
            .ok_or("missing 'find' array")?;
        let mut find_elems: Vec<FindElem> = Vec::with_capacity(find_arr.len());
        for item in find_arr {
            if let Some(s) = item.as_str() {
                find_elems.push(FindElem::parse(s)?);
            } else if let Some(obj) = item.as_object() {
                let func = obj
                    .get("agg")
                    .and_then(|x| x.as_str())
                    .ok_or("find object must have an 'agg' field")?;
                let var = obj
                    .get("var")
                    .and_then(|x| x.as_str())
                    .unwrap_or("*");
                find_elems.push(FindElem::parse(&format!("{}({})", func, var))?);
            } else {
                return Err("find element must be a string or object".into());
            }
        }
        // Effective projection: group vars + aggregate input vars, deduped,
        // preserving first-seen order. `count(*)` contributes no variable.
        let mut find: Vec<String> = Vec::new();
        for e in &find_elems {
            let var = match e {
                FindElem::Var(v) => Some(v.as_str()),
                FindElem::Agg { var, .. } if var != "*" => Some(var.as_str()),
                _ => None,
            };
            if let Some(v) = var {
                if !find.iter().any(|x| x == v) {
                    find.push(v.to_string());
                }
            }
        }

        // `where` is an array of clauses (implicit AND). Each clause may be a
        // pattern leaf or an or/and/not combinator (see Clause::from_json).
        let where_val = v.get("where").ok_or("missing 'where' array")?;
        let where_clause = Clause::from_json(where_val)?;

        let as_of = v.get("as_of").and_then(|a| a.as_u64());
        let as_of_time = v.get("as_of_time").and_then(|a| a.as_u64());
        let explain = v.get("explain").and_then(|e| e.as_bool()).unwrap_or(false);

        // order_by: array of either "?var" (ascending) or
        // {"var": "?var", "desc": true}.
        let order_by = match v.get("order_by") {
            None => Vec::new(),
            Some(arr) => {
                let arr = arr.as_array().ok_or("'order_by' must be an array")?;
                let mut keys = Vec::with_capacity(arr.len());
                for item in arr {
                    let key = if let Some(s) = item.as_str() {
                        OrderKey { var: s.to_string(), desc: false }
                    } else if let Some(obj) = item.as_object() {
                        let var = obj
                            .get("var")
                            .and_then(|x| x.as_str())
                            .ok_or("order_by entry missing 'var'")?
                            .to_string();
                        let desc = obj.get("desc").and_then(|d| d.as_bool()).unwrap_or(false);
                        OrderKey { var, desc }
                    } else {
                        return Err("order_by entry must be a string or object".into());
                    };
                    keys.push(key);
                }
                keys
            }
        };

        let limit = v
            .get("limit")
            .map(|l| {
                l.as_u64()
                    .map(|n| n as usize)
                    .ok_or("'limit' must be a non-negative integer")
            })
            .transpose()?;
        let offset = v
            .get("offset")
            .map(|o| {
                o.as_u64()
                    .map(|n| n as usize)
                    .ok_or("'offset' must be a non-negative integer")
            })
            .transpose()?;

        Ok(Query {
            find,
            find_elems,
            where_clause,
            as_of,
            as_of_time,
            explain,
            order_by,
            limit,
            offset,
        })
    }
}
