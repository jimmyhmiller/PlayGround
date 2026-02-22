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
}

impl Pattern {
    pub fn from_json(v: &serde_json::Value) -> Self {
        if let Some(s) = v.as_str() {
            if s.starts_with('?') {
                return Pattern::Variable(s.to_string());
            }
            return Pattern::Constant(Value::String(s.to_string()));
        }
        if let Some(n) = v.as_i64() {
            return Pattern::Constant(Value::I64(n));
        }
        if let Some(n) = v.as_f64() {
            return Pattern::Constant(Value::F64(n));
        }
        if let Some(b) = v.as_bool() {
            return Pattern::Constant(Value::Bool(b));
        }
        if let Some(obj) = v.as_object() {
            // Check for enum match: {"match": "Circle", ...}
            if let Some(match_val) = obj.get("match") {
                let variant = Box::new(Pattern::from_json(match_val));
                let field_patterns: Vec<_> = obj
                    .iter()
                    .filter(|(k, _)| k.as_str() != "match")
                    .map(|(k, v)| (k.clone(), Pattern::from_json(v)))
                    .collect();
                return Pattern::EnumMatch {
                    variant,
                    field_patterns,
                };
            }

            // Check for predicate operators
            for (key, val) in obj {
                let op = match key.as_str() {
                    "gt" => Some(PredOp::Gt),
                    "lt" => Some(PredOp::Lt),
                    "gte" => Some(PredOp::Gte),
                    "lte" => Some(PredOp::Lte),
                    "ne" => Some(PredOp::Ne),
                    _ => None,
                };
                if let Some(op) = op {
                    let value = value_from_json(val);
                    return Pattern::Predicate { op, value };
                }
            }
        }
        // Fallback
        Pattern::Constant(Value::String(format!("{}", v)))
    }
}

fn value_from_json(v: &serde_json::Value) -> Value {
    if let Some(s) = v.as_str() {
        Value::String(s.to_string())
    } else if let Some(n) = v.as_i64() {
        Value::I64(n)
    } else if let Some(n) = v.as_f64() {
        Value::F64(n)
    } else if let Some(b) = v.as_bool() {
        Value::Bool(b)
    } else {
        Value::String(format!("{}", v))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredOp {
    Gt,
    Lt,
    Gte,
    Lte,
    Ne,
}

impl PredOp {
    pub fn evaluate(&self, actual: &Value, target: &Value) -> bool {
        match (actual, target) {
            (Value::I64(a), Value::I64(b)) => match self {
                PredOp::Gt => a > b,
                PredOp::Lt => a < b,
                PredOp::Gte => a >= b,
                PredOp::Lte => a <= b,
                PredOp::Ne => a != b,
            },
            (Value::F64(a), Value::F64(b)) => match self {
                PredOp::Gt => a > b,
                PredOp::Lt => a < b,
                PredOp::Gte => a >= b,
                PredOp::Lte => a <= b,
                PredOp::Ne => a != b,
            },
            (Value::String(a), Value::String(b)) => match self {
                PredOp::Gt => a > b,
                PredOp::Lt => a < b,
                PredOp::Gte => a >= b,
                PredOp::Lte => a <= b,
                PredOp::Ne => a != b,
            },
            _ => match self {
                PredOp::Ne => actual != target,
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

/// The top-level query.
#[derive(Debug, Clone)]
pub struct Query {
    pub find: Vec<String>,
    pub where_clauses: Vec<WhereClause>,
    pub as_of: Option<TxId>,
    pub as_of_time: Option<u64>,
    /// If true, return the query plan instead of executing.
    pub explain: bool,
}

impl Query {
    /// Parse a query from JSON.
    pub fn from_json(v: &serde_json::Value) -> std::result::Result<Self, String> {
        let find = v
            .get("find")
            .and_then(|f| f.as_array())
            .ok_or("missing 'find' array")?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();

        let where_clauses = v
            .get("where")
            .and_then(|w| w.as_array())
            .ok_or("missing 'where' array")?
            .iter()
            .map(|clause| {
                let obj = clause.as_object().ok_or("where clause must be object")?;
                let bind = obj
                    .get("bind")
                    .and_then(|b| b.as_str())
                    .ok_or("missing 'bind'")?
                    .to_string();
                let entity_type = obj
                    .get("type")
                    .and_then(|t| t.as_str())
                    .ok_or("missing 'type'")?
                    .to_string();

                let mut field_patterns = Vec::new();
                for (key, val) in obj {
                    if key == "bind" || key == "type" {
                        continue;
                    }
                    field_patterns.push((key.clone(), Pattern::from_json(val)));
                }

                Ok(WhereClause {
                    bind,
                    entity_type,
                    field_patterns,
                })
            })
            .collect::<std::result::Result<Vec<_>, String>>()?;

        let as_of = v.get("as_of").and_then(|a| a.as_u64());
        let as_of_time = v.get("as_of_time").and_then(|a| a.as_u64());
        let explain = v.get("explain").and_then(|e| e.as_bool()).unwrap_or(false);

        Ok(Query {
            find,
            where_clauses,
            as_of,
            as_of_time,
            explain,
        })
    }
}
