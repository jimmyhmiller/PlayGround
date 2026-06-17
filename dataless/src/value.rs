//! Runtime values.
//!
//! A `Member` is the paper's "pointer to a member of a collection" (a "bug"):
//! a stable handle that dereferences a single datum in O(1) regardless of the
//! collection's representation. Positional lookup of "the k-th member" is what
//! costs differently between an ARRAY and a LIST — not the handle itself.

#[derive(Clone, Debug)]
pub enum Value {
    Int(i64),
    Text(String),
    Bool(bool),
    Nil,
    /// A handle to a member of a collection: (collection id, stable node id).
    Member(usize, usize),
}

impl Value {
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Text(s) => !s.is_empty(),
            Value::Nil => false,
            Value::Member(_, _) => true,
        }
    }

    pub fn as_int(&self) -> Result<i64, String> {
        match self {
            Value::Int(i) => Ok(*i),
            Value::Bool(b) => Ok(*b as i64),
            _ => Err(format!("expected an integer, found {}", self.type_name())),
        }
    }

    pub fn as_member(&self) -> Result<(usize, usize), String> {
        match self {
            Value::Member(c, id) => Ok((*c, *id)),
            _ => Err(format!(
                "expected a collection member handle, found {}",
                self.type_name()
            )),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "int",
            Value::Text(_) => "text",
            Value::Bool(_) => "bool",
            Value::Nil => "nil",
            Value::Member(_, _) => "member",
        }
    }

    pub fn display(&self) -> String {
        match self {
            Value::Int(i) => i.to_string(),
            Value::Text(s) => s.clone(),
            Value::Bool(b) => b.to_string(),
            Value::Nil => "nil".to_string(),
            Value::Member(c, id) => format!("<member c{}#{}>", c, id),
        }
    }
}

/// Structural equality for `==` / `!=` and search conditions.
pub fn values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Text(x), Value::Text(y)) => x == y,
        (Value::Nil, Value::Nil) => true,
        (Value::Member(c1, i1), Value::Member(c2, i2)) => c1 == c2 && i1 == i2,
        // Allow int/bool cross-compare so conditions stay convenient.
        (Value::Int(x), Value::Bool(y)) | (Value::Bool(y), Value::Int(x)) => *x == *y as i64,
        _ => false,
    }
}

/// Ordering for `<`, `<=`, `>`, `>=`.
pub fn values_cmp(a: &Value, b: &Value) -> Result<std::cmp::Ordering, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(x.cmp(y)),
        (Value::Text(x), Value::Text(y)) => Ok(x.cmp(y)),
        (Value::Bool(x), Value::Bool(y)) => Ok(x.cmp(y)),
        _ => Err(format!(
            "cannot compare {} and {}",
            a.type_name(),
            b.type_name()
        )),
    }
}
