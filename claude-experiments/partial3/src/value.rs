use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use crate::ast::Expr;

pub type EnvInner = HashMap<String, Value>;
pub type Env = Rc<RefCell<EnvInner>>;

/// Object inner type - mutable hashmap of properties
pub type ObjectInner = HashMap<String, Value>;
pub type ObjectRef = Rc<RefCell<ObjectInner>>;

/// Array inner type - mutable vector
pub type ArrayInner = Vec<Value>;
pub type ArrayRef = Rc<RefCell<ArrayInner>>;

pub fn new_env() -> Env {
    Rc::new(RefCell::new(HashMap::new()))
}

pub fn env_with_parent(parent: &Env) -> Env {
    Rc::new(RefCell::new(parent.borrow().clone()))
}

pub fn new_object() -> ObjectRef {
    Rc::new(RefCell::new(HashMap::new()))
}

pub fn new_array() -> ArrayRef {
    Rc::new(RefCell::new(Vec::new()))
}

pub fn array_from_vec(v: Vec<Value>) -> ArrayRef {
    Rc::new(RefCell::new(v))
}

pub fn object_with_props(props: Vec<(String, Value)>) -> ObjectRef {
    let obj = new_object();
    for (k, v) in props {
        obj.borrow_mut().insert(k, v);
    }
    obj
}

#[derive(Clone, Debug)]
pub enum Value {
    Int(i64),
    Bool(bool),
    String(String),
    Undefined,
    Null,
    Array(ArrayRef),
    Object(ObjectRef),
    Closure {
        params: Vec<String>,
        body: Expr,
        env: Env,
    },
    // Opaque values that can't be inspected - used for `new` expressions etc.
    Opaque {
        label: String,
        expr: Expr, // The original expression for residualization
        state: Option<Rc<dyn Any>>, // Type-erased pluggable state for semantic modeling
    },
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Undefined => write!(f, "undefined"),
            Value::Null => write!(f, "null"),
            Value::Array(elements) => {
                write!(f, "[")?;
                for (i, elem) in elements.borrow().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")
            }
            Value::Object(obj) => {
                write!(f, "{{")?;
                let borrowed = obj.borrow();
                for (i, (k, v)) in borrowed.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Closure { params, .. } => {
                write!(f, "<closure ({})>", params.join(", "))
            }
            Value::Opaque { label, .. } => {
                write!(f, "<opaque: {}>", label)
            }
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Undefined, Value::Undefined) => true,
            (Value::Null, Value::Null) => true,
            (Value::Array(a), Value::Array(b)) => a == b,
            // Objects compare by reference
            (Value::Object(a), Value::Object(b)) => Rc::ptr_eq(a, b),
            // Opaque values compare by label and state reference
            (Value::Opaque { label: l1, state: s1, .. }, Value::Opaque { label: l2, state: s2, .. }) => {
                l1 == l2 && match (s1, s2) {
                    (Some(a), Some(b)) => Rc::ptr_eq(a, b),
                    (None, None) => true,
                    _ => false,
                }
            }
            // Closures and other opaque values are not comparable
            _ => false,
        }
    }
}

/// Convert a Value back to an Expr (for residualization)
pub fn value_to_expr(v: &Value) -> Expr {
    match v {
        Value::Int(n) => Expr::Int(*n),
        Value::Bool(b) => Expr::Bool(*b),
        Value::String(s) => Expr::String(s.clone()),
        Value::Undefined => Expr::Undefined,
        Value::Null => Expr::Null,
        Value::Array(elements) => Expr::Array(elements.borrow().iter().map(value_to_expr).collect()),
        Value::Object(obj) => {
            let borrowed = obj.borrow();
            let props: Vec<(String, Expr)> = borrowed
                .iter()
                .map(|(k, v)| (k.clone(), value_to_expr(v)))
                .collect();
            Expr::Object(props)
        }
        Value::Closure { params, body, .. } => {
            // When residualizing a closure, we just emit the fn form
            // The captured environment's static values are already inlined in the body
            Expr::Fn(params.clone(), Box::new(body.clone()))
        }
        Value::Opaque { expr, .. } => expr.clone(),
    }
}
