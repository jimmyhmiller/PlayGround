//! Runtime values for the interpreter
//!
//! Values are the results of evaluation. They include:
//! - Primitives (bool, int, string)
//! - Closures (captured environment + body)
//! - Objects (maps of field names to values)

use crate::expr::Expr;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

/// A runtime value
#[derive(Clone)]
pub enum Value {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// String value
    String(String),
    /// Closure: captures environment, parameter name, and body
    Closure {
        param: String,
        body: Expr,
        env: Env,
    },
    /// Object: a map of field names to values
    Object(HashMap<String, Value>),
    /// A reference cell - used for recursive bindings
    Ref(Rc<RefCell<Value>>),
    /// Placeholder for uninitialized recursive bindings
    Uninitialized,
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{:?}", s),
            Value::Closure { param, .. } => write!(f, "<closure: {} => ...>", param),
            Value::Object(fields) => {
                write!(f, "{{ ")?;
                let mut first = true;
                for (name, val) in fields {
                    if !first {
                        write!(f, ", ")?;
                    }
                    first = false;
                    // Avoid infinite recursion for self-referential objects
                    match val {
                        Value::Object(_) => write!(f, "{}: {{...}}", name)?,
                        Value::Ref(_) => write!(f, "{}: <ref>", name)?,
                        _ => write!(f, "{}: {:?}", name, val)?,
                    }
                }
                write!(f, " }}")
            }
            Value::Ref(r) => {
                let inner = r.borrow();
                write!(f, "{:?}", *inner)
            }
            Value::Uninitialized => write!(f, "<uninitialized>"),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{}", s),
            Value::Closure { param, .. } => write!(f, "<function: {} => ...>", param),
            Value::Object(fields) => {
                write!(f, "{{ ")?;
                let mut first = true;
                for (name, val) in fields {
                    if !first {
                        write!(f, ", ")?;
                    }
                    first = false;
                    match val {
                        Value::Object(_) => write!(f, "{}: {{...}}", name)?,
                        Value::Ref(_) => write!(f, "{}: <self>", name)?,
                        _ => write!(f, "{}: {}", name, val)?,
                    }
                }
                write!(f, " }}")
            }
            Value::Ref(r) => {
                let inner = r.borrow();
                write!(f, "{}", *inner)
            }
            Value::Uninitialized => write!(f, "<uninitialized>"),
        }
    }
}

/// Environment: maps variable names to values
/// Uses Rc for cheap cloning when extending
#[derive(Clone, Default)]
pub struct Env {
    bindings: Rc<EnvInner>,
}

#[derive(Default)]
struct EnvInner {
    parent: Option<Rc<EnvInner>>,
    bindings: HashMap<String, Value>,
}

impl Env {
    /// Create an empty environment
    pub fn new() -> Self {
        Env {
            bindings: Rc::new(EnvInner::default()),
        }
    }

    /// Look up a variable in the environment
    pub fn get(&self, name: &str) -> Option<Value> {
        let mut current = Some(&self.bindings);
        while let Some(env) = current {
            if let Some(val) = env.bindings.get(name) {
                return Some(val.clone());
            }
            current = env.parent.as_ref();
        }
        None
    }

    /// Extend the environment with a new binding
    pub fn extend(&self, name: String, value: Value) -> Self {
        let mut bindings = HashMap::new();
        bindings.insert(name, value);
        Env {
            bindings: Rc::new(EnvInner {
                parent: Some(self.bindings.clone()),
                bindings,
            }),
        }
    }

    /// Extend the environment with multiple bindings
    pub fn extend_many(&self, new_bindings: HashMap<String, Value>) -> Self {
        Env {
            bindings: Rc::new(EnvInner {
                parent: Some(self.bindings.clone()),
                bindings: new_bindings,
            }),
        }
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<env>")
    }
}
