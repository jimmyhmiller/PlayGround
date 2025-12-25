use std::collections::HashMap;
use std::fmt;

use crate::namespace::Namespace;

/// Symbol with namespace tracking
#[derive(Debug, Clone, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub namespace: Option<Namespace>,
    /// True if this symbol uses slash notation (alias/name)
    pub uses_alias: bool,
    /// True if this symbol uses dot notation (namespace.name)
    pub uses_dot: bool,
}

impl Symbol {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            namespace: None,
            uses_alias: false,
            uses_dot: false,
        }
    }

    pub fn with_namespace(name: impl Into<String>, namespace: Namespace) -> Self {
        Self {
            name: name.into(),
            namespace: Some(namespace),
            uses_alias: false,
            uses_dot: false,
        }
    }

    /// Get the full qualified name (namespace.name or alias/name)
    pub fn qualified_name(&self) -> String {
        if let Some(ref ns) = self.namespace {
            if self.uses_alias {
                if let Some(ref alias) = ns.alias {
                    return format!("{}/{}", alias, self.name);
                }
            }
            if self.uses_dot {
                return format!("{}.{}", ns.name, self.name);
            }
            // Default to dot notation
            return format!("{}.{}", ns.name, self.name);
        }
        self.name.clone()
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Don't print the "user" namespace - it's the default/implicit namespace
        if let Some(ref ns) = self.namespace {
            if ns.name == "user" {
                return write!(f, "{}", self.name);
            }
        }
        write!(f, "{}", self.qualified_name())
    }
}

/// Main value type for the reader
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    List(Vec<Value>),
    Vector(Vec<Value>),
    Map(HashMap<String, Value>),
    Symbol(Symbol),
    String(String),
    Number(f64),
    Keyword(String),
    Boolean(bool),
    Nil,
}

impl Value {
    pub fn list() -> Self {
        Value::List(Vec::new())
    }

    pub fn vector() -> Self {
        Value::Vector(Vec::new())
    }

    pub fn map() -> Self {
        Value::Map(HashMap::new())
    }

    pub fn symbol(name: impl Into<String>) -> Self {
        Value::Symbol(Symbol::new(name))
    }

    pub fn symbol_with_namespace(name: impl Into<String>, namespace: Namespace) -> Self {
        Value::Symbol(Symbol::with_namespace(name, namespace))
    }

    pub fn string(s: impl Into<String>) -> Self {
        Value::String(s.into())
    }

    pub fn number(n: f64) -> Self {
        Value::Number(n)
    }

    pub fn keyword(k: impl Into<String>) -> Self {
        Value::Keyword(k.into())
    }

    pub fn boolean(b: bool) -> Self {
        Value::Boolean(b)
    }

    pub fn nil() -> Self {
        Value::Nil
    }

    /// Check if this is a list
    pub fn is_list(&self) -> bool {
        matches!(self, Value::List(_))
    }

    /// Check if this is a vector
    pub fn is_vector(&self) -> bool {
        matches!(self, Value::Vector(_))
    }

    /// Check if this is a map
    pub fn is_map(&self) -> bool {
        matches!(self, Value::Map(_))
    }

    /// Check if this is a symbol
    pub fn is_symbol(&self) -> bool {
        matches!(self, Value::Symbol(_))
    }

    /// Get as list (panics if not a list)
    pub fn as_list(&self) -> &[Value] {
        match self {
            Value::List(l) => l,
            _ => panic!("expected list"),
        }
    }

    /// Get as list mut (panics if not a list)
    pub fn as_list_mut(&mut self) -> &mut Vec<Value> {
        match self {
            Value::List(l) => l,
            _ => panic!("expected list"),
        }
    }

    /// Get as vector (panics if not a vector)
    pub fn as_vector(&self) -> &[Value] {
        match self {
            Value::Vector(v) => v,
            _ => panic!("expected vector"),
        }
    }

    /// Get as vector mut (panics if not a vector)
    pub fn as_vector_mut(&mut self) -> &mut Vec<Value> {
        match self {
            Value::Vector(v) => v,
            _ => panic!("expected vector"),
        }
    }

    /// Get as map (panics if not a map)
    pub fn as_map(&self) -> &HashMap<String, Value> {
        match self {
            Value::Map(m) => m,
            _ => panic!("expected map"),
        }
    }

    /// Get as map mut (panics if not a map)
    pub fn as_map_mut(&mut self) -> &mut HashMap<String, Value> {
        match self {
            Value::Map(m) => m,
            _ => panic!("expected map"),
        }
    }

    /// Get as symbol (panics if not a symbol)
    pub fn as_symbol(&self) -> &Symbol {
        match self {
            Value::Symbol(s) => s,
            _ => panic!("expected symbol"),
        }
    }

    /// Get as symbol mut (panics if not a symbol)
    pub fn as_symbol_mut(&mut self) -> &mut Symbol {
        match self {
            Value::Symbol(s) => s,
            _ => panic!("expected symbol"),
        }
    }

    /// Get as string (panics if not a string)
    pub fn as_string(&self) -> &str {
        match self {
            Value::String(s) => s,
            _ => panic!("expected string"),
        }
    }

    /// Get as number (panics if not a number)
    pub fn as_number(&self) -> f64 {
        match self {
            Value::Number(n) => *n,
            _ => panic!("expected number"),
        }
    }

    /// Get as keyword (panics if not a keyword)
    pub fn as_keyword(&self) -> &str {
        match self {
            Value::Keyword(k) => k,
            _ => panic!("expected keyword"),
        }
    }

    /// Get as boolean (panics if not a boolean)
    pub fn as_boolean(&self) -> bool {
        match self {
            Value::Boolean(b) => *b,
            _ => panic!("expected boolean"),
        }
    }

    /// Append item to list
    pub fn list_append(&mut self, item: Value) {
        match self {
            Value::List(l) => l.push(item),
            _ => panic!("expected list"),
        }
    }

    /// Append item to vector
    pub fn vector_append(&mut self, item: Value) {
        match self {
            Value::Vector(v) => v.push(item),
            _ => panic!("expected vector"),
        }
    }

    /// Put key-value into map
    pub fn map_put(&mut self, key: impl Into<String>, value: Value) {
        match self {
            Value::Map(m) => {
                m.insert(key.into(), value);
            }
            _ => panic!("expected map"),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::List(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, ")")
            }
            Value::Vector(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::Map(items) => {
                write!(f, "{{")?;
                for (i, (k, v)) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, ":{} {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Symbol(s) => write!(f, "{}", s),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Number(n) => write!(f, "{}", n),
            Value::Keyword(k) => write!(f, "{}", k),
            Value::Boolean(b) => write!(f, "{}", b),
            Value::Nil => write!(f, "nil"),
        }
    }
}
