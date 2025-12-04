use im::{Vector, HashMap};
use std::fmt;
use std::collections::HashMap as StdHashMap;

/// Function arity definition
/// Represents one overload of a multi-arity function
#[derive(Debug, Clone, PartialEq)]
pub struct FnArity {
    pub params: Vec<String>,           // Positional parameters
    pub rest_param: Option<String>,    // Variadic rest parameter (after &)
    pub body: Vec<crate::clojure_ast::Expr>,  // Body expressions (implicit do)
    pub pre_conditions: Vec<crate::clojure_ast::Expr>,   // :pre assertions
    pub post_conditions: Vec<crate::clojure_ast::Expr>,  // :post assertions (can use %)
}

/// Clojure value representation
///
/// For Stage 0, we use im-rs for persistent data structures (temporary)
/// Later in Stage 3, we'll replace with compiled ClojureScript implementations
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    // Scalar types (can be embedded with tagging)
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),

    // Symbol and Keyword (heap allocated strings)
    Symbol(String),
    Keyword(String),

    // String (heap allocated)
    String(String),

    // Collections (using im-rs temporarily)
    List(Vector<Value>),      // Clojure lists - sequential
    Vector(Vector<Value>),     // Clojure vectors - indexed
    Map(HashMap<Value, Value>), // Clojure maps - key-value
    Set(im::HashSet<Value>),   // Clojure sets

    // Metadata wrapper
    WithMeta(HashMap<String, Value>, Box<Value>),

    // Functions - first-class values implementing IFn interface
    // Supports multi-arity dispatch (0-20 arities + optional variadic)
    Function {
        name: Option<String>,                      // Optional name for self-recursion
        arity_map: StdHashMap<usize, FnArity>,     // Fixed arities: arity_num → implementation
        variadic_arity: Option<Box<FnArity>>,      // Optional variadic overload (& rest-param)
    },

    // Namespace object (for future heap allocation)
    // For now, used as documentation of structure
    // In Phase 2, this will be heap-allocated with GC
    Namespace {
        name: String,
        mappings: std::collections::HashMap<String, isize>, // symbol → tagged value
        used_namespaces: Vec<String>,                        // namespace names that are used
    },
}

impl Value {}

// Implement Hash for Value (needed for HashMap keys)
impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::Nil => 0.hash(state),
            Value::Bool(b) => b.hash(state),
            Value::Int(i) => i.hash(state),
            Value::Float(f) => f.to_bits().hash(state), // Use bits for float hashing
            Value::Symbol(s) | Value::Keyword(s) | Value::String(s) => s.hash(state),
            Value::List(v) | Value::Vector(v) => {
                for item in v {
                    item.hash(state);
                }
            }
            Value::Map(_) => {
                // Maps are harder to hash consistently due to ordering
                // For now, just hash a marker
                "map".hash(state);
            }
            Value::Set(_) => {
                // Sets are harder to hash consistently
                "set".hash(state);
            }
            Value::WithMeta(_, inner) => {
                // Hash the inner value, metadata doesn't affect hash
                inner.hash(state);
            }
            Value::Function { name, arity_map, .. } => {
                name.hash(state);
                arity_map.len().hash(state);
            }
            Value::Namespace { name, .. } => name.hash(state),
        }
    }
}

impl Eq for Value {}

// Display implementation for REPL output
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::Symbol(s) => write!(f, "{}", s),
            Value::Keyword(k) => write!(f, ":{}", k),
            Value::String(s) => write!(f, "\"{}\"", s),
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
            Value::Map(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Set(set) => {
                write!(f, "#{{")?;
                for (i, item) in set.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "}}")
            }
            Value::WithMeta(_, inner) => {
                // Display the inner value (metadata is invisible in output)
                write!(f, "{}", inner)
            }
            Value::Function { name, .. } => {
                write!(f, "#<fn {}>", name.as_deref().unwrap_or("anonymous"))
            }
            Value::Namespace { name, mappings, .. } => {
                write!(f, "#<Namespace {} with {} vars>", name, mappings.len())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display() {
        assert_eq!(Value::Nil.to_string(), "nil");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Int(42).to_string(), "42");
        assert_eq!(Value::Keyword("foo".to_string()).to_string(), ":foo");
        assert_eq!(Value::Symbol("x".to_string()).to_string(), "x");
    }
}
