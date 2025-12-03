use crate::value::Value;

/// Clojure Abstract Syntax Tree
///
/// This represents Clojure code after parsing but before compilation.
/// Much simpler than Beagle's AST since Clojure has uniform S-expression syntax.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // Literals
    Literal(Value),

    // Variable reference
    Var {
        namespace: Option<String>,  // None = unqualified, Some("user") = qualified
        name: String,
    },

    // Namespace special forms
    Ns {
        name: String,
    },

    Use {
        namespace: String,
    },

    // Special forms
    Def {
        name: String,
        value: Box<Expr>,
        metadata: Option<im::HashMap<String, Value>>,
    },

    /// (set! var value)
    /// Modifies a thread-local binding (only works within binding context)
    Set {
        var: Box<Expr>,
        value: Box<Expr>,
    },

    If {
        test: Box<Expr>,
        then: Box<Expr>,
        else_: Option<Box<Expr>>,
    },

    Do {
        exprs: Vec<Expr>,
    },

    /// (binding [var1 val1 var2 val2 ...] body)
    /// Establishes thread-local bindings for dynamic vars
    Binding {
        bindings: Vec<(String, Box<Expr>)>,  // [(var-name, value-expr), ...]
        body: Vec<Expr>,  // Multiple expressions in body (like do)
    },

    /// (let [x 10 y 20] (+ x y))
    /// Establishes lexical (stack-allocated) local bindings
    /// Bindings are sequential - each can see prior bindings
    Let {
        bindings: Vec<(String, Box<Expr>)>,  // [(name, value-expr), ...]
        body: Vec<Expr>,  // Body expressions (returns last)
    },

    // Function call (for now, all calls are the same)
    // Later we'll distinguish between special forms at parse time
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
    },

    // Quote - return value unevaluated
    Quote(Value),
}

/// Convert parsed Value to AST
///
/// This is the analyzer phase - we recognize special forms and
/// build an AST suitable for compilation.
pub fn analyze(value: &Value) -> Result<Expr, String> {
    match value {
        // Literals pass through
        Value::Nil | Value::Bool(_) | Value::Int(_) | Value::Float(_) |
        Value::String(_) | Value::Keyword(_) => {
            Ok(Expr::Literal(value.clone()))
        }

        // Vectors and other data structures are literals for now
        // (until we implement vector/map construction)
        Value::Vector(_) | Value::Map(_) | Value::Set(_) => {
            Ok(Expr::Literal(value.clone()))
        }

        // WithMeta wraps a value with metadata - analyze the inner value
        Value::WithMeta(_, inner) => {
            analyze(inner)
        }

        // Symbols become variable references
        // Parse qualified symbols: "user/foo" → Var { namespace: Some("user"), name: "foo" }
        // Special case: "/" is the division operator, not a qualified symbol
        Value::Symbol(s) => {
            if s == "/" {
                // Division operator - treat as unqualified symbol
                Ok(Expr::Var {
                    namespace: None,
                    name: s.clone(),
                })
            } else if let Some(idx) = s.find('/') {
                let namespace = s[..idx].to_string();
                let name = s[idx+1..].to_string();
                Ok(Expr::Var {
                    namespace: Some(namespace),
                    name,
                })
            } else {
                Ok(Expr::Var {
                    namespace: None,
                    name: s.clone(),
                })
            }
        }

        // Lists are either special forms or function calls
        Value::List(items) if !items.is_empty() => {
            // Check if first element is a special form
            if let Some(Value::Symbol(name)) = items.get(0) {
                match name.as_str() {
                    "def" => analyze_def(items),
                    "set!" => analyze_set(items),
                    "if" => analyze_if(items),
                    "do" => analyze_do(items),
                    "let" => analyze_let(items),
                    "quote" => analyze_quote(items),
                    "ns" => analyze_ns(items),
                    "use" => analyze_use(items),
                    "binding" => analyze_binding(items),
                    _ => analyze_call(items),
                }
            } else {
                analyze_call(items)
            }
        }

        Value::List(_) => {
            Err("Cannot evaluate empty list".to_string())
        }

        Value::Function { .. } => {
            Err("Functions not yet implemented".to_string())
        }

        Value::Namespace { .. } => {
            Err("Namespace values not yet implemented".to_string())
        }
    }
}

fn analyze_def(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() != 3 {
        return Err(format!("def requires 2 arguments, got {}", items.len() - 1));
    }

    // Extract name and metadata from the symbol (which might have metadata attached)
    let (name, metadata) = match &items[1] {
        Value::WithMeta(meta, inner) => {
            match **inner {
                Value::Symbol(ref s) => (s.clone(), Some(meta.clone())),
                _ => return Err("def requires a symbol".to_string()),
            }
        }
        Value::Symbol(s) => (s.clone(), None),
        _ => return Err("def requires a symbol as first argument".to_string()),
    };

    let value = analyze(&items[2])?;

    Ok(Expr::Def {
        name,
        value: Box::new(value),
        metadata,
    })
}

fn analyze_set(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() != 3 {
        return Err(format!("set! requires 2 arguments, got {}", items.len() - 1));
    }

    let var = analyze(&items[1])?;
    let value = analyze(&items[2])?;

    Ok(Expr::Set {
        var: Box::new(var),
        value: Box::new(value),
    })
}

fn analyze_if(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() < 3 || items.len() > 4 {
        return Err(format!("if requires 2 or 3 arguments, got {}", items.len() - 1));
    }

    let test = analyze(&items[1])?;
    let then = analyze(&items[2])?;
    let else_ = if items.len() == 4 {
        Some(Box::new(analyze(&items[3])?))
    } else {
        None
    };

    Ok(Expr::If {
        test: Box::new(test),
        then: Box::new(then),
        else_,
    })
}

fn analyze_do(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() < 2 {
        return Err("do requires at least 1 argument".to_string());
    }

    let mut exprs = Vec::new();
    for i in 1..items.len() {
        exprs.push(analyze(&items[i])?);
    }

    Ok(Expr::Do { exprs })
}

fn analyze_quote(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() != 2 {
        return Err(format!("quote requires 1 argument, got {}", items.len() - 1));
    }

    Ok(Expr::Quote(items[1].clone()))
}

fn analyze_ns(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() != 2 {
        return Err(format!("ns requires 1 argument, got {}", items.len() - 1));
    }

    let ns_name = match &items[1] {
        Value::Symbol(s) => s.clone(),
        _ => return Err("ns requires a symbol as namespace name".to_string()),
    };

    Ok(Expr::Ns { name: ns_name })
}

fn analyze_use(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() != 2 {
        return Err(format!("use requires 1 argument, got {}", items.len() - 1));
    }

    let ns_name = match &items[1] {
        // Accept both symbols and quoted symbols: (use clojure.core) or (use 'clojure.core)
        Value::Symbol(s) => s.clone(),
        Value::List(quoted) if quoted.len() == 2 => {
            if let (Some(Value::Symbol(q)), Some(Value::Symbol(ns))) = (quoted.get(0), quoted.get(1)) {
                if q == "quote" {
                    ns.clone()
                } else {
                    return Err("use requires a symbol or quoted symbol".to_string());
                }
            } else {
                return Err("use requires a symbol or quoted symbol".to_string());
            }
        }
        _ => return Err("use requires a symbol or quoted symbol".to_string()),
    };

    Ok(Expr::Use { namespace: ns_name })
}

fn analyze_binding(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (binding [var1 val1 var2 val2 ...] body...)
    if items.len() < 3 {
        return Err("binding requires at least 2 arguments: bindings vector and body".to_string());
    }

    // Parse bindings vector
    let bindings_vec = match &items[1] {
        Value::Vector(v) => v,
        _ => return Err("binding requires a vector of bindings as first argument".to_string()),
    };

    // Bindings must be even (pairs of var/value)
    if bindings_vec.len() % 2 != 0 {
        return Err("binding vector must contain an even number of forms (var/value pairs)".to_string());
    }

    // Parse each binding pair
    let mut bindings = Vec::new();
    for i in (0..bindings_vec.len()).step_by(2) {
        let var_name = match &bindings_vec[i] {
            Value::Symbol(s) => {
                // Handle both qualified and unqualified symbols
                // For now, we'll store the full symbol name
                s.clone()
            }
            _ => return Err(format!("binding requires symbols as variable names, got {:?}", bindings_vec[i])),
        };

        let value_expr = analyze(&bindings_vec[i + 1])?;
        bindings.push((var_name, Box::new(value_expr)));
    }

    // Parse body expressions (like do - all expressions evaluated, last one returned)
    let mut body = Vec::new();
    for i in 2..items.len() {
        body.push(analyze(&items[i])?);
    }

    Ok(Expr::Binding { bindings, body })
}

fn analyze_let(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (let [x 10 y 20] (+ x y))
    //      ^^^^^^^^^  ^^^^^^^^^^
    //      bindings   body
    // Body is optional: (let [x 10]) returns nil

    if items.len() < 2 {
        return Err("let requires at least 1 argument: bindings vector".to_string());
    }

    // Parse bindings vector
    let bindings_vec = match &items[1] {
        Value::Vector(v) => v,
        _ => return Err("let requires a vector of bindings as first argument".to_string()),
    };

    // Bindings must be even (pairs of name/value)
    if bindings_vec.len() % 2 != 0 {
        return Err("let bindings vector must contain an even number of forms (name/value pairs)".to_string());
    }

    // Parse each binding pair
    let mut bindings = Vec::new();
    for i in (0..bindings_vec.len()).step_by(2) {
        let name = match &bindings_vec[i] {
            Value::Symbol(s) => s.clone(),
            _ => return Err(format!("let binding names must be symbols, got {:?}", bindings_vec[i])),
        };

        let value_expr = analyze(&bindings_vec[i + 1])?;
        bindings.push((name, Box::new(value_expr)));
    }

    // Parse body expressions (all evaluated, last one returned)
    // If empty body, return nil
    let mut body = Vec::new();
    for i in 2..items.len() {
        body.push(analyze(&items[i])?);
    }

    if body.is_empty() {
        // Empty body returns nil
        body.push(Expr::Literal(Value::Nil));
    }

    Ok(Expr::Let { bindings, body })
}

fn analyze_call(items: &im::Vector<Value>) -> Result<Expr, String> {
    let func = analyze(&items[0])?;
    let mut args = Vec::new();

    for i in 1..items.len() {
        args.push(analyze(&items[i])?);
    }

    Ok(Expr::Call {
        func: Box::new(func),
        args,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::read;

    #[test]
    fn test_analyze_literals() {
        let val = read("42").unwrap();
        let expr = analyze(&val).unwrap();
        assert!(matches!(expr, Expr::Literal(Value::Int(42))));
    }

    #[test]
    fn test_analyze_var() {
        let val = read("x").unwrap();
        let expr = analyze(&val).unwrap();
        assert!(matches!(expr, Expr::Var { namespace: None, ref name } if name == "x"));
    }

    #[test]
    fn test_analyze_def() {
        let val = read("(def x 42)").unwrap();
        let expr = analyze(&val).unwrap();
        match expr {
            Expr::Def { name, value, .. } => {
                assert_eq!(name, "x");
                assert!(matches!(*value, Expr::Literal(Value::Int(42))));
            }
            _ => panic!("Expected Def"),
        }
    }

    #[test]
    fn test_analyze_if() {
        let val = read("(if true 1 2)").unwrap();
        let expr = analyze(&val).unwrap();
        match expr {
            Expr::If { test, then, else_ } => {
                assert!(matches!(*test, Expr::Literal(Value::Bool(true))));
                assert!(matches!(*then, Expr::Literal(Value::Int(1))));
                assert!(matches!(else_, Some(_)));
            }
            _ => panic!("Expected If"),
        }
    }

    #[test]
    fn test_analyze_call() {
        let val = read("(+ 1 2)").unwrap();
        let expr = analyze(&val).unwrap();
        match expr {
            Expr::Call { func, args } => {
                assert!(matches!(*func, Expr::Var { namespace: None, ref name } if name == "+"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Call"),
        }
    }
}
