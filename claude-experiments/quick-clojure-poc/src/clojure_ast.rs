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

    /// (loop [x 0 sum 0] body...)
    /// Establishes bindings (like let) and a recursion point for recur
    Loop {
        bindings: Vec<(String, Box<Expr>)>,
        body: Vec<Expr>,
    },

    /// (recur expr1 expr2 ...)
    /// Jumps back to nearest loop/fn with new values for bindings
    Recur {
        args: Vec<Expr>,
    },

    /// (fn name? [params*] exprs*)
    /// (fn name? ([params*] exprs*)+)
    /// Function definition - first-class values
    /// Supports multi-arity dispatch and closures
    Fn {
        name: Option<String>,                      // Optional self-recursion name
        arities: Vec<crate::value::FnArity>,       // One or more arity overloads
    },

    // Function call (for now, all calls are the same)
    // Later we'll distinguish between special forms at parse time
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
    },

    // Quote - return value unevaluated
    Quote(Value),

    /// (var symbol) or #'symbol
    /// Returns the Var object itself, not its value
    VarRef {
        namespace: Option<String>,
        name: String,
    },

    /// (deftype* TypeName [field1 field2 ...])
    /// Defines a new type with named fields
    DefType {
        name: String,
        fields: Vec<String>,
    },

    /// (TypeName. arg1 arg2 ...) - constructor call
    /// Creates an instance of a deftype
    TypeConstruct {
        type_name: String,
        args: Vec<Expr>,
    },

    /// (.-field obj) - field access
    /// Reads a field from a deftype instance
    FieldAccess {
        field: String,
        object: Box<Expr>,
    },
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
                    "loop" => analyze_loop(items),
                    "recur" => analyze_recur(items),
                    "fn" => analyze_fn(items),
                    "quote" => analyze_quote(items),
                    "ns" => analyze_ns(items),
                    "use" => analyze_use(items),
                    "binding" => analyze_binding(items),
                    "var" => analyze_var_ref(items),
                    "deftype*" | "deftype" => analyze_deftype(items),
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

fn analyze_var_ref(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (var symbol) - returns the Var object itself
    if items.len() != 2 {
        return Err(format!("var requires 1 argument, got {}", items.len() - 1));
    }

    // Parse the symbol (may be qualified: ns/name or unqualified: name)
    match &items[1] {
        Value::Symbol(s) => {
            if s == "/" {
                Ok(Expr::VarRef {
                    namespace: None,
                    name: s.clone(),
                })
            } else if let Some(idx) = s.find('/') {
                let namespace = s[..idx].to_string();
                let name = s[idx+1..].to_string();
                Ok(Expr::VarRef {
                    namespace: Some(namespace),
                    name,
                })
            } else {
                Ok(Expr::VarRef {
                    namespace: None,
                    name: s.clone(),
                })
            }
        }
        _ => Err("var requires a symbol".to_string()),
    }
}

fn analyze_deftype(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (deftype* TypeName [field1 field2 ...])
    if items.len() != 3 {
        return Err(format!("deftype* requires 2 arguments (name and fields), got {}", items.len() - 1));
    }

    // Get type name
    let name = match &items[1] {
        Value::Symbol(s) => s.clone(),
        _ => return Err("deftype* requires a symbol as type name".to_string()),
    };

    // Get field names from vector
    let fields = match &items[2] {
        Value::Vector(v) => {
            let mut field_names = Vec::new();
            for field in v.iter() {
                match field {
                    Value::Symbol(s) => field_names.push(s.clone()),
                    _ => return Err(format!("deftype* field must be a symbol, got {:?}", field)),
                }
            }
            field_names
        }
        _ => return Err("deftype* requires a vector of field names".to_string()),
    };

    Ok(Expr::DefType { name, fields })
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

fn analyze_loop(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (loop [x 0 sum 0] body...)
    if items.len() < 2 {
        return Err("loop requires at least 1 argument: bindings vector".to_string());
    }

    let bindings_vec = match &items[1] {
        Value::Vector(v) => v,
        _ => return Err("loop requires a vector of bindings as first argument".to_string()),
    };

    if bindings_vec.len() % 2 != 0 {
        return Err("loop bindings must contain an even number of forms".to_string());
    }

    let mut bindings = Vec::new();
    for i in (0..bindings_vec.len()).step_by(2) {
        let name = match &bindings_vec[i] {
            Value::Symbol(s) => s.clone(),
            _ => return Err(format!("loop binding names must be symbols, got {:?}", bindings_vec[i])),
        };
        let value_expr = analyze(&bindings_vec[i + 1])?;
        bindings.push((name, Box::new(value_expr)));
    }

    let mut body = Vec::new();
    for i in 2..items.len() {
        body.push(analyze(&items[i])?);
    }

    if body.is_empty() {
        body.push(Expr::Literal(Value::Nil));
    }

    Ok(Expr::Loop { bindings, body })
}

fn analyze_recur(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (recur expr1 expr2 ...)
    let mut args = Vec::new();
    for i in 1..items.len() {
        args.push(analyze(&items[i])?);
    }
    Ok(Expr::Recur { args })
}

fn analyze_fn(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (fn name? [params] body*)
    // (fn name? ([params] body*)+)
    if items.len() < 2 {
        return Err("fn requires at least a parameter vector".to_string());
    }

    let mut idx = 1;

    // Check for optional name
    let name = if let Some(Value::Symbol(s)) = items.get(idx) {
        // Could be name or could be param vector
        // Name is a symbol, params is a vector
        // Peek ahead to disambiguate
        if idx + 1 < items.len() {
            match items.get(idx + 1) {
                Some(Value::Vector(_)) => {
                    // This is a name, next is params
                    idx += 1;
                    Some(s.clone())
                }
                Some(Value::List(_)) => {
                    // Multi-arity form starting with (params) list, no name
                    None
                }
                _ => {
                    // Ambiguous - assume no name if current is a vector
                    if matches!(items.get(idx), Some(Value::Vector(_))) {
                        None
                    } else {
                        return Err("fn requires parameter vector or arity forms".to_string());
                    }
                }
            }
        } else {
            // Only one element after fn - must be params vector
            None
        }
    } else {
        None
    };

    // Parse arities
    let mut arities = Vec::new();

    // Check if single-arity or multi-arity form
    let is_multi_arity = matches!(items.get(idx), Some(Value::List(_)));

    if is_multi_arity {
        // Multi-arity: (fn name? ([params] body)+ )
        for i in idx..items.len() {
            let arity_form = match &items[i] {
                Value::List(arity_items) => arity_items,
                _ => return Err("Multi-arity fn requires lists for each arity".to_string()),
            };

            let arity = parse_fn_arity(arity_form)?;
            arities.push(arity);
        }
    } else {
        // Single-arity: (fn name? [params] body*)
        let params_vec = match &items[idx] {
            Value::Vector(v) => v,
            _ => return Err("fn requires parameter vector".to_string()),
        };

        // Collect body expressions (everything after params)
        let body_start = idx + 1;
        let mut body_values = Vec::new();
        for i in body_start..items.len() {
            body_values.push(items[i].clone());
        }

        let arity = parse_fn_params_and_body(params_vec, &body_values)?;
        arities.push(arity);
    }

    // Validate arities
    validate_fn_arities(&arities)?;

    Ok(Expr::Fn { name, arities })
}

fn parse_fn_arity(arity_items: &im::Vector<Value>) -> Result<crate::value::FnArity, String> {
    // ([params] condition-map? body*)
    if arity_items.is_empty() {
        return Err("fn arity form cannot be empty".to_string());
    }

    let params_vec = match &arity_items[0] {
        Value::Vector(v) => v,
        _ => return Err("fn arity form must start with parameter vector".to_string()),
    };

    let mut body_values = Vec::new();
    for i in 1..arity_items.len() {
        body_values.push(arity_items[i].clone());
    }
    parse_fn_params_and_body(params_vec, &body_values)
}

fn parse_fn_params_and_body(
    params_vec: &im::Vector<Value>,
    body_items: &[Value],
) -> Result<crate::value::FnArity, String> {
    // Parse parameters: [x y] or [x y & rest]
    let mut params = Vec::new();
    let mut rest_param = None;
    let mut found_ampersand = false;

    for (i, param) in params_vec.iter().enumerate() {
        match param {
            Value::Symbol(s) if s == "&" => {
                if found_ampersand {
                    return Err("Only one & allowed in parameter list".to_string());
                }
                if i == params_vec.len() - 1 {
                    return Err("& must be followed by rest parameter".to_string());
                }
                found_ampersand = true;
            }
            Value::Symbol(s) => {
                if found_ampersand {
                    if rest_param.is_some() {
                        return Err("Only one rest parameter allowed after &".to_string());
                    }
                    rest_param = Some(s.clone());
                } else {
                    params.push(s.clone());
                }
            }
            _ => return Err(format!("fn parameters must be symbols, got {:?}", param)),
        }
    }

    // Parse condition map and body
    let (pre_conditions, post_conditions, body_start_idx) =
        if !body_items.is_empty() {
            if let Some(Value::Map(cond_map)) = body_items.first() {
                // Check if this is truly a condition map or just the body
                // Condition map has :pre and/or :post keys
                let has_pre = cond_map.contains_key(&Value::Keyword("pre".to_string()));
                let has_post = cond_map.contains_key(&Value::Keyword("post".to_string()));

                if has_pre || has_post {
                    // Parse condition map
                    let pre = if let Some(Value::Vector(pre_vec)) = cond_map.get(&Value::Keyword("pre".to_string())) {
                        pre_vec.iter()
                            .map(|v| analyze(v))
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        Vec::new()
                    };

                    let post = if let Some(Value::Vector(post_vec)) = cond_map.get(&Value::Keyword("post".to_string())) {
                        post_vec.iter()
                            .map(|v| analyze(v))
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        Vec::new()
                    };

                    (pre, post, 1)
                } else {
                    // Just a map in the body, not a condition map
                    (Vec::new(), Vec::new(), 0)
                }
            } else {
                (Vec::new(), Vec::new(), 0)
            }
        } else {
            (Vec::new(), Vec::new(), 0)
        };

    // Parse body expressions
    let mut body = Vec::new();
    for i in body_start_idx..body_items.len() {
        body.push(analyze(&body_items[i])?);
    }

    if body.is_empty() {
        // Empty body returns nil
        body.push(Expr::Literal(Value::Nil));
    }

    Ok(crate::value::FnArity {
        params,
        rest_param,
        body,
        pre_conditions,
        post_conditions,
    })
}

fn validate_fn_arities(arities: &[crate::value::FnArity]) -> Result<(), String> {
    if arities.is_empty() {
        return Err("fn requires at least one arity".to_string());
    }

    let mut seen_arities = std::collections::HashSet::new();
    let mut variadic_count = 0;

    for arity in arities {
        let arity_num = arity.params.len();

        if arity.rest_param.is_some() {
            variadic_count += 1;
            if variadic_count > 1 {
                return Err("fn can have at most one variadic arity".to_string());
            }
        }

        if !seen_arities.insert(arity_num) {
            return Err(format!("Duplicate arity {} in fn", arity_num));
        }
    }

    Ok(())
}

fn analyze_call(items: &im::Vector<Value>) -> Result<Expr, String> {
    // Check for special call patterns based on the first symbol
    if let Some(Value::Symbol(sym)) = items.get(0) {
        // Check for constructor call: (TypeName. arg1 arg2 ...)
        if sym.ends_with('.') && sym.len() > 1 && !sym.starts_with('.') {
            let type_name = sym[..sym.len()-1].to_string();
            let mut args = Vec::new();
            for i in 1..items.len() {
                args.push(analyze(&items[i])?);
            }
            return Ok(Expr::TypeConstruct { type_name, args });
        }

        // Check for field access: (.-field obj)
        if sym.starts_with(".-") && sym.len() > 2 {
            if items.len() != 2 {
                return Err(format!("Field access {} requires exactly 1 argument", sym));
            }
            let field = sym[2..].to_string();
            let object = analyze(&items[1])?;
            return Ok(Expr::FieldAccess {
                field,
                object: Box::new(object),
            });
        }
    }

    // Regular function call
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
