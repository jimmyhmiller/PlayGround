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
    Var(String),

    // Special forms
    Def {
        name: String,
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

        // Symbols become variable references
        Value::Symbol(name) => {
            Ok(Expr::Var(name.clone()))
        }

        // Lists are either special forms or function calls
        Value::List(items) if !items.is_empty() => {
            // Check if first element is a special form
            if let Some(Value::Symbol(name)) = items.get(0) {
                match name.as_str() {
                    "def" => analyze_def(items),
                    "if" => analyze_if(items),
                    "do" => analyze_do(items),
                    "quote" => analyze_quote(items),
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
    }
}

fn analyze_def(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() != 3 {
        return Err(format!("def requires 2 arguments, got {}", items.len() - 1));
    }

    let name = match &items[1] {
        Value::Symbol(s) => s.clone(),
        _ => return Err("def requires a symbol as first argument".to_string()),
    };

    let value = analyze(&items[2])?;

    Ok(Expr::Def {
        name,
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
        assert!(matches!(expr, Expr::Var(ref s) if s == "x"));
    }

    #[test]
    fn test_analyze_def() {
        let val = read("(def x 42)").unwrap();
        let expr = analyze(&val).unwrap();
        match expr {
            Expr::Def { name, value } => {
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
                assert!(matches!(*func, Expr::Var(ref s) if s == "+"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Call"),
        }
    }
}
