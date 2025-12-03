use crate::clojure_ast::Expr;
use crate::value::Value;
use std::collections::HashMap;

/// Simple interpreter for testing
///
/// This evaluates the AST directly without compiling to IR/ARM64.
/// Good for testing Stage 1 logic before we wire up the full compiler.
pub struct Evaluator {
    /// Global variables (def'd values)
    globals: HashMap<String, Value>,
}

impl Evaluator {
    pub fn new() -> Self {
        Evaluator {
            globals: HashMap::new(),
        }
    }

    pub fn eval(&mut self, expr: &Expr) -> Result<Value, String> {
        match expr {
            Expr::Literal(value) => Ok(value.clone()),

            Expr::Quote(value) => Ok(value.clone()),

            Expr::Var { namespace, name } => {
                // For now, ignore namespaces in the simple evaluator
                // Just look up the bare name
                let _ = namespace;  // Suppress warning
                self.globals
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("Undefined variable: {}", name))
            }

            Expr::Ns { .. } => {
                // Namespace declaration - just return nil in simple evaluator
                Ok(Value::Nil)
            }

            Expr::Use { .. } => {
                // Use declaration - just return nil in simple evaluator
                Ok(Value::Nil)
            }

            Expr::Def { name, value, metadata: _ } => {
                // Simple evaluator ignores metadata
                let val = self.eval(value)?;
                self.globals.insert(name.clone(), val.clone());
                Ok(val)
            }

            Expr::Set { .. } => {
                Err("set! not supported in simple evaluator (use JIT compiler)".to_string())
            }

            Expr::If { test, then, else_ } => {
                let test_val = self.eval(test)?;
                if test_val.is_truthy() {
                    self.eval(then)
                } else if let Some(else_expr) = else_ {
                    self.eval(else_expr)
                } else {
                    Ok(Value::Nil)
                }
            }

            Expr::Do { exprs } => {
                let mut last = Value::Nil;
                for expr in exprs {
                    last = self.eval(expr)?;
                }
                Ok(last)
            }

            Expr::Binding { .. } => {
                // Dynamic bindings not supported in simple evaluator
                Err("binding form not supported in simple evaluator (use JIT compiler)".to_string())
            }

            Expr::Call { func, args } => {
                // Evaluate function position
                if let Expr::Var { name, .. } = &**func {
                    // Evaluate arguments
                    let arg_vals: Result<Vec<_>, _> =
                        args.iter().map(|arg| self.eval(arg)).collect();
                    let arg_vals = arg_vals?;

                    // Call builtin (ignore namespaces for now in simple evaluator)
                    match name.as_str() {
                        "+" => self.builtin_add(&arg_vals),
                        "-" => self.builtin_sub(&arg_vals),
                        "*" => self.builtin_mul(&arg_vals),
                        "/" => self.builtin_div(&arg_vals),
                        "<" => self.builtin_lt(&arg_vals),
                        ">" => self.builtin_gt(&arg_vals),
                        "=" => self.builtin_eq(&arg_vals),
                        _ => Err(format!("Unknown function: {}", name)),
                    }
                } else {
                    Err("Function position must be a symbol".to_string())
                }
            }
        }
    }

    fn builtin_add(&self, args: &[Value]) -> Result<Value, String> {
        if args.len() != 2 {
            return Err(format!("+ requires 2 arguments, got {}", args.len()));
        }

        match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + *b as f64)),
            _ => Err("+ requires numeric arguments".to_string()),
        }
    }

    fn builtin_sub(&self, args: &[Value]) -> Result<Value, String> {
        if args.len() != 2 {
            return Err(format!("- requires 2 arguments, got {}", args.len()));
        }

        match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - *b as f64)),
            _ => Err("- requires numeric arguments".to_string()),
        }
    }

    fn builtin_mul(&self, args: &[Value]) -> Result<Value, String> {
        if args.len() != 2 {
            return Err(format!("* requires 2 arguments, got {}", args.len()));
        }

        match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * *b as f64)),
            _ => Err("* requires numeric arguments".to_string()),
        }
    }

    fn builtin_div(&self, args: &[Value]) -> Result<Value, String> {
        if args.len() != 2 {
            return Err(format!("/ requires 2 arguments, got {}", args.len()));
        }

        match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(b)) => {
                if *b == 0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(Value::Int(a / b))
                }
            }
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 / b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / *b as f64)),
            _ => Err("/ requires numeric arguments".to_string()),
        }
    }

    fn builtin_lt(&self, args: &[Value]) -> Result<Value, String> {
        if args.len() != 2 {
            return Err(format!("< requires 2 arguments, got {}", args.len()));
        }

        let result = match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(b)) => a < b,
            (Value::Float(a), Value::Float(b)) => a < b,
            (Value::Int(a), Value::Float(b)) => (*a as f64) < *b,
            (Value::Float(a), Value::Int(b)) => *a < (*b as f64),
            _ => return Err("< requires numeric arguments".to_string()),
        };

        Ok(Value::Bool(result))
    }

    fn builtin_gt(&self, args: &[Value]) -> Result<Value, String> {
        if args.len() != 2 {
            return Err(format!("> requires 2 arguments, got {}", args.len()));
        }

        let result = match (&args[0], &args[1]) {
            (Value::Int(a), Value::Int(b)) => a > b,
            (Value::Float(a), Value::Float(b)) => a > b,
            (Value::Int(a), Value::Float(b)) => (*a as f64) > *b,
            (Value::Float(a), Value::Int(b)) => *a > (*b as f64),
            _ => return Err("> requires numeric arguments".to_string()),
        };

        Ok(Value::Bool(result))
    }

    fn builtin_eq(&self, args: &[Value]) -> Result<Value, String> {
        if args.len() != 2 {
            return Err(format!("= requires 2 arguments, got {}", args.len()));
        }

        Ok(Value::Bool(args[0] == args[1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clojure_ast::analyze;
    use crate::reader::read;

    #[test]
    fn test_eval_literal() {
        let mut eval = Evaluator::new();
        let val = read("42").unwrap();
        let ast = analyze(&val).unwrap();
        let result = eval.eval(&ast).unwrap();
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn test_eval_add() {
        let mut eval = Evaluator::new();
        let val = read("(+ 1 2)").unwrap();
        let ast = analyze(&val).unwrap();
        let result = eval.eval(&ast).unwrap();
        assert_eq!(result, Value::Int(3));
    }

    #[test]
    fn test_eval_def() {
        let mut eval = Evaluator::new();
        let val = read("(def x 42)").unwrap();
        let ast = analyze(&val).unwrap();
        eval.eval(&ast).unwrap();

        // Now use x
        let val2 = read("x").unwrap();
        let ast2 = analyze(&val2).unwrap();
        let result = eval.eval(&ast2).unwrap();
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn test_eval_if() {
        let mut eval = Evaluator::new();
        let val = read("(if true 1 2)").unwrap();
        let ast = analyze(&val).unwrap();
        let result = eval.eval(&ast).unwrap();
        assert_eq!(result, Value::Int(1));

        let val2 = read("(if false 1 2)").unwrap();
        let ast2 = analyze(&val2).unwrap();
        let result2 = eval.eval(&ast2).unwrap();
        assert_eq!(result2, Value::Int(2));
    }

    #[test]
    fn test_eval_complex() {
        let mut eval = Evaluator::new();

        // (def x 42)
        let val = read("(def x 42)").unwrap();
        let ast = analyze(&val).unwrap();
        eval.eval(&ast).unwrap();

        // (if (< x 100) (+ x 1) x)
        let val2 = read("(if (< x 100) (+ x 1) x)").unwrap();
        let ast2 = analyze(&val2).unwrap();
        let result = eval.eval(&ast2).unwrap();
        assert_eq!(result, Value::Int(43));
    }
}
