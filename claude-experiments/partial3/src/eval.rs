#![allow(dead_code)]
// This module provides a standard interpreter for validation and testing.
// The partial evaluator's results can be verified against this interpreter.

use crate::ast::{BinOp, Expr};
use crate::value::{array_from_vec, env_with_parent, new_object, Env, Value};

pub fn eval(expr: &Expr, env: &Env) -> Result<Value, String> {
    match expr {
        Expr::Int(n) => Ok(Value::Int(*n)),

        Expr::Bool(b) => Ok(Value::Bool(*b)),

        Expr::Var(name) => env
            .borrow()
            .get(name)
            .cloned()
            .ok_or_else(|| format!("Undefined variable: {}", name)),

        Expr::BinOp(op, left, right) => {
            let left_val = eval(left, env)?;
            let right_val = eval(right, env)?;
            eval_binop(op, &left_val, &right_val)
        }

        Expr::If(cond, then_branch, else_branch) => {
            let cond_val = eval(cond, env)?;
            match cond_val {
                Value::Bool(true) => eval(then_branch, env),
                Value::Bool(false) => eval(else_branch, env),
                _ => Err("If condition must be a boolean".to_string()),
            }
        }

        Expr::Let(name, value, body) => {
            let val = eval(value, env)?;
            let new_env = env_with_parent(env);
            new_env.borrow_mut().insert(name.clone(), val);
            eval(body, &new_env)
        }

        Expr::Fn(params, body) => Ok(Value::Closure {
            params: params.clone(),
            body: (**body).clone(),
            env: env.clone(),
        }),

        Expr::Call(func, args) => {
            let func_val = eval(func, env)?;
            match func_val {
                Value::Closure {
                    params,
                    body,
                    env: closure_env,
                } => {
                    if args.len() != params.len() {
                        return Err(format!(
                            "Function expects {} arguments, got {}",
                            params.len(),
                            args.len()
                        ));
                    }

                    let call_env = env_with_parent(&closure_env);
                    for (param, arg) in params.iter().zip(args.iter()) {
                        let arg_val = eval(arg, env)?;
                        call_env.borrow_mut().insert(param.clone(), arg_val);
                    }

                    eval(&body, &call_env)
                }
                _ => Err("Cannot call non-function".to_string()),
            }
        }

        Expr::Array(elements) => {
            let values: Result<Vec<Value>, String> =
                elements.iter().map(|e| eval(e, env)).collect();
            Ok(Value::Array(array_from_vec(values?)))
        }

        Expr::Index(arr, idx) => {
            let arr_val = eval(arr, env)?;
            let idx_val = eval(idx, env)?;

            match (&arr_val, &idx_val) {
                (Value::Array(elements), Value::Int(i)) => {
                    let i = *i as usize;
                    let borrowed = elements.borrow();
                    if i < borrowed.len() {
                        Ok(borrowed[i].clone())
                    } else {
                        Err(format!(
                            "Index {} out of bounds for array of length {}",
                            i,
                            borrowed.len()
                        ))
                    }
                }
                _ => Err("Index requires array and integer".to_string()),
            }
        }

        Expr::Len(arr) => {
            let arr_val = eval(arr, env)?;
            match arr_val {
                Value::Array(elements) => Ok(Value::Int(elements.borrow().len() as i64)),
                _ => Err("Len requires an array".to_string()),
            }
        }

        Expr::While(cond, body) => {
            loop {
                let cond_val = eval(cond, env)?;
                match cond_val {
                    Value::Bool(false) => return Ok(Value::Bool(false)),
                    Value::Bool(true) => {
                        eval(body, env)?;
                    }
                    _ => return Err("While condition must be a boolean".to_string()),
                }
            }
        }

        Expr::Set(name, value) => {
            let val = eval(value, env)?;
            if env.borrow().contains_key(name) {
                env.borrow_mut().insert(name.clone(), val.clone());
                Ok(val)
            } else {
                Err(format!("Cannot set! undefined variable: {}", name))
            }
        }

        Expr::Begin(exprs) => {
            if exprs.is_empty() {
                return Ok(Value::Undefined);
            }
            let mut result = Value::Undefined;
            for e in exprs {
                result = eval(e, env)?;
            }
            Ok(result)
        }

        // New literal types
        Expr::String(s) => Ok(Value::String(s.clone())),
        Expr::Undefined => Ok(Value::Undefined),
        Expr::Null => Ok(Value::Null),

        // Unary operations
        Expr::BitNot(inner) => {
            let v = eval(inner, env)?;
            match v {
                Value::Int(n) => Ok(Value::Int(!n)),
                _ => Err("Bitwise NOT requires an integer".to_string()),
            }
        }

        Expr::LogNot(inner) => {
            let v = eval(inner, env)?;
            // JavaScript truthiness
            let result = match v {
                Value::Bool(b) => !b,
                Value::Int(0) => true,
                Value::Int(_) => false,
                Value::String(ref s) if s.is_empty() => true,
                Value::String(_) => false,
                Value::Undefined => true,
                Value::Null => true,
                Value::Array(_) => false,
                Value::Object(_) => false,
                Value::Closure { .. } => false,
                Value::Opaque { .. } => false,
            };
            Ok(Value::Bool(result))
        }

        // Object literals
        Expr::Object(props) => {
            let obj = new_object();
            for (k, v) in props {
                let val = eval(v, env)?;
                obj.borrow_mut().insert(k.clone(), val);
            }
            Ok(Value::Object(obj))
        }

        // Property access: obj.prop
        Expr::PropAccess(obj_expr, prop) => {
            let obj_val = eval(obj_expr, env)?;
            match obj_val {
                Value::Object(obj_ref) => {
                    Ok(obj_ref.borrow().get(prop).cloned().unwrap_or(Value::Undefined))
                }
                Value::Array(arr) if prop == "length" => {
                    Ok(Value::Int(arr.borrow().len() as i64))
                }
                Value::String(s) if prop == "length" => {
                    Ok(Value::Int(s.len() as i64))
                }
                _ => Err(format!("Cannot access property '{}' on {:?}", prop, obj_val)),
            }
        }

        // Property set: obj.prop = value
        Expr::PropSet(obj_expr, prop, val_expr) => {
            let obj_val = eval(obj_expr, env)?;
            let value = eval(val_expr, env)?;
            match obj_val {
                Value::Object(obj_ref) => {
                    obj_ref.borrow_mut().insert(prop.clone(), value.clone());
                    Ok(value)
                }
                _ => Err(format!("Cannot set property '{}' on {:?}", prop, obj_val)),
            }
        }

        // Computed property access: obj[key]
        Expr::ComputedAccess(obj_expr, key_expr) => {
            let obj_val = eval(obj_expr, env)?;
            let key_val = eval(key_expr, env)?;

            match (&obj_val, &key_val) {
                (Value::Object(obj_ref), Value::String(k)) => {
                    Ok(obj_ref.borrow().get(k).cloned().unwrap_or(Value::Undefined))
                }
                (Value::Array(arr), Value::Int(i)) => {
                    let i = *i as usize;
                    let borrowed = arr.borrow();
                    if i < borrowed.len() {
                        Ok(borrowed[i].clone())
                    } else {
                        Ok(Value::Undefined)
                    }
                }
                (Value::Array(arr), Value::String(s)) if s == "length" => {
                    Ok(Value::Int(arr.borrow().len() as i64))
                }
                _ => Err(format!("Cannot access {:?} with key {:?}", obj_val, key_val)),
            }
        }

        // Computed property set: obj[key] = value
        Expr::ComputedSet(obj_expr, key_expr, val_expr) => {
            let obj_val = eval(obj_expr, env)?;
            let key_val = eval(key_expr, env)?;
            let value = eval(val_expr, env)?;

            match (&obj_val, &key_val) {
                (Value::Object(obj_ref), Value::String(k)) => {
                    obj_ref.borrow_mut().insert(k.clone(), value.clone());
                    Ok(value)
                }
                _ => Err(format!("Cannot set {:?}[{:?}]", obj_val, key_val)),
            }
        }

        // Switch statement
        Expr::Switch { discriminant, cases, default } => {
            let disc_val = eval(discriminant, env)?;

            // Find matching case
            for (case_val_expr, body) in cases {
                let case_val = eval(case_val_expr, env)?;
                if case_val == disc_val {
                    // Execute this case body
                    let mut result = Value::Undefined;
                    for stmt in body {
                        result = eval(stmt, env)?;
                        // TODO: handle break properly
                    }
                    return Ok(result);
                }
            }

            // No match - execute default if present
            if let Some(default_body) = default {
                let mut result = Value::Undefined;
                for stmt in default_body {
                    result = eval(stmt, env)?;
                }
                return Ok(result);
            }

            Ok(Value::Undefined)
        }

        // For loop
        Expr::For { init, cond, update, body } => {
            // Handle initialization
            if let Some(init_expr) = init {
                eval(init_expr, env)?;
            }

            loop {
                // Check condition
                let should_continue = if let Some(c) = cond {
                    let cond_val = eval(c, env)?;
                    match cond_val {
                        Value::Bool(b) => b,
                        _ => return Err("For condition must be a boolean".to_string()),
                    }
                } else {
                    true // Infinite loop if no condition
                };

                if !should_continue {
                    return Ok(Value::Undefined);
                }

                // Execute body
                eval(body, env)?;

                // Execute update
                if let Some(upd) = update {
                    eval(upd, env)?;
                }
            }
        }

        // Break and Continue - would need special handling with control flow
        Expr::Break => Err("Break outside of loop".to_string()),
        Expr::Continue => Err("Continue outside of loop".to_string()),

        // Throw
        Expr::Throw(inner) => {
            let val = eval(inner, env)?;
            Err(format!("Thrown: {:?}", val))
        }

        // New expression - creates an opaque value
        Expr::New(constructor, args) => {
            let ctor = eval(constructor, env)?;
            let arg_vals: Result<Vec<Value>, String> = args.iter().map(|a| eval(a, env)).collect();
            let _ = arg_vals?;
            // Create an opaque value representing the new expression
            Ok(Value::Opaque {
                label: format!("new {:?}", ctor),
                expr: expr.clone(),
                state: None,
            })
        }

        // Opaque expressions
        Expr::Opaque(label) => Ok(Value::Opaque {
            label: label.clone(),
            expr: expr.clone(),
            state: None,
        }),

        // TryCatch - simplified execution (no proper exception propagation)
        Expr::TryCatch { try_block, catch_param, catch_block, finally_block } => {
            let try_result = eval(try_block, env);

            let result = match try_result {
                Ok(v) => Ok(v),
                Err(e) => {
                    // Execute catch block
                    let catch_env = env_with_parent(env);
                    if let Some(param) = catch_param {
                        // Bind error to catch parameter as a string
                        catch_env.borrow_mut().insert(param.clone(), Value::String(e));
                    }
                    eval(catch_block, &catch_env)
                }
            };

            // Execute finally block if present
            if let Some(finally) = finally_block {
                eval(finally, env)?;
            }

            result
        }
    }
}

fn eval_binop(op: &BinOp, left: &Value, right: &Value) -> Result<Value, String> {
    match (op, left, right) {
        // Arithmetic
        (BinOp::Add, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_add(*b))),
        (BinOp::Sub, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_sub(*b))),
        (BinOp::Mul, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_mul(*b))),
        (BinOp::Div, Value::Int(a), Value::Int(b)) => {
            if *b == 0 {
                Err("Division by zero".to_string())
            } else {
                Ok(Value::Int(a / b))
            }
        }
        (BinOp::Mod, Value::Int(a), Value::Int(b)) => {
            if *b == 0 {
                Err("Modulo by zero".to_string())
            } else {
                Ok(Value::Int(a % b))
            }
        }

        // String concatenation
        (BinOp::Add, Value::String(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
        (BinOp::Add, Value::String(a), Value::Int(b)) => Ok(Value::String(format!("{}{}", a, b))),
        (BinOp::Add, Value::Int(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),

        // Comparison
        (BinOp::Lt, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
        (BinOp::Gt, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
        (BinOp::Lte, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
        (BinOp::Gte, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
        (BinOp::Eq, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a == b)),
        (BinOp::NotEq, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a != b)),
        (BinOp::Eq, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a == b)),
        (BinOp::NotEq, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a != b)),
        (BinOp::Eq, Value::String(a), Value::String(b)) => Ok(Value::Bool(a == b)),
        (BinOp::NotEq, Value::String(a), Value::String(b)) => Ok(Value::Bool(a != b)),
        (BinOp::Eq, Value::Undefined, Value::Undefined) => Ok(Value::Bool(true)),
        (BinOp::Eq, Value::Null, Value::Null) => Ok(Value::Bool(true)),
        (BinOp::Eq, Value::Undefined, Value::Null) => Ok(Value::Bool(true)),
        (BinOp::Eq, Value::Null, Value::Undefined) => Ok(Value::Bool(true)),

        // Logical
        (BinOp::And, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a && *b)),
        (BinOp::Or, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a || *b)),

        // Bitwise operations
        (BinOp::BitAnd, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a & b)),
        (BinOp::BitOr, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a | b)),
        (BinOp::BitXor, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a ^ b)),
        (BinOp::Shl, Value::Int(a), Value::Int(b)) => {
            let a32 = *a as i32;
            let shift = (*b as u32) & 0x1f;
            Ok(Value::Int((a32 << shift) as i64))
        }
        (BinOp::Shr, Value::Int(a), Value::Int(b)) => {
            let a32 = *a as i32;
            let shift = (*b as u32) & 0x1f;
            Ok(Value::Int((a32 >> shift) as i64))
        }
        (BinOp::UShr, Value::Int(a), Value::Int(b)) => {
            let a32 = *a as u32;
            let shift = (*b as u32) & 0x1f;
            Ok(Value::Int((a32 >> shift) as i64))
        }

        _ => Err(format!(
            "Invalid operands for {:?}: {:?} and {:?}",
            op, left, right
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;
    use crate::value::new_env;

    fn eval_str(s: &str) -> Result<Value, String> {
        let expr = parse(s)?;
        eval(&expr, &new_env())
    }

    #[test]
    fn test_eval_arithmetic() {
        assert_eq!(eval_str("(+ 1 2)").unwrap(), Value::Int(3));
        assert_eq!(eval_str("(* 3 4)").unwrap(), Value::Int(12));
        assert_eq!(eval_str("(- 10 3)").unwrap(), Value::Int(7));
    }

    #[test]
    fn test_eval_comparison() {
        assert_eq!(eval_str("(< 1 2)").unwrap(), Value::Bool(true));
        assert_eq!(eval_str("(< 2 1)").unwrap(), Value::Bool(false));
        assert_eq!(eval_str("(== 5 5)").unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_eval_let() {
        assert_eq!(eval_str("(let x 5 (+ x 3))").unwrap(), Value::Int(8));
    }

    #[test]
    fn test_eval_if() {
        assert_eq!(eval_str("(if true 1 2)").unwrap(), Value::Int(1));
        assert_eq!(eval_str("(if false 1 2)").unwrap(), Value::Int(2));
        assert_eq!(
            eval_str("(if (< 1 2) 10 20)").unwrap(),
            Value::Int(10)
        );
    }

    #[test]
    fn test_eval_function() {
        assert_eq!(
            eval_str("(let f (fn (x) (+ x 1)) (call f 5))").unwrap(),
            Value::Int(6)
        );
    }

    #[test]
    fn test_eval_closure() {
        assert_eq!(
            eval_str("(let y 10 (let f (fn (x) (+ x y)) (call f 5)))").unwrap(),
            Value::Int(15)
        );
    }

    #[test]
    fn test_eval_array() {
        assert_eq!(
            eval_str("(index (array 1 2 3) 1)").unwrap(),
            Value::Int(2)
        );
        assert_eq!(
            eval_str("(len (array 1 2 3 4 5))").unwrap(),
            Value::Int(5)
        );
    }

    #[test]
    fn test_eval_while() {
        // Count from 0 to 5 using mutable variable
        let code = "(let x 0
            (begin
                (while (< x 5)
                    (set! x (+ x 1)))
                x))";
        assert_eq!(eval_str(code).unwrap(), Value::Int(5));
    }

    #[test]
    fn test_eval_while_sum() {
        // Sum 1+2+3+4+5 = 15
        let code = "(let i 0
            (let sum 0
                (begin
                    (while (< i 5)
                        (begin
                            (set! i (+ i 1))
                            (set! sum (+ sum i))))
                    sum)))";
        assert_eq!(eval_str(code).unwrap(), Value::Int(15));
    }

    #[test]
    fn test_eval_set() {
        let code = "(let x 1 (begin (set! x 42) x))";
        assert_eq!(eval_str(code).unwrap(), Value::Int(42));
    }

    #[test]
    fn test_eval_begin() {
        let code = "(begin 1 2 3)";
        assert_eq!(eval_str(code).unwrap(), Value::Int(3));
    }
}
