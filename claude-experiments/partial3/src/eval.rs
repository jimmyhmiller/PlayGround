#![allow(dead_code)]
// This module provides a standard interpreter for validation and testing.
// The partial evaluator's results can be verified against this interpreter.

use crate::ast::{BinOp, Expr};
use crate::value::{array_from_vec, env_with_parent, new_object, Env, Value};
use std::cell::Cell;

thread_local! {
    static STEP_COUNT: Cell<u64> = Cell::new(0);
    static VERBOSE: Cell<bool> = Cell::new(false);
}

pub fn set_verbose(v: bool) {
    VERBOSE.with(|verbose| verbose.set(v));
}

/// Control flow signals for break/continue/return
#[derive(Debug, Clone)]
enum ControlFlow {
    Break,
    Continue,
    Return(Value),
}

/// Internal error type that handles both errors and control flow
#[derive(Debug)]
enum EvalError {
    Error(String),
    Control(ControlFlow),
}

impl From<String> for EvalError {
    fn from(s: String) -> Self {
        EvalError::Error(s)
    }
}

type EvalResult = Result<Value, EvalError>;

fn err(s: &str) -> EvalError {
    EvalError::Error(s.to_string())
}

pub fn eval(expr: &Expr, env: &Env) -> Result<Value, String> {
    match eval_inner(expr, env) {
        Ok(v) => Ok(v),
        Err(EvalError::Error(e)) => Err(e),
        Err(EvalError::Control(ControlFlow::Break)) => Err("Break outside of loop".to_string()),
        Err(EvalError::Control(ControlFlow::Continue)) => Err("Continue outside of loop".to_string()),
        Err(EvalError::Control(ControlFlow::Return(v))) => Ok(v),
    }
}

fn expr_kind(expr: &Expr) -> &'static str {
    match expr {
        Expr::Int(_) => "Int",
        Expr::Bool(_) => "Bool",
        Expr::String(_) => "String",
        Expr::Var(_) => "Var",
        Expr::BinOp(..) => "BinOp",
        Expr::If(..) => "If",
        Expr::Let(..) => "Let",
        Expr::Fn(..) => "Fn",
        Expr::Call(..) => "Call",
        Expr::Array(..) => "Array",
        Expr::Index(..) => "Index",
        Expr::Len(..) => "Len",
        Expr::While(..) => "While",
        Expr::For { .. } => "For",
        Expr::Set(..) => "Set",
        Expr::Begin(..) => "Begin",
        Expr::Undefined => "Undefined",
        Expr::Null => "Null",
        Expr::BitNot(..) => "BitNot",
        Expr::LogNot(..) => "LogNot",
        Expr::Object(..) => "Object",
        Expr::PropAccess(..) => "PropAccess",
        Expr::PropSet(..) => "PropSet",
        Expr::ComputedAccess(..) => "ComputedAccess",
        Expr::ComputedSet(..) => "ComputedSet",
        Expr::Switch { .. } => "Switch",
        Expr::Break => "Break",
        Expr::Continue => "Continue",
        Expr::Return(..) => "Return",
        Expr::Throw(..) => "Throw",
        Expr::New(..) => "New",
        Expr::Opaque(..) => "Opaque",
        Expr::TryCatch { .. } => "TryCatch",
    }
}

fn eval_inner(expr: &Expr, env: &Env) -> EvalResult {
    STEP_COUNT.with(|count| {
        let c = count.get() + 1;
        count.set(c);
        VERBOSE.with(|verbose| {
            if verbose.get() && c % 50_000 == 0 {
                eprintln!("[eval] step {}: {} ", c, expr_kind(expr));
            }
        });
    });
    match expr {
        Expr::Int(n) => Ok(Value::Int(*n)),

        Expr::Bool(b) => Ok(Value::Bool(*b)),

        Expr::Var(name) => env
            .borrow()
            .get(name)
            .cloned()
            .ok_or_else(|| err(&format!("Undefined variable: {}", name))),

        Expr::BinOp(op, left, right) => {
            let left_val = eval_inner(left, env)?;
            let right_val = eval_inner(right, env)?;
            eval_binop(op, &left_val, &right_val).map_err(EvalError::Error)
        }

        Expr::If(cond, then_branch, else_branch) => {
            let cond_val = eval_inner(cond, env)?;
            match cond_val {
                Value::Bool(true) => eval_inner(then_branch, env),
                Value::Bool(false) => eval_inner(else_branch, env),
                _ => Err(err("If condition must be a boolean")),
            }
        }

        Expr::Let(name, value, body) => {
            let val = eval_inner(value, env)?;
            let new_env = env_with_parent(env);
            new_env.borrow_mut().insert(name.clone(), val);
            eval_inner(body, &new_env)
        }

        Expr::Fn(params, body) => Ok(Value::Closure {
            params: params.clone(),
            body: (**body).clone(),
            env: env.clone(),
        }),

        Expr::Call(func, args) => {
            let func_val = eval_inner(func, env)?;
            match func_val {
                Value::Closure {
                    params,
                    body,
                    env: closure_env,
                } => {
                    if args.len() != params.len() {
                        return Err(err(&format!(
                            "Function expects {} arguments, got {}",
                            params.len(),
                            args.len()
                        )));
                    }

                    let call_env = env_with_parent(&closure_env);
                    for (param, arg) in params.iter().zip(args.iter()) {
                        let arg_val = eval_inner(arg, env)?;
                        call_env.borrow_mut().insert(param.clone(), arg_val);
                    }

                    // Handle return from function body
                    match eval_inner(&body, &call_env) {
                        Ok(v) => Ok(v),
                        Err(EvalError::Control(ControlFlow::Return(v))) => Ok(v),
                        Err(e) => Err(e),
                    }
                }
                _ => Err(err("Cannot call non-function")),
            }
        }

        Expr::Array(elements) => {
            let mut values = Vec::new();
            for e in elements {
                values.push(eval_inner(e, env)?);
            }
            Ok(Value::Array(array_from_vec(values)))
        }

        Expr::Index(arr, idx) => {
            let arr_val = eval_inner(arr, env)?;
            let idx_val = eval_inner(idx, env)?;

            match (&arr_val, &idx_val) {
                (Value::Array(elements), Value::Int(i)) => {
                    let i = *i as usize;
                    let borrowed = elements.borrow();
                    if i < borrowed.len() {
                        Ok(borrowed[i].clone())
                    } else {
                        Err(err(&format!(
                            "Index {} out of bounds for array of length {}",
                            i,
                            borrowed.len()
                        )))
                    }
                }
                _ => Err(err("Index requires array and integer")),
            }
        }

        Expr::Len(arr) => {
            let arr_val = eval_inner(arr, env)?;
            match arr_val {
                Value::Array(elements) => Ok(Value::Int(elements.borrow().len() as i64)),
                _ => Err(err("Len requires an array")),
            }
        }

        Expr::While(cond, body) => {
            loop {
                let cond_val = eval_inner(cond, env)?;
                match cond_val {
                    Value::Bool(false) => return Ok(Value::Undefined),
                    Value::Bool(true) => {
                        match eval_inner(body, env) {
                            Ok(_) => {}
                            Err(EvalError::Control(ControlFlow::Break)) => return Ok(Value::Undefined),
                            Err(EvalError::Control(ControlFlow::Continue)) => continue,
                            Err(e) => return Err(e),
                        }
                    }
                    _ => return Err(err("While condition must be a boolean")),
                }
            }
        }

        Expr::Set(name, value) => {
            let val = eval_inner(value, env)?;
            // JavaScript allows implicit global creation, so always allow set
            env.borrow_mut().insert(name.clone(), val.clone());
            Ok(val)
        }

        Expr::Begin(exprs) => {
            if exprs.is_empty() {
                return Ok(Value::Undefined);
            }
            let mut result = Value::Undefined;
            for e in exprs {
                result = eval_inner(e, env)?;
            }
            Ok(result)
        }

        // New literal types
        Expr::String(s) => Ok(Value::String(s.clone())),
        Expr::Undefined => Ok(Value::Undefined),
        Expr::Null => Ok(Value::Null),

        // Unary operations
        Expr::BitNot(inner) => {
            let v = eval_inner(inner, env)?;
            match v {
                Value::Int(n) => Ok(Value::Int(!n)),
                _ => Err(err("Bitwise NOT requires an integer")),
            }
        }

        Expr::LogNot(inner) => {
            let v = eval_inner(inner, env)?;
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
                let val = eval_inner(v, env)?;
                obj.borrow_mut().insert(k.clone(), val);
            }
            Ok(Value::Object(obj))
        }

        // Property access: obj.prop
        Expr::PropAccess(obj_expr, prop) => {
            let obj_val = eval_inner(obj_expr, env)?;
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
                _ => Err(err(&format!("Cannot access property '{}' on {:?}", prop, obj_val))),
            }
        }

        // Property set: obj.prop = value
        Expr::PropSet(obj_expr, prop, val_expr) => {
            let obj_val = eval_inner(obj_expr, env)?;
            let value = eval_inner(val_expr, env)?;
            match obj_val {
                Value::Object(obj_ref) => {
                    obj_ref.borrow_mut().insert(prop.clone(), value.clone());
                    Ok(value)
                }
                _ => Err(err(&format!("Cannot set property '{}' on {:?}", prop, obj_val))),
            }
        }

        // Computed property access: obj[key]
        Expr::ComputedAccess(obj_expr, key_expr) => {
            let obj_val = eval_inner(obj_expr, env)?;
            let key_val = eval_inner(key_expr, env)?;

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
                _ => Err(err(&format!("Cannot access {:?} with key {:?}", obj_val, key_val))),
            }
        }

        // Computed property set: obj[key] = value
        Expr::ComputedSet(obj_expr, key_expr, val_expr) => {
            let obj_val = eval_inner(obj_expr, env)?;
            let key_val = eval_inner(key_expr, env)?;
            let value = eval_inner(val_expr, env)?;

            match (&obj_val, &key_val) {
                (Value::Object(obj_ref), Value::String(k)) => {
                    obj_ref.borrow_mut().insert(k.clone(), value.clone());
                    Ok(value)
                }
                (Value::Array(arr), Value::Int(i)) => {
                    let i = *i as usize;
                    let mut borrowed = arr.borrow_mut();
                    // Extend array if needed
                    while borrowed.len() <= i {
                        borrowed.push(Value::Undefined);
                    }
                    borrowed[i] = value.clone();
                    Ok(value)
                }
                _ => Err(err(&format!("Cannot set {:?}[{:?}]", obj_val, key_val))),
            }
        }

        // Switch statement
        Expr::Switch { discriminant, cases, default } => {
            let disc_val = eval_inner(discriminant, env)?;

            // Find matching case
            for (case_val_expr, body) in cases {
                let case_val = eval_inner(case_val_expr, env)?;
                if values_equal(&case_val, &disc_val) {
                    // Execute this case body
                    for stmt in body {
                        match eval_inner(stmt, env) {
                            Ok(_) => {}
                            Err(EvalError::Control(ControlFlow::Break)) => return Ok(Value::Undefined),
                            Err(e) => return Err(e),
                        }
                    }
                    return Ok(Value::Undefined);
                }
            }

            // No match - execute default if present
            if let Some(default_body) = default {
                for stmt in default_body {
                    match eval_inner(stmt, env) {
                        Ok(_) => {}
                        Err(EvalError::Control(ControlFlow::Break)) => return Ok(Value::Undefined),
                        Err(e) => return Err(e),
                    }
                }
            }

            Ok(Value::Undefined)
        }

        // For loop
        Expr::For { init, cond, update, body } => {
            // Handle initialization
            if let Some(init_expr) = init {
                eval_inner(init_expr, env)?;
            }

            loop {
                // Check condition
                let should_continue = if let Some(c) = cond {
                    let cond_val = eval_inner(c, env)?;
                    match cond_val {
                        Value::Bool(b) => b,
                        Value::Int(0) => false,
                        Value::Int(_) => true,
                        _ => return Err(err("For condition must be a boolean or number")),
                    }
                } else {
                    true // Infinite loop if no condition
                };

                if !should_continue {
                    return Ok(Value::Undefined);
                }

                // Execute body
                match eval_inner(body, env) {
                    Ok(_) => {}
                    Err(EvalError::Control(ControlFlow::Break)) => return Ok(Value::Undefined),
                    Err(EvalError::Control(ControlFlow::Continue)) => {}
                    Err(e) => return Err(e),
                }

                // Execute update
                if let Some(upd) = update {
                    eval_inner(upd, env)?;
                }
            }
        }

        // Break and Continue
        Expr::Break => Err(EvalError::Control(ControlFlow::Break)),
        Expr::Continue => Err(EvalError::Control(ControlFlow::Continue)),

        // Return
        Expr::Return(inner) => {
            let v = eval_inner(inner, env)?;
            Err(EvalError::Control(ControlFlow::Return(v)))
        }

        // Throw
        Expr::Throw(inner) => {
            let val = eval_inner(inner, env)?;
            Err(err(&format!("Thrown: {:?}", val)))
        }

        // New expression - creates an opaque value
        Expr::New(constructor, args) => {
            let ctor = eval_inner(constructor, env)?;
            let mut arg_vals = Vec::new();
            for a in args {
                arg_vals.push(eval_inner(a, env)?);
            }
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

        // TryCatch
        Expr::TryCatch { try_block, catch_param, catch_block, finally_block } => {
            let try_result = eval_inner(try_block, env);

            let result = match try_result {
                Ok(v) => Ok(v),
                Err(EvalError::Error(e)) => {
                    // Execute catch block
                    let catch_env = env_with_parent(env);
                    if let Some(param) = catch_param {
                        // Bind error to catch parameter as a string
                        catch_env.borrow_mut().insert(param.clone(), Value::String(e));
                    }
                    eval_inner(catch_block, &catch_env)
                }
                Err(EvalError::Control(cf)) => {
                    // Control flow passes through try/catch
                    Err(EvalError::Control(cf))
                }
            };

            // Execute finally block if present
            if let Some(finally) = finally_block {
                eval_inner(finally, env)?;
            }

            result
        }
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Undefined, Value::Undefined) => true,
        (Value::Null, Value::Null) => true,
        (Value::Undefined, Value::Null) => true,
        (Value::Null, Value::Undefined) => true,
        _ => false,
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
