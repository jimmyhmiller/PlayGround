//! Interpreter for the structural OOP language
//!
//! A tree-walking interpreter that evaluates expressions to values.

use crate::expr::{ClassDef, Expr, ObjectField};
use crate::value::{Env, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

/// Evaluation error
#[derive(Debug)]
pub struct EvalError {
    pub message: String,
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for EvalError {}

fn error<T>(msg: impl Into<String>) -> Result<T, EvalError> {
    Err(EvalError {
        message: msg.into(),
    })
}

/// Resolve a value through any Ref wrappers
fn resolve(val: Value) -> Value {
    match val {
        Value::Ref(r) => {
            let inner = r.borrow().clone();
            resolve(inner)
        }
        other => other,
    }
}

/// Evaluate an expression in the given environment
pub fn eval(expr: &Expr, env: &Env) -> Result<Value, EvalError> {
    match expr {
        // === Literals ===
        Expr::Bool(b) => Ok(Value::Bool(*b)),
        Expr::Int(n) => Ok(Value::Int(*n)),
        Expr::String(s) => Ok(Value::String(s.clone())),

        // === Variables ===
        Expr::Var(name) => env
            .get(name)
            .map(resolve)
            .ok_or_else(|| EvalError {
                message: format!("Unbound variable: {}", name),
            }),

        // === Functions ===
        Expr::Lambda(param, body) => Ok(Value::Closure {
            param: param.clone(),
            body: (**body).clone(),
            env: env.clone(),
        }),

        Expr::App(func, arg) => {
            let func_val = resolve(eval(func, env)?);
            let arg_val = eval(arg, env)?;

            match func_val {
                Value::Closure { param, body, env: closure_env } => {
                    let new_env = closure_env.extend(param, arg_val);
                    eval(&body, &new_env)
                }
                _ => error(format!("Cannot apply non-function: {:?}", func_val)),
            }
        }

        // === Objects ===
        Expr::Object(fields) => eval_object(fields, env, None),

        Expr::FieldAccess(obj, field) => {
            let obj_val = resolve(eval(obj, env)?);
            match obj_val {
                Value::Object(fields) => fields
                    .get(field)
                    .cloned()
                    .map(resolve)
                    .ok_or_else(|| EvalError {
                        message: format!("Object has no field: {}", field),
                    }),
                _ => error(format!("Cannot access field on non-object: {:?}", obj_val)),
            }
        }

        Expr::This => {
            // Look for __this__ in the environment (captured by closures in objects)
            env.get("__this__")
                .map(resolve)
                .ok_or_else(|| EvalError {
                    message: "'this' used outside of object context".to_string(),
                })
        }

        // === Control ===
        Expr::If(cond, then_, else_) => {
            let cond_val = resolve(eval(cond, env)?);
            match cond_val {
                Value::Bool(true) => eval(then_, env),
                Value::Bool(false) => eval(else_, env),
                _ => error(format!("Condition must be boolean, got: {:?}", cond_val)),
            }
        }

        Expr::Let(name, value, body) => {
            let val = eval(value, env)?;
            let new_env = env.extend(name.clone(), val);
            eval(body, &new_env)
        }

        Expr::LetRec(name, value, body) => {
            // Create a reference cell for the recursive binding
            let placeholder = Rc::new(RefCell::new(Value::Uninitialized));
            let rec_env = env.extend(name.clone(), Value::Ref(placeholder.clone()));

            // Evaluate the value in the recursive environment
            let val = eval(value, &rec_env)?;

            // Backpatch the placeholder
            *placeholder.borrow_mut() = val;

            // Evaluate the body
            eval(body, &rec_env)
        }

        Expr::LetRecMutual(bindings, body) => {
            // Create placeholders for all bindings
            let mut placeholders: Vec<Rc<RefCell<Value>>> = Vec::new();
            let mut rec_bindings = HashMap::new();

            for (name, _) in bindings {
                let placeholder = Rc::new(RefCell::new(Value::Uninitialized));
                rec_bindings.insert(name.clone(), Value::Ref(placeholder.clone()));
                placeholders.push(placeholder);
            }

            let rec_env = env.extend_many(rec_bindings);

            // Evaluate each binding and backpatch
            for (i, (_, expr)) in bindings.iter().enumerate() {
                let val = eval(expr, &rec_env)?;
                *placeholders[i].borrow_mut() = val;
            }

            // Evaluate the body
            eval(body, &rec_env)
        }

        Expr::Block(classes, body) => {
            // Desugar classes to lambdas and treat as mutual recursion
            let bindings: Vec<(String, Expr)> = classes
                .iter()
                .map(|c| (c.name.clone(), c.to_lambda()))
                .collect();

            if bindings.is_empty() {
                eval(body, env)
            } else {
                // Create placeholders for all class constructors
                let mut placeholders: Vec<Rc<RefCell<Value>>> = Vec::new();
                let mut rec_bindings = HashMap::new();

                for (name, _) in &bindings {
                    let placeholder = Rc::new(RefCell::new(Value::Uninitialized));
                    rec_bindings.insert(name.clone(), Value::Ref(placeholder.clone()));
                    placeholders.push(placeholder);
                }

                let rec_env = env.extend_many(rec_bindings);

                // Evaluate each class constructor and backpatch
                for (i, (_, expr)) in bindings.iter().enumerate() {
                    let val = eval(expr, &rec_env)?;
                    *placeholders[i].borrow_mut() = val;
                }

                // Evaluate the body
                eval(body, &rec_env)
            }
        }

        Expr::Call(func, args) => {
            // Multi-argument call: f(a, b, c) = f(a)(b)(c)
            let mut result = eval(func, env)?;

            // Handle zero-arg call: f() where f is () => expr
            if args.is_empty() {
                result = resolve(result);
                match result {
                    Value::Closure { ref param, ref body, ref env } if param == "_unit" => {
                        // Zero-arg function: apply with unit value
                        let new_env = env.extend(param.clone(), Value::Bool(false)); // dummy value
                        return eval(body, &new_env);
                    }
                    _ => return Ok(result), // Not a thunk, return as-is
                }
            }

            for arg in args {
                let arg_val = eval(arg, env)?;
                result = resolve(result);
                match result {
                    Value::Closure { param, body, env: closure_env } => {
                        let new_env = closure_env.extend(param, arg_val);
                        result = eval(&body, &new_env)?;
                    }
                    _ => return error(format!("Cannot apply non-function: {:?}", result)),
                }
            }
            Ok(result)
        }

        // === Binary Operators ===
        Expr::Eq(left, right) => {
            let l = resolve(eval(left, env)?);
            let r = resolve(eval(right, env)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a == b)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a == b)),
                (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a == b)),
                (l, r) => error(format!("Cannot compare {:?} and {:?}", l, r)),
            }
        }

        Expr::And(left, right) => {
            let l = resolve(eval(left, env)?);
            match l {
                Value::Bool(false) => Ok(Value::Bool(false)),
                Value::Bool(true) => {
                    let r = resolve(eval(right, env)?);
                    match r {
                        Value::Bool(b) => Ok(Value::Bool(b)),
                        _ => error(format!("'&&' requires booleans, got: {:?}", r)),
                    }
                }
                _ => error(format!("'&&' requires booleans, got: {:?}", l)),
            }
        }

        Expr::Or(left, right) => {
            let l = resolve(eval(left, env)?);
            match l {
                Value::Bool(true) => Ok(Value::Bool(true)),
                Value::Bool(false) => {
                    let r = resolve(eval(right, env)?);
                    match r {
                        Value::Bool(b) => Ok(Value::Bool(b)),
                        _ => error(format!("'||' requires booleans, got: {:?}", r)),
                    }
                }
                _ => error(format!("'||' requires booleans, got: {:?}", l)),
            }
        }

        Expr::Add(left, right) => {
            let l = resolve(eval(left, env)?);
            let r = resolve(eval(right, env)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (l, r) => error(format!("'+' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Sub(left, right) => {
            let l = resolve(eval(left, env)?);
            let r = resolve(eval(right, env)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
                (l, r) => error(format!("'-' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Mul(left, right) => {
            let l = resolve(eval(left, env)?);
            let r = resolve(eval(right, env)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
                (l, r) => error(format!("'*' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Div(left, right) => {
            let l = resolve(eval(left, env)?);
            let r = resolve(eval(right, env)?);
            match (l, r) {
                (Value::Int(_), Value::Int(0)) => error("Division by zero"),
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)),
                (l, r) => error(format!("'/' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Concat(left, right) => {
            let l = resolve(eval(left, env)?);
            let r = resolve(eval(right, env)?);
            // Flexible concat: auto-convert to strings
            let l_str = value_to_string(&l);
            let r_str = value_to_string(&r);
            Ok(Value::String(l_str + &r_str))
        }

        Expr::Mod(left, right) => {
            let l = resolve(eval(left, env)?);
            let r = resolve(eval(right, env)?);
            match (l, r) {
                (Value::Int(_), Value::Int(0)) => error("Modulo by zero"),
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a % b)),
                (l, r) => error(format!("'%' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Lt(left, right) => {
            let l = resolve(eval(left, env)?);
            let r = resolve(eval(right, env)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
                (l, r) => error(format!("'<' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::LtEq(left, right) => {
            let l = resolve(eval(left, env)?);
            let r = resolve(eval(right, env)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
                (l, r) => error(format!("'<=' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Gt(left, right) => {
            let l = resolve(eval(left, env)?);
            let r = resolve(eval(right, env)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
                (l, r) => error(format!("'>' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::GtEq(left, right) => {
            let l = resolve(eval(left, env)?);
            let r = resolve(eval(right, env)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
                (l, r) => error(format!("'>=' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Not(expr) => {
            let v = resolve(eval(expr, env)?);
            match v {
                Value::Bool(b) => Ok(Value::Bool(!b)),
                v => error(format!("'!' requires boolean, got: {:?}", v)),
            }
        }
    }
}

/// Convert a value to a string for flexible concatenation
fn value_to_string(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        Value::Int(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        other => format!("{}", other),
    }
}

/// Evaluate an object literal with support for `this` and spread
fn eval_object(
    fields: &[ObjectField],
    env: &Env,
    this_val: Option<Rc<RefCell<Value>>>,
) -> Result<Value, EvalError> {
    // Create a placeholder for `this`
    let this_ref = this_val.unwrap_or_else(|| Rc::new(RefCell::new(Value::Uninitialized)));

    // Extend environment with `this`
    let obj_env = env.extend("__this__".to_string(), Value::Ref(this_ref.clone()));

    // Evaluate fields
    let mut result_fields: HashMap<String, Value> = HashMap::new();

    for field in fields {
        match field {
            ObjectField::Field(name, expr) => {
                let val = eval_with_this(expr, &obj_env, &this_ref)?;
                result_fields.insert(name.clone(), val);
            }
            ObjectField::Spread(expr) => {
                let spread_val = resolve(eval_with_this(expr, &obj_env, &this_ref)?);
                match spread_val {
                    Value::Object(spread_fields) => {
                        result_fields.extend(spread_fields);
                    }
                    _ => {
                        return error(format!("Cannot spread non-object: {:?}", spread_val));
                    }
                }
            }
        }
    }

    let obj = Value::Object(result_fields);

    // Backpatch `this`
    *this_ref.borrow_mut() = obj.clone();

    Ok(obj)
}

/// Evaluate an expression, replacing `This` with a reference to the object
fn eval_with_this(
    expr: &Expr,
    env: &Env,
    this_ref: &Rc<RefCell<Value>>,
) -> Result<Value, EvalError> {
    match expr {
        Expr::This => Ok(Value::Ref(this_ref.clone())),

        // For nested objects, they get their own `this`
        Expr::Object(fields) => eval_object(fields, env, None),

        // For lambdas, we need to capture `this` in the closure environment
        Expr::Lambda(param, body) => {
            // Create a closure that remembers `this`
            let closure_env = env.extend("__this__".to_string(), Value::Ref(this_ref.clone()));
            Ok(Value::Closure {
                param: param.clone(),
                body: (**body).clone(),
                env: closure_env,
            })
        }

        // For other expressions, recursively evaluate
        _ => eval_expr_with_this(expr, env, this_ref),
    }
}

/// Helper to evaluate expressions while preserving `this` context
fn eval_expr_with_this(
    expr: &Expr,
    env: &Env,
    this_ref: &Rc<RefCell<Value>>,
) -> Result<Value, EvalError> {
    match expr {
        // === Literals - no this needed ===
        Expr::Bool(b) => Ok(Value::Bool(*b)),
        Expr::Int(n) => Ok(Value::Int(*n)),
        Expr::String(s) => Ok(Value::String(s.clone())),

        // === Variables ===
        Expr::Var(name) => env
            .get(name)
            .map(resolve)
            .ok_or_else(|| EvalError {
                message: format!("Unbound variable: {}", name),
            }),

        // === This ===
        Expr::This => Ok(Value::Ref(this_ref.clone())),

        // === Functions ===
        Expr::Lambda(param, body) => {
            let closure_env = env.extend("__this__".to_string(), Value::Ref(this_ref.clone()));
            Ok(Value::Closure {
                param: param.clone(),
                body: (**body).clone(),
                env: closure_env,
            })
        }

        Expr::App(func, arg) => {
            let func_val = resolve(eval_with_this(func, env, this_ref)?);
            let arg_val = eval_with_this(arg, env, this_ref)?;

            match func_val {
                Value::Closure { param, body, env: closure_env } => {
                    let new_env = closure_env.extend(param, arg_val);
                    // When applying a closure, use regular eval (the closure has its own this)
                    eval(&body, &new_env)
                }
                _ => error(format!("Cannot apply non-function: {:?}", func_val)),
            }
        }

        // === Objects ===
        Expr::Object(fields) => eval_object(fields, env, None),

        Expr::FieldAccess(obj, field) => {
            let obj_val = resolve(eval_with_this(obj, env, this_ref)?);
            match obj_val {
                Value::Object(fields) => fields
                    .get(field)
                    .cloned()
                    .map(resolve)
                    .ok_or_else(|| EvalError {
                        message: format!("Object has no field: {}", field),
                    }),
                _ => error(format!("Cannot access field on non-object: {:?}", obj_val)),
            }
        }

        // === Control ===
        Expr::If(cond, then_, else_) => {
            let cond_val = resolve(eval_with_this(cond, env, this_ref)?);
            match cond_val {
                Value::Bool(true) => eval_with_this(then_, env, this_ref),
                Value::Bool(false) => eval_with_this(else_, env, this_ref),
                _ => error(format!("Condition must be boolean, got: {:?}", cond_val)),
            }
        }

        Expr::Let(name, value, body) => {
            let val = eval_with_this(value, env, this_ref)?;
            let new_env = env.extend(name.clone(), val);
            eval_with_this(body, &new_env, this_ref)
        }

        Expr::LetRec(name, value, body) => {
            let placeholder = Rc::new(RefCell::new(Value::Uninitialized));
            let rec_env = env.extend(name.clone(), Value::Ref(placeholder.clone()));
            let val = eval_with_this(value, &rec_env, this_ref)?;
            *placeholder.borrow_mut() = val;
            eval_with_this(body, &rec_env, this_ref)
        }

        Expr::LetRecMutual(bindings, body) => {
            let mut placeholders: Vec<Rc<RefCell<Value>>> = Vec::new();
            let mut rec_bindings = HashMap::new();

            for (name, _) in bindings {
                let placeholder = Rc::new(RefCell::new(Value::Uninitialized));
                rec_bindings.insert(name.clone(), Value::Ref(placeholder.clone()));
                placeholders.push(placeholder);
            }

            let rec_env = env.extend_many(rec_bindings);

            for (i, (_, expr)) in bindings.iter().enumerate() {
                let val = eval_with_this(expr, &rec_env, this_ref)?;
                *placeholders[i].borrow_mut() = val;
            }

            eval_with_this(body, &rec_env, this_ref)
        }

        Expr::Block(classes, body) => {
            let bindings: Vec<(String, Expr)> = classes
                .iter()
                .map(|c| (c.name.clone(), c.to_lambda()))
                .collect();

            if bindings.is_empty() {
                eval_with_this(body, env, this_ref)
            } else {
                let mut placeholders: Vec<Rc<RefCell<Value>>> = Vec::new();
                let mut rec_bindings = HashMap::new();

                for (name, _) in &bindings {
                    let placeholder = Rc::new(RefCell::new(Value::Uninitialized));
                    rec_bindings.insert(name.clone(), Value::Ref(placeholder.clone()));
                    placeholders.push(placeholder);
                }

                let rec_env = env.extend_many(rec_bindings);

                for (i, (_, expr)) in bindings.iter().enumerate() {
                    let val = eval_with_this(expr, &rec_env, this_ref)?;
                    *placeholders[i].borrow_mut() = val;
                }

                eval_with_this(body, &rec_env, this_ref)
            }
        }

        Expr::Call(func, args) => {
            let mut result = eval_with_this(func, env, this_ref)?;

            // Handle zero-arg call: f() where f is () => expr
            if args.is_empty() {
                result = resolve(result);
                match result {
                    Value::Closure { ref param, ref body, ref env } if param == "_unit" => {
                        let new_env = env.extend(param.clone(), Value::Bool(false));
                        return eval(body, &new_env);
                    }
                    _ => return Ok(result),
                }
            }

            for arg in args {
                let arg_val = eval_with_this(arg, env, this_ref)?;
                result = resolve(result);
                match result {
                    Value::Closure { param, body, env: closure_env } => {
                        let new_env = closure_env.extend(param, arg_val);
                        result = eval(&body, &new_env)?;
                    }
                    _ => return error(format!("Cannot apply non-function: {:?}", result)),
                }
            }
            Ok(result)
        }

        // === Binary Operators ===
        Expr::Eq(left, right) => {
            let l = resolve(eval_with_this(left, env, this_ref)?);
            let r = resolve(eval_with_this(right, env, this_ref)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a == b)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a == b)),
                (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a == b)),
                (l, r) => error(format!("Cannot compare {:?} and {:?}", l, r)),
            }
        }

        Expr::And(left, right) => {
            let l = resolve(eval_with_this(left, env, this_ref)?);
            match l {
                Value::Bool(false) => Ok(Value::Bool(false)),
                Value::Bool(true) => {
                    let r = resolve(eval_with_this(right, env, this_ref)?);
                    match r {
                        Value::Bool(b) => Ok(Value::Bool(b)),
                        _ => error(format!("'&&' requires booleans, got: {:?}", r)),
                    }
                }
                _ => error(format!("'&&' requires booleans, got: {:?}", l)),
            }
        }

        Expr::Or(left, right) => {
            let l = resolve(eval_with_this(left, env, this_ref)?);
            match l {
                Value::Bool(true) => Ok(Value::Bool(true)),
                Value::Bool(false) => {
                    let r = resolve(eval_with_this(right, env, this_ref)?);
                    match r {
                        Value::Bool(b) => Ok(Value::Bool(b)),
                        _ => error(format!("'||' requires booleans, got: {:?}", r)),
                    }
                }
                _ => error(format!("'||' requires booleans, got: {:?}", l)),
            }
        }

        Expr::Add(left, right) => {
            let l = resolve(eval_with_this(left, env, this_ref)?);
            let r = resolve(eval_with_this(right, env, this_ref)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (l, r) => error(format!("'+' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Sub(left, right) => {
            let l = resolve(eval_with_this(left, env, this_ref)?);
            let r = resolve(eval_with_this(right, env, this_ref)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
                (l, r) => error(format!("'-' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Mul(left, right) => {
            let l = resolve(eval_with_this(left, env, this_ref)?);
            let r = resolve(eval_with_this(right, env, this_ref)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
                (l, r) => error(format!("'*' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Div(left, right) => {
            let l = resolve(eval_with_this(left, env, this_ref)?);
            let r = resolve(eval_with_this(right, env, this_ref)?);
            match (l, r) {
                (Value::Int(_), Value::Int(0)) => error("Division by zero"),
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)),
                (l, r) => error(format!("'/' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Concat(left, right) => {
            let l = resolve(eval_with_this(left, env, this_ref)?);
            let r = resolve(eval_with_this(right, env, this_ref)?);
            // Flexible concat: auto-convert to strings
            let l_str = value_to_string(&l);
            let r_str = value_to_string(&r);
            Ok(Value::String(l_str + &r_str))
        }

        Expr::Mod(left, right) => {
            let l = resolve(eval_with_this(left, env, this_ref)?);
            let r = resolve(eval_with_this(right, env, this_ref)?);
            match (l, r) {
                (Value::Int(_), Value::Int(0)) => error("Modulo by zero"),
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a % b)),
                (l, r) => error(format!("'%' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Lt(left, right) => {
            let l = resolve(eval_with_this(left, env, this_ref)?);
            let r = resolve(eval_with_this(right, env, this_ref)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
                (l, r) => error(format!("'<' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::LtEq(left, right) => {
            let l = resolve(eval_with_this(left, env, this_ref)?);
            let r = resolve(eval_with_this(right, env, this_ref)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
                (l, r) => error(format!("'<=' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Gt(left, right) => {
            let l = resolve(eval_with_this(left, env, this_ref)?);
            let r = resolve(eval_with_this(right, env, this_ref)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
                (l, r) => error(format!("'>' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::GtEq(left, right) => {
            let l = resolve(eval_with_this(left, env, this_ref)?);
            let r = resolve(eval_with_this(right, env, this_ref)?);
            match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
                (l, r) => error(format!("'>=' requires integers, got: {:?} and {:?}", l, r)),
            }
        }

        Expr::Not(expr) => {
            let v = resolve(eval_with_this(expr, env, this_ref)?);
            match v {
                Value::Bool(b) => Ok(Value::Bool(!b)),
                v => error(format!("'!' requires boolean, got: {:?}", v)),
            }
        }
    }
}

/// Convenience function to evaluate a string
pub fn eval_str(input: &str) -> Result<Value, String> {
    use crate::parser::parse;

    let expr = parse(input).map_err(|e| format!("Parse error: {}", e))?;
    let env = Env::new();
    eval(&expr, &env).map_err(|e| format!("Eval error: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literals() {
        assert!(matches!(eval_str("42").unwrap(), Value::Int(42)));
        assert!(matches!(eval_str("true").unwrap(), Value::Bool(true)));
        assert!(matches!(eval_str("\"hello\"").unwrap(), Value::String(s) if s == "hello"));
    }

    #[test]
    fn test_arithmetic() {
        assert!(matches!(eval_str("1 + 2").unwrap(), Value::Int(3)));
        assert!(matches!(eval_str("5 - 3").unwrap(), Value::Int(2)));
        assert!(matches!(eval_str("4 * 3").unwrap(), Value::Int(12)));
        assert!(matches!(eval_str("10 / 2").unwrap(), Value::Int(5)));
    }

    #[test]
    fn test_lambda() {
        assert!(matches!(eval_str("((x) => x)(42)").unwrap(), Value::Int(42)));
        assert!(matches!(eval_str("((x, y) => x + y)(1, 2)").unwrap(), Value::Int(3)));
    }

    #[test]
    fn test_object() {
        let val = eval_str("{ x: 42 }.x").unwrap();
        assert!(matches!(val, Value::Int(42)));
    }

    #[test]
    fn test_this_in_method() {
        // Object with method that returns this
        let val = eval_str("{ x: 42, getSelf: () => this }.getSelf().x").unwrap();
        assert!(matches!(val, Value::Int(42)));
    }

    #[test]
    fn test_this_returns_self() {
        // Classic set-like pattern: insert returns this
        let val = eval_str("{ val: 1, inc: () => this }.inc().val").unwrap();
        assert!(matches!(val, Value::Int(1)));
    }

    #[test]
    fn test_block_with_class() {
        // Use classes for recursion - zero-param classes work as thunks
        let result = eval_str(
            "{ class Fact() { call: (n) => n == 0 ? 1 : n * Fact().call(n - 1) } Fact().call(5) }"
        ).unwrap();
        assert!(matches!(result, Value::Int(120)));
    }

    #[test]
    fn test_counter_class() {
        // Classic counter pattern with increment using zero-arg lambdas
        let result = eval_str(
            "{ class Counter(n) { val: n, inc: () => Counter(n + 1) } Counter(0).inc().inc().val }"
        ).unwrap();
        assert!(matches!(result, Value::Int(2)));
    }

    #[test]
    fn test_zero_arg_lambda() {
        // Zero-arg lambdas now work properly
        let result = eval_str("(() => 42)()").unwrap();
        assert!(matches!(result, Value::Int(42)));

        let result = eval_str("(() => (() => 1)())()").unwrap();
        assert!(matches!(result, Value::Int(1)));
    }

    #[test]
    fn test_spread() {
        let val = eval_str("{ ...{ x: 1, y: 2 }, z: 3 }.y").unwrap();
        assert!(matches!(val, Value::Int(2)));
    }
}
