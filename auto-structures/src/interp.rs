//! Tree-walking interpreter. It runs the program using whichever concrete
//! data structures the analysis selected (specialized) or, when `naive` is
//! set, the slow linear equivalents — same answers, different performance.

use crate::analysis::Analysis;
use crate::ast::*;
use crate::value::{format_value, values_eq, Coll, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Interp<'a> {
    analysis: &'a Analysis,
    naive: bool,
    scopes: Vec<HashMap<String, Value>>,
    pub output: Vec<String>,
}

enum Flow {
    Normal,
    Break,
    Continue,
}

impl<'a> Interp<'a> {
    pub fn new(analysis: &'a Analysis, naive: bool) -> Self {
        Interp {
            analysis,
            naive,
            scopes: vec![HashMap::new()],
            output: Vec::new(),
        }
    }

    pub fn run(&mut self, program: &Program) -> Result<(), String> {
        self.exec_block(program)?;
        Ok(())
    }

    // ----- environment --------------------------------------------------

    fn define(&mut self, name: &str, v: Value) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), v);
    }

    fn assign(&mut self, name: &str, v: Value) -> Result<(), String> {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), v);
                return Ok(());
            }
        }
        Err(format!("assignment to undefined variable '{}'", name))
    }

    fn lookup(&self, name: &str) -> Result<Value, String> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Ok(v.clone());
            }
        }
        Err(format!("undefined variable '{}'", name))
    }

    // ----- statements ---------------------------------------------------

    fn exec_block(&mut self, stmts: &[Stmt]) -> Result<Flow, String> {
        for s in stmts {
            match self.exec(s)? {
                Flow::Normal => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Normal)
    }

    fn exec(&mut self, stmt: &Stmt) -> Result<Flow, String> {
        match stmt {
            Stmt::Let { name, value } => {
                // The special case that makes the whole thing tick: when the
                // right-hand side is `collection()`, we don't evaluate it
                // generically — we instantiate the structure that analysis
                // chose for this particular variable.
                let v = if is_collection_ctor(value) {
                    let kind = self
                        .analysis
                        .kind_of(name)
                        .expect("collection var must have an inferred kind");
                    Coll::new(kind, self.naive).into_value()
                } else {
                    self.eval(value)?
                };
                self.define(name, v);
                Ok(Flow::Normal)
            }
            Stmt::Assign { name, value } => {
                let v = self.eval(value)?;
                self.assign(name, v)?;
                Ok(Flow::Normal)
            }
            Stmt::Expr(e) => {
                self.eval(e)?;
                Ok(Flow::Normal)
            }
            Stmt::If { cond, then, els } => {
                if self.eval(cond)?.is_truthy() {
                    self.scoped(then)
                } else {
                    self.scoped(els)
                }
            }
            Stmt::While { cond, body } => {
                while self.eval(cond)?.is_truthy() {
                    match self.scoped(body)? {
                        Flow::Break => break,
                        Flow::Continue | Flow::Normal => {}
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::For { var, iter, body } => {
                let items = self.eval_iterable(iter)?;
                for item in items {
                    self.scopes.push(HashMap::new());
                    self.define(var, item);
                    let flow = self.exec_block(body);
                    self.scopes.pop();
                    match flow? {
                        Flow::Break => break,
                        Flow::Continue | Flow::Normal => {}
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::Break => Ok(Flow::Break),
            Stmt::Continue => Ok(Flow::Continue),
        }
    }

    fn scoped(&mut self, body: &[Stmt]) -> Result<Flow, String> {
        self.scopes.push(HashMap::new());
        let flow = self.exec_block(body);
        self.scopes.pop();
        flow
    }

    fn eval_iterable(&mut self, e: &Expr) -> Result<Vec<Value>, String> {
        let v = self.eval(e)?;
        match v {
            Value::Coll(c) => Ok(c.borrow().iter_values()),
            other => Err(format!(
                "cannot iterate over {}",
                type_name(&other)
            )),
        }
    }

    // ----- expressions --------------------------------------------------

    fn eval(&mut self, e: &Expr) -> Result<Value, String> {
        match e {
            Expr::Int(n) => Ok(Value::Int(*n)),
            Expr::Str(s) => Ok(Value::Str(s.clone())),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Nil => Ok(Value::Nil),
            Expr::Var(name) => self.lookup(name),
            Expr::List(items) => {
                let mut vals = Vec::with_capacity(items.len());
                for it in items {
                    vals.push(self.eval(it)?);
                }
                Ok(Coll::seq(vals).into_value())
            }
            Expr::Unary(op, inner) => {
                let v = self.eval(inner)?;
                match op {
                    UnOp::Neg => match v {
                        Value::Int(i) => Ok(Value::Int(-i)),
                        _ => Err("unary '-' expects an integer".into()),
                    },
                    UnOp::Not => Ok(Value::Bool(!v.is_truthy())),
                }
            }
            Expr::Binary(op, l, r) => self.eval_binary(*op, l, r),
            Expr::Call(name, args) => self.eval_call(name, args),
        }
    }

    fn eval_binary(&mut self, op: BinOp, l: &Expr, r: &Expr) -> Result<Value, String> {
        // Short-circuiting logical operators.
        match op {
            BinOp::And => {
                let lv = self.eval(l)?;
                if !lv.is_truthy() {
                    return Ok(Value::Bool(false));
                }
                return Ok(Value::Bool(self.eval(r)?.is_truthy()));
            }
            BinOp::Or => {
                let lv = self.eval(l)?;
                if lv.is_truthy() {
                    return Ok(Value::Bool(true));
                }
                return Ok(Value::Bool(self.eval(r)?.is_truthy()));
            }
            _ => {}
        }

        let lv = self.eval(l)?;
        let rv = self.eval(r)?;
        match op {
            BinOp::Add => match (&lv, &rv) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (Value::Str(a), Value::Str(b)) => Ok(Value::Str(format!("{}{}", a, b))),
                (Value::Str(a), b) => Ok(Value::Str(format!("{}{}", a, format_value(b)))),
                (a, Value::Str(b)) => Ok(Value::Str(format!("{}{}", format_value(a), b))),
                _ => Err("'+' expects two ints or strings".into()),
            },
            BinOp::Sub => int_op(&lv, &rv, |a, b| a - b),
            BinOp::Mul => int_op(&lv, &rv, |a, b| a * b),
            BinOp::Div => {
                if let (Value::Int(_), Value::Int(0)) = (&lv, &rv) {
                    return Err("division by zero".into());
                }
                int_op(&lv, &rv, |a, b| a / b)
            }
            BinOp::Mod => {
                if let (Value::Int(_), Value::Int(0)) = (&lv, &rv) {
                    return Err("modulo by zero".into());
                }
                int_op(&lv, &rv, |a, b| a % b)
            }
            BinOp::Eq => Ok(Value::Bool(values_eq(&lv, &rv))),
            BinOp::Ne => Ok(Value::Bool(!values_eq(&lv, &rv))),
            BinOp::Lt => cmp_op(&lv, &rv, |o| o.is_lt()),
            BinOp::Le => cmp_op(&lv, &rv, |o| o.is_le()),
            BinOp::Gt => cmp_op(&lv, &rv, |o| o.is_gt()),
            BinOp::Ge => cmp_op(&lv, &rv, |o| o.is_ge()),
            BinOp::And | BinOp::Or => unreachable!(),
        }
    }

    fn eval_call(&mut self, name: &str, args: &[Expr]) -> Result<Value, String> {
        // Evaluate arguments first.
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            vals.push(self.eval(a)?);
        }

        // Builtins that don't operate on an existing collection.
        match name {
            "collection" => {
                // Reachable only if `collection()` is used outside a `let`.
                return Ok(Coll::new(crate::value::Kind::Set { ordered: false }, self.naive).into_value());
            }
            "print" => {
                let line = vals.iter().map(format_value).collect::<Vec<_>>().join(" ");
                self.output.push(line);
                return Ok(Value::Nil);
            }
            "range" => {
                let n = as_int(vals.first(), "range")?;
                let seq: Vec<Value> = (0..n).map(Value::Int).collect();
                return Ok(Coll::seq(seq).into_value());
            }
            "str" => {
                let v = vals.first().cloned().unwrap_or(Value::Nil);
                return Ok(Value::Str(format_value(&v)));
            }
            "sorted" => {
                let c = as_coll(vals.first(), "sorted")?;
                let sorted = c.borrow().sorted_values()?;
                return Ok(Coll::seq(sorted).into_value());
            }
            _ => {}
        }

        // Operations whose first argument is the collection.
        let first = vals.first().cloned();
        let arg1 = vals.get(1).cloned();
        let arg2 = vals.get(2).cloned();

        match name {
            "add" => with_coll_mut(&first, name, |c| c.add(req(arg1, name)?)),
            "has" => with_coll(&first, name, |c| c.has(&req(arg1, name)?)),
            "del" => with_coll_mut(&first, name, |c| c.del(&req(arg1, name)?)),
            "put" => with_coll_mut(&first, name, |c| c.put(req(arg1, name)?, req(arg2, name)?)),
            "get" => with_coll(&first, name, |c| c.get(&req(arg1, name)?)),
            "keys" => with_coll(&first, name, |c| Ok(Coll::seq(c.keys()).into_value())),
            "append" => with_coll_mut(&first, name, |c| c.append(req(arg1, name)?)),
            "at" => with_coll(&first, name, |c| c.at(as_int(arg1.as_ref(), name)?)),
            "set_at" => {
                with_coll_mut(&first, name, |c| c.set_at(as_int(arg1.as_ref(), name)?, req(arg2, name)?))
            }
            "push" => with_coll_mut(&first, name, |c| c.push(req(arg1, name)?)),
            "pop" => with_coll_mut(&first, name, |c| c.pop()),
            "peek" => with_coll(&first, name, |c| c.peek()),
            "enqueue" => with_coll_mut(&first, name, |c| c.enqueue(req(arg1, name)?)),
            "dequeue" => with_coll_mut(&first, name, |c| c.dequeue()),
            "front" => with_coll(&first, name, |c| c.front()),
            "min" => with_coll(&first, name, |c| c.min()),
            "max" => with_coll(&first, name, |c| c.max()),
            "size" | "len" => with_coll(&first, name, |c| Ok(Value::Int(c.size() as i64))),
            other => Err(format!("unknown function '{}'", other)),
        }
    }
}

// ----- small helpers ----------------------------------------------------

fn is_collection_ctor(e: &Expr) -> bool {
    matches!(e, Expr::Call(name, args) if name == "collection" && args.is_empty())
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Int(_) => "int",
        Value::Bool(_) => "bool",
        Value::Str(_) => "string",
        Value::Nil => "nil",
        Value::Coll(_) => "collection",
    }
}

fn int_op(l: &Value, r: &Value, f: impl Fn(i64, i64) -> i64) -> Result<Value, String> {
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(f(*a, *b))),
        _ => Err("arithmetic expects two integers".into()),
    }
}

fn cmp_op(l: &Value, r: &Value, f: impl Fn(std::cmp::Ordering) -> bool) -> Result<Value, String> {
    use std::cmp::Ordering;
    let ord = match (l, r) {
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Str(a), Value::Str(b)) => a.cmp(b),
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        _ => return Err("comparison expects two values of the same scalar type".into()),
    };
    let _ = Ordering::Equal;
    Ok(Value::Bool(f(ord)))
}

fn as_int(v: Option<&Value>, ctx: &str) -> Result<i64, String> {
    match v {
        Some(Value::Int(i)) => Ok(*i),
        _ => Err(format!("'{}' expects an integer argument", ctx)),
    }
}

fn as_coll(v: Option<&Value>, ctx: &str) -> Result<Rc<RefCell<Coll>>, String> {
    match v {
        Some(Value::Coll(c)) => Ok(c.clone()),
        _ => Err(format!("'{}' expects a collection as its first argument", ctx)),
    }
}

fn req(v: Option<Value>, ctx: &str) -> Result<Value, String> {
    v.ok_or_else(|| format!("'{}' is missing an argument", ctx))
}

fn with_coll(
    first: &Option<Value>,
    ctx: &str,
    f: impl FnOnce(&Coll) -> Result<Value, String>,
) -> Result<Value, String> {
    match first {
        Some(Value::Coll(c)) => f(&c.borrow()),
        _ => Err(format!("'{}' expects a collection as its first argument", ctx)),
    }
}

fn with_coll_mut(
    first: &Option<Value>,
    ctx: &str,
    f: impl FnOnce(&mut Coll) -> Result<Value, String>,
) -> Result<Value, String> {
    match first {
        Some(Value::Coll(c)) => f(&mut c.borrow_mut()),
        _ => Err(format!("'{}' expects a collection as its first argument", ctx)),
    }
}
