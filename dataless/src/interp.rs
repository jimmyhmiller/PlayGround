//! The interpreter. It runs a representation-independent program against a set
//! of declarations. The same program runs unchanged whatever the declarations
//! say — that is the whole point.
//!
//! Reference resolution is the crux: `name(handle)` looks at the handle's
//! collection. If `name` is a stored field there, it is read/written directly;
//! if `name` is declared `computed`, the function body runs. The program cannot
//! tell the difference — a data reference and a function reference are
//! syntactically identical and interchanged purely in the declarations.

use crate::ast::*;
use crate::repr::{Collection, FieldLayout, Type};
use crate::value::{values_cmp, values_eq, Value};
use std::collections::HashMap;

struct Computed {
    param: String,
    body: Expr,
}

struct GenState {
    coll: usize,
    cond: Expr,
    next_pos: usize,
    current: Value,
}

struct Monitor {
    cond: Expr,
    body: Vec<Stmt>,
    last: bool,
}

pub struct Interp {
    pub colls: Vec<Collection>,
    coll_id: HashMap<String, usize>,
    computed: HashMap<String, Computed>,
    scopes: Vec<HashMap<String, Value>>,
    /// Active iterations: (collection id, current member id). Drives `it` and
    /// implied qualification.
    iter_stack: Vec<(usize, usize)>,
    gens: HashMap<String, GenState>,
    monitors: Vec<Monitor>,
    in_monitor: bool,
    pub output: Vec<String>,
}

impl Interp {
    pub fn from_decls(decls: &Decls) -> Result<Interp, String> {
        let mut colls = Vec::new();
        let mut coll_id = HashMap::new();
        let mut computed = HashMap::new();

        for d in decls {
            match d {
                Decl::Collection { name, rep, fields } => {
                    if coll_id.contains_key(name) {
                        return Err(format!("collection `{}` declared twice", name));
                    }
                    let id = colls.len();
                    let layouts = build_layouts(fields)?;
                    colls.push(Collection::new(name.clone(), *rep, layouts));
                    coll_id.insert(name.clone(), id);
                }
                Decl::Computed { name, param, body } => {
                    computed.insert(
                        name.clone(),
                        Computed {
                            param: param.clone(),
                            body: body.clone(),
                        },
                    );
                }
            }
        }

        Ok(Interp {
            colls,
            coll_id,
            computed,
            scopes: vec![HashMap::new()],
            iter_stack: Vec::new(),
            gens: HashMap::new(),
            monitors: Vec::new(),
            in_monitor: false,
            output: Vec::new(),
        })
    }

    pub fn run(&mut self, program: &Program) -> Result<(), String> {
        self.exec_block(program)
    }

    /// Total positional-walk/shift work across all collections.
    pub fn total_steps(&self) -> u64 {
        self.colls.iter().map(|c| c.steps).sum()
    }

    // ===== scopes =======================================================

    fn define(&mut self, name: &str, v: Value) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), v);
    }

    fn scope_get(&self, name: &str) -> Option<Value> {
        for s in self.scopes.iter().rev() {
            if let Some(v) = s.get(name) {
                return Some(v.clone());
            }
        }
        None
    }

    fn scope_set(&mut self, name: &str, v: Value) -> bool {
        for s in self.scopes.iter_mut().rev() {
            if s.contains_key(name) {
                s.insert(name.to_string(), v);
                return true;
            }
        }
        false
    }

    fn coll(&self, name: &str) -> Result<usize, String> {
        self.coll_id
            .get(name)
            .copied()
            .ok_or_else(|| format!("unknown collection `{}`", name))
    }

    /// Resolve a bare field name through the active iterations (implied
    /// qualification): innermost iteration whose collection has that field.
    fn implied_field(&self, name: &str) -> Option<(usize, usize, usize)> {
        for &(cid, id) in self.iter_stack.iter().rev() {
            if let Some(fid) = self.colls[cid].field_id(name) {
                return Some((cid, id, fid));
            }
        }
        None
    }

    // ===== statements ===================================================

    fn exec_block(&mut self, stmts: &[Stmt]) -> Result<(), String> {
        for s in stmts {
            self.exec(s)?;
        }
        Ok(())
    }

    fn exec(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Let(name, e) => {
                let v = self.eval(e)?;
                self.define(name, v);
                Ok(())
            }
            Stmt::AssignVar(name, e) => {
                let v = self.eval(e)?;
                if self.scope_set(name, v.clone()) {
                    return Ok(());
                }
                // Not a known local: maybe a bare field (implied qualification).
                if let Some((cid, id, fid)) = self.implied_field(name) {
                    self.colls[cid].set(id, fid, v);
                    return self.check_monitors();
                }
                // Otherwise introduce a new local.
                self.define(name, v);
                Ok(())
            }
            Stmt::AssignRef(name, idx, rhs) => {
                let handle = self.eval(idx)?;
                let v = self.eval(rhs)?;
                // Writing a computed (derived) reference is a no-op: the
                // declaration says how to *produce* it, so the program's store
                // is simply ignored. This is what lets a stored field and a
                // computed function be swapped without touching the program.
                if self.computed.contains_key(name) {
                    return Ok(());
                }
                let (cid, id) = handle.as_member()?;
                let fid = self.colls[cid]
                    .field_id(name)
                    .ok_or_else(|| format!("`{}` is not a field of `{}`", name, self.colls[cid].name))?;
                self.colls[cid].set(id, fid, v);
                self.check_monitors()
            }
            Stmt::Print(args) => {
                let mut parts = Vec::with_capacity(args.len());
                for a in args {
                    parts.push(self.eval(a)?.display());
                }
                self.output.push(parts.join(" "));
                Ok(())
            }
            Stmt::ExprStmt(e) => {
                self.eval(e)?;
                Ok(())
            }
            Stmt::Delete(e) => {
                let (cid, id) = self.eval(e)?.as_member()?;
                self.colls[cid].delete(id)?;
                self.check_monitors()
            }
            Stmt::If(cond, then, els) => {
                if self.eval(cond)?.is_truthy() {
                    self.scoped(then)
                } else {
                    self.scoped(els)
                }
            }
            Stmt::While(cond, body) => {
                while self.eval(cond)?.is_truthy() {
                    self.scoped(body)?;
                }
                Ok(())
            }
            Stmt::Repeat(n, var, body) => {
                let count = self.eval(n)?.as_int()?;
                for i in 1..=count {
                    self.scopes.push(HashMap::new());
                    self.define(var, Value::Int(i));
                    let r = self.exec_block(body);
                    self.scopes.pop();
                    r?;
                }
                Ok(())
            }
            Stmt::ForEach { coll, cond, body } => {
                let cid = self.coll(coll)?;
                let members = self.colls[cid].members();
                for id in members {
                    self.iter_stack.push((cid, id));
                    self.scopes.push(HashMap::new());
                    let keep = match cond {
                        Some(c) => self.eval(c)?.is_truthy(),
                        None => true,
                    };
                    let r = if keep { self.exec_block(body) } else { Ok(()) };
                    self.scopes.pop();
                    self.iter_stack.pop();
                    r?;
                }
                Ok(())
            }
            Stmt::Generate { gen, coll, cond } => {
                let cid = self.coll(coll)?;
                self.gens.insert(
                    gen.clone(),
                    GenState {
                        coll: cid,
                        cond: cond.clone(),
                        next_pos: 1,
                        current: Value::Nil,
                    },
                );
                Ok(())
            }
            Stmt::Whenever(cond, body) => {
                let last = self.eval(cond)?.is_truthy();
                self.monitors.push(Monitor {
                    cond: cond.clone(),
                    body: body.clone(),
                    last,
                });
                Ok(())
            }
        }
    }

    fn scoped(&mut self, body: &[Stmt]) -> Result<(), String> {
        self.scopes.push(HashMap::new());
        let r = self.exec_block(body);
        self.scopes.pop();
        r
    }

    // ===== expressions ==================================================

    fn eval(&mut self, e: &Expr) -> Result<Value, String> {
        match e {
            Expr::Int(n) => Ok(Value::Int(*n)),
            Expr::Text(s) => Ok(Value::Text(s.clone())),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Nil => Ok(Value::Nil),
            Expr::It => {
                let &(cid, id) = self
                    .iter_stack
                    .last()
                    .ok_or_else(|| "`it` used outside an iteration".to_string())?;
                Ok(Value::Member(cid, id))
            }
            Expr::Var(name) => self.eval_var(name),
            Expr::Ref(name, idx) => {
                let arg = self.eval(idx)?;
                self.eval_ref(name, arg)
            }
            Expr::Size(coll) => {
                let cid = self.coll(coll)?;
                Ok(Value::Int(self.colls[cid].len() as i64))
            }
            Expr::MemberAt(coll, idx) => {
                let cid = self.coll(coll)?;
                let pos = self.eval(idx)?.as_int()?;
                let id = self.colls[cid].member_at(pos as usize)?;
                Ok(Value::Member(cid, id))
            }
            Expr::Insert(coll) => {
                let cid = self.coll(coll)?;
                let id = self.colls[cid].append();
                self.check_monitors()?;
                Ok(Value::Member(cid, id))
            }
            Expr::InsertAfter(coll, handle) => {
                let (hc, hid) = self.eval(handle)?.as_member()?;
                let cid = self.coll(coll)?;
                if hc != cid {
                    return Err("insert after: handle is from a different collection".into());
                }
                let id = self.colls[cid].insert_after(hid);
                self.check_monitors()?;
                Ok(Value::Member(cid, id))
            }
            Expr::Exists { var, coll, cond } => {
                let cid = self.coll(coll)?;
                let members = self.colls[cid].members();
                for id in members {
                    self.iter_stack.push((cid, id));
                    self.scopes.push(HashMap::new());
                    self.define(var, Value::Member(cid, id));
                    let hit = self.eval(cond)?.is_truthy();
                    self.scopes.pop();
                    self.iter_stack.pop();
                    if hit {
                        return Ok(Value::Bool(true));
                    }
                }
                Ok(Value::Bool(false))
            }
            Expr::Next(gen) => self.gen_next(gen),
            Expr::Unary(op, inner) => {
                let v = self.eval(inner)?;
                match op {
                    UnOp::Neg => Ok(Value::Int(-v.as_int()?)),
                    UnOp::Not => Ok(Value::Bool(!v.is_truthy())),
                }
            }
            Expr::Binary(op, l, r) => self.eval_binary(*op, l, r),
        }
    }

    fn eval_var(&mut self, name: &str) -> Result<Value, String> {
        if let Some(v) = self.scope_get(name) {
            return Ok(v);
        }
        if let Some(g) = self.gens.get(name) {
            return Ok(g.current.clone());
        }
        // Implied qualification: a bare field of the current member.
        if let Some((cid, id, fid)) = self.implied_field(name) {
            return Ok(self.colls[cid].get(id, fid));
        }
        Err(format!("undefined name `{}`", name))
    }

    /// The single canonical reference `name(handle)`.
    fn eval_ref(&mut self, name: &str, arg: Value) -> Result<Value, String> {
        // Computed function reference?
        if self.computed.contains_key(name) {
            return self.call_computed(name, arg);
        }
        // Otherwise a stored field, addressed through the handle's collection.
        let (cid, id) = arg.as_member().map_err(|_| {
            format!("`{}` is referenced as data/function but its argument is not a member handle", name)
        })?;
        let fid = self.colls[cid]
            .field_id(name)
            .ok_or_else(|| format!("`{}` is neither a field of `{}` nor a computed function", name, self.colls[cid].name))?;
        Ok(self.colls[cid].get(id, fid))
    }

    fn call_computed(&mut self, name: &str, arg: Value) -> Result<Value, String> {
        // Clone the small body so we can borrow self mutably while evaluating.
        let (param, body) = {
            let c = &self.computed[name];
            (c.param.clone(), c.body.clone())
        };
        // The parameter is a member handle; binding it and pushing its
        // collection as the current iteration makes implied qualification work
        // inside computed bodies (e.g. bare `radius` means `radius(param)`).
        self.scopes.push(HashMap::new());
        if let Value::Member(cid, id) = arg {
            self.iter_stack.push((cid, id));
            self.define(&param, arg);
            let r = self.eval(&body);
            self.iter_stack.pop();
            self.scopes.pop();
            r
        } else {
            self.define(&param, arg);
            let r = self.eval(&body);
            self.scopes.pop();
            r
        }
    }

    fn gen_next(&mut self, gen: &str) -> Result<Value, String> {
        let (cid, cond, start) = {
            let g = self
                .gens
                .get(gen)
                .ok_or_else(|| format!("`{}` is not a generator", gen))?;
            (g.coll, g.cond.clone(), g.next_pos)
        };
        let len = self.colls[cid].len();
        let mut pos = start;
        while pos <= len {
            let id = self.colls[cid].member_at(pos)?;
            self.iter_stack.push((cid, id));
            self.scopes.push(HashMap::new());
            let hit = self.eval(&cond)?.is_truthy();
            self.scopes.pop();
            self.iter_stack.pop();
            if hit {
                let found = Value::Member(cid, id);
                let g = self.gens.get_mut(gen).unwrap();
                g.next_pos = pos + 1;
                g.current = found.clone();
                return Ok(found);
            }
            pos += 1;
        }
        let g = self.gens.get_mut(gen).unwrap();
        g.next_pos = pos;
        g.current = Value::Nil;
        Ok(Value::Nil)
    }

    fn eval_binary(&mut self, op: BinOp, l: &Expr, r: &Expr) -> Result<Value, String> {
        match op {
            BinOp::And => {
                return Ok(Value::Bool(
                    self.eval(l)?.is_truthy() && self.eval(r)?.is_truthy(),
                ))
            }
            BinOp::Or => {
                return Ok(Value::Bool(
                    self.eval(l)?.is_truthy() || self.eval(r)?.is_truthy(),
                ))
            }
            _ => {}
        }
        let a = self.eval(l)?;
        let b = self.eval(r)?;
        match op {
            BinOp::Add => match (&a, &b) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x + y)),
                (Value::Text(x), Value::Text(y)) => Ok(Value::Text(format!("{}{}", x, y))),
                (Value::Text(x), other) => Ok(Value::Text(format!("{}{}", x, other.display()))),
                (other, Value::Text(y)) => Ok(Value::Text(format!("{}{}", other.display(), y))),
                _ => Err("`+` expects two ints or text".into()),
            },
            BinOp::Sub => Ok(Value::Int(a.as_int()? - b.as_int()?)),
            BinOp::Mul => Ok(Value::Int(a.as_int()? * b.as_int()?)),
            BinOp::Div => {
                let d = b.as_int()?;
                if d == 0 {
                    return Err("division by zero".into());
                }
                Ok(Value::Int(a.as_int()? / d))
            }
            BinOp::Mod => {
                let d = b.as_int()?;
                if d == 0 {
                    return Err("modulo by zero".into());
                }
                Ok(Value::Int(a.as_int()? % d))
            }
            BinOp::Eq => Ok(Value::Bool(values_eq(&a, &b))),
            BinOp::Ne => Ok(Value::Bool(!values_eq(&a, &b))),
            BinOp::Lt => Ok(Value::Bool(values_cmp(&a, &b)?.is_lt())),
            BinOp::Le => Ok(Value::Bool(values_cmp(&a, &b)?.is_le())),
            BinOp::Gt => Ok(Value::Bool(values_cmp(&a, &b)?.is_gt())),
            BinOp::Ge => Ok(Value::Bool(values_cmp(&a, &b)?.is_ge())),
            BinOp::And | BinOp::Or => unreachable!(),
        }
    }

    // ===== STATE monitors ==============================================

    fn check_monitors(&mut self) -> Result<(), String> {
        if self.in_monitor || self.monitors.is_empty() {
            return Ok(());
        }
        self.in_monitor = true;
        let mut result = Ok(());
        for i in 0..self.monitors.len() {
            let cond = self.monitors[i].cond.clone();
            let now = match self.eval(&cond) {
                Ok(v) => v.is_truthy(),
                Err(e) => {
                    result = Err(e);
                    break;
                }
            };
            let was = self.monitors[i].last;
            self.monitors[i].last = now;
            if now && !was {
                let body = self.monitors[i].body.clone();
                if let Err(e) = self.exec_block(&body) {
                    result = Err(e);
                    break;
                }
            }
        }
        self.in_monitor = false;
        result
    }
}

fn build_layouts(fields: &[FieldDecl]) -> Result<Vec<FieldLayout>, String> {
    let mut out = Vec::with_capacity(fields.len());
    for f in fields {
        let default = match &f.init {
            Some(e) => eval_const(e)?,
            None => match f.ty {
                Type::Int => Value::Int(0),
                Type::Text => Value::Text(String::new()),
                Type::Bool => Value::Bool(false),
            },
        };
        out.push(FieldLayout {
            name: f.name.clone(),
            default,
        });
    }
    Ok(out)
}

/// Evaluate a constant initializer (literals and arithmetic only).
fn eval_const(e: &Expr) -> Result<Value, String> {
    match e {
        Expr::Int(n) => Ok(Value::Int(*n)),
        Expr::Text(s) => Ok(Value::Text(s.clone())),
        Expr::Bool(b) => Ok(Value::Bool(*b)),
        Expr::Nil => Ok(Value::Nil),
        Expr::Unary(UnOp::Neg, inner) => Ok(Value::Int(-eval_const(inner)?.as_int()?)),
        Expr::Binary(op, l, r) => {
            let a = eval_const(l)?.as_int()?;
            let b = eval_const(r)?.as_int()?;
            let v = match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => a / b,
                BinOp::Mod => a % b,
                _ => return Err("only arithmetic is allowed in field initializers".into()),
            };
            Ok(Value::Int(v))
        }
        _ => Err("field initializers must be constants".into()),
    }
}
