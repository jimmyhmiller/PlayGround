//! The lambda-Tally type checker (v0.3): a quantitative, usage-context checker
//! with functions. Every binding has a multiplicity BUDGET; the checker counts
//! actual USAGE (a multiplicity) and requires `usage ⊑ budget`. At a call, an
//! argument's usage is SCALED by the parameter's budget (the rig `·`), so linear
//! capabilities are threaded across function boundaries:
//!
//!   * an `Own<S>` binding has budget 1 — it must be consumed exactly once
//!     (freed, moved, or returned); dropping or leaking it fails `0 ⋢ 1`, and
//!     re-using it fails `ω ⋢ 1`.
//!   * a copyable binding defaults to budget ω, but a parameter may be declared
//!     `1` (use exactly once) or `0` (erased — unused at runtime).
//!   * passing a value to a parameter of budget π uses it π times; passing a
//!     linear `Own` to a ω-parameter is rejected (it would alias it).
//!
//! This is the algorithmic reflection of QTT (`../../agda/Syntax.agda`): borrows
//! (field reads, `addr`, write bases) cost 0; only moves/consumes spend budget.

use crate::ast::{Expr, Func, Idx, Param, Program, Stmt, StructDecl, Ty};
use crate::mult::Mult;
use std::collections::{HashMap, HashSet};

// ---- index substitution & unification (for erased/implicit index params) ----

fn subst_idx(i: &Idx, s: &HashMap<String, Idx>) -> Idx {
    match i {
        Idx::Lit(k) => Idx::Lit(*k),
        Idx::Var(n, k) => match s.get(n) {
            Some(Idx::Lit(m)) => Idx::Lit(m + k),
            Some(Idx::Var(p, j)) => Idx::Var(p.clone(), j + k),
            None => Idx::Var(n.clone(), *k),
        },
    }
}

fn subst_ty(t: &Ty, s: &HashMap<String, Idx>) -> Ty {
    match t {
        Ty::Vec(i) => Ty::Vec(subst_idx(i, s)),
        other => other.clone(),
    }
}

/// Unify a parameter index against an argument index, solving implicit vars.
fn unify_idx(
    p: &Idx,
    a: &Idx,
    s: &mut HashMap<String, Idx>,
    implicit: &HashSet<String>,
) -> Result<(), String> {
    let pr = subst_idx(p, s);
    if let Idx::Var(n, k) = &pr {
        if implicit.contains(n) {
            return match a {
                Idx::Lit(m) if m >= k => {
                    s.insert(n.clone(), Idx::Lit(m - k));
                    Ok(())
                }
                Idx::Var(q, j) if j >= k => {
                    s.insert(n.clone(), Idx::Var(q.clone(), j - k));
                    Ok(())
                }
                _ => Err(format!("length mismatch: cannot solve {p} = {a}")),
            };
        }
    }
    if &pr == a {
        Ok(())
    } else {
        Err(format!("length mismatch: expected {pr}, found {a}"))
    }
}

fn unify(
    p: &Ty,
    a: &Ty,
    s: &mut HashMap<String, Idx>,
    implicit: &HashSet<String>,
) -> Result<(), String> {
    match (p, a) {
        (Ty::Vec(pi), Ty::Vec(ai)) => unify_idx(pi, ai, s, implicit),
        (Ty::Own(x), Ty::Own(y)) if x == y => Ok(()),
        (Ty::Ptr(x), Ty::Ptr(y)) if x == y => Ok(()),
        (Ty::Lst(x), Ty::Lst(y)) if x == y => Ok(()),
        (Ty::Cursor(x), Ty::Cursor(y)) if x == y => Ok(()),
        (Ty::Pair(p1, p2), Ty::Pair(a1, a2)) => {
            unify(p1, a1, s, implicit)?;
            unify(p2, a2, s, implicit)
        }
        (Ty::Int, Ty::Int) | (Ty::Unit, Ty::Unit) | (Ty::Nat, Ty::Nat) => Ok(()),
        _ => Err(format!("type mismatch: expected {p}, found {a}")),
    }
}

struct Slot {
    ty: Ty,
    budget: Mult,
    alive: bool,  // false once consumed by a move/free (for ordering diagnostics)
    usage: Mult,  // accumulated consuming uses, checked against budget at scope end
}

type Sig = (Vec<(Mult, String, Ty)>, Ty); // (param budget+name+type, return type)

struct Checker {
    structs: HashMap<String, StructDecl>,
    fns: HashMap<String, Sig>,
    env: HashMap<String, Slot>,
    errs: Vec<String>,
    tag: u32, // fresh type-level list-identity tags
}

fn budget_of(mult: Option<Mult>, ty: &Ty, errs: &mut Vec<String>, what: &str) -> Mult {
    match mult {
        Some(m) => {
            if ty.is_linear() && m != Mult::One {
                errs.push(format!(
                    "{what}: a linear `{ty}` must have multiplicity 1, not {m}"
                ));
                Mult::One
            } else {
                m
            }
        }
        None => {
            if ty.is_linear() {
                Mult::One
            } else {
                Mult::Omega
            }
        }
    }
}

impl Checker {
    fn err(&mut self, m: impl Into<String>) {
        self.errs.push(m.into());
    }

    fn fresh_tag(&mut self) -> String {
        let t = format!("$L{}", self.tag);
        self.tag += 1;
        t
    }

    fn field_ty(&mut self, sname: &str, f: &str) -> Ty {
        let res = match self.structs.get(sname) {
            None => Err(format!("unknown struct `{sname}`")),
            Some(sd) => match sd.fields.iter().find(|(n, _)| n == f) {
                Some((_, t)) => Ok(t.clone()),
                None => Err(format!("struct `{sname}` has no field `{f}`")),
            },
        };
        match res {
            Ok(t) => t,
            Err(m) => {
                self.err(m);
                Ty::Unit
            }
        }
    }

    /// Borrow a variable (read its type without spending budget).
    fn borrow(&mut self, x: &str) -> Ty {
        match self.env.get(x).map(|s| (s.ty.clone(), s.alive)) {
            None => {
                self.err(format!("`{x}`: unbound"));
                Ty::Unit
            }
            Some((t, alive)) => {
                if !alive {
                    self.err(format!("`{x}`: borrow after move/free"));
                }
                t
            }
        }
    }

    /// Use a variable as a value at multiplicity `m` (a move when linear).
    fn use_at(&mut self, x: &str, m: Mult) -> Ty {
        match self.env.get(x).map(|s| (s.ty.clone(), s.alive)) {
            None => {
                self.err(format!("`{x}`: unbound"));
                Ty::Unit
            }
            Some((t, alive)) => {
                if !alive {
                    if m != Mult::Zero {
                        self.err(format!("`{x}`: use after move/free"));
                    }
                    return t;
                }
                if t.is_linear() && m == Mult::Omega {
                    self.err(format!(
                        "`{x}` : {t} is linear but used at multiplicity ω (it cannot be aliased)"
                    ));
                }
                let s = self.env.get_mut(x).unwrap();
                s.usage = s.usage.add(m);
                if t.is_linear() && m != Mult::Zero {
                    s.alive = false;
                }
                t
            }
        }
    }

    /// The base of a read/write/addr borrows (when a var) instead of consuming.
    fn check_base(&mut self, e: &Expr) -> Ty {
        match e {
            Expr::Var(x) => self.borrow(x),
            _ => self.check_expr(e, None, Mult::Zero),
        }
    }

    /// Bidirectional, usage-aware: `pi` is the multiplicity at which the context
    /// consumes this expression's result.
    fn check_expr(&mut self, e: &Expr, expected: Option<&Ty>, pi: Mult) -> Ty {
        let got = match e {
            Expr::Int(_) => Ty::Int,
            Expr::Unit => Ty::Unit,
            Expr::Null => match expected {
                Some(Ty::Ptr(s)) => Ty::Ptr(s.clone()),
                _ => {
                    self.err("cannot infer the type of `null` here (use it in a Ptr<S> position)");
                    expected.cloned().unwrap_or(Ty::Unit)
                }
            },
            Expr::Var(x) => self.use_at(x, pi),
            Expr::AddrOf(x) => match self.borrow(x) {
                Ty::Own(s) => Ty::Ptr(s),
                t => {
                    self.err(format!("addr({x}): expects an owned value, but `{x}` : {t}"));
                    Ty::Unit
                }
            },
            Expr::Alloc(name, fields) => self.check_alloc(name, fields, pi),
            Expr::Field(obj, f) => match self.check_base(obj) {
                Ty::Own(s) => self.field_ty(&s, f),
                Ty::Ptr(s) => {
                    self.err(format!(
                        "no capability: the base is a bare `Ptr<{s}>`; you hold no Perm for it, \
                         so it cannot be dereferenced"
                    ));
                    self.field_ty(&s, f)
                }
                t => {
                    self.err(format!("`.{f}`: base has non-record type {t}"));
                    Ty::Unit
                }
            },
            Expr::Call(fname, args) => self.check_call(fname, args, pi),
        };
        if let Some(exp) = expected {
            if &got != exp && !matches!(e, Expr::Null) {
                self.err(format!("type mismatch: expected {exp}, found {got}"));
            }
        }
        got
    }

    fn check_alloc(&mut self, name: &str, fields: &[(String, Expr)], pi: Mult) -> Ty {
        let decl = match self.structs.get(name).cloned() {
            Some(d) => d,
            None => {
                self.err(format!("alloc of unknown struct `{name}`"));
                return Ty::Unit;
            }
        };
        for (fname, fty) in &decl.fields {
            match fields.iter().find(|(n, _)| n == fname) {
                Some((_, fe)) => {
                    self.check_expr(fe, Some(fty), pi);
                }
                None => self.err(format!("alloc {name}: missing field `{fname}`")),
            }
        }
        for (fname, _) in fields {
            if !decl.fields.iter().any(|(n, _)| n == fname) {
                self.err(format!("alloc {name}: unknown field `{fname}`"));
            }
        }
        Ty::Own(name.to_string())
    }

    /// The dependent, length-indexed `Vec<n>` built-ins. The length arithmetic
    /// (`n`, `n+1`, `0`) is the type-level dependency; `vhead`/`vtail` require a
    /// statically non-empty vector, so pop-from-empty is a *type* error.
    fn check_builtin(&mut self, fname: &str, args: &[Expr], pi: Mult) -> Option<Ty> {
        let as_vec = |c: &mut Self, t: Ty, op: &str| -> Option<Idx> {
            match t {
                Ty::Vec(i) => Some(i),
                other => {
                    c.err(format!("{op}: expected a Vec, found {other}"));
                    None
                }
            }
        };
        match fname {
            "vnew" => {
                if !args.is_empty() {
                    self.err("vnew expects no arguments");
                }
                Some(Ty::Vec(Idx::Lit(0)))
            }
            "vpush" => {
                if args.len() != 2 {
                    self.err("vpush(v, x) expects 2 arguments");
                    return Some(Ty::Vec(Idx::Lit(0)));
                }
                let vt = self.check_expr(&args[0], None, pi); // consume the vector
                self.check_expr(&args[1], Some(&Ty::Int), pi); // the element
                match as_vec(self, vt, "vpush") {
                    Some(i) => Some(Ty::Vec(i.succ())),
                    None => Some(Ty::Vec(Idx::Lit(0))),
                }
            }
            "vhead" => {
                if args.len() != 1 {
                    self.err("vhead(v) expects 1 argument");
                    return Some(Ty::Int);
                }
                let vt = self.check_base(&args[0]); // borrow (read), do not consume
                if let Some(i) = as_vec(self, vt, "vhead") {
                    if !i.nonempty() {
                        self.err(format!(
                            "vhead: the vector may be empty (its length is {i}); need a Vec<n+1>"
                        ));
                    }
                }
                Some(Ty::Int)
            }
            "vtail" => {
                if args.len() != 1 {
                    self.err("vtail(v) expects 1 argument");
                    return Some(Ty::Vec(Idx::Lit(0)));
                }
                let vt = self.check_expr(&args[0], None, pi); // consume
                match as_vec(self, vt, "vtail") {
                    Some(i) if i.nonempty() => Some(Ty::Vec(i.pred())),
                    Some(i) => {
                        self.err(format!(
                            "vtail: the vector may be empty (its length is {i}); need a Vec<n+1>"
                        ));
                        Some(Ty::Vec(i))
                    }
                    None => Some(Ty::Vec(Idx::Lit(0))),
                }
            }
            "vfree" => {
                if args.len() != 1 {
                    self.err("vfree(v) expects 1 argument");
                    return Some(Ty::Unit);
                }
                let vt = self.check_expr(&args[0], None, pi); // consume
                if let Some(i) = as_vec(self, vt, "vfree") {
                    if !i.is_zero() {
                        self.err(format!(
                            "vfree requires an empty Vec<0>, but this vector has length {i}"
                        ));
                    }
                }
                Some(Ty::Unit)
            }

            // ---- linear-cursor list (the Vale `LinearKey` model) -------------
            // a cursor is LINEAR, so `lremove` consumes it: no double-remove, no
            // use-after-remove; the list tag stops cursors crossing lists.
            "lnew" => {
                if !args.is_empty() {
                    self.err("lnew expects no arguments");
                }
                Some(Ty::Lst(self.fresh_tag()))
            }
            "linsert" => {
                if args.len() != 2 {
                    self.err("linsert(l, x) expects 2 arguments");
                    return Some(Ty::Unit);
                }
                let lt = self.check_expr(&args[0], None, pi); // consume the list
                self.check_expr(&args[1], Some(&Ty::Int), pi);
                match lt {
                    Ty::Lst(tag) => Some(Ty::Pair(
                        Box::new(Ty::Cursor(tag.clone())),
                        Box::new(Ty::Lst(tag)),
                    )),
                    other => {
                        self.err(format!("linsert: expected a list, found {other}"));
                        Some(Ty::Unit)
                    }
                }
            }
            "lremove" => {
                if args.len() != 2 {
                    self.err("lremove(l, c) expects 2 arguments");
                    return Some(Ty::Unit);
                }
                let lt = self.check_expr(&args[0], None, pi); // consume the list
                let ct = self.check_expr(&args[1], None, pi); // consume the cursor
                let ltag = match lt {
                    Ty::Lst(t) => Some(t),
                    other => {
                        self.err(format!("lremove: first argument is not a list ({other})"));
                        None
                    }
                };
                let ctag = match ct {
                    Ty::Cursor(t) => Some(t),
                    other => {
                        self.err(format!("lremove: second argument is not a cursor ({other})"));
                        None
                    }
                };
                if let (Some(l), Some(c)) = (&ltag, &ctag) {
                    if l != c {
                        self.err(format!(
                            "lremove: cursor belongs to a different list (cursor {c}, list {l})"
                        ));
                    }
                }
                let rtag = match ltag {
                    Some(t) => t,
                    None => self.fresh_tag(),
                };
                Some(Ty::Pair(Box::new(Ty::Int), Box::new(Ty::Lst(rtag))))
            }
            "lfree" => {
                if args.len() != 1 {
                    self.err("lfree(l) expects 1 argument");
                    return Some(Ty::Unit);
                }
                let lt = self.check_expr(&args[0], None, pi); // consume the list
                match lt {
                    Ty::Lst(_) => {}
                    other => self.err(format!("lfree: expected a list, found {other}")),
                }
                Some(Ty::Unit)
            }
            _ => None,
        }
    }

    fn check_call(&mut self, fname: &str, args: &[Expr], pi: Mult) -> Ty {
        if let Some(t) = self.check_builtin(fname, args, pi) {
            return t;
        }
        let sig = match self.fns.get(fname).cloned() {
            Some(s) => s,
            None => {
                self.err(format!("call to unknown function `{fname}`"));
                return Ty::Unit;
            }
        };
        let (params, ret) = sig;
        // multiplicity-0 parameters are IMPLICIT (erased): not passed at the
        // call site, but solved by unifying the explicit argument types.
        let implicit: HashSet<String> = params
            .iter()
            .filter(|(m, _, _)| *m == Mult::Zero)
            .map(|(_, n, _)| n.clone())
            .collect();
        let explicit: Vec<&(Mult, String, Ty)> =
            params.iter().filter(|(m, _, _)| *m != Mult::Zero).collect();
        if args.len() != explicit.len() {
            self.err(format!(
                "`{fname}` expects {} explicit argument(s), got {}",
                explicit.len(),
                args.len()
            ));
        }
        let mut subst: HashMap<String, Idx> = HashMap::new();
        for (arg, (budget, _name, pty)) in args.iter().zip(explicit.iter()) {
            // the argument is consumed at  pi · (parameter budget)
            let at = self.check_expr(arg, None, pi.mul(*budget));
            if let Err(m) = unify(pty, &at, &mut subst, &implicit) {
                self.err(format!("`{fname}`: argument {m}"));
            }
        }
        subst_ty(&ret, &subst)
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let(name, ann, rhs) => {
                if let Some((lty, leak)) = self
                    .env
                    .get(name)
                    .map(|s| (s.ty.clone(), s.ty.is_linear() && s.alive && s.usage == Mult::Zero))
                {
                    if leak {
                        self.err(format!(
                            "let {name}: rebinding drops the live linear `{name}` : {lty} (leak)"
                        ));
                    }
                }
                let ty = self.check_expr(rhs, ann.as_ref(), Mult::One);
                let ty = ann.clone().unwrap_or(ty);
                let budget = if ty.is_linear() {
                    Mult::One
                } else {
                    Mult::Omega
                };
                self.env.insert(
                    name.clone(),
                    Slot {
                        ty,
                        budget,
                        alive: true,
                        usage: Mult::Zero,
                    },
                );
            }
            Stmt::LetPair(x, y, rhs) => {
                let t = self.check_expr(rhs, None, Mult::One);
                let (t1, t2) = match t {
                    Ty::Pair(a, b) => (*a, *b),
                    other => {
                        self.err(format!("let ({x}, {y}): expected a pair, found {other}"));
                        (Ty::Unit, Ty::Unit)
                    }
                };
                for (name, ty) in [(x, t1), (y, t2)] {
                    let budget = if ty.is_linear() { Mult::One } else { Mult::Omega };
                    self.env.insert(
                        name.clone(),
                        Slot {
                            ty,
                            budget,
                            alive: true,
                            usage: Mult::Zero,
                        },
                    );
                }
            }
            Stmt::Write(base, fld, rhs) => match self.check_base(base) {
                Ty::Own(s) => {
                    let ft = self.field_ty(&s, fld);
                    self.check_expr(rhs, Some(&ft), Mult::One);
                }
                Ty::Ptr(s) => self.err(format!(
                    "cannot write through a bare `Ptr<{s}>` (no capability held)"
                )),
                t => self.err(format!("write: base has non-record type {t}")),
            },
            Stmt::Free(name) => {
                match self
                    .env
                    .get(name)
                    .map(|s| (s.ty.clone(), s.alive, s.ty.is_linear()))
                {
                    None => self.err(format!("free {name}: unbound")),
                    Some((_, false, _)) => {
                        self.err(format!("free {name}: double free / use after free"))
                    }
                    Some((_, true, true)) => {
                        let s = self.env.get_mut(name).unwrap();
                        s.alive = false;
                        s.usage = s.usage.add(Mult::One);
                    }
                    Some((t, true, false)) => self
                        .err(format!("free {name}: `{name}` : {t} is not an owned capability")),
                }
            }
            Stmt::Expr(e) => {
                let t = self.check_expr(e, None, Mult::One);
                if t.is_linear() {
                    self.err(format!(
                        "a linear value of type {t} is computed and dropped (must be consumed)"
                    ));
                }
            }
        }
    }

    fn check_fn(&mut self, f: &Func) {
        self.env.clear();
        for Param { mult, name, ty } in &f.params {
            let budget = budget_of(*mult, ty, &mut self.errs, &format!("fn {} param `{name}`", f.name));
            self.env.insert(
                name.clone(),
                Slot {
                    ty: ty.clone(),
                    budget,
                    alive: true,
                    usage: Mult::Zero,
                },
            );
        }
        for s in &f.body {
            self.stmt(s);
        }
        match &f.tail {
            Some(e) => {
                self.check_expr(e, Some(&f.ret), Mult::One);
            }
            None => {
                if f.ret != Ty::Unit {
                    self.err(format!(
                        "fn {}: missing return value of type {}",
                        f.name, f.ret
                    ));
                }
            }
        }
        // end-of-scope: every binding must satisfy  usage ⊑ budget
        let mut slots: Vec<_> = self
            .env
            .iter()
            .map(|(k, s)| (k.clone(), s.ty.clone(), s.budget, s.usage))
            .collect();
        slots.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, ty, budget, usage) in slots {
            if usage.leq(budget) {
                continue;
            }
            let msg = match (budget, usage) {
                (Mult::One, Mult::Zero) => format!(
                    "fn {}: `{name}` : {ty} (linear, budget 1) is never consumed (leak)",
                    f.name
                ),
                (Mult::One, _) => format!(
                    "fn {}: `{name}` : {ty} is used more than once (budget 1, used {usage})",
                    f.name
                ),
                (Mult::Zero, _) => format!(
                    "fn {}: `{name}` : {ty} is erased (budget 0) but used at runtime",
                    f.name
                ),
                _ => format!(
                    "fn {}: `{name}` : {ty} usage {usage} exceeds budget {budget}",
                    f.name
                ),
            };
            self.err(msg);
        }
    }
}

/// Check a program; return diagnostics (empty == accepted).
pub fn check(prog: &Program) -> Vec<String> {
    let mut structs = HashMap::new();
    let mut errs = Vec::new();
    for sd in &prog.structs {
        if structs.insert(sd.name.clone(), sd.clone()).is_some() {
            errs.push(format!("duplicate struct `{}`", sd.name));
        }
    }
    for sd in &prog.structs {
        for (f, t) in &sd.fields {
            if t.is_linear() {
                errs.push(format!(
                    "struct `{}`: field `{f}` has linear type {t}; fields must be copyable \
                     (store a Ptr<S> instead of an Own<S>)",
                    sd.name
                ));
            }
            if let Ty::Ptr(s) | Ty::Own(s) = t {
                if !structs.contains_key(s) {
                    errs.push(format!(
                        "struct `{}`: field `{f}` references unknown struct `{s}`",
                        sd.name
                    ));
                }
            }
        }
    }

    // collect function signatures first (so calls and recursion resolve)
    let mut fns: HashMap<String, Sig> = HashMap::new();
    for f in &prog.funcs {
        let params = f
            .params
            .iter()
            .map(|p| (budget_of(p.mult, &p.ty, &mut errs, "param"), p.name.clone(), p.ty.clone()))
            .collect();
        if fns.insert(f.name.clone(), (params, f.ret.clone())).is_some() {
            errs.push(format!("duplicate function `{}`", f.name));
        }
    }
    if !fns.contains_key("main") {
        errs.push("program has no `fn main`".to_string());
    }

    let mut c = Checker {
        structs,
        fns,
        env: HashMap::new(),
        errs,
        tag: 0,
    };
    for f in &prog.funcs {
        c.check_fn(f);
    }
    c.errs
}

#[cfg(test)]
mod tests {
    use super::check;
    use crate::parser::parse;

    fn errs(src: &str) -> Vec<String> {
        check(&parse(src).expect("parse"))
    }
    fn accept(src: &str) {
        let e = errs(src);
        assert!(e.is_empty(), "expected accept, got {e:?}\n{src}");
    }
    fn reject(src: &str) {
        assert!(!errs(src).is_empty(), "expected reject:\n{src}");
    }

    const C: &str = "struct C { val: Int }";

    #[test]
    fn good() {
        accept(&format!("{C} fn main() -> Int {{ let a = alloc C {{ val: 41 }}; a.val = 42; let r = a.val; free a; r }}"));
        // a function that CONSUMES a linear capability, threaded across the call
        accept(&format!(
            "{C}
             fn consume(c: Own<C>) -> Int {{ let r = c.val; free c; r }}
             fn main() -> Int {{ let a = alloc C {{ val: 7 }}; consume(a) }}"
        ));
        // a factory that RETURNS a linear capability (ownership out)
        accept(&format!(
            "{C}
             fn make(n: Int) -> Own<C> {{ alloc C {{ val: n }} }}
             fn main() -> Int {{ let a = make(5); let r = a.val; free a; r }}"
        ));
        // copyable param used many times (default budget ω)
        accept(&format!("{C} fn dup(w n: Int) -> Int {{ n }} fn main() -> Int {{ dup(3) }}"));
        // explicit linear use of a copyable param: used exactly once
        accept(&format!("{C} fn once(1 n: Int) -> Int {{ n }} fn main() -> Int {{ once(3) }}"));
    }

    #[test]
    fn bad() {
        // a function that takes an Own and neither frees nor returns it -> leak
        reject(&format!(
            "{C} fn drop_it(c: Own<C>) -> Int {{ 0 }} fn main() -> Int {{ let a = alloc C {{ val: 1 }}; drop_it(a) }}"
        ));
        // passing the same Own twice (use after move)
        reject(&format!(
            "{C} fn two(a: Own<C>, b: Own<C>) -> Int {{ free a; free b; 0 }}
             fn main() -> Int {{ let x = alloc C {{ val: 1 }}; two(x, x) }}"
        ));
        // an Own parameter cannot be declared ω
        reject(&format!("{C} fn bad(w c: Own<C>) -> Int {{ free c; 0 }} fn main() -> Int {{ 0 }}"));
        // a `1` (linear) copyable param used twice
        reject(&format!("{C} fn lin(1 n: Int) -> Int {{ add(n, n) }} fn add(w a: Int, w b: Int) -> Int {{ a }} fn main() -> Int {{ lin(2) }}"));
        // a `1` copyable param never used
        reject(&format!("{C} fn lin(1 n: Int) -> Int {{ 0 }} fn main() -> Int {{ lin(2) }}"));
        // wrong arity / unknown function / wrong arg type
        reject(&format!("{C} fn f(a: Int) -> Int {{ a }} fn main() -> Int {{ f(1, 2) }}"));
        reject(&format!("{C} fn main() -> Int {{ nope(1) }}"));
        reject(&format!("{C} fn main() -> Int {{ let a = alloc C {{ val: 1 }}; a }}")); // returns Own where Int expected, and... a:Own escapes
        // still catches the basics
        reject(&format!("{C} fn main() -> Int {{ let a = alloc C {{ val: 1 }}; free a; free a; 0 }}"));
        reject(&format!("{C} fn main() -> Int {{ let a = alloc C {{ val: 1 }}; 0 }}")); // leak
    }

    #[test]
    fn dependent() {
        // length-indexed vector: the type tracks the length, which is erased.
        accept(
            "fn main() -> Int {
               let v0 = vnew();
               let v1 = vpush(v0, 10);
               let v2 = vpush(v1, 20);
               let top = vhead(v2);
               let v1b = vtail(v2);
               let v0b = vtail(v1b);
               vfree(v0b);
               top
             }",
        );
        // a dependent function over an ERASED length index (the 0-fragment):
        // n is implicit, solved by unifying Vec<n> with the argument.
        accept(
            "fn push0(0 n: Nat, v: Vec<n>) -> Vec<n+1> { vpush(v, 0) }
             fn main() -> Int {
               let a = vnew();
               let b = push0(a);
               let c = vtail(b);
               vfree(c);
               0
             }",
        );
        accept(
            "fn pop2(0 n: Nat, v: Vec<n+2>) -> Vec<n> { vtail(vtail(v)) }
             fn main() -> Int {
               let a = vpush(vpush(vnew(), 1), 2);
               let b = pop2(a);
               vfree(b);
               0
             }",
        );

        // pop-from-empty is a TYPE error
        reject("fn main() -> Int { let v = vnew(); let w = vtail(v); vfree(w); 0 }");
        reject("fn main() -> Int { let v = vnew(); let x = vhead(v); vfree(v); x }");
        // free a non-empty vector
        reject("fn main() -> Int { let v = vpush(vnew(), 1); vfree(v); 0 }");
        // leak a vector
        reject("fn main() -> Int { let v = vnew(); 0 }");
        // cannot prove a Vec<n> (abstract, possibly empty) is non-empty
        reject("fn bad(0 n: Nat, v: Vec<n>) -> Vec<n> { vtail(v) } fn main() -> Int { 0 }");
        // an erased index used at runtime
        reject("fn bad(0 n: Nat) -> Int { n } fn main() -> Int { 0 }");
    }

    #[test]
    fn linear_cursors() {
        // O(1) remove-by-handle: insert three, remove the MIDDLE by its cursor,
        // then account for the rest. The cursor is linear, so this is sound.
        accept(
            "fn main() -> Int {
               let l0 = lnew();
               let (c1, l1) = linsert(l0, 10);
               let (c2, l2) = linsert(l1, 20);
               let (c3, l3) = linsert(l2, 30);
               let (x, l4) = lremove(l3, c2);
               let (y, l5) = lremove(l4, c1);
               let (z, l6) = lremove(l5, c3);
               lfree(l6);
               x
             }",
        );

        // double-remove: c is consumed by the first remove
        reject(
            "fn main() -> Int {
               let l0 = lnew();
               let (c, l1) = linsert(l0, 1);
               let (x, l2) = lremove(l1, c);
               let (y, l3) = lremove(l2, c);
               lfree(l3);
               x
             }",
        );
        // forget to remove a cursor -> the linear cursor leaks
        reject(
            "fn main() -> Int {
               let l0 = lnew();
               let (c, l1) = linsert(l0, 1);
               lfree(l1);
               0
             }",
        );
        // cross-list: a cursor from list `a` used to remove from list `b`
        reject(
            "fn main() -> Int {
               let a0 = lnew();
               let b0 = lnew();
               let (ca, a1) = linsert(a0, 1);
               let (cb, b1) = linsert(b0, 2);
               let (x, b2) = lremove(b1, ca);
               let (y, a2) = lremove(a1, cb);
               lfree(a2); lfree(b2);
               x
             }",
        );
        // leak the list itself
        reject("fn main() -> Int { let l = lnew(); 0 }");
    }
}
