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

use crate::ast::{Expr, Func, Param, Program, Stmt, StructDecl, Ty};
use crate::mult::Mult;
use std::collections::HashMap;

struct Slot {
    ty: Ty,
    budget: Mult,
    alive: bool,  // false once consumed by a move/free (for ordering diagnostics)
    usage: Mult,  // accumulated consuming uses, checked against budget at scope end
}

type Sig = (Vec<(Mult, Ty)>, Ty); // (param budgets+types, return type)

struct Checker {
    structs: HashMap<String, StructDecl>,
    fns: HashMap<String, Sig>,
    env: HashMap<String, Slot>,
    errs: Vec<String>,
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

    fn check_call(&mut self, fname: &str, args: &[Expr], pi: Mult) -> Ty {
        let sig = match self.fns.get(fname).cloned() {
            Some(s) => s,
            None => {
                self.err(format!("call to unknown function `{fname}`"));
                return Ty::Unit;
            }
        };
        let (params, ret) = sig;
        if args.len() != params.len() {
            self.err(format!(
                "`{fname}` expects {} argument(s), got {}",
                params.len(),
                args.len()
            ));
        }
        for (arg, (budget, pty)) in args.iter().zip(params.iter()) {
            // the argument is consumed at  pi · (parameter budget)
            self.check_expr(arg, Some(pty), pi.mul(*budget));
        }
        ret
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
            .map(|p| (budget_of(p.mult, &p.ty, &mut errs, "param"), p.ty.clone()))
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
}
