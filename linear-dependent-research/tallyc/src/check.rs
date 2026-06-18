//! The lambda-Tally type checker (v0.2): a type-directed, structurally-linear
//! discipline. The L3 split lives in the types (`Own<S>` linear vs `Ptr<S>`
//! copyable), and memory safety (use-after-free / double-free / leak) falls out
//! of linearity. The discipline matches `../../agda/CombinedSound.agda`.
//!
//! Typing rules, informally:
//!   * struct fields must be COPYABLE (never `Own`): no permission escapes into
//!     the heap, so reading a field can never hand you a capability.
//!   * `alloc S {..}` : `Own<S>` (a fresh linear capability).
//!   * `addr(x)` borrows `x : Own<S>` and yields a copyable `Ptr<S>`.
//!   * `e.f` requires the base to be `Own<S>` (you hold the Perm); a `Ptr<S>`
//!     base is rejected — you cannot dereference a bare address.
//!   * an `Own` value must be used exactly once (free or move); dropping or
//!     re-using it is an error, and an `Own` still live at end of scope leaks.

use crate::ast::{Expr, Program, Stmt, StructDecl, Ty};
use std::collections::HashMap;

struct Slot {
    ty: Ty,
    consumed: bool,
}

struct Checker<'a> {
    structs: HashMap<String, &'a StructDecl>,
    env: HashMap<String, Slot>,
    errs: Vec<String>,
}

impl<'a> Checker<'a> {
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

    /// Borrow a variable (read its type without consuming it).
    fn borrow_var(&mut self, x: &str) -> Ty {
        match self.env.get(x).map(|s| (s.ty.clone(), s.consumed)) {
            None => {
                self.err(format!("`{x}`: unbound"));
                Ty::Unit
            }
            Some((t, consumed)) => {
                if consumed {
                    self.err(format!("`{x}`: use after move/free"));
                }
                t
            }
        }
    }

    /// Use a variable as a value: consume it if it is linear (a move).
    fn use_var(&mut self, x: &str) -> Ty {
        match self.env.get(x).map(|s| (s.ty.clone(), s.consumed)) {
            None => {
                self.err(format!("`{x}`: unbound"));
                Ty::Unit
            }
            Some((t, true)) => {
                self.err(format!("`{x}`: use after move/free"));
                t
            }
            Some((t, false)) => {
                if t.is_linear() {
                    self.env.get_mut(x).unwrap().consumed = true;
                }
                t
            }
        }
    }

    /// The base of a read/write/addr borrows (never consumes) when it is a var.
    fn check_base(&mut self, e: &Expr) -> Ty {
        match e {
            Expr::Var(x) => self.borrow_var(x),
            _ => self.check_expr(e, None),
        }
    }

    /// Bidirectional: `expected = Some(t)` checks against `t`, else synthesises.
    fn check_expr(&mut self, e: &Expr, expected: Option<&Ty>) -> Ty {
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
            Expr::Var(x) => self.use_var(x),
            Expr::AddrOf(x) => match self.borrow_var(x) {
                Ty::Own(s) => Ty::Ptr(s),
                t => {
                    self.err(format!(
                        "addr({x}): expects an owned value, but `{x}` : {t}"
                    ));
                    Ty::Unit
                }
            },
            Expr::Alloc(name, fields) => self.check_alloc(name, fields),
            Expr::Field(obj, f) => match self.check_base(obj) {
                Ty::Own(s) => self.field_ty(&s, f),
                Ty::Ptr(s) => {
                    self.err(format!(
                        "no capability: the base is a bare `Ptr<{s}>`; you hold no Perm \
                         for it, so it cannot be dereferenced"
                    ));
                    self.field_ty(&s, f)
                }
                t => {
                    self.err(format!("`.{f}`: base has non-record type {t}"));
                    Ty::Unit
                }
            },
        };
        if let Some(exp) = expected {
            if &got != exp && !matches!(e, Expr::Null) {
                self.err(format!("type mismatch: expected {exp}, found {got}"));
            }
        }
        got
    }

    fn check_alloc(&mut self, name: &str, fields: &[(String, Expr)]) -> Ty {
        let decl = match self.structs.get(name).map(|d| (*d).clone()) {
            Some(d) => d,
            None => {
                self.err(format!("alloc of unknown struct `{name}`"));
                return Ty::Unit;
            }
        };
        for (fname, fty) in &decl.fields {
            match fields.iter().find(|(n, _)| n == fname) {
                Some((_, fe)) => {
                    self.check_expr(fe, Some(fty));
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

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let(name, ann, rhs) => {
                if let Some((lty, leak)) = self
                    .env
                    .get(name)
                    .map(|s| (s.ty.clone(), s.ty.is_linear() && !s.consumed))
                {
                    if leak {
                        self.err(format!(
                            "let {name}: rebinding drops the live linear `{name}` : {lty} (leak)"
                        ));
                    }
                }
                let ty = self.check_expr(rhs, ann.as_ref());
                let ty = ann.clone().unwrap_or(ty);
                self.env.insert(name.clone(), Slot { ty, consumed: false });
            }
            Stmt::Write(base, fld, rhs) => match self.check_base(base) {
                Ty::Own(s) => {
                    let ft = self.field_ty(&s, fld);
                    self.check_expr(rhs, Some(&ft));
                }
                Ty::Ptr(s) => self.err(format!(
                    "cannot write through a bare `Ptr<{s}>` (no capability held)"
                )),
                t => self.err(format!("write: base has non-record type {t}")),
            },
            Stmt::Free(name) => {
                match self.env.get(name).map(|s| (s.ty.clone(), s.consumed)) {
                    None => self.err(format!("free {name}: unbound")),
                    Some((_, true)) => {
                        self.err(format!("free {name}: double free / use after free"))
                    }
                    Some((t, false)) if t.is_linear() => {
                        self.env.get_mut(name).unwrap().consumed = true;
                    }
                    Some((t, false)) => self
                        .err(format!("free {name}: `{name}` : {t} is not an owned capability")),
                }
            }
            Stmt::Expr(e) => {
                let t = self.check_expr(e, None);
                if t.is_linear() {
                    self.err(format!(
                        "a linear value of type {t} is computed and dropped (must be consumed)"
                    ));
                }
            }
        }
    }
}

/// Check a program; return diagnostics (empty == accepted).
pub fn check(prog: &Program) -> Vec<String> {
    let mut structs = HashMap::new();
    let mut errs = Vec::new();
    // register structs; field types must be copyable and reference known structs
    for sd in &prog.structs {
        if structs.insert(sd.name.clone(), sd).is_some() {
            errs.push(format!("duplicate struct `{}`", sd.name));
        }
    }
    for sd in &prog.structs {
        for (f, t) in &sd.fields {
            if t.is_linear() {
                errs.push(format!(
                    "struct `{}`: field `{f}` has linear type {t}; fields must be \
                     copyable (store a Ptr<S> instead of an Own<S>)",
                    sd.name
                ));
            }
            if let Ty::Ptr(s) | Ty::Own(s) = t {
                if !structs.contains_key(s) {
                    errs.push(format!("struct `{}`: field `{f}` references unknown struct `{s}`", sd.name));
                }
            }
        }
    }

    let mut c = Checker {
        structs,
        env: HashMap::new(),
        errs,
    };
    for s in &prog.body {
        c.stmt(s);
    }
    // leak check: any linear local still live at end of scope
    let mut leaked: Vec<_> = c
        .env
        .iter()
        .filter(|(_, s)| s.ty.is_linear() && !s.consumed)
        .map(|(k, s)| (k.clone(), s.ty.clone()))
        .collect();
    leaked.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, ty) in leaked {
        c.err(format!(
            "leak: `{name}` : {ty} is still live at end of scope (linear value never consumed)"
        ));
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

    const NODE: &str = "struct Node { next: Ptr<Node>, prev: Ptr<Node>, elem: Int }";

    #[test]
    fn good() {
        accept("struct C { val: Int } let a = alloc C { val: 41 }; a.val = 42; free a;");
        // two nodes with mutual back-pointers (the doubly-linked essence), typed
        accept(&format!(
            "{NODE}
             let a = alloc Node {{ next: null, prev: null, elem: 1 }};
             let b = alloc Node {{ next: null, prev: null, elem: 2 }};
             a.next = addr(b); b.prev = addr(a);
             free a; free b;"
        ));
        // linear move
        accept("struct C { val: Int } let a = alloc C { val: 1 }; let b = a; free b;");
        // reading a Ptr field yields a copyable Ptr; binding it is fine
        accept(&format!(
            "{NODE} let a = alloc Node {{ next: null, prev: null, elem: 1 }};
             let p = a.next; free a;"
        ));
    }

    #[test]
    fn bad() {
        reject("struct C { val: Int } let a = alloc C { val: 1 }; free a; free a;"); // double free
        reject("struct C { val: Int } let a = alloc C { val: 1 }; free a; a.val = 2;"); // UAF
        reject("struct C { val: Int } let a = alloc C { val: 1 }; let b = a; free a; free b;"); // use after move
        reject("struct C { val: Int } let a = alloc C { val: 1 };"); // leak
        // dereference a bare aliased pointer (no capability)
        reject(&format!(
            "{NODE} let a = alloc Node {{ next: null, prev: null, elem: 0 }};
             let b = alloc Node {{ next: null, prev: null, elem: 0 }};
             a.next = addr(b); free b; a.next.elem = 1; free a;"
        ));
        // a field may not have a linear type
        reject("struct C { val: Int } struct Bad { c: Own<C> }");
        // type mismatch: writing an Int where a Ptr<Node> is expected
        reject(&format!(
            "{NODE} let a = alloc Node {{ next: null, prev: null, elem: 0 }};
             a.next = 5; free a;"
        ));
        // missing field in alloc
        reject(&format!(
            "{NODE} let a = alloc Node {{ next: null, elem: 0 }}; free a;"
        ));
        // free something that is not an owned capability
        reject("struct C { val: Int } let a = alloc C { val: 1 }; let p = addr(a); free p; free a;");
    }
}
