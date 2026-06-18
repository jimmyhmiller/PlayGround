//! The lambda-Tally checker (v0 core): the L3 linear / permission discipline.
//!
//! Each variable has a static STATE:
//!   Own   -> holds a live linear Perm to a heap cell (must be consumed by `free`)
//!   Addr  -> a copyable address / value -- NO permission
//!   Moved -> an Own whose Perm was moved or freed away (using it is an error)
//!
//! Field reads always yield a copyable value (Addr): you can NEVER fabricate a
//! capability by reading memory -- Perms come only from `alloc`. That single
//! invariant is what makes aliased back-pointers safe. The discipline is the one
//! proved sound in ../../agda/CombinedSound.agda.

use crate::ast::{Expr, Program, Stmt};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Debug)]
enum St {
    Own,
    Addr,
    Moved,
}

struct Checker {
    st: HashMap<String, St>,
    errs: Vec<String>,
}

impl Checker {
    fn err(&mut self, m: impl Into<String>) {
        self.errs.push(m.into());
    }

    /// kind of an expression used as a *value*; `move_owns` allows a linear move.
    fn kind_of(&mut self, e: &Expr, move_owns: bool) -> St {
        match e {
            Expr::Int(_) | Expr::Null | Expr::Unit => St::Addr,
            Expr::Alloc(fields) => {
                for (_, fe) in fields {
                    self.require_copyable(fe);
                }
                St::Own
            }
            Expr::AddrOf(x) => {
                match self.st.get(x).copied() {
                    Some(St::Own) | Some(St::Addr) => {}
                    Some(St::Moved) => self.err(format!("addr({x}): {x} was already moved/freed")),
                    None => self.err(format!("addr({x}): unbound")),
                }
                St::Addr // an address is always copyable, Perm untouched
            }
            Expr::Field(obj, _) => {
                self.require_perm_for(obj);
                St::Addr // reading a field yields a copyable value, never a Perm
            }
            Expr::Var(x) => match self.st.get(x).copied() {
                None => {
                    self.err(format!("{x}: unbound"));
                    St::Addr
                }
                Some(St::Moved) => {
                    self.err(format!("{x}: use after move/free"));
                    St::Addr
                }
                Some(St::Addr) => St::Addr,
                Some(St::Own) => {
                    if move_owns {
                        self.st.insert(x.clone(), St::Moved); // linear move
                        St::Own
                    } else {
                        self.err(format!(
                            "{x}: linear owned value used where a plain value is required \
                             (use addr({x}) to copy its address)"
                        ));
                        St::Addr
                    }
                }
            },
        }
    }

    /// a value going into a field / a plain slot must be copyable (no Perm escapes)
    fn require_copyable(&mut self, e: &Expr) {
        self.kind_of(e, false);
    }

    /// the base of a read/write must currently hold a Perm
    fn require_perm_for(&mut self, e: &Expr) {
        match e {
            Expr::Var(x) => match self.st.get(x).copied() {
                Some(St::Own) => {}
                Some(St::Moved) => self.err(format!("{x}.<field>: use after move/free")),
                None => self.err(format!("{x}.<field>: unbound")),
                Some(St::Addr) => self.err(format!(
                    "{x}.<field>: no capability for this address \
                     (`{x}` is a bare Addr; you hold no Perm for it)"
                )),
            },
            _ => {
                self.kind_of(e, false);
                self.err(
                    "deref of an interior/aliased address: no capability held for it".to_string(),
                );
            }
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let(name, rhs) => {
                if self.st.get(name) == Some(&St::Own) {
                    self.err(format!(
                        "let {name}: rebinding drops `{name}`'s live Perm (leak)"
                    ));
                }
                let k = self.kind_of(rhs, true);
                self.st
                    .insert(name.clone(), if k == St::Own { St::Own } else { St::Addr });
            }
            Stmt::Write(base, _fld, rhs) => {
                self.require_perm_for(base);
                self.require_copyable(rhs);
            }
            Stmt::Free(name) => match self.st.get(name).copied() {
                Some(St::Own) => {
                    self.st.insert(name.clone(), St::Moved);
                }
                Some(St::Moved) => self.err(format!("free {name}: double free / use after free")),
                None => self.err(format!("free {name}: unbound")),
                Some(St::Addr) => self.err(format!("free {name}: no capability (bare Addr)")),
            },
            Stmt::Expr(e) => {
                self.kind_of(e, false);
            }
        }
    }
}

/// Check a program; return the list of diagnostics (empty == accepted).
pub fn check(prog: &Program) -> Vec<String> {
    let mut c = Checker {
        st: HashMap::new(),
        errs: Vec::new(),
    };
    for s in prog {
        c.stmt(s);
    }
    // leak check: any Perm still held at end of scope is a leak
    let mut leaked: Vec<_> = c
        .st
        .iter()
        .filter(|(_, v)| **v == St::Own)
        .map(|(k, _)| k.clone())
        .collect();
    leaked.sort();
    for name in leaked {
        c.err(format!(
            "leak: `{name}` still owns a live cell at end of scope (linear Perm never consumed)"
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

    #[test]
    fn good_programs_accept() {
        let good = [
            "let a = alloc { val: 41, next: null }; a.val = 42; free a;",
            // two nodes with mutual back-pointers (the doubly-linked essence)
            "let a = alloc { next: null, prev: null, elem: 1 };
             let b = alloc { next: null, prev: null, elem: 2 };
             a.next = addr(b); b.prev = addr(a);
             free a; free b;",
            // linear move
            "let a = alloc { val: 1 }; let b = a; free b;",
        ];
        for g in good {
            assert!(errs(g).is_empty(), "expected accept:\n{g}\ngot {:?}", errs(g));
        }
    }

    #[test]
    fn bad_programs_reject() {
        let bad = [
            "let a = alloc { val: 1 }; free a; free a;",          // double free
            "let a = alloc { val: 1 }; free a; a.val = 2;",        // use after free
            "let a = alloc { val: 1 }; let b = a; free a; free b;", // use after move
            "let a = alloc { val: 1 };",                           // leak
            // dangling pointer via alias
            "let a = alloc { next: null }; let b = alloc { next: null };
             a.next = addr(b); free b; a.next.next = null;",
        ];
        for b in bad {
            assert!(!errs(b).is_empty(), "expected reject:\n{b}");
        }
    }
}
