//! The High IR: the compiler-internal form a `CodeSpace` executes.
//!
//! Crucially, this is *not* the surface syntax. The surface is `Val` (code is
//! data). `analyze` turns a macroexpanded `Val` into `Ir`. Macros run *before*
//! analysis and never appear here. Arithmetic primitives get their own node
//! (`Prim`) so the value-model fast path is a first-class lowering, the way
//! `dyn_add` is in the real toolkit — not a call into an opaque builtin.

use std::rc::Rc;

use crate::value::Sym;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Prim {
    Add,
    Sub,
    Mul,
    Lt,
    Eq,
    List,
    Cons,
    First,
    Rest,
    IsNil,
    Println,
}

#[derive(Clone)]
pub enum Ir {
    /// Pre-encoded literal (already boxed/immediate per the model).
    Const(u64),
    /// A quoted datum: literal code-as-data.
    Quote(u64),
    /// Variable reference: resolved lexically first, then via a global Var.
    Var(Sym),
    If(Box<Ir>, Box<Ir>, Box<Ir>),
    Do(Vec<Ir>),
    /// `def` / `defmacro`. The `is_macro` flag rides through to the Var.
    Def {
        name: Sym,
        init: Box<Ir>,
        is_macro: bool,
    },
    Let(Vec<(Sym, Ir)>, Box<Ir>),
    Lambda {
        params: Vec<Sym>,
        variadic: Option<Sym>,
        body: Rc<Ir>,
    },
    Call(Box<Ir>, Vec<Ir>),
    Prim(Prim, Vec<Ir>),
}
