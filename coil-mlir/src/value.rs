//! The `Val` type — the single value universe shared by data, syntax, and
//! (eventually) reflected MLIR handles.
//!
//! This first increment defines the *syntax-level* cases the reader produces.
//! The reflected-MLIR cases (`Type`, `Attr`, `Value`, `Op`, …) from
//! `mlir-lisp-design/KERNEL.md §1` are added once the backend is wired in; they
//! live behind the `Backend` boundary (see `backend`) so the core stays
//! buildable without MLIR.

use std::rc::Rc;

/// A coil value. `Clone` is cheap: composites are `Rc`-shared.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    /// The absence of a value (e.g. the result of a void op). Not readable.
    Unit,
    /// The literal `nil`.
    Nil,
    Bool(bool),
    /// Integer literal. Width/signedness are attached later by `(: v t)`, not
    /// by the reader, so a bare literal carries none.
    Int(i64),
    Float(f64),
    Str(Rc<str>),
    /// A symbol: `foo`, `arith.addi`, `my/helper`, `+`, `->`, `@printf`, `^bb1`,
    /// `%0`. Sigils (`@ ^ %`) are kept as part of the name in this increment;
    /// the expander gives them meaning (see SPEC §1).
    Sym(Rc<str>),
    /// `:keyword`. Stored without the leading colon.
    Keyword(Rc<str>),
    List(Rc<[Val]>),
    Vec(Rc<[Val]>),
    /// Insertion-ordered key/value pairs (`{:a 1 :b 2}`).
    Map(Rc<[(Val, Val)]>),
    /// `!…` type literal, stored without the `!`. Backend parses it to a Type.
    TypeLit(Rc<str>),
    /// `#…` attribute literal, stored without the `#`. Backend parses it.
    AttrLit(Rc<str>),
}

impl Val {
    pub fn sym(s: impl AsRef<str>) -> Val {
        Val::Sym(Rc::from(s.as_ref()))
    }
    pub fn str(s: impl AsRef<str>) -> Val {
        Val::Str(Rc::from(s.as_ref()))
    }
    pub fn keyword(s: impl AsRef<str>) -> Val {
        Val::Keyword(Rc::from(s.as_ref()))
    }
    pub fn list(items: Vec<Val>) -> Val {
        Val::List(Rc::from(items))
    }
    pub fn vector(items: Vec<Val>) -> Val {
        Val::Vec(Rc::from(items))
    }
    pub fn map(pairs: Vec<(Val, Val)>) -> Val {
        Val::Map(Rc::from(pairs))
    }

    /// The symbol name, if this is a symbol.
    pub fn as_sym(&self) -> Option<&str> {
        match self {
            Val::Sym(s) => Some(s),
            _ => None,
        }
    }

    /// The head symbol of a list, if any — `(foo …)` → `"foo"`.
    pub fn head_sym(&self) -> Option<&str> {
        match self {
            Val::List(items) => items.first().and_then(Val::as_sym),
            _ => None,
        }
    }

    /// True for `false`, `nil`, and `Unit`; everything else is truthy.
    pub fn is_truthy(&self) -> bool {
        !matches!(self, Val::Bool(false) | Val::Nil | Val::Unit)
    }
}
