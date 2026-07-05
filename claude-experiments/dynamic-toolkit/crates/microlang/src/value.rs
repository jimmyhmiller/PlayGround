//! The neutral value vocabulary.
//!
//! The key design decision lives here: `Cat` is the *closed, small* set of
//! categories that matter for **layout and fast-path selection**, and `Val`
//! names each of them as a first-class variant. The original toolkit's
//! `Decoded { Tagged | Float }` privileged float (NaN-boxing's ontology) and
//! forced integers to be "just another tagged payload", which is exactly what
//! made an integer-primary language (Clojure) second-class by construction.
//!
//! Everything unbounded (lists, closures, boxed numbers, strings, records...)
//! lives behind `Ref` + a heap object. Symbols are immediate (interned id) so
//! macro-time identity comparison is a cheap integer compare.

use std::rc::Rc;

use crate::ir::Ir;

/// Interned symbol id. The name lives in the runtime's symbol table.
pub type Sym = u32;

/// Heap object id (index into the runtime heap).
pub type HeapId = u32;

/// The closed set of layout categories. This is deliberately small: these are
/// the only things a *value model* must reason about to decide immediacy and
/// pick arithmetic fast paths. The open-ended type space lives behind `Ref`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Cat {
    Int,
    Float,
    Bool,
    Nil,
    Sym,
    Ref,
}

/// The physical tag actually present in a word for a given `Repr`. A `LowBit`
/// word never physically carries `Float` (floats are boxed); a `NanBox` word
/// never physically carries `Int` (ints are boxed). `tag_of` reports what is
/// physically there; `Val`/`Cat` is the semantic view.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RawTag {
    Int,
    Float,
    Bool,
    Nil,
    Sym,
    Ref,
}

/// Neutral, owned view of a value. Replaces `Decoded`. Note `Int` is a peer of
/// `Float`, not a subordinate of "tagged".
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Val {
    Int(i64),
    Float(f64),
    Bool(bool),
    Nil,
    Sym(Sym),
    Ref(HeapId),
}

/// Heap objects. Everything that is not an immediate lives here.
#[derive(Clone)]
pub enum Obj {
    /// A cons cell. The empty list is `Val::Nil`, never a heap object.
    Cons { head: u64, tail: u64 },
    Str(String),
    /// A boxed integer: only allocated by a model where `Int` is *not*
    /// immediate (e.g. `NanBox`). This allocation is the cost the value axis
    /// is about.
    BoxInt(i64),
    /// A boxed float: allocated by a model where `Float` is not immediate
    /// (e.g. `LowBit`).
    BoxFloat(f64),
    /// A user closure. Captures its defining lexical frame (`env`). The
    /// `is_macro` flag is carried by the *Var*, not the closure, so this is
    /// just a callable.
    Closure {
        params: Vec<Sym>,
        variadic: Option<Sym>,
        body: Rc<Ir>,
        env: Locals,
    },
}

/// A lexical frame: name -> value bindings, linked to a parent. Persistent via
/// `Rc` so closures can capture it cheaply. A real toolkit would resolve these
/// to slots at analyze time; the sketch keeps name lookup for legibility.
pub struct Frame {
    pub vars: Vec<(Sym, u64)>,
    pub parent: Locals,
}

pub type Locals = Option<Rc<Frame>>;

pub fn frame_lookup(mut locals: &Locals, sym: Sym) -> Option<u64> {
    while let Some(fr) = locals {
        for &(s, v) in fr.vars.iter().rev() {
            if s == sym {
                return Some(v);
            }
        }
        locals = &fr.parent;
    }
    None
}
