//! The neutral value vocabulary.
//!
//! `Cat` is the closed, small set of categories that matter for layout and
//! fast-path selection; `Val` names each. Everything unbounded lives behind
//! `Ref` + a heap object. Symbols are immediate.
//!
//! For the MOVING collector, the key design choice is here: lexical frames stay
//! `Rc`-managed (not in the moving heap), but their slots are `Cell<u64>`. The
//! `Rc` pointer never moves, so the mutator's `locals` reference survives a
//! collection; the GC rewrites the heap pointers *inside* the cells in place, so
//! reading a variable through `frame_get` always sees the relocated address.
//! The only values that go stale on a move are bare `u64`s the mutator holds
//! directly (the compiler's in-flight form) — which is exactly what handles fix.

use std::cell::Cell;
use std::rc::Rc;

use crate::ir::Ir;

pub type Sym = u32;
pub type HeapId = u32;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Cat {
    Int,
    Float,
    Bool,
    Nil,
    Sym,
    Ref,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RawTag {
    Int,
    Float,
    Bool,
    Nil,
    Sym,
    Ref,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Val {
    Int(i64),
    Float(f64),
    Bool(bool),
    Nil,
    Sym(Sym),
    Ref(HeapId),
}

/// Heap objects. Frames are NOT here (they are `Rc`-managed, see `Frame`), so a
/// closure's only heap child is nothing structural in its body — `Ir` carries
/// no heap pointers (literals live in the constant pool). A `Closure`'s captured
/// `env` is an `Rc<Frame>` the collector reaches to rewrite its cells.
#[derive(Clone)]
pub enum Obj {
    Cons {
        head: u64,
        tail: u64,
    },
    Str(String),
    BoxInt(i64),
    BoxFloat(f64),
    Closure {
        nparams: usize,
        variadic: bool,
        body: Rc<Ir>,
        env: Locals,
    },
    /// A user record: a type tag (interned symbol) plus positional fields. The
    /// thing polymorphic dispatch dispatches ON.
    Record {
        type_id: Sym,
        fields: Vec<u64>,
    },
    /// An escape continuation: invoking it does a non-local exit back to the
    /// `call-with-escaping-continuation` that created it (matched by `tag`).
    /// One-shot, upward-only — enough for early exit and generators-lite; full
    /// multi-shot continuations would need CPS or stack copying.
    Escape {
        tag: u64,
    },
    /// Forwarding marker left in from-space by the copying collector: the object
    /// now lives at index `.0` (or `u32::MAX` for reclaimed garbage). Any
    /// attempt to dereference a stale from-space pointer hits this and errors
    /// loudly — the moving-GC analogue of use-after-free.
    Moved(u32),
}

/// A lexical frame: `Cell` slots (so the GC can rewrite the heap pointers they
/// hold in place) plus a parent. `Rc` so closures capture it cheaply and so the
/// pointer is stable across a collection.
pub struct Frame {
    pub slots: Vec<Cell<u64>>,
    pub parent: Locals,
}

pub type Locals = Option<Rc<Frame>>;

/// Read slot `idx` in the frame `up` levels out. Reads through the `Cell`, so a
/// value relocated by a prior collection is seen at its new address.
pub fn frame_get(env: &Locals, up: u16, idx: u16) -> u64 {
    let mut f = env.as_ref().expect("local reference in empty environment");
    for _ in 0..up {
        f = f.parent.as_ref().expect("local reference past root frame");
    }
    f.slots[idx as usize].get()
}

/// Mutate slot `idx` in the frame `up` levels out. The slots are already `Cell`s
/// (for GC), so local assignment is just a cell write.
pub fn frame_set(env: &Locals, up: u16, idx: u16, v: u64) {
    let mut f = env.as_ref().expect("assignment in empty environment");
    for _ in 0..up {
        f = f.parent.as_ref().expect("assignment past root frame");
    }
    f.slots[idx as usize].set(v);
}
