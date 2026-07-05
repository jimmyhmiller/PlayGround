//! The High IR: the compiler-internal form a `CodeSpace` executes.
//!
//! Crucially, this is *not* the surface syntax. The surface is `Val` (code is
//! data). `analyze` turns a macroexpanded `Val` into `Ir`. Macros run *before*
//! analysis and never appear here. Arithmetic primitives get their own node
//! (`Prim`) so the value-model fast path is a first-class lowering, the way
//! `dyn_add` is in the real toolkit — not a call into an opaque builtin.

use std::rc::Rc;

use crate::value::Sym;

/// Index into the runtime constant pool. Literals are indirected through the
/// pool (which is a GC root) so `Ir` itself holds NO heap pointers — a moving
/// collector could not rewrite pointers embedded in an immutable `Rc<Ir>`.
pub type ConstId = u32;

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
    /// Force a garbage collection. A safepoint you can place explicitly, so the
    /// GC-during-macro hazard is deterministic to demonstrate.
    Gc,
    /// `(record 'Type f0 f1 ...)` -> a record value.
    Record,
    /// `(field r i)` -> the i-th field of a record.
    Field,
    /// `(%callec f)` — call `f` with a fresh escape continuation. Backend-handled
    /// (needs to invoke a closure and catch a non-local exit), like `Gc`.
    CallEc,
}

#[derive(Clone)]
pub enum Ir {
    /// A literal, held in the constant pool (no embedded heap pointer).
    Const(ConstId),
    /// A quoted datum, held in the constant pool.
    Quote(ConstId),
    /// Lexical variable, resolved to a slot at analyze time: `up` frames out,
    /// slot `idx`. No name, no search — a pointer walk plus an index.
    Local { up: u16, idx: u16 },
    /// Global variable: resolved through the Var table at RUN time, so a
    /// reference can precede the definition (late binding).
    Global(Sym),
    /// Assign an existing lexical slot (`set!` on a local). Returns the value.
    SetLocal { up: u16, idx: u16, val: Box<Ir> },
    /// Assign an existing global Var. Returns the value.
    SetGlobal { name: Sym, val: Box<Ir> },
    If(Box<Ir>, Box<Ir>, Box<Ir>),
    Do(Vec<Ir>),
    /// `def` / `defmacro`. The `is_macro` flag rides through to the Var.
    Def {
        name: Sym,
        init: Box<Ir>,
        is_macro: bool,
    },
    /// `let*`: binding inits in order (each occupies the next slot of a single
    /// frame and can see earlier ones), then the body.
    Let(Vec<Ir>, Box<Ir>),
    /// A closure: arity only; the body's locals are already slot-resolved.
    Lambda {
        nparams: usize,
        variadic: bool,
        body: Rc<Ir>,
    },
    Call(Box<Ir>, Vec<Ir>),
    Prim(Prim, Vec<Ir>),
    /// `(defmethod name Type impl)`: register an impl for `(name, Type)`.
    DefMethod {
        name: Sym,
        ty: Sym,
        imp: Box<Ir>,
    },
    /// A polymorphic call site: `(method recv args...)`. `site` is a stable id
    /// used as the inline-cache key; the dispatch strategy resolves it.
    Dispatch {
        site: usize,
        method: Sym,
        args: Vec<Ir>,
    },
}
