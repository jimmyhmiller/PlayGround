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
    /// `(string-length s)` -> the character count of a string.
    StrLen,
    /// `(char->integer c)` -> the Unicode scalar value of a char.
    CharToInt,
    /// `(integer->char n)` -> the char with Unicode scalar value `n`.
    IntToChar,
    /// `(vector e ...)` -> a fresh vector of the arguments.
    Vector,
    /// `(vector-ref v i)` -> the i-th element.
    VectorRef,
    /// `(vector-set! v i x)` -> set the i-th element to `x`; returns nil.
    VectorSet,
    /// `(vector-length v)` -> the element count.
    VectorLen,
    /// `(values e ...)` -> a multiple-values packet.
    Values,
    /// `(%values->list v)` -> the list of values in a packet (a lone non-packet
    /// value becomes a one-element list). Bridges `values` to `apply`.
    ValuesToList,
    /// `(apply f a ... lst)` -> apply `f` to the leading args followed by the
    /// elements of the final list. Requires a backend that can invoke closures
    /// (intercepted by the `CekMachine`), like `%callcc`.
    Apply,
    /// `(%eq a b)` — object identity (`eq?`/`eqv?`): equal iff the encoded bits
    /// are equal. For immediates that is value equality; for heap values it is
    /// pointer identity. (Contrast `Eq`, which is structural `equal?`.)
    Identical,
    /// `(%callcc f)` — full call-with-current-continuation. Only the stackless
    /// `CekMachine` supports it (the continuation is a first-class, multi-shot,
    /// re-installable value); host-stack tiers cannot.
    CallCc,
    /// `(%reset body)` — install a continuation delimiter (prompt) and evaluate
    /// `body` under it. A NATIVE delimited-control primitive; only the stackless
    /// `CekMachine` supports it.
    Reset,
    /// `(%shift f)` — capture the continuation from here up to the nearest
    /// enclosing `%reset`, reify it as a COMPOSABLE (re-delimited, multi-shot)
    /// procedure, and apply `f` to it under a re-established prompt. Native
    /// delimited control; `CekMachine` only.
    Shift,

    // ── optimizer-introduced fixnum specializations ──────────────────────────
    // These are produced ONLY by the `optimize` nanopass (never by `analyze`).
    // `FxAdd/FxSub/FxMul/FxLt/FxEq` mean "same as `Add/Sub/Mul/Lt/Eq`, but the
    // operands have been PROVEN to be immediate fixnums" (by a dominating
    // `AllFixnum` guard the pass inserts). Semantically identical to the checked
    // op — every interpreter tier lowers them to the very same `prim` — so a
    // backend that ignores the distinction is still correct. Only the JIT reads
    // it, to skip the per-op tag check (it still keeps the overflow → promotion
    // path, since two fixnums can still overflow to a bignum).
    FxAdd,
    FxSub,
    FxMul,
    FxLt,
    FxEq,
    /// `(%all-fixnum? a b ...)` — true iff EVERY argument is an immediate fixnum.
    /// The guard the specializer places at a lambda's entry; when it holds, the
    /// body's `Fx*` ops are valid. On the JIT it lowers to a single combined
    /// tag test; on the interpreter tiers it is a normal predicate (and the
    /// result only picks between two equivalent bodies, so it can't be wrong).
    AllFixnum,
}

impl Prim {
    /// The checked arithmetic op a fixnum-specialized `Fx*` op stands for
    /// (identity for everything else). Interpreter tiers use this to give `Fx*`
    /// exactly the semantics of the base op.
    pub fn dechecked(self) -> Prim {
        match self {
            Prim::FxAdd => Prim::Add,
            Prim::FxSub => Prim::Sub,
            Prim::FxMul => Prim::Mul,
            Prim::FxLt => Prim::Lt,
            Prim::FxEq => Prim::Eq,
            other => other,
        }
    }
}

#[derive(Clone, PartialEq)]
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
