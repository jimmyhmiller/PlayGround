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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::ir::Ir;

/// Memory ordering for frame/cell slot access. Mutators run these single-threaded
/// between safepoints; cross-thread visibility is established by the acquire/
/// release fences at the safepoint park/resume boundary (see the GC), so plain
/// `Relaxed` on the slots themselves is sufficient and cheap.
const SLOT_ORDER: Ordering = Ordering::Relaxed;

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
    /// The neutral integer is `i128`-backed: small values ride in an immediate
    /// fixnum, larger ones are promoted to a boxed `BigInt`. (`i128` is a bounded
    /// stand-in for arbitrary precision — the promotion MECHANISM is the point;
    /// swapping the storage for a real bignum is orthogonal.)
    Int(i128),
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
    /// A character (R7RS `char?`), disjoint from integers and strings.
    Char(char),
    /// A vector: a fixed, mutable, index-addressed sequence of values.
    Vector(Vec<u64>),
    /// A multiple-values packet produced by `(values …)` and consumed by
    /// `call-with-values`. Distinct from a list so `(values (list 1 2))` (one
    /// value that is a list) differs from `(values 1 2)` (two values).
    Values(Vec<u64>),
    /// A promoted integer that did not fit the immediate fixnum range (but still
    /// fits `i128` — the common promotion).
    BigInt(i128),
    /// An integer beyond `i128`: true arbitrary precision. Reached only when the
    /// `i128` arithmetic path overflows.
    HugeInt(crate::bigint::BigInt),
    BoxFloat(f64),
    Closure {
        nparams: usize,
        variadic: bool,
        body: Arc<Ir>,
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
    /// A FULL, multi-shot continuation: a reified `CekMachine` continuation.
    /// Because it is an immutable `Rc`-linked structure, invoking it re-installs
    /// the captured continuation any number of times — enabling generators,
    /// coroutines, and backtracking. Only the stackless machine produces these.
    Cont(Arc<crate::cek::Kont>),
    /// A COMPOSABLE (delimited) continuation: the slice of a `CekMachine`
    /// continuation between a `%shift` and its enclosing `%reset`. Unlike `Cont`,
    /// invoking it does NOT abort — it splices the captured slice onto the
    /// caller's continuation under a fresh prompt and RETURNS, so it composes and
    /// can be invoked any number of times. Only the stackless machine produces
    /// these. Its captured frames are traced by the moving GC (see `gc.rs`), so
    /// it survives collection like any other heap value.
    PartialCont(Arc<crate::cek::Kont>),
    /// A FUTURE: the pending result of a thunk running on another OS thread. The
    /// `Arc<Mutex<..>>` is shared with the worker; `%await` joins the thread and
    /// caches its value. Cloneable (Arc), so the GC can relocate the enclosing
    /// object; the contained result is a raw heap id (see the rooting note in the
    /// spawn handler).
    Future(Arc<std::sync::Mutex<FutureSlot>>),
    /// Forwarding marker left in from-space by the copying collector: the object
    /// now lives at index `.0` (or `u32::MAX` for reclaimed garbage). Any
    /// attempt to dereference a stale from-space pointer hits this and errors
    /// loudly — the moving-GC analogue of use-after-free.
    Moved(u32),
}

/// The shared state of a `Future`: the worker's join handle (taken on first
/// await) and the cached result value once it has finished.
pub struct FutureSlot {
    pub handle: Option<std::thread::JoinHandle<u64>>,
    pub result: Option<u64>,
}

/// A lexical frame: `AtomicU64` slots (so the GC can rewrite the heap pointers
/// they hold in place, and so a frame captured by a closure can be shared across
/// threads) plus a parent. `Arc` so closures capture it cheaply, the pointer is
/// stable across a collection, and it is `Send + Sync`.
pub struct Frame {
    pub slots: Vec<AtomicU64>,
    pub parent: Locals,
}

pub type Locals = Option<Arc<Frame>>;

/// Build a frame's slots from raw values (replaces the old `Cell::new` splat).
pub fn slots_from(vals: impl IntoIterator<Item = u64>) -> Vec<AtomicU64> {
    vals.into_iter().map(AtomicU64::new).collect()
}

/// Snapshot existing slots into fresh atomics (each frame/`done` vector is
/// immutable-once-shared, so extending it must copy rather than mutate the prefix).
pub fn clone_slots(slots: &[AtomicU64]) -> Vec<AtomicU64> {
    slots.iter().map(|c| AtomicU64::new(c.load(SLOT_ORDER))).collect()
}

/// Read a slot atomic with the frame ordering.
pub fn slot_load(a: &AtomicU64) -> u64 {
    a.load(SLOT_ORDER)
}
/// Store a slot atomic with the frame ordering.
pub fn slot_store(a: &AtomicU64, v: u64) {
    a.store(v, SLOT_ORDER)
}

/// Read slot `idx` in the frame `up` levels out. Reads through the atomic, so a
/// value relocated by a prior collection is seen at its new address.
pub fn frame_get(env: &Locals, up: u16, idx: u16) -> u64 {
    let mut f = env.as_ref().expect("local reference in empty environment");
    for _ in 0..up {
        f = f.parent.as_ref().expect("local reference past root frame");
    }
    f.slots[idx as usize].load(SLOT_ORDER)
}

/// Mutate slot `idx` in the frame `up` levels out. The slots are already atomics
/// (for GC + sharing), so local assignment is just an atomic store.
pub fn frame_set(env: &Locals, up: u16, idx: u16, v: u64) {
    let mut f = env.as_ref().expect("assignment in empty environment");
    for _ in 0..up {
        f = f.parent.as_ref().expect("assignment past root frame");
    }
    f.slots[idx as usize].store(v, SLOT_ORDER);
}
