//! The neutral value vocabulary.
//!
//! `Cat` is the closed, small set of categories that matter for layout and
//! fast-path selection; `Val` names each. Everything unbounded lives behind
//! `Ref` + a heap object. Symbols are immediate.
//!
//! Stage D: a reference IS an address into the real heap (`heap::Gc`). `Obj`
//! is no longer the heap's storage — it is the ALLOCATION REQUEST vocabulary:
//! a constructor-shaped description the runtime's `alloc` lowers into a raw
//! heap object (header + inline fields). Reading heap data goes through
//! `Runtime::view`, which reconstructs the same shapes as borrows.
//!
//! For the MOVING collector, the key design choice is here: lexical frames
//! stay `Arc`-managed (not in the moving heap), but their slots are atomics.
//! The `Arc` pointer never moves, so the mutator's `locals` reference survives
//! a collection; the GC rewrites the heap pointers *inside* the slots in
//! place, so reading a variable through `frame_get` always sees the relocated
//! address. The only values that go stale on a move are bare `u64`s the
//! mutator holds directly — which is exactly what handles fix.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::heap::{Gc, CLOSURE_CAPS_OFF};
use crate::ir::Ir;
use crate::model::Repr;

/// Memory ordering for frame/cell slot access. A frame can be captured by a
/// closure that runs on ANOTHER thread, so a slot holding a heap ref is a
/// cross-thread channel: the store must publish the pointed-to object (`Release`)
/// and the load acquire it (`Acquire`). Cheap on x86, and correct on weaker isas.
const SLOT_LOAD: Ordering = Ordering::Acquire;
const SLOT_STORE: Ordering = Ordering::Release;

pub type Sym = u32;

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
    Ref(Gc),
}

/// ALLOCATION REQUESTS: the constructor vocabulary `Runtime::alloc` lowers
/// into raw heap objects (see `heap::kind` for the layouts). Cold/simple call
/// sites build one of these; hot paths use the typed `alloc_*` constructors
/// directly and never materialize the enum.
pub enum Obj {
    Cons {
        head: u64,
        tail: u64,
    },
    /// The empty list `()` — a distinct value from `nil` (Clojure's
    /// `PersistentList/EMPTY`). One canonical singleton per runtime.
    EmptyList,
    Str(String),
    /// A character (R7RS `char?`), disjoint from integers and strings.
    /// A UTF-16 code unit VALUE (any scalar, or a lone surrogate half —
    /// `charAt` on astral content yields halves, exactly like Java's char).
    Char(u32),
    /// A growable, index-addressed sequence: lowered to a HANDLE object (the
    /// identity, carrying the logical length) plus a DATA blob (the elements,
    /// carrying the capacity) — the ArrayList shape. Growth allocates a new
    /// blob and re-points the handle, so every reference observes it.
    Vector(Vec<u64>),
    /// A multiple-values packet produced by `(values …)` and consumed by
    /// `call-with-values`. Distinct from a list so `(values (list 1 2))` (one
    /// value that is a list) differs from `(values 1 2)` (two values).
    Values(Vec<u64>),
    /// A promoted integer that did not fit the immediate fixnum range (but still
    /// fits `i128` — the common promotion).
    BigInt(i128),
    /// An integer beyond `i128`: true arbitrary precision, limbs stored inline.
    HugeInt(crate::bigint::BigInt),
    /// An exact rational `num/den`, always in lowest terms with `den > 1`.
    Ratio(i128, i128),
    BoxFloat(f64),
    /// A flat closure: the body is registered in the append-only TEMPLATE
    /// registry (code, like gc-rust's type table); the captured VALUES are
    /// copied inline into the object at creation.
    Closure {
        nparams: usize,
        variadic: bool,
        /// Size of the ONE flat activation frame a call allocates (params, the
        /// rest arg, every let/catch slot) — assigned by the `flatten` pass.
        nslots: u16,
        body: Arc<Ir>,
        caps: Vec<u64>,
    },
    /// A MULTI-ARITY function: per-arity closures selected by argument count at
    /// call time. `fixed[k]` holds the k-param closure's bits (0 = no such
    /// arity); `variadic` is the `[… & rest]` clause (min fixed count, closure
    /// bits).
    MultiFn {
        fixed: Vec<u64>,
        variadic: Option<(usize, u64)>,
    },
    /// A user record: a type tag (interned symbol) plus positional fields.
    Record {
        type_id: Sym,
        fields: Vec<u64>,
    },
    /// An escape continuation: invoking it does a non-local exit back to the
    /// `call-with-escaping-continuation` that created it (matched by `tag`).
    Escape {
        tag: u64,
    },
    /// A FULL, multi-shot continuation: a reified `CekMachine` continuation.
    /// The `Arc<Kont>` is execution-machine state (never raw heap data); the
    /// object stores an index into the runtime's kont registry.
    Cont(Arc<crate::cek::Kont>),
    /// A COMPOSABLE (delimited) continuation — same registry shape as `Cont`;
    /// invoking it splices rather than aborts (see cek.rs).
    PartialCont(Arc<crate::cek::Kont>),
    /// An ATOM: a single atomically-updated cell, initialized to the given
    /// value. The object is `[hdr | slot]` and swap!/CAS operate on the slot
    /// word directly (STW keeps a moving collector safe).
    Atom(u64),
    /// A FUTURE: the pending result of a thunk running on another OS thread.
    /// The `Arc<Mutex<..>>` is an OS resource: it lives in the runtime's
    /// future registry and the object stores its index.
    Future(Arc<std::sync::Mutex<FutureSlot>>),
}

/// The shared state of a `Future`: the worker's join handle (taken on first
/// await) and the cached result value once it has finished.
pub struct FutureSlot {
    pub handle: Option<std::thread::JoinHandle<u64>>,
    pub result: Option<u64>,
}

/// ONE flat activation frame per call — no parent chain (the `flatten` pass
/// resolves every variable to a slot of the single frame or to a closure
/// capture). `AtomicU64` slots so the GC can rewrite the heap pointers they
/// hold in place, and so a frame published at a safepoint (or captured by a
/// CEK continuation) can be traced from another thread.
///
/// `caps_src` holds the RUNNING CLOSURE's bits (0 = none): captures are read
/// through it out of the closure object itself. It is an atomic root slot the
/// collector forwards, so capture reads re-decode the closure's CURRENT
/// address — GC-safe by construction, with no capture-array copy per call.
pub struct Frame {
    pub slots: Vec<AtomicU64>,
    pub caps_src: AtomicU64,
}

pub type Locals = Option<Arc<Frame>>;

/// Build a frame's slots from raw values (replaces the old `Cell::new` splat).
pub fn slots_from(vals: impl IntoIterator<Item = u64>) -> Vec<AtomicU64> {
    vals.into_iter().map(AtomicU64::new).collect()
}

/// Snapshot existing slots into fresh atomics (each frame/`done` vector is
/// immutable-once-shared, so extending it must copy rather than mutate the prefix).
pub fn clone_slots(slots: &[AtomicU64]) -> Vec<AtomicU64> {
    slots.iter().map(|c| AtomicU64::new(c.load(SLOT_LOAD))).collect()
}

/// Read a slot atomic with the frame ordering.
pub fn slot_load(a: &AtomicU64) -> u64 {
    a.load(SLOT_LOAD)
}
/// Store a slot atomic with the frame ordering.
pub fn slot_store(a: &AtomicU64, v: u64) {
    a.store(v, SLOT_STORE)
}

/// Read slot `idx` of the activation frame. Reads through the atomic, so a
/// value relocated by a prior collection is seen at its new address. `up` must
/// be 0 — the `flatten` pass eliminated frame chains; a non-zero `up` means
/// unflattened Ir reached an execution tier.
pub fn frame_get(env: &Locals, up: u16, idx: u16) -> u64 {
    assert_eq!(up, 0, "unflattened Ir reached a tier: Local up={up} (run flatten::flatten)");
    let f = env.as_ref().expect("local reference in empty environment");
    f.slots[idx as usize].load(SLOT_LOAD)
}

/// Mutate slot `idx` of the activation frame (see `frame_get` for the `up`
/// contract). The slots are already atomics (for GC + sharing), so local
/// assignment is just an atomic store.
pub fn frame_set(env: &Locals, up: u16, idx: u16, v: u64) {
    assert_eq!(up, 0, "unflattened Ir reached a tier: SetLocal up={up} (run flatten::flatten)");
    let f = env.as_ref().expect("assignment in empty environment");
    f.slots[idx as usize].store(v, SLOT_STORE);
}

/// Read capture `idx` of the running closure: decode the frame's `caps_src`
/// (kept current by the collector) and load the capture word straight out of
/// the closure object's inline cap array.
pub fn frame_cap<R: Repr>(env: &Locals, idx: u16) -> u64 {
    let f = env.as_ref().expect("capture reference in empty environment");
    let bits = f.caps_src.load(SLOT_LOAD);
    assert_ne!(bits, 0, "capture read in a frame with no running closure");
    let g = R::as_ref(bits);
    unsafe { *(g.0.add(CLOSURE_CAPS_OFF + idx as usize * 8) as *const u64) }
}

/// Read capture `idx` directly off a CLOSURE OBJECT (creation-time transitive
/// capture; the shared building block under `frame_cap` and `build_caps`).
#[inline(always)]
pub fn closure_cap(g: Gc, idx: u16) -> u64 {
    unsafe { *(g.0.add(CLOSURE_CAPS_OFF + idx as usize * 8) as *const u64) }
}

/// Compute a new closure's capture VALUES at creation time: copy each source
/// out of the current activation (its slots, or the running closure's own
/// captures for transitive capture). The values are written inline into the
/// closure object by `alloc`. Shared by every backend.
pub fn build_caps<R: Repr>(captures: &[crate::ir::CapSrc], env: &Locals) -> Vec<u64> {
    use crate::ir::CapSrc;
    if captures.is_empty() {
        return Vec::new();
    }
    let f = env.as_ref().expect("closure captures in empty environment");
    captures
        .iter()
        .map(|c| match c {
            CapSrc::Slot(i) => f.slots[*i as usize].load(SLOT_LOAD),
            CapSrc::Cap(i) => {
                let bits = f.caps_src.load(SLOT_LOAD);
                assert_ne!(bits, 0, "transitive capture with no running closure");
                closure_cap(R::as_ref(bits), *i)
            }
        })
        .collect()
}
