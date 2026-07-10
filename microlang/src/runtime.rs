//! The runtime CORE: heap, moving GC roots, symbol table, global environment,
//! the dispatch axis, and the value-model-aware primitives. It knows NOTHING
//! about s-expressions, special forms, or `analyze` — those live in the optional
//! `sexpr` frontend (or a frontend compiles to `Ir` directly).
//!
//! `encode`/`decode` are the seam where the value axis meets everything else:
//! they box a non-immediate category and unbox on the way out, and `allocs`
//! counts the boxing so the micro-languages can *show* the cost.

use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::{HashMap, HashSet};

/// Memory ordering for the global-slot array. Reads/writes are `Relaxed`; a
/// cross-thread global definition is made visible to another thread by the
/// happens-before edge on whatever channel passes the value (an atom / a join),
/// not by the slot store itself.
const GLOBAL_ORDER: Ordering = Ordering::Relaxed;
use std::hash::BuildHasherDefault;

/// A cheap hasher for `Sym` (a `u32`) keys. The global environment is consulted
/// on every free-variable reference — a hot path in every tier — and the default
/// `SipHash` is far more than a 32-bit symbol id needs. Fibonacci multiply gives
/// good spread at a fraction of the cost. (Dep-free, like the rest of the core.)
#[derive(Default)]
pub struct SymHasher(u64);
impl std::hash::Hasher for SymHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }
    fn write_u32(&mut self, i: u32) {
        self.0 = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}
/// A `HashMap` keyed by `Sym` with the fast symbol hasher.
pub type SymMap<V> = HashMap<Sym, V, BuildHasherDefault<SymHasher>>;
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use crate::bigint::BigInt;
use crate::dispatch::{Dispatch, Megamorphic, MethodRegistry};
use crate::ir::{ConstId, Prim};
use crate::model::{Repr, ValueModel};
use crate::value::{Cat, Frame, HeapId, Locals, Obj, RawTag, Sym, Val};

/// Sentinel for an unbound slot in `global_slots`. `u64::MAX` has an invalid tag
/// under every value model (`LowBit`/`HighBit`/`NanBox`), so it can never collide
/// with a real encoded value — reading it back means "not bound, use slow path".
pub const GLOBAL_UNBOUND: u64 = u64::MAX;

/// The panic payload of a `(throw v)`: the thrown runtime value, carried up the
/// stack until an `Ir::Try` catches it. Neutral control-flow, not Clojure-specific.
pub struct Thrown {
    pub value: u64,
}

/// Objects per heap chunk. A power of two so index split is shift/mask.
const CHUNK_BITS: u32 = 12;
const CHUNK: usize = 1 << CHUNK_BITS;

/// A SEGMENTED heap: a vector of fixed-capacity chunks. Appending only ever adds
/// to (or pushes) a chunk, so the buffer holding any existing object NEVER moves
/// — an `&Obj` stays valid while other objects are allocated. That address
/// stability is what makes a shared heap safe to read concurrently (only chunk
/// acquisition needs synchronization) and is the prerequisite for true
/// system-thread parallelism over one heap. A `HeapId` is a flat index; the
/// chunk is `id >> CHUNK_BITS`, the offset `id & (CHUNK-1)`. `Index`/`IndexMut`
/// preserve the `heap[id]` call sites verbatim.
pub struct Heap {
    chunks: Vec<Vec<Obj>>,
    len: usize,
}

impl Heap {
    pub fn new() -> Self {
        Heap { chunks: Vec::new(), len: 0 }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Append an object; returns its flat id. The current chunk never grows past
    /// its reserved capacity, so existing object addresses are stable.
    pub fn push(&mut self, o: Obj) -> usize {
        let i = self.len;
        let c = i >> CHUNK_BITS;
        if c >= self.chunks.len() {
            self.chunks.push(Vec::with_capacity(CHUNK));
        }
        self.chunks[c].push(o);
        self.len += 1;
        i
    }
    pub fn iter(&self) -> impl Iterator<Item = &Obj> {
        self.chunks.iter().flat_map(|c| c.iter())
    }
}

impl Default for Heap {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Index<usize> for Heap {
    type Output = Obj;
    #[inline]
    fn index(&self, i: usize) -> &Obj {
        &self.chunks[i >> CHUNK_BITS][i & (CHUNK - 1)]
    }
}

impl std::ops::IndexMut<usize> for Heap {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Obj {
        &mut self.chunks[i >> CHUNK_BITS][i & (CHUNK - 1)]
    }
}

/// Reserved capacity for the append-only, stable-base tables (`consts`,
/// `sym_names`, `global_slots`). They must never reallocate, because concurrent
/// readers hold raw/unsafe references into them (the JIT reads `consts`/globals
/// through a base pointer; `sym_name` returns a borrowed `&str`). Growth past
/// this panics loudly rather than dangling — a clear error, not UB.
const TABLE_CAP: usize = 1 << 20;

/// The mutable dispatch tables, guarded by one lock. `resolve` copies a `u64`
/// impl out and drops the guard before the caller invokes it, so no lock is held
/// across a method call (no reentrancy).
pub(crate) struct Tables {
    pub(crate) methods: MethodRegistry,
    method_names: HashSet<Sym>,
    pub(crate) dispatch: Box<dyn Dispatch>,
}

/// State SHARED by every thread over one `Arc`. Its interior mutability is the
/// whole game: the heap is read lock-free (stable segmented addresses) and
/// mutated only under `heap_lock` (alloc / GC / in-place set); the append-only
/// tables are read lock-free from their reserved, never-reallocated buffers and
/// appended under a lock; globals are atomic; dispatch is behind a short lock.
/// No lock is ever held across a callback into the interpreter, so a thread
/// blocked in `deref` holds nothing another thread needs.
pub struct Shared<M: ValueModel> {
    pub(crate) heap: UnsafeCell<Heap>,
    /// Serializes heap MUTATION (alloc, GC, in-place vector set). Reads are
    /// lock-free (segmented heap => stable object addresses).
    pub(crate) heap_lock: Mutex<()>,
    allocs: AtomicU64,
    pub(crate) relocated: AtomicU64,
    pub(crate) freed: AtomicU64,
    /// Constant pool (reserved, stable base for the JIT's inline reads).
    pub(crate) consts: UnsafeCell<Vec<u64>>,
    consts_lock: Mutex<()>,
    /// Interner: names in a reserved, stable buffer (lock-free `&str` reads);
    /// the dedup map + append serialized by `sym_lock`.
    sym_names: UnsafeCell<Vec<String>>,
    sym_ids: UnsafeCell<HashMap<String, Sym>>,
    sym_lock: Mutex<()>,
    /// Global environment: atomic slots, reserved stable base (see `global`).
    pub global_slots: Vec<AtomicU64>,
    pub(crate) tables: Mutex<Tables>,
    apply_fn: AtomicU64, // Sym+1, or 0 for None
    escape_tags: AtomicU64,
    _pd: PhantomData<fn() -> M>,
}

// SAFETY: every field is accessed under the discipline documented on `Shared`:
// heap/consts/interner reads are to stable addresses that are never freed or
// moved except under a lock (heap) or never during a run (append-only tables);
// all mutation is serialized by the corresponding lock; globals/counters are
// atomic. No `&`/`&mut` into the interior outlives the operation that takes it.
unsafe impl<M: ValueModel> Send for Shared<M> {}
unsafe impl<M: ValueModel> Sync for Shared<M> {}

pub struct Runtime<M: ValueModel> {
    /// State shared with sibling threads (heap, globals, interner, dispatch).
    pub(crate) shared: Arc<Shared<M>>,
    /// The shadow stack: this THREAD's GC root set for transient values.
    /// Per-thread (not shared) — each mutator handle owns its own.
    pub(crate) shadow: Vec<u64>,
    _pd: PhantomData<fn() -> M>,
}

impl<M: ValueModel> Runtime<M> {
    pub fn new() -> Self {
        let shared = Shared {
            heap: UnsafeCell::new(Heap::new()),
            heap_lock: Mutex::new(()),
            allocs: AtomicU64::new(0),
            relocated: AtomicU64::new(0),
            freed: AtomicU64::new(0),
            consts: UnsafeCell::new(Vec::with_capacity(TABLE_CAP)),
            consts_lock: Mutex::new(()),
            sym_names: UnsafeCell::new(Vec::with_capacity(TABLE_CAP)),
            sym_ids: UnsafeCell::new(HashMap::new()),
            sym_lock: Mutex::new(()),
            global_slots: (0..0).map(|_| AtomicU64::new(GLOBAL_UNBOUND)).collect(),
            tables: Mutex::new(Tables {
                methods: HashMap::new(),
                method_names: HashSet::new(),
                dispatch: Box::new(Megamorphic::new()),
            }),
            apply_fn: AtomicU64::new(0),
            escape_tags: AtomicU64::new(0),
            _pd: PhantomData,
        };
        // Pre-size the global array to its reserved cap so its base is stable and
        // slots exist for every future symbol (index == Sym).
        let mut slots = Vec::with_capacity(TABLE_CAP);
        slots.extend((0..TABLE_CAP).map(|_| AtomicU64::new(GLOBAL_UNBOUND)));
        let shared = Shared { global_slots: slots, ..shared };
        Runtime { shared: Arc::new(shared), shadow: Vec::new(), _pd: PhantomData }
    }

    /// A fresh mutator handle for another OS thread, sharing this runtime's heap,
    /// globals, interner, and dispatch. The new handle has its OWN (empty) shadow
    /// stack. This is what `spawn`/`future` hands to a `std::thread`.
    pub fn thread_handle(&self) -> Self {
        Runtime { shared: self.shared.clone(), shadow: Vec::new(), _pd: PhantomData }
    }

    // ── heap access (lock-free reads to stable addresses) ───
    /// The heap for READING. Object addresses are stable (segmented), so a shared
    /// reference is sound even while another thread allocates.
    #[inline]
    pub fn heap(&self) -> &Heap {
        unsafe { &*self.shared.heap.get() }
    }
    /// The heap for MUTATION. Callers MUST hold `heap_lock` (alloc/GC/set take it).
    #[inline]
    fn heap_mut(&self) -> &mut Heap {
        unsafe { &mut *self.shared.heap.get() }
    }

    // ── symbols ─────────────────────────────────────────────
    pub fn intern(&self, s: &str) -> Sym {
        let _g = self.shared.sym_lock.lock().unwrap();
        // SAFETY: the interner is only mutated here, under `sym_lock`; the
        // reserved buffers never reallocate (TABLE_CAP), so concurrent lock-free
        // readers of `sym_name` see stable `&str`s.
        let ids = unsafe { &mut *self.shared.sym_ids.get() };
        if let Some(&id) = ids.get(s) {
            return id;
        }
        let names = unsafe { &mut *self.shared.sym_names.get() };
        let id = names.len() as Sym;
        assert!(names.len() < TABLE_CAP, "interner overflow: raise TABLE_CAP");
        names.push(s.to_string());
        ids.insert(s.to_string(), id);
        id
    }

    /// Read a global's value, or `None` if unbound. THE global read path — the
    /// dense `global_slots` array IS the store (no separate map), so this is a
    /// single indexed load, and a `Sym` beyond the current slot count is unbound.
    pub fn global(&self, sym: Sym) -> Option<u64> {
        match self.shared.global_slots.get(sym as usize).map(|a| a.load(GLOBAL_ORDER)) {
            Some(v) if v != GLOBAL_UNBOUND => Some(v),
            _ => None,
        }
    }

    /// Is this global bound? (`set!`/redefinition checks.)
    pub fn global_defined(&self, sym: Sym) -> bool {
        matches!(self.shared.global_slots.get(sym as usize).map(|a| a.load(GLOBAL_ORDER)), Some(v) if v != GLOBAL_UNBOUND)
    }

    /// Define (or redefine) a global. Atomic store, so `&self` suffices — a step
    /// toward the shared runtime where a thread defines through an `Arc`.
    pub fn define_global(&self, sym: Sym, val: u64) {
        if let Some(slot) = self.shared.global_slots.get(sym as usize) {
            slot.store(val, GLOBAL_ORDER);
        }
    }

    /// Assign an existing global's value (`set!`). Returns `false` if unbound.
    pub fn set_global_val(&self, sym: Sym, val: u64) -> bool {
        match self.shared.global_slots.get(sym as usize) {
            Some(slot) if slot.load(GLOBAL_ORDER) != GLOBAL_UNBOUND => {
                slot.store(val, GLOBAL_ORDER);
                true
            }
            _ => false,
        }
    }

    /// Stable base pointer + length of the dense global mirror (for inline reads).
    /// `AtomicU64` is layout-identical to `u64`, so the native tier reads slots
    /// through this `*const u64` directly.
    pub fn global_slots_ptr(&self) -> *const u64 {
        self.shared.global_slots.as_ptr() as *const u64
    }
    pub fn global_slots_len(&self) -> usize {
        self.shared.global_slots.len()
    }
    pub fn sym_name(&self, s: Sym) -> &str {
        // SAFETY: `s` was interned (< names.len()), the buffer never reallocates
        // (TABLE_CAP), and a `String` at an index is never mutated once written,
        // so this borrow is valid for as long as the `Shared` (and thus `&self`).
        unsafe { (&*self.shared.sym_names.get()).get_unchecked(s as usize) }
    }

    // ── heap ────────────────────────────────────────────────
    pub fn alloc(&self, o: Obj) -> HeapId {
        let _g = self.shared.heap_lock.lock().unwrap();
        self.shared.allocs.fetch_add(1, Ordering::Relaxed);
        // SAFETY: heap mutation is serialized by `heap_lock`; the segmented heap
        // never relocates existing objects on push, so lock-free readers are safe.
        self.heap_mut().push(o) as HeapId
    }

    /// Intern a literal into the constant pool, returning its id. The pool is a
    /// GC root, so the literal survives collection and is rewritten if it moves.
    pub fn intern_const(&self, bits: u64) -> ConstId {
        let _g = self.shared.consts_lock.lock().unwrap();
        // SAFETY: appends serialized by `consts_lock`; reserved buffer never
        // reallocates, so the JIT's inline base-pointer reads stay valid.
        let consts = unsafe { &mut *self.shared.consts.get() };
        assert!(consts.len() < TABLE_CAP, "const pool overflow: raise TABLE_CAP");
        consts.push(bits);
        (consts.len() - 1) as ConstId
    }

    pub fn get_const(&self, id: ConstId) -> u64 {
        // SAFETY: `id` was handed out by `intern_const` (< len); reserved buffer
        // is never reallocated and the slot is written before the id escapes.
        unsafe { *(&*self.shared.consts.get()).get_unchecked(id as usize) }
    }

    /// Base pointer of the constant pool, for a native backend that loads
    /// constants inline instead of through a call. Valid until the pool grows
    /// (which only happens at analyze time, never during execution).
    pub fn consts_ptr(&self) -> *const u64 {
        // SAFETY: reserved buffer, never reallocated; base is stable for a run.
        unsafe { (*self.shared.consts.get()).as_ptr() }
    }

    /// Instrumentation counters (shared across sibling threads).
    pub fn allocs(&self) -> u64 {
        self.shared.allocs.load(Ordering::Relaxed)
    }
    pub fn relocated(&self) -> u64 {
        self.shared.relocated.load(Ordering::Relaxed)
    }
    pub fn freed(&self) -> u64 {
        self.shared.freed.load(Ordering::Relaxed)
    }

    /// Box a non-immediate category, encode an immediate one. THE value-axis
    /// seam: whether Int or Float takes the heap path is the model's call.
    pub fn encode(&mut self, v: Val) -> u64 {
        match v {
            Val::Int(i) => {
                let fixnum = M::R::is_immediate(Cat::Int)
                    && i >= i64::MIN as i128
                    && i <= i64::MAX as i128
                    && M::R::imm_fits(i as i64);
                if fixnum {
                    M::R::enc_int(i as i64)
                } else {
                    let id = self.alloc(Obj::BigInt(i)); // promoted to a boxed bignum
                    M::R::enc_ref(id)
                }
            }
            Val::Float(f) => {
                if M::R::is_immediate(Cat::Float) {
                    M::R::enc_float(f)
                } else {
                    let id = self.alloc(Obj::BoxFloat(f));
                    M::R::enc_ref(id)
                }
            }
            Val::Bool(b) => M::R::enc_bool(b),
            Val::Nil => M::R::enc_nil(),
            Val::Sym(s) => M::R::enc_sym(s),
            Val::Ref(id) => M::R::enc_ref(id),
        }
    }

    pub fn decode(&self, bits: u64) -> Val {
        match M::R::tag_of(bits) {
            RawTag::Int => Val::Int(M::R::imm_int(bits) as i128),
            RawTag::Float => Val::Float(M::R::imm_float(bits)),
            RawTag::Bool => Val::Bool(M::R::as_bool(bits)),
            RawTag::Nil => Val::Nil,
            RawTag::Sym => Val::Sym(M::R::as_sym(bits)),
            RawTag::Ref => {
                let id = M::R::as_ref(bits);
                match &self.heap()[id as usize] {
                    Obj::BigInt(i) => Val::Int(*i),
                    Obj::BoxFloat(f) => Val::Float(*f),
                    Obj::Moved(_) => panic!(
                        "use-after-move: 0x{bits:x} is a stale pointer into from-space; \
                         the collector relocated it — re-read through its root/handle"
                    ),
                    _ => Val::Ref(id),
                }
            }
        }
    }

    /// Allocate a string value (used by the frontend reader for string literals).
    pub fn alloc_str(&mut self, s: String) -> u64 {
        let id = self.alloc(Obj::Str(s));
        M::R::enc_ref(id)
    }
    /// Allocate a character value (used by the frontend reader for `#\c` literals).
    pub fn alloc_char(&mut self, c: char) -> u64 {
        let id = self.alloc(Obj::Char(c));
        M::R::enc_ref(id)
    }

    // ── lists ───────────────────────────────────────────────
    pub fn cons(&mut self, head: u64, tail: u64) -> u64 {
        let id = self.alloc(Obj::Cons { head, tail });
        M::R::enc_ref(id)
    }
    pub fn as_cons(&self, bits: u64) -> Option<(u64, u64)> {
        if let RawTag::Ref = M::R::tag_of(bits) {
            match &self.heap()[M::R::as_ref(bits) as usize] {
                Obj::Cons { head, tail } => return Some((*head, *tail)),
                Obj::Moved(_) => panic!(
                    "use-after-move: 0x{bits:x} is a stale pointer into from-space; \
                     the collector relocated it — re-read through its root/handle"
                ),
                _ => {}
            }
        }
        None
    }
    pub fn list_to_vec(&self, mut bits: u64) -> Vec<u64> {
        let mut out = Vec::new();
        while let Some((h, t)) = self.as_cons(bits) {
            out.push(h);
            bits = t;
        }
        out
    }
    pub fn vec_to_list(&mut self, items: &[u64]) -> u64 {
        let mut tail = self.encode(Val::Nil);
        for &it in items.iter().rev() {
            tail = self.cons(it, tail);
        }
        tail
    }

    // ── equality / compare ──────────────────────────────────
    pub fn equal(&self, a: u64, b: u64) -> bool {
        // Huge integers decode to opaque refs; compare them by value (two equal
        // huge results live at different heap addresses).
        if let (Some(x), Some(y)) = (self.as_huge(a), self.as_huge(b)) {
            return x == y;
        }
        match (self.decode(a), self.decode(b)) {
            (Val::Int(x), Val::Int(y)) => x == y,
            (Val::Float(x), Val::Float(y)) => x == y,
            (Val::Bool(x), Val::Bool(y)) => x == y,
            (Val::Nil, Val::Nil) => true,
            (Val::Sym(x), Val::Sym(y)) => x == y,
            (Val::Ref(x), Val::Ref(y)) => {
                if x == y {
                    return true; // same object
                }
                match (&self.heap()[x as usize], &self.heap()[y as usize]) {
                    (Obj::Cons { .. }, Obj::Cons { .. }) => {
                        let (ha, ta) = self.as_cons(a).unwrap();
                        let (hb, tb) = self.as_cons(b).unwrap();
                        self.equal(ha, hb) && self.equal(ta, tb)
                    }
                    // Structural equality for aggregates (R7RS `equal?` on
                    // strings/vectors; ordered field equality for records — the
                    // general aggregate case). Order-INSENSITIVE collections
                    // (e.g. a hash-map) are a frontend concern layered on top.
                    (Obj::Str(sa), Obj::Str(sb)) => sa == sb,
                    (Obj::Char(ca), Obj::Char(cb)) => ca == cb,
                    (Obj::Vector(va), Obj::Vector(vb)) => {
                        va.len() == vb.len()
                            && va.clone().iter().zip(vb.clone().iter()).all(|(&x, &y)| self.equal(x, y))
                    }
                    (
                        Obj::Record { type_id: ta, fields: fa },
                        Obj::Record { type_id: tb, fields: fb },
                    ) => {
                        ta == tb
                            && fa.len() == fb.len()
                            && fa.clone().iter().zip(fb.clone().iter()).all(|(&x, &y)| self.equal(x, y))
                    }
                    _ => false, // identity already handled; distinct other objects differ
                }
            }
            _ => false,
        }
    }
    fn num_lt(&self, a: u64, b: u64) -> bool {
        // Exact when both are integers of any size; falls back to f64 only when a
        // float is involved.
        if let (Some(x), Some(y)) = (self.as_int_big(a), self.as_int_big(b)) {
            return x.cmp(&y) == std::cmp::Ordering::Less;
        }
        let x = self.num_as_f64(a).expect("< on non-number");
        let y = self.num_as_f64(b).expect("< on non-number");
        x < y
    }

    // ── primitives (value-model fast paths live here) ───────
    pub fn prim(&mut self, op: Prim, args: &[u64]) -> u64 {
        match op {
            // The fixnum-specialized `Fx*` ops are semantically the checked op:
            // interpreter tiers give them identical meaning (only the JIT reads
            // the distinction, to skip a tag check). So they share these arms.
            Prim::Add | Prim::FxAdd => self.arith(args[0], args[1], i64::checked_add, i128::checked_add, |a, b| a + b, BigInt::add),
            Prim::Sub | Prim::FxSub => self.arith(args[0], args[1], i64::checked_sub, i128::checked_sub, |a, b| a - b, BigInt::sub),
            Prim::Mul | Prim::FxMul => self.arith(args[0], args[1], i64::checked_mul, i128::checked_mul, |a, b| a * b, BigInt::mul),
            Prim::Lt | Prim::FxLt => {
                let r = self.num_lt(args[0], args[1]);
                self.encode(Val::Bool(r))
            }
            Prim::Eq | Prim::FxEq => {
                let r = self.equal(args[0], args[1]);
                self.encode(Val::Bool(r))
            }
            // The specializer's fixnum guard: true iff every arg is an immediate
            // fixnum. (Only picks between two equivalent bodies, so its exact
            // value never affects correctness on an interpreter tier.)
            Prim::AllFixnum => {
                let all = M::R::is_immediate(Cat::Int)
                    && args.iter().all(|&a| matches!(self.decode(a), Val::Int(v) if (-(1i128 << 60)..(1i128 << 60)).contains(&v)));
                self.encode(Val::Bool(all))
            }
            // Identity (`eq?`/`eqv?`): equal encoded bits. Immediates compare by
            // value; heap objects by pointer.
            Prim::Identical => {
                let r = args[0] == args[1];
                self.encode(Val::Bool(r))
            }
            Prim::StrLen => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("string-length: not a string");
                };
                let Obj::Str(s) = &self.heap()[id as usize] else {
                    panic!("string-length: not a string");
                };
                let n = s.chars().count() as i128;
                self.encode(Val::Int(n))
            }
            Prim::CharToInt => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("char->integer: not a char");
                };
                let Obj::Char(c) = &self.heap()[id as usize] else {
                    panic!("char->integer: not a char");
                };
                self.encode(Val::Int(*c as i128))
            }
            Prim::IntToChar => {
                let Val::Int(n) = self.decode(args[0]) else {
                    panic!("integer->char: not an integer");
                };
                let c = char::from_u32(n as u32)
                    .unwrap_or_else(|| panic!("integer->char: {n} is not a Unicode scalar value"));
                let id = self.alloc(Obj::Char(c));
                M::R::enc_ref(id)
            }
            Prim::Vector => {
                let id = self.alloc(Obj::Vector(args.to_vec()));
                M::R::enc_ref(id)
            }
            Prim::VectorRef => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("vector-ref: not a vector");
                };
                let Val::Int(i) = self.decode(args[1]) else {
                    panic!("vector-ref: index must be an int");
                };
                let Obj::Vector(elems) = &self.heap()[id as usize] else {
                    panic!("vector-ref: not a vector");
                };
                *elems
                    .get(i as usize)
                    .unwrap_or_else(|| panic!("vector-ref: index {i} out of range"))
            }
            Prim::VectorSet => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("vector-set!: not a vector");
                };
                let Val::Int(i) = self.decode(args[1]) else {
                    panic!("vector-set!: index must be an int");
                };
                let Obj::Vector(elems) = &mut self.heap_mut()[id as usize] else {
                    panic!("vector-set!: not a vector");
                };
                let slot = elems
                    .get_mut(i as usize)
                    .unwrap_or_else(|| panic!("vector-set!: index {i} out of range"));
                *slot = args[2];
                self.enc_nil()
            }
            Prim::VectorLen => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("vector-length: not a vector");
                };
                let Obj::Vector(elems) = &self.heap()[id as usize] else {
                    panic!("vector-length: not a vector");
                };
                self.encode(Val::Int(elems.len() as i128))
            }
            Prim::Values => {
                let id = self.alloc(Obj::Values(args.to_vec()));
                M::R::enc_ref(id)
            }
            Prim::ValuesToList => {
                // Unpack a `values` packet into a list; a lone value becomes a
                // one-element list so single-valued producers work too.
                if let Val::Ref(id) = self.decode(args[0]) {
                    if let Obj::Values(vals) = &self.heap()[id as usize] {
                        let vals = vals.clone();
                        return self.vec_to_list(&vals);
                    }
                }
                self.vec_to_list(&[args[0]])
            }
            Prim::Apply => {
                // `apply` must invoke a closure, which only a backend can do; the
                // CekMachine intercepts it before reaching here.
                panic!("apply requires a backend that can invoke closures (CekMachine)");
            }
            Prim::List => self.vec_to_list(args),
            Prim::Cons => self.cons(args[0], args[1]),
            Prim::First => self.as_cons(args[0]).map(|(h, _)| h).unwrap_or_else(|| self.enc_nil()),
            Prim::Rest => self.as_cons(args[0]).map(|(_, t)| t).unwrap_or_else(|| self.enc_nil()),
            Prim::IsNil => {
                let r = matches!(self.decode(args[0]), Val::Nil);
                self.encode(Val::Bool(r))
            }
            Prim::Println => {
                let s = self.print(args[0]);
                println!("{s}");
                self.enc_nil()
            }
            Prim::Gc => {
                // The collector needs the live environment as a root, which
                // only the backend holds (it is the safepoint). Backends
                // intercept `Gc` in their `Prim` arm and call `collect(locals)`;
                // reaching here means a caller invoked the prim without one.
                panic!("gc must be evaluated at a safepoint with a live environment");
            }
            Prim::CallEc => {
                // Escape continuations need to invoke a closure and catch a
                // non-local exit, which only a backend can do; backends that
                // support it intercept `CallEc` before reaching here.
                panic!("%callec requires a backend that supports escape continuations");
            }
            Prim::CallCc => {
                panic!("%callcc requires the stackless CekMachine (full continuations)");
            }
            Prim::Reset | Prim::Shift => {
                panic!("%reset/%shift require the stackless CekMachine (delimited continuations)");
            }
            Prim::Record => {
                let Val::Sym(type_id) = self.decode(args[0]) else {
                    panic!("record: first arg must be a (quoted) type symbol");
                };
                let fields = args[1..].to_vec();
                let id = self.alloc(Obj::Record { type_id, fields });
                M::R::enc_ref(id)
            }
            Prim::TypeOf => {
                // A record reports its own type; everything else a built-in tag.
                let name = match self.decode(args[0]) {
                    Val::Int(_) => "Long",
                    Val::Float(_) => "Double",
                    Val::Bool(_) => "Boolean",
                    Val::Nil => "nil",
                    Val::Sym(_) => "Symbol",
                    Val::Ref(id) => match &self.heap()[id as usize] {
                        Obj::Record { type_id, .. } => {
                            let s = *type_id;
                            return M::R::enc_sym(s);
                        }
                        Obj::Cons { .. } => "List",
                        Obj::Vector(_) => "Vector",
                        Obj::Str(_) => "String",
                        Obj::Char(_) => "Char",
                        Obj::Closure { .. } => "Fn",
                        Obj::BigInt(_) | Obj::HugeInt(_) => "Long",
                        Obj::BoxFloat(_) => "Double",
                        Obj::Future(_) => "Future",
                        _ => "Object",
                    },
                };
                let s = self.intern(name);
                M::R::enc_sym(s)
            }
            Prim::Throw => {
                // Unwind with the thrown VALUE as the payload, so an enclosing
                // `Ir::Try` can catch it (via catch_unwind + downcast). Uncaught,
                // it aborts like any panic. Neutral: the payload is a raw runtime
                // value, not a Clojure exception object.
                std::panic::panic_any(Thrown { value: args[0] });
            }
            Prim::NFields => {
                let n = match self.decode(args[0]) {
                    Val::Ref(id) => match &self.heap()[id as usize] {
                        Obj::Record { fields, .. } => fields.len() as i128,
                        _ => 0,
                    },
                    _ => 0,
                };
                self.encode(Val::Int(n))
            }
            // Join a future's worker thread and cache its value. No backend
            // needed — this only blocks and reads a shared slot.
            Prim::Await => {
                let id = M::R::as_ref(args[0]);
                let slot = match &self.heap()[id as usize] {
                    Obj::Future(s) => s.clone(),
                    _ => panic!("await: not a future"),
                };
                let mut g = slot.lock().unwrap();
                if let Some(r) = g.result {
                    return r;
                }
                let handle = g.handle.take().expect("future already awaited");
                drop(g); // release the lock before blocking on the join
                let r = handle.join().expect("future thread panicked");
                slot.lock().unwrap().result = Some(r);
                r
            }
            // `spawn` needs to invoke a closure on the child thread, so it is
            // handled in the backend (like `Gc`/`CallEc`), never here.
            Prim::Spawn => unreachable!("Prim::Spawn is backend-handled"),
            Prim::Field => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("field: not a record");
                };
                let Val::Int(i) = self.decode(args[1]) else {
                    panic!("field: index must be an int");
                };
                match &self.heap()[id as usize] {
                    Obj::Record { fields, .. } => fields[i as usize],
                    _ => panic!("field: not a record"),
                }
            }
        }
    }

    fn enc_nil(&self) -> u64 {
        M::R::enc_nil()
    }

    // ── dispatch axis ───────────────────────────────────────
    /// Swap the dispatch strategy. Nothing else changes — the axis is free.
    pub fn set_dispatch(&self, d: Box<dyn Dispatch>) {
        self.shared.tables.lock().unwrap().dispatch = d;
    }
    pub fn dispatch_stats(&self) -> crate::dispatch::DispatchStats {
        self.shared.tables.lock().unwrap().dispatch.stats()
    }
    /// The receiver's type tag (a record's `type_id`). `None` for non-records.
    pub fn type_of(&self, bits: u64) -> Option<Sym> {
        if let Val::Ref(id) = self.decode(bits) {
            if let Obj::Record { type_id, .. } = &self.heap()[id as usize] {
                return Some(*type_id);
            }
        }
        None
    }
    pub fn register_method(&self, name: Sym, ty: Sym, imp: u64) {
        let mut t = self.shared.tables.lock().unwrap();
        t.method_names.insert(name);
        t.methods.insert((name, ty), imp);
    }
    /// Is `name` a registered method (so a frontend should compile `(name recv)`
    /// to a `Dispatch`)? The dispatch axis's compile-time query.
    pub fn is_method_name(&self, name: Sym) -> bool {
        self.shared.tables.lock().unwrap().method_names.contains(&name)
    }
    /// Register the global fn a backend should invoke when a non-closure object
    /// is called (see `apply_fn`). The frontend sets this to a callable-object
    /// dispatcher. Stored as `Sym + 1` (0 = None) in an atomic.
    pub fn set_apply_fn(&self, name: Sym) {
        self.shared.apply_fn.store(name as u64 + 1, Ordering::Relaxed);
    }
    /// The current apply-handler fn value (re-read from globals, so GC-safe), if any.
    pub fn apply_handler(&self) -> Option<u64> {
        match self.shared.apply_fn.load(Ordering::Relaxed) {
            0 => None,
            v => self.global((v - 1) as Sym),
        }
    }
    /// Resolve a call site via the current dispatch strategy (reads registry +
    /// updates the strategy's per-site cache), then invoke happens in the backend.
    /// The impl is copied out and the lock dropped before the caller invokes it.
    pub fn resolve_method(&self, site: usize, method: Sym, ty: Sym) -> Option<u64> {
        let t = self.shared.tables.lock().unwrap();
        t.dispatch.resolve(&t.methods, site, method, ty)
    }

    /// A fresh tag for an escape continuation.
    pub fn fresh_escape_tag(&self) -> u64 {
        self.shared.escape_tags.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// The generic arithmetic path with a numeric tower. The immediate-int fast
    /// path uses a CHECKED op and the model's fixnum-range check; on overflow it
    /// PROMOTES to a boxed `BigInt` (computed in `i128`) instead of wrapping.
    /// That is the value axis's contribution to the tower — the fast path stays
    /// alloc-free for small ints, and big results box automatically.
    ///
    ///   LowBit small ints -> fixnum fast path,     0 allocations
    ///   LowBit big result -> checked op overflows -> promote to boxed BigInt
    ///   NanBox floats      -> float fast path,      0 allocations
    ///   NanBox ints        -> no immediate int, slow path, boxes (BigInt)
    fn arith(
        &mut self,
        a: u64,
        b: u64,
        iop64: fn(i64, i64) -> Option<i64>,
        iop128: fn(i128, i128) -> Option<i128>,
        fop: fn(f64, f64) -> f64,
        bigop: fn(&BigInt, &BigInt) -> BigInt,
    ) -> u64 {
        if M::R::is_immediate(Cat::Int)
            && M::R::tag_of(a) == RawTag::Int
            && M::R::tag_of(b) == RawTag::Int
        {
            let (x, y) = (M::R::imm_int(a), M::R::imm_int(b));
            if let Some(r) = iop64(x, y) {
                if M::R::imm_fits(r) {
                    return M::R::enc_int(r); // stays a fixnum, no allocation
                }
            }
            // Overflowed the fixnum: promote. Stay in i128 if it fits, else go to
            // true arbitrary precision.
            if let Some(r) = iop128(x as i128, y as i128) {
                let id = self.alloc(Obj::BigInt(r));
                return M::R::enc_ref(id);
            }
            let big = bigop(&BigInt::from_i128(x as i128), &BigInt::from_i128(y as i128));
            return self.alloc_bigint(big);
        }
        if M::R::is_immediate(Cat::Float)
            && M::R::tag_of(a) == RawTag::Float
            && M::R::tag_of(b) == RawTag::Float
        {
            return M::R::enc_float(fop(M::R::imm_float(a), M::R::imm_float(b)));
        }
        // Both operands are integers of any size (fixnum, i128-boxed, or huge):
        // do the whole operation in arbitrary precision, staying in i128 when it
        // fits so small results do not carry a BigInt.
        if let (Some(x), Some(y)) = (self.as_int_big(a), self.as_int_big(b)) {
            if let (Some(xi), Some(yi)) = (x.to_i128(), y.to_i128()) {
                if let Some(r) = iop128(xi, yi) {
                    let id = self.alloc(Obj::BigInt(r));
                    return M::R::enc_ref(id);
                }
            }
            return self.alloc_bigint(bigop(&x, &y));
        }
        // A float is involved: compute in f64 (huge ints degrade to an f64).
        let r = match (self.num_as_f64(a), self.num_as_f64(b)) {
            (Some(x), Some(y)) => Val::Float(fop(x, y)),
            _ => panic!("arith on non-numbers"),
        };
        self.encode(r)
    }

    /// The integer value of `bits` as a `BigInt`, if it is any kind of integer
    /// (fixnum, `i128`-boxed, or huge). `None` for non-integers.
    fn as_int_big(&self, bits: u64) -> Option<BigInt> {
        if let Val::Int(i) = self.decode(bits) {
            return Some(BigInt::from_i128(i));
        }
        self.as_huge(bits).cloned()
    }

    /// The `BigInt` behind a `HugeInt` heap value, if `bits` is one.
    fn as_huge(&self, bits: u64) -> Option<&BigInt> {
        if let RawTag::Ref = M::R::tag_of(bits) {
            if let Obj::HugeInt(b) = &self.heap()[M::R::as_ref(bits) as usize] {
                return Some(b);
            }
        }
        None
    }

    /// Any number (int of any size, or float) as an `f64`; `None` if not a number.
    fn num_as_f64(&self, bits: u64) -> Option<f64> {
        match self.decode(bits) {
            Val::Int(i) => Some(i as f64),
            Val::Float(x) => Some(x),
            _ => self.as_huge(bits).map(|b| b.to_f64()),
        }
    }

    /// Store a `BigInt`, normalizing down to a fixnum / `i128` box when it fits so
    /// only genuinely-huge values carry the arbitrary-precision representation.
    fn alloc_bigint(&mut self, b: BigInt) -> u64 {
        if let Some(i) = b.to_i128() {
            return self.encode(Val::Int(i));
        }
        let id = self.alloc(Obj::HugeInt(b));
        M::R::enc_ref(id)
    }

    // ── printing ────────────────────────────────────────────
    pub fn print(&self, bits: u64) -> String {
        match self.decode(bits) {
            Val::Int(i) => i.to_string(),
            Val::Float(f) => {
                if f.is_finite() && f == f.trunc() {
                    format!("{f:.1}")
                } else {
                    format!("{f}")
                }
            }
            Val::Bool(b) => b.to_string(),
            Val::Nil => "nil".to_string(),
            Val::Sym(s) => self.sym_name(s).to_string(),
            Val::Ref(id) => match &self.heap()[id as usize] {
                Obj::Cons { .. } => {
                    let items = self.list_to_vec(bits);
                    let inner: Vec<String> = items.iter().map(|&x| self.print(x)).collect();
                    format!("({})", inner.join(" "))
                }
                Obj::Str(s) => format!("\"{s}\""),
                Obj::Char(c) => c.to_string(),
                Obj::Vector(elems) => {
                    let inner: Vec<String> = elems.iter().map(|&x| self.print(x)).collect();
                    format!("#({})", inner.join(" "))
                }
                Obj::Values(vals) => {
                    let inner: Vec<String> = vals.iter().map(|&x| self.print(x)).collect();
                    inner.join(" ")
                }
                Obj::Closure { .. } => "#<closure>".to_string(),
                Obj::BigInt(i) => i.to_string(),
                Obj::HugeInt(b) => b.to_string(),
                Obj::BoxFloat(f) => format!("{f}"),
                Obj::Record { type_id, fields } => {
                    let inner: Vec<String> = fields.iter().map(|&x| self.print(x)).collect();
                    format!("#{}[{}]", self.sym_name(*type_id), inner.join(" "))
                }
                Obj::Escape { .. } => "#<continuation>".to_string(),
                Obj::Cont(_) => "#<continuation>".to_string(),
                Obj::PartialCont(_) => "#<partial-continuation>".to_string(),
                Obj::Future(_) => "#<future>".to_string(),
                Obj::Moved(_) => "#<moved>".to_string(),
            },
        }
    }

    /// Build a callee's slot frame from evaluated args. Slots `0..nparams` are
    /// the positional args; a variadic rest arg is the slot after them, holding
    /// the collected list. Shared by every backend so the frame layout has one
    /// definition matching what `analyze` assigned.
    pub fn build_call_frame(
        &mut self,
        nparams: usize,
        variadic: bool,
        args: &[u64],
        env: Locals,
    ) -> Locals {
        let mut slots: Vec<AtomicU64> = Vec::with_capacity(nparams + variadic as usize);
        if variadic {
            assert!(
                args.len() >= nparams,
                "arity: expected at least {nparams}, got {}",
                args.len()
            );
            slots.extend(args[..nparams].iter().map(|&a| AtomicU64::new(a)));
            let restlist = self.vec_to_list(&args[nparams..]);
            slots.push(AtomicU64::new(restlist));
        } else {
            assert!(
                args.len() == nparams,
                "arity: expected {nparams}, got {}",
                args.len()
            );
            slots.extend(args.iter().map(|&a| AtomicU64::new(a)));
        }
        Some(Arc::new(Frame { slots, parent: env }))
    }

}
