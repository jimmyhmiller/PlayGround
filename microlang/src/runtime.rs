//! The runtime CORE: heap, moving GC roots, symbol table, global environment,
//! the dispatch axis, and the value-model-aware primitives. It knows NOTHING
//! about s-expressions, special forms, or `analyze` — those live in the optional
//! `sexpr` frontend (or a frontend compiles to `Ir` directly).
//!
//! `encode`/`decode` are the seam where the value axis meets everything else:
//! they box a non-immediate category and unbox on the way out, and `allocs`
//! counts the boxing so the micro-languages can *show* the cost.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
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
use std::marker::PhantomData;
use std::rc::Rc;

use crate::bigint::BigInt;
use crate::dispatch::{Dispatch, Megamorphic, MethodRegistry};
use crate::ir::{ConstId, Prim};
use crate::model::{Repr, ValueModel};
use crate::value::{Cat, Frame, HeapId, Locals, Obj, RawTag, Sym, Val};

/// Sentinel for an unbound slot in `global_slots`. `u64::MAX` has an invalid tag
/// under every value model (`LowBit`/`HighBit`/`NanBox`), so it can never collide
/// with a real encoded value — reading it back means "not bound, use slow path".
pub const GLOBAL_UNBOUND: u64 = u64::MAX;

/// A global binding — the single resolution table the compiler and runtime share.
pub struct Var {
    pub val: u64,
}

/// The panic payload of a `(throw v)`: the thrown runtime value, carried up the
/// stack until an `Ir::Try` catches it. Neutral control-flow, not Clojure-specific.
pub struct Thrown {
    pub value: u64,
}

pub struct Runtime<M: ValueModel> {
    pub heap: Vec<Obj>,
    /// Count of heap allocations. Instrumentation so the value axis is visible.
    pub allocs: u64,
    /// Objects relocated (copied to to-space) by the moving collector so far.
    pub relocated: u64,
    /// Objects reclaimed (not carried forward) by the collector so far.
    pub freed: u64,
    /// The constant pool: literal/quoted values referenced by `Ir` via a
    /// `ConstId`. A GC root the collector rewrites — this indirection is why
    /// `Ir` holds no heap pointers a moving collector could not relocate.
    pub(crate) consts: Vec<u64>,
    /// The shadow stack: the GC root set for transient mutator/compiler values.
    /// `push_root`/`pop_root` (LIFO) manage it; the collector rewrites entries
    /// here to the relocated address, which is what a `Handle` re-reads.
    pub(crate) shadow: Vec<u64>,
    sym_names: Vec<String>,
    sym_ids: HashMap<String, Sym>,
    pub globals: SymMap<Var>,
    /// A dense mirror of global VALUES, indexed by `Sym`, for O(1) pointer-based
    /// reads. `intern` extends it (so it is always sym-sized and only grows at
    /// analyze time, never during a run), and `define_global`/`set_global_val`
    /// keep it current. A native backend loads a global inline from this array's
    /// stable base instead of an FFI into the hash map. Slots default to
    /// `GLOBAL_UNBOUND` (a bit pattern no value model can produce), which the JIT
    /// treats as "fall back to the slow, late-binding lookup".
    pub global_slots: Vec<u64>,
    // ── dispatch axis ──
    /// `(method, type) -> impl closure`. The source of truth; a GC root.
    pub(crate) methods: MethodRegistry,
    /// Symbols known to be method names, so a frontend turns `(name recv ..)`
    /// into a `Dispatch` site. Updated when a `defmethod` evaluates.
    method_names: HashSet<Sym>,
    /// The swappable dispatch strategy (megamorphic / mono IC / poly IC).
    pub(crate) dispatch: Box<dyn Dispatch>,
    /// Optional "apply handler": a frontend may name a global fn that the backend
    /// invokes when a NON-closure heap object is called (like Python `__call__`),
    /// e.g. to make Clojure keywords callable. Stored as a `Sym` (re-read from
    /// `globals`) so it is GC-safe. `None` => calling a non-closure is an error.
    apply_fn: Option<Sym>,
    /// Monotonic tag for escape continuations.
    escape_tags: u64,
    _pd: PhantomData<fn() -> M>,
}

impl<M: ValueModel> Runtime<M> {
    pub fn new() -> Self {
        Runtime {
            heap: Vec::new(),
            allocs: 0,
            relocated: 0,
            freed: 0,
            consts: Vec::new(),
            shadow: Vec::new(),
            sym_names: Vec::new(),
            sym_ids: HashMap::new(),
            globals: Default::default(),
            global_slots: Vec::new(),
            methods: HashMap::new(),
            method_names: HashSet::new(),
            dispatch: Box::new(Megamorphic::new()),
            apply_fn: None,
            escape_tags: 0,
            _pd: PhantomData,
        }
    }

    // ── symbols ─────────────────────────────────────────────
    pub fn intern(&mut self, s: &str) -> Sym {
        if let Some(&id) = self.sym_ids.get(s) {
            return id;
        }
        let id = self.sym_names.len() as Sym;
        self.sym_names.push(s.to_string());
        self.sym_ids.insert(s.to_string(), id);
        // Keep the dense global mirror sym-sized. Growing it here (analyze time)
        // means it never reallocates during a run, so the native backend can hold
        // a stable base pointer to it across a whole call chain.
        self.global_slots.push(GLOBAL_UNBOUND);
        id
    }

    /// Define (or redefine) a global, updating both the map and the dense mirror.
    /// The one place a global binding is created; routing writes through here is
    /// what keeps `global_slots` a faithful cache.
    pub fn define_global(&mut self, sym: Sym, val: u64) {
        self.globals.insert(sym, Var { val });
        if let Some(slot) = self.global_slots.get_mut(sym as usize) {
            *slot = val;
        }
    }

    /// Assign an existing global's value (`set!`), updating the dense mirror too.
    /// Returns `false` if the global is unbound.
    pub fn set_global_val(&mut self, sym: Sym, val: u64) -> bool {
        match self.globals.get_mut(&sym) {
            Some(var) => {
                var.val = val;
                if let Some(slot) = self.global_slots.get_mut(sym as usize) {
                    *slot = val;
                }
                true
            }
            None => false,
        }
    }

    /// Stable base pointer + length of the dense global mirror (for inline reads).
    pub fn global_slots_ptr(&self) -> *const u64 {
        self.global_slots.as_ptr()
    }
    pub fn global_slots_len(&self) -> usize {
        self.global_slots.len()
    }
    pub fn sym_name(&self, s: Sym) -> &str {
        &self.sym_names[s as usize]
    }

    // ── heap ────────────────────────────────────────────────
    pub fn alloc(&mut self, o: Obj) -> HeapId {
        self.allocs += 1;
        self.heap.push(o);
        (self.heap.len() - 1) as HeapId
    }

    /// Intern a literal into the constant pool, returning its id. The pool is a
    /// GC root, so the literal survives collection and is rewritten if it moves.
    pub fn intern_const(&mut self, bits: u64) -> ConstId {
        self.consts.push(bits);
        (self.consts.len() - 1) as ConstId
    }

    pub fn get_const(&self, id: ConstId) -> u64 {
        self.consts[id as usize]
    }

    /// Base pointer of the constant pool, for a native backend that loads
    /// constants inline instead of through a call. Valid until the pool grows
    /// (which only happens at analyze time, never during execution).
    pub fn consts_ptr(&self) -> *const u64 {
        self.consts.as_ptr()
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
                match &self.heap[id as usize] {
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
            match &self.heap[M::R::as_ref(bits) as usize] {
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
                match (&self.heap[x as usize], &self.heap[y as usize]) {
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
                let Obj::Str(s) = &self.heap[id as usize] else {
                    panic!("string-length: not a string");
                };
                let n = s.chars().count() as i128;
                self.encode(Val::Int(n))
            }
            Prim::CharToInt => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("char->integer: not a char");
                };
                let Obj::Char(c) = &self.heap[id as usize] else {
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
                let Obj::Vector(elems) = &self.heap[id as usize] else {
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
                let Obj::Vector(elems) = &mut self.heap[id as usize] else {
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
                let Obj::Vector(elems) = &self.heap[id as usize] else {
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
                    if let Obj::Values(vals) = &self.heap[id as usize] {
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
                    Val::Ref(id) => match &self.heap[id as usize] {
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
                    Val::Ref(id) => match &self.heap[id as usize] {
                        Obj::Record { fields, .. } => fields.len() as i128,
                        _ => 0,
                    },
                    _ => 0,
                };
                self.encode(Val::Int(n))
            }
            Prim::Field => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("field: not a record");
                };
                let Val::Int(i) = self.decode(args[1]) else {
                    panic!("field: index must be an int");
                };
                match &self.heap[id as usize] {
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
    pub fn set_dispatch(&mut self, d: Box<dyn Dispatch>) {
        self.dispatch = d;
    }
    pub fn dispatch_stats(&self) -> crate::dispatch::DispatchStats {
        self.dispatch.stats()
    }
    /// The receiver's type tag (a record's `type_id`). `None` for non-records.
    pub fn type_of(&self, bits: u64) -> Option<Sym> {
        if let Val::Ref(id) = self.decode(bits) {
            if let Obj::Record { type_id, .. } = &self.heap[id as usize] {
                return Some(*type_id);
            }
        }
        None
    }
    pub fn register_method(&mut self, name: Sym, ty: Sym, imp: u64) {
        self.method_names.insert(name);
        self.methods.insert((name, ty), imp);
    }
    /// Is `name` a registered method (so a frontend should compile `(name recv)`
    /// to a `Dispatch`)? The dispatch axis's compile-time query.
    pub fn is_method_name(&self, name: Sym) -> bool {
        self.method_names.contains(&name)
    }
    /// Register the global fn a backend should invoke when a non-closure object
    /// is called (see `apply_fn`). The frontend sets this to a callable-object
    /// dispatcher.
    pub fn set_apply_fn(&mut self, name: Sym) {
        self.apply_fn = Some(name);
    }
    /// The current apply-handler fn value (re-read from `globals`, so GC-safe), if any.
    pub fn apply_handler(&self) -> Option<u64> {
        self.apply_fn.and_then(|s| self.globals.get(&s).map(|v| v.val))
    }
    /// Resolve a call site via the current dispatch strategy (reads registry +
    /// updates the strategy's per-site cache), then invoke happens in the backend.
    pub fn resolve_method(&self, site: usize, method: Sym, ty: Sym) -> Option<u64> {
        self.dispatch.resolve(&self.methods, site, method, ty)
    }

    /// A fresh tag for an escape continuation.
    pub fn fresh_escape_tag(&mut self) -> u64 {
        self.escape_tags += 1;
        self.escape_tags
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
            if let Obj::HugeInt(b) = &self.heap[M::R::as_ref(bits) as usize] {
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
            Val::Ref(id) => match &self.heap[id as usize] {
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
        let mut slots: Vec<Cell<u64>> = Vec::with_capacity(nparams + variadic as usize);
        if variadic {
            assert!(
                args.len() >= nparams,
                "arity: expected at least {nparams}, got {}",
                args.len()
            );
            slots.extend(args[..nparams].iter().map(|&a| Cell::new(a)));
            let restlist = self.vec_to_list(&args[nparams..]);
            slots.push(Cell::new(restlist));
        } else {
            assert!(
                args.len() == nparams,
                "arity: expected {nparams}, got {}",
                args.len()
            );
            slots.extend(args.iter().map(|&a| Cell::new(a)));
        }
        Some(Rc::new(Frame { slots, parent: env }))
    }

}
