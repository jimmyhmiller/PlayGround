//! The runtime: heap, symbol table, global environment, reader, macroexpander,
//! analyzer, and the value-model-aware primitives.
//!
//! `encode`/`decode` are the seam where the value axis meets everything else:
//! they box a non-immediate category and unbox on the way out, and `allocs`
//! counts the boxing so the micro-languages can *show* the cost. The reader
//! produces `Val` (code is data); `macroexpand` re-enters compiled code
//! through `CodeSpace::invoke`; `analyze` lowers a macroexpanded `Val` to `Ir`.

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
use crate::code::CodeSpace;
use crate::dispatch::{Dispatch, Megamorphic, MethodRegistry};
use crate::ir::{ConstId, Ir, Prim};
use crate::model::{Repr, ValueModel};
use crate::value::{Cat, Frame, HeapId, Locals, Obj, RawTag, Sym, Val};

/// Sentinel for an unbound slot in `global_slots`. `u64::MAX` has an invalid tag
/// under every value model (`LowBit`/`HighBit`/`NanBox`), so it can never collide
/// with a real encoded value — reading it back means "not bound, use slow path".
pub const GLOBAL_UNBOUND: u64 = u64::MAX;

/// A global binding. `is_macro` is what makes `macroexpand` treat a call head
/// specially — the single resolution table the compiler and runtime share.
pub struct Var {
    pub val: u64,
    pub is_macro: bool,
}

struct Specials {
    quote: Sym,
    if_: Sym,
    do_: Sym,
    def: Sym,
    defmacro: Sym,
    defmethod: Sym,
    fn_: Sym,
    let_: Sym,
    set_: Sym,
    amp: Sym,
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
    sf: Specials,
    prims: HashMap<Sym, Prim>,
    /// Compile-time lexical scope, innermost frame last. Used only during
    /// `analyze` to resolve names to `(up, idx)` slots; empty between forms.
    scope: Vec<Vec<Sym>>,
    // ── dispatch axis ──
    /// `(method, type) -> impl closure`. The source of truth; a GC root.
    pub(crate) methods: MethodRegistry,
    /// Symbols known to be method names, so `analyze` turns `(name recv ..)`
    /// into a `Dispatch` site. Updated when a `defmethod` evaluates.
    method_names: HashSet<Sym>,
    /// The swappable dispatch strategy (megamorphic / mono IC / poly IC).
    pub(crate) dispatch: Box<dyn Dispatch>,
    /// Monotonic call-site id counter, assigned at analyze time.
    next_site: usize,
    /// Monotonic tag for escape continuations.
    escape_tags: u64,
    _pd: PhantomData<fn() -> M>,
}

impl<M: ValueModel> Runtime<M> {
    pub fn new() -> Self {
        let mut rt = Runtime {
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
            sf: Specials {
                quote: 0,
                if_: 0,
                do_: 0,
                def: 0,
                defmacro: 0,
                defmethod: 0,
                fn_: 0,
                let_: 0,
                set_: 0,
                amp: 0,
            },
            prims: HashMap::new(),
            scope: Vec::new(),
            methods: HashMap::new(),
            method_names: HashSet::new(),
            dispatch: Box::new(Megamorphic::new()),
            next_site: 0,
            escape_tags: 0,
            _pd: PhantomData,
        };
        rt.sf = Specials {
            quote: rt.intern("quote"),
            if_: rt.intern("if"),
            do_: rt.intern("do"),
            def: rt.intern("def"),
            defmacro: rt.intern("defmacro"),
            defmethod: rt.intern("defmethod"),
            fn_: rt.intern("fn"),
            let_: rt.intern("let"),
            set_: rt.intern("set!"),
            amp: rt.intern("&"),
        };
        use Prim::*;
        for (name, p) in [
            ("+", Add),
            ("-", Sub),
            ("*", Mul),
            ("<", Lt),
            ("=", Eq),
            ("list", List),
            ("cons", Cons),
            ("first", First),
            ("rest", Rest),
            ("nil?", IsNil),
            ("println", Println),
            ("gc", Gc),
            ("record", Record),
            ("field", Field),
            ("%eq", Identical),
            // Binary aliases so arithmetic/comparison operators can be given
            // first-class prelude bindings (the surface `+`/`<`/… fold in head
            // position, but must also be callable values, e.g. `(apply + xs)`).
            ("%add", Add),
            ("%sub", Sub),
            ("%mul", Mul),
            ("%lt", Lt),
            ("%num-eq", Eq),
            ("string-length", StrLen),
            ("char->integer", CharToInt),
            ("integer->char", IntToChar),
            ("vector", Vector),
            ("vector-ref", VectorRef),
            ("vector-set!", VectorSet),
            ("vector-length", VectorLen),
            ("values", Values),
            ("%values->list", ValuesToList),
            ("apply", Apply),
            ("%callec", CallEc),
            ("%callcc", CallCc),
            ("%reset", Reset),
            ("%shift", Shift),
        ] {
            let s = rt.intern(name);
            rt.prims.insert(s, p);
        }
        rt
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
    pub fn define_global(&mut self, sym: Sym, val: u64, is_macro: bool) {
        self.globals.insert(sym, Var { val, is_macro });
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
            (Val::Ref(_), Val::Ref(_)) => match (self.as_cons(a), self.as_cons(b)) {
                (Some((ha, ta)), Some((hb, tb))) => self.equal(ha, hb) && self.equal(ta, tb),
                _ => a == b, // identity for other objects
            },
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
    /// Resolve a call site via the current dispatch strategy (reads registry +
    /// updates the strategy's per-site cache), then invoke happens in the backend.
    pub fn resolve_method(&self, site: usize, method: Sym, ty: Sym) -> Option<u64> {
        self.dispatch.resolve(&self.methods, site, method, ty)
    }
    fn fresh_site(&mut self) -> usize {
        let s = self.next_site;
        self.next_site += 1;
        s
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

    // ── reader: &str -> Vec<Val> (code is data) ─────────────
    pub fn read_all(&mut self, src: &str) -> Vec<u64> {
        let toks = tokenize(src);
        let mut p = 0;
        let mut out = Vec::new();
        while p < toks.len() {
            let (v, np) = self.read_form(&toks, p);
            out.push(v);
            p = np;
        }
        out
    }

    fn read_form(&mut self, toks: &[Tok], p: usize) -> (u64, usize) {
        match &toks[p] {
            Tok::LParen => {
                let mut items = Vec::new();
                let mut q = p + 1;
                while !matches!(toks.get(q), Some(Tok::RParen)) {
                    if q >= toks.len() {
                        panic!("unbalanced (");
                    }
                    let (v, nq) = self.read_form(toks, q);
                    items.push(v);
                    q = nq;
                }
                (self.vec_to_list(&items), q + 1)
            }
            Tok::RParen => panic!("unexpected )"),
            Tok::Quote => {
                let (v, nq) = self.read_form(toks, p + 1);
                let qs = self.sf.quote;
                let qsym = self.encode(Val::Sym(qs));
                let inner = self.vec_to_list(&[qsym, v]);
                (inner, nq)
            }
            Tok::Atom(a) => (self.read_atom(a), p + 1),
        }
    }

    fn read_atom(&mut self, a: &str) -> u64 {
        let v = if a == "nil" {
            Val::Nil
        } else if a == "true" {
            Val::Bool(true)
        } else if a == "false" {
            Val::Bool(false)
        } else if let Ok(i) = a.parse::<i128>() {
            Val::Int(i)
        } else if let Ok(f) = a.parse::<f64>() {
            Val::Float(f)
        } else {
            Val::Sym(self.intern(a))
        };
        self.encode(v)
    }

    // ── macroexpand: re-enters compiled code mid-compile ────
    //
    // The backend `cs` is passed as a value, not baked into `Runtime`'s type.
    // `cs` and `self` are disjoint borrows, so `cs.invoke(self, ...)` type-
    // checks with any backend — including a wrapping one (see `code::Traced`).
    pub fn macroexpand(&mut self, cs: &dyn CodeSpace<M>, form: u64) -> u64 {
        // THE FUSION POINT. `invoke` runs a macro mid-compilation; the macro may
        // allocate and trigger a MOVING collection, which relocates the form we
        // are expanding. `form` is a bare `u64` the GC cannot see or update, so
        // we publish it to the shadow stack and re-read it through that root
        // after every `invoke` (`self.shadow[slot]` is rewritten to the new
        // address). Using the stale bare `form` instead is the clojure-jvm
        // form-609 corruption — now a use-after-move, not silent.
        let slot = self.push_root(form);
        loop {
            let f = self.shadow[slot];
            let Some((head, _)) = self.as_cons(f) else { break };
            let Val::Sym(hs) = self.decode(head) else { break };
            let (is_macro, mfn) = match self.globals.get(&hs) {
                Some(v) => (v.is_macro, v.val),
                None => break,
            };
            if !is_macro {
                break;
            }
            // Args are sublists of the rooted form (kept alive + forwarded);
            // `invoke` binds them into the macro's call frame before the macro
            // body runs, so they are read while still valid.
            let args = self.list_to_vec(self.shadow[slot]);
            let result = cs.invoke(cs, self, mfn, &args[1..]);
            self.shadow[slot] = result; // re-root to the expansion
        }
        let out = self.shadow[slot];
        self.pop_root();
        out
    }

    // ── analyze: macroexpanded Val -> Ir ────────────────────
    //
    // The compiler-side handle discipline: `analyze` roots the form for the
    // whole of its work, and every child access (`child`) re-derives from that
    // root rather than caching a bare pointer. So a moving GC inside a nested
    // macro relocates the form, updates the root, and sibling children are read
    // at their new addresses. Caching `list_to_vec(form)` up front (as the code
    // did before this collector) is precisely the form-609 stale-pointer bug.
    pub fn analyze(&mut self, cs: &dyn CodeSpace<M>, form: u64) -> Ir {
        let slot = self.push_root(form);
        self.shadow[slot] = self.macroexpand(cs, self.shadow[slot]);
        let r = match self.decode(self.shadow[slot]) {
            Val::Int(_) | Val::Float(_) | Val::Bool(_) | Val::Nil => {
                let f = self.shadow[slot];
                Ir::Const(self.intern_const(f))
            }
            Val::Sym(s) => match self.resolve_local(s) {
                Some((up, idx)) => Ir::Local { up, idx },
                None => Ir::Global(s),
            },
            Val::Ref(_) => {
                if self.as_cons(self.shadow[slot]).is_some() {
                    self.analyze_list(cs, slot)
                } else {
                    let f = self.shadow[slot];
                    Ir::Const(self.intern_const(f))
                }
            }
        };
        self.shadow.truncate(slot);
        r
    }

    /// Re-derive the k-th child of the rooted form. Cheap (forms are short) and
    /// always current after a relocation.
    fn child(&self, slot: usize, k: usize) -> u64 {
        self.list_to_vec(self.shadow[slot])[k]
    }
    fn child_count(&self, slot: usize) -> usize {
        self.list_to_vec(self.shadow[slot]).len()
    }

    fn analyze_list(&mut self, cs: &dyn CodeSpace<M>, slot: usize) -> Ir {
        let head = self.child(slot, 0);
        if let Val::Sym(hs) = self.decode(head) {
            if hs == self.sf.quote {
                let q = self.child(slot, 1);
                return Ir::Quote(self.intern_const(q));
            }
            if hs == self.sf.if_ {
                let c1 = self.child(slot, 1);
                let c = self.analyze(cs, c1);
                let t1 = self.child(slot, 2);
                let t = self.analyze(cs, t1);
                let e = if self.child_count(slot) > 3 {
                    let e1 = self.child(slot, 3);
                    self.analyze(cs, e1)
                } else {
                    let nil = self.enc_nil();
                    Ir::Const(self.intern_const(nil))
                };
                return Ir::If(Box::new(c), Box::new(t), Box::new(e));
            }
            if hs == self.sf.do_ {
                let n = self.child_count(slot);
                let mut body = Vec::new();
                for k in 1..n {
                    let f = self.child(slot, k);
                    body.push(self.analyze(cs, f));
                }
                return Ir::Do(body);
            }
            if hs == self.sf.def {
                let Val::Sym(name) = self.decode(self.child(slot, 1)) else {
                    panic!("def: name must be a symbol");
                };
                let i1 = self.child(slot, 2);
                let init = self.analyze(cs, i1);
                return Ir::Def {
                    name,
                    init: Box::new(init),
                    is_macro: false,
                };
            }
            if hs == self.sf.defmacro {
                let Val::Sym(name) = self.decode(self.child(slot, 1)) else {
                    panic!("defmacro: name must be a symbol");
                };
                let lam = self.analyze_fn(cs, slot, 2, 3);
                return Ir::Def {
                    name,
                    init: Box::new(lam),
                    is_macro: true,
                };
            }
            if hs == self.sf.defmethod {
                let Val::Sym(name) = self.decode(self.child(slot, 1)) else {
                    panic!("defmethod: method name must be a symbol");
                };
                let Val::Sym(ty) = self.decode(self.child(slot, 2)) else {
                    panic!("defmethod: type must be a symbol");
                };
                let imp_form = self.child(slot, 3);
                let imp = self.analyze(cs, imp_form);
                return Ir::DefMethod {
                    name,
                    ty,
                    imp: Box::new(imp),
                };
            }
            if hs == self.sf.fn_ {
                return self.analyze_fn(cs, slot, 1, 2);
            }
            if hs == self.sf.let_ {
                return self.analyze_let(cs, slot);
            }
            if hs == self.sf.set_ {
                // (set! name val) — assign an existing local or global binding.
                let Val::Sym(name) = self.decode(self.child(slot, 1)) else {
                    panic!("set!: target must be a symbol");
                };
                let vform = self.child(slot, 2);
                let val = Box::new(self.analyze(cs, vform));
                return match self.resolve_local(name) {
                    Some((up, idx)) => Ir::SetLocal { up, idx, val },
                    None => Ir::SetGlobal { name, val },
                };
            }
            if let Some(&p) = self.prims.get(&hs) {
                let n = self.child_count(slot);
                let mut a = Vec::new();
                for k in 1..n {
                    let f = self.child(slot, k);
                    a.push(self.analyze(cs, f));
                }
                return Ir::Prim(p, a);
            }
            // A registered method name -> a polymorphic dispatch site. Checked
            // after specials/prims so those cannot be shadowed by a method.
            if self.method_names.contains(&hs) {
                let site = self.fresh_site();
                let n = self.child_count(slot);
                let mut a = Vec::new();
                for k in 1..n {
                    let f = self.child(slot, k);
                    a.push(self.analyze(cs, f));
                }
                return Ir::Dispatch {
                    site,
                    method: hs,
                    args: a,
                };
            }
        }
        let h = self.child(slot, 0);
        let f = self.analyze(cs, h);
        let n = self.child_count(slot);
        let mut a = Vec::new();
        for k in 1..n {
            let c = self.child(slot, k);
            a.push(self.analyze(cs, c));
        }
        Ir::Call(Box::new(f), a)
    }

    fn analyze_let(&mut self, cs: &dyn CodeSpace<M>, slot: usize) -> Ir {
        // Root the binding list too; its inits can macroexpand-and-GC.
        let binds_form = self.child(slot, 1);
        let bslot = self.push_root(binds_form);
        self.scope.push(Vec::new());
        let mut inits = Vec::new();
        let mut i = 0;
        while i + 1 < self.list_to_vec(self.shadow[bslot]).len() {
            let bl = self.list_to_vec(self.shadow[bslot]);
            let Val::Sym(s) = self.decode(bl[i]) else {
                panic!("let: binding name must be a symbol");
            };
            let initform = bl[i + 1];
            inits.push(self.analyze(cs, initform));
            self.scope.last_mut().unwrap().push(s);
            i += 2;
        }
        self.shadow.truncate(bslot);
        let n = self.child_count(slot);
        let mut body = Vec::new();
        for k in 2..n {
            let f = self.child(slot, k);
            body.push(self.analyze(cs, f));
        }
        self.scope.pop();
        Ir::Let(inits, Box::new(Ir::Do(body)))
    }

    /// Resolve a name to `(up, idx)` against the compile-time scope, innermost
    /// frame first (so inner bindings shadow outer). `None` => it is a global.
    fn resolve_local(&self, sym: Sym) -> Option<(u16, u16)> {
        for (up, frame) in self.scope.iter().rev().enumerate() {
            if let Some(idx) = frame.iter().rposition(|&s| s == sym) {
                return Some((up as u16, idx as u16));
            }
        }
        None
    }

    /// Analyze an `fn`/`defmacro` body (children `body_start..` of the rooted
    /// form at `slot`, params at child `params_k`) under a fresh param frame.
    fn analyze_fn(
        &mut self,
        cs: &dyn CodeSpace<M>,
        slot: usize,
        params_k: usize,
        body_start: usize,
    ) -> Ir {
        let params_form = self.child(slot, params_k);
        let (params, variadic) = self.parse_params(params_form);
        let nparams = params.len();
        let mut frame = params;
        if let Some(rest) = variadic {
            frame.push(rest);
        }
        self.scope.push(frame);
        let n = self.child_count(slot);
        let mut body = Vec::new();
        for k in body_start..n {
            let f = self.child(slot, k);
            body.push(self.analyze(cs, f));
        }
        self.scope.pop();
        Ir::Lambda {
            nparams,
            variadic: variadic.is_some(),
            body: std::rc::Rc::new(Ir::Do(body)),
        }
    }

    fn parse_params(&self, form: u64) -> (Vec<Sym>, Option<Sym>) {
        let items = self.list_to_vec(form);
        let mut params = Vec::new();
        let mut variadic = None;
        let mut i = 0;
        while i < items.len() {
            let Val::Sym(s) = self.decode(items[i]) else {
                panic!("param must be a symbol");
            };
            if s == self.sf.amp {
                if let Some(&rest) = items.get(i + 1) {
                    let Val::Sym(r) = self.decode(rest) else {
                        panic!("variadic param must be a symbol");
                    };
                    variadic = Some(r);
                }
                break;
            }
            params.push(s);
            i += 1;
        }
        (params, variadic)
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

    // ── top-level eval ──────────────────────────────────────
    pub fn eval_top(&mut self, cs: &dyn CodeSpace<M>, form: u64) -> u64 {
        let ir = self.analyze(cs, form);
        cs.eval_ir(cs, self, &ir, &None)
    }

    pub fn eval_str(&mut self, cs: &dyn CodeSpace<M>, src: &str) -> u64 {
        let forms = self.read_all(src);
        // Root the whole read buffer: a GC during an earlier form must not
        // reclaim (nor leave a stale pointer to) later, not-yet-analyzed source.
        // Same discipline as the macro case, one level out. The forms are read
        // BY VALUE into `eval_top`, which re-reads through the shadow slot, so
        // relocation is handled.
        let base = self.shadow.len();
        self.shadow.extend(forms.iter().copied());
        let mut last = self.enc_nil();
        for k in 0..forms.len() {
            let f = self.shadow[base + k];
            last = self.eval_top(cs, f);
        }
        self.shadow.truncate(base);
        last
    }
}

// ── tokenizer ───────────────────────────────────────────────
enum Tok {
    LParen,
    RParen,
    Quote,
    Atom(String),
}

fn tokenize(src: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            c if c.is_whitespace() => i += 1,
            ';' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '(' | '[' => {
                out.push(Tok::LParen);
                i += 1;
            }
            ')' | ']' => {
                out.push(Tok::RParen);
                i += 1;
            }
            '\'' => {
                out.push(Tok::Quote);
                i += 1;
            }
            _ => {
                let start = i;
                while i < chars.len()
                    && !chars[i].is_whitespace()
                    && !matches!(chars[i], '(' | ')' | '[' | ']' | '\'' | ';')
                {
                    i += 1;
                }
                out.push(Tok::Atom(chars[start..i].iter().collect()));
            }
        }
    }
    out
}
