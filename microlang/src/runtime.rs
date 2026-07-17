//! The runtime CORE: heap, moving GC roots, symbol table, global environment,
//! the dispatch axis, and the value-model-aware primitives. It knows NOTHING
//! about s-expressions, special forms, or `analyze` — those live in the optional
//! `sexpr` frontend (or a frontend compiles to `Ir` directly).
//!
//! `encode`/`decode` are the seam where the value axis meets everything else:
//! they box a non-immediate category and unbox on the way out, and `allocs`
//! counts the boxing so the micro-languages can *show* the cost.

use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::collections::{HashMap, HashSet};

/// Memory ordering for the global-slot array. A global's VALUE is often a heap
/// object (a closure) that a sibling thread reads through this slot, so the store
/// must publish the allocation (`Release`) and the load must acquire it
/// (`Acquire`) — otherwise a reader can observe the id before the object's writes
/// are visible (a real cross-thread use-before-init, not just a TSan report).
const GLOBAL_LOAD: Ordering = Ordering::Acquire;
const GLOBAL_STORE: Ordering = Ordering::Release;
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

/// A TCP handle owned by the `%tcp-*` prims: a listener or a connected stream.
/// `Arc` so blocking I/O runs on a clone OUTSIDE the registry lock.
#[derive(Clone)]
pub(crate) enum TcpHandle {
    Listener(Arc<std::net::TcpListener>),
    Stream(Arc<std::net::TcpStream>),
}
use crate::ir::{ConstId, Prim};
use crate::model::{Repr, ValueModel};
use crate::heap::Gc;
use crate::value::{Cat, Frame, Locals, Obj, RawTag, Sym, Val};

/// Sentinel for an unbound slot in `global_slots`. `u64::MAX` has an invalid tag
/// under every value model (`LowBit`/`HighBit`/`NanBox`), so it can never collide
/// with a real encoded value — reading it back means "not bound, use slow path".
pub const GLOBAL_UNBOUND: u64 = u64::MAX;

/// Non-local control-flow signal for one thread. A `(throw v)` or an escape SETS
/// this and returns a dummy value; every eval site checks `pending()` and
/// short-circuits, bubbling the value up until an `Ir::Try` / `%callec` handles
/// it. This replaces the old Rust-panic-based control flow: it is a plain branch
/// (fast), correct under `panic = "abort"`, needs no global panic-hook swap, and
/// the JIT lowers it natively (a load + conditional return after each call).
///
/// `#[repr(C)]` with `kind` first so the JIT can read the flag at
/// `offset_of!(Runtime, signal)` with an `i8` load.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Signal {
    /// 0 = none, 1 = throw, 2 = escape.
    pub kind: u8,
    pub value: u64,
    /// The escape-continuation tag (only meaningful when `kind == 2`).
    pub tag: u64,
}

/// A HAMT trie node's kind, classified by interned-sym compare.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum HamtKind {
    Bitmap,
    Array,
    Collision,
    Other,
}

/// A read VIEW of a heap object: the enum shape the old `Vec<Obj>` heap used
/// to store, reconstructed as borrows from the raw object (`Runtime::view`).
/// Slices borrow the heap — valid until the next collection (same discipline
/// as every raw `u64` value word: re-read through a root across safepoints).
pub enum ObjView<'h> {
    Cons { head: u64, tail: u64 },
    EmptyList,
    Str(&'h str),
    Char(char),
    /// A growable array: `elems` is the LIVE prefix (logical length) of the
    /// data blob; `gc` is the handle (identity, for mutation); `cap` the
    /// blob's capacity.
    Vector { elems: &'h [u64], gc: Gc, cap: u32 },
    Values(&'h [u64]),
    BigInt(i128),
    /// Reassembled from the inline sign + limbs (cold path by construction).
    HugeInt(BigInt),
    Ratio(i128, i128),
    BoxFloat(f64),
    Closure { nparams: usize, variadic: bool, nslots: u16, template: u32, gc: Gc },
    MultiFn { fixed: &'h [u64], variadic: Option<(usize, u64)> },
    Record { type_id: Sym, fields: &'h [u64] },
    Escape { tag: u64 },
    Cont(Arc<crate::cek::Kont>),
    PartialCont(Arc<crate::cek::Kont>),
    /// The atom's slot word, CAS-able in place.
    Atom(&'h AtomicU64),
    Future(Arc<std::sync::Mutex<crate::value::FutureSlot>>),
}

/// Reserved capacity for the append-only, stable-base tables (`consts`,
/// `sym_names`, `global_slots`). They must never reallocate, because concurrent
/// readers hold raw/unsafe references into them (the JIT reads `consts`/globals
/// through a base pointer; `sym_name` returns a borrowed `&str`). Growth past
/// this panics loudly rather than dangling — a clear error, not UB.
const TABLE_CAP: usize = 1 << 20;

/// Reserved capacity for the per-site field inline-cache (`field_ic`). One slot
/// per `.-field` call site in the whole program; a stable base so the cache is
/// lock-free. `u64::MAX` is the empty sentinel (its high 32 bits, `0xffffffff`,
/// exceed any real type `Sym` (< `TABLE_CAP`), so it never false-hits).
const FIELD_SITE_CAP: usize = 1 << 17;
const FIELD_IC_EMPTY: u64 = u64::MAX;

/// Per-mutator-thread state the STW collector needs: where a parked thread
/// publishes its GC roots. When a thread reaches a safepoint during a GC request
/// it copies its shadow stack into `roots` and its live environment into `env`,
/// flips `parked`, and waits; the collector rewrites `roots` in place (and the
/// `env`'s frame cells) and the thread copies the rewritten roots back on resume.
pub(crate) struct MutatorState {
    pub(crate) roots: Mutex<Vec<u64>>,
    /// The thread's whole DYNAMIC activation chain of environments (one per live
    /// function call), not just the innermost — a moving collector must trace
    /// every live frame, and a deep caller's env is not reachable from the
    /// callee's lexical parent chain.
    pub(crate) envs: Mutex<Vec<Locals>>,
    /// This thread's dynamic-var binding stack, published when it parks so a
    /// moving collector traces (and rewrites) the bound values. `(Sym, value)`;
    /// `Sym == u32::MAX` is a `binding`-scope delimiter (see `DYN_MARK`).
    pub(crate) dyn_roots: Mutex<Vec<(Sym, u64)>>,
    /// The live CEK continuation, when this thread parked from the CEK step
    /// loop — its `done` cells and captured frames are roots (Stage E: the
    /// CEK now parks/collects at its step safepoint, so the kont must ride
    /// along like the env chain does).
    pub(crate) kont: Mutex<Option<Arc<crate::cek::Kont>>>,
    /// NATIVE root slot addresses (Stage E): the addresses of this thread's
    /// JIT stack-map spill slots, published by the frame walker when the
    /// thread parks inside native frames. The collector rewrites them in
    /// place; emitted code reloads them after its call returns.
    pub(crate) native_roots: Mutex<Vec<usize>>,
    pub(crate) parked: std::sync::atomic::AtomicBool,
}

/// The installed NATIVE frame walker (Stage E): a `fn() -> Vec<usize>` (stored
/// as a usize; 0 = none) that walks the CURRENT thread's frame-pointer chain,
/// resolves return addresses against the JIT's code-map registry, and returns
/// the addresses of every live stack-map slot. Installed once by the JIT
/// backend; the runtime calls it at every park and at collection time (for the
/// collector's own frames). Without a JIT there are no native frames and the
/// hook stays 0.
pub static NATIVE_ROOT_WALKER: AtomicU64 = AtomicU64::new(0);

/// Is allocation-driven collection on? Stage E4 flipped the default: ON.
/// `MICROLANG_PRESSURE_GC=0` restores the explicit-`(gc)`-only behavior.
fn pressure_gc_default() -> bool {
    // gc-stress is a pressure mode: it needs the response enabled too.
    if std::env::var("MICROLANG_GC_STRESS").is_ok_and(|v| v != "0" && !v.is_empty()) {
        return true;
    }
    match std::env::var("MICROLANG_PRESSURE_GC") {
        Ok(v) => v != "0" && !v.is_empty(),
        Err(_) => true,
    }
}

/// Run the installed native walker (empty when none is installed).
pub(crate) fn native_roots_now() -> Vec<usize> {
    let f = NATIVE_ROOT_WALKER.load(Ordering::Acquire);
    if f == 0 {
        Vec::new()
    } else {
        let f: fn() -> Vec<usize> = unsafe { std::mem::transmute::<usize, fn() -> Vec<usize>>(f as usize) };
        f()
    }
}

/// Delimiter sentinel on the dynamic-binding stack (never a real interned `Sym`,
/// which are `< TABLE_CAP`). Pushed by `%dyn-mark`, popped through by `%dyn-unwind`.
pub(crate) const DYN_MARK: Sym = u32::MAX;

/// The mutable dispatch tables, guarded by one lock. `resolve` copies a `u64`
/// impl out and drops the guard before the caller invokes it, so no lock is held
/// across a method call (no reentrancy).
pub(crate) struct Tables {
    pub(crate) methods: MethodRegistry,
    method_names: HashSet<Sym>,
    pub(crate) dispatch: Box<dyn Dispatch>,
}

/// Indices into `Shared::type_tag_cache` for `type_tag`'s built-in category
/// names (everything that isn't a `Record`, which returns its own `type_id`
/// directly with no interning at all).
const TYPE_TAG_LONG: usize = 0;
const TYPE_TAG_DOUBLE: usize = 1;
const TYPE_TAG_BOOLEAN: usize = 2;
const TYPE_TAG_NIL: usize = 3;
const TYPE_TAG_SYMBOL: usize = 4;
const TYPE_TAG_LIST: usize = 5;
const TYPE_TAG_EMPTYLIST: usize = 6;
const TYPE_TAG_VECTOR: usize = 7;
const TYPE_TAG_STRING: usize = 8;
const TYPE_TAG_CHAR: usize = 9;
const TYPE_TAG_FN: usize = 10;
const TYPE_TAG_RATIO: usize = 11;
const TYPE_TAG_ATOM: usize = 12;
const TYPE_TAG_FUTURE: usize = 13;
const TYPE_TAG_OBJECT: usize = 14;
const TYPE_TAG_CACHE_LEN: usize = 15;

/// State SHARED by every thread over one `Arc`. Its interior mutability is the
/// whole game: the heap is read lock-free (stable segmented addresses) and
/// mutated only under `heap_lock` (alloc / GC / in-place set); the append-only
/// tables are read lock-free from their reserved, never-reallocated buffers and
/// appended under a lock; globals are atomic; dispatch is behind a short lock.
/// No lock is ever held across a callback into the interpreter, so a thread
/// blocked in `deref` holds nothing another thread needs.
pub struct Shared<M: ValueModel> {
    /// THE heap: raw objects in bump semi-spaces (`heap.rs`). Allocation is a
    /// lock-free atomic bump; objects move only under the STW rendezvous.
    pub(crate) heap: crate::heap::Heap,
    /// The `TypeInfo` table driving allocation sizes + the generic GC scan.
    pub(crate) types: Vec<crate::heap::TypeInfo>,
    /// Closure BODY registry: templates are CODE (like gc-rust's type table),
    /// append-only with a reserved stable base for lock-free reads.
    templates: UnsafeCell<Vec<Arc<crate::ir::Ir>>>,
    /// Also the dedup map for `register_template`: `Arc::as_ptr(body) -> id`.
    /// The SAME lock serializes both the map and the `templates` append, so a
    /// racing pair of `Ir::Lambda` evaluations of the same closure body always
    /// converge on one id (the `Arc` the map keys on is kept alive by the
    /// registry's own clone in `templates`, so the pointer key never dangles).
    templates_lock: Mutex<HashMap<*const crate::ir::Ir, u32>>,
    /// Reified CEK continuations (`Cont`/`PartialCont` objects hold an index).
    /// Execution-machine state, never raw heap data; every entry is traced as
    /// a root (append-only — entries are never reclaimed, a bounded leak).
    pub(crate) konts: Mutex<Vec<Arc<crate::cek::Kont>>>,
    /// Future slots (OS resources: join handles + cached results). The cached
    /// results are traced as roots.
    pub(crate) futures: Mutex<Vec<Arc<std::sync::Mutex<crate::value::FutureSlot>>>>,
    /// The canonical `()` singleton's bits (a GC root; 0 until first use).
    pub(crate) empty_list: AtomicU64,
    /// Serializes MULTI-WORD in-place heap mutation (growable-array extend).
    /// Single-word stores/CAS and allocation need no lock.
    pub(crate) heap_lock: Mutex<()>,
    allocs: AtomicU64,
    pub(crate) relocated: AtomicU64,
    /// Bumped on every `register_method`, so per-site dispatch caches see
    /// protocol (re)definitions immediately.
    pub(crate) dispatch_version: AtomicU64,
    pub(crate) freed: AtomicU64,
    /// Constant pool (reserved, stable base for the JIT's inline reads).
    pub(crate) consts: UnsafeCell<Vec<u64>>,
    consts_lock: Mutex<()>,
    /// Interner: names in a reserved, stable buffer (lock-free `&str` reads);
    /// the dedup map + append serialized by `sym_lock`.
    sym_names: UnsafeCell<Vec<String>>,
    sym_ids: UnsafeCell<HashMap<String, Sym>>,
    sym_lock: Mutex<()>,
    /// Keyword interner: name `Sym` -> the ONE canonical keyword object for
    /// that name, held as a `ConstId` rather than raw bits. The const pool is
    /// a GC root that the collector rewrites on a move, so going through it
    /// keeps this table valid across a collection for free — a `HashMap<Sym,
    /// u64>` of bare bits would go stale the first time a keyword moved.
    ///
    /// Interning is what makes `(identical? :a :a)` true, as it is in BOTH
    /// Clojure and ClojureScript. It also makes a keyword literal FREE: `:foo`
    /// resolves to a constant instead of allocating a fresh record on every
    /// single evaluation, which is what it used to do.
    keywords: Mutex<HashMap<Sym, ConstId>>,
    /// Global environment: atomic slots, reserved stable base (see `global`).
    pub global_slots: Vec<AtomicU64>,
    /// Record type tag (`Sym`) -> a leaked, immutable `Vec<Sym>` of its field
    /// names in slot order (populated by `deftype`). Reserved stable base like
    /// `global_slots`, so `(.-field x)` resolves LOCK-FREE: an atomic pointer load
    /// per type, then a scan. A type's field list is write-once, so the boxed Vec
    /// is leaked and readers deref it without synchronization.
    field_names: Vec<AtomicPtr<Vec<Sym>>>,
    /// Per-site inline cache for `(.-field x)`: packs `(type Sym << 32) | index`,
    /// keyed by the `FieldGet` site id. Reserved stable base, lock-free: a hit is
    /// one relaxed load + a type-tag compare; a miss scans the field list and
    /// refills. `FIELD_IC_EMPTY` sentinel until first filled.
    field_ic: Vec<AtomicU64>,
    pub(crate) tables: Mutex<Tables>,
    apply_fn: AtomicU64, // Sym+1, or 0 for None
    seq_fn: AtomicU64,   // Sym+1, or 0 for None — the frontend's `seq` (forces one lazy node)
    escape_tags: AtomicU64,
    /// TCP handles for the `%tcp-*` prims (listener/stream by index). `Arc`s are
    /// cloned OUT under the lock, then blocking I/O happens lock-free on `&TcpStream`
    /// / `&TcpListener` (std impls Read/accept on shared refs), so a blocked reader
    /// never stalls other threads' socket ops.
    pub(crate) tcp: Mutex<Vec<Option<TcpHandle>>>,
    /// Set when a thread requests a stop-the-world collection; every other
    /// mutator parks at its next safepoint until it clears.
    pub(crate) gc_requested: std::sync::atomic::AtomicBool,
    /// Allocation-driven collection enabled (Stage E)? Default ON;
    /// `MICROLANG_PRESSURE_GC=0` opts out. When off, only the explicit `(gc)`
    /// prim collects (the pre-Stage-E behavior); parking at safepoints for a
    /// sibling's collection is always on.
    pub(crate) pressure_gc: bool,
    /// Every live mutator handle, so the collector can find + rewrite all roots.
    pub(crate) mutators: Mutex<Vec<Arc<MutatorState>>>,
    /// Frontend var/namespace registry (metadata only — `Sym`s, flag bits, and
    /// name strings, never heap values — so the collector ignores it). A frontend
    /// records each `def`'d var here so it can reflect over namespaces (ns-interns,
    /// all-ns) and per-var flags (dynamic/private/macro) at runtime.
    pub(crate) var_flags: Mutex<HashMap<Sym, u8>>,
    /// namespace name -> the vars interned there, in definition order.
    pub(crate) ns_vars: Mutex<HashMap<String, Vec<Sym>>>,
    /// qualified sym -> its `:arglists` value (a heap datum captured at `def` time,
    /// so it IS traced by the collector, unlike the flags/ns tables above).
    pub(crate) var_arglists: Mutex<HashMap<Sym, u64>>,
    /// Cached `Sym`s for the record-type tags native prims stamp on every node
    /// they allocate (HAMT trie nodes, PersistentVector trie nodes, chunked
    /// seqs). `intern` takes a process-wide mutex + does a `HashMap<String,_>`
    /// lookup even on a hit; these ops allocate one such node per trie LEVEL on
    /// every `assoc`/`conj`, so re-interning the same handful of literal type
    /// names on every call was showing up as real time in a profile. `u32::MAX`
    /// = "not yet cached" (never a real `Sym`, since `TABLE_CAP` is far smaller).
    sym_cache_bitmap_node: AtomicU32,
    sym_cache_array_node: AtomicU32,
    sym_cache_collision_node: AtomicU32,
    sym_cache_vector_node: AtomicU32,
    sym_cache_persistent_vector: AtomicU32,
    sym_cache_chunked_cons: AtomicU32,
    sym_cache_transient_vector: AtomicU32,
    sym_cache_transient_array_map: AtomicU32,
    sym_cache_transient_hash_map: AtomicU32,
    sym_cache_persistent_hash_map: AtomicU32,
    sym_cache_persistent_array_map: AtomicU32,
    /// Stage F3: the EDIT-SESSION counter for transients (see
    /// `Runtime::fresh_session`). Starts at 1 and never repeats.
    edit_sessions: AtomicU64,
    /// Same lock-free-on-hit cache, for `type_tag`'s BUILT-IN category names
    /// (see the `TYPE_TAG_*` indices below) — `type_tag` backs `type-of`,
    /// which nearly every predicate (`vector?`, `map?`, `string?`, `seq?`, a
    /// LazySeq's `-seq`, `chunked?`, ...) calls on every single invocation, so
    /// this was showing up as real time in profiles even outside string/HAMT
    /// code.
    type_tag_cache: [AtomicU32; TYPE_TAG_CACHE_LEN],
    _pd: PhantomData<fn() -> M>,
}

/// Var-flag bits stored in `Shared::var_flags`.
pub const VAR_DYNAMIC: u8 = 1;
pub const VAR_PRIVATE: u8 = 2;
pub const VAR_MACRO: u8 = 4;

/// Create + register a fresh mutator root-slot in the shared registry.
fn register_mutator<M: ValueModel>(shared: &Arc<Shared<M>>) -> Arc<MutatorState> {
    let me = Arc::new(MutatorState {
        roots: Mutex::new(Vec::new()),
        envs: Mutex::new(Vec::new()),
        dyn_roots: Mutex::new(Vec::new()),
        kont: Mutex::new(None),
        native_roots: Mutex::new(Vec::new()),
        parked: std::sync::atomic::AtomicBool::new(false),
    });
    shared.mutators.lock().unwrap().push(me.clone());
    me
}

// SAFETY: every field is accessed under the discipline documented on `Shared`:
// heap/consts/interner reads are to stable addresses that are never freed or
// moved except under a lock (heap) or never during a run (append-only tables);
// all mutation is serialized by the corresponding lock; globals/counters are
// atomic. No `&`/`&mut` into the interior outlives the operation that takes it.
unsafe impl<M: ValueModel> Send for Shared<M> {}
unsafe impl<M: ValueModel> Sync for Shared<M> {}

/// How `apply` should call a callee, decided BEFORE the frame is built (see
/// `Runtime::plan_apply`). The split exists because Clojure's `apply` shares
/// structure: the rest arg is the applied seq, not a copy of it, and knowing
/// that requires resolving the callee's arity up front.
pub(crate) enum ApplyPlan {
    /// Call with these exact positional args — a non-variadic callee, or one
    /// whose arity is wrong (`invoke` raises the arity throw).
    Flat { callee: u64, args: Vec<u64> },
    /// Call `callee` (an already-resolved variadic Closure — never a MultiFn,
    /// whose clause selection has already happened) with `fixed` positional
    /// args and `rest` installed verbatim as the rest slot.
    WithRest { callee: u64, fixed: Vec<u64>, rest: u64 },
    /// A signal (arity throw / escape) was raised while planning.
    Signal,
}

pub struct Runtime<M: ValueModel> {
    /// State shared with sibling threads (heap, globals, interner, dispatch).
    pub(crate) shared: Arc<Shared<M>>,
    /// The shadow stack: this THREAD's GC root set for transient values.
    /// Per-thread (not shared) — each mutator handle owns its own.
    pub(crate) shadow: Vec<u64>,
    /// This thread's dynamic env stack: one entry per active function call, so a
    /// collection traces every live frame in the whole call chain (see `invoke`,
    /// which pushes/pops it). Published to `me.envs` when the thread parks.
    pub(crate) env_stack: Vec<Locals>,
    /// `apply`'s pending rest arg: the SHADOW-STACK SLOT holding the seq that
    /// the next variadic frame must install verbatim, instead of collecting
    /// `args[nparams..]` into a fresh spine. `Prim::Apply` sets it immediately
    /// before invoking an already-resolved variadic callee; `mk_rest_seq` —
    /// the ONE place any backend materializes a rest arg — takes it.
    ///
    /// This is what makes `apply` share structure the way Clojure does:
    /// `(apply f coll)` hands `coll`'s seq to `f` untouched, so `identical?`
    /// holds, a lazy tail stays unforced, and the rest arg keeps the source's
    /// shape (chunked stays chunked, a list stays a list).
    ///
    /// A SLOT, not the bits: allocation sets `gc_requested`, so a collection
    /// can land at the next safepoint and relocate the seq. The shadow stack
    /// is traced and rewritten; a bare `u64` here would silently go stale the
    /// day someone adds a safepoint between the set and the frame build.
    pub(crate) apply_rest_slot: Option<usize>,
    /// This thread's slot in the STW registry (where it publishes roots to park).
    pub(crate) me: Arc<MutatorState>,
    /// This thread's pending non-local control-flow signal (throw / escape). See
    /// `Signal`. `kind == 0` in the common (no-signal) case.
    pub(crate) signal: Signal,
    /// This thread's dynamic-var binding stack (`^:dynamic` + `binding`). Innermost
    /// binding last; a `DYN_MARK` entry delimits each `binding` scope. Traced by the
    /// collector (self here, published via `me.dyn_roots` when parked).
    pub(crate) dyn_stack: Vec<(Sym, u64)>,
    /// Per-HANDLE (thread-local, lock-free) dispatch-site inline cache:
    /// `site -> (gc_epoch, receiver type, impl)`. A hit skips the registry
    /// mutex + hash lookup entirely. Epoch-invalidated: `relocated` advances on
    /// every moving collection, so an entry holding a moved impl never hits.
    site_ic: std::cell::RefCell<Vec<(u64, Sym, u64)>>,
    /// Frontend-installed bridge back into the reader + compiler, for the runtime
    /// ops `read-string`/`eval`/`macroexpand-1`. Set on the MAIN handle only (worker
    /// threads get `None` — they never re-enter the compiler).
    pub(crate) eval_bridge: Option<Arc<dyn EvalBridge<M>>>,
    _pd: PhantomData<fn() -> M>,
}

/// A frontend-installed bridge letting runtime code re-enter the reader + compiler
/// (`read-string`/`eval`/`macroexpand-1`). The frontend owns the live `Compiler` +
/// macro set, so only it can implement this. The impl typically holds raw pointers
/// to that compiler state and is `unsafe impl Send + Sync`; those pointers are only
/// dereferenced on the thread that installed the bridge (the main REPL thread).
pub trait EvalBridge<M: ValueModel>: Send + Sync {
    /// Read the first datum from a string value `s`.
    fn read_string(&self, rt: &mut Runtime<M>, s: u64) -> u64;
    /// Compile & run a form (datum) in the current namespace; returns its value.
    fn eval(&self, rt: &mut Runtime<M>, form: u64) -> u64;
    /// Expand a form by one macro step, or return it unchanged.
    fn macroexpand_1(&self, rt: &mut Runtime<M>, form: u64) -> u64;
    /// The current namespace's name symbol (`*ns*` reflection). Namespace state
    /// is frontend policy; frontends without namespaces return nil.
    fn current_ns(&self, rt: &mut Runtime<M>) -> u64 {
        rt.encode(Val::Nil)
    }
}

impl<M: ValueModel> Drop for Runtime<M> {
    fn drop(&mut self) {
        // Deregister this thread's root slot so a future collector won't wait on
        // a thread that no longer exists.
        let mut ms = self.shared.mutators.lock().unwrap();
        ms.retain(|m| !Arc::ptr_eq(m, &self.me));
    }
}

impl<M: ValueModel> Runtime<M> {
    pub fn new() -> Self {
        let shared = Shared {
            heap: crate::heap::Heap::new(),
            types: crate::heap::type_table(),
            templates: UnsafeCell::new(Vec::with_capacity(TABLE_CAP)),
            templates_lock: Mutex::new(HashMap::new()),
            konts: Mutex::new(Vec::new()),
            futures: Mutex::new(Vec::new()),
            empty_list: AtomicU64::new(0),
            heap_lock: Mutex::new(()),
            allocs: AtomicU64::new(0),
            relocated: AtomicU64::new(0),
            dispatch_version: AtomicU64::new(0),
            freed: AtomicU64::new(0),
            consts: UnsafeCell::new(Vec::with_capacity(TABLE_CAP)),
            consts_lock: Mutex::new(()),
            sym_names: UnsafeCell::new(Vec::with_capacity(TABLE_CAP)),
            sym_ids: UnsafeCell::new(HashMap::new()),
            sym_lock: Mutex::new(()),
            keywords: Mutex::new(HashMap::new()),
            global_slots: (0..0).map(|_| AtomicU64::new(GLOBAL_UNBOUND)).collect(),
            field_names: Vec::new(),
            field_ic: Vec::new(),
            tables: Mutex::new(Tables {
                methods: HashMap::new(),
                method_names: HashSet::new(),
                dispatch: Box::new(Megamorphic::new()),
            }),
            apply_fn: AtomicU64::new(0),
            seq_fn: AtomicU64::new(0),
            tcp: Mutex::new(Vec::new()),
            escape_tags: AtomicU64::new(0),
            gc_requested: std::sync::atomic::AtomicBool::new(false),
            pressure_gc: pressure_gc_default(),
            mutators: Mutex::new(Vec::new()),
            var_flags: Mutex::new(HashMap::new()),
            ns_vars: Mutex::new(HashMap::new()),
            var_arglists: Mutex::new(HashMap::new()),
            sym_cache_bitmap_node: AtomicU32::new(u32::MAX),
            sym_cache_array_node: AtomicU32::new(u32::MAX),
            sym_cache_collision_node: AtomicU32::new(u32::MAX),
            sym_cache_vector_node: AtomicU32::new(u32::MAX),
            sym_cache_persistent_vector: AtomicU32::new(u32::MAX),
            sym_cache_chunked_cons: AtomicU32::new(u32::MAX),
            sym_cache_transient_vector: AtomicU32::new(u32::MAX),
            sym_cache_transient_array_map: AtomicU32::new(u32::MAX),
            sym_cache_transient_hash_map: AtomicU32::new(u32::MAX),
            sym_cache_persistent_hash_map: AtomicU32::new(u32::MAX),
            sym_cache_persistent_array_map: AtomicU32::new(u32::MAX),
            edit_sessions: AtomicU64::new(1),
            type_tag_cache: std::array::from_fn(|_| AtomicU32::new(u32::MAX)),
            _pd: PhantomData,
        };
        // Pre-size the global array to its reserved cap so its base is stable and
        // slots exist for every future symbol (index == Sym).
        let mut slots = Vec::with_capacity(TABLE_CAP);
        slots.extend((0..TABLE_CAP).map(|_| AtomicU64::new(GLOBAL_UNBOUND)));
        let mut fnames = Vec::with_capacity(TABLE_CAP);
        fnames.extend((0..TABLE_CAP).map(|_| AtomicPtr::new(std::ptr::null_mut())));
        let mut fic = Vec::with_capacity(FIELD_SITE_CAP);
        fic.extend((0..FIELD_SITE_CAP).map(|_| AtomicU64::new(FIELD_IC_EMPTY)));
        let shared = Arc::new(Shared {
            global_slots: slots,
            field_names: fnames,
            field_ic: fic,
            ..shared
        });
        // The heap has its FINAL address now (inside the Arc): open the JIT's
        // inline-allocation window. Arming any earlier would leave its cursor
        // mirror pointing at a moved-from temporary.
        shared.heap.arm_window();
        let me = register_mutator(&shared);
        Runtime { shared, shadow: Vec::new(), env_stack: Vec::new(), apply_rest_slot: None, me, signal: Signal::default(), dyn_stack: Vec::new(), site_ic: std::cell::RefCell::new(Vec::new()), eval_bridge: None, _pd: PhantomData }
    }

    /// A fresh mutator handle for another OS thread, sharing this runtime's heap,
    /// globals, interner, and dispatch. The new handle has its OWN (empty) shadow
    /// stack and its own STW registry slot. This is what `spawn`/`future` hands to
    /// a `std::thread`.
    pub fn thread_handle(&self) -> Self {
        let me = register_mutator(&self.shared);
        Runtime { shared: self.shared.clone(), shadow: Vec::new(), env_stack: Vec::new(), apply_rest_slot: None, me, signal: Signal::default(), dyn_stack: Vec::new(), site_ic: std::cell::RefCell::new(Vec::new()), eval_bridge: None, _pd: PhantomData }
    }

    // ── heap access ─────────────────────────────────────────
    /// The raw heap (allocation is `&self`, lock-free atomic bump; objects
    /// move only under the STW rendezvous, so reads between safepoints are
    /// sound exactly as before).
    #[inline]
    pub fn heap(&self) -> &crate::heap::Heap {
        &self.shared.heap
    }
    /// The `TypeInfo` for a heap object.
    #[inline]
    pub(crate) fn type_info(&self, g: Gc) -> &crate::heap::TypeInfo {
        &self.shared.types[unsafe { g.type_id() } as usize]
    }

    /// Reconstruct the enum-shaped VIEW of the object `bits` references.
    /// Panics loudly (verify-style) on a non-ref or a poisoned/invalid header.
    #[inline]
    pub fn view(&self, bits: u64) -> ObjView<'_> {
        debug_assert_eq!(M::R::tag_of(bits), RawTag::Ref, "view of a non-ref");
        self.view_gc(M::R::as_ref(bits))
    }

    /// The view, from an already-decoded object pointer.
    pub fn view_gc(&self, g: Gc) -> ObjView<'_> {
        use crate::heap::kind;
        unsafe {
            let tid = g.type_id();
            match tid {
                kind::CONS => ObjView::Cons { head: g.field(0), tail: g.field(1) },
                kind::EMPTY_LIST => ObjView::EmptyList,
                kind::STR => {
                    let info = &self.shared.types[kind::STR as usize];
                    ObjView::Str(std::str::from_utf8_unchecked(g.bytes(info)))
                }
                kind::CHAR => {
                    let info = &self.shared.types[kind::CHAR as usize];
                    ObjView::Char(char::from_u32_unchecked(g.raw_word_u32(info)))
                }
                kind::ARRAY => {
                    let len = g.aux();
                    // The handle's one traced field is an ENCODED ref to the blob.
                    let data = M::R::as_ref(g.field(0));
                    let dinfo = &self.shared.types[kind::ARRAY_DATA as usize];
                    let cap = data.aux();
                    ObjView::Vector { elems: &data.values(dinfo)[..len as usize], gc: g, cap }
                }
                kind::VALUES => {
                    let info = &self.shared.types[kind::VALUES as usize];
                    ObjView::Values(g.values(info))
                }
                kind::BIGINT => {
                    let info = &self.shared.types[kind::BIGINT as usize];
                    ObjView::BigInt(g.raw_i128(info, 0))
                }
                kind::HUGEINT => {
                    let info = &self.shared.types[kind::HUGEINT as usize];
                    let neg = g.raw_word(info, 0) != 0;
                    let bytes = g.bytes(info);
                    let mag: Vec<u32> = bytes
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    ObjView::HugeInt(BigInt::from_parts(neg, mag))
                }
                kind::RATIO => {
                    let info = &self.shared.types[kind::RATIO as usize];
                    ObjView::Ratio(g.raw_i128(info, 0), g.raw_i128(info, 16))
                }
                kind::BOXFLOAT => {
                    let info = &self.shared.types[kind::BOXFLOAT as usize];
                    ObjView::BoxFloat(f64::from_bits(g.raw_word(info, 0)))
                }
                kind::CLOSURE => {
                    let meta = *(g.0.add(crate::heap::CLOSURE_META_OFF) as *const u64);
                    ObjView::Closure {
                        nparams: crate::heap::meta_nparams(meta),
                        variadic: crate::heap::meta_variadic(meta),
                        nslots: crate::heap::meta_nslots(meta),
                        template: crate::heap::meta_template(meta),
                        gc: g,
                    }
                }
                kind::MULTIFN => {
                    let info = &self.shared.types[kind::MULTIFN as usize];
                    let vbits = g.field(0);
                    let vmin = g.raw_word(info, 0);
                    let variadic = if vmin == u64::MAX { None } else { Some((vmin as usize, vbits)) };
                    ObjView::MultiFn { fixed: g.values(info), variadic }
                }
                kind::RECORD => {
                    let info = &self.shared.types[kind::RECORD as usize];
                    ObjView::Record { type_id: g.raw_word(info, 0) as Sym, fields: g.values(info) }
                }
                kind::ESCAPE => {
                    let info = &self.shared.types[kind::ESCAPE as usize];
                    ObjView::Escape { tag: g.raw_word(info, 0) }
                }
                kind::CONT | kind::PARTIAL_CONT => {
                    let info = &self.shared.types[tid as usize];
                    let idx = g.raw_word(info, 0) as usize;
                    let k = self.shared.konts.lock().unwrap()[idx].clone();
                    if tid == kind::CONT { ObjView::Cont(k) } else { ObjView::PartialCont(k) }
                }
                kind::ATOM => ObjView::Atom(g.field_atomic(0)),
                kind::FUTURE => {
                    let info = &self.shared.types[kind::FUTURE as usize];
                    let idx = g.raw_word(info, 0) as usize;
                    ObjView::Future(self.shared.futures.lock().unwrap()[idx].clone())
                }
                other => panic!(
                    "view: object at {:p} has type_id {other} — {} (stale pointer into a \
                     collected space, or heap corruption)",
                    g.0,
                    if other == kind::INVALID || other == 0x5A5A { "poisoned/invalid" } else { "unknown" }
                ),
            }
        }
    }

    /// The template registry: a closure BODY, by id (append-only, stable base).
    #[inline]
    pub fn template(&self, id: u32) -> &Arc<crate::ir::Ir> {
        // SAFETY: id was handed out by `register_template` (< len); reserved
        // buffer never reallocates; a slot is written before its id escapes.
        unsafe { (&*self.shared.templates.get()).get_unchecked(id as usize) }
    }
    pub fn register_template(&self, body: &Arc<crate::ir::Ir>) -> u32 {
        // Registering the same Arc'd body again returns the existing id (bodies
        // are compiled once per template; an `Ir::Lambda` re-evaluates many
        // times — once per closure creation — but always over the SAME body
        // `Arc`, so a naive append would grow the registry unboundedly in a
        // hot loop). Dedup by pointer identity under the same lock that
        // serializes the append, so a race between two threads registering the
        // same body converges on one id.
        let key = Arc::as_ptr(body);
        let mut ids = self.shared.templates_lock.lock().unwrap();
        if let Some(&id) = ids.get(&key) {
            return id;
        }
        let ts = unsafe { &mut *self.shared.templates.get() };
        assert!(ts.len() < TABLE_CAP, "template registry overflow: raise TABLE_CAP");
        ts.push(body.clone());
        let id = (ts.len() - 1) as u32;
        ids.insert(key, id);
        id
    }

    // ── typed allocation (writes headers + fields; hot paths) ──
    /// Allocate a growable ARRAY (handle + data blob) over a copy of `xs`.
    pub fn alloc_vector(&self, xs: &[u64]) -> Gc {
        self.alloc_vector_cap(xs, xs.len())
    }
    /// Allocate an ARRAY with logical contents `xs` and capacity `cap >= xs.len()`.
    pub fn alloc_vector_cap(&self, xs: &[u64], cap: usize) -> Gc {
        use crate::heap::kind;
        self.shared.allocs.fetch_add(1, Ordering::Relaxed);
        let dinfo = &self.shared.types[kind::ARRAY_DATA as usize];
        let data = self.shared.heap.alloc(dinfo, cap as u32);
        unsafe {
            data.values_mut(dinfo)[..xs.len()].copy_from_slice(xs);
            let ainfo = &self.shared.types[kind::ARRAY as usize];
            let h = self.shared.heap.alloc(ainfo, xs.len() as u32);
            h.set_field(0, M::R::enc_ref(data));
            h
        }
    }
    /// Allocate an ARRAY holding `xs ++ [x]` — the persistent-vector tail conj.
    /// Fused (Stage H) so the copy goes heap→heap ONCE: the obvious
    /// `Vec::with_capacity` + `extend_from_slice` + `alloc_vector` spelling paid
    /// a malloc/free pair and a SECOND memmove per conj, and both were top
    /// frames of the vecbuild profile.
    pub fn alloc_vector_append1(&self, xs: &[u64], x: u64) -> Gc {
        use crate::heap::kind;
        self.shared.allocs.fetch_add(1, Ordering::Relaxed);
        let n = xs.len();
        let dinfo = &self.shared.types[kind::ARRAY_DATA as usize];
        let data = self.shared.heap.alloc(dinfo, (n + 1) as u32);
        unsafe {
            let slots = data.values_mut(dinfo);
            slots[..n].copy_from_slice(xs);
            slots[n] = x;
            let ainfo = &self.shared.types[kind::ARRAY as usize];
            let h = self.shared.heap.alloc(ainfo, (n + 1) as u32);
            h.set_field(0, M::R::enc_ref(data));
            h
        }
    }
    /// Allocate an ARRAY holding `xs` with index `i` replaced by `x` — the
    /// trie path-copy step. Fused for the same reason as `alloc_vector_append1`
    /// (the `arr_clone` + mutate + `mk_array` spelling round-tripped a Vec).
    pub fn alloc_vector_set1(&self, xs: &[u64], i: usize, x: u64) -> Gc {
        use crate::heap::kind;
        assert!(i < xs.len(), "alloc_vector_set1: index {i} out of bounds (len {})", xs.len());
        self.shared.allocs.fetch_add(1, Ordering::Relaxed);
        let n = xs.len();
        let dinfo = &self.shared.types[kind::ARRAY_DATA as usize];
        let data = self.shared.heap.alloc(dinfo, n as u32);
        unsafe {
            let slots = data.values_mut(dinfo);
            slots[..n].copy_from_slice(xs);
            slots[i] = x;
            let ainfo = &self.shared.types[kind::ARRAY as usize];
            let h = self.shared.heap.alloc(ainfo, n as u32);
            h.set_field(0, M::R::enc_ref(data));
            h
        }
    }
    /// Allocate an ARRAY of `n` `nil` slots.
    pub fn alloc_vector_nil(&self, n: usize, nil: u64) -> Gc {
        use crate::heap::kind;
        self.shared.allocs.fetch_add(1, Ordering::Relaxed);
        let dinfo = &self.shared.types[kind::ARRAY_DATA as usize];
        let data = self.shared.heap.alloc(dinfo, n as u32);
        unsafe {
            data.values_mut(dinfo).fill(nil);
            let ainfo = &self.shared.types[kind::ARRAY as usize];
            let h = self.shared.heap.alloc(ainfo, n as u32);
            h.set_field(0, M::R::enc_ref(data));
            h
        }
    }
    /// Allocate a Values packet over a copy of `xs`.
    pub fn alloc_values(&self, xs: &[u64]) -> Gc {
        use crate::heap::kind;
        self.shared.allocs.fetch_add(1, Ordering::Relaxed);
        let info = &self.shared.types[kind::VALUES as usize];
        let g = self.shared.heap.alloc(info, xs.len() as u32);
        unsafe { g.values_mut(info).copy_from_slice(xs) };
        g
    }
    /// Allocate a Record over a copy of `fields`.
    pub fn alloc_record(&self, type_id: Sym, fields: &[u64]) -> Gc {
        use crate::heap::kind;
        self.shared.allocs.fetch_add(1, Ordering::Relaxed);
        let info = &self.shared.types[kind::RECORD as usize];
        let g = self.shared.heap.alloc(info, fields.len() as u32);
        unsafe {
            g.set_raw_word(info, 0, type_id as u64);
            g.values_mut(info).copy_from_slice(fields);
        }
        g
    }
    /// Allocate a flat CLOSURE: body registered as a template, captures inline.
    pub fn alloc_closure(
        &self,
        nparams: usize,
        variadic: bool,
        nslots: u16,
        template: u32,
        caps: &[u64],
    ) -> Gc {
        use crate::heap::{closure_meta, kind, CLOSURE_CAPS_OFF, CLOSURE_META_OFF};
        self.shared.allocs.fetch_add(1, Ordering::Relaxed);
        let info = &self.shared.types[kind::CLOSURE as usize];
        let g = self.shared.heap.alloc(info, caps.len() as u32);
        unsafe {
            *(g.0.add(CLOSURE_META_OFF) as *mut u64) =
                closure_meta(template, nparams as u16, nslots, variadic);
            // code word stays 0 (not compiled); the JIT fills it on publish.
            let base = g.0.add(CLOSURE_CAPS_OFF) as *mut u64;
            std::ptr::copy_nonoverlapping(caps.as_ptr(), base, caps.len());
        }
        g
    }
    /// Read a growable array's live elements (logical length off the handle).
    #[inline]
    pub fn arr_slice(&self, g: Gc) -> &[u64] {
        use crate::heap::kind;
        unsafe {
            debug_assert_eq!(g.type_id(), kind::ARRAY);
            let len = g.aux() as usize;
            let data = M::R::as_ref(g.field(0));
            let dinfo = &self.shared.types[kind::ARRAY_DATA as usize];
            &data.values(dinfo)[..len]
        }
    }
    /// THE WRITE BARRIER, runtime-side (Stage I2). Remember that `g` — an
    /// OBJECT BASE — was written, so a minor GC finds the young objects it now
    /// names even though no root does.
    ///
    /// Every barriered site in this file goes through one of the three CHOKE
    /// POINTS below (`arr_slice_mut`, `values_mut`, `arr_extend`) or the two
    /// atom prims, rather than calling this directly at each store: an edge is
    /// only lost ONCE and then corrupts the heap arbitrarily later, so the
    /// invariant has to live where the capability is handed out, not where each
    /// caller remembers it.
    #[inline(always)]
    pub(crate) fn write_barrier(&self, g: Gc) {
        self.shared.heap.write_barrier(g);
    }

    /// The ATOM object behind `bits` — `ObjView::Atom` hands out the cell but
    /// not the base, and the barrier marks the base.
    fn atom_gc(&self, bits: u64, op: &str) -> Gc {
        match self.decode(bits) {
            Val::Ref(g) if unsafe { g.type_id() } == crate::heap::kind::ATOM => g,
            _ => panic!("{op}: not an atom"),
        }
    }

    /// Mutate a growable array's live elements in place (`%aset`).
    ///
    /// BARRIERED: the words live in the DATA BLOB, so the blob — not the handle
    /// — is the object that gets written and therefore the base that is marked.
    /// The mark happens when the `&mut` is handed out rather than at each store
    /// through it, which is what makes it impossible for a caller to forget;
    /// the price is marking a card for a caller that only writes non-pointers
    /// (a scan of that card's objects at the next minor, which finds nothing).
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn arr_slice_mut(&self, g: Gc) -> &mut [u64] {
        use crate::heap::kind;
        unsafe {
            debug_assert_eq!(g.type_id(), kind::ARRAY);
            let len = g.aux() as usize;
            let data = M::R::as_ref(g.field(0));
            let dinfo = &self.shared.types[kind::ARRAY_DATA as usize];
            self.write_barrier(data);
            &mut data.values_mut(dinfo)[..len]
        }
    }
    /// Mutable access to a fixed-shape object's varlen Values tail (record
    /// fields, a Values packet, ...) — the non-ARRAY mirror of `arr_slice_mut`
    /// (records/values are fixed-length once allocated, so there is no
    /// separate handle/data-blob split to go through).
    ///
    /// BARRIERED, on the same handing-out-the-capability principle. This is the
    /// one that covers the F3 transients: a transient record can be PROMOTED
    /// between two `assoc!`s, so from then on every in-place edit of it is an
    /// old→young store.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn values_mut(&self, g: Gc) -> &mut [u64] {
        let info = self.type_info(g);
        self.write_barrier(g);
        unsafe { g.values_mut(info) }
    }
    /// Append `xs` to a growable array: within capacity, write into the blob
    /// and bump the handle's logical length; past it, allocate a doubled blob
    /// and re-point the handle (identity is the handle, so every reference
    /// observes the growth). Serialized by `heap_lock` against racing extends.
    ///
    /// BARRIERED on BOTH objects, because it writes both: `xs` goes into the
    /// DATA BLOB (which an earlier minor may have promoted while `xs` is fresh
    /// from the nursery), and the grow path re-points the HANDLE's field 0 at a
    /// brand-new blob — a long-lived handle acquiring a young child, which is
    /// the same shape as the atom swap! that beagle shipped.
    pub fn arr_extend(&self, g: Gc, xs: &[u64]) {
        use crate::heap::kind;
        let _lk = self.shared.heap_lock.lock().unwrap();
        unsafe {
            assert_eq!(g.type_id(), kind::ARRAY, "apush: not an array");
            let len = g.aux() as usize;
            let data = M::R::as_ref(g.field(0));
            let dinfo = &self.shared.types[kind::ARRAY_DATA as usize];
            let cap = data.aux() as usize;
            let need = len + xs.len();
            if need <= cap {
                self.write_barrier(data);
                data.values_mut(dinfo)[len..need].copy_from_slice(xs);
                g.set_aux(need as u32);
                return;
            }
            let ncap = need.next_power_of_two().max(4);
            self.shared.allocs.fetch_add(1, Ordering::Relaxed);
            let ndata = self.shared.heap.alloc(dinfo, ncap as u32);
            let dst = ndata.values_mut(dinfo);
            dst[..len].copy_from_slice(&data.values(dinfo)[..len]);
            dst[len..need].copy_from_slice(xs);
            // `ndata` is fresh (nursery), so its own initializing stores above
            // need nothing; the handle is what may be old.
            g.set_field(0, M::R::enc_ref(ndata));
            self.write_barrier(g);
            g.set_aux(need as u32);
        }
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
    /// `intern(name)`, but through a lock-free cache slot for a name a native
    /// prim re-interns on every call (see the `sym_cache_*` fields' doc comment).
    /// A benign race on first use (two threads both miss and both `intern`) just
    /// stores the same `Sym` twice — `intern` itself is the source of truth.
    fn intern_cached(&self, cache: &AtomicU32, name: &str) -> Sym {
        let v = cache.load(Ordering::Relaxed);
        if v != u32::MAX {
            return v;
        }
        let s = self.intern(name);
        cache.store(s, Ordering::Relaxed);
        s
    }

    /// Read a global's value, or `None` if unbound. THE global read path — the
    /// dense `global_slots` array IS the store (no separate map), so this is a
    /// single indexed load, and a `Sym` beyond the current slot count is unbound.
    pub fn global(&self, sym: Sym) -> Option<u64> {
        match self.shared.global_slots.get(sym as usize).map(|a| a.load(GLOBAL_LOAD)) {
            Some(v) if v != GLOBAL_UNBOUND => Some(v),
            _ => None,
        }
    }

    /// Is this global bound? (`set!`/redefinition checks.)
    pub fn global_defined(&self, sym: Sym) -> bool {
        matches!(self.shared.global_slots.get(sym as usize).map(|a| a.load(GLOBAL_LOAD)), Some(v) if v != GLOBAL_UNBOUND)
    }

    /// Register a var in the frontend var/namespace registry: record it under its
    /// namespace (in definition order, once) and OR in its flag bits. Called by a
    /// frontend at `def` time so it can reflect over namespaces + var flags.
    pub fn register_var(&self, sym: Sym, ns: &str, flags: u8) {
        {
            let mut nv = self.shared.ns_vars.lock().unwrap();
            let list = nv.entry(ns.to_string()).or_default();
            if !list.contains(&sym) {
                list.push(sym);
            }
        }
        if flags != 0 {
            let mut vf = self.shared.var_flags.lock().unwrap();
            *vf.entry(sym).or_insert(0) |= flags;
        }
    }

    /// OR extra flag bits onto an existing var (e.g. `VAR_MACRO`, set after the def).
    pub fn set_var_flags(&self, sym: Sym, flags: u8) {
        let mut vf = self.shared.var_flags.lock().unwrap();
        *vf.entry(sym).or_insert(0) |= flags;
    }

    /// This var's flag bits (`VAR_DYNAMIC | VAR_PRIVATE | VAR_MACRO`), 0 if none.
    pub fn var_flags(&self, sym: Sym) -> u8 {
        self.shared.var_flags.lock().unwrap().get(&sym).copied().unwrap_or(0)
    }

    /// The vars interned in a namespace, in definition order.
    pub fn ns_var_syms(&self, ns: &str) -> Vec<Sym> {
        self.shared.ns_vars.lock().unwrap().get(ns).cloned().unwrap_or_default()
    }

    /// The names of all registered namespaces.
    pub fn all_ns_names(&self) -> Vec<String> {
        self.shared.ns_vars.lock().unwrap().keys().cloned().collect()
    }

    /// Record / read a var's `:arglists` datum (a heap value; captured at def time).
    pub fn set_var_arglists(&self, sym: Sym, val: u64) {
        self.shared.var_arglists.lock().unwrap().insert(sym, val);
    }
    pub fn get_var_arglists(&self, sym: Sym) -> Option<u64> {
        self.shared.var_arglists.lock().unwrap().get(&sym).copied()
    }

    /// Define (or redefine) a global. Atomic store, so `&self` suffices — a step
    /// toward the shared runtime where a thread defines through an `Arc`.
    pub fn define_global(&self, sym: Sym, val: u64) {
        if let Some(slot) = self.shared.global_slots.get(sym as usize) {
            slot.store(val, GLOBAL_STORE);
        }
    }

    /// Assign an existing global's value (`set!`). Returns `false` if unbound.
    pub fn set_global_val(&self, sym: Sym, val: u64) -> bool {
        match self.shared.global_slots.get(sym as usize) {
            Some(slot) if slot.load(GLOBAL_LOAD) != GLOBAL_UNBOUND => {
                slot.store(val, GLOBAL_STORE);
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
    /// Lower an allocation REQUEST into a raw heap object (header + inline
    /// fields). Cold/simple call sites use this; hot paths call the typed
    /// `alloc_*` constructors directly.
    pub fn alloc(&self, o: Obj) -> Gc {
        use crate::heap::kind;
        self.shared.allocs.fetch_add(1, Ordering::Relaxed);
        let heap = &self.shared.heap;
        let t = |k: u16| &self.shared.types[k as usize];
        unsafe {
            match o {
                Obj::Cons { head, tail } => {
                    let g = heap.alloc(t(kind::CONS), 0);
                    g.set_field(0, head);
                    g.set_field(1, tail);
                    g
                }
                Obj::EmptyList => heap.alloc(t(kind::EMPTY_LIST), 0),
                Obj::Str(s) => {
                    let info = t(kind::STR);
                    let g = heap.alloc(info, s.len() as u32);
                    g.bytes_mut(info).copy_from_slice(s.as_bytes());
                    g
                }
                Obj::Char(c) => {
                    let info = t(kind::CHAR);
                    let g = heap.alloc(info, 0);
                    g.set_raw_word_u32(info, c as u32);
                    g
                }
                Obj::Vector(xs) => {
                    self.shared.allocs.fetch_sub(1, Ordering::Relaxed); // counted below
                    self.alloc_vector(&xs)
                }
                Obj::Values(xs) => {
                    self.shared.allocs.fetch_sub(1, Ordering::Relaxed);
                    self.alloc_values(&xs)
                }
                Obj::BigInt(i) => {
                    let info = t(kind::BIGINT);
                    let g = heap.alloc(info, 0);
                    g.set_raw_i128(info, 0, i);
                    g
                }
                Obj::HugeInt(b) => {
                    let info = t(kind::HUGEINT);
                    let limbs = b.limbs();
                    let g = heap.alloc(info, (limbs.len() * 4) as u32);
                    g.set_raw_word(info, 0, b.is_negative() as u64);
                    let bytes = g.bytes_mut(info);
                    for (i, l) in limbs.iter().enumerate() {
                        bytes[i * 4..i * 4 + 4].copy_from_slice(&l.to_le_bytes());
                    }
                    g
                }
                Obj::Ratio(n, d) => {
                    let info = t(kind::RATIO);
                    let g = heap.alloc(info, 0);
                    g.set_raw_i128(info, 0, n);
                    g.set_raw_i128(info, 16, d);
                    g
                }
                Obj::BoxFloat(f) => {
                    let info = t(kind::BOXFLOAT);
                    let g = heap.alloc(info, 0);
                    g.set_raw_word(info, 0, f.to_bits());
                    g
                }
                Obj::Closure { nparams, variadic, nslots, body, caps } => {
                    self.shared.allocs.fetch_sub(1, Ordering::Relaxed);
                    let template = self.register_template(&body);
                    self.alloc_closure(nparams, variadic, nslots, template, &caps)
                }
                Obj::MultiFn { fixed, variadic } => {
                    let info = t(kind::MULTIFN);
                    let g = heap.alloc(info, fixed.len() as u32);
                    let (min, vbits) = match variadic {
                        Some((m, v)) => (m as u64, v),
                        None => (u64::MAX, 0),
                    };
                    g.set_field(0, vbits);
                    g.set_raw_word(info, 0, min);
                    g.values_mut(info).copy_from_slice(&fixed);
                    g
                }
                Obj::Record { type_id, fields } => {
                    self.shared.allocs.fetch_sub(1, Ordering::Relaxed);
                    self.alloc_record(type_id, &fields)
                }
                Obj::Escape { tag } => {
                    let info = t(kind::ESCAPE);
                    let g = heap.alloc(info, 0);
                    g.set_raw_word(info, 0, tag);
                    g
                }
                Obj::Cont(k) => {
                    let idx = self.register_kont(k);
                    let info = t(kind::CONT);
                    let g = heap.alloc(info, 0);
                    g.set_raw_word(info, 0, idx as u64);
                    g
                }
                Obj::PartialCont(k) => {
                    let idx = self.register_kont(k);
                    let info = t(kind::PARTIAL_CONT);
                    let g = heap.alloc(info, 0);
                    g.set_raw_word(info, 0, idx as u64);
                    g
                }
                Obj::Atom(init) => {
                    let g = heap.alloc(t(kind::ATOM), 0);
                    g.set_field(0, init);
                    g
                }
                Obj::Future(slot) => {
                    let idx = {
                        let mut fs = self.shared.futures.lock().unwrap();
                        fs.push(slot);
                        fs.len() - 1
                    };
                    let info = t(kind::FUTURE);
                    let g = heap.alloc(info, 0);
                    g.set_raw_word(info, 0, idx as u64);
                    g
                }
            }
        }
    }

    /// Register a reified continuation, returning its registry index. Entries
    /// are roots (traced every collection) and never reclaimed — reifying a
    /// continuation is rare and this keeps heap objects free of Rust `Arc`s.
    pub(crate) fn register_kont(&self, k: Arc<crate::cek::Kont>) -> usize {
        let mut ks = self.shared.konts.lock().unwrap();
        ks.push(k);
        ks.len() - 1
    }

    /// Insert a TCP handle into the registry, returning its index.
    pub(crate) fn tcp_insert(&self, h: TcpHandle) -> i64 {
        let mut reg = self.shared.tcp.lock().unwrap();
        reg.push(Some(h));
        (reg.len() - 1) as i64
    }

    /// Clone the `Arc` for a live handle OUT of the registry (blocking I/O then
    /// happens without the lock). Panics (catchably at the frontend boundary)
    /// for a closed or bogus handle.
    pub(crate) fn tcp_get(&self, bits: u64, who: &str) -> TcpHandle {
        let h = match self.decode(bits) {
            Val::Int(n) => n as usize,
            _ => panic!("{who}: not a tcp handle"),
        };
        let reg = self.shared.tcp.lock().unwrap();
        match reg.get(h) {
            Some(Some(t)) => t.clone(),
            _ => panic!("{who}: closed or unknown tcp handle {h}"),
        }
    }

    /// Intern a literal into the constant pool, returning its id. The pool is a
    /// GC root, so the literal survives collection and is rewritten if it moves.
    /// The ONE canonical keyword object named `name` — allocating it on first
    /// use and returning that same object forever after. Every keyword
    /// construction funnels here (the reader's `:foo`, `::foo` resolution, and
    /// the `keyword` fn), which is what makes keyword identity work:
    /// `(identical? :a :a)`, `(identical? (keyword "a") :a)` are true, exactly
    /// as in Clojure AND ClojureScript.
    ///
    /// Keyword identity IS name identity: `name` is the interned name symbol
    /// (already namespace-qualified for `:ns/foo`), so two spellings of the
    /// same keyword agree by construction.
    pub fn intern_keyword(&mut self, name: Sym) -> u64 {
        if let Some(&id) = self.shared.keywords.lock().unwrap().get(&name) {
            return self.get_const(id);
        }
        // Build OUTSIDE the lock: `alloc_record` can take heap locks, and
        // holding the keyword lock across them is a lock-order inversion.
        let ty = self.intern("Keyword");
        let namebits = self.encode(Val::Sym(name));
        let bits = M::R::enc_ref(self.alloc_record(ty, &[namebits]));
        let id = self.intern_const(bits);
        // Re-check under the lock: a sibling thread may have interned the same
        // name while we were building. First writer wins, so the canonical
        // object stays unique; ours is garbage the collector will take.
        let mut tbl = self.shared.keywords.lock().unwrap();
        let winner = *tbl.entry(name).or_insert(id);
        drop(tbl);
        self.get_const(winner)
    }

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
                let g = M::R::as_ref(bits);
                match unsafe { g.type_id() } {
                    crate::heap::kind::BIGINT => Val::Int(unsafe {
                        g.raw_i128(&self.shared.types[crate::heap::kind::BIGINT as usize], 0)
                    }),
                    crate::heap::kind::BOXFLOAT => Val::Float(unsafe {
                        f64::from_bits(
                            g.raw_word(&self.shared.types[crate::heap::kind::BOXFLOAT as usize], 0),
                        )
                    }),
                    t if t == crate::heap::kind::INVALID || t as usize >= crate::heap::kind::COUNT => {
                        panic!(
                            "use-after-move: 0x{bits:x} points into a collected space \
                             (header type_id {t}); re-read through its root/handle"
                        )
                    }
                    _ => Val::Ref(g),
                }
            }
        }
    }

    /// Allocate a string value (used by the frontend reader for string literals).
    pub fn alloc_str(&mut self, s: String) -> u64 {
        let g = self.alloc(Obj::Str(s));
        M::R::enc_ref(g)
    }
    /// Allocate a character value (used by the frontend reader for `#\c` literals).
    pub fn alloc_char(&mut self, c: char) -> u64 {
        let g = self.alloc(Obj::Char(c));
        M::R::enc_ref(g)
    }
    /// A string's bytes as `&str` (hot read path; `view` in one step).
    #[inline]
    pub fn str_view(&self, bits: u64) -> Option<&str> {
        if M::R::tag_of(bits) != RawTag::Ref {
            return None;
        }
        let g = M::R::as_ref(bits);
        unsafe {
            if g.type_id() != crate::heap::kind::STR {
                return None;
            }
            let info = &self.shared.types[crate::heap::kind::STR as usize];
            Some(std::str::from_utf8_unchecked(g.bytes(info)))
        }
    }

    // ── lists ───────────────────────────────────────────────
    /// The empty list `()` — a heap value distinct from nil. ONE canonical
    /// singleton per runtime (its bits are a GC root, re-encoded on moves).
    pub fn enc_empty_list(&mut self) -> u64 {
        let cur = self.shared.empty_list.load(GLOBAL_LOAD);
        if cur != 0 {
            return cur;
        }
        let bits = M::R::enc_ref(self.alloc(Obj::EmptyList));
        // A racing thread may have won; keep whichever landed first.
        match self.shared.empty_list.compare_exchange(0, bits, GLOBAL_STORE, GLOBAL_LOAD) {
            Ok(_) => bits,
            Err(existing) => existing,
        }
    }
    pub fn is_empty_list(&self, bits: u64) -> bool {
        M::R::tag_of(bits) == RawTag::Ref
            && unsafe { M::R::as_ref(bits).type_id() } == crate::heap::kind::EMPTY_LIST
    }
    pub fn cons(&mut self, head: u64, tail: u64) -> u64 {
        // A cons chain terminates in nil, never in the standalone `()` value — so
        // `(cons x ())` is `(x)` and iteration (which stops at nil) is unaffected.
        let tail = if self.is_empty_list(tail) { self.enc_nil() } else { tail };
        let g = self.alloc(Obj::Cons { head, tail });
        M::R::enc_ref(g)
    }
    pub fn as_cons(&self, bits: u64) -> Option<(u64, u64)> {
        if M::R::tag_of(bits) != RawTag::Ref {
            return None;
        }
        let g = M::R::as_ref(bits);
        unsafe {
            match g.type_id() {
                crate::heap::kind::CONS => Some((g.field(0), g.field(1))),
                t if t == crate::heap::kind::INVALID
                    || t as usize >= crate::heap::kind::COUNT =>
                {
                    panic!(
                        "use-after-move: 0x{bits:x} points into a collected space \
                         (header type_id {t}); re-read through its root/handle"
                    )
                }
                _ => None,
            }
        }
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
        // Stage F1 fast paths: `equal` is the HAMT's key comparator, called
        // per probed entry — the decode/view glue here was a top profile
        // frame. Identical bits are equal for every category except a FLOAT
        // (NaN != NaN — immediate or boxed), and two non-ref words can only
        // be equal bit-identically (immediates are canonical) or as two
        // floats (+0.0 == -0.0). Everything else falls to the slow structure
        // below, whose semantics are unchanged.
        if a == b {
            match M::R::tag_of(a) {
                RawTag::Float => {} // possibly NaN: value compare below
                RawTag::Ref => {
                    let g = M::R::as_ref(a);
                    if self.raw_type_id(g) != crate::heap::kind::BOXFLOAT {
                        return true; // same object (non-float): equal
                    }
                    // The identical BOXED float: NaN must still be unequal.
                    let info = &self.shared.types[crate::heap::kind::BOXFLOAT as usize];
                    let f = f64::from_bits(unsafe { g.raw_word(info, 0) });
                    return f == f;
                }
                _ => return true,
            }
        }
        let (ta, tb) = (M::R::tag_of(a), M::R::tag_of(b));
        if ta != RawTag::Ref && tb != RawTag::Ref {
            return match (ta, tb) {
                (RawTag::Float, RawTag::Float) => M::R::imm_float(a) == M::R::imm_float(b),
                _ => false, // bit-unequal immediates: unequal (or cross-category)
            };
        }
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
                // RAW structural dispatch (Stage F1): one type_id load each —
                // no ObjView construction per node. BigInt/BoxFloat never
                // reach here (`decode` peels them to Int/Float above); HugeInt
                // was peeled by `as_huge`.
                use crate::heap::kind;
                let (tx, ty) = (self.raw_type_id(x), self.raw_type_id(y));
                if tx != ty {
                    return false;
                }
                unsafe {
                    match tx {
                        kind::CONS => {
                            self.equal(x.field(0), y.field(0)) && self.equal(x.field(1), y.field(1))
                        }
                        // All empty lists are equal (and only to each other, never nil).
                        kind::EMPTY_LIST => true,
                        // Ratios are reduced, so equal iff same num & den (never = an
                        // integer, since a reduced ratio has den > 1).
                        kind::RATIO => {
                            let info = &self.shared.types[kind::RATIO as usize];
                            x.raw_i128(info, 0) == y.raw_i128(info, 0)
                                && x.raw_i128(info, 16) == y.raw_i128(info, 16)
                        }
                        // Structural equality for aggregates (R7RS `equal?` on
                        // strings/vectors; ordered field equality for records — the
                        // general aggregate case). Order-INSENSITIVE collections
                        // (e.g. a hash-map) are a frontend concern layered on top.
                        kind::STR => {
                            let info = &self.shared.types[kind::STR as usize];
                            x.bytes(info) == y.bytes(info)
                        }
                        kind::CHAR => {
                            let info = &self.shared.types[kind::CHAR as usize];
                            x.raw_word_u32(info) == y.raw_word_u32(info)
                        }
                        kind::ARRAY => {
                            let ea = self.arr_slice(x);
                            let eb = self.arr_slice(y);
                            ea.len() == eb.len()
                                && ea.iter().zip(eb.iter()).all(|(&p, &q)| self.equal(p, q))
                        }
                        kind::RECORD => {
                            let info = &self.shared.types[kind::RECORD as usize];
                            x.raw_word(info, 0) == y.raw_word(info, 0)
                                && x.aux() == y.aux()
                                && x.values(info)
                                    .iter()
                                    .zip(y.values(info).iter())
                                    .all(|(&p, &q)| self.equal(p, q))
                        }
                        _ => false, // identity already handled; distinct other objects differ
                    }
                }
            }
            _ => false,
        }
    }
    fn num_lt(&self, a: u64, b: u64) -> bool {
        // Fast path: two immediate fixnums — raw i64 compare, no BigInt allocation.
        if M::R::is_immediate(Cat::Int)
            && M::R::tag_of(a) == RawTag::Int
            && M::R::tag_of(b) == RawTag::Int
        {
            return M::R::imm_int(a) < M::R::imm_int(b);
        }
        // Exact when both are integers of any size; falls back to f64 only when a
        // float is involved.
        if let (Some(x), Some(y)) = (self.as_int_big(a), self.as_int_big(b)) {
            return x.cmp(&y) == std::cmp::Ordering::Less;
        }
        // Exact rational comparison when a Ratio is involved and neither is a float:
        // na/da < nb/db  <=>  na*db < nb*da (denominators are positive).
        if self.as_ratio(a).is_some() || self.as_ratio(b).is_some() {
            if let (Some((na, da)), Some((nb, db))) = (self.as_exact_ratio(a), self.as_exact_ratio(b)) {
                if let (Some(l), Some(r)) = (na.checked_mul(db), nb.checked_mul(da)) {
                    return l < r;
                }
            }
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
            Prim::Add | Prim::FxAdd => self.arith(args[0], args[1], 0, i64::checked_add, i128::checked_add, |a, b| a + b, BigInt::add),
            Prim::Sub | Prim::FxSub => self.arith(args[0], args[1], 1, i64::checked_sub, i128::checked_sub, |a, b| a - b, BigInt::sub),
            Prim::Mul | Prim::FxMul => self.arith(args[0], args[1], 2, i64::checked_mul, i128::checked_mul, |a, b| a * b, BigInt::mul),
            Prim::Quot => self.int_div(args[0], args[1], "quot", |a, b| a / b),
            Prim::Rem => self.int_div(args[0], args[1], "rem", |a, b| a % b),
            Prim::Mod => self.int_div(args[0], args[1], "mod", |a, b| ((a % b) + b) % b),
            Prim::Div => self.divide(args[0], args[1]),
            Prim::StrCat => {
                let (a, b) = (self.as_str(args[0], "str-cat"), self.as_str(args[1], "str-cat"));
                let id = self.alloc(Obj::Str(format!("{a}{b}")));
                M::R::enc_ref(id)
            }
            Prim::StrOf => {
                // A string is its own raw content; anything else uses the neutral
                // printer (correct for int/float/bool/nil/sym/char).
                let s = match self.decode(args[0]) {
                    Val::Ref(id) => match self.view_gc(id) {
                        ObjView::Str(s) => s.to_string(),
                        _ => self.print(args[0]),
                    },
                    _ => self.print(args[0]),
                };
                let id = self.alloc(Obj::Str(s));
                M::R::enc_ref(id)
            }
            Prim::Lt | Prim::FxLt => {
                // Fast path: two immediate fixnums compare as raw i64 — no BigInt
                // for the type check OR the comparison. Without this, every `<` in a
                // hot loop allocated FOUR transient BigInts (two in the non-number
                // guard below, two more in `num_lt`), which dominated numeric code.
                if M::R::is_immediate(Cat::Int)
                    && M::R::tag_of(args[0]) == RawTag::Int
                    && M::R::tag_of(args[1]) == RawTag::Int
                {
                    return self.encode(Val::Bool(M::R::imm_int(args[0]) < M::R::imm_int(args[1])));
                }
                // `<` on a non-number is a CATCHABLE error (Clojure throws a
                // ClassCastException), not a hard abort.
                if self.as_int_big(args[0]).is_none() && self.num_as_f64(args[0]).is_none()
                    || self.as_int_big(args[1]).is_none() && self.num_as_f64(args[1]).is_none()
                {
                    let id = self.alloc(Obj::Str("< on non-number".to_string()));
                    self.signal_throw(M::R::enc_ref(id));
                    return self.enc_nil();
                }
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
            Prim::Keyword => {
                let Val::Sym(name) = self.decode(args[0]) else {
                    panic!("%keyword: name must be a symbol, got {}", self.print(args[0]))
                };
                self.intern_keyword(name)
            }
            Prim::StrLen => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("string-length: not a string");
                };
                let ObjView::Str(s) = self.view_gc(id) else {
                    panic!("string-length: not a string");
                };
                let n = s.chars().count() as i128;
                self.encode(Val::Int(n))
            }
            Prim::CharToInt => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("char->integer: not a char");
                };
                let ObjView::Char(c) = self.view_gc(id) else {
                    panic!("char->integer: not a char");
                };
                self.encode(Val::Int(c as i128))
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
                let id = self.alloc_vector(args);
                M::R::enc_ref(id)
            }
            Prim::VectorRef => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("vector-ref: not a vector");
                };
                let Val::Int(i) = self.decode(args[1]) else {
                    panic!("vector-ref: index must be an int");
                };
                let ObjView::Vector { elems, .. } = self.view_gc(id) else {
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
                assert_eq!(unsafe { id.type_id() }, crate::heap::kind::ARRAY, "vector-set!: not a vector");
                let slot = self
                    .arr_slice_mut(id)
                    .get_mut(i as usize)
                    .unwrap_or_else(|| panic!("vector-set!: index {i} out of range"));
                *slot = args[2];
                self.enc_nil()
            }
            Prim::MakeArray => {
                let Val::Int(n) = self.decode(args[0]) else {
                    panic!("make-array: size must be an int");
                };
                let nil = self.enc_nil();
                let id = self.alloc_vector_nil(n as usize, nil);
                M::R::enc_ref(id)
            }
            Prim::AClone => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("aclone: not an array");
                };
                let ObjView::Vector { elems, .. } = self.view_gc(id) else {
                    panic!("aclone: not an array");
                };
                let nid = self.alloc_vector(elems);
                M::R::enc_ref(nid)
            }
            Prim::PvConj => self.pv_conj(args[0], args[1]),
            // Conj a whole chunk's run onto a PV in one native call (a pv_conj
            // loop in Rust), avoiding a %pv-conj FFI per element.
            Prim::PvConjChunk => {
                let arr = self.arr_clone(args[1]);
                let off = match self.decode(args[2]) { Val::Int(i) => i as usize, _ => panic!("pv-conj-chunk: off") };
                let end = match self.decode(args[3]) { Val::Int(i) => i as usize, _ => panic!("pv-conj-chunk: end") };
                let mut pv = args[0];
                for &e in &arr[off..end] {
                    pv = self.pv_conj(pv, e);
                }
                pv
            }
            Prim::PvFromArray => self.pv_from_array(args[0]),
            Prim::SortArr => {
                let Val::Ref(id) = self.decode(args[0]) else { panic!("sort-arr: not an array") };
                let ObjView::Vector { elems, .. } = self.view_gc(id) else {
                    panic!("sort-arr: not an array")
                };
                let elems = elems.to_vec();
                // Homogeneous immediate fixnums?
                if elems.iter().all(|&e| matches!(M::R::tag_of(e), RawTag::Int)) {
                    let mut keyed: Vec<(i64, u64)> = elems
                        .iter()
                        .map(|&e| match self.decode(e) {
                            Val::Int(i) => (i as i64, e),
                            _ => unreachable!("tag-checked"),
                        })
                        .collect();
                    keyed.sort_by_key(|&(k, _)| k);
                    let sorted: Vec<u64> = keyed.into_iter().map(|(_, e)| e).collect();
                    return M::R::enc_ref(self.alloc_vector(&sorted));
                }
                // Homogeneous strings? (code-point order == `%str-cmp` == Rust str cmp)
                let all_str = elems.iter().all(|&e| {
                    matches!(self.decode(e), Val::Ref(i) if matches!(self.view_gc(i), ObjView::Str(_)))
                });
                if all_str {
                    let mut v = elems;
                    v.sort_by(|&a, &b| {
                        let (Val::Ref(ia), Val::Ref(ib)) = (self.decode(a), self.decode(b)) else {
                            unreachable!("checked str")
                        };
                        let (ObjView::Str(sa), ObjView::Str(sb)) = (self.view_gc(ia), self.view_gc(ib))
                        else {
                            unreachable!("checked str")
                        };
                        sa.cmp(sb)
                    });
                    return M::R::enc_ref(self.alloc_vector(&v));
                }
                self.enc_nil()
            }
            Prim::MultiFnNew => {
                // Build a multi-arity fn from per-clause closures: each fixed
                // clause registers under its own param count; the (at most one)
                // variadic clause serves every count >= its fixed params.
                let mut fixed: Vec<u64> = Vec::new();
                let mut variadic: Option<(usize, u64)> = None;
                for &f in args {
                    let Val::Ref(id) = self.decode(f) else {
                        panic!("%multifn: argument is not a closure");
                    };
                    let (np, va) = match self.view_gc(id) {
                        ObjView::Closure { nparams, variadic, .. } => (nparams, variadic),
                        _ => panic!("%multifn: argument is not a closure"),
                    };
                    if va {
                        assert!(variadic.is_none(), "%multifn: two variadic clauses");
                        variadic = Some((np, f));
                    } else {
                        if fixed.len() <= np {
                            fixed.resize(np + 1, 0);
                        }
                        assert!(fixed[np] == 0, "%multifn: duplicate arity {np}");
                        fixed[np] = f;
                    }
                }
                let id = self.alloc(Obj::MultiFn { fixed, variadic });
                M::R::enc_ref(id)
            }
            Prim::ApushChunk => {
                let src = self.arr_clone(args[1]);
                let off = match self.decode(args[2]) { Val::Int(i) => i as usize, _ => panic!("apush-chunk: off") };
                let end = match self.decode(args[3]) { Val::Int(i) => i as usize, _ => panic!("apush-chunk: end") };
                let Val::Ref(id) = self.decode(args[0]) else { panic!("apush-chunk: not an array"); };
                self.arr_extend(id, &src[off..end]);
                args[0]
            }
            Prim::PvNth => {
                let Val::Int(i) = self.decode(args[1]) else { panic!("pv-nth: index must be an int"); };
                self.pv_nth(args[0], i as i64)
            }
            Prim::PvAssoc => {
                let Val::Int(i) = self.decode(args[1]) else { panic!("pv-assoc: index must be an int"); };
                self.pv_assoc(args[0], i as i64, args[2])
            }
            // Cache a LazySeq's forced value in place: field 0 = v, field 1 = true.
            Prim::LazyRealize => {
                let Val::Ref(id) = self.decode(args[0]) else { panic!("lazy-realize!: not a record"); };
                let tru = self.encode(Val::Bool(true));
                assert_eq!(unsafe { id.type_id() }, crate::heap::kind::RECORD, "lazy-realize!: not a record");
                let fs = self.values_mut(id);
                fs[0] = args[1];
                fs[1] = tru;
                args[1]
            }
            // Fill a whole range chunk (up to 32 fixnums) in one native call — the
            // producer for `range`'s ChunkedCons, replacing 32 interpreted
            // `%cell-set!` calls with one Rust loop.
            Prim::RangeFill => {
                let Val::Int(start) = self.decode(args[0]) else { panic!("range-fill: start must be an int"); };
                let Val::Int(end) = self.decode(args[1]) else { panic!("range-fill: end must be an int"); };
                let Val::Int(step) = self.decode(args[2]) else { panic!("range-fill: step must be an int"); };
                let mut v = Vec::with_capacity(32);
                let mut j = start;
                if step > 0 {
                    while j < end && v.len() < 32 {
                        v.push(self.encode(Val::Int(j)));
                        j += step;
                    }
                } else {
                    while j > end && v.len() < 32 {
                        v.push(self.encode(Val::Int(j)));
                        j += step;
                    }
                }
                self.mk_array(v)
            }
            // Native HAMT trie ops — see the `hamt_*` methods below for the
            // full port of PersistentHashMap's -inode-assoc/-lookup/-without.
            Prim::HamtAssoc => {
                let (new_root, added) = self.hamt_map_assoc(args[0], args[1], args[2]);
                let addedb = self.encode(Val::Bool(added));
                self.mk_array(vec![new_root, addedb])
            }
            Prim::HamtLookup => self.hamt_map_lookup(args[0], args[1], args[2]),
            Prim::HamtWithout => self.hamt_map_without(args[0], args[1]),
            // Stage F3 transients (see the `tv_*`/`tam_*`/`thm_*` methods).
            Prim::TvNew => self.tv_new(args[0]),
            Prim::TvConj => self.tv_conj(args[0], args[1]),
            Prim::TvAssoc => {
                let i = self.raw_i64(args[1], "assoc!: index must be an int");
                self.tv_assoc(args[0], i, args[2])
            }
            Prim::TvNth => {
                let i = self.raw_i64(args[1], "tv-nth: index must be an int");
                self.tv_nth(args[0], i)
            }
            Prim::TvPop => self.tv_pop(args[0]),
            Prim::TvPersistent => self.tv_persistent(args[0]),
            Prim::TamNew => self.tam_new(args[0]),
            Prim::TamAssoc => self.tam_assoc(args[0], args[1], args[2]),
            Prim::TamDissoc => self.tam_dissoc(args[0], args[1]),
            Prim::TamPersistent => self.tam_persistent(args[0]),
            Prim::ThmNew => self.thm_new(args[0]),
            Prim::ThmAssoc => self.thm_assoc(args[0], args[1], args[2]),
            Prim::ThmDissoc => self.thm_dissoc(args[0], args[1]),
            Prim::ThmPersistent => self.thm_persistent(args[0]),
            // Join an array of already-stringified elements in ONE native pass
            // (see `Prim::StrJoinArr`'s doc comment in ir.rs).
            Prim::StrJoinArr => {
                let Val::Ref(id) = self.decode(args[0]) else { panic!("str-join-arr: not an array"); };
                let elems: Vec<u64> = match self.view_gc(id) {
                    ObjView::Vector { elems, .. } => elems.to_vec(),
                    _ => panic!("str-join-arr: not an array"),
                };
                let sep = self.as_str(args[1], "str-join-arr");
                let mut out = String::new();
                for (i, &e) in elems.iter().enumerate() {
                    if i > 0 {
                        out.push_str(&sep);
                    }
                    out.push_str(&self.as_str(e, "str-join-arr"));
                }
                let sid = self.alloc(Obj::Str(out));
                M::R::enc_ref(sid)
            }
            // clojure.core's `compare` on strings is Java String.compareTo: the
            // code-point difference at the first mismatch, else the length
            // difference (NOT clamped to ±1). Matches for all BMP text (Rust
            // char == UTF-16 unit there); the previous in-language `-str<` only
            // returned ±1/0, so this is strictly closer to Clojure.
            Prim::StrCmp => {
                let a = self.as_str(args[0], "str-cmp");
                let b = self.as_str(args[1], "str-cmp");
                let mut ai = a.chars();
                let mut bi = b.chars();
                let c: i32 = loop {
                    match (ai.next(), bi.next()) {
                        (Some(x), Some(y)) => {
                            if x != y {
                                break x as i32 - y as i32;
                            }
                        }
                        (Some(_), None) => break 1 + ai.count() as i32,
                        (None, Some(_)) => break -1 - bi.count() as i32,
                        (None, None) => break 0,
                    }
                };
                self.encode(Val::Int(c as i128))
            }
            Prim::ArrPush => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("apush: not an array");
                };
                self.arr_extend(id, &[args[1]]);
                args[0]
            }
            Prim::ArrShift => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("ashift: not an array");
                };
                assert_eq!(unsafe { id.type_id() }, crate::heap::kind::ARRAY, "ashift: not an array");
                // Shift left in place and shrink (object identity preserved).
                // Held for the whole read-modify-write, like `arr_extend`.
                let _g = self.shared.heap_lock.lock().unwrap();
                let ws = self.arr_slice_mut(id);
                if ws.is_empty() {
                    self.enc_nil()
                } else {
                    let first = ws[0];
                    let len = ws.len();
                    ws.copy_within(1.., 0);
                    unsafe { id.set_aux((len - 1) as u32) };
                    first
                }
            }
            Prim::ArrClear => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("aclear: not an array");
                };
                let _g = self.shared.heap_lock.lock().unwrap();
                assert_eq!(unsafe { id.type_id() }, crate::heap::kind::ARRAY, "aclear: not an array");
                unsafe { id.set_aux(0) };
                args[0]
            }
            Prim::BitAnd => self.encode(Val::Int(self.as_i128(args[0]) & self.as_i128(args[1]))),
            Prim::BitOr => self.encode(Val::Int(self.as_i128(args[0]) | self.as_i128(args[1]))),
            Prim::BitXor => self.encode(Val::Int(self.as_i128(args[0]) ^ self.as_i128(args[1]))),
            Prim::BitShl => self.encode(Val::Int(self.as_i128(args[0]) << self.as_i128(args[1]))),
            Prim::BitShr => self.encode(Val::Int(self.as_i128(args[0]) >> self.as_i128(args[1]))),
            Prim::BitCount => self.encode(Val::Int(self.as_i128(args[0]).count_ones() as i128)),
            Prim::RegisterFields => {
                let Val::Sym(ty) = self.decode(args[0]) else {
                    panic!("register-fields: type must be a symbol");
                };
                let names: Vec<Sym> = self
                    .list_to_vec(args[1])
                    .into_iter()
                    .map(|f| match self.decode(f) {
                        Val::Sym(s) => s,
                        _ => panic!("register-fields: field names must be symbols"),
                    })
                    .collect();
                // Leak an immutable Vec and publish its pointer atomically (Release).
                // A prior registration for this type (rare re-deftype) is superseded;
                // the old box leaks, which is bounded (one per type definition).
                let boxed = Box::into_raw(Box::new(names));
                self.shared.field_names[ty as usize].store(boxed, Ordering::Release);
                self.enc_nil()
            }
            // The registered field-name symbols of a value's type, as a list (empty
            // if unregistered). Lets a frontend give records generic map behavior.
            Prim::MakeRecord => {
                let Val::Sym(type_id) = self.decode(args[0]) else { panic!("make-record: type must be a symbol"); };
                let fields = self.list_to_vec(args[1]);
                let id = self.alloc_record(type_id, &fields);
                M::R::enc_ref(id)
            }
            Prim::FieldNames => {
                let ty = self.type_tag(args[0]);
                let ptr = self.shared.field_names[ty as usize].load(Ordering::Acquire);
                if ptr.is_null() {
                    return self.vec_to_list(&[]);
                }
                let names: &Vec<Sym> = unsafe { &*ptr };
                let vals: Vec<u64> = names.iter().map(|&n| self.encode(Val::Sym(n))).collect();
                self.vec_to_list(&vals)
            }
            Prim::FieldByName => {
                let ty = self.type_tag(args[0]);
                let Val::Sym(name) = self.decode(args[1]) else {
                    panic!("field-by-name: field must be a symbol");
                };
                // Lock-free: one atomic pointer load per type, then a scan.
                let ptr = self.shared.field_names[ty as usize].load(Ordering::Acquire);
                if ptr.is_null() {
                    panic!("field access .-{} on unregistered type '{}'", self.sym_name(name), self.sym_name(ty));
                }
                let names: &Vec<Sym> = unsafe { &*ptr };
                let idx = names.iter().position(|&n| n == name).unwrap_or_else(|| {
                    panic!("no field '{}' on type '{}'", self.sym_name(name), self.sym_name(ty))
                });
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("field access on non-record");
                };
                let ObjView::Record { fields, .. } = self.view_gc(id) else {
                    panic!("field access on non-record");
                };
                fields[idx]
            }
            Prim::Hash => {
                let h = self.hash_value(args[0]);
                self.encode(Val::Int((h & 0x7fff_ffff) as i128))
            }
            Prim::VectorLen => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("vector-length: not a vector");
                };
                let ObjView::Vector { elems, .. } = self.view_gc(id) else {
                    panic!("vector-length: not a vector");
                };
                self.encode(Val::Int(elems.len() as i128))
            }
            Prim::Values => {
                let id = self.alloc_values(args);
                M::R::enc_ref(id)
            }
            Prim::ValuesToList => {
                // Unpack a `values` packet into a list; a lone value becomes a
                // one-element list so single-valued producers work too.
                if let Val::Ref(id) = self.decode(args[0]) {
                    if let ObjView::Values(vals) = self.view_gc(id) {
                        let vals = vals.to_vec();
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
            Prim::List => {
                if args.is_empty() {
                    self.enc_empty_list()
                } else {
                    self.vec_to_list(args)
                }
            }
            Prim::Cons => self.cons(args[0], args[1]),
            Prim::First => {
                if let Some((h, _)) = self.as_cons(args[0]) {
                    h
                } else if let Some((arr, off, _, _)) = self.as_chunked(args[0]) {
                    self.arr_at(arr, off as usize)
                } else {
                    self.enc_nil()
                }
            }
            Prim::Rest => {
                if let Some((_, t)) = self.as_cons(args[0]) {
                    t
                } else if let Some((arr, off, end, more)) = self.as_chunked(args[0]) {
                    if off + 1 < end {
                        self.mk_chunked(arr, off + 1, end, more)
                    } else {
                        more
                    }
                } else {
                    self.enc_nil()
                }
            }
            Prim::IsNil => {
                let r = matches!(self.decode(args[0]), Val::Nil);
                self.encode(Val::Bool(r))
            }
            Prim::Println => {
                let s = self.str_form(args[0]);
                println!("{s}");
                self.enc_nil()
            }
            Prim::Print => {
                let s = self.str_form(args[0]);
                print!("{s}");
                use std::io::Write as _;
                let _ = std::io::stdout().flush();
                self.enc_nil()
            }
            Prim::ErrPrint => {
                let s = self.str_form(args[0]);
                eprint!("{s}");
                use std::io::Write as _;
                let _ = std::io::stderr().flush();
                self.enc_nil()
            }
            Prim::CurrentNs => match self.eval_bridge.clone() {
                Some(b) => b.current_ns(self),
                None => panic!("%current-ns: no eval bridge installed"),
            },
            Prim::Nanos => {
                // Monotonic, arbitrary origin (first use) — System/nanoTime's shape.
                static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let e = EPOCH.get_or_init(std::time::Instant::now);
                self.encode(Val::Int(e.elapsed().as_nanos() as i128))
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
                let id = self.alloc_record(type_id, &args[1..]);
                M::R::enc_ref(id)
            }
            Prim::TypeOf => {
                let s = self.type_tag(args[0]);
                M::R::enc_sym(s)
            }
            Prim::Throw => {
                // Set the thread's throw signal and return a dummy; the caller
                // checks `pending()` and bubbles up to the nearest `Ir::Try`. No
                // panic — a plain signal (see `Signal`).
                self.signal_throw(args[0]);
                self.enc_nil()
            }
            Prim::NFields => {
                let n = match self.decode(args[0]) {
                    Val::Ref(id) => match self.view_gc(id) {
                        ObjView::Record { fields, .. } => fields.len() as i128,
                        _ => 0,
                    },
                    _ => 0,
                };
                self.encode(Val::Int(n))
            }
            // Join a future's worker thread and cache its value. No backend
            // needed — this only blocks and reads a shared slot.
            Prim::Await => {
                let slot = match self.view(args[0]) {
                    ObjView::Future(s) => s,
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
            // ── dynamic vars: a per-thread binding stack, so every tier runs them.
            Prim::DynGet => {
                let s = match self.decode(args[0]) {
                    Val::Sym(s) => s,
                    _ => panic!("%dyn-get: not a symbol"),
                };
                match self.dyn_stack.iter().rev().find(|e| e.0 == s) {
                    Some(&(_, v)) => v,
                    None => self.global(s).unwrap_or_else(|| self.encode(Val::Nil)),
                }
            }
            Prim::DynSet => {
                let s = match self.decode(args[0]) {
                    Val::Sym(s) => s,
                    _ => panic!("%dyn-set: not a symbol"),
                };
                match self.dyn_stack.iter_mut().rev().find(|e| e.0 == s) {
                    Some(e) => e.1 = args[1],
                    // set! on a dynamic with no active binding mutates the root.
                    None => {
                        self.set_global_val(s, args[1]);
                    }
                }
                args[1]
            }
            Prim::DynMark => {
                self.dyn_stack.push((DYN_MARK, 0));
                self.encode(Val::Nil)
            }
            Prim::DynBind => {
                let s = match self.decode(args[0]) {
                    Val::Sym(s) => s,
                    _ => panic!("%dyn-bind: not a symbol"),
                };
                self.dyn_stack.push((s, args[1]));
                self.encode(Val::Nil)
            }
            Prim::DynUnwind => {
                while let Some((s, _)) = self.dyn_stack.pop() {
                    if s == DYN_MARK {
                        break;
                    }
                }
                self.encode(Val::Nil)
            }
            // First-class vars: read/write a global by symbol (the Var handle wraps
            // the qualified sym). `%global-get` throws a catchable error if unbound.
            Prim::GlobalGet => {
                let s = match self.decode(args[0]) {
                    Val::Sym(s) => s,
                    _ => panic!("%global-get: not a symbol"),
                };
                match self.global(s) {
                    Some(v) => v,
                    None => {
                        let name = self.sym_name(s).to_string();
                        let id = self.alloc(Obj::Str(format!("Unbound var: {name}")));
                        self.signal_throw(M::R::enc_ref(id));
                        self.enc_nil()
                    }
                }
            }
            Prim::GlobalSet => {
                let s = match self.decode(args[0]) {
                    Val::Sym(s) => s,
                    _ => panic!("%global-set: not a symbol"),
                };
                self.define_global(s, args[1]);
                args[1]
            }
            Prim::GlobalBound => {
                let s = match self.decode(args[0]) {
                    Val::Sym(s) => s,
                    _ => panic!("%global-bound?: not a symbol"),
                };
                M::R::enc_bool(self.global_defined(s))
            }
            Prim::SymName => {
                let s = match self.decode(args[0]) {
                    Val::Sym(s) => s,
                    _ => panic!("%sym-name: not a symbol"),
                };
                let full = self.sym_name(s);
                let name = full.rsplit_once('/').map(|(_, n)| n).unwrap_or(full).to_string();
                let id = self.alloc(Obj::Str(name));
                M::R::enc_ref(id)
            }
            Prim::SymNs => {
                let s = match self.decode(args[0]) {
                    Val::Sym(s) => s,
                    _ => panic!("%sym-ns: not a symbol"),
                };
                let full = self.sym_name(s);
                match full.rsplit_once('/') {
                    Some((ns, _)) => {
                        let id = self.alloc(Obj::Str(ns.to_string()));
                        M::R::enc_ref(id)
                    }
                    None => self.enc_nil(),
                }
            }
            Prim::VarFlags => {
                let s = match self.decode(args[0]) {
                    Val::Sym(s) => s,
                    _ => panic!("%var-flags: not a symbol"),
                };
                self.encode(Val::Int(self.var_flags(s) as i128))
            }
            Prim::NsInterns => {
                let ns = match self.decode(args[0]) {
                    Val::Sym(s) => self.sym_name(s).to_string(),
                    _ => panic!("%ns-interns: not a symbol"),
                };
                let vals: Vec<u64> =
                    self.ns_var_syms(&ns).into_iter().map(|s| self.encode(Val::Sym(s))).collect();
                self.vec_to_list(&vals)
            }
            Prim::AllNs => {
                let names = self.all_ns_names();
                let vals: Vec<u64> =
                    names.iter().map(|n| self.encode(Val::Sym(self.intern(n)))).collect();
                self.vec_to_list(&vals)
            }
            Prim::MethodTypes => {
                let method = match self.decode(args[0]) {
                    Val::Sym(s) => s,
                    _ => panic!("%method-types: not a symbol"),
                };
                let sentinel = self.intern("-protocol-default");
                let tys: Vec<Sym> = {
                    let tables = self.shared.tables.lock().unwrap();
                    tables
                        .methods
                        .keys()
                        .filter(|(m, ty)| *m == method && *ty != sentinel)
                        .map(|(_, ty)| *ty)
                        .collect()
                };
                let vals: Vec<u64> = tys.iter().map(|&t| self.encode(Val::Sym(t))).collect();
                self.vec_to_list(&vals)
            }
            Prim::MethodHasType => {
                let Val::Sym(method) = self.decode(args[0]) else {
                    panic!("%method-has-type?: method must be a symbol")
                };
                let Val::Sym(ty) = self.decode(args[1]) else {
                    panic!("%method-has-type?: type must be a symbol")
                };
                // The `-protocol-default` fallback entry is not a satisfying
                // type — `MethodTypes` filters it out and so must this.
                let sentinel = self.intern("-protocol-default");
                let r = if ty == sentinel {
                    false
                } else {
                    self.shared.tables.lock().unwrap().methods.contains_key(&(method, ty))
                };
                self.encode(Val::Bool(r))
            }
            // read-string / eval / macroexpand-1: re-enter the reader + compiler via
            // the frontend-installed bridge. Clone the Arc out first so `self` is free
            // to be borrowed mutably by the bridge call.
            Prim::ReadString => match self.eval_bridge.clone() {
                Some(b) => b.read_string(self, args[0]),
                None => panic!("read-string: no eval bridge installed"),
            },
            Prim::Eval => match self.eval_bridge.clone() {
                Some(b) => b.eval(self, args[0]),
                None => panic!("eval: no eval bridge installed"),
            },
            Prim::MacroExpand1 => match self.eval_bridge.clone() {
                Some(b) => b.macroexpand_1(self, args[0]),
                None => panic!("macroexpand-1: no eval bridge installed"),
            },
            Prim::Numerator => {
                let n = match self.as_ratio(args[0]) {
                    Some((n, _)) => n,
                    None => self.as_int_big(args[0]).and_then(|b| b.to_i128()).unwrap_or_else(|| {
                        panic!("numerator: not a ratio/integer")
                    }),
                };
                self.alloc_bigint(BigInt::from_i128(n))
            }
            Prim::Denominator => {
                let d = self.as_ratio(args[0]).map(|(_, d)| d).unwrap_or(1);
                self.alloc_bigint(BigInt::from_i128(d))
            }
            Prim::BigIntP => {
                // A boxed integer `decode`s transparently to `Val::Int`, so check the
                // RAW heap representation instead.
                let is_big = M::R::tag_of(args[0]) == RawTag::Ref
                    && matches!(self.view(args[0]), ObjView::BigInt(_) | ObjView::HugeInt(_));
                self.encode(Val::Bool(is_big))
            }
            Prim::ToLong => {
                // `(int \a)` => 97: a char coerces to its code point, same as
                // `(int c)` does in real Clojure.
                if let Val::Ref(id) = self.decode(args[0]) {
                    if let ObjView::Char(c) = self.view_gc(id) {
                        return self.encode(Val::Int(c as i128));
                    }
                }
                if let Some((n, d)) = self.as_ratio(args[0]) {
                    return self.alloc_bigint(BigInt::from_i128(n / d)); // truncates toward zero
                }
                if let Some(b) = self.as_int_big(args[0]) {
                    return self.alloc_bigint(b); // already an integer of any size
                }
                match self.num_as_f64(args[0]) {
                    Some(f) => self.alloc_bigint(BigInt::from_i128(f.trunc() as i128)),
                    None => panic!("long: not a number"),
                }
            }
            Prim::SymbolOf => {
                let s = self.as_str(args[0], "symbol");
                let sym = self.intern(&s);
                self.encode(Val::Sym(sym))
            }
            Prim::VarArglists => {
                let s = match self.decode(args[0]) {
                    Val::Sym(s) => s,
                    _ => panic!("%var-arglists: not a symbol"),
                };
                self.get_var_arglists(s).unwrap_or_else(|| self.enc_nil())
            }
            Prim::StrChars => {
                let cs: Vec<char> = match self.decode(args[0]) {
                    Val::Ref(id) => match self.view_gc(id) {
                        ObjView::Str(s) => s.chars().collect(),
                        _ => panic!("%str->chars: not a string"),
                    },
                    _ => panic!("%str->chars: not a string"),
                };
                let vals: Vec<u64> = cs
                    .into_iter()
                    .map(|c| {
                        let id = self.alloc(Obj::Char(c));
                        M::R::enc_ref(id)
                    })
                    .collect();
                self.vec_to_list(&vals)
            }
            Prim::StrToBytes => {
                let bytes: Vec<u64> = match self.decode(args[0]) {
                    Val::Ref(id) => match self.view_gc(id) {
                        // SIGNED bytes (the JVM's `byte`), so in-language wire
                        // code round-trips exactly like Java's.
                        ObjView::Str(s) => s.bytes().map(|b| M::R::enc_int(b as i8 as i64)).collect(),
                        _ => panic!("%str->bytes: not a string"),
                    },
                    _ => panic!("%str->bytes: not a string"),
                };
                let id = self.alloc_vector(&bytes);
                M::R::enc_ref(id)
            }
            Prim::BytesToStr => {
                let bytes: Vec<u8> = match self.decode(args[0]) {
                    Val::Ref(id) => match self.view_gc(id) {
                        ObjView::Vector { elems, .. } => elems
                            .iter()
                            .map(|&b| match self.decode(b) {
                                Val::Int(n) => n as i8 as u8,
                                _ => panic!("%bytes->str: array element is not an int"),
                            })
                            .collect(),
                        _ => panic!("%bytes->str: not an array"),
                    },
                    _ => panic!("%bytes->str: not an array"),
                };
                // Invalid UTF-8 becomes U+FFFD, matching java.lang.String.
                let s = String::from_utf8_lossy(&bytes).into_owned();
                let id = self.alloc(Obj::Str(s));
                M::R::enc_ref(id)
            }
            Prim::TcpListen => {
                let port = match self.decode(args[0]) {
                    Val::Int(n) => n as u16,
                    _ => panic!("%tcp-listen: port must be an int"),
                };
                let l = std::net::TcpListener::bind(("127.0.0.1", port))
                    .unwrap_or_else(|e| panic!("%tcp-listen: {e}"));
                M::R::enc_int(self.tcp_insert(TcpHandle::Listener(Arc::new(l))))
            }
            Prim::TcpAccept => {
                let l = match self.tcp_get(args[0], "%tcp-accept") {
                    TcpHandle::Listener(l) => l,
                    _ => panic!("%tcp-accept: not a listener handle"),
                };
                // Blocking accept OUTSIDE the registry lock.
                let (s, _) = l.accept().unwrap_or_else(|e| panic!("%tcp-accept: {e}"));
                M::R::enc_int(self.tcp_insert(TcpHandle::Stream(Arc::new(s))))
            }
            Prim::TcpRead => {
                let s = match self.tcp_get(args[0], "%tcp-read") {
                    TcpHandle::Stream(s) => s,
                    _ => panic!("%tcp-read: not a stream handle"),
                };
                use std::io::Read;
                let mut buf = [0u8; 1];
                // EOF and connection errors both read as -1: stream-end, the
                // way in-language stream code already treats it.
                match (&*s).read(&mut buf) {
                    Ok(0) | Err(_) => M::R::enc_int(-1),
                    Ok(_) => M::R::enc_int(buf[0] as i64),
                }
            }
            Prim::TcpWrite => {
                let s = match self.tcp_get(args[0], "%tcp-write") {
                    TcpHandle::Stream(s) => s,
                    _ => panic!("%tcp-write: not a stream handle"),
                };
                let bytes: Vec<u8> = match self.decode(args[1]) {
                    Val::Ref(id) => match self.view_gc(id) {
                        ObjView::Vector { elems, .. } => elems
                            .iter()
                            .map(|&b| match self.decode(b) {
                                Val::Int(n) => n as i8 as u8,
                                _ => panic!("%tcp-write: array element is not an int"),
                            })
                            .collect(),
                        _ => panic!("%tcp-write: not an array"),
                    },
                    _ => panic!("%tcp-write: not an array"),
                };
                use std::io::Write;
                (&*s).write_all(&bytes).unwrap_or_else(|e| panic!("%tcp-write: {e}"));
                (&*s).flush().ok();
                M::R::enc_nil()
            }
            Prim::TcpClose => {
                let h = match self.decode(args[0]) {
                    Val::Int(n) => n as usize,
                    _ => panic!("%tcp-close: not a handle"),
                };
                let mut reg = self.shared.tcp.lock().unwrap();
                if let Some(slot) = reg.get_mut(h) {
                    if let Some(TcpHandle::Stream(s)) = slot {
                        s.shutdown(std::net::Shutdown::Both).ok();
                    }
                    *slot = None; // dropping a listener closes it
                }
                M::R::enc_nil()
            }
            Prim::TcpLocalPort => {
                let port = match self.tcp_get(args[0], "%tcp-local-port") {
                    TcpHandle::Listener(l) => l.local_addr().map(|a| a.port()).unwrap_or(0),
                    TcpHandle::Stream(s) => s.local_addr().map(|a| a.port()).unwrap_or(0),
                };
                M::R::enc_int(port as i64)
            }
            // ── atoms: real cross-thread compare-and-set ────────────────
            Prim::AtomNew => {
                let id = self.alloc(Obj::Atom(args[0]));
                M::R::enc_ref(id)
            }
            Prim::AtomGet => {
                let ObjView::Atom(a) = self.view(args[0]) else {
                    panic!("atom-get: not an atom");
                };
                a.load(Ordering::Acquire)
            }
            // BEAGLE'S EXACT CRASH SHAPE: a long-lived (so, promoted) atom
            // `swap!`-ed to a value freshly allocated this frame. The atom is
            // then the ONLY reference to that young object and no root names
            // it, so without the barrier the next minor reclaims it under the
            // atom's feet. Both stores are barriered — a failed CAS wrote
            // nothing, but marking a card costs one byte store and a spurious
            // card scan, where reasoning about "did the CAS win" costs a bug.
            Prim::AtomSet => {
                let g = self.atom_gc(args[0], "atom-set");
                let ObjView::Atom(a) = self.view(args[0]) else {
                    panic!("atom-set: not an atom");
                };
                a.store(args[1], Ordering::Release);
                self.write_barrier(g);
                args[1]
            }
            Prim::AtomCas => {
                let g = self.atom_gc(args[0], "atom-cas");
                let ObjView::Atom(a) = self.view(args[0]) else {
                    panic!("atom-cas: not an atom");
                };
                let ok = a
                    .compare_exchange(args[1], args[2], Ordering::AcqRel, Ordering::Acquire)
                    .is_ok();
                self.write_barrier(g);
                M::R::enc_bool(ok)
            }
            Prim::Field => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("field: not a record");
                };
                let Val::Int(i) = self.decode(args[1]) else {
                    panic!("field: index must be an int");
                };
                match self.view_gc(id) {
                    ObjView::Record { fields, .. } => fields[i as usize],
                    _ => panic!("field: not a record"),
                }
            }
        }
    }

    pub(crate) fn enc_nil(&self) -> u64 {
        M::R::enc_nil()
    }

    // ── dispatch axis ───────────────────────────────────────
    /// Swap the dispatch strategy. Nothing else changes — the axis is free.
    pub fn set_dispatch(&self, d: Box<dyn Dispatch>) {
        self.shared.tables.lock().unwrap().dispatch = d;
        // Swapping strategies invalidates every per-thread site cache (the new
        // strategy may resolve differently or need to observe every call).
        self.shared.dispatch_version.fetch_add(1, Ordering::Relaxed);
    }
    pub fn dispatch_stats(&self) -> crate::dispatch::DispatchStats {
        self.shared.tables.lock().unwrap().dispatch.stats()
    }
    /// The receiver's type tag (a record's `type_id`). `None` for non-records.
    pub fn type_of(&self, bits: u64) -> Option<Sym> {
        if let Val::Ref(id) = self.decode(bits) {
            if let ObjView::Record { type_id, .. } = self.view_gc(id) {
                return Some(type_id);
            }
        }
        None
    }

    /// Inline-cached `(.-field obj)`: read the field named `field` from record
    /// `obj`, using the per-site cache to skip the field-name scan on a monomorphic
    /// access. The cache packs `(type << 32) | index`; a hit needs only a type-tag
    /// compare (a field layout is fixed per type). Lock-free (relaxed load/store —
    /// the cache is a hint re-validated by the type check; the packed pair is
    /// written atomically, and a type's index is deterministic, so a stale hit is
    /// still correct).
    pub fn field_get(&self, site: usize, field: Sym, obj: u64) -> u64 {
        let ty = self.type_tag(obj);
        let cache = self.shared.field_ic[site].load(Ordering::Relaxed);
        let idx = if (cache >> 32) as u32 == ty {
            (cache & 0xffff_ffff) as usize
        } else {
            let ptr = self.shared.field_names[ty as usize].load(Ordering::Acquire);
            if ptr.is_null() {
                panic!(
                    "field access .-{} on unregistered type '{}'",
                    self.sym_name(field),
                    self.sym_name(ty)
                );
            }
            let names: &Vec<Sym> = unsafe { &*ptr };
            let idx = names.iter().position(|&n| n == field).unwrap_or_else(|| {
                panic!("no field '{}' on type '{}'", self.sym_name(field), self.sym_name(ty))
            });
            self.shared.field_ic[site].store(((ty as u64) << 32) | idx as u64, Ordering::Relaxed);
            idx
        };
        let Val::Ref(id) = self.decode(obj) else {
            panic!("field access on non-record");
        };
        let ObjView::Record { fields, .. } = self.view_gc(id) else {
            panic!("field access on non-record");
        };
        fields[idx]
    }

    // ── non-local control-flow signal (throw / escape) ──────────────────────
    /// Raise `(throw v)`: record the signal and return a dummy. The interpreter /
    /// JIT checks `pending()` after the throwing expression and bubbles up.
    pub fn signal_throw(&mut self, v: u64) {
        self.signal = Signal { kind: 1, value: v, tag: 0 };
    }
    /// Raise a non-local escape to the `%callec` with `tag`.
    pub fn signal_escape(&mut self, tag: u64, v: u64) {
        self.signal = Signal { kind: 2, value: v, tag };
    }
    /// Is a control-flow signal in flight on this thread?
    #[inline]
    pub fn pending(&self) -> bool {
        self.signal.kind != 0
    }
    /// Is the pending signal a `throw` (vs an escape)?
    #[inline]
    pub fn pending_throw(&self) -> bool {
        self.signal.kind == 1
    }
    /// The pending signal's carried value (thrown value / escape value).
    #[inline]
    pub fn signal_value(&self) -> u64 {
        self.signal.value
    }
    /// The pending escape's target tag (meaningful when `pending_escape()`).
    #[inline]
    pub fn signal_tag(&self) -> u64 {
        self.signal.tag
    }
    /// Take + clear the pending signal (a handler consumed it).
    pub fn take_signal(&mut self) -> Signal {
        std::mem::take(&mut self.signal)
    }
    /// Byte offset of the `signal.kind` flag, for the JIT's inline pending-check.
    pub fn signal_kind_offset() -> usize {
        core::mem::offset_of!(Runtime<M>, signal) + core::mem::offset_of!(Signal, kind)
    }

    /// A deterministic 32-bit content hash of any value (for the HAMT). Equal
    /// values (by `equal`) hash equal: ints by value, strings/symbols/chars by
    /// content, records/cons/arrays structurally, nil=0. FNV-1a style mixing; not
    /// tied to any host's exact hash (we don't interop with real cljs hashes).
    pub fn hash_value(&self, bits: u64) -> u32 {
        const FNV_OFFSET: u32 = 0x811c9dc5;
        const FNV_PRIME: u32 = 0x0100_0193;
        fn mix(h: u32, x: u32) -> u32 {
            (h ^ x).wrapping_mul(FNV_PRIME)
        }
        fn mix_str(mut h: u32, s: &str) -> u32 {
            for b in s.bytes() {
                h = mix(h, b as u32);
            }
            h
        }
        match self.decode(bits) {
            Val::Nil => 0,
            Val::Bool(b) => mix(FNV_OFFSET, b as u32 + 1),
            Val::Int(i) => mix(FNV_OFFSET, (i as u64 as u32) ^ ((i as u64 >> 32) as u32)),
            Val::Float(f) => mix(FNV_OFFSET, f.to_bits() as u32 ^ (f.to_bits() >> 32) as u32),
            Val::Sym(s) => mix_str(0x53_9d_11u32, self.sym_name(s)),
            // RAW dispatch on the header type id (Stage F1): hashing runs per
            // HAMT probe — no ObjView per object. BigInt/BoxFloat never reach
            // here (`decode` peeled them to Int/Float above).
            Val::Ref(id) => {
                use crate::heap::kind;
                match self.raw_type_id(id) {
                    kind::STR => {
                        let info = &self.shared.types[kind::STR as usize];
                        mix_str(FNV_OFFSET, unsafe {
                            std::str::from_utf8_unchecked(id.bytes(info))
                        })
                    }
                    kind::CHAR => {
                        let info = &self.shared.types[kind::CHAR as usize];
                        mix(0xc4a_u32, unsafe { id.raw_word_u32(info) })
                    }
                    kind::HUGEINT => {
                        // Cold by construction: reassemble via the view.
                        match self.view_gc(id) {
                            ObjView::HugeInt(b) => mix_str(FNV_OFFSET, &b.to_string()),
                            _ => unreachable!("type-checked"),
                        }
                    }
                    kind::RATIO => {
                        let info = &self.shared.types[kind::RATIO as usize];
                        let (n, d) = unsafe { (id.raw_i128(info, 0), id.raw_i128(info, 16)) };
                        mix(mix(FNV_OFFSET, n as u64 as u32), d as u64 as u32)
                    }
                    kind::RECORD => {
                        let info = &self.shared.types[kind::RECORD as usize];
                        let ty = unsafe { id.raw_word(info, 0) } as Sym;
                        let mut h = mix_str(0x9e37_79b9u32, self.sym_name(ty));
                        for &f in unsafe { id.values(info) } {
                            h = mix(h, self.hash_value(f));
                        }
                        h
                    }
                    kind::CONS => {
                        let mut h = 0x1000_193u32;
                        for x in self.list_to_vec(bits) {
                            h = mix(h, self.hash_value(x));
                        }
                        h
                    }
                    kind::ARRAY => {
                        let mut h = 0x27d4_eb2fu32;
                        for &x in self.arr_slice(id) {
                            h = mix(h, self.hash_value(x));
                        }
                        h
                    }
                    _ => mix(FNV_OFFSET, id.addr() as u32),
                }
            }
        }
    }

    /// The type TAG of ANY value as an interned symbol: a record's own `type_id`,
    /// or the built-in category tag (`List`/`Vector`/`String`/`Long`/`nil`/…).
    /// This is the general dispatch key (unlike `type_of`, which is records-only),
    /// so protocol/method dispatch can target primitives and built-in containers —
    /// exactly what an in-language collection library needs.
    pub fn type_tag(&self, bits: u64) -> Sym {
        let (idx, name) = match self.decode(bits) {
            Val::Int(_) => (TYPE_TAG_LONG, "Long"),
            Val::Float(_) => (TYPE_TAG_DOUBLE, "Double"),
            Val::Bool(_) => (TYPE_TAG_BOOLEAN, "Boolean"),
            Val::Nil => (TYPE_TAG_NIL, "nil"),
            Val::Sym(_) => (TYPE_TAG_SYMBOL, "Symbol"),
            Val::Ref(id) => match self.view_gc(id) {
                ObjView::Record { type_id, .. } => return type_id,
                ObjView::Cons { .. } => (TYPE_TAG_LIST, "List"),
                ObjView::EmptyList => (TYPE_TAG_EMPTYLIST, "EmptyList"),
                ObjView::Vector { .. } => (TYPE_TAG_VECTOR, "Vector"),
                ObjView::Str(_) => (TYPE_TAG_STRING, "String"),
                ObjView::Char(_) => (TYPE_TAG_CHAR, "Char"),
                ObjView::Closure { .. } | ObjView::MultiFn { .. } => (TYPE_TAG_FN, "Fn"),
                ObjView::BigInt(_) | ObjView::HugeInt(_) => (TYPE_TAG_LONG, "Long"),
                ObjView::Ratio(..) => (TYPE_TAG_RATIO, "Ratio"),
                ObjView::BoxFloat(_) => (TYPE_TAG_DOUBLE, "Double"),
                ObjView::Atom(_) => (TYPE_TAG_ATOM, "Atom"),
                ObjView::Future(_) => (TYPE_TAG_FUTURE, "Future"),
                _ => (TYPE_TAG_OBJECT, "Object"),
            },
        };
        self.intern_cached(&self.shared.type_tag_cache[idx], name)
    }
    pub fn register_method(&self, name: Sym, ty: Sym, imp: u64) {
        let mut t = self.shared.tables.lock().unwrap();
        t.method_names.insert(name);
        t.methods.insert((name, ty), imp);
        self.shared.dispatch_version.fetch_add(1, Ordering::Relaxed);
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
    /// Register the frontend's `seq` fn: backends call it (via `top`) to force
    /// a lazy node when they need to WALK a sequence natively (apply flatten).
    pub fn set_seq_fn(&self, name: Sym) {
        self.shared.seq_fn.store(name as u64 + 1, Ordering::Relaxed);
    }
    /// The current seq-fn value (re-read from globals, so GC-safe), if any.
    pub fn seq_handler(&self) -> Option<u64> {
        match self.shared.seq_fn.load(Ordering::Relaxed) {
            0 => None,
            v => self.global((v - 1) as Sym),
        }
    }
    /// One realized step of a sequence: `Some(None)` = end, `Some(Some((head,
    /// tail)))` = an element, `None` = an unrealizable node (no seq fn
    /// registered / not a sequence).
    pub fn seq_step(
        &mut self,
        top: &dyn crate::code::CodeSpace<M>,
        cur: u64,
    ) -> Option<Option<(u64, u64)>> {
        // Only the LAZY/FRONTEND arm invokes (a safepoint that relocates `cur`);
        // the CONS/chunked/nil arms neither allocate nor safepoint. This runs
        // once per SEQ ELEMENT, so the common case must pay NO rooting — root
        // ONLY across the invoke, not for the whole walk. (Stage J rooted the
        // whole walk with a flat push_root per element on EVERY sequence, lazy
        // or not; that showed up as the per-element tax on lazy-seq-heavy code
        // like `interleave`.)
        let mut cur = cur;
        loop {
            if self.is_nil_bits(cur) {
                return Some(None);
            }
            match self.raw_gc(cur) {
                Some(cid) => {
                    use crate::heap::kind;
                    match self.raw_type_id(cid) {
                        kind::CONS => return Some(Some(unsafe { (cid.field(0), cid.field(1)) })),
                        kind::EMPTY_LIST => return Some(None),
                        _ => {}
                    }
                    if let Some((arr, off, end, more)) = self.as_chunked(cur) {
                        let h = self.arr_at_pub(arr, off as usize);
                        let t = if off + 1 < end {
                            self.mk_chunked(arr, off + 1, end, more)
                        } else {
                            more
                        };
                        return Some(Some((h, t)));
                    }
                    // Lazy / frontend node: force ONE step through the registered
                    // `seq` — an INVOKE (safepoint), so `cur` rides the shadow
                    // stack ACROSS it and is re-read after. The no-progress check
                    // compares the post-invoke `forced` against the RE-READ `cur`;
                    // against a pre-invoke address it would miss "seq returned the
                    // same node" whenever a collection moved it — looping forever
                    // instead of erroring loudly. (`forced` is the invoke's return,
                    // current with no safepoint before the next iteration reads it.)
                    let Some(sf) = self.seq_handler() else { return None };
                    let slot = self.push_root(cur);
                    let forced = top.invoke(top, self, sf, &[cur]);
                    let cur_now = self.root_get(slot);
                    self.truncate_roots(slot);
                    if self.pending() {
                        return Some(None);
                    }
                    if forced == cur_now {
                        return None; // seq made no progress: not a sequence
                    }
                    cur = forced;
                }
                None => return None,
            }
        }
    }
    /// Flatten a whole sequence (cons / chunked / lazy) into a Vec, forcing
    /// through the registered `seq` fn as needed. The general `apply` path.
    /// The FIXED arities `callee` could serve, +1 — how far `plan_apply` must
    /// count the arglist before it can pick a clause. Clojure never counts the
    /// whole applied seq (`RestFn.applyTo` uses `RT.boundedLength`), and
    /// neither may we: `(apply + (range 1e9))` must not walk a billion
    /// elements to discover it wants the variadic clause, and an applied
    /// INFINITE seq must still work at all.
    fn apply_arity_bound(&mut self, callee: u64) -> usize {
        let Val::Ref(id) = self.decode(callee) else { return 1 };
        match self.view_gc(id) {
            ObjView::MultiFn { fixed, variadic } => {
                let vmin = variadic.map(|(min, _)| min + 1).unwrap_or(0);
                fixed.len().max(vmin)
            }
            ObjView::Closure { nparams, .. } => nparams + 1,
            _ => 1,
        }
    }

    /// Walk `seq` up to `bound` elements, forcing lazy nodes, and return
    /// `(count, tail_after_them)`. `count < bound` means the seq ENDED (an
    /// exact length); `count == bound` means "at least `bound`".
    fn seq_bounded_take(
        &mut self,
        top: &dyn crate::code::CodeSpace<M>,
        seq: u64,
        bound: usize,
    ) -> (Vec<u64>, u64) {
        // `seq_step` may INVOKE a lazy thunk (a safepoint), relocating both the
        // cursor and everything taken so far — same discipline as seq_flatten:
        // both live in the shadow stack for the walk.
        let cur_slot = self.push_root(seq);
        let out_base = self.root_depth();
        while self.root_depth() - out_base < bound {
            let cur = self.root_get(cur_slot);
            match self.seq_step(top, cur) {
                Some(Some((h, t))) => {
                    self.push_root(h);
                    self.set_root(cur_slot, t);
                }
                Some(None) => break,
                None => panic!(
                    "apply: unwalkable sequence node {} (no seq fn registered?)",
                    self.print(self.root_get(cur_slot))
                ),
            }
        }
        let taken: Vec<u64> = (out_base..self.root_depth()).map(|i| self.root_get(i)).collect();
        let tail = self.root_get(cur_slot);
        self.truncate_roots(cur_slot);
        (taken, tail)
    }

    /// `(apply callee leading… last)`. THE apply path — every backend's
    /// `Prim::Apply` calls this one, so the sharing semantics cannot drift
    /// between tiers.
    ///
    /// The rest arg rides `apply_rest_slot` rather than an extra `invoke`
    /// parameter, because that lands it at `mk_rest_seq` — the single point
    /// where ANY backend materializes a rest arg. One change, five tiers, no
    /// way for one of them to quietly keep copying.
    pub(crate) fn invoke_apply(
        &mut self,
        top: &dyn crate::code::CodeSpace<M>,
        callee: u64,
        leading: &[u64],
        last: u64,
    ) -> u64 {
        match self.plan_apply(top, callee, leading, last) {
            ApplyPlan::Signal => M::R::enc_nil(),
            ApplyPlan::Flat { callee, args } => top.invoke(top, self, callee, &args),
            ApplyPlan::WithRest { callee, fixed, rest } => {
                let slot = self.push_root(rest);
                self.apply_rest_slot = Some(slot);
                let r = top.invoke(top, self, callee, &fixed);
                // Clear unconditionally: an arity throw can return before any
                // frame is built, and a stale slot would silently become the
                // NEXT variadic call's rest arg.
                self.apply_rest_slot = None;
                self.truncate_roots(slot);
                r
            }
        }
    }

    /// Resolve `(apply callee leading… last)` into a concrete call.
    ///
    /// Clojure's `apply` does NOT copy: the callee's rest arg IS the applied
    /// seq (minus the fixed params taken off the front), so `identical?`
    /// holds, laziness survives, and the shape is preserved. Reproducing that
    /// means resolving the callee's arity HERE — before the frame is built —
    /// which is what this does.
    pub(crate) fn plan_apply(
        &mut self,
        top: &dyn crate::code::CodeSpace<M>,
        callee: u64,
        leading: &[u64],
        last: u64,
    ) -> ApplyPlan {
        let base = self.root_depth();
        let callee_slot = self.push_root(callee);
        for &a in leading {
            self.push_root(a);
        }
        let lead_base = base + 1;
        let last_slot = self.push_root(last);

        // `(seq last)` — Clojure applies the SEQ of the collection, which is
        // why `identical?` holds for a value that is its own seq (a list) but
        // not for a LazySeq (whose seq is the forced node, a different
        // object). Forces the HEAD only: an applied lazy tail must stay lazy.
        if let Some(sf) = self.seq_handler() {
            let l = self.root_get(last_slot);
            let seqd = top.invoke(top, self, sf, &[l]);
            if self.pending() {
                self.truncate_roots(base);
                return ApplyPlan::Signal;
            }
            self.set_root(last_slot, seqd);
        }

        let plan = loop {
            let c = self.root_get(callee_slot);
            let Val::Ref(id) = self.decode(c) else {
                panic!("value not callable: {}", self.print(c));
            };
            // Callable-object hook (e.g. keywords): redirect to
            // `(handler object args…)`, exactly as `invoke` does.
            if let ObjView::Record { .. } = self.view_gc(id) {
                if let Some(h) = self.apply_handler() {
                    let mut lead = vec![c];
                    lead.extend((lead_base..last_slot).map(|i| self.root_get(i)));
                    let last_now = self.root_get(last_slot);
                    self.truncate_roots(base);
                    return self.plan_apply(top, h, &lead, last_now);
                }
            }
            let bound = self.apply_arity_bound(c);
            // Only as far as arity selection needs — never the whole seq. The
            // probe's TAIL is discarded on purpose: the real split re-takes
            // from `last_slot` once the clause (and so `nparams`) is known.
            let want = bound.saturating_sub(last_slot - lead_base);
            let (taken, _probe_tail) =
                self.seq_bounded_take(top, self.root_get(last_slot), want);
            let argc_at_least = (last_slot - lead_base) + taken.len();

            // The take above forces lazy nodes, which INVOKES user code — a
            // safepoint. `c`/`id` were read BEFORE it and now point into
            // collected space, so re-read the callee through its root and
            // re-decode. (The verify heap catches this instantly: "stale
            // pointer into a collected space" out of the very next view_gc.)
            let c = self.root_get(callee_slot);
            let Val::Ref(id) = self.decode(c) else {
                panic!("value not callable: {}", self.print(c));
            };

            if matches!(self.view_gc(id), ObjView::MultiFn { .. }) {
                let sel = self.multifn_select(c, argc_at_least);
                if self.pending() {
                    break ApplyPlan::Signal;
                }
                let Some(sel) = sel else { break ApplyPlan::Signal };
                self.set_root(callee_slot, sel);
                continue;
            }
            // Escapes/continuations and non-variadic closures take every arg
            // positionally: flatten, and let `invoke` do the rest.
            //
            // ORDER MATTERS: `seq_flatten` forces lazy nodes, which INVOKES
            // user code — a safepoint that relocates the callee and the
            // leading args. So flatten FIRST, then re-read both through the
            // shadow stack, which the collector rewrites.
            let plain = match self.view_gc(id) {
                ObjView::Closure { variadic, .. } => !variadic,
                _ => true,
            };
            if plain {
                let rest = self.seq_flatten(top, self.root_get(last_slot));
                let mut args: Vec<u64> =
                    (lead_base..last_slot).map(|i| self.root_get(i)).collect();
                args.extend(rest);
                break ApplyPlan::Flat { callee: self.root_get(callee_slot), args };
            }
            let ObjView::Closure { nparams, .. } = self.view_gc(id) else {
                unreachable!("non-closure took the `plain` path above")
            };
            // Variadic: `nparams` come off the front of the arglist (leading
            // args first, then the seq); the REST IS THE REMAINING SEQ.
            let nlead = last_slot - lead_base;
            if taken.len() + nlead < nparams {
                // Too few args for this clause — `invoke` raises the arity
                // throw when it sees the short arg list.
                let mut args: Vec<u64> =
                    (lead_base..last_slot).map(|i| self.root_get(i)).collect();
                args.extend(taken);
                break ApplyPlan::Flat { callee: self.root_get(callee_slot), args };
            }
            // DROP the fixed params off the seq rather than re-consing the
            // ones the arity probe took: dropping preserves the shape (a
            // chunked tail stays chunked, which is exactly what Clojure
            // reports), whereas re-consing would hand the callee a Cons and
            // silently destroy the chunking.
            let need = nparams.saturating_sub(nlead);
            let (from_seq, tail) = self.seq_bounded_take(top, self.root_get(last_slot), need);
            // NORMALIZE the tail to a seq — Clojure's rest arg is `next`-shaped.
            // Only when we actually dropped something: with `need == 0` the tail
            // IS the top-level `(seq last)` already, and re-seq'ing it would
            // force a lazy node the callee has not asked for.
            //
            // This matters because `%cons` does NOT seq its tail (Clojure's
            // `RT.cons` does), and core's `apply` conses the leading args on:
            // `(apply f "L" (vec …))` reaches here as `(cons "L" <vector>)`, so
            // dropping "L" leaves the raw VECTOR. Handing that over made the
            // callee's rest arg a PersistentVector — `(seq? xs)` false.
            //
            // Everything the take produced rides the shadow stack across that
            // invoke: it is a safepoint like any other force.
            let scratch = self.root_depth();
            for &e in &from_seq {
                self.push_root(e);
            }
            let tail_slot = self.push_root(tail);
            if need > 0 {
                if let Some(sf) = self.seq_handler() {
                    let t = self.root_get(tail_slot);
                    let s = top.invoke(top, self, sf, &[t]);
                    if self.pending() {
                        self.truncate_roots(base);
                        return ApplyPlan::Signal;
                    }
                    self.set_root(tail_slot, s);
                }
            }
            let from_seq: Vec<u64> = (scratch..tail_slot).map(|i| self.root_get(i)).collect();
            let tail = self.root_get(tail_slot);
            self.truncate_roots(scratch);
            // Read the leading args AFTER all of the above: forcing INVOKES user
            // code — a safepoint that relocates every one of them. Values read
            // before it are stale the moment it returns; the shadow stack is
            // what the collector rewrites, so re-read from there.
            let leads: Vec<u64> = (lead_base..last_slot).map(|i| self.root_get(i)).collect();
            let mut fixed: Vec<u64> = Vec::with_capacity(nparams);
            fixed.extend_from_slice(&leads[..nparams.min(nlead)]);
            fixed.extend_from_slice(&from_seq);
            // Un-consumed LEADING args do go back on the front — Clojure conses
            // them too, which is why `(apply f 1 2 coll)` reports a plain Cons
            // rather than coll's shape.
            let rooted_rest = self.push_root(tail);
            for &e in leads[nparams.min(nlead)..].iter().rev() {
                let r = self.root_get(rooted_rest);
                let c2 = self.cons(e, r);
                self.set_root(rooted_rest, c2);
            }
            let rest = self.root_get(rooted_rest);
            self.truncate_roots(rooted_rest);
            // `(apply f [])` must give the callee nil, not `()` — Clojure's
            // rest arg is `next`-shaped: nil when exhausted.
            let rest = if self.is_nil_bits(rest) { self.enc_nil() } else { rest };
            // Re-read: the takes above safepoint, so `c` (read before them) is
            // stale. `fixed`/`rest` are already post-safepoint values.
            break ApplyPlan::WithRest { callee: self.root_get(callee_slot), fixed, rest };
        };
        self.truncate_roots(base);
        plan
    }

    pub fn seq_flatten(&mut self, top: &dyn crate::code::CodeSpace<M>, bits: u64) -> Vec<u64> {
        // `seq_step` INVOKES the frontend's seq fn to force a lazy node — a
        // safepoint, which relocates BOTH the cursor and every element already
        // collected. A bare `Vec<u64>` of accumulated elements would go stale
        // mid-walk and then be stored into the callee's frame by `invoke`, where
        // the NEXT collection trips over it in `visit_env`. So the cursor and the
        // accumulator both live in the shadow stack for the duration.
        let cur_slot = self.push_root(bits);
        let out_base = self.root_depth();
        loop {
            let cur = self.root_get(cur_slot);
            // Whole realized chunks copy in one extend (no per-element step).
            if let Some((arr, coff, cend, more)) = self.as_chunked(cur) {
                // Copy out before pushing: `arr_elems_pub` borrows the heap.
                let elems: Vec<u64> =
                    self.arr_elems_pub(arr)[coff as usize..cend as usize].to_vec();
                for e in elems {
                    self.push_root(e);
                }
                self.set_root(cur_slot, more);
                continue;
            }
            match self.seq_step(top, cur) {
                Some(Some((h, t))) => {
                    self.push_root(h);
                    self.set_root(cur_slot, t);
                }
                Some(None) => break,
                None => {
                    // A node we cannot walk and cannot force is a configuration
                    // error (frontend forgot set_seq_fn) — fail LOUDLY rather
                    // than silently truncating the argument list.
                    panic!(
                        "seq_flatten: unwalkable sequence node {} (no seq fn registered?)",
                        self.print(self.root_get(cur_slot))
                    );
                }
            }
        }
        let out: Vec<u64> = (out_base..self.root_depth()).map(|i| self.root_get(i)).collect();
        self.truncate_roots(cur_slot);
        out
    }
    /// Install the reader+compiler re-entry bridge (see `EvalBridge`). Set once, on
    /// the main handle, by the frontend's top-level driver.
    pub fn set_eval_bridge(&mut self, b: Arc<dyn EvalBridge<M>>) {
        self.eval_bridge = Some(b);
    }
    /// Drop the bridge (its raw pointers become invalid once the installer returns).
    pub fn clear_eval_bridge(&mut self) {
        self.eval_bridge = None;
    }
    /// The current apply-handler fn value (re-read from globals, so GC-safe), if any.
    pub fn apply_handler(&self) -> Option<u64> {
        match self.shared.apply_fn.load(Ordering::Relaxed) {
            0 => None,
            v => self.global((v - 1) as Sym),
        }
    }
    /// If `callee` is a multi-arity fn, the closure serving `argc` arguments
    /// (fixed arity first, else the variadic clause). `None` when `callee` is
    /// not a MultiFn; a catchable arity throw when no clause matches.
    /// `multifn_select` on the raw accessor layer (Stage G1): the select runs
    /// per BOUNCE in the JIT's tail chains — the decode/ObjView glue was a top
    /// profile frame. One type_id check (loud poison panic via `raw_type_id`),
    /// then direct reads of the clause table. Semantics identical to
    /// `multifn_select` below. (All callers live in the JIT tier.)
    #[cfg_attr(not(feature = "jit"), allow(dead_code))]
    pub(crate) fn multifn_select_raw(&mut self, callee: u64, argc: usize) -> Option<u64> {
        let g = self.raw_gc(callee)?;
        if self.raw_type_id(g) != crate::heap::kind::MULTIFN {
            return None;
        }
        let info = &self.shared.types[crate::heap::kind::MULTIFN as usize];
        let fixed = unsafe { g.values(info) };
        let mut sel = fixed.get(argc).copied().unwrap_or(0);
        if sel == 0 {
            let vmin = unsafe { g.raw_word(info, 0) };
            if vmin != u64::MAX && argc as u64 >= vmin {
                sel = unsafe { g.field(0) };
            }
        }
        if sel == 0 {
            let msg = format!("arity: no clause for {argc} args");
            let sid = self.alloc(Obj::Str(msg));
            self.signal_throw(M::R::enc_ref(sid));
            return Some(self.encode(Val::Nil));
        }
        Some(sel)
    }

    pub fn multifn_select(&mut self, callee: u64, argc: usize) -> Option<u64> {
        let Val::Ref(id) = self.decode(callee) else { return None };
        let (sel, known) = match self.view_gc(id) {
            ObjView::MultiFn { fixed, variadic } => {
                let f = fixed.get(argc).copied().unwrap_or(0);
                if f != 0 {
                    (f, true)
                } else if let Some((min, vf)) = variadic {
                    if argc >= min {
                        (vf, true)
                    } else {
                        (0, true)
                    }
                } else {
                    (0, true)
                }
            }
            _ => (0, false),
        };
        if !known {
            return None;
        }
        if sel == 0 {
            let msg = format!("arity: no clause for {argc} args");
            let sid = self.alloc(Obj::Str(msg));
            let thrown = M::R::enc_ref(sid);
            self.signal_throw(thrown);
            // Callers observe the pending signal and unwind.
            return Some(self.encode(Val::Nil));
        }
        Some(sel)
    }

    /// Resolve a call site via the current dispatch strategy (reads registry +
    /// updates the strategy's per-site cache), then invoke happens in the backend.
    /// The impl is copied out and the lock dropped before the caller invokes it.
    pub fn resolve_method(&self, site: usize, method: Sym, ty: Sym) -> Option<u64> {
        let t = self.shared.tables.lock().unwrap();
        t.dispatch.resolve(&t.methods, site, method, ty)
    }

    /// Resolve `method` for `ty`, falling back to an `Object` extension when the
    /// receiver's own type has no impl. `Object` is the universal root a protocol
    /// can be extended against to provide a DEFAULT (as ClojureScript's `default`
    /// / `Object` does), so e.g. `=` works on any value without every type
    /// implementing `IEquiv`.
    pub fn resolve_or_default(&self, site: usize, method: Sym, ty: Sym) -> Option<u64> {
        // Per-site, per-thread monomorphic cache: one epoch load + two compares
        // on the (overwhelmingly common) repeat-type case, instead of a mutex +
        // SipHash registry lookup per dispatch. The epoch folds the GC's
        // relocation count (a moved impl never hits) with the registry version
        // (a redefinition invalidates immediately).
        let reloc = self.shared.relocated.load(Ordering::Relaxed);
        let ver = self.shared.dispatch_version.load(Ordering::Relaxed);
        let epoch = reloc.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ver;
        {
            let ic = self.site_ic.borrow();
            if let Some(&(e, t, imp)) = ic.get(site) {
                if e == epoch && t == ty {
                    return Some(imp);
                }
            }
        }
        let resolved = self
            .resolve_method(site, method, ty)
            .or_else(|| self.resolve_method(site, method, self.intern("Object")));
        // Fill the per-thread cache only when the installed strategy is a pure
        // registry lookup (see `Dispatch::thread_cacheable`) — an observing
        // strategy (ICs, speculation) must see every repeat call.
        let cacheable = self.shared.tables.lock().unwrap().dispatch.thread_cacheable();
        if cacheable {
            if let Some(imp) = resolved {
                let mut ic = self.site_ic.borrow_mut();
                if ic.len() <= site {
                    // `Sym::MAX` is never a real type tag, so a fresh slot can't hit.
                    ic.resize(site + 1, (0, Sym::MAX, 0));
                }
                ic[site] = (epoch, ty, imp);
            }
        }
        resolved
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
    /// Integer division family (`quot`/`rem`/`mod`). Both operands must be
    /// integers that fit `i128` (the common case); arbitrary-precision operands
    /// and non-integers raise a clear error rather than silently degrading.
    /// `/`: exact integer division when it divides evenly; otherwise float
    /// division (no Ratio type). Integer division by zero errors; float `/0.0`
    /// follows IEEE (`inf`), as in Clojure.
    fn divide(&mut self, a: u64, b: u64) -> u64 {
        // Two ratios / ratio-and-integer: exact rational division = a * (1/b).
        if let (Some((na, da)), Some((nb, db))) = (self.as_exact_ratio(a), self.as_exact_ratio(b)) {
            if nb == 0 {
                panic!("Divide by zero");
            }
            if let (Some(n), Some(d)) = (na.checked_mul(db), da.checked_mul(nb)) {
                return self.make_ratio(n, d);
            }
        }
        if let (Some(x), Some(y)) = (self.as_int_big(a), self.as_int_big(b)) {
            if let (Some(xi), Some(yi)) = (x.to_i128(), y.to_i128()) {
                if yi == 0 {
                    panic!("Divide by zero");
                }
                // Exact integer quotient, else an exact Ratio (Clojure semantics —
                // `(/ 1 3)` is `1/3`, not `0.333…`).
                return self.make_ratio(xi, yi);
            }
        }
        let x = self.num_as_f64(a).expect("/: not a number");
        let y = self.num_as_f64(b).expect("/: not a number");
        self.encode(Val::Float(x / y))
    }

    fn int_div(&mut self, a: u64, b: u64, name: &str, op: fn(i128, i128) -> i128) -> u64 {
        let (Val::Int(x), Val::Int(y)) = (self.decode(a), self.decode(b)) else {
            if self.as_huge(a).is_some() || self.as_huge(b).is_some() {
                panic!("{name}: arbitrary-precision integers are unsupported");
            }
            panic!("{name}: both arguments must be integers");
        };
        if y == 0 {
            panic!("{name}: divide by zero");
        }
        let r = op(x, y);
        if M::R::is_immediate(Cat::Int) {
            if let Ok(r64) = i64::try_from(r) {
                if M::R::imm_fits(r64) {
                    return M::R::enc_int(r64);
                }
            }
        }
        let id = self.alloc(Obj::BigInt(r));
        M::R::enc_ref(id)
    }

    /// An integer operand as `i128`, or a clear error. For the bitwise ops,
    /// which operate on the (non-negative, bounded) indices/hashes/bitmaps the
    /// persistent structures compute.
    fn as_i128(&self, bits: u64) -> i128 {
        match self.decode(bits) {
            Val::Int(i) => i,
            _ => panic!("bitwise op: argument is not an integer"),
        }
    }

    /// The `String` behind a string value, or a clear error if it is not one.
    pub fn as_str(&self, bits: u64, who: &str) -> String {
        if let Val::Ref(id) = self.decode(bits) {
            if let ObjView::Str(s) = self.view_gc(id) {
                return s.to_string();
            }
        }
        panic!("{who}: argument is not a string");
    }

    /// `(num, den)` if `bits` is a Ratio, else `None`.
    fn as_ratio(&self, bits: u64) -> Option<(i128, i128)> {
        if let Val::Ref(id) = self.decode(bits) {
            if let ObjView::Ratio(n, d) = self.view_gc(id) {
                return Some((n, d));
            }
        }
        None
    }
    /// Build a value from `num/den`: reduce to lowest terms, force `den > 0`, and
    /// collapse a denominator of 1 back to a plain integer. Divide-by-zero panics.
    pub fn make_ratio(&mut self, num: i128, den: i128) -> u64 {
        if den == 0 {
            panic!("Divide by zero");
        }
        let (mut n, mut d) = (num, den);
        if d < 0 {
            n = -n;
            d = -d;
        }
        let g = {
            let (mut a, mut b) = (n.unsigned_abs(), d as u128);
            while b != 0 {
                let t = a % b;
                a = b;
                b = t;
            }
            a.max(1) as i128
        };
        n /= g;
        d /= g;
        if d == 1 {
            return self.alloc_bigint(BigInt::from_i128(n));
        }
        let id = self.alloc(Obj::Ratio(n, d));
        M::R::enc_ref(id)
    }
    /// `(num, den)` for any exact number: an integer `x` is `x/1`; a ratio is itself.
    /// `None` for floats / non-numbers.
    fn as_exact_ratio(&self, bits: u64) -> Option<(i128, i128)> {
        if let Some(r) = self.as_ratio(bits) {
            return Some(r);
        }
        if let Val::Int(i) = self.decode(bits) {
            return Some((i, 1));
        }
        // an i128-boxed integer counts too (huge ints don't fit and fall to float)
        self.as_int_big(bits).and_then(|b| b.to_i128()).map(|i| (i, 1))
    }
    /// Exact rational arithmetic for `+`/`-`/`*` when a Ratio is involved. `op`:
    /// 0=add, 1=sub, 2=mul. Falls back to f64 if a float operand is present or an
    /// i128 intermediate overflows.
    fn ratio_arith(&mut self, a: u64, b: u64, op: u8, fop: fn(f64, f64) -> f64) -> u64 {
        if let (Some((na, da)), Some((nb, db))) = (self.as_exact_ratio(a), self.as_exact_ratio(b)) {
            let r = match op {
                2 => na.checked_mul(nb).zip(da.checked_mul(db)),
                _ => da.checked_mul(db).and_then(|d| {
                    let l = na.checked_mul(db)?;
                    let rr = nb.checked_mul(da)?;
                    let n = if op == 0 { l.checked_add(rr) } else { l.checked_sub(rr) }?;
                    Some((n, d))
                }),
            };
            if let Some((n, d)) = r {
                return self.make_ratio(n, d);
            }
        }
        // float involved, or overflow: degrade to f64.
        let (x, y) = (self.num_as_f64(a), self.num_as_f64(b));
        match (x, y) {
            (Some(x), Some(y)) => self.encode(Val::Float(fop(x, y))),
            _ => panic!("arith on non-numbers"),
        }
    }
    // ── Stage F1: RAW accessors for the measured-hot collection internals ──
    // The profile said it plainly (docs/STAGE_F_COLLECTIONS.md): the
    // collection cluster's cost was per-node GLUE — every trie/PV node
    // touched constructed and dropped a full ObjView, plus `decode` for
    // category tests — not the algorithms and not copying. These few helpers
    // read the object layout DIRECTLY (one type_id check, then field/values
    // loads), mirroring the judgment the JIT's emitted fast paths already
    // apply. ObjView remains the general seam everywhere else; only the
    // measured-hot paths below (pv_*/hamt_*/equal/hash_value/seq_step) sit
    // on this raw layer. Soundness: every entry point still CHECKS the
    // header type_id and panics loudly on a mismatch or a poisoned header —
    // exactly where the view would have — and these prims run between
    // safepoints, so no collection can interleave (the existing contract).

    /// The raw object address of `bits`, if it is a reference at all.
    #[inline(always)]
    pub(crate) fn raw_gc(&self, bits: u64) -> Option<Gc> {
        <M::R as crate::heap::PtrPolicy>::try_decode_ptr(bits).map(Gc)
    }
    /// A reference's header type id, with the loud poison/corruption check
    /// `view_gc` would have performed.
    #[inline(always)]
    fn raw_type_id(&self, g: Gc) -> u16 {
        let t = unsafe { g.type_id() };
        if t == crate::heap::kind::INVALID || t as usize >= crate::heap::kind::COUNT {
            panic!(
                "raw read: object at {:p} has type_id {t} — poisoned/stale pointer or heap \
                 corruption (use-after-move?)",
                g.0
            );
        }
        t
    }
    /// A record's `(type sym, fields)` read raw; `None` for any other object.
    /// The 'static slice borrows the heap — valid between safepoints, same
    /// discipline as every raw value word (and as `ObjView`'s slices).
    #[inline(always)]
    fn raw_rec(&self, bits: u64) -> Option<(Sym, &'static [u64])> {
        let g = self.raw_gc(bits)?;
        if self.raw_type_id(g) != crate::heap::kind::RECORD {
            return None;
        }
        let info = &self.shared.types[crate::heap::kind::RECORD as usize];
        unsafe { Some((g.raw_word(info, 0) as Sym, g.values(info))) }
    }
    /// An integer field's value without constructing `Val`: immediate fixnum
    /// or promoted BigInt (the only shapes an int-valued record field takes).
    #[inline(always)]
    pub(crate) fn raw_i64(&self, bits: u64, what: &str) -> i64 {
        match M::R::tag_of(bits) {
            RawTag::Int => M::R::imm_int(bits),
            RawTag::Ref => {
                let g = M::R::as_ref(bits);
                if self.raw_type_id(g) == crate::heap::kind::BIGINT {
                    let info = &self.shared.types[crate::heap::kind::BIGINT as usize];
                    unsafe { g.raw_i128(info, 0) as i64 }
                } else {
                    panic!("{what}: expected an int")
                }
            }
            _ => panic!("{what}: expected an int"),
        }
    }

    // ── native persistent vector ──────────────────────────────────────────
    // `'PersistentVector [meta cnt shift root tail __hash]`; trie nodes are
    // `'VectorNode [edit arr]` wrapping 32-wide `Obj::Vector` arrays; `tail` is a
    // bare array. Ported byte-for-byte from cljs_types.clj so results are
    // identical. GC-safe WITHOUT rooting: `alloc` never relocates existing objects
    // and GC runs only at explicit safepoints (never mid-prim), so the raw `u64`
    // refs held across the allocations below stay valid. One native call replaces
    // the ~30 interpreted helper calls the deftype methods compiled to.

    /// (meta, cnt, shift, root, tail) of a `'PersistentVector`.
    fn pv_read(&self, pv: u64) -> (u64, i64, i64, u64, u64) {
        let Some((_, f)) = self.raw_rec(pv) else { panic!("pvec: not a record") };
        (f[0], self.raw_i64(f[1], "pvec cnt"), self.raw_i64(f[2], "pvec shift"), f[3], f[4])
    }
    /// The array inside a `'VectorNode` (field 1).
    fn node_arr(&self, node: u64) -> u64 {
        let Some((_, f)) = self.raw_rec(node) else { panic!("vnode: not a record") };
        f[1]
    }
    /// The ARRAY handle backing an array value (identity for mutation/growth).
    fn arr_handle(&self, arr: u64) -> Gc {
        let Some(g) = self.raw_gc(arr) else { panic!("pvec: array not a ref") };
        assert_eq!(self.raw_type_id(g), crate::heap::kind::ARRAY, "pvec: not an array");
        g
    }
    fn arr_clone(&self, arr: u64) -> Vec<u64> {
        self.arr_slice(self.arr_handle(arr)).to_vec()
    }
    fn arr_at(&self, arr: u64, i: usize) -> u64 {
        self.arr_slice(self.arr_handle(arr))[i]
    }
    pub(crate) fn arr_at_pub(&self, arr: u64, i: usize) -> u64 {
        self.arr_at(arr, i)
    }
    /// The live elements of an array value (replaces the old off/len "span"
    /// pair now that arrays are addressed by handle, not a word-arena range).
    pub(crate) fn arr_elems_pub(&self, arr: u64) -> &[u64] {
        self.arr_slice(self.arr_handle(arr))
    }
    fn mk_array(&mut self, v: Vec<u64>) -> u64 {
        M::R::enc_ref(self.alloc_vector(&v))
    }
    /// A fresh array = `arr` ++ [x]. The tail-conj hot path — one heap→heap
    /// copy, no Vec (allocation never collects, so the source borrow is live
    /// across the alloc).
    fn mk_array_append1(&mut self, arr: u64, x: u64) -> u64 {
        let src = self.arr_slice(self.arr_handle(arr));
        M::R::enc_ref(self.alloc_vector_append1(src, x))
    }
    /// A fresh array = `arr` with index `i` replaced by `x`. The trie
    /// path-copy hot path (same fusion as `mk_array_append1`).
    fn mk_array_set1(&mut self, arr: u64, i: usize, x: u64) -> u64 {
        let src = self.arr_slice(self.arr_handle(arr));
        M::R::enc_ref(self.alloc_vector_set1(src, i, x))
    }
    /// A fresh one-element array.
    fn mk_array1(&mut self, x: u64) -> u64 {
        M::R::enc_ref(self.alloc_vector(&[x]))
    }
    /// (arr, off, end, more) of a `'ChunkedCons` record, or None. Lets the cons
    /// prims (`%first`/`%rest`) treat a chunk transparently as a seq — only on the
    /// as_cons-miss path, so the common cons case pays nothing.
    /// HAMT node kind by interned-sym compare (no string allocation/compare).
    fn hamt_kind(&self, type_id: Sym) -> HamtKind {

        if type_id == self.intern_cached(&self.shared.sym_cache_bitmap_node, "BitmapIndexedNode") {
            HamtKind::Bitmap
        } else if type_id == self.intern_cached(&self.shared.sym_cache_array_node, "ArrayNode") {
            HamtKind::Array
        } else if type_id
            == self.intern_cached(&self.shared.sym_cache_collision_node, "HashCollisionNode")
        {
            HamtKind::Collision
        } else {
            HamtKind::Other
        }
    }

    pub(crate) fn as_chunked(&self, bits: u64) -> Option<(u64, i64, i64, u64)> {
        let (type_id, fields) = self.raw_rec(bits)?;
        // ONE interned-sym compare (this runs per seq step — a string compare
        // here dominated chunk iteration).
        let cc = self.intern_cached(&self.shared.sym_cache_chunked_cons, "ChunkedCons");
        if type_id != cc {
            return None;
        }
        Some((
            fields[0],
            self.raw_i64(fields[1], "chunked off"),
            self.raw_i64(fields[2], "chunked end"),
            fields[3],
        ))
    }
    pub(crate) fn mk_chunked(&mut self, arr: u64, off: i64, end: i64, more: u64) -> u64 {
        let offb = self.encode(Val::Int(off as i128));
        let endb = self.encode(Val::Int(end as i128));
        let ty = self.intern_cached(&self.shared.sym_cache_chunked_cons, "ChunkedCons");
        M::R::enc_ref(self.alloc_record(ty, &[arr, offb, endb, more]))
    }
    fn mk_node(&mut self, arr: u64) -> u64 {
        let nil = self.enc_nil();
        let ty = self.intern_cached(&self.shared.sym_cache_vector_node, "VectorNode");
        M::R::enc_ref(self.alloc_record(ty, &[nil, arr]))
    }
    fn mk_pv(&mut self, meta: u64, cnt: i64, shift: i64, root: u64, tail: u64) -> u64 {
        let cntb = self.encode(Val::Int(cnt as i128));
        let shiftb = self.encode(Val::Int(shift as i128));
        let nil = self.enc_nil();
        let ty = self.intern_cached(&self.shared.sym_cache_persistent_vector, "PersistentVector");
        M::R::enc_ref(self.alloc_record(ty, &[meta, cntb, shiftb, root, tail, nil]))
    }
    #[inline]
    fn tail_off(cnt: i64) -> i64 {
        if cnt < 32 { 0 } else { ((cnt - 1) >> 5) << 5 }
    }
    /// Wrap `node` under `level/5` single-child `'VectorNode` levels.
    fn pv_new_path(&mut self, level: i64, node: u64) -> u64 {
        let mut ret = node;
        let mut ll = level;
        while ll != 0 {
            let nil = self.enc_nil();
            let mut a = vec![nil; 32];
            a[0] = ret;
            let arr = self.mk_array(a);
            ret = self.mk_node(arr);
            ll -= 5;
        }
        ret
    }
    /// Path-copy from `parent`, inserting `tailnode` at the leaf slot for `cnt-1`.
    fn pv_push_tail(&mut self, cnt: i64, level: i64, parent: u64, tailnode: u64) -> u64 {
        let subidx = (((cnt - 1) >> level) & 31) as usize;
        let parr = self.node_arr(parent);
        let slot = if level == 5 {
            tailnode
        } else {
            let child = self.arr_at(parr, subidx);
            if matches!(self.decode(child), Val::Nil) {
                self.pv_new_path(level - 5, tailnode)
            } else {
                self.pv_push_tail(cnt, level - 5, child, tailnode)
            }
        };
        let arr = self.mk_array_set1(parr, subidx, slot);
        self.mk_node(arr)
    }
    pub(crate) fn pv_conj(&mut self, pv: u64, o: u64) -> u64 {
        let (meta, cnt, shift, root, tail) = self.pv_read(pv);
        // Common case: room in the tail — copy the tail array and append,
        // span-to-span (no heap-allocating temporary).
        if cnt - Self::tail_off(cnt) < 32 {
            let new_tail = self.mk_array_append1(tail, o);
            return self.mk_pv(meta, cnt + 1, shift, root, new_tail);
        }
        // Tail full: wrap it as a trie node and push into the tree.
        let tail_node = self.mk_node(tail);
        let root_overflow = (cnt >> 5) > (1i64 << shift);
        let (new_shift, new_root) = if root_overflow {
            let path = self.pv_new_path(shift, tail_node);
            let nil = self.enc_nil();
            let mut a = vec![nil; 32];
            a[0] = root;
            a[1] = path;
            let arr = self.mk_array(a);
            let nr = self.mk_node(arr);
            (shift + 5, nr)
        } else {
            (shift, self.pv_push_tail(cnt, shift, root, tail_node))
        };
        let new_tail = self.mk_array1(o);
        self.mk_pv(meta, cnt + 1, new_shift, new_root, new_tail)
    }
    /// Build a PersistentVector from a flat element array, bottom-up: chunk the
    /// non-tail elements into 32-wide leaves, then group nodes 32-at-a-time up to
    /// a single root. O(n) with no per-element tail clone. `shift` tracks the
    /// depth so nth/seq/pop stay consistent (the exact tree need not match what
    /// incremental conj builds — both are valid tries over the same elements).
    fn pv_from_array(&mut self, arr_bits: u64) -> u64 {
        let elems = self.arr_clone(arr_bits);
        let n = elems.len() as i64;
        let nil = self.enc_nil();
        let tailoff = Self::tail_off(n);
        let tail_vec: Vec<u64> = elems[tailoff as usize..].to_vec();
        let tail = self.mk_array(tail_vec);
        if tailoff == 0 {
            // All elements (0..32) live in the tail; root is an empty leaf level.
            let en = self.mk_array(vec![nil; 32]);
            let root = self.mk_node(en);
            return self.mk_pv(nil, n, 5, root, tail);
        }
        // Leaf nodes: each wraps an exactly-32 run of elements.
        let mut level: Vec<u64> = Vec::new();
        let mut i = 0i64;
        while i < tailoff {
            let leaf_vec: Vec<u64> = elems[i as usize..(i + 32) as usize].to_vec();
            let la = self.mk_array(leaf_vec);
            level.push(self.mk_node(la));
            i += 32;
        }
        // Group nodes into 32-wide (nil-padded) parents until one root remains.
        let mut shift = 5i64;
        loop {
            let mut parents: Vec<u64> = Vec::new();
            let mut j = 0usize;
            while j < level.len() {
                let end = (j + 32).min(level.len());
                let mut pa = vec![nil; 32];
                for (k, idx) in (j..end).enumerate() {
                    pa[k] = level[idx];
                }
                let paa = self.mk_array(pa);
                parents.push(self.mk_node(paa));
                j += 32;
            }
            if parents.len() == 1 {
                return self.mk_pv(nil, n, shift, parents[0], tail);
            }
            level = parents;
            shift += 5;
        }
    }
    pub(crate) fn pv_nth(&mut self, pv: u64, i: i64) -> u64 {
        let (_, cnt, shift, root, tail) = self.pv_read(pv);
        self.pv_nth_inner(cnt, shift, root, tail, i)
    }
    /// The nth walk over (cnt, shift, root, tail) — shared by the persistent
    /// vector and the transient (whose record layout differs).
    fn pv_nth_inner(&mut self, cnt: i64, shift: i64, root: u64, tail: u64, i: i64) -> u64 {
        if i < 0 || i >= cnt {
            let sid = self.alloc(Obj::Str(format!("No item {i} in vector of length {cnt}")));
            self.signal_throw(M::R::enc_ref(sid));
            return self.enc_nil();
        }
        let arr = if i >= Self::tail_off(cnt) {
            tail
        } else {
            let mut node = root;
            let mut level = shift;
            while level > 0 {
                let na = self.node_arr(node);
                node = self.arr_at(na, ((i >> level) & 31) as usize);
                level -= 5;
            }
            self.node_arr(node)
        };
        self.arr_at(arr, (i & 31) as usize)
    }
    /// Path-copy a node, setting index `i`'s leaf to `val`.
    fn pv_do_assoc(&mut self, level: i64, node: u64, i: i64, val: u64) -> u64 {
        let narr = self.node_arr(node);
        let (subidx, slot) = if level == 0 {
            ((i & 31) as usize, val)
        } else {
            let subidx = ((i >> level) & 31) as usize;
            let child = self.arr_at(narr, subidx);
            (subidx, self.pv_do_assoc(level - 5, child, i, val))
        };
        let arr = self.mk_array_set1(narr, subidx, slot);
        self.mk_node(arr)
    }
    pub(crate) fn pv_assoc(&mut self, pv: u64, n: i64, val: u64) -> u64 {
        let (meta, cnt, shift, root, tail) = self.pv_read(pv);
        if n >= 0 && n < cnt {
            if Self::tail_off(cnt) <= n {
                let new_tail = self.mk_array_set1(tail, (n & 31) as usize, val);
                self.mk_pv(meta, cnt, shift, root, new_tail)
            } else {
                let nr = self.pv_do_assoc(shift, root, n, val);
                self.mk_pv(meta, cnt, shift, nr, tail)
            }
        } else if n == cnt {
            self.pv_conj(pv, val)
        } else {
            let sid = self.alloc(Obj::Str(format!("Index {n} out of bounds  [0,{cnt}]")));
            self.signal_throw(M::R::enc_ref(sid));
            self.enc_nil()
        }
    }

    // ── native HAMT (PersistentHashMap's BitmapIndexedNode / ArrayNode /
    // HashCollisionNode trie) ──────────────────────────────────────────────
    // A faithful port of the `-inode-assoc`/`-inode-without`/`-inode-lookup`
    // algorithm in cljs_types.clj (itself ported from cljs/core.cljs) — same
    // node shapes, same bit tricks, same promotion/packing thresholds — just
    // run natively instead of walking the trie through interpreted protocol
    // dispatch + per-level allocation via `aset`/`aclone` calls. One native
    // call replaces the whole recursive interpreted descent.
    //
    // Key equality: `self.equal` (== the `%num-eq` prim == clojure.core's
    // `-eq2` default `Object` impl — see `extend-type Object IEquiv (-equiv
    // [a b] (%num-eq a b))` in core.clj). This matches for every ordinary key
    // (numbers, strings, keywords/syms, chars, nested vectors of these). It
    // does NOT replicate `-equiv`'s CROSS-TYPE sequential override (a vector
    // key is never `equal` to an `=`-equal list here) or a custom deftype's
    // hand-written `-equiv` — both are exceedingly rare as hash-map keys, and
    // this is a documented, honest limitation, not a silent wrong answer for
    // the common case.
    fn hash_masked(&self, k: u64) -> u32 {
        self.hash_value(k) & 0x7fff_ffff
    }
    fn is_nil_bits(&self, b: u64) -> bool {
        // nil has exactly ONE encoding under every model: a bit compare.
        b == M::R::enc_nil()
    }
    fn mk_bitmap_node(&mut self, bitmap: u32, arr: Vec<u64>) -> u64 {
        let nil = self.enc_nil();
        let bm = self.encode(Val::Int(bitmap as i128));
        let arrb = self.mk_array(arr);
        let ty = self.intern_cached(&self.shared.sym_cache_bitmap_node, "BitmapIndexedNode");
        M::R::enc_ref(self.alloc_record(ty, &[nil, bm, arrb]))
    }
    fn mk_array_node(&mut self, cnt: i64, arr: Vec<u64>) -> u64 {
        let nil = self.enc_nil();
        let cb = self.encode(Val::Int(cnt as i128));
        let arrb = self.mk_array(arr);
        let ty = self.intern_cached(&self.shared.sym_cache_array_node, "ArrayNode");
        M::R::enc_ref(self.alloc_record(ty, &[nil, cb, arrb]))
    }
    fn mk_collision_node(&mut self, chash: u32, cnt: i64, arr: Vec<u64>) -> u64 {
        let nil = self.enc_nil();
        let hb = self.encode(Val::Int(chash as i128));
        let cb = self.encode(Val::Int(cnt as i128));
        let arrb = self.mk_array(arr);
        let ty = self.intern_cached(&self.shared.sym_cache_collision_node, "HashCollisionNode");
        M::R::enc_ref(self.alloc_record(ty, &[nil, hb, cb, arrb]))
    }
    /// A brand-new single-entry `BitmapIndexedNode` — what
    /// `(-inode-assoc -EMPTY-BIN shift hash key val _)` always produces (an
    /// empty bitmap node's assoc always takes the "new bit, n=0 < 16" path).
    fn hamt_single(&mut self, hash: u32, shift: u32, key: u64, val: u64) -> u64 {
        let bit = 1u32 << ((hash >> shift) & 31);
        self.mk_bitmap_node(bit, vec![key, val])
    }
    /// `create-node`: build the 2-entry subtree resolving a collision between
    /// an existing leaf (`key1`/`val1`, whose hash we must compute) and a new
    /// leaf (`key2`/`val2`, hash already known).
    fn hamt_create_node(&mut self, shift: u32, key1: u64, val1: u64, key2hash: u32, key2: u64, val2: u64) -> u64 {
        let key1hash = self.hash_masked(key1);
        if key1hash == key2hash {
            self.mk_collision_node(key1hash, 2, vec![key1, val1, key2, val2])
        } else {
            let n1 = self.hamt_single(key1hash, shift, key1, val1);
            self.hamt_assoc(n1, shift, key2hash, key2, val2).0
        }
    }
    /// `(-inode-assoc node shift hash key val)` -> `(new_node, added_leaf?)`.
    /// `added_leaf?` is true iff `key` was NOT already present (so the caller
    /// should bump `cnt`); when nothing changed at all, `new_node == node`
    /// (raw bit identity — matches the original's `identical?` short-circuit).
    fn hamt_assoc(&mut self, node: u64, shift: u32, hash: u32, key: u64, val: u64) -> (u64, bool) {
        // RAW node reads (Stage F1): the trie descent touches a record header
        // + two int fields per level — the ObjView/decode glue here was the
        // top of the assoc-build profile. The `fields` slice stays valid
        // across the allocations below (alloc never moves objects; no
        // safepoints mid-prim — the same contract the old `to_vec` relied on,
        // minus the copy).
        let Some((type_id, fields)) = self.raw_rec(node) else { panic!("hamt-assoc: not a node") };
        let kind = self.hamt_kind(type_id);
        match kind {
            HamtKind::Bitmap => {
                let bitmap = self.raw_i64(fields[1], "hamt bitmap") as u32;
                let arr = self.arr_clone(fields[2]);
                let bit = 1u32 << ((hash >> shift) & 31);
                let idx = (bitmap & bit.wrapping_sub(1)).count_ones() as usize;
                if bitmap & bit == 0 {
                    let n = bitmap.count_ones();
                    if n >= 16 {
                        let nil = self.enc_nil();
                        let mut nodes = vec![nil; 32];
                        let jdx = ((hash >> shift) & 31) as usize;
                        nodes[jdx] = self.hamt_single(hash, shift + 5, key, val);
                        let mut j = 0usize;
                        for i in 0..32u32 {
                            if (bitmap >> i) & 1 == 0 {
                                continue;
                            }
                            let key_or_nil = arr[j];
                            let val_or_node = arr[j + 1];
                            nodes[i as usize] = if !self.is_nil_bits(key_or_nil) {
                                let h2 = self.hash_masked(key_or_nil);
                                self.hamt_single(h2, shift + 5, key_or_nil, val_or_node)
                            } else {
                                val_or_node
                            };
                            j += 2;
                        }
                        (self.mk_array_node((n + 1) as i64, nodes), true)
                    } else {
                        let mut new_arr = Vec::with_capacity(arr.len() + 2);
                        new_arr.extend_from_slice(&arr[0..2 * idx]);
                        new_arr.push(key);
                        new_arr.push(val);
                        new_arr.extend_from_slice(&arr[2 * idx..]);
                        (self.mk_bitmap_node(bitmap | bit, new_arr), true)
                    }
                } else {
                    let key_or_nil = arr[2 * idx];
                    let val_or_node = arr[2 * idx + 1];
                    if self.is_nil_bits(key_or_nil) {
                        let (n, added) = self.hamt_assoc(val_or_node, shift + 5, hash, key, val);
                        if n == val_or_node {
                            (node, false)
                        } else {
                            let mut na = arr.clone();
                            na[2 * idx + 1] = n;
                            (self.mk_bitmap_node(bitmap, na), added)
                        }
                    } else if self.equal(key, key_or_nil) {
                        if val == val_or_node {
                            (node, false)
                        } else {
                            let mut na = arr.clone();
                            na[2 * idx + 1] = val;
                            (self.mk_bitmap_node(bitmap, na), false)
                        }
                    } else {
                        let sub = self.hamt_create_node(shift + 5, key_or_nil, val_or_node, hash, key, val);
                        let mut na = arr.clone();
                        na[2 * idx] = self.enc_nil();
                        na[2 * idx + 1] = sub;
                        (self.mk_bitmap_node(bitmap, na), true)
                    }
                }
            }
            HamtKind::Array => {
                let cnt = self.raw_i64(fields[1], "hamt cnt");
                let arr = self.arr_clone(fields[2]);
                let idx = ((hash >> shift) & 31) as usize;
                let child = arr[idx];
                if self.is_nil_bits(child) {
                    let sub = self.hamt_single(hash, shift + 5, key, val);
                    let mut na = arr.clone();
                    na[idx] = sub;
                    (self.mk_array_node(cnt + 1, na), true)
                } else {
                    let (n, added) = self.hamt_assoc(child, shift + 5, hash, key, val);
                    if n == child {
                        (node, false)
                    } else {
                        let mut na = arr.clone();
                        na[idx] = n;
                        (self.mk_array_node(cnt, na), added)
                    }
                }
            }
            HamtKind::Collision => {
                let chash = self.raw_i64(fields[1], "hamt chash") as u32;
                let cnt = self.raw_i64(fields[2], "hamt cnt");
                let arr = self.arr_clone(fields[3]);
                if hash == chash {
                    let mut found = None;
                    let mut i = 0usize;
                    while i < arr.len() {
                        if self.equal(key, arr[i]) {
                            found = Some(i);
                            break;
                        }
                        i += 2;
                    }
                    match found {
                        None => {
                            let mut na = arr.clone();
                            na.push(key);
                            na.push(val);
                            (self.mk_collision_node(chash, cnt + 1, na), true)
                        }
                        Some(i) => {
                            if self.equal(arr[i + 1], val) {
                                (node, false)
                            } else {
                                let mut na = arr.clone();
                                na[i + 1] = val;
                                (self.mk_collision_node(chash, cnt, na), false)
                            }
                        }
                    }
                } else {
                    let wrapper = self.mk_bitmap_node(1u32 << ((chash >> shift) & 31), vec![self.enc_nil(), node]);
                    self.hamt_assoc(wrapper, shift, hash, key, val)
                }
            }
            HamtKind::Other => panic!("hamt-assoc: not a HAMT node"),
        }
    }
    /// `(-inode-lookup node shift hash key not-found)`.
    fn hamt_lookup(&self, node: u64, shift: u32, hash: u32, key: u64, not_found: u64) -> u64 {
        // RAW node reads (Stage F1) — see `hamt_assoc`.
        let Some((type_id, fields)) = self.raw_rec(node) else { return not_found };
        match self.hamt_kind(type_id) {
            HamtKind::Bitmap => {
                let bitmap = self.raw_i64(fields[1], "hamt bitmap") as u32;
                let bit = 1u32 << ((hash >> shift) & 31);
                if bitmap & bit == 0 {
                    return not_found;
                }
                let idx = (bitmap & bit.wrapping_sub(1)).count_ones() as usize;
                let key_or_nil = self.arr_at(fields[2], 2 * idx);
                let val_or_node = self.arr_at(fields[2], 2 * idx + 1);
                if self.is_nil_bits(key_or_nil) {
                    self.hamt_lookup(val_or_node, shift + 5, hash, key, not_found)
                } else if self.equal(key, key_or_nil) {
                    val_or_node
                } else {
                    not_found
                }
            }
            HamtKind::Array => {
                let idx = ((hash >> shift) & 31) as usize;
                let child = self.arr_at(fields[2], idx);
                if self.is_nil_bits(child) {
                    not_found
                } else {
                    self.hamt_lookup(child, shift + 5, hash, key, not_found)
                }
            }
            HamtKind::Collision => {
                let cnt = self.raw_i64(fields[2], "hamt cnt") as usize;
                // No clone: lookup only READS (and `equal` allocates nothing).
                let arr = self.arr_slice(self.arr_handle(fields[3]));
                let mut i = 0usize;
                while i < 2 * cnt {
                    if self.equal(key, arr[i]) {
                        return arr[i + 1];
                    }
                    i += 2;
                }
                not_found
            }
            _ => not_found,
        }
    }
    fn hamt_pack_array_node(&mut self, arr: &[u64], cnt: i64, skip_idx: usize) -> u64 {
        let mut new_arr = vec![self.enc_nil(); 2 * (cnt as usize - 1)];
        let mut j = 1usize;
        let mut bitmap = 0u32;
        for (i, &child) in arr.iter().enumerate() {
            if i == skip_idx || self.is_nil_bits(child) {
                continue;
            }
            new_arr[j] = child;
            j += 2;
            bitmap |= 1u32 << i;
        }
        self.mk_bitmap_node(bitmap, new_arr)
    }
    /// `(-inode-without node shift hash key)` -> the new node, `node` itself
    /// (raw bit identity) if nothing changed, or `nil` if `node` becomes empty.
    fn hamt_without(&mut self, node: u64, shift: u32, hash: u32, key: u64) -> u64 {
        // RAW node reads (Stage F1) — see `hamt_assoc`.
        let Some((type_id, fields)) = self.raw_rec(node) else { return node };
        match self.hamt_kind(type_id) {
            HamtKind::Bitmap => {
                let bitmap = self.raw_i64(fields[1], "hamt bitmap") as u32;
                let bit = 1u32 << ((hash >> shift) & 31);
                if bitmap & bit == 0 {
                    return node;
                }
                let idx = (bitmap & bit.wrapping_sub(1)).count_ones() as usize;
                let arr = self.arr_clone(fields[2]);
                let key_or_nil = arr[2 * idx];
                let val_or_node = arr[2 * idx + 1];
                if self.is_nil_bits(key_or_nil) {
                    let n = self.hamt_without(val_or_node, shift + 5, hash, key);
                    if n == val_or_node {
                        node
                    } else if !self.is_nil_bits(n) {
                        let mut na = arr.clone();
                        na[2 * idx + 1] = n;
                        self.mk_bitmap_node(bitmap, na)
                    } else if bitmap == bit {
                        self.enc_nil()
                    } else {
                        let na = self.hamt_remove_pair(&arr, idx);
                        self.mk_bitmap_node(bitmap ^ bit, na)
                    }
                } else if self.equal(key, key_or_nil) {
                    if bitmap == bit {
                        self.enc_nil()
                    } else {
                        let na = self.hamt_remove_pair(&arr, idx);
                        self.mk_bitmap_node(bitmap ^ bit, na)
                    }
                } else {
                    node
                }
            }
            HamtKind::Array => {
                let cnt = self.raw_i64(fields[1], "hamt cnt");
                let idx = ((hash >> shift) & 31) as usize;
                let arr = self.arr_clone(fields[2]);
                let child = arr[idx];
                if self.is_nil_bits(child) {
                    return node;
                }
                let n = self.hamt_without(child, shift + 5, hash, key);
                if n == child {
                    node
                } else if self.is_nil_bits(n) {
                    if cnt <= 8 {
                        self.hamt_pack_array_node(&arr, cnt, idx)
                    } else {
                        let mut na = arr.clone();
                        na[idx] = n;
                        self.mk_array_node(cnt - 1, na)
                    }
                } else {
                    let mut na = arr.clone();
                    na[idx] = n;
                    self.mk_array_node(cnt, na)
                }
            }
            HamtKind::Collision => {
                let chash = self.raw_i64(fields[1], "hamt chash") as u32;
                let cnt = self.raw_i64(fields[2], "hamt cnt");
                let arr = self.arr_clone(fields[3]);
                let mut found = None;
                let mut i = 0usize;
                while i < arr.len() {
                    if self.equal(key, arr[i]) {
                        found = Some(i);
                        break;
                    }
                    i += 2;
                }
                match found {
                    None => node,
                    Some(_) if cnt == 1 => self.enc_nil(),
                    Some(i) => {
                        let na = self.hamt_remove_pair(&arr, i / 2);
                        self.mk_collision_node(chash, cnt - 1, na)
                    }
                }
            }
            _ => node,
        }
    }
    fn hamt_remove_pair(&self, arr: &[u64], i: usize) -> Vec<u64> {
        let mut na = Vec::with_capacity(arr.len() - 2);
        na.extend_from_slice(&arr[0..2 * i]);
        na.extend_from_slice(&arr[2 * (i + 1)..]);
        na
    }
    /// Map-level entry points (`root` may be `nil`, the empty-map case the
    /// per-node functions above don't handle on their own): compute the key's
    /// hash once and start the trie descent at `shift = 0`.
    pub(crate) fn hamt_map_assoc(&mut self, root: u64, key: u64, val: u64) -> (u64, bool) {
        let hash = self.hash_masked(key);
        if self.is_nil_bits(root) {
            (self.hamt_single(hash, 0, key, val), true)
        } else {
            self.hamt_assoc(root, 0, hash, key, val)
        }
    }
    pub(crate) fn hamt_map_lookup(&self, root: u64, key: u64, not_found: u64) -> u64 {
        let hash = self.hash_masked(key);
        self.hamt_lookup(root, 0, hash, key, not_found)
    }
    fn hamt_map_without(&mut self, root: u64, key: u64) -> u64 {
        let hash = self.hash_masked(key);
        self.hamt_without(root, 0, hash, key)
    }

    // ── Stage F3: TRANSIENTS — ownership-stamped in-place builders ─────────
    //
    // An EDIT SESSION id (one fresh value per `transient`, from a per-runtime
    // counter that never repeats) marks ownership. Trie nodes carry it in
    // their existing `edit` slot (field 0 — the field the cljs port already
    // reserved and always left nil), NOT in the header's spare u16: a u16
    // stamp wraps after 65k transients (a real-program killer — group-by
    // creates one per call), the node layouts already have the slot, and a
    // traced value field is copied by the collector like any other value, so
    // GC neutrality is free (a collection mid-transient rewrites the session
    // box and every stamp consistently — under NaN-boxing the session is a
    // boxed int and all stamps share the ONE box, so bit-equality survives
    // relocation by construction).
    //
    // Rules (exactly Clojure's): a node is EDITABLE by a session iff its edit
    // field == the session bits; editable nodes are mutated in place (their
    // arrays are exclusively owned — every copy-on-first-touch clones the
    // array too); everything else is copied on first touch and the copy is
    // stamped. `persistent!` clears the TRANSIENT's own session field (O(1)
    // freeze) — stale stamps on nodes are inert because ids never repeat.
    // Transients are single-threaded by contract; use-after-persistent!
    // throws catchably with cljs' messages. Like every collection prim,
    // these run between safepoints (no collection can interleave).

    /// A fresh session id, ENCODED (fixnum, or one shared box under models
    /// that box ints — bit-identical across all its uses either way).
    fn fresh_session(&mut self) -> u64 {
        let n = self.shared.edit_sessions.fetch_add(1, Ordering::Relaxed);
        assert!(n < (1 << 59), "transient edit-session counter overflow (2^59 sessions?!)");
        self.encode(Val::Int(n as i128))
    }
    /// Is `node` (a trie node record) editable by `session`?
    fn node_editable(&self, node: u64, session: u64) -> bool {
        match self.raw_rec(node) {
            Some((_, f)) => f[0] == session,
            None => false,
        }
    }
    fn rec_set(&self, rec: u64, i: usize, v: u64) {
        let g = self.raw_gc(rec).expect("transient: not a record");
        self.values_mut(g)[i] = v;
    }
    fn arr_set(&self, arr: u64, i: usize, v: u64) {
        let h = self.arr_handle(arr);
        self.arr_slice_mut(h)[i] = v;
    }
    fn mk_node_edited(&mut self, session: u64, arr: u64) -> u64 {
        let ty = self.intern_cached(&self.shared.sym_cache_vector_node, "VectorNode");
        M::R::enc_ref(self.alloc_record(ty, &[session, arr]))
    }
    fn mk_bitmap_node_edited(&mut self, session: u64, bitmap: u32, arr: Vec<u64>) -> u64 {
        let bm = self.encode(Val::Int(bitmap as i128));
        let arrb = self.mk_array(arr);
        let ty = self.intern_cached(&self.shared.sym_cache_bitmap_node, "BitmapIndexedNode");
        M::R::enc_ref(self.alloc_record(ty, &[session, bm, arrb]))
    }
    fn mk_array_node_edited(&mut self, session: u64, cnt: i64, arr: Vec<u64>) -> u64 {
        let cb = self.encode(Val::Int(cnt as i128));
        let arrb = self.mk_array(arr);
        let ty = self.intern_cached(&self.shared.sym_cache_array_node, "ArrayNode");
        M::R::enc_ref(self.alloc_record(ty, &[session, cb, arrb]))
    }
    fn mk_collision_node_edited(&mut self, session: u64, chash: u32, cnt: i64, arr: Vec<u64>) -> u64 {
        let hb = self.encode(Val::Int(chash as i128));
        let cb = self.encode(Val::Int(cnt as i128));
        let arrb = self.mk_array(arr);
        let ty = self.intern_cached(&self.shared.sym_cache_collision_node, "HashCollisionNode");
        M::R::enc_ref(self.alloc_record(ty, &[session, hb, cb, arrb]))
    }
    /// Read a transient record's fields, checking type + liveness. Throws
    /// catchably (Clojure semantics) and returns None on a dead session.
    fn transient_fields(&mut self, t: u64, want: Sym, name: &str, op: &str) -> Option<&'static [u64]> {
        let Some((ty, f)) = self.raw_rec(t) else {
            let sid = self.alloc(Obj::Str(format!("{op}: not a {name}")));
            self.signal_throw(M::R::enc_ref(sid));
            return None;
        };
        if ty != want {
            let sid = self.alloc(Obj::Str(format!("{op}: not a {name}")));
            self.signal_throw(M::R::enc_ref(sid));
            return None;
        }
        if self.is_nil_bits(f[0]) {
            let sid = self.alloc(Obj::Str(format!("{op} after persistent!")));
            self.signal_throw(M::R::enc_ref(sid));
            return None;
        }
        Some(f)
    }

    // ── TransientVector [session cnt shift root tail meta] ──
    pub(crate) fn tv_new(&mut self, pv: u64) -> u64 {
        let (meta, cnt, shift, root, tail) = self.pv_read(pv);
        let session = self.fresh_session();
        // The tail is OWNED from birth: a copy at full 32 capacity, so conj!
        // appends in place (this is the "real 32-wide tail" — 31 of 32 conj!s
        // are one in-place array push).
        let telems = self.arr_slice(self.arr_handle(tail)).to_vec();
        let ntail = M::R::enc_ref(self.alloc_vector_cap(&telems, 32));
        let cntb = self.encode(Val::Int(cnt as i128));
        let shiftb = self.encode(Val::Int(shift as i128));
        let ty = self.intern_cached(&self.shared.sym_cache_transient_vector, "TransientVector");
        M::R::enc_ref(self.alloc_record(ty, &[session, cntb, shiftb, root, ntail, meta]))
    }
    pub(crate) fn tv_conj(&mut self, tv: u64, x: u64) -> u64 {
        let want = self.intern_cached(&self.shared.sym_cache_transient_vector, "TransientVector");
        let Some(f) = self.transient_fields(tv, want, "TransientVector", "conj!") else {
            return self.enc_nil();
        };
        let session = f[0];
        let cnt = self.raw_i64(f[1], "tv cnt");
        let shift = self.raw_i64(f[2], "tv shift");
        let (root, tail) = (f[3], f[4]);
        let th = self.arr_handle(tail);
        if (unsafe { th.aux() } as i64) < 32 {
            self.arr_extend(th, &[x]);
            let ncnt = self.encode(Val::Int((cnt + 1) as i128));
            self.rec_set(tv, 1, ncnt);
            return tv;
        }
        // Tail full: spill it into the trie as an EDITED node (the tail array
        // moves in wholesale — it was ours), start a fresh owned tail.
        let tail_node = self.mk_node_edited(session, tail);
        let root_overflow = (cnt >> 5) > (1i64 << shift);
        let (new_shift, new_root) = if root_overflow {
            let path = self.tv_new_path(session, shift, tail_node);
            let nil = self.enc_nil();
            let mut a = vec![nil; 32];
            a[0] = root;
            a[1] = path;
            let arr = self.mk_array(a);
            let nr = self.mk_node_edited(session, arr);
            (shift + 5, nr)
        } else {
            (shift, self.tv_push_tail(session, cnt, shift, root, tail_node))
        };
        let ntail = M::R::enc_ref(self.alloc_vector_cap(&[x], 32));
        let ncnt = self.encode(Val::Int((cnt + 1) as i128));
        let nshift = self.encode(Val::Int(new_shift as i128));
        let g = self.raw_gc(tv).expect("tv");
        let fs = self.values_mut(g);
        fs[1] = ncnt;
        fs[2] = nshift;
        fs[3] = new_root;
        fs[4] = ntail;
        tv
    }
    fn tv_new_path(&mut self, session: u64, level: i64, node: u64) -> u64 {
        let mut ret = node;
        let mut ll = level;
        while ll != 0 {
            let nil = self.enc_nil();
            let mut a = vec![nil; 32];
            a[0] = ret;
            let arr = self.mk_array(a);
            ret = self.mk_node_edited(session, arr);
            ll -= 5;
        }
        ret
    }
    /// `pv_push_tail`, transient: mutate session-owned nodes in place, copy
    /// (and stamp) on first touch.
    fn tv_push_tail(&mut self, session: u64, cnt: i64, level: i64, parent: u64, tailnode: u64) -> u64 {
        let subidx = (((cnt - 1) >> level) & 31) as usize;
        let editable = self.node_editable(parent, session);
        let parr = self.node_arr(parent);
        let newchild = if level == 5 {
            tailnode
        } else {
            let child = self.arr_at(parr, subidx);
            if self.is_nil_bits(child) {
                self.tv_new_path(session, level - 5, tailnode)
            } else {
                self.tv_push_tail(session, cnt, level - 5, child, tailnode)
            }
        };
        if editable {
            self.arr_set(parr, subidx, newchild);
            parent
        } else {
            let mut ret = self.arr_clone(parr);
            ret[subidx] = newchild;
            let arr = self.mk_array(ret);
            self.mk_node_edited(session, arr)
        }
    }
    pub(crate) fn tv_assoc(&mut self, tv: u64, n: i64, val: u64) -> u64 {
        let want = self.intern_cached(&self.shared.sym_cache_transient_vector, "TransientVector");
        let Some(f) = self.transient_fields(tv, want, "TransientVector", "assoc!") else {
            return self.enc_nil();
        };
        let session = f[0];
        let cnt = self.raw_i64(f[1], "tv cnt");
        let shift = self.raw_i64(f[2], "tv shift");
        let (root, tail) = (f[3], f[4]);
        if n >= 0 && n < cnt {
            if Self::tail_off(cnt) <= n {
                self.arr_set(tail, (n & 31) as usize, val);
            } else {
                let nr = self.tv_do_assoc(session, shift, root, n, val);
                if nr != root {
                    self.rec_set(tv, 3, nr);
                }
            }
            tv
        } else if n == cnt {
            self.tv_conj(tv, val)
        } else {
            let sid = self.alloc(Obj::Str(format!("Index {n} out of bounds  [0,{cnt}]")));
            self.signal_throw(M::R::enc_ref(sid));
            self.enc_nil()
        }
    }
    fn tv_do_assoc(&mut self, session: u64, level: i64, node: u64, n: i64, val: u64) -> u64 {
        let editable = self.node_editable(node, session);
        let arr = self.node_arr(node);
        let subidx = ((n >> level) & 31) as usize;
        let newchild = if level == 0 {
            val
        } else {
            let child = self.arr_at(arr, subidx);
            self.tv_do_assoc(session, level - 5, child, n, val)
        };
        if editable {
            self.arr_set(arr, subidx, newchild);
            node
        } else {
            let mut na = self.arr_clone(arr);
            na[subidx] = newchild;
            let a = self.mk_array(na);
            self.mk_node_edited(session, a)
        }
    }
    pub(crate) fn tv_nth(&mut self, tv: u64, n: i64) -> u64 {
        let Some((_, f)) = self.raw_rec(tv) else { panic!("tv-nth: not a transient vector") };
        let cnt = self.raw_i64(f[1], "tv cnt");
        let shift = self.raw_i64(f[2], "tv shift");
        self.pv_nth_inner(cnt, shift, f[3], f[4], n)
    }
    pub(crate) fn tv_pop(&mut self, tv: u64) -> u64 {
        let want = self.intern_cached(&self.shared.sym_cache_transient_vector, "TransientVector");
        let Some(f) = self.transient_fields(tv, want, "TransientVector", "pop!") else {
            return self.enc_nil();
        };
        let session = f[0];
        let cnt = self.raw_i64(f[1], "tv cnt");
        let shift = self.raw_i64(f[2], "tv shift");
        let (root, tail) = (f[3], f[4]);
        if cnt == 0 {
            let sid = self.alloc(Obj::Str("Can't pop empty vector".to_string()));
            self.signal_throw(M::R::enc_ref(sid));
            return self.enc_nil();
        }
        let th = self.arr_handle(tail);
        let tlen = unsafe { th.aux() } as i64;
        if cnt == 1 || tlen > 1 {
            // Shrink the owned tail in place (cnt == 1 empties it).
            unsafe { th.set_aux((tlen - 1) as u32) };
            let ncnt = self.encode(Val::Int((cnt - 1) as i128));
            self.rec_set(tv, 1, ncnt);
            return tv;
        }
        // The tail had one element: the previous leaf becomes the new OWNED tail
        // (copied at capacity 32) and the trie pops it.
        let leaf = {
            // The leaf array holding index cnt-2 (which is below tail_off here).
            let i = cnt - 2;
            let mut node = root;
            let mut level = shift;
            while level > 0 {
                let na = self.node_arr(node);
                node = self.arr_at(na, ((i >> level) & 31) as usize);
                level -= 5;
            }
            self.node_arr(node)
        };
        let lelems = self.arr_slice(self.arr_handle(leaf)).to_vec();
        let ntail = M::R::enc_ref(self.alloc_vector_cap(&lelems, 32));
        let nr = self.tv_pop_tail(session, cnt, shift, root);
        let (new_shift, new_root) = if self.is_nil_bits(nr) {
            let nil = self.enc_nil();
            let a = self.mk_array(vec![nil; 32]);
            (shift, self.mk_node_edited(session, a))
        } else if shift > 5 && self.is_nil_bits(self.arr_at(self.node_arr(nr), 1)) {
            (shift - 5, self.arr_at(self.node_arr(nr), 0))
        } else {
            (shift, nr)
        };
        let ncnt = self.encode(Val::Int((cnt - 1) as i128));
        let nshift = self.encode(Val::Int(new_shift as i128));
        let g = self.raw_gc(tv).expect("tv");
        let fs = self.values_mut(g);
        fs[1] = ncnt;
        fs[2] = nshift;
        fs[3] = new_root;
        fs[4] = ntail;
        tv
    }
    /// `pop_tail`, transient: editable-aware (mirror of the persistent
    /// `pop_tail` used by `-pop`).
    fn tv_pop_tail(&mut self, session: u64, cnt: i64, level: i64, node: u64) -> u64 {
        let subidx = (((cnt - 2) >> level) & 31) as usize;
        let nil = self.enc_nil();
        if level > 5 {
            let child = self.arr_at(self.node_arr(node), subidx);
            let nchild = self.tv_pop_tail(session, cnt, level - 5, child);
            if self.is_nil_bits(nchild) && subidx == 0 {
                nil
            } else if self.node_editable(node, session) {
                self.arr_set(self.node_arr(node), subidx, nchild);
                node
            } else {
                let mut na = self.arr_clone(self.node_arr(node));
                na[subidx] = nchild;
                let a = self.mk_array(na);
                self.mk_node_edited(session, a)
            }
        } else if subidx == 0 {
            nil
        } else if self.node_editable(node, session) {
            self.arr_set(self.node_arr(node), subidx, nil);
            node
        } else {
            let mut na = self.arr_clone(self.node_arr(node));
            na[subidx] = nil;
            let a = self.mk_array(na);
            self.mk_node_edited(session, a)
        }
    }
    pub(crate) fn tv_persistent(&mut self, tv: u64) -> u64 {
        let want = self.intern_cached(&self.shared.sym_cache_transient_vector, "TransientVector");
        let Some(f) = self.transient_fields(tv, want, "TransientVector", "persistent!") else {
            return self.enc_nil();
        };
        let cnt = self.raw_i64(f[1], "tv cnt");
        let shift = self.raw_i64(f[2], "tv shift");
        let (root, tail, meta) = (f[3], f[4], f[5]);
        let nil = self.enc_nil();
        self.rec_set(tv, 0, nil); // O(1) freeze: the session dies here
        self.mk_pv(meta, cnt, shift, root, tail)
    }

    // ── TransientArrayMap [session arr cnt meta] ──
    pub(crate) fn tam_new(&mut self, pam: u64) -> u64 {
        let Some((_, f)) = self.raw_rec(pam) else { panic!("tam-new: not a record") };
        // PAM layout: [meta cnt arr __hash]
        let (meta, cnt, arr) = (f[0], f[1], f[2]);
        let session = self.fresh_session();
        let elems = self.arr_slice(self.arr_handle(arr)).to_vec();
        let narr = M::R::enc_ref(self.alloc_vector_cap(&elems, 16));
        let ty = self.intern_cached(&self.shared.sym_cache_transient_array_map, "TransientArrayMap");
        M::R::enc_ref(self.alloc_record(ty, &[session, narr, cnt, meta]))
    }
    /// assoc! on an array-map transient. Past the 8-pair threshold this
    /// PROMOTES: the return value is a `'TransientHashMap` carrying the SAME
    /// session (cljs' TransientArrayMap behavior) — callers use the return
    /// value, per the transient contract.
    pub(crate) fn tam_assoc(&mut self, tam: u64, k: u64, v: u64) -> u64 {
        let want = self.intern_cached(&self.shared.sym_cache_transient_array_map, "TransientArrayMap");
        let Some(f) = self.transient_fields(tam, want, "TransientArrayMap", "assoc!") else {
            return self.enc_nil();
        };
        let session = f[0];
        let (arr, cnt, meta) = (f[1], self.raw_i64(f[2], "tam cnt"), f[3]);
        let h = self.arr_handle(arr);
        let n = unsafe { h.aux() } as usize;
        // Linear key scan (same comparator as the persistent PAM/HAMT).
        let mut i = 0usize;
        while i < n {
            let ki = self.arr_slice(h)[i];
            if self.equal(k, ki) {
                self.arr_slice_mut(h)[i + 1] = v;
                return tam;
            }
            i += 2;
        }
        if cnt < 8 {
            self.arr_extend(h, &[k, v]);
            let ncnt = self.encode(Val::Int((cnt + 1) as i128));
            self.rec_set(tam, 2, ncnt);
            return tam;
        }
        // Promote to a TransientHashMap under the same session.
        let pairs = self.arr_slice(h).to_vec();
        let nil = self.enc_nil();
        let mut root = nil;
        let mut count = 0i64;
        let mut j = 0usize;
        while j < pairs.len() {
            let (added, nr) = {
                let hash = self.hash_masked(pairs[j]);
                let r = self.hamt_assoc_t(session, root, 0, hash, pairs[j], pairs[j + 1]);
                (r.1, r.0)
            };
            root = nr;
            count += added as i64;
            j += 2;
        }
        let hash = self.hash_masked(k);
        let (nr, added) = self.hamt_assoc_t(session, root, 0, hash, k, v);
        root = nr;
        count += added as i64;
        // Invalidate the TAM (it must not be used again) and hand back a THM.
        self.rec_set(tam, 0, nil);
        let cntb = self.encode(Val::Int(count as i128));
        let fal = self.encode(Val::Bool(false));
        let ty = self.intern_cached(&self.shared.sym_cache_transient_hash_map, "TransientHashMap");
        M::R::enc_ref(self.alloc_record(ty, &[session, root, cntb, fal, nil, meta]))
    }
    pub(crate) fn tam_dissoc(&mut self, tam: u64, k: u64) -> u64 {
        let want = self.intern_cached(&self.shared.sym_cache_transient_array_map, "TransientArrayMap");
        let Some(f) = self.transient_fields(tam, want, "TransientArrayMap", "dissoc!") else {
            return self.enc_nil();
        };
        let (arr, cnt) = (f[1], self.raw_i64(f[2], "tam cnt"));
        let h = self.arr_handle(arr);
        let n = unsafe { h.aux() } as usize;
        let mut i = 0usize;
        while i < n {
            let ki = self.arr_slice(h)[i];
            if self.equal(k, ki) {
                // Order-preserving in-place removal: shift left by one pair.
                let ws = self.arr_slice_mut(h);
                ws.copy_within(i + 2.., i);
                unsafe { h.set_aux((n - 2) as u32) };
                let ncnt = self.encode(Val::Int((cnt - 1) as i128));
                self.rec_set(tam, 2, ncnt);
                return tam;
            }
            i += 2;
        }
        tam
    }
    pub(crate) fn tam_persistent(&mut self, tam: u64) -> u64 {
        let want = self.intern_cached(&self.shared.sym_cache_transient_array_map, "TransientArrayMap");
        let Some(f) = self.transient_fields(tam, want, "TransientArrayMap", "persistent!") else {
            return self.enc_nil();
        };
        let (arr, cnt, meta) = (f[1], f[2], f[3]);
        let nil = self.enc_nil();
        self.rec_set(tam, 0, nil);
        let ty = self.intern_cached(&self.shared.sym_cache_persistent_array_map, "PersistentArrayMap");
        M::R::enc_ref(self.alloc_record(ty, &[meta, cnt, arr, nil]))
    }

    // ── TransientHashMap [session root cnt has-nil? nil-val meta] ──
    pub(crate) fn thm_new(&mut self, phm: u64) -> u64 {
        let Some((_, f)) = self.raw_rec(phm) else { panic!("thm-new: not a record") };
        // PHM layout: [meta cnt root has-nil? nil-val __hash]
        let (meta, cnt, root, has_nil, nil_val) = (f[0], f[1], f[2], f[3], f[4]);
        let session = self.fresh_session();
        let ty = self.intern_cached(&self.shared.sym_cache_transient_hash_map, "TransientHashMap");
        M::R::enc_ref(self.alloc_record(ty, &[session, root, cnt, has_nil, nil_val, meta]))
    }
    pub(crate) fn thm_assoc(&mut self, thm: u64, k: u64, v: u64) -> u64 {
        let want = self.intern_cached(&self.shared.sym_cache_transient_hash_map, "TransientHashMap");
        let Some(f) = self.transient_fields(thm, want, "TransientHashMap", "assoc!") else {
            return self.enc_nil();
        };
        let session = f[0];
        let (root, cnt) = (f[1], self.raw_i64(f[2], "thm cnt"));
        if self.is_nil_bits(k) {
            let had = f[3] == self.encode(Val::Bool(true));
            self.rec_set(thm, 4, v);
            if !had {
                let tru = self.encode(Val::Bool(true));
                self.rec_set(thm, 3, tru);
                let ncnt = self.encode(Val::Int((cnt + 1) as i128));
                self.rec_set(thm, 2, ncnt);
            }
            return thm;
        }
        let hash = self.hash_masked(k);
        let (nroot, added) = self.hamt_assoc_t(session, root, 0, hash, k, v);
        if nroot != root {
            self.rec_set(thm, 1, nroot);
        }
        if added {
            let ncnt = self.encode(Val::Int((cnt + 1) as i128));
            self.rec_set(thm, 2, ncnt);
        }
        thm
    }
    pub(crate) fn thm_dissoc(&mut self, thm: u64, k: u64) -> u64 {
        let want = self.intern_cached(&self.shared.sym_cache_transient_hash_map, "TransientHashMap");
        let Some(f) = self.transient_fields(thm, want, "TransientHashMap", "dissoc!") else {
            return self.enc_nil();
        };
        let (root, cnt) = (f[1], self.raw_i64(f[2], "thm cnt"));
        if self.is_nil_bits(k) {
            if f[3] == self.encode(Val::Bool(true)) {
                let fal = self.encode(Val::Bool(false));
                let nil = self.enc_nil();
                self.rec_set(thm, 3, fal);
                self.rec_set(thm, 4, nil);
                let ncnt = self.encode(Val::Int((cnt - 1) as i128));
                self.rec_set(thm, 2, ncnt);
            }
            return thm;
        }
        // Persistent trie removal (copies; sound — the copies are unstamped
        // and get copy-on-touch semantics from later assoc!s). dissoc! is not
        // the transient hot path.
        let nroot = self.hamt_map_without(root, k);
        if nroot != root {
            self.rec_set(thm, 1, nroot);
            let ncnt = self.encode(Val::Int((cnt - 1) as i128));
            self.rec_set(thm, 2, ncnt);
        }
        thm
    }
    pub(crate) fn thm_persistent(&mut self, thm: u64) -> u64 {
        let want = self.intern_cached(&self.shared.sym_cache_transient_hash_map, "TransientHashMap");
        let Some(f) = self.transient_fields(thm, want, "TransientHashMap", "persistent!") else {
            return self.enc_nil();
        };
        let (root, cnt, has_nil, nil_val, meta) = (f[1], f[2], f[3], f[4], f[5]);
        let nil = self.enc_nil();
        self.rec_set(thm, 0, nil);
        let ty = self.intern_cached(&self.shared.sym_cache_persistent_hash_map, "PersistentHashMap");
        M::R::enc_ref(self.alloc_record(ty, &[meta, cnt, root, has_nil, nil_val, nil]))
    }

    /// `-inode-assoc!` — the transient trie descent: session-owned nodes are
    /// edited in place; others are copied-and-stamped on first touch. Same
    /// algorithm and thresholds as `hamt_assoc` otherwise.
    fn hamt_assoc_t(&mut self, session: u64, node: u64, shift: u32, hash: u32, key: u64, val: u64) -> (u64, bool) {
        if self.is_nil_bits(node) {
            let bit = 1u32 << ((hash >> shift) & 31);
            return (self.mk_bitmap_node_edited(session, bit, vec![key, val]), true);
        }
        let Some((type_id, fields)) = self.raw_rec(node) else { panic!("hamt-assoc!: not a node") };
        let editable = fields[0] == session;
        match self.hamt_kind(type_id) {
            HamtKind::Bitmap => {
                let bitmap = self.raw_i64(fields[1], "hamt bitmap") as u32;
                let arrb = fields[2];
                let bit = 1u32 << ((hash >> shift) & 31);
                let idx = (bitmap & bit.wrapping_sub(1)).count_ones() as usize;
                if bitmap & bit == 0 {
                    let n = bitmap.count_ones();
                    if n >= 16 {
                        // Promote to an (edited) ArrayNode — always fresh.
                        let arr = self.arr_clone(arrb);
                        let nil = self.enc_nil();
                        let mut nodes = vec![nil; 32];
                        let jdx = ((hash >> shift) & 31) as usize;
                        nodes[jdx] = {
                            let bit2 = 1u32 << ((hash >> (shift + 5)) & 31);
                            self.mk_bitmap_node_edited(session, bit2, vec![key, val])
                        };
                        let mut j = 0usize;
                        for i in 0..32u32 {
                            if (bitmap >> i) & 1 == 0 {
                                continue;
                            }
                            let key_or_nil = arr[j];
                            let val_or_node = arr[j + 1];
                            nodes[i as usize] = if !self.is_nil_bits(key_or_nil) {
                                let h2 = self.hash_masked(key_or_nil);
                                let (sub, _) = self.hamt_assoc_t(
                                    session,
                                    self.enc_nil(),
                                    shift + 5,
                                    h2,
                                    key_or_nil,
                                    val_or_node,
                                );
                                sub
                            } else {
                                val_or_node
                            };
                            j += 2;
                        }
                        (self.mk_array_node_edited(session, (n + 1) as i64, nodes), true)
                    } else if editable {
                        // In-place pair insertion: grow by 2, shift right.
                        let h = self.arr_handle(arrb);
                        let nil = self.enc_nil();
                        self.arr_extend(h, &[nil, nil]);
                        let ws = self.arr_slice_mut(h);
                        ws.copy_within(2 * idx..ws.len() - 2, 2 * idx + 2);
                        ws[2 * idx] = key;
                        ws[2 * idx + 1] = val;
                        let bm = self.encode(Val::Int((bitmap | bit) as i128));
                        self.rec_set(node, 1, bm);
                        (node, true)
                    } else {
                        let arr = self.arr_clone(arrb);
                        let mut new_arr = Vec::with_capacity(arr.len() + 2);
                        new_arr.extend_from_slice(&arr[0..2 * idx]);
                        new_arr.push(key);
                        new_arr.push(val);
                        new_arr.extend_from_slice(&arr[2 * idx..]);
                        (self.mk_bitmap_node_edited(session, bitmap | bit, new_arr), true)
                    }
                } else {
                    let key_or_nil = self.arr_at(arrb, 2 * idx);
                    let val_or_node = self.arr_at(arrb, 2 * idx + 1);
                    if self.is_nil_bits(key_or_nil) {
                        let (nsub, added) = self.hamt_assoc_t(session, val_or_node, shift + 5, hash, key, val);
                        if nsub == val_or_node {
                            (node, added)
                        } else if editable {
                            self.arr_set(arrb, 2 * idx + 1, nsub);
                            (node, added)
                        } else {
                            let mut na = self.arr_clone(arrb);
                            na[2 * idx + 1] = nsub;
                            (self.mk_bitmap_node_edited(session, bitmap, na), added)
                        }
                    } else if self.equal(key, key_or_nil) {
                        if val == val_or_node {
                            (node, false)
                        } else if editable {
                            self.arr_set(arrb, 2 * idx + 1, val);
                            (node, false)
                        } else {
                            let mut na = self.arr_clone(arrb);
                            na[2 * idx + 1] = val;
                            (self.mk_bitmap_node_edited(session, bitmap, na), false)
                        }
                    } else {
                        let sub = self.hamt_create_node_t(session, shift + 5, key_or_nil, val_or_node, hash, key, val);
                        let nil = self.enc_nil();
                        if editable {
                            self.arr_set(arrb, 2 * idx, nil);
                            self.arr_set(arrb, 2 * idx + 1, sub);
                            (node, true)
                        } else {
                            let mut na = self.arr_clone(arrb);
                            na[2 * idx] = nil;
                            na[2 * idx + 1] = sub;
                            (self.mk_bitmap_node_edited(session, bitmap, na), true)
                        }
                    }
                }
            }
            HamtKind::Array => {
                let cnt = self.raw_i64(fields[1], "hamt cnt");
                let arrb = fields[2];
                let idx = ((hash >> shift) & 31) as usize;
                let child = self.arr_at(arrb, idx);
                if self.is_nil_bits(child) {
                    let (sub, _) = self.hamt_assoc_t(session, self.enc_nil(), shift + 5, hash, key, val);
                    if editable {
                        self.arr_set(arrb, idx, sub);
                        let cb = self.encode(Val::Int((cnt + 1) as i128));
                        self.rec_set(node, 1, cb);
                        (node, true)
                    } else {
                        let mut na = self.arr_clone(arrb);
                        na[idx] = sub;
                        (self.mk_array_node_edited(session, cnt + 1, na), true)
                    }
                } else {
                    let (nsub, added) = self.hamt_assoc_t(session, child, shift + 5, hash, key, val);
                    if nsub == child {
                        (node, added)
                    } else if editable {
                        self.arr_set(arrb, idx, nsub);
                        (node, added)
                    } else {
                        let mut na = self.arr_clone(arrb);
                        na[idx] = nsub;
                        (self.mk_array_node_edited(session, cnt, na), added)
                    }
                }
            }
            HamtKind::Collision => {
                let chash = self.raw_i64(fields[1], "hamt chash") as u32;
                let cnt = self.raw_i64(fields[2], "hamt cnt");
                let arrb = fields[3];
                if hash == chash {
                    let n = unsafe { self.arr_handle(arrb).aux() } as usize;
                    let mut found = None;
                    let mut i = 0usize;
                    while i < n {
                        let ki = self.arr_at(arrb, i);
                        if self.equal(key, ki) {
                            found = Some(i);
                            break;
                        }
                        i += 2;
                    }
                    match found {
                        None => {
                            if editable {
                                let h = self.arr_handle(arrb);
                                self.arr_extend(h, &[key, val]);
                                let cb = self.encode(Val::Int((cnt + 1) as i128));
                                self.rec_set(node, 2, cb);
                                (node, true)
                            } else {
                                let mut na = self.arr_clone(arrb);
                                na.push(key);
                                na.push(val);
                                (self.mk_collision_node_edited(session, chash, cnt + 1, na), true)
                            }
                        }
                        Some(i) => {
                            if self.arr_at(arrb, i + 1) == val {
                                (node, false)
                            } else if editable {
                                self.arr_set(arrb, i + 1, val);
                                (node, false)
                            } else {
                                let mut na = self.arr_clone(arrb);
                                na[i + 1] = val;
                                (self.mk_collision_node_edited(session, chash, cnt, na), false)
                            }
                        }
                    }
                } else {
                    let nil = self.enc_nil();
                    let wrapper = self.mk_bitmap_node_edited(
                        session,
                        1u32 << ((chash >> shift) & 31),
                        vec![nil, node],
                    );
                    self.hamt_assoc_t(session, wrapper, shift, hash, key, val)
                }
            }
            HamtKind::Other => panic!("hamt-assoc!: not a HAMT node"),
        }
    }
    /// Transient `create-node`: both fresh subnodes are session-stamped.
    fn hamt_create_node_t(&mut self, session: u64, shift: u32, key1: u64, val1: u64, key2hash: u32, key2: u64, val2: u64) -> u64 {
        let key1hash = self.hash_masked(key1);
        if key1hash == key2hash {
            self.mk_collision_node_edited(session, key1hash, 2, vec![key1, val1, key2, val2])
        } else {
            let nil = self.enc_nil();
            let (n1, _) = self.hamt_assoc_t(session, nil, shift, key1hash, key1, val1);
            self.hamt_assoc_t(session, n1, shift, key2hash, key2, val2).0
        }
    }

    fn arith(
        &mut self,
        a: u64,
        b: u64,
        op: u8,
        iop64: fn(i64, i64) -> Option<i64>,
        iop128: fn(i128, i128) -> Option<i128>,
        fop: fn(f64, f64) -> f64,
        bigop: fn(&BigInt, &BigInt) -> BigInt,
    ) -> u64 {
        // A Ratio operand takes the exact-rational path.
        if self.as_ratio(a).is_some() || self.as_ratio(b).is_some() {
            return self.ratio_arith(a, b, op, fop);
        }
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
        self.as_huge(bits)
    }

    /// The `BigInt` behind a `HugeInt` heap value, if `bits` is one. Owned:
    /// the view reassembles it from the object's sign word + limb bytes on
    /// every call (a cold path — huge integers are the rare arbitrary-
    /// precision overflow case), so there is no live `&BigInt` to borrow.
    fn as_huge(&self, bits: u64) -> Option<BigInt> {
        if let RawTag::Ref = M::R::tag_of(bits) {
            if let ObjView::HugeInt(b) = self.view(bits) {
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
            _ => {
                if let Some((n, d)) = self.as_ratio(bits) {
                    return Some(n as f64 / d as f64);
                }
                self.as_huge(bits).map(|b| b.to_f64())
            }
        }
    }

    /// Store a `BigInt`, normalizing down to a fixnum / `i128` box when it fits so
    /// only genuinely-huge values carry the arbitrary-precision representation.
    pub fn alloc_bigint(&mut self, b: BigInt) -> u64 {
        if let Some(i) = b.to_i128() {
            return self.encode(Val::Int(i));
        }
        let id = self.alloc(Obj::HugeInt(b));
        M::R::enc_ref(id)
    }

    // ── printing ────────────────────────────────────────────
    /// The `str`-form of a value: a string yields its raw content (no quotes);
    /// everything else uses the neutral printer. Mirrors `StrOf`.
    pub fn str_form(&self, bits: u64) -> String {
        if let Val::Ref(id) = self.decode(bits) {
            if let ObjView::Str(s) = self.view_gc(id) {
                return s.to_string();
            }
        }
        self.print(bits)
    }
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
            Val::Ref(id) => match self.view_gc(id) {
                ObjView::Cons { .. } => {
                    let items = self.list_to_vec(bits);
                    let inner: Vec<String> = items.iter().map(|&x| self.print(x)).collect();
                    format!("({})", inner.join(" "))
                }
                ObjView::EmptyList => "()".to_string(),
                ObjView::Str(s) => {
                    // Readable form: escape the reader-significant chars so the
                    // output round-trips (matches Clojure's pr on strings).
                    let mut out = String::with_capacity(s.len() + 2);
                    out.push('"');
                    for c in s.chars() {
                        match c {
                            '"' => out.push_str("\\\""),
                            '\\' => out.push_str("\\\\"),
                            '\n' => out.push_str("\\n"),
                            '\t' => out.push_str("\\t"),
                            '\r' => out.push_str("\\r"),
                            _ => out.push(c),
                        }
                    }
                    out.push('"');
                    out
                }
                ObjView::Char(c) => c.to_string(),
                ObjView::Vector { elems, .. } => {
                    let inner: Vec<String> = elems.iter().map(|&x| self.print(x)).collect();
                    format!("#({})", inner.join(" "))
                }
                ObjView::Values(vals) => {
                    let inner: Vec<String> = vals.iter().map(|&x| self.print(x)).collect();
                    inner.join(" ")
                }
                ObjView::Closure { .. } => "#<closure>".to_string(),
                ObjView::MultiFn { .. } => "#<closure>".to_string(),
                ObjView::BigInt(i) => i.to_string(),
                ObjView::HugeInt(b) => b.to_string(),
                ObjView::Ratio(n, d) => format!("{n}/{d}"),
                ObjView::BoxFloat(f) => format!("{f}"),
                ObjView::Record { type_id, fields } => {
                    let inner: Vec<String> = fields.iter().map(|&x| self.print(x)).collect();
                    format!("#{}[{}]", self.sym_name(type_id), inner.join(" "))
                }
                ObjView::Escape { .. } => "#<continuation>".to_string(),
                ObjView::Cont(_) => "#<continuation>".to_string(),
                ObjView::PartialCont(_) => "#<partial-continuation>".to_string(),
                ObjView::Atom(_) => "#<atom>".to_string(),
                ObjView::Future(_) => "#<future>".to_string(),
            },
        }
    }

    /// Materialize a DIRECT variadic call's rest arg from its evaluated args.
    /// THE definition — every backend's frame builder calls this one, because
    /// a second copy of this policy is a silent divergence between tiers.
    ///
    /// `apply` does NOT come through here: Clojure hands the applied seq to
    /// the callee untouched (see `Prim::Apply`), and copying it into a fresh
    /// spine here would break `identical?` and force lazy tails.
    ///
    /// NB: the shape must not depend on how MANY args there are. An earlier
    /// attempt returned a chunked seq above a length threshold, which made
    /// `(list? (apply list (range 3)))` true and `(list? (apply list (range
    /// 100)))` false — a semantic that varies with arg count is a bug, however
    /// fast it is.
    pub(crate) fn mk_rest_seq(&mut self, xs: &[u64]) -> u64 {
        // An `apply` in progress has already computed this callee's rest arg —
        // the applied seq itself, shared. `take` so only the FIRST frame built
        // consumes it: the callee's own. A tail call out of that body builds
        // further frames, and those must collect their rest args normally.
        if let Some(slot) = self.apply_rest_slot.take() {
            return self.root_get(slot);
        }
        self.vec_to_list(xs)
    }

    /// Build a callee's FLAT activation frame from evaluated args. Slots
    /// `0..nparams` are the positional args; a variadic rest arg is the slot
    /// after them, holding the collected list; the remaining slots (up to
    /// `nslots`, assigned by the `flatten` pass for the body's let/catch
    /// bindings) start nil. `callee_bits` is the CALLEE closure's own encoded
    /// bits (0 for a top-level/no-closure frame) — stored into the frame's
    /// `caps_src` root slot, which `frame_cap` decodes to load captures
    /// straight out of the closure object (no per-call capture-array copy).
    /// Shared by every backend so the frame layout has one definition.
    pub fn build_call_frame(
        &mut self,
        nparams: usize,
        variadic: bool,
        nslots: u16,
        args: &[u64],
        callee_bits: u64,
    ) -> Locals {
        let nfixed = nparams + variadic as usize;
        debug_assert!(
            nslots as usize >= nfixed,
            "unflattened Lambda reached a call: nslots={nslots} < params={nfixed} (run flatten::flatten)"
        );
        let nslots = (nslots as usize).max(nfixed);
        let mut slots: Vec<AtomicU64> = Vec::with_capacity(nslots);
        if variadic {
            assert!(
                args.len() >= nparams,
                "arity: expected at least {nparams}, got {}",
                args.len()
            );
            slots.extend(args[..nparams].iter().map(|&a| AtomicU64::new(a)));
            let restseq = self.mk_rest_seq(&args[nparams..]);
            slots.push(AtomicU64::new(restseq));
        } else {
            assert!(
                args.len() == nparams,
                "arity: expected {nparams}, got {}",
                args.len()
            );
            slots.extend(args.iter().map(|&a| AtomicU64::new(a)));
        }
        let nil = self.encode(Val::Nil);
        while slots.len() < nslots {
            slots.push(AtomicU64::new(nil));
        }
        Some(Arc::new(Frame { slots, caps_src: AtomicU64::new(callee_bits) }))
    }

}
