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
use crate::value::{Caps, Cat, Frame, HeapId, Locals, Obj, RawTag, Sym, Val};

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

/// Objects per heap chunk. A power of two so index split is shift/mask.
const CHUNK_BITS: u32 = 12;
const CHUNK: usize = 1 << CHUNK_BITS;
/// Reserved slots in the chunk INDEX. The index Vec must never reallocate: a
/// lock-free reader traverses `chunks[c]` while another thread (under `heap_lock`)
/// may be appending a new chunk, and an index realloc would move the chunk
/// headers out from under the reader. Reserving a large index avoids that
/// (soft cap = CHUNKS_CAP * CHUNK objects; a lock-free/STW-growable index is the
/// clean unbounded fix, a later-phase item). Cheap: 24 bytes/slot.
const CHUNKS_CAP: usize = 1 << 16;

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
        Heap { chunks: Vec::with_capacity(CHUNKS_CAP), len: 0 }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Append an object; returns its flat id. A new chunk is PRE-INITIALIZED to
    /// sentinels (never `Vec::with_capacity`), so a lock-free reader never touches
    /// uninitialized memory or the mutable `Vec` length; allocation ASSIGNS an
    /// existing slot rather than pushing, and existing object addresses are stable.
    pub fn push(&mut self, o: Obj) -> usize {
        let i = self.len;
        let c = i >> CHUNK_BITS;
        if c >= self.chunks.len() {
            assert!(c < CHUNKS_CAP, "heap chunk index overflow: raise CHUNKS_CAP");
            self.chunks.push(vec![Obj::Moved(u32::MAX); CHUNK]);
        }
        // Assign the (already-initialized) slot via the unchecked path.
        *std::ops::IndexMut::index_mut(self, i) = o;
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
        // UNCHECKED on purpose: a bounds check reads the (inner and outer) Vec
        // `len`, which a concurrent `push` on the current fill-chunk WRITES — a
        // data race even when the two touch different objects (TSan-confirmed).
        // A `HeapId` is only ever handed out by `push`, so it is always in range;
        // `get_unchecked` reads the stable base pointer + a computed offset and
        // never the mutable length. (Object addresses are stable — segmented
        // heap — so the returned `&Obj` is sound while other objects allocate.)
        unsafe {
            self.chunks
                .get_unchecked(i >> CHUNK_BITS)
                .get_unchecked(i & (CHUNK - 1))
        }
    }
}

impl std::ops::IndexMut<usize> for Heap {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Obj {
        unsafe {
            self.chunks
                .get_unchecked_mut(i >> CHUNK_BITS)
                .get_unchecked_mut(i & (CHUNK - 1))
        }
    }
}

/// Words per arena chunk (2^18 = 262144 words = 2 MiB). Powers of two so the
/// split is shift/mask; big enough that chunk turnover is rare.
const WCHUNK_BITS: u32 = 18;
const WCHUNK: usize = 1 << WCHUNK_BITS;
/// Reserved slots in the arena chunk INDEX (same never-realloc discipline as
/// the object heap's chunk index — lock-free readers hold `&[u64]` into
/// chunks while another thread appends new ones).
const WCHUNKS_CAP: usize = 1 << 15;

/// The bump WORD ARENA holding hot object payloads (`Vector`/`Values`/
/// `Record` elements). Allocation is an atomic bump — no malloc, no free, no
/// lock on the fast path; a span is owned by exactly one Obj. Semi-space
/// discipline matches the object heap: the GC copies live spans to the top
/// and abandons the old words.
pub struct WordArena {
    chunks: Vec<Box<[u64]>>,
    /// Next free word index (atomic bump). May briefly exceed the initialized
    /// area during a chunk race; `word_alloc` repairs under the heap lock.
    len: AtomicU64,
    /// OVERSIZE spans (> one chunk): exact-size boxed slices, addressed as
    /// `BIG_BIT | index`. Rare (big flat arrays); appended under the heap lock.
    big: Vec<Box<[u64]>>,
}

/// High bit of a span offset marks an oversize allocation (index into `big`).
const BIG_BIT: u32 = 1 << 31;

impl WordArena {
    fn new() -> Self {
        WordArena {
            chunks: Vec::with_capacity(WCHUNKS_CAP),
            len: AtomicU64::new(0),
            big: Vec::with_capacity(1 << 12),
        }
    }
    #[inline]
    pub(crate) fn slice(&self, off: u32, len: u32) -> &[u64] {
        if off & BIG_BIT != 0 {
            let b = &self.big[(off & !BIG_BIT) as usize];
            return &b[..len as usize];
        }
        let (c, o) = ((off as usize) >> WCHUNK_BITS, (off as usize) & (WCHUNK - 1));
        // Spans never cross a chunk boundary (word_alloc pads); bounds are
        // guaranteed by construction, mirroring `Heap`'s unchecked access.
        unsafe { self.chunks.get_unchecked(c).get_unchecked(o..o + len as usize) }
    }
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn slice_mut(&self, off: u32, len: u32) -> &mut [u64] {
        if off & BIG_BIT != 0 {
            unsafe {
                let base = self.big[(off & !BIG_BIT) as usize].as_ptr() as *mut u64;
                return std::slice::from_raw_parts_mut(base, len as usize);
            }
        }
        let (c, o) = ((off as usize) >> WCHUNK_BITS, (off as usize) & (WCHUNK - 1));
        unsafe {
            let base = self.chunks.get_unchecked(c).as_ptr() as *mut u64;
            std::slice::from_raw_parts_mut(base.add(o), len as usize)
        }
    }
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
    pub(crate) parked: std::sync::atomic::AtomicBool,
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
    pub(crate) heap: UnsafeCell<Heap>,
    pub(crate) words: UnsafeCell<WordArena>,
    /// Serializes heap MUTATION (alloc, GC, in-place vector set). Reads are
    /// lock-free (segmented heap => stable object addresses).
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
            heap: UnsafeCell::new(Heap::new()),
            words: UnsafeCell::new(WordArena::new()),
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
        let me = register_mutator(&shared);
        Runtime { shared, shadow: Vec::new(), env_stack: Vec::new(), me, signal: Signal::default(), dyn_stack: Vec::new(), site_ic: std::cell::RefCell::new(Vec::new()), eval_bridge: None, _pd: PhantomData }
    }

    /// A fresh mutator handle for another OS thread, sharing this runtime's heap,
    /// globals, interner, and dispatch. The new handle has its OWN (empty) shadow
    /// stack and its own STW registry slot. This is what `spawn`/`future` hands to
    /// a `std::thread`.
    pub fn thread_handle(&self) -> Self {
        let me = register_mutator(&self.shared);
        Runtime { shared: self.shared.clone(), shadow: Vec::new(), env_stack: Vec::new(), me, signal: Signal::default(), dyn_stack: Vec::new(), site_ic: std::cell::RefCell::new(Vec::new()), eval_bridge: None, _pd: PhantomData }
    }

    // ── heap access (lock-free reads to stable addresses) ───
    /// The heap for READING. Object addresses are stable (segmented), so a shared
    /// reference is sound even while another thread allocates.
    #[inline]
    pub fn heap(&self) -> &Heap {
        unsafe { &*self.shared.heap.get() }
    }

    /// The payload word arena (see `WordArena`).
    #[inline]
    pub fn arena(&self) -> &WordArena {
        unsafe { &*self.shared.words.get() }
    }
    /// Read a payload span.
    #[inline]
    pub fn words(&self, off: u32, len: u32) -> &[u64] {
        self.arena().slice(off, len)
    }
    /// Mutate a payload span in place (array `%aset` etc.). Sound under the
    /// same discipline as `Heap`'s unchecked access: spans are exclusively
    /// owned, addresses are stable, and cross-thread visibility rides the
    /// existing safepoint/park protocol.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn words_mut(&self, off: u32, len: u32) -> &mut [u64] {
        self.arena().slice_mut(off, len)
    }
    /// Bump-allocate `n` UNINITIALIZED-to-zero words; the span never crosses a
    /// chunk boundary (the boundary remainder is padded away). Lock-free fast
    /// path; takes the heap lock only to append a chunk.
    pub fn word_alloc(&self, n: usize) -> u32 {
        use std::sync::atomic::Ordering::Relaxed;
        let n = n.max(1); // zero-length spans still get a distinct off
        if n > WCHUNK {
            // Oversize: an exact-size boxed slice in the side table.
            let _g = self.shared.heap_lock.lock().unwrap();
            let a = unsafe { &mut *self.shared.words.get() };
            let idx = a.big.len() as u32;
            assert!(idx < BIG_BIT, "word arena: big-span index overflow");
            a.big.push(vec![0u64; n].into_boxed_slice());
            return BIG_BIT | idx;
        }
        loop {
            let a = self.arena();
            let cur = a.len.load(Relaxed) as usize;
            // Pad to the next chunk when the span would cross a boundary.
            let start = if (cur & (WCHUNK - 1)) + n > WCHUNK {
                ((cur >> WCHUNK_BITS) + 1) << WCHUNK_BITS
            } else {
                cur
            };
            let end = start + n;
            if (end - 1) >> WCHUNK_BITS >= a.chunks.len() {
                // Missing chunk: append under the heap lock, then retry.
                let _g = self.shared.heap_lock.lock().unwrap();
                let a = unsafe { &mut *self.shared.words.get() };
                while (end - 1) >> WCHUNK_BITS >= a.chunks.len() {
                    assert!(a.chunks.len() < WCHUNKS_CAP, "word arena chunk index overflow");
                    a.chunks.push(vec![0u64; WCHUNK].into_boxed_slice());
                }
                continue;
            }
            if a.len.compare_exchange_weak(cur as u64, end as u64, Relaxed, Relaxed).is_ok() {
                return start as u32;
            }
        }
    }

    /// Bump-allocate a span holding `xs` (with `extra` reserved words after it,
    /// for growable arrays); `(off, len, cap)`.
    pub fn word_alloc_from(&self, xs: &[u64], extra: usize) -> (u32, u32, u32) {
        let cap = xs.len() + extra;
        let off = self.word_alloc(cap);
        self.words_mut(off, xs.len() as u32).copy_from_slice(xs);
        (off, xs.len() as u32, cap as u32)
    }
    /// Allocate a Vector object over a copy of `xs`.
    pub fn alloc_vector(&self, xs: &[u64]) -> HeapId {
        let (off, len, cap) = self.word_alloc_from(xs, 0);
        self.alloc(Obj::Vector { off, len, cap })
    }
    /// Allocate a Vector of `n` nil slots.
    pub fn alloc_vector_nil(&self, n: usize, nil: u64) -> HeapId {
        let off = self.word_alloc(n);
        self.words_mut(off, n as u32).fill(nil);
        self.alloc(Obj::Vector { off, len: n as u32, cap: n as u32 })
    }
    /// Allocate a Values packet over a copy of `xs`.
    pub fn alloc_values(&self, xs: &[u64]) -> HeapId {
        let (off, len, _cap) = self.word_alloc_from(xs, 0);
        self.alloc(Obj::Values { off, len })
    }
    /// Allocate a Record over a copy of `fields`.
    pub fn alloc_record(&self, type_id: Sym, fields: &[u64]) -> HeapId {
        let (off, len, _cap) = self.word_alloc_from(fields, 0);
        self.alloc(Obj::Record { type_id, off, len })
    }
    /// Append `xs` to a growable array in place: within `cap`, extend the span;
    /// past it, bump-allocate a doubled span and re-point the SAME Obj (object
    /// identity is the heap slot, so every reference observes the growth).
    pub fn arr_extend(&self, id: HeapId, xs: &[u64]) {
        loop {
            let (off, len, cap) = match &self.heap()[id as usize] {
                &Obj::Vector { off, len, cap } => (off, len, cap),
                _ => panic!("apush: not an array"),
            };
            let need = len as usize + xs.len();
            if need <= cap as usize {
                let _g = self.shared.heap_lock.lock().unwrap();
                // Re-check under the lock (a concurrent extend may have won).
                match &self.heap()[id as usize] {
                    &Obj::Vector { off: o2, len: l2, cap: c2 }
                        if (o2, l2, c2) == (off, len, cap) => {}
                    _ => continue,
                }
                self.arena().slice_mut(off, need as u32)[len as usize..].copy_from_slice(xs);
                if let Obj::Vector { len, .. } = &mut self.heap_mut()[id as usize] {
                    *len = need as u32;
                }
                return;
            }
            // Growth: bump-allocate the new span OUTSIDE the heap lock
            // (`word_alloc` may itself take it to append a chunk).
            let ncap = need.next_power_of_two().max(4);
            let noff = self.word_alloc(ncap);
            let _g = self.shared.heap_lock.lock().unwrap();
            match &self.heap()[id as usize] {
                &Obj::Vector { off: o2, len: l2, cap: c2 }
                    if (o2, l2, c2) == (off, len, cap) => {}
                _ => continue, // lost a race: retry (the fresh span is abandoned)
            }
            let dst = self.arena().slice_mut(noff, need as u32);
            dst[..len as usize].copy_from_slice(self.arena().slice(off, len));
            dst[len as usize..].copy_from_slice(xs);
            if let Obj::Vector { off, len, cap } = &mut self.heap_mut()[id as usize] {
                *off = noff;
                *len = need as u32;
                *cap = ncap as u32;
            }
            return;
        }
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
    pub fn alloc(&self, o: Obj) -> HeapId {
        let _g = self.shared.heap_lock.lock().unwrap();
        self.shared.allocs.fetch_add(1, Ordering::Relaxed);
        // SAFETY: heap mutation is serialized by `heap_lock`; the segmented heap
        // never relocates existing objects on push, so lock-free readers are safe.
        self.heap_mut().push(o) as HeapId
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
    /// The empty list `()` — a heap value distinct from nil.
    pub fn enc_empty_list(&mut self) -> u64 {
        let id = self.alloc(Obj::EmptyList);
        M::R::enc_ref(id)
    }
    pub fn is_empty_list(&self, bits: u64) -> bool {
        if let RawTag::Ref = M::R::tag_of(bits) {
            return matches!(&self.heap()[M::R::as_ref(bits) as usize], Obj::EmptyList);
        }
        false
    }
    pub fn cons(&mut self, head: u64, tail: u64) -> u64 {
        // A cons chain terminates in nil, never in the standalone `()` value — so
        // `(cons x ())` is `(x)` and iteration (which stops at nil) is unaffected.
        let tail = if self.is_empty_list(tail) { self.enc_nil() } else { tail };
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
                    // All empty lists are equal (and only to each other, never nil).
                    (Obj::EmptyList, Obj::EmptyList) => true,
                    // Ratios are reduced, so equal iff same num & den (never = an
                    // integer, since a reduced ratio has den > 1).
                    (Obj::Ratio(na, da), Obj::Ratio(nb, db)) => na == nb && da == db,
                    // Structural equality for aggregates (R7RS `equal?` on
                    // strings/vectors; ordered field equality for records — the
                    // general aggregate case). Order-INSENSITIVE collections
                    // (e.g. a hash-map) are a frontend concern layered on top.
                    (Obj::Str(sa), Obj::Str(sb)) => sa == sb,
                    (Obj::Char(ca), Obj::Char(cb)) => ca == cb,
                    (&Obj::Vector { off: oa, len: la, .. }, &Obj::Vector { off: ob, len: lb, .. }) => {
                        la == lb
                            && (0..la).all(|k| {
                                let (x, y) =
                                    (self.words(oa, la)[k as usize], self.words(ob, lb)[k as usize]);
                                self.equal(x, y)
                            })
                    }
                    (
                        &Obj::Record { type_id: ta, off: oa, len: la },
                        &Obj::Record { type_id: tb, off: ob, len: lb },
                    ) => {
                        ta == tb
                            && la == lb
                            && (0..la).all(|k| {
                                let (x, y) =
                                    (self.words(oa, la)[k as usize], self.words(ob, lb)[k as usize]);
                                self.equal(x, y)
                            })
                    }
                    _ => false, // identity already handled; distinct other objects differ
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
                    Val::Ref(id) => match &self.heap()[id as usize] {
                        Obj::Str(s) => s.clone(),
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
                let &Obj::Vector { off, len, .. } = &self.heap()[id as usize] else {
                    panic!("vector-ref: not a vector");
                };
                *self
                    .words(off, len)
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
                let &Obj::Vector { off, len, .. } = &self.heap()[id as usize] else {
                    panic!("vector-set!: not a vector");
                };
                let slot = self
                    .words_mut(off, len)
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
                let &Obj::Vector { off, len, .. } = &self.heap()[id as usize] else {
                    panic!("aclone: not an array");
                };
                let noff = self.word_alloc(len.max(1) as usize);
                self.arena().slice_mut(noff, len).copy_from_slice(self.words(off, len));
                let nid = self.alloc(Obj::Vector { off: noff, len, cap: len });
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
                let &Obj::Vector { off, len, .. } = &self.heap()[id as usize] else {
                    panic!("sort-arr: not an array")
                };
                let elems = self.words(off, len).to_vec();
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
                    matches!(self.decode(e), Val::Ref(i) if matches!(&self.heap()[i as usize], Obj::Str(_)))
                });
                if all_str {
                    let mut v = elems;
                    v.sort_by(|&a, &b| {
                        let (Val::Ref(ia), Val::Ref(ib)) = (self.decode(a), self.decode(b)) else {
                            unreachable!("checked str")
                        };
                        let (Obj::Str(sa), Obj::Str(sb)) =
                            (&self.heap()[ia as usize], &self.heap()[ib as usize])
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
                    let (np, va) = match &self.heap()[id as usize] {
                        Obj::Closure { nparams, variadic, .. } => (*nparams, *variadic),
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
                let &Obj::Record { off, len, .. } = &self.heap()[id as usize] else {
                    panic!("lazy-realize!: not a record");
                };
                let fs = self.words_mut(off, len);
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
            // Join an array of already-stringified elements in ONE native pass
            // (see `Prim::StrJoinArr`'s doc comment in ir.rs).
            Prim::StrJoinArr => {
                let Val::Ref(id) = self.decode(args[0]) else { panic!("str-join-arr: not an array"); };
                let elems: Vec<u64> = match &self.heap()[id as usize] {
                    &Obj::Vector { off, len, .. } => self.words(off, len).to_vec(),
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
                let &Obj::Vector { off, len, .. } = &self.heap()[id as usize] else {
                    panic!("ashift: not an array");
                };
                if len == 0 {
                    self.enc_nil()
                } else {
                    let first = self.words(off, len)[0];
                    // Shift left in place and shrink (object identity preserved).
                    let ws = self.words_mut(off, len);
                    ws.copy_within(1.., 0);
                    let _g = self.shared.heap_lock.lock().unwrap();
                    if let Obj::Vector { len, .. } = &mut self.heap_mut()[id as usize] {
                        *len -= 1;
                    }
                    first
                }
            }
            Prim::ArrClear => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("aclear: not an array");
                };
                let _g = self.shared.heap_lock.lock().unwrap();
                let Obj::Vector { len, .. } = &mut self.heap_mut()[id as usize] else {
                    panic!("aclear: not an array");
                };
                *len = 0;
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
                let &Obj::Record { off, len, .. } = &self.heap()[id as usize] else {
                    panic!("field access on non-record");
                };
                self.words(off, len)[idx]
            }
            Prim::Hash => {
                let h = self.hash_value(args[0]);
                self.encode(Val::Int((h & 0x7fff_ffff) as i128))
            }
            Prim::VectorLen => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("vector-length: not a vector");
                };
                let &Obj::Vector { len, .. } = &self.heap()[id as usize] else {
                    panic!("vector-length: not a vector");
                };
                self.encode(Val::Int(len as i128))
            }
            Prim::Values => {
                let id = self.alloc_values(args);
                M::R::enc_ref(id)
            }
            Prim::ValuesToList => {
                // Unpack a `values` packet into a list; a lone value becomes a
                // one-element list so single-valued producers work too.
                if let Val::Ref(id) = self.decode(args[0]) {
                    if let &Obj::Values { off, len } = &self.heap()[id as usize] {
                        let vals = self.words(off, len).to_vec();
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
                    Val::Ref(id) => match &self.heap()[id as usize] {
                        Obj::Record { len, .. } => *len as i128,
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
                    && matches!(
                        &self.heap()[M::R::as_ref(args[0]) as usize],
                        Obj::BigInt(_) | Obj::HugeInt(_)
                    );
                self.encode(Val::Bool(is_big))
            }
            Prim::ToLong => {
                // `(int \a)` => 97: a char coerces to its code point, same as
                // `(int c)` does in real Clojure.
                if let Val::Ref(id) = self.decode(args[0]) {
                    if let Obj::Char(c) = &self.heap()[id as usize] {
                        return self.encode(Val::Int(*c as i128));
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
                    Val::Ref(id) => match &self.heap()[id as usize] {
                        Obj::Str(s) => s.chars().collect(),
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
                    Val::Ref(id) => match &self.heap()[id as usize] {
                        // SIGNED bytes (the JVM's `byte`), so in-language wire
                        // code round-trips exactly like Java's.
                        Obj::Str(s) => s.bytes().map(|b| M::R::enc_int(b as i8 as i64)).collect(),
                        _ => panic!("%str->bytes: not a string"),
                    },
                    _ => panic!("%str->bytes: not a string"),
                };
                let id = self.alloc_vector(&bytes);
                M::R::enc_ref(id)
            }
            Prim::BytesToStr => {
                let bytes: Vec<u8> = match self.decode(args[0]) {
                    Val::Ref(id) => match &self.heap()[id as usize] {
                        &Obj::Vector { off, len, .. } => self
                            .words(off, len)
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
                    Val::Ref(id) => match &self.heap()[id as usize] {
                        &Obj::Vector { off, len, .. } => self
                            .words(off, len)
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
                let id = self.alloc(Obj::Atom(Arc::new(AtomicU64::new(args[0]))));
                M::R::enc_ref(id)
            }
            Prim::AtomGet => {
                let Obj::Atom(a) = &self.heap()[M::R::as_ref(args[0]) as usize] else {
                    panic!("atom-get: not an atom");
                };
                a.load(Ordering::Acquire)
            }
            Prim::AtomSet => {
                let Obj::Atom(a) = &self.heap()[M::R::as_ref(args[0]) as usize] else {
                    panic!("atom-set: not an atom");
                };
                a.store(args[1], Ordering::Release);
                args[1]
            }
            Prim::AtomCas => {
                let Obj::Atom(a) = &self.heap()[M::R::as_ref(args[0]) as usize] else {
                    panic!("atom-cas: not an atom");
                };
                let ok = a
                    .compare_exchange(args[1], args[2], Ordering::AcqRel, Ordering::Acquire)
                    .is_ok();
                M::R::enc_bool(ok)
            }
            Prim::Field => {
                let Val::Ref(id) = self.decode(args[0]) else {
                    panic!("field: not a record");
                };
                let Val::Int(i) = self.decode(args[1]) else {
                    panic!("field: index must be an int");
                };
                match &self.heap()[id as usize] {
                    &Obj::Record { off, len, .. } => self.words(off, len)[i as usize],
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
            if let Obj::Record { type_id, .. } = &self.heap()[id as usize] {
                return Some(*type_id);
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
        let &Obj::Record { off, len, .. } = &self.heap()[id as usize] else {
            panic!("field access on non-record");
        };
        self.words(off, len)[idx]
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
            Val::Ref(id) => match &self.heap()[id as usize] {
                Obj::Str(s) => mix_str(FNV_OFFSET, s),
                Obj::Char(c) => mix(0xc4a_u32, *c as u32),
                Obj::BigInt(i) => mix(FNV_OFFSET, *i as u64 as u32),
                Obj::HugeInt(b) => mix_str(FNV_OFFSET, &b.to_string()),
                Obj::Ratio(n, d) => mix(mix(FNV_OFFSET, *n as u64 as u32), *d as u64 as u32),
                Obj::BoxFloat(f) => mix(FNV_OFFSET, f.to_bits() as u32),
                &Obj::Record { type_id, off, len } => {
                    let mut h = mix_str(0x9e37_79b9u32, self.sym_name(type_id));
                    for k in 0..len {
                        let f = self.words(off, len)[k as usize];
                        h = mix(h, self.hash_value(f));
                    }
                    h
                }
                Obj::Cons { .. } => {
                    let mut h = 0x1000_193u32;
                    for x in self.list_to_vec(bits) {
                        h = mix(h, self.hash_value(x));
                    }
                    h
                }
                &Obj::Vector { off, len, .. } => {
                    let mut h = 0x27d4_eb2fu32;
                    for k in 0..len {
                        let x = self.words(off, len)[k as usize];
                        h = mix(h, self.hash_value(x));
                    }
                    h
                }
                _ => mix(FNV_OFFSET, id),
            },
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
            Val::Ref(id) => match &self.heap()[id as usize] {
                Obj::Record { type_id, .. } => return *type_id,
                Obj::Cons { .. } => (TYPE_TAG_LIST, "List"),
                Obj::EmptyList => (TYPE_TAG_EMPTYLIST, "EmptyList"),
                Obj::Vector { .. } => (TYPE_TAG_VECTOR, "Vector"),
                Obj::Str(_) => (TYPE_TAG_STRING, "String"),
                Obj::Char(_) => (TYPE_TAG_CHAR, "Char"),
                Obj::Closure { .. } | Obj::MultiFn { .. } => (TYPE_TAG_FN, "Fn"),
                Obj::BigInt(_) | Obj::HugeInt(_) => (TYPE_TAG_LONG, "Long"),
                Obj::Ratio(..) => (TYPE_TAG_RATIO, "Ratio"),
                Obj::BoxFloat(_) => (TYPE_TAG_DOUBLE, "Double"),
                Obj::Atom(_) => (TYPE_TAG_ATOM, "Atom"),
                Obj::Future(_) => (TYPE_TAG_FUTURE, "Future"),
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
        let mut cur = cur;
        loop {
            match self.decode(cur) {
                Val::Nil => return Some(None),
                Val::Ref(cid) => {
                    match &self.heap()[cid as usize] {
                        Obj::Cons { head, tail } => return Some(Some((*head, *tail))),
                        Obj::EmptyList => return Some(None),
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
                    // Lazy / frontend node: force ONE step through the
                    // registered `seq` and retry (it yields cons/chunked/nil).
                    let sf = self.seq_handler()?;
                    let forced = top.invoke(top, self, sf, &[cur]);
                    if self.pending() {
                        return Some(None);
                    }
                    if forced == cur {
                        return None; // seq made no progress: not a sequence
                    }
                    cur = forced;
                }
                _ => None?,
            }
        }
    }
    /// Flatten a whole sequence (cons / chunked / lazy) into a Vec, forcing
    /// through the registered `seq` fn as needed. The general `apply` path.
    pub fn seq_flatten(&mut self, top: &dyn crate::code::CodeSpace<M>, bits: u64) -> Vec<u64> {
        let mut out = Vec::new();
        let mut cur = bits;
        loop {
            // Whole realized chunks copy in one extend (no per-element step).
            if let Some((arr, coff, cend, more)) = self.as_chunked(cur) {
                let (aoff, alen) = self.arr_span_pub(arr);
                let words = self.words(aoff, alen);
                out.extend_from_slice(&words[coff as usize..cend as usize]);
                cur = more;
                continue;
            }
            match self.seq_step(top, cur) {
                Some(Some((h, t))) => {
                    out.push(h);
                    cur = t;
                }
                Some(None) => return out,
                None => {
                    // A node we cannot walk and cannot force is a configuration
                    // error (frontend forgot set_seq_fn) — fail LOUDLY rather
                    // than silently truncating the argument list.
                    panic!(
                        "seq_flatten: unwalkable sequence node {} (no seq fn registered?)",
                        self.print(cur)
                    );
                }
            }
        }
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
    pub fn multifn_select(&mut self, callee: u64, argc: usize) -> Option<u64> {
        let Val::Ref(id) = self.decode(callee) else { return None };
        let (sel, known) = match &self.heap()[id as usize] {
            Obj::MultiFn { fixed, variadic } => {
                let f = fixed.get(argc).copied().unwrap_or(0);
                if f != 0 {
                    (f, true)
                } else if let Some((min, vf)) = variadic {
                    if argc >= *min {
                        (*vf, true)
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
            if let Obj::Str(s) = &self.heap()[id as usize] {
                return s.clone();
            }
        }
        panic!("{who}: argument is not a string");
    }

    /// `(num, den)` if `bits` is a Ratio, else `None`.
    fn as_ratio(&self, bits: u64) -> Option<(i128, i128)> {
        if let Val::Ref(id) = self.decode(bits) {
            if let Obj::Ratio(n, d) = &self.heap()[id as usize] {
                return Some((*n, *d));
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
        let Val::Ref(id) = self.decode(pv) else { panic!("pvec: not a record") };
        let &Obj::Record { off, len, .. } = &self.heap()[id as usize] else { panic!("pvec: not a record") };
        let fields = self.words(off, len);
        let cnt = match self.decode(fields[1]) { Val::Int(i) => i as i64, _ => panic!("pvec cnt") };
        let shift = match self.decode(fields[2]) { Val::Int(i) => i as i64, _ => panic!("pvec shift") };
        (fields[0], cnt, shift, fields[3], fields[4])
    }
    /// The array inside a `'VectorNode` (field 1).
    fn node_arr(&self, node: u64) -> u64 {
        let Val::Ref(id) = self.decode(node) else { panic!("vnode: not a record") };
        let &Obj::Record { off, len, .. } = &self.heap()[id as usize] else { panic!("vnode: not a record") };
        self.words(off, len)[1]
    }
    /// The (off, len) span of an array value.
    fn arr_span(&self, arr: u64) -> (u32, u32) {
        let Val::Ref(id) = self.decode(arr) else { panic!("pvec: array not a ref") };
        let &Obj::Vector { off, len, .. } = &self.heap()[id as usize] else { panic!("pvec: not an array") };
        (off, len)
    }
    fn arr_clone(&self, arr: u64) -> Vec<u64> {
        let (off, len) = self.arr_span(arr);
        self.words(off, len).to_vec()
    }
    fn arr_at(&self, arr: u64, i: usize) -> u64 {
        let (off, len) = self.arr_span(arr);
        self.words(off, len)[i]
    }
    pub(crate) fn arr_at_pub(&self, arr: u64, i: usize) -> u64 {
        self.arr_at(arr, i)
    }
    pub(crate) fn arr_span_pub(&self, arr: u64) -> (u32, u32) {
        self.arr_span(arr)
    }
    fn mk_array(&mut self, v: Vec<u64>) -> u64 {
        M::R::enc_ref(self.alloc_vector(&v))
    }
    /// A fresh array = `arr` ++ [x], copied span-to-span (no Vec temporary).
    /// The tail-conj hot path.
    fn mk_array_append1(&mut self, arr: u64, x: u64) -> u64 {
        let (soff, slen) = self.arr_span(arr);
        let noff = self.word_alloc(slen as usize + 1);
        let dst = self.arena().slice_mut(noff, slen + 1);
        dst[..slen as usize].copy_from_slice(self.arena().slice(soff, slen));
        dst[slen as usize] = x;
        M::R::enc_ref(self.alloc(Obj::Vector { off: noff, len: slen + 1, cap: slen + 1 }))
    }
    /// A fresh one-element array (no Vec temporary).
    fn mk_array1(&mut self, x: u64) -> u64 {
        let noff = self.word_alloc(1);
        self.arena().slice_mut(noff, 1)[0] = x;
        M::R::enc_ref(self.alloc(Obj::Vector { off: noff, len: 1, cap: 1 }))
    }
    /// (arr, off, end, more) of a `'ChunkedCons` record, or None. Lets the cons
    /// prims (`%first`/`%rest`) treat a chunk transparently as a seq — only on the
    /// as_cons-miss path, so the common cons case pays nothing.
    pub(crate) fn as_chunked(&self, bits: u64) -> Option<(u64, i64, i64, u64)> {
        let Val::Ref(id) = self.decode(bits) else { return None };
        let &Obj::Record { type_id, off: foff, len: flen } = &self.heap()[id as usize] else {
            return None;
        };
        if self.sym_name(type_id) != "ChunkedCons" {
            return None;
        }
        let fields = self.words(foff, flen);
        let (f0, f1, f2, f3) = (fields[0], fields[1], fields[2], fields[3]);
        let off = match self.decode(f1) { Val::Int(i) => i as i64, _ => return None };
        let end = match self.decode(f2) { Val::Int(i) => i as i64, _ => return None };
        Some((f0, off, end, f3))
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
        let mut ret = self.arr_clone(parr);
        if level == 5 {
            ret[subidx] = tailnode;
        } else {
            let child = self.arr_at(parr, subidx);
            ret[subidx] = if matches!(self.decode(child), Val::Nil) {
                self.pv_new_path(level - 5, tailnode)
            } else {
                self.pv_push_tail(cnt, level - 5, child, tailnode)
            };
        }
        let arr = self.mk_array(ret);
        self.mk_node(arr)
    }
    fn pv_conj(&mut self, pv: u64, o: u64) -> u64 {
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
    fn pv_nth(&mut self, pv: u64, i: i64) -> u64 {
        let (_, cnt, shift, root, tail) = self.pv_read(pv);
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
        let mut ret = self.arr_clone(narr);
        if level == 0 {
            ret[(i & 31) as usize] = val;
        } else {
            let subidx = ((i >> level) & 31) as usize;
            let child = self.arr_at(narr, subidx);
            ret[subidx] = self.pv_do_assoc(level - 5, child, i, val);
        }
        let arr = self.mk_array(ret);
        self.mk_node(arr)
    }
    fn pv_assoc(&mut self, pv: u64, n: i64, val: u64) -> u64 {
        let (meta, cnt, shift, root, tail) = self.pv_read(pv);
        if n >= 0 && n < cnt {
            if Self::tail_off(cnt) <= n {
                let mut nt = self.arr_clone(tail);
                nt[(n & 31) as usize] = val;
                let new_tail = self.mk_array(nt);
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
        matches!(self.decode(b), Val::Nil)
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
        let Val::Ref(id) = self.decode(node) else { panic!("hamt-assoc: not a node") };
        let &Obj::Record { type_id, off: foff, len: flen } = &self.heap()[id as usize] else { panic!("hamt-assoc: not a node") };
        let fields: Vec<u64> = self.words(foff, flen).to_vec();
        let kind = self.sym_name(type_id).to_string();
        match kind.as_str() {
            "BitmapIndexedNode" => {
                let bitmap = match self.decode(fields[1]) { Val::Int(i) => i as u32, _ => panic!("bitmap") };
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
            "ArrayNode" => {
                let cnt = match self.decode(fields[1]) { Val::Int(i) => i as i64, _ => panic!("cnt") };
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
            "HashCollisionNode" => {
                let chash = match self.decode(fields[1]) { Val::Int(i) => i as u32, _ => panic!("chash") };
                let cnt = match self.decode(fields[2]) { Val::Int(i) => i as i64, _ => panic!("cnt") };
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
            _ => panic!("hamt-assoc: unknown node kind {kind}"),
        }
    }
    /// `(-inode-lookup node shift hash key not-found)`.
    fn hamt_lookup(&self, node: u64, shift: u32, hash: u32, key: u64, not_found: u64) -> u64 {
        let Val::Ref(id) = self.decode(node) else { return not_found };
        let &Obj::Record { type_id, off: foff, len: flen } = &self.heap()[id as usize] else { return not_found };
        let fields = self.words(foff, flen);
        match self.sym_name(type_id) {
            "BitmapIndexedNode" => {
                let bitmap = match self.decode(fields[1]) { Val::Int(i) => i as u32, _ => return not_found };
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
            "ArrayNode" => {
                let idx = ((hash >> shift) & 31) as usize;
                let child = self.arr_at(fields[2], idx);
                if self.is_nil_bits(child) {
                    not_found
                } else {
                    self.hamt_lookup(child, shift + 5, hash, key, not_found)
                }
            }
            "HashCollisionNode" => {
                let cnt = match self.decode(fields[2]) { Val::Int(i) => i as usize, _ => return not_found };
                let arr = self.arr_clone(fields[3]);
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
        let Val::Ref(id) = self.decode(node) else { return node };
        let &Obj::Record { type_id, off: foff, len: flen } = &self.heap()[id as usize] else { return node };
        let fields: Vec<u64> = self.words(foff, flen).to_vec();
        let fields = &fields[..];
        match self.sym_name(type_id) {
            "BitmapIndexedNode" => {
                let bitmap = match self.decode(fields[1]) { Val::Int(i) => i as u32, _ => return node };
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
            "ArrayNode" => {
                let cnt = match self.decode(fields[1]) { Val::Int(i) => i as i64, _ => return node };
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
            "HashCollisionNode" => {
                let chash = match self.decode(fields[1]) { Val::Int(i) => i as u32, _ => return node };
                let cnt = match self.decode(fields[2]) { Val::Int(i) => i as i64, _ => return node };
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
    fn hamt_map_assoc(&mut self, root: u64, key: u64, val: u64) -> (u64, bool) {
        let hash = self.hash_masked(key);
        if self.is_nil_bits(root) {
            (self.hamt_single(hash, 0, key, val), true)
        } else {
            self.hamt_assoc(root, 0, hash, key, val)
        }
    }
    fn hamt_map_lookup(&self, root: u64, key: u64, not_found: u64) -> u64 {
        let hash = self.hash_masked(key);
        self.hamt_lookup(root, 0, hash, key, not_found)
    }
    fn hamt_map_without(&mut self, root: u64, key: u64) -> u64 {
        let hash = self.hash_masked(key);
        self.hamt_without(root, 0, hash, key)
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
            if let Obj::Str(s) = &self.heap()[id as usize] {
                return s.clone();
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
            Val::Ref(id) => match &self.heap()[id as usize] {
                Obj::Cons { .. } => {
                    let items = self.list_to_vec(bits);
                    let inner: Vec<String> = items.iter().map(|&x| self.print(x)).collect();
                    format!("({})", inner.join(" "))
                }
                Obj::EmptyList => "()".to_string(),
                Obj::Str(s) => {
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
                Obj::Char(c) => c.to_string(),
                &Obj::Vector { off, len, .. } => {
                    let inner: Vec<String> =
                        self.words(off, len).to_vec().iter().map(|&x| self.print(x)).collect();
                    format!("#({})", inner.join(" "))
                }
                &Obj::Values { off, len } => {
                    let inner: Vec<String> =
                        self.words(off, len).to_vec().iter().map(|&x| self.print(x)).collect();
                    inner.join(" ")
                }
                Obj::Closure { .. } => "#<closure>".to_string(),
                Obj::MultiFn { .. } => "#<closure>".to_string(),
                Obj::BigInt(i) => i.to_string(),
                Obj::HugeInt(b) => b.to_string(),
                Obj::Ratio(n, d) => format!("{n}/{d}"),
                Obj::BoxFloat(f) => format!("{f}"),
                &Obj::Record { type_id, off, len } => {
                    let inner: Vec<String> =
                        self.words(off, len).to_vec().iter().map(|&x| self.print(x)).collect();
                    format!("#{}[{}]", self.sym_name(type_id), inner.join(" "))
                }
                Obj::Escape { .. } => "#<continuation>".to_string(),
                Obj::Cont(_) => "#<continuation>".to_string(),
                Obj::PartialCont(_) => "#<partial-continuation>".to_string(),
                Obj::Atom(_) => "#<atom>".to_string(),
                Obj::Future(_) => "#<future>".to_string(),
                Obj::Moved(_) => "#<moved>".to_string(),
            },
        }
    }

    /// Build a callee's FLAT activation frame from evaluated args. Slots
    /// `0..nparams` are the positional args; a variadic rest arg is the slot
    /// after them, holding the collected list; the remaining slots (up to
    /// `nslots`, assigned by the `flatten` pass for the body's let/catch
    /// bindings) start nil. The closure's capture array rides along (one `Arc`
    /// bump). Shared by every backend so the frame layout has one definition.
    pub fn build_call_frame(
        &mut self,
        nparams: usize,
        variadic: bool,
        nslots: u16,
        args: &[u64],
        caps: Caps,
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
        let nil = self.encode(Val::Nil);
        while slots.len() < nslots {
            slots.push(AtomicU64::new(nil));
        }
        Some(Arc::new(Frame { slots, caps }))
    }

}
