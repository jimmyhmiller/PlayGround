//! Runtime support for JIT-compiled code.
//!
//! This module defines the structures and `extern "C"` functions that
//! JIT'd code interacts with at runtime:
//!
//! - [`Thread`] — the per-thread struct passed as the first parameter
//!   of every JIT'd function. Holds the safepoint state, the head of
//!   the per-thread shadow-stack chain, pointers to the GC heap and
//!   the JIT code table.
//! - [`Frame`] / [`FrameOrigin`] — per-call shadow-stack frames. JIT'd
//!   code allocates these on its native stack, links them into the
//!   chain on entry, and unlinks on return. The GC walks the chain
//!   to find live heap pointers.
//! - [`ClosureHeader`] — heap layout for closure objects.
//! - `ai_gc_*` extern fns — called by JIT'd code. Registered with the
//!   inkwell execution engine via `add_global_mapping`.
//!
//! The integration with the extracted `gc` module:
//! - One `Heap` per `Runtime`, plus one `ThreadState` per mutator
//!   thread (just one in v1).
//! - We register a `walk_jit_frames` walker with the heap so GCs can
//!   scan our shadow-stack chain. Before allocating, we publish the
//!   chain head into the thread's `parked_jit_fp` so the GC picks it
//!   up; we clear it on return.
//!
//! The shape of `Thread`, `Frame`, and `FrameOrigin` is part of the
//! ABI between Rust and JIT'd code. Layouts are asserted at compile
//! time (see `const _ : () = …` blocks) so a layout change breaks
//! the build rather than silently corrupting at runtime.

use crate::codegen::{FieldMeta, ShapeMeta};
use crate::gc::{Full, Heap, ObjHeader, ThreadState, TypeInfo};
use crate::hash::Hash;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// =============================================================================
// Thread
// =============================================================================

/// Per-thread state visible to JIT'd code.
///
/// `state`, `top_frame`, `heap`, `code_table`, and `dyna_thread` are
/// accessed by JIT'd LLVM IR via fixed byte offsets — see
/// [`thread_offsets`]. The layout asserts below enforce the ABI.
#[repr(C)]
pub struct Thread {
    /// Safepoint flag. `0` = running normally. Non-zero means the GC
    /// has requested a safepoint and the next JIT-side check will
    /// trampoline into [`ai_gc_pollcheck_slow`].
    pub state: u8,
    _pad: [u8; 7],

    /// Head of this thread's JIT-side shadow-stack chain. Each
    /// JIT'd function alloca's its own [`Frame`] on entry, links
    /// it in here, and unlinks on return.
    pub top_frame: *mut Frame,

    /// Pointer to the GC heap. JIT'd code calls runtime fns
    /// (`ai_gc_alloc_*`) which dereference this to actually allocate.
    pub heap: *mut Heap,

    /// Pointer to the JIT code table for closure indirect calls
    /// (`hash → fn_ptr`). Stored on `Thread` so the runtime lookup
    /// fn doesn't need a global.
    pub code_table: *const CodeTable,

    /// Pointer to the dynalloc `ThreadState` that this `Thread`
    /// shadows. Used to set/clear `parked_jit_fp` around allocations
    /// so the GC walks our chain.
    pub dyna_thread: *const ThreadState,

    /// `TypeInfo` for the `BoxedInt` heap shape used when generic
    /// code stores an `Int` in a TypeVar slot. `ai_gc_box_int`
    /// reads this to know which shape to allocate.
    pub boxed_int_ti: *const TypeInfo,
    /// `TypeInfo` for the `String` heap shape. Layout: GC header +
    /// varlen-byte section (count + UTF-8 bytes). Used by string
    /// literal codegen and by `ai_str_*` runtime fns. Also backs the
    /// `Bytes` shape (identical layout).
    pub string_ti: *const TypeInfo,
    /// `TypeInfo` for the `Array` heap shape. Layout: GC header +
    /// varlen-Values section (count + N GC-traced pointer slots). Used
    /// by `ai_array_*` runtime fns. Elements are uniform boxed pointers.
    pub array_ti: *const TypeInfo,
    /// `TypeInfo` for the `Atom` heap shape. Layout: GC header + exactly
    /// one GC-traced pointer slot (the current value). A *distinct* shape
    /// from `Array` so the runtime, GC, and reflection can tell a shared
    /// mutable cell apart from immutable data. `ai_atom_new` reads this.
    pub atom_ti: *const TypeInfo,
}

pub mod thread_offsets {
    //! Byte offsets within [`Thread`]. Mirror these in LLVM codegen.
    pub const STATE: usize = 0;
    pub const TOP_FRAME: usize = 8;
    pub const HEAP: usize = 16;
    pub const CODE_TABLE: usize = 24;
    pub const DYNA_THREAD: usize = 32;
    pub const BOXED_INT_TI: usize = 40;
    pub const STRING_TI: usize = 48;
}

const _: () = {
    assert!(core::mem::offset_of!(Thread, state) == thread_offsets::STATE);
    assert!(core::mem::offset_of!(Thread, top_frame) == thread_offsets::TOP_FRAME);
    assert!(core::mem::offset_of!(Thread, heap) == thread_offsets::HEAP);
    assert!(core::mem::offset_of!(Thread, code_table) == thread_offsets::CODE_TABLE);
    assert!(core::mem::offset_of!(Thread, dyna_thread) == thread_offsets::DYNA_THREAD);
    assert!(core::mem::offset_of!(Thread, boxed_int_ti) == thread_offsets::BOXED_INT_TI);
    assert!(core::mem::offset_of!(Thread, string_ti) == thread_offsets::STRING_TI);
};

// =============================================================================
// Frame + FrameOrigin
// =============================================================================

/// Per-call shadow-stack frame header.
///
/// JIT'd code allocates a `{ Frame, [*mut u8; N] }` on the native stack
/// (where N is the function's GC-typed local count), links it into the
/// thread chain via [`prologue`-equivalent IR][crate::codegen], and
/// unlinks on return.
///
/// Layout (load-bearing — matches LLVM IR `{ ptr, ptr, [N x ptr] }`):
///
/// ```text
/// offset 0  : parent  : *mut Frame
/// offset 8  : origin  : *const FrameOrigin
/// offset 16 : roots   : [*mut u8; N]  (variable, walked via origin.num_roots)
/// ```
#[repr(C)]
pub struct Frame {
    pub parent: *mut Frame,
    pub origin: *const FrameOrigin,
    // followed by [*mut u8; num_roots] starting at offset 16
}

pub mod frame_offsets {
    pub const PARENT: usize = 0;
    pub const ORIGIN: usize = 8;
    /// First root slot. N slots of 8 bytes each follow.
    pub const ROOTS: usize = 16;
}

const _: () = {
    assert!(core::mem::offset_of!(Frame, parent) == frame_offsets::PARENT);
    assert!(core::mem::offset_of!(Frame, origin) == frame_offsets::ORIGIN);
    // Frame's declared size equals ROOTS — the trailing array is
    // outside the struct (variable-length).
    assert!(core::mem::size_of::<Frame>() == frame_offsets::ROOTS);
};

/// Static per-function descriptor. One emitted per JIT'd function as
/// a private constant global. The frame's `origin` field points at it.
///
/// The GC reads `num_roots` to know how many root slots follow the
/// frame header.
#[repr(C)]
pub struct FrameOrigin {
    pub num_roots: u32,
    _pad: u32,
    pub name: *const u8,
}

impl FrameOrigin {
    pub const fn new(num_roots: u32, name: *const u8) -> Self {
        FrameOrigin {
            num_roots,
            _pad: 0,
            name,
        }
    }
}

// =============================================================================
// Closure heap layout
// =============================================================================

/// The fixed prefix of a closure object on the heap, after the GC's
/// `Full` header.
///
/// The full closure layout is:
///
/// ```text
/// offset 0    : Full header           (16 bytes)
/// offset 16   : value field 0 (ptr)   ← first pointer capture, if any
/// offset 16+8 : value field 1 (ptr)   ← additional pointer captures
/// ...
/// offset hdr + K*8 : code_hash        (32 bytes)
/// offset       + 32 : n_captures      (u32)
/// offset       + 36 : _pad            (u32)
/// offset       + 40 : non-pointer capture 0 (u64)
/// offset       + 48 : non-pointer capture 1 (u64)
/// ...
/// ```
///
/// where K is the count of pointer captures (which sit in
/// `TypeInfo::value_field_count` slots so the GC traces them) and the
/// remaining captures live in `raw_byte_count` bytes after `_pad`.
///
/// `n_captures` is the total count (pointer + non-pointer).
#[repr(C)]
pub struct ClosureRaw {
    pub code_hash: [u8; 32],
    pub n_captures: u32,
    pub _pad: u32,
    // followed by non-pointer captures (u64 each) up to TypeInfo's raw_byte_count
}

/// Byte offsets within a closure's raw-bytes section
/// (i.e., relative to `header_size + value_field_count * 8`).
pub mod closure_offsets {
    pub const CODE_HASH: usize = 0;
    pub const N_CAPTURES: usize = 32;
    pub const NON_POINTER_CAPTURES: usize = 40;
}

/// Canonical content hash for the runtime-managed `BoxedInt` shape.
///
/// `BoxedInt` is not user-authored; it's how the uniform closure ABI
/// represents an `Int` when it has to fit in a generic / TypeVar slot.
/// To ship a BoxedInt across the wire we need it registered as a real
/// shape — which means it needs a stable, content-addressable hash.
///
/// We synthesize one by Blake3-hashing a fixed sentinel string. Two
/// runtimes built from any sources agree on this hash, and a real
/// user-authored def can't collide with it (Blake3 of our sentinel is
/// vanishingly unlikely to match Blake3 of any canonical AST encoding).
pub fn boxed_int_shape_hash() -> Hash {
    Hash::of_bytes(b"<runtime:BoxedInt>")
}

/// Canonical wire-shape hash for the heap `String` (varlen-bytes) shape.
/// Lets the wire codec recognize a String pointer and ship it. Part of
/// the wire format.
pub fn string_shape_hash() -> Hash {
    Hash::of_bytes(b"<runtime:String>")
}

/// Canonical wire-shape hash for the heap `Array` (varlen pointer slots)
/// shape. Part of the wire format.
pub fn array_shape_hash() -> Hash {
    Hash::of_bytes(b"<runtime:Array>")
}

/// Canonical shape hash for the heap `Atom` shape: a single GC-traced
/// pointer cell, atomically updated. Distinct from `Array` so a shared
/// mutable cell is recognizable as such (by the GC, reflection, and any
/// wire/`Any` boundary) rather than masquerading as a 1-element array.
pub fn atom_shape_hash() -> Hash {
    Hash::of_bytes(b"<runtime:Atom>")
}

const _: () = {
    assert!(core::mem::offset_of!(ClosureRaw, code_hash) == closure_offsets::CODE_HASH);
    assert!(core::mem::offset_of!(ClosureRaw, n_captures) == closure_offsets::N_CAPTURES);
    assert!(core::mem::size_of::<ClosureRaw>() == closure_offsets::NON_POINTER_CAPTURES);
};

// =============================================================================
// CodeTable — hash → JIT fn ptr lookup
// =============================================================================

/// Maps a content-addressed def hash to its JIT'd entry point.
/// Built once at JIT-init time; queried on every closure indirect call.
///
/// Thread-safe so multiple threads (future) can call closures concurrently.
pub struct CodeTable {
    table: Mutex<HashMap<Hash, *const u8>>,
    /// type_id → shape hash for closure shapes. Populated when
    /// closure shapes are registered; consulted by indirect-call
    /// lookups (`ai_gc_lookup_code`) which read the type_id from a
    /// closure's header (a fixed offset) and resolve the actual
    /// code_hash through this table. Avoids the JIT having to
    /// compute a variable code_hash offset that depends on the
    /// closure's pointer-capture count.
    type_id_to_hash: Mutex<Vec<Option<Hash>>>,
}

// The raw `*const u8` is a JIT'd function pointer — process-local but
// not tied to any one thread. Safe to share read-only.
unsafe impl Send for CodeTable {}
unsafe impl Sync for CodeTable {}

impl CodeTable {
    pub fn new() -> Self {
        CodeTable {
            table: Mutex::new(HashMap::new()),
            type_id_to_hash: Mutex::new(Vec::new()),
        }
    }

    pub fn insert(&self, hash: Hash, fn_ptr: *const u8) {
        self.table.lock().unwrap().insert(hash, fn_ptr);
    }

    pub fn lookup(&self, hash: &Hash) -> Option<*const u8> {
        self.table.lock().unwrap().get(hash).copied()
    }

    /// Record that closure shapes with `type_id` correspond to the
    /// given content hash. Idempotent — overwrites if called again
    /// for the same id.
    pub fn register_type_id(&self, type_id: u16, hash: Hash) {
        let mut tab = self.type_id_to_hash.lock().unwrap();
        while tab.len() <= type_id as usize {
            tab.push(None);
        }
        tab[type_id as usize] = Some(hash);
    }

    pub fn shape_hash_for_type_id(&self, type_id: u16) -> Option<Hash> {
        let tab = self.type_id_to_hash.lock().unwrap();
        tab.get(type_id as usize).copied().flatten()
    }
}

impl Default for CodeTable {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// JIT-side shadow-stack walker
// =============================================================================

/// Walk a JIT-side shadow-stack chain rooted at `top`, invoking
/// `visitor` for each root slot (the *address* of the slot — so the
/// GC can update it in place after relocation).
///
/// Registered with the dynalloc `Heap` via `set_jit_frame_walker`.
/// dynalloc invokes this with `parked_jit_fp`, which we set to be
/// the head of our JIT frame chain before each allocation.
///
/// # Safety
///
/// `top` must point to a valid `Frame` (or be null). Each frame in
/// the chain must have a valid `origin` pointing to a `FrameOrigin`
/// whose `num_roots` correctly describes the number of trailing
/// `*mut u8` slots.
pub unsafe fn walk_jit_frames(
    top: *const u8,
    visitor: &mut dyn FnMut(*mut u64),
) {
    let mut frame = top as *mut Frame;
    while !frame.is_null() {
        unsafe {
            let header = &*frame;
            if header.origin.is_null() {
                // No origin means this frame isn't yet fully linked
                // (between alloca-and-memset and the origin store in
                // the prologue). Treat as zero roots.
                frame = header.parent;
                continue;
            }
            let origin = &*header.origin;
            let roots_start = (frame as *mut u8).add(frame_offsets::ROOTS) as *mut u64;
            for i in 0..origin.num_roots as usize {
                let slot = roots_start.add(i);
                visitor(slot);
            }
            frame = header.parent;
        }
    }
}

// =============================================================================
// Runtime: top-level container
// =============================================================================

/// Bundles together everything a JIT'd program needs at runtime: the
/// heap, the per-thread state, the code-lookup table, and the JIT-side
/// `Thread` struct that JIT'd code reads.
///
/// Single-threaded for v1 — one `Runtime` corresponds to one mutator.
///
/// Memory ownership: the `Runtime` owns the `Heap`, `ThreadState`,
/// `CodeTable`, and `Thread`. JIT'd code receives only raw pointers
/// into these; the `Runtime` must outlive any JIT execution.
pub struct Runtime {
    pub heap: Arc<Heap>,
    pub dyna_thread: Arc<ThreadState>,
    pub code_table: Box<CodeTable>,
    /// Boxed so `Thread`'s address is stable across moves of `Runtime`.
    pub thread: Box<Thread>,
    /// The OS thread the `Runtime` (and its `thread`/`dyna_thread`) was
    /// built on. Code running on this thread reuses the existing
    /// `Thread`; code on any *other* OS thread must build its own via
    /// [`Runtime::new_thread_context`] so concurrent JIT execution does
    /// not corrupt the shared shadow-stack head.
    pub home_thread: std::thread::ThreadId,
    /// Boxed copies of each registered TypeInfo so their addresses are
    /// stable. The wire deserializer needs to pass these addresses to
    /// `ai_gc_alloc_closure` to allocate the right shape; we look them
    /// up by hash via the shape registry.
    pub type_infos: Vec<Box<TypeInfo>>,
    /// Maps a content hash (lambda code hash / struct hash / variant
    /// hash) to the index of its `TypeInfo` in `type_infos`.
    pub shape_registry: HashMap<Hash, u16>,

    /// Layout metadata per shape hash. Used by the wire encoder/decoder.
    pub shape_meta: HashMap<Hash, ShapeMeta>,

    /// `shape_by_type_id[type_id]` = the shape hash for that TypeInfo
    /// (or `None` for type_ids we didn't register). Encoder uses this
    /// to identify any pointer it pulls from a frame slot or field.
    pub shape_by_type_id: Vec<Option<Hash>>,

    /// Stable `TypeInfo` for the `BoxedInt` heap shape. Allocated by
    /// `ai_gc_box_int` when generic code boxes an `Int`. Layout: GC
    /// header (16B) + raw i64 (8B), so `value_field_count = 0` and
    /// `raw_byte_count = 8`.
    pub boxed_int_ti: Box<TypeInfo>,
    /// Stable storage for the heap-resident `String` shape's TypeInfo.
    /// Heap layout: header (16 B) + varlen-bytes (8 B count + raw
    /// UTF-8 bytes). Also backs `Bytes`.
    pub string_ti: Box<TypeInfo>,
    /// Stable storage for the `Array` shape's TypeInfo. Heap layout:
    /// header (16 B) + varlen-Values (8 B count + N×8 B pointer slots).
    pub array_ti: Box<TypeInfo>,
    /// Stable storage for the `Atom` shape's TypeInfo. Heap layout:
    /// header (16 B) + one GC-traced pointer slot (8 B). A distinct shape
    /// from `Array`; `ai_atom_new` allocates it and the CAS in
    /// `ai_atom_swap_local` runs on that single slot.
    pub atom_ti: Box<TypeInfo>,

    /// Server-side memoization cache: `blake3(call_payload) → reply
    /// frame body`. The key covers the closure's code hash AND its
    /// captures (the entire encoded Call payload), so two calls with
    /// the same code and the same captures hit the cache.
    ///
    /// This is the headline property of content-addressed
    /// distributed computing — repeated work is free without any
    /// language-level memoization annotation, because identical code
    /// + identical inputs deterministically produce identical hashes.
    ///
    /// `cache_hits` / `cache_misses` count consults via
    /// `try_cached_result` / `store_cached_result` so tests can
    /// verify the cache is doing its job.
    pub result_cache: Arc<Mutex<HashMap<Hash, Vec<u8>>>>,
    pub cache_hits: Arc<std::sync::atomic::AtomicUsize>,
    pub cache_misses: Arc<std::sync::atomic::AtomicUsize>,

    /// Lambda/def hashes whose bodies transitively touch a node `state`
    /// cell. The `at()` result cache memoizes by payload bytes, which is
    /// sound only for pure thunks; a stateful thunk must bypass the cache
    /// (else a repeated identical call would skip its mutation). Populated
    /// at JIT install (`Jit::new` / `IncrementalJit::install`) from
    /// `knowledge::stateful_hashes`; consulted by `serve_one`.
    pub stateful_hashes: Arc<Mutex<std::collections::HashSet<Hash>>>,
}

impl Runtime {
    /// Build a runtime with the given closure-shape type table.
    ///
    /// `closure_types` is the list of `TypeInfo`s registered for
    /// closure shapes; one per distinct lambda in the JIT'd module.
    /// The index in this list becomes the `type_id` JIT'd code passes
    /// to `ai_gc_alloc_closure`.
    pub fn new(closure_types: Vec<TypeInfo>) -> Self {
        Self::new_with_metadata(closure_types, HashMap::new(), HashMap::new(), Vec::new())
    }

    /// Build a runtime with the registry only — convenience for callers
    /// that don't need full shape metadata (the wire codec needs it,
    /// JIT-only tests don't).
    pub fn new_with_registry(
        closure_types: Vec<TypeInfo>,
        shape_registry: HashMap<Hash, u16>,
    ) -> Self {
        Self::new_with_metadata(closure_types, shape_registry, HashMap::new(), Vec::new())
    }

    /// Build a runtime with the full set of metadata produced by
    /// `CompiledModule::build`. This is what the wire encoder/decoder
    /// needs in order to walk heap values.
    pub fn new_with_metadata(
        closure_types: Vec<TypeInfo>,
        mut shape_registry: HashMap<Hash, u16>,
        mut shape_meta: HashMap<Hash, ShapeMeta>,
        mut shape_by_type_id: Vec<Option<Hash>>,
    ) -> Self {
        // Reserve distinct type_ids for runtime-managed shapes at the
        // END of the closure-types table so they don't collide with
        // any module's shapes.
        //
        // BoxedInt: 0 value fields, 8 raw bytes (one i64).
        // String:   0 value fields, 0 raw bytes, varlen-bytes section.
        // Array:    0 fixed fields, varlen-Values section (pointer slots).
        // Atom:     1 fixed GC-traced pointer field (the mutable cell).
        // These four are the RUNTIME_RESERVED_SHAPES; the count MUST
        // match `RUNTIME_RESERVED_SHAPES` in IncrementalJit so dynamic
        // installs append after this block.
        let boxed_int_type_id = closure_types.len() as u16;
        let string_type_id = boxed_int_type_id + 1;
        let array_type_id = string_type_id + 1;
        let atom_type_id = array_type_id + 1;
        let boxed_int = TypeInfo::for_header(crate::gc::Full::SIZE as usize)
            .with_type_id(boxed_int_type_id)
            .with_fields(0)
            .with_raw_bytes(8);
        let string_ti_val = TypeInfo::for_header(crate::gc::Full::SIZE as usize)
            .with_type_id(string_type_id)
            .with_fields(0)
            .with_varlen_bytes(0);
        let array_ti_val = TypeInfo::for_header(crate::gc::Full::SIZE as usize)
            .with_type_id(array_type_id)
            .with_varlen_values(0);
        // Atom: exactly one GC-traced pointer slot (the current value).
        // A dedicated shape, NOT a 1-slot Array, so the runtime/GC/wire
        // layer can recognize a shared mutable cell as such.
        let atom_ti_val = TypeInfo::for_header(crate::gc::Full::SIZE as usize)
            .with_type_id(atom_type_id)
            .with_fields(1);

        // Register BoxedInt as a synthetic wire shape so the wire
        // encoder/decoder can ship pointers to BoxedInt across nodes.
        // This is what makes `at(node, || some_int_returning_thunk())`
        // work under the "always ship as heap" return path: the
        // closure's lambda boxes its Int return into a BoxedInt
        // (uniform ABI), we ship the BoxedInt pointer, the receiver
        // decodes into a fresh BoxedInt on its own heap, and the
        // user's `Ok(v)` match arm unboxes back to Int.
        let bi_hash = boxed_int_shape_hash();
        shape_registry.insert(bi_hash, boxed_int_type_id);
        while shape_by_type_id.len() <= atom_type_id as usize {
            shape_by_type_id.push(None);
        }
        shape_by_type_id[boxed_int_type_id as usize] = Some(bi_hash);
        // Register the String and Array shapes so the wire codec can ship
        // them. They have no `shape_meta` entry — the encoder special-cases
        // these two canonical hashes and reads the varlen heap layout
        // directly; the decoder reconstructs from the kind tag. Immutable
        // container values (e.g. a persistent HAMT, which holds
        // `Array<HNode>` and `String` keys) must all be wire-portable.
        shape_by_type_id[string_type_id as usize] = Some(string_shape_hash());
        shape_by_type_id[array_type_id as usize] = Some(array_shape_hash());
        shape_by_type_id[atom_type_id as usize] = Some(atom_shape_hash());
        shape_meta.insert(
            bi_hash,
            ShapeMeta::Struct {
                struct_ref: bi_hash,
                fields: vec![FieldMeta {
                    offset: crate::gc::Full::SIZE as u32,
                    is_pointer: false,
                }],
            },
        );

        let mut all_types = closure_types.clone();
        all_types.push(boxed_int);
        all_types.push(string_ti_val);
        all_types.push(array_ti_val);
        all_types.push(atom_ti_val);

        // Boxed copies for stable addresses (used by the deserializer
        // when it needs to pass a `*const TypeInfo` to `ai_gc_alloc_closure`).
        // BoxedInt + String sit at the same trailing slots as in the
        // heap's type-table so subsequent dynamic installs append into
        // both tables in lockstep.
        let mut type_infos: Vec<Box<TypeInfo>> =
            closure_types.iter().map(|t| Box::new(*t)).collect();
        let boxed_int_ti_box: Box<TypeInfo> = Box::new(boxed_int);
        let string_ti_box: Box<TypeInfo> = Box::new(string_ti_val);
        let array_ti_box: Box<TypeInfo> = Box::new(array_ti_val);
        let atom_ti_box: Box<TypeInfo> = Box::new(atom_ti_val);
        type_infos.push(Box::new(boxed_int));
        type_infos.push(Box::new(string_ti_val));
        type_infos.push(Box::new(array_ti_val));
        type_infos.push(Box::new(atom_ti_val));

        // 32 MiB semi-space — plenty for tests, easily reconfigurable later.
        let heap = Arc::new(Heap::new::<Full>(32 * 1024 * 1024, all_types));
        heap.set_jit_frame_walker(walk_jit_frames);

        let (dyna_thread, _id) = heap.register_thread();
        let code_table = Box::new(CodeTable::new());

        // Mirror shape_by_type_id into the code table so
        // `ai_gc_lookup_code` can resolve a closure's code_hash from
        // its header type_id without traversing the closure body
        // (whose code_hash sits at a variable offset).
        for (type_id, hash_opt) in shape_by_type_id.iter().enumerate() {
            if let Some(hash) = hash_opt {
                code_table.register_type_id(type_id as u16, *hash);
            }
        }

        let mut thread = Box::new(Thread {
            state: 0,
            _pad: [0; 7],
            top_frame: core::ptr::null_mut(),
            heap: Arc::as_ptr(&heap) as *mut Heap,
            code_table: &*code_table,
            dyna_thread: &*dyna_thread,
            boxed_int_ti: &*boxed_int_ti_box,
            string_ti: &*string_ti_box,
            array_ti: &*array_ti_box,
            atom_ti: &*atom_ti_box,
        });

        let _ = &mut *thread;

        // Point the GC coordinator's poll flag at this Thread's `state`
        // byte so a STW request can stop us even mid-JIT-loop. The
        // address is stable (Thread is boxed).
        dyna_thread.set_poll_flag(&mut thread.state as *mut u8);

        Runtime {
            heap,
            dyna_thread,
            code_table,
            thread,
            home_thread: std::thread::current().id(),
            type_infos,
            shape_registry,
            shape_meta,
            shape_by_type_id,
            boxed_int_ti: boxed_int_ti_box,
            string_ti: string_ti_box,
            array_ti: array_ti_box,
            atom_ti: atom_ti_box,
            result_cache: Arc::new(Mutex::new(HashMap::new())),
            cache_hits: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            cache_misses: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            stateful_hashes: Arc::new(Mutex::new(std::collections::HashSet::new())),
        }
    }

    /// Mark a def/lambda hash as stateful (its thunk must bypass the
    /// `at()` result cache). Idempotent.
    pub fn mark_stateful(&self, hash: Hash) {
        self.stateful_hashes
            .lock()
            .expect("stateful_hashes poisoned")
            .insert(hash);
    }

    /// Whether a shipped closure's lambda hash is stateful (cache-unsafe).
    pub fn is_stateful(&self, hash: &Hash) -> bool {
        self.stateful_hashes
            .lock()
            .expect("stateful_hashes poisoned")
            .contains(hash)
    }

    /// Look up a cached result frame by the call-payload hash.
    /// Increments `cache_hits` on hit, `cache_misses` on miss.
    pub fn try_cached_result(&self, key: &Hash) -> Option<Vec<u8>> {
        let guard = self.result_cache.lock().expect("result cache poisoned");
        if let Some(bytes) = guard.get(key) {
            self.cache_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(bytes.clone())
        } else {
            self.cache_misses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }

    /// Store a reply frame in the cache under the given key.
    pub fn store_cached_result(&self, key: Hash, reply: Vec<u8>) {
        let mut guard = self.result_cache.lock().expect("result cache poisoned");
        guard.insert(key, reply);
    }

    /// Look up a `TypeInfo` by content hash. Returns a stable pointer
    /// suitable for passing to `ai_gc_alloc_closure`.
    pub fn type_info_for(&self, hash: &Hash) -> Option<*const TypeInfo> {
        let idx = *self.shape_registry.get(hash)? as usize;
        let ti: &TypeInfo = &self.type_infos[idx];
        Some(ti as *const TypeInfo)
    }

    /// Raw pointer to the JIT-visible `Thread`. JIT'd code receives
    /// this as its first parameter.
    pub fn thread_ptr(&self) -> *mut Thread {
        &*self.thread as *const Thread as *mut Thread
    }

    /// Build a fresh execution context for a *different* OS thread.
    ///
    /// Registers a new GC [`ThreadState`] with the shared heap and
    /// builds a [`Thread`] that shares this runtime's heap, code table,
    /// and runtime-shape `TypeInfo`s (all stable + `Sync`) but has its
    /// own shadow-stack head (`top_frame`) and its own `dyna_thread`.
    ///
    /// Each OS thread that invokes JIT'd code MUST run with its own
    /// context: the shadow-stack head lives inside `Thread`, so two
    /// threads sharing one `Thread` clobber each other's GC root chain.
    /// The returned [`ThreadContext`] deregisters itself on drop, so it
    /// must not outlive this `Runtime` (it borrows the runtime's stable
    /// pointers by value).
    pub fn new_thread_context(&self) -> ThreadContext {
        let (dyna_thread, _id) = self.heap.register_thread();
        let thread = Box::new(Thread {
            state: 0,
            _pad: [0; 7],
            top_frame: core::ptr::null_mut(),
            // Shared, stable pointers — copied from the home Thread.
            heap: self.thread.heap,
            code_table: self.thread.code_table,
            // This thread's own GC coordination state.
            dyna_thread: &*dyna_thread,
            boxed_int_ti: self.thread.boxed_int_ti,
            string_ti: self.thread.string_ti,
            array_ti: self.thread.array_ti,
            atom_ti: self.thread.atom_ti,
        });
        // Wire the coordinator's poll flag to this context's own `state`
        // byte (stable: boxed) so a STW request can park this thread even
        // while it spins in an allocation-free JIT loop.
        dyna_thread.set_poll_flag(&thread.state as *const u8 as *mut u8);
        ThreadContext {
            thread,
            dyna_thread,
            heap: self.heap.clone(),
        }
    }
}

/// A per-OS-thread JIT execution context.
///
/// Holds one OS thread's own [`Thread`] (with its own shadow-stack head)
/// and its own registered GC [`ThreadState`], both derived from a
/// [`Runtime`] via [`Runtime::new_thread_context`]. On drop it
/// deregisters the `ThreadState` from the heap so a later collection
/// won't try to scan this thread's now-dead frame chain.
///
/// `Send` because moving the context to the OS thread that will use it
/// is exactly the supported pattern; the raw pointers it carries all
/// reference data owned by the originating `Runtime`, which must outlive
/// the context (same contract as the network server threads).
pub struct ThreadContext {
    thread: Box<Thread>,
    dyna_thread: Arc<ThreadState>,
    /// Kept so the heap (and thus the registered `ThreadState`) outlives
    /// this context, and so `Drop` can deregister.
    heap: Arc<Heap>,
}

unsafe impl Send for ThreadContext {}

impl ThreadContext {
    /// Raw pointer to this thread's `Thread`, to pass as the first
    /// argument of any JIT'd function invoked on this OS thread.
    pub fn thread_ptr(&self) -> *mut Thread {
        &*self.thread as *const Thread as *mut Thread
    }

    /// This context's GC coordination state (for safepoint / blocked
    /// transitions around blocking calls).
    pub fn dyna_thread(&self) -> &Arc<ThreadState> {
        &self.dyna_thread
    }
}

impl Drop for ThreadContext {
    fn drop(&mut self) {
        self.heap.safe_deregister_thread(&self.dyna_thread);
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // Cleanly deregister the thread so the Heap doesn't try to
        // scan its (now-dangling) frame chain on a future collection.
        self.heap.safe_deregister_thread(&self.dyna_thread);
    }
}

// =============================================================================
// Extern "C" runtime functions exposed to JIT'd code
// =============================================================================

/// Allocate a closure (or any object whose TypeInfo lives at the
/// given pointer). The pointer returned is to the start of the GC
/// header — JIT'd code stores `code_hash`, captures, etc. at the
/// known offsets immediately after.
///
/// Sets `dyna_thread.parked_jit_fp` to the current `top_frame` for
/// the duration of the alloc so a GC triggered inside `Heap::alloc_obj`
/// walks our shadow-stack chain via the registered walker.
///
/// # Safety
///
/// Called from JIT'd code with a valid `Thread*` and `TypeInfo*`
/// (both live for the duration of this call).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_gc_alloc_closure(
    thread: *mut Thread,
    type_info: *const TypeInfo,
) -> *mut u8 {
    unsafe {
        let t = &*thread;
        let heap = &*t.heap;
        let ti = &*type_info;
        let dyna = &*t.dyna_thread;

        // Publish our chain head so a GC during alloc walks it.
        dyna.set_parked_jit_fp(t.top_frame as *const u8);
        let ptr = heap.alloc_obj::<Full>(ti, 0);
        if !ptr.is_null() {
            dyna.clear_parked_jit_fp();
            return ptr;
        }

        // Heap is full — run a collection (with parked_jit_fp still
        // pointed at our chain so roots get scanned) and retry once.
        heap.collect::<crate::gc::IdentityPtrPolicy>(&[]);
        let ptr = heap.alloc_obj::<Full>(ti, 0);
        dyna.clear_parked_jit_fp();
        if ptr.is_null() {
            panic!(
                "ai_gc_alloc_closure: heap exhausted after GC \
                 (object size > available space?)"
            );
        }
        ptr
    }
}

/// Look up a JIT'd function by its content hash. Used for closure
/// indirect calls — given a closure's `code_hash` (32 bytes pointed
/// to by `hash_ptr`), returns the JIT'd entry point.
///
/// # Safety
///
/// `thread` must be a valid `Thread*`. `hash_ptr` must point to a
/// Resolve a closure's JIT'd entry point given the closure pointer.
///
/// Reads the `type_id` from the closure's GC header, looks up the
/// shape hash via the thread's heap (which is also the closure's
/// code_hash because closure shapes are content-addressed), then
/// looks up the fn pointer in the code table.
///
/// Indirect calls invoke this rather than computing `closure + 16`
/// directly: with Phase 1B pointer captures, the closure's
/// code_hash sits at a variable offset (`Full::SIZE + ptr_count * 8`)
/// that the JIT call site can't compute statically. type_id-based
/// resolution sidesteps the offset entirely.
///
/// Panics if the closure's type_id isn't a registered shape or
/// if the resulting hash has no code table entry.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_gc_lookup_code(
    thread: *mut Thread,
    closure_ptr: *const u8,
) -> *const u8 {
    unsafe {
        let t = &*thread;
        let table = &*t.code_table;
        // type_id at the GC header's TYPE_ID_OFFSET.
        let type_id_off = <Full as crate::gc::ObjHeader>::TYPE_ID_OFFSET;
        let type_id = *(closure_ptr.add(type_id_off) as *const u16);
        let hash = table.shape_hash_for_type_id(type_id).unwrap_or_else(|| {
            panic!(
                "ai_gc_lookup_code: type_id {} not registered as a shape",
                type_id
            )
        });
        match table.lookup(&hash) {
            Some(p) => p,
            None => panic!(
                "ai_gc_lookup_code: hash {} (type_id {}) not registered in code table",
                hash, type_id
            ),
        }
    }
}

/// Box an `i64` into a `BoxedInt` heap object so it can be stored in
/// a generic-typed slot. Reads the `BoxedInt` TypeInfo from
/// `thread.boxed_int_ti` (set up by `Runtime::new_with_metadata`).
///
/// # Safety
/// `thread` must be a valid `Thread*` with `boxed_int_ti` initialised.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_gc_box_int(thread: *mut Thread, value: i64) -> *mut u8 {
    unsafe {
        let t = &*thread;
        let ti = t.boxed_int_ti;
        if ti.is_null() {
            panic!("ai_gc_box_int: thread.boxed_int_ti is null");
        }
        let heap = &*t.heap;
        let dyna = &*t.dyna_thread;
        dyna.set_parked_jit_fp(t.top_frame as *const u8);
        let mut ptr = heap.alloc_obj::<Full>(&*ti, 0);
        if ptr.is_null() {
            heap.collect::<crate::gc::IdentityPtrPolicy>(&[]);
            ptr = heap.alloc_obj::<Full>(&*ti, 0);
            if ptr.is_null() {
                dyna.clear_parked_jit_fp();
                panic!("ai_gc_box_int: heap exhausted after GC");
            }
        }
        dyna.clear_parked_jit_fp();
        // Write the i64 at offset Full::SIZE.
        let value_slot = ptr.add(<Full as crate::gc::ObjHeader>::SIZE) as *mut i64;
        *value_slot = value;
        ptr
    }
}

/// Unbox an `i64` from a `BoxedInt` heap object. JIT'd code calls this
/// when extracting a generic-typed payload whose instantiated type is
/// `Int`. No allocation, no GC, no thread argument needed.
///
/// # Safety
/// `ptr` must be a pointer to a `BoxedInt` heap object (as produced
/// by `ai_gc_box_int`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_gc_unbox_int(ptr: *const u8) -> i64 {
    unsafe {
        let value_slot = ptr.add(<Full as crate::gc::ObjHeader>::SIZE) as *const i64;
        *value_slot
    }
}

/// Print a panic message to stderr and abort the process.
///
/// `msg` is a heap `String` pointer (the same shape `ai_str_new`
/// produces). This is the single hard-error path for the language:
/// the user-visible `panic("...")` builtin lowers to it, and codegen
/// fallthroughs that were previously `build_unreachable` (undefined
/// behaviour — e.g. a non-exhaustive match hitting an unhandled
/// variant at runtime) now call it instead so the failure is a clear
/// message rather than silent corruption.
///
/// Never returns: aborts via `std::process::abort()` so the process
/// dies with a recognizable signal rather than unwinding through
/// JIT'd frames the Rust runtime can't unwind.
///
/// # Safety
/// `msg` must be null or a valid heap `String` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_panic(_thread: *mut Thread, msg: *const u8) -> ! {
    let text = if msg.is_null() {
        "(no message)".to_owned()
    } else {
        unsafe { crate::ffi::heap_str_to_owned(msg) }
    };
    eprintln!("ai-lang panic: {}", text);
    std::process::abort();
}

/// Allocate a heap String containing the bytes at `[src..src+len]`.
/// Used for string literal codegen and as the underlying allocator
/// for `ai_str_concat`.
///
/// The heap layout matches `Runtime.string_ti`: GC header + varlen
/// bytes (count followed by raw payload). The varlen count is `len`;
/// payload is a `memcpy` of `len` bytes from `src`.
///
/// # Safety
/// `src` must be readable for `len` bytes. `thread` must have
/// `string_ti` initialised (it is, by `Runtime::new_with_metadata`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_new(
    thread: *mut Thread,
    src: *const u8,
    len: i64,
) -> *mut u8 {
    unsafe {
        let t = &*thread;
        let ti = t.string_ti;
        if ti.is_null() {
            panic!("ai_str_new: thread.string_ti is null");
        }
        let heap = &*t.heap;
        let dyna = &*t.dyna_thread;
        let varlen = len as usize;
        dyna.set_parked_jit_fp(t.top_frame as *const u8);
        let mut ptr = heap.alloc_obj::<Full>(&*ti, varlen);
        if ptr.is_null() {
            heap.collect::<crate::gc::IdentityPtrPolicy>(&[]);
            ptr = heap.alloc_obj::<Full>(&*ti, varlen);
            if ptr.is_null() {
                dyna.clear_parked_jit_fp();
                panic!("ai_str_new: heap exhausted after GC");
            }
        }
        dyna.clear_parked_jit_fp();
        // alloc_obj initializes the varlen count word; we only need
        // to copy the bytes into the varlen payload section.
        let payload_off = (*ti).varlen_element_offset(0);
        if varlen > 0 {
            core::ptr::copy_nonoverlapping(src, ptr.add(payload_off), varlen);
        }
        ptr
    }
}

/// Length (in bytes) of a heap `String`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_len(s: *const u8) -> i64 {
    if s.is_null() {
        return 0;
    }
    unsafe {
        let count_off = <Full as crate::gc::ObjHeader>::SIZE;
        let count_ptr = s.add(count_off) as *const u64;
        *count_ptr as i64
    }
}

/// Byte-wise equality of two heap `String`s. Returns 1 if equal,
/// 0 otherwise. Same-length + same-bytes; encoding-agnostic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_eq(a: *const u8, b: *const u8) -> i64 {
    unsafe {
        let len_a = ai_str_len(a);
        let len_b = ai_str_len(b);
        if len_a != len_b {
            return 0;
        }
        if len_a == 0 {
            return 1;
        }
        let count_off = <Full as crate::gc::ObjHeader>::SIZE;
        // Payload sits after the count word (8 bytes), per varlen layout.
        let pa = a.add(count_off + 8);
        let pb = b.add(count_off + 8);
        let na = len_a as usize;
        for i in 0..na {
            if *pa.add(i) != *pb.add(i) {
                return 0;
            }
        }
        1
    }
}

/// Concatenate two heap `String`s into a fresh String. Allocates.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_str_concat(
    thread: *mut Thread,
    a: *const u8,
    b: *const u8,
) -> *mut u8 {
    unsafe {
        let len_a = ai_str_len(a) as usize;
        let len_b = ai_str_len(b) as usize;
        let total = len_a + len_b;
        let t = &*thread;
        let ti = t.string_ti;
        let heap = &*t.heap;
        let dyna = &*t.dyna_thread;
        dyna.set_parked_jit_fp(t.top_frame as *const u8);
        let mut ptr = heap.alloc_obj::<Full>(&*ti, total);
        if ptr.is_null() {
            heap.collect::<crate::gc::IdentityPtrPolicy>(&[]);
            ptr = heap.alloc_obj::<Full>(&*ti, total);
            if ptr.is_null() {
                dyna.clear_parked_jit_fp();
                panic!("ai_str_concat: heap exhausted after GC");
            }
        }
        dyna.clear_parked_jit_fp();
        let payload_off = (*ti).varlen_element_offset(0);
        let count_off = <Full as crate::gc::ObjHeader>::SIZE;
        if len_a > 0 {
            core::ptr::copy_nonoverlapping(
                a.add(count_off + 8),
                ptr.add(payload_off),
                len_a,
            );
        }
        if len_b > 0 {
            core::ptr::copy_nonoverlapping(
                b.add(count_off + 8),
                ptr.add(payload_off + len_a),
                len_b,
            );
        }
        ptr
    }
}

// =============================================================================
// Bytes runtime functions
//
// `Bytes` is a mutable, indexable byte buffer. It reuses the heap-resident
// `String` shape (`Runtime.string_ti`): GC header + varlen-byte section
// (8-byte count + raw payload). The two are byte-identical in memory; the
// distinction is purely at the type level (the typechecker keeps `String`
// and `Bytes` separate). Because the payload is raw bytes (no pointers),
// in-place mutation needs no write barrier.
// =============================================================================

/// Offset of the varlen payload (raw bytes) within a String/Bytes object:
/// past the GC header and the 8-byte varlen count word.
#[inline]
fn varlen_payload_offset() -> usize {
    <Full as crate::gc::ObjHeader>::SIZE + 8
}

/// Allocate a zero-filled `Bytes` of `len` bytes.
///
/// # Safety
/// `thread` must have `string_ti` initialised (it is, by
/// `Runtime::new_with_metadata`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_bytes_new(thread: *mut Thread, len: i64) -> *mut u8 {
    if len < 0 {
        panic!("ai_bytes_new: negative length {}", len);
    }
    unsafe {
        let t = &*thread;
        let ti = t.string_ti;
        if ti.is_null() {
            panic!("ai_bytes_new: thread.string_ti is null");
        }
        let heap = &*t.heap;
        let dyna = &*t.dyna_thread;
        let varlen = len as usize;
        dyna.set_parked_jit_fp(t.top_frame as *const u8);
        let mut ptr = heap.alloc_obj::<Full>(&*ti, varlen);
        if ptr.is_null() {
            heap.collect::<crate::gc::IdentityPtrPolicy>(&[]);
            ptr = heap.alloc_obj::<Full>(&*ti, varlen);
            if ptr.is_null() {
                dyna.clear_parked_jit_fp();
                panic!("ai_bytes_new: heap exhausted after GC");
            }
        }
        dyna.clear_parked_jit_fp();
        // alloc_obj initialises the varlen count word; zero the payload
        // so a fresh buffer reads back as all-zero bytes.
        if varlen > 0 {
            let payload = ptr.add(varlen_payload_offset());
            core::ptr::write_bytes(payload, 0u8, varlen);
        }
        ptr
    }
}

/// Length (in bytes) of a `Bytes`. Identical layout to `String`, so this
/// reads the same varlen count word as `ai_str_len`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_bytes_len(b: *const u8) -> i64 {
    unsafe { ai_str_len(b) }
}

/// Read the byte at index `i` (0..len), returned as an `Int` in 0..=255.
/// Panics on out-of-bounds access — a clear hard error beats silent UB.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_bytes_get(b: *const u8, i: i64) -> i64 {
    unsafe {
        let len = ai_str_len(b);
        if i < 0 || i >= len {
            panic!("ai_bytes_get: index {} out of bounds (len {})", i, len);
        }
        let payload = b.add(varlen_payload_offset());
        *payload.add(i as usize) as i64
    }
}

/// Write the low 8 bits of `v` to index `i` (0..len). Returns 0.
/// Panics on out-of-bounds access.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_bytes_set(b: *mut u8, i: i64, v: i64) -> i64 {
    unsafe {
        let len = ai_str_len(b);
        if i < 0 || i >= len {
            panic!("ai_bytes_set: index {} out of bounds (len {})", i, len);
        }
        let payload = b.add(varlen_payload_offset());
        *payload.add(i as usize) = (v & 0xff) as u8;
        0
    }
}

/// Allocate a fresh `Bytes` containing `[start..start+len]` of `src`.
/// Panics if the requested range is out of bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_bytes_slice(
    thread: *mut Thread,
    src: *const u8,
    start: i64,
    len: i64,
) -> *mut u8 {
    unsafe {
        let src_len = ai_str_len(src);
        if start < 0 || len < 0 || start + len > src_len {
            panic!(
                "ai_bytes_slice: range [{}, {}) out of bounds (len {})",
                start,
                start + len,
                src_len
            );
        }
        let src_payload = src.add(varlen_payload_offset() + start as usize);
        // ai_str_new allocates a varlen-bytes object and copies `len`
        // bytes from `src_payload` — exactly the slice we want.
        ai_str_new(thread, src_payload, len)
    }
}

/// Copy a `String`/`Bytes` into a fresh varlen-bytes object. Backs both
/// `bytes_from_string` and `string_from_bytes`: the representations are
/// identical, so the conversion is a defensive copy that gives the result
/// independent mutation semantics from the source.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_bytes_copy(thread: *mut Thread, src: *const u8) -> *mut u8 {
    unsafe {
        let len = ai_str_len(src);
        let src_payload = src.add(varlen_payload_offset());
        ai_str_new(thread, src_payload, len)
    }
}

// =============================================================================
// Array runtime functions
//
// `Array<T>` is a fixed-size, O(1)-indexable vector of GC-traced pointer
// slots (heap shape `Runtime.array_ti`, VarLenKind::Values). Elements use
// the uniform boxed representation: an element of type Int is a BoxedInt
// pointer (boxed/unboxed at the call site by codegen); any other element
// is a real heap pointer. The GC scanner walks every slot as a pointer, so
// `array_new` MUST zero-fill (IdentityPtrPolicy treats 0 as null/skip).
//
// No write barrier is needed: the collector is a single-space copying GC,
// so every collection traces all live objects from roots — storing a
// pointer into an existing array slot can't hide it from the next trace.
// =============================================================================

/// Offset of varlen Values element `i` (a pointer slot) in an Array:
/// past the GC header and the 8-byte count word, 8 bytes per element.
#[inline]
fn array_element_offset(i: usize) -> usize {
    <Full as crate::gc::ObjHeader>::SIZE + 8 + i * 8
}

/// Allocate a fixed-size `Array` of `n` null (zero) pointer slots.
///
/// # Safety
/// `thread` must have `array_ti` initialised (it is, by
/// `Runtime::new_with_metadata`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_array_new(thread: *mut Thread, n: i64) -> *mut u8 {
    if n < 0 {
        panic!("ai_array_new: negative length {}", n);
    }
    unsafe {
        let t = &*thread;
        let ti = t.array_ti;
        if ti.is_null() {
            panic!("ai_array_new: thread.array_ti is null");
        }
        let heap = &*t.heap;
        let dyna = &*t.dyna_thread;
        let count = n as usize;
        dyna.set_parked_jit_fp(t.top_frame as *const u8);
        let mut ptr = heap.alloc_obj::<Full>(&*ti, count);
        if ptr.is_null() {
            heap.collect::<crate::gc::IdentityPtrPolicy>(&[]);
            ptr = heap.alloc_obj::<Full>(&*ti, count);
            if ptr.is_null() {
                dyna.clear_parked_jit_fp();
                panic!("ai_array_new: heap exhausted after GC");
            }
        }
        dyna.clear_parked_jit_fp();
        // alloc_obj writes the varlen count word; zero the element slots
        // so the GC sees null (skipped) until the user stores pointers.
        if count > 0 {
            let base = ptr.add(array_element_offset(0));
            core::ptr::write_bytes(base, 0u8, count * 8);
        }
        ptr
    }
}

/// Number of slots in an `Array`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_array_len(a: *const u8) -> i64 {
    if a.is_null() {
        return 0;
    }
    unsafe {
        let count_off = <Full as crate::gc::ObjHeader>::SIZE;
        *(a.add(count_off) as *const u64) as i64
    }
}

/// Load the pointer in slot `i` (0..len). Panics on out-of-bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_array_get(a: *const u8, i: i64) -> *mut u8 {
    unsafe {
        let len = ai_array_len(a);
        if i < 0 || i >= len {
            panic!("ai_array_get: index {} out of bounds (len {})", i, len);
        }
        let slot = a.add(array_element_offset(i as usize)) as *const *mut u8;
        *slot
    }
}

/// Store pointer `p` into slot `i` (0..len). Returns 0. Panics on
/// out-of-bounds. No write barrier (single-space copying GC).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_array_set(a: *mut u8, i: i64, p: *mut u8) -> i64 {
    unsafe {
        let len = ai_array_len(a);
        if i < 0 || i >= len {
            panic!("ai_array_set: index {} out of bounds (len {})", i, len);
        }
        let slot = a.add(array_element_offset(i as usize)) as *mut *mut u8;
        *slot = p;
        0
    }
}

/// Byte offset of an `Atom`'s single value slot. The cell has no count
/// word (unlike `Array`), so the value sits immediately after the header.
fn atom_value_offset() -> usize {
    <Full as crate::gc::ObjHeader>::SIZE
}

/// Allocate a fresh `Atom` cell holding `init`. The cell is a dedicated
/// heap shape (one GC-traced pointer slot), NOT a 1-element `Array`.
///
/// # Safety
/// `thread` must have `atom_ti` initialised (it is, by
/// `Runtime::new_with_metadata`). `init` must be a live heap pointer
/// (Int values are boxed by codegen before reaching here).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_atom_new(thread: *mut Thread, init: *mut u8) -> *mut u8 {
    unsafe {
        let t = &*thread;
        let ti = t.atom_ti;
        if ti.is_null() {
            panic!("ai_atom_new: thread.atom_ti is null");
        }
        let heap = &*t.heap;
        let dyna = &*t.dyna_thread;
        dyna.set_parked_jit_fp(t.top_frame as *const u8);
        let mut ptr = heap.alloc_obj::<Full>(&*ti, 0);
        if ptr.is_null() {
            heap.collect::<crate::gc::IdentityPtrPolicy>(&[]);
            ptr = heap.alloc_obj::<Full>(&*ti, 0);
            if ptr.is_null() {
                dyna.clear_parked_jit_fp();
                panic!("ai_atom_new: heap exhausted after GC");
            }
        }
        dyna.clear_parked_jit_fp();
        // Store the initial value with Release so a subsequent acquiring
        // reader (deref / swap) sees a fully-constructed cell.
        let slot = ptr.add(atom_value_offset()) as *const core::sync::atomic::AtomicPtr<u8>;
        (*slot).store(init, core::sync::atomic::Ordering::Release);
        ptr
    }
}

/// Read an `Atom`'s current value with `Acquire` ordering, establishing
/// happens-before with the `Release` swap that installed it.
///
/// # Safety
/// `atom` must be a live `Atom` cell (as produced by `ai_atom_new`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_atom_load(atom: *const u8) -> *mut u8 {
    unsafe {
        let slot = atom.add(atom_value_offset()) as *const core::sync::atomic::AtomicPtr<u8>;
        (*slot).load(core::sync::atomic::Ordering::Acquire)
    }
}

/// The atom primitive: a lock-free swap. This is the ONE irreducible
/// piece a real Clojure-style atom needs; everything else (`atom`,
/// `deref`, `reset`) is plain ai-lang.
///
/// `atom` is a dedicated `Atom` cell whose single slot holds a pointer to
/// the current immutable value. `updater` is a closure `fn(T) -> T`. The
/// loop:
///   1. atomically load the slot's current pointer
///   2. invoke `updater(current)` — runs WITHOUT any lock — to build a
///      brand-new value object
///   3. `compare_exchange` the slot from the old pointer to the new one
///   4. on success, return the new pointer; on failure (another swap
///      committed first), retry from step 1
///
/// No mutex, no generation counter: one hardware compare-exchange. It is
/// correct precisely because values are immutable — the slot only ever
/// changes which object it points at, and each swap installs a *fresh*
/// object (fresh identity), so pointer-identity CAS is sufficient (no ABA).
///
/// Works for ANY value type with no user thought: the closure does its
/// own Int box/unbox via the uniform ABI, so the loop only ever moves raw
/// object pointers. The slot is a GC-traced field, so both the old and
/// newly-installed values stay rooted and the GC scans them.
///
/// # Safety
/// `atom` must be a live `Atom` cell; `updater` must be a live closure of
/// LLVM signature
/// `unsafe extern "C" fn(*mut Thread, *const u8, *const u8) -> *mut u8`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_atom_swap_local(
    thread: *mut Thread,
    atom: *mut u8,
    updater: *const u8,
) -> *mut u8 {
    unsafe {
        let slot =
            atom.add(atom_value_offset()) as *const core::sync::atomic::AtomicPtr<u8>;

        // Resolve the closure's code pointer once (it doesn't change
        // across retries).
        let fn_ptr = ai_gc_lookup_code(thread, updater);
        let lambda: unsafe extern "C" fn(*mut Thread, *const u8, *const u8) -> *mut u8 =
            core::mem::transmute(fn_ptr);

        loop {
            let current = (*slot).load(core::sync::atomic::Ordering::Acquire);
            // Run the user's updater on the current value — no lock held.
            let new = lambda(thread, updater, current as *const u8);
            // Try to install it. Identity CAS: succeeds iff nobody moved
            // the slot since our load.
            match (*slot).compare_exchange(
                current,
                new,
                core::sync::atomic::Ordering::AcqRel,
                core::sync::atomic::Ordering::Acquire,
            ) {
                Ok(_) => return new,
                Err(_) => continue, // lost the race; recompute on the new value
            }
        }
    }
}

/// Read the 32-byte content hash a node-`state` primitive was given.
unsafe fn read_state_key(hash_ptr: *const u8) -> [u8; 32] {
    let mut key = [0u8; 32];
    unsafe { core::ptr::copy_nonoverlapping(hash_ptr, key.as_mut_ptr(), 32) };
    key
}

/// `1` if a node `state` binding identified by `hash_ptr` (a pointer to
/// 32 content-hash bytes) is already installed on this node's heap, else
/// `0`. The installer thunk uses this to make installation idempotent:
/// the initializer runs exactly once per node per hash.
///
/// # Safety
/// `thread.heap` must be valid; `hash_ptr` must point to 32 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_state_present(thread: *mut Thread, hash_ptr: *const u8) -> i64 {
    unsafe {
        let heap = &*(*thread).heap;
        let key = read_state_key(hash_ptr);
        let slots = heap.state_slots.lock().expect("state_slots poisoned");
        if slots.contains_key(&key) { 1 } else { 0 }
    }
}

/// Install a node `state` binding: store `val` (the initializer's result,
/// typically an `Atom` pointer) in a fresh GC-root slot and record
/// `hash -> slot`. Idempotent by hash: if the hash is already live this is
/// a no-op and the existing cell is preserved (so a shipped handler that
/// re-installs a state the node already has never clobbers it). Returns 0.
///
/// # Safety
/// `thread.heap` must be valid; `hash_ptr` must point to 32 readable bytes;
/// `val` must be a live heap pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_state_set(
    thread: *mut Thread,
    hash_ptr: *const u8,
    val: *mut u8,
) -> i64 {
    unsafe {
        let heap = &*(*thread).heap;
        let key = read_state_key(hash_ptr);
        let mut slots = heap.state_slots.lock().expect("state_slots poisoned");
        if slots.contains_key(&key) {
            return 0; // already live — keep the node's existing cell
        }
        let idx = heap.globals.add(val as u64);
        slots.insert(key, idx);
        0
    }
}

/// Resolve a node `state` reference to its single live cell on this node.
/// Reads the value through the GC-root slot, so the pointer reflects any
/// relocation since install. Panics with a clear message if the state was
/// never installed (a hard error, never a silent null).
///
/// # Safety
/// `thread.heap` must be valid; `hash_ptr` must point to 32 readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_state_get(thread: *mut Thread, hash_ptr: *const u8) -> *mut u8 {
    unsafe {
        let heap = &*(*thread).heap;
        let key = read_state_key(hash_ptr);
        let slots = heap.state_slots.lock().expect("state_slots poisoned");
        match slots.get(&key) {
            Some(&idx) => heap.globals.get(idx) as *mut u8,
            None => panic!(
                "ai_state_get: node `state` {} not installed on this node \
                 (its install thunk must run before any reference)",
                hex_of(&key),
            ),
        }
    }
}

/// Lowercase hex of a 32-byte hash, for diagnostics.
fn hex_of(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

// =============================================================================
// Structural value hashing + equality
// =============================================================================
//
// The premise of the language is that any value has a canonical form, so any
// value is hashable. These walk a heap value by its shape (the same
// `TypeInfo` the GC walks) and produce a stable structural hash / decide
// structural equality. Stable across GC relocation (content-based, not
// address-based) and across nodes (same structure → same hash). This is what
// lets `HashMap<K, V>` key on ANY type, not just `String`.
//
// Values are immutable and acyclic (built bottom-up), so the recursion
// terminates. The ONE mutable shape is `Atom`: hashing/comparing it by its
// current contents would be unstable (a later `swap` changes the hash), and a
// cycle can only form THROUGH an atom. So both walkers hard-error if they
// reach an `Atom` (identified by its reserved `type_id`), rather than return a
// silently-wrong-and-unstable answer. Key on `deref(atom)` or a stable id.

const FNV64_OFFSET: u64 = 14695981039346656037;
const FNV64_PRIME: u64 = 1099511628211;

#[inline]
fn fnv_byte(acc: u64, b: u8) -> u64 {
    (acc ^ b as u64).wrapping_mul(FNV64_PRIME)
}
#[inline]
fn fnv_u64(acc: u64, v: u64) -> u64 {
    let mut a = acc;
    for b in v.to_le_bytes() {
        a = fnv_byte(a, b);
    }
    a
}

/// Structurally hash the heap object at `obj`, folding into `acc`.
/// `atom_tid` is the reserved `Atom` type_id; reaching one is a hard error
/// (a mutable cell has no stable structural hash).
unsafe fn hash_obj(heap: &Heap, obj: *const u8, acc: u64, atom_tid: u16) -> u64 {
    unsafe {
        if obj.is_null() {
            return fnv_u64(acc, 0x4e_55_4c_4c); // "NULL"
        }
        let tid = heap.obj_type_id(obj);
        if tid == atom_tid {
            panic!(
                "value_hash: cannot hash an `Atom` (a mutable cell has no \
                 stable hash). Key on `deref(atom)` or a stable id instead."
            );
        }
        let info = heap.type_info_by_id(tid);
        // Fold the shape id so values of different shapes don't collide.
        let mut a = fnv_u64(acc, tid as u64);
        // Fixed GC-traced pointer fields: recurse.
        for i in 0..info.value_field_count {
            let off = info.value_field_offset(i);
            let p = *(obj.add(off) as *const *const u8);
            a = hash_obj(heap, p, a, atom_tid);
        }
        // Untraced raw bytes (e.g. a BoxedInt's i64, an enum tag).
        for &b in crate::gc::read_raw_bytes(obj, info) {
            a = fnv_byte(a, b);
        }
        // Variable-length tail.
        match info.varlen {
            crate::gc::VarLenKind::None => {}
            crate::gc::VarLenKind::Bytes => {
                let bytes = crate::gc::read_varlen_bytes(obj, info);
                a = fnv_u64(a, bytes.len() as u64);
                for &b in bytes {
                    a = fnv_byte(a, b);
                }
            }
            crate::gc::VarLenKind::Values => {
                let n = crate::gc::read_varlen_count(obj, info);
                a = fnv_u64(a, n as u64);
                for j in 0..n {
                    let off = info.varlen_element_offset(j);
                    let p = *(obj.add(off) as *const *const u8);
                    a = hash_obj(heap, p, a, atom_tid);
                }
            }
        }
        a
    }
}

/// Structural equality of two heap objects (same shape + same contents).
/// Reaching an `Atom` (reserved `atom_tid`) is a hard error.
unsafe fn eq_obj(heap: &Heap, a: *const u8, b: *const u8, atom_tid: u16) -> bool {
    unsafe {
        if a == b {
            return true; // same object (incl. both null)
        }
        if a.is_null() || b.is_null() {
            return false;
        }
        let ta = heap.obj_type_id(a);
        let tb = heap.obj_type_id(b);
        if ta == atom_tid || tb == atom_tid {
            panic!(
                "value_eq: cannot compare an `Atom` by value (a mutable cell \
                 has no stable structural equality). Compare `deref(atom)` or \
                 a stable id instead."
            );
        }
        if ta != tb {
            return false;
        }
        let info = heap.type_info_by_id(ta);
        for i in 0..info.value_field_count {
            let off = info.value_field_offset(i);
            let pa = *(a.add(off) as *const *const u8);
            let pb = *(b.add(off) as *const *const u8);
            if !eq_obj(heap, pa, pb, atom_tid) {
                return false;
            }
        }
        if crate::gc::read_raw_bytes(a, info) != crate::gc::read_raw_bytes(b, info) {
            return false;
        }
        match info.varlen {
            crate::gc::VarLenKind::None => {}
            crate::gc::VarLenKind::Bytes => {
                if crate::gc::read_varlen_bytes(a, info) != crate::gc::read_varlen_bytes(b, info) {
                    return false;
                }
            }
            crate::gc::VarLenKind::Values => {
                let na = crate::gc::read_varlen_count(a, info);
                let nb = crate::gc::read_varlen_count(b, info);
                if na != nb {
                    return false;
                }
                for j in 0..na {
                    let off = info.varlen_element_offset(j);
                    let pa = *(a.add(off) as *const *const u8);
                    let pb = *(b.add(off) as *const *const u8);
                    if !eq_obj(heap, pa, pb, atom_tid) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Stable structural hash of any value (i64). Lets `HashMap<K, V>` key on
/// any type. `v` is a heap pointer (Int/Float keys arrive boxed under the
/// uniform ABI; codegen boxes a bare Int before the call).
///
/// # Safety
/// `thread.heap` valid; `v` a live heap object (or null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_value_hash(thread: *mut Thread, v: *const u8) -> i64 {
    unsafe {
        let t = &*thread;
        let heap = &*t.heap;
        let atom_tid = (*t.atom_ti).type_id;
        hash_obj(heap, v, FNV64_OFFSET, atom_tid) as i64
    }
}

/// Structural equality of two values: `1` if equal, `0` otherwise.
///
/// # Safety
/// `thread.heap` valid; `a`/`b` live heap objects (or null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_value_eq(thread: *mut Thread, a: *const u8, b: *const u8) -> i64 {
    unsafe {
        let t = &*thread;
        let heap = &*t.heap;
        let atom_tid = (*t.atom_ti).type_id;
        if eq_obj(heap, a, b, atom_tid) { 1 } else { 0 }
    }
}

// =============================================================================
// OS threads: spawn / join
// =============================================================================
//
// `spawn(|| ...)` runs a zero-arg closure on a fresh OS thread that has its
// OWN execution context (own shadow-stack head + GC `ThreadState`), sharing
// the heap and code table with its parent. `join(handle)` blocks until the
// thread finishes and returns its result.
//
// The language-level handle is a `BoxedInt` holding an index into the
// process-global `THREAD_REGISTRY`. The cross-thread state — the OS
// `JoinHandle`, the input closure pointer, and the result pointer — lives in
// the registry, which is a GC root source so the closure (until the worker
// reads it) and the result (until `join` hands it back) survive collections
// and are relocated in place.

/// One in-flight (or finished-but-unjoined) spawned thread.
struct ThreadSlot {
    /// Input closure pointer. GC-rooted here until the worker reads it.
    closure: std::sync::atomic::AtomicU64,
    /// Result pointer, written by the worker on completion. GC-rooted
    /// here until `join` returns it. `0` = not yet produced.
    result: std::sync::atomic::AtomicU64,
    done: std::sync::atomic::AtomicBool,
    handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

#[derive(Default)]
struct ThreadRegistry {
    slots: Mutex<Vec<Option<Arc<ThreadSlot>>>>,
}

impl crate::gc::roots::RootSource for ThreadRegistry {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        let slots = self.slots.lock().unwrap();
        for slot in slots.iter().flatten() {
            // `0` (null) slots are ignored by the collector's slot
            // processing, so unset closure/result are harmless.
            visitor(slot.closure.as_ptr());
            visitor(slot.result.as_ptr());
        }
    }
}

static THREAD_REGISTRY: std::sync::OnceLock<ThreadRegistry> = std::sync::OnceLock::new();
static REGISTRY_ROOTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

fn thread_registry() -> &'static ThreadRegistry {
    THREAD_REGISTRY.get_or_init(ThreadRegistry::default)
}

/// Register the thread registry as a permanent GC root source, once.
fn ensure_registry_rooted(heap: &Heap) {
    REGISTRY_ROOTED.get_or_init(|| {
        let src: &'static dyn crate::gc::roots::RootSource = thread_registry();
        unsafe { heap.register_permanent_extra(src as *const dyn crate::gc::roots::RootSource) };
    });
}

/// Build a worker `Thread` that shares the parent's heap / code-table /
/// runtime-shape pointers but has its own null shadow-stack head and the
/// supplied `dyna_thread`. Exposed here (rather than constructed inline in
/// the thread module) because `Thread`'s padding field is private.
///
/// # Safety
/// All pointer arguments must be valid for the worker's lifetime; the
/// parent runtime must outlive the worker.
#[allow(clippy::too_many_arguments)]
unsafe fn build_worker_thread(
    heap: *mut Heap,
    code_table: *const CodeTable,
    dyna: *const ThreadState,
    boxed_int_ti: *const TypeInfo,
    string_ti: *const TypeInfo,
    array_ti: *const TypeInfo,
    atom_ti: *const TypeInfo,
) -> Box<Thread> {
    Box::new(Thread {
        state: 0,
        _pad: [0; 7],
        top_frame: core::ptr::null_mut(),
        heap,
        code_table,
        dyna_thread: dyna,
        boxed_int_ti,
        string_ti,
        array_ti,
        atom_ti,
    })
}

/// Body of a spawned worker OS thread: register a fresh GC context, read the
/// (possibly relocated) closure pointer from the slot, invoke it, publish the
/// result, and deregister.
#[allow(clippy::too_many_arguments)]
fn run_worker(
    heap_addr: usize,
    code_table: usize,
    boxed_int_ti: usize,
    string_ti: usize,
    array_ti: usize,
    atom_ti: usize,
    slot: Arc<ThreadSlot>,
) {
    use std::sync::atomic::Ordering;
    let heap = unsafe { &*(heap_addr as *const Heap) };
    // Register THIS OS thread's GC state (records its own os_thread id).
    let (worker_ts, _id) = heap.register_thread();
    let mut worker_thread = unsafe {
        build_worker_thread(
            heap_addr as *mut Heap,
            code_table as *const CodeTable,
            &*worker_ts as *const ThreadState,
            boxed_int_ti as *const TypeInfo,
            string_ti as *const TypeInfo,
            array_ti as *const TypeInfo,
            atom_ti as *const TypeInfo,
        )
    };
    // Let the GC coordinator stop us mid-loop via the inline poll.
    worker_ts.set_poll_flag(&mut worker_thread.state as *mut u8);
    let tptr = &mut *worker_thread as *mut Thread;

    // Read the closure from the slot (the registry kept it rooted across
    // any GC that ran between spawn and now, updating it on relocation).
    let closure = slot.closure.load(Ordering::Acquire) as *const u8;
    let result = unsafe {
        let fn_ptr = ai_gc_lookup_code(tptr, closure);
        let lambda: unsafe extern "C" fn(*mut Thread, *const u8) -> *mut u8 =
            core::mem::transmute(fn_ptr);
        lambda(tptr, closure)
    };
    // Publish the result BEFORE deregistering: `safe_deregister_thread` may
    // park us at a safepoint for an in-flight collection, which scans the
    // slot — so the result must already be rooted there.
    slot.result.store(result as u64, Ordering::Release);
    slot.done.store(true, Ordering::Release);
    heap.safe_deregister_thread(&worker_ts);
}

/// `spawn(thunk)` — start `thunk` on a new OS thread; returns a handle
/// (a `BoxedInt` registry id).
///
/// # Safety
/// `thread` is a valid parent `Thread*`; `closure_ptr` is a zero-arg closure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_thread_spawn(thread: *mut Thread, closure_ptr: *const u8) -> *mut u8 {
    use std::sync::atomic::{AtomicBool, AtomicU64};
    unsafe {
        let parent = &*thread;
        let heap = &*parent.heap;
        ensure_registry_rooted(heap);

        // Snapshot the shared pointers (read on the parent thread) to hand
        // to the worker as plain integers (Send-safe).
        let heap_addr = parent.heap as usize;
        let code_table = parent.code_table as usize;
        let boxed_int_ti = parent.boxed_int_ti as usize;
        let string_ti = parent.string_ti as usize;
        let array_ti = parent.array_ti as usize;
        let atom_ti = parent.atom_ti as usize;

        // Create the slot and root the input closure BEFORE any allocation
        // (the box below may GC).
        let slot = Arc::new(ThreadSlot {
            closure: AtomicU64::new(closure_ptr as u64),
            result: AtomicU64::new(0),
            done: AtomicBool::new(false),
            handle: Mutex::new(None),
        });
        let id = {
            let mut slots = thread_registry().slots.lock().unwrap();
            slots.push(Some(slot.clone()));
            (slots.len() - 1) as i64
        };

        let worker_slot = slot.clone();
        let jh = std::thread::spawn(move || {
            run_worker(
                heap_addr,
                code_table,
                boxed_int_ti,
                string_ti,
                array_ti,
                atom_ti,
                worker_slot,
            );
        });
        *slot.handle.lock().unwrap() = Some(jh);

        // Hand back a BoxedInt holding the slot id.
        ai_gc_box_int(thread, id)
    }
}

/// `join(handle)` — block until the spawned thread finishes; return its
/// result pointer.
///
/// # Safety
/// `thread` is a valid `Thread*`; `handle_ptr` is a `spawn` handle (BoxedInt).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_thread_join(thread: *mut Thread, handle_ptr: *const u8) -> *mut u8 {
    use std::sync::atomic::Ordering;
    unsafe {
        let parent = &*thread;
        let heap = &*parent.heap;
        let joiner = &*parent.dyna_thread;
        let id = ai_gc_unbox_int(handle_ptr) as usize;

        let slot = {
            let slots = thread_registry().slots.lock().unwrap();
            slots
                .get(id)
                .and_then(|s| s.clone())
                .expect("ai_thread_join: invalid or already-joined handle")
        };

        // Take the OS handle (releasing the registry lock first), then wait
        // in a BLOCKED region so a STW collection on another thread scans us
        // in place instead of hanging.
        let jh = slot.handle.lock().unwrap().take();
        joiner.enter_blocked();
        match jh {
            Some(jh) => {
                let _ = jh.join();
            }
            None => {
                // Handle not yet recorded (worker finished extremely fast)
                // or already joined: wait on the done flag.
                while !slot.done.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
            }
        }
        joiner.exit_blocked(heap);

        let result = slot.result.load(Ordering::Acquire) as *mut u8;
        // Drop the slot: the result is returned straight into a JIT root
        // slot with no intervening safepoint, so it stays alive.
        {
            let mut slots = thread_registry().slots.lock().unwrap();
            if let Some(s) = slots.get_mut(id) {
                *s = None;
            }
        }
        result
    }
}

/// Force a full stop-the-world collection. Exposed to the language
/// as `gc_collect()` so tests and stress harnesses can verify that
/// roots get scanned + relocated correctly.
///
/// Sets `parked_jit_fp` to the current JIT frame head for the
/// duration of the collect so the GC walks the live shadow-stack
/// chain (same protocol as `ai_gc_alloc_closure`). Returns 0 so the
/// language-level signature can be a plain `() -> Int`.
///
/// # Safety
/// `thread` must be a valid `Thread*` with `heap` and `dyna_thread`
/// initialised. All other mutator threads (if any in future
/// multi-threaded mode) must be at safepoints.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_gc_force_collect(thread: *mut Thread) -> i64 {
    unsafe {
        let t = &*thread;
        let heap = &*t.heap;
        let dyna = &*t.dyna_thread;
        dyna.set_parked_jit_fp(t.top_frame as *const u8);
        // Use the mutator-triggered STW path (not bare `collect`) so that
        // if other mutator threads are running — e.g. workers spawned via
        // `spawn` — they are stopped at safepoints before we relocate.
        // Single-threaded, the snapshot is just us and this is a plain
        // collection.
        heap.mutator_triggered_gc::<crate::gc::IdentityPtrPolicy>(dyna);
        dyna.clear_parked_jit_fp();
    }
    0
}

/// Slow-path safepoint handler. Called from JIT'd code when the
/// inline `thread.state` check finds a non-zero value.
///
/// In v1 (single-threaded), this is unreachable in normal flow —
/// nothing else sets `state` non-zero. Wired up for completeness;
/// becomes load-bearing when we add concurrent GC threads.
///
/// # Safety
///
/// `thread` must be valid; `_origin` is the `FrameOrigin` of the
/// calling function (used by future diagnostics).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_gc_pollcheck_slow(
    thread: *mut Thread,
    _origin: *const FrameOrigin,
) {
    unsafe {
        let t = &*thread;
        let dyna = &*t.dyna_thread;
        dyna.set_parked_jit_fp(t.top_frame as *const u8);
        dyna.enter_safepoint();
        dyna.clear_parked_jit_fp();
        // Clear the state flag — the GC has finished.
        let state_ptr = (thread as *mut u8).add(thread_offsets::STATE);
        *state_ptr = 0;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A durable cell store survives a "restart": values written on one
    /// Runtime are reloaded by a fresh Runtime pointed at the same dir.
    #[test]
    fn thread_layout_matches_offsets() {
        // The const _ asserts above run at compile time; this is a
        // belt-and-braces runtime check that catches additions to the
        // struct that happen to not break asserts (e.g. accidentally
        // adding a 7-byte field that compiler-pads to 8).
        assert_eq!(core::mem::offset_of!(Thread, state), 0);
        assert_eq!(core::mem::offset_of!(Thread, top_frame), 8);
        assert_eq!(core::mem::offset_of!(Thread, heap), 16);
        assert_eq!(core::mem::offset_of!(Thread, code_table), 24);
        assert_eq!(core::mem::offset_of!(Thread, dyna_thread), 32);
    }

    #[test]
    fn frame_layout_matches_offsets() {
        assert_eq!(core::mem::offset_of!(Frame, parent), 0);
        assert_eq!(core::mem::offset_of!(Frame, origin), 8);
        assert_eq!(core::mem::size_of::<Frame>(), 16);
    }

    #[test]
    fn closure_raw_layout_matches_offsets() {
        assert_eq!(core::mem::offset_of!(ClosureRaw, code_hash), 0);
        assert_eq!(core::mem::offset_of!(ClosureRaw, n_captures), 32);
        assert_eq!(core::mem::size_of::<ClosureRaw>(), 40);
    }

    #[test]
    fn runtime_constructs_with_empty_type_table() {
        let rt = Runtime::new(vec![]);
        assert_eq!(rt.thread.state, 0);
        assert!(rt.thread.top_frame.is_null());
        assert!(!rt.thread.heap.is_null());
        assert!(!rt.thread.code_table.is_null());
        assert!(!rt.thread.dyna_thread.is_null());
        drop(rt);
    }

    #[test]
    fn code_table_insert_and_lookup_roundtrip() {
        let table = CodeTable::new();
        let h = Hash([0x42; 32]);
        let dummy_fn = 0xdeadbeefusize as *const u8;
        table.insert(h, dummy_fn);
        assert_eq!(table.lookup(&h), Some(dummy_fn));
        assert_eq!(table.lookup(&Hash([0; 32])), None);
    }

    #[test]
    fn walk_empty_chain_visits_nothing() {
        let mut count = 0;
        unsafe {
            walk_jit_frames(core::ptr::null(), &mut |_| count += 1);
        }
        assert_eq!(count, 0);
    }

    #[test]
    fn walk_single_frame_visits_all_roots() {
        // Build a fake frame with 3 roots on the stack manually.
        // The frame storage is a (Frame, [*mut u8; 3]) laid out in
        // a backing array of u64s for size + alignment certainty.
        // 16 bytes Frame header + 24 bytes for 3 *mut u8 = 40 bytes = 5 u64s.
        let origin = FrameOrigin::new(3, b"test\0".as_ptr());
        let mut storage: [u64; 5] = [
            0,                                 // parent = null
            &origin as *const FrameOrigin as u64, // origin
            0x1111,                            // root[0]
            0x2222,                            // root[1]
            0x3333,                            // root[2]
        ];

        let mut visited = Vec::new();
        unsafe {
            walk_jit_frames(storage.as_mut_ptr() as *const u8, &mut |slot| {
                visited.push(*slot);
            });
        }
        assert_eq!(visited, vec![0x1111, 0x2222, 0x3333]);
    }

    #[test]
    fn walk_chained_frames_visits_parent_after_child() {
        // Parent frame: 1 root.
        let parent_origin = FrameOrigin::new(1, b"parent\0".as_ptr());
        let mut parent_storage: [u64; 3] = [
            0,                                          // parent = null
            &parent_origin as *const FrameOrigin as u64, // origin
            0xAAAA,                                     // root[0]
        ];
        let parent_ptr = parent_storage.as_mut_ptr() as *mut Frame;

        // Child frame: 2 roots. Parent = parent_ptr.
        let child_origin = FrameOrigin::new(2, b"child\0".as_ptr());
        let mut child_storage: [u64; 4] = [
            parent_ptr as u64,                         // parent
            &child_origin as *const FrameOrigin as u64, // origin
            0xBBBB,                                    // root[0]
            0xCCCC,                                    // root[1]
        ];

        let mut visited = Vec::new();
        unsafe {
            walk_jit_frames(child_storage.as_mut_ptr() as *const u8, &mut |slot| {
                visited.push(*slot);
            });
        }
        // Walk visits child's roots first, then parent's.
        assert_eq!(visited, vec![0xBBBB, 0xCCCC, 0xAAAA]);
    }

    /// End-to-end GC verification: forge a fake JIT frame on the test
    /// stack, allocate two heap objects via `ai_gc_alloc_closure`,
    /// store their pointers in the frame's root slots, force a
    /// collection, then verify (a) the slots were rewritten to to-space
    /// addresses and (b) the relocated objects still contain the
    /// data we wrote into them.
    ///
    /// This is the lowest-level proof that:
    ///   - `walk_jit_frames` is reachable from `Heap::collect`
    ///   - `process_slot` rewrites our root slots in place
    ///   - Object payloads are correctly copied to to-space
    #[test]
    fn forged_frame_survives_collect() {
        use crate::gc::{IdentityPtrPolicy, TypeInfo};
        use std::sync::Arc;

        // Two BoxedInt-like shapes: 0 value fields, 8 raw bytes
        // (so we can stuff an i64 sentinel into each).
        let shape = TypeInfo::for_header(<Full as crate::gc::ObjHeader>::SIZE)
            .with_type_id(0)
            .with_fields(0)
            .with_raw_bytes(8);

        // Build a runtime with this single shape so we have a real
        // heap + walker wired up.
        let rt = Runtime::new(vec![shape]);
        let ti_ptr: *const TypeInfo = &*rt.type_infos[0];

        // Allocate two objects + stamp them with sentinel payloads.
        const SENTINEL_A: i64 = 0xA1A1_A1A1_A1A1_A1A1u64 as i64;
        const SENTINEL_B: i64 = 0xB2B2_B2B2_B2B2_B2B2u64 as i64;
        let obj_a = unsafe { ai_gc_alloc_closure(rt.thread_ptr(), ti_ptr) };
        let obj_b = unsafe { ai_gc_alloc_closure(rt.thread_ptr(), ti_ptr) };
        assert!(!obj_a.is_null() && !obj_b.is_null());
        unsafe {
            *(obj_a.add(<Full as crate::gc::ObjHeader>::SIZE) as *mut i64) = SENTINEL_A;
            *(obj_b.add(<Full as crate::gc::ObjHeader>::SIZE) as *mut i64) = SENTINEL_B;
        }

        let from_base = rt.heap.from_base() as usize;
        let from_end = from_base + rt.heap.space_size();
        let to_base = rt.heap.to_base() as usize;
        let to_end = to_base + rt.heap.space_size();
        let in_from =
            |p: usize| -> bool { p >= from_base && p < from_end };
        let in_to = |p: usize| -> bool { p >= to_base && p < to_end };
        assert!(in_from(obj_a as usize), "fresh alloc should land in from-space");
        assert!(in_from(obj_b as usize));

        // Build a forged frame on the test's stack with the two
        // pointers held as roots.
        let origin = FrameOrigin::new(2, b"forged\0".as_ptr());
        let mut storage: [u64; 4] = [
            0,                                          // parent
            &origin as *const FrameOrigin as u64,       // origin
            obj_a as u64,                               // root[0]
            obj_b as u64,                               // root[1]
        ];

        // Publish the forged frame as the parked JIT fp and run a
        // collection. (Mimics what `ai_gc_alloc_closure` does around
        // every alloc, except we hold the frame across the collect
        // so the GC sees our roots.)
        rt.dyna_thread
            .set_parked_jit_fp(storage.as_mut_ptr() as *const u8);
        unsafe { rt.heap.collect::<IdentityPtrPolicy>(&[]) };
        rt.dyna_thread.clear_parked_jit_fp();

        // After collect, from/to swapped — the live objects are now
        // in the NEW from-space (= old to-space). Slot pointers must
        // have been updated to point there.
        let new_a = storage[2] as *const u8;
        let new_b = storage[3] as *const u8;
        assert_ne!(
            new_a as usize, obj_a as usize,
            "GC should have relocated obj_a; slot was not updated"
        );
        assert_ne!(new_b as usize, obj_b as usize);

        // The relocated objects should be in either the new from-space
        // (post-swap, formerly to-space) — i.e. in the to_base..to_end
        // window we captured before the swap.
        assert!(
            in_to(new_a as usize),
            "relocated obj_a should be in former to-space (now from); got {:p}",
            new_a
        );
        assert!(in_to(new_b as usize));

        // Payloads must survive the copy unchanged.
        let recovered_a = unsafe {
            *(new_a.add(<Full as crate::gc::ObjHeader>::SIZE) as *const i64)
        };
        let recovered_b = unsafe {
            *(new_b.add(<Full as crate::gc::ObjHeader>::SIZE) as *const i64)
        };
        assert_eq!(recovered_a, SENTINEL_A, "obj_a payload corrupted by GC");
        assert_eq!(recovered_b, SENTINEL_B, "obj_b payload corrupted by GC");

        let _ = Arc::strong_count(&rt.heap); // pin runtime for lifetime clarity
    }

    /// Sanity: if a heap object is NOT held by any root, GC reclaims it.
    /// We allocate then drop the pointer, GC, then allocate again —
    /// the new allocation should use the space the prior alloc held.
    #[test]
    fn unrooted_alloc_is_reclaimed() {
        use crate::gc::{IdentityPtrPolicy, TypeInfo};

        let shape = TypeInfo::for_header(<Full as crate::gc::ObjHeader>::SIZE)
            .with_type_id(0)
            .with_fields(0)
            .with_raw_bytes(8);
        let rt = Runtime::new(vec![shape]);
        let ti_ptr: *const TypeInfo = &*rt.type_infos[0];

        let used_before = rt.heap.from_used();
        // Allocate something that nobody roots.
        let _orphan = unsafe { ai_gc_alloc_closure(rt.thread_ptr(), ti_ptr) };
        let used_after_alloc = rt.heap.from_used();
        assert!(
            used_after_alloc > used_before,
            "alloc should bump from_used"
        );

        // No parked_jit_fp → no JIT roots seen. The thread also has no
        // registered roots in `globals` / per-thread roots that the
        // alloc would have surfaced. So the object is collectible.
        unsafe { rt.heap.collect::<IdentityPtrPolicy>(&[]) };
        let used_after_gc = rt.heap.from_used();
        assert!(
            used_after_gc < used_after_alloc,
            "GC should reclaim the unrooted object; before={}, after={}",
            used_after_alloc,
            used_after_gc
        );
    }
}
