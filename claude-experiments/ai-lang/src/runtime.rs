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

use crate::codegen::ShapeMeta;
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
    /// literal codegen and by `ai_str_*` runtime fns.
    pub string_ti: *const TypeInfo,
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
}

// The raw `*const u8` is a JIT'd function pointer — process-local but
// not tied to any one thread. Safe to share read-only.
unsafe impl Send for CodeTable {}
unsafe impl Sync for CodeTable {}

impl CodeTable {
    pub fn new() -> Self {
        CodeTable {
            table: Mutex::new(HashMap::new()),
        }
    }

    pub fn insert(&self, hash: Hash, fn_ptr: *const u8) {
        self.table.lock().unwrap().insert(hash, fn_ptr);
    }

    pub fn lookup(&self, hash: &Hash) -> Option<*const u8> {
        self.table.lock().unwrap().get(hash).copied()
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
    /// UTF-8 bytes).
    pub string_ti: Box<TypeInfo>,
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
        shape_registry: HashMap<Hash, u16>,
        shape_meta: HashMap<Hash, ShapeMeta>,
        shape_by_type_id: Vec<Option<Hash>>,
    ) -> Self {
        // Reserve distinct type_ids for runtime-managed shapes at the
        // END of the closure-types table so they don't collide with
        // any module's shapes.
        //
        // BoxedInt: 0 value fields, 8 raw bytes (one i64).
        // String:   0 value fields, 0 raw bytes, varlen-bytes section.
        let boxed_int_type_id = closure_types.len() as u16;
        let string_type_id = boxed_int_type_id + 1;
        let boxed_int = TypeInfo::for_header(crate::gc::Full::SIZE as usize)
            .with_type_id(boxed_int_type_id)
            .with_fields(0)
            .with_raw_bytes(8);
        let string_ti_val = TypeInfo::for_header(crate::gc::Full::SIZE as usize)
            .with_type_id(string_type_id)
            .with_fields(0)
            .with_varlen_bytes(0);

        let mut all_types = closure_types.clone();
        all_types.push(boxed_int);
        all_types.push(string_ti_val);

        // Boxed copies for stable addresses (used by the deserializer
        // when it needs to pass a `*const TypeInfo` to `ai_gc_alloc_closure`).
        // BoxedInt + String sit at the same trailing slots as in the
        // heap's type-table so subsequent dynamic installs append into
        // both tables in lockstep.
        let mut type_infos: Vec<Box<TypeInfo>> =
            closure_types.iter().map(|t| Box::new(*t)).collect();
        let boxed_int_ti_box: Box<TypeInfo> = Box::new(boxed_int);
        let string_ti_box: Box<TypeInfo> = Box::new(string_ti_val);
        type_infos.push(Box::new(boxed_int));
        type_infos.push(Box::new(string_ti_val));

        // 32 MiB semi-space — plenty for tests, easily reconfigurable later.
        let heap = Arc::new(Heap::new::<Full>(32 * 1024 * 1024, all_types));
        heap.set_jit_frame_walker(walk_jit_frames);

        let (dyna_thread, _id) = heap.register_thread();
        let code_table = Box::new(CodeTable::new());

        let mut thread = Box::new(Thread {
            state: 0,
            _pad: [0; 7],
            top_frame: core::ptr::null_mut(),
            heap: Arc::as_ptr(&heap) as *mut Heap,
            code_table: &*code_table,
            dyna_thread: &*dyna_thread,
            boxed_int_ti: &*boxed_int_ti_box,
            string_ti: &*string_ti_box,
        });

        let _ = &mut *thread;

        Runtime {
            heap,
            dyna_thread,
            code_table,
            thread,
            type_infos,
            shape_registry,
            shape_meta,
            shape_by_type_id,
            boxed_int_ti: boxed_int_ti_box,
            string_ti: string_ti_box,
        }
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
/// 32-byte hash. Panics if the hash isn't in the table — that means
/// a closure references code that wasn't JIT'd, which is a bug.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ai_gc_lookup_code(
    thread: *mut Thread,
    hash_ptr: *const u8,
) -> *const u8 {
    unsafe {
        let t = &*thread;
        let table = &*t.code_table;
        let mut h = [0u8; Hash::SIZE];
        core::ptr::copy_nonoverlapping(hash_ptr, h.as_mut_ptr(), Hash::SIZE);
        let hash = Hash(h);
        match table.lookup(&hash) {
            Some(p) => p,
            None => panic!(
                "ai_gc_lookup_code: hash {} not registered in code table",
                hash
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
        heap.collect::<crate::gc::IdentityPtrPolicy>(&[]);
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
