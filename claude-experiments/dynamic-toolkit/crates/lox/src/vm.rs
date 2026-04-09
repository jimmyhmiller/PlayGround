//! Lox VM: parse → resolve → lower → optimize → run (interpreter or JIT).

use std::cell::Cell;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use dynalloc::{Heap, PtrPolicy};
use dynexec::{PreciseStackRoots, RootTransport, ValueLayout};
use dynir::interp::{ExternCallResult, InterpResult, ModuleInterpreter};
use dynir::ir::{FuncDef, Module};
use dynir::InterpRootManager;
use dynir::opt;
use dynlower::{JitModule, JitOutcome};
use dynobj::{Compact, DynRootFrame, FrameChain, ObjHeader, RootFrame, TypeInfo};
use dynvalue::NanBox;

use crate::lower::{self, LoxGcTypes};

/// Read the current frame pointer (X29 on aarch64).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn frame_pointer() -> *const u8 {
    let fp: *const u8;
    unsafe { std::arch::asm!("mov {}, x29", out(reg) fp) };
    fp
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn frame_pointer() -> *const u8 {
    std::ptr::null()
}
use crate::parser::Parser;
use crate::value::*;

pub enum InterpretResult {
    Ok,
    CompileError,
    RuntimeError,
}

// ── NanBox PtrPolicy for Lox ─────────────────────────────────────

const LOX_FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
const LOX_TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;

/// PtrPolicy that decodes TAG_OBJ (tag 0) NanBox values as heap pointers.
pub struct LoxPtrPolicy;

impl PtrPolicy for LoxPtrPolicy {
    fn try_decode_ptr(bits: u64) -> Option<*mut u8> {
        // TAG_OBJ = 0, so pattern is TAG_PATTERN | (0 << 48) = TAG_PATTERN
        let mask = LOX_FULL_MASK | (0x0003u64 << 48);
        let expected = LOX_TAG_PATTERN; // tag 0
        if (bits & mask) != expected {
            return None;
        }
        let payload = bits & PAYLOAD_MASK;
        if payload == 0 {
            return None;
        }
        Some(payload as *mut u8)
    }

    fn encode_ptr(ptr: *mut u8) -> u64 {
        LOX_TAG_PATTERN | (ptr as u64)
    }
}

// ── LoxGcRuntime: proper GC with roots ───────────────────────────

/// GC runtime for Lox. Owns the heap, frame chain for interpreter roots,
/// and root sets for globals and strings stored in heap.globals.
pub struct LoxGcRuntime {
    pub heap: Heap,
    chain: FrameChain,
    frames: std::cell::RefCell<Vec<DynRootFrame>>,
    /// Globals: maps global ID → index in heap.globals
    pub globals: GlobalRoots,
    /// Interned strings: maps text → index in heap.globals
    pub string_roots: StringRoots,
    /// Type infos for alloc dispatch. Owned by value — no leak.
    type_infos: Vec<TypeInfo>,
    /// Alloc counter for triggering GC.
    alloc_count: Cell<usize>,
    gc_threshold: usize,
}

/// Globals: maps Lox global ID → index in Heap.globals (AtomicRootSet).
/// Values stored in the heap's global roots are automatically scanned by GC.
pub struct GlobalRoots {
    /// global_id → index in heap.globals
    slot_indices: Vec<Option<usize>>,
    defined: std::collections::HashSet<usize>,
}

impl GlobalRoots {
    fn new() -> Self { GlobalRoots { slot_indices: Vec::new(), defined: std::collections::HashSet::new() } }

    fn ensure(&mut self, heap: &Heap, id: usize) {
        if id >= self.slot_indices.len() {
            self.slot_indices.resize(id + 1, None);
        }
        if self.slot_indices[id].is_none() {
            let idx = heap.globals.add(nil_val());
            self.slot_indices[id] = Some(idx);
        }
    }

    fn get(&self, heap: &Heap, id: usize) -> u64 {
        heap.globals.get(self.slot_indices[id].unwrap())
    }

    fn set(&self, heap: &Heap, id: usize, val: u64) {
        heap.globals.set(self.slot_indices[id].unwrap(), val);
    }

    fn define(&mut self, id: usize) { self.defined.insert(id); }
    fn is_defined(&self, id: usize) -> bool {
        id < self.slot_indices.len() && self.defined.contains(&id) && self.slot_indices[id].is_some()
    }
}

/// Interned strings: maps text → index in Heap.globals (AtomicRootSet).
pub struct StringRoots {
    /// compile-time string ID → index in heap.globals
    table: Vec<usize>,
    /// text → position in table (for dedup)
    intern_map: HashMap<String, usize>,
}

impl StringRoots {
    fn new() -> Self { StringRoots { table: Vec::new(), intern_map: HashMap::new() } }

    fn get(&self, heap: &Heap, idx: usize) -> u64 {
        heap.globals.get(self.table[idx])
    }

    fn add(&mut self, heap: &Heap, val: u64) -> usize {
        let heap_idx = heap.globals.add(val);
        let idx = self.table.len();
        self.table.push(heap_idx);
        idx
    }

    fn lookup(&self, heap: &Heap, text: &str) -> Option<u64> {
        self.intern_map.get(text).map(|&idx| heap.globals.get(self.table[idx]))
    }

    fn insert(&mut self, heap: &Heap, text: &str, val: u64) -> u64 {
        if let Some(&idx) = self.intern_map.get(text) {
            return heap.globals.get(self.table[idx]);
        }
        let idx = self.add(heap, val);
        self.intern_map.insert(text.to_string(), idx);
        val
    }

    fn len(&self) -> usize { self.table.len() }
}

impl LoxGcRuntime {
    fn new(heap_size: usize, type_infos: Vec<TypeInfo>) -> Self {
        LoxGcRuntime {
            heap: Heap::new::<Compact>(heap_size, type_infos.clone()),
            chain: FrameChain::new(),
            frames: std::cell::RefCell::new(Vec::new()),
            globals: GlobalRoots::new(),
            string_roots: StringRoots::new(),
            type_infos,
            alloc_count: Cell::new(0),
            gc_threshold: 1024,
        }
    }

    /// Allocate without triggering GC. Used by Rust-side helpers where
    /// local variables may hold unrooted GC pointers.
    fn alloc_raw(&self, type_id: usize, varlen_len: usize) -> *mut u8 {
        let info = &self.type_infos[type_id];
        self.heap.alloc_obj::<Compact>(info, varlen_len)
    }

    /// Allocate with GC trigger. Only safe to call when all live GC pointers
    /// are in root slots (i.e., from the __gc_alloc__ extern during IR execution).
    fn alloc_with_gc(&self, type_id: usize, varlen_len: usize) -> *mut u8 {
        let info = &self.type_infos[type_id];

        // Check if we should collect
        let count = self.alloc_count.get() + 1;
        self.alloc_count.set(count);
        if count >= self.gc_threshold {
            self.do_collect();
            self.alloc_count.set(0);
        }

        let ptr = self.heap.alloc_obj::<Compact>(info, varlen_len);
        if ptr.is_null() {
            // OOM — try collecting and retry
            self.do_collect();
            self.heap.alloc_obj::<Compact>(info, varlen_len)
        } else {
            ptr
        }
    }

    fn do_collect(&self) {
        // Sync interpreter frame values → root slots before collection.
        let (data, pre_fn, post_fn) = GC_SYNC.with(|cell| cell.get());
        unsafe { pre_fn(data) };

        // Walk the native FP chain to find JIT frames (if we're being called
        // from a JIT extern callback).
        let jit_roots = dynlower::JitFrameRoots {
            jit_fp: frame_pointer() as *const u8,
        };
        unsafe {
            self.heap.collect::<LoxPtrPolicy>(&[&self.chain, &jit_roots]);
        }

        // Reload interpreter frame values ← root slots (pointers may have moved).
        unsafe { post_fn(data) };
    }
}

// InterpRootManager impls — the interpreter calls these to track roots
impl<L, Transport> InterpRootManager<L, PreciseStackRoots, Transport> for LoxGcRuntime
where
    L: ValueLayout,
    Transport: RootTransport<L, PreciseStackRoots>,
{
    fn push_frame(&self, gc_slot_count: usize) -> usize {
        let mut frames = self.frames.borrow_mut();
        let idx = frames.len();
        let frame = DynRootFrame::new(gc_slot_count);
        unsafe { self.chain.push_raw_unguarded(frame.header_ptr()); }
        frames.push(frame);
        idx
    }

    fn pop_frame(&self) {
        unsafe { self.chain.pop_raw(); }
        self.frames.borrow_mut().pop().expect("no frame to pop");
    }

    fn set_root(&self, frame: usize, slot: usize, value: u64) {
        self.frames.borrow()[frame].set(slot, value);
    }

    fn get_root(&self, frame: usize, slot: usize) -> u64 {
        self.frames.borrow()[frame].get(slot)
    }

    fn clear_frame(&self, frame: usize) {
        self.frames.borrow()[frame].clear_all();
    }

    fn collect(&self) {
        // Only collect if we've done enough allocations since last collection.
        let count = self.alloc_count.get();
        if count >= self.gc_threshold {
            self.do_collect();
            self.alloc_count.set(0);
        }
    }
}

impl<L, Transport> InterpRootManager<L, dynexec::ConservativeWordRoots, Transport> for LoxGcRuntime
where
    L: ValueLayout,
    Transport: RootTransport<L, dynexec::ConservativeWordRoots>,
{
    fn push_frame(&self, gc_slot_count: usize) -> usize {
        let mut frames = self.frames.borrow_mut();
        let idx = frames.len();
        let frame = DynRootFrame::new(gc_slot_count);
        unsafe { self.chain.push_raw_unguarded(frame.header_ptr()); }
        frames.push(frame);
        idx
    }
    fn pop_frame(&self) {
        unsafe { self.chain.pop_raw(); }
        self.frames.borrow_mut().pop().expect("no frame to pop");
    }
    fn set_root(&self, frame: usize, slot: usize, value: u64) {
        self.frames.borrow()[frame].set(slot, value);
    }
    fn get_root(&self, frame: usize, slot: usize) -> u64 {
        self.frames.borrow()[frame].get(slot)
    }
    fn clear_frame(&self, frame: usize) {
        self.frames.borrow()[frame].clear_all();
    }
    fn collect(&self) {
        self.do_collect();
    }
}

// ── VM struct ────────────────────────────────────────────────────

pub struct VM {
    had_error: bool,
    pub use_jit: bool,
    jit_call_table: Vec<*const u8>,
    gc_runtime: Option<LoxGcRuntime>,
    gc_types: Option<LoxGcTypes>,
    compile_strings: Vec<String>,
    /// Result kind from last invoke_lookup: 0=field, 1=method, 2=not_found
    last_invoke_kind: u64,
    /// Closure value from last invoke_fast call (used by JIT to pass as arg).
    last_invoke_closure: u64,
    /// Init method info from last instantiate call.
    last_init_arity: u64,
    last_init_func_ptr: u64,
}

// ── Thread-local VM pointer for JIT extern "C" callbacks ──────────

/// Raw thread-local VM pointer — avoids `thread_local!` overhead for
/// the hot path. Only safe because Lox is single-threaded.
static mut RAW_VM: *mut VM = std::ptr::null_mut();

thread_local! {
    /// GC sync hooks: called before/after collection to sync interpreter
    /// frame values to/from GC root slots. Stored as (data_ptr, pre_fn, post_fn).
    static GC_SYNC: Cell<(usize, unsafe fn(usize), unsafe fn(usize))> = const { Cell::new((0, noop_sync, noop_sync)) };
}

unsafe fn noop_sync(_: usize) {}

#[inline(always)]
fn with_vm<R>(f: impl FnOnce(&mut VM) -> R) -> R {
    let ptr = unsafe { RAW_VM };
    debug_assert!(!ptr.is_null(), "no active VM");
    f(unsafe { &mut *ptr })
}

// ── GC object helpers on VM ──────────────────────────────────────

impl VM {
    fn rt(&self) -> &LoxGcRuntime {
        self.gc_runtime.as_ref().expect("GC runtime not initialized")
    }

    fn types(&self) -> &LoxGcTypes {
        self.gc_types.as_ref().expect("GC types not initialized")
    }

    /// Allocate a GC object. Increments the allocation counter but does NOT
    /// trigger collection directly — GC runs at IR-level safepoints where
    /// the interpreter/JIT can properly sync frame values to root slots.
    fn gc_alloc(&self, type_id: usize, varlen_len: usize) -> *mut u8 {
        let rt = self.rt();
        rt.alloc_count.set(rt.alloc_count.get() + 1);
        let ptr = rt.alloc_raw(type_id, varlen_len);
        if ptr.is_null() {
            // OOM — force collection and retry (called from rooted context).
            rt.do_collect();
            rt.alloc_count.set(0);
            rt.alloc_raw(type_id, varlen_len)
        } else {
            ptr
        }
    }

    /// Read the type_id directly from the object header. One u16 read.
    /// Returns: 0=String, 1=Closure, 2=Upvalue, 3=Class, 4=Instance, 5=BoundMethod, 6=NativeFn, etc.
    #[inline(always)]
    fn obj_type_tag(&self, val: u64) -> u64 {
        if !is_obj(val) { return 255; }
        let ptr = obj_ptr(val);
        if ptr.is_null() { return 255; }
        unsafe { dynobj::read_type_id(ptr, dynobj::Compact::TYPE_ID_OFFSET) as u64 }
    }

    fn is_closure(&self, val: u64) -> bool {
        let tag = self.obj_type_tag(val);
        tag == 1 || tag == 6 // Closure or NativeFn
    }

    fn is_class(&self, val: u64) -> bool { self.obj_type_tag(val) == 3 }
    fn is_instance(&self, val: u64) -> bool { self.obj_type_tag(val) == 4 }
    fn is_bound_method(&self, val: u64) -> bool { self.obj_type_tag(val) == 5 }

    // ── String helpers ───────────────────────────────────────────

    /// Allocate a GC string from a Rust &str. Returns NanBox-tagged ptr.
    fn alloc_string(&self, s: &str) -> u64 {
        let bytes = s.as_bytes();
        let types = self.types();
        let string_id = types.string_id;
        let len_off = types.string_len_off;
        let data_off = types.string_data_base_off;

        let ptr = self.gc_alloc(string_id, bytes.len());
        unsafe {
            gc_write_field(ptr, len_off, bytes.len() as u64);
            gc_write_bytes(ptr, data_off, bytes);
        }
        obj_val(ptr)
    }

    /// Intern a string: same content → same NanBox value.
    /// After GC, the pointer inside the root slot may be updated,
    /// so we always read from the root slot, not from a cached value.
    fn intern_string(&mut self, s: &str) -> u64 {
        let rt = self.gc_runtime.as_mut().expect("GC runtime not initialized");
        if let Some(val) = rt.string_roots.lookup(&rt.heap, s) {
            return val;
        }
        let val = self.alloc_string(s);
        let rt = self.gc_runtime.as_mut().expect("GC runtime not initialized");
        rt.string_roots.insert(&rt.heap, s, val)
    }

    /// Read a GC string as a Rust String. `val` must be a string object.
    fn read_string(&self, val: u64) -> String {
        let ptr = obj_ptr(val);
        let types = self.types();
        unsafe {
            let len = gc_read_field(ptr, types.string_len_off) as usize;
            let bytes = gc_read_bytes(ptr, types.string_data_base_off, len);
            String::from_utf8_lossy(bytes).into_owned()
        }
    }

    /// Resolve a compile-time string ID to a NanBox GC string value.
    /// Reads from the root slot so we get the GC-updated pointer.
    fn resolve_string(&mut self, id: usize) -> u64 {
        let rt = self.gc_runtime.as_ref().expect("GC runtime not initialized");
        if id < rt.string_roots.len() {
            return rt.string_roots.get(&rt.heap, id);
        }
        self.intern_string(&format!("#{}", id))
    }

    /// Look up a compile-time string's integer ID by name.
    /// Used for runtime lookups of known strings like "init".
    fn compile_string_id(&self, name: &str) -> u64 {
        self.compile_strings.iter().position(|s| s == name).unwrap_or(u64::MAX as usize) as u64
    }

    /// Get the text for a compile-time string ID (for error messages).
    fn string_text(&self, id: usize) -> String {
        if id < self.compile_strings.len() {
            self.compile_strings[id].clone()
        } else {
            format!("#{}", id)
        }
    }

    // ── Upvalue helpers ──────────────────────────────────────────

    fn alloc_upvalue(&mut self, init: u64) -> u64 {
        let frame = RootFrame::<1>::new();
        unsafe { self.rt().chain.push_raw_unguarded(&frame.header as *const _ as *mut _) };
        frame.slots[0].set(init);

        let types = self.types();
        let upvalue_id = types.upvalue_id;
        let off = types.upvalue_value_off;
        let ptr = self.gc_alloc(upvalue_id, 0);
        let init = frame.slots[0].get();
        unsafe {
            gc_write_field(ptr, off, init);
            self.rt().chain.pop_raw();
        }
        obj_val(ptr)
    }

    fn get_upvalue(&self, cell: u64) -> u64 {
        let ptr = obj_ptr(cell);
        unsafe { gc_read_field(ptr, self.types().upvalue_value_off) }
    }

    fn set_upvalue(&self, cell: u64, val: u64) {
        let ptr = obj_ptr(cell);
        unsafe { gc_write_field(ptr, self.types().upvalue_value_off, val); }
    }

    // ── Closure helpers ──────────────────────────────────────────

    fn alloc_closure(&mut self, func_idx: u64, num_upvalues: usize, arity: u64, name_val: u64) -> u64 {
        // Root name_val (GC string) across the allocation.
        let frame = RootFrame::<1>::new();
        unsafe { self.rt().chain.push_raw_unguarded(&frame.header as *const _ as *mut _) };
        frame.slots[0].set(name_val);

        let types = self.types();
        let closure_id = types.closure_id;
        let func_idx_off = types.closure_func_idx_off;
        let arity_off = types.closure_arity_off;
        let name_off = types.closure_name_off;
        let upval_base = types.closure_upval_base_off;

        let ptr = self.gc_alloc(closure_id, num_upvalues);
        let name_val = frame.slots[0].get();
        unsafe {
            self.rt().chain.pop_raw();
            gc_write_field(ptr, func_idx_off, func_idx);
            gc_write_field(ptr, arity_off, arity);
            gc_write_field(ptr, name_off, name_val);
            // Initialize all upvalue slots to nil
            for i in 0..num_upvalues {
                gc_write_elem(ptr, upval_base, i, nil_val());
            }
        }
        obj_val(ptr)
    }

    fn closure_func_idx(&self, closure: u64) -> u64 {
        let ptr = obj_ptr(closure);
        unsafe { gc_read_field(ptr, self.types().closure_func_idx_off) }
    }

    fn closure_arity(&self, closure: u64) -> u64 {
        let ptr = obj_ptr(closure);
        unsafe { gc_read_field(ptr, self.types().closure_arity_off) }
    }

    fn closure_name(&self, closure: u64) -> u64 {
        let ptr = obj_ptr(closure);
        unsafe { gc_read_field(ptr, self.types().closure_name_off) }
    }

    fn closure_upvalue(&self, closure: u64, idx: usize) -> u64 {
        let ptr = obj_ptr(closure);
        unsafe { gc_read_elem(ptr, self.types().closure_upval_base_off, idx) }
    }

    fn set_closure_upvalue(&self, closure: u64, idx: usize, cell: u64) {
        let ptr = obj_ptr(closure);
        unsafe { gc_write_elem(ptr, self.types().closure_upval_base_off, idx, cell); }
    }

    // ── GC Table helpers ─────────────────────────────────────────
    // Table stores [key, val, key, val, ...] as varlen values with a count field.

    /// Hash a u64 key for open-addressing table lookup.
    /// Uses FxHash-style multiply-shift (keys are interned NaN-boxed string pointers).
    #[inline(always)]
    fn table_hash(key: u64) -> usize {
        // FxHash constant
        let h = key.wrapping_mul(0x517cc1b727220a95);
        (h >> 32) as usize
    }

    /// Allocate an empty hash table with the given capacity (must be power of 2, or 0).
    /// Uses open addressing with linear probing; nil keys = empty slots.
    fn alloc_table(&mut self, capacity: usize) -> u64 {
        // Round up to power of 2 (minimum 8 if non-zero)
        let capacity = if capacity == 0 {
            0
        } else {
            let min = capacity.max(8);
            min.next_power_of_two()
        };
        let types = self.types();
        let table_id = types.table_id;
        let count_off = types.table_count_off;
        let base_off = types.table_data_base_off;

        let ptr = self.gc_alloc(table_id, capacity * 2); // 2 slots per pair (key, value)
        unsafe {
            gc_write_field(ptr, count_off, 0);
            // Initialize all slots to nil (nil key = empty slot)
            for i in 0..(capacity * 2) {
                gc_write_elem(ptr, base_off, i, nil_val());
            }
        }
        obj_val(ptr)
    }

    /// Read the hash table capacity (number of key-value slots, not entries).
    fn table_capacity(&self, table: u64) -> usize {
        let ptr = obj_ptr(table);
        let types = self.types();
        let varlen_count = unsafe {
            let vc_off = types.type_infos[types.table_id].varlen_count_offset();
            gc_read_field(ptr, vc_off as i32) as usize
        };
        varlen_count / 2  // varlen has 2 slots per entry (key + value)
    }

    /// Read the count (number of key-value pairs) from a table.
    fn table_count(&self, table: u64) -> usize {
        let ptr = obj_ptr(table);
        unsafe { gc_read_field(ptr, self.types().table_count_off) as usize }
    }

    /// Look up a key in a hash table using open addressing + linear probing.
    fn table_get(&self, table: u64, key: u64) -> Option<u64> {
        let ptr = obj_ptr(table);
        let types = self.types();
        let capacity = self.table_capacity(table);
        if capacity == 0 {
            return None;
        }
        let base = types.table_data_base_off;
        let mask = capacity - 1; // capacity is power of 2
        let mut idx = Self::table_hash(key) & mask;
        loop {
            let k = unsafe { gc_read_elem(ptr, base, idx * 2) };
            if k == key {
                return Some(unsafe { gc_read_elem(ptr, base, idx * 2 + 1) });
            }
            if is_nil(k) {
                return None; // empty slot = key not present
            }
            idx = (idx + 1) & mask;
        }
    }

    /// Insert all entries from old_ptr (with old_capacity) into new_ptr (with new_capacity).
    /// Used during table growth. No allocation happens here.
    unsafe fn table_rehash(
        &self,
        old_ptr: *mut u8, old_capacity: usize,
        new_ptr: *mut u8, new_capacity: usize,
    ) {
        let base = self.types().table_data_base_off;
        let new_mask = new_capacity - 1;
        for i in 0..old_capacity {
            let k = gc_read_elem(old_ptr, base, i * 2);
            if is_nil(k) {
                continue;
            }
            let v = gc_read_elem(old_ptr, base, i * 2 + 1);
            let mut idx = Self::table_hash(k) & new_mask;
            loop {
                let slot_k = gc_read_elem(new_ptr, base, idx * 2);
                if is_nil(slot_k) {
                    gc_write_elem(new_ptr, base, idx * 2, k);
                    gc_write_elem(new_ptr, base, idx * 2 + 1, v);
                    break;
                }
                idx = (idx + 1) & new_mask;
            }
        }
    }

    /// Set a key-value pair in a hash table. If key exists, updates in place.
    /// If key doesn't exist, inserts. May grow the table — returns the
    /// (possibly new) table value.
    fn table_set(&mut self, table: u64, key: u64, value: u64) -> u64 {
        let capacity = self.table_capacity(table);

        // If capacity is 0, must grow first
        if capacity == 0 {
            return self.table_set_grow(table, 0, key, value);
        }

        let ptr = obj_ptr(table);
        let types = self.types();
        let base = types.table_data_base_off;
        let count_off = types.table_count_off;
        let mask = capacity - 1;
        let mut idx = Self::table_hash(key) & mask;

        loop {
            let k = unsafe { gc_read_elem(ptr, base, idx * 2) };
            if k == key {
                // Key exists — update in place
                unsafe { gc_write_elem(ptr, base, idx * 2 + 1, value); }
                return table;
            }
            if is_nil(k) {
                // Empty slot — check load factor before inserting
                let count = unsafe { gc_read_field(ptr, count_off) as usize };
                // Grow if load factor > 75%
                if (count + 1) * 4 > capacity * 3 {
                    return self.table_set_grow(table, capacity, key, value);
                }
                // Insert here
                unsafe {
                    gc_write_elem(ptr, base, idx * 2, key);
                    gc_write_elem(ptr, base, idx * 2 + 1, value);
                    gc_write_field(ptr, count_off, (count + 1) as u64);
                }
                return table;
            }
            idx = (idx + 1) & mask;
        }
    }

    /// Grow the table and insert the new key-value pair.
    fn table_set_grow(&mut self, table: u64, old_capacity: usize, key: u64, value: u64) -> u64 {
        let new_cap = if old_capacity == 0 { 8 } else { old_capacity * 2 };

        let frame = RootFrame::<3>::new();
        unsafe { self.rt().chain.push_raw_unguarded(&frame.header as *const _ as *mut _) };
        frame.slots[0].set(table);
        frame.slots[1].set(key);
        frame.slots[2].set(value);

        let new_table = self.alloc_table(new_cap);

        let table = frame.slots[0].get();
        let key = frame.slots[1].get();
        let value = frame.slots[2].get();
        unsafe { self.rt().chain.pop_raw() };

        let old_ptr = obj_ptr(table);
        let new_ptr = obj_ptr(new_table);

        // Rehash old entries into new table
        if old_capacity > 0 {
            unsafe { self.table_rehash(old_ptr, old_capacity, new_ptr, new_cap); }
        }

        // Now insert the new key-value pair
        let base = self.types().table_data_base_off;
        let count_off = self.types().table_count_off;
        let old_count = if old_capacity > 0 {
            unsafe { gc_read_field(old_ptr, count_off) as usize }
        } else {
            0
        };
        let new_mask = new_cap - 1;
        let mut idx = Self::table_hash(key) & new_mask;
        unsafe {
            loop {
                let k = gc_read_elem(new_ptr, base, idx * 2);
                if is_nil(k) {
                    gc_write_elem(new_ptr, base, idx * 2, key);
                    gc_write_elem(new_ptr, base, idx * 2 + 1, value);
                    break;
                }
                idx = (idx + 1) & new_mask;
            }
            gc_write_field(new_ptr, count_off, (old_count + 1) as u64);
        }
        new_table
    }

    /// Copy all entries from src table into dst table (for inheritance).
    /// Only inserts keys that don't already exist in dst.
    /// Returns the (possibly grown) dst table.
    fn table_merge(&mut self, dst: u64, src: u64) -> u64 {
        let frame = RootFrame::<2>::new();
        unsafe { self.rt().chain.push_raw_unguarded(&frame.header as *const _ as *mut _) };
        frame.slots[0].set(src);
        frame.slots[1].set(dst);

        let src_capacity = self.table_capacity(src);

        let base = self.types().table_data_base_off;

        for i in 0..src_capacity {
            let src = frame.slots[0].get();
            let src_ptr = obj_ptr(src);
            let k = unsafe { gc_read_elem(src_ptr, base, i * 2) };
            if is_nil(k) {
                continue; // skip empty hash slots
            }
            let v = unsafe { gc_read_elem(src_ptr, base, i * 2 + 1) };
            let result = frame.slots[1].get();
            if self.table_get(result, k).is_none() {
                let new_result = self.table_set(result, k, v);
                frame.slots[1].set(new_result);
            }
        }
        let result = frame.slots[1].get();
        unsafe { self.rt().chain.pop_raw() };
        result
    }

    // ── Class helpers ────────────────────────────────────────────

    fn alloc_class(&mut self, name_val: u64) -> u64 {
        // Root `name_val` and `empty_table` across allocations.
        let frame = RootFrame::<2>::new();
        unsafe { self.rt().chain.push_raw_unguarded(&frame.header as *const _ as *mut _) };
        frame.slots[0].set(name_val);  // slot 0 = name

        let empty_table = self.alloc_table(0);
        frame.slots[1].set(empty_table);  // slot 1 = methods table

        let types = self.types();
        let class_id = types.class_id;
        let name_off = types.class_name_off;
        let super_off = types.class_super_off;
        let methods_off = types.class_methods_off;

        let ptr = self.gc_alloc(class_id, 0);
        let name_val = frame.slots[0].get();
        let empty_table = frame.slots[1].get();
        unsafe {
            gc_write_field(ptr, name_off, name_val);
            gc_write_field(ptr, super_off, nil_val());
            gc_write_field(ptr, methods_off, empty_table);
            self.rt().chain.pop_raw();
        }
        obj_val(ptr)
    }

    fn class_name(&self, class: u64) -> u64 {
        let ptr = obj_ptr(class);
        unsafe { gc_read_field(ptr, self.types().class_name_off) }
    }

    fn class_superclass(&self, class: u64) -> u64 {
        let ptr = obj_ptr(class);
        unsafe { gc_read_field(ptr, self.types().class_super_off) }
    }

    fn class_methods_table(&self, class: u64) -> u64 {
        let ptr = obj_ptr(class);
        unsafe { gc_read_field(ptr, self.types().class_methods_off) }
    }

    fn set_class_methods_table(&self, class: u64, table: u64) {
        let ptr = obj_ptr(class);
        unsafe { gc_write_field(ptr, self.types().class_methods_off, table); }
    }

    fn class_get_method(&self, class: u64, key: u64) -> Option<u64> {
        let table = self.class_methods_table(class);
        self.table_get(table, key)
    }

    fn class_set_method(&mut self, class: u64, key: u64, value: u64) {
        let frame = RootFrame::<1>::new();
        unsafe { self.rt().chain.push_raw_unguarded(&frame.header as *const _ as *mut _) };
        frame.slots[0].set(class);

        let table = self.class_methods_table(class);
        let new_table = self.table_set(table, key, value);

        let class = frame.slots[0].get();
        unsafe { self.rt().chain.pop_raw() };
        if new_table != table {
            self.set_class_methods_table(class, new_table);
        }
    }

    fn set_class_superclass(&mut self, class: u64, super_val: u64) {
        let ptr = obj_ptr(class);
        unsafe { gc_write_field(ptr, self.types().class_super_off, super_val); }
    }

    // ── Instance helpers ─────────────────────────────────────────

    fn alloc_instance(&mut self, class: u64) -> u64 {
        // Root `class` and `empty_table` across allocations.
        let frame = RootFrame::<2>::new();
        unsafe { self.rt().chain.push_raw_unguarded(&frame.header as *const _ as *mut _) };
        frame.slots[0].set(class);  // slot 0 = class

        let empty_table = self.alloc_table(0);
        frame.slots[1].set(empty_table);  // slot 1 = table

        let types = self.types();
        let instance_id = types.instance_id;
        let class_off = types.instance_class_off;
        let fields_off = types.instance_fields_off;

        let ptr = self.gc_alloc(instance_id, 0);
        // Re-read rooted values (GC may have moved them).
        let class = frame.slots[0].get();
        let empty_table = frame.slots[1].get();
        unsafe {
            gc_write_field(ptr, class_off, class);
            gc_write_field(ptr, fields_off, empty_table);
            self.rt().chain.pop_raw();
        }
        obj_val(ptr)
    }

    fn instance_class(&self, inst: u64) -> u64 {
        let ptr = obj_ptr(inst);
        unsafe { gc_read_field(ptr, self.types().instance_class_off) }
    }

    fn instance_fields_table(&self, inst: u64) -> u64 {
        let ptr = obj_ptr(inst);
        unsafe { gc_read_field(ptr, self.types().instance_fields_off) }
    }

    fn set_instance_fields_table(&self, inst: u64, table: u64) {
        let ptr = obj_ptr(inst);
        unsafe { gc_write_field(ptr, self.types().instance_fields_off, table); }
    }

    fn instance_get_field(&self, inst: u64, key: u64) -> Option<u64> {
        let table = self.instance_fields_table(inst);
        self.table_get(table, key)
    }

    fn instance_set_field(&mut self, inst: u64, key: u64, value: u64) {
        let frame = RootFrame::<1>::new();
        unsafe { self.rt().chain.push_raw_unguarded(&frame.header as *const _ as *mut _) };
        frame.slots[0].set(inst);

        let table = self.instance_fields_table(inst);
        let new_table = self.table_set(table, key, value);

        let inst = frame.slots[0].get();
        unsafe { self.rt().chain.pop_raw() };
        if new_table != table {
            self.set_instance_fields_table(inst, new_table);
        }
    }

    // ── BoundMethod helpers ──────────────────────────────────────

    fn alloc_bound_method(&mut self, receiver: u64, method: u64) -> u64 {
        let frame = RootFrame::<2>::new();
        unsafe { self.rt().chain.push_raw_unguarded(&frame.header as *const _ as *mut _) };
        frame.slots[0].set(receiver);
        frame.slots[1].set(method);

        let types = self.types();
        let bm_id = types.bound_method_id;
        let recv_off = types.bound_receiver_off;
        let method_off = types.bound_method_off;

        let ptr = self.gc_alloc(bm_id, 0);
        let receiver = frame.slots[0].get();
        let method = frame.slots[1].get();
        unsafe {
            gc_write_field(ptr, recv_off, receiver);
            gc_write_field(ptr, method_off, method);
            self.rt().chain.pop_raw();
        }
        obj_val(ptr)
    }

    fn bound_receiver(&self, bm: u64) -> u64 {
        let ptr = obj_ptr(bm);
        unsafe { gc_read_field(ptr, self.types().bound_receiver_off) }
    }

    fn bound_method_closure(&self, bm: u64) -> u64 {
        let ptr = obj_ptr(bm);
        unsafe { gc_read_field(ptr, self.types().bound_method_off) }
    }

    // ── NativeFn helpers ─────────────────────────────────────────

    fn alloc_native_fn(&mut self, name_val: u64, func_ptr: u64) -> u64 {
        let frame = RootFrame::<1>::new();
        unsafe { self.rt().chain.push_raw_unguarded(&frame.header as *const _ as *mut _) };
        frame.slots[0].set(name_val);

        let types = self.types();
        let nf_id = types.native_fn_id;
        let name_off = types.native_name_off;
        let fp_off = types.native_func_ptr_off;

        let ptr = self.gc_alloc(nf_id, 0);
        let name_val = frame.slots[0].get();
        unsafe {
            gc_write_field(ptr, name_off, name_val);
            gc_write_field(ptr, fp_off, func_ptr);
            self.rt().chain.pop_raw();
        }
        obj_val(ptr)
    }

    fn native_func_ptr(&self, nf: u64) -> u64 {
        let ptr = obj_ptr(nf);
        unsafe { gc_read_field(ptr, self.types().native_func_ptr_off) }
    }

    // ── Value formatting ─────────────────────────────────────────

    fn value_to_string(&self, v: u64) -> String {
        if is_nil(v) {
            "nil".to_string()
        } else if is_bool(v) {
            if as_bool(v) { "true".to_string() } else { "false".to_string() }
        } else if is_number(v) {
            format_number(as_number(v))
        } else if is_obj(v) {
            let tag = self.obj_type_tag(v);
            match tag {
                0 => self.read_string(v), // String
                1 => { // Closure
                    let name_val = self.closure_name(v);
                    if is_obj(name_val) {
                        format!("<fn {}>", self.read_string(name_val))
                    } else {
                        "<fn>".to_string()
                    }
                }
                2 => "<upvalue>".to_string(), // Upvalue
                3 => { // Class
                    let name_val = self.class_name(v);
                    if is_obj(name_val) {
                        self.read_string(name_val)
                    } else {
                        "<class>".to_string()
                    }
                }
                4 => { // Instance
                    let class = self.instance_class(v);
                    let name_val = self.class_name(class);
                    if is_obj(name_val) {
                        format!("{} instance", self.read_string(name_val))
                    } else {
                        "<instance>".to_string()
                    }
                }
                5 => { // BoundMethod
                    let method = self.bound_method_closure(v);
                    let name_val = self.closure_name(method);
                    if is_obj(name_val) {
                        format!("<fn {}>", self.read_string(name_val))
                    } else {
                        "<fn>".to_string()
                    }
                }
                6 => "<native fn>".to_string(), // NativeFn
                _ => format!("<obj>"),
            }
        } else {
            "unknown".to_string()
        }
    }
}

// ── JIT extern "C" functions ──────────────────────────────────────

extern "C" fn jit_lox_print(v: u64) {
    with_vm(|vm| {
        if vm.had_error { return; }
        let s = vm.value_to_string(v);
        println!("{}", s);
    });
}

extern "C" fn jit_lox_define_global(name_id: u64, value: u64) {
    with_vm(|vm| {
        let id = name_id as usize;
        vm.ensure_global(id);
        let rt = vm.gc_runtime.as_mut().unwrap();
        rt.globals.set(&rt.heap, id, value);
        rt.globals.define(id);
    });
}

extern "C" fn jit_lox_get_global(name_id: u64) -> u64 {
    with_vm(|vm| {
        let id = name_id as usize;
        let rt = vm.gc_runtime.as_ref().unwrap();
        if rt.globals.is_defined(id) {
            rt.globals.get(&rt.heap, id)
        } else {
            vm.global_error(id);
            nil_val()
        }
    })
}

extern "C" fn jit_lox_set_global(name_id: u64, value: u64) -> u64 {
    with_vm(|vm| {
        let id = name_id as usize;
        let rt = vm.gc_runtime.as_ref().unwrap();
        if rt.globals.is_defined(id) {
            rt.globals.set(&rt.heap, id, value);
            value
        } else {
            vm.global_error(id);
            nil_val()
        }
    })
}

extern "C" fn jit_lox_add(a: u64, b: u64) -> u64 {
    with_vm(|vm| {
        if is_obj(a) && is_obj(b) {
            let sa = vm.read_string(a);
            let sb = vm.read_string(b);
            return vm.intern_string(&format!("{}{}", sa, sb));
        }
        vm.runtime_error("Operands must be two numbers or two strings.");
        nil_val()
    })
}

extern "C" fn jit_lox_sub(_: u64, _: u64) -> u64 { with_vm(|vm| { vm.runtime_error("Operands must be numbers."); nil_val() }) }
extern "C" fn jit_lox_mul(_: u64, _: u64) -> u64 { with_vm(|vm| { vm.runtime_error("Operands must be numbers."); nil_val() }) }
extern "C" fn jit_lox_div(_: u64, _: u64) -> u64 { with_vm(|vm| { vm.runtime_error("Operands must be numbers."); nil_val() }) }
extern "C" fn jit_lox_neg(_: u64) -> u64 { with_vm(|vm| { vm.runtime_error("Operand must be a number."); nil_val() }) }
extern "C" fn jit_lox_eq(a: u64, b: u64) -> u64 { bool_val(values_equal(a, b)) }
extern "C" fn jit_lox_lt(_: u64, _: u64) -> u64 { with_vm(|vm| { vm.runtime_error("Operands must be numbers."); nil_val() }) }
extern "C" fn jit_lox_gt(_: u64, _: u64) -> u64 { with_vm(|vm| { vm.runtime_error("Operands must be numbers."); nil_val() }) }
extern "C" fn jit_lox_not(v: u64) -> u64 { bool_val(is_falsey(v)) }

extern "C" fn jit_lox_clock() -> u64 {
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
    number_val(t)
}

// ── Upvalue / Closure JIT externs ─────────────────────────────────

extern "C" fn jit_lox_alloc_upvalue(init: u64) -> u64 {
    with_vm(|vm| vm.alloc_upvalue(init))
}

extern "C" fn jit_lox_get_upvalue(cell: u64) -> u64 {
    with_vm(|vm| vm.get_upvalue(cell))
}

extern "C" fn jit_lox_set_upvalue(cell: u64, val: u64) {
    with_vm(|vm| vm.set_upvalue(cell, val));
}

extern "C" fn jit_lox_make_closure(func_idx: u64, num_upvalues: u64, arity: u64, name_id: u64) -> u64 {
    with_vm(|vm| {
        let name_val = name_id;
        vm.alloc_closure(func_idx, num_upvalues as usize, arity, name_val)
    })
}

extern "C" fn jit_lox_closure_upvalue(closure: u64, idx: u64) -> u64 {
    with_vm(|vm| vm.closure_upvalue(closure, idx as usize))
}

extern "C" fn jit_lox_set_closure_upvalue(closure: u64, idx: u64, cell: u64) {
    with_vm(|vm| vm.set_closure_upvalue(closure, idx as usize, cell));
}

extern "C" fn jit_lox_closure_func_ptr(closure: u64) -> u64 {
    with_vm(|vm| {
        let tag = vm.obj_type_tag(closure);
        match tag {
            1 => { // Closure
                let idx = vm.closure_func_idx(closure) as usize;
                if idx < vm.jit_call_table.len() {
                    vm.jit_call_table[idx] as u64
                } else {
                    idx as u64
                }
            }
            6 => { // NativeFn
                vm.native_func_ptr(closure)
            }
            _ => 0,
        }
    })
}

extern "C" fn jit_lox_obj_type(v: u64) -> u64 {
    with_vm(|vm| vm.obj_type_tag(v))
}

// ── Class JIT externs ─────────────────────────────────────────────

extern "C" fn jit_lox_make_class(name_id: u64) -> u64 {
    with_vm(|vm| {
        let name_val = name_id;
        vm.alloc_class(name_val)
    })
}

extern "C" fn jit_lox_make_native_fn(name_id: u64) -> u64 {
    with_vm(|vm| {
        let name_val = name_id;
        // Store the clock function pointer for native fns
        vm.alloc_native_fn(name_val, jit_lox_clock as u64)
    })
}

extern "C" fn jit_lox_check_arity(_callee: u64, expected: u64, got: u64) {
    with_vm(|vm| {
        if expected != got {
            vm.runtime_error(&format!("Expected {} arguments but got {}.", expected, got));
        }
    });
}

extern "C" fn jit_lox_call_non_callable() {
    with_vm(|vm| {
        vm.runtime_error("Can only call functions and classes.");
    });
}

extern "C" fn jit_lox_get_closure_arity(callee: u64) -> u64 {
    with_vm(|vm| {
        if !is_obj(callee) { return 0; }
        let tag = vm.obj_type_tag(callee);
        match tag {
            1 => vm.closure_arity(callee), // Closure
            6 => 0, // NativeFn (clock takes 0 args)
            _ => 0,
        }
    })
}

extern "C" fn jit_lox_get_class_init_arity(callee: u64) -> u64 {
    with_vm(|vm| {
        let init_name = vm.compile_string_id("init");
        if let Some(closure_val) = vm.class_get_method(callee, init_name) {
            return vm.closure_arity(closure_val);
        }
        255 // sentinel: no init
    })
}

extern "C" fn jit_lox_get_bound_arity(callee: u64) -> u64 {
    with_vm(|vm| {
        let method = vm.bound_method_closure(callee);
        vm.closure_arity(method)
    })
}

extern "C" fn jit_lox_class_inherit(class: u64, super_val: u64) {
    with_vm(|vm| {
        if !is_obj(super_val) || !vm.is_class(super_val) {
            vm.runtime_error("Superclass must be a class.");
            return;
        }
        let super_table = vm.class_methods_table(super_val);
        let own_table = vm.class_methods_table(class);
        let merged = vm.table_merge(own_table, super_table);
        vm.set_class_methods_table(class, merged);
        vm.set_class_superclass(class, super_val);
    });
}

extern "C" fn jit_lox_class_add_method(class: u64, name_id: u64, method_closure: u64) {
    with_vm(|vm| {
        let name_val = name_id;
        vm.class_set_method(class, name_val, method_closure);
    });
}

extern "C" fn jit_lox_construct_instance(class: u64) -> u64 {
    with_vm(|vm| vm.alloc_instance(class))
}

extern "C" fn jit_lox_class_init_ptr(class: u64) -> u64 {
    with_vm(|vm| {
        let init_name = vm.compile_string_id("init");
        if let Some(closure_val) = vm.class_get_method(class, init_name) {
            let idx = vm.closure_func_idx(closure_val) as usize;
            if idx < vm.jit_call_table.len() {
                return vm.jit_call_table[idx] as u64;
            }
            return idx as u64;
        }
        0
    })
}

extern "C" fn jit_lox_class_init_closure(class: u64) -> u64 {
    with_vm(|vm| {
        let init_name = vm.compile_string_id("init");
        vm.class_get_method(class, init_name).unwrap_or(nil_val())
    })
}

// ── Property JIT externs ──────────────────────────────────────────

extern "C" fn jit_lox_get_property(obj: u64, name_id: u64) -> u64 {
    with_vm(|vm| {
        let name_val = name_id;

        if !is_obj(obj) || !vm.is_instance(obj) {
            vm.runtime_error("Only instances have properties.");
            return nil_val();
        }

        // Check fields first
        if let Some(val) = vm.instance_get_field(obj, name_val) {
            return val;
        }

        // Check methods on the class
        let class = vm.instance_class(obj);
        if let Some(method) = vm.class_get_method(class, name_val) {
            return vm.alloc_bound_method(obj, method);
        }

        let prop_name = vm.string_text(name_id as usize);
        vm.runtime_error(&format!("Undefined property '{}'.", prop_name));
        nil_val()
    })
}

/// Fast field-only get: returns field value or nil (no method fallback).
extern "C" fn jit_lox_get_field(obj: u64, name_id: u64) -> u64 {
    with_vm(|vm| {
        let name_val = name_id;
        if !is_obj(obj) || !vm.is_instance(obj) {
            return nil_val();
        }
        vm.instance_get_field(obj, name_val).unwrap_or_else(nil_val)
    })
}

extern "C" fn jit_lox_set_property(obj: u64, name_id: u64, val: u64) -> u64 {
    with_vm(|vm| {
        let name_val = name_id;

        if !is_obj(obj) || !vm.is_instance(obj) {
            vm.runtime_error("Only instances have fields.");
            return nil_val();
        }

        vm.instance_set_field(obj, name_val, val);
        val
    })
}

// ── Invoke (optimized method call without BoundMethod allocation) ──

/// Invoke lookup: check fields first (Lox semantics: fields shadow methods).
/// Returns:
///   - Field found: (field_value, 0) via two-call convention
///   - Method found: (method_closure, 1)
///   - Not found: (nil, 2)
/// Uses a two-call pattern: invoke_lookup returns the value,
/// invoke_lookup_kind returns 0=field, 1=method, 2=not_found.
extern "C" fn jit_lox_invoke_lookup(obj: u64, name_id: u64) -> u64 {
    with_vm(|vm| {
        let name_val = name_id;
        if !is_obj(obj) || !vm.is_instance(obj) {
            vm.last_invoke_kind = 2;
            return nil_val();
        }
        // Fields shadow methods
        if let Some(val) = vm.instance_get_field(obj, name_val) {
            vm.last_invoke_kind = 0;
            return val;
        }
        let class = vm.instance_class(obj);
        if let Some(method) = vm.class_get_method(class, name_val) {
            vm.last_invoke_kind = 1;
            return method;
        }
        vm.last_invoke_kind = 2;
        nil_val()
    })
}

extern "C" fn jit_lox_invoke_kind() -> u64 {
    with_vm(|vm| vm.last_invoke_kind)
}

extern "C" fn jit_lox_call_table_base() -> u64 {
    with_vm(|vm| {
        if vm.jit_call_table.is_empty() { 0 } else { vm.jit_call_table.as_ptr() as u64 }
    })
}

/// Look up a method on an instance's class (NOT fields).
/// Returns the method closure if found, nil otherwise.
/// Does NOT allocate a BoundMethod.
extern "C" fn jit_lox_invoke_func_ptr(closure: u64) -> u64 {
    with_vm(|vm| {
        let idx = vm.closure_func_idx(closure) as usize;
        if idx < vm.jit_call_table.len() {
            vm.jit_call_table[idx] as u64
        } else {
            idx as u64
        }
    })
}

/// Combined fast-path invoke: lookup method, check arity, return closure.
/// Returns the method closure (NaN-boxed) if method found with matching arity.
/// Returns nil if slow path needed (field, not found, or arity mismatch).
/// Stores closure in last_invoke_closure, kind in last_invoke_kind for slow path.
extern "C" fn jit_lox_invoke_fast(obj: u64, name_id: u64, num_args: u64) -> u64 {
    with_vm(|vm| {
        let name_val = name_id;
        if !is_obj(obj) || !vm.is_instance(obj) {
            vm.last_invoke_kind = 2;
            return nil_val();
        }
        // Fields shadow methods — check fields first
        if let Some(val) = vm.instance_get_field(obj, name_val) {
            vm.last_invoke_kind = 0;
            vm.last_invoke_closure = val;
            return nil_val();
        }
        let class = vm.instance_class(obj);
        if let Some(method) = vm.class_get_method(class, name_val) {
            vm.last_invoke_closure = method;
            vm.last_invoke_kind = 1;
            let arity = vm.closure_arity(method);
            if arity != num_args {
                return nil_val();
            }
            // Fast path: return the closure directly
            return method;
        }
        vm.last_invoke_kind = 2;
        nil_val()
    })
}

/// Retrieve the closure from last invoke_fast call (for slow path).
extern "C" fn jit_lox_invoke_closure() -> u64 {
    with_vm(|vm| vm.last_invoke_closure)
}

/// Combined instantiate: allocate instance + look up init method in one call.
/// Returns the new instance. Stores init info in globals:
///   last_invoke_closure = init closure (or nil)
///   last_init_arity = init arity (or 255 if no init)
///   last_init_func_ptr = init func_ptr (or 0 if no init)
extern "C" fn jit_lox_instantiate(class: u64) -> u64 {
    with_vm(|vm| {
        let instance = vm.alloc_instance(class);
        let init_name = vm.compile_string_id("init");
        if let Some(closure_val) = vm.class_get_method(class, init_name) {
            vm.last_invoke_closure = closure_val;
            vm.last_init_arity = vm.closure_arity(closure_val);
            let idx = vm.closure_func_idx(closure_val) as usize;
            vm.last_init_func_ptr = if idx < vm.jit_call_table.len() {
                vm.jit_call_table[idx] as u64
            } else {
                idx as u64
            };
        } else {
            vm.last_invoke_closure = nil_val();
            vm.last_init_arity = 255;
            vm.last_init_func_ptr = 0;
        }
        instance
    })
}

extern "C" fn jit_lox_last_init_arity() -> u64 {
    with_vm(|vm| vm.last_init_arity)
}

extern "C" fn jit_lox_last_init_func_ptr() -> u64 {
    with_vm(|vm| vm.last_init_func_ptr)
}

// ── Bound method / Super JIT externs ─────────────────────────────

extern "C" fn jit_lox_get_super(this: u64, class_val: u64, method_name_id: u64) -> u64 {
    with_vm(|vm| {
        let method_name_str = vm.string_text(method_name_id as usize);
        let method_name = method_name_id;

        let superclass = vm.class_superclass(class_val);
        if is_nil(superclass) { return nil_val(); }

        if let Some(method_closure) = vm.class_get_method(superclass, method_name) {
            return vm.alloc_bound_method(this, method_closure);
        }

        vm.runtime_error(&format!("Undefined property '{}'.", method_name_str));
        nil_val()
    })
}

extern "C" fn jit_lox_bound_receiver(bm: u64) -> u64 {
    with_vm(|vm| vm.bound_receiver(bm))
}

extern "C" fn jit_lox_bound_method_closure(bm: u64) -> u64 {
    with_vm(|vm| vm.bound_method_closure(bm))
}

extern "C" fn jit_lox_bound_closure_func_ptr(bm: u64) -> u64 {
    with_vm(|vm| {
        let method = vm.bound_method_closure(bm);
        let idx = vm.closure_func_idx(method) as usize;
        if idx < vm.jit_call_table.len() {
            vm.jit_call_table[idx] as u64
        } else {
            idx as u64
        }
    })
}

extern "C" fn jit_lox_resolve_string(id: u64) -> u64 {
    with_vm(|vm| vm.resolve_string(id as usize))
}

/// Build JIT extern pointers matching module.func_table order.
fn build_jit_externs(module: &Module) -> Vec<*const u8> {
    let mut ptrs = Vec::new();
    for def in &module.func_table {
        if let FuncDef::Extern(ext) = def {
            let ptr: *const u8 = match ext.name.as_str() {
                "lox_add" => jit_lox_add as *const u8,
                "lox_sub" => jit_lox_sub as *const u8,
                "lox_mul" => jit_lox_mul as *const u8,
                "lox_div" => jit_lox_div as *const u8,
                "lox_neg" => jit_lox_neg as *const u8,
                "lox_eq" => jit_lox_eq as *const u8,
                "lox_lt" => jit_lox_lt as *const u8,
                "lox_gt" => jit_lox_gt as *const u8,
                "lox_not" => jit_lox_not as *const u8,
                "lox_print" => jit_lox_print as *const u8,
                "lox_define_global" => jit_lox_define_global as *const u8,
                "lox_get_global" => jit_lox_get_global as *const u8,
                "lox_set_global" => jit_lox_set_global as *const u8,
                "lox_clock" => jit_lox_clock as *const u8,
                "lox_alloc_upvalue" => jit_lox_alloc_upvalue as *const u8,
                "lox_get_upvalue" => jit_lox_get_upvalue as *const u8,
                "lox_set_upvalue" => jit_lox_set_upvalue as *const u8,
                "lox_make_closure" => jit_lox_make_closure as *const u8,
                "lox_check_arity" => jit_lox_check_arity as *const u8,
                "lox_call_non_callable" => jit_lox_call_non_callable as *const u8,
                "lox_get_closure_arity" => jit_lox_get_closure_arity as *const u8,
                "lox_get_class_init_arity" => jit_lox_get_class_init_arity as *const u8,
                "lox_get_bound_arity" => jit_lox_get_bound_arity as *const u8,
                "lox_closure_upvalue" => jit_lox_closure_upvalue as *const u8,
                "lox_closure_func_ptr" => jit_lox_closure_func_ptr as *const u8,
                "lox_set_closure_upvalue" => jit_lox_set_closure_upvalue as *const u8,
                "lox_obj_type" => jit_lox_obj_type as *const u8,
                "lox_make_class" => jit_lox_make_class as *const u8,
                "lox_class_inherit" => jit_lox_class_inherit as *const u8,
                "lox_class_add_method" => jit_lox_class_add_method as *const u8,
                "lox_construct_instance" => jit_lox_construct_instance as *const u8,
                "lox_class_init_ptr" => jit_lox_class_init_ptr as *const u8,
                "lox_class_init_closure" => jit_lox_class_init_closure as *const u8,
                "lox_get_property" => jit_lox_get_property as *const u8,
                "lox_get_field" => jit_lox_get_field as *const u8,
                "lox_set_property" => jit_lox_set_property as *const u8,
                "lox_get_super" => jit_lox_get_super as *const u8,
                "lox_bound_receiver" => jit_lox_bound_receiver as *const u8,
                "lox_bound_method_closure" => jit_lox_bound_method_closure as *const u8,
                "lox_bound_closure_func_ptr" => jit_lox_bound_closure_func_ptr as *const u8,
                "lox_make_native_fn" => jit_lox_make_native_fn as *const u8,
                "lox_resolve_string" => jit_lox_resolve_string as *const u8,
                "lox_invoke_lookup" => jit_lox_invoke_lookup as *const u8,
                "lox_invoke_func_ptr" => jit_lox_invoke_func_ptr as *const u8,
                "lox_invoke_kind" => jit_lox_invoke_kind as *const u8,
                "lox_call_table_base" => jit_lox_call_table_base as *const u8,
                "lox_invoke_fast" => jit_lox_invoke_fast as *const u8,
                "lox_invoke_closure" => jit_lox_invoke_closure as *const u8,
                "lox_instantiate" => jit_lox_instantiate as *const u8,
                "lox_last_init_arity" => jit_lox_last_init_arity as *const u8,
                "lox_last_init_func_ptr" => jit_lox_last_init_func_ptr as *const u8,
                "__gc_alloc__" => jit_gc_alloc as *const u8,
                other => panic!("unknown extern for JIT: {other}"),
            };
            ptrs.push(ptr);
        }
    }
    ptrs
}

extern "C" fn jit_gc_alloc(type_id: u64, varlen_len: u64) -> u64 {
    with_vm(|vm| {
        // Safe to trigger GC here — called from IR, interpreter roots are synced
        let ptr = vm.rt().alloc_with_gc(type_id as usize, varlen_len as usize);
        ptr as u64
    })
}

// ── VM implementation ─────────────────────────────────────────────

impl VM {
    pub fn new() -> Self {
        VM {
            had_error: false,
            use_jit: false,
            jit_call_table: Vec::new(),
            gc_runtime: None,
            gc_types: None,
            compile_strings: Vec::new(),
            last_invoke_kind: 0,
            last_invoke_closure: 0,
            last_init_arity: 255,
            last_init_func_ptr: 0,
        }
    }

    pub fn reset(&mut self) {
        self.had_error = false;
        self.jit_call_table.clear();
        self.gc_runtime = None;
        self.gc_types = None;
        self.compile_strings.clear();
    }

    fn ensure_global(&mut self, id: usize) {
        let rt = self.gc_runtime.as_mut().unwrap();
        rt.globals.ensure(&rt.heap, id);
    }

    fn global_error(&mut self, id: usize) {
        let name = self.string_text(id);
        self.runtime_error(&format!("Undefined variable '{}'.", name));
    }

    pub fn interpret(&mut self, source: &str) -> InterpretResult {
        let program = match Parser::parse(source) {
            Some(p) => p,
            None => return InterpretResult::CompileError,
        };

        let mut lowered = lower::lower(&program);

        // Optimize to fixpoint
        for func in &mut lowered.module.functions {
            opt::optimize(func);
        }

        // Debug IR dump
        if std::env::var("DUMP_IR").is_ok() {
            for func in &lowered.module.functions {
                eprintln!("{}", func);
                eprintln!("---");
            }
        }

        // Initialize GC runtime with semi-space collector
        let type_infos = lowered.gc_types.type_infos.clone();
        self.gc_types = Some(lowered.gc_types);
        self.compile_strings = lowered.strings.clone();
        // 64MB heap (32MB each semi-space)
        self.gc_runtime = Some(LoxGcRuntime::new(32 * 1024 * 1024, type_infos));

        // Intern compile-time strings
        for s in &lowered.strings {
            self.intern_string(s);
        }

        self.had_error = false;

        if self.use_jit {
            self.run_jit(&lowered.module, lowered.entry)
        } else {
            self.run_interp(&lowered.module, lowered.entry)
        }
    }

    fn run_interp(&mut self, module: &Module, entry: dynlang::FuncRef) -> InterpretResult {
        // Safety: gc_runtime outlives the interpreter. We use a raw pointer to
        // avoid borrowing self, since bind_runtime needs &mut self.
        unsafe { RAW_VM = self as *mut VM; }
        let rt: &LoxGcRuntime = unsafe { &*(self.gc_runtime.as_ref().unwrap() as *const _) };
        let mut interp = ModuleInterpreter::<NanBox, _>::new(module, rt);
        self.bind_runtime(&mut interp);

        match interp.run(entry, &[nil_val()]) {
            Ok(InterpResult::Value(_)) | Ok(InterpResult::Void) => {
                if self.had_error { InterpretResult::RuntimeError }
                else { InterpretResult::Ok }
            }
            Ok(InterpResult::Deopt { .. }) => InterpretResult::RuntimeError,
            Err(e) => {
                eprintln!("Internal error: {:?}", e);
                InterpretResult::RuntimeError
            }
        }
    }

    fn run_jit(&mut self, module: &Module, entry: dynlang::FuncRef) -> InterpretResult {
        use dynruntime::{StackMapJitTransport, JitSafepointSession, active_jit_safepoint_handler};

        let externs = build_jit_externs(module);

        let jit = JitModule::compile_batch::<NanBox>(
            module, &externs, Some(active_jit_safepoint_handler as u64),
        );

        self.jit_call_table = jit.call_table().to_vec();

        if std::env::var("LOX_DUMP_JIT").is_ok() {
            jit.dump_code();
        }

        // Safety: gc_runtime outlives the session.
        let heap: &Heap = unsafe { &*((&self.gc_runtime.as_ref().unwrap().heap) as *const _) };
        let safepoints = jit.all_safepoints();
        let session = JitSafepointSession::<LoxPtrPolicy, _>::new(
            heap, StackMapJitTransport, &safepoints,
        ).with_gc_threshold(0.75);

        unsafe { RAW_VM = self as *mut VM; }
        let result = session.with_installed(|| {
            jit.call_outcome(entry, &[nil_val()])
        });
        unsafe { RAW_VM = std::ptr::null_mut(); }
        self.jit_call_table.clear();

        match result {
            JitOutcome::Value(_) | JitOutcome::Void => {
                if self.had_error { InterpretResult::RuntimeError }
                else { InterpretResult::Ok }
            }
            _ => { eprintln!("JIT error: {:?}", result); InterpretResult::RuntimeError }
        }
    }

    fn runtime_error(&mut self, msg: &str) {
        eprintln!("{}", msg);
        self.had_error = true;
    }

    fn bind_runtime<'a>(&mut self, interp: &mut ModuleInterpreter<'a, NanBox, LoxGcRuntime>) {
        let rt = self as *mut VM;

        interp.bind_by_name("lox_print", move |args| {
            let vm = unsafe { &*rt };
            if !vm.had_error {
                println!("{}", vm.value_to_string(args[0]));
            }
            ExternCallResult::Value(None)
        });

        interp.bind_by_name("lox_define_global", move |args| {
            let vm = unsafe { &mut *rt };
            let id = args[0] as usize;
            vm.ensure_global(id);
            let gcrt = vm.gc_runtime.as_mut().unwrap();
            gcrt.globals.set(&gcrt.heap, id, args[1]);
            gcrt.globals.define(id);
            ExternCallResult::Value(None)
        });

        interp.bind_by_name("lox_get_global", move |args| {
            let vm = unsafe { &mut *rt };
            let id = args[0] as usize;
            let gcrt = vm.gc_runtime.as_ref().unwrap();
            if gcrt.globals.is_defined(id) {
                ExternCallResult::Value(Some(gcrt.globals.get(&gcrt.heap, id)))
            } else {
                vm.global_error(id);
                ExternCallResult::Value(Some(nil_val()))
            }
        });

        interp.bind_by_name("lox_set_global", move |args| {
            let vm = unsafe { &mut *rt };
            let id = args[0] as usize;
            let val = args[1];
            let gcrt = vm.gc_runtime.as_ref().unwrap();
            if gcrt.globals.is_defined(id) {
                gcrt.globals.set(&gcrt.heap, id, val);
                ExternCallResult::Value(Some(val))
            } else {
                vm.global_error(id);
                ExternCallResult::Value(Some(nil_val()))
            }
        });

        // Arithmetic slow paths
        interp.bind_by_name("lox_add", move |args| {
            let vm = unsafe { &mut *rt };
            let (a, b) = (args[0], args[1]);
            if is_obj(a) && is_obj(b) {
                let sa = vm.read_string(a);
                let sb = vm.read_string(b);
                let r = vm.intern_string(&format!("{}{}", sa, sb));
                return ExternCallResult::Value(Some(r));
            }
            vm.runtime_error("Operands must be two numbers or two strings.");
            ExternCallResult::Value(Some(nil_val()))
        });
        interp.bind_by_name("lox_sub", move |_| { let vm = unsafe{&mut*rt}; vm.runtime_error("Operands must be numbers."); ExternCallResult::Value(Some(nil_val())) });
        interp.bind_by_name("lox_mul", move |_| { let vm = unsafe{&mut*rt}; vm.runtime_error("Operands must be numbers."); ExternCallResult::Value(Some(nil_val())) });
        interp.bind_by_name("lox_div", move |_| { let vm = unsafe{&mut*rt}; vm.runtime_error("Operands must be numbers."); ExternCallResult::Value(Some(nil_val())) });
        interp.bind_by_name("lox_neg", move |_| { let vm = unsafe{&mut*rt}; vm.runtime_error("Operand must be a number."); ExternCallResult::Value(Some(nil_val())) });
        interp.bind_by_name("lox_eq", |args| { ExternCallResult::Value(Some(bool_val(values_equal(args[0], args[1])))) });
        interp.bind_by_name("lox_lt", move |_| { let vm = unsafe{&mut*rt}; vm.runtime_error("Operands must be numbers."); ExternCallResult::Value(Some(nil_val())) });
        interp.bind_by_name("lox_gt", move |_| { let vm = unsafe{&mut*rt}; vm.runtime_error("Operands must be numbers."); ExternCallResult::Value(Some(nil_val())) });
        interp.bind_by_name("lox_not", |args| { ExternCallResult::Value(Some(bool_val(is_falsey(args[0])))) });
        interp.bind_by_name("lox_clock", |_| {
            let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
            ExternCallResult::Value(Some(number_val(t)))
        });

        // Upvalue / closure
        interp.bind_by_name("lox_alloc_upvalue", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.alloc_upvalue(args[0])))
        });
        interp.bind_by_name("lox_get_upvalue", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.get_upvalue(args[0])))
        });
        interp.bind_by_name("lox_set_upvalue", move |args| {
            let vm = unsafe{&mut*rt};
            vm.set_upvalue(args[0], args[1]);
            ExternCallResult::Value(None)
        });
        interp.bind_by_name("lox_make_closure", move |args| {
            let vm = unsafe{&mut*rt};
            let name_val = vm.resolve_string(args[3] as usize);
            let v = vm.alloc_closure(args[0], args[1] as usize, args[2], name_val);
            ExternCallResult::Value(Some(v))
        });
        interp.bind_by_name("lox_closure_upvalue", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.closure_upvalue(args[0], args[1] as usize)))
        });
        interp.bind_by_name("lox_set_closure_upvalue", move |args| {
            let vm = unsafe{&mut*rt};
            vm.set_closure_upvalue(args[0], args[1] as usize, args[2]);
            ExternCallResult::Value(None)
        });
        interp.bind_by_name("lox_closure_func_ptr", move |args| {
            let vm = unsafe{&mut*rt};
            let tag = vm.obj_type_tag(args[0]);
            let v = match tag {
                1 => vm.closure_func_idx(args[0]), // Closure: return func_table_idx for interp
                6 => 0, // NativeFn: not used in interp mode
                _ => 0,
            };
            ExternCallResult::Value(Some(v))
        });
        interp.bind_by_name("lox_check_arity", move |args| {
            let vm = unsafe{&mut*rt};
            let expected = args[1];
            let got = args[2];
            if expected != got {
                vm.runtime_error(&format!("Expected {} arguments but got {}.", expected, got));
            }
            ExternCallResult::Value(None)
        });
        interp.bind_by_name("lox_call_non_callable", move |_args| {
            let vm = unsafe{&mut*rt};
            vm.runtime_error("Can only call functions and classes.");
            ExternCallResult::Value(None)
        });
        interp.bind_by_name("lox_get_closure_arity", move |args| {
            let vm = unsafe{&mut*rt};
            if !is_obj(args[0]) { return ExternCallResult::Value(Some(0)); }
            let tag = vm.obj_type_tag(args[0]);
            let v = match tag {
                1 => vm.closure_arity(args[0]),
                6 => 0,
                _ => 0,
            };
            ExternCallResult::Value(Some(v))
        });
        interp.bind_by_name("lox_get_class_init_arity", move |args| {
            let vm = unsafe{&mut*rt};
            let init_name = vm.compile_string_id("init");
            let v = if let Some(closure_val) = vm.class_get_method(args[0], init_name) {
                vm.closure_arity(closure_val)
            } else { 255 };
            ExternCallResult::Value(Some(v))
        });
        interp.bind_by_name("lox_get_bound_arity", move |args| {
            let vm = unsafe{&mut*rt};
            let method = vm.bound_method_closure(args[0]);
            ExternCallResult::Value(Some(vm.closure_arity(method)))
        });
        interp.bind_by_name("lox_obj_type", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.obj_type_tag(args[0])))
        });

        // Class
        interp.bind_by_name("lox_make_class", move |args| {
            let vm = unsafe{&mut*rt};
            let name_val = vm.resolve_string(args[0] as usize);
            ExternCallResult::Value(Some(vm.alloc_class(name_val)))
        });
        interp.bind_by_name("lox_class_inherit", move |args| {
            let vm = unsafe{&mut*rt};
            if !is_obj(args[1]) || !vm.is_class(args[1]) {
                vm.runtime_error("Superclass must be a class.");
                return ExternCallResult::Value(None);
            }
            let super_table = vm.class_methods_table(args[1]);
            let own_table = vm.class_methods_table(args[0]);
            let merged = vm.table_merge(own_table, super_table);
            vm.set_class_methods_table(args[0], merged);
            vm.set_class_superclass(args[0], args[1]);
            ExternCallResult::Value(None)
        });
        interp.bind_by_name("lox_class_add_method", move |args| {
            let vm = unsafe{&mut*rt};
            vm.class_set_method(args[0], args[1], args[2]);
            ExternCallResult::Value(None)
        });
        interp.bind_by_name("lox_construct_instance", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.alloc_instance(args[0])))
        });
        interp.bind_by_name("lox_class_init_ptr", move |args| {
            let vm = unsafe{&mut*rt};
            let init_name = vm.compile_string_id("init");
            let v = if let Some(c) = vm.class_get_method(args[0], init_name) {
                vm.closure_func_idx(c)
            } else { 0 };
            ExternCallResult::Value(Some(v))
        });
        interp.bind_by_name("lox_class_init_closure", move |args| {
            let vm = unsafe{&mut*rt};
            let init_name = vm.compile_string_id("init");
            let v = vm.class_get_method(args[0], init_name).unwrap_or(nil_val());
            ExternCallResult::Value(Some(v))
        });

        // Combined instantiate
        interp.bind_by_name("lox_instantiate", move |args| {
            let vm = unsafe{&mut*rt};
            let instance = vm.alloc_instance(args[0]);
            let init_name = vm.compile_string_id("init");
            if let Some(closure_val) = vm.class_get_method(args[0], init_name) {
                vm.last_invoke_closure = closure_val;
                vm.last_init_arity = vm.closure_arity(closure_val);
                vm.last_init_func_ptr = vm.closure_func_idx(closure_val);
            } else {
                vm.last_invoke_closure = nil_val();
                vm.last_init_arity = 255;
                vm.last_init_func_ptr = 0;
            }
            ExternCallResult::Value(Some(instance))
        });
        interp.bind_by_name("lox_last_init_arity", move |_args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.last_init_arity))
        });
        interp.bind_by_name("lox_last_init_func_ptr", move |_args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.last_init_func_ptr))
        });

        // Properties
        interp.bind_by_name("lox_get_field", move |args| {
            let vm = unsafe{&mut*rt};
            if !is_obj(args[0]) || !vm.is_instance(args[0]) {
                return ExternCallResult::Value(Some(nil_val()));
            }
            let val = vm.instance_get_field(args[0], args[1]).unwrap_or_else(nil_val);
            ExternCallResult::Value(Some(val))
        });
        interp.bind_by_name("lox_get_property", move |args| {
            let vm = unsafe{&mut*rt};
            let name_id = args[1] as usize;
            let prop_name = vm.string_text(name_id);

            if !is_obj(args[0]) || !vm.is_instance(args[0]) {
                vm.runtime_error("Only instances have properties.");
                return ExternCallResult::Value(Some(nil_val()));
            }
            if let Some(val) = vm.instance_get_field(args[0], args[1]) {
                return ExternCallResult::Value(Some(val));
            }
            let class = vm.instance_class(args[0]);
            if let Some(method) = vm.class_get_method(class, args[1]) {
                let bm = vm.alloc_bound_method(args[0], method);
                return ExternCallResult::Value(Some(bm));
            }
            vm.runtime_error(&format!("Undefined property '{}'.", prop_name));
            ExternCallResult::Value(Some(nil_val()))
        });
        interp.bind_by_name("lox_set_property", move |args| {
            let vm = unsafe{&mut*rt};

            if !is_obj(args[0]) || !vm.is_instance(args[0]) {
                vm.runtime_error("Only instances have fields.");
                return ExternCallResult::Value(Some(nil_val()));
            }
            vm.instance_set_field(args[0], args[1], args[2]);
            ExternCallResult::Value(Some(args[2]))
        });

        // Super
        interp.bind_by_name("lox_get_super", move |args| {
            let vm = unsafe{&mut*rt};
            let method_name_str = vm.string_text(args[2] as usize);

            let superclass = vm.class_superclass(args[1]);
            if is_nil(superclass) { return ExternCallResult::Value(Some(nil_val())); }

            if let Some(mc) = vm.class_get_method(superclass, args[2]) {
                let bm = vm.alloc_bound_method(args[0], mc);
                return ExternCallResult::Value(Some(bm));
            }
            vm.runtime_error(&format!("Undefined property '{}'.", method_name_str));
            ExternCallResult::Value(Some(nil_val()))
        });

        // Native fn
        interp.bind_by_name("lox_make_native_fn", move |args| {
            let vm = unsafe{&mut*rt};
            let name_val = vm.resolve_string(args[0] as usize);
            let v = vm.alloc_native_fn(name_val, 0);
            ExternCallResult::Value(Some(v))
        });

        // Bound methods
        interp.bind_by_name("lox_bound_receiver", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.bound_receiver(args[0])))
        });
        interp.bind_by_name("lox_bound_method_closure", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.bound_method_closure(args[0])))
        });
        interp.bind_by_name("lox_bound_closure_func_ptr", move |args| {
            let vm = unsafe{&mut*rt};
            let method = vm.bound_method_closure(args[0]);
            ExternCallResult::Value(Some(vm.closure_func_idx(method)))
        });

        // Invoke (optimized method call)
        interp.bind_by_name("lox_invoke_lookup", move |args| {
            let vm = unsafe{&mut*rt};
            if !is_obj(args[0]) || !vm.is_instance(args[0]) {
                return ExternCallResult::Value(Some(nil_val()));
            }
            if let Some(val) = vm.instance_get_field(args[0], args[1]) {
                return ExternCallResult::Value(Some(val));
            }
            let class = vm.instance_class(args[0]);
            if let Some(method) = vm.class_get_method(class, args[1]) {
                return ExternCallResult::Value(Some(method));
            }
            ExternCallResult::Value(Some(nil_val()))
        });
        interp.bind_by_name("lox_invoke_func_ptr", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.closure_func_idx(args[0])))
        });
        interp.bind_by_name("lox_invoke_kind", move |_args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.last_invoke_kind))
        });
        interp.bind_by_name("lox_call_table_base", move |_args| {
            // Interpreter doesn't use code pointers — return 0.
            // The IR only uses call_table_base in call_indirect paths
            // that the interpreter handles differently.
            ExternCallResult::Value(Some(0))
        });

        // Combined fast-path invoke
        interp.bind_by_name("lox_invoke_fast", move |args| {
            let vm = unsafe{&mut*rt};
            let num_args = args[2];
            if !is_obj(args[0]) || !vm.is_instance(args[0]) {
                vm.last_invoke_kind = 2;
                return ExternCallResult::Value(Some(nil_val()));
            }
            if let Some(val) = vm.instance_get_field(args[0], args[1]) {
                vm.last_invoke_kind = 0;
                vm.last_invoke_closure = val;
                return ExternCallResult::Value(Some(nil_val()));
            }
            let class = vm.instance_class(args[0]);
            if let Some(method) = vm.class_get_method(class, args[1]) {
                vm.last_invoke_closure = method;
                vm.last_invoke_kind = 1;
                let arity = vm.closure_arity(method);
                if arity != num_args {
                    return ExternCallResult::Value(Some(nil_val()));
                }
                return ExternCallResult::Value(Some(method));
            }
            vm.last_invoke_kind = 2;
            ExternCallResult::Value(Some(nil_val()))
        });
        interp.bind_by_name("lox_invoke_closure", move |_args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.last_invoke_closure))
        });

        // String resolution
        interp.bind_by_name("lox_resolve_string", move |args| {
            let vm = unsafe{&mut*rt};
            ExternCallResult::Value(Some(vm.resolve_string(args[0] as usize)))
        });

        // GC alloc — safe to trigger GC here, interpreter roots are synced before extern calls
        interp.bind_by_name("__gc_alloc__", move |args| {
            let vm = unsafe{&mut*rt};
            let ptr = vm.rt().alloc_with_gc(args[0] as usize, args[1] as usize);
            ExternCallResult::Value(Some(ptr as u64))
        });
    }
}

