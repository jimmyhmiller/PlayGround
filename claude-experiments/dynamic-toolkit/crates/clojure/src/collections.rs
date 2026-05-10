//! Heap allocators / accessors for String, Keyword, Vector, Set.
//!
//! Layouts (mirror `types.rs` declaration order, header is `Compact`):
//!
//! ```text
//!   String   { hash: Raw64,    varlen_bytes }
//!   Keyword  { sym: Value,     hash: Raw64 }
//!   Vector   { root: Value, tail: Value, count: Raw64, shift: Raw64 }
//!     // v1: flat (no HAMT/RRB). `root` holds a flat varlen-values
//!     // node carrying every element; `tail` and `shift` unused.
//!   Set      { backing: Value }   // backing is a Vector of items.
//! ```
//!
//! The dynlang builder lays out value fields first, then raw64 fields,
//! then the varlen tail. Field offsets here mirror that.

use dynobj::roots::{Rooted, RootScope};
use dynobj::{Compact, ObjHeader};

use crate::host::with_host;
use crate::value::{self as v, NanBoxTag};

const HDR: usize = Compact::SIZE; // 8 bytes (see header.rs)

// ── String ──────────────────────────────────────────────────────────
//
// String { hash: Raw64; varlen_bytes }
// raw_data_offset = HDR
// hash at HDR
// varlen_count at HDR + 8
// bytes start at HDR + 16

const STR_HASH_OFFSET: usize = HDR;
const STR_VARLEN_COUNT_OFFSET: usize = HDR + 8;
const STR_BYTES_OFFSET: usize = HDR + 16;

pub fn alloc_string<'scope>(
    scope: &'scope RootScope<'_>,
    bytes: &[u8],
) -> Rooted<'scope, NanBoxTag> {
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.string.0;
        let raw = gc.alloc(type_id, bytes.len());
        assert!(!raw.is_null(), "alloc_string: GC alloc returned null");
        unsafe {
            (raw.add(STR_HASH_OFFSET) as *mut u64).write(string_hash_bytes(bytes));
            (raw.add(STR_VARLEN_COUNT_OFFSET) as *mut u64).write(bytes.len() as u64);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), raw.add(STR_BYTES_OFFSET), bytes.len());
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn string_len(s: u64) -> usize {
    let p = v::as_ptr(s);
    unsafe { (p.add(STR_VARLEN_COUNT_OFFSET) as *const u64).read() as usize }
}

pub fn string_bytes(s: u64) -> &'static [u8] {
    let p = v::as_ptr(s);
    let len = string_len(s);
    unsafe { std::slice::from_raw_parts(p.add(STR_BYTES_OFFSET), len) }
}

/// FNV-1a, 64-bit. Cheap-and-deterministic; fine for non-cryptographic
/// hash use (eq-based map keying). Replaceable later.
fn string_hash_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn is_string(v: u64) -> bool {
    is_obj_of(v, |t| t.string.0)
}

// ── Keyword ─────────────────────────────────────────────────────────
//
// Keyword { sym: Value; hash: Raw64 }
// sym at HDR
// hash at HDR + 8

const KW_SYM_OFFSET: usize = HDR;
const KW_HASH_OFFSET: usize = HDR + 8;

/// Allocate a Keyword wrapping the given symbol-id (its name lives in
/// the symbol table). Two `:foo` occurrences allocate distinct Keyword
/// objects today; `=` compares by inner sym-id so equality still works.
pub fn alloc_keyword<'scope>(
    scope: &'scope RootScope<'_>,
    sym_id: u32,
) -> Rooted<'scope, NanBoxTag> {
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.keyword.0;
        let raw = gc.alloc(type_id, 0);
        assert!(!raw.is_null(), "alloc_keyword: GC alloc returned null");
        unsafe {
            (raw.add(KW_SYM_OFFSET) as *mut u64).write(v::encode_sym_id(sym_id));
            (raw.add(KW_HASH_OFFSET) as *mut u64).write(sym_id as u64 ^ 0x9e3779b97f4a7c15);
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn keyword_sym(kw: u64) -> u64 {
    let p = v::as_ptr(kw);
    unsafe { (p.add(KW_SYM_OFFSET) as *const u64).read() }
}

pub fn keyword_sym_id(kw: u64) -> u32 {
    let s = keyword_sym(kw);
    debug_assert!(v::is_sym_id(s));
    v::as_sym_id(s)
}

pub fn is_keyword(v: u64) -> bool {
    is_obj_of(v, |t| t.keyword.0)
}

// ── Vector ──────────────────────────────────────────────────────────
//
// v1 layout: flat array.
// Vector { root: Value, tail: Value, count: Raw64, shift: Raw64 }
//   root → a "node" Map-shaped object reused: actually we allocate a
//          fresh varlen-values heap object using the `map` ObjType but
//          treating it as a flat value-array. Simpler: stash the
//          flat array directly in the Vector's own heap obj. Since
//          our Vector type doesn't have a varlen tail, we instead
//          allocate a tiny side-allocation and stash its pointer in
//          `root`.
//
// To avoid declaring yet another ObjType, we re-use the `map`'s shape
// for the side allocation: `map` has `count: Raw64; varlen_values`. We
// store `count` as the element count and pack values in the varlen
// tail. The flat-Vector code in this file owns the convention.

const VEC_ROOT_OFFSET: usize = HDR;          // Value
const VEC_TAIL_OFFSET: usize = HDR + 8;      // Value (unused in v1)
const VEC_COUNT_OFFSET: usize = HDR + 16;    // Raw64
const VEC_SHIFT_OFFSET: usize = HDR + 24;    // Raw64 (unused in v1)

pub fn alloc_vector<'scope>(
    scope: &'scope RootScope<'_>,
    items: &[u64],
) -> Rooted<'scope, NanBoxTag> {
    let node = alloc_array(scope, items);
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.vector.0;
        let raw = gc.alloc(type_id, 0);
        assert!(!raw.is_null(), "alloc_vector: GC alloc returned null");
        unsafe {
            (raw.add(VEC_ROOT_OFFSET) as *mut u64).write(node.get());
            (raw.add(VEC_TAIL_OFFSET) as *mut u64).write(v::NIL);
            (raw.add(VEC_COUNT_OFFSET) as *mut u64).write(items.len() as u64);
            (raw.add(VEC_SHIFT_OFFSET) as *mut u64).write(0);
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn vector_count(v_: u64) -> usize {
    let p = v::as_ptr(v_);
    unsafe { (p.add(VEC_COUNT_OFFSET) as *const u64).read() as usize }
}

pub fn vector_get(vec: u64, i: usize) -> u64 {
    let p = v::as_ptr(vec);
    let node = unsafe { (p.add(VEC_ROOT_OFFSET) as *const u64).read() };
    array_get(node, i)
}

/// Iterate a Vector. Each element is yielded once.
pub fn vector_iter(vec: u64) -> impl Iterator<Item = u64> {
    let n = vector_count(vec);
    (0..n).map(move |i| vector_get(vec, i))
}

pub fn is_vector(v: u64) -> bool {
    is_obj_of(v, |t| t.vector.0)
}

// ── Set ─────────────────────────────────────────────────────────────
//
// v1 layout: backing is a Vector of items. Linear scan. We'll swap to
// a hash-based representation when usage demands it.

const SET_BACKING_OFFSET: usize = HDR;

pub fn alloc_set<'scope>(
    scope: &'scope RootScope<'_>,
    items: &[u64],
) -> Rooted<'scope, NanBoxTag> {
    // De-dup by bitwise equality (sufficient for nil/bool/sym-id/int;
    // bigger story when strings/keywords are interned).
    let mut deduped: Vec<u64> = Vec::with_capacity(items.len());
    for x in items {
        if !deduped.iter().any(|y| y == x) {
            deduped.push(*x);
        }
    }
    let backing = alloc_vector(scope, &deduped);
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.set.0;
        let raw = gc.alloc(type_id, 0);
        assert!(!raw.is_null(), "alloc_set: GC alloc returned null");
        unsafe {
            (raw.add(SET_BACKING_OFFSET) as *mut u64).write(backing.get());
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn set_backing(s: u64) -> u64 {
    let p = v::as_ptr(s);
    unsafe { (p.add(SET_BACKING_OFFSET) as *const u64).read() }
}

pub fn is_set(v: u64) -> bool {
    is_obj_of(v, |t| t.set.0)
}

// ── Type-id discrimination ──────────────────────────────────────────
//
// For every TAG_PTR value, the first u16 of the heap object is the
// type-id (Compact header). We compare against the host's recorded
// ObjTypeId for the type we want.

fn is_obj_of(val: u64, pick: fn(&crate::types::Types) -> usize) -> bool {
    if !v::is_ptr(val) {
        return false;
    }
    let want = with_host(|h| pick(&h.types));
    let got = unsafe { v::read_type_id(v::as_ptr(val)) } as usize;
    got == want
}

pub fn is_list(v: u64) -> bool {
    is_obj_of(v, |t| t.list.0)
}

pub fn is_map(v: u64) -> bool {
    is_obj_of(v, |t| t.map.0)
}

pub fn is_fn(v: u64) -> bool {
    is_obj_of(v, |t| t.fn_obj.0)
}

pub fn is_var(v: u64) -> bool {
    is_obj_of(v, |t| t.var.0)
}

// ── Native mutable Array ────────────────────────────────────────────
//
// Layout: pure varlen-values (no fixed fields).
//   varlen_count at HDR
//   first element at HDR + 8
//
// This is the user-visible primitive on which core.clj's persistent
// collections (PersistentVector, PersistentHashMap, etc.) are built,
// AND the internal storage for the reader's transient Vector type.
// They share the same heap shape, the same type-id, and the same
// access fns.

const ARRAY_VARLEN_COUNT_OFFSET: usize = HDR;
const ARRAY_ELEM_BASE: usize = HDR + 8;

/// Allocate a fresh Array of `items.len()` slots, copying `items` in.
pub fn alloc_array<'scope>(
    scope: &'scope RootScope<'_>,
    items: &[u64],
) -> Rooted<'scope, NanBoxTag> {
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.array.0;
        let raw = gc.alloc(type_id, items.len());
        assert!(!raw.is_null(), "alloc_array: GC alloc returned null");
        unsafe {
            (raw.add(ARRAY_VARLEN_COUNT_OFFSET) as *mut u64).write(items.len() as u64);
            for (i, x) in items.iter().enumerate() {
                (raw.add(ARRAY_ELEM_BASE + i * 8) as *mut u64).write(*x);
            }
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

/// Allocate an Array of `n` slots filled with `nil`. The shape used
/// by `(make-array n)` from Clojure.
pub fn alloc_array_nil<'scope>(
    scope: &'scope RootScope<'_>,
    n: usize,
) -> Rooted<'scope, NanBoxTag> {
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.array.0;
        let raw = gc.alloc(type_id, n);
        assert!(!raw.is_null(), "alloc_array_nil: GC alloc returned null");
        unsafe {
            (raw.add(ARRAY_VARLEN_COUNT_OFFSET) as *mut u64).write(n as u64);
            for i in 0..n {
                (raw.add(ARRAY_ELEM_BASE + i * 8) as *mut u64).write(v::NIL);
            }
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn array_count(arr: u64) -> usize {
    let p = v::as_ptr(arr);
    unsafe { (p.add(ARRAY_VARLEN_COUNT_OFFSET) as *const u64).read() as usize }
}

pub fn array_get(arr: u64, i: usize) -> u64 {
    let p = v::as_ptr(arr);
    debug_assert!(i < array_count(arr), "array_get: out of bounds");
    unsafe { (p.add(ARRAY_ELEM_BASE + i * 8) as *const u64).read() }
}

/// Mutating store. Visible to user code via `aset!`. Two important
/// safety notes:
///   1. The new value `v` doesn't need a write-barrier here because
///      our generational GC handles old → young pointer writes via
///      the shadow stack at the next safepoint, not by intercepting
///      individual stores. (The current design is non-incremental
///      mark-compact within the generation; if/when that changes
///      this fn must grow a barrier.)
///   2. Mutating an Array that backs a reader-Vector that has
///      already been observed by the user as a "snapshot" violates
///      Clojure's value semantics. Don't do that — `(aset! …)` is
///      meant for collection-internals use only.
pub fn array_set(arr: u64, i: usize, val: u64) {
    let p = v::as_ptr(arr);
    debug_assert!(i < array_count(arr), "array_set: out of bounds");
    unsafe { (p.add(ARRAY_ELEM_BASE + i * 8) as *mut u64).write(val) }
}

pub fn is_array(v: u64) -> bool {
    is_obj_of(v, |t| t.array.0)
}

// ── Generic sequential iteration ────────────────────────────────────
//
// Many places in the compiler want to walk a "sequence of forms"
// without caring whether the source was a list or a vector
// (e.g. `(let [x 1 y 2] body)` reads its bindings as a Vector;
// `(unless c body)` reads its args as a List). `seq_iter` accepts
// either, plus nil.
//
// Returns a concrete enum cursor — no `Box<dyn Iterator>`, no
// per-call heap allocation. Hot paths in the compiler (lower_call's
// arg list walk, the freevars analyzer, the expander) all use this.

#[derive(Clone, Copy)]
enum SeqState {
    Empty,
    /// Walking a List: `cur` points to either a List cell or nil.
    List(u64),
    /// Walking a Vector by index: `(vec, next_idx, len)`.
    Vec(u64, usize, usize),
}

pub struct SeqCursor(SeqState);

impl Iterator for SeqCursor {
    type Item = u64;
    #[inline]
    fn next(&mut self) -> Option<u64> {
        match self.0 {
            SeqState::Empty => None,
            SeqState::List(cur) => {
                if !v::is_ptr(cur) {
                    self.0 = SeqState::Empty;
                    return None;
                }
                let f = v::first(cur);
                self.0 = SeqState::List(v::rest(cur));
                Some(f)
            }
            SeqState::Vec(vec, idx, len) => {
                if idx >= len {
                    self.0 = SeqState::Empty;
                    return None;
                }
                let v = vector_get(vec, idx);
                self.0 = SeqState::Vec(vec, idx + 1, len);
                Some(v)
            }
        }
    }
}

pub fn seq_iter(coll: u64) -> SeqCursor {
    if v::is_nil(coll) {
        SeqCursor(SeqState::Empty)
    } else if is_vector(coll) {
        let len = vector_count(coll);
        SeqCursor(SeqState::Vec(coll, 0, len))
    } else if v::is_ptr(coll) {
        // List or list-shaped (older callers may pass a list-tagged
        // ptr without an explicit type-id check). The cursor walks
        // the spine until it sees a non-ptr.
        SeqCursor(SeqState::List(coll))
    } else {
        SeqCursor(SeqState::Empty)
    }
}

pub fn seq_count(coll: u64) -> usize {
    if v::is_nil(coll) {
        0
    } else if is_vector(coll) {
        vector_count(coll)
    } else if v::is_ptr(coll) {
        v::list_len(coll)
    } else {
        0
    }
}
