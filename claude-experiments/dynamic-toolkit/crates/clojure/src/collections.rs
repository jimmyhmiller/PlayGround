//! Heap allocators / accessors for String, Keyword, Vector, Set.
//!
//! Field offsets come from `host::layouts()`, which is populated once
//! at engine init from the dynlang `ObjType` registry (see
//! `types::Layouts`). No `const FOO_OFFSET` constants here — that
//! data lives in one place and one place only.

use dynobj::roots::{RootScope, Rooted};

use crate::host::{layouts, with_host};
use crate::value::{self as v, NanBoxTag};

// ── String ──────────────────────────────────────────────────────────
//
// String { hash: Raw64; varlen_bytes }

pub fn alloc_string<'scope>(
    scope: &'scope RootScope<'_>,
    bytes: &[u8],
) -> Rooted<'scope, NanBoxTag> {
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.string.0;
        let raw = gc.alloc(type_id, bytes.len());
        assert!(!raw.is_null(), "alloc_string: GC alloc returned null");
        let l = h.layouts;
        unsafe {
            (raw.add(l.string_hash) as *mut u64).write(string_hash_bytes(bytes));
            (raw.add(l.string_varlen_count) as *mut u64).write(bytes.len() as u64);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), raw.add(l.string_bytes), bytes.len());
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn string_len(s: u64) -> usize {
    let p = v::as_ptr(s);
    unsafe { (p.add(layouts().string_varlen_count) as *const u64).read() as usize }
}

pub fn string_bytes(s: u64) -> &'static [u8] {
    let p = v::as_ptr(s);
    let len = string_len(s);
    unsafe { std::slice::from_raw_parts(p.add(layouts().string_bytes), len) }
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
        let l = h.layouts;
        unsafe {
            (raw.add(l.keyword_sym) as *mut u64).write(v::encode_sym_id(sym_id));
            (raw.add(l.keyword_hash) as *mut u64).write(sym_id as u64 ^ 0x9e3779b97f4a7c15);
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn keyword_sym(kw: u64) -> u64 {
    let p = v::as_ptr(kw);
    unsafe { (p.add(layouts().keyword_sym) as *const u64).read() }
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
//   root → a side-allocation that holds the flat element array
//          (uses the `Array` ObjType — pure varlen_values).
//   tail, shift — unused in v1; declared so the struct is HAMT/RRB-
//                 ready when we upgrade.
//
// Field offsets come from `host::layouts()`. The dynlang builder
// reorders to value-fields-first, so root/tail come before count/shift
// in memory regardless of declaration order in `types.rs`.

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
        let l = h.layouts;
        unsafe {
            (raw.add(l.vector_root) as *mut u64).write(node.get());
            (raw.add(l.vector_tail) as *mut u64).write(v::NIL);
            (raw.add(l.vector_count) as *mut u64).write(items.len() as u64);
            (raw.add(l.vector_shift) as *mut u64).write(0);
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn vector_count(v_: u64) -> usize {
    let p = v::as_ptr(v_);
    unsafe { (p.add(layouts().vector_count) as *const u64).read() as usize }
}

pub fn vector_get(vec: u64, i: usize) -> u64 {
    let p = v::as_ptr(vec);
    let node = unsafe { (p.add(layouts().vector_root) as *const u64).read() };
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

pub fn alloc_set<'scope>(scope: &'scope RootScope<'_>, items: &[u64]) -> Rooted<'scope, NanBoxTag> {
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
            (raw.add(h.layouts.set_backing) as *mut u64).write(backing.get());
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn set_backing(s: u64) -> u64 {
    let p = v::as_ptr(s);
    unsafe { (p.add(layouts().set_backing) as *const u64).read() }
}

pub fn is_set(v: u64) -> bool {
    is_obj_of(v, |t| t.set.0)
}

// ── Record (deftype instance) ───────────────────────────────────────
//
// Layout: HDR | type_name: Value | varlen_count | field 0 | field 1 …
//
// A single concrete ObjType backs every user-declared `deftype*`.
// The type's identity is a symbol stored as the `type_name` field;
// instances of `(deftype* MyType …)` carry `type_name = 'MyType` and
// the user's fields in the varlen tail.

/// Allocate a `Record` with `type_name` and the supplied field
/// values copied into the varlen tail.
pub fn alloc_record<'scope>(
    scope: &'scope RootScope<'_>,
    type_name: u64,
    fields: &[u64],
) -> Rooted<'scope, NanBoxTag> {
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.record.0;
        let raw = gc.alloc(type_id, fields.len());
        assert!(!raw.is_null(), "alloc_record: GC alloc returned null");
        let l = h.layouts;
        unsafe {
            (raw.add(l.record_type_name) as *mut u64).write(type_name);
            (raw.add(l.record_varlen_count) as *mut u64).write(fields.len() as u64);
            for (i, &f) in fields.iter().enumerate() {
                (raw.add(l.record_fields_base + i * 8) as *mut u64).write(f);
            }
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn record_type_name(rec: u64) -> u64 {
    let p = v::as_ptr(rec);
    unsafe { (p.add(layouts().record_type_name) as *const u64).read() }
}

pub fn record_field_count(rec: u64) -> usize {
    let p = v::as_ptr(rec);
    unsafe { (p.add(layouts().record_varlen_count) as *const u64).read() as usize }
}

pub fn record_field(rec: u64, i: usize) -> u64 {
    debug_assert!(i < record_field_count(rec));
    let p = v::as_ptr(rec);
    unsafe { (p.add(layouts().record_fields_base + i * 8) as *const u64).read() }
}

/// Overwrite a record field in place. Used by `(set! (.-field this) v)`
/// for `^:mutable` fields. The mutability check itself lives at the
/// extern boundary (see `clj_record_set_field`), which currently
/// refuses all writes until per-field mutability is tracked.
pub fn record_set_field(rec: u64, i: usize, val: u64) {
    debug_assert!(i < record_field_count(rec));
    let p = v::as_ptr(rec);
    unsafe { (p.add(layouts().record_fields_base + i * 8) as *mut u64).write(val) }
}

pub fn is_record(v: u64) -> bool {
    is_obj_of(v, |t| t.record.0)
}

// ── Atom ────────────────────────────────────────────────────────────
//
// `(atom v)` heap obj. One Value field. Reads/writes use the field's
// existing Relaxed atomic load/store. The cell value is GC-traced
// because `Value` fields are scanned automatically.

pub fn alloc_atom<'scope>(scope: &'scope RootScope<'_>, initial: u64) -> Rooted<'scope, NanBoxTag> {
    with_host(|h| {
        let gc = unsafe { &*h.gc };
        let type_id = h.types.atom.0;
        let raw = gc.alloc(type_id, 0);
        assert!(!raw.is_null(), "alloc_atom: GC alloc returned null");
        unsafe {
            (raw.add(h.layouts.atom_val) as *mut u64).write(initial);
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn atom_get(a: u64) -> u64 {
    let p = v::as_ptr(a);
    unsafe { (p.add(layouts().atom_val) as *const u64).read() }
}

pub fn atom_set(a: u64, val: u64) {
    let p = v::as_ptr(a);
    unsafe { (p.add(layouts().atom_val) as *mut u64).write(val) }
}

pub fn is_atom(v: u64) -> bool {
    is_obj_of(v, |t| t.atom.0)
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
//
// This is the user-visible primitive on which core.clj's persistent
// collections (PersistentVector, PersistentHashMap, etc.) are built,
// AND the internal storage for the reader's transient Vector type.
// They share the same heap shape, the same type-id, and the same
// access fns.

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
        let l = h.layouts;
        unsafe {
            (raw.add(l.array_varlen_count) as *mut u64).write(items.len() as u64);
            for (i, x) in items.iter().enumerate() {
                (raw.add(l.array_elem_base + i * 8) as *mut u64).write(*x);
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
        let l = h.layouts;
        unsafe {
            (raw.add(l.array_varlen_count) as *mut u64).write(n as u64);
            for i in 0..n {
                (raw.add(l.array_elem_base + i * 8) as *mut u64).write(v::NIL);
            }
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

pub fn array_count(arr: u64) -> usize {
    let p = v::as_ptr(arr);
    unsafe { (p.add(layouts().array_varlen_count) as *const u64).read() as usize }
}

pub fn array_get(arr: u64, i: usize) -> u64 {
    let p = v::as_ptr(arr);
    debug_assert!(i < array_count(arr), "array_get: out of bounds");
    unsafe { (p.add(layouts().array_elem_base + i * 8) as *const u64).read() }
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
    unsafe { (p.add(layouts().array_elem_base + i * 8) as *mut u64).write(val) }
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

/// Cursor over an arbitrary Clojure seqable.
///
/// Three internal shapes:
///   - `Empty`: end of iteration.
///   - `List(cur)`: walks built-in `__ReaderList` cells via the
///     `v::first` / `v::rest` accessors. Fast path for forms the
///     reader produced.
///   - `Vec(vec, idx, len)`: walks a built-in `Vector` by index.
///   - `Protocol(cur)`: everything else. `cur` is the result of
///     calling `(-seq receiver)` on the original input (or whatever
///     `(-next cur)` returned on the previous step). Each `next()`
///     dispatches `(-first cur)` and `(-next cur)` through
///     `protocol::invoke_method_0`. Records (PList, Cons,
///     PersistentVector, IndexedSeq, MapEntry, …) and any future
///     deftype that implements `ISeq` walk via this path uniformly.
///
/// We keep the two built-in fast paths because the runtime owns
/// those heap shapes — they're not user-extensible types we need
/// the protocol to abstract over.
#[derive(Clone, Copy)]
enum SeqState {
    Empty,
    List(u64),
    Vec(u64, usize, usize),
    Protocol(u64),
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
            SeqState::Protocol(cur) => {
                if v::is_nil(cur) {
                    self.0 = SeqState::Empty;
                    return None;
                }
                let (first_sym, next_sym) = with_host(|h| (h.first_method_sym, h.next_method_sym));
                let head = crate::protocol::invoke_method_0(first_sym, cur);
                let tail = crate::protocol::invoke_method_0(next_sym, cur);
                self.0 = SeqState::Protocol(tail);
                Some(head)
            }
        }
    }
}

pub fn seq_iter(coll: u64) -> SeqCursor {
    if v::is_nil(coll) {
        return SeqCursor(SeqState::Empty);
    }
    if is_vector(coll) {
        return SeqCursor(SeqState::Vec(coll, 0, vector_count(coll)));
    }
    if !v::is_ptr(coll) {
        return SeqCursor(SeqState::Empty);
    }
    if is_list(coll) {
        return SeqCursor(SeqState::List(coll));
    }
    // Anything else: ask `(-seq coll)` and walk via the protocol.
    // Whatever `-seq` returns becomes the cursor — for list-like
    // types it's typically the receiver itself; for sequence-views
    // (IndexedSeq, ChunkedSeq, …) it's a fresh seq value we walk
    // via `-first` / `-next` until nil.
    let seq_sym = with_host(|h| h.seq_method_sym);
    let s = crate::protocol::invoke_method_0(seq_sym, coll);
    if v::is_nil(s) {
        return SeqCursor(SeqState::Empty);
    }
    SeqCursor(SeqState::Protocol(s))
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
