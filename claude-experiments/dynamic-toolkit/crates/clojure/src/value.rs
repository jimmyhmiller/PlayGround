//! NanBox-encoded values for Clojure.
//!
//! Tag layout (NanBox: 4 tags × 48-bit payload):
//!   0 — nil singleton
//!   1 — bool (payload 0 = false, 1 = true)
//!   2 — heap pointer (Symbol, Keyword, String, List, Vector, Map,
//!                     Set, Fn, Var, Namespace, Registry — header
//!                     identifies the concrete type)
//!   3 — reserved for `char`
//!
//! Floats are stored directly (NanBox encodes ordinary IEEE 754 bits).

use dynobj::roots::{Rooted, RootScope};
use dynobj::{Compact, ObjHeader};
use dynvalue::{NanBox, TagScheme};

use crate::host::with_host;

/// Phantom tag for `Rooted<NanBoxTag>` slots that hold a Clojure value.
pub struct NanBoxTag;

pub const TAG_NIL: u32 = 0;
pub const TAG_BOOL: u32 = 1;
pub const TAG_PTR: u32 = 2;
/// Bootstrap symbol id. Payload is a `u32` index into the Rust-side
/// `SymbolTable`. We will switch to heap-allocated `Symbol` objects
/// when the namespace registry comes online; until then this lets the
/// reader produce symbol values without a heap allocation per
/// occurrence.
pub const TAG_SYMID: u32 = 3;

pub const NIL: u64 = NanBox::NIL;

pub fn encode_bool(b: bool) -> u64 {
    <NanBox as TagScheme>::encode_tagged(TAG_BOOL, b as u64)
}

pub const TRUE: u64 = 0x7FFC_0001_0000_0001;
pub const FALSE: u64 = 0x7FFC_0001_0000_0000;

pub fn encode_ptr(ptr: *const u8) -> u64 {
    <NanBox as TagScheme>::encode_tagged(TAG_PTR, ptr as u64 & 0x0000_FFFF_FFFF_FFFF)
}

pub fn encode_sym_id(id: u32) -> u64 {
    <NanBox as TagScheme>::encode_tagged(TAG_SYMID, id as u64)
}

pub fn is_sym_id(v: u64) -> bool {
    <NanBox as TagScheme>::has_tag(v, TAG_SYMID)
}

pub fn as_sym_id(v: u64) -> u32 {
    debug_assert!(is_sym_id(v));
    <NanBox as TagScheme>::extract_payload(v) as u32
}

pub fn encode_num(n: f64) -> u64 {
    NanBox::from_f64(n)
}

pub fn encode_int(n: i64) -> u64 {
    NanBox::from_int(n)
}

pub fn is_nil(v: u64) -> bool {
    v == NIL
}

/// Clojure truthiness: only `nil` and `false` are falsey.
pub fn is_truthy(v: u64) -> bool {
    v != NIL && v != FALSE
}

pub fn is_ptr(v: u64) -> bool {
    <NanBox as TagScheme>::has_tag(v, TAG_PTR)
}

pub fn is_bool(v: u64) -> bool {
    <NanBox as TagScheme>::has_tag(v, TAG_BOOL)
}

pub fn is_number(v: u64) -> bool {
    <NanBox as TagScheme>::is_float(v)
}

pub fn as_number(v: u64) -> f64 {
    f64::from_bits(v)
}

pub fn as_ptr(v: u64) -> *const u8 {
    debug_assert!(is_ptr(v));
    <NanBox as TagScheme>::extract_payload(v) as *const u8
}

// ── Heap object headers ─────────────────────────────────────────────
//
// Every heap object lives behind a `Compact` header. Field offsets
// match the declaration order in `types.rs`.

/// Byte offset of the first GC-traced Value field after the header.
pub const FIRST_VALUE_OFFSET: usize = Compact::SIZE;

/// Read the type id (index into the GC's type table) from a heap obj.
pub unsafe fn read_type_id(ptr: *const u8) -> u16 {
    unsafe { dynobj::read_type_id(ptr, 0) }
}

// ── List (cons) helpers ─────────────────────────────────────────────
//
// The `List` ObjType has three fields: first (Value), rest (Value),
// count (Raw64). Field offsets are computed by `dynlang` at type
// declaration time; we hardcode them here matching the declaration
// order in `types.rs`.

/// Byte offset of `List.first`.
pub fn list_first_offset() -> usize {
    Compact::SIZE
}

/// Byte offset of `List.rest`.
pub fn list_rest_offset() -> usize {
    Compact::SIZE + 8
}

/// Byte offset of `List.count` (Raw64, untraced).
pub fn list_count_offset() -> usize {
    Compact::SIZE + 16
}

/// Read the `first` field of a List. Caller must verify `is_ptr(v)`
/// and that the pointed-to object is a List.
pub fn first(v: u64) -> u64 {
    debug_assert!(is_ptr(v));
    let p = as_ptr(v);
    unsafe { (p.add(list_first_offset()) as *const u64).read() }
}

/// Read the `rest` field of a List.
pub fn rest(v: u64) -> u64 {
    debug_assert!(is_ptr(v));
    let p = as_ptr(v);
    unsafe { (p.add(list_rest_offset()) as *const u64).read() }
}

/// Read the cached `count` of a List. Caller must verify this value
/// is in fact a List before calling.
pub fn list_count(v: u64) -> u64 {
    debug_assert!(is_ptr(v));
    let p = as_ptr(v);
    unsafe { (p.add(list_count_offset()) as *const u64).read() }
}

/// Walk the spine of a list/cons-tree. `nil` terminates iteration.
pub fn list_iter(v: u64) -> impl Iterator<Item = u64> {
    let mut cur = v;
    std::iter::from_fn(move || {
        if is_ptr(cur) {
            // For checkpoint 2 the only heap objects are Lists, so a
            // ptr-tagged value here is always a List. Once we add other
            // heap types, gate this on a type-id check.
            let f = first(cur);
            cur = rest(cur);
            Some(f)
        } else {
            None
        }
    })
}

pub fn list_len(v: u64) -> usize {
    list_iter(v).count()
}

/// Allocate a List cell `(first . rest)` on the GC heap. Both inputs
/// must already be rooted in some scope; the result is rooted in
/// `scope`. Mirrors microlisp's `alloc_cons` discipline.
pub fn alloc_list_cell<'scope>(
    scope: &'scope RootScope<'_>,
    first_val: &Rooted<'_, NanBoxTag>,
    rest_val: &Rooted<'_, NanBoxTag>,
) -> Rooted<'scope, NanBoxTag> {
    with_host(|h| {
        debug_assert!(!h.gc.is_null(), "alloc_list_cell: Host has no GC");
        let gc = unsafe { &*h.gc };
        let type_id = h.types.list.0;
        let raw = gc.alloc(type_id, 0);
        assert!(!raw.is_null(), "clojure: GC alloc returned null");
        let f_bits = first_val.get();
        let r_bits = rest_val.get();
        // count = 1 + count(rest) when rest is a list, else 1.
        let r_count = if is_ptr(r_bits) {
            unsafe { (as_ptr(r_bits).add(list_count_offset()) as *const u64).read() }
        } else {
            0
        };
        unsafe {
            (raw.add(list_first_offset()) as *mut u64).write(f_bits);
            (raw.add(list_rest_offset()) as *mut u64).write(r_bits);
            (raw.add(list_count_offset()) as *mut u64).write(r_count + 1);
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

/// Convenience: allocate a list cell from raw NanBox bits. Roots them
/// internally for the duration of the allocation.
pub fn alloc_list_cell_from_raw<'scope>(
    scope: &'scope RootScope<'_>,
    first_bits: u64,
    rest_bits: u64,
) -> Rooted<'scope, NanBoxTag> {
    let f = scope.root::<NanBoxTag>(first_bits);
    let r = scope.root::<NanBoxTag>(rest_bits);
    alloc_list_cell(scope, &f, &r)
}
