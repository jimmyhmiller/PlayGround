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
use dynvalue::{NanBox, TagScheme};

use crate::host::{layouts, with_host};

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

/// Read the type id (index into the GC's type table) from a heap obj.
pub unsafe fn read_type_id(ptr: *const u8) -> u16 {
    unsafe { dynobj::read_type_id(ptr, 0) }
}

// ── List (cons) helpers ─────────────────────────────────────────────
//
// `List` is { first: Value, rest: Value, count: Raw64 }. All offsets
// resolved from `host::layouts()` (see `types::Layouts`).
//
// All public accessors here assert `is_list(v)`. The previous API
// silently read `first`/`rest`/`count` slots from any ptr-tagged
// value — passing in a Map/Vector/Record returned garbage. The
// assertions cost one type-id read per call (a thread-local
// `with_host` + a HashMap lookup avoided by the cached `Types`
// struct on Host) and turn the silent-corruption case into a clear
// panic. For hot paths that need to skip the check, use the
// `_unchecked` variants below.

/// Read the `first` field of a List. Panics if `v` isn't a List.
pub fn first(v: u64) -> u64 {
    assert!(
        crate::collections::is_list(v),
        "value::first: not a List (got 0x{v:016x})"
    );
    unsafe { first_unchecked(v) }
}

/// Read the `rest` field of a List. Panics if `v` isn't a List.
pub fn rest(v: u64) -> u64 {
    assert!(
        crate::collections::is_list(v),
        "value::rest: not a List (got 0x{v:016x})"
    );
    unsafe { rest_unchecked(v) }
}

/// Read the cached `count` of a List. Panics if `v` isn't a List.
pub fn list_count(v: u64) -> u64 {
    assert!(
        crate::collections::is_list(v),
        "value::list_count: not a List (got 0x{v:016x})"
    );
    unsafe { list_count_unchecked(v) }
}

/// Walk the spine of a List. `nil` terminates iteration. Panics on
/// non-list, non-nil input.
pub fn list_iter(v: u64) -> impl Iterator<Item = u64> {
    if !is_nil(v) {
        assert!(
            crate::collections::is_list(v),
            "value::list_iter: not a List (got 0x{v:016x})"
        );
    }
    let mut cur = v;
    std::iter::from_fn(move || {
        if is_ptr(cur) {
            // Safe: the entry assertion proved cur was a List, and
            // every cell's `rest` is either nil or another List.
            let f = unsafe { first_unchecked(cur) };
            cur = unsafe { rest_unchecked(cur) };
            Some(f)
        } else {
            None
        }
    })
}

// ── Unchecked variants ──────────────────────────────────────────────
//
// For hot paths that have already verified the input shape (e.g.
// inside `list_iter` after the entry check, or inside the typed
// `ListRef` wrapper). `unsafe` because reading `first`/`rest` from
// a non-List ptr-tagged value returns garbage.

/// Read `List.first` without a type check. UB if not a List.
#[inline]
pub unsafe fn first_unchecked(v: u64) -> u64 {
    let p = as_ptr(v);
    unsafe { (p.add(layouts().list_first) as *const u64).read() }
}

/// Read `List.rest` without a type check. UB if not a List.
#[inline]
pub unsafe fn rest_unchecked(v: u64) -> u64 {
    let p = as_ptr(v);
    unsafe { (p.add(layouts().list_rest) as *const u64).read() }
}

/// Read `List.count` without a type check. UB if not a List.
#[inline]
pub unsafe fn list_count_unchecked(v: u64) -> u64 {
    let p = as_ptr(v);
    unsafe { (p.add(layouts().list_count) as *const u64).read() }
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
        let l = h.layouts;
        let f_bits = first_val.get();
        let r_bits = rest_val.get();
        // count = 1 + count(rest) when rest is a list, else 1.
        let r_count = if is_ptr(r_bits) {
            unsafe { (as_ptr(r_bits).add(l.list_count) as *const u64).read() }
        } else {
            0
        };
        unsafe {
            (raw.add(l.list_first) as *mut u64).write(f_bits);
            (raw.add(l.list_rest) as *mut u64).write(r_bits);
            (raw.add(l.list_count) as *mut u64).write(r_count + 1);
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
