//! C-callable Rust externs that the Clojure JIT calls into.
//!
//! Each extern takes/returns NanBox-encoded `u64`. Numeric externs
//! always return floats — `from_int` and `from_f64` both encode into
//! the unboxed-float NaN payload, so an int+int still rides the
//! float fast path.

use crate::namespace as ns;
use crate::value::*;

// ── Helpers ────────────────────────────────────────────────────────

#[inline]
fn num(v: u64) -> f64 {
    if !is_number(v) {
        panic!("expected number, got 0x{:016x}", v);
    }
    as_number(v)
}

#[inline]
fn bool_(b: bool) -> u64 {
    if b { TRUE } else { FALSE }
}

// ── Numeric ────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn clj_add(a: u64, b: u64) -> u64 {
    encode_num(num(a) + num(b))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_sub(a: u64, b: u64) -> u64 {
    encode_num(num(a) - num(b))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_mul(a: u64, b: u64) -> u64 {
    encode_num(num(a) * num(b))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_div(a: u64, b: u64) -> u64 {
    encode_num(num(a) / num(b))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_lt(a: u64, b: u64) -> u64 {
    bool_(num(a) < num(b))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_gt(a: u64, b: u64) -> u64 {
    bool_(num(a) > num(b))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_le(a: u64, b: u64) -> u64 {
    bool_(num(a) <= num(b))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_ge(a: u64, b: u64) -> u64 {
    bool_(num(a) >= num(b))
}

/// `=` — Clojure equality.
///
/// - Numbers: value compare on the unboxed double.
/// - Strings: structural — same length and byte-wise equal contents.
///   Two `"foo"` literals from different reads allocate distinct
///   heap objects; bitwise compare would say they're inequal.
/// - Everything else (interned keywords/symbols, immediate `nil` /
///   `true` / `false`, identity references): bitwise compare. The
///   intern tables guarantee identity for the cases that need it.
///
/// Future-coverage TODO: structural compare for List/Vector/Map/Set
/// arrives with the persistent-collection rewrites (#13–#15).
#[unsafe(no_mangle)]
pub extern "C" fn clj_eq(a: u64, b: u64) -> u64 {
    if is_number(a) && is_number(b) {
        return bool_(num(a) == num(b));
    }
    if a == b {
        return TRUE;
    }
    // Structural string compare. Both must be String heap objects;
    // mismatched types fall through to inequal.
    if crate::collections::is_string(a) && crate::collections::is_string(b) {
        let bs_a = crate::collections::string_bytes(a);
        let bs_b = crate::collections::string_bytes(b);
        return bool_(bs_a == bs_b);
    }
    FALSE
}

// ── List construction ──────────────────────────────────────────────

/// `cons` — allocate a list cell `(head . tail)`. Both inputs are
/// rooted in a fresh scope before the alloc, then the result is
/// returned as raw bits to JIT (the JIT's safepoint stack map roots
/// it from there).
///
/// CALLERS THAT EMIT `cons` IN IR MUST emit a preceding safepoint
/// listing every live IR value. The JIT's stack-map mechanism is
/// what keeps the args alive across the alloc; without the
/// safepoint, GC fired by `cons` would relocate them and the IR
/// would carry stale pointers.
#[unsafe(no_mangle)]
pub extern "C" fn clj_cons(head: u64, tail: u64) -> u64 {
    dynobj::roots::with_scope(3, |scope| {
        alloc_list_cell_from_raw(scope, head, tail).get()
    })
}

// ── Binary concat (used by the quasiquote rewriter) ────────────────
//
// `__concat a b` — produce a fresh list containing every element of
// `a` followed by every element of `b`. Both inputs must be Lists or
// nil. The quasiquote expander always emits binary concats; user
// code can wrap a variadic `concat` on top in `core.clj` later.

#[unsafe(no_mangle)]
pub extern "C" fn clj_concat(a: u64, b: u64) -> u64 {
    // Snapshot elements off both source lists into a Vec BEFORE
    // allocating the new spine — list_iter holds raw pointers that
    // a moving GC fired during alloc would invalidate.
    let mut items: Vec<u64> = Vec::new();
    if !is_nil(a) {
        if !crate::collections::is_list(a) {
            panic!("__concat: first arg must be a list or nil");
        }
        items.extend(list_iter(a));
    }
    if !is_nil(b) {
        if !crate::collections::is_list(b) {
            panic!("__concat: second arg must be a list or nil");
        }
        items.extend(list_iter(b));
    }
    dynobj::roots::with_scope(items.len() + 4, |scope| {
        let acc = scope.root::<crate::value::NanBoxTag>(NIL);
        for x in items.iter().rev() {
            let cell = dynobj::roots::with_scope(3, |inner| {
                alloc_list_cell_from_raw(inner, *x, acc.get()).get()
            });
            acc.set(cell);
        }
        acc.get()
    })
}

// ── Native mutable Array ───────────────────────────────────────────
//
// The primitive on which core.clj builds PersistentVector and
// PersistentHashMap. Mutating, single-thread-of-control, no
// write-barrier (see `array_set` doc).

#[unsafe(no_mangle)]
pub extern "C" fn clj_make_array(n: u64) -> u64 {
    if !is_number(n) {
        panic!("make-array: expected a number, got 0x{:016x}", n);
    }
    let count = as_number(n) as usize;
    dynobj::roots::with_scope(2, |scope| {
        crate::collections::alloc_array_nil(scope, count).get()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_aget(arr: u64, i: u64) -> u64 {
    if !crate::collections::is_array(arr) {
        panic!("aget: expected an Array, got 0x{:016x}", arr);
    }
    if !is_number(i) {
        panic!("aget: index must be a number, got 0x{:016x}", i);
    }
    let idx = as_number(i) as usize;
    let len = crate::collections::array_count(arr);
    if idx >= len {
        panic!("aget: index {idx} out of bounds (length {len})");
    }
    crate::collections::array_get(arr, idx)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_aset(arr: u64, i: u64, val: u64) -> u64 {
    if !crate::collections::is_array(arr) {
        panic!("aset!: expected an Array, got 0x{:016x}", arr);
    }
    if !is_number(i) {
        panic!("aset!: index must be a number, got 0x{:016x}", i);
    }
    let idx = as_number(i) as usize;
    let len = crate::collections::array_count(arr);
    if idx >= len {
        panic!("aset!: index {idx} out of bounds (length {len})");
    }
    crate::collections::array_set(arr, idx, val);
    val
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_alength(arr: u64) -> u64 {
    if !crate::collections::is_array(arr) {
        panic!("alength: expected an Array, got 0x{:016x}", arr);
    }
    encode_int(crate::collections::array_count(arr) as i64)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_aclone(arr: u64) -> u64 {
    if !crate::collections::is_array(arr) {
        panic!("aclone: expected an Array, got 0x{:016x}", arr);
    }
    let n = crate::collections::array_count(arr);
    // Snapshot the source contents BEFORE allocating: alloc may move
    // `arr` out from under us, so the source-side pointer must be
    // resolved into a slice first. With_scope + a re-rooted source
    // would be more general; for an O(n) clone it's simpler to
    // collect into a Vec on the side and copy back.
    let src: Vec<u64> = (0..n).map(|i| crate::collections::array_get(arr, i)).collect();
    dynobj::roots::with_scope(2, |scope| {
        crate::collections::alloc_array(scope, &src).get()
    })
}

// ── Reader-collection accessor bridges ─────────────────────────────
//
// The reader produces opaque heap collections (List, Vector, Map,
// Set). `core.clj`'s `extend-type __ReaderXxx` blocks attach
// protocol methods to these by calling the externs below. This
// keeps the runtime side ignorant of protocol layout — the language
// side decides what counts as `count`/`first`/`rest`/`nth`/etc.

// ── List ──

#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_list_first(list: u64) -> u64 {
    if is_nil(list) {
        return NIL;
    }
    if !crate::collections::is_list(list) {
        panic!("__reader_list_first: expected a List, got 0x{list:016x}");
    }
    first(list)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_list_rest(list: u64) -> u64 {
    if is_nil(list) {
        return NIL;
    }
    if !crate::collections::is_list(list) {
        panic!("__reader_list_rest: expected a List, got 0x{list:016x}");
    }
    rest(list)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_list_count(list: u64) -> u64 {
    if is_nil(list) {
        return encode_int(0);
    }
    if !crate::collections::is_list(list) {
        panic!("__reader_list_count: expected a List, got 0x{list:016x}");
    }
    encode_int(list_count(list) as i64)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_list_nth(list: u64, n: u64) -> u64 {
    if !is_number(n) {
        panic!("__reader_list_nth: index must be a number");
    }
    let idx = as_number(n) as usize;
    let mut cur = list;
    for _ in 0..idx {
        if is_nil(cur) {
            panic!("__reader_list_nth: index {idx} past end");
        }
        cur = rest(cur);
    }
    if is_nil(cur) {
        panic!("__reader_list_nth: index {idx} past end");
    }
    first(cur)
}

// ── Vector ──

#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_vector_count(vec: u64) -> u64 {
    if !crate::collections::is_vector(vec) {
        panic!("__reader_vector_count: expected a Vector, got 0x{vec:016x}");
    }
    encode_int(crate::collections::vector_count(vec) as i64)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_vector_nth(vec: u64, n: u64) -> u64 {
    if !crate::collections::is_vector(vec) {
        panic!("__reader_vector_nth: expected a Vector, got 0x{vec:016x}");
    }
    if !is_number(n) {
        panic!("__reader_vector_nth: index must be a number");
    }
    let idx = as_number(n) as usize;
    let len = crate::collections::vector_count(vec);
    if idx >= len {
        panic!("__reader_vector_nth: index {idx} out of bounds (length {len})");
    }
    crate::collections::vector_get(vec, idx)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_vector_first(vec: u64) -> u64 {
    if !crate::collections::is_vector(vec) {
        panic!("__reader_vector_first: expected a Vector, got 0x{vec:016x}");
    }
    if crate::collections::vector_count(vec) == 0 {
        return NIL;
    }
    crate::collections::vector_get(vec, 0)
}

/// Returns a NEW Vector containing `(rest vec)` — i.e. with element 0
/// dropped. O(n) per call (allocates a fresh Array). Sufficient for
/// the bootstrap path in `core.clj`; user code that wants efficient
/// rest should call `(seq …)` to get an `IndexedSeq` instead.
#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_vector_rest(vec: u64) -> u64 {
    if !crate::collections::is_vector(vec) {
        panic!("__reader_vector_rest: expected a Vector, got 0x{vec:016x}");
    }
    let len = crate::collections::vector_count(vec);
    if len <= 1 {
        return dynobj::roots::with_scope(2, |scope| {
            crate::collections::alloc_vector(scope, &[]).get()
        });
    }
    // Snapshot the source elements before allocating: alloc may move
    // `vec`, so resolve to raw bits first.
    let items: Vec<u64> = (1..len).map(|i| crate::collections::vector_get(vec, i)).collect();
    dynobj::roots::with_scope(items.len() + 4, |scope| {
        crate::collections::alloc_vector(scope, &items).get()
    })
}

// ── Set ──

#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_set_count(set: u64) -> u64 {
    if !crate::collections::is_set(set) {
        panic!("__reader_set_count: expected a Set, got 0x{set:016x}");
    }
    encode_int(crate::collections::vector_count(crate::collections::set_backing(set)) as i64)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_set_get(set: u64, n: u64) -> u64 {
    if !crate::collections::is_set(set) {
        panic!("__reader_set_get: expected a Set, got 0x{set:016x}");
    }
    if !is_number(n) {
        panic!("__reader_set_get: index must be a number");
    }
    let idx = as_number(n) as usize;
    let backing = crate::collections::set_backing(set);
    let len = crate::collections::vector_count(backing);
    if idx >= len {
        panic!("__reader_set_get: index {idx} out of bounds (length {len})");
    }
    crate::collections::vector_get(backing, idx)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_set_contains(set: u64, val: u64) -> u64 {
    if !crate::collections::is_set(set) {
        panic!("__reader_set_contains: expected a Set, got 0x{set:016x}");
    }
    let backing = crate::collections::set_backing(set);
    let len = crate::collections::vector_count(backing);
    for i in 0..len {
        // Use `clj_eq` so structural equality (e.g. strings) hits.
        if clj_eq(crate::collections::vector_get(backing, i), val) == TRUE {
            return TRUE;
        }
    }
    FALSE
}

/// Returns a NEW Set with `val` added (or the same set if already a
/// member). The current Set type is structural-equality based on
/// linear scan; once HAMT-PersistentHashSet (in user-space, on top
/// of HAMT-Map) replaces this for non-trivial cases, callers will
/// switch to that.
#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_set_conj(set: u64, val: u64) -> u64 {
    if !crate::collections::is_set(set) {
        panic!("__reader_set_conj: expected a Set, got 0x{set:016x}");
    }
    if clj_reader_set_contains(set, val) == TRUE {
        return set;
    }
    let backing = crate::collections::set_backing(set);
    let n = crate::collections::vector_count(backing);
    let mut items: Vec<u64> = (0..n).map(|i| crate::collections::vector_get(backing, i)).collect();
    items.push(val);
    dynobj::roots::with_scope(items.len() + 4, |scope| {
        crate::collections::alloc_set(scope, &items).get()
    })
}

// ── Map ──

#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_map_count(map: u64) -> u64 {
    if !crate::collections::is_map(map) {
        panic!("__reader_map_count: expected a Map, got 0x{map:016x}");
    }
    encode_int(crate::namespace::map_count(map) as i64)
}

/// Look up `key` in `map`; return its value, or `not_found` if
/// absent. Equality is structural-via-clj_eq (so string keys work).
#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_map_lookup(map: u64, key: u64, not_found: u64) -> u64 {
    if !crate::collections::is_map(map) {
        panic!("__reader_map_lookup: expected a Map, got 0x{map:016x}");
    }
    let n = crate::namespace::map_count(map) as usize;
    for i in 0..n {
        let (k, val) = crate::namespace::map_entry(map, i);
        if clj_eq(k, key) == TRUE {
            return val;
        }
    }
    not_found
}

/// Build a List of the map's keys (insertion order).
#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_map_keys(map: u64) -> u64 {
    if !crate::collections::is_map(map) {
        panic!("__reader_map_keys: expected a Map, got 0x{map:016x}");
    }
    let n = crate::namespace::map_count(map) as usize;
    let keys: Vec<u64> = (0..n).map(|i| crate::namespace::map_entry(map, i).0).collect();
    dynobj::roots::with_scope(keys.len() + 4, |scope| {
        let acc = scope.root::<crate::value::NanBoxTag>(NIL);
        for k in keys.into_iter().rev() {
            let cell = dynobj::roots::with_scope(3, |inner| {
                alloc_list_cell_from_raw(inner, k, acc.get()).get()
            });
            acc.set(cell);
        }
        acc.get()
    })
}

/// Returns a NEW Map with `(key, val)` added (or replaced).
#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_map_assoc(map: u64, key: u64, val: u64) -> u64 {
    if !crate::collections::is_map(map) {
        panic!("__reader_map_assoc: expected a Map, got 0x{map:016x}");
    }
    dynobj::roots::with_scope(8, |scope| {
        crate::namespace::map_assoc(scope, map, key, val).get()
    })
}

// ── Predicates ─────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn clj_nil_p(v: u64) -> u64 {
    bool_(is_nil(v))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_number_p(v: u64) -> u64 {
    bool_(is_number(v))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_not(v: u64) -> u64 {
    bool_(!is_truthy(v))
}

// ── First-class fn allocation ──────────────────────────────────────
//
// `clj_alloc_fn(fr_id, arity)` produces a heap `Fn` object wrapping
// the given (already-compiled) FuncRef. With the closure-capture path
// we will switch to `clj_alloc_closure(fr_id, arity, n_caps, cap0,
// cap1, …)`; for now `(fn [...] ...)` without captures suffices.

#[unsafe(no_mangle)]
pub extern "C" fn clj_alloc_fn(fr_id: u64, arity: u64) -> u64 {
    dynobj::roots::with_scope(2, |scope| {
        ns::alloc_fn(scope, fr_id as u32, arity as usize).get()
    })
}

/// Allocate an `Fn` heap object with `n_caps` captures whose initial
/// values are read from a contiguous buffer at `caps_ptr`.
#[unsafe(no_mangle)]
pub extern "C" fn clj_alloc_closure(
    fr_id: u64,
    arity: u64,
    n_caps: u64,
    caps_ptr: *const u64,
) -> u64 {
    let n = n_caps as usize;
    // Snapshot the captures off the caller's stack before allocating
    // (so a GC during the alloc can't move the stack we're reading).
    let caps: Vec<u64> = unsafe { std::slice::from_raw_parts(caps_ptr, n).to_vec() };
    dynobj::roots::with_scope(n + 4, |scope| {
        ns::alloc_fn_with_captures(scope, fr_id as u32, arity as usize, &caps).get()
    })
}

// ── Runtime arity check ─────────────────────────────────────────────
//
// Higher-order calls are dispatched in IR via `call_indirect` through
// a code pointer fetched from the JitModule's call_table. There is no
// per-arity dispatch stub, so the runtime arity check lives here as
// its own extern. The caller emits a call to `clj_arity_check` BEFORE
// the indirect call; if the receiver is not an Fn or its declared
// arity differs from the call-site's argument count, we panic.

/// Returns 0 (a discardable u64) so it can share the unified
/// `(I64, I64, ...) -> I64` extern signature with everything else;
/// the caller emits the call as a side-effect and ignores the result.
#[unsafe(no_mangle)]
pub extern "C" fn clj_arity_check(fn_v: u64, expected: u64) -> u64 {
    if !is_ptr(fn_v) {
        panic!(
            "call: callee is not a function (got 0x{fn_v:016x})"
        );
    }
    let arity = ns::fn_arity(fn_v);
    let expected = expected as usize;
    if arity != expected {
        panic!(
            "ArityException: function expects {arity} arg(s) but was called with {expected}"
        );
    }
    0
}

// ── Registry: name → (fn pointer, arity) ───────────────────────────
//
// Used by the Engine to declare each primitive on the DynModule and
// to populate the externs vector that JitModule::extend reads to
// patch call-table slots. The name is what the source program uses;
// the arity tells the Engine what signature to declare.

pub struct Prim {
    pub name: &'static str,
    pub ptr: *const u8,
    pub arity: usize,
}

unsafe impl Sync for Prim {}

pub fn all_prims() -> &'static [Prim] {
    static PRIMS: &[Prim] = &[
        Prim { name: "+",  ptr: clj_add as *const u8,    arity: 2 },
        Prim { name: "-",  ptr: clj_sub as *const u8,    arity: 2 },
        Prim { name: "*",  ptr: clj_mul as *const u8,    arity: 2 },
        Prim { name: "/",  ptr: clj_div as *const u8,    arity: 2 },
        Prim { name: "<",  ptr: clj_lt as *const u8,     arity: 2 },
        Prim { name: ">",  ptr: clj_gt as *const u8,     arity: 2 },
        Prim { name: "<=", ptr: clj_le as *const u8,     arity: 2 },
        Prim { name: ">=", ptr: clj_ge as *const u8,     arity: 2 },
        Prim { name: "=",  ptr: clj_eq as *const u8,     arity: 2 },
        Prim { name: "nil?",    ptr: clj_nil_p as *const u8,    arity: 1 },
        Prim { name: "number?", ptr: clj_number_p as *const u8, arity: 1 },
        Prim { name: "not",     ptr: clj_not as *const u8,      arity: 1 },
        Prim { name: "cons",    ptr: clj_cons as *const u8,     arity: 2 },
        Prim { name: "__concat", ptr: clj_concat as *const u8,   arity: 2 },
        Prim { name: "make-array", ptr: clj_make_array as *const u8, arity: 1 },
        Prim { name: "aget",       ptr: clj_aget as *const u8,       arity: 2 },
        Prim { name: "aset",       ptr: clj_aset as *const u8,       arity: 3 },
        Prim { name: "alength",    ptr: clj_alength as *const u8,    arity: 1 },
        Prim { name: "aclone",     ptr: clj_aclone as *const u8,     arity: 1 },
        // ── Reader-collection bridges (called by core.clj's
        //    extend-type __ReaderXxx blocks) ────────────────────────
        Prim { name: "__reader_list_first",  ptr: clj_reader_list_first as *const u8, arity: 1 },
        Prim { name: "__reader_list_rest",   ptr: clj_reader_list_rest as *const u8,  arity: 1 },
        Prim { name: "__reader_list_count",  ptr: clj_reader_list_count as *const u8, arity: 1 },
        Prim { name: "__reader_list_nth",    ptr: clj_reader_list_nth as *const u8,   arity: 2 },
        Prim { name: "__reader_vector_count", ptr: clj_reader_vector_count as *const u8, arity: 1 },
        Prim { name: "__reader_vector_nth",   ptr: clj_reader_vector_nth as *const u8,   arity: 2 },
        Prim { name: "__reader_vector_first", ptr: clj_reader_vector_first as *const u8, arity: 1 },
        Prim { name: "__reader_vector_rest",  ptr: clj_reader_vector_rest as *const u8,  arity: 1 },
        Prim { name: "__reader_set_count",    ptr: clj_reader_set_count as *const u8,    arity: 1 },
        Prim { name: "__reader_set_get",      ptr: clj_reader_set_get as *const u8,      arity: 2 },
        Prim { name: "__reader_set_contains", ptr: clj_reader_set_contains as *const u8, arity: 2 },
        Prim { name: "__reader_set_conj",     ptr: clj_reader_set_conj as *const u8,     arity: 2 },
        Prim { name: "__reader_map_count",    ptr: clj_reader_map_count as *const u8,    arity: 1 },
        Prim { name: "__reader_map_lookup",   ptr: clj_reader_map_lookup as *const u8,   arity: 3 },
        Prim { name: "__reader_map_keys",     ptr: clj_reader_map_keys as *const u8,     arity: 1 },
        Prim { name: "__reader_map_assoc",    ptr: clj_reader_map_assoc as *const u8,    arity: 3 },
        Prim { name: "__alloc_fn",  ptr: clj_alloc_fn as *const u8, arity: 2 },
        Prim { name: "__alloc_closure", ptr: clj_alloc_closure as *const u8, arity: 4 },
        Prim { name: "__arity_check", ptr: clj_arity_check as *const u8, arity: 2 },
    ];
    PRIMS
}
