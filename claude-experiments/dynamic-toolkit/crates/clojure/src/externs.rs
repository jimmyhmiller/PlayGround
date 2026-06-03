//! C-callable Rust externs that the Clojure JIT calls into.
//!
//! Each extern takes/returns NanBox-encoded `u64`. Numeric externs
//! always return floats — `from_int` and `from_f64` both encode into
//! the unboxed-float NaN payload, so an int+int still rides the
//! float fast path.

use crate::namespace as ns;
use crate::value::*;
use dynir::ir::FuncRef;

// ── Exception object construction & accessors ──────────────────────
//
// Exceptions are heap Records with a uniform 3-field shape:
//   field 0: message  (String or NIL)
//   field 1: data     (Map or NIL)
//   field 2: cause    (an exception value or NIL)
//
// The record's `type_name` (sym-id stored in the record header)
// identifies the exception class — `'ExceptionInfo` for user
// `(ex-info ...)` constructions, `'ArityException`,
// `'ClassCastException`, etc. for runtime errors thrown by the JIT's
// own externs. `(catch T name body)` filters by this type_name via
// `instance?`-style semantics.
//
// We don't go through `deftype*` for these because deftype's
// `deftype_fields` registry is for `(.-field rec)` accessor lookup,
// and exception fields are accessed positionally via the dedicated
// `ex-message`/`ex-data`/`ex-cause` externs below.

/// Build an exception record with the given type_name symbol and
/// (message, data, cause) fields. Both Rust externs (constructing
/// internal exceptions like ArityException) and the user-facing
/// `ex-info` go through this.
pub fn build_exception_record(type_sym_id: u32, message: u64, data: u64, cause: u64) -> u64 {
    let type_name = encode_sym_id(type_sym_id);
    let fields = [message, data, cause];
    dynobj::roots::with_scope(8, |scope| {
        crate::collections::alloc_record(scope, type_name, &fields).get()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_ex_info_2(message: u64, data: u64) -> u64 {
    let type_sym = crate::host::with_host(|h| h.sym.intern("ExceptionInfo"));
    build_exception_record(type_sym, message, data, NIL)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_ex_info_3(message: u64, data: u64, cause: u64) -> u64 {
    let type_sym = crate::host::with_host(|h| h.sym.intern("ExceptionInfo"));
    build_exception_record(type_sym, message, data, cause)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_ex_message(e: u64) -> u64 {
    if !crate::collections::is_record(e) {
        // Non-record exception (e.g., a string or keyword the user
        // threw): return nil rather than panic.
        return NIL;
    }
    if crate::collections::record_field_count(e) < 1 {
        return NIL;
    }
    crate::collections::record_field(e, 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_ex_data(e: u64) -> u64 {
    if !crate::collections::is_record(e) {
        return NIL;
    }
    if crate::collections::record_field_count(e) < 2 {
        return NIL;
    }
    crate::collections::record_field(e, 1)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_ex_cause(e: u64) -> u64 {
    if !crate::collections::is_record(e) {
        return NIL;
    }
    if crate::collections::record_field_count(e) < 3 {
        return NIL;
    }
    crate::collections::record_field(e, 2)
}

/// True iff `e` is an exception record whose `type_name` matches
/// `expected_type_sym_id` (an interned symbol id). Used by the
/// compiler for `(catch T name body)` filtering.
#[unsafe(no_mangle)]
pub extern "C" fn clj_exception_type_matches(e: u64, expected_type_sym_id: u64) -> u64 {
    if !crate::collections::is_record(e) {
        return FALSE;
    }
    let type_name = crate::collections::record_type_name(e);
    if !is_sym_id(type_name) {
        return FALSE;
    }
    let actual = as_sym_id(type_name);
    if actual as u64 == expected_type_sym_id {
        TRUE
    } else {
        FALSE
    }
}

// ── Exception-raise stub (assembly) ─────────────────────────────────
//
// Returns a `JitOutcome::Exception(payload)` to the JIT call site
// instead of a normal return value. Follows dynlower's outcome
// register convention:
//   x0 = result        (unused for Exception)
//   x1 = outcome kind  (= JitOutcomeKind::Exception = 2)
//   x2 = payload0      (= the thrown value, our `v`)
//
// The JIT call site (control-aware) checks x1: if not ReturnValue/
// ReturnVoid, it propagates the outcome up — so a deep-nested throw
// bubbles through every intermediate call automatically. An `Invoke`
// site additionally routes Exception to its handler block.
//
// On entry, the value-to-throw is in x0 (first arg per AAPCS). We
// move it to x2 (payload0), set x1=2, and `ret`.

#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    ".globl _clj_raise_exception",
    "_clj_raise_exception:",
    "mov x2, x0", // payload0 = arg
    "mov x1, #2", // kind = Exception
    "ret",
);

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    pub fn clj_raise_exception(v: u64) -> u64;
}

#[cfg(not(target_arch = "aarch64"))]
compile_error!(
    "clj_raise_exception is only implemented for aarch64. Add an asm \
     stub for your architecture (set outcome kind = 2 in the \
     appropriate kind register, payload0 = arg, then ret)."
);

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

/// `(quot a b)` — integer division, truncating toward zero. Matches
/// Clojure's quot semantics (`(quot 7 2)` → 3, `(quot -7 2)` → -3).
/// Falls back to float trunc when the inputs aren't integers.
#[unsafe(no_mangle)]
pub extern "C" fn clj_quot(a: u64, b: u64) -> u64 {
    let q = num(a) / num(b);
    encode_num(q.trunc())
}

/// `(rem a b)` — the remainder of `(quot a b)` so that
/// `a == (quot a b) * b + (rem a b)`. Sign follows `a`.
#[unsafe(no_mangle)]
pub extern "C" fn clj_rem(a: u64, b: u64) -> u64 {
    let x = num(a);
    let y = num(b);
    encode_num(x - (x / y).trunc() * y)
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
    dynobj::roots::with_scope(3, |scope| alloc_list_cell_from_raw(scope, head, tail).get())
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

/// Build a Vector from a __ReaderList of element values. Used by
/// the compiler to evaluate vector literals containing live forms:
/// `[a b c]` compiles to a list-build of the lowered values, then
/// this extern converts the list into a Vector.
#[unsafe(no_mangle)]
pub extern "C" fn clj_vector_from_list(list: u64) -> u64 {
    let items: Vec<u64> = if is_nil(list) {
        Vec::new()
    } else if crate::collections::is_list(list) {
        crate::value::list_iter(list).collect()
    } else {
        panic!("__vector_from_list: expected a list, got 0x{list:016x}");
    };
    dynobj::roots::with_scope(items.len() + 4, |scope| {
        crate::collections::alloc_vector(scope, &items).get()
    })
}

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
    let src: Vec<u64> = (0..n)
        .map(|i| crate::collections::array_get(arr, i))
        .collect();
    dynobj::roots::with_scope(2, |scope| {
        crate::collections::alloc_array(scope, &src).get()
    })
}

// ── Bit ops ─────────────────────────────────────────────────────────
//
// core.clj uses these for hashing and persistent-collection
// internals. NanBox encodes integers as f64; we round-trip
// `f64 → i64 → bit-op → encoded i64`.

#[inline]
fn to_i64(v: u64) -> i64 {
    if !is_number(v) {
        panic!("bit op: argument is not a number (got 0x{v:016x})");
    }
    as_number(v) as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_bit_and(a: u64, b: u64) -> u64 {
    encode_int(to_i64(a) & to_i64(b))
}
#[unsafe(no_mangle)]
pub extern "C" fn clj_bit_or(a: u64, b: u64) -> u64 {
    encode_int(to_i64(a) | to_i64(b))
}
#[unsafe(no_mangle)]
pub extern "C" fn clj_bit_xor(a: u64, b: u64) -> u64 {
    encode_int(to_i64(a) ^ to_i64(b))
}
#[unsafe(no_mangle)]
pub extern "C" fn clj_bit_not(a: u64) -> u64 {
    encode_int(!to_i64(a))
}
#[unsafe(no_mangle)]
pub extern "C" fn clj_bit_shift_left(a: u64, b: u64) -> u64 {
    encode_int(to_i64(a).wrapping_shl(to_i64(b) as u32 & 63))
}
#[unsafe(no_mangle)]
pub extern "C" fn clj_bit_shift_right(a: u64, b: u64) -> u64 {
    // Arithmetic shift right (preserves sign).
    encode_int(to_i64(a).wrapping_shr(to_i64(b) as u32 & 63))
}
#[unsafe(no_mangle)]
pub extern "C" fn clj_unsigned_bit_shift_right(a: u64, b: u64) -> u64 {
    // Logical (zero-fill) shift right.
    encode_int(((to_i64(a) as u64).wrapping_shr(to_i64(b) as u32 & 63)) as i64)
}

// ── Numeric / type predicates ───────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn clj_integer_p(v: u64) -> u64 {
    if !is_number(v) {
        return FALSE;
    }
    let n = as_number(v);
    bool_(n.fract() == 0.0 && n.is_finite())
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_float_p(v: u64) -> u64 {
    if !is_number(v) {
        return FALSE;
    }
    let n = as_number(v);
    bool_(n.fract() != 0.0 || !n.is_finite())
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_string_p(v: u64) -> u64 {
    bool_(crate::collections::is_string(v))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_keyword_p(v: u64) -> u64 {
    bool_(crate::collections::is_keyword(v))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_symbol_p(v: u64) -> u64 {
    bool_(is_sym_id(v))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_fn_p(v: u64) -> u64 {
    bool_(crate::collections::is_fn(v))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_identical_p(a: u64, b: u64) -> u64 {
    bool_(a == b)
}

// ── List/cons primitives ────────────────────────────────────────────
//
// `cons-first` / `cons-rest` / `cons?` are the lowest-level list
// accessors core.clj expects (mirroring the POC's naming).

#[unsafe(no_mangle)]
pub extern "C" fn clj_cons_p(v: u64) -> u64 {
    bool_(crate::collections::is_list(v))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_cons_first(v: u64) -> u64 {
    if is_nil(v) {
        return NIL;
    }
    if !crate::collections::is_list(v) {
        panic!("cons-first: not a list (got 0x{v:016x})");
    }
    crate::value::first(v)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_cons_rest(v: u64) -> u64 {
    if is_nil(v) {
        return NIL;
    }
    if !crate::collections::is_list(v) {
        panic!("cons-rest: not a list (got 0x{v:016x})");
    }
    crate::value::rest(v)
}

// ── String ops ──────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn clj_string_count(s: u64) -> u64 {
    if !crate::collections::is_string(s) {
        panic!("__string_count: not a string");
    }
    encode_int(crate::collections::string_len(s) as i64)
}

/// `__str_concat seq` — concat the `(str x)`-form of each element in
/// the seq into a single string. The contract matches core.clj's
/// `(defn str …)`, which is the sole caller: `([x] (__str_concat (list x)))`
/// and `([x & ys] (__str_concat (cons x ys)))`. Each element is
/// stringified via `printer::str_repr` (bare contents for strings,
/// "" for nil, otherwise the standard pr-form).
#[unsafe(no_mangle)]
pub extern "C" fn clj_str_concat(seq: u64) -> u64 {
    let mut out = String::new();
    if !is_nil(seq) {
        for elem in crate::collections::seq_iter(seq) {
            crate::host::with_host(|h| {
                let sym = &h.sym;
                out.push_str(&crate::printer::str_repr(elem, &sym));
            });
        }
    }
    dynobj::roots::with_scope(2, |scope| {
        crate::collections::alloc_string(scope, out.as_bytes()).get()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_subs(s: u64, start: u64) -> u64 {
    if !crate::collections::is_string(s) {
        panic!("__subs: first arg must be a string");
    }
    let from = to_i64(start) as usize;
    let bs = crate::collections::string_bytes(s);
    let slice = &bs[from.min(bs.len())..];
    let owned: Vec<u8> = slice.to_vec();
    dynobj::roots::with_scope(2, |scope| {
        crate::collections::alloc_string(scope, &owned).get()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_subs_3(s: u64, start: u64, end: u64) -> u64 {
    if !crate::collections::is_string(s) {
        panic!("__subs: first arg must be a string");
    }
    let from = to_i64(start) as usize;
    let to = to_i64(end) as usize;
    let bs = crate::collections::string_bytes(s);
    let slice = &bs[from.min(bs.len())..to.min(bs.len())];
    let owned: Vec<u8> = slice.to_vec();
    dynobj::roots::with_scope(2, |scope| {
        crate::collections::alloc_string(scope, &owned).get()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_keyword_name(k: u64) -> u64 {
    if !crate::collections::is_keyword(k) {
        panic!("__keyword_name: not a keyword");
    }
    let sym_id = crate::collections::keyword_sym_id(k);
    let name = crate::host::with_host(|h| h.sym.name(sym_id).to_string());
    dynobj::roots::with_scope(2, |scope| {
        crate::collections::alloc_string(scope, name.as_bytes()).get()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_keyword_namespace(_k: u64) -> u64 {
    // Silently returning nil hides namespace data the caller passed in
    // (e.g. `(namespace :foo/bar)` would lose `"foo"`). Refuse until
    // namespaced keywords are actually tracked.
    unimplemented!(
        "(namespace ...) on keywords: namespaced keywords not yet \
         supported (see TODO.md). Refusing to silently return nil."
    );
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_keyword_from_string_1(s: u64) -> u64 {
    if !crate::collections::is_string(s) {
        panic!("__keyword_from_string: not a string");
    }
    let bytes = crate::collections::string_bytes(s);
    let name = std::str::from_utf8(bytes).expect("invalid utf-8 in keyword name");
    let id = crate::host::with_host(|h| h.sym.intern(name));
    crate::host::with_host(|h| h.intern_keyword(id))
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_keyword_from_string_2(_ns: u64, _name: u64) -> u64 {
    // Silently dropping the namespace produces a keyword that prints
    // as `:name` instead of `:ns/name` and compares unequal to itself.
    // Refuse until namespaced keywords are tracked.
    unimplemented!(
        "(keyword ns name): namespaced keywords not yet supported \
         (see TODO.md). Refusing to silently drop the namespace."
    );
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_compare(a: u64, b: u64) -> u64 {
    // Numeric ordering; extend later for strings/keywords.
    if is_number(a) && is_number(b) {
        let x = as_number(a);
        let y = as_number(b);
        if x < y {
            return encode_int(-1);
        }
        if x > y {
            return encode_int(1);
        }
        return encode_int(0);
    }
    panic!("__compare: only numeric compare supported in this checkpoint");
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_pr_str(v: u64) -> u64 {
    let s = crate::host::with_host(|h| {
        let sym = &h.sym;
        crate::printer::print(v, &sym)
    });
    dynobj::roots::with_scope(2, |scope| {
        crate::collections::alloc_string(scope, s.as_bytes()).get()
    })
}

/// Get a symbol's name as a String. Works on both interned sym-ids
/// (the common case for code-as-data) and heap Symbol pointers (which
/// we don't yet allocate but the reader could).
#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_symbol_name(s: u64) -> u64 {
    if !is_sym_id(s) {
        panic!("__reader_symbol_name: argument must be a symbol");
    }
    let id = as_sym_id(s);
    let name = crate::host::with_host(|h| h.sym.name(id).to_string());
    dynobj::roots::with_scope(2, |scope| {
        crate::collections::alloc_string(scope, name.as_bytes()).get()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_reader_symbol_namespace(_s: u64) -> u64 {
    // Silently returning nil hides the namespace half of `foo/bar`.
    // Refuse until namespaced symbols are tracked.
    unimplemented!(
        "(namespace ...) on symbols: namespaced symbols not yet \
         supported (see TODO.md). Refusing to silently return nil."
    );
}

// ── Symbol constructor / gensym ─────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn clj_symbol_1(name_str: u64) -> u64 {
    if !crate::collections::is_string(name_str) {
        panic!("symbol: argument must be a string");
    }
    let bs = crate::collections::string_bytes(name_str);
    let name = std::str::from_utf8(bs).expect("invalid utf-8 in symbol name");
    let id = crate::host::with_host(|h| h.sym.intern(name));
    encode_sym_id(id)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_symbol_2(_ns: u64, _name: u64) -> u64 {
    // Silently dropping the namespace produces a symbol that prints
    // as `name` instead of `ns/name`. Refuse until namespaced symbols
    // are tracked.
    unimplemented!(
        "(symbol ns name): namespaced symbols not yet supported \
         (see TODO.md). Refusing to silently drop the namespace."
    );
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_gensym_0() -> u64 {
    let id = crate::host::with_host(|h| h.sym.gensym("G__"));
    encode_sym_id(id)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_gensym_1(prefix: u64) -> u64 {
    if !crate::collections::is_string(prefix) {
        panic!("gensym: prefix must be a string");
    }
    let bs = crate::collections::string_bytes(prefix);
    let name = std::str::from_utf8(bs).expect("invalid utf-8 in gensym prefix");
    let id = crate::host::with_host(|h| h.sym.gensym(name));
    encode_sym_id(id)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_set_macro(var_or_sym: u64) -> u64 {
    // `(set-macro! NAME)` flips the :macro flag on a Var. We accept
    // either the Var pointer or the symbol-id of its name.
    let var = if is_sym_id(var_or_sym) {
        let core_ns = crate::host::with_host(|h| h.core_ns());
        crate::namespace::ns_lookup(core_ns, var_or_sym)
    } else {
        var_or_sym
    };
    if !is_ptr(var) {
        panic!("__set_macro!: target is not a Var");
    }
    crate::namespace::var_set_flag(var, crate::namespace::FLAG_MACRO);
    NIL
}

// ── I/O ─────────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn clj_println(v: u64) -> u64 {
    let s = crate::host::with_host(|h| {
        let sym = &h.sym;
        crate::printer::print(v, &sym)
    });
    println!("{s}");
    NIL
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_print(v: u64) -> u64 {
    let s = crate::host::with_host(|h| {
        let sym = &h.sym;
        crate::printer::print(v, &sym)
    });
    print!("{s}");
    NIL
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_newline() -> u64 {
    println!();
    NIL
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_print_space() -> u64 {
    print!(" ");
    NIL
}

// ── Hash primitive ──────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn clj_hash_primitive(v: u64) -> u64 {
    // FNV-1a over the value's bit pattern. core.clj layers
    // value-specific hashing on top.
    let bytes = v.to_le_bytes();
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    encode_int(h as i64)
}

// ── Throw ───────────────────────────────────────────────────────────
//
// `(throw v)` panics with `v`'s printed form. Until we wire up the
// dynlower prompt machinery for `try`/`catch`, throws abort the
// thread (and on the test path, the process). Returns a discardable
// `0` so it shares the unified extern signature; the caller treats
// the call as diverging.

#[unsafe(no_mangle)]
pub extern "C" fn clj_throw(v: u64) -> u64 {
    crate::host::with_host(|h| {
        let sym = &h.sym;
        let printed = crate::printer::print(v, &sym);
        panic!("Exception: {}", printed);
    })
}

// ── Atoms ───────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn clj_atom(initial: u64) -> u64 {
    dynobj::roots::with_scope(2, |scope| {
        crate::collections::alloc_atom(scope, initial).get()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_deref(a: u64) -> u64 {
    if !crate::collections::is_atom(a) {
        panic!("deref: not an atom (got 0x{a:016x})");
    }
    crate::collections::atom_get(a)
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_reset(a: u64, v: u64) -> u64 {
    if !crate::collections::is_atom(a) {
        panic!("reset!: not an atom (got 0x{a:016x})");
    }
    crate::collections::atom_set(a, v);
    v
}

// ── Late-bound Var resolution ───────────────────────────────────────
//
// `(__resolve_var 'name)` returns the current root of the Var named
// `name` in `clojure.core`, panicking if the Var doesn't exist yet.
// Used by the compiler when a bare symbol reference can't be resolved
// to a Var at compile time (forward reference). Allows core.clj's
// many forward references — `apply` defined late, used early — to
// work without an explicit `(declare apply)` pass.

#[unsafe(no_mangle)]
pub extern "C" fn clj_resolve_var(name_sym: u64) -> u64 {
    if !is_sym_id(name_sym) {
        panic!("__resolve_var: name must be a symbol");
    }
    let core_ns = crate::host::with_host(|h| h.core_ns());
    if !is_ptr(core_ns) {
        panic!("__resolve_var: clojure.core ns not initialized");
    }
    let var = crate::namespace::ns_lookup(core_ns, name_sym);
    if !is_ptr(var) {
        let name = crate::host::with_host(|h| h.sym.name(as_sym_id(name_sym)).to_string());
        panic!("undefined variable (runtime): {}", name);
    }
    crate::namespace::var_root(var)
}

// ── apply ───────────────────────────────────────────────────────────
//
// `(__apply f args-list)` invokes `f` with the elements of
// `args-list` as its arguments. With the unified single-list ABI
// every user-defined fn already takes `(self_fn, args_list)`, so
// `apply` is just an indirect call routed through the callee's
// `func_ref`.

#[unsafe(no_mangle)]
pub extern "C" fn clj_apply(f: u64, args_list: u64) -> u64 {
    if !is_ptr(f) {
        panic!("apply: callee is not a function (got 0x{f:016x})");
    }
    // Callee bodies walk `args_list` via __reader_list_first /
    // __reader_list_rest (the built-in __ReaderList accessors).
    // `seq_iter` already dispatches to `-seq` / `-first` / `-next`
    // for non-list inputs (PList/Cons/PersistentVector/IndexedSeq/
    // …), so collect through it and rebuild as a __ReaderList. The
    // fast-path skip avoids an alloc when the caller already passes
    // a built-in list.
    let reader_list = if is_nil(args_list) || crate::collections::is_list(args_list) {
        args_list
    } else {
        let items: Vec<u64> = crate::collections::seq_iter(args_list).collect();
        dynobj::roots::with_scope(items.len() + 4, |scope| {
            let acc = scope.root::<crate::value::NanBoxTag>(NIL);
            for &x in items.iter().rev() {
                let cell = dynobj::roots::with_scope(3, |inner| {
                    crate::value::alloc_list_cell_from_raw(inner, x, acc.get()).get()
                });
                acc.set(cell);
            }
            acc.get()
        })
    };
    let fr = ns::fn_func_ref(f);
    let code_ptr = crate::host::with_host(|h| {
        assert!(!h.jit.is_null(), "apply: host has no JitModule");
        let jit = unsafe { &*h.jit };
        jit.function_ptr(FuncRef::from_u32(fr))
    });
    // Single-list ABI: (self_fn, args_list).
    let typed: extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(code_ptr) };
    typed(f, reader_list)
}

// ── Bind a name to an arbitrary value at runtime ────────────────────
//
// Used by `deftype*` to make bare references to the type's name
// evaluate to its identifying symbol — so `(instance? MyType x)`
// can resolve `MyType` to a value (the symbol `'MyType`).

#[unsafe(no_mangle)]
pub extern "C" fn clj_def_value(name_sym: u64, value: u64) -> u64 {
    if !is_sym_id(name_sym) {
        panic!("__def_value: name must be a symbol");
    }
    let core_ns = crate::host::with_host(|h| h.core_ns());
    if !is_ptr(core_ns) {
        panic!("__def_value: clojure.core ns not initialized");
    }
    dynobj::roots::with_scope(16, |scope| {
        let _ = crate::namespace::ns_intern(scope, core_ns, name_sym, value);
    });
    value
}

// ── deftype / extend-type / .method dispatch ────────────────────────

/// Allocate a Record instance for `(MyType. field0 field1 …)`. The
/// caller passes the type-name symbol-id and the field values via a
/// stack buffer (same shape as the closure-capture allocator). The
/// fresh Record is rooted in a one-shot scope and returned as a raw
/// tagged pointer.
#[unsafe(no_mangle)]
pub extern "C" fn clj_alloc_record(type_name: u64, n_fields: u64, fields_ptr: *const u64) -> u64 {
    let n = n_fields as usize;
    let fields: Vec<u64> = unsafe { std::slice::from_raw_parts(fields_ptr, n).to_vec() };
    dynobj::roots::with_scope(n + 4, |scope| {
        crate::collections::alloc_record(scope, type_name, &fields).get()
    })
}

/// `(.-name instance)` — read a field from a Record by name. The
/// receiver's `type_name` indexes the host's `deftype_fields` map
/// to find the field's position in the varlen tail.
#[unsafe(no_mangle)]
pub extern "C" fn clj_record_get_field(rec: u64, field_name: u64) -> u64 {
    if !crate::collections::is_record(rec) {
        panic!("field access: not a record (got 0x{rec:016x})");
    }
    if !is_sym_id(field_name) {
        panic!("field access: field name must be a symbol");
    }
    let field_sym = as_sym_id(field_name);
    let type_name = crate::collections::record_type_name(rec);
    if !is_sym_id(type_name) {
        panic!("field access: record's type_name is not a symbol");
    }
    let type_sym = as_sym_id(type_name);
    let idx = crate::host::with_host(|h| {
        let map = h.deftype_fields.lock().unwrap();
        let fields = map.get(&type_sym).unwrap_or_else(|| {
            panic!("field access: type-name not registered in deftype_fields");
        });
        fields
            .iter()
            .position(|&f| f == field_sym)
            .unwrap_or_else(|| {
                panic!("field access: no such field on this type");
            })
    });
    crate::collections::record_field(rec, idx)
}

/// `(set! (.-field rec) val)` — store val into the named field.
///
/// Per-field `^:mutable` tracking isn't implemented yet, so we cannot
/// distinguish a legitimate mutable-field write from a contract
/// violation. Rather than silently overwrite a possibly-immutable
/// field, refuse all `set!` on records until deftype* records each
/// field's mutability. (See TODO.md.)
#[unsafe(no_mangle)]
pub extern "C" fn clj_record_set_field(_rec: u64, _field_name: u64, _val: u64) -> u64 {
    unimplemented!(
        "(set! (.-field rec) val): per-field `^:mutable` tracking not \
         yet implemented (see TODO.md). Refusing to silently overwrite \
         a possibly-immutable field."
    );
}

/// `(instance? Type x)` — true iff x is a Record whose type_name
/// matches the given symbol.
/// `(satisfies? Protocol x)` — true if x's type has any method
/// registered under any name that core.clj would associate with
/// the protocol. Our implementation is permissive: it always
/// returns false. This makes core.clj's polymorphic `=` skip the
/// `-equiv` fallback for non-bitwise-eq inputs, which is correct
/// for the value-types we already structurally compare via
/// `clj_eq`. A precise implementation needs us to track protocol
/// `(satisfies? Proto x)` — does x's type claim Proto via extend-type?
/// We track these claims explicitly (see `__register_protocol_member`)
/// rather than inferring from the method table — marker protocols
/// like `IList` declare no methods but still need to report true.
#[unsafe(no_mangle)]
pub extern "C" fn clj_satisfies_p(proto: u64, x: u64) -> u64 {
    if !is_sym_id(proto) || !is_ptr(x) {
        return FALSE;
    }
    let proto_sym = as_sym_id(proto);
    let type_sym = match value_type_name_sym(x) {
        Some(s) => s,
        None => return FALSE,
    };
    crate::host::with_host(|h| {
        let m = h.protocol_membership.lock().unwrap();
        bool_(m.contains(&(type_sym, proto_sym)))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn clj_instance_p(type_name: u64, x: u64) -> u64 {
    if !is_sym_id(type_name) || !is_ptr(x) {
        return FALSE;
    }
    let want = as_sym_id(type_name);
    let got = match value_type_name_sym(x) {
        Some(s) => s,
        None => return FALSE,
    };
    bool_(want == got)
}

/// Look up the user-facing type-name symbol for a heap value:
/// records carry their type name in the instance, built-ins use the
/// `builtin_type_names` map registered in the host. Returns None for
/// non-heap values (numbers, bools, nil) — which never match a
/// deftype-style protocol/instance check.
fn value_type_name_sym(x: u64) -> Option<u32> {
    if !is_ptr(x) {
        return None;
    }
    if crate::collections::is_record(x) {
        let t = crate::collections::record_type_name(x);
        if is_sym_id(t) {
            return Some(as_sym_id(t));
        }
        return None;
    }
    let type_id = unsafe { crate::value::read_type_id(crate::value::as_ptr(x)) } as usize;
    crate::host::with_host(|h| h.builtin_type_names.lock().unwrap().get(&type_id).copied())
}

/// `(.method-name receiver args…)` — dispatch by reading the
/// receiver's type_name and looking up the method in the host's
/// method_table. Returns the matching `Fn` heap pointer (the caller
/// then call-indirects through it).
#[unsafe(no_mangle)]
pub extern "C" fn clj_method_lookup(method_name: u64, receiver: u64) -> u64 {
    if !is_sym_id(method_name) {
        panic!("method dispatch: method name must be a symbol");
    }
    if !is_ptr(receiver) {
        panic!("method dispatch: receiver is not a heap object");
    }
    // Records carry their type name on the instance; built-ins look
    // it up in the host's builtin_type_names registry. Either way we
    // want a sym-id to use as the method-table key.
    let t = match value_type_name_sym(receiver) {
        Some(s) => s,
        None => panic!(
            "method dispatch: receiver has no registered type-name (heap value 0x{receiver:016x})"
        ),
    };
    let m = as_sym_id(method_name);
    let fn_obj = crate::host::with_host(|h| {
        let table = h.method_table.lock().unwrap();
        let idx = table.get(&(m, t)).copied().unwrap_or_else(|| {
            panic!(
                "method dispatch: no method '{}' for type '{}'",
                h.sym.name(m),
                h.sym.name(t)
            );
        });
        h.method_roots.get(idx)
    });
    fn_obj
}

/// Register a method implementation. Called by the runtime side of
/// `(extend-type T (Proto (m [this …] …)))` — it pins the Fn obj in
/// the GC-traced `method_roots` set and records its index in
/// `method_table[(method, type)]`.
#[unsafe(no_mangle)]
pub extern "C" fn clj_register_method(method_name: u64, type_name: u64, fn_obj: u64) -> u64 {
    if !is_sym_id(method_name) || !is_sym_id(type_name) {
        panic!("register-method: method-name and type-name must be symbols");
    }
    let m = as_sym_id(method_name);
    let t = as_sym_id(type_name);
    crate::host::with_host(|h| {
        let idx = h.method_roots.add(fn_obj);
        let mut tbl = h.method_table.lock().unwrap();
        tbl.insert((m, t), idx);
    });
    NIL
}

/// Register that a type satisfies a protocol. Called by the
/// `extend-type` registration thunk for every protocol named in the
/// extend body — both protocols whose methods got implemented and
/// marker protocols that just declare membership.
#[unsafe(no_mangle)]
pub extern "C" fn clj_register_protocol_member(type_name: u64, proto_name: u64) -> u64 {
    if !is_sym_id(type_name) || !is_sym_id(proto_name) {
        panic!("register-protocol-member: both names must be symbols");
    }
    let t = as_sym_id(type_name);
    let p = as_sym_id(proto_name);
    crate::host::with_host(|h| {
        let mut s = h.protocol_membership.lock().unwrap();
        s.insert((t, p));
    });
    NIL
}

/// Register a deftype's field list. Called once per
/// `(deftype* Name [fields…])` so the runtime field-name → index
/// lookup in `clj_record_get_field` works.
#[unsafe(no_mangle)]
pub extern "C" fn clj_register_deftype(
    type_name: u64,
    fields_count: u64,
    fields_ptr: *const u64,
) -> u64 {
    if !is_sym_id(type_name) {
        panic!("register-deftype: type-name must be a symbol");
    }
    let t = as_sym_id(type_name);
    let n = fields_count as usize;
    let fields: Vec<u32> = unsafe {
        std::slice::from_raw_parts(fields_ptr, n)
            .iter()
            .map(|&f| {
                if !is_sym_id(f) {
                    panic!("register-deftype: every field name must be a symbol");
                }
                as_sym_id(f)
            })
            .collect()
    };
    crate::host::with_host(|h| {
        let mut map = h.deftype_fields.lock().unwrap();
        map.insert(t, fields);
    });
    NIL
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
    let items: Vec<u64> = (1..len)
        .map(|i| crate::collections::vector_get(vec, i))
        .collect();
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
    let mut items: Vec<u64> = (0..n)
        .map(|i| crate::collections::vector_get(backing, i))
        .collect();
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

/// `(get coll key)` — 2-arg form, returns nil on miss. Works on
/// Maps today; should be extended to Vectors (index lookup) and
/// Sets (membership) when those layouts stabilize.
#[unsafe(no_mangle)]
pub extern "C" fn clj_get_2(coll: u64, key: u64) -> u64 {
    if is_nil(coll) {
        return NIL;
    }
    if crate::collections::is_map(coll) {
        return clj_reader_map_lookup(coll, key, NIL);
    }
    // For other collection types (vector, set, list, …) `get` either
    // doesn't apply or has different semantics. Return nil rather
    // than panicking — matches Clojure's "no match" convention.
    NIL
}

/// `(get coll key not-found)` — 3-arg form with explicit default.
#[unsafe(no_mangle)]
pub extern "C" fn clj_get_3(coll: u64, key: u64, not_found: u64) -> u64 {
    if is_nil(coll) {
        return not_found;
    }
    if crate::collections::is_map(coll) {
        return clj_reader_map_lookup(coll, key, not_found);
    }
    not_found
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
    let keys: Vec<u64> = (0..n)
        .map(|i| crate::namespace::map_entry(map, i).0)
        .collect();
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

// ── Runtime arity check ─────────────────────────────────────────────
//
// The IR-emitting allocation path lives in `dynlang::closure::
// ClosureKit::make` (it emits inline `__gc_alloc__` + field stores
// directly, no extern thunk). `namespace::alloc_fn` (Rust-only) stays
// for engine-init paths that wrap externs into Fn objects.

//
// Higher-order calls are dispatched in IR via `call_indirect` through
// a code pointer fetched from the JitModule's call_table. There is no
// per-arity dispatch stub, so the runtime arity check lives here as
// its own extern. The caller emits a call to `clj_arity_check` BEFORE
// the indirect call; if the receiver is not an Fn or its declared
// arity differs from the call-site's argument count, we panic.

/// Validate the length of a (already-packed) args_list against an
/// encoded arity word. Returns NIL on success, an ArityException
/// record on failure. The IR caller emits a branch on `!= NIL` to
/// route the failure path through `__raise_exception` (catchable as
/// `(catch ArityException e ...)`).
#[unsafe(no_mangle)]
pub extern "C" fn clj_check_args_list(args_list: u64, arity_word: u64) -> u64 {
    const VARIADIC_BIT: u64 = 1 << 63;
    let is_variadic = (arity_word & VARIADIC_BIT) != 0;
    let min_arity = (arity_word & !VARIADIC_BIT) as usize;
    let n = if is_nil(args_list) {
        0
    } else if !crate::collections::is_list(args_list) {
        let type_sym = crate::host::with_host(|h| h.sym.intern("ArityException"));
        let msg = make_string_value("callee body expected a list of args".to_string());
        return build_exception_record(type_sym, msg, NIL, NIL);
    } else {
        crate::value::list_count(args_list) as usize
    };
    if is_variadic && n < min_arity {
        let type_sym = crate::host::with_host(|h| h.sym.intern("ArityException"));
        let msg = make_string_value(format!(
            "function takes at least {min_arity} arg(s) but was called with {n}"
        ));
        return build_exception_record(type_sym, msg, NIL, NIL);
    }
    if !is_variadic && n != min_arity {
        let type_sym = crate::host::with_host(|h| h.sym.intern("ArityException"));
        let msg = make_string_value(format!(
            "function takes {min_arity} arg(s) but was called with {n}"
        ));
        return build_exception_record(type_sym, msg, NIL, NIL);
    }
    NIL
}

/// Returns an `ArityException` exception record describing a
/// multi-arity-dispatch failure. The caller compiles this as:
///   `exc = call make_no_matching_arity_exception(args_list)`
///   `call_via_func_ref __raise_exception(exc)` — propagates as
///   `JitOutcome::Exception`, catchable by `(catch ArityException
///   e ...)`.
#[unsafe(no_mangle)]
pub extern "C" fn clj_make_no_matching_arity_exception(args_list: u64) -> u64 {
    let n = if is_nil(args_list) {
        0
    } else if !crate::collections::is_list(args_list) {
        0
    } else {
        crate::value::list_count(args_list) as usize
    };
    let type_sym = crate::host::with_host(|h| h.sym.intern("ArityException"));
    let msg = make_string_value(format!("no matching arity for call with {n} arg(s)"));
    build_exception_record(type_sym, msg, NIL, NIL)
}

/// Legacy panicking form. Kept for now so existing call sites that
/// haven't been converted still work; new code should construct +
/// raise via `make_no_matching_arity_exception` + `__raise_exception`.
#[unsafe(no_mangle)]
pub extern "C" fn clj_no_matching_arity_panic(args_list: u64) -> u64 {
    let exc = clj_make_no_matching_arity_exception(args_list);
    let _ = exc; // we still panic for legacy callers
    panic!("ArityException: no matching arity");
}

/// Construct a Clojure `String` value from a Rust `String`. Helper
/// for building exception messages.
fn make_string_value(s: String) -> u64 {
    dynobj::roots::with_scope(2, |scope| {
        crate::collections::alloc_string(scope, s.as_bytes()).get()
    })
}

/// Validate a call site's arg count against a callee's declared
/// arity. The arity word stored in `Fn.arity` packs both `min_arity`
/// and a `variadic` flag (bit 63 set). Non-variadic callees require
/// exact arity; variadic callees require at least `min_arity` args.
///
/// Returns `NIL` on success, or an exception record on failure. The
/// IR caller emits this as a regular Call, then branches on
/// `result != NIL` to re-route through `__raise_exception` (which
/// propagates as `JitOutcome::Exception`, catchable by `(catch
/// ArityException e ...)` or `(catch CallError e ...)`).
#[unsafe(no_mangle)]
pub extern "C" fn clj_arity_check(fn_v: u64, n_args: u64) -> u64 {
    if !is_ptr(fn_v) {
        let type_sym = crate::host::with_host(|h| h.sym.intern("CallError"));
        let msg = make_string_value(format!(
            "call: callee is not a function (got 0x{fn_v:016x})"
        ));
        return build_exception_record(type_sym, msg, NIL, NIL);
    }
    const VARIADIC_BIT: u64 = 1 << 63;
    let arity_word = ns::fn_arity(fn_v) as u64;
    let is_variadic = (arity_word & VARIADIC_BIT) != 0;
    let min_arity = arity_word & !VARIADIC_BIT;
    if is_variadic && n_args < min_arity {
        let type_sym = crate::host::with_host(|h| h.sym.intern("ArityException"));
        let msg = make_string_value(format!(
            "function takes at least {min_arity} arg(s) but was called with {n_args}"
        ));
        return build_exception_record(type_sym, msg, NIL, NIL);
    }
    if !is_variadic && n_args != min_arity {
        let type_sym = crate::host::with_host(|h| h.sym.intern("ArityException"));
        let msg = make_string_value(format!(
            "function takes {min_arity} arg(s) but was called with {n_args}"
        ));
        return build_exception_record(type_sym, msg, NIL, NIL);
    }
    NIL
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
        Prim {
            name: "+",
            ptr: clj_add as *const u8,
            arity: 2,
        },
        Prim {
            name: "-",
            ptr: clj_sub as *const u8,
            arity: 2,
        },
        Prim {
            name: "*",
            ptr: clj_mul as *const u8,
            arity: 2,
        },
        Prim {
            name: "/",
            ptr: clj_div as *const u8,
            arity: 2,
        },
        Prim {
            name: "quot",
            ptr: clj_quot as *const u8,
            arity: 2,
        },
        Prim {
            name: "rem",
            ptr: clj_rem as *const u8,
            arity: 2,
        },
        // Private aliases so core.clj's `(defn quot …)` etc. (which
        // shadow the externs in func_refs) can still reach the
        // underlying primitives. Same trick as `__cons` for the
        // arg-list builder.
        Prim {
            name: "__quot",
            ptr: clj_quot as *const u8,
            arity: 2,
        },
        Prim {
            name: "__rem",
            ptr: clj_rem as *const u8,
            arity: 2,
        },
        // core.clj uses these "prim-*" names for the binary
        // primitives, then layers a variadic `+` / `-` / etc. on
        // top in user code.
        Prim {
            name: "prim-add",
            ptr: clj_add as *const u8,
            arity: 2,
        },
        Prim {
            name: "prim-sub",
            ptr: clj_sub as *const u8,
            arity: 2,
        },
        Prim {
            name: "prim-mul",
            ptr: clj_mul as *const u8,
            arity: 2,
        },
        Prim {
            name: "prim-div",
            ptr: clj_div as *const u8,
            arity: 2,
        },
        Prim {
            name: "prim-lt",
            ptr: clj_lt as *const u8,
            arity: 2,
        },
        Prim {
            name: "prim-gt",
            ptr: clj_gt as *const u8,
            arity: 2,
        },
        Prim {
            name: "prim-le",
            ptr: clj_le as *const u8,
            arity: 2,
        },
        Prim {
            name: "prim-ge",
            ptr: clj_ge as *const u8,
            arity: 2,
        },
        Prim {
            name: "prim-eq",
            ptr: clj_eq as *const u8,
            arity: 2,
        },
        // `__value_eq` in core.clj is the value-equality fallback
        // for strings, numbers, etc. Our `clj_eq` already does
        // structural compare for those, so aliasing is correct.
        Prim {
            name: "__value_eq",
            ptr: clj_eq as *const u8,
            arity: 2,
        },
        Prim {
            name: "<",
            ptr: clj_lt as *const u8,
            arity: 2,
        },
        Prim {
            name: ">",
            ptr: clj_gt as *const u8,
            arity: 2,
        },
        Prim {
            name: "<=",
            ptr: clj_le as *const u8,
            arity: 2,
        },
        Prim {
            name: ">=",
            ptr: clj_ge as *const u8,
            arity: 2,
        },
        Prim {
            name: "=",
            ptr: clj_eq as *const u8,
            arity: 2,
        },
        Prim {
            name: "==",
            ptr: clj_eq as *const u8,
            arity: 2,
        },
        Prim {
            name: "nil?",
            ptr: clj_nil_p as *const u8,
            arity: 1,
        },
        Prim {
            name: "number?",
            ptr: clj_number_p as *const u8,
            arity: 1,
        },
        Prim {
            name: "not",
            ptr: clj_not as *const u8,
            arity: 1,
        },
        Prim {
            name: "cons",
            ptr: clj_cons as *const u8,
            arity: 2,
        },
        // Private alias the compiler uses internally (e.g. when packing
        // call-site args into a list for the unified closure ABI). User
        // code redefines `cons` in core.clj as a seq-aware fn — that
        // would shadow the raw extern in `func_refs` and break the
        // arg-list builder. Keep a reserved name that user code can't
        // accidentally collide with.
        Prim {
            name: "__cons",
            ptr: clj_cons as *const u8,
            arity: 2,
        },
        Prim {
            name: "__concat",
            ptr: clj_concat as *const u8,
            arity: 2,
        },
        Prim {
            name: "make-array",
            ptr: clj_make_array as *const u8,
            arity: 1,
        },
        Prim {
            name: "__vector_from_list",
            ptr: clj_vector_from_list as *const u8,
            arity: 1,
        },
        Prim {
            name: "aget",
            ptr: clj_aget as *const u8,
            arity: 2,
        },
        Prim {
            name: "aset",
            ptr: clj_aset as *const u8,
            arity: 3,
        },
        Prim {
            name: "alength",
            ptr: clj_alength as *const u8,
            arity: 1,
        },
        Prim {
            name: "aclone",
            ptr: clj_aclone as *const u8,
            arity: 1,
        },
        // ── Reader-collection bridges (called by core.clj's
        //    extend-type __ReaderXxx blocks) ────────────────────────
        Prim {
            name: "__reader_list_first",
            ptr: clj_reader_list_first as *const u8,
            arity: 1,
        },
        Prim {
            name: "__reader_list_rest",
            ptr: clj_reader_list_rest as *const u8,
            arity: 1,
        },
        Prim {
            name: "__reader_list_count",
            ptr: clj_reader_list_count as *const u8,
            arity: 1,
        },
        Prim {
            name: "__reader_list_nth",
            ptr: clj_reader_list_nth as *const u8,
            arity: 2,
        },
        Prim {
            name: "__reader_vector_count",
            ptr: clj_reader_vector_count as *const u8,
            arity: 1,
        },
        Prim {
            name: "__reader_vector_nth",
            ptr: clj_reader_vector_nth as *const u8,
            arity: 2,
        },
        Prim {
            name: "__reader_vector_first",
            ptr: clj_reader_vector_first as *const u8,
            arity: 1,
        },
        Prim {
            name: "__reader_vector_rest",
            ptr: clj_reader_vector_rest as *const u8,
            arity: 1,
        },
        Prim {
            name: "__reader_set_count",
            ptr: clj_reader_set_count as *const u8,
            arity: 1,
        },
        Prim {
            name: "__reader_set_get",
            ptr: clj_reader_set_get as *const u8,
            arity: 2,
        },
        Prim {
            name: "__reader_set_contains",
            ptr: clj_reader_set_contains as *const u8,
            arity: 2,
        },
        Prim {
            name: "__reader_set_conj",
            ptr: clj_reader_set_conj as *const u8,
            arity: 2,
        },
        Prim {
            name: "__reader_map_count",
            ptr: clj_reader_map_count as *const u8,
            arity: 1,
        },
        Prim {
            name: "__reader_map_lookup",
            ptr: clj_reader_map_lookup as *const u8,
            arity: 3,
        },
        Prim {
            name: "get",
            ptr: clj_get_2 as *const u8,
            arity: 2,
        },
        Prim {
            name: "__get_3",
            ptr: clj_get_3 as *const u8,
            arity: 3,
        },
        Prim {
            name: "__reader_map_keys",
            ptr: clj_reader_map_keys as *const u8,
            arity: 1,
        },
        Prim {
            name: "__reader_map_assoc",
            ptr: clj_reader_map_assoc as *const u8,
            arity: 3,
        },
        // ── deftype / extend-type / .method ────────────────────────
        Prim {
            name: "__alloc_record",
            ptr: clj_alloc_record as *const u8,
            arity: 3,
        },
        Prim {
            name: "__record_get_field",
            ptr: clj_record_get_field as *const u8,
            arity: 2,
        },
        Prim {
            name: "__record_set_field",
            ptr: clj_record_set_field as *const u8,
            arity: 3,
        },
        Prim {
            name: "instance?",
            ptr: clj_instance_p as *const u8,
            arity: 2,
        },
        Prim {
            name: "satisfies?",
            ptr: clj_satisfies_p as *const u8,
            arity: 2,
        },
        Prim {
            name: "__method_lookup",
            ptr: clj_method_lookup as *const u8,
            arity: 2,
        },
        Prim {
            name: "__register_method",
            ptr: clj_register_method as *const u8,
            arity: 3,
        },
        Prim {
            name: "__register_protocol_member",
            ptr: clj_register_protocol_member as *const u8,
            arity: 2,
        },
        Prim {
            name: "__register_deftype",
            ptr: clj_register_deftype as *const u8,
            arity: 3,
        },
        Prim {
            name: "__def_value",
            ptr: clj_def_value as *const u8,
            arity: 2,
        },
        Prim {
            name: "__apply",
            ptr: clj_apply as *const u8,
            arity: 2,
        },
        Prim {
            name: "__resolve_var",
            ptr: clj_resolve_var as *const u8,
            arity: 1,
        },
        // ── Atoms ──────────────────────────────────────────────────
        Prim {
            name: "throw",
            ptr: clj_throw as *const u8,
            arity: 1,
        },
        // Real raise: returns JitOutcome::Exception, propagated by
        // every call site (and caught by Invoke sites). Use this from
        // compiled code; `clj_throw` above is only the
        // legacy/uncaught panic path.
        Prim {
            name: "__raise_exception",
            ptr: clj_raise_exception as *const u8,
            arity: 1,
        },
        // Exception construction & accessors.
        Prim {
            name: "__ex_info_2",
            ptr: clj_ex_info_2 as *const u8,
            arity: 2,
        },
        // User-facing arity-2 `ex-info` (most common). Arity-3 form
        // is `__ex_info_3` — wrap in a Clojure-side defn if needed.
        Prim {
            name: "ex-info",
            ptr: clj_ex_info_2 as *const u8,
            arity: 2,
        },
        Prim {
            name: "__ex_info_3",
            ptr: clj_ex_info_3 as *const u8,
            arity: 3,
        },
        Prim {
            name: "ex-message",
            ptr: clj_ex_message as *const u8,
            arity: 1,
        },
        Prim {
            name: "ex-data",
            ptr: clj_ex_data as *const u8,
            arity: 1,
        },
        Prim {
            name: "ex-cause",
            ptr: clj_ex_cause as *const u8,
            arity: 1,
        },
        Prim {
            name: "__exception_type_matches",
            ptr: clj_exception_type_matches as *const u8,
            arity: 2,
        },
        // ── Bit ops (used by core.clj's hashing and persistent
        //    collection internals) ────────────────────────────────
        Prim {
            name: "bit-and",
            ptr: clj_bit_and as *const u8,
            arity: 2,
        },
        Prim {
            name: "bit-or",
            ptr: clj_bit_or as *const u8,
            arity: 2,
        },
        Prim {
            name: "bit-xor",
            ptr: clj_bit_xor as *const u8,
            arity: 2,
        },
        Prim {
            name: "bit-not",
            ptr: clj_bit_not as *const u8,
            arity: 1,
        },
        Prim {
            name: "bit-shift-left",
            ptr: clj_bit_shift_left as *const u8,
            arity: 2,
        },
        Prim {
            name: "bit-shift-right",
            ptr: clj_bit_shift_right as *const u8,
            arity: 2,
        },
        Prim {
            name: "unsigned-bit-shift-right",
            ptr: clj_unsigned_bit_shift_right as *const u8,
            arity: 2,
        },
        Prim {
            name: "bit-shift-right-zero-fill",
            ptr: clj_unsigned_bit_shift_right as *const u8,
            arity: 2,
        },
        // ── Type predicates ───────────────────────────────────────
        Prim {
            name: "integer?",
            ptr: clj_integer_p as *const u8,
            arity: 1,
        },
        Prim {
            name: "int?",
            ptr: clj_integer_p as *const u8,
            arity: 1,
        },
        Prim {
            name: "float?",
            ptr: clj_float_p as *const u8,
            arity: 1,
        },
        Prim {
            name: "string?",
            ptr: clj_string_p as *const u8,
            arity: 1,
        },
        Prim {
            name: "keyword?",
            ptr: clj_keyword_p as *const u8,
            arity: 1,
        },
        Prim {
            name: "symbol?",
            ptr: clj_symbol_p as *const u8,
            arity: 1,
        },
        // POC-era aliases used by core.clj's `(defn string? [x] (__is_string x))` etc.
        Prim {
            name: "__is_string",
            ptr: clj_string_p as *const u8,
            arity: 1,
        },
        Prim {
            name: "__is_symbol",
            ptr: clj_symbol_p as *const u8,
            arity: 1,
        },
        Prim {
            name: "__is_keyword",
            ptr: clj_keyword_p as *const u8,
            arity: 1,
        },
        Prim {
            name: "__reader_symbol_name",
            ptr: clj_reader_symbol_name as *const u8,
            arity: 1,
        },
        Prim {
            name: "__reader_symbol_namespace",
            ptr: clj_reader_symbol_namespace as *const u8,
            arity: 1,
        },
        Prim {
            name: "fn?",
            ptr: clj_fn_p as *const u8,
            arity: 1,
        },
        Prim {
            name: "identical?",
            ptr: clj_identical_p as *const u8,
            arity: 2,
        },
        // ── List/cons primitives ──────────────────────────────────
        Prim {
            name: "cons?",
            ptr: clj_cons_p as *const u8,
            arity: 1,
        },
        Prim {
            name: "cons-first",
            ptr: clj_cons_first as *const u8,
            arity: 1,
        },
        Prim {
            name: "cons-rest",
            ptr: clj_cons_rest as *const u8,
            arity: 1,
        },
        // ── String / keyword / symbol ops ─────────────────────────
        Prim {
            name: "__string_count",
            ptr: clj_string_count as *const u8,
            arity: 1,
        },
        Prim {
            name: "__str_concat",
            ptr: clj_str_concat as *const u8,
            arity: 1,
        },
        Prim {
            name: "__subs",
            ptr: clj_subs as *const u8,
            arity: 2,
        },
        Prim {
            name: "__subs_3",
            ptr: clj_subs_3 as *const u8,
            arity: 3,
        },
        Prim {
            name: "__keyword_name",
            ptr: clj_keyword_name as *const u8,
            arity: 1,
        },
        Prim {
            name: "__keyword_namespace",
            ptr: clj_keyword_namespace as *const u8,
            arity: 1,
        },
        Prim {
            name: "__keyword_from_string_1",
            ptr: clj_keyword_from_string_1 as *const u8,
            arity: 1,
        },
        Prim {
            name: "__keyword_from_string_2",
            ptr: clj_keyword_from_string_2 as *const u8,
            arity: 2,
        },
        Prim {
            name: "__compare",
            ptr: clj_compare as *const u8,
            arity: 2,
        },
        Prim {
            name: "__pr_str",
            ptr: clj_pr_str as *const u8,
            arity: 1,
        },
        Prim {
            name: "__symbol_1",
            ptr: clj_symbol_1 as *const u8,
            arity: 1,
        },
        Prim {
            name: "__symbol_2",
            ptr: clj_symbol_2 as *const u8,
            arity: 2,
        },
        Prim {
            name: "__gensym_0",
            ptr: clj_gensym_0 as *const u8,
            arity: 0,
        },
        Prim {
            name: "__gensym_1",
            ptr: clj_gensym_1 as *const u8,
            arity: 1,
        },
        Prim {
            name: "__set_macro!",
            ptr: clj_set_macro as *const u8,
            arity: 1,
        },
        // ── I/O ───────────────────────────────────────────────────
        Prim {
            name: "_println",
            ptr: clj_println as *const u8,
            arity: 1,
        },
        Prim {
            name: "_print",
            ptr: clj_print as *const u8,
            arity: 1,
        },
        Prim {
            name: "_newline",
            ptr: clj_newline as *const u8,
            arity: 0,
        },
        Prim {
            name: "_print-space",
            ptr: clj_print_space as *const u8,
            arity: 0,
        },
        Prim {
            name: "hash-primitive",
            ptr: clj_hash_primitive as *const u8,
            arity: 1,
        },
        // ── Atom builtins (POC's `__` flavor) ─────────────────────
        Prim {
            name: "__atom_create",
            ptr: clj_atom as *const u8,
            arity: 1,
        },
        Prim {
            name: "__atom_deref",
            ptr: clj_deref as *const u8,
            arity: 1,
        },
        Prim {
            name: "__atom_reset",
            ptr: clj_reset as *const u8,
            arity: 2,
        },
        Prim {
            name: "atom",
            ptr: clj_atom as *const u8,
            arity: 1,
        },
        Prim {
            name: "deref",
            ptr: clj_deref as *const u8,
            arity: 1,
        },
        Prim {
            name: "reset!",
            ptr: clj_reset as *const u8,
            arity: 2,
        },
        Prim {
            name: "__arity_check",
            ptr: clj_arity_check as *const u8,
            arity: 2,
        },
        Prim {
            name: "__check_args_list",
            ptr: clj_check_args_list as *const u8,
            arity: 2,
        },
        Prim {
            name: "__no_matching_arity_panic",
            ptr: clj_no_matching_arity_panic as *const u8,
            arity: 1,
        },
        Prim {
            name: "__make_no_matching_arity_exception",
            ptr: clj_make_no_matching_arity_exception as *const u8,
            arity: 1,
        },
    ];
    PRIMS
}
