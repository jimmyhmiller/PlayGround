//! C-callable Rust externs that the Clojure JIT calls into.
//!
//! Each extern takes/returns NanBox-encoded `u64`. Numeric externs
//! always return floats — `from_int` and `from_f64` both encode into
//! the unboxed-float NaN payload, so an int+int still rides the
//! float fast path.

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

/// `=` — Clojure equality. For numbers, value equality. For other
/// reference types in this checkpoint, identity. Will gain
/// collection-aware comparison in a later checkpoint.
#[unsafe(no_mangle)]
pub extern "C" fn clj_eq(a: u64, b: u64) -> u64 {
    if is_number(a) && is_number(b) {
        bool_(num(a) == num(b))
    } else {
        bool_(a == b)
    }
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
    ];
    PRIMS
}
