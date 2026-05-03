//! NanBox-encoded values for microlisp.
//!
//! Tag layout (NanBox: 4 tags × 48-bit payload):
//!   0 — immediates: nil (0), true (1), false (2)
//!   1 — symbol id (interned, payload is u32)
//!   2 — cons pointer (8-aligned heap object, GC-managed)
//!   3 — other heap (reserved)
//!
//! Cons cells live on the GC heap behind a [`dynobj::Compact`] header. Their
//! two value fields (`car`, `cdr`) are GC-traced. Allocation routes through
//! `DynGcRuntime::alloc` via the active microlisp `Host`, which holds a
//! pointer to the engine's `DynGcRuntime`.

use dynobj::{Compact, ObjHeader};
use dynvalue::{NanBox, TagScheme};

use crate::host::with_host;

pub const TAG_IMM: u32 = 0;
pub const TAG_SYM: u32 = 1;
pub const TAG_CONS: u32 = 2;
pub const TAG_OBJ: u32 = 3;

pub const IMM_NIL: u64 = 0;
pub const IMM_TRUE: u64 = 1;
pub const IMM_FALSE: u64 = 2;

pub const NIL: u64 = NanBox::NIL; // tag 0, payload 0
pub const TRUE: u64 = encode_imm(IMM_TRUE);
pub const FALSE: u64 = encode_imm(IMM_FALSE);

pub const fn encode_imm(payload: u64) -> u64 {
    // tag 0
    0x7FFC_0000_0000_0000 | (payload & 0x0000_FFFF_FFFF_FFFF)
}

pub fn encode_sym(id: u32) -> u64 {
    <NanBox as TagScheme>::encode_tagged(TAG_SYM, id as u64)
}

/// Tag a raw cons-cell heap pointer as a NanBox value. Used by tests that
/// build cons cells out-of-band; production code goes through `alloc_cons`
/// which calls into the GC.
pub fn encode_cons_ptr(ptr: *const u8) -> u64 {
    <NanBox as TagScheme>::encode_tagged(TAG_CONS, ptr as u64 & 0x0000_FFFF_FFFF_FFFF)
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

pub fn is_true_value(v: u64) -> bool {
    // CL truthiness: only nil and false are falsy. All else (including 0, "", '()) — well
    // we treat nil and false as the only falses.
    v != NIL && v != FALSE
}

pub fn is_cons(v: u64) -> bool {
    <NanBox as TagScheme>::has_tag(v, TAG_CONS)
}

pub fn is_symbol(v: u64) -> bool {
    <NanBox as TagScheme>::has_tag(v, TAG_SYM)
}

pub fn is_number(v: u64) -> bool {
    <NanBox as TagScheme>::is_float(v)
}

pub fn as_number(v: u64) -> f64 {
    f64::from_bits(v)
}

pub fn as_symbol_id(v: u64) -> u32 {
    debug_assert!(is_symbol(v));
    (<NanBox as TagScheme>::extract_payload(v)) as u32
}

/// The GC's `ObjTypeId` for cons cells. We register cons as type 0 (the
/// only obj type microlisp v0 uses).
pub const CONS_TYPE_ID: usize = 0;

/// Byte offset of `car` within a cons object (after the Compact header).
pub fn car_offset() -> usize { Compact::SIZE }
/// Byte offset of `cdr` within a cons object.
pub fn cdr_offset() -> usize { Compact::SIZE + 8 }

/// Decode the raw heap pointer from a cons NanBox.
pub fn as_cons_ptr(v: u64) -> *mut u8 {
    debug_assert!(is_cons(v));
    <NanBox as TagScheme>::extract_payload(v) as *mut u8
}

/// Allocate a cons cell on the GC heap.
///
/// The active `Host` (installed by `Engine::run_source`) carries a pointer
/// to the engine's `DynGcRuntime`. We allocate a `cons` typed object and
/// initialize its two value fields. The returned NanBox is tagged with the
/// configured ptr tag (`TAG_CONS`), so a moving GC can find and trace cons
/// cells from any root.
pub fn alloc_cons(car: u64, cdr: u64) -> u64 {
    with_host(|h| {
        debug_assert!(!h.gc.is_null(), "alloc_cons: Host has no GC installed");
        let gc = unsafe { &*h.gc };
        let raw = gc.alloc(CONS_TYPE_ID, 0);
        assert!(!raw.is_null(), "microlisp: GC alloc returned null");
        unsafe {
            (raw.add(car_offset()) as *mut u64).write(car);
            (raw.add(cdr_offset()) as *mut u64).write(cdr);
        }
        gc.tag_ptr(raw)
    })
}

pub fn car(v: u64) -> u64 {
    assert!(is_cons(v), "car of non-cons: 0x{:016x}", v);
    unsafe { (as_cons_ptr(v).add(car_offset()) as *const u64).read() }
}

pub fn cdr(v: u64) -> u64 {
    assert!(is_cons(v), "cdr of non-cons: 0x{:016x}", v);
    unsafe { (as_cons_ptr(v).add(cdr_offset()) as *const u64).read() }
}

pub fn set_car(v: u64, x: u64) {
    assert!(is_cons(v));
    unsafe { (as_cons_ptr(v).add(car_offset()) as *mut u64).write(x) }
}

pub fn set_cdr(v: u64, x: u64) {
    assert!(is_cons(v));
    unsafe { (as_cons_ptr(v).add(cdr_offset()) as *mut u64).write(x) }
}

/// Build a list from a Rust slice, right-folded with `cons`.
pub fn list_from_slice(items: &[u64]) -> u64 {
    let mut tail = NIL;
    for &x in items.iter().rev() {
        tail = alloc_cons(x, tail);
    }
    tail
}

/// Iterate the spine of a proper list. Stops at nil. Improper lists yield the
/// car of every cell, then the (non-nil) final cdr is dropped — callers that
/// need to handle dotted tails should iterate manually with `car`/`cdr`.
pub fn list_iter(mut v: u64) -> impl Iterator<Item = u64> {
    std::iter::from_fn(move || {
        if is_cons(v) {
            let car_v = car(v);
            v = cdr(v);
            Some(car_v)
        } else {
            None
        }
    })
}

pub fn list_len(v: u64) -> usize {
    let mut n = 0;
    let mut p = v;
    while is_cons(p) {
        n += 1;
        p = cdr(p);
    }
    n
}

pub fn equal(a: u64, b: u64) -> bool {
    if a == b {
        return true;
    }
    if is_cons(a) && is_cons(b) {
        return equal(car(a), car(b)) && equal(cdr(a), cdr(b));
    }
    false
}
