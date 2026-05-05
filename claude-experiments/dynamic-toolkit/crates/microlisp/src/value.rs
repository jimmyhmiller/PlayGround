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

use dynobj::roots::{Rooted, RootScope};
use dynobj::{Compact, ObjHeader};
use dynvalue::{NanBox, TagScheme};

use crate::host::with_host;

/// Phantom type tag for `Rooted<NanBoxTag>` — identifies the rooted slot
/// as carrying a microlisp NanBox value (cons pointer, symbol, number,
/// nil, or boolean). The tag plays no runtime role; it's documentation
/// for code that's holding GC-managed handles.
pub struct NanBoxTag;

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

/// Allocate a cons cell on the GC heap, in `scope`.
///
/// **The toolkit-enforced safe allocator.** Both arguments must already
/// be rooted in some scope (theirs or ours), and the result is rooted in
/// `scope`. This makes "hold a raw `u64` GC pointer across an allocation"
/// statically impossible: there's no signature you can call where the
/// car/cdr pointers aren't visible to the GC.
///
/// Implementation: we re-fetch via `.get()` AFTER `gc.alloc` (which may
/// itself fire a moving collection in the future — currently `Generational`
/// doesn't, but the contract permits it), so the cons fields receive the
/// correct (post-GC) addresses.
pub fn alloc_cons<'scope>(
    scope: &'scope RootScope<'_>,
    car: &Rooted<'_, NanBoxTag>,
    cdr: &Rooted<'_, NanBoxTag>,
) -> Rooted<'scope, NanBoxTag> {
    with_host(|h| {
        debug_assert!(!h.gc.is_null(), "alloc_cons: Host has no GC installed");
        let gc = unsafe { &*h.gc };
        let raw = gc.alloc(CONS_TYPE_ID, 0);
        assert!(!raw.is_null(), "microlisp: GC alloc returned null");
        // Re-fetch after alloc: even though current backends don't auto-GC
        // inside alloc, the API contract permits it. Reading via .get()
        // now picks up any in-place GC update of the rooted slots.
        let car_bits = car.get();
        let cdr_bits = cdr.get();
        unsafe {
            (raw.add(car_offset()) as *mut u64).write(car_bits);
            (raw.add(cdr_offset()) as *mut u64).write(cdr_bits);
        }
        scope.root::<NanBoxTag>(gc.tag_ptr(raw))
    })
}

/// Convenience: allocate a cons from raw `u64` bits. Roots them in
/// `scope` for the duration of the allocation, then returns a `Rooted`
/// for the result. Use this at FFI boundaries where the JIT hands us
/// raw bits via the C ABI; for Rust-internal callers prefer
/// [`alloc_cons`] which forces explicit rooting at every site.
pub fn alloc_cons_from_raw<'scope>(
    scope: &'scope RootScope<'_>,
    car_bits: u64,
    cdr_bits: u64,
) -> Rooted<'scope, NanBoxTag> {
    let car = scope.root::<NanBoxTag>(car_bits);
    let cdr = scope.root::<NanBoxTag>(cdr_bits);
    alloc_cons(scope, &car, &cdr)
}

/// Compile-time-only cons builder. Self-contained `with_scope` per call,
/// returns raw `u64`. Use **only** in code paths that don't trigger GC
/// — currently the IR-construction helpers in `compile.rs` and the
/// `quasiquote_rewrite` family in `expand.rs`. Each call pays the cost
/// of a one-slot frame push/pop, but the result is always a valid GC
/// pointer at the moment of return.
///
/// Why this exists: those helpers chain many small `cons` calls in
/// expressions like `cons(a, cons(b, cons(c, NIL)))`. Forcing each
/// intermediate to be `Rooted` would require restructuring every
/// expression. Since the helpers themselves are GC-free, we accept the
/// raw-`u64` return and wall the rooting discipline behind a clear
/// "compile-time-only" name.
pub fn cons_compile_time(car: u64, cdr: u64) -> u64 {
    // Each call uses a fresh 3-slot scope (car + cdr + result) — the
    // minimum that `alloc_cons_from_raw` needs internally.
    dynobj::roots::with_scope(3, |scope| {
        alloc_cons_from_raw(scope, car, cdr).get()
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

/// Build a list from a Rust slice, right-folded with `cons`. Returns a
/// `Rooted` because each iteration's `tail` must survive the next
/// iteration's allocation.
pub fn list_from_slice<'scope>(
    scope: &'scope RootScope<'_>,
    items: &[u64],
) -> Rooted<'scope, NanBoxTag> {
    let tail = scope.root::<NanBoxTag>(NIL);
    for &x in items.iter().rev() {
        let new = alloc_cons_from_raw(scope, x, tail.get());
        tail.set(new.get());
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
