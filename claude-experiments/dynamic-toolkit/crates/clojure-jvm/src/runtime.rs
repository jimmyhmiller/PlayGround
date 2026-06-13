//! Runtime primitives the JIT calls back into.
//!
//! JIT-compiled Clojure code talks to the host through `extern "C"` functions
//! declared in [`lang::compiler::RuntimeExterns`] and bound into each
//! `JitModule` at compile time.
//!
//! Conventions:
//!   * All NanBox values cross the boundary as `u64`.
//!   * Var pointers cross as `u64` (the raw `*const Var`); the corresponding
//!     `Arc<Var>` lives in [`crate::lang::namespace::Namespace::mappings`]
//!     for the program's lifetime, so the pointer never dangles.

use std::cmp::Ordering;
use std::sync::Arc;

use crate::lang::object::Object;
use crate::lang::var::Var;
use dynobj::roots::{Heap, Raw, RootScope, Rooted};

// ── NanBox layout (matches dynlang's default NanBoxTags) ────────────────

const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
const TAG_MASK: u64 = 0x0003_0000_0000_0000;
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

// Tags (matches `dynlang::NanBoxTags::default()` + our own fn-handle tag).
pub const TAG_NIL: u32 = 0;
pub const TAG_BOOL: u32 = 1;
const TAG_PTR: u32 = 2;
const TAG_FN: u32 = 3;

fn nanbox_encode(tag: u32, payload: u64) -> u64 {
    TAG_PATTERN | ((tag as u64) << 48) | (payload & PAYLOAD_MASK)
}

pub fn nanbox_tag(bits: u64) -> Option<u32> {
    if (bits & FULL_MASK) != TAG_PATTERN {
        return None;
    }
    Some(((bits & TAG_MASK) >> 48) as u32)
}

pub fn nanbox_payload(bits: u64) -> u64 {
    bits & PAYLOAD_MASK
}

/// Public NanBox constructors callable from heap-population code that
/// stores Object fields (`alloc_object_as_nanbox` in compiler.rs).
pub fn nanbox_nil() -> u64 {
    nanbox_encode(TAG_NIL, 0)
}
pub fn nanbox_bool(b: bool) -> u64 {
    nanbox_encode(TAG_BOOL, b as u64)
}
/// NanBox-encode a heap pointer (returned by `gc.alloc`/`gc_alloc_thunk`).
/// The pointer's low 48 bits become the payload; tag bits go to TAG_PTR.
pub fn nanbox_ptr(raw: u64) -> u64 {
    nanbox_encode(TAG_PTR, raw & PAYLOAD_MASK)
}

/// Allocate a boxed `clojure.lang.Long` holding `n`, returning its NanBox.
/// Clojure integers are real longs (distinct from doubles); NaN-boxing has
/// no inline integer tag, so we box them. The i64 lives at offset 8 (after
/// the 8-byte Compact header), like other Raw64-cell types.
///
/// # Safety
/// Must run on a registered mutator thread (the GC alloc invariant).
pub unsafe fn box_long(n: i64) -> u64 {
    let ids = heap_type_ids();
    let raw = dynlang::gc::gc_alloc_thunk(ids.long as u64, 0);
    let p = raw as *mut u8;
    if p.is_null() {
        panic!("clojure-jvm: box_long: gc_alloc returned null");
    }
    unsafe { p.add(8).cast::<i64>().write_unaligned(n) };
    nanbox_ptr(raw)
}

/// Read the i64 out of a boxed `Long`. Precondition: `bits` is a TAG_PTR to
/// a Long cell (type_id == `heap_type_ids().long`).
///
/// # Safety
/// `bits` must point to a Long cell allocated by [`box_long`].
pub unsafe fn unbox_long(bits: u64) -> i64 {
    let p = nanbox_payload(bits) as *const u8;
    unsafe { p.add(8).cast::<i64>().read_unaligned() }
}

/// Is `bits` a boxed Long?
pub fn is_boxed_long(bits: u64) -> bool {
    if nanbox_tag(bits) != Some(TAG_PTR) {
        return false;
    }
    let p = nanbox_payload(bits) as *const u8;
    if p.is_null() {
        return false;
    }
    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
    tid == heap_type_ids().long
}

/// Allocate a boxed `clojure.lang.Character` holding codepoint `c`. Layout
/// mirrors [`box_long`] (Raw64 i64 at offset 8) so the codepoint flows
/// through JIT code the same way an integer does.
///
/// # Safety
/// Must run on a registered mutator thread (the GC alloc invariant).
pub unsafe fn box_char(c: u32) -> u64 {
    let ids = heap_type_ids();
    let raw = dynlang::gc::gc_alloc_thunk(ids.character as u64, 0);
    let p = raw as *mut u8;
    if p.is_null() {
        panic!("clojure-jvm: box_char: gc_alloc returned null");
    }
    unsafe { p.add(8).cast::<i64>().write_unaligned(c as i64) };
    nanbox_ptr(raw)
}

/// Read the codepoint out of a boxed `Character`. Precondition: `bits` is a
/// TAG_PTR to a Character cell.
///
/// # Safety
/// `bits` must point to a Character cell allocated by [`box_char`].
pub unsafe fn unbox_char(bits: u64) -> u32 {
    let p = nanbox_payload(bits) as *const u8;
    unsafe { p.add(8).cast::<i64>().read_unaligned() as u32 }
}

/// Is `bits` a boxed Character?
pub fn is_boxed_char(bits: u64) -> bool {
    if nanbox_tag(bits) != Some(TAG_PTR) {
        return false;
    }
    let p = nanbox_payload(bits) as *const u8;
    if p.is_null() {
        return false;
    }
    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
    tid == heap_type_ids().character
}

/// Decode a numeric argument to an `i64`, accepting either a boxed Long or
/// a native NanBox double. This is the polymorphic replacement for the bare
/// `f64::from_bits(bits) as i64` pattern at sites that read an integer
/// argument (indices, counts, method/type ids): once integer literals are
/// boxed Longs, those args arrive as TAG_PTR cells rather than floats, but
/// internally-baked ids and pre-flip integers still arrive as floats.
pub fn arg_to_i64(bits: u64) -> i64 {
    if is_boxed_long(bits) {
        unsafe { unbox_long(bits) }
    } else if is_boxed_char(bits) {
        // `(int \a)` / `(long \a)` decode a Character to its codepoint.
        unsafe { unbox_char(bits) as i64 }
    } else {
        f64::from_bits(bits) as i64
    }
}

/// Decode a numeric argument to an `f64`, accepting either a boxed Long or
/// a native NanBox double. The float-side companion to [`arg_to_i64`], for
/// sites that read a number into floating point (casts, `Math/*`, etc.).
pub fn arg_to_f64(bits: u64) -> f64 {
    if is_boxed_long(bits) {
        unsafe { unbox_long(bits) as f64 }
    } else {
        f64::from_bits(bits)
    }
}

/// Encode a double result as a NanBox. A raw f64 IS its own NanBox value,
/// except when its bit pattern collides with the tag pattern (a signalling
/// region) — then canonicalize to a quiet NaN.
fn nanbox_double(d: f64) -> u64 {
    let bits = d.to_bits();
    if (bits & FULL_MASK) == TAG_PATTERN {
        0x7FF8_0000_0000_0000
    } else {
        bits
    }
}

// ── Numeric-tower arithmetic externs ───────────────────────────────
//
// clojure-jvm integers are boxed Longs and doubles are native NanBox
// floats. These externs decode each operand to a `Num`, apply the tower
// (see `value_repr`), and re-encode (box a Long, NanBox a double). The JIT
// calls these from `PrimOpExpr` instead of dynlang's float-only `add`/etc.

use crate::lang::value_repr::{self, Num, NumResult};

/// Decode an operand into a `Num` (boxed Long → Long, else native double).
fn decode_num(bits: u64) -> Num {
    if is_boxed_long(bits) {
        Num::Long(unsafe { unbox_long(bits) })
    } else {
        Num::Double(f64::from_bits(bits))
    }
}

/// Encode a `Num` result. `# Safety`: boxing requires a mutator thread.
unsafe fn encode_num(n: Num) -> u64 {
    match n {
        Num::Long(v) => unsafe { box_long(v) },
        Num::Double(d) => nanbox_double(d),
    }
}

/// On long-overflow Clojure's `+`/`-`/`*` THROW; until exceptions are wired
/// through here we promote to double (documented deviation — the conformance
/// corpus never overflows). `NumResult::Num` encodes normally.
unsafe fn encode_numresult(r: NumResult, a: Num, b: Num, as_double: fn(f64, f64) -> f64) -> u64 {
    match r {
        NumResult::Num(n) => unsafe { encode_num(n) },
        NumResult::Overflow => nanbox_double(as_double(a.as_f64(), b.as_f64())),
    }
}

macro_rules! num_binop_extern {
    ($name:ident, $tower:path, $dbl:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(a: u64, b: u64) -> u64 {
            let (na, nb) = (decode_num(a), decode_num(b));
            unsafe { encode_numresult($tower(na, nb), na, nb, $dbl) }
        }
    };
}
num_binop_extern!(cljvm_num_add, value_repr::num_add, |x, y| x + y);
num_binop_extern!(cljvm_num_sub, value_repr::num_sub, |x, y| x - y);
num_binop_extern!(cljvm_num_mul, value_repr::num_mul, |x, y| x * y);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_num_div(a: u64, b: u64) -> u64 {
    unsafe { encode_num(value_repr::num_div(decode_num(a), decode_num(b))) }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_num_quot(a: u64, b: u64) -> u64 {
    unsafe { encode_num(value_repr::num_quot(decode_num(a), decode_num(b))) }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_num_rem(a: u64, b: u64) -> u64 {
    unsafe { encode_num(value_repr::num_rem(decode_num(a), decode_num(b))) }
}

/// Is `bits` a number (boxed Long or native double)?
fn is_number_val(bits: u64) -> bool {
    is_boxed_long(bits) || nanbox_tag(bits).is_none()
}

/// `=` (general equality). Numbers compare type-aware (`(= 1 1)` true,
/// `(= 1 1.0)` false); everything else delegates to `equiv_impl`, which does
/// value equality for the heap types it knows (strings, symbols, keywords,
/// cons lists, maps, sets, namespaces, vars). nil/bool/interned refs already
/// short-circuit on the bit-equality check.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_equals(a: u64, b: u64) -> u64 {
    if a == b {
        return nanbox_bool(true);
    }
    if is_number_val(a) && is_number_val(b) {
        return nanbox_bool(value_repr::val_num_eq(decode_num(a), decode_num(b)));
    }
    nanbox_bool(unsafe { equiv_impl(a, b) })
}

macro_rules! num_cmp_extern {
    ($name:ident, $tower:path) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(a: u64, b: u64) -> u64 {
            nanbox_bool($tower(decode_num(a), decode_num(b)))
        }
    };
}
num_cmp_extern!(cljvm_num_lt, value_repr::num_lt);
num_cmp_extern!(cljvm_num_gt, value_repr::num_gt);
num_cmp_extern!(cljvm_num_le, value_repr::num_le);
num_cmp_extern!(cljvm_num_ge, value_repr::num_ge);
/// `==` — numeric equality across long/double.
num_cmp_extern!(cljvm_num_equiv, value_repr::num_eq);

/// Marker wrapper for a heap-tagged NanBox stored opaquely in `Object::Host`.
/// Lets Var roundtrip heap pointers (string literals, future quoted-symbol /
/// keyword / list values) without prematurely decoding them into Rust-side
/// `Object` variants every bind/deref. The actual heap bytes only get
/// reconstructed into `Object::String` etc. when something asks for the
/// Object form (e.g. printing, equality, host-side inspection).
#[derive(Debug)]
pub struct HeapBits(pub u64);

/// Decode a NanBox u64 into an `Object`. The inverse of `object_to_nanbox`.
pub fn nanbox_to_object(bits: u64) -> Object {
    match nanbox_tag(bits) {
        None => {
            // Untagged → IEEE 754 float (or natural NaN).
            Object::Double(f64::from_bits(bits))
        }
        Some(TAG_NIL) => Object::Nil,
        Some(TAG_BOOL) => Object::Bool(nanbox_payload(bits) != 0),
        // Heap-tagged values roundtrip opaquely via `Object::Host(HeapBits)`.
        // Decoding them into Object::String / Object::Symbol / etc. requires
        // reading the heap object's header to dispatch by type_id, which we
        // do lazily in `heap_bits_to_object` — invoked by host code that
        // needs the actual contents (`print-str`, equality, reader bridges).
        Some(TAG_PTR) => Object::host(HeapBits(bits)),
        Some(TAG_FN) => Object::host(HeapBits(bits)),
        Some(_) => Object::Unported {
            java_class: "unknown NanBox tag",
        },
    }
}

/// Encode an `Object` into a NanBox u64. The inverse of `nanbox_to_object`.
pub fn object_to_nanbox(obj: &Object) -> u64 {
    match obj {
        Object::Nil => nanbox_encode(TAG_NIL, 0),
        Object::Bool(b) => nanbox_encode(TAG_BOOL, *b as u64),
        // Integers are boxed Longs. `# Safety`: boxing needs an installed
        // mutator thread; every `object_to_nanbox` caller (Var deref, heap
        // stores) runs on one. (Early init that has no thread goes through
        // `try_object_to_nanbox`, which returns None for Long instead.)
        Object::Long(n) => unsafe { box_long(*n) },
        Object::Char(c) => unsafe { box_char(*c) },
        Object::Double(x) => x.to_bits(),
        Object::Host(_) => {
            if let Some(hb) = obj.host_as::<HeapBits>() {
                return hb.0;
            }
            panic!(
                "clojure-jvm: object_to_nanbox: Object::Host wrapping non-HeapBits value not yet representable as NanBox"
            )
        }
        // `clojure.lang.Namespace` / `clojure.lang.Var` — heap cells holding a
        // leaked `Arc::as_ptr` in their Raw64 slot. Boxing requires an
        // installed mutator thread (every `object_to_nanbox` caller runs on
        // one); the Arc is kept alive on the Session roots.
        Object::Namespace(ns) => unsafe {
            alloc_arc_cell(
                heap_type_ids().namespace,
                ns.clone(),
                crate::lang::compiler::with_active_session_root_namespace,
            )
        },
        Object::Var(v) => unsafe {
            alloc_arc_cell(
                heap_type_ids().var,
                v.clone(),
                crate::lang::compiler::with_active_session_root_var,
            )
        },
        _ => {
            panic!("clojure-jvm: object_to_nanbox: variant {obj:?} not yet representable as NanBox")
        }
    }
}

/// Like [`object_to_nanbox`] but returns `None` for values that have no
/// NanBox representation (e.g. `Object::Namespace`, or `Object::Host`
/// wrapping arbitrary Rust state) instead of panicking. Used by Var root
/// storage to decide whether a value can live in the GC-rooted slot table
/// (NanBox-representable, including all heap pointers) or must be kept
/// Rust-side.
pub fn try_object_to_nanbox(obj: &Object) -> Option<u64> {
    match obj {
        Object::Nil => Some(nanbox_encode(TAG_NIL, 0)),
        Object::Bool(b) => Some(nanbox_encode(TAG_BOOL, *b as u64)),
        // A boxed Long can't be built here: this is the non-allocating
        // representability check (runs during `Var::bind_root`, before the
        // heap/thread exist). Keep Long roots Rust-side as `VarRoot::Object`;
        // they get boxed lazily on deref via `object_to_nanbox`.
        Object::Long(_) => None,
        // Like Long: a Character is a heap cell, so it can't be built in this
        // non-allocating check. Box lazily on deref via `object_to_nanbox`.
        Object::Char(_) => None,
        Object::Double(x) => Some(x.to_bits()),
        Object::Host(_) => obj.host_as::<HeapBits>().map(|hb| hb.0),
        _ => None,
    }
}

/// Type-id dispatch table for `heap_bits_to_object`. Mirrors the `ObjTypeId`s
/// declared in `Compiler::new`. Passing a struct rather than positional args
/// avoids ordering bugs as more heap types come online.
#[derive(Debug, Clone, Copy)]
pub struct HeapTypeIds {
    pub string: usize,
    pub symbol: usize,
    pub keyword: usize,
    pub cons: usize,
    pub vector: usize,
    pub map: usize,
    pub set: usize,
    pub tree_map: usize,
    pub tree_set: usize,
    pub string_builder: usize,
    pub chunk_buffer: usize,
    pub i_chunk: usize,
    pub lazy_seq: usize,
    pub delay: usize,
    pub multi_arity_fn: usize,
    pub class: usize,
    pub var: usize,
    /// `clojure.lang.Namespace` — a heap cell holding a leaked `Arc<Namespace>`
    /// pointer (Raw64 at offset 8), exactly like `var`. Lets namespace
    /// objects flow through JIT-compiled code: `*ns*`, `the-ns`, `ns-map`,
    /// and `(. *ns* (refer …))` all pass `Namespace` values.
    pub namespace: usize,
    pub with_meta: usize,
    /// `clojure.lang.Long` — boxed 64-bit integer. Clojure has a real
    /// integer type (`long`) distinct from `double`; we box it on the heap
    /// (a Raw64 i64 cell) so `(+ 1 2)` is `3` not `3.0`. (NaN-boxing left no
    /// inline integer tag; a future low-bit-tagging pass could inline it.)
    pub long: usize,
    /// `clojure.lang.Character` — a boxed Unicode codepoint (Raw64 i64
    /// cell, like `long`). Distinct type so `str`/`pr-str` render it as the
    /// character and `(= \a 97)` is false; `(int \a)` still unboxes to 97.
    pub character: usize,
    /// Shared ObjTypeId backing every `deftype`/`defrecord` instance.
    /// The actual user-type discriminator lives in the cell's Raw64
    /// `user_type_id` field and is read by `effective_type_id`.
    pub user_instance: usize,
    /// `clojure.lang.Reduced` — the wrapper produced by `(reduced x)` to
    /// signal early termination of `reduce`. One traced `Value` slot holds
    /// the wrapped value; `(deref r)` / `@r` unwraps it and
    /// `clojure.lang.RT/isReduced` tests for it.
    pub reduced: usize,
    /// `clojure.lang.Repeat` — a (possibly infinite) seq of one repeated
    /// value, backing `(repeat x)` / `(repeat n x)`. Layout (dynlang puts
    /// traced Value fields before Raw64 fields): traced `value` slot at
    /// offset 8, Raw64 `count` at offset 16 (i64; -1 = infinite, else
    /// remaining elements, always > 0 — exhaustion yields nil, zero-count
    /// creation yields nil). `seq` returns the cell itself; `first` reads
    /// `value`; `next` returns the cell for infinite, a fresh cell with
    /// `count-1` for bounded > 1, nil at 1.
    pub repeat_seq: usize,
    /// `clojure.lang.Atom` — mutable reference cell. Raw64 holds
    /// `Arc<RefCell<RefState>>`; the held value is GC-traced via the REF
    /// registry. Backs `atom`/`swap!`/`reset!`/`compare-and-set!`/`deref`.
    pub atom: usize,
    /// `clojure.lang.Volatile` — same cell shape as Atom; backs
    /// `volatile!`/`vreset!`/`vswap!`/`deref`.
    pub volatile_cell: usize,
    /// `clojure.lang.Closure` — JIT closure cells (fref + captures).
    /// Exposed here so host predicates (`fn?`, `ifn?`) can recognize
    /// first-class fns that aren't bare `TAG_FN` handles.
    pub closure: usize,
    /// `clojure.lang.Iterate` — infinite seq of x, (f x), (f (f x)) ….
    /// Two traced slots: `f`@8, `value`@16. `first` reads `value`;
    /// `next` allocates a fresh cell with `(f value)`.
    pub iterate_seq: usize,
    /// `clojure.lang.Cycle` — infinite repetition of a finite seq.
    /// Two traced slots: `all`@8 (the head of the cycle), `current`@16
    /// (the walk position). `first` is `(first current)`; `next` advances
    /// `current`, wrapping back to `all` at the end.
    pub cycle_seq: usize,
    /// `java.util.regex.Matcher` — Raw64 → Arc<RefCell<MatcherState>>.
    pub matcher: usize,
    /// `clojure.lang.MultiFn` — Raw64 → Arc<RefCell<MultiFnState>>.
    pub multifn_cell: usize,
    /// `clojure.lang.ExceptionInfo` — the `ex-info`/`ex-data` exception
    /// type. Three traced Value slots: message @8, data @16, cause @24
    /// (offset constants in `lang::exception_info`).
    pub exception_info: usize,
}

/// Process-global type-id registry. `Compiler::new` calls
/// `set_heap_type_ids` so externs (`cljvm_rt_cons`, etc.) called from
/// JIT-executing code can look up the right ObjTypeIds without threading
/// them through every call site. Stable assumption: every compilation
/// session declares the same set of types in the same order, so the ids
/// are reused across compilations.
static HEAP_TYPE_IDS: std::sync::OnceLock<HeapTypeIds> = std::sync::OnceLock::new();

pub fn set_heap_type_ids(ids: HeapTypeIds) {
    // Safe to ignore the result: subsequent calls with the same ids are a
    // no-op. If a future change introduces compilations with different
    // type_id assignments we'll need to revisit this; an assert would
    // surface the violation immediately.
    let _ = HEAP_TYPE_IDS.set(ids);
}

pub fn heap_type_ids() -> HeapTypeIds {
    *HEAP_TYPE_IDS
        .get()
        .expect("clojure-jvm: heap_type_ids() called before Compiler::new ran")
}

/// Byte-offset layout of `clojure.lang.UserInstance` cells. Captured at
/// `Compiler::new` time (same as `HeapTypeIds`) so runtime externs can
/// allocate and read user-deftype instances without threading the
/// `Compiler` through.
#[derive(Debug, Clone, Copy)]
pub struct UserInstanceLayout {
    /// Offset (from the heap cell base) of the Raw64 `user_type_id` slot.
    pub user_type_id_offset: i32,
    /// Offset of varlen value slot 0; slot `i` lives at
    /// `varlen_base + 8 * i`.
    pub varlen_base: i64,
}

static USER_INSTANCE_LAYOUT: std::sync::OnceLock<UserInstanceLayout> = std::sync::OnceLock::new();

pub fn set_user_instance_layout(layout: UserInstanceLayout) {
    let _ = USER_INSTANCE_LAYOUT.set(layout);
}

pub fn user_instance_layout() -> UserInstanceLayout {
    *USER_INSTANCE_LAYOUT
        .get()
        .expect("clojure-jvm: user_instance_layout() called before Compiler::new ran")
}

/// Decode an arbitrary NanBox into an `Object`. Handles immediates
/// (nil / bool / long / double) directly and dispatches to
/// `heap_bits_to_object` for TAG_PTR pointers.
pub fn any_bits_to_object(bits: u64, ids: HeapTypeIds) -> Object {
    match nanbox_tag(bits) {
        None => Object::Double(f64::from_bits(bits)),
        Some(TAG_NIL) => Object::Nil,
        Some(TAG_BOOL) => Object::Bool(nanbox_payload(bits) != 0),
        Some(TAG_PTR) => unsafe { heap_bits_to_object(bits, ids) },
        _ => Object::Unported {
            java_class: "unknown NanBox tag",
        },
    }
}

/// Decode a heap-tagged NanBox into a fully-materialized `Object` by reading
/// the heap object's header to dispatch by type_id. Used by tests + host
/// code that wants the Rust-side representation. Assumes the bits are
/// TAG_PTR — caller must check.
///
/// # Safety
/// `bits` must encode a live pointer to a `clojure.lang.*` heap object
/// allocated under the active DynGcRuntime. Reading the header + varlen
/// section is sound under the moving-GC-safepoint contract as long as
/// the caller is on a registered mutator thread.
pub unsafe fn heap_bits_to_object(bits: u64, ids: HeapTypeIds) -> Object {
    debug_assert_eq!(
        nanbox_tag(bits),
        Some(TAG_PTR),
        "heap_bits_to_object: not a TAG_PTR"
    );
    let payload = nanbox_payload(bits);
    let ptr = payload as *const u8;
    if ptr.is_null() {
        panic!("clojure-jvm: heap_bits_to_object: null payload");
    }
    // Compact header: type_id at offset 0, u16 little-endian.
    let type_id = unsafe { ptr.cast::<u16>().read_unaligned() } as usize;
    if type_id == ids.long {
        let n = unsafe { ptr.add(8).cast::<i64>().read_unaligned() };
        return Object::Long(n);
    }
    if type_id == ids.character {
        let c = unsafe { ptr.add(8).cast::<i64>().read_unaligned() };
        return Object::Char(c as u32);
    }
    if type_id == ids.symbol {
        // Sanity-check the arc_ptr stored in the cell. Real Symbol cells
        // hold a pointer to a leaked Arc<Symbol> — always > 0x10000.
        // A tiny value means the cell is misidentified (the byte at offset 0
        // happened to be 0x01 but it's not actually a Symbol cell). Panicking
        // loudly here is much friendlier than the SIGBUS that
        // `Arc::increment_strong_count` would produce.
        let arc_check = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
        if arc_check < 0x1_0000 {
            panic!(
                "clojure-jvm: heap_decode: 'Symbol' cell @ 0x{:x} has \
                 implausible arc_ptr=0x{arc_check:x} — cell is misidentified, \
                 not actually a Symbol. Pointer probably came from a stale \
                 NanBox (e.g., a Cons.rest field that wasn't updated when GC \
                 moved the original target).",
                ptr as u64
            );
        }
    }
    if type_id == ids.string {
        // `clojure.lang.String` (varlen_bytes): Header(8) + count word(8) +
        // N bytes. Count word lives at offset 8; bytes start at offset 16.
        let count = unsafe { ptr.add(8).cast::<u64>().read_unaligned() } as usize;
        let bytes = unsafe { std::slice::from_raw_parts(ptr.add(16), count) };
        let s = std::str::from_utf8(bytes)
            .expect("clojure.lang.String varlen bytes must be valid UTF-8")
            .to_string();
        return Object::String(std::sync::Arc::new(s));
    }
    if type_id == ids.symbol {
        let arc_ptr = unsafe { ptr.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::symbol::Symbol;
        unsafe { std::sync::Arc::increment_strong_count(arc_ptr) };
        let arc = unsafe { std::sync::Arc::from_raw(arc_ptr) };
        return Object::Symbol(arc);
    }
    if type_id == ids.keyword {
        // `clojure.lang.Keyword`: same layout as Symbol.
        let arc_ptr = unsafe { ptr.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::keyword::Keyword;
        unsafe { std::sync::Arc::increment_strong_count(arc_ptr) };
        let arc = unsafe { std::sync::Arc::from_raw(arc_ptr) };
        return Object::Keyword(arc);
    }
    if type_id == ids.var {
        // `clojure.lang.Var`: Raw64 holding `*const Var` (leaked Arc).
        let arc = unsafe { decode_arc_cell::<crate::lang::var::Var>(ptr) };
        return Object::Var(arc);
    }
    if type_id == ids.namespace {
        // `clojure.lang.Namespace`: Raw64 holding `*const Namespace`.
        let arc = unsafe { decode_arc_cell::<crate::lang::namespace::Namespace>(ptr) };
        return Object::Namespace(arc);
    }
    if type_id == ids.cons {
        // `clojure.lang.Cons`: Header(8) + value-field "first"(8) +
        // value-field "rest"(8) + value-field "meta"(8). Recursively
        // decode `first` (any Object) and `rest` (Object::Nil terminator
        // or another Cons). Reconstruct a `PersistentList` so the caller
        // gets a familiar Rust shape; if the meta slot holds a non-nil
        // map, wrap the result in `Object::WithMeta`.
        // Decoding `first`/`rest` can FORCE lazy seqs (run JIT thunks →
        // GC), now that cons tails carry unforced LazySeqs. Stage the
        // three field bits in a registered chunk buffer so the collector
        // forwards them, and re-read each one fresh before use.
        let buf = std::sync::Arc::new(std::cell::RefCell::new(vec![
            unsafe { ptr.add(8).cast::<u64>().read_unaligned() },
            unsafe { ptr.add(16).cast::<u64>().read_unaligned() },
            unsafe { ptr.add(24).cast::<u64>().read_unaligned() },
        ]));
        register_chunk_buffer(&buf);
        let first_now = buf.borrow()[0];
        let first_obj = decode_value_bits(first_now, ids);
        let rest_now = buf.borrow()[1];
        let rest_list = decode_rest_to_list(rest_now, ids);
        // The decoded rest knows its length — never re-walk raw bits.
        let count = 1 + rest_list.count();
        let meta_now = buf.borrow()[2];
        deregister_chunk_buffer(&buf);
        let list_obj = Object::List(std::sync::Arc::new(
            crate::lang::persistent_list::PersistentList::Cons {
                first: first_obj,
                rest: rest_list,
                count,
            },
        ));
        return wrap_with_meta_bits(list_obj, meta_now, ids);
    }
    if type_id == ids.vector {
        // `clojure.lang.PersistentVector` (our flat varlen-values shape):
        // Header(8) + varlen-count(8) + N * 8 byte slots. The count word
        // lives at offset 8; items start at offset 16.
        let count = unsafe { ptr.add(8).cast::<u64>().read_unaligned() } as usize;
        // Element decodes can force lazy seqs (GC) — stage all item bits
        // in a registered chunk buffer and re-read each fresh.
        let buf = std::sync::Arc::new(std::cell::RefCell::new(
            (0..count)
                .map(|i| unsafe { ptr.add(16 + i * 8).cast::<u64>().read_unaligned() })
                .collect::<Vec<u64>>(),
        ));
        register_chunk_buffer(&buf);
        let mut items: Vec<Object> = Vec::with_capacity(count);
        for i in 0..count {
            let bits = buf.borrow()[i];
            items.push(decode_value_bits(bits, ids));
        }
        deregister_chunk_buffer(&buf);
        return Object::Vector(crate::lang::persistent_vector::PersistentVector::create(
            items,
        ));
    }
    if type_id == ids.map {
        // `clojure.lang.PersistentHashMap`: Header(8) + Raw64 "arc_ptr"(8)
        // holding the host-side `Arc<PersistentHashMap>` pointer. Reconstruct
        // by `Arc::increment_strong_count` on the stored pointer — the Arc
        // is kept alive by `CompileRoots._maps`, so the pointer is valid.
        let arc_ptr = unsafe { ptr.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::persistent_hash_map::PersistentHashMap;
        unsafe { std::sync::Arc::increment_strong_count(arc_ptr) };
        let arc = unsafe { std::sync::Arc::from_raw(arc_ptr) };
        return Object::Map(arc);
    }
    if type_id == ids.set {
        // `clojure.lang.PersistentHashSet`: same layout as Map.
        let arc_ptr = unsafe { ptr.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::persistent_hash_set::PersistentHashSet;
        unsafe { std::sync::Arc::increment_strong_count(arc_ptr) };
        let arc = unsafe { std::sync::Arc::from_raw(arc_ptr) };
        return Object::Set(arc);
    }
    if type_id == ids.tree_map {
        let arc = unsafe { decode_tree_map(ptr) };
        return Object::TreeMap(arc);
    }
    if type_id == ids.tree_set {
        let arc = unsafe { decode_tree_set(ptr) };
        return Object::TreeSet(arc);
    }
    if type_id == ids.with_meta {
        // Generic IObj wrapper. inner @ 8, meta @ 16 (both NanBox).
        let inner_bits = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
        let meta_bits = unsafe { ptr.add(16).cast::<u64>().read_unaligned() };
        let inner_obj = unsafe { decode_value_bits(inner_bits, ids) };
        return wrap_with_meta_bits(inner_obj, meta_bits, ids);
    }
    if type_id == ids.lazy_seq {
        // A lazy seq appearing in host-bound data — e.g. macroexpansion
        // output where a macro splices `map`/`filter`/`remove` results via
        // `~@` (the `ns` macro does this). Force it to a concrete seq and
        // decode that. `cljvm_rt_seq` fully realizes to a cons chain or nil.
        let forced = unsafe { cljvm_rt_seq(bits) };
        return decode_value_bits(forced, ids);
    }
    if type_id == ids.iterate_seq || type_id == ids.cycle_seq {
        panic!(
            "clojure-jvm: infinite seq (iterate/cycle) reached host decode — \
             realize a bounded prefix (e.g. take) first"
        );
    }
    if type_id == ids.repeat_seq {
        // Bounded `(repeat n x)` reaching host-bound data: realize it as a
        // PersistentList of n copies. An infinite repeat cannot be decoded
        // — realizing it would never terminate, so fail loudly.
        let count = unsafe { ptr.add(16).cast::<u64>().read_unaligned() } as i64;
        if count < 0 {
            panic!(
                "clojure-jvm: infinite (repeat x) reached host decode — \
                 realize a bounded prefix (e.g. take) before splicing it \
                 into host-bound data"
            );
        }
        let value_bits = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
        let v = decode_value_bits(value_bits, ids);
        let items = vec![v; count as usize];
        return Object::List(crate::lang::persistent_list::PersistentList::create(items));
    }
    Object::Unported {
        java_class: "heap object of unrecognized type_id",
    }
}

/// If `meta_bits` is a non-nil heap pointer to a `PersistentHashMap`,
/// wrap `inner` in `Object::WithMeta(inner, map)`. Otherwise return
/// `inner` unchanged. Used by `heap_bits_to_object` for IObj-carrying
/// types so reader-attached metadata round-trips back to the host.
fn wrap_with_meta_bits(inner: Object, meta_bits: u64, ids: HeapTypeIds) -> Object {
    if matches!(nanbox_tag(meta_bits), Some(TAG_NIL)) {
        return inner;
    }
    let meta_obj = match nanbox_tag(meta_bits) {
        Some(TAG_PTR) => unsafe { heap_bits_to_object(meta_bits, ids) },
        _ => return inner,
    };
    if let Object::Map(m) = meta_obj {
        Object::with_meta_map(inner, m)
    } else {
        inner
    }
}

// ─── Arc-backed cell helpers ──────────────────────────────────────────
//
// Many host classes have the same shape: a fixed Raw64 slot that holds
// `Arc::as_ptr(&value)`. The Arc itself is rooted on the active Session
// (host-side, not GC heap) so the raw pointer outlives the heap cell.
// `alloc_arc_cell` and `decode_arc_cell` capture this pattern so
// individual class definitions don't have to re-implement it (and so
// they can't forget to root, which has historically been a SIGBUS magnet).

/// Allocate a fresh Raw64-arc-backed heap cell. `value` is moved into
/// `root_push` (callers thread it through `with_active_session_root_*`)
/// to extend the Arc's lifetime to the JIT module's; the raw pointer
/// goes into the cell's slot at offset 8.
///
/// # Safety
/// Caller must be on a registered mutator thread (the GC's allocation
/// invariant). `T` must be the type that the corresponding decoder
/// reads back via `decode_arc_cell`.
pub unsafe fn alloc_arc_cell<T: 'static>(
    type_id: usize,
    value: std::sync::Arc<T>,
    root_push: impl FnOnce(std::sync::Arc<T>),
) -> u64 {
    let raw_arc = std::sync::Arc::as_ptr(&value) as u64;
    root_push(value);
    let raw = dynlang::gc::gc_alloc_thunk(type_id as u64, 0);
    let p = raw as *mut u8;
    if p.is_null() {
        panic!("clojure-jvm: alloc_arc_cell: gc_alloc returned null for type_id {type_id}");
    }
    unsafe {
        p.add(8).cast::<u64>().write_unaligned(raw_arc);
    }
    nanbox_ptr(raw)
}

/// Reconstruct an `Arc<T>` from a Raw64-arc-backed cell allocated via
/// [`alloc_arc_cell`]. The original Arc is still live on the Session's
/// roots; this clones the strong count so the caller owns its own ref.
///
/// # Safety
/// `ptr` must point to a heap cell whose slot at offset 8 holds an
/// `Arc<T>::as_ptr` raw pointer (i.e., the cell was allocated via
/// `alloc_arc_cell::<T>`).
pub unsafe fn decode_arc_cell<T>(ptr: *const u8) -> std::sync::Arc<T> {
    let arc_ptr = unsafe { ptr.add(8).cast::<u64>().read_unaligned() } as *const T;
    unsafe { std::sync::Arc::increment_strong_count(arc_ptr) };
    unsafe { std::sync::Arc::from_raw(arc_ptr) }
}

/// Decode a NanBox-encoded value-field into an `Object`. Heap-tagged
/// pointers recurse through `heap_bits_to_object`; immediates go through
/// the standard NanBox decoder.
fn decode_value_bits(bits: u64, ids: HeapTypeIds) -> Object {
    match nanbox_tag(bits) {
        None => Object::Double(f64::from_bits(bits)),
        Some(TAG_NIL) => Object::Nil,
        Some(TAG_BOOL) => Object::Bool(nanbox_payload(bits) != 0),
        Some(TAG_PTR) => unsafe { heap_bits_to_object(bits, ids) },
        _ => Object::Unported {
            java_class: "unknown NanBox tag in value field",
        },
    }
}

/// Decode a Cons `rest` field into an `Arc<PersistentList>`. Nil terminates;
/// a Cons pointer continues the chain.
fn decode_rest_to_list(
    bits: u64,
    ids: HeapTypeIds,
) -> std::sync::Arc<crate::lang::persistent_list::PersistentList> {
    use crate::lang::persistent_list::PersistentList;
    match nanbox_tag(bits) {
        Some(TAG_NIL) => PersistentList::empty(),
        Some(TAG_PTR) => {
            let obj = unsafe { heap_bits_to_object(bits, ids) };
            match obj {
                Object::List(l) => l,
                // A lazy rest that forces to the empty seq decodes to Nil.
                Object::Nil => PersistentList::empty(),
                other => panic!("clojure-jvm: Cons.rest expected nil or list, got {other:?}"),
            }
        }
        _ => panic!("clojure-jvm: Cons.rest must be nil or a Cons pointer"),
    }
}


// ── Host-method externs (clojure.lang.RT) ──────────────────────────────
//
// These are direct Rust ports of `clojure.lang.RT` static methods that
// `clojure.core` invokes through `(. clojure.lang.RT (methodname ...))`.
// HostExpr's emit routes those forms to a `fb.call(fref, args)` on the
// matching extern. As more of clojure.core gets ported, more methods
// land here.

// ── First-class fn invocation (Java `IFn.invoke`) ──────────────────────
//
// A Clojure fn value is a NanBox-tagged `FuncRef` index (TAG_FN). To call
// one we need to (a) decode the index, (b) load the entry-point pointer
// out of the JitModule's call table, (c) call it indirectly. The call
// table base is installed per-thread by `compile_form_to_jit` before
// running the entry fn, mirroring how `DynGcRuntime::install_thread` sets
// the GC runtime pointer.

thread_local! {
    static CALL_TABLE_BASE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Install the current JIT module's call-table base address for the
/// duration of a `gc.run_jit` invocation. Returns a guard that restores
/// the previous value on drop.
pub fn install_call_table_base(base: u64) -> CallTableBaseGuard {
    let prev = CALL_TABLE_BASE.with(|c| c.replace(base));
    CallTableBaseGuard { prev }
}

pub struct CallTableBaseGuard {
    prev: u64,
}

impl Drop for CallTableBaseGuard {
    fn drop(&mut self) {
        CALL_TABLE_BASE.with(|c| c.set(self.prev));
    }
}

fn call_table_base() -> u64 {
    let base = CALL_TABLE_BASE.with(|c| c.get());
    if base != 0 {
        return base;
    }
    // Not installed on this thread-local — we're dispatching a JIT fn from a
    // Rust context OUTSIDE a `run_jit` scope. The canonical case: forcing a
    // lazy-seq / Delay thunk during a Rust-side realize (`pr_str_bits`,
    // diagnostics) AFTER `eval_form` returned and dropped its guard. Recover
    // the base from the active Session's JIT so the thunk can still run.
    if let Some(b) = crate::lang::compiler::active_session_call_table_base() {
        if b != 0 {
            return b;
        }
    }
    panic!(
        "clojure-jvm: call_table_base not installed and no active Session — \
         cannot dispatch a JIT fn from this context"
    );
}

/// Like `dispatch_target_with_idx`, but if the receiver is a
/// `clojure.lang.MultiArityFn` dispatcher cell, picks the clause whose
/// arity matches `n` first — so dynamic invokes through Var values land
/// on the right clause without help from compile-time `var_multi_arity`.
unsafe fn dispatch_with_arity(handle_bits: u64, n: usize) -> (*const u8, Option<u64>, u32) {
    if let Some(t) = unsafe { try_dispatch_multi_arity(handle_bits, n) } {
        return t;
    }
    unsafe { dispatch_target_with_idx(handle_bits) }
}

/// Decode a fn-value NanBox into a function pointer + optional self arg
/// + the FuncRef index (so callers can look up arity info).
unsafe fn dispatch_target_with_idx(handle_bits: u64) -> (*const u8, Option<u64>, u32) {
    match nanbox_tag(handle_bits) {
        Some(TAG_FN) => {
            let idx = nanbox_payload(handle_bits) as usize;
            let slot_addr = call_table_base() + (idx as u64) * 8;
            let ptr = unsafe { *(slot_addr as *const *const u8) };
            (ptr, None, idx as u32)
        }
        Some(TAG_PTR) => {
            let ids = heap_type_ids();
            let raw = nanbox_payload(handle_bits) as *const u8;
            if raw.is_null() {
                eprintln!(
                    "[cljvm-stub] cljvm_rt_invoke_*: receiver is nil — \
                     dispatching to nil-returning stub"
                );
                return (cljvm_unimpl_host_call_0 as *const u8, None, u32::MAX);
            }
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.string
                || type_id == ids.symbol
                || type_id == ids.keyword
                || type_id == ids.cons
            {
                eprintln!(
                    "[cljvm-stub] cljvm_rt_invoke_*: receiver is non-callable \
                     heap (type_id {type_id}) — nil-returning stub"
                );
                return (cljvm_unimpl_host_call_0 as *const u8, None, u32::MAX);
            }
            if type_id == ids.multi_arity_fn {
                panic!(
                    "clojure-jvm: cljvm_rt_invoke_*: MultiArityFn receiver \
                     reached arity-blind dispatch_target_with_idx — caller \
                     must use dispatch_with_arity(fn_bits, n) instead so we \
                     can pick the right clause."
                );
            }
            if type_id != ids.closure {
                // Only Closure cells carry a fref_index at offset 16 —
                // treating any other cell (LazySeq, Repeat, vector, …) as
                // a closure reads garbage and SIGBUSes on the call-table
                // load. Java throws ClassCastException here.
                panic!(
                    "clojure-jvm: cljvm_rt_invoke_*: receiver type_id {type_id} \
                     is not invokable (expected a fn/closure)"
                );
            }
            let fref_idx = unsafe { raw.add(16).cast::<u64>().read_unaligned() } as u32;
            let slot_addr = call_table_base() + (fref_idx as u64) * 8;
            let ptr = unsafe { *(slot_addr as *const *const u8) };
            (ptr, Some(handle_bits), fref_idx)
        }
        _ => {
            eprintln!(
                "[cljvm-stub] cljvm_rt_invoke_*: receiver is not a fn \
                 (bits 0x{handle_bits:x}) — nil-returning stub"
            );
            (cljvm_unimpl_host_call_0 as *const u8, None, u32::MAX)
        }
    }
}

/// Decode a fn-value NanBox into a function pointer + optional self arg.
/// Two shapes:
///   * TAG_FN: payload is the FuncRef index; no self arg.
///   * TAG_PTR pointing at a `clojure.lang.Closure`: read the
///     `fref_index` Raw64 field; the closure handle becomes the body's
///     implicit first arg.
unsafe fn dispatch_target(handle_bits: u64) -> (*const u8, Option<u64>) {
    match nanbox_tag(handle_bits) {
        Some(TAG_FN) => {
            let idx = nanbox_payload(handle_bits) as usize;
            let slot_addr = call_table_base() + (idx as u64) * 8;
            let ptr = unsafe { *(slot_addr as *const *const u8) };
            (ptr, None)
        }
        Some(TAG_PTR) => {
            let ids = heap_type_ids();
            let raw = nanbox_payload(handle_bits) as *const u8;
            if raw.is_null() {
                eprintln!(
                    "[cljvm-stub] dispatch_target: receiver is nil — \
                     nil-returning stub"
                );
                return (cljvm_unimpl_host_call_0 as *const u8, None);
            }
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.string
                || type_id == ids.symbol
                || type_id == ids.keyword
                || type_id == ids.cons
            {
                eprintln!(
                    "[cljvm-stub] dispatch_target: receiver is non-callable \
                     heap (type_id {type_id}) — nil-returning stub"
                );
                return (cljvm_unimpl_host_call_0 as *const u8, None);
            }
            if type_id == ids.multi_arity_fn {
                panic!(
                    "clojure-jvm: cljvm_rt_invoke_*: MultiArityFn receiver \
                     reached arity-blind dispatch_target — caller must use \
                     dispatch_with_arity(fn_bits, n) so we can pick the \
                     right clause."
                );
            }
            // Closure layout (from `dynlang`'s ObjType builder for a type
            // with `.field(_, Raw64).varlen_values()`):
            //   * [0..8]  Compact header (type_id at offset 0)
            //   * [8..16] One Value slot (unused — the builder reserves it
            //             because varlen_values's `fixed_fields` count
            //             includes the raw64 field)
            //   * [16..24] Raw64 `fref_index`
            //   * [24..32] varlen-count word
            //   * [32..]  varlen NanBox values (captures)
            // Read fref_index at offset 16.
            let fref_idx = unsafe { raw.add(16).cast::<u64>().read_unaligned() } as usize;
            let slot_addr = call_table_base() + (fref_idx as u64) * 8;
            let ptr = unsafe { *(slot_addr as *const *const u8) };
            (ptr, Some(handle_bits))
        }
        _ => {
            eprintln!(
                "[cljvm-stub] cljvm_rt_invoke_*: receiver is not a fn \
                 (bits 0x{handle_bits:x}) — nil-returning stub"
            );
            (cljvm_unimpl_host_call_0 as *const u8, None)
        }
    }
}

/// If the call site has more args than the body's `fixed_arity` AND
/// the body is variadic, fold the overflow into a list and return the
/// packed argument vector that should be passed to the body. The
/// returned vec contains `[(self_arg)?, fixed_args..., overflow_list]`.
/// Returns `None` when no packing is needed.
///
/// Every heap-pointer arg gets rooted on a `dynobj::roots::with_scope`
/// frame BEFORE we start cons-folding. Without that, intermediate
/// `cljvm_rt_cons` allocs may trigger a moving GC that relocates the
/// other args' heap cells — leaving stale `u64` NanBoxes in our slice
/// and producing a list with garbage heads. This was the root cause
/// of form 35's mysterious `Vector{[a,b]}` instead of `Symbol(def)`.
unsafe fn pack_variadic_args(
    self_arg: Option<u64>,
    fref_idx: u32,
    args: &[u64],
) -> Option<Vec<u64>> {
    let info = crate::lang::compiler::with_active_compiler_arity(fref_idx)?;
    if !info.is_variadic {
        return None;
    }
    // A variadic fn `[fixed… & rest]` is compiled to take `fixed_arity + 1`
    // params — the final one being the rest seq. So it must be packed even
    // when there is NO overflow (`args.len() == fixed_arity`): the rest is
    // then `nil` (matching Clojure's `((fn [& r] r))` => nil). Bailing here
    // would omit the rest param entirely, so the callee reads its
    // uninitialized rest register as garbage. Only `args.len() < fixed_arity`
    // (genuine under-arity) skips packing.
    if args.len() < info.fixed_arity {
        return None;
    }
    // Capacity: self_arg(1) + every arg + the accumulating tail.
    let cap = args.len() + 2;
    Some(dynobj::roots::gc_enter(cap, |heap, scope| {
        let self_root = self_arg.map(|s| scope.root::<()>(s));
        let arg_roots: Vec<_> = args.iter().map(|v| scope.root::<()>(*v)).collect();
        let tail_root = scope.root::<()>(nanbox_nil());
        // Fold overflow into a cons-list, right-to-left. `heap_cons` takes
        // `&mut Heap`, so the borrow checker forbids holding any un-rooted
        // reference across it — every value here is rooted.
        for i in (info.fixed_arity..args.len()).rev() {
            let new_tail = heap_cons(heap, arg_roots[i], tail_root);
            tail_root.set_raw(new_tail);
        }
        let mut packed: Vec<u64> = Vec::with_capacity(cap);
        if let Some(r) = &self_root {
            packed.push(r.get_raw(&*heap).bits());
        }
        for i in 0..info.fixed_arity {
            packed.push(arg_roots[i].get_raw(&*heap).bits());
        }
        packed.push(tail_root.get_raw(&*heap).bits());
        packed
    }))
}

/// Invoke `ptr` with `packed` args. Used by the variadic dispatch path
/// (`pack_variadic_args` returned `Some`) and by `(.applyTo …)`.
///
/// The match arms cover up to 20 args, matching upstream Clojure's
/// `IFn.invoke` overload count. A fn that needs more positional args
/// is variadic and the caller passes `fixed_arity + 1` here (with the
/// final arg being the packed tail list), so 20 covers fns up to
/// `fixed_arity = 19` — well past anything in clojure.core.
/// Call a JIT-compiled clojure fn (internal calling convention) with up to
/// 16 args placed in X0–X15. The C-ABI `transmute` path used for ≤8 args is
/// only correct because the AAPCS C ABI and the internal CC agree on the
/// first 8 GP arg registers (X0–X7). For 9–16 args they diverge: the C ABI
/// puts args 8+ on the stack, but JIT code reads them from X8–X15. This shim
/// loads X0–X15 directly so a 9-to-16-arg call into JIT code is correct.
///
/// It deliberately does NOT set up an FP fence (so a GC during the callee
/// still walks up through this Rust frame to the outer JIT frame, matching
/// the no-fence `transmute` path) and does NOT touch X23 (the callee
/// inherits the caller's control-context register, exactly as the transmute
/// path does — X23 is callee-saved and preserved across `clobber_abi("C")`).
///
/// # Safety
/// `ptr` must point to a JIT fn expecting `packed.len()` internal-CC args.
#[cfg(target_arch = "aarch64")]
unsafe fn call_jit_packed_regs(ptr: *const u8, packed: &[u64]) -> u64 {
    debug_assert!(
        (9..=16).contains(&packed.len()),
        "call_jit_packed_regs: arity {} out of the 9..=16 range",
        packed.len()
    );
    let mut regs = [0u64; 16];
    regs[..packed.len()].copy_from_slice(packed);
    let result: u64;
    unsafe {
        core::arch::asm!(
            "ldp x0, x1, [x16]",
            "ldp x2, x3, [x16, #16]",
            "ldp x4, x5, [x16, #32]",
            "ldp x6, x7, [x16, #48]",
            "ldp x8, x9, [x16, #64]",
            "ldp x10, x11, [x16, #80]",
            "ldp x12, x13, [x16, #96]",
            "ldp x14, x15, [x16, #112]",
            "blr x17",
            in("x16") regs.as_ptr(),
            in("x17") ptr,
            lateout("x0") result,
            clobber_abi("C"),
        );
    }
    result
}

unsafe fn call_with_packed(ptr: *const u8, packed: &[u64]) -> u64 {
    // 9..=16 args: the C-ABI `transmute` path would mis-place args 8+ (stack
    // vs X8–X15). Use the direct-register shim so JIT callees read them right.
    #[cfg(target_arch = "aarch64")]
    if (9..=16).contains(&packed.len()) {
        return unsafe { call_jit_packed_regs(ptr, packed) };
    }
    macro_rules! no_op_u64 {
        ($_:literal) => {
            u64
        };
    }
    macro_rules! call_n {
        ($n:literal, $($i:literal),*) => {{
            let f: unsafe extern "C" fn($(no_op_u64!($i)),*) -> u64 =
                unsafe { std::mem::transmute(ptr) };
            unsafe { f($(packed[$i]),*) }
        }};
    }
    match packed.len() {
        0 => {
            let f: unsafe extern "C" fn() -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f() }
        }
        1 => call_n!(1, 0),
        2 => call_n!(2, 0, 1),
        3 => call_n!(3, 0, 1, 2),
        4 => call_n!(4, 0, 1, 2, 3),
        5 => call_n!(5, 0, 1, 2, 3, 4),
        6 => call_n!(6, 0, 1, 2, 3, 4, 5),
        7 => call_n!(7, 0, 1, 2, 3, 4, 5, 6),
        8 => call_n!(8, 0, 1, 2, 3, 4, 5, 6, 7),
        9 => call_n!(9, 0, 1, 2, 3, 4, 5, 6, 7, 8),
        10 => call_n!(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        11 => call_n!(11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        12 => call_n!(12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        13 => call_n!(13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        14 => call_n!(14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        15 => call_n!(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
        16 => call_n!(16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        17 => call_n!(17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        18 => call_n!(
            18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
        ),
        19 => call_n!(
            19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
        ),
        20 => call_n!(
            20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
        ),
        n => panic!(
            "clojure-jvm: call_with_packed: arity {n} > 20; extend the match arms \
             (matches IFn.invoke's overload count). A variadic fn with fixed_arity \
             > 19 would land here — practically never, but raise the cap if seen."
        ),
    }
}

/// `IFn.invoke()` — 0 arity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_0(fn_bits: u64) -> u64 {
    if let Some(r) = unsafe { try_multifn_invoke(fn_bits, &[]) } {
        return r;
    }
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, 0) };
    // A 0-arg call of a variadic fn (e.g. `(list)` / `(concat)`, both
    // `[& xs]`) must still supply the rest param (an empty/`nil` seq).
    // `pack_variadic_args` builds `[self?, nil]`; without this the callee
    // reads its uninitialized rest register and returns garbage.
    if let Some(packed) = unsafe { pack_variadic_args(self_arg, fref_idx, &[]) } {
        return unsafe { call_with_packed(ptr, &packed) };
    }
    match self_arg {
        None => {
            let f: unsafe extern "C" fn() -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f() }
        }
        Some(s) => {
            let f: unsafe extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(s) }
        }
    }
}

/// `IFn.invoke(arg1)` — 1 arity. Also handles Clojure's keyword-as-fn
/// shortcut: `(:k m)` is `(get m :k nil)`. We dispatch that before
/// going through the regular fn-handle path so a Keyword receiver
/// doesn't trip the "non-callable heap value" panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_1(fn_bits: u64, a: u64) -> u64 {
    if let Some(r) = unsafe { try_multifn_invoke(fn_bits, &[a]) } {
        return r;
    }
    if let Some(v) = unsafe { keyword_as_fn_lookup(fn_bits, a, nanbox_nil()) } {
        return v;
    }
    if let Some(v) = unsafe { vector_or_map_or_set_as_fn_1(fn_bits, a) } {
        return v;
    }
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, 1) };
    if let Some(packed) = unsafe { pack_variadic_args(self_arg, fref_idx, &[a]) } {
        return unsafe { call_with_packed(ptr, &packed) };
    }
    let r = match self_arg {
        None => {
            let f: unsafe extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(a) }
        }
        Some(s) => {
            let f: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(s, a) }
        }
    };
    r
}

/// `(coll k)` shortcut for vector / map / set receivers — Clojure
/// makes these implement `IFn`. Vector: `(v idx)` = `(get v idx)`.
/// Map: `(m k)` = `(get m k)`. Set: `(s x)` = `x` if present else nil.
/// Returns `Some(result)` when fn_bits is one of these collections;
/// `None` to fall through to regular fn dispatch.
unsafe fn vector_or_map_or_set_as_fn_1(fn_bits: u64, a_bits: u64) -> Option<u64> {
    let ids = heap_type_ids();
    if !matches!(nanbox_tag(fn_bits), Some(TAG_PTR)) {
        return None;
    }
    let p = nanbox_payload(fn_bits) as *const u8;
    if p.is_null() {
        return None;
    }
    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
    if tid == ids.vector {
        if !nanbox_tag(a_bits).is_none() {
            return None;
        }
        let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as i64;
        let idx = arg_to_i64(a_bits);
        if idx < 0 || idx >= n {
            panic!("clojure-jvm: vector-as-fn index {idx} out of range [0,{n})");
        }
        return Some(unsafe {
            p.add(16 + (idx as usize) * 8)
                .cast::<u64>()
                .read_unaligned()
        });
    }
    if tid == ids.map {
        return Some(unsafe { cljvm_rt_get(fn_bits, a_bits) });
    }
    if tid == ids.set {
        let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::persistent_hash_set::PersistentHashSet;
        unsafe { Arc::increment_strong_count(arc_ptr) };
        let s = unsafe { Arc::from_raw(arc_ptr) };
        let key = any_bits_to_object(a_bits, ids);
        return Some(if s.contains(&key) {
            a_bits
        } else {
            nanbox_nil()
        });
    }
    None
}

/// `(:keyword map [not-found])` shortcut. Returns `Some(value)` when
/// `fn_bits` is a Keyword heap pointer, doing the lookup; `None`
/// otherwise so the caller can fall through to regular fn dispatch.
unsafe fn keyword_as_fn_lookup(fn_bits: u64, m_bits: u64, not_found: u64) -> Option<u64> {
    let ids = heap_type_ids();
    let raw = match nanbox_tag(fn_bits) {
        Some(TAG_PTR) => nanbox_payload(fn_bits) as *const u8,
        _ => return None,
    };
    if raw.is_null() {
        return None;
    }
    let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
    if type_id != ids.keyword {
        return None;
    }
    // Recover the Keyword's Arc.
    let kw_ptr = unsafe { raw.add(8).cast::<u64>().read_unaligned() }
        as *const crate::lang::keyword::Keyword;
    unsafe { Arc::increment_strong_count(kw_ptr) };
    let kw = unsafe { Arc::from_raw(kw_ptr) };
    let key = Object::Keyword(kw);

    // Map dispatch: nil → not_found; PHM → val_at; anything else →
    // not_found (Java throws on non-map; clojure-side we swallow until
    // we have IPersistentMap modeled for vectors/sets).
    match nanbox_tag(m_bits) {
        Some(TAG_NIL) => Some(not_found),
        Some(TAG_PTR) => {
            let mraw = nanbox_payload(m_bits) as *const u8;
            if mraw.is_null() {
                return Some(not_found);
            }
            let mtid = unsafe { mraw.cast::<u16>().read_unaligned() } as usize;
            if mtid != ids.map {
                return Some(not_found);
            }
            let arc_ptr = unsafe { mraw.add(8).cast::<u64>().read_unaligned() }
                as *const crate::lang::persistent_hash_map::PersistentHashMap;
            unsafe { Arc::increment_strong_count(arc_ptr) };
            let m = unsafe { Arc::from_raw(arc_ptr) };
            let v_obj = m.val_at(&key);
            if matches!(v_obj, Object::Nil) {
                return Some(not_found);
            }
            // Re-encode the Object back to a NanBox. For simple immediates,
            // use object_to_nanbox; for heap-resident Objects, the bits
            // are still live on the source map so we can fetch them via
            // alloc-style round-trip. Easiest: ask the active session.
            Some(crate::lang::compiler::with_active_session_encode_object(
                &v_obj,
            ))
        }
        _ => Some(not_found),
    }
}

/// `IFn.invoke(arg1, arg2)` — 2 arity. Handles `(:k m default)` and
/// any variadic target whose call-site arity exceeds its `fixed_arity`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_2(fn_bits: u64, a: u64, b: u64) -> u64 {
    if let Some(r) = unsafe { try_multifn_invoke(fn_bits, &[a, b]) } {
        return r;
    }
    if let Some(v) = unsafe { keyword_as_fn_lookup(fn_bits, a, b) } {
        return v;
    }
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, 2) };
    if let Some(packed) = unsafe { pack_variadic_args(self_arg, fref_idx, &[a, b]) } {
        return unsafe { call_with_packed(ptr, &packed) };
    }
    let r = match self_arg {
        None => {
            let f: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(a, b) }
        }
        Some(s) => {
            let f: unsafe extern "C" fn(u64, u64, u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(s, a, b) }
        }
    };
    if r < 0x1000 && std::env::var("CLJVM_INVOKE_TRACE").is_ok() {
        eprintln!(
            "[invoke_2] SUSPECT result=0x{r:x} fn_bits=0x{fn_bits:x} a=0x{a:x} b=0x{b:x} \
             ptr={ptr:p} self={}",
            self_arg.is_some()
        );
    }
    r
}

/// `IFn.invoke(arg1, arg2, arg3)` — 3 arity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_3(fn_bits: u64, a: u64, b: u64, c: u64) -> u64 {
    if let Some(r) = unsafe { try_multifn_invoke(fn_bits, &[a, b, c]) } {
        return r;
    }
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, 3) };
    if let Some(packed) = unsafe { pack_variadic_args(self_arg, fref_idx, &[a, b, c]) } {
        return unsafe { call_with_packed(ptr, &packed) };
    }
    match self_arg {
        None => {
            let f: unsafe extern "C" fn(u64, u64, u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(a, b, c) }
        }
        Some(s) => {
            let f: unsafe extern "C" fn(u64, u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
            unsafe { f(s, a, b, c) }
        }
    }
}

/// `IFn.invoke(a..d)` — 4 arity. See `cljvm_rt_invoke_3`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_4(fn_bits: u64, a: u64, b: u64, c: u64, d: u64) -> u64 {
    if let Some(r) = unsafe { try_multifn_invoke(fn_bits, &[a, b, c, d]) } {
        return r;
    }
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, 4) };
    if let Some(packed) = unsafe { pack_variadic_args(self_arg, fref_idx, &[a, b, c, d]) } {
        return unsafe { call_with_packed(ptr, &packed) };
    }
    match self_arg {
        None => {
            let f: unsafe extern "C" fn(u64, u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
            unsafe { f(a, b, c, d) }
        }
        Some(s) => {
            let f: unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
            unsafe { f(s, a, b, c, d) }
        }
    }
}

/// `IFn.invoke(a..e)` — 5 arity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_5(
    fn_bits: u64,
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    e: u64,
) -> u64 {
    if let Some(r) = unsafe { try_multifn_invoke(fn_bits, &[a, b, c, d, e]) } {
        return r;
    }
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, 5) };
    if let Some(packed) = unsafe { pack_variadic_args(self_arg, fref_idx, &[a, b, c, d, e]) } {
        return unsafe { call_with_packed(ptr, &packed) };
    }
    match self_arg {
        None => {
            let f: unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
            unsafe { f(a, b, c, d, e) }
        }
        Some(s) => {
            let f: unsafe extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
            unsafe { f(s, a, b, c, d, e) }
        }
    }
}

/// `IFn.invoke(a..f)` — 6 arity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_6(
    fn_bits: u64,
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    e: u64,
    f: u64,
) -> u64 {
    if let Some(r) = unsafe { try_multifn_invoke(fn_bits, &[a, b, c, d, e, f]) } {
        return r;
    }
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, 6) };
    if let Some(packed) = unsafe { pack_variadic_args(self_arg, fref_idx, &[a, b, c, d, e, f]) } {
        return unsafe { call_with_packed(ptr, &packed) };
    }
    match self_arg {
        None => {
            let g: unsafe extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
            unsafe { g(a, b, c, d, e, f) }
        }
        Some(s) => {
            let g: unsafe extern "C" fn(u64, u64, u64, u64, u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
            unsafe { g(s, a, b, c, d, e, f) }
        }
    }
}

/// `IFn.invoke(a..g)` — 7 arity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_7(
    fn_bits: u64,
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    e: u64,
    f: u64,
    g: u64,
) -> u64 {
    if let Some(r) = unsafe { try_multifn_invoke(fn_bits, &[a, b, c, d, e, f, g]) } {
        return r;
    }
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, 7) };
    if let Some(packed) = unsafe { pack_variadic_args(self_arg, fref_idx, &[a, b, c, d, e, f, g]) }
    {
        return unsafe { call_with_packed(ptr, &packed) };
    }
    match self_arg {
        None => {
            let h: unsafe extern "C" fn(u64, u64, u64, u64, u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
            unsafe { h(a, b, c, d, e, f, g) }
        }
        Some(s) => {
            let h: unsafe extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
            unsafe { h(s, a, b, c, d, e, f, g) }
        }
    }
}

/// `IFn.invoke(a..h)` — 8 arity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_8(
    fn_bits: u64,
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    e: u64,
    f: u64,
    g: u64,
    h: u64,
) -> u64 {
    if let Some(r) = unsafe { try_multifn_invoke(fn_bits, &[a, b, c, d, e, f, g, h]) } {
        return r;
    }
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, 8) };
    if let Some(packed) =
        unsafe { pack_variadic_args(self_arg, fref_idx, &[a, b, c, d, e, f, g, h]) }
    {
        return unsafe { call_with_packed(ptr, &packed) };
    }
    match self_arg {
        None => {
            let i: unsafe extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
            unsafe { i(a, b, c, d, e, f, g, h) }
        }
        Some(s) => {
            // 9 args (self + 8): a C-ABI transmute would put arg 8 on the
            // stack, but the JIT callee (internal CC) reads it from X8.
            // Route through `call_with_packed`, which uses the X0–X15 shim.
            unsafe { call_with_packed(ptr, &[s, a, b, c, d, e, f, g, h]) }
        }
    }
}

/// `IFn.invoke(a..i)` — 9 arity. From here up the C-ABI `transmute` path
/// is unusable in BOTH dispatch shapes (args ≥ 9 would go to the stack,
/// but the JIT callee's internal CC reads X0–X15), so everything routes
/// through `call_with_packed`'s direct-register shim.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_9(
    fn_bits: u64,
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    e: u64,
    f: u64,
    g: u64,
    h: u64,
    i: u64,
) -> u64 {
    if let Some(r) = unsafe { try_multifn_invoke(fn_bits, &[a, b, c, d, e, f, g, h, i]) } {
        return r;
    }
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, 9) };
    if let Some(packed) =
        unsafe { pack_variadic_args(self_arg, fref_idx, &[a, b, c, d, e, f, g, h, i]) }
    {
        return unsafe { call_with_packed(ptr, &packed) };
    }
    match self_arg {
        None => unsafe { call_with_packed(ptr, &[a, b, c, d, e, f, g, h, i]) },
        Some(s) => unsafe { call_with_packed(ptr, &[s, a, b, c, d, e, f, g, h, i]) },
    }
}

/// `IFn.invoke(a..j)` — 10 arity. See `cljvm_rt_invoke_9` for the
/// packed-register routing rationale.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_10(
    fn_bits: u64,
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    e: u64,
    f: u64,
    g: u64,
    h: u64,
    i: u64,
    j: u64,
) -> u64 {
    if let Some(r) = unsafe { try_multifn_invoke(fn_bits, &[a, b, c, d, e, f, g, h, i, j]) } {
        return r;
    }
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, 10) };
    if let Some(packed) =
        unsafe { pack_variadic_args(self_arg, fref_idx, &[a, b, c, d, e, f, g, h, i, j]) }
    {
        return unsafe { call_with_packed(ptr, &packed) };
    }
    match self_arg {
        None => unsafe { call_with_packed(ptr, &[a, b, c, d, e, f, g, h, i, j]) },
        Some(s) => unsafe { call_with_packed(ptr, &[s, a, b, c, d, e, f, g, h, i, j]) },
    }
}

/// `clojure.lang.RT.inc(Object x)` — for now, a primitive Long-only
/// increment. Real RT.inc dispatches on Number subtypes; we'll widen as
/// the Numbers port lands. This exists as the bring-up vehicle for the
/// `(. clojure.lang.RT (...))` codegen path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_inc(x_bits: u64) -> u64 {
    let x = arg_to_i64(x_bits);
    box_long(x + 1)
}

/// `clojure.lang.RT/intCast(n)` — truncate to 32-bit int. We model all
/// integers as i64 so this is a `(i64 as i32) as i64` round-trip.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_intCast(n_bits: u64) -> u64 {
    let n = arg_to_i64(n_bits) as i32 as i64;
    box_long(n)
}

/// `clojure.lang.RT/longCast(n)` — coerce to long. Already long for us.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_longCast(n_bits: u64) -> u64 {
    let n = arg_to_i64(n_bits);
    box_long(n)
}

/// `clojure.lang.RT/booleanCast(x)` — Clojure truthiness: nil and false
/// are falsey; everything else (including 0, "", '()) is truthy.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_booleanCast(x_bits: u64) -> u64 {
    let truthy = match nanbox_tag(x_bits) {
        Some(TAG_NIL) => false,
        Some(TAG_BOOL) => nanbox_payload(x_bits) != 0,
        _ => true,
    };
    nanbox_bool(truthy)
}

/// `clojure.lang.RT/doubleCast(n)` — coerce to double. f64 already.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_doubleCast(n_bits: u64) -> u64 {
    let n = arg_to_f64(n_bits);
    n.to_bits()
}

/// `clojure.lang.RT/floatCast(n)` — coerce to float (round-trip via f32).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_floatCast(n_bits: u64) -> u64 {
    let n = arg_to_f64(n_bits) as f32 as f64;
    n.to_bits()
}

/// `clojure.lang.RT/byteCast(n)` — coerce to i8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_byteCast(n_bits: u64) -> u64 {
    let n = arg_to_i64(n_bits) as i8 as i64;
    box_long(n)
}

/// `clojure.lang.RT/shortCast(n)` — coerce to i16.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_shortCast(n_bits: u64) -> u64 {
    let n = arg_to_i64(n_bits) as i16 as i64;
    box_long(n)
}

/// `clojure.lang.RT/charCast(n)` / `(char n)` — coerce to a Character. Returns
/// a boxed `clojure.lang.Character` (not a Long), so `(str (char 47))` is
/// `"/"`. Accepts a Long, a Character, or a native number.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_charCast(n_bits: u64) -> u64 {
    let n = arg_to_i64(n_bits) as u16 as u32;
    unsafe { box_char(n) }
}

// ── clojure.lang.Numbers — arithmetic + comparisons ──────────────────────
//
// All operate on NanBox-encoded numerics (Long stored as f64::to_bits, Double
// likewise). Round-trip through f64 then back. For now we don't distinguish
// Long vs Double — Clojure does (Long+Long = Long, mixed promotes to Double),
// but our wider-than-needed model just uses f64 throughout. When upstream
// behavior diverges we can split later.

fn num_to_f64(bits: u64) -> f64 {
    arg_to_f64(bits)
}

// These `Numbers.*` arithmetic ops route through the same boxed-Long/double
// tower as the `cljvm_num_*` externs, so `(. Numbers (add 1 2))` is `3` (a
// Long), not `3.0`. (`PrimOpExpr` calls `cljvm_num_*` directly; these exist
// for explicit `clojure.lang.Numbers` host-method calls.)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_add(a: u64, b: u64) -> u64 {
    let (na, nb) = (decode_num(a), decode_num(b));
    encode_numresult(value_repr::num_add(na, nb), na, nb, |x, y| x + y)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_minus(a: u64, b: u64) -> u64 {
    let (na, nb) = (decode_num(a), decode_num(b));
    encode_numresult(value_repr::num_sub(na, nb), na, nb, |x, y| x - y)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_minus_1(a: u64) -> u64 {
    let (na, nb) = (Num::Long(0), decode_num(a));
    encode_numresult(value_repr::num_sub(na, nb), na, nb, |x, y| x - y)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_multiply(a: u64, b: u64) -> u64 {
    let (na, nb) = (decode_num(a), decode_num(b));
    encode_numresult(value_repr::num_mul(na, nb), na, nb, |x, y| x * y)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_divide(a: u64, b: u64) -> u64 {
    encode_num(value_repr::num_div(decode_num(a), decode_num(b)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_inc(a: u64) -> u64 {
    let (na, nb) = (decode_num(a), Num::Long(1));
    encode_numresult(value_repr::num_add(na, nb), na, nb, |x, y| x + y)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_dec(a: u64) -> u64 {
    let (na, nb) = (decode_num(a), Num::Long(1));
    encode_numresult(value_repr::num_sub(na, nb), na, nb, |x, y| x - y)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_lt(a: u64, b: u64) -> u64 {
    nanbox_bool(num_to_f64(a) < num_to_f64(b))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_lte(a: u64, b: u64) -> u64 {
    nanbox_bool(num_to_f64(a) <= num_to_f64(b))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_gt(a: u64, b: u64) -> u64 {
    nanbox_bool(num_to_f64(a) > num_to_f64(b))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_gte(a: u64, b: u64) -> u64 {
    nanbox_bool(num_to_f64(a) >= num_to_f64(b))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_equiv(a: u64, b: u64) -> u64 {
    nanbox_bool(num_to_f64(a) == num_to_f64(b))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_isPos(a: u64) -> u64 {
    nanbox_bool(num_to_f64(a) > 0.0)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_isNeg(a: u64) -> u64 {
    nanbox_bool(num_to_f64(a) < 0.0)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_max(a: u64, b: u64) -> u64 {
    let (x, y) = (num_to_f64(a), num_to_f64(b));
    if x >= y { a } else { b }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_min(a: u64, b: u64) -> u64 {
    let (x, y) = (num_to_f64(a), num_to_f64(b));
    if x <= y { a } else { b }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_quotient(a: u64, b: u64) -> u64 {
    encode_num(value_repr::num_quot(decode_num(a), decode_num(b)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_remainder(a: u64, b: u64) -> u64 {
    encode_num(value_repr::num_rem(decode_num(a), decode_num(b)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_abs(a: u64) -> u64 {
    match decode_num(a) {
        Num::Long(v) => box_long(v.abs()),
        Num::Double(d) => nanbox_double(d.abs()),
    }
}
/// `java.lang.Math/pow(a, b)` — `a^b` as a double.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_math_pow(a: u64, b: u64) -> u64 {
    nanbox_double(arg_to_f64(a).powf(arg_to_f64(b)))
}
/// `java.lang.Math/sqrt(a)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_math_sqrt(a: u64) -> u64 {
    nanbox_double(arg_to_f64(a).sqrt())
}
/// `java.lang.Math/abs(a)` — Long in → Long out, Double in → Double out
/// (mirrors the Java overloads).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_math_abs(a: u64) -> u64 {
    if is_boxed_long(a) {
        unsafe { box_long(arg_to_i64(a).abs()) }
    } else {
        nanbox_double(arg_to_f64(a).abs())
    }
}
/// `java.lang.Math/floor(a)` — always returns a double, like Java.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_math_floor(a: u64) -> u64 {
    nanbox_double(arg_to_f64(a).floor())
}
/// `java.lang.Math/ceil(a)` — always returns a double, like Java.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_math_ceil(a: u64) -> u64 {
    nanbox_double(arg_to_f64(a).ceil())
}
/// `Integer/parseInt(s)` / `Long/parseLong(s)` — parse a base-10 integer.
/// Panics (like Java throws NumberFormatException) on a malformed string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_long_parse(s_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = unsafe { read_string_heap(s_bits, ids, "Long/parseLong") };
    match s.trim().parse::<i64>() {
        Ok(n) => unsafe { box_long(n) },
        Err(e) => panic!("clojure-jvm: NumberFormatException — parseLong(\"{s}\"): {e}"),
    }
}
/// `Double/parseDouble(s)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_double_parse(s_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = unsafe { read_string_heap(s_bits, ids, "Double/parseDouble") };
    match s.trim().parse::<f64>() {
        Ok(f) => nanbox_double(f),
        Err(e) => panic!("clojure-jvm: NumberFormatException — parseDouble(\"{s}\"): {e}"),
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_identity(a: u64) -> u64 {
    a
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_isInteger(a: u64) -> u64 {
    let f = num_to_f64(a);
    nanbox_bool(f.fract() == 0.0 && f.is_finite())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_isFloat(a: u64) -> u64 {
    nanbox_bool(matches!(nanbox_tag(a), None))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_isRational(a: u64) -> u64 {
    // We don't have ratios — only integers count as rational here.
    let f = num_to_f64(a);
    nanbox_bool(f.fract() == 0.0 && f.is_finite())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_isNaN(a: u64) -> u64 {
    nanbox_bool(num_to_f64(a).is_nan())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_isInfinite(a: u64) -> u64 {
    nanbox_bool(num_to_f64(a).is_infinite())
}
// Bitwise ops on Numbers (treat as i64).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_not(a: u64) -> u64 {
    let n = num_to_f64(a) as i64;
    box_long(!n)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_and(a: u64, b: u64) -> u64 {
    let (x, y) = (num_to_f64(a) as i64, num_to_f64(b) as i64);
    box_long(x & y)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_or(a: u64, b: u64) -> u64 {
    let (x, y) = (num_to_f64(a) as i64, num_to_f64(b) as i64);
    box_long(x | y)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_xor(a: u64, b: u64) -> u64 {
    let (x, y) = (num_to_f64(a) as i64, num_to_f64(b) as i64);
    box_long(x ^ y)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_andNot(a: u64, b: u64) -> u64 {
    let (x, y) = (num_to_f64(a) as i64, num_to_f64(b) as i64);
    box_long(x & !y)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_shiftLeft(a: u64, b: u64) -> u64 {
    let (x, y) = (num_to_f64(a) as i64, num_to_f64(b) as i64);
    box_long(x << (y & 63))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_shiftRight(a: u64, b: u64) -> u64 {
    let (x, y) = (num_to_f64(a) as i64, num_to_f64(b) as i64);
    box_long(x >> (y & 63))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_unsignedShiftRight(a: u64, b: u64) -> u64 {
    let (x, y) = (num_to_f64(a) as u64, num_to_f64(b) as i64);
    box_long((x >> (y & 63)) as i64)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_clearBit(a: u64, b: u64) -> u64 {
    let (x, y) = (num_to_f64(a) as i64, num_to_f64(b) as i64);
    box_long(x & !(1i64 << (y & 63)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_setBit(a: u64, b: u64) -> u64 {
    let (x, y) = (num_to_f64(a) as i64, num_to_f64(b) as i64);
    box_long(x | (1i64 << (y & 63)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_flipBit(a: u64, b: u64) -> u64 {
    let (x, y) = (num_to_f64(a) as i64, num_to_f64(b) as i64);
    box_long(x ^ (1i64 << (y & 63)))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_testBit(a: u64, b: u64) -> u64 {
    let (x, y) = (num_to_f64(a) as i64, num_to_f64(b) as i64);
    nanbox_bool((x & (1i64 << (y & 63))) != 0)
}

/// `clojure.lang.Numbers/isZero(n)` — true iff n is zero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_isZero(n_bits: u64) -> u64 {
    let obj = any_bits_to_object(n_bits, heap_type_ids());
    let is_zero = match obj {
        Object::Long(n) => n == 0,
        Object::Double(d) => d == 0.0,
        other => panic!("clojure-jvm: Numbers/isZero requires a Number, got {other:?}"),
    };
    nanbox_bool(is_zero)
}

/// `clojure.lang.RT.nth(coll, idx)` — get the `idx`-th element. Vector
/// is direct indexing; Cons walks the chain. Negative or out-of-range
/// throws `IndexOutOfBoundsException` (we panic).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_nth(coll_bits: u64, idx_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let idx = arg_to_i64(idx_bits);
    if idx < 0 {
        eprintln!("[cljvm-stub] RT.nth idx={idx} negative — nil");
        return nanbox_nil();
    }
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => {
            eprintln!("[cljvm-stub] RT.nth on nil — nil");
            nanbox_nil()
        }
        Some(TAG_PTR) => {
            let p = nanbox_payload(coll_bits) as *const u8;
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.vector {
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as i64;
                if idx >= n {
                    eprintln!("[cljvm-stub] RT.nth idx={idx} >= count={n} — nil");
                    return nanbox_nil();
                }
                unsafe {
                    p.add(16 + (idx as usize) * 8)
                        .cast::<u64>()
                        .read_unaligned()
                }
            } else if tid == ids.cons || tid == ids.lazy_seq || tid == ids.repeat_seq
                || tid == ids.iterate_seq || tid == ids.cycle_seq {
                // Walk a seq (cons chain, lazy seq, or repeat) via first/next, both of
                // which force lazy nodes. `cljvm_rt_next` returns the realized
                // tail, so `nth` on a lazy seq realizes only as far as `idx`.
                // `nth` over a lazy seq is what positional destructuring lowers
                // to (`(nth coll i nil)`), so handling `lazy_seq` here makes
                // `(let [[a b] (split-at 2 xs)] …)` bind correctly.
                let mut cur = coll_bits;
                for _ in 0..idx {
                    cur = unsafe { cljvm_rt_next(cur) };
                    if matches!(nanbox_tag(cur), Some(TAG_NIL)) {
                        eprintln!("[cljvm-stub] RT.nth past end — nil");
                        return nanbox_nil();
                    }
                }
                unsafe { cljvm_rt_first(cur) }
            } else {
                eprintln!("[cljvm-stub] RT.nth on unsupported type_id {tid} — nil");
                nanbox_nil()
            }
        }
        _ => {
            eprintln!("[cljvm-stub] RT.nth on non-coll NanBox — nil");
            nanbox_nil()
        }
    }
}

/// `clojure.lang.RT.nth(coll, idx, not-found)` — like nth/2 but returns
/// `not-found` instead of throwing on out-of-range.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_nth_3(coll_bits: u64, idx_bits: u64, not_found_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let idx = arg_to_i64(idx_bits);
    if idx < 0 {
        return not_found_bits;
    }
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => not_found_bits,
        Some(TAG_PTR) => {
            let p = nanbox_payload(coll_bits) as *const u8;
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.vector {
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as i64;
                if idx >= n {
                    return not_found_bits;
                }
                unsafe {
                    p.add(16 + (idx as usize) * 8)
                        .cast::<u64>()
                        .read_unaligned()
                }
            } else if tid == ids.cons || tid == ids.lazy_seq || tid == ids.repeat_seq
                || tid == ids.iterate_seq || tid == ids.cycle_seq {
                // Walk a seq (cons chain, lazy seq, or repeat) via first/next, same as
                // `cljvm_rt_nth` above. `nth_3` is what positional
                // destructuring lowers to (`(nth tmp i nil)`), and macros
                // destructure lazy-seq values at MACROEXPAND time — e.g. the
                // `case` macro's `(fn [m [test expr]] …)` over `partition`
                // output. Historically lazy forcing here aborted (JIT thunk
                // dispatch without an installed call-table base), but macro
                // bodies now execute via `run_jit` with the call table live —
                // the same context where `first`/`next` already force lazy
                // seqs — so the exclusion only produced silent wrong nils
                // (destructured bindings = not-found).
                let mut cur = coll_bits;
                for _ in 0..idx {
                    cur = unsafe { cljvm_rt_next(cur) };
                    if matches!(nanbox_tag(cur), Some(TAG_NIL)) {
                        return not_found_bits;
                    }
                }
                unsafe { cljvm_rt_first(cur) }
            } else {
                not_found_bits
            }
        }
        _ => not_found_bits,
    }
}

/// `clojure.lang.RT.nextID()` — monotonic counter for `gensym`. Java
/// uses an `AtomicInteger`; we use a process-global `AtomicI64`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_nextID() -> u64 {
    use std::sync::atomic::{AtomicI64, Ordering};
    static NEXT_ID: AtomicI64 = AtomicI64::new(1);
    let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    box_long(n)
}

/// `clojure.lang.RT.cons(Object x, Object seq)` — allocate a new Cons cell
/// linking `x` and `seq`.
///
/// Mirrors Java's `RT.cons`: if `seq` isn't already an `ISeq` (Cons or nil),
/// convert it via `RT.seq` first. Without this, `(cons x [a b c])` would
/// store the Vector directly in `rest`, breaking downstream walkers (e.g.,
/// `(vec (cons '&form (cons '&env [test & body])))` in defn's macro body
/// would otherwise produce a vector containing the inner vector as a
/// single element, and `test`/`body` would never become fn params).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_cons(x_bits: u64, seq_bits: u64) -> u64 {
    // The whole body runs under the `Heap` capability so the borrow checker
    // can enforce the GC contract: `x` and `seq` are rooted up front, and
    // every allocation (`heap_seq`, `heap_alloc`) takes `&mut Heap`, which
    // makes holding an un-rooted `Raw` across it a compile error. This is
    // the type-level version of the old form-430 stale-pointer fix.
    // 3 slots: x, seq, and the freshly allocated Cons cell.
    dynobj::roots::gc_enter(3, |heap, scope| {
        let x = scope.root::<()>(x_bits);
        let seq = scope.root::<()>(seq_bits);
        cons_inner(heap, scope, x, seq).bits()
    })
}

/// Read a NanBox **value slot** out of a heap object as a [`Raw`] tied to
/// `heap`. This is the type-safe replacement for a bare
/// `base.add(off).cast::<u64>().read_unaligned()` on a GC-value field: the
/// returned reference borrows `&Heap`, so the borrow checker forbids holding
/// it across a `&mut Heap` allocation (the form-430 stale-pointer bug).
///
/// Use ONLY for slots that hold a NanBox value (a GC pointer or immediate) —
/// NOT for object headers (`u16` type_id), varlen counts, or Rust-side `Arc`
/// pointers, which are not GC values and stay as plain reads.
///
/// # Safety
/// `base` must point at a live object of a type whose slot at `off` is a
/// NanBox value field.
#[inline]
unsafe fn read_slot<'h>(heap: &'h Heap, base: *const u8, off: usize) -> Raw<'h> {
    let bits = unsafe { base.add(off).cast::<u64>().read_unaligned() };
    Raw::from_bits(heap, bits)
}

/// Write a NanBox value `v` into a heap object's value slot. Takes a [`Raw`],
/// so the caller has already proven (via the borrow checker) that `v` is a
/// live, non-stale value at the point of the write — no bare bits.
///
/// # Safety
/// `base` must point at a live object with a NanBox value slot at `off`.
#[inline]
unsafe fn write_slot(base: *mut u8, off: usize, v: Raw<'_>) {
    unsafe {
        base.add(off).cast::<u64>().write_unaligned(v.bits());
    }
}

/// Write a non-NanBox machine word (a varlen count, a `user_type_id`, or a
/// Rust-side `Arc` pointer) into a heap object slot. This is NOT a GC value:
/// it can't dangle across a collection, so it needs no `Heap`/`Raw`. It
/// exists as the sanctioned counterpart to [`write_slot`] so the
/// `gc_slot_writes_are_typed` guard can forbid ALL other bare
/// `write_unaligned::<u64>` — every word write picks `write_slot` (GC value)
/// or `write_raw_word` (plain word) explicitly.
///
/// # Safety
/// `base` must point at a live object with a `u64`-sized slot at `off`.
#[inline]
unsafe fn write_raw_word(base: *mut u8, off: usize, word: u64) {
    unsafe {
        base.add(off).cast::<u64>().write_unaligned(word);
    }
}

/// Allocate a fixed-size GC object of `type_id` under the heap capability,
/// returning the boxed owner reference as a `Raw` tied to `heap`. Because
/// it takes `&mut Heap`, callers cannot hold an un-rooted `Raw` across it.
#[inline]
fn heap_alloc<'h>(heap: &'h mut Heap, type_id: u64, varlen_len: u64) -> Raw<'h> {
    let raw = dynlang::gc::gc_alloc_thunk(type_id, varlen_len);
    assert!(raw != 0, "clojure-jvm: gc_alloc returned null");
    Raw::from_bits(&*heap, nanbox_encode(TAG_PTR, raw & PAYLOAD_MASK))
}

/// `clojure.lang.RT.seq` as a GC-causing primitive. Takes `&mut Heap`
/// (it can allocate) and a *rooted* argument (so the value survives the
/// collection), returning a fresh `Raw` for the coerced seq.
#[inline]
fn heap_seq<'h>(heap: &'h mut Heap, v: Rooted<'_, ()>) -> Raw<'h> {
    // Reading the rooted bits to feed the primitive is safe: no GC happens
    // between the read and the call, and `cljvm_rt_seq` re-roots internally.
    let in_bits = v.get_raw(&*heap).bits();
    let out = unsafe { cljvm_rt_seq(in_bits) };
    Raw::from_bits(&*heap, out)
}

/// `RT.cons` as a GC-causing primitive. `&mut Heap` (it allocates) plus
/// rooted `x`/`tail` (so they survive). Returns a `Raw` for the new cell.
#[inline]
fn heap_cons<'h>(heap: &'h mut Heap, x: Rooted<'_, ()>, tail: Rooted<'_, ()>) -> Raw<'h> {
    let x_bits = x.get_raw(&*heap).bits();
    let tail_bits = tail.get_raw(&*heap).bits();
    let out = unsafe { cljvm_rt_cons(x_bits, tail_bits) };
    Raw::from_bits(&*heap, out)
}

/// Build a finite list of boxed Longs `start, start+step, …` (toward `end`,
/// exclusive). Backs `clojure.lang.LongRange/create` (the impl `range` calls).
/// Eager (not lazy) but produces a real seq (cons chain), so `pr-str` matches
/// Clojure's `(0 1 2 …)`. `step == 0` yields the empty list (Clojure's
/// infinite-repeat case is avoided to never hang).
unsafe fn longrange_build(start: i64, end: i64, step: i64) -> u64 {
    let mut vals: Vec<i64> = Vec::new();
    if step > 0 {
        let mut i = start;
        while i < end {
            vals.push(i);
            i = i.wrapping_add(step);
        }
    } else if step < 0 {
        let mut i = start;
        while i > end {
            vals.push(i);
            i = i.wrapping_add(step);
        }
    }
    let n = vals.len();
    if n == 0 {
        return nanbox_nil();
    }
    dynobj::roots::gc_enter(n + 1, |heap, scope| {
        let cur = scope.root::<()>(nanbox_nil());
        for &v in vals.iter().rev() {
            // box_long allocates; root it before heap_cons allocates again.
            // `cur` (the chain so far) is rooted, so it survives both.
            let boxed = scope.root::<()>(box_long(v));
            let new_cur = heap_cons(heap, boxed, cur);
            cur.set_raw(new_cur);
        }
        cur.get_raw(&*heap).bits()
    })
}

/// `clojure.lang.LongRange/create(end)` → `(range 0 end 1)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_longrange_create_1(end_bits: u64) -> u64 {
    unsafe { longrange_build(0, arg_to_i64(end_bits), 1) }
}
/// `clojure.lang.LongRange/create(start, end)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_longrange_create_2(start_bits: u64, end_bits: u64) -> u64 {
    unsafe { longrange_build(arg_to_i64(start_bits), arg_to_i64(end_bits), 1) }
}
/// `clojure.lang.LongRange/create(start, end, step)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_longrange_create_3(start_bits: u64, end_bits: u64, step_bits: u64) -> u64 {
    unsafe { longrange_build(arg_to_i64(start_bits), arg_to_i64(end_bits), arg_to_i64(step_bits)) }
}

/// Allocate a `clojure.lang.Repeat` cell (see `HeapTypeIds::repeat_seq`
/// for the layout/semantics). `count` is -1 (infinite) or > 0; callers
/// handle the <= 0 cases before allocating.
fn alloc_repeat(count: i64, value_bits: u64) -> u64 {
    debug_assert!(count == -1 || count > 0);
    let ids = heap_type_ids();
    dynobj::roots::gc_enter(2, |heap, scope| {
        let value = scope.root::<()>(value_bits);
        let cell = heap_alloc(heap, ids.repeat_seq as u64, 0).root(scope);
        let cell_bits = cell.get_raw(&*heap).bits();
        let ptr = nanbox_payload(cell_bits) as *mut u8;
        unsafe {
            write_slot(ptr, 8, value.get_raw(&*heap));
            write_raw_word(ptr, 16, count as u64);
        }
        cell_bits
    })
}

/// `clojure.lang.Repeat/create(x)` — the INFINITE `(repeat x)`. Safe to
/// represent now that seq/first/next dispatch on the Repeat type id: lazy
/// consumers (`take`, `interleave`, …) pull one element per step instead
/// of structurally walking a cons chain.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_repeat_create_1(x_bits: u64) -> u64 {
    alloc_repeat(-1, x_bits)
}

/// `clojure.lang.Repeat/create(n, x)` — bounded `(repeat n x)`. `n <= 0`
/// yields the empty seq (nil — same compromise as `longrange_build`; Java
/// returns `PersistentList.EMPTY`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_repeat_create_2(n_bits: u64, x_bits: u64) -> u64 {
    let n = arg_to_i64(n_bits);
    if n <= 0 {
        return nanbox_nil();
    }
    alloc_repeat(n, x_bits)
}

/// Allocate a two-traced-slot seq cell (Iterate / Cycle).
fn alloc_seq2(type_id: usize, a_bits: u64, b_bits: u64) -> u64 {
    dynobj::roots::gc_enter(3, |heap, scope| {
        let a = scope.root::<()>(a_bits);
        let b = scope.root::<()>(b_bits);
        let cell = heap_alloc(heap, type_id as u64, 0).root(scope);
        let cell_bits = cell.get_raw(&*heap).bits();
        let ptr = nanbox_payload(cell_bits) as *mut u8;
        unsafe {
            write_slot(ptr, 8, a.get_raw(&*heap));
            write_slot(ptr, 16, b.get_raw(&*heap));
        }
        cell_bits
    })
}

/// `clojure.lang.Iterate/create(f, x)` — backs `clojure.core/iterate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_iterate_create_2(f_bits: u64, x_bits: u64) -> u64 {
    alloc_seq2(heap_type_ids().iterate_seq, f_bits, x_bits)
}

/// `clojure.lang.Cycle/create(seq)` — backs `clojure.core/cycle`. The
/// caller passes `(seq coll)`; nil in → nil out (empty cycle).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_cycle_create_1(seq_bits: u64) -> u64 {
    if matches!(nanbox_tag(seq_bits), Some(TAG_NIL)) {
        return nanbox_nil();
    }
    alloc_seq2(heap_type_ids().cycle_seq, seq_bits, seq_bits)
}

/// `(f acc x)` as a GC-causing primitive. The user fn behind `f_bits` may
/// allocate, so `acc`/`x` arrive rooted and the result is returned as a
/// fresh `Raw` (the caller re-roots it).
#[inline]
fn heap_invoke_2<'h>(heap: &'h mut Heap, f_bits: u64, acc: Rooted<'_, ()>, x: Rooted<'_, ()>) -> Raw<'h> {
    let acc_bits = acc.get_raw(&*heap).bits();
    let x_bits = x.get_raw(&*heap).bits();
    let out = unsafe { cljvm_rt_invoke_2(f_bits, acc_bits, x_bits) };
    Raw::from_bits(&*heap, out)
}

/// Type-safe core of `RT.cons`. Mirrors Java's `RT.cons`: if `seq` isn't
/// already an ISeq (Cons or nil), coerce it via `RT.seq` first, then
/// allocate the Cons cell. `x` and `seq` arrive rooted; the moving GC that
/// `heap_seq`/`heap_alloc` may trigger updates their slots in place.
fn cons_inner<'h>(
    heap: &'h mut Heap,
    scope: &RootScope<'_>,
    x: Rooted<'_, ()>,
    seq: Rooted<'_, ()>,
) -> Raw<'h> {
    let ids = heap_type_ids();

    // Mirror Java's RT.cons: an ISeq tail (Cons, LazySeq, Repeat, Iterate,
    // Cycle) is stored AS-IS — coercing a LazySeq here would force the
    // whole chain eagerly, which turns `(cons x (map f infinite))` into a
    // stack overflow. Only non-ISeq collections (vectors, maps, strings,
    // sets, …) go through RT.seq.
    let needs_coerce = match nanbox_tag(seq.get_raw(&*heap).bits()) {
        Some(TAG_NIL) => false,
        Some(TAG_PTR) => {
            let p = nanbox_payload(seq.get_raw(&*heap).bits()) as *const u8;
            if p.is_null() {
                false
            } else {
                let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                tid != ids.cons
                    && tid != ids.lazy_seq
                    && tid != ids.repeat_seq
                    && tid != ids.iterate_seq
                    && tid != ids.cycle_seq
            }
        }
        _ => panic!(
            "clojure-jvm: cljvm_rt_cons: second arg must be a collection or nil, \
             got bits 0x{:x}",
            seq.get_raw(&*heap).bits()
        ),
    };
    if needs_coerce {
        // `x` stays valid across this allocation because it is rooted; the
        // borrow checker would reject the call otherwise.
        let coerced = heap_seq(heap, seq);
        seq.set_raw(coerced);
    }

    // Allocate the Cons cell; `x` and `seq` are rooted across it.
    let cell = heap_alloc(heap, ids.cons as u64, 0).root(scope);

    // Field writes are plain stores, not GC points. Re-read the rooted
    // values through `heap` so any relocation that happened above is seen.
    let cell_bits = cell.get_raw(&*heap).bits();
    trap_forwarded_first_result("cons.write.x", cell_bits, x.get_raw(&*heap).bits());
    trap_forwarded_first_result("cons.write.seq", cell_bits, seq.get_raw(&*heap).bits());
    let ptr = nanbox_payload(cell_bits) as *mut u8;
    unsafe {
        write_slot(ptr, 8, x.get_raw(&*heap));
        write_slot(ptr, 16, seq.get_raw(&*heap));
        write_slot(ptr, 24, Raw::from_bits(&*heap, nanbox_nil()));
    }
    cell.get_raw(&*heap)
}

/// `clojure.lang.RT.conj(Object coll, Object x)` — add `x` to `coll`.
/// Java dispatches on `IPersistentCollection.cons`. We cover what
/// the bootstrap of `clojure.core` exercises:
///   * nil      → `(x)` (one-element list)
///   * Cons     → `(x . coll)` (cons-prepend)
///   * Map+Map  → merge x's entries into coll
///   * Map+nil  → coll (Java: conj nil onto a map is a no-op)
/// Other receivers panic loudly until they're actually needed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_conj(coll_bits: u64, x_bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => unsafe { cljvm_rt_cons(x_bits, nanbox_nil()) },
        Some(TAG_PTR) => {
            let raw = nanbox_payload(coll_bits) as *const u8;
            if raw.is_null() {
                return unsafe { cljvm_rt_cons(x_bits, nanbox_nil()) };
            }
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.cons {
                return unsafe { cljvm_rt_cons(x_bits, coll_bits) };
            }
            if type_id == ids.vector {
                // (conj <vec> x) → append x at the end. Allocate a
                // fresh Vector heap cell with the existing items
                // copied + the new x at the tail.
                //
                // Snapshot existing items + root x BEFORE allocating
                // the new Vector — gc_alloc_thunk can move both the
                // source vector and any heap pointers in `x_bits`.
                // Without rooting, post-alloc reads off `raw` would
                // dereference stale memory and silently copy garbage
                // bits into the new cell. (This is the same class of
                // bug as the original vector→seq SIGBUS at form 46.)
                let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
                return dynobj::roots::gc_enter(n + 2, |heap, scope| {
                    let item_roots: Vec<dynobj::roots::Rooted<()>> = (0..n)
                        .map(|i| {
                            let v = unsafe { read_slot(&*heap, raw, 16 + i * 8) };
                            v.root(scope)
                        })
                        .collect();
                    let x_root = scope.root::<()>(x_bits);
                    let cell = heap_alloc(heap, ids.vector as u64, (n + 1) as u64).root(scope);
                    let cell_bits = cell.get_raw(&*heap).bits();
                    let new_ptr = nanbox_payload(cell_bits) as *mut u8;
                    for i in 0..n {
                        unsafe {
                            write_slot(new_ptr, 16 + i * 8, item_roots[i].get_raw(&*heap));
                        }
                    }
                    unsafe {
                        write_slot(new_ptr, 16 + n * 8, x_root.get_raw(&*heap));
                    }
                    cell_bits
                });
            }
            if type_id == ids.map {
                // (conj <map> nil) → <map>.
                if matches!(nanbox_tag(x_bits), Some(TAG_NIL)) {
                    return coll_bits;
                }
                // Decode receiver Arc to merge into. Use a clone we own
                // so we can fold sequential assocs.
                let recv_arc_ptr = unsafe { raw.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_hash_map::PersistentHashMap;
                unsafe { Arc::increment_strong_count(recv_arc_ptr) };
                let mut acc = unsafe { Arc::from_raw(recv_arc_ptr) };
                let x_obj = any_bits_to_object(x_bits, ids);
                let bare_x = x_obj.peel_meta_ref();
                match bare_x {
                    Object::Map(other) => {
                        for (k, v) in other.iter() {
                            acc = acc.assoc(k, v);
                        }
                    }
                    Object::Vector(v) if v.count() == 2 => {
                        // Java treats a 2-element [k v] vector as a MapEntry.
                        acc = acc.assoc(v.nth(0), v.nth(1));
                    }
                    other => panic!(
                        "clojure-jvm: RT.conj — can't conj {other:?} onto a Map \
                         (need Map / MapEntry / [k v] vector)"
                    ),
                }
                let raw_arc = Arc::as_ptr(&acc) as u64;
                crate::lang::compiler::with_active_session_root_map(acc);
                let new_raw = dynlang::gc::gc_alloc_thunk(ids.map as u64, 0);
                let new_ptr = new_raw as *mut u8;
                if new_ptr.is_null() {
                    panic!("clojure-jvm: RT.conj: gc_alloc returned null for Map");
                }
                unsafe {
                    write_raw_word(new_ptr, 8, raw_arc);
                }
                return nanbox_ptr(new_raw);
            }
            if type_id == ids.set {
                // (conj <set> x) → a new set with x added (Arc-shared).
                let recv_arc_ptr = unsafe { raw.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_hash_set::PersistentHashSet;
                unsafe { Arc::increment_strong_count(recv_arc_ptr) };
                let recv = unsafe { Arc::from_raw(recv_arc_ptr) };
                let x_obj = any_bits_to_object(x_bits, ids);
                let acc = recv.cons(x_obj);
                let raw_arc = Arc::as_ptr(&acc) as u64;
                crate::lang::compiler::with_active_session_root_set(acc);
                let new_raw = dynlang::gc::gc_alloc_thunk(ids.set as u64, 0);
                let new_ptr = new_raw as *mut u8;
                if new_ptr.is_null() {
                    panic!("clojure-jvm: RT.conj: gc_alloc returned null for Set");
                }
                unsafe {
                    write_raw_word(new_ptr, 8, raw_arc);
                }
                return nanbox_ptr(new_raw);
            }
            if type_id == ids.tree_map {
                // (conj <sorted-map> x): x is a Map (merge) or a [k v]/MapEntry.
                if matches!(nanbox_tag(x_bits), Some(TAG_NIL)) {
                    return coll_bits;
                }
                let mut acc = unsafe { decode_tree_map(raw) };
                match any_bits_to_object(x_bits, ids).peel_meta_ref() {
                    Object::Map(other) => {
                        for (k, v) in other.iter() {
                            acc = acc.assoc(k, v);
                        }
                    }
                    Object::Vector(v) if v.count() == 2 => {
                        acc = acc.assoc(v.nth(0), v.nth(1));
                    }
                    other => panic!(
                        "clojure-jvm: RT.conj onto sorted-map needs Map / [k v], got {other:?}"
                    ),
                }
                return alloc_tree_map_cell(acc);
            }
            if type_id == ids.tree_set {
                // (conj <sorted-set> x) → add x.
                let recv = unsafe { decode_tree_set(raw) };
                let x_obj = any_bits_to_object(x_bits, ids);
                return alloc_tree_set_cell(recv.cons(x_obj));
            }
            eprintln!(
                "[cljvm-stub] RT.conj — receiver type_id {type_id} not yet \
                 implemented, returning nil"
            );
            nanbox_nil()
        }
        _ => {
            eprintln!(
                "[cljvm-stub] RT.conj — first arg must be a collection or \
                 nil, got NanBox bits 0x{coll_bits:x} — returning nil"
            );
            nanbox_nil()
        }
    }
}

/// `(assoc <vector> idx val)` — a new vector with slot `idx` replaced
/// (`idx == count` appends, matching Clojure). Rooted-alloc pattern mirrors
/// `cljvm_rt_conj`'s vector path: snapshot items + val into roots BEFORE the
/// `heap_alloc` that may relocate them.
unsafe fn vector_assoc(vec_bits: u64, key_bits: u64, val_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let raw = nanbox_payload(vec_bits) as *const u8;
    let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
    let idx = arg_to_i64(key_bits);
    if idx < 0 || idx as usize > n {
        panic!("clojure-jvm: (assoc vec idx val): index {idx} out of bounds (count {n})");
    }
    let idx = idx as usize;
    let new_n = if idx == n { n + 1 } else { n };
    dynobj::roots::gc_enter(n + 2, |heap, scope| {
        let item_roots: Vec<dynobj::roots::Rooted<()>> = (0..n)
            .map(|i| {
                let v = unsafe { read_slot(&*heap, raw, 16 + i * 8) };
                v.root(scope)
            })
            .collect();
        let val_root = scope.root::<()>(val_bits);
        let cell = heap_alloc(heap, ids.vector as u64, new_n as u64).root(scope);
        let cell_bits = cell.get_raw(&*heap).bits();
        let new_ptr = nanbox_payload(cell_bits) as *mut u8;
        for i in 0..n {
            let src = if i == idx {
                val_root.get_raw(&*heap)
            } else {
                item_roots[i].get_raw(&*heap)
            };
            unsafe { write_slot(new_ptr, 16 + i * 8, src) };
        }
        if idx == n {
            unsafe { write_slot(new_ptr, 16 + n * 8, val_root.get_raw(&*heap)) };
        }
        cell_bits
    })
}

/// Allocate a `tree_map` heap cell pointing at `m` (rooted on the Session).
fn alloc_tree_map_cell(m: Arc<crate::lang::persistent_tree_map::PersistentTreeMap>) -> u64 {
    let ids = heap_type_ids();
    let raw_arc = Arc::as_ptr(&m) as u64;
    crate::lang::compiler::with_active_session_root_tree_map(m);
    let new_raw = dynlang::gc::gc_alloc_thunk(ids.tree_map as u64, 0);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: alloc_tree_map_cell: gc_alloc returned null");
    }
    unsafe { new_ptr.add(8).cast::<u64>().write_unaligned(raw_arc) };
    nanbox_ptr(new_raw)
}

/// Allocate a `tree_set` heap cell pointing at `s` (rooted on the Session).
fn alloc_tree_set_cell(s: Arc<crate::lang::persistent_tree_set::PersistentTreeSet>) -> u64 {
    let ids = heap_type_ids();
    let raw_arc = Arc::as_ptr(&s) as u64;
    crate::lang::compiler::with_active_session_root_tree_set(s);
    let new_raw = dynlang::gc::gc_alloc_thunk(ids.tree_set as u64, 0);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: alloc_tree_set_cell: gc_alloc returned null");
    }
    unsafe { new_ptr.add(8).cast::<u64>().write_unaligned(raw_arc) };
    nanbox_ptr(new_raw)
}

/// Decode a `tree_map` cell's `Arc<PersistentTreeMap>` (incrementing the
/// strong count so the returned Arc is owned).
unsafe fn decode_tree_map(raw: *const u8) -> Arc<crate::lang::persistent_tree_map::PersistentTreeMap> {
    let arc_ptr = unsafe { raw.add(8).cast::<u64>().read_unaligned() }
        as *const crate::lang::persistent_tree_map::PersistentTreeMap;
    unsafe { Arc::increment_strong_count(arc_ptr) };
    unsafe { Arc::from_raw(arc_ptr) }
}
/// Decode a `tree_set` cell's `Arc<PersistentTreeSet>`.
unsafe fn decode_tree_set(raw: *const u8) -> Arc<crate::lang::persistent_tree_set::PersistentTreeSet> {
    let arc_ptr = unsafe { raw.add(8).cast::<u64>().read_unaligned() }
        as *const crate::lang::persistent_tree_set::PersistentTreeSet;
    unsafe { Arc::increment_strong_count(arc_ptr) };
    unsafe { Arc::from_raw(arc_ptr) }
}

/// `clojure.lang.RT.assoc(Object coll, Object key, Object val)` —
/// associate `key → val` in `coll` and return the resulting collection.
/// Covers nil → singleton map, Map → assoc'd map, Vector → index replace.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_assoc(coll_bits: u64, key_bits: u64, val_bits: u64) -> u64 {
    let ids = heap_type_ids();
    // Vector receiver: integer-index replace/append. Handle before the
    // map-entry decode below (which is map-specific).
    if let Some(TAG_PTR) = nanbox_tag(coll_bits) {
        let raw = nanbox_payload(coll_bits) as *const u8;
        if !raw.is_null() {
            let tid = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.vector {
                return unsafe { vector_assoc(coll_bits, key_bits, val_bits) };
            }
            if tid == ids.tree_map {
                let arc = unsafe { decode_tree_map(raw) };
                let k = any_bits_to_object(key_bits, ids);
                let v = any_bits_to_object(val_bits, ids);
                return alloc_tree_map_cell(arc.assoc(k, v));
            }
        }
    }
    // Decode key + val to host-side Objects so the Arc<PersistentHashMap>
    // stores them by-value. (Maps don't put their entries on the GC heap
    // — the Arc owns a Vec<(Object, Object)>.)
    let key_obj = any_bits_to_object(key_bits, ids);
    let val_obj = any_bits_to_object(val_bits, ids);

    let new_map: Arc<crate::lang::persistent_hash_map::PersistentHashMap> =
        match nanbox_tag(coll_bits) {
            Some(TAG_NIL) => crate::lang::persistent_hash_map::PersistentHashMap::create_pairs(
                vec![(key_obj, val_obj)],
            ),
            Some(TAG_PTR) => {
                let raw = nanbox_payload(coll_bits) as *const u8;
                if raw.is_null() {
                    crate::lang::persistent_hash_map::PersistentHashMap::create_pairs(vec![(
                        key_obj, val_obj,
                    )])
                } else {
                    let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
                    if type_id != ids.map {
                        panic!(
                            "clojure-jvm: RT.assoc — receiver type_id {type_id} not yet \
                             implemented (extend cljvm_rt_assoc to handle this type)"
                        );
                    }
                    // Map heap cell: Raw64 at offset 8 holds Arc::as_ptr.
                    let arc_ptr = unsafe { raw.add(8).cast::<u64>().read_unaligned() }
                        as *const crate::lang::persistent_hash_map::PersistentHashMap;
                    unsafe { Arc::increment_strong_count(arc_ptr) };
                    let arc = unsafe { Arc::from_raw(arc_ptr) };
                    arc.assoc(key_obj, val_obj)
                }
            }
            _ => panic!(
                "clojure-jvm: RT.assoc — first arg must be a map or nil, got NanBox \
                 bits 0x{coll_bits:x}"
            ),
        };

    // Allocate a fresh Map heap cell, point its Raw64 at the new Arc,
    // and root the Arc on the active Session so it outlives the heap cell.
    let raw_arc = Arc::as_ptr(&new_map) as u64;
    crate::lang::compiler::with_active_session_root_map(new_map);
    let new_raw = dynlang::gc::gc_alloc_thunk(ids.map as u64, 0);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: RT.assoc: gc_alloc returned null for Map");
    }
    unsafe {
        new_ptr.add(8).cast::<u64>().write_unaligned(raw_arc);
    }
    nanbox_encode(TAG_PTR, new_raw & PAYLOAD_MASK)
}

/// `clojure.lang.RT.count(Object coll)` — return the element count
/// of `coll` as a NanBox-encoded Long. Java's RT.count dispatches via
/// `Counted` / `IPersistentCollection`; we cover the receivers our
/// bootstrap touches: nil/Cons/Vector/Map. Other types panic loudly.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_count(bits: u64) -> u64 {
    let ids = heap_type_ids();
    let n: i64 = match nanbox_tag(bits) {
        Some(TAG_NIL) => 0,
        Some(TAG_PTR) => {
            let ptr = nanbox_payload(bits) as *const u8;
            if ptr.is_null() {
                0
            } else {
                let type_id = unsafe { ptr.cast::<u16>().read_unaligned() } as usize;
                if type_id == ids.cons || type_id == ids.lazy_seq {
                    // Generic seq walk: `seq` normalizes/forces the head and
                    // `next` forces each subsequent step, so this counts
                    // both plain cons chains and lazy seqs (e.g. the result
                    // of `map`/`filter`) without assuming the rest is a cons.
                    let mut count: i64 = 0;
                    let mut cur = unsafe { cljvm_rt_seq(bits) };
                    loop {
                        match nanbox_tag(cur) {
                            Some(TAG_NIL) => break,
                            _ => {
                                let p = nanbox_payload(cur) as *const u8;
                                if p.is_null() {
                                    break;
                                }
                                count += 1;
                                cur = unsafe { cljvm_rt_next(cur) };
                            }
                        }
                    }
                    count
                } else if type_id == ids.repeat_seq {
                    let c = unsafe { ptr.add(16).cast::<u64>().read_unaligned() } as i64;
                    if c < 0 {
                        panic!("clojure-jvm: count on an infinite (repeat x) never terminates");
                    }
                    c
                } else if type_id == ids.iterate_seq || type_id == ids.cycle_seq {
                    panic!("clojure-jvm: count on an infinite seq (iterate/cycle) never terminates");
                } else if type_id == ids.vector {
                    // Vector layout: Header(8) + count word(8) + items.
                    let c = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
                    c as i64
                } else if type_id == ids.map {
                    // Map: Raw64 holds the Arc<PersistentHashMap> pointer.
                    let arc_ptr = unsafe { ptr.add(8).cast::<u64>().read_unaligned() }
                        as *const crate::lang::persistent_hash_map::PersistentHashMap;
                    unsafe { (*arc_ptr).count() as i64 }
                } else if type_id == ids.set {
                    let arc_ptr = unsafe { ptr.add(8).cast::<u64>().read_unaligned() }
                        as *const crate::lang::persistent_hash_set::PersistentHashSet;
                    unsafe { (*arc_ptr).count() as i64 }
                } else if type_id == ids.tree_map {
                    let arc = unsafe { decode_tree_map(ptr) };
                    arc.count() as i64
                } else if type_id == ids.tree_set {
                    let arc = unsafe { decode_tree_set(ptr) };
                    arc.count() as i64
                } else if type_id == ids.string {
                    let c = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
                    c as i64
                } else {
                    eprintln!(
                        "[cljvm-stub] RT.count: heap type_id {type_id} not \
                         yet supported — 0"
                    );
                    0
                }
            }
        }
        _ => {
            eprintln!(
                "[cljvm-stub] RT.count: receiver bits 0x{bits:x} not \
                 countable — 0"
            );
            0
        }
    };
    box_long(n)
}

/// `clojure.lang.RT.subvec(Object v, int start, int end)` — return a
/// sub-vector covering `[start, end)`. Allocates a fresh Vector heap
/// cell whose items are copied (NanBox-bit-for-bit) from the source
/// vector — heap pointers in the source remain valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_subvec(v_bits: u64, start_bits: u64, end_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let start = arg_to_i64(start_bits) as usize;
    let end = arg_to_i64(end_bits) as usize;
    if end < start {
        panic!("clojure-jvm: RT.subvec — end ({end}) < start ({start})");
    }
    // Validate the source is a Vector heap cell.
    let src_raw = match nanbox_tag(v_bits) {
        Some(TAG_PTR) => nanbox_payload(v_bits) as *const u8,
        _ => panic!("clojure-jvm: RT.subvec — first arg must be a Vector"),
    };
    if src_raw.is_null() {
        panic!("clojure-jvm: RT.subvec — null Vector pointer");
    }
    let src_type = unsafe { src_raw.cast::<u16>().read_unaligned() } as usize;
    if src_type != ids.vector {
        panic!("clojure-jvm: RT.subvec — first arg type_id {src_type} is not a Vector");
    }
    let src_count = unsafe { src_raw.add(8).cast::<u64>().read_unaligned() } as usize;
    if end > src_count {
        panic!("clojure-jvm: RT.subvec — end ({end}) > count ({src_count})");
    }

    let n = end - start;
    // Snapshot the source slice BEFORE allocating — the alloc can relocate
    // both the source vector (`src_raw` would dangle) and the element
    // targets. Root the snapshot, then write the relocated values back.
    let slice: Vec<u64> = (0..n)
        .map(|i| unsafe { src_raw.add(16 + (start + i) * 8).cast::<u64>().read_unaligned() })
        .collect();
    dynobj::roots::gc_enter(n + 1, |heap, scope| {
        let roots: Vec<dynobj::roots::Rooted<()>> =
            slice.iter().map(|&b| scope.root::<()>(b)).collect();
        let cell = heap_alloc(heap, ids.vector as u64, n as u64).root(scope);
        let cell_bits = cell.get_raw(&*heap).bits();
        let new_ptr = nanbox_payload(cell_bits) as *mut u8;
        for (i, r) in roots.iter().enumerate() {
            unsafe {
                write_slot(new_ptr, 16 + i * 8, r.get_raw(&*heap));
            }
        }
        cell_bits
    })
}

/// `clojure.lang.PersistentHashSet.create(Object coll)` — build a set
/// from a sequence of items. Upstream's `(defn hash-set [& keys] (PHS/create keys))`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_phs_create(coll_bits: u64) -> u64 {
    let ids = heap_type_ids();
    // Walk the coll into a Vec of NanBox bits.
    let mut item_bits: Vec<u64> = Vec::new();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => {}
        Some(TAG_PTR) => {
            let raw = nanbox_payload(coll_bits) as *const u8;
            if !raw.is_null() {
                let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
                if type_id == ids.cons || type_id == ids.lazy_seq {
                    // Generic GC-safe walk — cons tails may be
                    // unforced LazySeqs (RT.cons stores ISeq tails
                    // as-is); the old raw walk truncated there.
                    item_bits = unsafe { seq_to_items(coll_bits) };
                } else if type_id == ids.vector {
                    let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
                    item_bits.reserve(n);
                    for i in 0..n {
                        item_bits
                            .push(unsafe { raw.add(16 + i * 8).cast::<u64>().read_unaligned() });
                    }
                } else {
                    panic!(
                        "clojure-jvm: PersistentHashSet/create — receiver \
                         type_id {type_id} not yet supported"
                    );
                }
            }
        }
        _ => panic!(
            "clojure-jvm: PersistentHashSet/create — first arg must be \
             a collection or nil, got bits 0x{coll_bits:x}"
        ),
    }
    let items: Vec<Object> = item_bits
        .into_iter()
        .map(|b| any_bits_to_object(b, ids))
        .collect();
    let s = crate::lang::persistent_hash_set::PersistentHashSet::create(items);
    let raw_arc = std::sync::Arc::as_ptr(&s) as u64;
    crate::lang::compiler::with_active_session_root_set(s);
    let new_raw = dynlang::gc::gc_alloc_thunk(ids.set as u64, 0);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: PHS/create: gc_alloc returned null");
    }
    unsafe {
        new_ptr.add(8).cast::<u64>().write_unaligned(raw_arc);
    }
    nanbox_ptr(new_raw)
}

/// `clojure.lang.PersistentTreeMap.create(Object coll)` — sorted-map
/// counterpart of `cljvm_phm_create`. Walks `coll` (a cons-list or
/// vector) into a flat sequence of alternating key/value items and
/// builds an `Arc<PersistentTreeMap>` whose keys are kept sorted by
/// `compare_objects`. The Arc is rooted on the active Session's
/// `_tree_maps` and a fresh `clojure.lang.PersistentTreeMap` heap cell
/// is returned with the Arc raw pointer in its `arc_ptr` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_ptm_create(coll_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let mut items: Vec<u64> = Vec::new();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => {}
        Some(TAG_PTR) => {
            let raw = nanbox_payload(coll_bits) as *const u8;
            if !raw.is_null() {
                let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
                if type_id == ids.cons || type_id == ids.lazy_seq {
                    // Generic GC-safe walk — cons tails may be
                    // unforced LazySeqs (RT.cons stores ISeq tails
                    // as-is); the old raw walk truncated there.
                    items = unsafe { seq_to_items(coll_bits) };
                } else if type_id == ids.vector {
                    let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
                    items.reserve(n);
                    for i in 0..n {
                        items.push(unsafe { raw.add(16 + i * 8).cast::<u64>().read_unaligned() });
                    }
                } else {
                    panic!(
                        "clojure-jvm: PersistentTreeMap/create — receiver \
                         type_id {type_id} not yet supported"
                    );
                }
            }
        }
        _ => panic!(
            "clojure-jvm: PersistentTreeMap/create — first arg must be \
             a collection or nil, got bits 0x{coll_bits:x}"
        ),
    }
    if items.len() % 2 != 0 {
        panic!(
            "clojure-jvm: IllegalArgumentException — No value supplied for \
             key in PersistentTreeMap/create ({} items)",
            items.len()
        );
    }
    let mut flat: Vec<Object> = Vec::with_capacity(items.len());
    for b in items {
        flat.push(any_bits_to_object(b, ids));
    }
    let m = crate::lang::persistent_tree_map::PersistentTreeMap::create_flat(flat);
    let raw_arc = std::sync::Arc::as_ptr(&m) as u64;
    crate::lang::compiler::with_active_session_root_tree_map(m);
    let new_raw = dynlang::gc::gc_alloc_thunk(ids.tree_map as u64, 0);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: PTM/create: gc_alloc returned null");
    }
    unsafe {
        new_ptr.add(8).cast::<u64>().write_unaligned(raw_arc);
    }
    nanbox_ptr(new_raw)
}

/// `clojure.lang.PersistentTreeSet.create(Object coll)` — natural-order
/// sorted-set builder. Walks `coll` (cons-list or vector) into items
/// and builds a `PersistentTreeSet` keyed by `compare_objects`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_pts_create(coll_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let item_bits = unsafe { walk_coll_to_bits(coll_bits, ids, "PersistentTreeSet/create") };
    let items: Vec<Object> = item_bits
        .into_iter()
        .map(|b| any_bits_to_object(b, ids))
        .collect();
    let s = crate::lang::persistent_tree_set::PersistentTreeSet::create(items);
    let raw_arc = std::sync::Arc::as_ptr(&s) as u64;
    crate::lang::compiler::with_active_session_root_tree_set(s);
    let new_raw = dynlang::gc::gc_alloc_thunk(ids.tree_set as u64, 0);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: PTS/create: gc_alloc returned null");
    }
    unsafe {
        new_ptr.add(8).cast::<u64>().write_unaligned(raw_arc);
    }
    nanbox_ptr(new_raw)
}

/// `clojure.lang.PersistentTreeSet.create(Comparator c, Object coll)`.
/// Uses the user-provided comparator (a fn) for ordering, dispatched via
/// `cljvm_rt_invoke_2` like sorted-map's arity-2 form.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_pts_create_cmp(cmp_bits: u64, coll_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let item_bits = unsafe { walk_coll_to_bits(coll_bits, ids, "PersistentTreeSet/create") };
    // Insertion-sort by comparator.
    let mut sorted_bits: Vec<u64> = Vec::with_capacity(item_bits.len());
    for kb in item_bits {
        let mut placed = false;
        let mut idx = sorted_bits.len();
        for i in 0..sorted_bits.len() {
            let cmp_ret = unsafe { cljvm_rt_invoke_2(cmp_bits, kb, sorted_bits[i]) };
            match decode_cmp_ret(cmp_ret) {
                Ordering::Equal => {
                    placed = true;
                    break;
                }
                Ordering::Less => {
                    idx = i;
                    break;
                }
                Ordering::Greater => {}
            }
        }
        if !placed {
            sorted_bits.insert(idx, kb);
        }
    }
    let items: Vec<Object> = sorted_bits
        .into_iter()
        .map(|b| any_bits_to_object(b, ids))
        .collect();
    let s = crate::lang::persistent_tree_set::PersistentTreeSet::create_cmp(
        items,
        |_a, _b| Ordering::Greater, // preserve insertion order; no dups
    );
    let raw_arc = std::sync::Arc::as_ptr(&s) as u64;
    crate::lang::compiler::with_active_session_root_tree_set(s);
    let new_raw = dynlang::gc::gc_alloc_thunk(ids.tree_set as u64, 0);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: PTS/create (cmp): gc_alloc returned null");
    }
    unsafe {
        new_ptr.add(8).cast::<u64>().write_unaligned(raw_arc);
    }
    nanbox_ptr(new_raw)
}

/// Decode a coll (cons or vector) into a Vec of NanBox bits. Used by all
/// the `PersistentX/create` builders. `caller` shows up in panic messages.
unsafe fn walk_coll_to_bits(coll_bits: u64, ids: HeapTypeIds, caller: &'static str) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::new();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => {}
        Some(TAG_PTR) => {
            let raw = nanbox_payload(coll_bits) as *const u8;
            if !raw.is_null() {
                let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
                if type_id == ids.cons || type_id == ids.lazy_seq {
                    // Generic GC-safe walk — cons tails may be
                    // unforced LazySeqs (RT.cons stores ISeq tails
                    // as-is); the old raw walk truncated there.
                    out = unsafe { seq_to_items(coll_bits) };
                } else if type_id == ids.vector {
                    let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
                    out.reserve(n);
                    for i in 0..n {
                        out.push(unsafe { raw.add(16 + i * 8).cast::<u64>().read_unaligned() });
                    }
                } else {
                    panic!("clojure-jvm: {caller} — receiver type_id {type_id} not yet supported");
                }
            }
        }
        _ => panic!(
            "clojure-jvm: {caller} — coll arg must be a collection or nil, got bits 0x{coll_bits:x}"
        ),
    }
    out
}

/// `clojure.lang.PersistentTreeMap.create(Comparator c, Object coll)` —
/// arity-2 sorted-map builder. Uses `c` to order keys: each insertion
/// invokes `(c k1 k2)` via the standard 2-arg dispatch and treats the
/// returned Long as `<0 / 0 / >0` per Java's `Comparator` contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_ptm_create_cmp(cmp_bits: u64, coll_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let mut items: Vec<u64> = Vec::new();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => {}
        Some(TAG_PTR) => {
            let raw = nanbox_payload(coll_bits) as *const u8;
            if !raw.is_null() {
                let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
                if type_id == ids.cons || type_id == ids.lazy_seq {
                    // Generic GC-safe walk — cons tails may be
                    // unforced LazySeqs (RT.cons stores ISeq tails
                    // as-is); the old raw walk truncated there.
                    items = unsafe { seq_to_items(coll_bits) };
                } else if type_id == ids.vector {
                    let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
                    items.reserve(n);
                    for i in 0..n {
                        items.push(unsafe { raw.add(16 + i * 8).cast::<u64>().read_unaligned() });
                    }
                } else {
                    panic!(
                        "clojure-jvm: PersistentTreeMap/create — receiver \
                         type_id {type_id} not yet supported"
                    );
                }
            }
        }
        _ => panic!(
            "clojure-jvm: PersistentTreeMap/create — second arg must be \
             a collection or nil, got bits 0x{coll_bits:x}"
        ),
    }
    if items.len() % 2 != 0 {
        panic!(
            "clojure-jvm: IllegalArgumentException — No value supplied for \
             key in PersistentTreeMap/create ({} items)",
            items.len()
        );
    }
    // Sort key insertion uses the user-provided comparator. We keep
    // ordering in NanBox space by indexing into a parallel Vec<u64> for
    // keys+vals; pair Objects are decoded on the fly for `assoc`-style
    // overwrite checks via the comparator's 0-return.
    let mut pair_bits: Vec<(u64, u64)> = Vec::with_capacity(items.len() / 2);
    {
        let mut it = items.into_iter();
        while let (Some(kb), Some(vb)) = (it.next(), it.next()) {
            pair_bits.push((kb, vb));
        }
    }
    // Insertion sort using the comparator extern.
    let mut sorted: Vec<(u64, u64)> = Vec::with_capacity(pair_bits.len());
    for (kb, vb) in pair_bits.into_iter() {
        let mut placed = false;
        let mut idx = sorted.len();
        for i in 0..sorted.len() {
            let cmp_ret = unsafe { cljvm_rt_invoke_2(cmp_bits, kb, sorted[i].0) };
            let ord = decode_cmp_ret(cmp_ret);
            match ord {
                Ordering::Equal => {
                    sorted[i].1 = vb;
                    placed = true;
                    break;
                }
                Ordering::Less => {
                    idx = i;
                    break;
                }
                Ordering::Greater => {}
            }
        }
        if !placed {
            sorted.insert(idx, (kb, vb));
        }
    }
    // Now decode pairs into Objects for the host-side Arc storage.
    let mut flat: Vec<Object> = Vec::with_capacity(sorted.len() * 2);
    for (kb, vb) in sorted.into_iter() {
        flat.push(any_bits_to_object(kb, ids));
        flat.push(any_bits_to_object(vb, ids));
    }
    // Bypass the comparator at host-storage time: insertion already
    // produced the canonical order. Use `create_flat_cmp` with a
    // monotone comparator that preserves insertion order so
    // duplicate-overwrite semantics agree with what we just decided.
    let m = crate::lang::persistent_tree_map::PersistentTreeMap::create_flat_cmp(
        flat,
        |_a, _b| Ordering::Greater, // keep insertion order; no duplicates by construction
    );
    let raw_arc = std::sync::Arc::as_ptr(&m) as u64;
    crate::lang::compiler::with_active_session_root_tree_map(m);
    let new_raw = dynlang::gc::gc_alloc_thunk(ids.tree_map as u64, 0);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: PTM/create (cmp): gc_alloc returned null");
    }
    unsafe {
        new_ptr.add(8).cast::<u64>().write_unaligned(raw_arc);
    }
    nanbox_ptr(new_raw)
}

/// Decode a comparator's NanBox return value to an `Ordering`. Accepts
/// Long (preferred) and Double; panics on anything else (matches
/// Comparator's `int` contract).
fn decode_cmp_ret(bits: u64) -> Ordering {
    let ids = heap_type_ids();
    let obj = any_bits_to_object(bits, ids);
    match obj {
        Object::Long(n) => n.cmp(&0),
        Object::Double(x) => x.partial_cmp(&0.0).unwrap_or(Ordering::Equal),
        other => panic!(
            "clojure-jvm: comparator returned non-numeric {other:?} \
             (Java Comparator must return int)"
        ),
    }
}

/// `clojure.lang.PersistentHashMap.create(Object coll)` — build a map
/// from a sequence of alternating key/value items. Upstream's
/// `(defn hash-map [& keyvals] (. PHM (create keyvals)))` calls this
/// with a cons-list of even length. We walk the list and accumulate
/// `assoc` calls onto an empty map. Roots accumulator across each
/// alloc so the in-progress `Arc<PHM>` survives nursery collections.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_phm_create(coll_bits: u64) -> u64 {
    let ids = heap_type_ids();
    // Decode coll into a host-side vec of NanBox bits. Walk through
    // cons cells (or vector) like LPV/create.
    let mut items: Vec<u64> = Vec::new();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => {}
        Some(TAG_PTR) => {
            let raw = nanbox_payload(coll_bits) as *const u8;
            if !raw.is_null() {
                let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
                if type_id == ids.cons || type_id == ids.lazy_seq {
                    // Generic GC-safe walk — cons tails may be
                    // unforced LazySeqs (RT.cons stores ISeq tails
                    // as-is); the old raw walk truncated there.
                    items = unsafe { seq_to_items(coll_bits) };
                } else if type_id == ids.vector {
                    let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
                    items.reserve(n);
                    for i in 0..n {
                        items.push(unsafe { raw.add(16 + i * 8).cast::<u64>().read_unaligned() });
                    }
                } else {
                    panic!(
                        "clojure-jvm: PersistentHashMap/create — receiver \
                         type_id {type_id} not yet supported"
                    );
                }
            }
        }
        _ => panic!(
            "clojure-jvm: PersistentHashMap/create — first arg must be \
             a collection or nil, got bits 0x{coll_bits:x}"
        ),
    }
    if items.len() % 2 != 0 {
        panic!(
            "clojure-jvm: IllegalArgumentException — No value supplied for \
             key in PersistentHashMap/create ({} items)",
            items.len()
        );
    }
    // Decode each k/v to Object. Build the Arc<PHM> via create_pairs.
    let mut pairs: Vec<(Object, Object)> = Vec::with_capacity(items.len() / 2);
    let mut it = items.into_iter();
    while let (Some(k_bits), Some(v_bits)) = (it.next(), it.next()) {
        let k = any_bits_to_object(k_bits, ids);
        let v = any_bits_to_object(v_bits, ids);
        pairs.push((k, v));
    }
    let m = crate::lang::persistent_hash_map::PersistentHashMap::create_pairs(pairs);
    let raw_arc = std::sync::Arc::as_ptr(&m) as u64;
    crate::lang::compiler::with_active_session_root_map(m);
    let new_raw = dynlang::gc::gc_alloc_thunk(ids.map as u64, 0);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: PHM/create: gc_alloc returned null");
    }
    unsafe {
        new_ptr.add(8).cast::<u64>().write_unaligned(raw_arc);
    }
    nanbox_ptr(new_raw)
}

/// Walk any seqable into a Vec of element NanBox bits, GC-safely.
/// `RT.seq` normalizes/forces the head and `RT.next` forces each tail, so
/// this realizes lazy seqs (e.g. `partition-all` segments, `map` output)
/// as well as plain cons chains, strings, maps, sets, and sorted
/// collections — anything `cljvm_rt_seq` accepts. Forcing runs JIT thunks
/// that may allocate and move the heap, so the collected element bits live
/// in a registered chunk buffer that `scan_roots` forwards across GC; the
/// cursor itself follows the same bare-bits walk `cljvm_rt_count` uses
/// (verified under EveryPoint), since each step's bits come fresh from
/// `rt_next`'s return value.
unsafe fn seq_to_items(coll_bits: u64) -> Vec<u64> {
    let buf = std::sync::Arc::new(std::cell::RefCell::new(Vec::<u64>::new()));
    register_chunk_buffer(&buf);
    let mut cur = unsafe { cljvm_rt_seq(coll_bits) };
    loop {
        match nanbox_tag(cur) {
            Some(TAG_PTR) => {
                let p = nanbox_payload(cur) as *const u8;
                if p.is_null() {
                    break;
                }
                let f = unsafe { cljvm_rt_first(cur) };
                // Short borrow: released before the next (possibly
                // GC-triggering) rt_next call, per the chunk-buffer
                // root source's "no cell mutably borrowed at a
                // safepoint" invariant.
                buf.borrow_mut().push(f);
                cur = unsafe { cljvm_rt_next(cur) };
            }
            _ => break,
        }
    }
    let items = buf.borrow().clone();
    deregister_chunk_buffer(&buf);
    items
}

/// `clojure.lang.LazilyPersistentVector.create(Object coll)` — Java's
/// builder produces a lazily-realized vector wrapping the seq; we
/// eagerly walk the seq into a fresh `PersistentVector` heap cell.
/// Sufficient for upstream `(defn vector …)`'s variadic clause:
///   `(. LazilyPersistentVector (create (cons a (cons b … args))))`
/// and for `(vec coll)` over ANY seqable, including lazy seqs (which
/// `seq_to_items` forces tail-by-tail).
/// Items are copied as raw NanBox bits — preserves any heap pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_lpv_create(coll_bits: u64) -> u64 {
    let ids = heap_type_ids();
    // Vector input: copy items directly (no forcing can occur, so the
    // bare reads are safe). Everything else: generic GC-safe seq walk.
    let mut items: Vec<u64> = Vec::new();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => {}
        Some(TAG_PTR) => {
            let raw = nanbox_payload(coll_bits) as *const u8;
            if !raw.is_null() {
                let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
                if type_id == ids.vector {
                    let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
                    items.reserve(n);
                    for i in 0..n {
                        let bits = unsafe { raw.add(16 + i * 8).cast::<u64>().read_unaligned() };
                        items.push(bits);
                    }
                } else {
                    items = unsafe { seq_to_items(coll_bits) };
                }
            }
        }
        _ => panic!(
            "clojure-jvm: LazilyPersistentVector/create — first arg must \
             be a collection or nil, got NanBox bits 0x{coll_bits:x}"
        ),
    }
    // `items` holds source element NanBoxes (GC pointers). Root them all
    // before allocating the result Vector, then write the (possibly
    // relocated) values read back through the heap. heap_alloc's `&mut Heap`
    // makes holding an un-rooted element across the alloc a compile error.
    let n = items.len();
    dynobj::roots::gc_enter(n + 1, |heap, scope| {
        let roots: Vec<dynobj::roots::Rooted<()>> =
            items.iter().map(|&b| scope.root::<()>(b)).collect();
        let cell = heap_alloc(heap, ids.vector as u64, n as u64).root(scope);
        let cell_bits = cell.get_raw(&*heap).bits();
        let new_ptr = nanbox_payload(cell_bits) as *mut u8;
        for (i, r) in roots.iter().enumerate() {
            unsafe {
                write_slot(new_ptr, 16 + i * 8, r.get_raw(&*heap));
            }
        }
        cell_bits
    })
}

/// `clojure.lang.RT.toArray(Object coll)` — copy `coll` into a fresh
/// Object[]. We don't model Java arrays distinctly; we use our own
/// Vector heap cell as the closest analog (callers in Clojure-land
/// almost always pass the result to seq operations that work on
/// vectors). Handles nil → empty vector, Vector → copy, anything else →
/// generic GC-safe seq walk (forces lazy tails — `apply` over a lazy seq
/// routes its argument list through here).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_toArray(coll_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let mut items: Vec<u64> = Vec::new();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => {}
        Some(TAG_PTR) => {
            let raw = nanbox_payload(coll_bits) as *const u8;
            if raw.is_null() {
                // empty
            } else {
                let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
                if type_id == ids.vector {
                    let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
                    items.reserve(n);
                    for i in 0..n {
                        let bits = unsafe { raw.add(16 + i * 8).cast::<u64>().read_unaligned() };
                        items.push(bits);
                    }
                } else {
                    items = unsafe { seq_to_items(coll_bits) };
                }
            }
        }
        _ => panic!(
            "clojure-jvm: RT.toArray — first arg must be a collection or nil, \
             got NanBox bits 0x{coll_bits:x}"
        ),
    }
    // Root every collected element before allocating the result Vector
    // (the alloc can relocate their target objects), then write the values
    // read back through the heap.
    let n = items.len();
    dynobj::roots::gc_enter(n + 1, |heap, scope| {
        let roots: Vec<dynobj::roots::Rooted<()>> =
            items.iter().map(|&b| scope.root::<()>(b)).collect();
        let cell = heap_alloc(heap, ids.vector as u64, n as u64).root(scope);
        let cell_bits = cell.get_raw(&*heap).bits();
        let new_ptr = nanbox_payload(cell_bits) as *mut u8;
        for (i, r) in roots.iter().enumerate() {
            unsafe {
                write_slot(new_ptr, 16 + i * 8, r.get_raw(&*heap));
            }
        }
        cell_bits
    })
}

/// Coerce a Clojure comparator's result NanBox into an `Ordering`, mirroring
/// `clojure.lang.AFn.compare`: a Boolean `true` means "less than"; a Boolean
/// `false` re-tests the reversed pair (`true` → "greater", else "equal");
/// otherwise the value is a Number whose sign is the ordering. Calls the
/// comparator via `cljvm_rt_invoke_2` (safe: the call-table base is installed
/// during JIT execution, which is the only context `Arrays/sort` runs in).
///
/// # Safety
/// `comp_bits` must be an invokable 2-arity fn value; `a`/`b` are passed
/// straight through and become roots in the comparator's frame before it can
/// allocate, so no GC window exposes them.
unsafe fn sort_compare(comp_bits: u64, a: u64, b: u64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let r = unsafe { cljvm_rt_invoke_2(comp_bits, a, b) };
    if nanbox_tag(r) == Some(TAG_BOOL) {
        if nanbox_payload(r) != 0 {
            Ordering::Less
        } else {
            let r2 = unsafe { cljvm_rt_invoke_2(comp_bits, b, a) };
            if nanbox_tag(r2) == Some(TAG_BOOL) && nanbox_payload(r2) != 0 {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
    } else {
        arg_to_i64(r).cmp(&0)
    }
}

/// Identity instance method — backs `(.asTransient coll)` and
/// `(.persistent coll)`, which for our delegation-based transients just
/// return the (persistent) collection unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_identity(recv_bits: u64) -> u64 {
    recv_bits
}

/// `java.util.Comparator.compare(a, b)` invoked on a Clojure `IFn`. `sort-by`
/// expands to `(fn [x y] (. comp (compare (keyfn x) (keyfn y))))`, where `comp`
/// is a fn (default `compare`). A fn-as-Comparator's `.compare` is just its
/// invocation coerced to an int (clojure.lang.AFn.compare): Boolean `true` →
/// -1, `false` → re-test reversed (`true` → 1, else 0), otherwise the returned
/// number itself.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_compare(recv_bits: u64, a: u64, b: u64) -> u64 {
    let r = unsafe { cljvm_rt_invoke_2(recv_bits, a, b) };
    if nanbox_tag(r) == Some(TAG_BOOL) {
        if nanbox_payload(r) != 0 {
            unsafe { box_long(-1) }
        } else {
            let r2 = unsafe { cljvm_rt_invoke_2(recv_bits, b, a) };
            if nanbox_tag(r2) == Some(TAG_BOOL) && nanbox_payload(r2) != 0 {
                unsafe { box_long(1) }
            } else {
                unsafe { box_long(0) }
            }
        }
    } else {
        r
    }
}

/// `java.util.Arrays.sort(Object[] a, Comparator c)` — stable in-place sort.
/// Our "array" is the fresh Vector that `to-array` produces, so we sort its
/// flat element slots in place; upstream `sort`/`sort-by` then call `(seq a)`.
///
/// GC-safety is the crux: each comparator call can allocate and the GC can
/// relocate every element AND the array cell itself. We therefore stage the
/// elements (plus the array's own NanBox) in a chunk-buffer-style `Vec<u64>`
/// registered as a GC root source, so every comparison reads forwarded bits.
/// Indices are sorted (reading the staged buffer fresh per comparison); the
/// final write-back reads the staged buffer and the forwarded array pointer
/// with no intervening allocation, so no stale pointer is ever stored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_arrays_sort(arr_bits: u64, comp_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let raw = match nanbox_tag(arr_bits) {
        Some(TAG_PTR) => nanbox_payload(arr_bits) as *const u8,
        _ => return arr_bits, // nil → nothing to sort
    };
    if raw.is_null() {
        return arr_bits;
    }
    let tid = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
    if tid != ids.vector {
        panic!("clojure-jvm: Arrays/sort — first arg must be an array (Vector), got type_id {tid}");
    }
    let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
    if n < 2 {
        return arr_bits;
    }

    // Stage [elem0 … elem_{n-1}, arr_bits] in a GC-traced buffer. After this,
    // `buf[i]` is the forwarded bits of element i and `buf[n]` is the
    // forwarded array NanBox — both updated across any GC.
    let buf = std::sync::Arc::new(std::cell::RefCell::new(Vec::<u64>::with_capacity(n + 1)));
    {
        let mut v = buf.borrow_mut();
        for i in 0..n {
            v.push(unsafe { raw.add(16 + i * 8).cast::<u64>().read_unaligned() });
        }
        v.push(arr_bits);
    }
    register_chunk_buffer(&buf);

    // Sort indices. The comparator reads its two operands from the staged
    // buffer under a short borrow that is released before the (possibly
    // GC-triggering) comparator call, satisfying the "no cell mutably borrowed
    // at a safepoint" invariant of the chunk root source.
    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.sort_by(|&i, &j| {
        let (a, b) = {
            let v = buf.borrow();
            (v[i], v[j])
        };
        unsafe { sort_compare(comp_bits, a, b) }
    });

    // Write back. Read the forwarded array pointer and forwarded elements from
    // the staged buffer; this loop allocates nothing, so every bit stays valid.
    {
        let v = buf.borrow();
        let arr_now = nanbox_payload(v[n]) as *mut u8;
        for (k, &src_i) in idxs.iter().enumerate() {
            let bits = v[src_i];
            unsafe {
                arr_now.add(16 + k * 8).cast::<u64>().write_unaligned(bits);
            }
        }
    }

    deregister_chunk_buffer(&buf);
    // Return the (possibly forwarded) array NanBox.
    let final_bits = buf.borrow()[n];
    final_bits
}

/// `clojure.lang.RT.first(Object coll)` — return the first item or nil.
/// Java: handles ISeq, Seqable, nil. We cover nil/Cons/Vector now;
/// Map iteration goes through `seq` first (Java does the same).
pub(crate) fn trap_forwarded_first_result(source: &str, owner_bits: u64, result_bits: u64) {
    if std::env::var("CLJVM_TRAP").is_err() || nanbox_tag(result_bits) != Some(TAG_PTR) {
        return;
    }
    let ptr = nanbox_payload(result_bits) as *const u8;
    if ptr.is_null() {
        return;
    }
    let hdr = unsafe { ptr.cast::<u64>().read_unaligned() };
    if hdr & dynalloc::FORWARDING_BIT == 0 {
        return;
    }
    let owner_ptr = if nanbox_tag(owner_bits) == Some(TAG_PTR) {
        nanbox_payload(owner_bits) as *const u8
    } else {
        core::ptr::null()
    };
    let fwd_ptr = (hdr & !dynalloc::FORWARDING_BIT) as *const u8;
    let owner_region = dynlang::gc::debug_current_ptr_region(owner_ptr).unwrap_or("no-runtime");
    let result_region = dynlang::gc::debug_current_ptr_region(ptr).unwrap_or("no-runtime");
    let fwd_region = dynlang::gc::debug_current_ptr_region(fwd_ptr).unwrap_or("no-runtime");
    eprintln!(
        "[first-ret] {source} owner={owner_bits:#x}/{owner_region} result={result_bits:#x}/{result_region} hdr={hdr:#x} fwd={:#x}/{fwd_region} max_gen={}",
        hdr & !dynalloc::FORWARDING_BIT,
        dynalloc::max_visit_gen()
    );
    if std::env::var("CLJVM_BACKTRACE").is_ok() {
        eprintln!(
            "[first-ret] backtrace:\n{}",
            std::backtrace::Backtrace::force_capture()
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        let mut fp: usize;
        unsafe {
            core::arch::asm!("mov {}, fp", out(reg) fp, options(nomem, nostack, preserves_flags));
        }
        for (addr, line) in dynlower::trap_find_value(fp as *const u8, owner_bits) {
            eprintln!(
                "[first-ret]   owner {line} last_visit_gen={:?}",
                dynalloc::slot_last_visit_gen(addr)
            );
        }
        for (addr, line) in dynlower::trap_find_value(fp as *const u8, result_bits) {
            eprintln!(
                "[first-ret]   result {line} last_visit_gen={:?}",
                dynalloc::slot_last_visit_gen(addr)
            );
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_first(bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(bits) {
        Some(TAG_NIL) => nanbox_nil(),
        Some(TAG_PTR) => {
            let ptr = nanbox_payload(bits) as *const u8;
            if ptr.is_null() {
                return nanbox_nil();
            }
            let type_id = unsafe { ptr.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.cons {
                let result = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
                trap_forwarded_first_result("cons.first", bits, result);
                result
            } else if type_id == ids.repeat_seq {
                let result = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
                trap_forwarded_first_result("repeat.first", bits, result);
                result
            } else if type_id == ids.iterate_seq {
                let result = unsafe { ptr.add(16).cast::<u64>().read_unaligned() };
                trap_forwarded_first_result("iterate.first", bits, result);
                result
            } else if type_id == ids.cycle_seq {
                let current = unsafe { ptr.add(16).cast::<u64>().read_unaligned() };
                unsafe { cljvm_rt_first(current) }
            } else if type_id == ids.vector {
                // Vector: header(8) + count(8) + items at offset 16.
                let n = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
                if n == 0 {
                    nanbox_nil()
                } else {
                    let result = unsafe { ptr.add(16).cast::<u64>().read_unaligned() };
                    trap_forwarded_first_result("vector.0", bits, result);
                    result
                }
            } else if type_id == ids.string {
                // (first "abc") is treated as nil — Java strings ARE
                // seqable into chars, but we don't have chars yet.
                nanbox_nil()
            } else if type_id == ids.lazy_seq {
                // Force to a concrete seq (nil or cons) via RT.seq, then
                // take its first — mirrors Java's `RT.first` calling
                // `seq(coll)` before reading the head.
                let s = unsafe { cljvm_rt_seq(bits) };
                match nanbox_tag(s) {
                    Some(TAG_NIL) => nanbox_nil(),
                    _ => unsafe { cljvm_rt_first(s) },
                }
            } else {
                if std::env::var("CLJVM_TRAP").is_ok() {
                    let hdr = unsafe { ptr.cast::<u64>().read_unaligned() };
                    #[cfg(target_arch = "aarch64")]
                    {
                        let mut fp: usize;
                        unsafe {
                            core::arch::asm!("mov {}, fp", out(reg) fp, options(nomem, nostack, preserves_flags));
                        }
                        eprintln!(
                            "[trap] rt_first bits={bits:#x} hdr={hdr:#x} type_id={type_id} fp={fp:#x} max_gen={}",
                            dynalloc::max_visit_gen()
                        );
                        for (addr, line) in dynlower::trap_find_value(fp as *const u8, bits) {
                            eprintln!(
                                "[trap]   {line} last_visit_gen={:?}",
                                dynalloc::slot_last_visit_gen(addr)
                            );
                        }
                        // Dump the machine code preceding each JIT frame's
                        // current call site, so the slot/register feeding the
                        // outgoing argument can be disassembled.
                        for (i, ret_lr) in dynlower::frame_return_addrs(fp as *const u8)
                            .into_iter()
                            .enumerate()
                            .take(4)
                        {
                            if let Some((cs, ro, bytes)) = dynlower::jit_code_window(ret_lr, 0x400)
                            {
                                let hex: String =
                                    bytes.iter().map(|b| format!("{b:02x}")).collect();
                                eprintln!(
                                    "[trap]   code frame#{i} code_start={cs:#x} call_off={ro:#x} bytes_before_call={hex}"
                                );
                            }
                        }
                    }
                }
                panic!("clojure-jvm: RT.first on unsupported heap type_id {type_id}");
            }
        }
        _ => panic!("clojure-jvm: RT.first on non-seqable NanBox tag"),
    }
}

/// `clojure.lang.Var/pushThreadBindings(map)` — Stub. Real dynamic
/// var binding isn't wired (we have a static thread-bindings stack
/// from `Var::push_thread_bindings` for COMPILE-TIME var bindings,
/// but runtime user-level `(binding [...] ...)` macros expect the
/// fn-level entry point with the JVM Var class).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_pushThreadBindings(_m: u64) -> u64 {
    panic!(
        "clojure-jvm: clojure.lang.Var/pushThreadBindings — dynamic var \
         binding not yet implemented. `binding` macro analyzes but using \
         it at runtime panics here."
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_popThreadBindings() -> u64 {
    eprintln!("[cljvm-stub] Var/popThreadBindings not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_getThreadBindingFrame() -> u64 {
    eprintln!("[cljvm-stub] Var/getThreadBindingFrame not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_getThreadBindings() -> u64 {
    eprintln!("[cljvm-stub] Var/getThreadBindings not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_find(_sym: u64) -> u64 {
    eprintln!("[cljvm-stub] Var/find not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_intern_2(_ns: u64, _sym: u64) -> u64 {
    eprintln!("[cljvm-stub] Var/intern (2-arg) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_intern_3(_ns: u64, _sym: u64, _val: u64) -> u64 {
    eprintln!("[cljvm-stub] Var/intern (3-arg) not yet implemented");
    return nanbox_nil();
}

/// `(.resetMeta v m)` — replaces a Var's metadata. Stub; no-op return.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_resetMeta(_recv: u64, m: u64) -> u64 {
    // Real impl would update var.meta; we no-op and return the new meta.
    m
}

/// `(.alterMeta v f & args)` — apply f to var's meta. Stub.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_alterMeta(_recv: u64, _f: u64, _args: u64) -> u64 {
    eprintln!("[cljvm-stub] (.alterMeta) not yet implemented");
    return nanbox_nil();
}

/// `(.bindRoot v val)` — set a Var's root binding.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_bindRoot(_recv: u64, _val: u64) -> u64 {
    eprintln!(
        "[cljvm-stub] (.bindRoot) not yet implemented — defn handles binding via the def special form"
    );
    return nanbox_nil();
}

/// `(.hasRoot v)` — does this Var have a root binding? Backs `defonce`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_hasRoot(recv: u64) -> u64 {
    let ids = heap_type_ids();
    if let Some(TAG_PTR) = nanbox_tag(recv) {
        let raw = nanbox_payload(recv) as *const u8;
        if !raw.is_null() {
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.var {
                let var: std::sync::Arc<crate::lang::var::Var> = unsafe { decode_arc_cell(raw) };
                return nanbox_bool(var.has_root());
            }
        }
    }
    panic!("clojure-jvm: (.hasRoot v) — receiver is not a Var (bits 0x{recv:x})");
}

/// `(.getRawRoot v)` — read Var's root without dynamic-binding lookup.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getRawRoot(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getRawRoot) not yet implemented");
    return nanbox_nil();
}

// IRef validators / watches / state — stub instance methods.
// Real impls would live on the Atom / Ref / Agent heap cells which
// aren't modeled yet. These are present so defns/defmacros referencing
// them analyze, even though calling them would panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_setValidator(_recv: u64, _v: u64) -> u64 {
    eprintln!("[cljvm-stub] (.setValidator) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getValidator(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getValidator) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getWatches(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getWatches) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_addWatch(_recv: u64, _key: u64, _f: u64) -> u64 {
    eprintln!("[cljvm-stub] (.addWatch) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_removeWatch(_recv: u64, _key: u64) -> u64 {
    eprintln!("[cljvm-stub] (.removeWatch) not yet implemented");
    return nanbox_nil();
}
/// Decode an Atom/Volatile receiver into its `RefState`. Panics loudly on
/// any other receiver — these methods are only reachable through the
/// `atom`/`volatile!` constructors.
unsafe fn decode_ref_cell(recv: u64, ctx: &str) -> std::sync::Arc<std::cell::RefCell<RefState>> {
    let ids = heap_type_ids();
    if let Some(TAG_PTR) = nanbox_tag(recv) {
        let raw = nanbox_payload(recv) as *const u8;
        if !raw.is_null() {
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.atom || type_id == ids.volatile_cell {
                return unsafe { decode_arc_cell(raw) };
            }
        }
    }
    panic!("clojure-jvm: {ctx} — receiver is not an Atom/Volatile (bits 0x{recv:x})");
}

/// Shared core of `(.swap atom f extra…)`: read the current value, invoke
/// `f` on it (plus extras), store and return the result. The current-value
/// read happens under a short borrow released before the (allocating,
/// GC-triggering) invoke; the GC forwards `RefState.value_bits` in place via
/// the REF registry, and the invoke's result arrives as fresh bits.
/// Single-threaded runtime — no CAS retry loop needed.
unsafe fn ref_swap(recv: u64, invoke: impl FnOnce(u64) -> u64, ctx: &str) -> u64 {
    let cell = unsafe { decode_ref_cell(recv, ctx) };
    let cur = cell.borrow().value_bits;
    let new = invoke(cur);
    cell.borrow_mut().value_bits = new;
    new
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_swap(recv: u64, f: u64) -> u64 {
    unsafe { ref_swap(recv, |cur| cljvm_rt_invoke_1(f, cur), "(.swap a f)") }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_swap_3(recv: u64, f: u64, x: u64) -> u64 {
    unsafe { ref_swap(recv, |cur| cljvm_rt_invoke_2(f, cur, x), "(.swap a f x)") }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_swap_4(recv: u64, f: u64, x: u64, y: u64) -> u64 {
    unsafe { ref_swap(recv, |cur| cljvm_rt_invoke_3(f, cur, x, y), "(.swap a f x y)") }
}
/// `(.swap a f x y args-seq)` — the variadic swap! tail: apply `f` to
/// `cur x y arg…`. Realizes the extra-args seq, then dispatches on total
/// argument count through the fixed-arity invokes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_swap_5(recv: u64, f: u64, x: u64, y: u64, args: u64) -> u64 {
    unsafe {
        ref_swap(
            recv,
            |cur| {
                let extras = seq_to_items(args);
                let mut all = vec![cur, x, y];
                all.extend_from_slice(&extras);
                match all.len() {
                    3 => cljvm_rt_invoke_3(f, all[0], all[1], all[2]),
                    4 => cljvm_rt_invoke_4(f, all[0], all[1], all[2], all[3]),
                    5 => cljvm_rt_invoke_5(f, all[0], all[1], all[2], all[3], all[4]),
                    6 => cljvm_rt_invoke_6(f, all[0], all[1], all[2], all[3], all[4], all[5]),
                    n => panic!(
                        "clojure-jvm: (.swap a f x y args) — {n} total args not \
                         supported yet (extend cljvm_inst_swap_5)"
                    ),
                }
            },
            "(.swap a f x y args)",
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_reset(recv: u64, new: u64) -> u64 {
    let cell = unsafe { decode_ref_cell(recv, "(.reset a v)") };
    cell.borrow_mut().value_bits = new;
    new
}
/// `(.compareAndSet a old new)`. Java's Atom CAS is reference identity,
/// which works there because small Longs/keywords are interned instances;
/// our boxed values aren't canonical, so value-equiv is the faithful
/// observable behavior for the cases real programs hit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_compareAndSet(recv: u64, old: u64, new: u64) -> u64 {
    let cell = unsafe { decode_ref_cell(recv, "(.compareAndSet a old new)") };
    let cur = cell.borrow().value_bits;
    let eq = unsafe { equiv_impl(cur, old) };
    if eq {
        cell.borrow_mut().value_bits = new;
    }
    nanbox_bool(eq)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_deref(recv: u64) -> u64 {
    let ids = heap_type_ids();
    if let Some(TAG_PTR) = nanbox_tag(recv) {
        let raw = nanbox_payload(recv) as *const u8;
        if !raw.is_null() {
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            // `@(reduced x)` → x.
            if type_id == ids.reduced {
                return unsafe { reduced_value(recv) };
            }
            // `@(delay …)` → forced value.
            if type_id == ids.delay {
                return unsafe { cljvm_delay_force(recv) };
            }
            // `@atom` / `@volatile` → the held value.
            if type_id == ids.atom || type_id == ids.volatile_cell {
                let cell: std::sync::Arc<std::cell::RefCell<RefState>> =
                    unsafe { decode_arc_cell(raw) };
                let v = cell.borrow().value_bits;
                return v;
            }
        }
    }
    eprintln!("[cljvm-stub] (.deref) on unsupported receiver bits 0x{recv:x} — nil");
    nanbox_nil()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_iterator(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.iterator) not yet implemented");
    return nanbox_nil();
}

// Agent error-handling stubs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_setErrorHandler(_recv: u64, _h: u64) -> u64 {
    eprintln!("[cljvm-stub] (.setErrorHandler) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getErrorHandler(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getErrorHandler) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_setErrorMode(_recv: u64, _m: u64) -> u64 {
    eprintln!("[cljvm-stub] (.setErrorMode) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getErrorMode(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getErrorMode) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getError(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getError) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_dispatch(_recv: u64, _f: u64, _args: u64, _exec: u64) -> u64 {
    eprintln!("[cljvm-stub] (.dispatch) on Agent not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_restart(_recv: u64, _new: u64, _clear: u64) -> u64 {
    eprintln!("[cljvm-stub] (.restart) on Agent not yet implemented");
    return nanbox_nil();
}

// Ref stubs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_set(_recv: u64, _val: u64) -> u64 {
    eprintln!("[cljvm-stub] (.set) on Ref not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_alter(_recv: u64, _f: u64, _args: u64) -> u64 {
    eprintln!("[cljvm-stub] (.alter) on Ref not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_commute(_recv: u64, _f: u64, _args: u64) -> u64 {
    eprintln!("[cljvm-stub] (.commute) on Ref not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_ensure(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.ensure) on Ref not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_setMinHistory(_recv: u64, _n: u64) -> u64 {
    eprintln!("[cljvm-stub] (.setMinHistory) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_setMaxHistory(_recv: u64, _n: u64) -> u64 {
    eprintln!("[cljvm-stub] (.setMaxHistory) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getMinHistory(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getMinHistory) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getMaxHistory(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getMaxHistory) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getHistoryCount(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getHistoryCount) not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_trimHistory(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.trimHistory) not yet implemented");
    return nanbox_nil();
}

// Generic "unimplemented host call" stubs. Used by the analyzer's
// fallback for unregistered `(. Class method args…)` calls so defns
// referencing Java statics we haven't ported still compile.
//
// At runtime, each prints a one-line diagnostic to stderr and returns
// nil. Returning nil (rather than panicking via `extern "C"`) is
// deliberate: a panic at the FFI boundary aborts the process,
// halting the loader. nil-substitution lets the loader make progress
// at the cost of silently-wrong subsequent behavior. The stderr line
// makes the substitution visible. When something downstream breaks,
// grep for `cljvm-unimpl-host-call` to find the culprit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_unimpl_host_call_0() -> u64 {
    eprintln!("[cljvm-unimpl-host-call] arity 0 — returning nil");
    nanbox_nil()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_unimpl_host_call_1(_a: u64) -> u64 {
    eprintln!("[cljvm-unimpl-host-call] arity 1 — returning nil");
    nanbox_nil()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_unimpl_host_call_2(_a: u64, _b: u64) -> u64 {
    eprintln!("[cljvm-unimpl-host-call] arity 2 — returning nil");
    nanbox_nil()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_unimpl_host_call_3(_a: u64, _b: u64, _c: u64) -> u64 {
    eprintln!("[cljvm-unimpl-host-call] arity 3 — returning nil");
    nanbox_nil()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_unimpl_host_call_4(_a: u64, _b: u64, _c: u64, _d: u64) -> u64 {
    eprintln!("[cljvm-unimpl-host-call] arity 4 — returning nil");
    nanbox_nil()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_unimpl_host_call_5(
    _a: u64,
    _b: u64,
    _c: u64,
    _d: u64,
    _e: u64,
) -> u64 {
    eprintln!("[cljvm-unimpl-host-call] arity 5 — returning nil");
    nanbox_nil()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_unimpl_host_call_6(
    _a: u64,
    _b: u64,
    _c: u64,
    _d: u64,
    _e: u64,
    _f: u64,
) -> u64 {
    eprintln!("[cljvm-unimpl-host-call] arity 6 — returning nil");
    nanbox_nil()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_resetThreadBindingFrame(_f: u64) -> u64 {
    eprintln!("[cljvm-stub] Var/resetThreadBindingFrame not yet implemented");
    return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_cloneThreadBindingFrame() -> u64 {
    eprintln!("[cljvm-stub] Var/cloneThreadBindingFrame not yet implemented");
    return nanbox_nil();
}

// Multimethod *registration* (`.reset`/`.addMethod`/`.removeMethod`/
// `.preferMethod`) runs at LOAD time when `defmethod`/`remove-all-methods`
// top-level forms execute. We don't model multimethods, so these can't
// register anything — but they must NOT hard-panic, because a panic across
// the `extern "C"` boundary aborts the whole loader (it can't unwind). Skip
// the registration with a visible `[cljvm-stub]` note and return nil, exactly
// like other unmodeled load-time host ops (e.g. `cloneThreadBindingFrame`).
// The *dispatch*-side externs below (getMethod/methodTable/…) still panic with
// a clear message, since hitting them means a multimethod was actually called.

/// `(.reset multifn)` — `clojure.lang.MultiFn.reset()`. No-op (see above).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_multifn_reset(recv: u64) -> u64 {
    eprintln!("[cljvm-stub] MultiFn.reset() — multimethods not modeled; ignoring (recv 0x{recv:x})");
    nanbox_nil()
}
/// `(new clojure.lang.MultiFn name dispatch-fn default hierarchy)` —
/// upstream `defmulti` expands to this. The hierarchy arg (a Var) is
/// accepted and ignored: dispatch is exact-equiv plus the default key;
/// `isa?` hierarchies panic in the lookup when no exact match exists
/// and would be needed.
pub fn multifn_ctor(args: &[u64], ids: HeapTypeIds) -> u64 {
    if args.len() != 4 {
        panic!(
            "clojure-jvm: (new clojure.lang.MultiFn …) — expected 4 args \
             (name dispatch default hierarchy), got {}",
            args.len()
        );
    }
    let name = unsafe { read_string_heap(args[0], ids, "MultiFn ctor — name") }.to_string();
    let st = std::sync::Arc::new(std::cell::RefCell::new(MultiFnState {
        name,
        dispatch_bits: args[1],
        default_bits: args[2],
        methods: Vec::new(),
    }));
    register_multifn_state(&st);
    unsafe {
        alloc_arc_cell(
            ids.multifn_cell,
            st,
            crate::lang::compiler::with_active_session_root_multifn,
        )
    }
}

unsafe fn decode_multifn(recv: u64) -> Option<std::sync::Arc<std::cell::RefCell<MultiFnState>>> {
    let ids = heap_type_ids();
    if let Some(TAG_PTR) = nanbox_tag(recv) {
        let raw = nanbox_payload(recv) as *const u8;
        if !raw.is_null() {
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.multifn_cell {
                return Some(unsafe { decode_arc_cell(raw) });
            }
        }
    }
    None
}

/// If `fn_bits` is a MultiFn cell, dispatch and invoke; None otherwise.
/// Dispatch: run the dispatch fn on the args, find the method whose key
/// is equiv to the dispatch value, else the default-key method, else
/// panic with the dispatch value (mirrors Java's IllegalArgumentException).
pub unsafe fn try_multifn_invoke(fn_bits: u64, args: &[u64]) -> Option<u64> {
    let st = unsafe { decode_multifn(fn_bits) }?;
    // Stage args (and the eventual dispatch value) in a registered chunk
    // buffer — both invokes below can GC.
    let buf = std::sync::Arc::new(std::cell::RefCell::new(args.to_vec()));
    register_chunk_buffer(&buf);
    let dv = {
        let dispatch = st.borrow().dispatch_bits;
        let a = buf.borrow().clone();
        unsafe { invoke_n(dispatch, &a) }
    };
    // equiv_impl is Rust-side (no JIT, no GC) — safe to compare bare bits.
    let method = {
        let stb = st.borrow();
        let exact = stb.methods.iter().find(|(k, _)| unsafe { equiv_impl(*k, dv) });
        match exact {
            Some((_, m)) => Some(*m),
            None => stb
                .methods
                .iter()
                .find(|(k, _)| unsafe { equiv_impl(*k, stb.default_bits) })
                .map(|(_, m)| *m),
        }
    };
    let method = method.unwrap_or_else(|| {
        panic!(
            "clojure-jvm: IllegalArgumentException — No method in multimethod \
             '{}' for dispatch value: {}",
            st.borrow().name,
            pr_str_bits(dv)
        )
    });
    let a = buf.borrow().clone();
    let out = unsafe { invoke_n(method, &a) };
    deregister_chunk_buffer(&buf);
    Some(out)
}

/// Invoke `f` with a runtime-length arg list via the fixed-arity invokes.
unsafe fn invoke_n(f: u64, a: &[u64]) -> u64 {
    unsafe {
        match a.len() {
            0 => cljvm_rt_invoke_0(f),
            1 => cljvm_rt_invoke_1(f, a[0]),
            2 => cljvm_rt_invoke_2(f, a[0], a[1]),
            3 => cljvm_rt_invoke_3(f, a[0], a[1], a[2]),
            4 => cljvm_rt_invoke_4(f, a[0], a[1], a[2], a[3]),
            5 => cljvm_rt_invoke_5(f, a[0], a[1], a[2], a[3], a[4]),
            6 => cljvm_rt_invoke_6(f, a[0], a[1], a[2], a[3], a[4], a[5]),
            n => panic!("clojure-jvm: invoke_n — arity {n} not wired (extend invoke_n)"),
        }
    }
}

/// `(.addMethod multifn dispatch-val f)` — upstream `defmethod` expands
/// to this. Returns the multifn (Java returns `this`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_multifn_addMethod(recv: u64, dv: u64, f: u64) -> u64 {
    match unsafe { decode_multifn(recv) } {
        Some(st) => {
            let mut stb = st.borrow_mut();
            // Re-defining a method for the same dispatch value replaces it.
            if let Some(slot) = stb.methods.iter_mut().find(|(k, _)| unsafe { equiv_impl(*k, dv) }) {
                slot.1 = f;
            } else {
                stb.methods.push((dv, f));
            }
            recv
        }
        None => {
            if matches!(nanbox_tag(recv), Some(TAG_NIL)) {
                // The multifn var is nil — its defmulti lives in a
                // sub-file we don't embed (e.g. print-method in
                // clojure/core_print). Defer instead of aborting the
                // core load; calling the multimethod still fails loudly.
                eprintln!(
                    "[cljvm-defer] (.addMethod nil …) — defmulti not loaded \
                     (sub-file not embedded); method NOT registered"
                );
                return nanbox_nil();
            }
            panic!(
                "clojure-jvm: (.addMethod m dv f) — receiver is not a MultiFn \
                 (bits 0x{recv:x})"
            )
        }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_multifn_removeMethod(_a: u64, _b: u64) -> u64 {
    eprintln!("[cljvm-stub] MultiFn.removeMethod — multimethods not modeled; ignoring");
    nanbox_nil()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_multifn_preferMethod(_a: u64, _b: u64, _c: u64) -> u64 {
    eprintln!("[cljvm-stub] MultiFn.preferMethod — multimethods not modeled; ignoring");
    nanbox_nil()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_multifn_methodTable(_a: u64) -> u64 {
    panic!("clojure-jvm: MultiFn.getMethodTable not yet implemented");
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_multifn_preferTable(_a: u64) -> u64 {
    panic!("clojure-jvm: MultiFn.getPreferTable not yet implemented");
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_multifn_getMethod(_a: u64, _b: u64) -> u64 {
    panic!("clojure-jvm: MultiFn.getMethod not yet implemented");
}

/// `(.getNamespace named)` — Symbol or Keyword. Returns the namespace
/// String, or nil. Mirrors Java's `Named.getNamespace`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getNamespace(x_bits: u64) -> u64 {
    let ids = heap_type_ids();
    if let Some(TAG_PTR) = nanbox_tag(x_bits) {
        let p = nanbox_payload(x_bits) as *const u8;
        if !p.is_null() {
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.symbol {
                let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::symbol::Symbol;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let s = unsafe { Arc::from_raw(arc_ptr) };
                return match s.get_namespace() {
                    Some(ns) => unsafe { alloc_string_heap(ns, ids) },
                    None => nanbox_nil(),
                };
            }
            if tid == ids.keyword {
                let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::keyword::Keyword;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let k = unsafe { Arc::from_raw(arc_ptr) };
                return match k.get_namespace() {
                    Some(ns) => unsafe { alloc_string_heap(ns, ids) },
                    None => nanbox_nil(),
                };
            }
        }
    }
    panic!("clojure-jvm: (.getNamespace) on non-Named receiver bits 0x{x_bits:x}");
}

/// `(.rseq rev)` — Reversible.rseq. For a Vector, return a cons-seq of
/// items in reverse order; nil if empty. (We don't have sorted-map yet.)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_rseq(rev_bits: u64) -> u64 {
    let ids = heap_type_ids();
    if let Some(TAG_PTR) = nanbox_tag(rev_bits) {
        let p = nanbox_payload(rev_bits) as *const u8;
        if !p.is_null() {
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.vector {
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as usize;
                if n == 0 {
                    return nanbox_nil();
                }
                let items: Vec<u64> = (0..n)
                    .map(|i| unsafe { p.add(16 + i * 8).cast::<u64>().read_unaligned() })
                    .collect();
                return dynobj::roots::gc_enter(items.len() + 1, |heap, scope| {
                    let roots: Vec<_> = items.iter().map(|v| scope.root::<()>(*v)).collect();
                    let tail = scope.root::<()>(nanbox_nil());
                    // Forward iteration (NOT reversed) since we want reverse-order seq:
                    // walk left→right and prepend so the last item ends up at head.
                    for r in roots.iter() {
                        let new_tail = heap_cons(heap, *r, tail);
                        tail.set_raw(new_tail);
                    }
                    tail.get_raw(&*heap).bits()
                });
            }
        }
    }
    eprintln!("[cljvm-stub] (.rseq) on unsupported receiver bits 0x{rev_bits:x}");
    return nanbox_nil();
}

/// `(.getKey entry)` — `Map.Entry.getKey()`. We model map entries as
/// 2-element vectors, so this is `(nth e 0)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getKey(e_bits: u64) -> u64 {
    let ids = heap_type_ids();
    if let Some(TAG_PTR) = nanbox_tag(e_bits) {
        let p = nanbox_payload(e_bits) as *const u8;
        if !p.is_null() {
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.vector {
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as usize;
                if n >= 1 {
                    return unsafe { p.add(16).cast::<u64>().read_unaligned() };
                }
            }
        }
    }
    eprintln!("[cljvm-stub] (.getKey) on non-MapEntry receiver bits 0x{e_bits:x}");
    return nanbox_nil();
}

/// `(.getValue entry)` — `Map.Entry.getValue()`. `(nth e 1)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getValue(e_bits: u64) -> u64 {
    let ids = heap_type_ids();
    if let Some(TAG_PTR) = nanbox_tag(e_bits) {
        let p = nanbox_payload(e_bits) as *const u8;
        if !p.is_null() {
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.vector {
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as usize;
                if n >= 2 {
                    return unsafe { p.add(24).cast::<u64>().read_unaligned() };
                }
            }
        }
    }
    eprintln!("[cljvm-stub] (.getValue) on non-MapEntry receiver bits 0x{e_bits:x}");
    return nanbox_nil();
}

/// `(.disjoin set key)` — return a new set without `key`. Receiver
/// must be a PersistentHashSet (or nil → nil).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_disjoin(set_bits: u64, key_bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(set_bits) {
        Some(TAG_NIL) => nanbox_nil(),
        Some(TAG_PTR) => {
            let p = nanbox_payload(set_bits) as *const u8;
            if p.is_null() {
                return nanbox_nil();
            }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.tree_set {
                let s = unsafe { decode_tree_set(p) };
                let key = any_bits_to_object(key_bits, ids);
                return alloc_tree_set_cell(s.disjoin(&key));
            }
            if tid != ids.set {
                eprintln!("[cljvm-stub] (.disjoin) on non-set type_id {tid}");
                return nanbox_nil();
            }
            let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                as *const crate::lang::persistent_hash_set::PersistentHashSet;
            unsafe { Arc::increment_strong_count(arc_ptr) };
            let s = unsafe { Arc::from_raw(arc_ptr) };
            let key = any_bits_to_object(key_bits, ids);
            let new_s = s.disjoin(&key);
            let raw_arc = Arc::as_ptr(&new_s) as u64;
            crate::lang::compiler::with_active_session_root_set(new_s);
            let raw = dynlang::gc::gc_alloc_thunk(ids.set as u64, 0);
            let np = raw as *mut u8;
            if np.is_null() {
                panic!("clojure-jvm: (.disjoin) alloc null");
            }
            unsafe {
                np.add(8).cast::<u64>().write_unaligned(raw_arc);
            }
            nanbox_encode(TAG_PTR, raw & PAYLOAD_MASK)
        }
        _ => panic!("clojure-jvm: (.disjoin) on non-set non-nil receiver"),
    }
}

/// `clojure.lang.RT.contains(coll, key)` — true if `key` is present
/// in the map (or in vector index range). Returns NanBox-encoded bool.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_contains(coll_bits: u64, key_bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => nanbox_bool(false),
        Some(TAG_PTR) => {
            let p = nanbox_payload(coll_bits) as *const u8;
            if p.is_null() {
                return nanbox_bool(false);
            }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.map {
                let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_hash_map::PersistentHashMap;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let m = unsafe { Arc::from_raw(arc_ptr) };
                let key = any_bits_to_object(key_bits, ids);
                return nanbox_bool(m.contains_key(&key));
            }
            if tid == ids.vector {
                if !nanbox_tag(key_bits).is_none() {
                    return nanbox_bool(false);
                }
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as i64;
                let idx = arg_to_i64(key_bits);
                return nanbox_bool(idx >= 0 && idx < n);
            }
            if tid == ids.set {
                let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_hash_set::PersistentHashSet;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let s = unsafe { Arc::from_raw(arc_ptr) };
                let key = any_bits_to_object(key_bits, ids);
                return nanbox_bool(s.contains(&key));
            }
            if tid == ids.tree_map {
                let m = unsafe { decode_tree_map(p) };
                let key = any_bits_to_object(key_bits, ids);
                return nanbox_bool(m.contains_key(&key));
            }
            if tid == ids.tree_set {
                let s = unsafe { decode_tree_set(p) };
                let key = any_bits_to_object(key_bits, ids);
                return nanbox_bool(s.contains(&key));
            }
            panic!("clojure-jvm: RT.contains on unsupported type_id {tid}");
        }
        _ => nanbox_bool(false),
    }
}

/// `clojure.lang.RT.get(map, key)` — `(get m k)`. Returns nil if not present.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_get(coll_bits: u64, key_bits: u64) -> u64 {
    unsafe { cljvm_rt_get_3(coll_bits, key_bits, nanbox_nil()) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_get_3(coll_bits: u64, key_bits: u64, not_found: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => not_found,
        Some(TAG_PTR) => {
            let p = nanbox_payload(coll_bits) as *const u8;
            if p.is_null() {
                return not_found;
            }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.map {
                let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_hash_map::PersistentHashMap;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let m = unsafe { Arc::from_raw(arc_ptr) };
                let key = any_bits_to_object(key_bits, ids);
                let v = m.val_at(&key);
                if matches!(v, Object::Nil) {
                    if !m.contains_key(&key) {
                        return not_found;
                    }
                }
                return crate::lang::compiler::with_active_session_encode_object(&v);
            }
            if tid == ids.vector {
                // Vectors index by integer only. The key is either a boxed
                // Long (`(get v 2)`) or a native number; anything else
                // (keyword/string/…) is `not_found`.
                let is_int = is_boxed_long(key_bits) || nanbox_tag(key_bits).is_none();
                if !is_int {
                    return not_found;
                }
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as i64;
                let idx = arg_to_i64(key_bits);
                if idx < 0 || idx >= n {
                    return not_found;
                }
                return unsafe {
                    p.add(16 + (idx as usize) * 8)
                        .cast::<u64>()
                        .read_unaligned()
                };
            }
            if tid == ids.tree_map {
                let m = unsafe { decode_tree_map(p) };
                let key = any_bits_to_object(key_bits, ids);
                if !m.contains_key(&key) {
                    return not_found;
                }
                let v = m.val_at(&key);
                return crate::lang::compiler::with_active_session_encode_object(&v);
            }
            if tid == ids.tree_set {
                let s = unsafe { decode_tree_set(p) };
                let key = any_bits_to_object(key_bits, ids);
                return if s.contains(&key) { key_bits } else { not_found };
            }
            if tid == ids.set {
                let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_hash_set::PersistentHashSet;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let s = unsafe { Arc::from_raw(arc_ptr) };
                let key = any_bits_to_object(key_bits, ids);
                if s.contains(&key) {
                    key_bits
                } else {
                    not_found
                }
            } else {
                not_found
            }
        }
        _ => not_found,
    }
}

/// `clojure.lang.RT.find(map, key)` — returns the [k v] map-entry as a
/// 2-vector, or nil if not present.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_find(coll_bits: u64, key_bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => nanbox_nil(),
        Some(TAG_PTR) => {
            let p = nanbox_payload(coll_bits) as *const u8;
            if p.is_null() {
                return nanbox_nil();
            }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid != ids.map {
                panic!("clojure-jvm: RT.find on non-map type_id {tid}");
            }
            let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                as *const crate::lang::persistent_hash_map::PersistentHashMap;
            unsafe { Arc::increment_strong_count(arc_ptr) };
            let m = unsafe { Arc::from_raw(arc_ptr) };
            let key = any_bits_to_object(key_bits, ids);
            match m.entry_at(&key) {
                None => nanbox_nil(),
                Some((k, v)) => {
                    let kbits = crate::lang::compiler::with_active_session_encode_object(&k);
                    let vbits = crate::lang::compiler::with_active_session_encode_object(&v);
                    // Build [k v] vector.
                    dynobj::roots::gc_enter(3, |heap, scope| {
                        let kr = scope.root::<()>(kbits);
                        let vr = scope.root::<()>(vbits);
                        let cell = heap_alloc(heap, ids.vector as u64, 2).root(scope);
                        let cell_bits = cell.get_raw(&*heap).bits();
                        let np = nanbox_payload(cell_bits) as *mut u8;
                        unsafe {
                            write_raw_word(np, 8, 2);
                            write_slot(np, 16, kr.get_raw(&*heap));
                            write_slot(np, 24, vr.get_raw(&*heap));
                        }
                        cell_bits
                    })
                }
            }
        }
        _ => nanbox_nil(),
    }
}

/// `clojure.lang.RT.dissoc(map, key)` — return new map without key.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_dissoc(coll_bits: u64, key_bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => nanbox_nil(),
        Some(TAG_PTR) => {
            let p = nanbox_payload(coll_bits) as *const u8;
            if p.is_null() {
                return nanbox_nil();
            }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.tree_map {
                let m = unsafe { decode_tree_map(p) };
                let key = any_bits_to_object(key_bits, ids);
                return alloc_tree_map_cell(m.without(&key));
            }
            if tid != ids.map {
                panic!("clojure-jvm: RT.dissoc on non-map type_id {tid}");
            }
            let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                as *const crate::lang::persistent_hash_map::PersistentHashMap;
            unsafe { Arc::increment_strong_count(arc_ptr) };
            let m = unsafe { Arc::from_raw(arc_ptr) };
            let key = any_bits_to_object(key_bits, ids);
            let new_m = m.without(&key);
            let raw_arc = Arc::as_ptr(&new_m) as u64;
            crate::lang::compiler::with_active_session_root_map(new_m);
            let raw = dynlang::gc::gc_alloc_thunk(ids.map as u64, 0);
            let np = raw as *mut u8;
            if np.is_null() {
                panic!("clojure-jvm: RT.dissoc alloc null");
            }
            unsafe {
                np.add(8).cast::<u64>().write_unaligned(raw_arc);
            }
            nanbox_encode(TAG_PTR, raw & PAYLOAD_MASK)
        }
        _ => panic!("clojure-jvm: RT.dissoc on non-map non-nil receiver"),
    }
}

/// `clojure.lang.RT.keys(map)` — return a seq of keys, or nil if empty.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_keys(coll_bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => nanbox_nil(),
        Some(TAG_PTR) => {
            let p = nanbox_payload(coll_bits) as *const u8;
            if p.is_null() {
                return nanbox_nil();
            }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            let keys: Vec<u64> = if tid == ids.tree_map {
                let m = unsafe { decode_tree_map(p) };
                m.iter()
                    .map(|(k, _)| crate::lang::compiler::with_active_session_encode_object(&k))
                    .collect()
            } else if tid == ids.map {
                let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_hash_map::PersistentHashMap;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let m = unsafe { Arc::from_raw(arc_ptr) };
                m.iter()
                    .map(|(k, _)| crate::lang::compiler::with_active_session_encode_object(&k))
                    .collect()
            } else {
                panic!("clojure-jvm: RT.keys on non-map type_id {tid}");
            };
            if keys.is_empty() {
                return nanbox_nil();
            }
            dynobj::roots::gc_enter(keys.len() + 1, |heap, scope| {
                let roots: Vec<_> = keys.iter().map(|v| scope.root::<()>(*v)).collect();
                let tail = scope.root::<()>(nanbox_nil());
                for r in roots.iter().rev() {
                    let new_tail = heap_cons(heap, *r, tail);
                    tail.set_raw(new_tail);
                }
                tail.get_raw(&*heap).bits()
            })
        }
        _ => panic!("clojure-jvm: RT.keys on unsupported NanBox"),
    }
}

/// `clojure.lang.RT.vals(map)` — seq of values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_vals(coll_bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => nanbox_nil(),
        Some(TAG_PTR) => {
            let p = nanbox_payload(coll_bits) as *const u8;
            if p.is_null() {
                return nanbox_nil();
            }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            let vals: Vec<u64> = if tid == ids.tree_map {
                let m = unsafe { decode_tree_map(p) };
                m.iter()
                    .map(|(_, v)| crate::lang::compiler::with_active_session_encode_object(&v))
                    .collect()
            } else if tid == ids.map {
                let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_hash_map::PersistentHashMap;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let m = unsafe { Arc::from_raw(arc_ptr) };
                m.iter()
                    .map(|(_, v)| crate::lang::compiler::with_active_session_encode_object(&v))
                    .collect()
            } else {
                panic!("clojure-jvm: RT.vals on non-map type_id {tid}");
            };
            if vals.is_empty() {
                return nanbox_nil();
            }
            dynobj::roots::gc_enter(vals.len() + 1, |heap, scope| {
                let roots: Vec<_> = vals.iter().map(|v| scope.root::<()>(*v)).collect();
                let tail = scope.root::<()>(nanbox_nil());
                for r in roots.iter().rev() {
                    let new_tail = heap_cons(heap, *r, tail);
                    tail.set_raw(new_tail);
                }
                tail.get_raw(&*heap).bits()
            })
        }
        _ => panic!("clojure-jvm: RT.vals on unsupported NanBox"),
    }
}

/// `clojure.lang.RT.peek(coll)` — for a list/cons, same as `first`;
/// for a vector, the LAST element; nil for empty / nil. We treat
/// any seq-like (cons) as list-style peek (= first).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_peek(bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(bits) {
        Some(TAG_NIL) => nanbox_nil(),
        Some(TAG_PTR) => {
            let p = nanbox_payload(bits) as *const u8;
            if p.is_null() {
                return nanbox_nil();
            }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.cons {
                unsafe { p.add(8).cast::<u64>().read_unaligned() }
            } else if tid == ids.vector {
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as usize;
                if n == 0 {
                    nanbox_nil()
                } else {
                    unsafe { p.add(16 + (n - 1) * 8).cast::<u64>().read_unaligned() }
                }
            } else {
                panic!("clojure-jvm: RT.peek on unsupported heap type_id {tid}");
            }
        }
        _ => panic!("clojure-jvm: RT.peek on non-seqable NanBox"),
    }
}

/// `clojure.lang.RT.pop(coll)` — for a list/cons, the rest; for a
/// vector, a vector with the last item dropped; nil → nil.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_pop(bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(bits) {
        Some(TAG_NIL) => nanbox_nil(),
        Some(TAG_PTR) => {
            let p = nanbox_payload(bits) as *const u8;
            if p.is_null() {
                return nanbox_nil();
            }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.cons {
                unsafe { p.add(16).cast::<u64>().read_unaligned() }
            } else if tid == ids.vector {
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as usize;
                if n == 0 {
                    panic!("clojure-jvm: RT.pop on empty vector");
                }
                let new_n = n - 1;
                dynobj::roots::gc_enter(new_n + 2, |heap, scope| {
                    let item_roots: Vec<_> = (0..new_n)
                        .map(|i| {
                            let v = unsafe { read_slot(&*heap, p, 16 + i * 8) };
                            v.root(scope)
                        })
                        .collect();
                    let cell = heap_alloc(heap, ids.vector as u64, new_n as u64).root(scope);
                    let cell_bits = cell.get_raw(&*heap).bits();
                    let nptr = nanbox_payload(cell_bits) as *mut u8;
                    unsafe {
                        write_raw_word(nptr, 8, new_n as u64);
                        for (i, r) in item_roots.iter().enumerate() {
                            write_slot(nptr, 16 + i * 8, r.get_raw(&*heap));
                        }
                    }
                    cell_bits
                })
            } else {
                panic!("clojure-jvm: RT.pop on unsupported heap type_id {tid}");
            }
        }
        _ => panic!("clojure-jvm: RT.pop on non-seqable NanBox"),
    }
}

/// `clojure.lang.Util.equiv(a, b)` — Clojure's structural equality.
/// Java dispatches through `Util.equiv(Object, Object)` → `IPersistentCollection`
/// equality, primitive Number equality, etc. We port the cases we have:
///   * Same NanBox bits → equal (covers nil, bool, identical doubles).
///   * Same heap type_id: dispatch per type.
///     - String: byte-compare.
///     - Symbol: compare ns + name (no global interning yet).
///     - Keyword: Arc::ptr_eq via the stored Raw64 pointer (interned).
///     - Cons: structural recursion on first + rest.
///   * Otherwise: not equal.
///
/// Returns a NanBox-encoded bool.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_equiv(a_bits: u64, b_bits: u64) -> u64 {
    let result = unsafe { equiv_impl(a_bits, b_bits) };
    nanbox_bool(result)
}

/// Implementation of `equiv`. Unsafe because it dereferences NanBox-encoded
/// heap pointers; caller must ensure both args are valid (live or nil).
unsafe fn equiv_impl(a: u64, b: u64) -> bool {
    if a == b {
        return true;
    }
    // Two pointers or two non-pointers with different bits can still be equal
    // if they're both heap objects of the same type.
    let a_tag = nanbox_tag(a);
    let b_tag = nanbox_tag(b);
    if a_tag != b_tag {
        return false;
    }
    let Some(TAG_PTR) = a_tag else {
        // Both immediates with different bits → not equal.
        return false;
    };
    let ids = heap_type_ids();
    let a_ptr = nanbox_payload(a) as *const u8;
    let b_ptr = nanbox_payload(b) as *const u8;
    if a_ptr.is_null() || b_ptr.is_null() {
        return false;
    }
    let a_type = unsafe { a_ptr.cast::<u16>().read_unaligned() } as usize;
    let b_type = unsafe { b_ptr.cast::<u16>().read_unaligned() } as usize;
    if a_type != b_type {
        return false;
    }
    if a_type == ids.long {
        // Two distinct boxed-Long cells are `=` iff they hold the same value.
        return unsafe { unbox_long(a) == unbox_long(b) };
    }
    if a_type == ids.character {
        // Two distinct Character cells are `=` iff they hold the same
        // codepoint. (A Character is never `=` to a Long: the `a_type !=
        // b_type` check above already returned false for `(= \a 97)`.)
        return unsafe { unbox_char(a) == unbox_char(b) };
    }
    if a_type == ids.string {
        // varlen_bytes: count at offset 8, bytes at offset 16.
        let a_count = unsafe { a_ptr.add(8).cast::<u64>().read_unaligned() } as usize;
        let b_count = unsafe { b_ptr.add(8).cast::<u64>().read_unaligned() } as usize;
        if a_count != b_count {
            return false;
        }
        let a_bytes = unsafe { std::slice::from_raw_parts(a_ptr.add(16), a_count) };
        let b_bytes = unsafe { std::slice::from_raw_parts(b_ptr.add(16), b_count) };
        return a_bytes == b_bytes;
    }
    if a_type == ids.symbol {
        let a_arc = unsafe { a_ptr.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::symbol::Symbol;
        let b_arc = unsafe { b_ptr.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::symbol::Symbol;
        if a_arc == b_arc {
            return true;
        }
        // Value equality on (ns, name).
        let a_sym = unsafe { &*a_arc };
        let b_sym = unsafe { &*b_arc };
        return a_sym.get_namespace() == b_sym.get_namespace()
            && a_sym.get_name() == b_sym.get_name();
    }
    if a_type == ids.keyword {
        let a_arc = unsafe { a_ptr.add(8).cast::<u64>().read_unaligned() };
        let b_arc = unsafe { b_ptr.add(8).cast::<u64>().read_unaligned() };
        // Keywords ARE globally interned in our port — pointer equality is
        // value equality.
        return a_arc == b_arc;
    }
    if a_type == ids.cons {
        let a_first = unsafe { a_ptr.add(8).cast::<u64>().read_unaligned() };
        let b_first = unsafe { b_ptr.add(8).cast::<u64>().read_unaligned() };
        if !unsafe { equiv_impl(a_first, b_first) } {
            return false;
        }
        let a_rest = unsafe { a_ptr.add(16).cast::<u64>().read_unaligned() };
        let b_rest = unsafe { b_ptr.add(16).cast::<u64>().read_unaligned() };
        return unsafe { equiv_impl(a_rest, b_rest) };
    }
    if a_type == ids.vector {
        // Vector layout: count (Raw64) at offset 8, elements (NanBox) from
        // offset 16. Equal iff same length and element-wise `=`.
        let a_n = unsafe { a_ptr.add(8).cast::<u64>().read_unaligned() };
        let b_n = unsafe { b_ptr.add(8).cast::<u64>().read_unaligned() };
        if a_n != b_n {
            return false;
        }
        for i in 0..a_n as usize {
            let ae = unsafe { a_ptr.add(16 + i * 8).cast::<u64>().read_unaligned() };
            let be = unsafe { b_ptr.add(16 + i * 8).cast::<u64>().read_unaligned() };
            if !unsafe { equiv_impl(ae, be) } {
                return false;
            }
        }
        return true;
    }
    if a_type == ids.map {
        // Read the host-side Arc pointers stored as Raw64. Pointer-equal
        // Arcs (same instance) short-circuit; otherwise dispatch to the
        // host-side structural comparator.
        let a_arc = unsafe { a_ptr.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::persistent_hash_map::PersistentHashMap;
        let b_arc = unsafe { b_ptr.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::persistent_hash_map::PersistentHashMap;
        if a_arc == b_arc {
            return true;
        }
        let a_map = unsafe { &*a_arc };
        let b_map = unsafe { &*b_arc };
        return a_map.equiv(b_map);
    }
    if a_type == ids.set {
        let a_arc = unsafe { a_ptr.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::persistent_hash_set::PersistentHashSet;
        let b_arc = unsafe { b_ptr.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::persistent_hash_set::PersistentHashSet;
        if a_arc == b_arc {
            return true;
        }
        let a_set = unsafe { &*a_arc };
        let b_set = unsafe { &*b_arc };
        return a_set.equiv(b_set);
    }
    if a_type == ids.namespace || a_type == ids.var {
        // Both cells hold a leaked `Arc::as_ptr` at offset 8. Namespaces are
        // interned in the global registry and each Var is a single Arc, so
        // canonical instances share that raw pointer — pointer equality is
        // value equality. This is what makes `(= ns (.ns v))` in `ns-publics`
        // hold for two freshly-boxed cells that name the same namespace.
        let a_arc = unsafe { a_ptr.add(8).cast::<u64>().read_unaligned() };
        let b_arc = unsafe { b_ptr.add(8).cast::<u64>().read_unaligned() };
        return a_arc == b_arc;
    }
    false
}

// ─── `clojure.lang.Util.hash` + `case*` dispatch ─────────────────────────
//
// The `case` macro computes `(clojure.lang.Util/hash test-constant)` at
// EXPANSION time to build the case-map keys, and the compiled `case*`
// dispatch recomputes the hash of the runtime value to index the switch.
// Both sides MUST use the same function or every hash-mode `case` silently
// falls through to its default. `cljvm_util_hash` (the host static method)
// and `cljvm_case_dispatch` (CaseExpr's switch-index helper) therefore both
// route through `util_hash_bits` below.
//
// The hash itself ports Java's `Util.hash(Object)` (`o.hashCode()`),
// following each type's documented `hashCode` formula from the upstream
// sources (Long/Double/Boolean/Character/String per java.lang, Symbol /
// Keyword / collections per clojure.lang). Exact numeric parity with the
// JVM holds for the types whose hashCode is specified; identity-based
// hashes (fns, unmodeled host cells) are arbitrary-but-harmless because no
// case test constant can be `=` to such a value — a spurious bucket match
// is rejected by the post-switch equivalence check.

/// Java `String.hashCode()`: `h = 31*h + c` over UTF-16 code units.
fn java_string_hash_code(s: &str) -> i32 {
    s.encode_utf16()
        .fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(c as i32))
}

/// Java `Util.hashCombine(seed, hash)` (boost-style):
/// `seed ^ (hash + 0x9e3779b9 + (seed << 6) + (seed >> 2))`.
fn java_hash_combine(seed: i32, hash: i32) -> i32 {
    seed ^ hash
        .wrapping_add(0x9e3779b9u32 as i32)
        .wrapping_add(seed.wrapping_shl(6))
        .wrapping_add(seed >> 2)
}

/// Java `Symbol.hashCode()`: `hashCombine(name.hashCode(), Util.hash(ns))`.
fn java_symbol_hash_code(sym: &crate::lang::symbol::Symbol) -> i32 {
    let name_h = java_string_hash_code(sym.get_name());
    let ns_h = sym
        .get_namespace()
        .map(java_string_hash_code)
        .unwrap_or(0);
    java_hash_combine(name_h, ns_h)
}

/// Java `clojure.lang.Util.hash(Object)` over our host-side `Object`
/// representation. Total for every value-modeled variant; panics for
/// variants that have no value hash (callers pre-handle those with
/// `identity_hash_bits` at the NanBox level).
pub fn util_hash_object(o: &Object) -> i32 {
    match o {
        Object::Nil => 0,
        // Java `Boolean.hashCode()`.
        Object::Bool(b) => {
            if *b {
                1231
            } else {
                1237
            }
        }
        // Java `Long.hashCode()`: `(int)(value ^ (value >>> 32))`.
        Object::Long(v) => ((*v as u64) ^ ((*v as u64) >> 32)) as u32 as i32,
        // Java `Double.hashCode()`: same fold over `doubleToLongBits`.
        Object::Double(x) => {
            let bits = x.to_bits();
            (bits ^ (bits >> 32)) as u32 as i32
        }
        // Java `Character.hashCode()`: the char value itself.
        Object::Char(c) => *c as i32,
        Object::String(s) => java_string_hash_code(s),
        Object::Symbol(s) => java_symbol_hash_code(s),
        // Java `Keyword.hashCode()`: `sym.hashCode() + 0x9e3779b9`.
        Object::Keyword(k) => {
            java_symbol_hash_code(&k.sym).wrapping_add(0x9e3779b9u32 as i32)
        }
        // Java `ASeq.hashCode()` / `APersistentVector.hashCode()`:
        // `h = 1; h = 31*h + Util.hash(e)` — identical formulas, so a list
        // and a vector of equal elements hash alike (as on the JVM).
        Object::List(l) => l
            .iter()
            .fold(1i32, |h, e| {
                h.wrapping_mul(31).wrapping_add(util_hash_object(&e))
            }),
        Object::Vector(v) => v
            .iter()
            .fold(1i32, |h, e| {
                h.wrapping_mul(31).wrapping_add(util_hash_object(&e))
            }),
        // Java `APersistentMap.mapHash`: sum of `hash(k) ^ hash(v)`.
        Object::Map(m) => m.iter().fold(0i32, |h, (k, v)| {
            h.wrapping_add(util_hash_object(&k) ^ util_hash_object(&v))
        }),
        Object::TreeMap(m) => m.iter().fold(0i32, |h, (k, v)| {
            h.wrapping_add(util_hash_object(&k) ^ util_hash_object(&v))
        }),
        // Java `APersistentSet.hashCode()`: sum of `Util.hash(e)`.
        Object::Set(s) => s
            .iter()
            .fold(0i32, |h, e| h.wrapping_add(util_hash_object(&e))),
        Object::TreeSet(s) => s
            .iter()
            .fold(0i32, |h, e| h.wrapping_add(util_hash_object(&e))),
        // `hashCode` ignores metadata.
        Object::WithMeta(inner, _) => util_hash_object(inner),
        // Vars/Namespaces are canonical instances — Java uses the default
        // identity hashCode. Our Arcs are the canonical identities; fold
        // the pointer the same way Long folds its value.
        Object::Var(v) => {
            let p = std::sync::Arc::as_ptr(v) as usize as u64;
            (p ^ (p >> 32)) as u32 as i32
        }
        Object::Namespace(n) => {
            let p = std::sync::Arc::as_ptr(n) as usize as u64;
            (p ^ (p >> 32)) as u32 as i32
        }
        // Non-value objects (fn handles, unmodeled heap cells) NESTED inside
        // a collection being hashed (e.g. `case` dispatching on
        // `[(fn [] 1)]`). At the top level `util_hash_bits` pre-handles them
        // with the bits-identity hash; nested, the original bits are gone.
        // Any deterministic value is provably correct here: such an object
        // can never be `=` to a component of a literal test constant (reader
        // literals only contain value types), so a colliding bucket is always
        // rejected by the post-switch equivalence check, and no real bucket
        // is ever missed (a composite containing a non-value object equals no
        // constant). This mirrors Java, where `Object.hashCode` is an
        // arbitrary identity hash with no cross-run meaning.
        Object::Host(_) | Object::Unported { .. } => 0,
    }
}

/// Identity hash for values with no modeled value-hash (fn handles,
/// unrecognized heap cells): fold the NanBox bits like `Long.hashCode`.
/// This mirrors Java's default identity `Object.hashCode()` — arbitrary
/// but harmless for `case` dispatch, since no test constant can be `=`
/// to such a value and any spurious bucket match fails the post-switch
/// equivalence check. (Unlike the JVM's, this hash is not stable across
/// a moving collection — fine for `case*`, which consumes it immediately.)
fn identity_hash_bits(bits: u64) -> i32 {
    (bits ^ (bits >> 32)) as u32 as i32
}

/// `Util.hash` over arbitrary NanBox bits. The single hash function shared
/// by macro-expansion-time `(clojure.lang.Util/hash x)` and `case*` runtime
/// dispatch.
///
/// # Safety
/// `bits` must be a valid live NanBox value (heap pointers traced by the
/// active GC); must run on a registered mutator thread (lazy seqs are
/// realized, which allocates).
unsafe fn util_hash_bits(bits: u64) -> i32 {
    let ids = heap_type_ids();
    let obj = any_bits_to_object(bits, ids);
    match obj {
        // TAG_FN handles and unrecognized heap type_ids decode to
        // `Unported`; `Host` is defensive (no decode path produces it).
        Object::Unported { .. } | Object::Host(_) => identity_hash_bits(bits),
        o => util_hash_object(&o),
    }
}

/// `(clojure.lang.Util/hash x)` host static method — returns a boxed Long
/// holding the (sign-extended) i32 hash, exactly what the `case` macro's
/// `prep-hashes` / `merge-hash-collisions` arithmetic expects.
///
/// # Safety
/// See [`util_hash_bits`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_util_hash(x_bits: u64) -> u64 {
    let h = unsafe { util_hash_bits(x_bits) };
    unsafe { box_long(h as i64) }
}

/// Switch-index values for [`cljvm_case_dispatch`]'s `test_type` argument.
pub const CASE_TEST_INT: i64 = 0;
pub const CASE_TEST_HASH: i64 = 1;

/// Sentinel index returned when an `:int`-mode dispatch value is not a
/// number (Java: `instanceOf Number` check jumps straight to the default
/// label). `i64::MIN` can never collide with a real case-map key: int-mode
/// keys are i32-ranged test values or shift-masked (small, non-negative)
/// indexes — `parse_case_form` hard-asserts this.
pub const CASE_DISPATCH_NO_MATCH: i64 = i64::MIN;

/// Compute the `case*` switch index for `val_bits`. Ports the dispatch
/// half of Java `CaseExpr.doEmit`:
///   * `:int` (`test_type` = [`CASE_TEST_INT`]): `((Number)v).intValue()`,
///     non-Numbers go to the default branch (sentinel).
///   * `:hash-equiv` / `:hash-identity` ([`CASE_TEST_HASH`]): `Util.hash(v)`.
/// Then `emitShiftMask`: `(h >> shift) & mask` when `mask != 0`.
///
/// Returns a RAW i64 (NOT a NanBox). The value never looks like a tagged
/// pointer (i32-sign-extended values and the sentinel all fail the NanBox
/// tag-pattern check), so holding it across safepoints is GC-safe — same
/// contract as `emit_unboxed` primitives.
///
/// # Safety
/// See [`util_hash_bits`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_case_dispatch(
    val_bits: u64,
    test_type: u64,
    shift: u64,
    mask: u64,
) -> u64 {
    let test_type = test_type as i64;
    let shift = shift as i64;
    let mask = mask as i64;
    let h: i64 = match test_type {
        CASE_TEST_INT => {
            if is_boxed_long(val_bits) {
                // Java `Long.intValue()`: truncate to i32, sign-extend.
                (unsafe { unbox_long(val_bits) }) as i32 as i64
            } else if nanbox_tag(val_bits).is_none() {
                // Untagged → double. Java `Double.intValue()` is the JLS
                // narrowing cast: saturating, NaN → 0 — same as Rust `as`.
                (f64::from_bits(val_bits) as i32) as i64
            } else {
                // Not a Number → default branch.
                return CASE_DISPATCH_NO_MATCH as u64;
            }
        }
        CASE_TEST_HASH => (unsafe { util_hash_bits(val_bits) }) as i64,
        other => panic!("clojure-jvm: cljvm_case_dispatch: unknown test_type {other}"),
    };
    let idx = if mask != 0 { (h >> shift) & mask } else { h };
    idx as u64
}

/// `clojure.lang.Util.nil?(x)` — NanBox tag check.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_is_nil(x_bits: u64) -> u64 {
    nanbox_bool(nanbox_tag(x_bits) == Some(TAG_NIL))
}

/// `clojure.lang.Util.compare(a, b)` — three-way numeric/lexical compare.
/// Returns NanBox-encoded Long: -1 / 0 / 1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_util_compare(a_bits: u64, b_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let a = any_bits_to_object(a_bits, ids);
    let b = any_bits_to_object(b_bits, ids);
    let ord = crate::lang::persistent_tree_map::compare_objects(&a, &b);
    let n: i64 = match ord {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    };
    box_long(n)
}

/// `clojure.lang.Compiler/macroexpand1(form)` — backs the `macroexpand-1`
/// and `macroexpand` core fns. Decodes the form to a host Object, runs one
/// expansion step (host intercepts + upstream defmacros, same path the
/// analyzer uses), and re-encodes. Non-macro forms come back unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_compiler_macroexpand1(form_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let form = any_bits_to_object(form_bits, ids);
    match crate::lang::compiler::macroexpand1_for_runtime(&form) {
        Some(expanded) => crate::lang::compiler::with_active_session_encode_object(&expanded),
        None => form_bits,
    }
}

/// `clojure.lang.Compiler/eval(form)` — backs `clojure.core/eval`.
/// Decodes the form to a host Object and runs a full nested
/// compile+execute in the active Session (the same reentrant-run_jit
/// shape macro bodies already exercise during compilation).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_compiler_eval(form_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let form = any_bits_to_object(form_bits, ids);
    crate::lang::compiler::eval_in_active_session(form)
}

/// `clojure.lang.Util.identical(a, b)` — Java reference equality.
/// Bit-equality on NanBox values handles nil/bool/long/double directly
/// and pointer-identity for heap cells. Keywords need one more step:
/// Java interns Keyword INSTANCES (so `(identical? :a :a)` is true), but
/// our heap cells are per-occurrence wrappers around the interned Arc —
/// compare the Arc pointer when both sides are keyword cells. (Symbols
/// are NOT interned instances in Java; `(identical? 'a 'a)` is false
/// there, and distinct cells give the same answer here.)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_util_identical(a_bits: u64, b_bits: u64) -> u64 {
    if a_bits == b_bits {
        return nanbox_bool(true);
    }
    let ids = heap_type_ids();
    if let (Some(TAG_PTR), Some(TAG_PTR)) = (nanbox_tag(a_bits), nanbox_tag(b_bits)) {
        let pa = nanbox_payload(a_bits) as *const u8;
        let pb = nanbox_payload(b_bits) as *const u8;
        if !pa.is_null() && !pb.is_null() {
            let ta = unsafe { pa.cast::<u16>().read_unaligned() } as usize;
            let tb = unsafe { pb.cast::<u16>().read_unaligned() } as usize;
            if ta == ids.keyword && tb == ids.keyword {
                let aa = unsafe { pa.add(8).cast::<u64>().read_unaligned() };
                let ab = unsafe { pb.add(8).cast::<u64>().read_unaligned() };
                return nanbox_bool(aa == ab);
            }
        }
    }
    nanbox_bool(false)
}

/// `clojure.lang.RT.more(Object coll)` — same shape as `next` for our
/// Cons-only model. Real Clojure distinguishes seq-empty (returns
/// `()`) from nil (returns `()`); for us, both return nil. Used by
/// `clojure.core/rest`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_more(bits: u64) -> u64 {
    unsafe { cljvm_rt_next(bits) }
}

/// `clojure.lang.RT.seq(Object coll)` — returns a seq on coll, nil for
/// empty/nil. For our Cons-only model: nil-in → nil-out; Cons-in →
/// the cons itself (already a seq).
///
/// Java distinguishes seqable types here (Iterable, CharSequence,
/// Map, native arrays, etc.). We only handle Cons + nil for now.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_seq(bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(bits) {
        Some(TAG_NIL) => nanbox_nil(),
        Some(TAG_PTR) => {
            let ptr = nanbox_payload(bits) as *const u8;
            if ptr.is_null() {
                return nanbox_nil();
            }
            let type_id = unsafe { ptr.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.cons {
                bits
            } else if type_id == ids.repeat_seq
                || type_id == ids.iterate_seq
                || type_id == ids.cycle_seq
            {
                // Repeat/Iterate/Cycle cells are already (non-empty) seqs.
                bits
            } else if type_id == ids.with_meta {
                // Metadata wrapper: seq the inner value.
                let inner = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
                unsafe { cljvm_rt_seq(inner) }
            } else if type_id == ids.vector {
                // Vector → seq: walk items right-to-left into a cons-list.
                // Empty vector seqs to nil (matches Java `RT.seq`).
                //
                // Each item bits may be a heap pointer; cljvm_rt_cons
                // triggers GC which moves cells. Root the WHOLE snapshot
                // in a with_scope so each iteration's read of items[i]
                // sees the post-GC forwarded address. Without rooting,
                // items[i+1]'s heap pointer goes stale during the cons of
                // item i and the next iteration writes garbage into
                // the new cons cell's first field. Symptom was form 46
                // SIGBUS when decoding a macro result whose chain pointed
                // `rest` into the interior of an unrelated string cell.
                let n = unsafe { ptr.add(8).cast::<u64>().read_unaligned() } as usize;
                if n == 0 {
                    return nanbox_nil();
                }
                dynobj::roots::gc_enter(n + 1, |heap, scope| {
                    let item_roots: Vec<dynobj::roots::Rooted<()>> = (0..n)
                        .map(|i| {
                            let bits =
                                unsafe { ptr.add(16 + i * 8).cast::<u64>().read_unaligned() };
                            scope.root::<()>(bits)
                        })
                        .collect();
                    // Root `cur` so it survives each cons-call's GC.
                    let cur_root = scope.root::<()>(nanbox_nil());
                    for i in (0..n).rev() {
                        let new_cur = heap_cons(heap, item_roots[i], cur_root);
                        cur_root.set_raw(new_cur);
                    }
                    cur_root.get_raw(&*heap).bits()
                })
            } else if type_id == ids.map {
                // Map → seq of `[k v]` MapEntry vectors (Java's `RT.seq`
                // on an IPersistentMap). `(key e)`/`(val e)` read the 2-elem
                // vector's slots. Order is the map's iteration order.
                let arc_ptr = unsafe { ptr.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_hash_map::PersistentHashMap;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let m = unsafe { Arc::from_raw(arc_ptr) };
                // Snapshot the (Object, Object) pairs first: they're Rust
                // Arcs, so they stay live across the cell allocations below
                // even if those trigger GC.
                let pairs: Vec<(Object, Object)> = m.iter().collect();
                let n = pairs.len();
                if n == 0 {
                    return nanbox_nil();
                }
                dynobj::roots::gc_enter(n + 1, |heap, scope| {
                    let tail = scope.root::<()>(nanbox_nil());
                    for (k, v) in pairs.into_iter() {
                        let entry = Object::Vector(
                            crate::lang::persistent_vector::PersistentVector::create(vec![k, v]),
                        );
                        let entry_bits =
                            crate::lang::compiler::with_active_session_encode_object(&entry);
                        let e = scope.root::<()>(entry_bits);
                        let new_tail = heap_cons(heap, e, tail);
                        tail.set_raw(new_tail);
                    }
                    tail.get_raw(&*heap).bits()
                })
            } else if type_id == ids.set {
                // Set → seq of its elements (Java's `RT.seq` on an
                // IPersistentSet). Same GC-safe snapshot-then-cons pattern.
                let arc_ptr = unsafe { ptr.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_hash_set::PersistentHashSet;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let s = unsafe { Arc::from_raw(arc_ptr) };
                let elems: Vec<Object> = s.iter().collect();
                let n = elems.len();
                if n == 0 {
                    return nanbox_nil();
                }
                dynobj::roots::gc_enter(n + 1, |heap, scope| {
                    let tail = scope.root::<()>(nanbox_nil());
                    for e in elems.into_iter() {
                        let e_bits = crate::lang::compiler::with_active_session_encode_object(&e);
                        let er = scope.root::<()>(e_bits);
                        let new_tail = heap_cons(heap, er, tail);
                        tail.set_raw(new_tail);
                    }
                    tail.get_raw(&*heap).bits()
                })
            } else if type_id == ids.tree_map {
                // Sorted map → seq of `[k v]` entries in ASCENDING key order.
                // (We cons from the back, so iterate reversed to keep order.)
                let arc_ptr = unsafe { ptr.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_tree_map::PersistentTreeMap;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let m = unsafe { Arc::from_raw(arc_ptr) };
                let pairs: Vec<(Object, Object)> = m.iter().collect();
                if pairs.is_empty() {
                    return nanbox_nil();
                }
                dynobj::roots::gc_enter(pairs.len() + 1, |heap, scope| {
                    let tail = scope.root::<()>(nanbox_nil());
                    for (k, v) in pairs.into_iter().rev() {
                        let entry = Object::Vector(
                            crate::lang::persistent_vector::PersistentVector::create(vec![k, v]),
                        );
                        let e = scope
                            .root::<()>(crate::lang::compiler::with_active_session_encode_object(&entry));
                        let new_tail = heap_cons(heap, e, tail);
                        tail.set_raw(new_tail);
                    }
                    tail.get_raw(&*heap).bits()
                })
            } else if type_id == ids.tree_set {
                // Sorted set → seq of elements in ASCENDING order.
                let arc_ptr = unsafe { ptr.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_tree_set::PersistentTreeSet;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let s = unsafe { Arc::from_raw(arc_ptr) };
                let elems: Vec<Object> = s.iter().collect();
                if elems.is_empty() {
                    return nanbox_nil();
                }
                dynobj::roots::gc_enter(elems.len() + 1, |heap, scope| {
                    let tail = scope.root::<()>(nanbox_nil());
                    for e in elems.into_iter().rev() {
                        let er = scope
                            .root::<()>(crate::lang::compiler::with_active_session_encode_object(&e));
                        let new_tail = heap_cons(heap, er, tail);
                        tail.set_raw(new_tail);
                    }
                    tail.get_raw(&*heap).bits()
                })
            } else if type_id == ids.string {
                // String → seq of Character (one per Unicode scalar). Same
                // GC-safe snapshot-then-cons pattern as the collection
                // branches: snapshot the codepoints into a Rust Vec first
                // (no heap pointers), then box each Character and cons it.
                let s = unsafe { read_string_heap(bits, ids, "RT.seq on String") };
                let codepoints: Vec<u32> = s.chars().map(|c| c as u32).collect();
                if codepoints.is_empty() {
                    return nanbox_nil();
                }
                dynobj::roots::gc_enter(codepoints.len() + 1, |heap, scope| {
                    let tail = scope.root::<()>(nanbox_nil());
                    for &cp in codepoints.iter().rev() {
                        let cbits = unsafe { box_char(cp) };
                        let cr = scope.root::<()>(cbits);
                        let new_tail = heap_cons(heap, cr, tail);
                        tail.set_raw(new_tail);
                    }
                    tail.get_raw(&*heap).bits()
                })
            } else if type_id == ids.lazy_seq {
                // Force the thunk; recurse on the realized value.
                let arc: std::sync::Arc<std::cell::RefCell<LazyState>> =
                    unsafe { decode_arc_cell(ptr) };
                let cached = {
                    let st = arc.borrow();
                    if st.realized {
                        Some(st.value_bits)
                    } else {
                        None
                    }
                };
                let realized = match cached {
                    Some(v) => v,
                    None => {
                        let thunk = arc.borrow().thunk_bits;
                        let v = unsafe { cljvm_rt_invoke_0(thunk) };
                        let mut st = arc.borrow_mut();
                        st.realized = true;
                        st.value_bits = v;
                        v
                    }
                };
                unsafe { cljvm_rt_seq(realized) }
            } else {
                eprintln!(
                    "[cljvm-stub] RT.seq on unsupported heap type_id \
                     {type_id} — returning nil"
                );
                nanbox_nil()
            }
        }
        _ => {
            eprintln!("[cljvm-stub] RT.seq on non-seqable NanBox tag — nil");
            nanbox_nil()
        }
    }
}

/// `clojure.lang.RT.next(Object coll)` — return the next seq, or nil if
/// the seq has one or fewer elements. Java distinguishes more vs next;
/// we narrow to the Cons/nil pair.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_next(bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(bits) {
        Some(TAG_NIL) => nanbox_nil(),
        Some(TAG_PTR) => {
            let ptr = nanbox_payload(bits) as *const u8;
            if ptr.is_null() {
                return nanbox_nil();
            }
            let type_id = unsafe { ptr.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.cons {
                // `next` returns a *seq* (Java forces one step), so route
                // the cons tail through RT.seq: a plain cons tail returns
                // itself, a nil tail returns nil, and a lazy-seq tail gets
                // forced. Without this, `(next coll)` on a one-element seq
                // backed by a lazy rest would return a non-nil unrealized
                // seq instead of nil.
                let tail = unsafe { ptr.add(16).cast::<u64>().read_unaligned() };
                unsafe { cljvm_rt_seq(tail) }
            } else if type_id == ids.lazy_seq {
                // Force to a concrete seq, then take its next.
                let s = unsafe { cljvm_rt_seq(bits) };
                unsafe { cljvm_rt_next(s) }
            } else if type_id == ids.repeat_seq {
                let count = unsafe { ptr.add(16).cast::<u64>().read_unaligned() } as i64;
                if count == -1 {
                    // Infinite: the rest of (repeat x) is (repeat x) itself.
                    bits
                } else if count > 1 {
                    let value = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
                    alloc_repeat(count - 1, value)
                } else {
                    nanbox_nil()
                }
            } else if type_id == ids.iterate_seq {
                // next = Iterate(f, (f value)). The invoke may allocate and
                // move the heap; root f/value across it via the chunk-buffer
                // root source, then read the forwarded bits back.
                let buf = std::sync::Arc::new(std::cell::RefCell::new(vec![
                    unsafe { ptr.add(8).cast::<u64>().read_unaligned() },
                    unsafe { ptr.add(16).cast::<u64>().read_unaligned() },
                ]));
                register_chunk_buffer(&buf);
                let (f_now, v_now) = {
                    let v = buf.borrow();
                    (v[0], v[1])
                };
                let new_val = unsafe { cljvm_rt_invoke_1(f_now, v_now) };
                buf.borrow_mut()[1] = new_val;
                let (f_final, val_final) = {
                    let v = buf.borrow();
                    (v[0], v[1])
                };
                deregister_chunk_buffer(&buf);
                alloc_seq2(ids.iterate_seq, f_final, val_final)
            } else if type_id == ids.cycle_seq {
                // next = Cycle(all, (next current)), wrapping to all at the
                // end. rt_next may force a lazy tail (allocating), so stage
                // all/current in a registered chunk buffer first.
                let buf = std::sync::Arc::new(std::cell::RefCell::new(vec![
                    unsafe { ptr.add(8).cast::<u64>().read_unaligned() },
                    unsafe { ptr.add(16).cast::<u64>().read_unaligned() },
                ]));
                register_chunk_buffer(&buf);
                let cur_now = buf.borrow()[1];
                let n = unsafe { cljvm_rt_next(cur_now) };
                buf.borrow_mut()[1] = n;
                let (all_final, n_final) = {
                    let v = buf.borrow();
                    (v[0], v[1])
                };
                deregister_chunk_buffer(&buf);
                let next_cur = if matches!(nanbox_tag(n_final), Some(TAG_NIL)) {
                    all_final
                } else {
                    n_final
                };
                alloc_seq2(ids.cycle_seq, all_final, next_cur)
            } else if type_id == ids.vector {
                // Vector → seq of items[1..count): build a cons-list so
                // callers see a Seq, matching upstream's `RT.next` which
                // returns an `ISeq` (not a subvec). Empty / single-element
                // vectors yield nil. Items must be rooted across the cons
                // allocations because each `cljvm_rt_cons` triggers a GC
                // safepoint that may relocate any prior cell.
                let n = unsafe { ptr.add(8).cast::<u64>().read_unaligned() } as usize;
                if n <= 1 {
                    return nanbox_nil();
                }
                let mut items: Vec<u64> = Vec::with_capacity(n - 1);
                for i in 1..n {
                    items.push(unsafe { ptr.add(16 + i * 8).cast::<u64>().read_unaligned() });
                }
                dynobj::roots::gc_enter(items.len() + 1, |heap, scope| {
                    let roots: Vec<_> = items.iter().map(|v| scope.root::<()>(*v)).collect();
                    let acc_root = scope.root::<()>(nanbox_nil());
                    for r in roots.iter().rev() {
                        let acc = heap_cons(heap, *r, acc_root);
                        acc_root.set_raw(acc);
                    }
                    acc_root.get_raw(&*heap).bits()
                })
            } else {
                eprintln!("[cljvm-stub] RT.next on unsupported heap type_id {type_id} — nil");
                return nanbox_nil();
            }
        }
        _ => panic!("clojure-jvm: RT.next on non-seqable NanBox tag"),
    }
}

// ── Var externs ────────────────────────────────────────────────────────

/// JIT extern: bind `val_bits` (NanBox) as the root of the Var at `var_ptr`.
/// Returns `val_bits` so the caller can use it as the expression value.
///
/// Safety: `var_ptr` must be a pointer obtained from `Arc::as_ptr` on a
/// `Var` that remains alive (the global namespace mapping holds it). The
/// caller (`DefExpr.emit`) bakes the pointer in as a compile-time constant
/// after looking up the Var via `Namespace::intern`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_bind_root(var_ptr: u64, val_bits: u64) -> u64 {
    let v: &Var = unsafe { &*(var_ptr as *const Var) };
    // Store the NanBox straight into the Var's GC-rooted slot: a heap
    // pointer here is a real root that the collector forwards, so the Var
    // never dangles after a move.
    v.bind_root_bits(val_bits);
    val_bits
}

/// JIT extern: return the current value of the Var at `var_ptr`, NanBox-encoded.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_deref(var_ptr: u64) -> u64 {
    let v: &Var = unsafe { &*(var_ptr as *const Var) };
    v.deref_bits()
}

/// Resolve an `Arc<Var>` to a `u64` suitable for baking into IR. Holds onto
/// the Arc through the namespace mapping (the caller's responsibility), so
/// the pointer remains valid for the program's lifetime.
pub fn var_to_jit_ptr(v: &Arc<Var>) -> u64 {
    Arc::as_ptr(v) as u64
}

// ── Static-field externs ───────────────────────────────────────────────
//
// JVM Clojure exposes `clojure.lang.PersistentList/creator` as a `static
// IFn` field initialized at class load. We model it as a 0-arg "static
// getter" registered in `Compiler.host_methods`: `(. PL creator)` lowers
// to a `Call(cljvm_pl_creator, [])`, and this extern returns the cached
// NanBox handle of the singleton variadic-identity fn that
// `Session::new` compiles once via `init_static_singletons`.

/// `clojure.lang.PersistentList/creator` — return the cached singleton
/// NanBox handle for the variadic-identity fn `(fn* [& xs] xs)`.
///
/// The handle is per-Session because it indexes into that Session's JIT
/// call table; we resolve the active Session through `ACTIVE_SESSION`
/// (which `Session::eval_form` installs around the JIT execution that
/// triggers this call).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_pl_creator() -> u64 {
    crate::lang::compiler::active_session_pl_creator_handle()
}

// ── Instance-method externs (`(.method recv args…)`) ──────────────────
//
// Compiled `(.method recv args…)` forms lower to a direct `Call` to one
// of these externs. Each handles the receiver-type dispatch the JVM
// would do via interface vtables. Add a new extern (and register it
// in `Compiler.instance_methods`) when a new instance method is needed.

/// `(.meta x)` — return `x`'s metadata map, or nil if it has none.
///   * `Cons` carries metadata in its dedicated meta slot (offset 24).
///   * `WithMeta` wrapper exposes its meta slot.
///   * Everything else returns nil (Java's `IMeta.meta()` on non-IMeta
///     returns nil too).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_meta(recv_bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(recv_bits) {
        Some(TAG_PTR) => {
            let raw = nanbox_payload(recv_bits) as *const u8;
            if raw.is_null() {
                return nanbox_nil();
            }
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.cons {
                return unsafe { raw.add(24).cast::<u64>().read_unaligned() };
            }
            if type_id == ids.with_meta {
                // WithMeta layout: header(8) + inner(8) + meta(8).
                return unsafe { raw.add(16).cast::<u64>().read_unaligned() };
            }
            nanbox_nil()
        }
        _ => nanbox_nil(),
    }
}

/// `(.empty coll)` — the empty collection of the same category. Backs
/// `clojure.core/empty`. Cons-backed lists return nil (the same
/// empty-list-as-nil compromise as `longrange_build`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_empty(recv: u64) -> u64 {
    let ids = heap_type_ids();
    if let Some(TAG_PTR) = nanbox_tag(recv) {
        let raw = nanbox_payload(recv) as *const u8;
        if !raw.is_null() {
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.vector {
                return dynobj::roots::gc_enter(1, |heap, _scope| {
                    heap_alloc(heap, ids.vector as u64, 0).bits()
                });
            }
            if type_id == ids.map {
                return unsafe { cljvm_phm_create(nanbox_nil()) };
            }
            if type_id == ids.set {
                return unsafe { cljvm_phs_create(nanbox_nil()) };
            }
            if type_id == ids.tree_map {
                return unsafe { cljvm_ptm_create(nanbox_nil()) };
            }
            if type_id == ids.tree_set {
                return unsafe { cljvm_pts_create(nanbox_nil()) };
            }
            if type_id == ids.cons || type_id == ids.lazy_seq || type_id == ids.repeat_seq {
                return nanbox_nil();
            }
        }
    }
    nanbox_nil()
}

// ── Constructor externs (`(new ClassName args)`) ──────────────────────
//
// One extern per arity (class + N args). Each decodes the Class id off
// the receiver heap cell, looks up the registered constructor in
// `host_class`, and dispatches. Classes without a constructor panic
// with a clear message — adding `Some(ctor)` to a class entry is a
// one-place edit.

unsafe fn dispatch_new(class_bits: u64, args: &[u64]) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(class_bits) {
        Some(TAG_PTR) => {
            let raw = nanbox_payload(class_bits) as *const u8;
            if raw.is_null() {
                panic!("clojure-jvm: (new nil ...) — null Class receiver");
            }
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id != ids.class {
                panic!(
                    "clojure-jvm: (new ...) — receiver is not a Class \
                     (heap type_id {type_id})"
                );
            }
            let raw_id = unsafe { raw.add(8).cast::<u64>().read_unaligned() };
            let info =
                crate::lang::host_class::by_id(crate::lang::host_class::ClassId(raw_id as u16));
            match info.ctor {
                Some(c) => c(args, ids),
                None => panic!(
                    "clojure-jvm: (new {} ...) — class has no registered \
                     constructor (add `Some(ctor)` to its `host_class` entry)",
                    info.name
                ),
            }
        }
        _ => panic!(
            "clojure-jvm: (new ...) — receiver bits 0x{class_bits:x} is \
             not a Class heap pointer"
        ),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_new_1(class_bits: u64) -> u64 {
    unsafe { dispatch_new(class_bits, &[]) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_new_2(class_bits: u64, a: u64) -> u64 {
    unsafe { dispatch_new(class_bits, &[a]) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_new_3(class_bits: u64, a: u64, b: u64) -> u64 {
    unsafe { dispatch_new(class_bits, &[a, b]) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_new_4(class_bits: u64, a: u64, b: u64, c: u64) -> u64 {
    unsafe { dispatch_new(class_bits, &[a, b, c]) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_new_5(class_bits: u64, a: u64, b: u64, c: u64, d: u64) -> u64 {
    unsafe { dispatch_new(class_bits, &[a, b, c, d]) }
}

// ── clojure.lang.ExceptionInfo ─────────────────────────────────────────
//
// The `ex-info`/`ex-data` exception type — a real heap-allocated object
// (three GC-traced Value slots; offsets in `lang::exception_info`), not a
// host-side Arc cell. Constructed by the registered host-class ctor
// (`host_class::exception_info_ctor` → `clj_exception_info_{2,3}`); the
// slots are read back by the `.getData`/`.getMessage`/`.getCause`
// instance-method externs below.

/// Allocate a `clojure.lang.ExceptionInfo` cell. Mirrors Java's
/// `ExceptionInfo(String msg, IPersistentMap data, Throwable cause)`
/// constructor contract: `data` must be non-nil (Java throws
/// `IllegalArgumentException("Additional data must be non-nil.")`).
fn exception_info_alloc(msg_bits: u64, data_bits: u64, cause_bits: u64) -> u64 {
    use crate::lang::exception_info::{
        CAUSE_OFFSET, DATA_OFFSET, MESSAGE_OFFSET, STACK_TRACE_OFFSET,
    };
    let ids = heap_type_ids();
    if matches!(nanbox_tag(data_bits), Some(TAG_NIL)) {
        panic!(
            "clojure-jvm: IllegalArgumentException — \
             (new clojure.lang.ExceptionInfo …): Additional data must be non-nil."
        );
    }
    dynobj::roots::gc_enter(5, |heap, scope| {
        let msg = scope.root::<()>(msg_bits);
        let data = scope.root::<()>(data_bits);
        let cause = scope.root::<()>(cause_bits);
        // The trace slot starts as a zero-length Java-array-as-Vector
        // (this runtime records no JVM frames — the Throwable contract's
        // zero-length-array case). `Throwable.getStackTrace` never
        // returns nil, so the slot is never nil.
        let trace = heap_alloc(heap, ids.vector as u64, 0).root(scope);
        let cell = heap_alloc(heap, ids.exception_info as u64, 0).root(scope);
        let owner_bits = cell.get_raw(&*heap).bits();
        let new_ptr = nanbox_payload(owner_bits) as *mut u8;
        unsafe {
            write_slot(new_ptr, MESSAGE_OFFSET, msg.get_raw(&*heap));
            write_slot(new_ptr, DATA_OFFSET, data.get_raw(&*heap));
            write_slot(new_ptr, CAUSE_OFFSET, cause.get_raw(&*heap));
            write_slot(new_ptr, STACK_TRACE_OFFSET, trace.get_raw(&*heap));
        }
        owner_bits
    })
}

/// `(ExceptionInfo. msg data)` — 2-arg constructor (no cause).
pub fn clj_exception_info_2(msg_bits: u64, data_bits: u64) -> u64 {
    exception_info_alloc(msg_bits, data_bits, nanbox_nil())
}

/// `(ExceptionInfo. msg data cause)` — 3-arg constructor.
pub fn clj_exception_info_3(msg_bits: u64, data_bits: u64, cause_bits: u64) -> u64 {
    exception_info_alloc(msg_bits, data_bits, cause_bits)
}

/// Shared receiver check + slot read for the ExceptionInfo accessors.
/// Panics with a clear message when the receiver is not an
/// ExceptionInfo — in Java these methods don't exist on other types
/// (core.clj guards each call site with the matching `instance?` test).
unsafe fn exception_info_slot(recv_bits: u64, off: usize, method: &str) -> u64 {
    let ids = heap_type_ids();
    if let Some(TAG_PTR) = nanbox_tag(recv_bits) {
        let raw = nanbox_payload(recv_bits) as *const u8;
        if !raw.is_null() {
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.exception_info {
                return unsafe { raw.add(off).cast::<u64>().read_unaligned() };
            }
        }
    }
    panic!(
        "clojure-jvm: (.{method} x) — receiver is not a \
         clojure.lang.ExceptionInfo (bits 0x{recv_bits:x})"
    );
}

/// `(.getData ex)` — `clojure.lang.IExceptionInfo.getData()`. Backs
/// `clojure.core/ex-data`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getData(recv_bits: u64) -> u64 {
    use crate::lang::exception_info::DATA_OFFSET;
    unsafe { exception_info_slot(recv_bits, DATA_OFFSET, "getData") }
}

/// `(.getMessage ex)` — `java.lang.Throwable.getMessage()`. Backs
/// `clojure.core/ex-message` (ExceptionInfo is our only Throwable-shaped
/// heap type today; other receivers panic).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getMessage(recv_bits: u64) -> u64 {
    use crate::lang::exception_info::MESSAGE_OFFSET;
    unsafe { exception_info_slot(recv_bits, MESSAGE_OFFSET, "getMessage") }
}

/// `(.getCause ex)` — `java.lang.Throwable.getCause()`. Backs
/// `clojure.core/ex-cause`. nil when constructed without a cause.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getCause(recv_bits: u64) -> u64 {
    use crate::lang::exception_info::CAUSE_OFFSET;
    unsafe { exception_info_slot(recv_bits, CAUSE_OFFSET, "getCause") }
}

/// `(.getStackTrace ex)` — `java.lang.Throwable.getStackTrace()`.
/// Returns the trace slot's Java-array-as-Vector (zero-length at
/// construction; whatever `.setStackTrace` stored afterwards). Never nil.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getStackTrace(recv_bits: u64) -> u64 {
    use crate::lang::exception_info::STACK_TRACE_OFFSET;
    unsafe { exception_info_slot(recv_bits, STACK_TRACE_OFFSET, "getStackTrace") }
}

/// `(.setStackTrace ex trace)` — `java.lang.Throwable.setStackTrace`.
/// Stores `trace` (a Java-array-as-Vector) into the receiver's trace
/// slot. Java's contract rejects a null trace array with
/// NullPointerException; we panic the same way. Returns nil (void).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_setStackTrace(recv_bits: u64, trace_bits: u64) -> u64 {
    use crate::lang::exception_info::STACK_TRACE_OFFSET;
    let ids = heap_type_ids();
    if matches!(nanbox_tag(trace_bits), Some(TAG_NIL)) {
        panic!(
            "clojure-jvm: NullPointerException — (.setStackTrace ex trace): \
             stack trace array must be non-nil"
        );
    }
    if let Some(TAG_PTR) = nanbox_tag(recv_bits) {
        let raw = nanbox_payload(recv_bits) as *mut u8;
        if !raw.is_null() {
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.exception_info {
                // Plain slot store: no allocation (and hence no safepoint)
                // happens between reading `raw` and the write, and the slot
                // is a declared traced Value field, so the GC scans the new
                // reference on the next collection.
                unsafe {
                    raw.add(STACK_TRACE_OFFSET)
                        .cast::<u64>()
                        .write_unaligned(trace_bits);
                }
                return nanbox_nil();
            }
        }
    }
    panic!(
        "clojure-jvm: (.setStackTrace x trace) — receiver is not a \
         clojure.lang.ExceptionInfo (bits 0x{recv_bits:x})"
    );
}

/// `clojure.lang.RT.seqToTypedArray(ISeq seq)` — backs `into-array`.
/// Arrays are modeled as Vector cells (the same analog `RT.toArray`
/// uses), so this is exactly toArray's collection walk.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_seqToTypedArray(seq_bits: u64) -> u64 {
    unsafe { cljvm_rt_toArray(seq_bits) }
}

/// `clojure.lang.RT.seqToTypedArray(Class type, ISeq seq)` — the typed
/// 2-arity. Our Vector cells are untyped, so the component-type argument
/// is accepted and unused (Java uses it only to pick the array's
/// component class; element values are unchanged either way).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_seqToTypedArray2(_type_bits: u64, seq_bits: u64) -> u64 {
    unsafe { cljvm_rt_toArray(seq_bits) }
}

/// `(.cast c x)` — Java's `Class.cast`. Returns `x` if `x` is an
/// instance of class `c`; otherwise throws (well, raises) a
/// `ClassCastException`. We use the same isInstance dispatch, then
/// fall through to a panic — runtime cast failures should be loud
/// until try/catch is wired.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_cast(c_bits: u64, x_bits: u64) -> u64 {
    let yes = unsafe { cljvm_inst_isInstance(c_bits, x_bits) };
    if matches!(nanbox_tag(yes), Some(TAG_BOOL)) && nanbox_payload(yes) != 0 {
        return x_bits;
    }
    panic!(
        "clojure-jvm: ClassCastException — `(.cast c x)` failed: \
         x is not an instance of c (c bits 0x{c_bits:x}, x bits 0x{x_bits:x})"
    );
}

/// `(.applyTo f args)` — Java's `IFn.applyTo(ISeq args)`. Walks the
/// args seq, looks up the fn's arity info, and dispatches to the
/// underlying fn pointer with the right shape. No artificial argcount
/// cap — the only limit is the fn's actual signature (matching
/// Clojure: an arity exception comes from the fn, not from `apply`).
///
/// For variadic fns: takes `fixed_arity` args from the seq, packs the
/// rest into a cons-list as the tail, and calls the fn with
/// `fixed_arity + 1` args via [`call_with_packed`].
/// For non-variadic fns: arg count must equal `fixed_arity`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_applyTo(fn_bits: u64, args_bits: u64) -> u64 {
    // Realize ALL args via the generic GC-safe seq walk — RT.cons stores
    // ISeq tails unforced now, so an apply args-chain may carry lazy
    // rests; the old raw cons walk silently truncated at the first one
    // (dropping the variadic tail of e.g. `load-libs`' interleave opts).
    // Forcing runs JIT thunks that may GC, and `fn_bits` is a bare
    // extern-arg copy the collector cannot update — root it in a
    // registered chunk buffer across the walk and re-read it after.
    let fnbuf = std::sync::Arc::new(std::cell::RefCell::new(vec![fn_bits]));
    register_chunk_buffer(&fnbuf);
    let walked: Vec<u64> = match nanbox_tag(args_bits) {
        Some(TAG_NIL) | Some(TAG_PTR) => unsafe { seq_to_items(args_bits) },
        _ => panic!("clojure-jvm: applyTo: args must be a seqable, got bits 0x{args_bits:x}"),
    };
    let fn_bits = fnbuf.borrow()[0];
    deregister_chunk_buffer(&fnbuf);
    if let Some(r) = unsafe { try_multifn_invoke(fn_bits, &walked) } {
        return r;
    }

    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, walked.len()) };
    let info = crate::lang::compiler::with_active_compiler_arity(fref_idx);

    // Build the final arg vector that gets passed to the native fn.
    // Must be rooted because cons-folding the variadic tail can trigger GC.
    let final_args = dynobj::roots::gc_enter(walked.len() + 2, |heap, scope| {
        let arg_roots: Vec<_> = walked.iter().map(|v| scope.root::<()>(*v)).collect();
        match info {
            Some(i) if i.is_variadic => {
                // Fold the variadic rest into a cons-list tail FIRST (this
                // is what triggers GC), then read every rooted value via
                // `get_raw` to assemble `packed` from post-GC addresses.
                let tail_root = scope.root::<()>(nanbox_nil());
                for j in (i.fixed_arity..arg_roots.len()).rev() {
                    let new_tail = heap_cons(heap, arg_roots[j], tail_root);
                    tail_root.set_raw(new_tail);
                }
                let mut packed: Vec<u64> = Vec::with_capacity(walked.len() + 2);
                if let Some(s) = self_arg {
                    packed.push(s);
                }
                // Take fixed_arity args directly.
                for j in 0..i.fixed_arity.min(arg_roots.len()) {
                    packed.push(arg_roots[j].get_raw(&*heap).bits());
                }
                // Pad missing fixed args with nil (matches Clojure's
                // arity behavior for too-few args to a variadic fn —
                // it accepts as long as args.len() >= fixed_arity, but
                // we err on the side of nil-padding for tolerance).
                let self_offset = if self_arg.is_some() { 1 } else { 0 };
                while packed.len() - self_offset < i.fixed_arity {
                    packed.push(nanbox_nil());
                }
                packed.push(tail_root.get_raw(&*heap).bits());
                packed
            }
            _ => {
                // Non-variadic (or unknown arity — assume fixed).
                // Pass args through 1:1; arity mismatch surfaces as a
                // wrong-shape native call at the C ABI boundary. No
                // allocation happens here.
                let mut packed: Vec<u64> = Vec::with_capacity(walked.len() + 2);
                if let Some(s) = self_arg {
                    packed.push(s);
                }
                for r in &arg_roots {
                    packed.push(r.get_raw(&*heap).bits());
                }
                packed
            }
        }
    });
    unsafe { call_with_packed(ptr, &final_args) }
}

/// `(.toString x)` — Java's `Object.toString`. Per Clojure's `(str ...)`
/// docs: nil → "" (handled before this extern is called by `(str x)`),
/// string → string, others → printed representation. We cover what the
/// bootstrap exercises: String, Symbol, Keyword, Long, Double, Bool,
/// Cons (printed as a list), Vector (printed as `[...]`), Nil → "nil".
///
/// Receivers with their own `.toString` (StringBuilder, etc.) get
/// dispatched to their per-type extern based on the heap cell's
/// type_id. This is a pragmatic stand-in for proper type-keyed
/// instance dispatch.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_toString(x_bits: u64) -> u64 {
    let ids = heap_type_ids();
    if let Some(TAG_PTR) = nanbox_tag(x_bits) {
        let p = nanbox_payload(x_bits) as *const u8;
        if !p.is_null() {
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.string_builder {
                return unsafe { cljvm_inst_StringBuilder_toString(x_bits) };
            }
        }
    }
    let obj = any_bits_to_object(x_bits, ids);
    let s = format_object_for_str(&obj);
    unsafe { alloc_string_heap(&s, ids) }
}

fn format_object_for_str(obj: &Object) -> String {
    match obj {
        Object::Nil => String::new(),
        Object::Bool(b) => b.to_string(),
        Object::Long(n) => n.to_string(),
        Object::Char(c) => char::from_u32(*c).map(String::from).unwrap_or_default(),
        Object::Double(d) => {
            // Match Clojure's printer: integral doubles print as "1.0".
            if d.fract() == 0.0 && d.is_finite() {
                format!("{d:.1}")
            } else {
                d.to_string()
            }
        }
        Object::String(s) => (**s).clone(),
        Object::Symbol(s) => match s.get_namespace() {
            Some(ns) => format!("{ns}/{}", s.get_name()),
            None => s.get_name().to_string(),
        },
        Object::Keyword(k) => match k.get_namespace() {
            Some(ns) => format!(":{ns}/{}", k.get_name()),
            None => format!(":{}", k.get_name()),
        },
        Object::List(l) => {
            let items: Vec<String> = l.iter().map(|o| format_object_pr(&o)).collect();
            format!("({})", items.join(" "))
        }
        Object::Vector(v) => {
            let mut items: Vec<String> = Vec::with_capacity(v.count() as usize);
            for i in 0..v.count() {
                items.push(format_object_pr(&v.nth(i)));
            }
            format!("[{}]", items.join(" "))
        }
        Object::Map(m) => {
            // `{k v, k v}`. Iteration order is unspecified (hash order), so
            // this matches Clojure only for single-entry / sorted maps.
            let items: Vec<String> = m
                .iter()
                .map(|(k, v)| format!("{} {}", format_object_pr(&k), format_object_pr(&v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        Object::Set(s) => {
            let items: Vec<String> = s.iter().map(|o| format_object_pr(&o)).collect();
            format!("#{{{}}}", items.join(" "))
        }
        Object::TreeMap(m) => {
            let items: Vec<String> = m
                .iter()
                .map(|(k, v)| format!("{} {}", format_object_pr(&k), format_object_pr(&v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        Object::TreeSet(s) => {
            let items: Vec<String> = s.iter().map(|o| format_object_pr(&o)).collect();
            format!("#{{{}}}", items.join(" "))
        }
        Object::WithMeta(inner, _) => format_object_for_str(inner),
        // Java `Var.toString()`: `#'ns/sym` when interned, else
        // `#<Var: sym-or---unnamed-->`. `(pr-str (def x 5))` must print
        // `#'user/x` like real Clojure, not a Rust Debug dump.
        Object::Var(v) => match (&v.ns, &v.sym) {
            (Some(ns), Some(sym)) => {
                format!("#'{}/{}", ns.name.get_name(), sym.get_name())
            }
            (_, Some(sym)) => format!("#<Var: {}>", sym.get_name()),
            _ => "#<Var: --unnamed-->".to_string(),
        },
        other => format!("{other:?}"),
    }
}

/// Conformance-harness helper: render a result NanBox the way `pr-str`
/// would, for oracle comparison against real Clojure. Host-side printer
/// (does not depend on core's `pr-str`), so it works for any eval result.
/// Heap pointers are decoded via `heap_bits_to_object` so collections /
/// strings / keywords print as their real value, not `#<host>`.
pub fn pr_str_bits(bits: u64) -> String {
    let obj = match nanbox_tag(bits) {
        Some(TAG_PTR) | Some(TAG_FN) => unsafe { heap_bits_to_object(bits, heap_type_ids()) },
        _ => nanbox_to_object(bits),
    };
    format_object_pr(&obj)
}

/// Print-readable form (`pr`-style) — strings get wrapping quotes, etc.
/// Used by `format_object_for_str` when rendering nested elements.
fn format_object_pr(obj: &Object) -> String {
    match obj {
        Object::String(s) => format!("\"{}\"", s),
        Object::Nil => "nil".to_string(),
        // Clojure prints chars with a leading backslash, using the named
        // forms for the non-printing ones.
        Object::Char(c) => match char::from_u32(*c) {
            Some(' ') => "\\space".to_string(),
            Some('\n') => "\\newline".to_string(),
            Some('\t') => "\\tab".to_string(),
            Some('\r') => "\\return".to_string(),
            Some('\u{0c}') => "\\formfeed".to_string(),
            Some('\u{08}') => "\\backspace".to_string(),
            Some(ch) => format!("\\{ch}"),
            None => format!("\\u{c:04x}"),
        },
        _ => format_object_for_str(obj),
    }
}

/// `(.isInstance c x)` — `(instance? c x)` minus the macro layer.
/// `c_bits` must be a TAG_PTR pointing at a `clojure.lang.Class` heap
/// cell whose Raw64 slot holds a `ClassId`. Decodes the id, dispatches
/// through `host_class::is_instance` (each registered class brings its
/// own predicate). Result is a NanBox-encoded bool.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_isInstance(c_bits: u64, x_bits: u64) -> u64 {
    let ids = heap_type_ids();
    match nanbox_tag(c_bits) {
        Some(TAG_PTR) => {
            let raw = nanbox_payload(c_bits) as *const u8;
            if raw.is_null() {
                eprintln!("[cljvm-stub] (.isInstance nil x) — null Class — false");
                return nanbox_bool(false);
            }
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id != ids.class {
                eprintln!(
                    "[cljvm-stub] (.isInstance c x) — receiver type_id \
                     {type_id} is not a Class — false"
                );
                return nanbox_bool(false);
            }
            let raw_id = unsafe { raw.add(8).cast::<u64>().read_unaligned() };
            let class_id = crate::lang::host_class::ClassId(raw_id as u16);
            let yes = crate::lang::host_class::is_instance(class_id, x_bits, ids);
            nanbox_bool(yes)
        }
        _ => {
            eprintln!(
                "[cljvm-stub] (.isInstance c x) — receiver bits 0x{c_bits:x} \
                 not a Class heap pointer — false"
            );
            nanbox_bool(false)
        }
    }
}

/// `clojure.lang.Compiler$HostExpr/maybeSpecialTag(Symbol tag)` — Java
/// returns one of the primitive Class constants (`long`, `double`,
/// `int`, …) when `tag` matches a special primitive name, else null.
/// We don't model primitive special tags, so always return nil — the
/// caller (defn's `sigs`) treats nil as "fall through to maybeClass".
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_compiler_hostexpr_maybeSpecialTag(_tag_bits: u64) -> u64 {
    nanbox_nil()
}

/// `clojure.lang.Compiler$HostExpr/maybeClass(Object form, boolean stringOk)` —
/// look up a class by symbol/string name. Returns nil when not found.
/// We use our `host_class` registry; symbols and strings (when stringOk
/// is true) both map to the same canonical name.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_compiler_hostexpr_maybeClass(
    form_bits: u64,
    _string_ok_bits: u64,
) -> u64 {
    let ids = heap_type_ids();
    // Decode the form to an Object so we can normalize symbol/string.
    let obj = any_bits_to_object(form_bits, ids);
    let bare = obj.peel_meta_ref();
    let name: String = match bare {
        Object::Symbol(s) => match s.get_namespace() {
            Some(ns) => format!("{ns}.{}", s.get_name()),
            None => s.get_name().to_string(),
        },
        Object::String(s) => (**s).clone(),
        _ => return nanbox_nil(),
    };
    match crate::lang::host_class::lookup(&name) {
        Some(info) => {
            // Allocate a Class heap cell. Mirrors alloc_class_as_nanbox
            // in compiler.rs — kept inline here so we don't need to
            // thread an &Compiler into runtime.
            let raw = dynlang::gc::gc_alloc_thunk(ids.class as u64, 0);
            let ptr = raw as *mut u8;
            if ptr.is_null() {
                panic!("clojure-jvm: maybeClass: gc_alloc returned null");
            }
            unsafe {
                ptr.add(8).cast::<u64>().write_unaligned(info.id.0 as u64);
            }
            nanbox_ptr(raw)
        }
        None => nanbox_nil(),
    }
}

/// `clojure.lang.Symbol/intern(String name)` — single-argument
/// interning of a Symbol. Receiver is the class itself; we model that
/// via the `Symbol/intern` static-method dispatch in `parse_dot_form`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_keyword_intern_1(name_bits: u64) -> u64 {
    let ids = heap_type_ids();
    // Accept either a String (raw heap String) or a Symbol heap cell.
    let kw = match nanbox_tag(name_bits) {
        Some(TAG_PTR) => {
            let p = nanbox_payload(name_bits) as *const u8;
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.string {
                let s = unsafe { read_string_heap(name_bits, ids, "Keyword/intern: name arg") };
                crate::lang::keyword::Keyword::intern_nsname(s)
            } else if tid == ids.symbol {
                let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::symbol::Symbol;
                unsafe { std::sync::Arc::increment_strong_count(arc_ptr) };
                let sym = unsafe { std::sync::Arc::from_raw(arc_ptr) };
                crate::lang::keyword::Keyword::intern(sym)
            } else {
                panic!(
                    "clojure-jvm: Keyword/intern arity-1: receiver type_id {tid} not String or Symbol"
                );
            }
        }
        _ => panic!("clojure-jvm: Keyword/intern arity-1 needs a String or Symbol"),
    };
    let raw = dynlang::gc::gc_alloc_thunk(ids.keyword as u64, 0);
    let p = raw as *mut u8;
    let arc_bits = std::sync::Arc::as_ptr(&kw) as u64;
    // Globally interned — no Session-roots push required (the keyword
    // table keeps the Arc alive for the program's lifetime).
    std::mem::forget(kw);
    unsafe {
        p.add(8).cast::<u64>().write_unaligned(arc_bits);
    }
    nanbox_ptr(raw)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_keyword_find_1(name_bits: u64) -> u64 {
    // Java has overloads for Symbol and String. We accept both like
    // `Keyword/intern` arity-1 does.
    let ids = heap_type_ids();
    let sym_arc = match nanbox_tag(name_bits) {
        Some(TAG_PTR) => {
            let p = nanbox_payload(name_bits) as *const u8;
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.symbol {
                let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::symbol::Symbol;
                unsafe { std::sync::Arc::increment_strong_count(arc_ptr) };
                unsafe { std::sync::Arc::from_raw(arc_ptr) }
            } else if tid == ids.string {
                let s = unsafe { read_string_heap(name_bits, ids, "Keyword/find: name arg") };
                crate::lang::symbol::Symbol::intern(s)
            } else {
                panic!(
                    "clojure-jvm: Keyword/find arity-1: receiver type_id {tid} not Symbol or String"
                );
            }
        }
        _ => panic!("clojure-jvm: Keyword/find arity-1 needs a Symbol or String"),
    };
    let kw = match crate::lang::keyword::Keyword::find(&sym_arc) {
        Some(k) => k,
        None => return nanbox_nil(),
    };
    let raw = dynlang::gc::gc_alloc_thunk(ids.keyword as u64, 0);
    let p = raw as *mut u8;
    let arc_bits = std::sync::Arc::as_ptr(&kw) as u64;
    std::mem::forget(kw);
    unsafe {
        p.add(8).cast::<u64>().write_unaligned(arc_bits);
    }
    nanbox_ptr(raw)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_keyword_find_2(ns_bits: u64, name_bits: u64) -> u64 {
    // find-keyword: returns the keyword if already interned, nil otherwise.
    let ids = heap_type_ids();
    let ns_str = unsafe { read_string_heap(ns_bits, ids, "Keyword/find: ns arg") };
    let name = unsafe { read_string_heap(name_bits, ids, "Keyword/find: name arg") };
    let sym = crate::lang::symbol::Symbol::intern_ns_name(Some(ns_str), name);
    let kw = match crate::lang::keyword::Keyword::find(&sym) {
        Some(k) => k,
        None => return nanbox_nil(),
    };
    let raw = dynlang::gc::gc_alloc_thunk(ids.keyword as u64, 0);
    let p = raw as *mut u8;
    let arc_bits = std::sync::Arc::as_ptr(&kw) as u64;
    std::mem::forget(kw);
    unsafe {
        p.add(8).cast::<u64>().write_unaligned(arc_bits);
    }
    nanbox_ptr(raw)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_keyword_intern_2(ns_bits: u64, name_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let ns_str = unsafe { read_string_heap(ns_bits, ids, "Keyword/intern: ns arg") };
    let name = unsafe { read_string_heap(name_bits, ids, "Keyword/intern: name arg") };
    let kw = crate::lang::keyword::Keyword::intern_ns_name(Some(ns_str), name);
    let raw = dynlang::gc::gc_alloc_thunk(ids.keyword as u64, 0);
    let p = raw as *mut u8;
    let arc_bits = std::sync::Arc::as_ptr(&kw) as u64;
    std::mem::forget(kw);
    unsafe {
        p.add(8).cast::<u64>().write_unaligned(arc_bits);
    }
    nanbox_ptr(raw)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_symbol_intern_2(ns_bits: u64, name_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let ns_str = unsafe { read_string_heap(ns_bits, ids, "Symbol/intern: ns arg") };
    let name = unsafe { read_string_heap(name_bits, ids, "Symbol/intern: name arg") };
    let s = crate::lang::symbol::Symbol::intern_ns_name(Some(ns_str), name);
    crate::lang::compiler::with_active_session_root_symbol(s.clone());
    let raw = dynlang::gc::gc_alloc_thunk(ids.symbol as u64, 0);
    let ptr = raw as *mut u8;
    if ptr.is_null() {
        panic!("clojure-jvm: Symbol/intern: gc_alloc returned null");
    }
    let arc_bits = std::sync::Arc::as_ptr(&s) as u64;
    unsafe {
        ptr.add(8).cast::<u64>().write_unaligned(arc_bits);
    }
    nanbox_ptr(raw)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_symbol_intern_1(name_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let name = read_string_heap(name_bits, ids, "Symbol/intern: name arg");
    let s = crate::lang::symbol::Symbol::intern(name);
    // Allocate a Symbol heap cell holding `Arc::as_ptr(s)`. Mirrors
    // `alloc_symbol` (compiler.rs) — root the Arc on the active
    // Session so the pointer outlives the heap cell.
    crate::lang::compiler::with_active_session_root_symbol(s.clone());
    let raw = dynlang::gc::gc_alloc_thunk(ids.symbol as u64, 0);
    let ptr = raw as *mut u8;
    if ptr.is_null() {
        panic!("clojure-jvm: Symbol/intern: gc_alloc returned null");
    }
    let arc_bits = std::sync::Arc::as_ptr(&s) as u64;
    unsafe {
        ptr.add(8).cast::<u64>().write_unaligned(arc_bits);
    }
    nanbox_ptr(raw)
}

/// `(.toSymbol v)` — Java's `Var.toSymbol`. Returns the Var's symbol.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_toSymbol(v_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let p = nanbox_payload(v_bits) as *const u8;
    if p.is_null() || nanbox_tag(v_bits) != Some(TAG_PTR) {
        panic!("clojure-jvm: (.toSymbol v): receiver is not a Var");
    }
    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
    if tid != ids.var {
        panic!("clojure-jvm: (.toSymbol v): receiver type_id {tid} is not a Var");
    }
    let var_ptr =
        unsafe { p.add(8).cast::<u64>().read_unaligned() } as *const crate::lang::var::Var;
    let v: &crate::lang::var::Var = unsafe { &*var_ptr };
    let sym = v.sym.as_ref().expect("Var.sym is None").clone();
    crate::lang::compiler::with_active_session_root_symbol(sym.clone());
    let raw = dynlang::gc::gc_alloc_thunk(ids.symbol as u64, 0);
    let sp = raw as *mut u8;
    let arc_bits = std::sync::Arc::as_ptr(&sym) as u64;
    unsafe {
        sp.add(8).cast::<u64>().write_unaligned(arc_bits);
    }
    nanbox_ptr(raw)
}

/// `(.sym k)` — Java's `Keyword.sym`. Returns the keyword's underlying Symbol.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_sym(k_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let p = nanbox_payload(k_bits) as *const u8;
    if p.is_null() || nanbox_tag(k_bits) != Some(TAG_PTR) {
        panic!("clojure-jvm: (.sym k): receiver is not a Keyword");
    }
    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
    if tid != ids.keyword {
        panic!("clojure-jvm: (.sym k): receiver type_id {tid} is not a Keyword");
    }
    let arc_ptr =
        unsafe { p.add(8).cast::<u64>().read_unaligned() } as *const crate::lang::keyword::Keyword;
    unsafe { std::sync::Arc::increment_strong_count(arc_ptr) };
    let kw = unsafe { std::sync::Arc::from_raw(arc_ptr) };
    let sym = kw.sym.clone();
    crate::lang::compiler::with_active_session_root_symbol(sym.clone());
    let raw = dynlang::gc::gc_alloc_thunk(ids.symbol as u64, 0);
    let sp = raw as *mut u8;
    let arc_bits = std::sync::Arc::as_ptr(&sym) as u64;
    unsafe {
        sp.add(8).cast::<u64>().write_unaligned(arc_bits);
    }
    nanbox_ptr(raw)
}

// ── Namespace / Var reflective accessors (the `refer` chain) ─────────────
//
// These are the host primitives that `clojure.core`'s `refer` / `ns-map` /
// `ns-publics` / `the-ns` / `find-ns` bottom out in. With them, the `ns`
// macro's `(refer 'clojure.core)` runs end-to-end exactly as in Clojure.

/// Decode a `clojure.lang.Namespace` heap cell to its `Arc<Namespace>`.
unsafe fn namespace_from_bits(
    bits: u64,
    ctx: &str,
) -> std::sync::Arc<crate::lang::namespace::Namespace> {
    let ids = heap_type_ids();
    let p = nanbox_payload(bits) as *const u8;
    if p.is_null() || nanbox_tag(bits) != Some(TAG_PTR) {
        panic!("clojure-jvm: {ctx}: receiver is not a Namespace (bits 0x{bits:x})");
    }
    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
    if tid != ids.namespace {
        panic!("clojure-jvm: {ctx}: receiver type_id {tid} is not a Namespace");
    }
    unsafe { decode_arc_cell(p) }
}

/// Decode a `clojure.lang.Var` heap cell to its `Arc<Var>`.
unsafe fn var_from_bits(bits: u64, ctx: &str) -> std::sync::Arc<crate::lang::var::Var> {
    let ids = heap_type_ids();
    let p = nanbox_payload(bits) as *const u8;
    if p.is_null() || nanbox_tag(bits) != Some(TAG_PTR) {
        panic!("clojure-jvm: {ctx}: receiver is not a Var (bits 0x{bits:x})");
    }
    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
    if tid != ids.var {
        panic!("clojure-jvm: {ctx}: receiver type_id {tid} is not a Var");
    }
    unsafe { decode_arc_cell(p) }
}

/// `(.ns v)` — Java `Var.ns`. Returns the Var's home namespace (or nil).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_ns(v_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let var = unsafe { var_from_bits(v_bits, "(.ns v)") };
    match &var.ns {
        Some(ns) => unsafe {
            alloc_arc_cell(
                ids.namespace,
                ns.clone(),
                crate::lang::compiler::with_active_session_root_namespace,
            )
        },
        None => nanbox_nil(),
    }
}

/// `(.isPublic v)` — Java `Var.isPublic`. We don't track var privacy
/// (`:private` meta) yet, so every interned var reports public. Referring a
/// private core var is harmless: a local `def` still shadows it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_isPublic(v_bits: u64) -> u64 {
    let _var = unsafe { var_from_bits(v_bits, "(.isPublic v)") };
    nanbox_bool(true)
}

/// `(.getMappings ns)` — Java `Namespace.getMappings`. Returns a Clojure map
/// (Symbol → Var) snapshot of the namespace's bindings. Backs `ns-map`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getMappings(ns_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let ns = unsafe { namespace_from_bits(ns_bits, "(.getMappings ns)") };
    let pairs: Vec<(Object, Object)> = ns
        .mappings_snapshot()
        .into_iter()
        .map(|(k, v)| (Object::Symbol(k), v))
        .collect();
    let m = crate::lang::persistent_hash_map::PersistentHashMap::create_pairs(pairs);
    let raw_arc = std::sync::Arc::as_ptr(&m) as u64;
    crate::lang::compiler::with_active_session_root_map(m);
    let new_raw = dynlang::gc::gc_alloc_thunk(ids.map as u64, 0);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: (.getMappings ns): gc_alloc returned null for Map");
    }
    unsafe {
        new_ptr.add(8).cast::<u64>().write_unaligned(raw_arc);
    }
    nanbox_encode(TAG_PTR, new_raw & PAYLOAD_MASK)
}

/// `(. ns (refer sym var))` — Java `Namespace.refer(Symbol, Var)`. Adds the
/// mapping `sym -> var` to `ns`. Returns nil. The primitive at the bottom of
/// `clojure.core/refer`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_refer(ns_bits: u64, sym_bits: u64, var_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let ns = unsafe { namespace_from_bits(ns_bits, "(. ns (refer sym var))") };
    let sym = match any_bits_to_object(sym_bits, ids) {
        Object::Symbol(s) => s,
        other => panic!("clojure-jvm: Namespace.refer: sym arg is not a Symbol: {other:?}"),
    };
    let var = unsafe { var_from_bits(var_bits, "Namespace.refer: var arg") };
    ns.refer(sym, var);
    nanbox_nil()
}

/// `clojure.lang.Namespace/find(Symbol)` — static. Returns the named
/// namespace's heap cell, or nil. Backs `find-ns`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_ns_find(sym_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let sym = match any_bits_to_object(sym_bits, ids) {
        Object::Symbol(s) => s,
        other => panic!("clojure-jvm: Namespace/find: arg is not a Symbol: {other:?}"),
    };
    match crate::lang::namespace::Namespace::find(&sym) {
        Some(ns) => unsafe {
            alloc_arc_cell(
                ids.namespace,
                ns,
                crate::lang::compiler::with_active_session_root_namespace,
            )
        },
        None => nanbox_nil(),
    }
}

/// `clojure.lang.Namespace/findOrCreate(Symbol)` — static. Returns the named
/// namespace's heap cell, creating it if absent. Backs `create-ns`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_ns_findOrCreate(sym_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let sym = match any_bits_to_object(sym_bits, ids) {
        Object::Symbol(s) => s,
        other => {
            panic!("clojure-jvm: Namespace/findOrCreate: arg is not a Symbol: {other:?}")
        }
    };
    let ns = crate::lang::namespace::Namespace::find_or_create(sym);
    unsafe {
        alloc_arc_cell(
            ids.namespace,
            ns,
            crate::lang::compiler::with_active_session_root_namespace,
        )
    }
}

/// `(.setMacro v)` — flag the Var `v` as a macro. `v_bits` is a
/// `clojure.lang.Var` heap pointer; we read the `*const Var` Raw64
/// and call `Var.set_macro()` directly. Returns nil.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_setMacro(v_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let raw = match nanbox_tag(v_bits) {
        Some(TAG_PTR) => nanbox_payload(v_bits) as *const u8,
        _ => panic!("clojure-jvm: (.setMacro v) — receiver is not a Var heap pointer"),
    };
    if raw.is_null() {
        panic!("clojure-jvm: (.setMacro nil) — null receiver");
    }
    let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
    if type_id != ids.var {
        panic!("clojure-jvm: (.setMacro v) — receiver type_id {type_id} is not a Var");
    }
    let var_ptr =
        unsafe { raw.add(8).cast::<u64>().read_unaligned() } as *const crate::lang::var::Var;
    let v: &crate::lang::var::Var = unsafe { &*var_ptr };
    v.set_macro();
    nanbox_nil()
}

/// `(.getName x)` — name accessor. Java's `Named.getName()` is the
/// receiver-polymorphic extractor used on Symbol, Keyword, and Var.
/// Returns a fresh `String` heap value containing the name. Receivers
/// without a name field panic loudly.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getName(recv_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let raw = match nanbox_tag(recv_bits) {
        Some(TAG_PTR) => nanbox_payload(recv_bits) as *const u8,
        _ => panic!(
            "clojure-jvm: (.getName x) — receiver bits 0x{recv_bits:x} \
             is not a heap value with a name"
        ),
    };
    if raw.is_null() {
        panic!("clojure-jvm: (.getName nil) — null receiver");
    }
    let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
    if type_id == ids.namespace {
        // `clojure.lang.Namespace.getName()` returns the namespace's name
        // *Symbol* (not a String like Symbol/Keyword's getName). Backs
        // `(defn ns-name [ns] (.getName (the-ns ns)))`. We box the name
        // Symbol via the active session's encoder (GC-safe: it decodes a
        // Rust-owned `Object` before allocating, so no NanBox pointer is
        // held across the alloc).
        let arc_ptr = unsafe { raw.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::namespace::Namespace;
        let ns: &crate::lang::namespace::Namespace = unsafe { &*arc_ptr };
        return crate::lang::compiler::with_active_session_encode_object(&Object::Symbol(
            ns.name.clone(),
        ));
    }
    let name: String = if type_id == ids.symbol {
        let arc_ptr = unsafe { raw.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::symbol::Symbol;
        let s: &crate::lang::symbol::Symbol = unsafe { &*arc_ptr };
        s.get_name().to_string()
    } else if type_id == ids.keyword {
        let arc_ptr = unsafe { raw.add(8).cast::<u64>().read_unaligned() }
            as *const crate::lang::keyword::Keyword;
        let k: &crate::lang::keyword::Keyword = unsafe { &*arc_ptr };
        k.get_name().to_string()
    } else if type_id == ids.with_meta {
        // Metadata wrapper (e.g. defmulti's ^{:doc …} name symbol) —
        // names ignore meta; recurse on the inner value.
        let inner = unsafe { raw.add(8).cast::<u64>().read_unaligned() };
        return unsafe { cljvm_inst_getName(inner) };
    } else {
        let decoded = unsafe { heap_bits_to_object(recv_bits, ids) };
        panic!(
            "clojure-jvm: (.getName x) — receiver type_id {type_id} has no \
             name (extend cljvm_inst_getName); decoded receiver = {decoded:?}"
        );
    };
    alloc_string_heap(&name, ids)
}

/// Allocate a fresh `clojure.lang.String` heap cell holding `s`'s
/// bytes; returns its NanBox handle. Helper for runtime externs that
/// produce String values (`.getName`, `.concat`, …).
unsafe fn alloc_string_heap(s: &str, ids: HeapTypeIds) -> u64 {
    let bytes = s.as_bytes();
    let raw = dynlang::gc::gc_alloc_thunk(ids.string as u64, bytes.len() as u64);
    let ptr = raw as *mut u8;
    if ptr.is_null() {
        panic!("clojure-jvm: alloc_string_heap: gc_alloc returned null");
    }
    // String layout: header(8) + count word(8) + bytes.
    unsafe {
        ptr.add(16)
            .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
    }
    nanbox_ptr(raw)
}

/// `(.concat a b)` — String concatenation. Both `a` and `b` must be
/// `clojure.lang.String` heap values; result is a fresh String.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_concat(a_bits: u64, b_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let a = read_string_heap(a_bits, ids, "(.concat a b) — first");
    let b = read_string_heap(b_bits, ids, "(.concat a b) — second");
    let mut out = String::with_capacity(a.len() + b.len());
    out.push_str(a);
    out.push_str(b);
    alloc_string_heap(&out, ids)
}

/// `(.indexOf s sub)` — index of `sub` within `s`, or -1 if absent.
/// Java overload: also accepts a single char argument; we only handle
/// String-String since that's what `clojure.core` uses on `Symbol`-name
/// strings ("." / "/" lookups).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_indexOf(s_bits: u64, sub_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = read_string_heap(s_bits, ids, "(.indexOf s sub) — first");
    let sub = read_string_heap(sub_bits, ids, "(.indexOf s sub) — second");
    let n: i64 = match s.find(sub) {
        Some(i) => i as i64,
        None => -1,
    };
    box_long(n)
}

/// `(.startsWith s prefix)` — true iff String `s` begins with String
/// `prefix`. Both args must be `clojure.lang.String` heap values. An empty
/// `prefix` returns true (Java semantics). Used by `clojure.core`'s load to
/// test path prefixes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_startsWith(s_bits: u64, prefix_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = read_string_heap(s_bits, ids, "(.startsWith s prefix) — receiver");
    let prefix = read_string_heap(prefix_bits, ids, "(.startsWith s prefix) — prefix");
    nanbox_bool(s.starts_with(prefix))
}

/// `(.substring s start)` — the suffix of String `s` from byte index
/// `start` to the end. `start` is a boxed Long. Matches Java's
/// `String.substring(int)`: a `start` past the end yields the empty string;
/// a negative `start` is treated as an error (empty string). Used by
/// `clojure.core`'s load to strip a leading slash.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_substring(s_bits: u64, start_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = read_string_heap(s_bits, ids, "(.substring s start) — receiver");
    if !is_boxed_long(start_bits) {
        eprintln!(
            "[cljvm-stub] (.substring s start) — start 0x{start_bits:x} is not a \
             boxed Long, returning empty string"
        );
        return alloc_string_heap("", ids);
    }
    let start = unbox_long(start_bits);
    if start < 0 {
        eprintln!(
            "[cljvm-stub] (.substring s start) — negative start {start}, returning \
             empty string"
        );
        return alloc_string_heap("", ids);
    }
    let start = start as usize;
    let out: String = if start >= s.len() {
        String::new()
    } else {
        // Byte-index semantics (Java String is char-indexed, but clojure.core's
        // usage here only strips a leading ASCII '/'; byte == char index there).
        s[start..].to_string()
    };
    alloc_string_heap(&out, ids)
}

/// `(.substring s start end)` — the substring of String `s` between byte
/// indices `start` (inclusive) and `end` (exclusive). Matches Java's
/// `String.substring(int,int)`, which `clojure.core/subs` lowers to. Both
/// indices arrive as boxed Longs. Out-of-range / inverted ranges clamp to
/// the empty string (we panic only on non-Long args).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_substring2(s_bits: u64, start_bits: u64, end_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = read_string_heap(s_bits, ids, "(.substring s start end) — receiver");
    if !is_boxed_long(start_bits) || !is_boxed_long(end_bits) {
        panic!(
            "clojure-jvm: (.substring s start end) — indices must be boxed Longs \
             (got start 0x{start_bits:x}, end 0x{end_bits:x})"
        );
    }
    let start = unbox_long(start_bits);
    let end = unbox_long(end_bits);
    // Byte-index semantics (clojure.core's path usage is ASCII).
    let len = s.len() as i64;
    if start < 0 || end > len || start > end {
        return alloc_string_heap("", ids);
    }
    let out = &s[start as usize..end as usize];
    alloc_string_heap(out, ids)
}

/// `(.toUpperCase s)` — uppercased String. Backs `clojure.string/upper-case`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_toUpperCase(s_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = read_string_heap(s_bits, ids, "(.toUpperCase s)");
    alloc_string_heap(&s.to_uppercase(), ids)
}
/// `(.toLowerCase s)` — lowercased String. Backs `clojure.string/lower-case`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_toLowerCase(s_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = read_string_heap(s_bits, ids, "(.toLowerCase s)");
    alloc_string_heap(&s.to_lowercase(), ids)
}
/// `(.trim s)` — String with leading/trailing ASCII whitespace removed.
/// Backs `clojure.string/trim`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_trim(s_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = read_string_heap(s_bits, ids, "(.trim s)");
    alloc_string_heap(s.trim(), ids)
}
/// Reverse a String by Unicode scalar. Backs `clojure.string/reverse`
/// (registered as a static so the shim doesn't need char-seq machinery).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_str_reverse(s_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = read_string_heap(s_bits, ids, "string/reverse");
    let r: String = s.chars().rev().collect();
    alloc_string_heap(&r, ids)
}
/// Left-trim whitespace. Backs `clojure.string/triml`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_str_triml(s_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = read_string_heap(s_bits, ids, "string/triml");
    alloc_string_heap(s.trim_start(), ids)
}
/// Right-trim whitespace. Backs `clojure.string/trimr`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_str_trimr(s_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = read_string_heap(s_bits, ids, "string/trimr");
    alloc_string_heap(s.trim_end(), ids)
}

/// `clojure.string.Native/split(s, re)` — split String `s` on regex `re`,
/// returning a Vector of Strings. Backs `clojure.string/split` (the shim
/// passes the reader's regex literal through; our reader models `#"…"` as
/// a plain String of the pattern source). Mirrors Java's
/// `Pattern.split(s)` with limit 0: if the pattern never matches the
/// whole input is returned as a single element, otherwise trailing empty
/// strings are removed (possibly leaving an empty vector). The pattern is
/// compiled by the Rust `regex` crate — the common Java syntax subset
/// (literals, classes, `\d`/`\s`/`\w`, anchors, quantifiers, alternation)
/// matches; an unsupported pattern panics loudly with the compile error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_string_split(s_bits: u64, re_bits: u64) -> u64 {
    let ids = heap_type_ids();
    // Copy both heap strings into Rust-owned values BEFORE any allocation:
    // read_string_heap returns a borrow into the (movable) heap cell.
    let s: String = read_string_heap(s_bits, ids, "clojure.string/split — s").to_string();
    let pat: String = read_string_heap(re_bits, ids, "clojure.string/split — re").to_string();
    let re = regex::Regex::new(&pat).unwrap_or_else(|e| {
        panic!("clojure-jvm: clojure.string/split — cannot compile regex #\"{pat}\": {e}")
    });
    let mut parts: Vec<String> = re.split(&s).map(|p| p.to_string()).collect();
    if parts.len() > 1 {
        while parts.last().map(|p| p.is_empty()).unwrap_or(false) {
            parts.pop();
        }
    }
    // Allocate each piece as a String cell, rooting the accumulated cells
    // across subsequent allocations, then build the result Vector.
    let n = parts.len();
    dynobj::roots::gc_enter(n + 1, |heap, scope| {
        let mut roots: Vec<dynobj::roots::Rooted<()>> = Vec::with_capacity(n);
        for p in &parts {
            let bits = unsafe { alloc_string_heap(p, ids) };
            roots.push(scope.root::<()>(bits));
        }
        let cell = heap_alloc(heap, ids.vector as u64, n as u64).root(scope);
        let cell_bits = cell.get_raw(&*heap).bits();
        let new_ptr = nanbox_payload(cell_bits) as *mut u8;
        for (i, r) in roots.iter().enumerate() {
            unsafe {
                write_slot(new_ptr, 16 + i * 8, r.get_raw(&*heap));
            }
        }
        cell_bits
    })
}

// ── java.util.regex.Pattern / Matcher ─────────────────────────────────
//
// Our reader models `#"…"` as a plain String of the pattern source, and
// `re-pattern` treats Strings as already-compiled Patterns (the
// registered `java.util.regex.Pattern` class predicate accepts strings).
// `(.matcher re s)` builds a Rust-side `MatcherState` (regex crate);
// `.find`/`.matches`/`.group`/`.groupCount` drive it, which makes the
// upstream `re-find`/`re-seq`/`re-matches`/`re-groups` definitions work
// as written.

/// Rust-side state behind a `java.util.regex.Matcher` cell. Holds owned
/// copies only (no heap bits), so no GC root registry is needed.
pub struct MatcherState {
    re: regex::Regex,
    /// Lazily-built fully-anchored variant for `.matches`.
    full_re: Option<regex::Regex>,
    hay: String,
    pos: usize,
    /// Byte ranges of the last successful match's groups (0 = whole).
    groups: Option<Vec<Option<(usize, usize)>>>,
}

fn compile_regex(pat: &str) -> regex::Regex {
    regex::Regex::new(pat).unwrap_or_else(|e| {
        panic!("clojure-jvm: cannot compile regex #\"{pat}\": {e}")
    })
}

/// `java.util.regex.Pattern/compile(s)` — patterns ARE strings here;
/// validate eagerly (so a bad pattern fails at compile-site like Java)
/// and return the string unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_pattern_compile(s_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let pat = read_string_heap(s_bits, ids, "Pattern/compile");
    let _ = compile_regex(pat);
    s_bits
}

/// `(.matcher re s)` — allocate a Matcher cell.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_matcher(re_bits: u64, s_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let pat = read_string_heap(re_bits, ids, "(.matcher re s) — pattern").to_string();
    let hay = read_string_heap(s_bits, ids, "(.matcher re s) — input").to_string();
    let st = std::sync::Arc::new(std::cell::RefCell::new(MatcherState {
        re: compile_regex(&pat),
        full_re: None,
        hay,
        pos: 0,
        groups: None,
    }));
    unsafe {
        alloc_arc_cell(
            ids.matcher,
            st,
            crate::lang::compiler::with_active_session_root_matcher,
        )
    }
}

unsafe fn decode_matcher(recv: u64, ctx: &str) -> std::sync::Arc<std::cell::RefCell<MatcherState>> {
    let ids = heap_type_ids();
    if let Some(TAG_PTR) = nanbox_tag(recv) {
        let raw = nanbox_payload(recv) as *const u8;
        if !raw.is_null() {
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.matcher {
                return unsafe { decode_arc_cell(raw) };
            }
        }
    }
    panic!("clojure-jvm: {ctx} — receiver is not a Matcher (bits 0x{recv:x})");
}

/// Owned snapshot of a match: per-group byte ranges (0 = whole).
fn capture_ranges(caps: &regex::Captures<'_>) -> Vec<Option<(usize, usize)>> {
    (0..caps.len())
        .map(|i| caps.get(i).map(|m| (m.start(), m.end())))
        .collect()
}

fn record_captures(st: &mut MatcherState, ranges: Vec<Option<(usize, usize)>>) {
    let (start, end) = ranges[0].expect("group 0 always present");
    st.groups = Some(ranges);
    // Advance past the match; an empty match advances one char so the
    // scan terminates (Java does the same).
    st.pos = if end > start {
        end
    } else {
        let mut p = end + 1;
        while p < st.hay.len() && !st.hay.is_char_boundary(p) {
            p += 1;
        }
        p
    };
}

/// `(.find m)` — advance to the next match; true on success.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_find_next(recv: u64) -> u64 {
    let cell = unsafe { decode_matcher(recv, "(.find m)") };
    let mut st = cell.borrow_mut();
    if st.pos > st.hay.len() {
        st.groups = None;
        return nanbox_bool(false);
    }
    let ranges = {
        let MatcherState { re, hay, pos, .. } = &*st;
        re.captures_at(hay, *pos).map(|c| capture_ranges(&c))
    };
    match ranges {
        Some(r) => {
            record_captures(&mut st, r);
            nanbox_bool(true)
        }
        None => {
            st.groups = None;
            st.pos = st.hay.len() + 1;
            nanbox_bool(false)
        }
    }
}

/// `(.matches m)` — does the regex match the ENTIRE input?
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_matches(recv: u64) -> u64 {
    let cell = unsafe { decode_matcher(recv, "(.matches m)") };
    let mut st = cell.borrow_mut();
    if st.full_re.is_none() {
        let anchored = format!(r"\A(?:{})\z", st.re.as_str());
        st.full_re = Some(compile_regex(&anchored));
    }
    let ranges = {
        let MatcherState { full_re, hay, .. } = &*st;
        full_re.as_ref().unwrap().captures(hay).map(|c| capture_ranges(&c))
    };
    match ranges {
        Some(r) => {
            record_captures(&mut st, r);
            nanbox_bool(true)
        }
        None => {
            st.groups = None;
            nanbox_bool(false)
        }
    }
}

unsafe fn matcher_group(recv: u64, idx: usize) -> u64 {
    let ids = heap_type_ids();
    let cell = unsafe { decode_matcher(recv, "(.group m i)") };
    let piece: Option<String> = {
        let st = cell.borrow();
        let groups = st
            .groups
            .as_ref()
            .unwrap_or_else(|| panic!("clojure-jvm: (.group m) — no match available"));
        groups
            .get(idx)
            .unwrap_or_else(|| panic!("clojure-jvm: (.group m {idx}) — no such group"))
            .map(|(a, b)| st.hay[a..b].to_string())
    };
    match piece {
        Some(p) => unsafe { alloc_string_heap(&p, ids) },
        None => nanbox_nil(),
    }
}

/// `(.group m)` — the whole last match.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_group_0(recv: u64) -> u64 {
    unsafe { matcher_group(recv, 0) }
}
/// `(.group m i)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_group_i(recv: u64, idx_bits: u64) -> u64 {
    unsafe { matcher_group(recv, arg_to_i64(idx_bits) as usize) }
}
/// `(.groupCount m)` — number of capture groups (excluding group 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_groupCount(recv: u64) -> u64 {
    let cell = unsafe { decode_matcher(recv, "(.groupCount m)") };
    let n = cell.borrow().re.captures_len() - 1;
    unsafe { box_long(n as i64) }
}

/// `String/format(fmt, args-array)` — a printf subset (`%s` `%d` `%f`
/// `%x` `%%`) sufficient for `clojure.core/format`'s common uses. The
/// args arrive as our Vector (to-array's result). Unsupported directives
/// panic loudly.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_string_format(fmt_bits: u64, args_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let fmt = read_string_heap(fmt_bits, ids, "String/format — fmt").to_string();
    // Snapshot arg bits out of the vector (no allocation between the
    // reads and each use's decode — decodes don't allocate; only the
    // final string alloc does, after all reads).
    let mut args: Vec<u64> = Vec::new();
    if let Some(TAG_PTR) = nanbox_tag(args_bits) {
        let raw = nanbox_payload(args_bits) as *const u8;
        if !raw.is_null() {
            let tid = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.vector {
                let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
                for i in 0..n {
                    args.push(unsafe { raw.add(16 + i * 8).cast::<u64>().read_unaligned() });
                }
            }
        }
    }
    let mut out = String::new();
    let mut ai = 0;
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('%') => out.push('%'),
            Some('s') => {
                let a = args.get(ai).copied().unwrap_or_else(|| {
                    panic!("clojure-jvm: format — missing argument for %s")
                });
                ai += 1;
                let obj = any_bits_to_object(a, ids);
                out.push_str(&format_object_for_str(&obj));
            }
            Some('d') => {
                let a = args.get(ai).copied().unwrap_or_else(|| {
                    panic!("clojure-jvm: format — missing argument for %d")
                });
                ai += 1;
                out.push_str(&arg_to_i64(a).to_string());
            }
            Some('f') => {
                let a = args.get(ai).copied().unwrap_or_else(|| {
                    panic!("clojure-jvm: format — missing argument for %f")
                });
                ai += 1;
                out.push_str(&format!("{:.6}", arg_to_f64(a)));
            }
            Some('x') => {
                let a = args.get(ai).copied().unwrap_or_else(|| {
                    panic!("clojure-jvm: format — missing argument for %x")
                });
                ai += 1;
                out.push_str(&format!("{:x}", arg_to_i64(a)));
            }
            other => panic!(
                "clojure-jvm: format — unsupported directive %{} (extend cljvm_string_format)",
                other.map(String::from).unwrap_or_default()
            ),
        }
    }
    unsafe { alloc_string_heap(&out, ids) }
}

// ── Transducers: TransformerIterator / sequence ───────────────────────

/// `clojure.lang.RT/iter(coll)` — Java returns an Iterator; our
/// TransformerIterator driver consumes the coll directly, so this is
/// the identity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_iter(coll_bits: u64) -> u64 {
    coll_bits
}

/// `clojure.lang.RT/chunkIteratorSeq(it)` — our TransformerIterator
/// "iterator" is already the realized result seq; identity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_chunk_iterator_seq(bits: u64) -> u64 {
    bits
}

/// `clojure.lang.TransformerIterator/create(xform, coll)` — drives the
/// transducer protocol for `(sequence xform coll)` (and through it
/// `dedupe`, `eduction`-style uses):
///
///   rf2 = (xform conj)          ; conj is the appending reducing fn
///   acc = []                    ; fold every element
///   acc = (rf2 acc x) …         ; unwrap (reduced v) and stop early
///   acc = (rf2 acc)             ; completion arity
///   → (seq acc)
///
/// EAGER over the (realized) input — Java's TransformerIterator is
/// incremental, so `(sequence xform infinite)` hangs here instead of
/// streaming. Correct for every finite input.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_transformer_iterator_create(xform_bits: u64, coll_bits: u64) -> u64 {
    let ids = heap_type_ids();
    // Stage everything GC-reachable in one registered chunk buffer:
    // [xform, rf2, acc, item0..itemN]. Every invoke below may GC.
    let buf = std::sync::Arc::new(std::cell::RefCell::new(vec![xform_bits, 0, 0]));
    register_chunk_buffer(&buf);
    let items = unsafe { seq_to_items(coll_bits) };
    let n = items.len();
    buf.borrow_mut().extend_from_slice(&items);
    drop(items);

    let conj_bits = crate::lang::compiler::resolve_core_fn_bits("conj");
    let rf2 = unsafe { cljvm_rt_invoke_1(buf.borrow()[0], conj_bits) };
    buf.borrow_mut()[1] = rf2;
    // acc = [] — empty vector.
    let empty_vec = dynobj::roots::gc_enter(1, |heap, _scope| {
        heap_alloc(heap, ids.vector as u64, 0).bits()
    });
    buf.borrow_mut()[2] = empty_vec;

    for i in 0..n {
        let (rf2_now, acc_now, item_now) = {
            let v = buf.borrow();
            (v[1], v[2], v[3 + i])
        };
        let next_acc = unsafe { cljvm_rt_invoke_2(rf2_now, acc_now, item_now) };
        // (reduced v) → unwrap and stop.
        if let Some(TAG_PTR) = nanbox_tag(next_acc) {
            let raw = nanbox_payload(next_acc) as *const u8;
            if !raw.is_null() {
                let tid = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
                if tid == ids.reduced {
                    buf.borrow_mut()[2] = unsafe { reduced_value(next_acc) };
                    break;
                }
            }
        }
        buf.borrow_mut()[2] = next_acc;
    }
    // Completion arity.
    let (rf2_now, acc_now) = {
        let v = buf.borrow();
        (v[1], v[2])
    };
    let done = unsafe { cljvm_rt_invoke_1(rf2_now, acc_now) };
    buf.borrow_mut()[2] = done;
    let final_acc = buf.borrow()[2];
    let out = unsafe { cljvm_rt_seq(final_acc) };
    deregister_chunk_buffer(&buf);
    out
}

// ── Murmur3 / hasheq ───────────────────────────────────────────────────
//
// Faithful port of Clojure's Murmur3.java (the x86 32-bit variant with
// seed 0) — `hasheq` is Murmur3-based and DIFFERS from `hashCode`
// (`util_hash_bits`), which `case*` and Java-interop hashing use. Both
// must exist; do not unify them.

fn m3_mix_k1(k1: i32) -> i32 {
    k1.wrapping_mul(0xcc9e2d51u32 as i32)
        .rotate_left(15)
        .wrapping_mul(0x1b873593u32 as i32)
}
fn m3_mix_h1(h1: i32, k1: i32) -> i32 {
    (h1 ^ k1)
        .rotate_left(13)
        .wrapping_mul(5)
        .wrapping_add(0xe6546b64u32 as i32)
}
fn m3_fmix(mut h1: i32, len: i32) -> i32 {
    h1 ^= len;
    h1 ^= ((h1 as u32) >> 16) as i32;
    h1 = h1.wrapping_mul(0x85ebca6bu32 as i32);
    h1 ^= ((h1 as u32) >> 13) as i32;
    h1 = h1.wrapping_mul(0xc2b2ae35u32 as i32);
    h1 ^= ((h1 as u32) >> 16) as i32;
    h1
}
fn m3_hash_int(input: i32) -> i32 {
    if input == 0 {
        return 0;
    }
    m3_fmix(m3_mix_h1(0, m3_mix_k1(input)), 4)
}
fn m3_hash_long(input: i64) -> i32 {
    if input == 0 {
        return 0;
    }
    let low = input as i32;
    let high = ((input as u64) >> 32) as i32;
    let h1 = m3_mix_h1(0, m3_mix_k1(low));
    m3_fmix(m3_mix_h1(h1, m3_mix_k1(high)), 8)
}

/// `clojure.lang.Util/hasheq(x)` — backs `clojure.core/hash`. Covers the
/// scalar types; collections/keywords panic with a clear message until
/// their (ordered/unordered Murmur3) formulas are ported.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_util_hasheq(bits: u64) -> u64 {
    let ids = heap_type_ids();
    let h: i32 = match nanbox_tag(bits) {
        Some(TAG_NIL) => 0,
        Some(TAG_BOOL) => {
            if nanbox_payload(bits) != 0 {
                1231
            } else {
                1237
            }
        }
        None => {
            // Double — hasheq = Double.hashCode().
            let d = f64::from_bits(bits);
            let b = d.to_bits() as i64;
            (b ^ ((b as u64 >> 32) as i64)) as i32
        }
        Some(TAG_PTR) => {
            let raw = nanbox_payload(bits) as *const u8;
            if raw.is_null() {
                0
            } else {
                let tid = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
                if tid == ids.long {
                    m3_hash_long(unsafe { raw.add(8).cast::<i64>().read_unaligned() })
                } else if tid == ids.string {
                    let st = unsafe { read_string_heap(bits, ids, "Util/hasheq — string") };
                    m3_hash_int(java_string_hash_code(st))
                } else {
                    panic!(
                        "clojure-jvm: Util/hasheq — receiver type_id {tid} not yet \
                         ported (collections/keywords need hashOrdered/hashUnordered)"
                    );
                }
            }
        }
        _ => panic!("clojure-jvm: Util/hasheq — unsupported NanBox tag"),
    };
    unsafe { box_long(h as i64) }
}

/// `(.lastIndexOf s sub)` — byte index of the LAST occurrence of String
/// `sub` within String `s`, or -1 if absent. Java overload also accepts a
/// char; `clojure.core` only uses the String form here (`root-directory`'s
/// `(.lastIndexOf d "/")` to trim a trailing path segment).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_lastIndexOf(s_bits: u64, sub_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = read_string_heap(s_bits, ids, "(.lastIndexOf s sub) — receiver");
    let sub = read_string_heap(sub_bits, ids, "(.lastIndexOf s sub) — needle");
    let n: i64 = match s.rfind(sub) {
        Some(i) => i as i64,
        None => -1,
    };
    box_long(n)
}

/// `(.replace s old new)` — String with every occurrence of `old` replaced
/// by `new`. Covers both `java.lang.String` overloads:
///   * `replace(char, char)` — `old`/`new` are boxed `Character`s. Used by
///     `clojure.core`'s `root-resource`: `(.. (name lib) (replace \- \_)
///     (replace \. \/))`.
///   * `replace(CharSequence, CharSequence)` — `old`/`new` are Strings.
/// The two are distinguished by the arg cell type.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_replace(s_bits: u64, old_bits: u64, new_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let s = read_string_heap(s_bits, ids, "(.replace s old new) — receiver");
    if is_boxed_char(old_bits) && is_boxed_char(new_bits) {
        // (char, char) overload.
        let old_ch = char::from_u32(unbox_char(old_bits))
            .unwrap_or_else(|| panic!("clojure-jvm: (.replace) — invalid `old` char"));
        let new_ch = char::from_u32(unbox_char(new_bits))
            .unwrap_or_else(|| panic!("clojure-jvm: (.replace) — invalid `new` char"));
        let out = s.replace(old_ch, &new_ch.to_string());
        return alloc_string_heap(&out, ids);
    }
    // (CharSequence, CharSequence) overload — both args are Strings.
    let old = read_string_heap(old_bits, ids, "(.replace s old new) — `old` target");
    let new = read_string_heap(new_bits, ids, "(.replace s old new) — `new` replacement");
    let out = s.replace(old, new);
    alloc_string_heap(&out, ids)
}

// ─── clojure.lang.MultiArityFn ────────────────────────────────────────
//
// Dispatcher cell for defns / defmacros with multiple clauses. Used by
// `(apply f args)` (which calls `.applyTo`) and by dynamic `cljvm_rt_invoke_*`
// when the receiver is a multi-arity defn. Static InvokeExpr dispatch
// already picks the right clause at compile time via `var_multi_arity`,
// but anything dynamic needs this cell.

#[derive(Debug, Clone, Copy)]
pub struct MultiArityEntry {
    pub fixed_arity: u32,
    pub is_variadic: bool,
    pub fref_idx: u32,
}

/// Pick the entry whose arity matches `n`. Exact match wins for
/// non-variadic; variadic catches anything ≥ its fixed_arity.
pub fn pick_multi_arity(table: &[MultiArityEntry], n: usize) -> Option<&MultiArityEntry> {
    // Try exact non-variadic first, then variadic catch-all.
    if let Some(e) = table
        .iter()
        .find(|e| !e.is_variadic && e.fixed_arity as usize == n)
    {
        return Some(e);
    }
    table
        .iter()
        .find(|e| e.is_variadic && (e.fixed_arity as usize) <= n)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_alloc_multi_arity_fn(table_ptr: u64, table_len: u64) -> u64 {
    let ids = heap_type_ids();
    // table_ptr/table_len point to a Vec<MultiArityEntry> built host-side.
    // Caller passes Box::into_raw(Box::new(Vec<MultiArityEntry>)) and len.
    // We rebuild the Vec, wrap in Arc, allocate the heap cell.
    let vec: Vec<MultiArityEntry> = unsafe {
        Vec::from_raw_parts(
            table_ptr as *mut MultiArityEntry,
            table_len as usize,
            table_len as usize,
        )
    };
    let arc = std::sync::Arc::new(vec);
    unsafe {
        alloc_arc_cell(
            ids.multi_arity_fn,
            arc,
            crate::lang::compiler::with_active_session_root_multi_arity_fn,
        )
    }
}

/// Try to dispatch through a MultiArityFn cell. Returns the resolved
/// (fn_ptr, self_arg, fref_idx) if `handle_bits` IS such a cell,
/// matching arity `n` (call's user-arg count). Returns None if the
/// handle isn't a MultiArityFn — caller should fall through to the
/// regular TAG_FN / Closure dispatch.
pub unsafe fn try_dispatch_multi_arity(
    handle_bits: u64,
    n: usize,
) -> Option<(*const u8, Option<u64>, u32)> {
    if nanbox_tag(handle_bits) != Some(TAG_PTR) {
        return None;
    }
    let p = nanbox_payload(handle_bits) as *const u8;
    if p.is_null() {
        return None;
    }
    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
    let ids = heap_type_ids();
    if tid != ids.multi_arity_fn {
        return None;
    }
    let arc: std::sync::Arc<Vec<MultiArityEntry>> = unsafe { decode_arc_cell(p) };
    let entry = pick_multi_arity(&arc, n).copied().unwrap_or_else(|| {
        panic!("clojure-jvm: MultiArityFn dispatch — no clause matches {n} args")
    });
    let slot_addr = call_table_base() + (entry.fref_idx as u64) * 8;
    let ptr = unsafe { *(slot_addr as *const *const u8) };
    Some((ptr, None, entry.fref_idx))
}

/// `cljvm_rt_sq_concat(xss)` — concat all seqs in `xss` into one
/// cons-chain. `xss` is itself a seq (cons-list, vector, or nil) of
/// seqs. Used by the syntax-quote walker which emits
/// `(. clojure.lang.RT (sqConcat (list itemA itemB itemC)))` —
/// each item is itself a seq (either `(list value)` for a normal
/// quoted entry or `value` for an `~@` splice). We need this because
/// `clojure.core/concat` is a multi-arity defn whose Var binds to nil
/// in the current state of multi-arity-as-value support; calling it
/// through the regular invoke path SIGABRTs. This extern dispatches
/// via host-method `(. clojure.lang.RT (sqConcat …))`, bypassing
/// multi-arity Var lookup entirely.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_sq_concat(xss_bits: u64) -> u64 {
    // First flatten every input seq into one list of rooted items. Each input
    // is either nil, a Cons, or a Vector. Anything else is coerced through
    // RT.seq; that path can allocate, so raw `u64` item snapshots would go
    // stale before the final cons-building pass.
    if std::env::var("CLJVM_SQ_TRACE").is_ok() {
        eprintln!(
            "[sqConcat] xss=0x{xss_bits:x} tag={:?}",
            nanbox_tag(xss_bits)
        );
    }
    let ids = heap_type_ids();
    const SQ_CONCAT_ROOT_CAP: usize = 8192;
    dynobj::roots::gc_enter(SQ_CONCAT_ROOT_CAP, |heap, scope| {
        let mut all: Vec<dynobj::roots::Rooted<'_, ()>> = Vec::new();
        let cur_root = scope.root::<()>(xss_bits);
        let head_root = scope.root::<()>(nanbox_nil());
        let tail_root = scope.root::<()>(nanbox_nil());
        loop {
            let cur = cur_root.get_raw(&*heap).bits();
            match nanbox_tag(cur) {
                Some(TAG_NIL) => break,
                Some(TAG_PTR) => {
                    let p = nanbox_payload(cur) as *const u8;
                    if p.is_null() {
                        break;
                    }
                    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                    if tid != ids.cons {
                        panic!(
                            "clojure-jvm: sqConcat: outer container must be a cons-list \
                             (the syntax-quote walker emits (list …)); got tid={tid}"
                        );
                    }
                    head_root.set_raw(Raw::from_bits(
                        &*heap,
                        unsafe { p.add(8).cast::<u64>().read_unaligned() },
                    ));
                    tail_root.set_raw(Raw::from_bits(
                        &*heap,
                        unsafe { p.add(16).cast::<u64>().read_unaligned() },
                    ));
                    let head_bits = head_root.get_raw(&*heap).bits();
                    walk_seq_into_roots(head_bits, &ids, heap, scope, &mut all);
                    let next = tail_root.get_raw(&*heap);
                    cur_root.set_raw(next);
                }
                _ => panic!(
                    "clojure-jvm: sqConcat: outer arg must be a seq, got bits 0x{cur:x} \
                     (xss=0x{xss_bits:x}, items_so_far={})",
                    all.len(),
                ),
            }
        }
        // Build the result as a cons-chain right-to-left, rooting the
        // accumulator so allocations don't strand earlier nodes.
        let acc_root = scope.root::<()>(nanbox_nil());
        for r in all.iter().rev() {
            let new_tail = heap_cons(heap, *r, acc_root);
            acc_root.set_raw(new_tail);
        }
        acc_root.get_raw(&*heap).bits()
    })
}

/// Append every element of `seq_bits` (a Cons / Vector / nil) to `out`.
/// Anything else is appended as a single element (matches Clojure's
/// concat behavior on non-seq args, modulo lazy-seq).
fn walk_seq_into_roots<'scope>(
    seq_bits: u64,
    ids: &HeapTypeIds,
    heap: &mut Heap,
    scope: &'scope dynobj::roots::RootScope<'_>,
    out: &mut Vec<dynobj::roots::Rooted<'scope, ()>>,
) {
    let seq_root = scope.root::<()>(seq_bits);
    match nanbox_tag(seq_root.get_raw(&*heap).bits()) {
        Some(TAG_NIL) => {}
        Some(TAG_PTR) => {
            let p = nanbox_payload(seq_root.get_raw(&*heap).bits()) as *const u8;
            if p.is_null() {
                return;
            }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.cons {
                let cur_root = scope.root::<()>(seq_root.get_raw(&*heap).bits());
                while let Some(TAG_PTR) = nanbox_tag(cur_root.get_raw(&*heap).bits()) {
                    let p = nanbox_payload(cur_root.get_raw(&*heap).bits()) as *const u8;
                    if p.is_null() {
                        break;
                    }
                    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                    if tid != ids.cons {
                        // Unforced lazy (or Repeat/Iterate/Cycle) tail —
                        // keep walking through it instead of truncating.
                        walk_seq_into_roots(
                            cur_root.get_raw(&*heap).bits(),
                            ids,
                            heap,
                            scope,
                            out,
                        );
                        break;
                    }
                    out.push(scope.root::<()>(unsafe { p.add(8).cast::<u64>().read_unaligned() }));
                    cur_root.set_raw(Raw::from_bits(
                        &*heap,
                        unsafe { p.add(16).cast::<u64>().read_unaligned() },
                    ));
                }
            } else if tid == ids.vector {
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as usize;
                for i in 0..n {
                    out.push(
                        scope.root::<()>(unsafe {
                            p.add(16 + i * 8).cast::<u64>().read_unaligned()
                        }),
                    );
                }
            } else if tid == ids.lazy_seq
                || tid == ids.repeat_seq
                || tid == ids.iterate_seq
                || tid == ids.cycle_seq
            {
                // A seq-typed value: force one step and walk the result.
                // Empty (nil) contributes NOTHING — pushing the cell
                // itself here used to inject a phantom element (odd-length
                // binding vectors out of `~@(interleave …)`).
                let coerced = heap_seq(heap, seq_root);
                let s_root = coerced.root(scope);
                let s_bits = s_root.get_raw(&*heap).bits();
                if !matches!(nanbox_tag(s_bits), Some(TAG_NIL))
                    && s_bits != seq_root.get_raw(&*heap).bits()
                {
                    walk_seq_into_roots(s_bits, ids, heap, scope, out);
                }
            } else {
                // Coerce via cljvm_rt_seq, then walk the result; a value
                // that does not coerce to a seq is appended as a single
                // element (concat behavior on non-seq args).
                let coerced = heap_seq(heap, seq_root);
                let s_root = coerced.root(scope);
                if !matches!(nanbox_tag(s_root.get_raw(&*heap).bits()), Some(TAG_NIL))
                    && s_root.get_raw(&*heap).bits() != seq_root.get_raw(&*heap).bits()
                {
                    walk_seq_into_roots(s_root.get_raw(&*heap).bits(), ids, heap, scope, out);
                } else {
                    out.push(scope.root::<()>(seq_root.get_raw(&*heap).bits()));
                }
            }
        }
        _ => out.push(scope.root::<()>(seq_root.get_raw(&*heap).bits())),
    }
}

// ─── clojure.lang.LazySeq / Delay ─────────────────────────────────────
//
// Both wrap a thunk (TAG_FN handle) that's called at most once to produce
// a value. Backing: Arc<RefCell<LazyState>> where state is the original
// thunk + cached realized value. Single-threaded so RefCell is fine.

#[derive(Debug)]
pub struct LazyState {
    pub thunk_bits: u64, // TAG_FN handle of the deferred fn
    pub realized: bool,
    pub value_bits: u64, // valid only when realized
}

// ─── LazySeq/Delay GC roots ───────────────────────────────────────────
//
// `LazyState` lives in a Rust `Arc<RefCell<…>>` (kept alive by
// `CompileRoots._lazy_states`). Its `thunk_bits` (the deferred fn) and,
// once realized, `value_bits` (the cached result) are NaN-boxed GC-heap
// pointers. Keeping the Arc alive only protects the Rust allocation — it
// does NOT make those heap pointers GC roots, so a moving collection
// relocates the thunk/value without updating the cached bits. The cached
// pointer then dangles, and a later force returns a stale address (under
// `CLJVM_GC=every` this surfaces as `RT.first` on a forwarded/garbage
// object — the form-430 loader crash).
//
// Fix: register every live `LazyState` as a GC root, exactly like
// `var_roots::VarRoots` does for Var root values. On each collection the
// GC scans `thunk_bits` (always — the thunk must survive until realized)
// and `value_bits` (once realized) and rewrites moved pointers in place.
thread_local! {
    static LAZY_REGISTRY: std::cell::RefCell<Vec<std::sync::Arc<std::cell::RefCell<LazyState>>>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// `ChunkBuffer`/`IChunk` cells: a Rust `Vec<u64>` of NaN-boxed element
    /// pointers. Same unrooted-cache hazard as `LazyState`.
    static CHUNK_REGISTRY: std::cell::RefCell<Vec<std::sync::Arc<std::cell::RefCell<Vec<u64>>>>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// `Atom`/`Volatile` reference cells: the held value is a NaN-boxed
    /// heap pointer cached Rust-side. Same unrooted-cache hazard as
    /// `LazyState` — every live `RefState` is scanned/forwarded by the GC.
    static REF_REGISTRY: std::cell::RefCell<Vec<std::sync::Arc<std::cell::RefCell<RefState>>>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// `MultiFn` cells: dispatch fn / default / method table, all
    /// NaN-boxed heap values cached Rust-side.
    static MULTIFN_REGISTRY: std::cell::RefCell<Vec<std::sync::Arc<std::cell::RefCell<MultiFnState>>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Register a freshly-created `MultiFnState` for GC scanning.
pub fn register_multifn_state(arc: &std::sync::Arc<std::cell::RefCell<MultiFnState>>) {
    MULTIFN_REGISTRY.with(|r| r.borrow_mut().push(arc.clone()));
}

/// The mutable state behind an `Atom` or `Volatile` cell.
pub struct RefState {
    pub value_bits: u64,
}

/// The state behind a `clojure.lang.MultiFn` cell: dispatch fn, default
/// dispatch value, and the (dispatch-value, method-fn) table. Every u64
/// here is a NaN-box that may point into the GC heap — all are scanned
/// and forwarded via the MULTIFN registry.
pub struct MultiFnState {
    pub name: String,
    pub dispatch_bits: u64,
    pub default_bits: u64,
    pub methods: Vec<(u64, u64)>,
}

/// Register a freshly-created `RefState` so its held value is
/// scanned/forwarded by the GC. Must be called for every Atom/Volatile cell.
pub fn register_ref_state(arc: &std::sync::Arc<std::cell::RefCell<RefState>>) {
    REF_REGISTRY.with(|r| r.borrow_mut().push(arc.clone()));
}

/// Shared `(new Atom x)` / `(new Volatile x)` constructor body.
fn ref_ctor(args: &[u64], type_id: usize, class: &str) -> u64 {
    if args.len() != 1 {
        panic!(
            "clojure-jvm: (new clojure.lang.{class} …) — only the 1-arg \
             constructor is supported, got {} args",
            args.len()
        );
    }
    let arc = std::sync::Arc::new(std::cell::RefCell::new(RefState { value_bits: args[0] }));
    register_ref_state(&arc);
    unsafe {
        alloc_arc_cell(
            type_id,
            arc,
            crate::lang::compiler::with_active_session_root_ref,
        )
    }
}

/// `(clojure.lang.Atom. x)` — backs `clojure.core/atom`.
pub fn atom_ctor(args: &[u64], ids: HeapTypeIds) -> u64 {
    ref_ctor(args, ids.atom, "Atom")
}

/// `(clojure.lang.Volatile. x)` — backs `clojure.core/volatile!`.
pub fn volatile_ctor(args: &[u64], ids: HeapTypeIds) -> u64 {
    ref_ctor(args, ids.volatile_cell, "Volatile")
}

/// Register a freshly-created `LazyState` so its cached heap pointers are
/// scanned/forwarded by the GC. Must be called for every LazySeq/Delay cell.
pub fn register_lazy_state(arc: &std::sync::Arc<std::cell::RefCell<LazyState>>) {
    LAZY_REGISTRY.with(|r| r.borrow_mut().push(arc.clone()));
}

/// Register a `ChunkBuffer`/`IChunk` element buffer so its NaN-boxed element
/// pointers are scanned/forwarded by the GC.
pub fn register_chunk_buffer(arc: &std::sync::Arc<std::cell::RefCell<Vec<u64>>>) {
    CHUNK_REGISTRY.with(|r| r.borrow_mut().push(arc.clone()));
}

/// Remove a buffer previously registered with [`register_chunk_buffer`],
/// matched by `Arc` identity. Used by transient GC roots (e.g. the sort
/// scratch buffer) that must not leak into the registry after use.
pub fn deregister_chunk_buffer(arc: &std::sync::Arc<std::cell::RefCell<Vec<u64>>>) {
    CHUNK_REGISTRY.with(|r| {
        let mut v = r.borrow_mut();
        if let Some(pos) = v.iter().position(|a| std::sync::Arc::ptr_eq(a, arc)) {
            v.swap_remove(pos);
        }
    });
}

/// `RootSource` over all live `LazyState`s and chunk buffers on this thread.
pub struct LazyRootSource;

impl dynobj::RootSource for LazyRootSource {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        // No cell is mutably borrowed at a GC safepoint (forcing / `.add`
        // have completed), so `as_ptr` access is sound here.
        LAZY_REGISTRY.with(|r| {
            for arc in r.borrow().iter() {
                let p = arc.as_ptr();
                unsafe {
                    visitor(std::ptr::addr_of_mut!((*p).thunk_bits));
                    if (*p).realized {
                        visitor(std::ptr::addr_of_mut!((*p).value_bits));
                    }
                }
            }
        });
        CHUNK_REGISTRY.with(|r| {
            for arc in r.borrow().iter() {
                let p = arc.as_ptr();
                unsafe {
                    let v = &mut *p;
                    for i in 0..v.len() {
                        visitor(std::ptr::addr_of_mut!(v[i]));
                    }
                }
            }
        });
        REF_REGISTRY.with(|r| {
            for arc in r.borrow().iter() {
                let p = arc.as_ptr();
                unsafe {
                    visitor(std::ptr::addr_of_mut!((*p).value_bits));
                }
            }
        });
        MULTIFN_REGISTRY.with(|r| {
            for arc in r.borrow().iter() {
                let p = arc.as_ptr();
                unsafe {
                    visitor(std::ptr::addr_of_mut!((*p).dispatch_bits));
                    visitor(std::ptr::addr_of_mut!((*p).default_bits));
                    let methods = &mut (*p).methods;
                    for pair in methods.iter_mut() {
                        visitor(std::ptr::addr_of_mut!(pair.0));
                        visitor(std::ptr::addr_of_mut!(pair.1));
                    }
                }
            }
        });
    }
}

static LAZY_ROOT_SOURCE: LazyRootSource = LazyRootSource;

/// A `'static` `RootSource` handle for `register_extra_root_source`.
pub fn lazy_root_source() -> *const dyn dynobj::RootSource {
    &LAZY_ROOT_SOURCE as *const dyn dynobj::RootSource
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_LazySeq_new1(thunk_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let arc = std::sync::Arc::new(std::cell::RefCell::new(LazyState {
        thunk_bits,
        realized: false,
        value_bits: nanbox_nil(),
    }));
    register_lazy_state(&arc);
    unsafe {
        alloc_arc_cell(
            ids.lazy_seq,
            arc,
            crate::lang::compiler::with_active_session_root_lazy_seq,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_Delay_new1(thunk_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let arc = std::sync::Arc::new(std::cell::RefCell::new(LazyState {
        thunk_bits,
        realized: false,
        value_bits: nanbox_nil(),
    }));
    register_lazy_state(&arc);
    unsafe {
        alloc_arc_cell(
            ids.delay,
            arc,
            crate::lang::compiler::with_active_session_root_delay,
        )
    }
}

/// `clojure.lang.Delay/force` — static method. Returns the (possibly
/// cached) value of the Delay. If `x` isn't a Delay, returns x as-is.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_delay_force(x_bits: u64) -> u64 {
    let ids = heap_type_ids();
    if let Some(TAG_PTR) = nanbox_tag(x_bits) {
        let p = nanbox_payload(x_bits) as *const u8;
        if !p.is_null() {
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.delay {
                let arc: std::sync::Arc<std::cell::RefCell<LazyState>> =
                    unsafe { decode_arc_cell(p) };
                let cached = {
                    let st = arc.borrow();
                    if st.realized {
                        Some(st.value_bits)
                    } else {
                        None
                    }
                };
                if let Some(v) = cached {
                    return v;
                }
                let thunk = arc.borrow().thunk_bits;
                let v = unsafe { cljvm_rt_invoke_0(thunk) };
                let mut st = arc.borrow_mut();
                st.realized = true;
                st.value_bits = v;
                return v;
            }
        }
    }
    x_bits
}

// ─── clojure.lang.ChunkBuffer / IChunk / ChunkedCons ──────────────────
//
// Mutable buffer + immutable chunk view. ChunkBuffer is what upstream
// `chunk-buffer` returns; `(.add b x)` appends; `(.chunk b)` snapshots
// into an IChunk and clears. IChunk is a fixed-size NanBox array. Both
// are Arc-backed via the new `alloc_arc_cell` helper.
//
// We use shared `Arc<RefCell<Vec<u64>>>` for both — IChunk just promises
// not to mutate. Items stay as raw NanBox bits to avoid round-tripping
// heap pointers through Object.

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_ChunkBuffer_new1(capacity_bits: u64) -> u64 {
    let ids = heap_type_ids();
    // capacity is a Long NanBox (raw f64 bits round-tripped through i64).
    let cap = arg_to_i64(capacity_bits);
    if cap < 0 {
        panic!("clojure-jvm: ChunkBuffer ctor: negative capacity {cap}");
    }
    let arc = std::sync::Arc::new(std::cell::RefCell::new(Vec::<u64>::with_capacity(
        cap as usize,
    )));
    register_chunk_buffer(&arc);
    unsafe {
        alloc_arc_cell(
            ids.chunk_buffer,
            arc,
            crate::lang::compiler::with_active_session_root_chunk_buffer,
        )
    }
}

/// `(.add buffer x)` — append. Returns nil (Java returns void).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_ChunkBuffer_add(recv_bits: u64, x_bits: u64) -> u64 {
    let p = nanbox_payload(recv_bits) as *const u8;
    let recv: std::sync::Arc<std::cell::RefCell<Vec<u64>>> = unsafe { decode_arc_cell(p) };
    recv.borrow_mut().push(x_bits);
    nanbox_nil()
}

/// `(.reduce chunk f init)` — Java's IChunk.reduce. Folds f over items
/// starting with init. Used by `reduce1`'s chunked-seq fast path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_reduce_3(recv_bits: u64, f_bits: u64, init_bits: u64) -> u64 {
    let p = nanbox_payload(recv_bits) as *const u8;
    let arc: std::sync::Arc<std::cell::RefCell<Vec<u64>>> = unsafe { decode_arc_cell(p) };
    let items: Vec<u64> = arc.borrow().clone();
    if items.is_empty() {
        return init_bits;
    }
    // Each `cljvm_rt_invoke_2` runs the user fn, which may allocate and
    // trigger a moving GC. The folded items and the running accumulator are
    // NaN-boxed heap pointers held only in this Rust frame, so they must be
    // rooted across every call or a relocation leaves them dangling (the
    // form-430 `reduce1`-over-chunked-seq crash under `CLJVM_GC=every`).
    dynobj::roots::gc_enter(items.len() + 1, |heap, scope| {
        let item_roots: Vec<dynobj::roots::Rooted<()>> =
            items.iter().map(|&b| scope.root::<()>(b)).collect();
        let acc = scope.root::<()>(init_bits);
        for ir in &item_roots {
            let next = heap_invoke_2(heap, f_bits, acc, *ir);
            acc.set_raw(next);
        }
        acc.get_raw(&*heap).bits()
    })
}

/// `(.chunkedFirst s)` — return the first chunk of an IChunkedSeq.
/// Java throws ClassCastException when receiver isn't IChunkedSeq; we
/// panic with the same intent. No values currently flow through the
/// system as IChunkedSeq (we don't yet build any), so this only runs
/// when user code calls `chunk-first` on a regular seq — which is also
/// an error in upstream Clojure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_chunkedFirst(recv_bits: u64) -> u64 {
    let ids = heap_type_ids();
    panic!(
        "clojure-jvm: ClassCastException — `(.chunkedFirst s)` requires \
         IChunkedSeq, got bits 0x{recv_bits:x} (no chunked-seq producers \
         exist yet; this matches upstream Clojure's behavior). type_ids: \
         i_chunk={} chunk_buffer={}",
        ids.i_chunk, ids.chunk_buffer,
    );
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_chunkedMore(recv_bits: u64) -> u64 {
    panic!(
        "clojure-jvm: ClassCastException — `(.chunkedMore s)` requires \
         IChunkedSeq, got bits 0x{recv_bits:x}"
    );
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_chunkedNext(recv_bits: u64) -> u64 {
    panic!(
        "clojure-jvm: ClassCastException — `(.chunkedNext s)` requires \
         IChunkedSeq, got bits 0x{recv_bits:x}"
    );
}

/// `(.chunk buffer)` — snapshot the current contents into a fresh IChunk
/// and clear the buffer. Both share the Arc pattern; the new cell gets
/// a fresh Vec so the buffer can keep growing independently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_ChunkBuffer_chunk(recv_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let p = nanbox_payload(recv_bits) as *const u8;
    let recv: std::sync::Arc<std::cell::RefCell<Vec<u64>>> = unsafe { decode_arc_cell(p) };
    let snapshot: Vec<u64> = std::mem::take(&mut *recv.borrow_mut());
    let arc = std::sync::Arc::new(std::cell::RefCell::new(snapshot));
    unsafe {
        alloc_arc_cell(
            ids.i_chunk,
            arc,
            crate::lang::compiler::with_active_session_root_i_chunk,
        )
    }
}

// ─── java.lang.StringBuilder ──────────────────────────────────────────
//
// First user of `alloc_arc_cell` / `decode_arc_cell`. Backing is
// `RefCell<String>` (single-threaded interior mutability — there's no
// concurrent JIT execution). The Arc lives in
// `CompileRoots._string_builders` for the JIT module's lifetime;
// the heap cell holds `Arc::as_ptr` in its Raw64 slot.

/// `(new StringBuilder s)` — arity-1 ctor.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_StringBuilder_new1(s_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let init = unsafe { read_string_heap(s_bits, ids, "(new StringBuilder s)") };
    let arc = std::sync::Arc::new(std::cell::RefCell::new(init.to_string()));
    unsafe {
        alloc_arc_cell(
            ids.string_builder,
            arc,
            crate::lang::compiler::with_active_session_root_string_builder,
        )
    }
}

/// `(.append sb x)` — append `x.toString()` to the builder, return the
/// builder. Java returns `this`; we return the receiver bits unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_StringBuilder_append(recv_bits: u64, x_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let p = nanbox_payload(recv_bits) as *const u8;
    let recv: std::sync::Arc<std::cell::RefCell<String>> = unsafe { decode_arc_cell(p) };
    let x_obj = any_bits_to_object(x_bits, ids);
    recv.borrow_mut().push_str(&format_object_for_str(&x_obj));
    recv_bits
}

/// `(.toString sb)` — return the accumulated buffer as a clojure.lang.String.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_StringBuilder_toString(recv_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let p = nanbox_payload(recv_bits) as *const u8;
    let recv: std::sync::Arc<std::cell::RefCell<String>> = unsafe { decode_arc_cell(p) };
    let snapshot = recv.borrow().clone();
    unsafe { alloc_string_heap(&snapshot, ids) }
}

unsafe fn read_string_heap<'a>(bits: u64, ids: HeapTypeIds, ctx: &str) -> &'a str {
    let raw = match nanbox_tag(bits) {
        Some(TAG_PTR) => nanbox_payload(bits) as *const u8,
        _ => {
            eprintln!("[cljvm-stub] {ctx} — non-heap NanBox 0x{bits:x}, treating as empty string");
            return "";
        }
    };
    if raw.is_null() {
        eprintln!("[cljvm-stub] {ctx} — null receiver, treating as empty string");
        return "";
    }
    let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
    if type_id != ids.string {
        eprintln!("[cljvm-stub] {ctx} — type_id {type_id} not a String, treating as empty");
        return "";
    }
    let count = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
    let bytes = unsafe { std::slice::from_raw_parts(raw.add(16), count) };
    std::str::from_utf8(bytes).expect("String heap bytes must be UTF-8")
}

// ── clojure.lang.Reduced ───────────────────────────────────────────────
//
// `(reduced x)` wraps `x` in a Reduced cell to signal early termination of
// `reduce`. The cell has one traced Value slot at offset 8. `reduced?` /
// `RT.isReduced` test for it; `deref` / `@` unwrap it.

/// `(clojure.lang.Reduced. x)` — allocate a Reduced cell wrapping `x`.
///
/// # Safety
/// Runs on a mutator thread (may trigger GC); `x_bits` is rooted across
/// the allocation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_Reduced_new1(x_bits: u64) -> u64 {
    let ids = heap_type_ids();
    dynobj::roots::gc_enter(2, |heap, scope| {
        let x = scope.root::<()>(x_bits);
        let cell = heap_alloc(heap, ids.reduced as u64, 0).root(scope);
        let cell_bits = cell.get_raw(&*heap).bits();
        let ptr = nanbox_payload(cell_bits) as *mut u8;
        unsafe {
            write_slot(ptr, 8, x.get_raw(&*heap));
        }
        cell_bits
    })
}

/// True iff `bits` is a `clojure.lang.Reduced` cell.
pub fn is_reduced(bits: u64) -> bool {
    if let Some(TAG_PTR) = nanbox_tag(bits) {
        let raw = nanbox_payload(bits) as *const u8;
        if raw.is_null() {
            return false;
        }
        let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
        return type_id == heap_type_ids().reduced;
    }
    false
}

/// Unwrap a Reduced cell's wrapped value.
///
/// # Safety
/// `bits` must be a Reduced cell (`is_reduced(bits)` is true).
pub unsafe fn reduced_value(bits: u64) -> u64 {
    let raw = nanbox_payload(bits) as *const u8;
    unsafe { raw.add(8).cast::<u64>().read_unaligned() }
}

/// `clojure.lang.RT/isReduced` — backs `(reduced? x)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_isReduced(x_bits: u64) -> u64 {
    nanbox_bool(is_reduced(x_bits))
}

/// `(clojure.lang.RT/load "path")` — the runtime resource loader behind
/// `clojure.core/load`. Reads the embedded source for `path` and evaluates
/// each form through the active Session, reentrantly (the call happens
/// while the JIT-compiled `load` fn is on the stack — see
/// `with_active_session_load_resource`). Returns nil.
///
/// # Safety
/// Runs on a mutator thread inside `Session::eval_form`; reenters the
/// session via the `ACTIVE_SESSION` thread-local raw pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_load(path_bits: u64) -> u64 {
    let path = unsafe { read_string_heap(path_bits, heap_type_ids(), "RT.load: path") };
    crate::lang::compiler::with_active_session_load_resource(path);
    nanbox_nil()
}

/// `(.withMeta x m)` — return a value structurally equal to `x` but
/// carrying metadata `m`. Cons gets its dedicated meta slot updated
/// (cheap path); everything else gets wrapped in a `WithMeta` heap
/// cell. If `x` is itself a `WithMeta`, we collapse the wrapper so
/// metadata wrappers never nest.
///
/// Both args are rooted on a `with_scope` frame so a GC during the
/// `gc_alloc_thunk` call doesn't relocate them out from under us.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_with_meta(recv_bits: u64, meta_bits: u64) -> u64 {
    let ids = heap_type_ids();
    if !matches!(nanbox_tag(recv_bits), Some(TAG_PTR)) {
        if matches!(nanbox_tag(recv_bits), Some(TAG_NIL)) {
            panic!("clojure-jvm: (.withMeta nil m) — nil is not IObj");
        }
        panic!(
            "clojure-jvm: (.withMeta x m) — receiver bits 0x{recv_bits:x} \
             is an immediate that does not carry metadata"
        );
    }
    // `(.withMeta x nil)`: dropping metadata. A bare (unwrapped, non-cons)
    // receiver carries none — return it unchanged; an existing WithMeta
    // wrapper unwraps to its inner value. This keeps the common
    // `(with-meta ret (meta m))` tail of select-keys/update-vals from
    // wrapping plain maps in WithMeta cells that the seq walks would
    // then have to unwrap.
    if matches!(nanbox_tag(meta_bits), Some(TAG_NIL)) {
        let ids2 = heap_type_ids();
        let raw = nanbox_payload(recv_bits) as *const u8;
        if !raw.is_null() {
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids2.with_meta {
                return unsafe { raw.add(8).cast::<u64>().read_unaligned() };
            }
            if type_id != ids2.cons {
                return recv_bits;
            }
        }
    }
    dynobj::roots::gc_enter(3, |heap, scope| {
        let recv = scope.root::<()>(recv_bits);
        let meta = scope.root::<()>(meta_bits);

        // Read receiver type once. `gc_alloc` may move things, so
        // read header / first / rest fresh after each alloc by going
        // back through `recv.get_raw(&*heap).bits()`.
        let raw = nanbox_payload(recv.get_raw(&*heap).bits()) as *const u8;
        if raw.is_null() {
            panic!("clojure-jvm: (.withMeta nil m) — null receiver");
        }
        let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
        if type_id == ids.cons {
            let cell = heap_alloc(heap, ids.cons as u64, 0).root(scope);
            let owner_bits = cell.get_raw(&*heap).bits();
            let new_ptr = nanbox_payload(owner_bits) as *mut u8;
            // Re-read the receiver pointer AFTER alloc; root may have
            // been forwarded.
            let raw = nanbox_payload(recv.get_raw(&*heap).bits()) as *const u8;
            let first = unsafe { read_slot(&*heap, raw, 8) };
            let rest = unsafe { read_slot(&*heap, raw, 16) };
            trap_forwarded_first_result("withMeta.cons.first", owner_bits, first.bits());
            unsafe {
                write_slot(new_ptr, 8, first);
                write_slot(new_ptr, 16, rest);
                write_slot(new_ptr, 24, meta.get_raw(&*heap));
            }
            return owner_bits;
        }
        // Generic path: wrap in `clojure.lang.WithMeta`. Layout:
        // header(8) + inner(8) + meta(8). Collapse nested wrappers so
        // the inner is always a "bare" value.
        let cell = heap_alloc(heap, ids.with_meta as u64, 0).root(scope);
        let owner_bits = cell.get_raw(&*heap).bits();
        let new_ptr = nanbox_payload(owner_bits) as *mut u8;
        // Re-read AFTER alloc.
        let raw = nanbox_payload(recv.get_raw(&*heap).bits()) as *const u8;
        let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
        let inner = if type_id == ids.with_meta {
            unsafe { read_slot(&*heap, raw, 8) }
        } else {
            recv.get_raw(&*heap)
        };
        unsafe {
            write_slot(new_ptr, 8, inner);
            write_slot(new_ptr, 16, meta.get_raw(&*heap));
        }
        owner_bits
    })
}

// ── User types / protocols ─────────────────────────────────────────────
//
// Runtime support for `deftype`/`defrecord`/`defprotocol`/`extend-type`.
// The registries themselves live in `lang::user_types`; the functions
// below are the bridge between NanBox values and those registries.
//
// Heap layout for a UserInstance cell (one ObjType shared across all
// user-defined types):
//
//     +0   u16   type_id          (= ids.user_instance for all instances)
//     +2   u16   _gc/header pad
//     +4   u32   _header
//     +offs Raw64 user_type_id    (encoded as u64; the high 32 bits are 0)
//     +base Value0, Value1, …    (one slot per declared field)
//
// `user_type_id_offset` and `varlen_base` are captured in
// `UserInstanceLayout` at `Compiler::new` time.

/// Map an arbitrary NanBox value to its `LogicalTypeId` for protocol
/// dispatch lookup. The id is the union of:
///   * Built-in `ObjTypeId.0` values for heap-allocated cells.
///   * `BUILTIN_*` synthetic ids for tag-encoded primitives.
///   * `USER_TYPE_BASE + user_type_id` for `UserInstance` cells
///     (read off the cell's Raw64 discriminator slot).
pub fn effective_type_id(bits: u64) -> crate::lang::user_types::LogicalTypeId {
    use crate::lang::user_types::{
        BUILTIN_BOOL, BUILTIN_DOUBLE, BUILTIN_FN, BUILTIN_LONG, BUILTIN_NIL, user_type_logical,
    };
    match nanbox_tag(bits) {
        None => BUILTIN_DOUBLE,
        Some(TAG_NIL) => BUILTIN_NIL,
        Some(TAG_BOOL) => BUILTIN_BOOL,
        Some(TAG_FN) => BUILTIN_FN,
        Some(TAG_PTR) => {
            let raw = nanbox_payload(bits) as *const u8;
            if raw.is_null() {
                return BUILTIN_NIL;
            }
            let cell_type = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            // For UserInstance cells, read the Raw64 discriminator and
            // shift into the user-type range. For everything else,
            // widen the ObjTypeId to LogicalTypeId space directly.
            let ids = heap_type_ids();
            // A boxed Long is a TAG_PTR heap cell, but dispatch treats it as
            // a number under the synthetic BUILTIN_LONG id (parallel to
            // BUILTIN_DOUBLE for native floats).
            if cell_type == ids.long {
                return BUILTIN_LONG;
            }
            if cell_type == ids.user_instance {
                let layout = user_instance_layout();
                let off = layout.user_type_id_offset as isize;
                let user_tid = unsafe { raw.offset(off).cast::<u64>().read_unaligned() } as u32;
                user_type_logical(user_tid)
            } else {
                cell_type as u32
            }
        }
        _ => crate::lang::user_types::BUILTIN_OBJECT,
    }
}

/// Read field index `i` (zero-based, in declaration order) off a
/// UserInstance NanBox. Panics if `bits` does not encode a UserInstance
/// cell or if `i` is out of range for that user type. Callers that
/// can't statically know the receiver shape should look up the field
/// count via `lang::user_types::user_type_info` first.
pub fn user_instance_field_get(bits: u64, i: usize) -> u64 {
    let layout = user_instance_layout();
    let ids = heap_type_ids();
    let raw = match nanbox_tag(bits) {
        Some(TAG_PTR) => {
            let p = nanbox_payload(bits) as *const u8;
            if p.is_null() {
                panic!("clojure-jvm: user_instance_field_get: null TAG_PTR payload");
            }
            p
        }
        other => panic!(
            "clojure-jvm: user_instance_field_get: expected TAG_PTR receiver, \
             got tag {other:?}",
        ),
    };
    let cell_type = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
    if cell_type != ids.user_instance {
        panic!(
            "clojure-jvm: user_instance_field_get: receiver type_id {cell_type} \
             is not UserInstance ({})",
            ids.user_instance,
        );
    }
    // Without field-count metadata on the cell itself we trust the
    // analyzer to bake in a valid index. Out-of-range reads would walk
    // off the end of the varlen payload — read_unaligned would still
    // succeed but return garbage. The user-types registry has the
    // declared count; callers needing safety should query it.
    let off = layout.varlen_base + (i as i64) * 8;
    unsafe { raw.offset(off as isize).cast::<u64>().read_unaligned() }
}

/// Allocate a new `clojure.lang.UserInstance` cell stamped with
/// `user_type_id` and populated with the given field values (in
/// declaration order). Returns the NanBox-tagged pointer.
///
/// Caller responsibilities:
///   * Must be running with an installed mutator thread (the dyngc
///     contract — same as `cljvm_rt_cons` and friends).
///   * `fields` must be already-rooted; this function does not push
///     them into the rootset before alloc. The expected pattern when
///     called from JIT code is for codegen to materialize the field
///     values, root them via `dynobj::roots::with_scope`, then call this.
///
/// # Safety
/// See above — runs on a mutator thread, may trigger GC.
pub unsafe fn alloc_user_instance(user_type_id: u32, fields: &[u64]) -> u64 {
    // Previously a latent form-430: `fields` are heap pointers held in a
    // Rust slice across `gc_alloc_thunk`; a moving GC would relocate them
    // and the writes below would store stale pointers. Now every field is
    // rooted before the allocation, and the writes re-read the rooted
    // (relocated) values. The `&mut Heap` on `heap_alloc` makes the old
    // un-rooted version a compile error.
    dynobj::roots::gc_enter(fields.len() + 1, |heap, scope| {
        let ids = heap_type_ids();
        let layout = user_instance_layout();
        let rooted: Vec<Rooted<'_, ()>> =
            fields.iter().map(|&bits| scope.root::<()>(bits)).collect();
        let cell = heap_alloc(heap, ids.user_instance as u64, fields.len() as u64).root(scope);
        let cell_bits = cell.get_raw(&*heap).bits();
        let ptr = nanbox_payload(cell_bits) as *mut u8;
        let off = layout.user_type_id_offset as isize;
        unsafe {
            write_raw_word(ptr, off as usize, user_type_id as u64);
            for (i, r) in rooted.iter().enumerate() {
                let slot_off = layout.varlen_base + (i as i64) * 8;
                write_slot(ptr, slot_off as usize, r.get_raw(&*heap));
            }
        }
        cell_bits
    })
}

/// Dispatch a protocol method call. `method_id` identifies the
/// `ProtoMethod` (allocated by `register_protocol`). Looks up an impl
/// for `effective_type_id(this)`, falling back to `BUILTIN_OBJECT`.
/// Returns the FnHandle (NanBox bits) of the implementation — caller
/// is responsible for invoking it with the original args.
///
/// Panics with a Clojure-style "No implementation of method" message
/// when no entry exists (matching the runtime behavior of
/// `clojure.core/-cache-protocol-fn`'s miss path).
pub fn lookup_protocol_method(method_id: u32, this_bits: u64) -> u64 {
    use crate::lang::user_types::{BUILTIN_OBJECT, lookup_impl};
    let tid = effective_type_id(this_bits);
    if let Some(handle) = lookup_impl(tid, method_id) {
        return handle;
    }
    if let Some(handle) = lookup_impl(BUILTIN_OBJECT, method_id) {
        return handle;
    }
    panic!(
        "clojure-jvm: protocol dispatch: no impl for method_id {method_id} on \
         type_id {tid} (0x{tid:x}); receiver bits 0x{this_bits:x}",
    );
}

// ── JIT-callable protocol dispatch externs ─────────────────────────────
//
// `defprotocol` lowers each declared method `(m [this …N args])` to a
// `(def m (fn* [this …N args] (cljvm_rt_protocol_dispatch_<N+1> <method_id> this …N args)))`.
// The method_id is a Long literal baked at expansion time. These
// externs read it as a NanBox u64 and decode back to u32.
//
// Decoding: the method_id literal arrives as `f64::to_bits(method_id as f64)`
// in NanBox form (our long encoding). `decode_method_id` reverses that.

fn decode_method_id_bits(method_id_bits: u64) -> u32 {
    // method_id is baked as an integer literal — a boxed Long once the flip
    // lands, or a NanBox double pre-flip. `arg_to_i64` handles both.
    let v = arg_to_i64(method_id_bits);
    if v < 0 || v > u32::MAX as i64 {
        panic!(
            "clojure-jvm: cljvm_rt_protocol_dispatch: implausible \
             method_id {v} (decoded from bits 0x{method_id_bits:x})"
        );
    }
    v as u32
}

/// Arity 1: `(m this)` — a no-extra-args protocol method.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_protocol_dispatch_1(method_id_bits: u64, this: u64) -> u64 {
    let mid = decode_method_id_bits(method_id_bits);
    let handle = lookup_protocol_method(mid, this);
    unsafe { cljvm_rt_invoke_1(handle, this) }
}

/// Arity 2: `(m this a)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_protocol_dispatch_2(
    method_id_bits: u64,
    this: u64,
    a: u64,
) -> u64 {
    let mid = decode_method_id_bits(method_id_bits);
    let handle = lookup_protocol_method(mid, this);
    unsafe { cljvm_rt_invoke_2(handle, this, a) }
}

/// Arity 3: `(m this a b)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_protocol_dispatch_3(
    method_id_bits: u64,
    this: u64,
    a: u64,
    b: u64,
) -> u64 {
    let mid = decode_method_id_bits(method_id_bits);
    let handle = lookup_protocol_method(mid, this);
    unsafe { cljvm_rt_invoke_3(handle, this, a, b) }
}

/// Arity 4: `(m this a b c)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_protocol_dispatch_4(
    method_id_bits: u64,
    this: u64,
    a: u64,
    b: u64,
    c: u64,
) -> u64 {
    let mid = decode_method_id_bits(method_id_bits);
    let handle = lookup_protocol_method(mid, this);
    unsafe { cljvm_rt_invoke_4(handle, this, a, b, c) }
}

/// `(satisfies? P x)` query — returns true iff at least one method of
/// protocol `proto_id` has an impl registered for `effective_type_id(x)`
/// (exact match, no `BUILTIN_OBJECT` fallback). Matches Clojure's
/// `clojure.core/satisfies?` semantics: a type satisfies a protocol
/// when it has been extended via `extend-type` / `extend-protocol` /
/// inline `deftype`. The Object fallback bucket is intentionally
/// excluded; otherwise every receiver would "satisfy" every protocol
/// that has an `Object` default.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_satisfies(proto_id_bits: u64, x_bits: u64) -> u64 {
    let pid = arg_to_i64(proto_id_bits);
    if pid < 0 || pid > u32::MAX as i64 {
        panic!("clojure-jvm: cljvm_rt_satisfies: implausible proto_id {pid}");
    }
    let info = crate::lang::user_types::protocol_info(pid as u32).unwrap_or_else(|| {
        panic!("clojure-jvm: cljvm_rt_satisfies: protocol id {pid} not registered")
    });
    let tid = effective_type_id(x_bits);
    for m in &info.methods {
        if crate::lang::user_types::lookup_impl(tid, m.id).is_some() {
            return nanbox_bool(true);
        }
    }
    nanbox_bool(false)
}

// ── User-instance allocation externs ───────────────────────────────────
//
// `deftype` expands its factory `(def Foo (fn* [a b]
//   (. clojure.lang.RT (allocUserInstanceN <user_type_id> a b))))`.
// One extern per supported field arity; raise the cap by adding more
// here and registering them in `Compiler::new`. v1 covers 0..4 fields.

fn decode_user_type_id_bits(type_id_bits: u64) -> u32 {
    let v = arg_to_i64(type_id_bits);
    if v < 0 || v > u32::MAX as i64 {
        panic!(
            "clojure-jvm: alloc_user_instance: implausible user_type_id {v} \
             (decoded from bits 0x{type_id_bits:x})"
        );
    }
    v as u32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_alloc_user_instance_0(type_id_bits: u64) -> u64 {
    let tid = decode_user_type_id_bits(type_id_bits);
    unsafe { alloc_user_instance(tid, &[]) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_alloc_user_instance_1(type_id_bits: u64, a: u64) -> u64 {
    let tid = decode_user_type_id_bits(type_id_bits);
    unsafe { alloc_user_instance(tid, &[a]) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_alloc_user_instance_2(type_id_bits: u64, a: u64, b: u64) -> u64 {
    let tid = decode_user_type_id_bits(type_id_bits);
    unsafe { alloc_user_instance(tid, &[a, b]) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_alloc_user_instance_3(
    type_id_bits: u64,
    a: u64,
    b: u64,
    c: u64,
) -> u64 {
    let tid = decode_user_type_id_bits(type_id_bits);
    unsafe { alloc_user_instance(tid, &[a, b, c]) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_alloc_user_instance_4(
    type_id_bits: u64,
    a: u64,
    b: u64,
    c: u64,
    d: u64,
) -> u64 {
    let tid = decode_user_type_id_bits(type_id_bits);
    unsafe { alloc_user_instance(tid, &[a, b, c, d]) }
}

/// `(.-field-name inst)` lowers to a call to this extern. Reads the
/// receiver's `user_type_id` off the cell, looks up the type info,
/// scans for the field by name, and returns the corresponding slot's
/// NanBox value. Panics on non-UserInstance receivers or missing
/// field name.
///
/// `field_name_bits` is the NanBox of a Symbol heap cell. We use
/// Symbol rather than String because field names in source are
/// already symbols — the analyzer can hand them across without
/// allocating a String.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_user_field_get_by_name(
    inst_bits: u64,
    field_name_bits: u64,
) -> u64 {
    let ids = heap_type_ids();
    // Decode receiver.
    let raw = match nanbox_tag(inst_bits) {
        Some(TAG_PTR) => {
            let p = nanbox_payload(inst_bits) as *const u8;
            if p.is_null() {
                panic!("clojure-jvm: user_field_get_by_name: null receiver");
            }
            p
        }
        other => panic!(
            "clojure-jvm: user_field_get_by_name: receiver must be a \
             UserInstance, got NanBox tag {other:?}"
        ),
    };
    let cell_type = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
    if cell_type != ids.user_instance {
        panic!(
            "clojure-jvm: user_field_get_by_name: receiver type_id \
             {cell_type} is not UserInstance ({})",
            ids.user_instance,
        );
    }
    let layout = user_instance_layout();
    let off = layout.user_type_id_offset as isize;
    let user_tid = unsafe { raw.offset(off).cast::<u64>().read_unaligned() } as u32;
    // Decode field name from the Symbol heap cell.
    let field_obj = unsafe { heap_bits_to_object(field_name_bits, ids) };
    let field_sym = match field_obj {
        Object::Symbol(s) => s,
        other => panic!(
            "clojure-jvm: user_field_get_by_name: field name must decode \
             to a Symbol, got {other:?}",
        ),
    };
    let idx =
        crate::lang::user_types::user_type_field_index(user_tid, &field_sym).unwrap_or_else(|| {
            let info = crate::lang::user_types::user_type_info(user_tid);
            let known: Vec<String> = info
                .map(|t| t.fields.iter().map(|s| s.get_name().to_string()).collect())
                .unwrap_or_default();
            panic!(
                "clojure-jvm: user_field_get_by_name: field `{}` not found on \
             user type id {user_tid}; known fields: {known:?}",
                field_sym.get_name(),
            )
        });
    user_instance_field_get(inst_bits, idx)
}

/// `cljvm_rt_install_impl(type_id_bits, method_id_bits, fn_handle_bits)`:
/// register `fn_handle_bits` as the impl of `(type_id, method_id)`. Both
/// type_id and method_id arrive as Long-encoded NanBox bits. `fn_handle_bits`
/// is the raw NanBox of the implementation fn (TAG_FN or TAG_PTR closure).
/// Returns `nil` for convenience as a top-level statement.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_install_impl(
    type_id_bits: u64,
    method_id_bits: u64,
    fn_handle_bits: u64,
) -> u64 {
    let tid = arg_to_i64(type_id_bits);
    if tid < 0 || tid > u32::MAX as i64 {
        panic!("clojure-jvm: cljvm_rt_install_impl: implausible type_id {tid}");
    }
    let mid = decode_method_id_bits(method_id_bits);
    crate::lang::user_types::install_impl(tid as u32, mid, fn_handle_bits);
    nanbox_nil()
}

/// `clojure.lang.Namespace/setCurrent(Symbol nsname)` — implements
/// Clojure's `in-ns`. Returns nil (upstream returns the Namespace;
/// we don't have a Namespace NanBox encoding yet, and the loader
/// only cares about the side effect of switching the current ns).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_ns_set_current(sym_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let sym_arc = match nanbox_tag(sym_bits) {
        Some(TAG_PTR) => {
            let p = nanbox_payload(sym_bits) as *const u8;
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.symbol {
                let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::symbol::Symbol;
                unsafe { std::sync::Arc::increment_strong_count(arc_ptr) };
                unsafe { std::sync::Arc::from_raw(arc_ptr) }
            } else {
                panic!("clojure-jvm: in-ns: arg must be a Symbol, got type_id {tid}");
            }
        }
        _ => panic!("clojure-jvm: in-ns: arg must be a Symbol"),
    };
    let ns = crate::lang::namespace::Namespace::find_or_create(sym_arc);
    crate::lang::rt::CURRENT_NS.bind_root(crate::lang::object::Object::Namespace(ns));
    nanbox_nil()
}

#[cfg(test)]
mod user_type_runtime_tests {
    use super::*;
    use crate::lang::symbol::Symbol;
    use crate::lang::user_types::{
        self as ut, BUILTIN_DOUBLE, BUILTIN_NIL, BUILTIN_OBJECT, USER_TYPE_BASE, install_impl,
        register_protocol, register_user_type, user_type_logical,
    };
    use std::sync::Arc;

    fn sym(n: &str) -> Arc<Symbol> {
        Symbol::intern_ns_name(None, n)
    }

    // These tests share the user_types global registries with the
    // registry-touching tests in `compiler.rs` and `user_types.rs`. They
    // MUST serialize via the one crate-level lock (not a local one), or a
    // reset here wipes a compiler test's protocols mid-run.
    fn guard() -> std::sync::MutexGuard<'static, ()> {
        crate::lang::user_types::registry_test_guard()
    }

    #[test]
    fn effective_type_id_handles_tag_encoded_primitives() {
        let _g = guard();
        // Don't need set_heap_type_ids / set_user_instance_layout for
        // primitives — they're decided by nanbox_tag alone.
        assert_eq!(effective_type_id(nanbox_nil()), BUILTIN_NIL);
        assert_eq!(effective_type_id(nanbox_bool(true)), ut::BUILTIN_BOOL);
        // f64::to_bits gives an untagged double.
        assert_eq!(effective_type_id(1.5_f64.to_bits()), BUILTIN_DOUBLE);
    }

    #[test]
    fn dispatch_falls_back_to_object_when_no_exact_impl() {
        let _g = guard();
        let (_pid, mids) = register_protocol(sym("ICountable"), vec![(sym("-count"), vec![1])]);
        // Install on Object only.
        install_impl(BUILTIN_OBJECT, mids[0], 0xABCDABCD);
        // Nil should still hit the Object fallback (no Nil-specific entry).
        let handle = lookup_protocol_method(mids[0], nanbox_nil());
        assert_eq!(handle, 0xABCDABCD);
    }

    #[test]
    #[should_panic(expected = "no impl for method_id")]
    fn dispatch_panics_with_no_impl() {
        let _g = guard();
        let (_pid, mids) = register_protocol(sym("IUnknown"), vec![(sym("-mystery"), vec![1])]);
        let _ = lookup_protocol_method(mids[0], nanbox_nil());
    }

    #[test]
    fn user_type_logical_in_user_range() {
        let _g = guard();
        let uid = register_user_type(sym("Foo"), vec![]);
        let logical = user_type_logical(uid);
        assert!(logical >= USER_TYPE_BASE);
    }
}

/// GC alloc-side lock-in. The borrow-checked `Heap`/`Raw` discipline only
/// bites when allocation goes through `heap_alloc(&mut Heap)`: a direct
/// `dynlang::gc::gc_alloc_thunk` call lets you hold a bare NanBox GC pointer
/// across the allocation (the form-430 stale-pointer bug) with no compile
/// error, because raw bits don't borrow the `Heap`.
///
/// This test freezes the set of functions allowed to call `gc_alloc_thunk`
/// directly. Each one was audited to be safe: it either *is* `heap_alloc`,
/// stores only a numeric immediate, or decodes its heap-pointer args into
/// Rust-owned data (`Arc`/`Object`/`String`) BEFORE allocating and stores
/// only stable Rust pointers — so no NanBox GC pointer is live across the
/// alloc. Any NEW direct call fails this test, forcing the author to either
/// route the allocation through `heap_alloc` (the rooted, borrow-checked
/// path) or audit the new site and add it here with the same justification.
#[cfg(test)]
mod gc_alloc_lockin {
    /// Functions audited safe to call `gc_alloc_thunk` directly. See the
    /// module doc for the safety criterion.
    const ALLOWLIST: &[&str] = &[
        "heap_alloc",        // the sanctioned rooted allocator
        "box_long",          // boxes an i64 immediate
        "box_char",          // boxes a u32 codepoint immediate
        "alloc_arc_cell",    // takes a Rust value, stores its Arc ptr
        "alloc_string_heap", // copies a Rust-owned &str
        "alloc_tree_map_cell", // takes Arc<PersistentTreeMap>, stores its Arc ptr
        "alloc_tree_set_cell", // takes Arc<PersistentTreeSet>, stores its Arc ptr
        // Arc-backed persistent collections: decode to Arc before alloc,
        // store only the stable Rust Arc pointer.
        "cljvm_rt_assoc",
        "cljvm_rt_conj",
        "cljvm_rt_dissoc",
        "cljvm_inst_disjoin",
        "cljvm_inst_getMappings",
        "cljvm_phm_create",
        "cljvm_phs_create",
        "cljvm_ptm_create",
        "cljvm_ptm_create_cmp",
        "cljvm_pts_create",
        "cljvm_pts_create_cmp",
        // Symbol/Keyword/Class interning: decode arg to a Rust String/Arc
        // before alloc, store the interned Rust Arc pointer.
        "cljvm_keyword_intern_1",
        "cljvm_keyword_intern_2",
        "cljvm_keyword_find_1",
        "cljvm_keyword_find_2",
        "cljvm_symbol_intern_1",
        "cljvm_symbol_intern_2",
        "cljvm_inst_toSymbol",
        "cljvm_inst_sym",
        "cljvm_compiler_hostexpr_maybeClass",
    ];

    #[test]
    fn no_unreviewed_direct_gc_alloc() {
        let src = include_str!("runtime.rs");
        let fn_re = |line: &str| -> Option<String> {
            let t = line.trim_start();
            let t = t.strip_prefix("pub ").unwrap_or(t);
            let t = t.strip_prefix("unsafe ").unwrap_or(t);
            let t = t.strip_prefix("extern \"C\" ").unwrap_or(t);
            let t = t.strip_prefix("fn ")?;
            let name: String = t
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            (!name.is_empty()).then_some(name)
        };

        let mut current = String::new();
        let mut offenders: Vec<(String, usize)> = Vec::new();
        for (i, line) in src.lines().enumerate() {
            // Stop before this guard module so its own mention of the call
            // string (in the .contains check) isn't counted as a call site.
            if line.contains("mod gc_alloc_lockin") {
                break;
            }
            if let Some(name) = fn_re(line) {
                current = name;
            }
            let code = line.trim_start();
            if code.contains("dynlang::gc::gc_alloc_thunk(")
                && !code.starts_with("//")
                && !code.starts_with("///")
                && !ALLOWLIST.contains(&current.as_str())
            {
                offenders.push((current.clone(), i + 1));
            }
        }

        assert!(
            offenders.is_empty(),
            "New direct `gc_alloc_thunk` call(s) outside the audited allowlist:\n{}\n\n\
             A direct alloc lets a bare NanBox GC pointer go stale across it \
             (the form-430 bug). Route the allocation through `heap_alloc(&mut Heap)` \
             (which roots and borrow-checks), or — if the site provably holds no NanBox \
             GC pointer across the alloc — audit it and add the function to ALLOWLIST \
             in this module with a justification.",
            offenders
                .iter()
                .map(|(f, ln)| format!("  - {f} (runtime.rs:{ln})"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

/// Guard that every type-safe allocating builder (any fn whose body calls
/// `heap_alloc(`) writes its heap slots through the typed helpers
/// [`write_slot`] (NanBox GC value) or [`write_raw_word`] (plain machine
/// word), never via a bare `cast::<u64>().write_unaligned(`. A bare word
/// write hides whether the stored bits are a live GC value or an opaque
/// word, and historically let stale (un-relocated) NanBox pointers slip in
/// across a collection. Routing through the helpers makes the choice
/// explicit and (for `write_slot`) borrow-checks the value at the store.
#[cfg(test)]
mod gc_slot_writes {
    /// The slot helpers themselves legitimately contain the bare write —
    /// they are the single sanctioned implementation. Exclude them.
    const HELPER_DEFS: &[&str] = &["write_slot", "write_raw_word", "read_slot"];

    #[test]
    fn value_writes_in_builders_are_typed() {
        let src = include_str!("runtime.rs");
        let fn_name = |line: &str| -> Option<String> {
            let t = line.trim_start();
            let t = t.strip_prefix("pub ").unwrap_or(t);
            let t = t.strip_prefix("unsafe ").unwrap_or(t);
            let t = t.strip_prefix("extern \"C\" ").unwrap_or(t);
            let t = t.strip_prefix("fn ")?;
            let name: String = t
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            (!name.is_empty()).then_some(name)
        };

        // First pass: for each function, record whether its body calls
        // `heap_alloc(` and on which lines it does a bare u64 slot write.
        // Stop before the first guard module so neither this test's nor the
        // sibling guard's own string literals are scanned as real code.
        let mut current = String::new();
        let mut calls_heap_alloc: std::collections::HashMap<String, bool> =
            std::collections::HashMap::new();
        let mut bare_writes: Vec<(String, usize)> = Vec::new();
        for (i, line) in src.lines().enumerate() {
            if line.contains("mod gc_alloc_lockin") {
                break;
            }
            if let Some(name) = fn_name(line) {
                current = name;
            }
            let code = line.trim_start();
            if code.starts_with("//") || code.starts_with("///") {
                continue;
            }
            if code.contains("heap_alloc(") {
                calls_heap_alloc.insert(current.clone(), true);
            }
            if code.contains("cast::<u64>().write_unaligned(")
                && !HELPER_DEFS.contains(&current.as_str())
            {
                bare_writes.push((current.clone(), i + 1));
            }
        }

        // Offenders: bare writes living inside a heap_alloc-allocating builder.
        let offenders: Vec<(String, usize)> = bare_writes
            .into_iter()
            .filter(|(f, _)| *calls_heap_alloc.get(f).unwrap_or(&false))
            .collect();

        assert!(
            offenders.is_empty(),
            "Bare `cast::<u64>().write_unaligned(` inside type-safe allocating \
             builder(s):\n{}\n\n\
             Functions that call `heap_alloc(` must write heap slots through \
             `write_slot(base, off, <Raw>)` for NanBox GC values or \
             `write_raw_word(base, off, <word>)` for plain machine words \
             (counts, user_type_id, Arc pointers). A bare word write can store \
             a stale NanBox pointer across a collection.",
            offenders
                .iter()
                .map(|(f, ln)| format!("  - {f} (runtime.rs:{ln})"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}
