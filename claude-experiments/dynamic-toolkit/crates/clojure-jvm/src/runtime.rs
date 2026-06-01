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

pub fn nanbox_payload(bits: u64) -> u64 { bits & PAYLOAD_MASK }

/// Public NanBox constructors callable from heap-population code that
/// stores Object fields (`alloc_object_as_nanbox` in compiler.rs).
pub fn nanbox_nil() -> u64 { nanbox_encode(TAG_NIL, 0) }
pub fn nanbox_bool(b: bool) -> u64 { nanbox_encode(TAG_BOOL, b as u64) }
/// NanBox-encode a heap pointer (returned by `gc.alloc`/`gc_alloc_thunk`).
/// The pointer's low 48 bits become the payload; tag bits go to TAG_PTR.
pub fn nanbox_ptr(raw: u64) -> u64 { nanbox_encode(TAG_PTR, raw & PAYLOAD_MASK) }

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
    if nanbox_tag(bits) != Some(TAG_PTR) { return false; }
    let p = nanbox_payload(bits) as *const u8;
    if p.is_null() { return false; }
    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
    tid == heap_type_ids().long
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
    if (bits & FULL_MASK) == TAG_PATTERN { 0x7FF8_0000_0000_0000 } else { bits }
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
        _ => panic!(
            "clojure-jvm: object_to_nanbox: variant {obj:?} not yet representable as NanBox"
        ),
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
    /// Shared ObjTypeId backing every `deftype`/`defrecord` instance.
    /// The actual user-type discriminator lives in the cell's Raw64
    /// `user_type_id` field and is read by `effective_type_id`.
    pub user_instance: usize,
    /// `clojure.lang.Reduced` — the wrapper produced by `(reduced x)` to
    /// signal early termination of `reduce`. One traced `Value` slot holds
    /// the wrapped value; `(deref r)` / `@r` unwraps it and
    /// `clojure.lang.RT/isReduced` tests for it.
    pub reduced: usize,
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

static USER_INSTANCE_LAYOUT: std::sync::OnceLock<UserInstanceLayout> =
    std::sync::OnceLock::new();

pub fn set_user_instance_layout(layout: UserInstanceLayout) {
    let _ = USER_INSTANCE_LAYOUT.set(layout);
}

pub fn user_instance_layout() -> UserInstanceLayout {
    *USER_INSTANCE_LAYOUT.get().expect(
        "clojure-jvm: user_instance_layout() called before Compiler::new ran",
    )
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
        _ => Object::Unported { java_class: "unknown NanBox tag" },
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
    debug_assert_eq!(nanbox_tag(bits), Some(TAG_PTR), "heap_bits_to_object: not a TAG_PTR");
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
        let arc_ptr = unsafe { ptr.add(8).cast::<u64>().read_unaligned() } as *const crate::lang::keyword::Keyword;
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
        let first_bits = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
        let rest_bits = unsafe { ptr.add(16).cast::<u64>().read_unaligned() };
        let meta_bits = unsafe { ptr.add(24).cast::<u64>().read_unaligned() };
        let first_obj = decode_value_bits(first_bits, ids);
        let rest_list = decode_rest_to_list(rest_bits, ids);
        let list_obj = Object::List(std::sync::Arc::new(crate::lang::persistent_list::PersistentList::Cons {
            first: first_obj,
            rest: rest_list,
            count: 1 + count_list(&rest_bits, ids),
        }));
        return wrap_with_meta_bits(list_obj, meta_bits, ids);
    }
    if type_id == ids.vector {
        // `clojure.lang.PersistentVector` (our flat varlen-values shape):
        // Header(8) + varlen-count(8) + N * 8 byte slots. The count word
        // lives at offset 8; items start at offset 16.
        let count = unsafe { ptr.add(8).cast::<u64>().read_unaligned() } as usize;
        let mut items: Vec<Object> = Vec::with_capacity(count);
        for i in 0..count {
            let off = 16 + i * 8;
            let bits = unsafe { ptr.add(off).cast::<u64>().read_unaligned() };
            items.push(decode_value_bits(bits, ids));
        }
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
        panic!(
            "clojure-jvm: alloc_arc_cell: gc_alloc returned null for type_id {type_id}"
        );
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
        _ => Object::Unported { java_class: "unknown NanBox tag in value field" },
    }
}

/// Decode a Cons `rest` field into an `Arc<PersistentList>`. Nil terminates;
/// a Cons pointer continues the chain.
fn decode_rest_to_list(bits: u64, ids: HeapTypeIds) -> std::sync::Arc<crate::lang::persistent_list::PersistentList> {
    use crate::lang::persistent_list::PersistentList;
    match nanbox_tag(bits) {
        Some(TAG_NIL) => PersistentList::empty(),
        Some(TAG_PTR) => {
            let obj = unsafe { heap_bits_to_object(bits, ids) };
            match obj {
                Object::List(l) => l,
                other => panic!(
                    "clojure-jvm: Cons.rest expected nil or list, got {other:?}"
                ),
            }
        }
        _ => panic!("clojure-jvm: Cons.rest must be nil or a Cons pointer"),
    }
}

/// Count the elements of a list-tail bits sequence (for setting count on the
/// reconstructed `PersistentList::Cons`). O(n) walk through the chain.
fn count_list(bits: &u64, ids: HeapTypeIds) -> i32 {
    let payload = nanbox_payload(*bits);
    if nanbox_tag(*bits) != Some(TAG_PTR) { return 0; }
    let ptr = payload as *const u8;
    if ptr.is_null() { return 0; }
    let type_id = unsafe { ptr.cast::<u16>().read_unaligned() } as usize;
    if type_id != ids.cons { return 0; }
    let next_bits = unsafe { ptr.add(16).cast::<u64>().read_unaligned() };
    1 + count_list(&next_bits, ids)
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
    assert!(
        base != 0,
        "clojure-jvm: call_table_base not installed — eval helpers must use install_call_table_base"
    );
    base
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
unsafe fn pack_variadic_args(self_arg: Option<u64>, fref_idx: u32, args: &[u64]) -> Option<Vec<u64>> {
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
    Some(dynobj::roots::with_scope(cap, |scope| {
        let self_root = self_arg.map(|s| scope.root::<()>(s));
        let arg_roots: Vec<_> = args.iter().map(|v| scope.root::<()>(*v)).collect();
        let tail_root = scope.root::<()>(nanbox_nil());
        // Fold overflow into a cons-list, right-to-left. Every read of
        // the args goes through `arg_roots[i].get()` so a GC during
        // a `cljvm_rt_cons` call relocates the slot in place.
        for i in (info.fixed_arity..args.len()).rev() {
            let v = arg_roots[i].get();
            let new_tail = unsafe { cljvm_rt_cons(v, tail_root.get()) };
            tail_root.set(new_tail);
        }
        let mut packed: Vec<u64> = Vec::with_capacity(cap);
        if let Some(r) = &self_root {
            packed.push(r.get());
        }
        for i in 0..info.fixed_arity {
            packed.push(arg_roots[i].get());
        }
        packed.push(tail_root.get());
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
    macro_rules! no_op_u64 { ($_:literal) => { u64 }; }
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
        18 => call_n!(18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17),
        19 => call_n!(19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
        20 => call_n!(20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
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
    if !matches!(nanbox_tag(fn_bits), Some(TAG_PTR)) { return None; }
    let p = nanbox_payload(fn_bits) as *const u8;
    if p.is_null() { return None; }
    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
    if tid == ids.vector {
        if !nanbox_tag(a_bits).is_none() { return None; }
        let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as i64;
        let idx = arg_to_i64(a_bits);
        if idx < 0 || idx >= n {
            panic!(
                "clojure-jvm: vector-as-fn index {idx} out of range [0,{n})"
            );
        }
        return Some(unsafe {
            p.add(16 + (idx as usize) * 8).cast::<u64>().read_unaligned()
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
        return Some(if s.contains(&key) { a_bits } else { nanbox_nil() });
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
            Some(crate::lang::compiler::with_active_session_encode_object(&v_obj))
        }
        _ => Some(not_found),
    }
}

/// `IFn.invoke(arg1, arg2)` — 2 arity. Handles `(:k m default)` and
/// any variadic target whose call-site arity exceeds its `fixed_arity`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_2(fn_bits: u64, a: u64, b: u64) -> u64 {
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
            let f: unsafe extern "C" fn(u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
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
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, 3) };
    if let Some(packed) = unsafe { pack_variadic_args(self_arg, fref_idx, &[a, b, c]) } {
        return unsafe { call_with_packed(ptr, &packed) };
    }
    match self_arg {
        None => {
            let f: unsafe extern "C" fn(u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
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
pub unsafe extern "C" fn cljvm_rt_invoke_4(
    fn_bits: u64, a: u64, b: u64, c: u64, d: u64,
) -> u64 {
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
    fn_bits: u64, a: u64, b: u64, c: u64, d: u64, e: u64,
) -> u64 {
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
    fn_bits: u64, a: u64, b: u64, c: u64, d: u64, e: u64, f: u64,
) -> u64 {
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
    fn_bits: u64, a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64,
) -> u64 {
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_with_arity(fn_bits, 7) };
    if let Some(packed) = unsafe { pack_variadic_args(self_arg, fref_idx, &[a, b, c, d, e, f, g]) } {
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
    fn_bits: u64, a: u64, b: u64, c: u64, d: u64, e: u64, f: u64, g: u64, h: u64,
) -> u64 {
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

/// `clojure.lang.RT/charCast(n)` — coerce to a u16 char value.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_charCast(n_bits: u64) -> u64 {
    let n = arg_to_i64(n_bits) as u16 as i64;
    box_long(n)
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_numbers_identity(a: u64) -> u64 { a }
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
        other => panic!(
            "clojure-jvm: Numbers/isZero requires a Number, got {other:?}"
        ),
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
                unsafe { p.add(16 + (idx as usize) * 8).cast::<u64>().read_unaligned() }
            } else if tid == ids.cons {
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
pub unsafe extern "C" fn cljvm_rt_nth_3(
    coll_bits: u64, idx_bits: u64, not_found_bits: u64,
) -> u64 {
    let ids = heap_type_ids();
    let idx = arg_to_i64(idx_bits);
    if idx < 0 { return not_found_bits; }
    match nanbox_tag(coll_bits) {
        Some(TAG_NIL) => not_found_bits,
        Some(TAG_PTR) => {
            let p = nanbox_payload(coll_bits) as *const u8;
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.vector {
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as i64;
                if idx >= n { return not_found_bits; }
                unsafe { p.add(16 + (idx as usize) * 8).cast::<u64>().read_unaligned() }
            } else if tid == ids.cons {
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
    let ids = heap_type_ids();
    let coerced_seq = match nanbox_tag(seq_bits) {
        Some(TAG_NIL) => seq_bits,
        Some(TAG_PTR) => {
            let p = nanbox_payload(seq_bits) as *const u8;
            if p.is_null() {
                seq_bits
            } else {
                let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                if tid == ids.cons {
                    seq_bits
                } else {
                    unsafe { cljvm_rt_seq(seq_bits) }
                }
            }
        }
        _ => panic!(
            "clojure-jvm: cljvm_rt_cons: second arg must be a collection or nil, \
             got bits 0x{seq_bits:x}"
        ),
    };
    dynobj::roots::with_scope(2, |scope| {
        let x = scope.root::<()>(x_bits);
        let seq = scope.root::<()>(coerced_seq);
        let raw = dynlang::gc::gc_alloc_thunk(ids.cons as u64, 0);
        let ptr = raw as *mut u8;
        if ptr.is_null() {
            panic!("clojure-jvm: cljvm_rt_cons: gc_alloc returned null");
        }
        let nil_bits = nanbox_nil();
        unsafe {
            ptr.add(8).cast::<u64>().write_unaligned(x.get());
            ptr.add(16).cast::<u64>().write_unaligned(seq.get());
            ptr.add(24).cast::<u64>().write_unaligned(nil_bits);
        }
        nanbox_encode(TAG_PTR, raw & PAYLOAD_MASK)
    })
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
                return dynobj::roots::with_scope(n + 1, |scope| {
                    let item_roots: Vec<dynobj::roots::Rooted<()>> = (0..n)
                        .map(|i| {
                            let bits = unsafe {
                                raw.add(16 + i * 8).cast::<u64>().read_unaligned()
                            };
                            scope.root::<()>(bits)
                        })
                        .collect();
                    let x_root = scope.root::<()>(x_bits);
                    let new_raw = dynlang::gc::gc_alloc_thunk(
                        ids.vector as u64,
                        (n + 1) as u64,
                    );
                    let new_ptr = new_raw as *mut u8;
                    if new_ptr.is_null() {
                        panic!("clojure-jvm: RT.conj: gc_alloc returned null for Vector");
                    }
                    for i in 0..n {
                        unsafe {
                            new_ptr
                                .add(16 + i * 8)
                                .cast::<u64>()
                                .write_unaligned(item_roots[i].get());
                        }
                    }
                    unsafe {
                        new_ptr
                            .add(16 + n * 8)
                            .cast::<u64>()
                            .write_unaligned(x_root.get());
                    }
                    nanbox_ptr(new_raw)
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
                    new_ptr.add(8).cast::<u64>().write_unaligned(raw_arc);
                }
                return nanbox_ptr(new_raw);
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

/// `clojure.lang.RT.assoc(Object coll, Object key, Object val)` —
/// associate `key → val` in `coll` and return the resulting collection.
/// Java dispatches via `IPersistentCollection`; we cover the cases the
/// bootstrap of `clojure.core` actually exercises:
///   * nil  → a fresh singleton map `{key val}`
///   * Map  → a new map with the entry assoc'd (Arc-shared with `coll`)
/// Vectors are panic'd until the bootstrap actually needs them; that
/// keeps the failure loud rather than silently succeeding for a wrong
/// receiver type.
///
/// Newly-built `Arc<PersistentHashMap>` values are pushed onto the
/// active Session's `roots._maps` so the heap-cell's Raw64 pointer
/// stays valid for the lifetime of the JIT module.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_assoc(coll_bits: u64, key_bits: u64, val_bits: u64) -> u64 {
    let ids = heap_type_ids();
    // Decode key + val to host-side Objects so the Arc<PersistentHashMap>
    // stores them by-value. (Maps don't put their entries on the GC heap
    // — the Arc owns a Vec<(Object, Object)>.)
    let key_obj = any_bits_to_object(key_bits, ids);
    let val_obj = any_bits_to_object(val_bits, ids);

    let new_map: Arc<crate::lang::persistent_hash_map::PersistentHashMap> =
        match nanbox_tag(coll_bits) {
            Some(TAG_NIL) => {
                crate::lang::persistent_hash_map::PersistentHashMap::create_pairs(vec![(
                    key_obj, val_obj,
                )])
            }
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
        panic!(
            "clojure-jvm: RT.subvec — first arg type_id {src_type} is not a Vector"
        );
    }
    let src_count = unsafe { src_raw.add(8).cast::<u64>().read_unaligned() } as usize;
    if end > src_count {
        panic!(
            "clojure-jvm: RT.subvec — end ({end}) > count ({src_count})"
        );
    }

    let n = end - start;
    let new_raw = dynlang::gc::gc_alloc_thunk(ids.vector as u64, n as u64);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: RT.subvec: gc_alloc returned null");
    }
    for i in 0..n {
        let src_off = 16 + (start + i) * 8;
        let bits = unsafe { src_raw.add(src_off).cast::<u64>().read_unaligned() };
        unsafe {
            new_ptr.add(16 + i * 8).cast::<u64>().write_unaligned(bits);
        }
    }
    nanbox_ptr(new_raw)
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
                if type_id == ids.cons {
                    let mut cur = coll_bits;
                    loop {
                        match nanbox_tag(cur) {
                            Some(TAG_NIL) => break,
                            Some(TAG_PTR) => {
                                let p = nanbox_payload(cur) as *const u8;
                                if p.is_null() { break; }
                                let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                                if tid != ids.cons { break; }
                                item_bits.push(unsafe { p.add(8).cast::<u64>().read_unaligned() });
                                cur = unsafe { p.add(16).cast::<u64>().read_unaligned() };
                            }
                            _ => break,
                        }
                    }
                } else if type_id == ids.vector {
                    let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
                    item_bits.reserve(n);
                    for i in 0..n {
                        item_bits.push(unsafe { raw.add(16 + i * 8).cast::<u64>().read_unaligned() });
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
                if type_id == ids.cons {
                    let mut cur = coll_bits;
                    loop {
                        match nanbox_tag(cur) {
                            Some(TAG_NIL) => break,
                            Some(TAG_PTR) => {
                                let p = nanbox_payload(cur) as *const u8;
                                if p.is_null() { break; }
                                let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                                if tid != ids.cons { break; }
                                items.push(unsafe { p.add(8).cast::<u64>().read_unaligned() });
                                cur = unsafe { p.add(16).cast::<u64>().read_unaligned() };
                            }
                            _ => break,
                        }
                    }
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
                Ordering::Equal => { placed = true; break; }
                Ordering::Less => { idx = i; break; }
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
                if type_id == ids.cons {
                    let mut cur = coll_bits;
                    loop {
                        match nanbox_tag(cur) {
                            Some(TAG_NIL) => break,
                            Some(TAG_PTR) => {
                                let p = nanbox_payload(cur) as *const u8;
                                if p.is_null() { break; }
                                let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                                if tid != ids.cons { break; }
                                out.push(unsafe { p.add(8).cast::<u64>().read_unaligned() });
                                cur = unsafe { p.add(16).cast::<u64>().read_unaligned() };
                            }
                            _ => break,
                        }
                    }
                } else if type_id == ids.vector {
                    let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
                    out.reserve(n);
                    for i in 0..n {
                        out.push(unsafe { raw.add(16 + i * 8).cast::<u64>().read_unaligned() });
                    }
                } else {
                    panic!(
                        "clojure-jvm: {caller} — receiver type_id {type_id} not yet supported"
                    );
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
                if type_id == ids.cons {
                    let mut cur = coll_bits;
                    loop {
                        match nanbox_tag(cur) {
                            Some(TAG_NIL) => break,
                            Some(TAG_PTR) => {
                                let p = nanbox_payload(cur) as *const u8;
                                if p.is_null() { break; }
                                let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                                if tid != ids.cons { break; }
                                items.push(unsafe { p.add(8).cast::<u64>().read_unaligned() });
                                cur = unsafe { p.add(16).cast::<u64>().read_unaligned() };
                            }
                            _ => break,
                        }
                    }
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
            let cmp_ret = unsafe {
                cljvm_rt_invoke_2(cmp_bits, kb, sorted[i].0)
            };
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
                if type_id == ids.cons {
                    let mut cur = coll_bits;
                    loop {
                        match nanbox_tag(cur) {
                            Some(TAG_NIL) => break,
                            Some(TAG_PTR) => {
                                let p = nanbox_payload(cur) as *const u8;
                                if p.is_null() { break; }
                                let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                                if tid != ids.cons { break; }
                                items.push(unsafe { p.add(8).cast::<u64>().read_unaligned() });
                                cur = unsafe { p.add(16).cast::<u64>().read_unaligned() };
                            }
                            _ => break,
                        }
                    }
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

/// `clojure.lang.LazilyPersistentVector.create(Object coll)` — Java's
/// builder produces a lazily-realized vector wrapping the seq; we
/// eagerly walk the seq into a fresh `PersistentVector` heap cell.
/// Sufficient for upstream `(defn vector …)`'s variadic clause:
///   `(. LazilyPersistentVector (create (cons a (cons b … args))))`
/// Items are copied as raw NanBox bits — preserves any heap pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_lpv_create(coll_bits: u64) -> u64 {
    let ids = heap_type_ids();
    // Collect items by walking the seq via cons-rest. Bottom-out at nil
    // or any non-cons (per `RT.next`'s contract). For Vector input,
    // copy items directly.
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
                        let bits =
                            unsafe { raw.add(16 + i * 8).cast::<u64>().read_unaligned() };
                        items.push(bits);
                    }
                } else if type_id == ids.cons {
                    let mut cur = coll_bits;
                    loop {
                        match nanbox_tag(cur) {
                            Some(TAG_NIL) => break,
                            Some(TAG_PTR) => {
                                let p = nanbox_payload(cur) as *const u8;
                                if p.is_null() { break; }
                                let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                                if tid != ids.cons { break; }
                                let f = unsafe { p.add(8).cast::<u64>().read_unaligned() };
                                items.push(f);
                                cur = unsafe { p.add(16).cast::<u64>().read_unaligned() };
                            }
                            _ => break,
                        }
                    }
                } else {
                    // Unsupported source type: produce empty vector
                    // rather than panic. extern "C" panics here abort
                    // the process via panic-during-panic.
                    eprintln!(
                        "[cljvm-stub] LazilyPersistentVector/create: \
                         receiver type_id {type_id} not yet supported, \
                         returning empty vector"
                    );
                }
            }
        }
        _ => panic!(
            "clojure-jvm: LazilyPersistentVector/create — first arg must \
             be a collection or nil, got NanBox bits 0x{coll_bits:x}"
        ),
    }
    let n = items.len();
    let new_raw = dynlang::gc::gc_alloc_thunk(ids.vector as u64, n as u64);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: LazilyPersistentVector/create: gc_alloc returned null");
    }
    for (i, bits) in items.iter().enumerate() {
        unsafe {
            new_ptr.add(16 + i * 8).cast::<u64>().write_unaligned(*bits);
        }
    }
    nanbox_ptr(new_raw)
}

/// `clojure.lang.RT.toArray(Object coll)` — copy `coll` into a fresh
/// Object[]. We don't model Java arrays distinctly; we use our own
/// Vector heap cell as the closest analog (callers in Clojure-land
/// almost always pass the result to seq operations that work on
/// vectors). Handles nil → empty vector, Vector → copy, Cons → walk
/// and collect.
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
                        let bits =
                            unsafe { raw.add(16 + i * 8).cast::<u64>().read_unaligned() };
                        items.push(bits);
                    }
                } else if type_id == ids.cons {
                    let mut cur = coll_bits;
                    loop {
                        match nanbox_tag(cur) {
                            Some(TAG_NIL) => break,
                            Some(TAG_PTR) => {
                                let p = nanbox_payload(cur) as *const u8;
                                if p.is_null() {
                                    break;
                                }
                                let tid =
                                    unsafe { p.cast::<u16>().read_unaligned() } as usize;
                                if tid != ids.cons {
                                    break;
                                }
                                let f =
                                    unsafe { p.add(8).cast::<u64>().read_unaligned() };
                                items.push(f);
                                cur =
                                    unsafe { p.add(16).cast::<u64>().read_unaligned() };
                            }
                            _ => break,
                        }
                    }
                } else {
                    panic!(
                        "clojure-jvm: RT.toArray — receiver type_id {type_id} not yet \
                         supported (extend cljvm_rt_toArray)"
                    );
                }
            }
        }
        _ => panic!(
            "clojure-jvm: RT.toArray — first arg must be a collection or nil, \
             got NanBox bits 0x{coll_bits:x}"
        ),
    }
    let n = items.len();
    let new_raw = dynlang::gc::gc_alloc_thunk(ids.vector as u64, n as u64);
    let new_ptr = new_raw as *mut u8;
    if new_ptr.is_null() {
        panic!("clojure-jvm: RT.toArray: gc_alloc returned null for Vector");
    }
    for (i, bits) in items.iter().enumerate() {
        unsafe {
            new_ptr.add(16 + i * 8).cast::<u64>().write_unaligned(*bits);
        }
    }
    nanbox_ptr(new_raw)
}

/// `clojure.lang.RT.first(Object coll)` — return the first item or nil.
/// Java: handles ISeq, Seqable, nil. We cover nil/Cons/Vector now;
/// Map iteration goes through `seq` first (Java does the same).
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
                unsafe { ptr.add(8).cast::<u64>().read_unaligned() }
            } else if type_id == ids.vector {
                // Vector: header(8) + count(8) + items at offset 16.
                let n = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
                if n == 0 {
                    nanbox_nil()
                } else {
                    unsafe { ptr.add(16).cast::<u64>().read_unaligned() }
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
    eprintln!("[cljvm-stub] Var/popThreadBindings not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_getThreadBindingFrame() -> u64 {
    eprintln!("[cljvm-stub] Var/getThreadBindingFrame not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_getThreadBindings() -> u64 {
    eprintln!("[cljvm-stub] Var/getThreadBindings not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_find(_sym: u64) -> u64 {
    eprintln!("[cljvm-stub] Var/find not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_intern_2(_ns: u64, _sym: u64) -> u64 {
    eprintln!("[cljvm-stub] Var/intern (2-arg) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_intern_3(_ns: u64, _sym: u64, _val: u64) -> u64 {
    eprintln!("[cljvm-stub] Var/intern (3-arg) not yet implemented");     return nanbox_nil();
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
    eprintln!("[cljvm-stub] (.alterMeta) not yet implemented");     return nanbox_nil();
}

/// `(.bindRoot v val)` — set a Var's root binding.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_bindRoot(_recv: u64, _val: u64) -> u64 {
    eprintln!("[cljvm-stub] (.bindRoot) not yet implemented — defn handles binding via the def special form");     return nanbox_nil();
}

/// `(.hasRoot v)` — does this Var have a root binding?
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_hasRoot(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.hasRoot) not yet implemented");     return nanbox_nil();
}

/// `(.getRawRoot v)` — read Var's root without dynamic-binding lookup.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getRawRoot(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getRawRoot) not yet implemented");     return nanbox_nil();
}

// IRef validators / watches / state — stub instance methods.
// Real impls would live on the Atom / Ref / Agent heap cells which
// aren't modeled yet. These are present so defns/defmacros referencing
// them analyze, even though calling them would panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_setValidator(_recv: u64, _v: u64) -> u64 {
    eprintln!("[cljvm-stub] (.setValidator) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getValidator(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getValidator) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getWatches(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getWatches) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_addWatch(_recv: u64, _key: u64, _f: u64) -> u64 {
    eprintln!("[cljvm-stub] (.addWatch) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_removeWatch(_recv: u64, _key: u64) -> u64 {
    eprintln!("[cljvm-stub] (.removeWatch) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_swap(_recv: u64, _f: u64) -> u64 {
    eprintln!("[cljvm-stub] (.swap) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_reset(_recv: u64, _new: u64) -> u64 {
    eprintln!("[cljvm-stub] (.reset) on IAtom not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_compareAndSet(_recv: u64, _old: u64, _new: u64) -> u64 {
    eprintln!("[cljvm-stub] (.compareAndSet) not yet implemented");     return nanbox_nil();
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
        }
    }
    eprintln!("[cljvm-stub] (.deref) on unsupported receiver bits 0x{recv:x} — nil");
    nanbox_nil()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_iterator(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.iterator) not yet implemented");     return nanbox_nil();
}

// Agent error-handling stubs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_setErrorHandler(_recv: u64, _h: u64) -> u64 {
    eprintln!("[cljvm-stub] (.setErrorHandler) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getErrorHandler(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getErrorHandler) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_setErrorMode(_recv: u64, _m: u64) -> u64 {
    eprintln!("[cljvm-stub] (.setErrorMode) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getErrorMode(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getErrorMode) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getError(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getError) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_dispatch(_recv: u64, _f: u64, _args: u64, _exec: u64) -> u64 {
    eprintln!("[cljvm-stub] (.dispatch) on Agent not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_restart(_recv: u64, _new: u64, _clear: u64) -> u64 {
    eprintln!("[cljvm-stub] (.restart) on Agent not yet implemented");     return nanbox_nil();
}

// Ref stubs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_set(_recv: u64, _val: u64) -> u64 {
    eprintln!("[cljvm-stub] (.set) on Ref not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_alter(_recv: u64, _f: u64, _args: u64) -> u64 {
    eprintln!("[cljvm-stub] (.alter) on Ref not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_commute(_recv: u64, _f: u64, _args: u64) -> u64 {
    eprintln!("[cljvm-stub] (.commute) on Ref not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_ensure(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.ensure) on Ref not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_setMinHistory(_recv: u64, _n: u64) -> u64 {
    eprintln!("[cljvm-stub] (.setMinHistory) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_setMaxHistory(_recv: u64, _n: u64) -> u64 {
    eprintln!("[cljvm-stub] (.setMaxHistory) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getMinHistory(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getMinHistory) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getMaxHistory(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getMaxHistory) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_getHistoryCount(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.getHistoryCount) not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_trimHistory(_recv: u64) -> u64 {
    eprintln!("[cljvm-stub] (.trimHistory) not yet implemented");     return nanbox_nil();
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
pub unsafe extern "C" fn cljvm_unimpl_host_call_5(_a: u64, _b: u64, _c: u64, _d: u64, _e: u64) -> u64 {
    eprintln!("[cljvm-unimpl-host-call] arity 5 — returning nil");
    nanbox_nil()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_unimpl_host_call_6(_a: u64, _b: u64, _c: u64, _d: u64, _e: u64, _f: u64) -> u64 {
    eprintln!("[cljvm-unimpl-host-call] arity 6 — returning nil");
    nanbox_nil()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_resetThreadBindingFrame(_f: u64) -> u64 {
    eprintln!("[cljvm-stub] Var/resetThreadBindingFrame not yet implemented");     return nanbox_nil();
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_cloneThreadBindingFrame() -> u64 {
    eprintln!("[cljvm-stub] Var/cloneThreadBindingFrame not yet implemented");     return nanbox_nil();
}

/// `(.reset multifn)` — `clojure.lang.MultiFn.reset()`. We don't model
/// multimethods yet; calling this raises a clear error at runtime, but
/// upstream defns that only DECLARE wrappers around it (`remove-all-methods`)
/// can still compile because the call site lowers to this extern.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_multifn_reset(_recv: u64) -> u64 {
    panic!(
        "clojure-jvm: clojure.lang.MultiFn.reset() — multimethods not yet \
         implemented. Defmulti/defmethod aren't wired; only the wrapper \
         defns analyze cleanly."
    );
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_multifn_addMethod(_a: u64, _b: u64, _c: u64) -> u64 {
    panic!("clojure-jvm: MultiFn.addMethod not yet implemented");
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_multifn_removeMethod(_a: u64, _b: u64) -> u64 {
    panic!("clojure-jvm: MultiFn.removeMethod not yet implemented");
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_multifn_preferMethod(_a: u64, _b: u64, _c: u64) -> u64 {
    panic!("clojure-jvm: MultiFn.preferMethod not yet implemented");
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
    panic!(
        "clojure-jvm: (.getNamespace) on non-Named receiver bits 0x{x_bits:x}"
    );
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
                if n == 0 { return nanbox_nil(); }
                let items: Vec<u64> = (0..n)
                    .map(|i| unsafe { p.add(16 + i * 8).cast::<u64>().read_unaligned() })
                    .collect();
                return dynobj::roots::with_scope(items.len() + 1, |scope| {
                    let roots: Vec<_> = items.iter().map(|v| scope.root::<()>(*v)).collect();
                    let tail = scope.root::<()>(nanbox_nil());
                    // Forward iteration (NOT reversed) since we want reverse-order seq:
                    // walk left→right and prepend so the last item ends up at head.
                    for r in roots.iter() {
                        let new_tail = unsafe { cljvm_rt_cons(r.get(), tail.get()) };
                        tail.set(new_tail);
                    }
                    tail.get()
                });
            }
        }
    }
    eprintln!("[cljvm-stub] (.rseq) on unsupported receiver bits 0x{rev_bits:x}");     return nanbox_nil();
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
    eprintln!("[cljvm-stub] (.getKey) on non-MapEntry receiver bits 0x{e_bits:x}");     return nanbox_nil();
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
    eprintln!("[cljvm-stub] (.getValue) on non-MapEntry receiver bits 0x{e_bits:x}");     return nanbox_nil();
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
            if p.is_null() { return nanbox_nil(); }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid != ids.set {
                eprintln!("[cljvm-stub] (.disjoin) on non-set type_id {tid}");                 return nanbox_nil();
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
            if np.is_null() { panic!("clojure-jvm: (.disjoin) alloc null"); }
            unsafe { np.add(8).cast::<u64>().write_unaligned(raw_arc); }
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
            if p.is_null() { return nanbox_bool(false); }
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
                if !nanbox_tag(key_bits).is_none() { return nanbox_bool(false); }
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
            if p.is_null() { return not_found; }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.map {
                let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_hash_map::PersistentHashMap;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let m = unsafe { Arc::from_raw(arc_ptr) };
                let key = any_bits_to_object(key_bits, ids);
                let v = m.val_at(&key);
                if matches!(v, Object::Nil) {
                    if !m.contains_key(&key) { return not_found; }
                }
                return crate::lang::compiler::with_active_session_encode_object(&v);
            }
            if tid == ids.vector {
                if !nanbox_tag(key_bits).is_none() { return not_found; }
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as i64;
                let idx = arg_to_i64(key_bits);
                if idx < 0 || idx >= n { return not_found; }
                return unsafe {
                    p.add(16 + (idx as usize) * 8).cast::<u64>().read_unaligned()
                };
            }
            if tid == ids.set {
                let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                    as *const crate::lang::persistent_hash_set::PersistentHashSet;
                unsafe { Arc::increment_strong_count(arc_ptr) };
                let s = unsafe { Arc::from_raw(arc_ptr) };
                let key = any_bits_to_object(key_bits, ids);
                if s.contains(&key) { key_bits } else { not_found }
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
            if p.is_null() { return nanbox_nil(); }
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
                    dynobj::roots::with_scope(2, |scope| {
                        let kr = scope.root::<()>(kbits);
                        let vr = scope.root::<()>(vbits);
                        let raw = dynlang::gc::gc_alloc_thunk(ids.vector as u64, 2);
                        let np = raw as *mut u8;
                        unsafe {
                            np.add(8).cast::<u64>().write_unaligned(2);
                            np.add(16).cast::<u64>().write_unaligned(kr.get());
                            np.add(24).cast::<u64>().write_unaligned(vr.get());
                        }
                        nanbox_encode(TAG_PTR, raw & PAYLOAD_MASK)
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
            if p.is_null() { return nanbox_nil(); }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
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
            if np.is_null() { panic!("clojure-jvm: RT.dissoc alloc null"); }
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
            if p.is_null() { return nanbox_nil(); }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid != ids.map {
                panic!("clojure-jvm: RT.keys on non-map type_id {tid}");
            }
            let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                as *const crate::lang::persistent_hash_map::PersistentHashMap;
            unsafe { Arc::increment_strong_count(arc_ptr) };
            let m = unsafe { Arc::from_raw(arc_ptr) };
            let keys: Vec<u64> = m.iter()
                .map(|(k, _)| crate::lang::compiler::with_active_session_encode_object(&k))
                .collect();
            if keys.is_empty() { return nanbox_nil(); }
            dynobj::roots::with_scope(keys.len() + 1, |scope| {
                let roots: Vec<_> = keys.iter().map(|v| scope.root::<()>(*v)).collect();
                let tail = scope.root::<()>(nanbox_nil());
                for r in roots.iter().rev() {
                    let new_tail = unsafe { cljvm_rt_cons(r.get(), tail.get()) };
                    tail.set(new_tail);
                }
                tail.get()
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
            if p.is_null() { return nanbox_nil(); }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid != ids.map {
                panic!("clojure-jvm: RT.vals on non-map type_id {tid}");
            }
            let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
                as *const crate::lang::persistent_hash_map::PersistentHashMap;
            unsafe { Arc::increment_strong_count(arc_ptr) };
            let m = unsafe { Arc::from_raw(arc_ptr) };
            let vals: Vec<u64> = m.iter()
                .map(|(_, v)| crate::lang::compiler::with_active_session_encode_object(&v))
                .collect();
            if vals.is_empty() { return nanbox_nil(); }
            dynobj::roots::with_scope(vals.len() + 1, |scope| {
                let roots: Vec<_> = vals.iter().map(|v| scope.root::<()>(*v)).collect();
                let tail = scope.root::<()>(nanbox_nil());
                for r in roots.iter().rev() {
                    let new_tail = unsafe { cljvm_rt_cons(r.get(), tail.get()) };
                    tail.set(new_tail);
                }
                tail.get()
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
            if p.is_null() { return nanbox_nil(); }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.cons {
                unsafe { p.add(8).cast::<u64>().read_unaligned() }
            } else if tid == ids.vector {
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as usize;
                if n == 0 { nanbox_nil() } else {
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
            if p.is_null() { return nanbox_nil(); }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.cons {
                unsafe { p.add(16).cast::<u64>().read_unaligned() }
            } else if tid == ids.vector {
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as usize;
                if n == 0 {
                    panic!("clojure-jvm: RT.pop on empty vector");
                }
                let new_n = n - 1;
                dynobj::roots::with_scope(new_n + 1, |scope| {
                    let item_roots: Vec<_> = (0..new_n)
                        .map(|i| {
                            let v = unsafe {
                                p.add(16 + i * 8).cast::<u64>().read_unaligned()
                            };
                            scope.root::<()>(v)
                        })
                        .collect();
                    let new_raw = dynlang::gc::gc_alloc_thunk(ids.vector as u64, new_n as u64);
                    let nptr = new_raw as *mut u8;
                    if nptr.is_null() {
                        panic!("clojure-jvm: RT.pop alloc returned null");
                    }
                    unsafe {
                        nptr.add(8).cast::<u64>().write_unaligned(new_n as u64);
                        for (i, r) in item_roots.iter().enumerate() {
                            nptr.add(16 + i * 8).cast::<u64>().write_unaligned(r.get());
                        }
                    }
                    nanbox_encode(TAG_PTR, new_raw & PAYLOAD_MASK)
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
        let a_arc = unsafe { a_ptr.add(8).cast::<u64>().read_unaligned() } as *const crate::lang::symbol::Symbol;
        let b_arc = unsafe { b_ptr.add(8).cast::<u64>().read_unaligned() } as *const crate::lang::symbol::Symbol;
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

/// `clojure.lang.Util.identical(a, b)` — Java reference equality.
/// Bit-equality on NanBox values handles nil/bool/long/double directly,
/// pointer-identity for heap cells, and (because Symbol/Keyword interning
/// produces a single Arc per name) reduces to pointer-equality for
/// interned identity types as well.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_util_identical(a_bits: u64, b_bits: u64) -> u64 {
    nanbox_bool(a_bits == b_bits)
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
                dynobj::roots::with_scope(n + 1, |scope| {
                    let item_roots: Vec<dynobj::roots::Rooted<()>> = (0..n)
                        .map(|i| {
                            let bits = unsafe {
                                ptr.add(16 + i * 8).cast::<u64>().read_unaligned()
                            };
                            scope.root::<()>(bits)
                        })
                        .collect();
                    // Root `cur` so it survives each cons-call's GC.
                    let cur_root = scope.root::<()>(nanbox_nil());
                    for i in (0..n).rev() {
                        let item = item_roots[i].get();
                        let new_cur = unsafe { cljvm_rt_cons(item, cur_root.get()) };
                        cur_root.set(new_cur);
                    }
                    cur_root.get()
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
                dynobj::roots::with_scope(n + 1, |scope| {
                    let tail = scope.root::<()>(nanbox_nil());
                    for (k, v) in pairs.into_iter() {
                        let entry = Object::Vector(
                            crate::lang::persistent_vector::PersistentVector::create(
                                vec![k, v],
                            ),
                        );
                        let entry_bits =
                            crate::lang::compiler::with_active_session_encode_object(&entry);
                        let e = scope.root::<()>(entry_bits);
                        let new_tail = unsafe { cljvm_rt_cons(e.get(), tail.get()) };
                        tail.set(new_tail);
                    }
                    tail.get()
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
                dynobj::roots::with_scope(n + 1, |scope| {
                    let tail = scope.root::<()>(nanbox_nil());
                    for e in elems.into_iter() {
                        let e_bits =
                            crate::lang::compiler::with_active_session_encode_object(&e);
                        let er = scope.root::<()>(e_bits);
                        let new_tail = unsafe { cljvm_rt_cons(er.get(), tail.get()) };
                        tail.set(new_tail);
                    }
                    tail.get()
                })
            } else if type_id == ids.string {
                // String → seq of Character. Bootstrap doesn't need this
                // path yet; surface loudly so callers see what to add.
                panic!(
                    "clojure-jvm: RT.seq on String — needs Character heap type"
                );
            } else if type_id == ids.lazy_seq {
                // Force the thunk; recurse on the realized value.
                let arc: std::sync::Arc<std::cell::RefCell<LazyState>> =
                    unsafe { decode_arc_cell(ptr) };
                let cached = {
                    let st = arc.borrow();
                    if st.realized { Some(st.value_bits) } else { None }
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
                dynobj::roots::with_scope(items.len() + 1, |scope| {
                    let roots: Vec<_> =
                        items.iter().map(|v| scope.root::<()>(*v)).collect();
                    let mut acc = nanbox_nil();
                    let acc_root = scope.root::<()>(acc);
                    for r in roots.iter().rev() {
                        acc = unsafe { cljvm_rt_cons(r.get(), acc_root.get()) };
                        acc_root.set(acc);
                    }
                    acc_root.get()
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
            let info = crate::lang::host_class::by_id(
                crate::lang::host_class::ClassId(raw_id as u16),
            );
            match info.ctor {
                Some(c) => c(args, ids),
                None => {
                    eprintln!(
                        "[cljvm-stub] (new {} ...) — class has no registered \
                         constructor, returning nil",
                        info.name
                    );
                    nanbox_nil()
                }
            }
        }
        _ => {
            eprintln!(
                "[cljvm-stub] (new ...) — receiver bits 0x{class_bits:x} is \
                 not a Class heap pointer, returning nil"
            );
            nanbox_nil()
        }
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
    let ids = heap_type_ids();
    // Coerce non-cons seqables (Vector, etc.) into a cons-list.
    let coerced = match nanbox_tag(args_bits) {
        Some(TAG_NIL) => args_bits,
        Some(TAG_PTR) => {
            let p = nanbox_payload(args_bits) as *const u8;
            if p.is_null() {
                args_bits
            } else {
                let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                if tid == ids.cons {
                    args_bits
                } else {
                    unsafe { cljvm_rt_seq(args_bits) }
                }
            }
        }
        _ => panic!(
            "clojure-jvm: applyTo: args must be a seqable, got bits 0x{args_bits:x}"
        ),
    };
    // Walk all args from the cons-chain.
    let mut walked: Vec<u64> = Vec::new();
    let mut cur = coerced;
    loop {
        match nanbox_tag(cur) {
            Some(TAG_NIL) => break,
            Some(TAG_PTR) => {
                let p = nanbox_payload(cur) as *const u8;
                if p.is_null() { break; }
                let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                if tid != ids.cons { break; }
                walked.push(unsafe { p.add(8).cast::<u64>().read_unaligned() });
                cur = unsafe { p.add(16).cast::<u64>().read_unaligned() };
            }
            _ => break,
        }
    }

    let (ptr, self_arg, fref_idx) =
        unsafe { dispatch_with_arity(fn_bits, walked.len()) };
    let info = crate::lang::compiler::with_active_compiler_arity(fref_idx);

    // Build the final arg vector that gets passed to the native fn.
    // Must be rooted because cons-folding the variadic tail can trigger GC.
    let final_args = dynobj::roots::with_scope(walked.len() + 2, |scope| {
        let arg_roots: Vec<_> = walked.iter().map(|v| scope.root::<()>(*v)).collect();
        let mut packed: Vec<u64> = Vec::with_capacity(walked.len() + 2);
        if let Some(s) = self_arg {
            packed.push(s);
        }
        match info {
            Some(i) if i.is_variadic => {
                // Take fixed_arity args directly, fold the rest into a cons-list tail.
                for j in 0..i.fixed_arity.min(arg_roots.len()) {
                    packed.push(arg_roots[j].get());
                }
                // Pad missing fixed args with nil (matches Clojure's
                // arity behavior for too-few args to a variadic fn —
                // it accepts as long as args.len() >= fixed_arity, but
                // we err on the side of nil-padding for tolerance).
                let self_offset = if self_arg.is_some() { 1 } else { 0 };
                while packed.len() - self_offset < i.fixed_arity {
                    packed.push(nanbox_nil());
                }
                let tail_root = scope.root::<()>(nanbox_nil());
                for j in (i.fixed_arity..arg_roots.len()).rev() {
                    let v = arg_roots[j].get();
                    let new_tail = unsafe { cljvm_rt_cons(v, tail_root.get()) };
                    tail_root.set(new_tail);
                }
                packed.push(tail_root.get());
            }
            _ => {
                // Non-variadic (or unknown arity — assume fixed).
                // Pass args through 1:1; arity mismatch surfaces as a
                // wrong-shape native call at the C ABI boundary.
                for r in &arg_roots {
                    packed.push(r.get());
                }
            }
        }
        packed
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
            let items: Vec<String> =
                l.iter().map(|o| format_object_pr(&o)).collect();
            format!("({})", items.join(" "))
        }
        Object::Vector(v) => {
            let mut items: Vec<String> = Vec::with_capacity(v.count() as usize);
            for i in 0..v.count() {
                items.push(format_object_pr(&v.nth(i)));
            }
            format!("[{}]", items.join(" "))
        }
        Object::WithMeta(inner, _) => format_object_for_str(inner),
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
                panic!("clojure-jvm: Keyword/find arity-1: receiver type_id {tid} not Symbol or String");
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
    let var_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
        as *const crate::lang::var::Var;
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
    let arc_ptr = unsafe { p.add(8).cast::<u64>().read_unaligned() }
        as *const crate::lang::keyword::Keyword;
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
pub unsafe extern "C" fn cljvm_inst_refer(
    ns_bits: u64,
    sym_bits: u64,
    var_bits: u64,
) -> u64 {
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
        panic!(
            "clojure-jvm: (.setMacro v) — receiver type_id {type_id} is not a Var"
        );
    }
    let var_ptr = unsafe { raw.add(8).cast::<u64>().read_unaligned() }
        as *const crate::lang::var::Var;
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
    } else {
        panic!(
            "clojure-jvm: (.getName x) — receiver type_id {type_id} has no \
             name (extend cljvm_inst_getName)"
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
        ptr.add(16).copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
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
    if let Some(e) = table.iter().find(|e| !e.is_variadic && e.fixed_arity as usize == n) {
        return Some(e);
    }
    table.iter().find(|e| e.is_variadic && (e.fixed_arity as usize) <= n)
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
        panic!(
            "clojure-jvm: MultiArityFn dispatch — no clause matches {n} args"
        )
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
    // First flatten every input seq into one big Vec<u64> of items.
    // Each input is either nil, a Cons, or a Vector. Anything else
    // panics — syntax-quote should only produce these shapes.
    if std::env::var("CLJVM_SQ_TRACE").is_ok() {
        eprintln!("[sqConcat] xss=0x{xss_bits:x} tag={:?}", nanbox_tag(xss_bits));
    }
    let ids = heap_type_ids();
    let mut all: Vec<u64> = Vec::new();
    let mut cur = xss_bits;
    loop {
        match nanbox_tag(cur) {
            Some(TAG_NIL) => break,
            Some(TAG_PTR) => {
                let p = nanbox_payload(cur) as *const u8;
                if p.is_null() { break; }
                let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                if tid != ids.cons {
                    panic!(
                        "clojure-jvm: sqConcat: outer container must be a cons-list \
                         (the syntax-quote walker emits (list …)); got tid={tid}"
                    );
                }
                let head = unsafe { p.add(8).cast::<u64>().read_unaligned() };
                let tail = unsafe { p.add(16).cast::<u64>().read_unaligned() };
                // Walk `head` (a single seq) and append its items.
                walk_seq_into(head, &ids, &mut all);
                cur = tail;
            }
            _ => panic!(
                "clojure-jvm: sqConcat: outer arg must be a seq, got bits 0x{cur:x} \
                 (xss=0x{xss_bits:x}, items_so_far={})",
                all.len(),
            ),
        }
    }
    // Build the result as a cons-chain right-to-left, rooting accumulators
    // so allocations don't strand earlier nodes.
    dynobj::roots::with_scope(all.len() + 1, |scope| {
        let item_roots: Vec<_> = all.iter().map(|v| scope.root::<()>(*v)).collect();
        let tail_root = scope.root::<()>(nanbox_nil());
        for r in item_roots.iter().rev() {
            let new_tail = unsafe { cljvm_rt_cons(r.get(), tail_root.get()) };
            tail_root.set(new_tail);
        }
        tail_root.get()
    })
}

/// Append every element of `seq_bits` (a Cons / Vector / nil) to `out`.
/// Anything else is appended as a single element (matches Clojure's
/// concat behavior on non-seq args, modulo lazy-seq).
fn walk_seq_into(seq_bits: u64, ids: &HeapTypeIds, out: &mut Vec<u64>) {
    match nanbox_tag(seq_bits) {
        Some(TAG_NIL) => {}
        Some(TAG_PTR) => {
            let p = nanbox_payload(seq_bits) as *const u8;
            if p.is_null() { return; }
            let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
            if tid == ids.cons {
                let mut cur = seq_bits;
                while let Some(TAG_PTR) = nanbox_tag(cur) {
                    let p = nanbox_payload(cur) as *const u8;
                    if p.is_null() { break; }
                    let tid = unsafe { p.cast::<u16>().read_unaligned() } as usize;
                    if tid != ids.cons { break; }
                    out.push(unsafe { p.add(8).cast::<u64>().read_unaligned() });
                    cur = unsafe { p.add(16).cast::<u64>().read_unaligned() };
                }
            } else if tid == ids.vector {
                let n = unsafe { p.add(8).cast::<u64>().read_unaligned() } as usize;
                for i in 0..n {
                    out.push(unsafe { p.add(16 + i * 8).cast::<u64>().read_unaligned() });
                }
            } else {
                // Coerce via cljvm_rt_seq, then walk the result.
                let s = unsafe { cljvm_rt_seq(seq_bits) };
                if !matches!(nanbox_tag(s), Some(TAG_NIL)) && s != seq_bits {
                    walk_seq_into(s, ids, out);
                } else {
                    out.push(seq_bits);
                }
            }
        }
        _ => out.push(seq_bits),
    }
}

// ─── clojure.lang.LazySeq / Delay ─────────────────────────────────────
//
// Both wrap a thunk (TAG_FN handle) that's called at most once to produce
// a value. Backing: Arc<RefCell<LazyState>> where state is the original
// thunk + cached realized value. Single-threaded so RefCell is fine.

#[derive(Debug)]
pub struct LazyState {
    pub thunk_bits: u64,        // TAG_FN handle of the deferred fn
    pub realized: bool,
    pub value_bits: u64,        // valid only when realized
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_LazySeq_new1(thunk_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let arc = std::sync::Arc::new(std::cell::RefCell::new(LazyState {
        thunk_bits,
        realized: false,
        value_bits: nanbox_nil(),
    }));
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
                    if st.realized { Some(st.value_bits) } else { None }
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
    let arc = std::sync::Arc::new(std::cell::RefCell::new(
        Vec::<u64>::with_capacity(cap as usize),
    ));
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
    let recv: std::sync::Arc<std::cell::RefCell<Vec<u64>>> =
        unsafe { decode_arc_cell(p) };
    recv.borrow_mut().push(x_bits);
    nanbox_nil()
}

/// `(.reduce chunk f init)` — Java's IChunk.reduce. Folds f over items
/// starting with init. Used by `reduce1`'s chunked-seq fast path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_reduce_3(
    recv_bits: u64, f_bits: u64, init_bits: u64,
) -> u64 {
    let p = nanbox_payload(recv_bits) as *const u8;
    let arc: std::sync::Arc<std::cell::RefCell<Vec<u64>>> =
        unsafe { decode_arc_cell(p) };
    let items: Vec<u64> = arc.borrow().clone();
    let mut acc = init_bits;
    for item in items {
        acc = unsafe { cljvm_rt_invoke_2(f_bits, acc, item) };
    }
    acc
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
    let recv: std::sync::Arc<std::cell::RefCell<Vec<u64>>> =
        unsafe { decode_arc_cell(p) };
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
pub unsafe extern "C" fn cljvm_inst_StringBuilder_append(
    recv_bits: u64,
    x_bits: u64,
) -> u64 {
    let ids = heap_type_ids();
    let p = nanbox_payload(recv_bits) as *const u8;
    let recv: std::sync::Arc<std::cell::RefCell<String>> =
        unsafe { decode_arc_cell(p) };
    let x_obj = any_bits_to_object(x_bits, ids);
    recv.borrow_mut().push_str(&format_object_for_str(&x_obj));
    recv_bits
}

/// `(.toString sb)` — return the accumulated buffer as a clojure.lang.String.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_inst_StringBuilder_toString(recv_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let p = nanbox_payload(recv_bits) as *const u8;
    let recv: std::sync::Arc<std::cell::RefCell<String>> =
        unsafe { decode_arc_cell(p) };
    let snapshot = recv.borrow().clone();
    unsafe { alloc_string_heap(&snapshot, ids) }
}

unsafe fn read_string_heap<'a>(bits: u64, ids: HeapTypeIds, ctx: &str) -> &'a str {
    let raw = match nanbox_tag(bits) {
        Some(TAG_PTR) => nanbox_payload(bits) as *const u8,
        _ => {
            eprintln!(
                "[cljvm-stub] {ctx} — non-heap NanBox 0x{bits:x}, treating as empty string"
            );
            return "";
        }
    };
    if raw.is_null() {
        eprintln!("[cljvm-stub] {ctx} — null receiver, treating as empty string");
        return "";
    }
    let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
    if type_id != ids.string {
        eprintln!(
            "[cljvm-stub] {ctx} — type_id {type_id} not a String, treating as empty"
        );
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
    dynobj::roots::with_scope(1, |scope| {
        let x = scope.root::<()>(x_bits);
        let raw = dynlang::gc::gc_alloc_thunk(ids.reduced as u64, 0);
        let ptr = raw as *mut u8;
        if ptr.is_null() {
            panic!("clojure-jvm: Reduced ctor: gc_alloc returned null");
        }
        unsafe {
            ptr.add(8).cast::<u64>().write_unaligned(x.get());
        }
        nanbox_encode(TAG_PTR, raw & PAYLOAD_MASK)
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
    dynobj::roots::with_scope(2, |scope| {
        let recv = scope.root::<()>(recv_bits);
        let meta = scope.root::<()>(meta_bits);

        // Read receiver type once. `gc_alloc` may move things, so
        // read header / first / rest fresh after each alloc by going
        // back through `recv.get()`.
        let raw = nanbox_payload(recv.get()) as *const u8;
        if raw.is_null() {
            panic!("clojure-jvm: (.withMeta nil m) — null receiver");
        }
        let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
        if type_id == ids.cons {
            let new_raw = dynlang::gc::gc_alloc_thunk(ids.cons as u64, 0);
            let new_ptr = new_raw as *mut u8;
            if new_ptr.is_null() {
                panic!("clojure-jvm: (.withMeta cons): gc_alloc returned null");
            }
            // Re-read the receiver pointer AFTER alloc; root may have
            // been forwarded.
            let raw = nanbox_payload(recv.get()) as *const u8;
            let first_bits = unsafe { raw.add(8).cast::<u64>().read_unaligned() };
            let rest_bits = unsafe { raw.add(16).cast::<u64>().read_unaligned() };
            unsafe {
                new_ptr.add(8).cast::<u64>().write_unaligned(first_bits);
                new_ptr.add(16).cast::<u64>().write_unaligned(rest_bits);
                new_ptr.add(24).cast::<u64>().write_unaligned(meta.get());
            }
            return nanbox_ptr(new_raw);
        }
        // Generic path: wrap in `clojure.lang.WithMeta`. Layout:
        // header(8) + inner(8) + meta(8). Collapse nested wrappers so
        // the inner is always a "bare" value.
        let new_raw = dynlang::gc::gc_alloc_thunk(ids.with_meta as u64, 0);
        let new_ptr = new_raw as *mut u8;
        if new_ptr.is_null() {
            panic!("clojure-jvm: (.withMeta x m): gc_alloc returned null for WithMeta");
        }
        // Re-read AFTER alloc.
        let raw = nanbox_payload(recv.get()) as *const u8;
        let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
        let inner_bits = if type_id == ids.with_meta {
            unsafe { raw.add(8).cast::<u64>().read_unaligned() }
        } else {
            recv.get()
        };
        unsafe {
            new_ptr.add(8).cast::<u64>().write_unaligned(inner_bits);
            new_ptr.add(16).cast::<u64>().write_unaligned(meta.get());
        }
        nanbox_ptr(new_raw)
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
                let user_tid =
                    unsafe { raw.offset(off).cast::<u64>().read_unaligned() }
                        as u32;
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
                panic!(
                    "clojure-jvm: user_instance_field_get: null TAG_PTR payload"
                );
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
    let ids = heap_type_ids();
    let layout = user_instance_layout();
    let raw = dynlang::gc::gc_alloc_thunk(ids.user_instance as u64, fields.len() as u64);
    let ptr = raw as *mut u8;
    if ptr.is_null() {
        panic!("clojure-jvm: alloc_user_instance: gc_alloc returned null");
    }
    let off = layout.user_type_id_offset as isize;
    unsafe {
        ptr.offset(off).cast::<u64>().write_unaligned(user_type_id as u64);
        for (i, bits) in fields.iter().enumerate() {
            let slot_off = layout.varlen_base + (i as i64) * 8;
            ptr.offset(slot_off as isize)
                .cast::<u64>()
                .write_unaligned(*bits);
        }
    }
    nanbox_encode(TAG_PTR, raw & PAYLOAD_MASK)
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
    use crate::lang::user_types::{lookup_impl, BUILTIN_OBJECT};
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
pub unsafe extern "C" fn cljvm_rt_protocol_dispatch_1(
    method_id_bits: u64,
    this: u64,
) -> u64 {
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
pub unsafe extern "C" fn cljvm_rt_satisfies(
    proto_id_bits: u64,
    x_bits: u64,
) -> u64 {
    let pid = arg_to_i64(proto_id_bits);
    if pid < 0 || pid > u32::MAX as i64 {
        panic!(
            "clojure-jvm: cljvm_rt_satisfies: implausible proto_id {pid}"
        );
    }
    let info = crate::lang::user_types::protocol_info(pid as u32)
        .unwrap_or_else(|| {
            panic!(
                "clojure-jvm: cljvm_rt_satisfies: protocol id {pid} not registered"
            )
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
pub unsafe extern "C" fn cljvm_rt_alloc_user_instance_1(
    type_id_bits: u64,
    a: u64,
) -> u64 {
    let tid = decode_user_type_id_bits(type_id_bits);
    unsafe { alloc_user_instance(tid, &[a]) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_alloc_user_instance_2(
    type_id_bits: u64,
    a: u64,
    b: u64,
) -> u64 {
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
                panic!(
                    "clojure-jvm: user_field_get_by_name: null receiver"
                );
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
    let idx = crate::lang::user_types::user_type_field_index(
        user_tid,
        &field_sym,
    )
    .unwrap_or_else(|| {
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
        panic!(
            "clojure-jvm: cljvm_rt_install_impl: implausible type_id {tid}"
        );
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
                panic!(
                    "clojure-jvm: in-ns: arg must be a Symbol, got type_id {tid}"
                );
            }
        }
        _ => panic!("clojure-jvm: in-ns: arg must be a Symbol"),
    };
    let ns = crate::lang::namespace::Namespace::find_or_create(sym_arc);
    crate::lang::rt::CURRENT_NS
        .bind_root(crate::lang::object::Object::Namespace(ns));
    nanbox_nil()
}

#[cfg(test)]
mod user_type_runtime_tests {
    use super::*;
    use crate::lang::user_types::{
        self as ut, install_impl, register_protocol, register_user_type,
        user_type_logical, BUILTIN_DOUBLE, BUILTIN_NIL, BUILTIN_OBJECT,
        USER_TYPE_BASE,
    };
    use crate::lang::symbol::Symbol;
    use std::sync::Arc;

    fn sym(n: &str) -> Arc<Symbol> {
        Symbol::intern_ns_name(None, n)
    }

    // These tests share the user_types global registries with each
    // other; serialize via a Mutex so allocator state doesn't
    // interleave.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn guard() -> std::sync::MutexGuard<'static, ()> {
        let g = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        ut::_reset_for_tests();
        g
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
        let (_pid, mids) = register_protocol(
            sym("ICountable"),
            vec![(sym("-count"), vec![1])],
        );
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
        let (_pid, mids) = register_protocol(
            sym("IUnknown"),
            vec![(sym("-mystery"), vec![1])],
        );
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
