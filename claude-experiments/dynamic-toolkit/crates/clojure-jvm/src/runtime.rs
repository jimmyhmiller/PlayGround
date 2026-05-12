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

use std::sync::Arc;

use crate::lang::object::Object;
use crate::lang::var::Var;

// ── NanBox layout (matches dynlang's default NanBoxTags) ────────────────

const FULL_MASK: u64 = 0xFFFC_0000_0000_0000;
const TAG_PATTERN: u64 = 0x7FFC_0000_0000_0000;
const TAG_MASK: u64 = 0x0003_0000_0000_0000;
const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

// Tags (matches `dynlang::NanBoxTags::default()` + our own fn-handle tag).
const TAG_NIL: u32 = 0;
const TAG_BOOL: u32 = 1;
const TAG_PTR: u32 = 2;
const TAG_FN: u32 = 3;

fn nanbox_encode(tag: u32, payload: u64) -> u64 {
    TAG_PATTERN | ((tag as u64) << 48) | (payload & PAYLOAD_MASK)
}

fn nanbox_tag(bits: u64) -> Option<u32> {
    if (bits & FULL_MASK) != TAG_PATTERN {
        return None;
    }
    Some(((bits & TAG_MASK) >> 48) as u32)
}

fn nanbox_payload(bits: u64) -> u64 { bits & PAYLOAD_MASK }

/// Public NanBox constructors callable from heap-population code that
/// stores Object fields (`alloc_object_as_nanbox` in compiler.rs).
pub fn nanbox_nil() -> u64 { nanbox_encode(TAG_NIL, 0) }
pub fn nanbox_bool(b: bool) -> u64 { nanbox_encode(TAG_BOOL, b as u64) }

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
        Object::Long(n) => (*n as f64).to_bits(),
        Object::Double(x) => x.to_bits(),
        Object::Host(_) => {
            if let Some(hb) = obj.host_as::<HeapBits>() {
                return hb.0;
            }
            panic!(
                "clojure-jvm: object_to_nanbox: Object::Host wrapping non-HeapBits value not yet representable as NanBox"
            )
        }
        _ => panic!(
            "clojure-jvm: object_to_nanbox: variant {obj:?} not yet representable as NanBox"
        ),
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
        // `clojure.lang.Symbol`: Header(8) + Raw64 "arc_ptr"(8). Reconstruct
        // the Arc<Symbol> by `Arc::increment_strong_count` on the stored
        // pointer. The Arc is rooted by `CompileRoots` for the JIT module's
        // lifetime, so the pointer is valid as long as the compile output
        // outlives this call.
        let arc_ptr = unsafe { ptr.add(8).cast::<u64>().read_unaligned() } as *const crate::lang::symbol::Symbol;
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
    if type_id == ids.cons {
        // `clojure.lang.Cons`: Header(8) + value-field "first"(8) +
        // value-field "rest"(8). Recursively decode `first` (any Object)
        // and `rest` (Object::Nil terminator or another Cons). Reconstruct
        // a `PersistentList` so the caller gets a familiar Rust shape.
        let first_bits = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
        let rest_bits = unsafe { ptr.add(16).cast::<u64>().read_unaligned() };
        let first_obj = decode_value_bits(first_bits, ids);
        let rest_list = decode_rest_to_list(rest_bits, ids);
        return Object::List(std::sync::Arc::new(crate::lang::persistent_list::PersistentList::Cons {
            first: first_obj,
            rest: rest_list,
            count: 1 + count_list(&rest_bits, ids),
        }));
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
    Object::Unported {
        java_class: "heap object of unrecognized type_id",
    }
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
                panic!("clojure-jvm: cljvm_rt_invoke_*: receiver is nil");
            }
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id == ids.string
                || type_id == ids.symbol
                || type_id == ids.keyword
                || type_id == ids.cons
            {
                panic!(
                    "clojure-jvm: cljvm_rt_invoke_*: receiver is a non-callable \
                     heap value (type_id {type_id})"
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
        _ => panic!(
            "clojure-jvm: cljvm_rt_invoke_*: receiver is not a fn (bits 0x{handle_bits:x})"
        ),
    }
}

/// `IFn.invoke()` — 0 arity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_0(fn_bits: u64) -> u64 {
    let (ptr, self_arg) = unsafe { dispatch_target(fn_bits) };
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

/// `IFn.invoke(arg1)` — 1 arity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_1(fn_bits: u64, a: u64) -> u64 {
    let (ptr, self_arg) = unsafe { dispatch_target(fn_bits) };
    match self_arg {
        None => {
            let f: unsafe extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(a) }
        }
        Some(s) => {
            let f: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(s, a) }
        }
    }
}

/// `IFn.invoke(arg1, arg2)` — 2 arity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_2(fn_bits: u64, a: u64, b: u64) -> u64 {
    let (ptr, self_arg) = unsafe { dispatch_target(fn_bits) };
    match self_arg {
        None => {
            let f: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(a, b) }
        }
        Some(s) => {
            let f: unsafe extern "C" fn(u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
            unsafe { f(s, a, b) }
        }
    }
}

/// `IFn.invoke(arg1, arg2, arg3)` — 3 arity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_3(fn_bits: u64, a: u64, b: u64, c: u64) -> u64 {
    let (ptr, self_arg) = unsafe { dispatch_target(fn_bits) };
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

/// `clojure.lang.RT.inc(Object x)` — for now, a primitive Long-only
/// increment. Real RT.inc dispatches on Number subtypes; we'll widen as
/// the Numbers port lands. This exists as the bring-up vehicle for the
/// `(. clojure.lang.RT (...))` codegen path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_inc(x_bits: u64) -> u64 {
    // NanBox-encoded Long round-trips as f64 bits (per our object_to_nanbox).
    let x = f64::from_bits(x_bits);
    ((x as i64 + 1) as f64).to_bits()
}

/// `clojure.lang.RT.cons(Object x, Object seq)` — allocate a new Cons cell
/// linking `x` and `seq`. Java's RT.cons handles nil/Empty specially; we
/// match that behavior. Both args + return are NanBox-encoded.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_cons(x_bits: u64, seq_bits: u64) -> u64 {
    let ids = heap_type_ids();
    // Allocate raw pointer via dynlang's gc_alloc_thunk (reads the
    // installed thread-local DynGcRuntime).
    let raw = dynlang::gc::gc_alloc_thunk(ids.cons as u64, 0);
    let ptr = raw as *mut u8;
    if ptr.is_null() {
        panic!("clojure-jvm: cljvm_rt_cons: gc_alloc returned null");
    }
    // Cons layout: Compact header(8) + first(8) + rest(8).
    unsafe {
        ptr.add(8).cast::<u64>().write_unaligned(x_bits);
        ptr.add(16).cast::<u64>().write_unaligned(seq_bits);
    }
    nanbox_encode(TAG_PTR, raw & PAYLOAD_MASK)
}

/// `clojure.lang.RT.first(Object coll)` — return the first item or nil.
/// Java: handles ISeq, Seqable, nil. We support nil + Cons for now.
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
            } else {
                panic!("clojure-jvm: RT.first on unsupported heap type_id {type_id}");
            }
        }
        _ => panic!("clojure-jvm: RT.first on non-seqable NanBox tag"),
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
    false
}

/// `clojure.lang.Util.nil?(x)` — NanBox tag check.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_is_nil(x_bits: u64) -> u64 {
    nanbox_bool(nanbox_tag(x_bits) == Some(TAG_NIL))
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
            } else {
                panic!("clojure-jvm: RT.seq on unsupported heap type_id {type_id} (only Cons supported yet)");
            }
        }
        _ => panic!("clojure-jvm: RT.seq on non-seqable NanBox tag"),
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
                // Java's `next` returns nil for one-element seqs (the `rest`
                // would be Empty). For our Cons-only model the `rest` field
                // is already either nil or another Cons — passing it through
                // matches Java's `more` semantics; for `next` we'd need to
                // distinguish empty. With Cons-only this is the same answer.
                unsafe { ptr.add(16).cast::<u64>().read_unaligned() }
            } else {
                panic!("clojure-jvm: RT.next on unsupported heap type_id {type_id}");
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
    let obj = nanbox_to_object(val_bits);
    v.bind_root(obj);
    val_bits
}

/// JIT extern: return the current value of the Var at `var_ptr`, NanBox-encoded.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_var_deref(var_ptr: u64) -> u64 {
    let v: &Var = unsafe { &*(var_ptr as *const Var) };
    let obj = v.deref();
    object_to_nanbox(&obj)
}

/// Resolve an `Arc<Var>` to a `u64` suitable for baking into IR. Holds onto
/// the Arc through the namespace mapping (the caller's responsibility), so
/// the pointer remains valid for the program's lifetime.
pub fn var_to_jit_ptr(v: &Arc<Var>) -> u64 {
    Arc::as_ptr(v) as u64
}
