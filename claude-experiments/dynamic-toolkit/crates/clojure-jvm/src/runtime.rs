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
const TAG_NIL: u32 = 0;
const TAG_BOOL: u32 = 1;
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
    pub tree_map: usize,
    pub tree_set: usize,
    pub class: usize,
    pub var: usize,
    pub with_meta: usize,
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
            let fref_idx = unsafe { raw.add(16).cast::<u64>().read_unaligned() } as u32;
            let slot_addr = call_table_base() + (fref_idx as u64) * 8;
            let ptr = unsafe { *(slot_addr as *const *const u8) };
            (ptr, Some(handle_bits), fref_idx)
        }
        _ => panic!(
            "clojure-jvm: cljvm_rt_invoke_*: receiver is not a fn (bits 0x{handle_bits:x})"
        ),
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
    if args.len() <= info.fixed_arity {
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
/// (`pack_variadic_args` returned `Some`). Supports up to 8 args
/// (matches the maximum we'd plausibly need before more invoke_N
/// arities land).
unsafe fn call_with_packed(ptr: *const u8, packed: &[u64]) -> u64 {
    match packed.len() {
        1 => {
            let f: unsafe extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(packed[0]) }
        }
        2 => {
            let f: unsafe extern "C" fn(u64, u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(packed[0], packed[1]) }
        }
        3 => {
            let f: unsafe extern "C" fn(u64, u64, u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(packed[0], packed[1], packed[2]) }
        }
        4 => {
            let f: unsafe extern "C" fn(u64, u64, u64, u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(packed[0], packed[1], packed[2], packed[3]) }
        }
        5 => {
            let f: unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64 = unsafe { std::mem::transmute(ptr) };
            unsafe { f(packed[0], packed[1], packed[2], packed[3], packed[4]) }
        }
        n => panic!("clojure-jvm: call_with_packed: unsupported arity {n}"),
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

/// `IFn.invoke(arg1)` — 1 arity. Also handles Clojure's keyword-as-fn
/// shortcut: `(:k m)` is `(get m :k nil)`. We dispatch that before
/// going through the regular fn-handle path so a Keyword receiver
/// doesn't trip the "non-callable heap value" panic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_invoke_1(fn_bits: u64, a: u64) -> u64 {
    if let Some(v) = unsafe { keyword_as_fn_lookup(fn_bits, a, nanbox_nil()) } {
        return v;
    }
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_target_with_idx(fn_bits) };
    if let Some(packed) = unsafe { pack_variadic_args(self_arg, fref_idx, &[a]) } {
        return unsafe { call_with_packed(ptr, &packed) };
    }
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
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_target_with_idx(fn_bits) };
    if let Some(packed) = unsafe { pack_variadic_args(self_arg, fref_idx, &[a, b]) } {
        return unsafe { call_with_packed(ptr, &packed) };
    }
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
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_target_with_idx(fn_bits) };
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
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_target_with_idx(fn_bits) };
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
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_target_with_idx(fn_bits) };
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
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_target_with_idx(fn_bits) };
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
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_target_with_idx(fn_bits) };
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
    let (ptr, self_arg, fref_idx) = unsafe { dispatch_target_with_idx(fn_bits) };
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
            let i: unsafe extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64 =
                unsafe { std::mem::transmute(ptr) };
            unsafe { i(s, a, b, c, d, e, f, g, h) }
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
/// linking `x` and `seq`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_cons(x_bits: u64, seq_bits: u64) -> u64 {
    let ids = heap_type_ids();
    dynobj::roots::with_scope(2, |scope| {
        let x = scope.root::<()>(x_bits);
        let seq = scope.root::<()>(seq_bits);
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
                let n = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
                let new_raw = dynlang::gc::gc_alloc_thunk(ids.vector as u64, (n + 1) as u64);
                let new_ptr = new_raw as *mut u8;
                if new_ptr.is_null() {
                    panic!("clojure-jvm: RT.conj: gc_alloc returned null for Vector");
                }
                for i in 0..n {
                    let src_off = 16 + i * 8;
                    let bits = unsafe { raw.add(src_off).cast::<u64>().read_unaligned() };
                    unsafe {
                        new_ptr.add(16 + i * 8).cast::<u64>().write_unaligned(bits);
                    }
                }
                unsafe {
                    new_ptr.add(16 + n * 8).cast::<u64>().write_unaligned(x_bits);
                }
                return nanbox_ptr(new_raw);
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
            panic!(
                "clojure-jvm: RT.conj — receiver type_id {type_id} not yet \
                 implemented (extend cljvm_rt_conj to dispatch this type)"
            );
        }
        _ => panic!(
            "clojure-jvm: RT.conj — first arg must be a collection or nil, \
             got NanBox bits 0x{coll_bits:x}"
        ),
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
                if type_id == ids.cons {
                    // Walk rest-chain. Each rest is either another Cons or nil.
                    let mut count: i64 = 0;
                    let mut cur = bits;
                    loop {
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
                                        "clojure-jvm: RT.count: cons rest pointed at \
                                         non-cons heap type_id {tid}"
                                    );
                                }
                                count += 1;
                                cur = unsafe { p.add(16).cast::<u64>().read_unaligned() };
                            }
                            _ => panic!(
                                "clojure-jvm: RT.count: cons rest is non-pointer NanBox \
                                 (bits 0x{cur:x})"
                            ),
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
                } else if type_id == ids.string {
                    let c = unsafe { ptr.add(8).cast::<u64>().read_unaligned() };
                    c as i64
                } else {
                    panic!(
                        "clojure-jvm: RT.count: heap type_id {type_id} not yet supported"
                    );
                }
            }
        }
        _ => panic!("clojure-jvm: RT.count: receiver bits 0x{bits:x} is not countable"),
    };
    (n as f64).to_bits()
}

/// `clojure.lang.RT.subvec(Object v, int start, int end)` — return a
/// sub-vector covering `[start, end)`. Allocates a fresh Vector heap
/// cell whose items are copied (NanBox-bit-for-bit) from the source
/// vector — heap pointers in the source remain valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cljvm_rt_subvec(v_bits: u64, start_bits: u64, end_bits: u64) -> u64 {
    let ids = heap_type_ids();
    let start = f64::from_bits(start_bits) as usize;
    let end = f64::from_bits(end_bits) as usize;
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
                    panic!(
                        "clojure-jvm: LazilyPersistentVector/create — \
                         receiver type_id {type_id} not yet supported"
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
                let n = unsafe { ptr.add(8).cast::<u64>().read_unaligned() } as usize;
                if n == 0 {
                    return nanbox_nil();
                }
                let mut cur = nanbox_nil();
                for i in (0..n).rev() {
                    let item = unsafe { ptr.add(16 + i * 8).cast::<u64>().read_unaligned() };
                    cur = unsafe { cljvm_rt_cons(item, cur) };
                }
                cur
            } else if type_id == ids.string {
                // String → seq of Character. Bootstrap doesn't need this
                // path yet; surface loudly so callers see what to add.
                panic!(
                    "clojure-jvm: RT.seq on String — needs Character heap type"
                );
            } else {
                panic!(
                    "clojure-jvm: RT.seq on unsupported heap type_id {type_id} \
                     (Cons / Vector / nil supported)"
                );
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
                unsafe { ptr.add(16).cast::<u64>().read_unaligned() }
            } else if type_id == ids.vector {
                // Vector: subvec [1..count) as a fresh Vector heap cell.
                // Returns nil for empty/single-element vectors.
                let n = unsafe { ptr.add(8).cast::<u64>().read_unaligned() } as usize;
                if n <= 1 {
                    return nanbox_nil();
                }
                let new_n = n - 1;
                let new_raw = dynlang::gc::gc_alloc_thunk(ids.vector as u64, new_n as u64);
                let new_ptr = new_raw as *mut u8;
                if new_ptr.is_null() {
                    panic!("clojure-jvm: RT.next: gc_alloc returned null for Vector tail");
                }
                for i in 0..new_n {
                    let src_off = 16 + (i + 1) * 8;
                    let bits = unsafe { ptr.add(src_off).cast::<u64>().read_unaligned() };
                    unsafe {
                        new_ptr.add(16 + i * 8).cast::<u64>().write_unaligned(bits);
                    }
                }
                nanbox_ptr(new_raw)
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
                None => panic!(
                    "clojure-jvm: (new {} ...) — class has no registered constructor",
                    info.name
                ),
            }
        }
        _ => panic!(
            "clojure-jvm: (new ...) — receiver bits 0x{class_bits:x} \
             is not a Class heap pointer"
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
                panic!("clojure-jvm: (.isInstance nil x) — null Class receiver");
            }
            let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
            if type_id != ids.class {
                panic!(
                    "clojure-jvm: (.isInstance c x) — receiver is not a Class \
                     (heap type_id {type_id})"
                );
            }
            // Class layout: Header(8) + Raw64 "class_id"(8). Read u16 from
            // the low end of the Raw64 slot.
            let raw_id = unsafe { raw.add(8).cast::<u64>().read_unaligned() };
            let class_id = crate::lang::host_class::ClassId(raw_id as u16);
            let yes = crate::lang::host_class::is_instance(class_id, x_bits, ids);
            nanbox_bool(yes)
        }
        _ => panic!(
            "clojure-jvm: (.isInstance c x) — receiver bits 0x{c_bits:x} \
             is not a Class heap pointer"
        ),
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
    (n as f64).to_bits()
}

unsafe fn read_string_heap<'a>(bits: u64, ids: HeapTypeIds, ctx: &str) -> &'a str {
    let raw = match nanbox_tag(bits) {
        Some(TAG_PTR) => nanbox_payload(bits) as *const u8,
        _ => panic!("clojure-jvm: {ctx} — not a heap String (NanBox 0x{bits:x})"),
    };
    if raw.is_null() {
        panic!("clojure-jvm: {ctx} — null receiver");
    }
    let type_id = unsafe { raw.cast::<u16>().read_unaligned() } as usize;
    if type_id != ids.string {
        panic!("clojure-jvm: {ctx} — heap type_id {type_id} is not a String");
    }
    let count = unsafe { raw.add(8).cast::<u64>().read_unaligned() } as usize;
    let bytes = unsafe { std::slice::from_raw_parts(raw.add(16), count) };
    std::str::from_utf8(bytes).expect("String heap bytes must be UTF-8")
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
