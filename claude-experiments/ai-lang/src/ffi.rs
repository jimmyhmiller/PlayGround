//! FFI extern registry.
//!
//! Host runtimes register Rust functions here under a string name.
//! ai-lang `extern fn` declarations with that name get linked into
//! the JIT at module-build time via `add_global_mapping`.
//!
//! ## Layer 1 only (today)
//!
//! Only Int parameters and Int return values are supported. Each
//! extern function pointer is treated as
//! `unsafe extern "C" fn(*mut Thread, i64, i64, ...) -> i64`.
//! The first argument is always the JIT-visible `Thread*` (the
//! same convention as ai-lang's own runtime fns); the remaining
//! parameters are i64s carrying the user's `Int` args. The return
//! is i64 carrying the `Int` result.
//!
//! Layer 2 (heap-resident String / Bytes pass-through) and Layer 3
//! (opaque Rust handles) will extend this; for now keep it simple.
//!
//! ## Two-tier, race-free by design
//!
//! Extern resolution happens only at build time: the codegen reads this
//! registry to `add_global_mapping` each `ext/<name>` to a fn pointer,
//! and once mapped the JIT'd code calls the address directly and never
//! consults the registry again. So the registry only needs to be correct
//! at the moment of a build, on the thread doing that build. Two kinds of
//! extern live here, in two tiers:
//!
//! - **The fixed runtime I/O externs** (`print_int`, `print_string`, ...)
//!   have static fn pointers, are identical for every `Runtime`, and must
//!   be visible from every thread that builds a module. They live in a
//!   process-global, **write-once** table (`install_io_externs`): set
//!   exactly once, never mutated or cleared, so nothing can ever race on
//!   them.
//! - **Dynamic externs** — C symbols resolved during a build (the codegen
//!   registers them moments before JIT-init, on the build thread) and
//!   test-registered overrides — live in a **thread-local** registry.
//!   Because build + JIT-init happen on the same thread, this is always
//!   correct, and parallel tests (or concurrent builds on different
//!   threads) can never stomp on each other's externs.
//!
//! Resolution checks the thread-local tier first, so a build's C externs
//! and a test's override win over the global I/O defaults.
//!
//! ## Safety
//!
//! The registered function pointer must match the declared arity.
//! Mismatch is a soundness hole — the JIT will happily call with
//! the wrong number of args and corrupt the stack.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::ast::Type;
use crate::gc::Full;
use crate::runtime::{Thread, ai_str_new};

#[derive(Clone, Debug)]
pub struct ExternEntry {
    /// Declared parameter types as seen by ai-lang. Used by the
    /// codegen to construct the LLVM function type at JIT-init time
    /// and to validate that the registered pointer matches what the
    /// module declared.
    pub params: Vec<Type>,
    pub ret: Type,
    /// Raw function pointer, treated as
    /// `unsafe extern "C" fn(*mut Thread, i64, ...) -> i64` for
    /// Layer 1. Stored as `usize` so the entry is `Send + Sync`.
    pub fn_ptr: usize,
}

/// Process-global, write-once table of the fixed runtime I/O externs.
/// Visible from every thread; never mutated or cleared after init.
static IO_REGISTRY: OnceLock<HashMap<String, ExternEntry>> = OnceLock::new();

thread_local! {
    /// Per-thread registry for dynamic externs (build-time C symbols and
    /// test overrides). Build + JIT-init run on the same thread, so this
    /// is always correct, and parallel work never collides.
    static DYNAMIC: RefCell<HashMap<String, ExternEntry>> = RefCell::new(HashMap::new());
}

/// Install the fixed runtime I/O externs exactly once. Idempotent and
/// race-free: the first caller wins, every later call is a no-op, and the
/// table is never overwritten or cleared. (The I/O externs never change,
/// so there is nothing to update.)
pub fn install_io_externs(entries: Vec<(String, ExternEntry)>) {
    IO_REGISTRY.get_or_init(move || entries.into_iter().collect());
}

/// Register a dynamic extern under `name` in the calling thread's
/// registry, replacing any prior thread-local entry. Used by the C-FFI
/// build path (a resolved C symbol, registered just before JIT-init on
/// the build thread) and by tests that override an extern.
///
/// # Safety
/// `fn_ptr` must be a valid `extern "C"` function pointer whose
/// signature matches `params` / `ret` under the Layer-1 ABI:
/// `unsafe extern "C" fn(*mut Thread, i64 × params.len()) -> i64`.
/// (Layer 1 only handles Int params / Int return.)
pub unsafe fn register_extern(name: &str, params: Vec<Type>, ret: Type, fn_ptr: usize) {
    DYNAMIC.with(|r| {
        r.borrow_mut()
            .insert(name.to_owned(), ExternEntry { params, ret, fn_ptr });
    });
}

/// Look up an extern: thread-local dynamic registry first (so build-time
/// C externs and test overrides win), then the global I/O table.
pub fn lookup_extern(name: &str) -> Option<ExternEntry> {
    if let Some(e) = DYNAMIC.with(|r| r.borrow().get(name).cloned()) {
        return Some(e);
    }
    IO_REGISTRY.get().and_then(|m| m.get(name).cloned())
}

/// Remove the calling thread's dynamic externs. Does NOT touch the
/// global I/O table or any other thread's registry. Useful for test
/// isolation.
pub fn clear_externs() {
    DYNAMIC.with(|r| r.borrow_mut().clear());
}

/// Snapshot of all currently-resolvable extern names (dynamic + I/O), for
/// diagnostic logging.
pub fn registered_extern_names() -> Vec<String> {
    let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    DYNAMIC.with(|r| {
        for k in r.borrow().keys() {
            set.insert(k.clone());
        }
    });
    if let Some(m) = IO_REGISTRY.get() {
        for k in m.keys() {
            set.insert(k.clone());
        }
    }
    set.into_iter().collect()
}

// =============================================================================
// Layer 2 marshaling: ai-lang heap String ↔ Rust String
// =============================================================================
//
// String marshaling is the smallest interesting Layer-2 case. The
// pattern: copy in / copy out — never hold an ai-lang heap pointer
// across a GC point.

/// Borrow the bytes of a heap-resident ai-lang `String` as a `&[u8]`.
///
/// # Safety
/// `s` must be a live ai-lang `String` heap pointer. The returned
/// slice MUST NOT be held across any operation that can trigger GC
/// (calling another ai_gc_* fn, invoking JIT'd code, etc.). For
/// anything beyond a single read, copy to an owned `Vec<u8>` /
/// `String` first via [`heap_str_to_owned`].
pub unsafe fn heap_str_bytes<'a>(s: *const u8) -> &'a [u8] {
    if s.is_null() {
        return &[];
    }
    unsafe {
        let count_off = <Full as crate::gc::ObjHeader>::SIZE;
        let len = *(s.add(count_off) as *const u64) as usize;
        let data = s.add(count_off + 8);
        std::slice::from_raw_parts(data, len)
    }
}

/// Copy a heap-resident ai-lang `String` into a Rust-owned `String`.
/// Safe to hold across GC: the bytes are duplicated into Rust-managed
/// memory.
///
/// Replaces invalid UTF-8 with the U+FFFD replacement character so
/// downstream Rust code can use it as a regular `String`.
///
/// # Safety
/// `s` must be a live ai-lang `String` heap pointer.
pub unsafe fn heap_str_to_owned(s: *const u8) -> String {
    let bytes = unsafe { heap_str_bytes(s) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// Allocate a new ai-lang `String` containing the given Rust string.
/// The allocation goes through the GC (may collect under pressure);
/// no live ai-lang pointers should be held in Rust-local variables
/// across this call.
///
/// # Safety
/// `thread` must be the JIT-visible Thread pointer with `string_ti`
/// initialised (every Runtime constructed via `Runtime::new_with_*`
/// has this).
pub unsafe fn owned_str_to_heap(thread: *mut Thread, s: &str) -> *mut u8 {
    unsafe { ai_str_new(thread, s.as_ptr(), s.len() as i64) }
}

/// Allocate a new ai-lang `String` (a heap byte buffer) holding the raw
/// bytes in `b`. Unlike [`owned_str_to_heap`] this does NOT require valid
/// UTF-8 — it copies bytes verbatim, so it's the right tool for binary
/// payloads (e.g. an intermediate HMAC digest used as the key for the
/// next HMAC in a SigV4 chain). Reading it back with [`heap_str_bytes`]
/// returns the same bytes; reading it with [`heap_str_to_owned`] would
/// lossily UTF-8-convert, so only use byte-level ops on the result.
///
/// # Safety
/// `thread` must be the JIT-visible Thread pointer with `string_ti`
/// initialised. No live ai-lang pointers may be held in Rust locals
/// across this call (it may GC).
pub unsafe fn owned_bytes_to_heap(thread: *mut Thread, b: &[u8]) -> *mut u8 {
    unsafe { ai_str_new(thread, b.as_ptr(), b.len() as i64) }
}
