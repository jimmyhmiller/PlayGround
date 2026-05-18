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
//! ## Process-global by design
//!
//! The registry is a `Mutex<HashMap<String, ExternEntry>>` shared
//! across all `Runtime` instances in the same process. Tests can
//! register / clear via `register_extern` / `clear_externs`. This
//! matches how the existing `core/net.at`, `core/gc.collect` etc.
//! work — they're process-wide function pointers that get mapped
//! by name during JIT init.
//!
//! ## Safety
//!
//! The registered function pointer must match the declared arity.
//! Mismatch is a soundness hole — the JIT will happily call with
//! the wrong number of args and corrupt the stack.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

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

static REGISTRY: OnceLock<Mutex<HashMap<String, ExternEntry>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<String, ExternEntry>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register an extern under `name`. Replaces any prior entry.
///
/// # Safety
/// `fn_ptr` must be a valid `extern "C"` function pointer whose
/// signature matches `params` / `ret` under the Layer-1 ABI:
/// `unsafe extern "C" fn(*mut Thread, i64 × params.len()) -> i64`.
/// (Layer 1 only handles Int params / Int return.)
pub unsafe fn register_extern(name: &str, params: Vec<Type>, ret: Type, fn_ptr: usize) {
    registry()
        .lock()
        .unwrap()
        .insert(name.to_owned(), ExternEntry { params, ret, fn_ptr });
}

/// Look up a previously-registered extern.
pub fn lookup_extern(name: &str) -> Option<ExternEntry> {
    registry().lock().unwrap().get(name).cloned()
}

/// Remove all registered externs. Useful for test isolation.
pub fn clear_externs() {
    registry().lock().unwrap().clear();
}

/// Snapshot the current registry. Useful for diagnostic logging.
pub fn registered_extern_names() -> Vec<String> {
    let g = registry().lock().unwrap();
    let mut v: Vec<String> = g.keys().cloned().collect();
    v.sort();
    v
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
