//! Serverless function backend: invoke a Rust app compiled as a `cdylib` in
//! process via `dlopen`, instead of proxying to a long-running upstream.
//!
//! A function route points at a dynamic library (`.so`/`.dylib`/`.dll`) built
//! against `gatekeeper-fn`. The gate loads it once (lazily, on first request),
//! checks its ABI version, caches the open handle, and on each request marshals
//! the request into the `#[repr(C)]` [`GkRequest`], calls the exported
//! `gk_handle`, copies the response out, and hands the response pointer back to
//! the library's `gk_free`.
//!
//! ## Why this is safe to share-fate with the gate
//!
//! The function side (in `gatekeeper-fn`) catches panics and returns a 500, so a
//! handler panic does not unwind across the ABI into the gate. The remaining
//! shared-fate risk is genuine UB inside a function (a function dylib is trusted
//! native code — it can do anything). That is acceptable here for the same
//! reason `proxy` upstreams are trusted: you deploy your own functions. We do
//! NOT load arbitrary untrusted dylibs.
//!
//! ## Ownership across the boundary
//!
//! The gate owns the request buffers (kept alive on the stack across the call).
//! The function owns the response; the gate copies it into a [`Reply`] and then
//! calls `gk_free`. Each side frees only what it allocated — see `gatekeeper-abi`.

use std::collections::HashMap;
use std::ffi::OsString;
use std::os::raw::c_char;
use std::sync::Mutex;

use gatekeeper_abi::{GkHeader, GkRequest, GkResponse, GK_ABI_VERSION};

use crate::reply::Reply;

/// A loaded function dylib plus its resolved symbols. The `Library` must outlive
/// every raw `fn` pointer taken from it, so we keep it boxed alongside them and
/// only ever hand out copies of the (Copy) fn pointers while the cache holds the
/// `Library`.
struct LoadedFn {
    // Field order matters for drop: symbols are pointers into `_lib`, so `_lib`
    // must be dropped last. Rust drops fields in declaration order, so keep
    // `_lib` last.
    handle: HandleFn,
    free: FreeFn,
    _lib: libloading::Library,
}

type VersionFn = unsafe extern "C" fn() -> u32;
type HandleFn = unsafe extern "C" fn(*const GkRequest) -> *mut GkResponse;
type FreeFn = unsafe extern "C" fn(*mut GkResponse);

// Safe to share across worker threads: after load it is immutable, and the
// underlying handler is required to be thread-safe (it is pure per-call in the
// SDK). libloading::Library is Send + Sync; raw fn pointers are Send + Sync.
unsafe impl Send for LoadedFn {}
unsafe impl Sync for LoadedFn {}

/// Process-wide cache of loaded function dylibs, keyed by library path. Lazily
/// populated: a function is loaded on its first request and kept resident.
pub struct FunctionRegistry {
    loaded: Mutex<HashMap<OsString, std::sync::Arc<LoadedFn>>>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        FunctionRegistry {
            loaded: Mutex::new(HashMap::new()),
        }
    }

    /// Handle a request against the function dylib at `lib_path`. Loads (and
    /// version-checks) the library on first use; reuses the cached handle after.
    /// Any failure fails closed with a 5xx Reply rather than panicking.
    pub fn invoke(
        &self,
        lib_path: &std::path::Path,
        method: &str,
        path: &str,
        query: &str,
        headers: &[tiny_http::Header],
        body: &[u8],
    ) -> Reply {
        let func = match self.get_or_load(lib_path) {
            Ok(f) => f,
            Err(e) => return Reply::status(502, &format!("Bad Gateway: function load failed: {e}")),
        };
        call_function(&func, method, path, query, headers, body)
    }

    /// Return the cached function for `lib_path`, loading it if absent.
    fn get_or_load(
        &self,
        lib_path: &std::path::Path,
    ) -> Result<std::sync::Arc<LoadedFn>, String> {
        let key = lib_path.as_os_str().to_os_string();
        {
            let map = self.loaded.lock().unwrap();
            if let Some(f) = map.get(&key) {
                return Ok(std::sync::Arc::clone(f));
            }
        }
        // Load outside any double-checked race: two threads may both load on the
        // very first concurrent hit; the second insert just wins. Harmless (the
        // loser's Library drops). Keeping the load off the lock avoids holding it
        // across dlopen.
        let loaded = std::sync::Arc::new(load_library(lib_path)?);
        let mut map = self.loaded.lock().unwrap();
        let entry = map
            .entry(key)
            .or_insert_with(|| std::sync::Arc::clone(&loaded));
        Ok(std::sync::Arc::clone(entry))
    }
}

impl Default for FunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Open a dylib and resolve + version-check its symbols. Fails closed on any
/// missing symbol or ABI mismatch.
fn load_library(lib_path: &std::path::Path) -> Result<LoadedFn, String> {
    // SAFETY: loading a dylib runs its initializers; we trust our own functions
    // (same trust level as a proxy upstream — see module docs).
    let lib = unsafe {
        libloading::Library::new(lib_path)
            .map_err(|e| format!("dlopen {}: {e}", lib_path.display()))?
    };

    // Resolve symbols. We immediately copy the fn pointers out (they're Copy);
    // they stay valid as long as `lib` is alive, which we guarantee by storing
    // `lib` in the same struct.
    let version: VersionFn = unsafe {
        *lib.get(gatekeeper_abi::GK_ABI_VERSION_SYMBOL)
            .map_err(|e| format!("symbol gk_abi_version: {e}"))?
    };
    let handle: HandleFn = unsafe {
        *lib.get(gatekeeper_abi::GK_HANDLE_SYMBOL)
            .map_err(|e| format!("symbol gk_handle: {e}"))?
    };
    let free: FreeFn = unsafe {
        *lib.get(gatekeeper_abi::GK_FREE_SYMBOL)
            .map_err(|e| format!("symbol gk_free: {e}"))?
    };

    let got = unsafe { version() };
    if got != GK_ABI_VERSION {
        return Err(format!(
            "ABI version mismatch: dylib reports {got}, gate expects {GK_ABI_VERSION} \
             (rebuild the function against this gatekeeper)"
        ));
    }

    Ok(LoadedFn {
        handle,
        free,
        _lib: lib,
    })
}

/// Marshal a request, call `gk_handle`, copy the response out, free it. All the
/// borrowed buffers (method/path/query/headers/body) live on this stack frame
/// for the whole duration of the call, satisfying the ABI's borrow contract.
fn call_function(
    func: &LoadedFn,
    method: &str,
    path: &str,
    query: &str,
    headers: &[tiny_http::Header],
    body: &[u8],
) -> Reply {
    // Stage header name/value bytes into owned buffers we control, then build
    // the GkHeader array pointing into them. Both buffers and the array outlive
    // the call.
    let header_bytes: Vec<(Vec<u8>, Vec<u8>)> = headers
        .iter()
        .map(|h| {
            (
                h.field.as_str().as_str().as_bytes().to_vec(),
                h.value.as_str().as_bytes().to_vec(),
            )
        })
        .collect();
    let gk_headers: Vec<GkHeader> = header_bytes
        .iter()
        .map(|(n, v)| GkHeader {
            name_ptr: n.as_ptr() as *const c_char,
            name_len: n.len(),
            value_ptr: v.as_ptr() as *const c_char,
            value_len: v.len(),
        })
        .collect();

    let req = GkRequest {
        method_ptr: method.as_ptr() as *const c_char,
        method_len: method.len(),
        path_ptr: path.as_ptr() as *const c_char,
        path_len: path.len(),
        query_ptr: query.as_ptr() as *const c_char,
        query_len: query.len(),
        headers_ptr: gk_headers.as_ptr(),
        header_count: gk_headers.len(),
        body_ptr: body.as_ptr(),
        body_len: body.len(),
    };

    // SAFETY: req's buffers are all alive on this frame; handle is a valid fn
    // pointer from the loaded library. The function side catches its own panics.
    let resp_ptr = unsafe { (func.handle)(&req) };
    if resp_ptr.is_null() {
        return Reply::status(500, "function returned no response");
    }

    // Copy the response out of the function-owned memory, then free it. We copy
    // before freeing and never retain any function pointer past gk_free.
    let reply = unsafe { copy_response(resp_ptr) };
    unsafe { (func.free)(resp_ptr) };
    reply
}

/// Copy a function-owned `GkResponse` into an owned [`Reply`]. Does not free the
/// response — the caller does that via the function's `gk_free` afterward.
///
/// # Safety
/// `resp_ptr` must be a non-null pointer returned by the function's `gk_handle`.
unsafe fn copy_response(resp_ptr: *const GkResponse) -> Reply {
    let resp = &*resp_ptr;

    let body = if resp.body_ptr.is_null() || resp.body_len == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(resp.body_ptr, resp.body_len).to_vec()
    };

    let mut reply = Reply::new(resp.status, body);

    if !resp.headers_ptr.is_null() {
        let hdrs = std::slice::from_raw_parts(resp.headers_ptr, resp.header_count);
        for h in hdrs {
            let name = bytes_to_string(h.name_ptr, h.name_len);
            let value = bytes_to_string(h.value_ptr, h.value_len);
            if !name.is_empty() {
                reply = reply.with_header(&name, &value);
            }
        }
    }
    reply
}

unsafe fn bytes_to_string(ptr: *const c_char, len: usize) -> String {
    if ptr.is_null() || len == 0 {
        return String::new();
    }
    let slice = std::slice::from_raw_parts(ptr as *const u8, len);
    String::from_utf8_lossy(slice).into_owned()
}
