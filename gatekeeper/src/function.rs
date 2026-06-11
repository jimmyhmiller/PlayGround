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
    /// Optional self-description symbol (ABI v2). `None` if the dylib doesn't
    /// export `gk_describe` — describing it then yields "(no description)".
    describe: Option<DescribeFn>,
    _lib: libloading::Library,
    /// Each version is loaded from a UNIQUE private copy of the dylib (see
    /// `load_library`), so the dynamic linker treats a reloaded build as a
    /// distinct library instead of returning the still-mapped old handle. We
    /// own that temp file and delete it when this `LoadedFn` drops (after the
    /// library is unmapped).
    temp_path: Option<std::path::PathBuf>,
}

impl Drop for LoadedFn {
    fn drop(&mut self) {
        // The library is unmapped as part of this same drop (fields drop in
        // order; `_lib` after these). Removing the file now is fine on Unix:
        // an unlinked-but-mapped file stays valid until unmapped, and the OS
        // reclaims it after. Best-effort; a leftover temp is harmless.
        if let Some(p) = &self.temp_path {
            let _ = std::fs::remove_file(p);
        }
    }
}

type VersionFn = unsafe extern "C" fn() -> u32;
type DescribeFn = unsafe extern "C" fn() -> *mut GkResponse;
type HandleFn = unsafe extern "C" fn(*const GkRequest) -> *mut GkResponse;
type FreeFn = unsafe extern "C" fn(*mut GkResponse);

// Safe to share across worker threads: after load it is immutable, and the
// underlying handler is required to be thread-safe (it is pure per-call in the
// SDK). libloading::Library is Send + Sync; raw fn pointers are Send + Sync.
unsafe impl Send for LoadedFn {}
unsafe impl Sync for LoadedFn {}

/// A fingerprint of the dylib file on disk, used to detect that a function has
/// been rebuilt/replaced so we re-`dlopen` it. We compare modification time,
/// size, and inode: any change means a different build, even one that reuses the
/// same path and the same mtime-second (size/inode still move on a fresh write).
#[derive(Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    mtime: std::time::SystemTime,
    size: u64,
    inode: u64,
}

impl FileStamp {
    /// Stat `path` into a stamp. An error (missing file, no perms) yields `None`;
    /// callers treat that as "can't refresh" and keep any currently-cached lib.
    fn of(path: &std::path::Path) -> Option<FileStamp> {
        use std::os::unix::fs::MetadataExt;
        let md = std::fs::metadata(path).ok()?;
        Some(FileStamp {
            mtime: md.modified().ok()?,
            size: md.len(),
            inode: md.ino(),
        })
    }
}

/// A cache entry: the loaded library plus the stamp of the file it was loaded
/// from. When the on-disk stamp no longer matches, the entry is stale.
struct CacheEntry {
    func: std::sync::Arc<LoadedFn>,
    stamp: FileStamp,
}

/// Process-wide cache of loaded function dylibs, keyed by library path. Lazily
/// populated: a function is loaded on its first request and kept resident until
/// the file on disk changes, at which point it is transparently reloaded — so
/// shipping a new build of a function (same path) takes effect on the next
/// request, no restart or reload needed.
pub struct FunctionRegistry {
    loaded: Mutex<HashMap<OsString, CacheEntry>>,
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

    /// Ask the function dylib at `lib_path` to describe itself (ABI v2). Loads it
    /// if needed. Returns `Ok(Some(json))` if it exports `gk_describe`,
    /// `Ok(None)` if it doesn't (no description), or `Err` if the dylib can't be
    /// loaded at all. Used to build the `/_gatekeeper/describe` catalog.
    pub fn describe(&self, lib_path: &std::path::Path) -> Result<Option<String>, String> {
        let func = self.get_or_load(lib_path)?;
        let Some(describe_fn) = func.describe else {
            return Ok(None);
        };
        // SAFETY: describe_fn is a valid fn pointer from the loaded library; the
        // function side catches its own panics. It returns an owned GkResponse we
        // copy out and then free via the dylib's own gk_free (same contract as a
        // handler response).
        let resp_ptr = unsafe { describe_fn() };
        if resp_ptr.is_null() {
            return Err("gk_describe returned null".into());
        }
        let reply = unsafe { copy_response(resp_ptr) };
        unsafe { (func.free)(resp_ptr) };
        Ok(Some(String::from_utf8_lossy(&reply.body).into_owned()))
    }

    /// Return the function for `lib_path`, (re)loading it if absent or if the
    /// file on disk has changed since we cached it.
    ///
    /// Hot code update: each call stats the file. If the stamp matches the cached
    /// entry, we return the warm handle (the common, fast path — one stat, no
    /// dlopen). If it differs, the dylib was rebuilt/replaced, so we load the new
    /// one and swap it in. The old `Arc<LoadedFn>` lives until in-flight requests
    /// release it, then unloads — so a request running on the old code finishes
    /// safely.
    ///
    /// Fail-safe: if the new file fails to load (bad build, ABI mismatch) but we
    /// still hold a working cached version, we keep serving the old one rather
    /// than erroring — a botched function build can't take the route down.
    fn get_or_load(
        &self,
        lib_path: &std::path::Path,
    ) -> Result<std::sync::Arc<LoadedFn>, String> {
        let key = lib_path.as_os_str().to_os_string();
        let disk = FileStamp::of(lib_path);

        // Fast path: cached entry whose stamp still matches the file on disk.
        {
            let map = self.loaded.lock().unwrap();
            if let Some(entry) = map.get(&key) {
                match disk {
                    // Unchanged (or we couldn't stat — then we can't know it
                    // changed, so keep the warm copy): serve the cache.
                    Some(s) if s == entry.stamp => return Ok(std::sync::Arc::clone(&entry.func)),
                    None => return Ok(std::sync::Arc::clone(&entry.func)),
                    // Changed on disk: fall through to reload below.
                    Some(_) => {}
                }
            }
        }

        // Load the (new or first) version off-lock — dlopen can be slow and we
        // don't want to hold the map mutex across it. Re-stat right before the
        // load so the stamp we store matches the bytes we actually loaded (closes
        // a write-during-load race: if it changes again we'll just reload next
        // time).
        let pre_stamp = FileStamp::of(lib_path);
        match load_library(lib_path) {
            Ok(lib) => {
                let func = std::sync::Arc::new(lib);
                let stamp = pre_stamp.or(disk);
                let mut map = self.loaded.lock().unwrap();
                if let Some(stamp) = stamp {
                    map.insert(key, CacheEntry { func: std::sync::Arc::clone(&func), stamp });
                } else {
                    // No stamp available at all (file vanished mid-flight); serve
                    // this load but don't cache it under an unknown identity.
                    map.remove(&key);
                }
                Ok(func)
            }
            Err(e) => {
                // New build failed to load. If we have a working older version,
                // keep serving it (fail-safe); otherwise surface the error.
                let map = self.loaded.lock().unwrap();
                if let Some(entry) = map.get(&key) {
                    eprintln!(
                        "gatekeeper: function {}: reload failed, keeping previous version: {e}",
                        lib_path.display()
                    );
                    Ok(std::sync::Arc::clone(&entry.func))
                } else {
                    Err(e)
                }
            }
        }
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
    // Two dynamic-linker hazards make a naive `dlopen(lib_path)` wrong for
    // hot-reload, and we defuse both here:
    //
    // 1. SAME-PATH CACHING. glibc keys loaded libraries by resolved pathname and
    //    refcounts them. While the OLD build is still mapped (an in-flight
    //    request holds its Arc), `dlopen` of the SAME path returns the OLD
    //    handle — so a rebuilt function would keep running the old code. Fix:
    //    copy the new bytes to a UNIQUE temp path and dlopen THAT, so the linker
    //    sees a distinct library every time. We delete the temp on drop.
    //
    // 2. SHARED SYMBOL NAMES. Every function dylib exports the same
    //    `gk_handle`/`gk_free`/`gk_abi_version` (from the SDK). Under the default
    //    RTLD_GLOBAL, those names bind to whichever library loaded first, so a
    //    second function would call the first one's code. Fix: RTLD_LOCAL keeps
    //    each library's symbols private to its own handle. RTLD_NOW resolves
    //    eagerly so a missing symbol fails here, not mid-request.
    let temp_path = unique_temp_path(lib_path);
    std::fs::copy(lib_path, &temp_path).map_err(|e| {
        format!(
            "staging {} -> {}: {e}",
            lib_path.display(),
            temp_path.display()
        )
    })?;

    // SAFETY: loading a dylib runs its initializers; we trust our own functions
    // (same trust level as a proxy upstream — see module docs).
    let lib = unsafe {
        use libloading::os::unix as ldunix;
        match ldunix::Library::open(
            Some(&temp_path),
            ldunix::RTLD_NOW | ldunix::RTLD_LOCAL,
        ) {
            Ok(l) => libloading::Library::from(l),
            Err(e) => {
                let _ = std::fs::remove_file(&temp_path);
                return Err(format!("dlopen {}: {e}", lib_path.display()));
            }
        }
    };

    // Resolve symbols + version-check. On any failure here we must delete the
    // staged temp copy (the early `?`s won't run the Drop, since we haven't built
    // the LoadedFn yet), so wrap the fallible part and clean up on error.
    let built = (|| -> Result<LoadedFn, String> {
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

        // gk_describe is OPTIONAL (ABI v2). Resolve it if present; a missing
        // symbol is fine — that function just has no self-description.
        let describe: Option<DescribeFn> = unsafe {
            lib.get(gatekeeper_abi::GK_DESCRIBE_SYMBOL).ok().map(|s| *s)
        };

        Ok(LoadedFn {
            handle,
            free,
            describe,
            _lib: lib,
            temp_path: Some(temp_path.clone()),
        })
    })();

    if built.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    built
}

/// A unique path in the system temp dir to stage a private copy of `lib_path`
/// before `dlopen`, so each load is a distinct library to the dynamic linker.
/// Unique across processes and concurrent loads via pid + a counter + nanos.
fn unique_temp_path(lib_path: &std::path::Path) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let stem = lib_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("gkfn");
    // Keep a `.so` extension so the loader is happy on all platforms.
    let name = format!("{stem}.gk.{}.{n}.{nanos}.so", std::process::id());
    std::env::temp_dir().join(name)
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
