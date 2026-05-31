//! Generic C FFI symbol resolution.
//!
//! This is the real foreign-function interface: ai-lang declares C
//! function signatures with `extern "C" lib "<name>" { fn ... }`, and at
//! JIT-link time we resolve each symbol from a real shared library via
//! `dlopen`/`dlsym` and map the JIT'd call straight to its address. No
//! per-function Rust glue — calling `getenv` calls libc's `getenv`,
//! calling `curl_easy_init` calls libcurl's.
//!
//! `dlopen`/`dlsym`/`dlerror` live in libSystem (macOS) / libdl (Linux),
//! both linked into every process, so we declare them directly rather
//! than pulling in the `libc` crate.

use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;

/// Pseudo-handle that searches every loaded image for a symbol. Value
/// differs by platform.
#[cfg(target_os = "macos")]
const RTLD_DEFAULT: *mut c_void = (-2isize) as *mut c_void;
#[cfg(not(target_os = "macos"))]
const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();

/// Cache of opened library handles (as `usize`, since `*mut c_void`
/// isn't `Send`). Keyed by the library name the user wrote.
static HANDLES: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();

fn handles() -> &'static Mutex<HashMap<String, usize>> {
    HANDLES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Candidate filenames to try for a user-written library name. A name
/// with a path separator or an extension is tried verbatim; a bare name
/// like "curl" expands to the platform's conventional forms.
///
/// Full paths in standard library directories (Homebrew, /usr/local) are
/// tried BEFORE the bare soname. This finds Homebrew-installed libraries
/// such as OpenSSL's libcrypto, and on macOS it matters for correctness:
/// `dlopen("libcrypto.dylib")` would load Apple's restricted system
/// libcrypto, which aborts the process ("loading in an unsafe way"); the
/// full Homebrew path loads a normal, usable copy instead.
fn candidates(lib: &str) -> Vec<String> {
    if lib.contains('/') || lib.contains('.') {
        return vec![lib.to_owned()];
    }
    #[cfg(target_os = "macos")]
    let exts = ["dylib"];
    #[cfg(not(target_os = "macos"))]
    let exts = ["so"];
    #[cfg(target_os = "macos")]
    let dirs = ["/opt/homebrew/lib", "/usr/local/lib"];
    #[cfg(not(target_os = "macos"))]
    let dirs = [
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
    ];
    let mut v = Vec::new();
    for dir in dirs {
        for ext in exts {
            v.push(format!("{dir}/lib{lib}.{ext}"));
        }
    }
    for ext in exts {
        v.push(format!("lib{lib}.{ext}"));
    }
    // On Linux the unversioned `libX.so` symlink only ships in the `-dev`
    // package; a stripped runtime (e.g. a Lambda container) has only the
    // versioned soname `libX.so.N`. Try a range of versioned sonames so
    // ai-lang resolves the library without dev packages installed.
    #[cfg(not(target_os = "macos"))]
    for ver in ["5", "4", "3", "2", "1", "0"] {
        for dir in dirs {
            v.push(format!("{dir}/lib{lib}.so.{ver}"));
        }
        v.push(format!("lib{lib}.so.{ver}"));
    }
    // Also try the raw name (e.g. a framework or already-suffixed name).
    v.push(lib.to_owned());
    v
}

/// Resolve (and cache) a handle for `lib`. The names "", "c", and "m"
/// resolve to `RTLD_DEFAULT` — libc and libm symbols are already in the
/// process image (libSystem on macOS), so no `dlopen` is needed.
fn open_lib(lib: &str) -> Option<*mut c_void> {
    if lib.is_empty() || lib == "c" || lib == "m" || lib == "System" {
        return Some(RTLD_DEFAULT);
    }
    let mut cache = handles().lock().unwrap();
    if let Some(&addr) = cache.get(lib) {
        return Some(addr as *mut c_void);
    }
    for cand in candidates(lib) {
        let cs = match CString::new(cand.as_str()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let h = unsafe { dlopen(cs.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            cache.insert(lib.to_owned(), h as usize);
            return Some(h);
        }
    }
    None
}

/// Resolve a symbol `name` from library `lib` to its absolute address,
/// or `None` if the library or symbol can't be found.
pub fn resolve_symbol(lib: &str, name: &str) -> Option<usize> {
    let handle = open_lib(lib)?;
    let cname = CString::new(name).ok()?;
    let addr = unsafe { dlsym(handle, cname.as_ptr()) };
    if addr.is_null() {
        None
    } else {
        Some(addr as usize)
    }
}
