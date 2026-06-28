//! Inject memscope into an **unmodified** program — no `#[global_allocator]`, no
//! source changes. Build this as a `cdylib` and launch the target under it:
//!
//! ```sh
//! DYLD_INSERT_LIBRARIES=libmemscope_preload.dylib ./your_program   # macOS
//! LD_PRELOAD=libmemscope_preload.so ./your_program                  # Linux
//! kill -USR1 <pid>     # → writes /tmp/memscope-<pid>-<n>.hprof (open in MAT/heapster)
//! ```
//!
//! How it works: a dyld `__interpose` table redirects the target's
//! `malloc`/`calloc`/`realloc`/`free`/`posix_memalign` to wrappers here. Each
//! wrapper calls the *real* function (interposition is per-image, so the calls
//! made inside THIS dylib — including all of memscope's own machinery — bind to
//! the real allocator and never recurse) and forwards the address + size to the
//! recorder, which captures a backtrace and recovers the Rust type from the
//! target's own DWARF. A background thread waits for `SIGUSR1` and writes a
//! type-resolved HPROF heap dump.
//!
//! Requirements (none touch the target's source): the program uses the default
//! system allocator (the Rust default), and — for *type names* — has debug info
//! (a `.dSYM` / `debug = true`); without it you still get a complete untyped
//! dump (sizes, references, dominators, root paths).

#![allow(clippy::missing_safety_doc)]

use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Once;

// --- the real libc allocator (self-exempt inside the interposing image) -------

extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn calloc(count: usize, size: usize) -> *mut c_void;
    fn realloc(ptr: *mut c_void, size: usize) -> *mut c_void;
    fn free(ptr: *mut c_void);
    fn posix_memalign(memptr: *mut *mut c_void, align: usize, size: usize) -> c_int;
    /// macOS: the usable size of a malloc block (so a `free` shim knows the size).
    fn malloc_size(ptr: *const c_void) -> usize;
}

/// Recording is gated until the constructor has set up memscope, so allocations
/// during early dyld/runtime bringup pass straight through.
static RECORDING: AtomicBool = AtomicBool::new(false);
/// Set by the SIGUSR1 handler; the dumper thread polls it.
static DUMP_REQUESTED: AtomicBool = AtomicBool::new(false);
/// Dump counter, for the `{n}` slot in the output path.
static DUMP_SEQ: AtomicU32 = AtomicU32::new(0);

#[inline]
fn recording() -> bool {
    RECORDING.load(Ordering::Relaxed)
}

/// Spawn the pump + dumper thread once, from a normal program thread (not from
/// the dyld constructor, to avoid pre-main thread-spawn fragility).
fn ensure_workers() {
    static START: Once = Once::new();
    START.call_once(|| {
        // Kick the recorder's off-thread reconstructor so the live set stays
        // current (otherwise per-thread rings would overflow before a dump).
        let _ = memscope_core::stats();
        std::thread::Builder::new()
            .name("memscope-dumper".into())
            .spawn(|| {
                // This thread's own allocations must never be tracked.
                memscope_core::exclude_current_thread();
                // Optional "dump when the heap first gets big" trigger — robust
                // for short-lived programs where dump-at-exit is too late
                // (RAII has already dropped everything by then).
                let threshold: Option<u64> = std::env::var("MEMSCOPE_HPROF_AT_BYTES")
                    .ok()
                    .and_then(|s| parse_bytes(&s));
                let mut fired_threshold = false;
                loop {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    if DUMP_REQUESTED.swap(false, Ordering::AcqRel) {
                        do_dump();
                    }
                    if let Some(t) = threshold {
                        if !fired_threshold && memscope_core::stats().live_bytes >= t {
                            fired_threshold = true;
                            eprintln!("[memscope-preload] live heap crossed {t} bytes — dumping");
                            do_dump();
                        }
                    }
                }
            })
            .ok();
    });
}

// --- interpose wrappers -------------------------------------------------------

unsafe extern "C" fn my_malloc(size: usize) -> *mut c_void {
    let p = malloc(size);
    if !p.is_null() && recording() {
        ensure_workers();
        memscope_core::note_alloc(p as u64, malloc_size(p) as u64, 16);
    }
    p
}

unsafe extern "C" fn my_calloc(count: usize, size: usize) -> *mut c_void {
    let p = calloc(count, size);
    if !p.is_null() && recording() {
        ensure_workers();
        memscope_core::note_alloc(p as u64, malloc_size(p) as u64, 16);
    }
    p
}

unsafe extern "C" fn my_realloc(ptr: *mut c_void, size: usize) -> *mut c_void {
    let old_size = if ptr.is_null() { 0 } else { malloc_size(ptr) };
    let new = realloc(ptr, size);
    if !new.is_null() && recording() {
        ensure_workers();
        if ptr.is_null() {
            memscope_core::note_alloc(new as u64, malloc_size(new) as u64, 16);
        } else {
            memscope_core::note_realloc(
                ptr as u64,
                old_size as u64,
                new as u64,
                malloc_size(new) as u64,
                16,
            );
        }
    }
    new
}

unsafe extern "C" fn my_free(ptr: *mut c_void) {
    if !ptr.is_null() && recording() {
        memscope_core::note_free(ptr as u64, malloc_size(ptr) as u64, 16);
    }
    free(ptr);
}

unsafe extern "C" fn my_posix_memalign(
    memptr: *mut *mut c_void,
    align: usize,
    size: usize,
) -> c_int {
    let rc = posix_memalign(memptr, align, size);
    if rc == 0 && recording() {
        let p = *memptr;
        if !p.is_null() {
            ensure_workers();
            memscope_core::note_alloc(p as u64, malloc_size(p) as u64, align as u32);
        }
    }
    rc
}

// --- the dyld interpose table -------------------------------------------------

#[repr(C)]
struct Interpose {
    replacement: *const c_void,
    original: *const c_void,
}
unsafe impl Sync for Interpose {}

#[used]
#[link_section = "__DATA,__interpose"]
static INTERPOSERS: [Interpose; 5] = [
    Interpose { replacement: my_malloc as *const c_void, original: malloc as *const c_void },
    Interpose { replacement: my_calloc as *const c_void, original: calloc as *const c_void },
    Interpose { replacement: my_realloc as *const c_void, original: realloc as *const c_void },
    Interpose { replacement: my_free as *const c_void, original: free as *const c_void },
    Interpose {
        replacement: my_posix_memalign as *const c_void,
        original: posix_memalign as *const c_void,
    },
];

// --- constructor: set up memscope before main --------------------------------

#[used]
#[link_section = "__DATA,__mod_init_func"]
static CTOR: extern "C" fn() = init;

extern "C" fn init() {
    memscope_core::set_mode(memscope_core::Mode::Full);
    // Reliable ring so nothing is dropped between dumps (the pump applies
    // backpressure rather than overwriting).
    memscope_core::set_ring_mode(memscope_core::RingMode::Reliable);

    // SIGUSR1 -> request a dump (handled off-signal by the dumper thread).
    unsafe {
        libc::signal(libc::SIGUSR1, on_sigusr1 as *const () as usize);
    }
    // Optional dump-at-exit.
    if std::env::var_os("MEMSCOPE_HPROF_ON_EXIT").is_some() {
        unsafe {
            libc::atexit(at_exit);
        }
    }

    RECORDING.store(true, Ordering::Release);
    eprintln!(
        "[memscope-preload] attached to pid {} — kill -USR1 {} for a heap dump",
        std::process::id(),
        std::process::id()
    );
}

extern "C" fn on_sigusr1(_sig: c_int) {
    DUMP_REQUESTED.store(true, Ordering::Release);
}

extern "C" fn at_exit() {
    do_dump();
}

/// Resolve the output path from `MEMSCOPE_HPROF_OUT` (default
/// `/tmp/memscope-<pid>-<n>.hprof`), expanding `{pid}` and `{n}`.
fn dump_path() -> String {
    let pid = std::process::id();
    let n = DUMP_SEQ.fetch_add(1, Ordering::Relaxed);
    match std::env::var("MEMSCOPE_HPROF_OUT") {
        Ok(tpl) => tpl.replace("{pid}", &pid.to_string()).replace("{n}", &n.to_string()),
        Err(_) => format!("/tmp/memscope-{pid}-{n}.hprof"),
    }
}

/// Parse a byte count like `5MB`, `512KB`, `1GB`, or a plain number.
fn parse_bytes(s: &str) -> Option<u64> {
    let s = s.trim();
    let (num, mult) = if let Some(n) = s.strip_suffix("GB").or_else(|| s.strip_suffix("G")) {
        (n, 1u64 << 30)
    } else if let Some(n) = s.strip_suffix("MB").or_else(|| s.strip_suffix("M")) {
        (n, 1 << 20)
    } else if let Some(n) = s.strip_suffix("KB").or_else(|| s.strip_suffix("K")) {
        (n, 1 << 10)
    } else {
        (s, 1)
    };
    num.trim().parse::<f64>().ok().map(|v| (v * mult as f64) as u64)
}

fn do_dump() {
    let path = dump_path();
    match memscope_agent::heap_dump(&path) {
        Ok(s) => eprintln!(
            "[memscope-preload] heap dump -> {path}: {} objects, {} classes",
            s.objects, s.classes
        ),
        Err(e) => eprintln!("[memscope-preload] heap dump failed: {e}"),
    }
}
