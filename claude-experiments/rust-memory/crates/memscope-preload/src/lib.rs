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
use std::sync::atomic::{AtomicBool, Ordering};
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

#[inline]
fn recording() -> bool {
    RECORDING.load(Ordering::Relaxed)
}

/// Spawn the pump + trigger watcher once, from a normal program thread (not
/// from the dyld constructor, to avoid pre-main thread-spawn fragility). The
/// triggers themselves (SIGUSR1 / at-exit / at-bytes) live in memscope-agent —
/// the same contract an integrated `start_agent` program installs.
fn ensure_workers() {
    static START: Once = Once::new();
    START.call_once(|| {
        memscope_agent::spawn_trigger_thread();
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

    // If MEMSCOPE_RECORD is set, stream the full allocation trace to that path
    // as a self-contained `.mscope` (binary, or `.json`/`.jsonl` if the extension
    // matches). This is what `memscope perfetto|flamegraph|analyze` consume for
    // churn / timeline / perfetto — not just the final live-heap HPROF.
    //
    // The preload lib already links memscope-agent + memscope-core, so it can
    // call record_to_file directly without the target needing #[global_allocator].
    // This makes zero-instrumentation injection produce a full alloc-stream,
    // not just a live-heap snapshot.
    if let Ok(rec_path) = std::env::var("MEMSCOPE_RECORD") {
        if !rec_path.is_empty() {
            // We live in an injected dylib, but the frames we capture belong to
            // the target executable — symbolicate the recording against it, not us.
            memscope_agent::record_against_main_executable();
            // Start the file recorder; it switches the ring to Reliable and spawns
            // its own pump. Ignore errors — falling back to HPROF mode is still
            // useful for live-heap debugging.
            if let Err(e) = memscope_agent::record_to_file(&rec_path) {
                eprintln!("[memscope-preload] MEMSCOPE_RECORD={rec_path}: failed to start file recorder: {e}");
            } else {
                eprintln!("[memscope-preload] recording full alloc-stream to {rec_path}");
            }
        }
    }

    // SIGUSR1 + at-exit dump — the shared env-trigger contract (also installed
    // by `start_agent` in integrated programs). Signal/atexit registration is
    // pre-main-safe; the watcher thread is spawned later by ensure_workers().
    memscope_agent::install_env_triggers();

    RECORDING.store(true, Ordering::Release);
    eprintln!(
        "[memscope-preload] attached to pid {} — kill -USR1 {} for a heap dump",
        std::process::id(),
        std::process::id()
    );
}
