//! Env-driven heap-dump triggers — one contract, both capture paths.
//!
//! `memscope run` drives its target with environment variables; these used to
//! be honored only by the preload shim, which meant an *integrated* program
//! (the 2-line `start_agent` setup) couldn't be driven the same way. Now both
//! install the same triggers:
//!
//! * `MEMSCOPE_HPROF_ON_EXIT=1`     — dump the live heap at process exit
//! * `MEMSCOPE_HPROF_AT_BYTES=50MB` — dump when the live heap first crosses N
//! * `MEMSCOPE_HPROF_OUT=path`      — output template (`{pid}`/`{n}` expand)
//! * `SIGUSR1`                      — dump now (what `memscope dump-pid` sends)
//!
//! Registration is split in two because the preload shim starts before `main`:
//! [`install_env_triggers`] only registers the signal handler + atexit hook
//! (pre-main-safe); [`spawn_trigger_thread`] starts the watcher thread and must
//! be called from a normal program thread. `start_agent` calls both.

use std::os::raw::c_int;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Once;

/// Set by the SIGUSR1 handler; the watcher thread polls it.
static DUMP_REQUESTED: AtomicBool = AtomicBool::new(false);
/// Dump counter, for the `{n}` slot in the output path.
static DUMP_SEQ: AtomicU32 = AtomicU32::new(0);

/// Register the SIGUSR1 handler and (when `MEMSCOPE_HPROF_ON_EXIT` is set) the
/// at-exit dump. Safe to call before `main`; idempotent.
pub fn install_env_triggers() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // SAFETY: registering an async-signal-safe handler (it only stores an
        // atomic) and a plain extern "C" atexit hook.
        unsafe {
            libc::signal(libc::SIGUSR1, on_sigusr1 as *const () as usize);
            if std::env::var_os("MEMSCOPE_HPROF_ON_EXIT").is_some() {
                libc::atexit(dump_at_exit);
            }
        }
    });
}

/// Start the watcher thread: services SIGUSR1 requests and the
/// `MEMSCOPE_HPROF_AT_BYTES` threshold. Call from a normal program thread
/// (NOT a pre-main constructor); idempotent.
pub fn spawn_trigger_thread() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Kick the recorder's off-thread reconstructor so the live set stays
        // current (otherwise per-thread rings would overflow before a dump).
        let _ = memscope_core::stats();
        std::thread::Builder::new()
            .name("memscope-dumper".into())
            .spawn(|| {
                // This thread's own allocations must never be tracked.
                memscope_core::exclude_current_thread();
                // "Dump when the heap first gets big" — robust for short-lived
                // programs where dump-at-exit is too late (RAII has already
                // dropped everything by then).
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
                            eprintln!("[memscope] live heap crossed {t} bytes — dumping");
                            do_dump();
                        }
                    }
                }
            })
            .ok();
    });
}

extern "C" fn on_sigusr1(_sig: c_int) {
    DUMP_REQUESTED.store(true, Ordering::Release);
}

extern "C" fn dump_at_exit() {
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

fn do_dump() {
    let path = dump_path();
    match crate::heap_dump(&path) {
        Ok(s) => {
            eprintln!("[memscope] heap dump -> {path}: {} objects, {} classes", s.objects, s.classes)
        }
        Err(e) => eprintln!("[memscope] heap dump failed: {e}"),
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
