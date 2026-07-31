//! `memscope` — drop-in JVM-style memory tooling for Rust.
//!
//! Add the allocator and start the agent; then attach the `memscope` CLI (or
//! your own UI) over the Unix socket for a live allocation monitor and
//! type-resolved heap dumps.
//!
//! ```ignore
//! #[global_allocator]
//! static GLOBAL: memscope::MemScope = memscope::MemScope::system();
//!
//! fn main() {
//!     memscope::set_mode(memscope::Mode::Full);
//!     memscope::start_agent();            // prints the socket path
//!     // ... your program ...
//! }
//! ```
//!
//! Requirements: build with debug info (`[profile.*] debug = true`) so DWARF
//! type recovery works. On macOS a `.dSYM` is generated automatically on first
//! snapshot. No nightly, no toolchain changes.

pub use memscope_core::{
    drain_events, key_id, key_name, mark, mark_label, meta_context, push_meta, ring_dropped,
    set_backtrace_depth, set_capture_sites, set_event_streaming, set_frame_pointer_unwinding,
    set_mode, set_ring_mode, set_sample_rate, snapshot, spawn_consumer, stats, Consumer, EventSink,
    FanOut, FnSink, LiveRec, LiveSet, MemScope, MetaGuard, Mode, RingMode, Stats,
};
pub use memscope_proto::{
    AllocShape, EventKind, Frame, LiveAlloc, MetaValue, RawEvent, SiteInfo, Snapshot, TypeId,
    TypeInfo,
};

/// Attach arbitrary key/value metadata to every allocation made in the current
/// scope. Returns a guard; the scope ends when it drops. Scopes nest and merge.
///
/// ```ignore
/// let _m = memscope::meta!(subsystem = "parser", file = path);
/// parse(input);                       // allocs tagged { subsystem: "parser", file: … }
///
/// let _m = memscope::meta!(request = req.id);   // dynamic values are fine
/// ```
///
/// Values may be any type implementing `Into<MetaValue>` (`&str`/`String`, the
/// integer types, `f64`, `bool`). Keep the guard bound (`let _m = …`); a bare
/// `let _ = …` would drop it immediately and tag nothing.
#[macro_export]
macro_rules! meta {
    ($($key:ident = $val:expr),+ $(,)?) => {
        $crate::push_meta(&[
            $( ($crate::key_id(stringify!($key)), $crate::MetaValue::from($val)) ),+
        ])
    };
}

/// Start the transport agent on a background thread. Returns the socket path a
/// consumer should connect to (also printed to stderr). Override the path with
/// the `MEMSCOPE_SOCK` environment variable.
///
/// Also honors `MEMSCOPE_RECORD=<path>`: when set, the full allocation stream
/// is additionally recorded to that file (as if [`record_to_file`] were
/// called) — so an integrated program supports env-driven recording with no
/// extra code. An explicit earlier `record_to_file` call wins.
pub fn start_agent() -> std::io::Result<String> {
    maybe_record_from_env();
    // Same env-trigger contract as injection (`memscope run`): SIGUSR1,
    // MEMSCOPE_HPROF_ON_EXIT / _AT_BYTES / _OUT — so `run` can drive an
    // integrated binary identically, just without the preload shim.
    memscope_agent::install_env_triggers();
    memscope_agent::spawn_trigger_thread();
    memscope_agent::start()
}

/// Start the agent at an explicit socket path. Honors `MEMSCOPE_RECORD` like
/// [`start_agent`].
pub fn start_agent_at(path: &str) -> std::io::Result<()> {
    maybe_record_from_env();
    memscope_agent::install_env_triggers();
    memscope_agent::spawn_trigger_thread();
    memscope_agent::start_at(path)
}

/// Bring memscope up from the environment — the entry point for code with no
/// `fn main()` to edit: a **Node `.node` addon**, a Python extension, any Rust
/// cdylib a host process loads.
///
/// Call it once from your module's init hook (napi-rs: `#[napi::module_init]`;
/// neon: the `#[neon::main]` fn; raw N-API: `napi_register_module_v1`). Nothing
/// else is needed — everything is driven by env vars, so the module can ship
/// with this call permanently in place and cost nothing when they're unset:
///
/// * `MEMSCOPE_MODE=full|sampled[:rate]|off` — tracking mode (default `full`
///   when any other memscope env var is set, otherwise `off`).
/// * `MEMSCOPE_RECORD=<path.mscope>` — record the whole allocation stream
///   (`{pid}` expands, which matters when the host spawns worker processes).
/// * `MEMSCOPE_LIVE=1` — also start the agent, for `memscope monitor`/`graph`.
/// * `MEMSCOPE_HPROF_ON_EXIT` / `_AT_BYTES` / `_OUT` and `SIGUSR1` — heap-dump
///   triggers, identical to the ones `memscope run` drives.
///
/// ```ignore
/// #[global_allocator]
/// static GLOBAL: memscope::MemScope = memscope::MemScope::system();
///
/// #[napi::module_init]
/// fn init() {
///     memscope::init();
/// }
/// ```
///
/// Then: `MEMSCOPE_RECORD=/tmp/addon.mscope node app.js && memscope analyze /tmp/addon.mscope`.
///
/// Recordings and dumps from a module symbolicate against **the module's own
/// DWARF** (found via `dladdr`), not the host executable's — `node` has none.
pub fn init() {
    // Off unless asked: a module linking memscope must be able to ship as-is.
    let mode = std::env::var("MEMSCOPE_MODE").unwrap_or_default();
    let anything_asked = mode_is_on(&mode)
        || ["MEMSCOPE_RECORD", "MEMSCOPE_LIVE", "MEMSCOPE_SOCK", "MEMSCOPE_HPROF_ON_EXIT",
            "MEMSCOPE_HPROF_AT_BYTES", "MEMSCOPE_HPROF_OUT"]
            .iter()
            .any(|k| std::env::var_os(k).is_some_and(|v| !v.is_empty()));
    if !anything_asked {
        return;
    }
    match parse_mode(&mode) {
        Some((m, rate)) => {
            if let Some(r) = rate {
                set_sample_rate(r);
            }
            set_mode(m);
        }
        // Any of the capture env vars, but no explicit mode: they asked for data.
        None => set_mode(Mode::Full),
    }

    maybe_record_from_env();
    memscope_agent::install_env_triggers();
    memscope_agent::spawn_trigger_thread();

    if std::env::var_os("MEMSCOPE_LIVE").is_some_and(|v| v == "1" || v == "true")
        || std::env::var_os("MEMSCOPE_SOCK").is_some()
    {
        // `start` announces the socket path itself; only its failure is news.
        if let Err(e) = memscope_agent::start() {
            eprintln!("[memscope] could not start the agent: {e}");
        }
    }
}

/// `full` | `sampled` | `sampled:200` | `off` → (mode, sample rate).
fn parse_mode(s: &str) -> Option<(Mode, Option<u32>)> {
    let s = s.trim();
    let (name, rate) = match s.split_once(':') {
        Some((n, r)) => (n, r.parse::<u32>().ok()),
        None => (s, None),
    };
    match name.to_ascii_lowercase().as_str() {
        "full" => Some((Mode::Full, None)),
        "sampled" | "sample" => Some((Mode::Sampled, rate)),
        "off" | "none" => Some((Mode::Off, None)),
        _ => None,
    }
}

/// Does `MEMSCOPE_MODE` name a tracking mode (i.e. not unset, not `off`)?
fn mode_is_on(s: &str) -> bool {
    matches!(parse_mode(s), Some((Mode::Full | Mode::Sampled, _)))
}

fn maybe_record_from_env() {
    let Ok(path) = std::env::var("MEMSCOPE_RECORD") else { return };
    if path.is_empty() {
        return;
    }
    match memscope_agent::record_to_file(&path) {
        Ok(()) => {}
        // An explicit record_to_file already started — env is a no-op then.
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => eprintln!("[memscope] MEMSCOPE_RECORD={path}: failed to start recording: {e}"),
    }
}

/// Stream the full allocation event stream to a self-contained file (resolved
/// types + stacks, newline-JSON). Switches the ring to Reliable mode so nothing
/// is dropped. Read it back posthoc with `memscope replay <file>` or your own
/// viewer. Requires `set_mode(Full)` for a complete trace.
pub use memscope_agent::record_to_file;

/// Write a JVM **HPROF** heap dump of the current process to `path`, openable in
/// Eclipse MAT / VisualVM (dominator tree, retained sizes, paths-to-GC-roots).
///
/// Recovers types + layout from DWARF, then **`fork()`s** so the heap is walked
/// against a frozen copy-on-write image — a consistent point-in-time snapshot
/// without pausing the program (like Redis BGSAVE / `gcore`). Memory is read
/// *safely* (via Mach `mach_vm_read_overwrite`), so a since-freed address can't
/// crash the dump. Requires `[profile.*] debug = true` for type recovery.
pub use memscope_agent::{heap_dump, HprofStats};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_env_parses() {
        assert!(matches!(parse_mode("full"), Some((Mode::Full, None))));
        assert!(matches!(parse_mode("FULL"), Some((Mode::Full, None))));
        assert!(matches!(parse_mode(" sampled "), Some((Mode::Sampled, None))));
        assert!(matches!(parse_mode("sampled:200"), Some((Mode::Sampled, Some(200)))));
        assert!(matches!(parse_mode("off"), Some((Mode::Off, None))));
        // Unset / nonsense must not silently mean "off": init() decides that
        // from whether anything was asked for at all.
        assert!(parse_mode("").is_none());
        assert!(parse_mode("verbose").is_none());
    }

    #[test]
    fn only_a_tracking_mode_counts_as_on() {
        assert!(mode_is_on("full"));
        assert!(mode_is_on("sampled:50"));
        assert!(!mode_is_on("off"));
        assert!(!mode_is_on(""));
    }
}
