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
pub fn start_agent() -> std::io::Result<String> {
    memscope_agent::start()
}

/// Start the agent at an explicit socket path.
pub fn start_agent_at(path: &str) -> std::io::Result<()> {
    memscope_agent::start_at(path)
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
