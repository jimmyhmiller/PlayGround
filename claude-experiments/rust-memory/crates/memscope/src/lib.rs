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
    drain_events, set_backtrace_depth, set_capture_sites, set_mode, set_sample_rate, snapshot,
    stats, MemScope, Mode, Stats,
};
pub use memscope_proto::{
    AllocShape, Frame, LiveAlloc, RawEvent, SiteInfo, Snapshot, TypeId, TypeInfo,
};

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
