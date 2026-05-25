//! PTY-related types used across the crate.
//!
//! Historical note: this module used to host `Pty`, `Child`, and the
//! actual `forkpty()` + non-blocking I/O plumbing. The editor process no
//! longer owns the PTY directly — see `daemon.rs` for the per-session
//! daemon that does, and `daemon_client.rs` for the editor-side
//! connection. What's left here is the `PtySize` value type, which is
//! convenient shared vocabulary between the worker, the resize systems,
//! and any future place that wants to talk about terminal dimensions.

/// Pixel + cell dimensions for a terminal grid.
#[derive(Clone, Copy, Debug)]
pub struct PtySize {
    pub cols: u16,
    pub rows: u16,
    pub cell_width_px: u16,
    pub cell_height_px: u16,
}
