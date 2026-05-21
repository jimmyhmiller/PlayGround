//! Per-session terminal daemon — headless, no GUI deps.
//!
//! This crate hosts the long-lived process that owns the PTY + child
//! shell behind one `TerminalSession`, plus the wire protocol the
//! editor's GUI crate uses to attach over a Unix socket.
//!
//! Living in its own crate (deliberately not linked against libghostty,
//! Bevy, or any rendering code) keeps the daemon binary deployable on
//! its own and out of the editor's heavy dynamic-link surface.

pub mod daemon;
pub mod proto;

use std::path::PathBuf;

/// Root for daemon-owned on-disk state (pid files, scrollback). Mirrors
/// `terminal_bevy::data_dir`.
pub fn data_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".terminal-bevy");
    Some(p)
}

/// Runtime directory for short-lived ephemera that must respect path-
/// length limits. On macOS `sockaddr_un.sun_path` is only 104 bytes, so
/// nested $HOME/.terminal-bevy/... paths blow it out on most systems.
/// Default `/tmp/.terminal-bevy-<uid>/`; overridable via the
/// `TERMINAL_BEVY_RUNTIME_DIR` env var (used by tests to isolate).
pub fn runtime_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("TERMINAL_BEVY_RUNTIME_DIR") {
        return PathBuf::from(p);
    }
    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("/tmp/.terminal-bevy-{}", uid))
}

/// Unix socket the per-session daemon listens on. Short path on purpose
/// (see [`runtime_dir`]).
pub fn socket_path(session_id: u64) -> Option<PathBuf> {
    Some(runtime_dir().join(format!("{}.sock", session_id)))
}

/// PID file the daemon writes on startup. Co-located with the socket so
/// cleanup is one directory.
pub fn pid_path(session_id: u64) -> Option<PathBuf> {
    Some(runtime_dir().join(format!("{}.pid", session_id)))
}

/// Side-channel Unix socket for one-shot input injection. The daemon
/// accepts on this socket, reads raw bytes from the connection until
/// the peer closes, and writes them to the PTY master — without
/// disturbing whatever client is attached to the main socket.
pub fn inject_socket_path(session_id: u64) -> Option<PathBuf> {
    Some(runtime_dir().join(format!("{}.inject", session_id)))
}
