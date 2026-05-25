//! Central Claude-Code event bus.
//!
//! One long-running daemon, one unix socket. Producers (the hook
//! event-logger, terminal-daemons announcing themselves, etc.) publish
//! frames; subscribers (the editor app, reactors) attach and receive
//! live + replayed events. Every published event is also appended to
//! `~/.claude/events.jsonl` so a crashed daemon doesn't lose history —
//! on restart the bus replays from the file's tail into its ring buffer
//! and picks up live publishes again.
//!
//! Wire format mirrors `terminal-daemon::proto` (length-prefixed
//! bincode). Path resolution lives here so producers and the daemon
//! agree on where the socket is.

pub mod client;
pub mod daemon;
pub mod proto;

use std::path::PathBuf;

/// Where the bus listens. Picked under `$HOME/.claude` so it sits next
/// to the JSONL fallback and Claude Code's own state. The path is short
/// enough on every system we ship to to stay inside the 104-byte
/// `sockaddr_un.sun_path` cap on macOS.
pub fn socket_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".claude");
    p.push("bus.sock");
    Some(p)
}

/// PID file written by the daemon on startup. Lets a fresh daemon detect
/// a stale predecessor and (eventually) lets clients query liveness
/// without opening the socket.
pub fn pid_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".claude");
    p.push("bus.pid");
    Some(p)
}

/// The durable JSONL log every event lands in. Same path the existing
/// event-logger writes to today, so the on-disk format stays compatible
/// across the rewire.
pub fn events_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".claude");
    p.push("events.jsonl");
    Some(p)
}
