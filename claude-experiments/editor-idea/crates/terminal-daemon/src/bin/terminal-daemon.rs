//! `terminal-daemon` — long-lived per-session shell host.
//!
//! Spawned by the editor when no daemon for a given `TerminalSession` is
//! alive. Detaches into the background and serves the session over a
//! Unix socket at `~/.terminal-bevy/sessions/<session_id>.sock` until
//! the child exits or the editor sends `Kill`.
//!
//! Usage:
//!
//! ```text
//! terminal-daemon <session_id> <program> [args...]
//! ```
//!
//! Internal — the editor process is the only intended caller. argv is
//! deliberately positional and unforgiving.

fn main() {
    let mut args = std::env::args().skip(1);
    let session_id: u64 = match args.next().and_then(|s| s.parse().ok()) {
        Some(id) => id,
        None => {
            eprintln!(
                "usage: terminal-daemon <session_id> <program> [args...]"
            );
            std::process::exit(2);
        }
    };
    let command: Vec<String> = args.collect();
    if command.is_empty() {
        eprintln!("terminal-daemon: missing program to run");
        std::process::exit(2);
    }
    terminal_daemon::daemon::run(session_id, command);
}
