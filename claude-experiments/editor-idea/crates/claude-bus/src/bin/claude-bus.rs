//! `claude-bus` daemon entry point.
//!
//! Set `CLAUDE_BUS_FOREGROUND=1` to skip daemonization (useful for
//! launchd, tests, and `tail -f` debugging). `CLAUDE_BUS_LOG=/path`
//! routes stderr to a file when daemonized.

fn main() {
    claude_bus::daemon::daemonize_if_requested();

    let socket = match claude_bus::socket_path() {
        Some(p) => p,
        None => {
            eprintln!("[claude-bus] HOME not set; refusing to start");
            std::process::exit(1);
        }
    };
    let jsonl = match claude_bus::events_path() {
        Some(p) => p,
        None => {
            eprintln!("[claude-bus] HOME not set; refusing to start");
            std::process::exit(1);
        }
    };
    let pid_path = match claude_bus::pid_path() {
        Some(p) => p,
        None => {
            eprintln!("[claude-bus] HOME not set; refusing to start");
            std::process::exit(1);
        }
    };

    if let Err(e) = claude_bus::daemon::run(socket, jsonl, pid_path) {
        eprintln!("[claude-bus] fatal: {}", e);
        std::process::exit(1);
    }
}
