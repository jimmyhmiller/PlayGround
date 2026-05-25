//! `claude-bus-tail` — live-watch the bus.
//!
//! Connects as a subscriber, prints each event as a single JSON line on
//! stdout, and stays connected forever (reconnecting on bus restart).
//! Useful for ad-hoc observability and as a worked example of the
//! client API.
//!
//! Usage:
//!   claude-bus-tail               # follow live, no replay
//!   claude-bus-tail --since 1000  # replay from seq 1000 if still buffered

use std::io::Write;

use claude_bus::client::{BusItem, Subscriber};

fn parse_since(args: &[String]) -> Option<u64> {
    let mut iter = args.iter().skip(1);
    while let Some(arg) = iter.next() {
        if arg == "--since" {
            return iter.next().and_then(|v| v.parse().ok());
        }
        if let Some(rest) = arg.strip_prefix("--since=") {
            return rest.parse().ok();
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let since = parse_since(&args);

    let socket = match claude_bus::socket_path() {
        Some(p) => p,
        None => {
            eprintln!("claude-bus-tail: HOME not set");
            std::process::exit(1);
        }
    };
    if !socket.exists() {
        eprintln!(
            "claude-bus-tail: {} does not exist (is claude-bus running?)",
            socket.display()
        );
        // Don't exit — the worker will retry, and the bus may come up
        // momentarily. The user can ^C if they don't want to wait.
    }

    let sub = Subscriber::spawn(socket, since);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    loop {
        match sub.recv_timeout(std::time::Duration::from_secs(60)) {
            Some(BusItem::Event(ev)) => {
                // Re-stitch into the same on-disk envelope shape so
                // operators can pipe this into the same tools that
                // read events.jsonl.
                let line = format!(
                    "{{\"seq\":{},\"kind\":\"{}\",\"ts\":{},\"terminal_session_id\":\"{}\",\"claude_pid\":{},\"payload\":{}}}",
                    ev.seq,
                    escape(&ev.kind),
                    ev.ts,
                    escape(&ev.terminal_session_id),
                    ev.claude_pid,
                    if ev.payload_json.is_empty() { "null" } else { &ev.payload_json },
                );
                let _ = writeln!(out, "{}", line);
                let _ = out.flush();
            }
            Some(BusItem::Gap {
                last_delivered_seq,
                replay_from,
            }) => {
                let _ = writeln!(
                    out,
                    "{{\"_meta\":\"gap\",\"last_delivered_seq\":{:?},\"replay_from\":{}}}",
                    last_delivered_seq, replay_from
                );
                let _ = out.flush();
            }
            Some(BusItem::Disconnected) => {
                let _ = writeln!(out, "{{\"_meta\":\"disconnected\"}}");
                let _ = out.flush();
            }
            Some(BusItem::Reconnected) => {
                let _ = writeln!(out, "{{\"_meta\":\"reconnected\"}}");
                let _ = out.flush();
            }
            None => continue,
        }
    }
}

/// Minimal JSON string escape. We control `kind` and
/// `terminal_session_id` from elsewhere in the codebase, so the only
/// realistic specials are `"` and `\`.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
