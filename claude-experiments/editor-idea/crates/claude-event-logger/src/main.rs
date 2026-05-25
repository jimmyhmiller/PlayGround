//! Event logger for Claude Code hooks.
//!
//! Wire as the command for every hook in `~/.claude/settings.json`:
//!
//! ```json
//! "hooks": {
//!   "SessionStart":     [{"hooks": [{"type":"command","command":"/path/to/claude-event-logger session_start"}]}],
//!   ...
//! }
//! ```
//!
//! Stdin is Claude Code's hook payload (a JSON object). We wrap it in
//! `{kind, ts, terminal_session_id, claude_pid, payload}` and ship it
//! to the central `claude-bus` daemon over its Unix socket.
//!
//! Fast path (daemon up): connect → write Hello + Publish frames →
//! close. Total cost a couple of syscalls; the daemon does the JSONL
//! append on its side, plus broadcasting to live subscribers.
//!
//! Fallback (daemon down): write the same envelope directly to
//! `~/.claude/events.jsonl` via O_APPEND, exactly like the original
//! logger did. Multi-writer atomicity comes from the kernel's PIPE_BUF
//! guarantee for sub-4KB writes — we shrink payloads to stay inside it.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;

use claude_bus::proto::{encode, ClientFrame, Role};

const MAX_PAYLOAD_BYTES: usize = 3072;

/// Outer envelope for the JSONL fallback. Kept identical to the format
/// the bus daemon writes when it succeeds, so consumers (drainer, raw
/// tail) don't care which path the event took.
#[derive(Serialize)]
struct Envelope<'a> {
    kind: &'a str,
    ts: u64,
    terminal_session_id: &'a str,
    claude_pid: u32,
    payload: &'a Value,
}

fn read_payload() -> Value {
    let mut buf = String::new();
    if std::io::stdin().read_to_string(&mut buf).is_err() {
        return Value::Null;
    }
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        return Value::Null;
    }
    serde_json::from_str(trimmed).unwrap_or(Value::Null)
}

/// Cap the payload's serialized size by stripping known-large fields
/// when we'd otherwise overflow the atomicity budget. Today the only
/// realistic offender is the full text of a `UserPromptSubmit`; we
/// drop it to a short prefix when needed. Other hooks fit comfortably.
fn shrink_payload(mut payload: Value, headroom: usize) -> Value {
    let serialized_len = serde_json::to_string(&payload).map(|s| s.len()).unwrap_or(0);
    if serialized_len <= headroom {
        return payload;
    }
    if let Some(obj) = payload.as_object_mut() {
        for key in ["prompt", "text", "content", "tool_input"] {
            if let Some(v) = obj.get_mut(key)
                && let Some(s) = v.as_str()
                && s.len() > 256
            {
                *v = Value::String(format!("{}…[truncated]", &s[..256]));
            }
        }
    }
    payload
}

/// Try to publish to the bus. Returns `true` on success. Caller uses the
/// JSONL fallback on `false`.
///
/// We *don't* wait for any acknowledgement — the protocol is
/// fire-and-forget on the publish side. As long as the kernel accepts
/// our write into the socket's buffer, the daemon will read it on its
/// next poll cycle.
fn try_publish_to_bus(
    socket: &Path,
    kind: &str,
    ts: u64,
    terminal_session_id: &str,
    claude_pid: u32,
    payload_json: &str,
) -> bool {
    if !socket.exists() {
        return false;
    }
    let mut s = match UnixStream::connect(socket) {
        Ok(s) => s,
        Err(_) => return false,
    };
    // Bound the worst case if the daemon is wedged. Hook latency is
    // visible to the user, so we'd rather fall back to JSONL than
    // stall for seconds.
    let _ = s.set_write_timeout(Some(Duration::from_millis(200)));

    let hello = match encode(&ClientFrame::Hello { role: Role::Publisher }) {
        Ok(b) => b,
        Err(_) => return false,
    };
    let publish = match encode(&ClientFrame::Publish {
        kind: kind.to_string(),
        ts,
        terminal_session_id: terminal_session_id.to_string(),
        claude_pid,
        payload_json: payload_json.to_string(),
    }) {
        Ok(b) => b,
        Err(_) => return false,
    };
    if s.write_all(&hello).is_err() {
        return false;
    }
    if s.write_all(&publish).is_err() {
        return false;
    }
    // Drop drops the socket cleanly; the kernel still delivers the
    // buffered bytes to the daemon. No need to wait.
    true
}

fn fallback_append_jsonl(envelope: &Envelope) -> std::io::Result<()> {
    let Some(path) = claude_bus::events_path() else {
        return Err(std::io::Error::other("no HOME"));
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut line = serde_json::to_string(envelope).map_err(std::io::Error::other)?;
    line.push('\n');
    // O_APPEND guarantees the write goes at EOF atomically relative to
    // other O_APPEND writers (the daemon, other hook fires), as long as
    // the buffer is under PIPE_BUF. We shrink payloads above to stay
    // there.
    let mut f = OpenOptions::new()
        .append(true)
        .create(true)
        .mode(0o600)
        .open(&path)?;
    f.write_all(line.as_bytes())?;
    Ok(())
}

fn main() {
    let kind = std::env::args().nth(1).unwrap_or_else(|| "unknown".into());

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let claude_pid = unsafe { libc::getppid() } as u32;
    let terminal_session_id =
        std::env::var("EDITOR_IDEA_TERMINAL_SESSION_ID").unwrap_or_default();

    let payload = read_payload();
    let payload = shrink_payload(payload, MAX_PAYLOAD_BYTES);
    let payload_json = serde_json::to_string(&payload).unwrap_or_else(|_| "null".into());

    let bus_socket = claude_bus::socket_path();

    let published = if let Some(socket) = bus_socket.as_deref() {
        try_publish_to_bus(
            socket,
            &kind,
            ts,
            &terminal_session_id,
            claude_pid,
            &payload_json,
        )
    } else {
        false
    };

    if !published {
        let envelope = Envelope {
            kind: &kind,
            ts,
            terminal_session_id: &terminal_session_id,
            claude_pid,
            payload: &payload,
        };
        if let Err(e) = fallback_append_jsonl(&envelope) {
            eprintln!("[event-logger] fallback append: {}", e);
        }
    }
}
