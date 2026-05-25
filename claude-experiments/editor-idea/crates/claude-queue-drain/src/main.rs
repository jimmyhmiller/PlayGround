//! Noop drainer for `~/.claude/events.jsonl`.
//!
//! Tails the file from a persistent cursor and parses each line into
//! a typed `Event`. Today it does nothing with parsed events except
//! log them at debug level. That's intentional — the cursor + parser
//! are the parts that need to be right; deciding what to *do* with
//! events comes later (rules, weak models, or a full agent).
//!
//! ## Cursor
//!
//! Persisted at `~/.claude/.queue-cursor`. A single JSON object:
//! `{"byte_offset": <u64>, "inode": <u64>}`. Inode is stored so a
//! log-rotation (file deleted + recreated with same path) is detected
//! and the cursor resets to 0 instead of pointing into the void.
//!
//! ## Multiple consumers
//!
//! Each consumer should use its own cursor file (pass `--cursor
//! ~/.claude/.queue-cursor.foo`). The events.jsonl itself is
//! append-only, so any number of cursors can advance independently.
//!
//! ## Future shape
//!
//! When we add real handlers, the dispatch point is the `handle()`
//! function near the bottom. Add a `match ev.kind` and route each
//! variant. Keep this binary single-threaded and side-effect-free
//! beyond reading the log — anything that wants to take action
//! (inject, notify, kill) should spawn its own subprocess so a
//! handler bug can't wedge the drainer.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

const POLL_INTERVAL: Duration = Duration::from_millis(250);
const DEFAULT_CURSOR: &str = ".queue-cursor";
const EVENTS_FILENAME: &str = "events.jsonl";

#[derive(Debug, Deserialize)]
struct Event {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    ts: u64,
    #[serde(default)]
    terminal_session_id: String,
    #[serde(default)]
    claude_pid: u32,
    #[serde(default)]
    payload: Value,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Cursor {
    #[serde(default)]
    byte_offset: u64,
    /// Inode of the file we were reading. If it changes, the file was
    /// rotated/recreated and `byte_offset` no longer applies.
    #[serde(default)]
    inode: u64,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME unset"))
}

fn events_path() -> PathBuf {
    let mut p = home();
    p.push(".claude");
    p.push(EVENTS_FILENAME);
    p
}

fn default_cursor_path() -> PathBuf {
    let mut p = home();
    p.push(".claude");
    p.push(DEFAULT_CURSOR);
    p
}

fn load_cursor(path: &std::path::Path) -> Cursor {
    let Ok(bytes) = std::fs::read(path) else {
        return Cursor::default();
    };
    serde_json::from_slice(&bytes).unwrap_or_default()
}

fn save_cursor(path: &std::path::Path, cursor: &Cursor) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec(cursor)?;
    // Tmp + rename so we never observe a half-written cursor file.
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

fn handle(event: &Event) {
    // The noop. Print to stderr so it's visible when running
    // foreground but never pollutes stdout (which a future
    // supervisor might use for IPC).
    eprintln!(
        "[drain] {} pid={} term={} payload_keys={:?}",
        event.kind,
        event.claude_pid,
        if event.terminal_session_id.is_empty() {
            "-"
        } else {
            event.terminal_session_id.as_str()
        },
        event
            .payload
            .as_object()
            .map(|o| o.keys().take(6).cloned().collect::<Vec<_>>())
            .unwrap_or_default(),
    );
    let _ = event.ts; // unused for now; reserved for future ordering checks
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut cursor_path = default_cursor_path();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--cursor" => {
                if let Some(v) = args.next() {
                    cursor_path = PathBuf::from(v);
                }
            }
            "-h" | "--help" => {
                eprintln!(
                    "claude-queue-drain [--cursor PATH]\n\
                     \n\
                     Tail ~/.claude/events.jsonl from a persistent cursor and\n\
                     parse each event. Currently logs and discards. Each\n\
                     consumer should pass a unique --cursor path."
                );
                return;
            }
            other => {
                eprintln!("[drain] unknown flag: {}", other);
                std::process::exit(2);
            }
        }
    }

    let events = events_path();
    let mut cursor = load_cursor(&cursor_path);
    eprintln!(
        "[drain] events={} cursor={} start_offset={} start_inode={}",
        events.display(),
        cursor_path.display(),
        cursor.byte_offset,
        cursor.inode
    );

    loop {
        if let Err(e) = drain_once(&events, &mut cursor) {
            eprintln!("[drain] {}", e);
        }
        if let Err(e) = save_cursor(&cursor_path, &cursor) {
            eprintln!("[drain] save cursor: {}", e);
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

fn drain_once(events: &std::path::Path, cursor: &mut Cursor) -> std::io::Result<()> {
    let Ok(meta) = std::fs::metadata(events) else {
        // File doesn't exist yet — nothing to do. Don't reset cursor;
        // the file may reappear later (Claude Code not yet started).
        return Ok(());
    };
    let current_inode = meta.ino();

    // Detect rotation: same path, different inode → previous file
    // gone, start fresh from byte 0 of the new one.
    if cursor.inode != 0 && cursor.inode != current_inode {
        eprintln!(
            "[drain] inode changed ({} → {}), resetting cursor",
            cursor.inode, current_inode
        );
        cursor.byte_offset = 0;
    }
    cursor.inode = current_inode;

    // Same-file truncation (`> events.jsonl`) — file is shorter than
    // where we left off. Reset to start of file.
    if meta.len() < cursor.byte_offset {
        eprintln!(
            "[drain] file truncated ({} < {}), resetting cursor",
            meta.len(),
            cursor.byte_offset
        );
        cursor.byte_offset = 0;
    }

    if meta.len() == cursor.byte_offset {
        return Ok(()); // No new bytes.
    }

    let mut file = OpenOptions::new().read(true).open(events)?;
    file.seek(SeekFrom::Start(cursor.byte_offset))?;
    let mut reader = BufReader::new(file);

    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        // Only advance the cursor when we've fully consumed a line —
        // if the producer is mid-write and we read a partial line, we
        // bail and try again next tick. `read_line` returns when it
        // hits \n; if the file ends mid-line the trailing data is
        // returned without \n, which we detect and skip.
        if !buf.ends_with('\n') {
            break;
        }
        cursor.byte_offset += n as u64;

        let trimmed = buf.trim_end_matches(['\n', '\r']);
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<Event>(trimmed) {
            Ok(ev) => handle(&ev),
            Err(e) => eprintln!("[drain] parse error at byte {}: {}", cursor.byte_offset, e),
        }
    }
    Ok(())
}
