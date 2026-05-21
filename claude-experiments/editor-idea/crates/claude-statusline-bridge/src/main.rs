//! Claude Code statusLine bridge.
//!
//! Wire into `~/.claude/settings.json`:
//!
//! ```json
//! { "statusLine": { "type": "command",
//!                   "command": "/path/to/claude-statusline-bridge" } }
//! ```
//!
//! Claude Code pipes a JSON object to stdin every ~300ms with the
//! current session info: `{session_id, transcript_path, cwd, model,
//! workspace, ...}`. We compute context-percent by reading the
//! transcript JSONL and summing the last assistant message's usage
//! tokens, divide by the model's context window, atomically write
//! `~/.claude/statusline-feed/<session_id>.json`, and print a one-line
//! status string back to Claude Code's UI.
//!
//! The output file is what the `claude-context-bars` widget polls.

use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ---------- Claude Code → us ----------

/// Subset of the statusLine input JSON that we care about. Unknown
/// fields are ignored (`serde(default)` everywhere) so future Claude
/// Code changes don't break us.
#[derive(Deserialize, Debug)]
struct StatusInput {
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    transcript_path: String,
    #[serde(default)]
    cwd: String,
    #[serde(default)]
    model: ModelInfo,
}

#[derive(Deserialize, Debug, Default)]
struct ModelInfo {
    #[serde(default)]
    id: String,
    #[serde(default)]
    display_name: String,
}

// ---------- us → widget ----------

#[derive(Serialize)]
struct FeedEntry<'a> {
    claude_session_id: &'a str,
    /// `EDITOR_IDEA_TERMINAL_SESSION_ID` from env. Empty when the
    /// bridge runs outside one of our terminals — the widget treats
    /// such entries as orphans.
    terminal_session_id: String,
    /// PID of the Claude Code process that invoked us (i.e. our
    /// parent). The widget probes this with `kill(pid, 0)` to drop
    /// entries whose Claude Code session has exited — far more
    /// reliable than waiting for the file's `ts` to age out, since
    /// the bridge has no exit hook to clean up after itself.
    claude_pid: u32,
    cwd: &'a str,
    model_id: &'a str,
    model_display: &'a str,
    /// Fill ratio in 0..=1. 0 when the transcript hasn't started.
    context_pct: f32,
    /// Raw token count we divided by `context_window` to get
    /// `context_pct` — useful for debugging.
    tokens: u64,
    context_window: u64,
    /// Seconds since UNIX epoch. Used as a coarse fallback when the
    /// PID happens to get reused by an unrelated process.
    ts: u64,
}

// ---------- Transcript parsing ----------

/// Minimum useful structure of a transcript line. The transcript is
/// JSONL where each line is a message; `message.usage` is present on
/// assistant turns from the API.
#[derive(Deserialize, Default)]
struct TranscriptLine {
    #[serde(default)]
    message: Option<TranscriptMessage>,
}

#[derive(Deserialize, Default)]
struct TranscriptMessage {
    #[serde(default)]
    role: String,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize, Default)]
struct Usage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
}

/// Walk the transcript backwards looking for the most recent assistant
/// message that carries a `usage` block. That gives the current
/// in-context token count (cached + uncached inputs the model just saw).
/// Returns 0 if no usage was found.
fn latest_input_tokens(path: &str) -> u64 {
    let Ok(text) = fs::read_to_string(path) else {
        return 0;
    };
    for line in text.lines().rev() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(parsed) = serde_json::from_str::<TranscriptLine>(line) else {
            continue;
        };
        let Some(msg) = parsed.message else { continue };
        if msg.role != "assistant" {
            continue;
        }
        let Some(u) = msg.usage else { continue };
        let total = u.input_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens;
        if total > 0 {
            return total;
        }
    }
    0
}

/// Map a model id to its context window in tokens. Conservative
/// defaults — when in doubt, assume 200k so the bar fills sooner
/// rather than masking a near-overflow.
fn context_window_for(model_id: &str) -> u64 {
    // Match on substrings so `claude-opus-4-7[1m]` and friends all
    // resolve correctly regardless of vendor-prefix variations.
    let id = model_id.to_ascii_lowercase();
    if id.contains("[1m]") || id.contains("-1m") {
        return 1_000_000;
    }
    if id.contains("haiku") {
        return 200_000;
    }
    if id.contains("sonnet") {
        return 200_000;
    }
    if id.contains("opus") {
        return 200_000;
    }
    200_000
}

// ---------- Feed file ----------

fn feed_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".claude");
    p.push("statusline-feed");
    Some(p)
}

fn write_feed(entry: &FeedEntry<'_>) -> std::io::Result<()> {
    let Some(dir) = feed_dir() else {
        return Err(std::io::Error::other("no HOME"));
    };
    fs::create_dir_all(&dir)?;
    let file = dir.join(format!("{}.json", entry.claude_session_id));
    let tmp = dir.join(format!("{}.json.tmp", entry.claude_session_id));
    let bytes = serde_json::to_vec(entry)?;
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, &file)
}

// ---------- main ----------

fn main() {
    // Read everything from stdin — Claude Code closes stdin after
    // delivering the JSON object.
    let mut buf = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
        eprintln!("[bridge] read stdin: {}", e);
        std::process::exit(1);
    }

    let input: StatusInput = match serde_json::from_str(&buf) {
        Ok(v) => v,
        Err(e) => {
            // Don't fail the statusLine — Claude Code falls back to a
            // blank one. Log to stderr so the user can debug.
            eprintln!("[bridge] parse stdin: {}", e);
            std::process::exit(0);
        }
    };

    let tokens = if input.transcript_path.is_empty() {
        0
    } else {
        latest_input_tokens(&input.transcript_path)
    };
    let context_window = context_window_for(&input.model.id);
    let pct = if context_window == 0 {
        0.0
    } else {
        (tokens as f32 / context_window as f32).clamp(0.0, 1.0)
    };

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let terminal_session_id =
        std::env::var("EDITOR_IDEA_TERMINAL_SESSION_ID").unwrap_or_default();

    // Claude Code invokes us as its direct child, so getppid() is
    // Claude Code's PID. Safe everywhere we run (unix only).
    let claude_pid = unsafe { libc::getppid() } as u32;

    let entry = FeedEntry {
        claude_session_id: &input.session_id,
        terminal_session_id,
        claude_pid,
        cwd: &input.cwd,
        model_id: &input.model.id,
        model_display: &input.model.display_name,
        context_pct: pct,
        tokens,
        context_window,
        ts: now,
    };

    if let Err(e) = write_feed(&entry) {
        eprintln!("[bridge] write feed: {}", e);
    }

    // Status line text — keep it terse so it fits a wide range of
    // terminal widths. Claude Code already prefixes its own info.
    let pct_int = (pct * 100.0).round() as u32;
    let display = if input.model.display_name.is_empty() {
        &input.model.id
    } else {
        &input.model.display_name
    };
    println!("{}  ctx {}%", display, pct_int);
}
