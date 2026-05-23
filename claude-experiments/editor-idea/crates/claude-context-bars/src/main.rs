//! `claude-context-bars` — widget process for editor-idea.
//!
//! Renders a bar chart of context-% for every Claude Code session
//! currently open in the editor, grouped by project. Wire it into a
//! widget pane with `command = claude-context-bars`.
//!
//! Data sources (both polled, no fs watcher):
//! - `~/.claude/statusline-feed/*.json` — written by the bridge once
//!   per Claude Code statusLine tick. Each file is one session.
//!   Entries older than `STALE_SECS` are dropped (Claude Code session
//!   stopped publishing → session ended).
//! - `~/.terminal-bevy/terminals.json` — written by the host whenever
//!   pane state changes. Tells us project name + pane title for each
//!   terminal session id.
//!
//! Join key is `terminal_session_id` (set by the host on the shell's
//! env so it propagates through Claude Code into the bridge). A feed
//! entry with no matching terminal is treated as orphaned and skipped.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use widget_bevy::protocol::{Align, Element, HostEvent, Weight, WidgetMsg};

/// How often we redraw. Claude Code's statusLine fires every ~300ms,
/// so 500ms gives us at-most-one-frame lag on changes without spinning.
const TICK: Duration = Duration::from_millis(500);

/// Backup staleness window for entries written by older bridge
/// versions that didn't stamp a PID. With PID-liveness in place,
/// truly-dead sessions are dropped instantly via `kill(pid, 0)`, so
/// this only catches the "no-pid" legacy case.
const STALE_SECS: u64 = 60;

// ---------- Data shapes ----------

#[derive(Deserialize, Debug, Clone)]
struct FeedEntry {
    #[serde(default)]
    terminal_session_id: String,
    #[serde(default)]
    cwd: String,
    #[serde(default)]
    context_pct: f32,
    #[serde(default)]
    ts: u64,
    /// PID of the Claude Code session that wrote this entry. Older
    /// bridge versions omit it; treat 0 as "unknown" and fall back to
    /// the `ts` staleness check.
    #[serde(default)]
    claude_pid: u32,
}

fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return true;
    }
    // `kill(pid, 0)` is the canonical "does this pid exist?" probe
    // on Unix — no signal is delivered, the only effect is errno.
    // ESRCH = gone; EPERM = exists but we're not allowed (still alive).
    let r = unsafe { libc::kill(pid as i32, 0) };
    if r == 0 {
        return true;
    }
    let err = std::io::Error::last_os_error().raw_os_error();
    err != Some(libc::ESRCH)
}

#[derive(Deserialize, Debug, Clone)]
struct LiveTerminals {
    #[serde(default)]
    terminals: Vec<LiveTerminalEntry>,
}

#[derive(Deserialize, Debug, Clone)]
struct LiveTerminalEntry {
    session_id: u64,
    #[serde(default)]
    project_name: String,
}

// ---------- Paths ----------

fn feed_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".claude");
    p.push("statusline-feed");
    Some(p)
}

fn terminals_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".terminal-bevy");
    p.push("terminals.json");
    Some(p)
}

// ---------- Loaders ----------

fn load_feed() -> Vec<FeedEntry> {
    let Some(dir) = feed_dir() else { return vec![] };
    let Ok(rd) = fs::read_dir(&dir) else {
        return vec![];
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut out = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else { continue };
        let Ok(parsed) = serde_json::from_slice::<FeedEntry>(&bytes) else {
            continue;
        };
        // PID liveness is the source of truth: an alive Claude Code
        // process means the session is still open even if it's idle
        // (statusLine stops firing during idle, so `ts` would age
        // misleadingly fast). Only fall back to `ts` for legacy
        // entries that don't carry a pid.
        if parsed.claude_pid != 0 {
            if !pid_alive(parsed.claude_pid) {
                let _ = fs::remove_file(&path);
                continue;
            }
        } else if parsed.ts == 0 || now.saturating_sub(parsed.ts) > STALE_SECS {
            continue;
        }
        out.push(parsed);
    }
    out
}

fn load_terminals() -> Vec<LiveTerminalEntry> {
    let Some(path) = terminals_path() else { return vec![] };
    let Ok(bytes) = fs::read(&path) else { return vec![] };
    serde_json::from_slice::<LiveTerminals>(&bytes)
        .map(|t| t.terminals)
        .unwrap_or_default()
}

// ---------- Frame building ----------

struct Row {
    pct: f32,
}

/// Pick a color for a bar based on its fill ratio: green → amber →
/// red so a glance tells the user which sessions are running out.
fn bar_color(pct: f32) -> &'static str {
    if pct >= 0.85 {
        "#e07a5f"
    } else if pct >= 0.6 {
        "#e0c45f"
    } else {
        "#5fa1ff"
    }
}

fn build_frame(rows_by_project: &BTreeMap<String, Vec<Row>>, content_w: f32) -> Element {
    if rows_by_project.is_empty() {
        return Element::Vstack {
            gap: 6.0,
            pad: 12.0,
            children: vec![
                Element::Text {
                    value: "No Claude Code sessions".into(),
                    color: Some("#cc8".into()),
                    size: Some(13.0),
                    weight: Some(Weight::Bold),
                },
                Element::Text {
                    value: "Configure ~/.claude/settings.json statusLine to claude-statusline-bridge".into(),
                    color: Some("#888".into()),
                    size: Some(11.0),
                    weight: None,
                },
            ],
        };
    }

    // Bar spans the full content width minus the outer pad on both
    // sides, the row's gap, and a small reservation for the trailing
    // "NN%" label.
    const OUTER_PAD: f32 = 12.0;
    const ROW_GAP: f32 = 8.0;
    const PCT_LABEL_W: f32 = 36.0;
    let bar_w = (content_w - OUTER_PAD * 2.0 - ROW_GAP - PCT_LABEL_W).max(40.0);

    let mut children: Vec<Element> = Vec::new();
    let mut first = true;
    for (project, rows) in rows_by_project {
        if !first {
            children.push(Element::Spacer { size: 6.0 });
            children.push(Element::Divider);
        }
        first = false;

        let project_label = if project.is_empty() {
            "(no project)".to_string()
        } else {
            project.clone()
        };
        children.push(Element::Text {
            value: project_label,
            color: Some("#b8bcc4".into()),
            size: Some(12.0),
            weight: Some(Weight::Bold),
        });

        for row in rows {
            let pct_int = (row.pct * 100.0).round() as u32;
            let pct_label = format!("{:>3}%", pct_int);
            let row_children: Vec<Element> = vec![
                Element::Bar {
                    value: row.pct,
                    max: 1.0,
                    color: Some(bar_color(row.pct).into()),
                    track: Some("#1c1f25".into()),
                    width: bar_w,
                    height: 10.0,
                },
                Element::Text {
                    value: pct_label,
                    color: Some("#aab0b8".into()),
                    size: Some(11.0),
                    weight: None,
                },
            ];
            children.push(Element::Hstack {
                gap: 8.0,
                pad: 0.0,
                align: Align::Center,
                children: row_children,
            });
        }
    }

    Element::Scroll {
        gap: 4.0,
        pad: 12.0,
        children,
    }
}

/// Join feed entries against the host's live terminal panes. Only
/// sessions that correspond to a currently-open terminal-bevy pane
/// show up — entries with no `terminal_session_id` (Claude running
/// outside the app) or with an ID whose pane has been closed are
/// dropped. PID-liveness alone is too permissive: `claude` processes
/// orphaned by a closed pane keep running for a long time and would
/// otherwise haunt the bars.
fn build_rows(
    feed: &[FeedEntry],
    terminals: &[LiveTerminalEntry],
) -> BTreeMap<String, Vec<Row>> {
    let mut by_id: std::collections::HashMap<String, &LiveTerminalEntry> =
        std::collections::HashMap::new();
    for t in terminals {
        by_id.insert(t.session_id.to_string(), t);
    }

    let mut groups: BTreeMap<String, Vec<Row>> = BTreeMap::new();
    for f in feed {
        if f.terminal_session_id.is_empty() {
            continue;
        }
        let Some(term) = by_id.get(f.terminal_session_id.as_str()).copied() else {
            continue;
        };

        groups
            .entry(term.project_name.clone())
            .or_default()
            .push(Row { pct: f.context_pct });
    }

    for rows in groups.values_mut() {
        rows.sort_by(|a, b| {
            b.pct
                .partial_cmp(&a.pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    groups
}

// ---------- main loop ----------

fn emit(line: &str) {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    if writeln!(lock, "{}", line).is_err() {
        std::process::exit(0);
    }
    let _ = lock.flush();
}

fn main() {
    // Spawn a stdin reader on a background thread so blocking on
    // host events doesn't stall the redraw loop. We only act on
    // init/resize/close — clicks aren't bound to anything yet.
    let (tx, rx) = std::sync::mpsc::channel::<HostEvent>();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let reader = BufReader::new(stdin.lock());
        for line in reader.lines() {
            let Ok(s) = line else { return };
            if s.trim().is_empty() {
                continue;
            }
            let Ok(ev) = serde_json::from_str::<HostEvent>(&s) else {
                continue;
            };
            if tx.send(ev).is_err() {
                return;
            }
        }
    });

    // Start with a reasonable default content width so the first
    // frame looks right before `init` arrives.
    let mut content_w: f32 = 360.0;
    let mut next_tick = Instant::now();
    let mut last_frame_hash: u64 = 0;
    // Once we've ever drawn a non-empty frame, never fall back to the
    // "no sessions configured" placeholder. Idle Claude Code sessions
    // stop publishing every few minutes; flashing the empty state in
    // those gaps is worse than showing slightly stale bars.
    let mut have_drawn_data = false;

    loop {
        // Drain whatever host events are ready without blocking.
        loop {
            match rx.try_recv() {
                Ok(HostEvent::Init { width, .. }) => content_w = width,
                Ok(HostEvent::Resize { width, .. }) => content_w = width,
                Ok(HostEvent::Refresh) => last_frame_hash = 0,
                Ok(HostEvent::Close) => std::process::exit(0),
                Ok(HostEvent::Click { .. }) => {}
                Ok(HostEvent::Tick { .. }) => {}
                Ok(HostEvent::ClaudeEvent { .. }) => {}
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => std::process::exit(0),
            }
        }

        let feed = load_feed();
        let terminals = load_terminals();
        let rows = build_rows(&feed, &terminals);

        if rows.is_empty() && have_drawn_data {
            // Skip this tick — keep showing the previous (good) frame
            // instead of flashing the placeholder.
            schedule_next(&mut next_tick);
            continue;
        }
        if !rows.is_empty() {
            have_drawn_data = true;
        }

        let frame = build_frame(&rows, content_w);

        // Hash to suppress redundant frames — keeps the host's render
        // path from re-spawning sprites every tick.
        let serialized = serde_json::to_string(&WidgetMsg::Frame {
            root: frame.clone(),
        })
        .expect("serialize frame");
        let h = fast_hash(&serialized);
        if h != last_frame_hash {
            last_frame_hash = h;
            emit(&serialized);
        }

        schedule_next(&mut next_tick);
    }
}

fn schedule_next(next_tick: &mut Instant) {
    *next_tick += TICK;
    let now = Instant::now();
    if *next_tick > now {
        std::thread::sleep(*next_tick - now);
    } else {
        *next_tick = now + TICK;
    }
}

fn fast_hash(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}
