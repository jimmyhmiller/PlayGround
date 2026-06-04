//! Per-terminal worker thread that owns the libghostty `Terminal`,
//! drives `vt_write` from the pty, and publishes a snapshot for the
//! Bevy main thread to render. This is the single biggest performance
//! lever in the crate — without it, `vt_write` runs on the render
//! thread and a multi-MiB `cat` blocks the frame.
//!
//! Architecture (Alacritty-style):
//!
//! - **Worker thread** owns `Terminal`, `RenderState`, the pty fd, and
//!   does all VT processing. It loops: drain pty (bounded, non-blocking),
//!   feed bytes to `vt_write`, drain pending input messages from main,
//!   take a `GridSnapshot`, briefly lock + publish.
//! - **Main thread** reads `Arc<Mutex<GridSnapshot>>` once per frame to
//!   render. Sends keystroke / resize / shutdown via an `mpsc` channel.
//! - The `!Send` libghostty `Terminal` never moves off the worker.
//!   `GridSnapshot` is plain POD (chars + RGB + bools) so it crosses
//!   the thread boundary as `Send + Sync`.
//!
//! Bounded read-per-tick (`MAX_READ_PER_TICK`) is the same trick
//! Alacritty uses (`MAX_LOCKED_READ`) — it caps how long the worker
//! holds the snapshot mutex without releasing, so the renderer never
//! waits more than a few KiB worth of parsing for the lock.

use std::fs::{File, OpenOptions};
use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::os::fd::{AsFd, OwnedFd};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use bevy::winit::{EventLoopProxy, WinitUserEvent};
use libghostty_vt::{
    paste,
    render::{CellIterator, Dirty, RenderState, RowIterator},
    style::RgbColor,
    terminal::{Mode, Point, PointCoordinate, ScrollViewport},
};
use nix::errno::Errno;
use nix::fcntl::{self, OFlag};
use nix::poll::{poll, PollFd, PollFlags, PollTimeout};

use crate::daemon_client::DaemonClient;

// ---------- Process-global wake throttle ----------

/// Maximum rate at which any worker may wake the Bevy event loop. We
/// share this across every terminal worker so N busy terminals can't
/// multiply the redraw rate (the previous per-worker throttle gave us
/// up to N × 60Hz wakes for N terminals).
///
/// A `cat bigfile` produces hundreds of pty chunks/sec; the display
/// can show maybe 60 frames/sec of them. Coalescing here turns every
/// chunk-driven wake inside the cooldown into a no-op — the snapshot
/// the worker just published will be picked up by the next scheduled
/// frame anyway.
const MIN_WAKE_INTERVAL: Duration = Duration::from_millis(16);

static WAKE_THROTTLE: std::sync::OnceLock<Mutex<Instant>> = std::sync::OnceLock::new();

fn wake_throttle() -> &'static Mutex<Instant> {
    WAKE_THROTTLE.get_or_init(|| Mutex::new(Instant::now() - Duration::from_secs(1)))
}

/// Send a `WinitUserEvent::WakeUp` iff the global cooldown has expired.
/// Returns `true` on success, `false` if suppressed — the caller is
/// expected to remember and retry later.
fn try_wake_winit_throttled(wakeup: &Option<EventLoopProxy<WinitUserEvent>>) -> bool {
    let mut last = wake_throttle().lock().expect("wake throttle poisoned");
    if last.elapsed() < MIN_WAKE_INTERVAL {
        return false;
    }
    *last = Instant::now();
    drop(last);
    if let Some(p) = wakeup.as_ref() {
        let _ = p.send_event(WinitUserEvent::WakeUp);
    }
    true
}

/// Milliseconds remaining until the next wake slot opens. Used by the
/// worker's `poll(2)` loop to size its timeout so the deferred wake
/// fires on time.
fn wake_cooldown_remaining_ms() -> u64 {
    let last = wake_throttle().lock().expect("wake throttle poisoned");
    let elapsed = last.elapsed();
    if elapsed >= MIN_WAKE_INTERVAL {
        0
    } else {
        (MIN_WAKE_INTERVAL - elapsed).as_millis() as u64 + 1
    }
}
use crate::daemon_proto::{ClientMessage, DaemonMessage};
use crate::pty::PtySize;
use crate::vt::{self, CellPx};

/// One grid cell — what the renderer actually needs. Plain POD so the
/// snapshot is `Send + Sync` and crosses thread boundaries cheaply.
#[derive(Clone, Copy, Debug)]
pub struct SnapCell {
    pub ch: char,
    pub fg: RgbColor,
    pub bg: RgbColor,
    pub inverse: bool,
}

impl Default for SnapCell {
    fn default() -> Self {
        Self {
            ch: ' ',
            fg: RgbColor {
                r: 220,
                g: 220,
                b: 220,
            },
            bg: RgbColor {
                r: 13,
                g: 15,
                b: 20,
            },
            inverse: false,
        }
    }
}

/// What the worker hands to the renderer. Reused (not reallocated) on
/// each update to avoid allocator churn under heavy output.
#[derive(Default, Debug)]
pub struct GridSnapshot {
    pub cols: u16,
    pub rows: u16,
    /// `cols * rows` cells, row-major.
    pub cells: Vec<SnapCell>,
    /// `rows` flags. `true` if the row was rewritten in this update —
    /// renderer can skip per-cell sprite mutation when false.
    pub dirty_rows: Vec<bool>,
    pub default_fg: RgbColor,
    pub default_bg: RgbColor,
    /// `Some((x, y))` if the cursor is visible.
    pub cursor: Option<(u16, u16)>,
    /// Bumped on every snapshot update — renderer compares against the
    /// last value it consumed to skip whole frames when nothing changed.
    pub generation: u64,
    pub child_alive: bool,
    /// libghostty's scrollbar offset — the row index at the top of the
    /// visible viewport, measured from row 0 of the total scrollable
    /// area (scrollback + active). Increases as new content arrives or
    /// the viewport is scrolled toward the active area. Selection coords
    /// are anchored against this so a selection follows its content
    /// while the user scrolls.
    pub viewport_offset: u64,
}

/// Messages the main (Bevy) thread sends to a worker.
pub enum WorkerMsg {
    /// Bytes to write to the pty (keystrokes).
    Input(Vec<u8>),
    /// Clipboard text to paste. The worker queries the VT's bracketed
    /// paste mode (DEC 2004) and wraps with `\x1b[200~`/`\x1b[201~` when
    /// enabled, so the shell sees one paste blob instead of executing
    /// each newline as Enter. When disabled, libghostty's encoder
    /// rewrites `\n` to `\r` (still one Enter per line) — that's the
    /// best we can do without bracketed paste support on the other end.
    Paste(String),
    /// New grid dimensions (in cells) and per-cell pixel size. Worker
    /// resizes both the libghostty `Terminal` and the pty winsize.
    Resize {
        cols: u16,
        rows: u16,
        cell_w_px: u32,
        cell_h_px: u32,
    },
    /// Scroll the viewport by `delta` lines. Negative = back into
    /// scrollback history; positive = forward toward the active area.
    /// `None` is interpreted as "snap to the bottom (active area)".
    ScrollDelta(isize),
    ScrollToBottom,
    /// Extract the text of a selection that may span scrollback the
    /// visible snapshot doesn't hold. `start`/`end` are normalised
    /// `(col, absolute_row)` cells (absolute_row is measured from row 0
    /// of the total scrollable area — the same space as
    /// `GridSnapshot::viewport_offset`). The worker reads the cells
    /// straight out of the libghostty `Terminal` (which owns the full
    /// scrollback) and ships the joined text back on `reply`. Done on
    /// the worker because the `!Send` `Terminal` never leaves this
    /// thread.
    ExtractText {
        start: (i32, i64),
        end: (i32, i64),
        reply: Sender<String>,
    },
    Shutdown,
}

/// Handle the main thread keeps for each terminal.
pub struct WorkerHandle {
    pub snapshot: Arc<Mutex<GridSnapshot>>,
    /// Monotonic count of BEL characters processed by the VT parser.
    /// The renderer compares it against its own per-terminal `last_seen`
    /// to fire a one-shot visual flash. Atomic so the worker thread
    /// can bump it from inside libghostty's bell callback without the
    /// snapshot mutex.
    pub bell_count: Arc<AtomicU64>,
    /// Main thread flips this when the pane's visibility changes. While
    /// false the worker keeps processing pty bytes (so the libghostty
    /// terminal state stays correct), but skips waking Bevy — there's
    /// nothing for the renderer to do for a pane the user can't see.
    /// Saves the full per-frame schedule cost for inactive-project
    /// terminals running TUIs like `top` or Claude.
    pub visible: Arc<AtomicBool>,
    tx: Sender<WorkerMsg>,
    /// Write end of the worker's wake pipe. Every channel send pokes
    /// one byte here so the worker's blocking `poll(2)` returns even
    /// if no pty data is pending.
    wake_w: OwnedFd,
    /// Held to keep the thread joinable; we never actually `join` from
    /// the main thread — workers exit on `Shutdown` or PTY EOF.
    _join: JoinHandle<()>,
}

impl WorkerHandle {
    /// Send a message to the worker and wake it from `poll(2)`.
    ///
    /// Returns silently on send error (worker exited) — every existing
    /// caller already used `let _ = ...` semantics.
    pub fn send(&self, msg: WorkerMsg) {
        let _ = self.tx.send(msg);
        // Best-effort poke. Pipe is non-blocking; if the kernel buffer
        // is full there's already a pending wake — no need to add more.
        let _ = nix::unistd::write(self.wake_w.as_fd(), b"x");
    }
}

impl WorkerHandle {
    /// Spawn the worker. Connects to (or forks) a per-session daemon and
    /// hands the socket + a fresh libghostty `Terminal` to a worker
    /// thread.
    ///
    /// `session_id` keys the daemon (socket path is derived from it).
    /// `command` is the shell+args the daemon should exec if it has to
    /// fork a fresh one — ignored if the daemon is already running.
    /// `scrollback_log` mirrors every byte the daemon delivers to disk
    /// so cold-starts after machine reboot can still recover the screen.
    /// `replay_bytes` is the previous run's disk log; fed into `vt_write`
    /// only when we had to fork a *fresh* daemon (no in-memory history).
    pub fn spawn(
        session_id: u64,
        command: Vec<String>,
        initial_cwd: Option<String>,
        size: PtySize,
        scrollback: usize,
        scrollback_log: Option<PathBuf>,
        replay_bytes: Option<Vec<u8>>,
        wakeup: Option<EventLoopProxy<WinitUserEvent>>,
    ) -> std::io::Result<Self> {
        let client = DaemonClient::open(session_id, size.cols, size.rows, command, initial_cwd)?;
        let attached_existing = client.attached_existing;

        let snapshot = Arc::new(Mutex::new(GridSnapshot {
            cols: size.cols,
            rows: size.rows,
            cells: vec![SnapCell::default(); size.cols as usize * size.rows as usize],
            dirty_rows: vec![true; size.rows as usize],
            default_fg: RgbColor {
                r: 220,
                g: 220,
                b: 220,
            },
            default_bg: RgbColor {
                r: 13,
                g: 15,
                b: 20,
            },
            cursor: None,
            generation: 0,
            child_alive: true,
            viewport_offset: 0,
        }));
        let snapshot_w = snapshot.clone();
        let bell_count = Arc::new(AtomicU64::new(0));
        let bell_count_w = bell_count.clone();
        let visible = Arc::new(AtomicBool::new(true));
        let visible_w = visible.clone();

        let (tx, rx) = channel::<WorkerMsg>();

        // Self-pipe used to wake the worker out of poll(2) when a
        // message lands on the channel. Non-blocking on both ends so
        // neither side ever stalls on it.
        let (wake_r, wake_w) = nix::unistd::pipe()?;
        set_nonblock(&wake_r)?;
        set_nonblock(&wake_w)?;

        let join = thread::Builder::new()
            .name("terminal-worker".into())
            .spawn(move || {
                worker_loop(
                    client,
                    attached_existing,
                    size,
                    scrollback,
                    scrollback_log,
                    replay_bytes,
                    snapshot_w,
                    bell_count_w,
                    visible_w,
                    rx,
                    wakeup,
                    wake_r,
                )
            })
            .expect("spawn worker");

        Ok(Self {
            snapshot,
            bell_count,
            visible,
            tx,
            wake_w,
            _join: join,
        })
    }
}

fn set_nonblock(fd: &OwnedFd) -> std::io::Result<()> {
    let raw = fcntl::fcntl(fd, fcntl::F_GETFL)?;
    let flags = OFlag::from_bits_retain(raw) | OFlag::O_NONBLOCK;
    fcntl::fcntl(fd, fcntl::F_SETFL(flags))?;
    Ok(())
}

/// Hard ceiling on a per-terminal scrollback log file. When the log
/// exceeds this, the worker keeps the trailing `SCROLLBACK_LOG_KEEP`
/// bytes and rewrites in place. Sized so a busy terminal can keep
/// roughly the libghostty 100k-line scrollback worth of styled output
/// without growing without bound.
const SCROLLBACK_LOG_MAX: u64 = 16 * 1024 * 1024;
const SCROLLBACK_LOG_KEEP: u64 = 12 * 1024 * 1024;
/// Check the log size every N bytes appended (cheap counter test in
/// the hot path; the actual fs metadata call only runs at boundaries).
const SCROLLBACK_ROTATE_CHECK_EVERY: u64 = 256 * 1024;

fn worker_loop(
    mut client: DaemonClient,
    attached_existing: bool,
    initial_size: PtySize,
    scrollback: usize,
    scrollback_log: Option<PathBuf>,
    replay_bytes: Option<Vec<u8>>,
    snapshot: Arc<Mutex<GridSnapshot>>,
    bell_count: Arc<AtomicU64>,
    visible: Arc<AtomicBool>,
    rx: Receiver<WorkerMsg>,
    wakeup: Option<EventLoopProxy<WinitUserEvent>>,
    wake_r: OwnedFd,
) {
    let cell_px = CellPx {
        width: initial_size.cell_width_px as u32,
        height: initial_size.cell_height_px as u32,
    };
    let (mut terminal, pty_response) = vt::build_terminal(
        initial_size.cols,
        initial_size.rows,
        scrollback,
        cell_px,
    );

    // Shared flag: are we currently feeding replay bytes into the VT?
    // BEL bytes from past sessions show up during disk-replay (line
    // ~387 below) and during the daemon's ReplayStart→ReplayEnd
    // window. Both would otherwise increment `bell_count` and bump
    // the project's unread counter for events the user already saw.
    // The on_bell closure below reads this and skips increments
    // while true.
    let in_replay_flag = Arc::new(AtomicBool::new(false));

    // Bell handler. vt.rs deliberately doesn't register one — we own it
    // here so the closure can poke the per-terminal counter and wake
    // winit (an idle terminal in reactive mode would otherwise sit on
    // the bell for up to 5s before flashing).
    {
        let bell_count = bell_count.clone();
        let bell_wakeup = wakeup.clone();
        let in_replay_flag = in_replay_flag.clone();
        terminal
            .on_bell(move |_term| {
                if in_replay_flag.load(Ordering::Relaxed) {
                    // Replay BEL — silently absorb; the user already
                    // experienced this bell in a prior session.
                    return;
                }
                let n = bell_count.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!("[bell] fired, count={}", n);
                if let Some(p) = &bell_wakeup {
                    let _ = p.send_event(WinitUserEvent::WakeUp);
                }
            })
            .expect("on_bell");
    }

    // If we had to fork a fresh daemon (no in-memory history) and we
    // have a disk-replay log for this session_id, feed it in locally so
    // the visible scrollback is back. We must clear pty_response after:
    // replaying old DA / size queries would otherwise ship stale replies
    // to the brand-new shell as if it had asked.
    //
    // If we attached to an *existing* daemon, ignore the disk log — the
    // daemon's ReplayStart/Output…/ReplayEnd is the authoritative state.
    if !attached_existing {
        if let Some(bytes) = replay_bytes
            && !bytes.is_empty()
        {
            in_replay_flag.store(true, Ordering::Relaxed);
            terminal.vt_write(&bytes);
            in_replay_flag.store(false, Ordering::Relaxed);
            pty_response.borrow_mut().clear();
        }
    }

    // Append-only log of every byte we feed to `vt_write`. This is what
    // gets replayed on the next launch.
    let mut log_writer: Option<ScrollbackLogWriter> = scrollback_log
        .as_ref()
        .and_then(|p| ScrollbackLogWriter::open(p.clone()));

    // OSC 7 (current-directory report) watcher. The shell-integration
    // shim makes zsh emit `\e]7;file://host/path\e\\` on every prompt
    // and on every `cd`. We feed the same byte stream we hand to
    // `vt_write` through this watcher and publish a bus event each
    // time the cwd changes. Suppressed during replay — those OSC 7s
    // are from prior sessions.
    let mut osc7 = crate::osc7::Osc7Watcher::default();
    let mut last_published_cwd: Option<String> = None;
    // OSC 133 command marks → `terminal.command_executed` events that
    // feed the command-suggestion classifier. Same byte stream, same
    // replay suppression as OSC 7.
    let mut cmd_watch = crate::command_watch::CommandWatcher::default();
    let session_id = client.session_id();

    let mut render_state = RenderState::new().expect("RenderState");
    let mut row_it = RowIterator::new().expect("RowIterator");
    let mut cell_it = CellIterator::new().expect("CellIterator");

    // Track last published dims so publish_snapshot can detect a resize
    // and force a full redraw — `g.cells` is row-major flat-indexed and
    // a cols change makes every linear index map to a different cell.
    let mut last_cols: u16 = initial_size.cols;
    let mut last_rows: u16 = initial_size.rows;

    // Lightweight throughput instrumentation, gated on TERMINAL_PROFILE
    // env var so it doesn't spam stderr in normal runs. Set
    // `TERMINAL_PROFILE=1` to re-enable for perf debugging.
    let profile = std::env::var("TERMINAL_PROFILE").is_ok();
    const PROFILE_INTERVAL: u64 = 64;
    let mut tick_count: u64 = 0;
    let mut bytes_since_log: u64 = 0;
    let mut publishes_since_log: u64 = 0;
    let mut vt_write_ns_since_log: u128 = 0;
    let mut publish_ns_since_log: u128 = 0;
    let mut log_window_start = Instant::now();

    // True between DaemonMessage::ReplayStart and ReplayEnd — during
    // this window we keep feeding bytes to `vt_write` but suppress
    // snapshot publishes, so the renderer doesn't flicker through dozens
    // of intermediate states on attach. Final state is published once
    // when ReplayEnd arrives.
    let mut in_replay = false;
    let mut replay_just_ended = false;
    // Tracked locally; the daemon drives this via DaemonMessage::ChildExited.
    // Starts true on attach because if the daemon's still serving, the
    // child is presumed alive — we'll get notified if it isn't.
    let mut child_alive = true;
    // Set when the daemon socket closes — we publish a final snapshot
    // and then idle waiting for Shutdown from the main thread.
    let mut daemon_gone = false;

    // Local "I have a wake pending" flag. The actual throttle is
    // process-global (`try_wake_winit_throttled`); this flag remembers
    // that we wanted to wake during the cooldown so the top of the
    // loop can retry once it expires.
    let mut pending_wake = false;

    loop {
        // Drain any deferred wake whose cooldown has now expired. Safety
        // net: a snapshot got published during the global cooldown and
        // no further chunks have arrived; we still need to notify Bevy
        // so the screen doesn't go stale. Hidden panes never wake — the
        // visibility sync will fire one for us on un-hide.
        if pending_wake && visible.load(Ordering::Relaxed) {
            if try_wake_winit_throttled(&wakeup) {
                pending_wake = false;
            }
        } else if pending_wake && !visible.load(Ordering::Relaxed) {
            pending_wake = false;
        }

        let mut did_anything = false;
        let mut force_full_publish = false;
        tick_count += 1;

        // 1. Drain socket: pull every available DaemonMessage frame.
        if !daemon_gone {
            let mut output_bytes: u64 = 0;
            let mut had_any = false;
            let vt_start = Instant::now();
            let alive = client
                .poll_frames(|m| {
                    had_any = true;
                    match m {
                        DaemonMessage::Output(bytes) => {
                            terminal.vt_write(&bytes);
                            if !in_replay {
                                // Log only LIVE output. Replayed history is
                                // the daemon's authoritative buffer; the
                                // disk log already holds that scrollback
                                // from the prior session, so re-appending it
                                // on every warm restart duplicates content —
                                // wasting the rotate cap (real deep history
                                // gets trimmed sooner) and doubling
                                // scrollback on the next cold start. The
                                // cold-start disk replay (above) is fed
                                // before this writer exists, so it's never
                                // logged either — same invariant. (osc7 /
                                // cmd_watch below are already live-only.)
                                if let Some(w) = log_writer.as_mut() {
                                    w.append(&bytes);
                                }
                                osc7.feed(&bytes, |cwd| {
                                    if last_published_cwd.as_deref() != Some(cwd.as_str()) {
                                        publish_cwd_changed(session_id, &cwd);
                                        last_published_cwd = Some(cwd);
                                    }
                                });
                                cmd_watch.feed(&bytes, |command, exit_code| {
                                    publish_command_executed(
                                        session_id,
                                        &command,
                                        exit_code,
                                        last_published_cwd.as_deref(),
                                    );
                                });
                            }
                            output_bytes += bytes.len() as u64;
                        }
                        DaemonMessage::ReplayStart => {
                            in_replay = true;
                            in_replay_flag.store(true, Ordering::Relaxed);
                        }
                        DaemonMessage::ReplayEnd => {
                            in_replay = false;
                            in_replay_flag.store(false, Ordering::Relaxed);
                            replay_just_ended = true;
                        }
                        DaemonMessage::ChildExited { code: _ } => {
                            child_alive = false;
                        }
                        DaemonMessage::Attached => {
                            // Already consumed during handshake — a
                            // second one shouldn't happen, ignore.
                        }
                    }
                })
                .unwrap_or(false);
            if had_any {
                vt_write_ns_since_log += vt_start.elapsed().as_nanos();
                bytes_since_log += output_bytes;
                did_anything = true;
            }
            if !alive {
                daemon_gone = true;
                child_alive = false;
                force_full_publish = true;
                did_anything = true;
            }
        }

        // 2. Forward libghostty's pty-response bytes (DA replies, etc.)
        //    back to the daemon as Input — but NOT while replaying
        //    history. Replayed output carries the PRIOR session's
        //    capability queries (Primary DA `…c`, XTVERSION `DCS>|…`,
        //    DECRQM mode 2026 `…$y`, etc.); `vt_write` regenerates their
        //    replies into `pty_response`, and shipping those to the LIVE
        //    child injects e.g. `>|terminal-bevy…62;1;6;22c…2026;2$y` into
        //    its stdin, which the shell then echoes at the prompt. Drop
        //    anything generated during the replay window (and the final
        //    batch as ReplayEnd lands) — mirrors the disk-replay clear in
        //    the attach path above. The live child's own queries arrive as
        //    output after ReplayEnd and are answered normally.
        {
            let mut response = pty_response.borrow_mut();
            if in_replay || replay_just_ended {
                response.clear();
            } else if !response.is_empty() {
                let bytes: Vec<u8> = response.drain(..).collect();
                client.send(&ClientMessage::Input(bytes));
                did_anything = true;
            }
        }

        // 3. Drain channel messages from the main thread.
        loop {
            match rx.try_recv() {
                Ok(WorkerMsg::Input(bytes)) => {
                    client.send(&ClientMessage::Input(bytes));
                    did_anything = true;
                }
                Ok(WorkerMsg::Paste(text)) => {
                    let bracketed = terminal.mode(Mode::BRACKETED_PASTE).unwrap_or(false);
                    let mut data = text.into_bytes();
                    let mut buf = vec![0u8; data.len() + 16];
                    match paste::encode(&mut data, bracketed, &mut buf) {
                        Ok(len) => {
                            client.send(&ClientMessage::Input(buf[..len].to_vec()));
                            did_anything = true;
                        }
                        Err(_) => {
                            panic!("paste::encode failed for {} bytes", data.len());
                        }
                    }
                }
                Ok(WorkerMsg::Resize {
                    cols,
                    rows,
                    cell_w_px,
                    cell_h_px,
                }) => {
                    let _ = terminal.resize(cols, rows, cell_w_px, cell_h_px);
                    client.send(&ClientMessage::Resize { cols, rows });
                    did_anything = true;
                }
                Ok(WorkerMsg::ScrollDelta(delta)) => {
                    terminal.scroll_viewport(ScrollViewport::Delta(delta));
                    did_anything = true;
                    force_full_publish = true;
                }
                Ok(WorkerMsg::ScrollToBottom) => {
                    terminal.scroll_viewport(ScrollViewport::Bottom);
                    did_anything = true;
                    force_full_publish = true;
                }
                Ok(WorkerMsg::ExtractText { start, end, reply }) => {
                    // Best-effort; if the receiver is gone the main
                    // thread already moved on.
                    let _ = reply.send(extract_screen_selection(&terminal, start, end));
                }
                Ok(WorkerMsg::Shutdown) => {
                    // Explicit pane close → tell the daemon to die.
                    // Flush, give the kernel a tick to deliver, then exit.
                    client.send(&ClientMessage::Kill);
                    client.try_flush();
                    return;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }

        // 3b. Push queued outbound frames to the socket.
        client.try_flush();

        // 4. Publish snapshot if anything changed, except during the
        //    replay window — that's a one-shot final publish at ReplayEnd.
        let want_publish = did_anything && !in_replay;
        if want_publish || replay_just_ended {
            let force = force_full_publish || replay_just_ended;
            let t = Instant::now();
            let (pc, pr, published) = publish_snapshot(
                &mut terminal,
                &mut render_state,
                &mut row_it,
                &mut cell_it,
                &snapshot,
                child_alive,
                force,
                last_cols,
                last_rows,
            );
            last_cols = pc;
            last_rows = pr;
            publish_ns_since_log += t.elapsed().as_nanos();
            publishes_since_log += 1;
            if published {
                // Skip waking the renderer for hidden panes — the
                // libghostty Terminal still tracks state correctly via
                // the snapshot mutex, so when the pane becomes visible
                // again `sync_grid` reads the up-to-date generation and
                // does a forced repaint. Cuts the per-frame schedule
                // cost of TUIs running in inactive-project terminals.
                if visible.load(Ordering::Relaxed) {
                    if try_wake_winit_throttled(&wakeup) {
                        pending_wake = false;
                    } else {
                        pending_wake = true;
                    }
                } else {
                    pending_wake = false;
                }
            }
            replay_just_ended = false;
        }

        if daemon_gone {
            // Daemon socket closed — nothing more we can do here, just
            // idle until main thread issues Shutdown / disconnects.
            thread::sleep(Duration::from_millis(50));
            continue;
        }

        if profile && tick_count % PROFILE_INTERVAL == 0 && bytes_since_log > 0 {
            let elapsed = log_window_start.elapsed();
            eprintln!(
                "[worker] {:>7} bytes in {:>5.1} ms  ({:>5.1} MiB/s) | \
                 {:>3} publishes (avg {:>5.1} ms) | vt_write avg {:>5.1} ms",
                bytes_since_log,
                elapsed.as_secs_f64() * 1000.0,
                (bytes_since_log as f64) / (1024.0 * 1024.0) / elapsed.as_secs_f64(),
                publishes_since_log,
                (publish_ns_since_log as f64 / 1_000_000.0)
                    / (publishes_since_log.max(1) as f64),
                (vt_write_ns_since_log as f64 / 1_000_000.0)
                    / (publishes_since_log.max(1) as f64),
            );
            bytes_since_log = 0;
            publishes_since_log = 0;
            vt_write_ns_since_log = 0;
            publish_ns_since_log = 0;
            log_window_start = Instant::now();
        }

        // 5. If we did work, loop right back to drain more frames —
        // under heavy output we want to keep chewing through the
        // socket. If nothing happened this tick, block in poll(2)
        // until the socket has more data or the main thread pokes
        // our wake pipe. Difference between zero idle CPU and a 2 kHz
        // spin loop.
        if !did_anything {
            let mut drain_buf = [0u8; 64];
            loop {
                match nix::unistd::read(&wake_r, &mut drain_buf) {
                    Ok(n) if n > 0 => continue,
                    _ => break,
                }
            }

            let sock_fd = client.as_fd();
            let wake_fd = wake_r.as_fd();
            let sock_flags = if client.pending_out_empty() {
                PollFlags::POLLIN
            } else {
                PollFlags::POLLIN | PollFlags::POLLOUT
            };
            let mut fds = [
                PollFd::new(sock_fd, sock_flags),
                PollFd::new(wake_fd, PollFlags::POLLIN),
            ];
            // If a wake was deferred by the rate limiter, cap the poll
            // timeout so we come back around in time to fire it. Without
            // this, an output burst that ends exactly when we'd suppress
            // the last wake could leave the screen stale for up to 5s.
            let timeout_ms: i32 = if pending_wake {
                wake_cooldown_remaining_ms().min(5_000) as i32
            } else {
                5_000
            };
            let timeout = PollTimeout::try_from(timeout_ms.max(0))
                .unwrap_or(PollTimeout::try_from(16i32).unwrap());
            match poll(&mut fds, timeout) {
                Ok(_) => {}
                Err(Errno::EINTR) => {}
                Err(_) => {
                    thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }
}

fn publish_snapshot(
    terminal: &mut libghostty_vt::Terminal<'static, 'static>,
    render_state: &mut RenderState<'static>,
    row_it: &mut RowIterator<'static>,
    cell_it: &mut CellIterator<'static>,
    snapshot_arc: &Arc<Mutex<GridSnapshot>>,
    child_alive: bool,
    force_full: bool,
    prev_cols: u16,
    prev_rows: u16,
) -> (u16, u16, bool) {
    let snap = match render_state.update(terminal) {
        Ok(s) => s,
        Err(_) => return (prev_cols, prev_rows, false),
    };
    let dirty = snap.dirty().unwrap_or(Dirty::Full);
    let cols = snap.cols().unwrap_or(0);
    let rows = snap.rows().unwrap_or(0);
    let dims_changed = cols != prev_cols || rows != prev_rows;
    if matches!(dirty, Dirty::Clean) && !force_full && !dims_changed {
        // Still update child_alive flag without touching cells.
        let mut g = snapshot_arc.lock().expect("snapshot lock");
        let mut published = false;
        if g.child_alive != child_alive {
            g.child_alive = child_alive;
            g.generation = g.generation.wrapping_add(1);
            published = true;
        }
        return (cols, rows, published);
    }
    let cursor_visible = snap.cursor_visible().unwrap_or(false);
    let cursor_pos = snap.cursor_viewport().ok().flatten();
    let palette = snap.colors().ok();
    let default_fg = palette.as_ref().map(|c| c.foreground).unwrap_or(RgbColor {
        r: 220,
        g: 220,
        b: 220,
    });
    let default_bg = palette.map(|c| c.background).unwrap_or(RgbColor {
        r: 13,
        g: 15,
        b: 20,
    });

    // A dim change relaying-out g.cells means every linear index now maps
    // to a different (row, col) — anything we don't overwrite from libghostty
    // would still hold OLD-layout content, leaking glyph fragments at
    // mis-mapped positions. Force a full repaint and wipe g.cells below.
    let force_all = matches!(dirty, Dirty::Full) || force_full || dims_changed;

    // Walk dirty rows under the snapshot iterator, accumulating into a
    // small local scratch so we hold the GridSnapshot mutex for the
    // shortest possible window. Per-cell FFI happens while we're NOT
    // holding the lock.
    struct PendingRow {
        idx: usize,
        cells: Vec<SnapCell>,
    }
    let mut pending: Vec<PendingRow> = Vec::with_capacity(rows as usize);
    let mut row_dirty_flags: Vec<bool> = Vec::with_capacity(rows as usize);

    {
        let mut iter = match row_it.update(&snap) {
            Ok(it) => it,
            Err(_) => return (cols, rows, false),
        };
        let mut r = 0usize;
        while let Some(row) = iter.next() {
            if r >= rows as usize {
                break;
            }
            let row_dirty = row.dirty().unwrap_or(true);
            let process = force_all || row_dirty;
            row_dirty_flags.push(process);
            if process {
                let mut cells = Vec::with_capacity(cols as usize);
                let mut cell_iter = match cell_it.update(row) {
                    Ok(it) => it,
                    Err(_) => {
                        // Pad with defaults so indexing stays sane.
                        cells.resize(cols as usize, SnapCell::default());
                        pending.push(PendingRow { idx: r, cells });
                        r += 1;
                        continue;
                    }
                };
                let mut c = 0usize;
                while let Some(cell) = cell_iter.next() {
                    if c >= cols as usize {
                        break;
                    }
                    let glen = cell.graphemes_len().unwrap_or(0);
                    let ch = if glen == 0 {
                        ' '
                    } else {
                        cell.graphemes()
                            .ok()
                            .and_then(|cs| cs.into_iter().next())
                            .unwrap_or(' ')
                    };
                    let fg = cell.fg_color().ok().flatten().unwrap_or(default_fg);
                    let bg = cell.bg_color().ok().flatten().unwrap_or(default_bg);
                    let inverse = cell.style().ok().map(|s| s.inverse).unwrap_or(false);
                    cells.push(SnapCell {
                        ch,
                        fg,
                        bg,
                        inverse,
                    });
                    c += 1;
                }
                while cells.len() < cols as usize {
                    cells.push(SnapCell {
                        ch: ' ',
                        fg: default_fg,
                        bg: default_bg,
                        inverse: false,
                    });
                }
                pending.push(PendingRow { idx: r, cells });
            }
            let _ = row.set_dirty(false);
            r += 1;
        }
        let _ = snap.set_dirty(Dirty::Clean);
    }

    // Now take the lock and publish — short critical section.
    let mut g = snapshot_arc.lock().expect("snapshot lock");
    g.cols = cols;
    g.rows = rows;
    let total = cols as usize * rows as usize;
    if dims_changed {
        // Drop stale OLD-layout content; pending rows below repopulate
        // whatever libghostty owns now, and unwritten slots stay default.
        g.cells.clear();
        g.cells.resize(total, SnapCell::default());
    } else if g.cells.len() != total {
        g.cells.resize(total, SnapCell::default());
    }
    if g.dirty_rows.len() != rows as usize {
        g.dirty_rows.resize(rows as usize, false);
    }
    for (i, flag) in row_dirty_flags.iter().enumerate() {
        if i < g.dirty_rows.len() {
            g.dirty_rows[i] = *flag;
        }
    }
    for prow in pending {
        let base = prow.idx * cols as usize;
        let end = (base + prow.cells.len()).min(g.cells.len());
        let take = end - base;
        if take > 0 {
            g.cells[base..end].copy_from_slice(&prow.cells[..take]);
        }
    }
    g.default_fg = default_fg;
    g.default_bg = default_bg;
    g.cursor = if cursor_visible {
        cursor_pos.map(|p| (p.x, p.y))
    } else {
        None
    };
    g.child_alive = child_alive;
    // libghostty docs warn scrollbar() may be expensive for arbitrary
    // pin positions; in practice the viewport sits at the bottom or
    // close to it most of the time and the lookup is cheap. We're
    // already inside the snapshot publish path which runs at most a
    // few times per second under steady-state output.
    g.viewport_offset = terminal.scrollbar().map(|s| s.offset).unwrap_or(0);
    g.generation = g.generation.wrapping_add(1);
    (cols, rows, true)
}

/// Extract the text covered by a `(col, absolute_row)` selection from
/// the terminal's full scrollable area (scrollback + active), trimming
/// trailing whitespace per row and joining rows with newlines.
///
/// `absolute_row` is in the same coordinate space as
/// `GridSnapshot::viewport_offset` (row 0 of the total scrollable area),
/// which is exactly libghostty's `Point::Screen` y. That's why this can
/// reach rows that have scrolled out of the visible snapshot — the
/// renderer's snapshot only holds the visible grid, but the `Terminal`
/// here still owns the whole history.
///
/// Mirrors the per-row column-span logic in `selection.rs`: first row
/// runs from the start col to the row end, last row from row start to
/// the end col, middle rows span fully.
fn extract_screen_selection(
    terminal: &libghostty_vt::Terminal<'static, 'static>,
    start: (i32, i64),
    end: (i32, i64),
) -> String {
    let cols = terminal.cols().unwrap_or(0) as i32;
    let total_rows = terminal.total_rows().unwrap_or(0) as i64;
    if cols == 0 || total_rows == 0 {
        return String::new();
    }
    let first = start.1.max(0);
    let last = end.1.min(total_rows - 1);
    if first > last {
        return String::new();
    }

    let mut out = String::new();
    let mut grapheme_buf = [' '; 16];
    for row in first..=last {
        let (col_start, col_end) = if start.1 == end.1 {
            (start.0.min(end.0), start.0.max(end.0))
        } else if row == start.1 {
            (start.0, cols - 1)
        } else if row == end.1 {
            (0, end.0)
        } else {
            (0, cols - 1)
        };
        let col_start = col_start.max(0);
        let col_end = col_end.min(cols - 1);
        if col_end < col_start {
            if row != last {
                out.push('\n');
            }
            continue;
        }
        let mut line = String::new();
        for col in col_start..=col_end {
            let point = Point::Screen(PointCoordinate {
                x: col as u16,
                y: row as u32,
            });
            let ch = terminal
                .grid_ref(point)
                .ok()
                .and_then(|g| {
                    let n = g.graphemes(&mut grapheme_buf).ok()?;
                    (n > 0).then(|| grapheme_buf[0])
                })
                .unwrap_or(' ');
            line.push(ch);
        }
        // Trim trailing whitespace — terminals pad rows with spaces and
        // copying a screenful of those is annoying. Leading spaces
        // (indentation) are preserved.
        out.push_str(line.trim_end());
        if row != last {
            out.push('\n');
        }
    }
    out
}

/// Marker type for OwnedFd round-trip if needed elsewhere.
#[allow(dead_code)]
pub struct WorkerFd(pub OwnedFd);

/// Best-effort publish of a `terminal.cwd_changed` event to the bus.
/// Called from the worker thread on each unique OSC 7 report. The
/// publish is fire-and-forget — if the bus daemon isn't running we
/// silently drop the event rather than perturb terminal output.
fn publish_cwd_changed(session_id: u64, cwd: &str) {
    let Some(socket) = claude_bus::socket_path() else {
        return;
    };
    let payload = serde_json::json!({
        "session_id": session_id,
        "cwd": cwd,
    })
    .to_string();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let _ = claude_bus::client::publish_oneshot(
        &socket,
        "terminal.cwd_changed",
        ts,
        &session_id.to_string(),
        std::process::id(),
        &payload,
    );
}

/// Best-effort publish of a `terminal.command_executed` event to the
/// bus. Called from the worker thread when an OSC 133 `D` mark pairs
/// with its `C`. Fire-and-forget like [`publish_cwd_changed`].
fn publish_command_executed(session_id: u64, command: &str, exit_code: i32, cwd: Option<&str>) {
    let Some(socket) = claude_bus::socket_path() else {
        return;
    };
    let payload = serde_json::json!({
        "session_id": session_id,
        "command": command,
        "cwd": cwd.unwrap_or(""),
        "exit_code": exit_code,
    })
    .to_string();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let _ = claude_bus::client::publish_oneshot(
        &socket,
        "terminal.command_executed",
        ts,
        &session_id.to_string(),
        std::process::id(),
        &payload,
    );
}

/// Append-only writer for a per-terminal scrollback log. Bytes written
/// here are exactly what was fed to `vt_write`, so the next launch can
/// `vt_write` them straight into a fresh Terminal to recover scrollback.
///
/// Self-trims when the file exceeds `SCROLLBACK_LOG_MAX` by reading the
/// trailing `SCROLLBACK_LOG_KEEP` bytes and rewriting via tmp+rename.
/// Trim points may land mid-escape; libghostty's parser is robust to
/// the resulting garbage prefix and the next prompt will overwrite it.
struct ScrollbackLogWriter {
    path: PathBuf,
    file: File,
    bytes_since_check: u64,
}

impl ScrollbackLogWriter {
    fn open(path: PathBuf) -> Option<Self> {
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            eprintln!("[worker] mkdir {}: {}", parent.display(), e);
            return None;
        }
        let file = match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[worker] open {}: {}", path.display(), e);
                return None;
            }
        };
        Some(Self {
            path,
            file,
            bytes_since_check: 0,
        })
    }

    fn append(&mut self, bytes: &[u8]) {
        if let Err(e) = self.file.write_all(bytes) {
            eprintln!("[worker] log write {}: {}", self.path.display(), e);
            return;
        }
        self.bytes_since_check = self.bytes_since_check.saturating_add(bytes.len() as u64);
        if self.bytes_since_check >= SCROLLBACK_ROTATE_CHECK_EVERY {
            self.bytes_since_check = 0;
            if let Ok(meta) = self.file.metadata()
                && meta.len() > SCROLLBACK_LOG_MAX
            {
                self.rotate();
            }
        }
    }

    fn rotate(&mut self) {
        // Read tail, rewrite atomically. The append-mode handle stays
        // open the whole time, but we reopen after rename so subsequent
        // appends go to the new file.
        let path = &self.path;
        let read_path = path.clone();
        let tmp_path = path.with_extension("bytes.tmp");

        let trimmed = match (|| -> std::io::Result<Vec<u8>> {
            let mut f = File::open(&read_path)?;
            let len = f.metadata()?.len();
            let keep = SCROLLBACK_LOG_KEEP.min(len);
            let start = len - keep;
            f.seek(SeekFrom::Start(start))?;
            let mut buf = Vec::with_capacity(keep as usize);
            f.read_to_end(&mut buf)?;
            Ok(buf)
        })() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[worker] log read-tail {}: {}", path.display(), e);
                return;
            }
        };

        let write_result = (|| -> std::io::Result<()> {
            let mut f = File::create(&tmp_path)?;
            f.write_all(&trimmed)?;
            f.sync_all()?;
            std::fs::rename(&tmp_path, path)
        })();
        if let Err(e) = write_result {
            eprintln!("[worker] log rotate {}: {}", path.display(), e);
            return;
        }

        match OpenOptions::new().create(true).append(true).open(path) {
            Ok(f) => self.file = f,
            Err(e) => eprintln!("[worker] log reopen {}: {}", path.display(), e),
        }
    }
}
