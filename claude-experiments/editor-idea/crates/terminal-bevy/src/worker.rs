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

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::os::fd::{AsFd, OwnedFd};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use bevy::winit::{EventLoopProxy, WinitUserEvent};
use libghostty_vt::{
    paste,
    render::{CellIterator, Dirty, RenderState, RowIterator},
    style::RgbColor,
    terminal::{Mode, ScrollViewport},
};
use nix::errno::Errno;
use nix::fcntl::{self, OFlag};
use nix::poll::{poll, PollFd, PollFlags, PollTimeout};

use crate::pty::{Child, Pty, PtySize};
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
    /// Spawn the worker. Forks the pty + child shell, then hands fd +
    /// ownership of `Terminal` to a fresh thread.
    ///
    /// `scrollback_log` is the path to append raw pty bytes to. When
    /// `Some`, every chunk read from the pty is mirrored to disk so the
    /// next launch can replay it. `replay_bytes` is the previous run's
    /// log — fed straight into `vt_write` before the pty drain loop
    /// starts, so the visible scrollback comes back across restarts.
    pub fn spawn(
        size: PtySize,
        scrollback: usize,
        scrollback_log: Option<PathBuf>,
        replay_bytes: Option<Vec<u8>>,
        wakeup: Option<EventLoopProxy<WinitUserEvent>>,
    ) -> std::io::Result<Self> {
        let (pty, child) = Pty::spawn(size)?;
        pty.set_nonblock()?;

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
        }));
        let snapshot_w = snapshot.clone();
        let bell_count = Arc::new(AtomicU64::new(0));
        let bell_count_w = bell_count.clone();

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
                    pty,
                    child,
                    size,
                    scrollback,
                    scrollback_log,
                    replay_bytes,
                    snapshot_w,
                    bell_count_w,
                    rx,
                    wakeup,
                    wake_r,
                )
            })
            .expect("spawn worker");

        Ok(Self {
            snapshot,
            bell_count,
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

/// Cap how many bytes the worker feeds to `vt_write` between snapshot
/// publishes. Same idea as Alacritty's `MAX_LOCKED_READ` — bound the
/// snapshot lock-hold time so the renderer never waits much.
const MAX_READ_PER_TICK: usize = 65536;

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
    pty: Pty,
    mut child: Child,
    initial_size: PtySize,
    scrollback: usize,
    scrollback_log: Option<PathBuf>,
    replay_bytes: Option<Vec<u8>>,
    snapshot: Arc<Mutex<GridSnapshot>>,
    bell_count: Arc<AtomicU64>,
    rx: Receiver<WorkerMsg>,
    wakeup: Option<EventLoopProxy<WinitUserEvent>>,
    wake_r: OwnedFd,
) {
    let wake_winit = || {
        if let Some(proxy) = wakeup.as_ref() {
            // Best-effort: if the event loop has already exited the
            // proxy returns Err — nothing meaningful to do, we're
            // shutting down anyway.
            let _ = proxy.send_event(WinitUserEvent::WakeUp);
        }
    };
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

    // Bell handler. vt.rs deliberately doesn't register one — we own it
    // here so the closure can poke the per-terminal counter and wake
    // winit (an idle terminal in reactive mode would otherwise sit on
    // the bell for up to 5s before flashing).
    {
        let bell_count = bell_count.clone();
        let bell_wakeup = wakeup.clone();
        terminal
            .on_bell(move |_term| {
                let n = bell_count.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!("[bell] fired, count={}", n);
                if let Some(p) = &bell_wakeup {
                    let _ = p.send_event(WinitUserEvent::WakeUp);
                }
            })
            .expect("on_bell");
    }

    // Replay previous-session bytes (if any) into the fresh Terminal so
    // its scrollback matches what was on screen before. We must clear
    // pty_response afterwards: replaying old content would otherwise
    // re-trigger every DA / size query the original session answered,
    // and we'd ship those replies back to the brand-new shell as if it
    // had asked.
    if let Some(bytes) = replay_bytes
        && !bytes.is_empty()
    {
        terminal.vt_write(&bytes);
        pty_response.borrow_mut().clear();
    }

    // Append-only log of every byte we feed to `vt_write`. This is what
    // gets replayed on the next launch.
    let mut log_writer: Option<ScrollbackLogWriter> = scrollback_log
        .as_ref()
        .and_then(|p| ScrollbackLogWriter::open(p.clone()));

    let mut render_state = RenderState::new().expect("RenderState");
    let mut row_it = RowIterator::new().expect("RowIterator");
    let mut cell_it = CellIterator::new().expect("CellIterator");

    // Track last published dims so publish_snapshot can detect a resize
    // and force a full redraw — `g.cells` is row-major flat-indexed and
    // a cols change makes every linear index map to a different cell.
    let mut last_cols: u16 = initial_size.cols;
    let mut last_rows: u16 = initial_size.rows;

    let mut read_buf = [0u8; 65536];

    // Bytes queued to write back to the pty (VT responses, keystrokes,
    // paste blobs). Single ordered queue so a paste followed by a
    // keystroke reaches the shell in the right order, and so the bytes
    // we couldn't squeeze into the kernel pty input buffer this tick
    // simply wait until POLLOUT instead of being silently dropped.
    let mut pending_out: VecDeque<u8> = VecDeque::new();

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

    loop {
        let mut did_anything = false;
        let mut force_full_publish = false;
        let mut bytes_processed_this_tick = 0usize;
        let mut hit_eof = false;
        tick_count += 1;

        // 1. Drain pty (non-blocking). Bounded per-tick so we don't
        // hold the snapshot lock for an entire 1 GiB cat.
        while bytes_processed_this_tick < MAX_READ_PER_TICK {
            match nix::unistd::read(&pty.0, &mut read_buf) {
                Ok(0) => {
                    hit_eof = true;
                    break;
                }
                Ok(n) => {
                    let t = Instant::now();
                    terminal.vt_write(&read_buf[..n]);
                    vt_write_ns_since_log += t.elapsed().as_nanos();
                    if let Some(w) = log_writer.as_mut() {
                        w.append(&read_buf[..n]);
                    }
                    bytes_processed_this_tick += n;
                    bytes_since_log += n as u64;
                    did_anything = true;
                }
                Err(Errno::EAGAIN) => break,
                Err(Errno::EINTR) => continue,
                Err(Errno::EIO) => {
                    hit_eof = true;
                    break;
                }
                Err(_) => {
                    hit_eof = true;
                    break;
                }
            }
        }

        // 2. Queue VT-effect responses (DA replies, etc.) for the pty.
        {
            let mut response = pty_response.borrow_mut();
            if !response.is_empty() {
                pending_out.extend(response.drain(..));
                did_anything = true;
            }
        }

        // 3. Drain channel messages from the main thread.
        loop {
            match rx.try_recv() {
                Ok(WorkerMsg::Input(bytes)) => {
                    pending_out.extend(bytes);
                    did_anything = true;
                }
                Ok(WorkerMsg::Paste(text)) => {
                    let bracketed = terminal.mode(Mode::BRACKETED_PASTE).unwrap_or(false);
                    let mut data = text.into_bytes();
                    // Encoder may rewrite bytes in place (control-byte
                    // scrub, `\n` → `\r` when not bracketed). Output
                    // adds at most the 6-byte prefix + 6-byte suffix
                    // when bracketed; +16 is a safe slack.
                    let mut buf = vec![0u8; data.len() + 16];
                    match paste::encode(&mut data, bracketed, &mut buf) {
                        Ok(len) => {
                            pending_out.extend(&buf[..len]);
                            did_anything = true;
                        }
                        Err(_) => {
                            // OutOfSpace shouldn't happen with the
                            // sizing above; bail loudly so a future
                            // change to the encoder surfaces here.
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
                    pty.resize(PtySize {
                        cols,
                        rows,
                        cell_width_px: cell_w_px as u16,
                        cell_height_px: cell_h_px as u16,
                    });
                    did_anything = true;
                }
                Ok(WorkerMsg::ScrollDelta(delta)) => {
                    terminal.scroll_viewport(ScrollViewport::Delta(delta));
                    did_anything = true;
                    // Scrolling rotates which scrollback rows the
                    // viewport refers to; libghostty's per-row dirty
                    // bits don't always cover that, so force a full
                    // repaint to guarantee the new content shows up.
                    force_full_publish = true;
                }
                Ok(WorkerMsg::ScrollToBottom) => {
                    terminal.scroll_viewport(ScrollViewport::Bottom);
                    did_anything = true;
                    force_full_publish = true;
                }
                Ok(WorkerMsg::Shutdown) => return,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }

        // 3b. Flush as much of the pending pty-output queue as the
        // kernel input buffer will take right now. If `try_write`
        // returns 0 the kernel is full — leave the rest queued; the
        // POLLOUT branch in step 5 will wake us when it drains.
        while !pending_out.is_empty() {
            let (a, b) = pending_out.as_slices();
            let slice = if !a.is_empty() { a } else { b };
            let n = pty.try_write(slice);
            if n == 0 {
                break;
            }
            pending_out.drain(..n);
            did_anything = true;
        }

        if hit_eof {
            if let Child::Active(pid) = child {
                child = Child::Exited(pid);
            }
            // Publish one last snapshot so the renderer sees `child_alive=false`.
            let (pc, pr, published) = publish_snapshot(
                &mut terminal,
                &mut render_state,
                &mut row_it,
                &mut cell_it,
                &snapshot,
                false,
                true,
                last_cols,
                last_rows,
            );
            last_cols = pc;
            last_rows = pr;
            if published {
                wake_winit();
            }
            // Don't exit; let the channel-disconnect or Shutdown end us.
            // Until then, just sleep — child is gone, no work to do.
            thread::sleep(Duration::from_millis(50));
            continue;
        }

        // 4. Publish snapshot if anything changed.
        if did_anything {
            let t = Instant::now();
            let (pc, pr, published) = publish_snapshot(
                &mut terminal,
                &mut render_state,
                &mut row_it,
                &mut cell_it,
                &snapshot,
                matches!(child, Child::Active(_)),
                force_full_publish,
                last_cols,
                last_rows,
            );
            last_cols = pc;
            last_rows = pr;
            publish_ns_since_log += t.elapsed().as_nanos();
            publishes_since_log += 1;
            // Wake the winit event loop so the Bevy renderer schedules
            // a frame. Without this, the loop sits idle in `Reactive`
            // mode and pty output never makes it to the screen.
            if published {
                wake_winit();
            }
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

        // 5. If we did work, loop right back to drain more bytes — under
        // heavy `cat` we want to keep chewing through the kernel pty
        // buffer as fast as possible. If we did nothing, block in
        // poll(2) until either the pty has data or the main thread
        // pokes our wake pipe. This is the difference between ~zero
        // idle CPU and a 2 kHz spin loop.
        if !did_anything {
            // Drain any wake bytes that accumulated while we were busy
            // — otherwise poll() would return immediately on stale data.
            let mut drain_buf = [0u8; 64];
            loop {
                match nix::unistd::read(&wake_r, &mut drain_buf) {
                    Ok(n) if n > 0 => continue,
                    _ => break,
                }
            }

            let pty_fd = pty.0.as_fd();
            let wake_fd = wake_r.as_fd();
            // If there are still bytes queued for the pty (a paste blob
            // that didn't fit in one kernel buffer), wait for POLLOUT
            // too so we wake the moment the shell drains some input.
            let pty_flags = if pending_out.is_empty() {
                PollFlags::POLLIN
            } else {
                PollFlags::POLLIN | PollFlags::POLLOUT
            };
            let mut fds = [
                PollFd::new(pty_fd, pty_flags),
                PollFd::new(wake_fd, PollFlags::POLLIN),
            ];
            // Cap the wait at a few seconds as a safety net — if a
            // wake somehow gets lost we still recover within a few
            // seconds instead of hanging. In practice every state
            // change pokes the pipe.
            let timeout = PollTimeout::try_from(5_000i32).unwrap();
            match poll(&mut fds, timeout) {
                Ok(_) => {}
                Err(Errno::EINTR) => {}
                Err(_) => {
                    // Unexpected — back off briefly so we don't spin if
                    // poll returns the same error every iteration.
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
    g.generation = g.generation.wrapping_add(1);
    (cols, rows, true)
}

/// Marker type for OwnedFd round-trip if needed elsewhere.
#[allow(dead_code)]
pub struct WorkerFd(pub OwnedFd);

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
