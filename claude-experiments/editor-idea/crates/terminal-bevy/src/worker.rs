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

use std::os::fd::OwnedFd;
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use libghostty_vt::{
    render::{CellIterator, Dirty, RenderState, RowIterator},
    style::RgbColor,
};
use nix::errno::Errno;

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
    /// Bytes to write to the pty (keystrokes, paste).
    Input(Vec<u8>),
    /// New grid dimensions (in cells) and per-cell pixel size. Worker
    /// resizes both the libghostty `Terminal` and the pty winsize.
    Resize {
        cols: u16,
        rows: u16,
        cell_w_px: u32,
        cell_h_px: u32,
    },
    Shutdown,
}

/// Handle the main thread keeps for each terminal.
pub struct WorkerHandle {
    pub snapshot: Arc<Mutex<GridSnapshot>>,
    pub tx: Sender<WorkerMsg>,
    /// Held to keep the thread joinable; we never actually `join` from
    /// the main thread — workers exit on `Shutdown` or PTY EOF.
    _join: JoinHandle<()>,
}

impl WorkerHandle {
    /// Spawn the worker. Forks the pty + child shell, then hands fd +
    /// ownership of `Terminal` to a fresh thread.
    pub fn spawn(size: PtySize, scrollback: usize) -> std::io::Result<Self> {
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

        let (tx, rx) = channel::<WorkerMsg>();

        let join = thread::Builder::new()
            .name("terminal-worker".into())
            .spawn(move || worker_loop(pty, child, size, scrollback, snapshot_w, rx))
            .expect("spawn worker");

        Ok(Self {
            snapshot,
            tx,
            _join: join,
        })
    }
}

/// Cap how many bytes the worker feeds to `vt_write` between snapshot
/// publishes. Same idea as Alacritty's `MAX_LOCKED_READ` — bound the
/// snapshot lock-hold time so the renderer never waits much.
const MAX_READ_PER_TICK: usize = 65536;

fn worker_loop(
    pty: Pty,
    mut child: Child,
    initial_size: PtySize,
    scrollback: usize,
    snapshot: Arc<Mutex<GridSnapshot>>,
    rx: Receiver<WorkerMsg>,
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

    let mut render_state = RenderState::new().expect("RenderState");
    let mut row_it = RowIterator::new().expect("RowIterator");
    let mut cell_it = CellIterator::new().expect("CellIterator");

    // Track last published dims so publish_snapshot can detect a resize
    // and force a full redraw — `g.cells` is row-major flat-indexed and
    // a cols change makes every linear index map to a different cell.
    let mut last_cols: u16 = initial_size.cols;
    let mut last_rows: u16 = initial_size.rows;

    let mut read_buf = [0u8; 65536];

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

        // 2. Flush VT-effect responses (DA replies, etc.) back to the pty.
        {
            let mut response = pty_response.borrow_mut();
            if !response.is_empty() {
                pty.write(&response);
                response.clear();
                did_anything = true;
            }
        }

        // 3. Drain channel messages from the main thread.
        loop {
            match rx.try_recv() {
                Ok(WorkerMsg::Input(bytes)) => {
                    pty.write(&bytes);
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
                Ok(WorkerMsg::Shutdown) => return,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }

        if hit_eof {
            if let Child::Active(pid) = child {
                child = Child::Exited(pid);
            }
            // Publish one last snapshot so the renderer sees `child_alive=false`.
            let (pc, pr) = publish_snapshot(
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
            // Don't exit; let the channel-disconnect or Shutdown end us.
            // Until then, just sleep — child is gone, no work to do.
            thread::sleep(Duration::from_millis(50));
            continue;
        }

        // 4. Publish snapshot if anything changed.
        if did_anything {
            let t = Instant::now();
            let (pc, pr) = publish_snapshot(
                &mut terminal,
                &mut render_state,
                &mut row_it,
                &mut cell_it,
                &snapshot,
                matches!(child, Child::Active(_)),
                false,
                last_cols,
                last_rows,
            );
            last_cols = pc;
            last_rows = pr;
            publish_ns_since_log += t.elapsed().as_nanos();
            publishes_since_log += 1;
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

        // 5. If we did nothing, take a tiny nap. Otherwise loop right
        // back to drain more bytes — under heavy `cat` we want to keep
        // chewing through the kernel pty buffer as fast as possible.
        if !did_anything {
            thread::sleep(Duration::from_micros(500));
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
) -> (u16, u16) {
    let snap = match render_state.update(terminal) {
        Ok(s) => s,
        Err(_) => return (prev_cols, prev_rows),
    };
    let dirty = snap.dirty().unwrap_or(Dirty::Full);
    let cols = snap.cols().unwrap_or(0);
    let rows = snap.rows().unwrap_or(0);
    let dims_changed = cols != prev_cols || rows != prev_rows;
    if matches!(dirty, Dirty::Clean) && !force_full && !dims_changed {
        // Still update child_alive flag without touching cells.
        let mut g = snapshot_arc.lock().expect("snapshot lock");
        if g.child_alive != child_alive {
            g.child_alive = child_alive;
            g.generation = g.generation.wrapping_add(1);
        }
        return (cols, rows);
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
            Err(_) => return (cols, rows),
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
    (cols, rows)
}

/// Marker type for OwnedFd round-trip if needed elsewhere.
#[allow(dead_code)]
pub struct WorkerFd(pub OwnedFd);
