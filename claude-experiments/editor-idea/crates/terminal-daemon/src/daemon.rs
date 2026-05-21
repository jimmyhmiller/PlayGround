//! Per-terminal-session daemon process.
//!
//! Owns a PTY + child shell + replay buffer; serves a single client at a
//! time over a Unix socket. Survives the editor process — when the editor
//! relaunches it reattaches and the daemon replays the prior output so
//! the screen comes back as it was.
//!
//! Adapted from `keep-running/src/daemon.rs`. We drop the human-readable
//! session naming (we key by `TerminalSession(u64)` instead) and use
//! `daemon_proto`'s bincode framing.
//!
//! ## Lifecycle
//! - On startup: write `pid_path(session_id)`, bind `socket_path(session_id)`,
//!   `forkpty` + exec the user's shell. Wait for an `Attach`.
//! - Each client connect kicks the previous (last-attach-wins). Output is
//!   buffered while no client is connected; first thing a new client sees
//!   after `Attached` is `ReplayStart` → chunks of history → `ReplayEnd`,
//!   followed by live output.
//! - On child exit: drain PTY, send `ChildExited`, wait for the client to
//!   disconnect (or a grace period), then quit and unlink the socket+pid.

#![allow(unsafe_code)]

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use nix::pty::{openpty, OpenptyResult, Winsize};
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::{setsid, Pid};

use crate::proto::{decode, encode, ClientMessage, DaemonMessage};

/// Rolling history buffer cap. When exceeded, oldest bytes are dropped.
/// Sized to cover typical scrollback (libghostty caps at 100k lines,
/// styled bytes average somewhere south of 200 B/line).
const MAX_BUFFER: usize = 64 * 1024 * 1024;

/// When `send_buf` exceeds this, stop queuing live output (client can't
/// keep up). The data is still in `buffer` for replay on reattach.
const MAX_SEND_BUF: usize = 4 * 1024 * 1024;

/// How long to keep the daemon alive after the child exits with no
/// client connected, so a detached editor has a chance to reattach and
/// see the exit code / final output.
const GRACE_PERIOD: Duration = Duration::from_secs(30);

/// How long `send_buf` can sit non-empty without making progress before
/// we treat the client as wedged and force-disconnect.
const ZOMBIE_TIMEOUT: Duration = Duration::from_secs(30);

/// Public entry point used by the `terminal-daemon` binary.
///
/// Daemonizes the current process (double-fork + setsid), opens the
/// listener, spawns the child, and runs the main loop until exit. Never
/// returns — the process exits when the daemon loop terminates.
pub fn run(session_id: u64, command: Vec<String>) -> ! {
    // Honour TERMINAL_DAEMON_FOREGROUND=1 for tests / debugging — skips
    // the double-fork so stderr stays attached. Production always
    // daemonizes.
    let foreground = std::env::var_os("TERMINAL_DAEMON_FOREGROUND").is_some();
    if !foreground {
        if let Err(e) = daemonize() {
            eprintln!("[terminal-daemon] daemonize failed: {}", e);
            std::process::exit(1);
        }
    }
    eprintln!(
        "[terminal-daemon {}] starting; pid={}, command={:?}",
        session_id,
        std::process::id(),
        command
    );
    match run_loop(session_id, command) {
        Ok(()) => {
            eprintln!("[terminal-daemon {}] exiting cleanly", session_id);
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("[terminal-daemon {}] error: {}", session_id, e);
            std::process::exit(1);
        }
    }
}

/// Double-fork + setsid to fully detach from the parent's controlling
/// terminal and process group. Standard Unix daemonize.
fn daemonize() -> std::io::Result<()> {
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if pid > 0 {
        std::process::exit(0);
    }

    setsid().map_err(std::io::Error::from)?;

    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if pid > 0 {
        std::process::exit(0);
    }

    // Redirect stdio. stdin/stdout always go to /dev/null. stderr goes
    // either to the path in TERMINAL_DAEMON_LOG (useful for tests +
    // debugging) or to /dev/null.
    let devnull = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/null")?;
    let dn_fd = devnull.as_raw_fd();
    let err_fd = if let Some(log_path) = std::env::var_os("TERMINAL_DAEMON_LOG") {
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            Ok(f) => {
                let raw = f.as_raw_fd();
                // Leak the file handle so its fd stays open for the
                // life of the process.
                std::mem::forget(f);
                raw
            }
            Err(_) => dn_fd,
        }
    } else {
        dn_fd
    };
    unsafe {
        libc::dup2(dn_fd, 0);
        libc::dup2(dn_fd, 1);
        libc::dup2(err_fd, 2);
    }
    Ok(())
}

/// Spawn `command` on a fresh PTY pair. Returns the master fd (kept open
/// in the parent) and the child PID.
fn spawn_in_pty(
    session_id: u64,
    command: &[String],
    cols: u16,
    rows: u16,
) -> std::io::Result<(i32, Pid)> {
    let winsize = Winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    let OpenptyResult { master, slave } =
        openpty(&winsize, None).map_err(std::io::Error::from)?;
    let master_fd = master.as_raw_fd();
    let slave_fd = slave.as_raw_fd();

    let (program, args) = command
        .split_first()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty command"))?;

    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(std::io::Error::last_os_error());
    }

    if pid == 0 {
        // Child
        drop(master);
        let _ = setsid();
        unsafe {
            libc::ioctl(slave_fd, libc::TIOCSCTTY as _, 0);
            libc::dup2(slave_fd, 0);
            libc::dup2(slave_fd, 1);
            libc::dup2(slave_fd, 2);
        }
        if slave_fd > 2 {
            drop(slave);
        }
        // Match what Pty::spawn used to set so the shell sees the same env.
        // EDITOR_IDEA_TERMINAL_SESSION_ID lets child processes (notably
        // claude-statusline-bridge) identify which of our terminal panes
        // they are running in, so per-terminal widgets can match by id.
        let err = Command::new(program)
            .args(args)
            .env("TERM", "xterm-256color")
            .env("EDITOR_IDEA_TERMINAL_SESSION_ID", session_id.to_string())
            .exec();
        eprintln!("[terminal-daemon] exec '{}' failed: {}", program, err);
        std::process::exit(127);
    }

    drop(slave);
    // Leak the master fd into our integer handle — we manage its lifetime
    // explicitly via close() at shutdown.
    std::mem::forget(master);
    Ok((master_fd, Pid::from_raw(pid)))
}

/// Cap on `pty_write_buf` (inbound, daemon → PTY). Pastes the size of
/// the whole `buffer` aren't a thing in practice; this is a sanity
/// stopgap so a runaway producer can't OOM the daemon.
const MAX_PTY_WRITE_BUF: usize = 16 * 1024 * 1024;

/// Internal daemon state. All members live on the single daemon thread.
struct State {
    buffer: Vec<u8>,
    client: Option<UnixStream>,
    send_buf: Vec<u8>,
    /// Bytes destined for the PTY master that haven't been accepted yet.
    /// The master fd is `O_NONBLOCK`, so big pastes can exceed the
    /// kernel's PTY input buffer; we hold the remainder here and drain
    /// it on POLLOUT instead of dropping it (which would strand the
    /// closing `\x1b[201~` and wedge the shell in bracketed-paste mode).
    pty_write_buf: VecDeque<u8>,
    master_fd: i32,
    child_pid: Pid,
    child_exited: bool,
    exit_code: Option<i32>,
    /// `Some(offset)` while replaying history; `None` outside the replay
    /// window.
    replay_offset: Option<usize>,
}

impl State {
    fn new(master_fd: i32, child_pid: Pid) -> Self {
        Self {
            buffer: Vec::new(),
            client: None,
            send_buf: Vec::new(),
            pty_write_buf: VecDeque::new(),
            master_fd,
            child_pid,
            child_exited: false,
            exit_code: None,
            replay_offset: None,
        }
    }

    /// Queue bytes for the PTY master and attempt an immediate drain.
    /// Bytes that can't be written right now stay in `pty_write_buf`
    /// and will be flushed on the next POLLOUT.
    fn enqueue_pty_write(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        if self.pty_write_buf.len() + data.len() > MAX_PTY_WRITE_BUF {
            eprintln!(
                "[daemon] pty_write_buf would exceed cap ({} + {} > {}); dropping input",
                self.pty_write_buf.len(),
                data.len(),
                MAX_PTY_WRITE_BUF
            );
            return;
        }
        self.pty_write_buf.extend(data.iter().copied());
        self.drain_pty_write();
    }

    /// Push as much of `pty_write_buf` as the kernel will accept right
    /// now. Stops on WouldBlock without losing data.
    fn drain_pty_write(&mut self) {
        while !self.pty_write_buf.is_empty() {
            let (a, b) = self.pty_write_buf.as_slices();
            let slice = if !a.is_empty() { a } else { b };
            let n = unsafe {
                libc::write(
                    self.master_fd,
                    slice.as_ptr() as *const libc::c_void,
                    slice.len(),
                )
            };
            if n < 0 {
                let err = std::io::Error::last_os_error();
                if err.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                if err.kind() == std::io::ErrorKind::WouldBlock {
                    return;
                }
                // Other errors: nothing we can do here. Leave the bytes
                // in the buffer; the next iteration will retry or give
                // up via the surrounding child-exit handling.
                return;
            }
            if n == 0 {
                return;
            }
            self.pty_write_buf.drain(..n as usize);
        }
    }

    fn queue<M: serde::Serialize>(&mut self, msg: &M) -> bool {
        if self.client.is_none() {
            return false;
        }
        match encode(msg) {
            Ok(bytes) => {
                self.send_buf.extend_from_slice(&bytes);
                true
            }
            Err(_) => false,
        }
    }

    /// Flush as much of `send_buf` as the kernel will take right now.
    /// Drops the client on a real write error.
    fn flush(&mut self) {
        if self.send_buf.is_empty() {
            return;
        }
        let Some(ref mut client) = self.client else {
            return;
        };
        loop {
            match client.write(&self.send_buf) {
                Ok(0) => break,
                Ok(n) => {
                    self.send_buf.drain(0..n);
                    if self.send_buf.is_empty() {
                        break;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => {
                    self.send_buf.clear();
                    self.replay_offset = None;
                    self.client = None;
                    return;
                }
            }
        }
    }

    fn record_output(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
        if self.buffer.len() > MAX_BUFFER {
            let trim = self.buffer.len() - MAX_BUFFER;
            self.buffer.drain(0..trim);
        }
        if self.client.is_some() && self.send_buf.len() <= MAX_SEND_BUF {
            let msg = DaemonMessage::Output(data.to_vec());
            self.queue(&msg);
        }
    }

    fn start_replay(&mut self) {
        if self.client.is_none() || self.buffer.is_empty() {
            return;
        }
        self.queue(&DaemonMessage::ReplayStart);
        self.replay_offset = Some(0);
    }

    /// Feed the next batch of replay bytes into `send_buf`. Pauses when
    /// `send_buf` is past the watermark so we don't outrun the client.
    fn pump_replay(&mut self) {
        const CHUNK: usize = 32 * 1024;
        const WATERMARK: usize = 256 * 1024;

        let Some(offset) = self.replay_offset else {
            return;
        };
        if self.client.is_none() {
            self.replay_offset = None;
            return;
        }
        if self.send_buf.len() > WATERMARK {
            return;
        }
        if offset >= self.buffer.len() {
            self.queue(&DaemonMessage::ReplayEnd);
            self.replay_offset = None;
            return;
        }
        let mut pos = offset;
        while pos < self.buffer.len() && self.send_buf.len() <= WATERMARK {
            let end = (pos + CHUNK).min(self.buffer.len());
            let chunk = self.buffer[pos..end].to_vec();
            pos = end;
            if !self.queue(&DaemonMessage::Output(chunk)) {
                self.replay_offset = None;
                return;
            }
        }
        self.replay_offset = Some(pos);
    }

    fn resize_pty(&self, cols: u16, rows: u16) {
        let ws = Winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe {
            libc::ioctl(self.master_fd, libc::TIOCSWINSZ as _, &ws);
        }
    }

    fn kill_child(&self) {
        let _ = nix::sys::signal::kill(self.child_pid, nix::sys::signal::SIGHUP);
    }
}

fn run_loop(session_id: u64, command: Vec<String>) -> std::io::Result<()> {
    // Prepare paths. Both are under data_dir()/sessions/.
    let socket_path = crate::socket_path(session_id)
        .ok_or_else(|| std::io::Error::other("no data_dir (HOME unset?)"))?;
    let pid_path = crate::pid_path(session_id)
        .ok_or_else(|| std::io::Error::other("no data_dir (HOME unset?)"))?;
    let inject_path = crate::inject_socket_path(session_id)
        .ok_or_else(|| std::io::Error::other("no data_dir (HOME unset?)"))?;
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Stale sockets from a previous (dead) daemon — clean up so bind() works.
    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }
    if inject_path.exists() {
        let _ = std::fs::remove_file(&inject_path);
    }

    let (master_fd, child_pid) = spawn_in_pty(session_id, &command, 80, 24)?;

    // Master non-blocking.
    unsafe {
        let flags = libc::fcntl(master_fd, libc::F_GETFL);
        libc::fcntl(master_fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
    }

    let listener = UnixListener::bind(&socket_path)?;
    listener.set_nonblocking(true)?;

    // Side-channel inject socket: anyone with access to the runtime
    // dir can connect, write raw bytes, and we'll forward them to the
    // PTY master. Lets external tools type into the session without
    // displacing whatever GUI is currently attached.
    let inject_listener = UnixListener::bind(&inject_path)?;
    inject_listener.set_nonblocking(true)?;

    write_pid_file(&pid_path)?;
    // Ensure we always tear down our footprint on exit.
    let cleanup = Cleanup {
        socket_path: socket_path.clone(),
        pid_path: pid_path.clone(),
        inject_path: inject_path.clone(),
    };

    let mut state = State::new(master_fd, child_pid);

    let mut pty_read_buf = [0u8; 65536];
    let mut client_read_buf = [0u8; 4096];
    let mut client_msg_buf: Vec<u8> = Vec::new();

    let mut exit_notified = false;
    let mut exit_notified_at: Option<Instant> = None;
    let mut pty_drained = false;
    let mut child_exit_time: Option<Instant> = None;
    let mut exit_delivered = false;
    let mut send_stalled_since: Option<Instant> = None;
    let mut prev_send_buf_len: usize = 0;
    let mut should_quit_after_flush = false;

    let listener_fd = listener.as_raw_fd();
    let inject_listener_fd = inject_listener.as_raw_fd();
    // In-flight ephemeral inject connections. We don't bind these to
    // the main client slot — they exist only long enough to drain
    // their bytes into the PTY, then they're dropped. Bounded
    // implicitly by `MAX_INJECT_CONNS`.
    let mut inject_conns: Vec<UnixStream> = Vec::new();
    const MAX_INJECT_CONNS: usize = 16;
    let mut inject_read_buf = [0u8; 4096];

    loop {
        // Reap child.
        if !state.child_exited {
            match waitpid(child_pid, Some(WaitPidFlag::WNOHANG)) {
                Ok(WaitStatus::Exited(_, code)) => {
                    state.child_exited = true;
                    state.exit_code = Some(code);
                    child_exit_time = Some(Instant::now());
                }
                Ok(WaitStatus::Signaled(_, _, _)) => {
                    state.child_exited = true;
                    state.exit_code = None;
                    child_exit_time = Some(Instant::now());
                }
                _ => {}
            }
        }

        // Termination decision.
        if state.child_exited && state.client.is_none() {
            let timed_out = child_exit_time
                .map(|t| t.elapsed() >= GRACE_PERIOD)
                .unwrap_or(false);
            if exit_delivered || timed_out {
                drop(cleanup);
                unsafe {
                    libc::close(master_fd);
                }
                return Ok(());
            }
        }
        if should_quit_after_flush && state.send_buf.is_empty() && state.client.is_none() {
            drop(cleanup);
            unsafe {
                libc::close(master_fd);
            }
            return Ok(());
        }

        // Poll fds.
        let mut pollfds: Vec<libc::pollfd> = Vec::with_capacity(3);
        let master_pollfd = if state.child_exited && pty_drained {
            -1
        } else {
            master_fd
        };
        let mut master_events = libc::POLLIN;
        if !state.pty_write_buf.is_empty() {
            master_events |= libc::POLLOUT;
        }
        pollfds.push(libc::pollfd {
            fd: master_pollfd,
            events: master_events,
            revents: 0,
        });
        pollfds.push(libc::pollfd {
            fd: listener_fd,
            events: libc::POLLIN,
            revents: 0,
        });
        pollfds.push(libc::pollfd {
            fd: inject_listener_fd,
            events: libc::POLLIN,
            revents: 0,
        });
        let inject_listener_idx = pollfds.len() - 1;
        // Each in-flight inject connection contributes one pollfd; we
        // remember the range so we can map revents back to indices.
        let inject_conns_start = pollfds.len();
        for c in &inject_conns {
            pollfds.push(libc::pollfd {
                fd: c.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            });
        }
        let client_poll_idx = if let Some(ref c) = state.client {
            let mut events = libc::POLLIN;
            if !state.send_buf.is_empty() {
                events |= libc::POLLOUT;
            }
            pollfds.push(libc::pollfd {
                fd: c.as_raw_fd(),
                events,
                revents: 0,
            });
            Some(pollfds.len() - 1)
        } else {
            None
        };

        let has_drainable_work =
            !state.send_buf.is_empty() || state.replay_offset.is_some();
        let timeout_ms = if has_drainable_work && state.client.is_some() {
            1
        } else {
            500
        };

        let poll_ret =
            unsafe { libc::poll(pollfds.as_mut_ptr(), pollfds.len() as libc::nfds_t, timeout_ms) };
        if poll_ret < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() != std::io::ErrorKind::Interrupted {
                std::thread::sleep(Duration::from_millis(10));
            }
            continue;
        }

        // Accept new client (last-wins: replaces any existing client).
        if pollfds[1].revents & libc::POLLIN != 0 {
            if let Ok((stream, _)) = listener.accept() {
                let _ = stream.set_nonblocking(true);
                state.client = Some(stream);
                state.send_buf.clear();
                client_msg_buf.clear();
                exit_notified = false;
                exit_notified_at = None;
                send_stalled_since = None;
                prev_send_buf_len = 0;
            }
        }

        // Accept inject-channel connection. We accept as many as
        // arrive in one tick, but we cap the in-flight set so a
        // misbehaving caller can't exhaust fds. Each connection is
        // ephemeral — read all bytes, write to PTY, drop.
        if pollfds[inject_listener_idx].revents & libc::POLLIN != 0 {
            while inject_conns.len() < MAX_INJECT_CONNS {
                match inject_listener.accept() {
                    Ok((stream, _)) => {
                        let _ = stream.set_nonblocking(true);
                        inject_conns.push(stream);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(_) => break,
                }
            }
        }

        // Drain bytes from each in-flight inject connection into the
        // PTY. Connections that hit EOF or error are removed.
        if !inject_conns.is_empty() {
            let mut keep: Vec<bool> = Vec::with_capacity(inject_conns.len());
            for (i, conn) in inject_conns.iter_mut().enumerate() {
                let pf_idx = inject_conns_start + i;
                let readable = pollfds
                    .get(pf_idx)
                    .map(|p| p.revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) != 0)
                    .unwrap_or(false);
                if !readable {
                    keep.push(true);
                    continue;
                }
                let mut alive = true;
                loop {
                    match conn.read(&mut inject_read_buf) {
                        Ok(0) => {
                            alive = false; // peer closed
                            break;
                        }
                        Ok(n) => {
                            state.enqueue_pty_write(&inject_read_buf[..n]);
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                        Err(_) => {
                            alive = false;
                            break;
                        }
                    }
                }
                keep.push(alive);
            }
            let mut iter = keep.into_iter();
            inject_conns.retain(|_| iter.next().unwrap_or(false));
        }

        // Drain pending writes to the PTY first, so any input collected
        // from clients or inject conns this tick gets flushed before we
        // do anything else. POLLOUT is requested only when we have data
        // queued; opportunistic drain here also clears anything that
        // sneaked in via `enqueue_pty_write` mid-tick.
        if pollfds[0].revents & libc::POLLOUT != 0 || !state.pty_write_buf.is_empty() {
            state.drain_pty_write();
        }

        // Drain PTY.
        if pollfds[0].revents & (libc::POLLIN | libc::POLLHUP) != 0 {
            loop {
                let n = unsafe {
                    libc::read(
                        master_fd,
                        pty_read_buf.as_mut_ptr() as *mut libc::c_void,
                        pty_read_buf.len(),
                    )
                };
                if n > 0 {
                    state.record_output(&pty_read_buf[..n as usize]);
                    pty_drained = false;
                } else if n < 0 {
                    let err = std::io::Error::last_os_error();
                    if err.kind() == std::io::ErrorKind::WouldBlock {
                        if state.child_exited {
                            pty_drained = true;
                        }
                    } else {
                        if !state.child_exited {
                            state.child_exited = true;
                            child_exit_time = Some(Instant::now());
                        }
                        pty_drained = true;
                    }
                    break;
                } else {
                    pty_drained = true;
                    break;
                }
            }
        } else if state.child_exited {
            pty_drained = true;
        }

        if state.child_exited && pty_drained && !exit_notified {
            let msg = DaemonMessage::ChildExited {
                code: state.exit_code,
            };
            state.queue(&msg);
            exit_notified = true;
            exit_notified_at = Some(Instant::now());
        }

        // Process client input.
        let mut client_disconnected = false;
        let mut should_replay = false;
        let mut resize_request: Option<(u16, u16)> = None;
        if let Some(idx) = client_poll_idx {
            let revents = pollfds[idx].revents;
            if revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) != 0 {
                if let Some(ref mut client) = state.client {
                    match client.read(&mut client_read_buf) {
                        Ok(0) => client_disconnected = true,
                        Ok(n) => {
                            client_msg_buf.extend_from_slice(&client_read_buf[..n]);
                            loop {
                                match decode::<ClientMessage>(&client_msg_buf) {
                                    Ok(Some((msg, consumed))) => {
                                        client_msg_buf.drain(0..consumed);
                                        match msg {
                                            ClientMessage::Attach { cols, rows } => {
                                                resize_request = Some((cols, rows));
                                                state.queue(&DaemonMessage::Attached);
                                                should_replay = true;
                                                if state.child_exited
                                                    && pty_drained
                                                    && !exit_notified
                                                {
                                                    let m = DaemonMessage::ChildExited {
                                                        code: state.exit_code,
                                                    };
                                                    state.queue(&m);
                                                    exit_notified = true;
                                                    exit_notified_at = Some(Instant::now());
                                                }
                                            }
                                            ClientMessage::Input(data) => {
                                                state.enqueue_pty_write(&data);
                                            }
                                            ClientMessage::Resize { cols, rows } => {
                                                resize_request = Some((cols, rows));
                                            }
                                            ClientMessage::Detach => {
                                                client_disconnected = true;
                                            }
                                            ClientMessage::Kill => {
                                                state.kill_child();
                                                // Reap + tear down on next iter.
                                                should_quit_after_flush = true;
                                                client_disconnected = true;
                                            }
                                        }
                                    }
                                    Ok(None) => break,
                                    Err(_) => {
                                        client_disconnected = true;
                                        break;
                                    }
                                }
                            }
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                        Err(_) => client_disconnected = true,
                    }
                }
            }
        }

        if let Some((cols, rows)) = resize_request {
            state.resize_pty(cols, rows);
        }
        if should_replay {
            state.start_replay();
        }
        state.pump_replay();

        if client_disconnected {
            if exit_notified {
                exit_delivered = true;
            }
            state.client = None;
            state.send_buf.clear();
            state.replay_offset = None;
            client_msg_buf.clear();
            send_stalled_since = None;
            prev_send_buf_len = 0;
            exit_notified_at = None;
        }

        state.flush();

        // Zombie detection variant A: stalled send_buf with a client.
        if state.client.is_some() && !state.send_buf.is_empty() {
            if state.send_buf.len() < prev_send_buf_len {
                send_stalled_since = None;
            } else {
                let started = send_stalled_since.get_or_insert_with(Instant::now);
                if started.elapsed() >= ZOMBIE_TIMEOUT {
                    if exit_notified {
                        exit_delivered = true;
                    }
                    state.client = None;
                    state.send_buf.clear();
                    state.replay_offset = None;
                    client_msg_buf.clear();
                    send_stalled_since = None;
                    exit_notified_at = None;
                }
            }
        } else {
            send_stalled_since = None;
        }
        prev_send_buf_len = state.send_buf.len();

        // Zombie detection variant B: ChildExited sent but client never closes.
        if state.client.is_some() && exit_notified {
            if let Some(t) = exit_notified_at {
                if t.elapsed() >= GRACE_PERIOD {
                    state.client = None;
                    state.send_buf.clear();
                    state.replay_offset = None;
                    client_msg_buf.clear();
                    exit_notified_at = None;
                }
            }
        }
    }
}

fn write_pid_file(p: &Path) -> std::io::Result<()> {
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(p, std::process::id().to_string().as_bytes())
}

struct Cleanup {
    socket_path: PathBuf,
    pid_path: PathBuf,
    inject_path: PathBuf,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket_path);
        let _ = std::fs::remove_file(&self.pid_path);
        let _ = std::fs::remove_file(&self.inject_path);
    }
}
