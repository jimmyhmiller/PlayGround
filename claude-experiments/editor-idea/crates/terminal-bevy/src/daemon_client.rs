//! Editor-side connection to a per-session daemon.
//!
//! Replaces what `Pty` used to be on the worker thread. The worker still
//! owns the libghostty `Terminal` and the render snapshot — it just
//! sources its bytes from a Unix socket instead of a PTY master.
//!
//! Construction either attaches to a live daemon for `session_id` or
//! re-execs the editor binary as `<self> --daemon <session_id> <cmd…>`
//! and waits for its socket to appear. Tests can override the daemon
//! binary via `TERMINAL_BEVY_DAEMON_BIN` (positional argv, no
//! `--daemon` flag — matches the standalone `terminal-daemon` bin).
//! The caller can inspect `attached_existing` afterwards: if true, the
//! daemon's `ReplayStart`/`Output…`/`ReplayEnd` is the authoritative
//! scrollback; if false, the worker should feed its on-disk replay log
//! into `vt_write` locally to recover history.

#![allow(unsafe_code)]

use std::collections::VecDeque;
use std::io::{Read as _, Write as _};
use std::os::fd::{AsFd, BorrowedFd};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use crate::daemon_proto::{decode, encode, ClientMessage, DaemonMessage};

/// Outbound frame buffer high-water mark — purely an early-warning so
/// runaway senders surface as something other than unbounded memory growth.
const OUT_BUF_SOFT_LIMIT: usize = 16 * 1024 * 1024;

/// How long we wait after fork-execing the daemon for its socket to
/// appear. Daemon does a double-fork + bind; usually a few ms.
const DAEMON_BOOT_TIMEOUT: Duration = Duration::from_secs(2);

pub struct DaemonClient {
    sock: UnixStream,
    /// Frames serialized but not yet pushed to the kernel socket buffer.
    out_buf: VecDeque<u8>,
    /// Partially-received bytes pending more data to form a full frame.
    in_buf: Vec<u8>,
    /// True iff `open` found a daemon already running for this session.
    /// False iff we had to fork-exec a fresh one.
    pub attached_existing: bool,
    session_id: u64,
}

impl DaemonClient {
    /// Connect to (or spawn + connect to) the daemon for `session_id`,
    /// send Attach{cols,rows}, and wait for the Attached ack. Subsequent
    /// frames flow through `poll_frames`.
    pub fn open(session_id: u64, cols: u16, rows: u16, command: Vec<String>) -> std::io::Result<Self> {
        let socket_path = crate::socket_path(session_id)
            .ok_or_else(|| std::io::Error::other("no data_dir (HOME unset?)"))?;

        let (sock, attached_existing) = match UnixStream::connect(&socket_path) {
            Ok(s) => (s, true),
            Err(_) => {
                spawn_daemon(session_id, command)?;
                let s = wait_for_socket(&socket_path, DAEMON_BOOT_TIMEOUT)?;
                (s, false)
            }
        };

        sock.set_nonblocking(false)?;
        let mut client = Self {
            sock,
            out_buf: VecDeque::new(),
            in_buf: Vec::new(),
            attached_existing,
            session_id,
        };
        client.send_blocking(&ClientMessage::Attach { cols, rows })?;
        client.await_attached()?;
        client.sock.set_nonblocking(true)?;
        Ok(client)
    }

    pub fn session_id(&self) -> u64 {
        self.session_id
    }

    pub fn as_fd(&self) -> BorrowedFd<'_> {
        self.sock.as_fd()
    }

    pub fn pending_out_empty(&self) -> bool {
        self.out_buf.is_empty()
    }

    /// Serialize and queue a message; does not write to the socket. Call
    /// `try_flush` to push bytes.
    pub fn send(&mut self, msg: &ClientMessage) {
        match encode(msg) {
            Ok(bytes) => {
                if self.out_buf.len() + bytes.len() > OUT_BUF_SOFT_LIMIT {
                    eprintln!(
                        "[daemon-client {}] out_buf grew past soft limit ({} B); daemon stuck?",
                        self.session_id,
                        self.out_buf.len()
                    );
                }
                self.out_buf.extend(bytes);
            }
            Err(e) => {
                eprintln!(
                    "[daemon-client {}] failed to encode {:?}: {}",
                    self.session_id, msg, e
                );
            }
        }
    }

    /// Non-blocking write of as much of `out_buf` as the kernel will
    /// take. Stops on WouldBlock without dropping the connection.
    pub fn try_flush(&mut self) {
        while !self.out_buf.is_empty() {
            let (a, b) = self.out_buf.as_slices();
            let slice = if !a.is_empty() { a } else { b };
            match self.sock.write(slice) {
                Ok(0) => break,
                Ok(n) => {
                    self.out_buf.drain(..n);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => {
                    // Connection broken — drop the queue. Caller will
                    // see EOF/error on the next read.
                    self.out_buf.clear();
                    return;
                }
            }
        }
    }

    /// Drain the socket non-blockingly, invoking `f` for each fully-
    /// decoded frame. Returns `Ok(false)` on clean EOF (daemon gone) or
    /// a fatal error; the caller should then stop using this client.
    pub fn poll_frames<F: FnMut(DaemonMessage)>(
        &mut self,
        mut f: F,
    ) -> std::io::Result<bool> {
        let mut tmp = [0u8; 8192];
        loop {
            match self.sock.read(&mut tmp) {
                Ok(0) => return Ok(false),
                Ok(n) => {
                    self.in_buf.extend_from_slice(&tmp[..n]);
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => return Ok(false),
            }
        }
        loop {
            match decode::<DaemonMessage>(&self.in_buf) {
                Ok(Some((msg, consumed))) => {
                    self.in_buf.drain(..consumed);
                    f(msg);
                }
                Ok(None) => break,
                Err(e) => {
                    eprintln!(
                        "[daemon-client {}] frame decode error: {}; dropping connection",
                        self.session_id, e
                    );
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Blocking write used during the Attach handshake. The socket is
    /// blocking at this point, so a single `write_all` is fine.
    fn send_blocking(&mut self, msg: &ClientMessage) -> std::io::Result<()> {
        let bytes = encode(msg).map_err(std::io::Error::other)?;
        self.sock.write_all(&bytes)
    }

    /// Read frames until the first `Attached`, draining everything else
    /// into `in_buf` so subsequent `poll_frames` calls see it.
    fn await_attached(&mut self) -> std::io::Result<()> {
        let deadline = Instant::now() + DAEMON_BOOT_TIMEOUT;
        let mut tmp = [0u8; 4096];
        loop {
            if Instant::now() >= deadline {
                return Err(std::io::Error::other(
                    "timed out waiting for daemon Attached ack",
                ));
            }
            match self.sock.read(&mut tmp) {
                Ok(0) => {
                    return Err(std::io::Error::other("daemon closed during handshake"));
                }
                Ok(n) => {
                    self.in_buf.extend_from_slice(&tmp[..n]);
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
            // Drain any complete frames; return the moment we see Attached.
            loop {
                match decode::<DaemonMessage>(&self.in_buf) {
                    Ok(Some((DaemonMessage::Attached, consumed))) => {
                        self.in_buf.drain(..consumed);
                        return Ok(());
                    }
                    Ok(Some((_other, _consumed))) => {
                        // Pre-Attached frames shouldn't exist but tolerate
                        // them by leaving them queued for the worker.
                        return Ok(());
                    }
                    Ok(None) => break,
                    Err(e) => return Err(std::io::Error::other(e)),
                }
            }
        }
    }
}

/// Fork-exec the daemon in the background. By default this re-execs
/// the editor's own binary with `--daemon <session_id> <cmd…>`; the
/// daemonize() inside the child's `terminal_daemon::daemon::run`
/// double-forks so the immediate child exits quickly. We `wait()` it
/// to avoid a zombie. Tests can point `TERMINAL_BEVY_DAEMON_BIN` at
/// the standalone `terminal-daemon` binary; in that mode argv is
/// positional with no `--daemon` flag.
fn spawn_daemon(session_id: u64, command: Vec<String>) -> std::io::Result<()> {
    let (bin, use_flag) = daemon_invocation()?;
    let mut cmd = Command::new(bin);
    if use_flag {
        cmd.arg("--daemon");
    }
    cmd.arg(session_id.to_string());
    cmd.args(&command);
    let mut child = cmd.spawn()?;
    let _ = child.wait();
    Ok(())
}

/// Resolve the daemon binary and whether to prepend `--daemon`. Honors
/// `TERMINAL_BEVY_DAEMON_BIN` (tests) — that path is the standalone
/// `terminal-daemon` and takes positional argv. Otherwise self-exec.
fn daemon_invocation() -> std::io::Result<(PathBuf, bool)> {
    if let Some(p) = std::env::var_os("TERMINAL_BEVY_DAEMON_BIN") {
        return Ok((PathBuf::from(p), false));
    }
    let exe = std::env::current_exe()?;
    Ok((exe, true))
}

/// Poll the filesystem + connect attempts until the daemon's socket is
/// usable or the deadline elapses.
fn wait_for_socket(path: &std::path::Path, timeout: Duration) -> std::io::Result<UnixStream> {
    let deadline = Instant::now() + timeout;
    let mut backoff = Duration::from_millis(5);
    loop {
        if path.exists() {
            if let Ok(sock) = UnixStream::connect(path) {
                return Ok(sock);
            }
        }
        if Instant::now() >= deadline {
            return Err(std::io::Error::other(format!(
                "daemon socket never appeared: {}",
                path.display()
            )));
        }
        std::thread::sleep(backoff);
        backoff = (backoff * 2).min(Duration::from_millis(50));
    }
}
