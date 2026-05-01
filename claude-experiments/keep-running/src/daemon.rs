use crate::protocol::{decode_message, encode_message, ClientMessage, DaemonMessage};
use crate::session::{self, SessionInfo};
use anyhow::{Context, Result};
use nix::pty::{openpty, OpenptyResult, Winsize};
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::{setsid, Pid};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::time::Duration;

const MAX_BUFFER_SIZE: usize = 5 * 1024 * 1024; // 5MB scrollback history
/// When send_buf exceeds this, stop queuing new live output (client can't keep up).
/// The data is still saved in the history buffer for replay on reattach.
const MAX_SEND_BUF: usize = 4 * 1024 * 1024; // 4MB

/// Shared state for the daemon
struct DaemonState {
    /// Output buffer for replay on reattach
    buffer: Vec<u8>,
    /// Current client connection (if any)
    client: Option<UnixStream>,
    /// Outgoing data queued for the client
    send_buf: Vec<u8>,
    /// Child process ID (for informational purposes)
    #[allow(dead_code)]
    child_pid: Pid,
    /// Whether child has exited
    child_exited: bool,
    /// Child exit code
    exit_code: Option<i32>,
    /// PTY master fd
    master_fd: i32,
    /// Replay cursor: how far into `buffer` we've sent for the current replay.
    /// None means no replay in progress.
    replay_offset: Option<usize>,
}

impl DaemonState {
    fn new(child_pid: Pid, master_fd: i32) -> Self {
        Self {
            buffer: Vec::new(),
            client: None,
            send_buf: Vec::new(),
            child_pid,
            child_exited: false,
            exit_code: None,
            master_fd,
            replay_offset: None,
        }
    }

    /// Queue an encoded message for sending to the client.
    /// Returns false if client was dropped (send buffer overflow or encode error).
    fn queue_message(&mut self, msg: &DaemonMessage) -> bool {
        if self.client.is_none() {
            return false;
        }
        match encode_message(msg) {
            Ok(encoded) => {
                self.send_buf.extend_from_slice(&encoded);
                true
            }
            Err(_) => false,
        }
    }

    /// Flush as much of send_buf as possible using non-blocking writes.
    /// Drops the client if the send buffer is too large and no replay is in progress.
    fn flush_send_buf(&mut self) {
        if self.send_buf.is_empty() || self.client.is_none() {
            return;
        }

        if let Some(ref mut client) = self.client {
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
                        // Real write error — client is gone
                        self.send_buf.clear();
                        self.replay_offset = None;
                        self.client = None;
                        return;
                    }
                }
            }
        }

    }

    /// Buffer output data AND send to client if connected.
    /// If the client can't keep up (send_buf too large), we skip sending
    /// live output — the child process is never blocked. The data is still
    /// in the history buffer and the client will catch up on reattach.
    fn handle_output(&mut self, data: &[u8]) {
        // Always buffer output for session history
        self.buffer_data(data);

        // Only queue for client if send_buf isn't backed up
        if self.send_buf.len() <= MAX_SEND_BUF {
            let msg = DaemonMessage::Output(data.to_vec());
            self.queue_message(&msg);
        }
    }

    fn buffer_data(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
        // Trim if too large (drop oldest)
        if self.buffer.len() > MAX_BUFFER_SIZE {
            let trim = self.buffer.len() - MAX_BUFFER_SIZE;
            self.buffer.drain(0..trim);
        }
    }

    /// Begin an incremental replay of the session history buffer.
    /// Sends a ReplayStart message and sets up the replay cursor.
    /// Actual data is fed via pump_replay() each poll iteration.
    fn start_replay(&mut self) {
        if self.client.is_none() || self.buffer.is_empty() {
            return;
        }
        self.queue_message(&DaemonMessage::ReplayStart);
        self.replay_offset = Some(0);
    }

    /// Feed the next batch of replay data into send_buf.
    /// Called each poll iteration while a replay is in progress.
    /// Only queues more data when send_buf is below a threshold so we
    /// don't accumulate faster than the client can drain.
    fn pump_replay(&mut self) {
        const CHUNK_SIZE: usize = 32 * 1024; // 32KB per message
        /// How much we allow in send_buf before pausing replay
        const REPLAY_SEND_WATERMARK: usize = 256 * 1024; // 256KB

        let offset = match self.replay_offset {
            Some(o) => o,
            None => return,
        };

        if self.client.is_none() {
            self.replay_offset = None;
            return;
        }

        // Don't queue more if send_buf is already full enough
        if self.send_buf.len() > REPLAY_SEND_WATERMARK {
            return;
        }

        if offset >= self.buffer.len() {
            // Replay complete
            self.queue_message(&DaemonMessage::ReplayEnd);
            self.replay_offset = None;
            return;
        }

        // Queue a batch of chunks up to the watermark
        let mut pos = offset;
        while pos < self.buffer.len() && self.send_buf.len() <= REPLAY_SEND_WATERMARK {
            let end = (pos + CHUNK_SIZE).min(self.buffer.len());
            let chunk = self.buffer[pos..end].to_vec();
            pos = end;
            let msg = DaemonMessage::Output(chunk);
            if !self.queue_message(&msg) {
                self.replay_offset = None;
                return;
            }
        }
        self.replay_offset = Some(pos);
    }

    /// Resize the PTY
    fn resize_pty(&self, cols: u16, rows: u16) {
        let winsize = Winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe {
            libc::ioctl(self.master_fd, libc::TIOCSWINSZ as _, &winsize);
        }
    }
}

/// Daemonize the current process
pub fn daemonize() -> Result<()> {
    // First fork
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(std::io::Error::last_os_error()).context("First fork failed");
    }
    if pid > 0 {
        // Parent exits
        std::process::exit(0);
    }

    // Create new session
    setsid().context("setsid failed")?;

    // Second fork (prevent acquiring controlling terminal)
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(std::io::Error::last_os_error()).context("Second fork failed");
    }
    if pid > 0 {
        // First child exits
        std::process::exit(0);
    }

    // Close stdin/stdout/stderr and redirect to /dev/null
    let devnull = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/null")?;
    let fd = devnull.as_raw_fd();
    unsafe {
        libc::dup2(fd, 0);
        libc::dup2(fd, 1);
        libc::dup2(fd, 2);
    }

    Ok(())
}

/// Spawn command in a PTY, return (master_fd, child_pid)
fn spawn_in_pty(command: &[String], cols: u16, rows: u16) -> Result<(i32, Pid)> {
    let winsize = Winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };

    let OpenptyResult { master, slave } = openpty(&winsize, None).context("Failed to open PTY")?;

    let master_fd = master.as_raw_fd();
    let slave_fd = slave.as_raw_fd();

    let (program, args) = command.split_first().context("Empty command")?;

    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(std::io::Error::last_os_error()).context("Failed to fork");
    }

    if pid == 0 {
        // Child process
        // Close master
        drop(master);

        // Create new session
        let _ = setsid();

        // Set slave as controlling terminal
        unsafe {
            libc::ioctl(slave_fd, libc::TIOCSCTTY as _, 0);
        }

        // Dup slave to stdin/stdout/stderr
        unsafe {
            libc::dup2(slave_fd, 0);
            libc::dup2(slave_fd, 1);
            libc::dup2(slave_fd, 2);
        }

        if slave_fd > 2 {
            drop(slave);
        }

        // Exec — set KEEP_RUNNING so nested sessions can be detected
        let err = Command::new(program)
            .args(args)
            .env("KEEP_RUNNING", "1")
            .exec();
        eprintln!("Failed to exec '{}': {}", program, err);
        std::process::exit(1);
    }

    // Parent
    drop(slave);

    // Forget master so it doesn't get closed when OwnedFd drops
    std::mem::forget(master);

    Ok((master_fd, Pid::from_raw(pid)))
}

/// Run the daemon for a session
pub fn run_daemon(name: String, command: Vec<String>) -> Result<()> {
    // Default size - will be updated when client attaches
    let cols = 80;
    let rows = 24;

    // Create PTY and spawn child
    let (master_fd, child_pid) = spawn_in_pty(&command, cols, rows)?;

    // Set up socket
    let socket_path = session::socket_path(&name)?;
    if socket_path.exists() {
        std::fs::remove_file(&socket_path)?;
    }

    let listener = UnixListener::bind(&socket_path).context("Failed to bind socket")?;
    listener.set_nonblocking(true)?;

    // Save session info
    let info = SessionInfo {
        name: name.clone(),
        command: command.clone(),
        pid: std::process::id(),
        created_at: session::timestamp(),
        socket_path: socket_path.to_string_lossy().to_string(),
    };
    session::save_session(&info)?;

    // Initialize state
    let mut state = DaemonState::new(child_pid, master_fd);

    // Set PTY master to non-blocking
    unsafe {
        let flags = libc::fcntl(master_fd, libc::F_GETFL);
        libc::fcntl(master_fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
    }

    let mut pty_read_buf = [0u8; 4096];
    let mut client_read_buf = [0u8; 4096];
    let mut client_msg_buf = Vec::new();

    // Track whether we've notified the client about exit
    let mut exit_notified = false;
    // When we queued ChildExited for the current client. A well-behaved client
    // disconnects shortly after receiving this; if the connection is still alive
    // `grace_period` later, the peer is a zombie.
    let mut exit_notified_at: Option<std::time::Instant> = None;
    // Track if PTY has been drained after child exit
    let mut pty_drained = false;
    // When child exited (for grace period)
    let mut child_exit_time: Option<std::time::Instant> = None;
    // True when a client received the ChildExited message and then disconnected
    let mut exit_delivered = false;
    // Zombie-client detection: when did `send_buf` last fail to make progress?
    // If a peer connects but never reads (SIGSTOP'd, kernel-buffer-deadlocked,
    // or just abandoned), `send_buf` accumulates and `flush_send_buf` keeps
    // returning WouldBlock without an error. Without this, the daemon would
    // think a client is attached forever.
    let mut send_stalled_since: Option<std::time::Instant> = None;
    let mut prev_send_buf_len: usize = 0;

    let listener_fd = listener.as_raw_fd();

    // Grace period: keep daemon alive this long after child exits with no client,
    // so detached clients have time to reattach and see the output.
    // Tunable via KEEP_RUNNING_GRACE_SECS so tests don't have to wait 30s.
    let grace_period: Duration = std::env::var("KEEP_RUNNING_GRACE_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(30));

    // Zombie-client timeout: how long send_buf can sit non-empty without making
    // progress before we treat the peer as unresponsive and force-disconnect.
    // Tunable via KEEP_RUNNING_ZOMBIE_SECS for tests.
    let zombie_timeout: Duration = std::env::var("KEEP_RUNNING_ZOMBIE_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(30));

    loop {
        // Check if child has exited
        if !state.child_exited {
            match waitpid(child_pid, Some(WaitPidFlag::WNOHANG)) {
                Ok(WaitStatus::Exited(_, code)) => {
                    state.child_exited = true;
                    state.exit_code = Some(code);
                    child_exit_time = Some(std::time::Instant::now());
                }
                Ok(WaitStatus::Signaled(_, _, _)) => {
                    state.child_exited = true;
                    state.exit_code = None;
                    child_exit_time = Some(std::time::Instant::now());
                }
                _ => {}
            }
        }

        // Daemon exits when child has exited, no client connected, AND either:
        // - a client received the ChildExited notification and then disconnected, OR
        // - the grace period has elapsed (no one is coming back).
        //
        // We deliberately do not require `pty_drained` here. If the PTY hasn't
        // drained yet (e.g. POLLHUP semantics differ across platforms) we still
        // want to exit once the grace window is up — anything we haven't read
        // by then is lost regardless. Previously this condition could deadlock
        // the daemon at ~95% CPU forever if `pty_drained` never flipped.
        if state.child_exited && state.client.is_none() {
            let should_exit = exit_delivered
                || child_exit_time
                    .map(|t| t.elapsed() >= grace_period)
                    .unwrap_or(false);
            if should_exit {
                let _ = session::remove_session(&name);
                return Ok(());
            }
        }

        // Build poll fds: [pty_master, listener, client?]
        let mut pollfds: Vec<libc::pollfd> = Vec::with_capacity(3);

        // Poll PTY master for readable data — never block the child.
        // Once the child has exited AND the PTY is drained, set fd to -1
        // (poll() ignores negative fds). Otherwise POLLHUP keeps firing on
        // the closed slave end and the loop spins at 100% CPU during the
        // grace period. We still keep the slot in pollfds so subsequent
        // index references (pollfds[0]) stay valid.
        let master_pollfd = if state.child_exited && pty_drained {
            -1
        } else {
            master_fd
        };
        pollfds.push(libc::pollfd {
            fd: master_pollfd,
            events: libc::POLLIN,
            revents: 0,
        });

        // Always poll listener for new connections
        pollfds.push(libc::pollfd {
            fd: listener_fd,
            events: libc::POLLIN,
            revents: 0,
        });

        // Poll client if connected
        let client_poll_idx = if let Some(ref client) = state.client {
            let mut events = libc::POLLIN;
            if !state.send_buf.is_empty() {
                events |= libc::POLLOUT;
            }
            pollfds.push(libc::pollfd {
                fd: client.as_raw_fd(),
                events,
                revents: 0,
            });
            Some(pollfds.len() - 1)
        } else {
            None
        };

        // Use a longer timeout when idle, short when we have pending sends or replay.
        // The short timeout is only useful when there's actually a client to drain
        // `send_buf` into — otherwise we'd burn ~95% CPU spinning on a buffer with
        // no consumer (this is what caused stuck daemons after a child exited but
        // a queued ChildExited message had no client to receive it).
        let has_drainable_work =
            !state.send_buf.is_empty() || state.replay_offset.is_some();
        let timeout_ms = if has_drainable_work && state.client.is_some() {
            1
        } else {
            500
        };

        let poll_ret = unsafe {
            libc::poll(pollfds.as_mut_ptr(), pollfds.len() as libc::nfds_t, timeout_ms)
        };

        // poll error (not EINTR)
        if poll_ret < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() != std::io::ErrorKind::Interrupted {
                // Unexpected poll error, brief sleep and retry
                std::thread::sleep(Duration::from_millis(10));
            }
            continue;
        }

        // Try to accept new connection (if listener is readable or on timeout)
        if pollfds[1].revents & libc::POLLIN != 0 {
            if let Ok((stream, _)) = listener.accept() {
                stream.set_nonblocking(true).ok();
                state.client = Some(stream);
                state.send_buf.clear();
                client_msg_buf.clear();
                exit_notified = false;
                exit_notified_at = None;
            }
        }

        // Read from PTY
        if pollfds[0].revents & (libc::POLLIN | libc::POLLHUP) != 0 {
            loop {
                let pty_read = unsafe {
                    libc::read(
                        master_fd,
                        pty_read_buf.as_mut_ptr() as *mut libc::c_void,
                        pty_read_buf.len(),
                    )
                };

                if pty_read > 0 {
                    let data = &pty_read_buf[..pty_read as usize];
                    state.handle_output(data);
                    pty_drained = false;
                } else if pty_read < 0 {
                    let err = std::io::Error::last_os_error();
                    if err.kind() == std::io::ErrorKind::WouldBlock {
                        if state.child_exited {
                            pty_drained = true;
                        }
                    } else {
                        if !state.child_exited {
                            state.child_exited = true;
                        }
                        pty_drained = true;
                    }
                    break;
                } else {
                    // EOF
                    pty_drained = true;
                    break;
                }
            }
        } else if state.child_exited {
            // Child exited and poll didn't report PTY readable — it's drained
            pty_drained = true;
        }

        // Send ChildExited only after PTY is drained
        if state.child_exited && pty_drained && !exit_notified {
            let msg = DaemonMessage::ChildExited {
                code: state.exit_code,
            };
            state.queue_message(&msg);
            exit_notified = true;
            exit_notified_at = Some(std::time::Instant::now());
        }

        // Read from client
        let mut client_disconnected = false;
        let mut should_replay = false;
        let mut resize_request: Option<(u16, u16)> = None;

        if let Some(idx) = client_poll_idx {
            let client_revents = pollfds[idx].revents;

            if client_revents & (libc::POLLIN | libc::POLLHUP | libc::POLLERR) != 0 {
                if let Some(ref mut client) = state.client {
                    match client.read(&mut client_read_buf) {
                        Ok(0) => {
                            client_disconnected = true;
                        }
                        Ok(n) => {
                            client_msg_buf.extend_from_slice(&client_read_buf[..n]);

                            // Process messages
                            loop {
                                match decode_message::<ClientMessage>(&client_msg_buf) {
                                    Ok(Some((msg, consumed))) => {
                                        client_msg_buf.drain(0..consumed);

                                        match msg {
                                            ClientMessage::Attach { cols, rows } => {
                                                resize_request = Some((cols, rows));
                                                state.queue_message(&DaemonMessage::Attached);
                                                should_replay = true;

                                                if state.child_exited && pty_drained && !exit_notified {
                                                    let msg = DaemonMessage::ChildExited {
                                                        code: state.exit_code,
                                                    };
                                                    state.queue_message(&msg);
                                                    exit_notified = true;
                                                    exit_notified_at = Some(std::time::Instant::now());
                                                }
                                            }
                                            ClientMessage::Input(data) => {
                                                unsafe {
                                                    libc::write(
                                                        master_fd,
                                                        data.as_ptr() as *const libc::c_void,
                                                        data.len(),
                                                    );
                                                }
                                            }
                                            ClientMessage::Resize { cols, rows } => {
                                                resize_request = Some((cols, rows));
                                            }
                                            ClientMessage::Detach => {
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
                        Err(_) => {
                            client_disconnected = true;
                        }
                    }
                }
            }
        }

        // Handle resize outside of borrow
        if let Some((cols, rows)) = resize_request {
            state.resize_pty(cols, rows);
        }

        // Start incremental replay after processing attach message
        if should_replay {
            state.start_replay();
        }

        // Feed more replay data if a replay is in progress
        state.pump_replay();

        if client_disconnected {
            // If we already sent ChildExited to this client, the info was delivered
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

        // Flush queued data to client (non-blocking)
        state.flush_send_buf();

        // Zombie-client detection (variant A): the peer connected but our
        // `send_buf` is stuck — the kernel-side socket buffer filled up and
        // they aren't reading. Happens when a client is SIGSTOP'd or wedged.
        if state.client.is_some() && !state.send_buf.is_empty() {
            if state.send_buf.len() < prev_send_buf_len {
                send_stalled_since = None;
            } else {
                let started = send_stalled_since.get_or_insert_with(std::time::Instant::now);
                if started.elapsed() >= zombie_timeout {
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

        // Zombie-client detection (variant B): we sent ChildExited and the
        // peer never disconnected. Real clients show a final status and exit
        // within milliseconds of receiving this; anyone still here a full
        // grace period later is wedged. This is the failure mode that lets
        // a backgrounded `keep-running run -- <cmd> &` (the client gets
        // SIGTTIN-stopped on its first stdin read) hold the daemon open
        // forever — `send_buf` empties because the kernel buffer absorbed
        // the small ChildExited frame, so variant A above never triggers.
        if state.client.is_some() && exit_notified {
            if let Some(t) = exit_notified_at {
                if t.elapsed() >= grace_period {
                    state.client = None;
                    state.send_buf.clear();
                    state.replay_offset = None;
                    client_msg_buf.clear();
                    exit_notified_at = None;
                    // Don't set exit_delivered — we don't know the peer
                    // actually saw the message. Cleanup will fall through
                    // to the grace-period branch above.
                }
            }
        }
    }
}

/// Fork and run daemon in background
pub fn start_daemon(name: String, command: Vec<String>) -> Result<()> {
    let pid = unsafe { libc::fork() };

    if pid < 0 {
        return Err(std::io::Error::last_os_error()).context("Fork failed");
    }

    if pid == 0 {
        // Child - become daemon
        if let Err(e) = daemonize() {
            eprintln!("Daemonize failed: {}", e);
            std::process::exit(1);
        }

        if let Err(_e) = run_daemon(name, command) {
            std::process::exit(1);
        }

        std::process::exit(0);
    }

    // Parent - wait for daemon to set up
    std::thread::sleep(Duration::from_millis(200));

    Ok(())
}

