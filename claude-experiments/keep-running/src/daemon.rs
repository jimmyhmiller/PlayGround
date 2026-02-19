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

const MAX_BUFFER_SIZE: usize = 100 * 1024 * 1024; // 100MB
/// If the client's send buffer exceeds this, the client is stuck and we drop it.
const MAX_SEND_BUF: usize = 16 * 1024 * 1024; // 16MB

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
    /// Drops the client if the send buffer is too large (client is stuck).
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
                        self.client = None;
                        return;
                    }
                }
            }
        }

        // If send buffer is huge, client isn't draining — drop it
        if self.send_buf.len() > MAX_SEND_BUF {
            self.send_buf.clear();
            self.client = None;
        }
    }

    /// Buffer output data AND send to client if connected
    fn handle_output(&mut self, data: &[u8]) {
        // Always buffer output for session history
        self.buffer_data(data);

        // Also queue for client if connected
        let msg = DaemonMessage::Output(data.to_vec());
        self.queue_message(&msg);
    }

    fn buffer_data(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
        // Trim if too large (drop oldest)
        if self.buffer.len() > MAX_BUFFER_SIZE {
            let trim = self.buffer.len() - MAX_BUFFER_SIZE;
            self.buffer.drain(0..trim);
        }
    }

    /// Queue buffered output (session history) for replay to new client.
    /// Sends in chunks to avoid creating one giant JSON message.
    fn replay_buffer(&mut self) {
        if self.client.is_none() || self.buffer.is_empty() {
            return;
        }
        const CHUNK_SIZE: usize = 32 * 1024; // 32KB per message
        let mut offset = 0;
        while offset < self.buffer.len() {
            let end = (offset + CHUNK_SIZE).min(self.buffer.len());
            let chunk = self.buffer[offset..end].to_vec();
            offset = end;
            let msg = DaemonMessage::Output(chunk);
            if !self.queue_message(&msg) {
                break; // Client was dropped
            }
        }
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

        // Exec
        let err = Command::new(program).args(args).exec();
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
    // Track if PTY has been drained after child exit
    let mut pty_drained = false;
    // When child exited (for grace period)
    let mut child_exit_time: Option<std::time::Instant> = None;
    // True when a client received the ChildExited message and then disconnected
    let mut exit_delivered = false;

    let listener_fd = listener.as_raw_fd();

    // Grace period: keep daemon alive this long after child exits with no client,
    // so detached clients have time to reattach and see the output.
    const EXIT_GRACE_PERIOD: Duration = Duration::from_secs(30);

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

        // Daemon exits when child has exited, PTY is drained, no client connected, AND either:
        // - a client received the ChildExited notification and then disconnected, OR
        // - the grace period has elapsed (no one is coming back)
        if state.child_exited && pty_drained && state.client.is_none() {
            let should_exit = exit_delivered
                || child_exit_time
                    .map(|t| t.elapsed() >= EXIT_GRACE_PERIOD)
                    .unwrap_or(false);
            if should_exit {
                let _ = session::remove_session(&name);
                return Ok(());
            }
        }

        // Build poll fds: [pty_master, listener, client?]
        let mut pollfds: Vec<libc::pollfd> = Vec::with_capacity(3);

        // Always poll PTY master for readable data
        pollfds.push(libc::pollfd {
            fd: master_fd,
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

        // Use a longer timeout when idle, short when we have pending sends
        let timeout_ms = if !state.send_buf.is_empty() {
            10 // Short timeout when we have data to flush
        } else {
            500 // Longer timeout when idle; still wakes to check child status
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

        // Replay buffer after processing attach message
        if should_replay {
            state.replay_buffer();
        }

        if client_disconnected {
            // If we already sent ChildExited to this client, the info was delivered
            if exit_notified {
                exit_delivered = true;
            }
            state.client = None;
            state.send_buf.clear();
            client_msg_buf.clear();
        }

        // Flush queued data to client (non-blocking)
        state.flush_send_buf();
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

/// Run the daemon in foreground with configurable paths (for testing)
/// This runs in the current process without daemonizing
pub fn run_daemon_foreground(
    name: String,
    command: Vec<String>,
    session_dir: std::path::PathBuf,
    socket_dir: std::path::PathBuf,
) -> Result<()> {
    // Set the directory overrides for this thread
    session::set_session_dir(Some(session_dir));
    session::set_socket_dir(Some(socket_dir));

    // Run the daemon (will use the overridden paths)
    run_daemon(name, command)
}
