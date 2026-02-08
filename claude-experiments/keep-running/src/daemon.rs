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
const WRITE_TIMEOUT: Duration = Duration::from_millis(200);

/// Write to client socket with a timeout to prevent deadlocking the daemon.
/// Uses blocking mode with SO_SNDTIMEO so write_all works correctly.
fn write_with_timeout(client: &mut UnixStream, data: &[u8]) -> std::io::Result<()> {
    client.set_nonblocking(false)?;
    client.set_write_timeout(Some(WRITE_TIMEOUT))?;
    let result = client.write_all(data);
    let _ = client.set_write_timeout(None);
    let _ = client.set_nonblocking(true);
    result
}

/// Shared state for the daemon
struct DaemonState {
    /// Output buffer for replay on reattach
    buffer: Vec<u8>,
    /// Current client connection (if any)
    client: Option<UnixStream>,
    /// Child process ID (for informational purposes)
    #[allow(dead_code)]
    child_pid: Pid,
    /// Whether child has exited
    child_exited: bool,
    /// Child exit code
    exit_code: Option<i32>,
    /// PTY master fd
    master_fd: i32,
    /// Whether a client has seen the exit notification
    client_saw_exit: bool,
}

impl DaemonState {
    fn new(child_pid: Pid, master_fd: i32) -> Self {
        Self {
            buffer: Vec::new(),
            client: None,
            child_pid,
            child_exited: false,
            exit_code: None,
            master_fd,
            client_saw_exit: false,
        }
    }

    /// Buffer output data AND send to client if connected
    fn handle_output(&mut self, data: &[u8]) {
        // Always buffer output for session history
        self.buffer_data(data);

        // Also send to client if connected
        if let Some(ref mut client) = self.client {
            let msg = DaemonMessage::Output(data.to_vec());
            if let Ok(encoded) = encode_message(&msg) {
                if write_with_timeout(client, &encoded).is_err() {
                    // Client disconnected or write timed out
                    self.client = None;
                }
            }
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

    /// Send buffered output (session history) to new client
    fn replay_buffer(&mut self) -> Result<()> {
        if let Some(ref mut client) = self.client {
            if !self.buffer.is_empty() {
                let msg = DaemonMessage::Output(self.buffer.clone());
                let encoded = encode_message(&msg)?;
                // Use a longer timeout for replay since buffer can be large
                let _ = client.set_nonblocking(false);
                let _ = client.set_write_timeout(Some(Duration::from_secs(5)));
                let result = client.write_all(&encoded);
                let _ = client.set_write_timeout(None)?;
                let _ = client.set_nonblocking(true);
                result?;
                // Don't clear buffer - it's the session history
            }
        }
        Ok(())
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

    loop {
        // Check if child has exited
        if !state.child_exited {
            match waitpid(child_pid, Some(WaitPidFlag::WNOHANG)) {
                Ok(WaitStatus::Exited(_, code)) => {
                    state.child_exited = true;
                    state.exit_code = Some(code);
                    // Don't notify client yet - drain PTY first
                }
                Ok(WaitStatus::Signaled(_, _, _)) => {
                    state.child_exited = true;
                    state.exit_code = None;
                    // Don't notify client yet - drain PTY first
                }
                _ => {}
            }
        }

        // If child exited and client disconnected after seeing it, daemon exits
        // (client_saw_exit is set when we send ChildExited and client later disconnects)
        if state.child_exited && state.client.is_none() && state.client_saw_exit {
            let _ = session::remove_session(&name);
            return Ok(());
        }

        // Try to accept new connection
        if let Ok((stream, _)) = listener.accept() {
            stream.set_nonblocking(true).ok();
            state.client = Some(stream);
            client_msg_buf.clear();
            // Reset exit notification for new client
            exit_notified = false;
        }

        // Read from PTY
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
            pty_drained = false; // More data might be coming
        } else if pty_read < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::WouldBlock {
                // No data available right now
                if state.child_exited {
                    pty_drained = true;
                }
            } else {
                // PTY error - child probably exited
                if !state.child_exited {
                    state.child_exited = true;
                }
                pty_drained = true;
            }
        } else {
            // pty_read == 0 means EOF
            pty_drained = true;
        }

        // Send ChildExited only after PTY is drained
        if state.child_exited && pty_drained && !exit_notified {
            if let Some(ref mut client) = state.client {
                let msg = DaemonMessage::ChildExited {
                    code: state.exit_code,
                };
                if let Ok(encoded) = encode_message(&msg) {
                    if write_with_timeout(client, &encoded).is_ok() {
                        state.client_saw_exit = true;
                    }
                }
            }
            exit_notified = true;
        }

        // Read from client
        let mut client_disconnected = false;
        let mut should_replay = false;
        let mut resize_request: Option<(u16, u16)> = None;
        let mut notified_exit = false;

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

                                        // Send attached confirmation
                                        let reply = DaemonMessage::Attached;
                                        if let Ok(encoded) = encode_message(&reply) {
                                            let _ = write_with_timeout(client, &encoded);
                                        }

                                        // Mark that we need to replay buffer
                                        should_replay = true;

                                        // If child already exited and PTY drained, notify client
                                        // Otherwise the main loop will send it after draining
                                        if state.child_exited && pty_drained && !exit_notified {
                                            let msg = DaemonMessage::ChildExited {
                                                code: state.exit_code,
                                            };
                                            if let Ok(encoded) = encode_message(&msg) {
                                                if write_with_timeout(client, &encoded).is_ok() {
                                                    notified_exit = true;
                                                }
                                            }
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
                            Ok(None) => break, // Need more data
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

        // Handle resize outside of borrow
        if let Some((cols, rows)) = resize_request {
            state.resize_pty(cols, rows);
        }

        // Replay buffer after processing attach message
        if should_replay {
            let _ = state.replay_buffer();
        }

        // Track if client was notified about exit
        if notified_exit {
            state.client_saw_exit = true;
            exit_notified = true;
        }

        if client_disconnected {
            state.client = None;
            client_msg_buf.clear();
        }

        // Small sleep to avoid busy-waiting
        std::thread::sleep(Duration::from_millis(1));
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
