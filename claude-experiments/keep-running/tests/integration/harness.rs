//! Test execution harness
//!
//! Provides TestContext for isolated test environments and TestClient for
//! communicating with the daemon.

use crate::parser::{Action, ExpectedFinalState, Scenario};
use keep_running::protocol::{decode_message, encode_message, ClientMessage, DaemonMessage};
use regex::Regex;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Global counter for unique session names
static SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Test context providing isolated directories and daemon management
pub struct TestContext {
    #[allow(dead_code)]
    pub temp_dir: TempDir,
    pub session_dir: PathBuf,
    pub socket_dir: PathBuf,
    pub session_name: String,
    daemon_process: Option<Child>,
}

impl TestContext {
    /// Create a new isolated test context
    pub fn new() -> Result<Self, String> {
        let temp_dir = TempDir::new().map_err(|e| format!("Failed to create temp dir: {}", e))?;

        let session_dir = temp_dir.path().join("sessions");
        let socket_dir = temp_dir.path().join("sockets");

        std::fs::create_dir_all(&session_dir)
            .map_err(|e| format!("Failed to create session dir: {}", e))?;
        std::fs::create_dir_all(&socket_dir)
            .map_err(|e| format!("Failed to create socket dir: {}", e))?;

        // Generate unique session name
        let counter = SESSION_COUNTER.fetch_add(1, Ordering::SeqCst);
        let session_name = format!("test-session-{}-{}", std::process::id(), counter);

        Ok(Self {
            temp_dir,
            session_dir,
            socket_dir,
            session_name,
            daemon_process: None,
        })
    }

    /// Get the socket path for the current session
    pub fn socket_path(&self) -> PathBuf {
        self.socket_dir.join(format!("{}.sock", self.session_name))
    }

    /// Get the session file path
    pub fn session_file_path(&self) -> PathBuf {
        self.session_dir.join(format!("{}.json", self.session_name))
    }

    /// Start the daemon process in the background
    pub fn start_daemon(&mut self, program: &str, args: &[String]) -> Result<(), String> {
        // Build the full command
        let mut cmd_args = vec![program.to_string()];
        cmd_args.extend(args.iter().cloned());

        // Find the keep-running binary
        let binary = std::env::var("CARGO_BIN_EXE_keep-running")
            .or_else(|_| {
                // Try to find it in target directory
                let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
                    .map_err(|_| "CARGO_MANIFEST_DIR not set")?;
                let path = PathBuf::from(manifest_dir)
                    .join("target")
                    .join("debug")
                    .join("keep-running");
                if path.exists() {
                    Ok(path.to_string_lossy().to_string())
                } else {
                    Err("Binary not found".to_string())
                }
            })
            .map_err(|e| format!("Failed to find keep-running binary: {}", e))?;

        // Spawn the daemon using the 'start' subcommand (starts without attaching)
        let child = Command::new(&binary)
            .arg("start")
            .arg("--name")
            .arg(&self.session_name)
            .arg("--")
            .args(&cmd_args)
            .env("KEEP_RUNNING_SESSION_DIR", &self.session_dir)
            .env("KEEP_RUNNING_SOCKET_DIR", &self.socket_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn daemon: {}", e))?;

        self.daemon_process = Some(child);

        // Wait for socket to appear
        let socket_path = self.socket_path();
        let start = Instant::now();
        let timeout = Duration::from_secs(5);

        while start.elapsed() < timeout {
            if socket_path.exists() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(50));
        }

        // Debug: check if session file was created in the right place
        let session_file = self.session_file_path();
        eprintln!("DEBUG: Looking for socket at: {}", socket_path.display());
        eprintln!("DEBUG: Session dir: {}", self.session_dir.display());
        eprintln!("DEBUG: Socket dir: {}", self.socket_dir.display());
        eprintln!("DEBUG: Session file exists: {}", session_file.exists());

        // List files in the socket dir
        if let Ok(entries) = std::fs::read_dir(&self.socket_dir) {
            for entry in entries.flatten() {
                eprintln!("DEBUG: Socket dir contains: {:?}", entry.path());
            }
        }

        // Check stderr from the child process
        if let Some(ref mut child) = self.daemon_process {
            if let Some(ref mut stderr) = child.stderr {
                let mut output = String::new();
                let _ = std::io::Read::read_to_string(stderr, &mut output);
                if !output.is_empty() {
                    eprintln!("DEBUG: Daemon stderr: {}", output);
                }
            }
        }

        Err(format!(
            "Daemon socket did not appear at {} within timeout",
            socket_path.display()
        ))
    }

    /// Check if the daemon process is alive.
    /// The `start` command forks the actual daemon and exits, so we can't
    /// just check the spawned child. Instead, read the daemon PID from the
    /// session file and check if that process is alive.
    pub fn is_daemon_alive(&mut self) -> bool {
        // Reap the 'start' wrapper if it hasn't been reaped yet
        if let Some(ref mut child) = self.daemon_process {
            let _ = child.try_wait();
        }

        // Read the session file to get the actual daemon PID
        let session_file = self.session_file_path();
        if !session_file.exists() {
            return false;
        }
        if let Ok(json) = std::fs::read_to_string(&session_file) {
            if let Ok(info) = serde_json::from_str::<serde_json::Value>(&json) {
                if let Some(pid) = info.get("pid").and_then(|v| v.as_u64()) {
                    return unsafe { libc::kill(pid as i32, 0) == 0 };
                }
            }
        }
        false
    }

    /// Check if the session file exists
    pub fn session_file_exists(&self) -> bool {
        self.session_file_path().exists()
    }

    /// Get the daemon PID from the session file
    fn daemon_pid(&self) -> Option<i32> {
        let session_file = self.session_file_path();
        if let Ok(json) = std::fs::read_to_string(&session_file) {
            if let Ok(info) = serde_json::from_str::<serde_json::Value>(&json) {
                return info.get("pid").and_then(|v| v.as_i64()).map(|p| p as i32);
            }
        }
        None
    }

    /// Kill the daemon if running
    pub fn kill_daemon(&mut self) {
        // Kill the 'start' wrapper process if still around
        if let Some(ref mut child) = self.daemon_process {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.daemon_process = None;

        // Also kill the actual daemon process (which is a separate forked process)
        if let Some(pid) = self.daemon_pid() {
            unsafe {
                libc::kill(pid, libc::SIGTERM);
            }
            // Give it a moment to clean up
            std::thread::sleep(Duration::from_millis(100));
            // Force kill if still alive
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
    }
}

impl Drop for TestContext {
    fn drop(&mut self) {
        self.kill_daemon();
    }
}

/// Test client for communicating with the daemon
pub struct TestClient {
    stream: UnixStream,
    buffer: Vec<u8>,
    pub accumulated_output: Vec<u8>,
    /// Stored exit code if we've already seen ChildExited
    pub exit_received: Option<Option<i32>>,
}

impl TestClient {
    /// Connect to the daemon
    pub fn connect(socket_path: &PathBuf) -> Result<Self, String> {
        let stream = UnixStream::connect(socket_path)
            .map_err(|e| format!("Failed to connect to socket: {}", e))?;

        stream
            .set_nonblocking(true)
            .map_err(|e| format!("Failed to set nonblocking: {}", e))?;

        Ok(Self {
            stream,
            buffer: Vec::new(),
            accumulated_output: Vec::new(),
            exit_received: None,
        })
    }

    /// Send attach message and wait for confirmation
    pub fn attach(&mut self, cols: u16, rows: u16) -> Result<(), String> {
        let msg = ClientMessage::Attach { cols, rows };
        self.send_message(&msg)?;

        // Wait for Attached confirmation
        let start = Instant::now();
        let timeout = Duration::from_secs(5);
        let mut attached = false;

        while start.elapsed() < timeout {
            match self.try_read_message()? {
                Some(msg) => {
                    match msg {
                        DaemonMessage::Attached => {
                            attached = true;
                            // Don't return yet - there may be replayed output following
                        }
                        DaemonMessage::Output(data) => {
                            self.accumulated_output.extend(data);
                        }
                        DaemonMessage::Error(e) => return Err(format!("Daemon error: {}", e)),
                        DaemonMessage::ChildExited { code } => {
                            // Process already exited, store the exit code
                            self.exit_received = Some(code);
                            // If we've seen Attached, we can return now
                            if attached {
                                return Ok(());
                            }
                        }
                    }
                }
                None => {
                    // No more messages - if we've seen Attached, we're done
                    if attached {
                        return Ok(());
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }

        if attached {
            Ok(())
        } else {
            Err("Timeout waiting for attach confirmation".to_string())
        }
    }

    /// Send a client message
    pub fn send_message(&mut self, msg: &ClientMessage) -> Result<(), String> {
        let encoded =
            encode_message(msg).map_err(|e| format!("Failed to encode message: {}", e))?;
        self.stream
            .write_all(&encoded)
            .map_err(|e| format!("Failed to send message: {}", e))?;
        Ok(())
    }

    /// Try to read a message (non-blocking)
    pub fn try_read_message(&mut self) -> Result<Option<DaemonMessage>, String> {
        // Keep reading until we would block or have a complete message
        let mut buf = [0u8; 8192];
        loop {
            match self.stream.read(&mut buf) {
                Ok(0) => return Err("Connection closed".to_string()),
                Ok(n) => {
                    self.buffer.extend_from_slice(&buf[..n]);
                    // Check if we might have a complete message
                    if self.buffer.len() >= 4 {
                        let len = u32::from_be_bytes([
                            self.buffer[0],
                            self.buffer[1],
                            self.buffer[2],
                            self.buffer[3],
                        ]) as usize;
                        // If we have enough data for the complete message, stop reading
                        if self.buffer.len() >= 4 + len {
                            break;
                        }
                    }
                    // Keep reading if there's more data
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => return Err(format!("Read error: {}", e)),
            }
        }

        // Try to decode a message
        match decode_message::<DaemonMessage>(&self.buffer) {
            Ok(Some((msg, consumed))) => {
                self.buffer.drain(0..consumed);
                Ok(Some(msg))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(format!("Decode error: {}", e)),
        }
    }

    /// Wait for output containing a pattern
    pub fn wait_for_output(
        &mut self,
        pattern: &OutputPattern,
        timeout: Duration,
    ) -> Result<(), String> {
        let start = Instant::now();

        // First check if we already have the pattern from previous messages
        if pattern.matches(&self.accumulated_output) {
            return Ok(());
        }

        while start.elapsed() < timeout {
            // Process any pending messages
            loop {
                match self.try_read_message() {
                    Ok(Some(msg)) => {
                        match msg {
                            DaemonMessage::Output(data) => {
                                self.accumulated_output.extend(data);
                                // Check after each output chunk
                                if pattern.matches(&self.accumulated_output) {
                                    return Ok(());
                                }
                            }
                            DaemonMessage::ChildExited { code } => {
                                // Store the exit for later use by wait_for_exit
                                self.exit_received = Some(code);
                                // Child exited - check if we have the pattern now
                                if pattern.matches(&self.accumulated_output) {
                                    return Ok(());
                                }
                                return Err(format!(
                                    "Child exited with code {:?} before pattern matched. Output: {}",
                                    code,
                                    String::from_utf8_lossy(&self.accumulated_output)
                                ));
                            }
                            _ => {}
                        }
                    }
                    Ok(None) => break, // No more messages available
                    Err(e) => return Err(e),
                }
            }

            // Check if accumulated output matches
            if pattern.matches(&self.accumulated_output) {
                return Ok(());
            }

            std::thread::sleep(Duration::from_millis(10));
        }

        let output_str = String::from_utf8_lossy(&self.accumulated_output);
        Err(format!(
            "Timeout waiting for pattern {:?}. Got: {}",
            pattern, output_str
        ))
    }

    /// Wait for child to exit
    pub fn wait_for_exit(
        &mut self,
        expected_code: Option<i32>,
        timeout: Duration,
    ) -> Result<(), String> {
        // Check if we already received the exit during attach
        if let Some(code) = self.exit_received.take() {
            if let Some(expected) = expected_code {
                if code != Some(expected) {
                    return Err(format!(
                        "Expected exit code {}, got {:?}",
                        expected, code
                    ));
                }
            }
            return Ok(());
        }

        let start = Instant::now();

        while start.elapsed() < timeout {
            while let Some(msg) = self.try_read_message()? {
                match msg {
                    DaemonMessage::Output(data) => {
                        self.accumulated_output.extend(data);
                    }
                    DaemonMessage::ChildExited { code } => {
                        if let Some(expected) = expected_code {
                            if code != Some(expected) {
                                return Err(format!(
                                    "Expected exit code {}, got {:?}",
                                    expected, code
                                ));
                            }
                        }
                        return Ok(());
                    }
                    _ => {}
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        Err("Timeout waiting for child exit".to_string())
    }

    /// Send detach message
    pub fn detach(&mut self) -> Result<(), String> {
        self.send_message(&ClientMessage::Detach)
    }

    /// Send input text
    pub fn send_input(&mut self, text: &str) -> Result<(), String> {
        self.send_message(&ClientMessage::Input(text.as_bytes().to_vec()))
    }

    /// Send raw bytes
    pub fn send_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.send_message(&ClientMessage::Input(bytes.to_vec()))
    }

    /// Send resize message
    pub fn resize(&mut self, cols: u16, rows: u16) -> Result<(), String> {
        self.send_message(&ClientMessage::Resize { cols, rows })
    }

    /// Get accumulated output
    pub fn get_output(&self) -> &[u8] {
        &self.accumulated_output
    }


    /// Check if replay contains expected content
    pub fn check_replay_contains(&self, expected: &str) -> bool {
        let output_str = String::from_utf8_lossy(&self.accumulated_output);
        output_str.contains(expected)
    }
}

/// Pattern for matching output
#[derive(Debug)]
pub enum OutputPattern {
    Contains(String),
    Regex(Regex),
}

impl OutputPattern {
    pub fn matches(&self, data: &[u8]) -> bool {
        let text = String::from_utf8_lossy(data);
        match self {
            OutputPattern::Contains(s) => text.contains(s),
            OutputPattern::Regex(r) => r.is_match(&text),
        }
    }
}

/// Run a complete scenario
pub fn run_scenario(scenario: &Scenario) -> Result<(), String> {
    let mut ctx = TestContext::new()?;
    let timeout = Duration::from_secs(scenario.timeout_secs);
    let start = Instant::now();

    // Start the daemon
    ctx.start_daemon(&scenario.command.program, &scenario.command.args)?;

    // Track client connection state
    let mut client: Option<TestClient> = None;

    // Execute actions
    for (i, action) in scenario.actions.iter().enumerate() {
        if start.elapsed() > timeout {
            return Err(format!("Scenario timeout exceeded at action {}", i));
        }

        match action {
            Action::Attach {
                expect_replay_contains,
            } => {
                let mut new_client = TestClient::connect(&ctx.socket_path())?;
                new_client.attach(80, 24)?;

                if let Some(expected) = expect_replay_contains {
                    // Wait for replay data to arrive (may come in chunks)
                    let replay_start = Instant::now();
                    let replay_timeout = Duration::from_secs(5);
                    loop {
                        // Drain all available messages
                        loop {
                            match new_client.try_read_message()? {
                                Some(DaemonMessage::Output(data)) => {
                                    new_client.accumulated_output.extend(data);
                                }
                                Some(DaemonMessage::ChildExited { code }) => {
                                    new_client.exit_received = Some(code);
                                }
                                Some(_) => {}
                                None => break,
                            }
                        }

                        if new_client.check_replay_contains(expected) {
                            break;
                        }

                        if replay_start.elapsed() > replay_timeout {
                            let output = String::from_utf8_lossy(new_client.get_output());
                            return Err(format!(
                                "Expected replay to contain '{}', got: {}",
                                expected, output
                            ));
                        }

                        std::thread::sleep(Duration::from_millis(10));
                    }
                }

                client = Some(new_client);
            }

            Action::SendInput { text } => {
                let c = client
                    .as_mut()
                    .ok_or("No client connected for send_input")?;
                c.send_input(text)?;
            }

            Action::SendBytes { bytes } => {
                let c = client
                    .as_mut()
                    .ok_or("No client connected for send_bytes")?;
                c.send_bytes(bytes)?;
            }

            Action::WaitForOutput {
                contains,
                regex,
                timeout_secs,
            } => {
                let c = client
                    .as_mut()
                    .ok_or("No client connected for wait_for_output")?;

                let pattern = if let Some(s) = contains {
                    OutputPattern::Contains(s.clone())
                } else if let Some(r) = regex {
                    let re = Regex::new(r).map_err(|e| format!("Invalid regex: {}", e))?;
                    OutputPattern::Regex(re)
                } else {
                    return Err("wait_for_output requires 'contains' or 'regex'".to_string());
                };

                c.wait_for_output(&pattern, Duration::from_secs(*timeout_secs))?;
            }

            Action::WaitForExit {
                expected_code,
                timeout_secs,
            } => {
                let c = client
                    .as_mut()
                    .ok_or("No client connected for wait_for_exit")?;
                c.wait_for_exit(*expected_code, Duration::from_secs(*timeout_secs))?;
            }

            Action::Detach => {
                if let Some(ref mut c) = client {
                    c.detach()?;
                }
                client = None;
            }

            Action::DisconnectRaw => {
                // Just drop the client without sending detach
                client = None;
            }

            Action::Sleep { duration_ms } => {
                std::thread::sleep(Duration::from_millis(*duration_ms));
            }

            Action::Resize { cols, rows } => {
                let c = client.as_mut().ok_or("No client connected for resize")?;
                c.resize(*cols, *rows)?;
            }

            Action::AssertSessionExists { name } => {
                let session_name = name.as_ref().unwrap_or(&ctx.session_name);
                let path = ctx.session_dir.join(format!("{}.json", session_name));
                if !path.exists() {
                    return Err(format!("Expected session file to exist: {}", path.display()));
                }
            }

            Action::AssertSessionGone { name } => {
                let session_name = name.as_ref().unwrap_or(&ctx.session_name);
                let path = ctx.session_dir.join(format!("{}.json", session_name));
                if path.exists() {
                    return Err(format!("Expected session file to be gone: {}", path.display()));
                }
            }
        }
    }

    // Drop the client to disconnect before checking final state
    // This allows the daemon to clean up properly
    drop(client);

    // Check expected final state
    if let Some(ref expected) = scenario.expected_final_state {
        verify_final_state(&mut ctx, expected)?;
    }

    Ok(())
}

/// Verify the expected final state
fn verify_final_state(ctx: &mut TestContext, expected: &ExpectedFinalState) -> Result<(), String> {
    // Give a moment for cleanup
    std::thread::sleep(Duration::from_millis(200));

    if let Some(daemon_alive) = expected.daemon_alive {
        let actual = ctx.is_daemon_alive();
        if actual != daemon_alive {
            return Err(format!(
                "Expected daemon_alive={}, got {}",
                daemon_alive, actual
            ));
        }
    }

    if let Some(session_exists) = expected.session_file_exists {
        let actual = ctx.session_file_exists();
        if actual != session_exists {
            return Err(format!(
                "Expected session_file_exists={}, got {}",
                session_exists, actual
            ));
        }
    }

    Ok(())
}
