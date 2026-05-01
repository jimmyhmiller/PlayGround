//! Pty-based test harness for keep-running's interactive client paths.

use portable_pty::{native_pty_system, Child, CommandBuilder, MasterPty, PtySize};
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitStatus;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tempfile::TempDir;

static SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct PtySession {
    /// Kept so the temp dirs survive for the lifetime of the session.
    _temp_dir: TempDir,
    session_dir: PathBuf,
    session_name: String,
    child: Option<Box<dyn Child + Send + Sync>>,
    master: Option<Box<dyn MasterPty + Send>>,
    writer: Box<dyn Write + Send>,
    output: Arc<Mutex<Vec<u8>>>,
    reader_thread: Option<JoinHandle<()>>,
}

impl PtySession {
    pub fn spawn(binary: &str, args: &[&str]) -> Result<Self, String> {
        let temp_dir = TempDir::new().map_err(|e| format!("tempdir: {e}"))?;
        let session_dir = temp_dir.path().join("sessions");
        let socket_dir = temp_dir.path().join("sockets");
        std::fs::create_dir_all(&session_dir).map_err(|e| format!("mkdir sessions: {e}"))?;
        std::fs::create_dir_all(&socket_dir).map_err(|e| format!("mkdir sockets: {e}"))?;

        let counter = SESSION_COUNTER.fetch_add(1, Ordering::SeqCst);
        let session_name = format!("pty-test-{}-{}", std::process::id(), counter);

        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows: 24,
                cols: 80,
                pixel_width: 0,
                pixel_height: 0,
            })
            .map_err(|e| format!("openpty: {e}"))?;

        let mut cmd = CommandBuilder::new(binary);
        // We always inject `--name <session_name>` after the first arg
        // (which is expected to be the subcommand, e.g. `run`).
        let mut injected = false;
        for arg in args {
            cmd.arg(arg);
            if !injected {
                cmd.arg("--name");
                cmd.arg(&session_name);
                injected = true;
            }
        }
        cmd.env("KEEP_RUNNING_SESSION_DIR", &session_dir);
        cmd.env("KEEP_RUNNING_SOCKET_DIR", &socket_dir);
        cmd.env("TERM", "xterm-256color");
        // Force-disable the cyan ANSI escapes in status() so substring matches
        // don't have to skip past `\x1b[36m`.
        cmd.env("NO_COLOR", "1");

        let child = pair
            .slave
            .spawn_command(cmd)
            .map_err(|e| format!("spawn: {e}"))?;
        // Drop our handle to the slave; the child has its own.
        drop(pair.slave);

        let mut reader = pair
            .master
            .try_clone_reader()
            .map_err(|e| format!("clone reader: {e}"))?;
        let writer = pair
            .master
            .take_writer()
            .map_err(|e| format!("take writer: {e}"))?;

        let output: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let output_clone = Arc::clone(&output);

        let reader_thread = thread::spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if let Ok(mut out) = output_clone.lock() {
                            out.extend_from_slice(&buf[..n]);
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        Ok(Self {
            _temp_dir: temp_dir,
            session_dir,
            session_name,
            child: Some(child),
            master: Some(pair.master),
            writer,
            output,
            reader_thread: Some(reader_thread),
        })
    }

    pub fn write(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.writer
            .write_all(bytes)
            .map_err(|e| format!("write: {e}"))?;
        self.writer.flush().map_err(|e| format!("flush: {e}"))?;
        Ok(())
    }

    pub fn wait_for(&self, needle: &str, timeout: Duration) -> Result<(), String> {
        let needle_bytes = needle.as_bytes();
        let start = Instant::now();
        loop {
            {
                let out = self.output.lock().unwrap();
                if find_subslice(&out, needle_bytes).is_some() {
                    return Ok(());
                }
            }
            if start.elapsed() >= timeout {
                let out = self.output.lock().unwrap();
                let preview = String::from_utf8_lossy(&out).into_owned();
                return Err(format!(
                    "timeout waiting for {:?}; got {} bytes:\n{}",
                    needle,
                    out.len(),
                    preview
                ));
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    pub fn wait_exit(&mut self, timeout: Duration) -> Result<ExitStatus, String> {
        let start = Instant::now();
        let child = self.child.as_mut().ok_or("child already taken")?;
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let exit = status
                        .exit_code()
                        .try_into()
                        .ok()
                        .and_then(|c: i32| Some(c))
                        .unwrap_or(0);
                    // Convert portable_pty::ExitStatus -> std::process::ExitStatus
                    // by faking one through libc. Easier: just return a synthetic.
                    return Ok(synth_exit_status(exit));
                }
                Ok(None) => {
                    if start.elapsed() >= timeout {
                        return Err("timeout waiting for client exit".into());
                    }
                    thread::sleep(Duration::from_millis(20));
                }
                Err(e) => return Err(format!("try_wait: {e}")),
            }
        }
    }

    /// Read the daemon's pid from the session JSON file. Returns None if
    /// the daemon hasn't registered itself yet.
    pub fn daemon_pid(&self) -> Option<i32> {
        let path = self.session_dir.join(format!("{}.json", self.session_name));
        let json = std::fs::read_to_string(&path).ok()?;
        let v: serde_json::Value = serde_json::from_str(&json).ok()?;
        v.get("pid").and_then(|p| p.as_i64()).map(|p| p as i32)
    }
}

impl Drop for PtySession {
    fn drop(&mut self) {
        // Force-kill any leftover child (e.g. test failed mid-run).
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        // Closing the master makes the reader thread see EOF.
        drop(self.master.take());
        if let Some(handle) = self.reader_thread.take() {
            let _ = handle.join();
        }
        // Best-effort: kill any orphaned daemon. The session file may have
        // already been removed; that's fine.
        if let Some(pid) = self.daemon_pid() {
            unsafe {
                libc::kill(pid, libc::SIGTERM);
            }
        }
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(unix)]
fn synth_exit_status(code: i32) -> ExitStatus {
    use std::os::unix::process::ExitStatusExt;
    ExitStatus::from_raw((code & 0xff) << 8)
}
