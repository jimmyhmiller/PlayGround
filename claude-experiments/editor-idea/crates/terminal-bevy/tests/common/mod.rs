//! Shared setup for integration tests that need a working daemon.

use std::path::PathBuf;

pub struct DaemonTestEnv {
    /// Held so the temp dir survives the test. Backing store for HOME +
    /// the daemon stderr log.
    pub home: tempfile::TempDir,
    /// Short path (under /tmp) where sockets / pid files live. Must be
    /// short to satisfy macOS's 104-char `SUN_LEN`.
    pub runtime_dir: PathBuf,
}

impl Drop for DaemonTestEnv {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.runtime_dir);
    }
}

/// Set up an isolated environment: HOME → tempdir, runtime dir →
/// short /tmp path, daemon binary path discovered from this test's own
/// `current_exe()`. Returns a guard whose drop cleans the runtime dir.
pub fn setup_isolated_daemon_env() -> DaemonTestEnv {
    let home = tempfile::Builder::new()
        .prefix("terminal-bevy-test-home-")
        .tempdir()
        .expect("tempdir");

    let test_exe = std::env::current_exe().expect("current_exe");
    let daemon_bin = test_exe
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("terminal-daemon"))
        .expect("derive daemon binary path");
    assert!(
        daemon_bin.exists(),
        "terminal-daemon binary not built at {}",
        daemon_bin.display()
    );

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let runtime_dir = PathBuf::from(format!(
        "/tmp/terminal-bevy-test-{}-{}",
        std::process::id(),
        nanos
    ));
    std::fs::create_dir_all(&runtime_dir).expect("create runtime_dir");

    // SAFETY: tests using this helper must run single-threaded
    // (--test-threads=1). We mutate shared process env.
    unsafe {
        std::env::set_var("HOME", home.path());
        std::env::set_var("TERMINAL_BEVY_DAEMON_BIN", &daemon_bin);
        std::env::set_var("TERMINAL_BEVY_RUNTIME_DIR", &runtime_dir);
        std::env::set_var(
            "TERMINAL_DAEMON_LOG",
            home.path().join("daemon.log"),
        );
    }

    DaemonTestEnv { home, runtime_dir }
}

/// Pick a session_id unlikely to collide with anything from a previous
/// crashed test run, even within the same runtime_dir.
pub fn random_session_id() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = std::process::id() as u64;
    900_000 + (nanos.wrapping_mul(pid) % 100_000)
}
