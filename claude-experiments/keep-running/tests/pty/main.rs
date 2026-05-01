//! Pty-driven integration tests.
//!
//! These tests cover the interactive raw-mode paths in `client::attach()`
//! that the protocol-level integration harness can't exercise: the welcome
//! banner, the `Ctrl+a d` / `Ctrl+a k` keystroke state machine, and the
//! `Ctrl+a Ctrl+a` literal-passthrough escape.
//!
//! The test process spawns `keep-running run -- ...` with a pty as its
//! controlling terminal, drives keystrokes by writing to the master, and
//! reads the binary's output (banner + program output) from the master.

mod harness;

use harness::PtySession;
use std::time::Duration;

fn binary_path() -> String {
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_keep-running") {
        return p;
    }
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set");
    std::path::PathBuf::from(manifest_dir)
        .join("target")
        .join("debug")
        .join("keep-running")
        .to_string_lossy()
        .into_owned()
}

#[test]
fn ctrl_a_d_detaches_and_leaves_daemon_running() {
    let mut session = PtySession::spawn(
        &binary_path(),
        &["run", "--", "sleep", "30"],
    )
    .expect("spawn");

    session
        .wait_for("attached to", Duration::from_secs(5))
        .expect("welcome banner");

    let daemon_pid = session
        .daemon_pid()
        .expect("daemon pid should be in session file once attached");
    assert!(pid_alive(daemon_pid), "daemon should be alive while attached");

    session.write(b"\x01d").expect("send Ctrl+a d");

    session
        .wait_for("detached from", Duration::from_secs(5))
        .expect("detach banner");

    let status = session
        .wait_exit(Duration::from_secs(5))
        .expect("client should exit after detach");
    assert!(status.success(), "client exit should be clean: {:?}", status);

    // Detach must leave the daemon alive so we can reattach.
    assert!(
        pid_alive(daemon_pid),
        "daemon should still be running after detach"
    );

    // Manual cleanup since detach left the daemon behind.
    unsafe {
        libc::kill(daemon_pid, libc::SIGTERM);
    }
}

#[test]
fn ctrl_a_k_kills_session() {
    let mut session = PtySession::spawn(
        &binary_path(),
        &["run", "--", "sleep", "30"],
    )
    .expect("spawn");

    session
        .wait_for("attached to", Duration::from_secs(5))
        .expect("welcome banner");

    let daemon_pid = session.daemon_pid().expect("daemon pid");

    session.write(b"\x01k").expect("send Ctrl+a k");

    session
        .wait_for("killed", Duration::from_secs(5))
        .expect("kill banner");

    let status = session
        .wait_exit(Duration::from_secs(5))
        .expect("client should exit after kill");
    assert!(status.success(), "client exit should be clean: {:?}", status);

    // The daemon was sent SIGHUP; give it a moment to die.
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while pid_alive(daemon_pid) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        !pid_alive(daemon_pid),
        "daemon should be gone after Ctrl+a k"
    );
}

#[test]
fn ctrl_a_ctrl_a_passes_through_literal_byte() {
    // `cat -v` echoes \x01 as the visible string `^A`. If the client
    // intercepted the second \x01 (treating it as a fresh prefix) we'd
    // never see it round-trip back through the daemon's pty.
    let mut session = PtySession::spawn(
        &binary_path(),
        &["run", "--", "cat", "-v"],
    )
    .expect("spawn");

    session
        .wait_for("attached to", Duration::from_secs(5))
        .expect("welcome banner");

    // Two literal Ctrl+A bytes (the client should fold these into one byte
    // sent to the child) followed by newline so `cat` flushes its line.
    session
        .write(b"\x01\x01\n")
        .expect("send doubled Ctrl+a + newline");

    session
        .wait_for("^A", Duration::from_secs(5))
        .expect("cat -v should echo ^A from a passed-through Ctrl+a");

    // Clean up: detach so the daemon doesn't linger past the test.
    let daemon_pid = session.daemon_pid().expect("daemon pid");
    session.write(b"\x01d").expect("send Ctrl+a d");
    let _ = session.wait_exit(Duration::from_secs(5));
    unsafe {
        libc::kill(daemon_pid, libc::SIGTERM);
    }
}

fn pid_alive(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 }
}
