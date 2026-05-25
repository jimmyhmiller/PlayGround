//! End-to-end reproduction attempt for the "100-line paste hangs the
//! terminal pane" bug. Spawns a real worker (which forks `$SHELL`),
//! sends a multi-line paste via `WorkerMsg::Paste`, then sends a
//! distinctive keystroke and asserts it shows up in the snapshot.
//!
//! If the bug is in the worker / pty / shell-bracketed-paste interaction,
//! this should hang or fail. If it passes, the bug is elsewhere
//! (main-thread input handling, focus loss, renderer choke, etc.).

mod common;

use std::time::{Duration, Instant};

use terminal_bevy::pty::PtySize;
use terminal_bevy::worker::{WorkerHandle, WorkerMsg};

const SIZE: PtySize = PtySize {
    cols: 80,
    rows: 24,
    cell_width_px: 8,
    cell_height_px: 18,
};

fn snapshot_text(handle: &WorkerHandle) -> String {
    let g = handle.snapshot.lock().expect("snapshot lock");
    g.cells.iter().map(|c| c.ch).collect()
}

fn wait_for<F: FnMut(&str) -> bool>(
    handle: &WorkerHandle,
    deadline: Instant,
    mut pred: F,
) -> Option<String> {
    while Instant::now() < deadline {
        let s = snapshot_text(handle);
        if pred(&s) {
            return Some(s);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    None
}

#[test]
fn keystroke_after_large_paste_reaches_pty() {
    // ~64 KB paste — well past the PTY's input buffer cap on macOS
    // (4 KB on the master side). Pre-fix the daemon's `pty_write_all`
    // hit WouldBlock partway through and silently dropped the tail,
    // including the closing `\x1b[201~`, which wedged the shell in
    // bracketed-paste mode forever.
    let _env = common::setup_isolated_daemon_env();
    let session = common::random_session_id();
    let handle = WorkerHandle::spawn(
        session,
        terminal_bevy::default_shell_command(),
        SIZE,
        1000,
        None,
        None,
        None,
    )
    .expect("spawn");

    // Wait for the shell prompt.
    std::thread::sleep(Duration::from_millis(800));

    // Build a ~64 KB paste body that, post-execution, will _not_ leave
    // 64 KB of output sitting in the buffer (we just want bytes to
    // travel through the PTY). Use a long single comment so the shell
    // sees it as one line.
    let big = "x".repeat(64 * 1024);
    let paste = format!("# {big}\n");
    let bytes = paste.len();
    handle.send(WorkerMsg::Paste(paste));

    std::thread::sleep(Duration::from_millis(2000));

    // Marker keystroke. If the closing bracket of bracketed-paste was
    // dropped, the shell is still in paste mode and AABBCC never runs.
    handle.send(WorkerMsg::Input(b"AABBCC\n".to_vec()));

    let deadline = Instant::now() + Duration::from_secs(5);
    let result = wait_for(&handle, deadline, |s| s.contains("AABBCC"));

    if result.is_none() {
        let final_snap = snapshot_text(&handle);
        panic!(
            "after a {bytes}-byte paste, the keystroke 'AABBCC' never \
             appeared in the terminal snapshot — pane is unresponsive. \
             Snapshot tail (last 400 chars):\n{}",
            &final_snap[final_snap.len().saturating_sub(400)..]
        );
    }
}

#[test]
fn keystroke_after_100_line_paste_reaches_pty() {
    let _env = common::setup_isolated_daemon_env();
    let session = common::random_session_id();
    let handle = WorkerHandle::spawn(
        session,
        terminal_bevy::default_shell_command(),
        SIZE,
        1000,
        None,
        None,
        None,
    )
    .expect("spawn");

    // Wait for the shell to come up — give it a generous window.
    std::thread::sleep(Duration::from_millis(800));

    // 100-line paste, no embedded escape sequences.
    let paste: String = (0..100).map(|i| format!("echo line_{i}\n")).collect();
    let bytes = paste.len();
    handle.send(WorkerMsg::Paste(paste));

    // Give the shell a moment to ingest the paste — bracketed-paste
    // mode in zsh/bash redraws the prompt, so we need to let the
    // worker drain the echo before testing keyboard responsiveness.
    std::thread::sleep(Duration::from_millis(1500));

    // Distinctive marker keystroke. If the shell is wedged in
    // bracketed-paste mode, this either gets buffered as more paste
    // content (and never echoes) or never reaches the shell at all.
    handle.send(WorkerMsg::Input(b"AABBCC".to_vec()));

    // Up to 3 seconds for the marker to land in the snapshot.
    let deadline = Instant::now() + Duration::from_secs(3);
    let result = wait_for(&handle, deadline, |s| s.contains("AABBCC"));

    if let Some(s) = result {
        // Sanity: confirm the marker is actually visible.
        assert!(s.contains("AABBCC"));
    } else {
        let final_snap = snapshot_text(&handle);
        panic!(
            "after a {bytes}-byte paste, the keystroke 'AABBCC' never \
             appeared in the terminal snapshot — pane is unresponsive. \
             Snapshot tail (last 400 chars):\n{}",
            &final_snap[final_snap.len().saturating_sub(400)..]
        );
    }
}
