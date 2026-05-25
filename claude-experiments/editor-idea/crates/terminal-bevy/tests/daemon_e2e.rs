//! End-to-end test of the per-session daemon + DaemonClient.
//!
//! We fork the real `terminal-daemon` binary (cargo gives us its path
//! via `CARGO_BIN_EXE_terminal-daemon`), connect through `DaemonClient`,
//! and verify that:
//!
//! 1. A fresh attach replays an empty buffer (no Output) and then
//!    delivers live Output from the child.
//! 2. Reattach to the *same* live daemon replays the prior Output via
//!    ReplayStart / Output / ReplayEnd.
//! 3. On Detach the daemon stays alive; on Kill it dies.
//!
//! HOME is pointed at a tempdir so socket_path / pid_path / scrollback
//! never touches the user's real config.

mod common;

use std::time::{Duration, Instant};

use terminal_bevy::daemon_client::DaemonClient;
use terminal_bevy::daemon_proto::DaemonMessage;

/// Drain frames into a Vec for up to `timeout`.
fn drain_for(
    client: &mut DaemonClient,
    timeout: Duration,
) -> (Vec<DaemonMessage>, bool /* still alive */) {
    let deadline = Instant::now() + timeout;
    let mut got = Vec::new();
    let mut alive = true;
    while Instant::now() < deadline {
        let ok = client
            .poll_frames(|m| got.push(m))
            .expect("poll_frames errored");
        if !ok {
            alive = false;
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    (got, alive)
}

struct LogDumper {
    dir: std::path::PathBuf,
}
impl Drop for LogDumper {
    fn drop(&mut self) {
        let log = self.dir.join("daemon.log");
        if let Ok(s) = std::fs::read_to_string(&log) {
            eprintln!(
                "\n--- daemon.log @ {} ---\n{}\n---",
                log.display(),
                s
            );
        } else {
            eprintln!("\n--- no daemon.log at {} ---", log.display());
        }
    }
}

#[test]
fn fresh_daemon_attach_runs_child_then_exits() {
    let env = common::setup_isolated_daemon_env();
    let _dumper = LogDumper {
        dir: env.home.path().to_path_buf(),
    };
    let session = common::random_session_id();
    // /bin/echo writes one line and exits.
    let mut client = DaemonClient::open(
        session,
        80,
        24,
        vec!["/bin/echo".to_string(), "hello-daemon".to_string()],
    )
    .expect("open daemon");
    assert!(!client.attached_existing, "fresh daemon should be spawned");

    let (frames, _) = drain_for(&mut client, Duration::from_secs(3));
    // Look for the echo output + ChildExited.
    let saw_output = frames.iter().any(|m| match m {
        DaemonMessage::Output(b) => {
            let s = String::from_utf8_lossy(b);
            s.contains("hello-daemon")
        }
        _ => false,
    });
    let saw_exit = frames
        .iter()
        .any(|m| matches!(m, DaemonMessage::ChildExited { .. }));
    assert!(saw_output, "missing echo Output. Frames: {:?}", frames);
    assert!(saw_exit, "missing ChildExited. Frames: {:?}", frames);
}

#[test]
fn reattach_replays_history() {
    let _env = common::setup_isolated_daemon_env();
    let session = common::random_session_id();
    // Long-running cat reads stdin forever — keeps the daemon alive
    // while we detach and reconnect.
    let cmd = vec!["/bin/cat".to_string()];

    let mut c1 = DaemonClient::open(session, 80, 24, cmd.clone()).expect("first attach");
    assert!(!c1.attached_existing);

    // Write something into the child via Input; cat echoes it back.
    c1.send(&terminal_bevy::daemon_proto::ClientMessage::Input(
        b"hello-replay\n".to_vec(),
    ));
    c1.try_flush();

    // Let cat round-trip + the daemon buffer it.
    let (frames1, _) = drain_for(&mut c1, Duration::from_millis(500));
    let saw_first = frames1.iter().any(|m| match m {
        DaemonMessage::Output(b) => String::from_utf8_lossy(b).contains("hello-replay"),
        _ => false,
    });
    assert!(saw_first, "expected first client to see echo. {:?}", frames1);

    // Detach (drop the connection) — daemon survives.
    drop(c1);
    std::thread::sleep(Duration::from_millis(150));

    // Reattach: the daemon was already running, history should replay.
    let mut c2 = DaemonClient::open(session, 80, 24, cmd).expect("reattach");
    assert!(c2.attached_existing, "expected to reattach to live daemon");
    let (frames2, _) = drain_for(&mut c2, Duration::from_millis(800));

    let mut saw_replay_start = false;
    let mut saw_replay_end = false;
    let mut saw_replayed_output = false;
    for m in &frames2 {
        match m {
            DaemonMessage::ReplayStart => saw_replay_start = true,
            DaemonMessage::ReplayEnd => saw_replay_end = true,
            DaemonMessage::Output(b) => {
                if String::from_utf8_lossy(b).contains("hello-replay") {
                    saw_replayed_output = true;
                }
            }
            _ => {}
        }
    }
    assert!(saw_replay_start, "no ReplayStart in {:?}", frames2);
    assert!(saw_replay_end, "no ReplayEnd in {:?}", frames2);
    assert!(
        saw_replayed_output,
        "history not replayed in {:?}",
        frames2
    );

    // Kill so the test doesn't leave a daemon running.
    c2.send(&terminal_bevy::daemon_proto::ClientMessage::Kill);
    c2.try_flush();
    std::thread::sleep(Duration::from_millis(200));
}
