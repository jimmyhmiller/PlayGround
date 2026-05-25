//! Subscriber-API integration test. Spin up the daemon in-process,
//! point a `Subscriber` at it, publish a few events, and verify the
//! subscriber receives them in order.

use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use claude_bus::client::{BusItem, Subscriber};
use claude_bus::daemon::Daemon;
use claude_bus::proto::{encode, ClientFrame, Role};

fn spawn_daemon(dir: &std::path::Path) -> PathBuf {
    let socket = dir.join("bus.sock");
    let jsonl = dir.join("events.jsonl");
    let s_for_caller = socket.clone();
    thread::spawn(move || {
        let d = Daemon::new(&socket, &jsonl).expect("daemon new");
        let _ = d.run();
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    while !s_for_caller.exists() {
        if Instant::now() > deadline {
            panic!("daemon never bound socket");
        }
        thread::sleep(Duration::from_millis(10));
    }
    s_for_caller
}

fn publish(socket: &std::path::Path, kind: &str, ts: u64) {
    let mut s = UnixStream::connect(socket).unwrap();
    s.write_all(&encode(&ClientFrame::Hello { role: Role::Publisher }).unwrap()).unwrap();
    s.write_all(
        &encode(&ClientFrame::Publish {
            kind: kind.into(),
            ts,
            terminal_session_id: "T".into(),
            claude_pid: 1,
            payload_json: "{}".into(),
        })
        .unwrap(),
    )
    .unwrap();
}

#[test]
fn subscriber_receives_live_events() {
    let dir = tempdir();
    let socket = spawn_daemon(&dir);

    let sub = Subscriber::spawn(socket.clone(), None);
    // Let the subscriber complete its Hello.
    thread::sleep(Duration::from_millis(100));

    publish(&socket, "k1", 100);
    publish(&socket, "k2", 200);

    let events = collect_events(&sub, 2, Duration::from_secs(2));
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].kind, "k1");
    assert_eq!(events[0].seq, 0);
    assert_eq!(events[1].kind, "k2");
    assert_eq!(events[1].seq, 1);
}

#[test]
fn subscriber_replays_since_seq() {
    let dir = tempdir();
    let socket = spawn_daemon(&dir);

    publish(&socket, "k0", 100);
    publish(&socket, "k1", 200);
    publish(&socket, "k2", 300);
    // Wait for the daemon to absorb the publishes.
    thread::sleep(Duration::from_millis(100));

    let sub = Subscriber::spawn(socket, Some(1));
    let events = collect_events(&sub, 2, Duration::from_secs(2));
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].seq, 1);
    assert_eq!(events[1].seq, 2);
}

fn collect_events(
    sub: &Subscriber,
    n: usize,
    timeout: Duration,
) -> Vec<claude_bus::client::BusEvent> {
    let mut out = Vec::new();
    let deadline = Instant::now() + timeout;
    while out.len() < n {
        match sub.recv_timeout(Duration::from_millis(100)) {
            Some(BusItem::Event(ev)) => out.push(ev),
            Some(_) => {}
            None => {
                if Instant::now() > deadline {
                    panic!("timed out waiting for {} events (got {})", n, out.len());
                }
            }
        }
    }
    out
}

fn tempdir() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "claude-bus-client-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}
