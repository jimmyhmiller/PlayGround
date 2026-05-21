//! End-to-end smoke test: publish via the wire protocol, verify the
//! JSONL gets the line and a subscriber sees the event.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use claude_bus::daemon::Daemon;
use claude_bus::proto::{decode, encode, BusFrame, ClientFrame, Role};

fn spawn_daemon(dir: &std::path::Path) -> (PathBuf, std::thread::JoinHandle<()>) {
    let socket = dir.join("bus.sock");
    let jsonl = dir.join("events.jsonl");
    let socket_for_caller = socket.clone();
    let handle = thread::spawn(move || {
        let d = Daemon::new(&socket, &jsonl).expect("daemon new");
        let _ = d.run();
    });
    // Wait for the socket to exist.
    let deadline = Instant::now() + Duration::from_secs(2);
    while !socket_for_caller.exists() {
        if Instant::now() > deadline {
            panic!("daemon never created socket");
        }
        thread::sleep(Duration::from_millis(10));
    }
    (socket_for_caller, handle)
}

#[test]
fn publish_writes_jsonl_and_reaches_subscriber() {
    let dir = tempdir();
    let (socket, _h) = spawn_daemon(&dir);

    // Subscriber first, so it gets the live event.
    let mut sub = UnixStream::connect(&socket).expect("sub connect");
    sub.set_nonblocking(false).unwrap();
    sub.write_all(
        &encode(&ClientFrame::Hello {
            role: Role::Subscriber { since_seq: None },
        })
        .unwrap(),
    )
    .unwrap();

    // Brief sleep so the daemon processes the Hello before we publish.
    thread::sleep(Duration::from_millis(50));

    // Publisher.
    let mut pub_ = UnixStream::connect(&socket).expect("pub connect");
    pub_.write_all(
        &encode(&ClientFrame::Hello {
            role: Role::Publisher,
        })
        .unwrap(),
    )
    .unwrap();
    pub_.write_all(
        &encode(&ClientFrame::Publish {
            kind: "pre_tool_use".into(),
            ts: 12345,
            terminal_session_id: "T1".into(),
            claude_pid: 999,
            payload_json: r#"{"tool":"Bash"}"#.into(),
        })
        .unwrap(),
    )
    .unwrap();
    drop(pub_);

    // Read one BusFrame::Event from the subscriber.
    let event = read_one_event(&mut sub, Duration::from_secs(2));
    match event {
        BusFrame::Event { seq, kind, ts, .. } => {
            assert_eq!(seq, 0);
            assert_eq!(kind, "pre_tool_use");
            assert_eq!(ts, 12345);
        }
        other => panic!("unexpected frame: {:?}", other),
    }

    // JSONL line should be present.
    let line = std::fs::read_to_string(dir.join("events.jsonl")).unwrap();
    assert!(line.contains("\"kind\":\"pre_tool_use\""), "got: {}", line);
    assert!(line.contains("\"ts\":12345"), "got: {}", line);
    assert!(line.contains("\"tool\":\"Bash\""), "got: {}", line);
}

#[test]
fn replay_serves_since_seq() {
    let dir = tempdir();
    let (socket, _h) = spawn_daemon(&dir);

    // Publish 3 events before any subscriber connects.
    let mut p = UnixStream::connect(&socket).unwrap();
    p.write_all(&encode(&ClientFrame::Hello { role: Role::Publisher }).unwrap()).unwrap();
    for i in 0..3 {
        p.write_all(
            &encode(&ClientFrame::Publish {
                kind: format!("k{}", i),
                ts: 100 + i as u64,
                terminal_session_id: "T".into(),
                claude_pid: 1,
                payload_json: "{}".into(),
            })
            .unwrap(),
        )
        .unwrap();
    }
    drop(p);
    thread::sleep(Duration::from_millis(100));

    // Subscribe from seq=1; should get events 1 and 2 then ReplayEnd.
    let mut s = UnixStream::connect(&socket).unwrap();
    s.set_nonblocking(false).unwrap();
    s.write_all(
        &encode(&ClientFrame::Hello {
            role: Role::Subscriber { since_seq: Some(1) },
        })
        .unwrap(),
    )
    .unwrap();

    let events = read_n_events(&mut s, 2, Duration::from_secs(2));
    match (&events[0], &events[1]) {
        (BusFrame::Event { seq: s1, kind: k1, .. }, BusFrame::Event { seq: s2, kind: k2, .. }) => {
            assert_eq!(*s1, 1);
            assert_eq!(k1, "k1");
            assert_eq!(*s2, 2);
            assert_eq!(k2, "k2");
        }
        _ => panic!("unexpected frames: {:?}", events),
    }
}

fn read_n_events(s: &mut UnixStream, n: usize, timeout: Duration) -> Vec<BusFrame> {
    s.set_read_timeout(Some(timeout)).unwrap();
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let mut out = Vec::new();
    let deadline = Instant::now() + timeout;
    while out.len() < n {
        loop {
            match decode::<BusFrame>(&buf) {
                Ok(Some((f, consumed))) => {
                    buf.drain(0..consumed);
                    match f {
                        BusFrame::ReplayEnd => continue,
                        other => {
                            out.push(other);
                            if out.len() == n {
                                return out;
                            }
                        }
                    }
                }
                Ok(None) => break,
                Err(e) => panic!("decode: {:?}", e),
            }
        }
        if Instant::now() > deadline {
            panic!("timed out waiting for {} events (got {}); buf={:?}", n, out.len(), buf);
        }
        let r = s.read(&mut tmp).expect("read");
        if r == 0 {
            panic!("socket closed before {} events (got {})", n, out.len());
        }
        buf.extend_from_slice(&tmp[..r]);
    }
    out
}

fn read_one_event(s: &mut UnixStream, timeout: Duration) -> BusFrame {
    s.set_read_timeout(Some(timeout)).unwrap();
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let deadline = Instant::now() + timeout;
    loop {
        // Try decoding what we have. Skip ReplayEnd frames so callers
        // can ignore the framing detail.
        loop {
            match decode::<BusFrame>(&buf) {
                Ok(Some((f, n))) => {
                    buf.drain(0..n);
                    match f {
                        BusFrame::ReplayEnd => continue,
                        other => return other,
                    }
                }
                Ok(None) => break,
                Err(e) => panic!("decode: {:?}", e),
            }
        }
        if Instant::now() > deadline {
            panic!("timed out waiting for event; buf={:?}", buf);
        }
        let n = s.read(&mut tmp).expect("read");
        buf.extend_from_slice(&tmp[..n]);
    }
}

fn tempdir() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "claude-bus-test-{}-{}",
        std::process::id(),
        rand_suffix()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn rand_suffix() -> String {
    // Cheap nondeterministic suffix without pulling in `rand`.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    format!("{:x}", nanos)
}
