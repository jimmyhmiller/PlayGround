//! Headless smoke test: spin a bus daemon in-thread, build a minimal
//! Bevy `App` with `BusEventPlugin`, publish a few events directly to
//! the daemon, and verify the plugin drains them into a Bevy message
//! reader.

use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use bevy::prelude::*;

use claude_bus::daemon::Daemon;
use claude_bus::proto::{encode, ClientFrame, Role};
use claude_bus_bevy::{BusEventPlugin, ClaudeBusEvent};

fn spawn_daemon(dir: &std::path::Path) -> PathBuf {
    // Mirror what `claude_bus::socket_path()` returns when $HOME == dir,
    // so the plugin (which derives paths from $HOME) and the daemon
    // agree on the same socket.
    let claude_dir = dir.join(".claude");
    std::fs::create_dir_all(&claude_dir).unwrap();
    let socket = claude_dir.join("bus.sock");
    let jsonl = claude_dir.join("events.jsonl");
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
            payload_json: "{\"x\":1}".into(),
        })
        .unwrap(),
    )
    .unwrap();
}

#[derive(Resource)]
struct Counter(Arc<AtomicUsize>);

fn count_events(mut ev: MessageReader<ClaudeBusEvent>, c: Res<Counter>) {
    for e in ev.read() {
        c.0.fetch_add(1, Ordering::SeqCst);
        // sanity: kind should match what we published
        assert!(e.kind.starts_with("k"));
    }
}

#[test]
fn plugin_delivers_events_into_bevy() {
    let dir = std::env::temp_dir().join(format!(
        "claude-bus-bevy-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let socket = spawn_daemon(&dir);

    // BusEventPlugin reads socket_path() from $HOME — override HOME so
    // it points at our tempdir. (Setting HOME at runtime is fine in
    // tests; the bus client only reads it once at plugin build.)
    // SAFETY: tests in this binary run sequentially with a single
    // thread for the App build, so no cross-test env race.
    unsafe { std::env::set_var("HOME", &dir) };
    assert_eq!(claude_bus::socket_path().unwrap(), socket);

    let count = Arc::new(AtomicUsize::new(0));
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(BusEventPlugin::default())
        .insert_resource(Counter(count.clone()))
        .add_systems(Update, count_events);

    // Let the subscriber thread complete its Hello before we publish.
    thread::sleep(Duration::from_millis(150));

    publish(&socket, "k1", 1);
    publish(&socket, "k2", 2);
    publish(&socket, "k3", 3);

    // Pump the app a few times; the subscriber thread will keep
    // feeding the channel between ticks.
    let deadline = Instant::now() + Duration::from_secs(2);
    while count.load(Ordering::SeqCst) < 3 && Instant::now() < deadline {
        app.update();
        thread::sleep(Duration::from_millis(20));
    }

    assert_eq!(
        count.load(Ordering::SeqCst),
        3,
        "expected 3 events, got {}",
        count.load(Ordering::SeqCst)
    );
}
