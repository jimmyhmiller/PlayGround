//! IPC between the running `terminal-bevy` app and external CLIs
//! (currently just `tbopen`).
//!
//! Wire format: one JSON object per connection, terminated by EOF —
//! keeps both ends trivial (no framing, no length prefix). The app
//! reads to EOF, parses, dispatches; the CLI sends one request and
//! shuts down its half of the socket.
//!
//! Socket lives at `<data_dir()>/socket`. We unlink the path on
//! startup so a stale socket from a previous crashed run doesn't
//! block bind. Stale-while-running detection (e.g. another instance
//! actually listening) isn't handled — first run wins, subsequent
//! ones fail to bind and log.

use std::io::Read;
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use bevy::winit::{EventLoopProxy, WinitUserEvent};
use serde::{Deserialize, Serialize};

use crate::data_dir;

/// Wire format for `tbopen` → app. Mirrors `OpenFileRequest` minus the
/// origin (the CLI never positions panes — that's the picker's job).
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenRequest {
    pub path: PathBuf,
    /// Project name (case-insensitive lookup) or `None` for active.
    #[serde(default)]
    pub project: Option<String>,
}

/// Path of the IPC socket. `None` if `$HOME` isn't set.
pub fn socket_path() -> Option<PathBuf> {
    Some(data_dir()?.join("socket"))
}

/// Spawn the listener thread. Returns the receiver half of an mpsc
/// channel that fires once per accepted connection. Returns `None` if
/// we can't open the socket — the app keeps running, just without IPC.
///
/// The optional `wakeup` is winit's event-loop proxy; without it, IPC
/// requests sit in the channel until the next reactive-mode tick (up
/// to 5s), which feels broken. With it, each accepted connection
/// immediately wakes the main loop so the drain system runs that frame.
pub fn spawn_listener(
    wakeup: Option<EventLoopProxy<WinitUserEvent>>,
) -> Option<Receiver<OpenRequest>> {
    let path = socket_path()?;
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("[ipc] mkdir {}: {}", parent.display(), e);
            return None;
        }
    }
    // Best-effort unlink of stale socket. If another instance is
    // actually listening, the bind below will still succeed (we just
    // stole the path) — that's a known quirk; first instance to bind
    // after the unlink wins.
    let _ = std::fs::remove_file(&path);
    let listener = match UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[ipc] bind {}: {}", path.display(), e);
            return None;
        }
    };
    let (tx, rx) = channel::<OpenRequest>();
    thread::Builder::new()
        .name("tb-ipc".into())
        .spawn(move || listener_loop(listener, tx, wakeup))
        .ok()?;
    Some(rx)
}

fn listener_loop(
    listener: UnixListener,
    tx: Sender<OpenRequest>,
    wakeup: Option<EventLoopProxy<WinitUserEvent>>,
) {
    for conn in listener.incoming() {
        let mut stream = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[ipc] accept: {}", e);
                continue;
            }
        };
        let mut buf = String::new();
        if let Err(e) = stream.read_to_string(&mut buf) {
            eprintln!("[ipc] read: {}", e);
            continue;
        }
        match serde_json::from_str::<OpenRequest>(&buf) {
            Ok(req) => {
                if tx.send(req).is_err() {
                    // Receiver dropped — app is shutting down.
                    return;
                }
                if let Some(p) = &wakeup {
                    let _ = p.send_event(WinitUserEvent::WakeUp);
                }
            }
            Err(e) => eprintln!("[ipc] parse: {} (raw: {:?})", e, buf),
        }
    }
}
