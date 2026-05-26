//! IPC between the running `terminal-bevy` app and external CLIs
//! (`tbopen`, `tbwidget`).
//!
//! Wire format: one JSON object per connection, terminated by EOF —
//! keeps both ends trivial (no framing, no length prefix). The app
//! reads to EOF, parses, dispatches; the CLI sends one request and
//! shuts down its half of the socket.
//!
//! Requests are tagged via the `action` field. Unknown actions are
//! logged and dropped so adding new ones never breaks older daemons.
//!
//! Socket lives at `<data_dir()>/socket`. We unlink the path on
//! startup so a stale socket from a previous crashed run doesn't
//! block bind. Stale-while-running detection (e.g. another instance
//! actually listening) isn't handled — first run wins, subsequent
//! ones fail to bind and log.

use std::io::Read;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use bevy::winit::{EventLoopProxy, WinitUserEvent};
use serde::{Deserialize, Serialize};

use crate::data_dir;

/// Tagged wire format for external CLI → app. Each variant maps onto a
/// `PendingActions` entry on the next frame.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum IpcRequest {
    /// `tbopen <file> [--project NAME]` — open a file in an editor pane.
    OpenFile {
        path: PathBuf,
        #[serde(default)]
        project: Option<String>,
    },
    /// `tbwidget [--title T] [--cwd D] [--project P] -- <cmd> [args...]` —
    /// spawn a new widget pane running `cmd`. When `args` is non-empty
    /// the command runs directly (no shell); otherwise `cmd` is fed to
    /// `sh -c`.
    ///
    /// `position` is an optional window-space top-left `[x, y]` for the
    /// new pane; `size` is an optional `[w, h]`. Both default to the
    /// project's normal cascade / widget kind's default size.
    SpawnWidget {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        title: Option<String>,
        #[serde(default)]
        cwd: Option<PathBuf>,
        #[serde(default)]
        project: Option<String>,
        #[serde(default)]
        position: Option<[f32; 2]>,
        #[serde(default)]
        size: Option<[f32; 2]>,
        /// Optional widget kind override. Default is the subprocess
        /// widget kind (`"widget"`). Pass `"rhai_widget"` to spawn a
        /// Rhai-scripted in-process widget — `command` is then
        /// interpreted as the script filename under
        /// `~/.terminal-bevy/widgets/` (no shell invocation).
        #[serde(default)]
        kind: Option<String>,
    },
    /// `widget projects` — list known projects so external tools can
    /// pick one by name. Response is written back over the same socket
    /// as a single JSON object then EOF: `{"projects":[{"id":N,
    /// "name":"…","active":bool},…]}`.
    ListProjects,
    /// `tbinbox --project NAME --sender X --body "..."` — append a
    /// message to a project's inbox. The receiver writes the message
    /// to `~/.terminal-bevy/inbox/<id>.jsonl`; the running app's
    /// `InboxPane` picks it up on its next disk poll. Fire-and-forget;
    /// no response body.
    SendInbox {
        /// Resolved on the GUI side against current `Projects`. May be
        /// a name (`"editor-idea"`) or `"active"` for the current one.
        #[serde(default)]
        project: Option<String>,
        #[serde(default)]
        sender: Option<String>,
        #[serde(default)]
        subject: Option<String>,
        body: String,
    },
}

/// One accepted IPC connection: the parsed request plus the open socket,
/// kept around so request-response variants (e.g. `ListProjects`) can
/// write a reply back from the main thread. For fire-and-forget variants
/// the receiver simply drops the stream.
pub struct IpcMessage {
    pub req: IpcRequest,
    pub stream: UnixStream,
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
) -> Option<Receiver<IpcMessage>> {
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
    let (tx, rx) = channel::<IpcMessage>();
    thread::Builder::new()
        .name("tb-ipc".into())
        .spawn(move || listener_loop(listener, tx, wakeup))
        .ok()?;
    Some(rx)
}

fn listener_loop(
    listener: UnixListener,
    tx: Sender<IpcMessage>,
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
        match serde_json::from_str::<IpcRequest>(&buf) {
            Ok(req) => {
                if tx.send(IpcMessage { req, stream }).is_err() {
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
