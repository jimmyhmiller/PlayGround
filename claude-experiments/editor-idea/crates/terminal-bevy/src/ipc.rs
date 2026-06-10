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
        /// `~/.jim/widgets/` (no shell invocation).
        #[serde(default)]
        kind: Option<String>,
    },
    /// `widget projects` — list known projects so external tools can
    /// pick one by name. Response is written back over the same socket
    /// as a single JSON object then EOF: `{"projects":[{"id":N,
    /// "name":"…","active":bool},…]}`.
    ListProjects,
    /// Toggle the 3D project-prism ("cube") overview on/off. Unit
    /// variant, so the wire form is the bare JSON string `"ToggleCube"`.
    /// Primarily a dev/scripting hook mirroring the Cmd+Shift+C keybind.
    ToggleCube,
    /// `tbinbox --project NAME --sender X --body "..."` — append a
    /// message to a project's inbox. The receiver writes the message
    /// to `~/.jim/inbox/<id>.jsonl`; the running app's
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
    /// `tbproject set-cwd [--project NAME] <path>` — write a project's
    /// `default_cwd`. `project` accepts a name or `"active"` (default).
    /// `cwd = None` clears the override so new terminals fall back to
    /// `$HOME`.
    SetProjectDefaultCwd {
        #[serde(default)]
        project: Option<String>,
        #[serde(default)]
        cwd: Option<PathBuf>,
    },
    /// `tbsuggest [--kind K] [--title T] [--command CMD] [--cwd D]
    /// [--reason R] [--config JSON] [--project P]` — park a *suggested*
    /// pane in the drawer (the Quake-style dropdown) rather than
    /// spawning it on the canvas. The AI uses this when it infers a
    /// pane might be useful (e.g. it just ran a command in a side
    /// terminal) but doesn't want to clutter the canvas: the user pulls
    /// it down later and picks it.
    ///
    /// The stored item is a generic `PaneSnapshot`-shaped record: any
    /// registered pane `kind` plus its JSON `config`. As a convenience
    /// for the common "command pane" case, passing `command` with no
    /// explicit `kind`/`config` builds a `run-button` config
    /// (`{title, command, cwd}`) automatically.
    ///
    /// `project` is a *hint*; it's resolved against the live project
    /// list only when the user materializes the suggestion. Fire-and-
    /// forget; no response body.
    SuggestPane {
        /// Registered pane kind. Defaults to `"run-button"` when
        /// `command` is given and no kind is specified.
        #[serde(default)]
        kind: Option<String>,
        #[serde(default)]
        title: Option<String>,
        /// Convenience: shell command for the default `run-button`
        /// kind. Ignored if an explicit `config` is supplied.
        #[serde(default)]
        command: Option<String>,
        #[serde(default)]
        cwd: Option<PathBuf>,
        /// One-line "why this might be useful", shown under the title
        /// in the drawer.
        #[serde(default)]
        reason: Option<String>,
        /// Explicit kind-specific config blob. When present it's stored
        /// verbatim and `command`/`cwd` are ignored.
        #[serde(default)]
        config: Option<serde_json::Value>,
        #[serde(default)]
        project: Option<String>,
        /// Invocation cwd of the CLI. When `project` is unset, the app
        /// maps this to the owning project (by `default_cwd`) so the
        /// suggestion is scoped to the terminal's project rather than
        /// the GUI's active one. Falls back to unscoped (global) if no
        /// project owns the dir.
        #[serde(default)]
        from_cwd: Option<PathBuf>,
    },
    /// Capture the primary window to a PNG at `path`, rendered by the app
    /// itself (Bevy's `Screenshot`), so it works without macOS screen-
    /// recording permission and never steals focus from the user. Wire
    /// form: `{"action":"screenshot","path":"/tmp/x.png"}`. Fire-and-
    /// forget; the file appears a frame or two later.
    Screenshot { path: PathBuf },
    /// `tbclose --project P [--kind K]` — close (despawn) panes in a
    /// project, optionally filtered to a pane `kind` (e.g. `rhai_widget`).
    /// Routes through the normal pane-close path (`on_close` + despawn),
    /// so it's the scriptable equivalent of clicking each close button.
    /// Fire-and-forget; no response body.
    CloseProjectPanes {
        #[serde(default)]
        project: Option<String>,
        /// Pane kind to close (e.g. `"rhai_widget"`, `"widget"`). None =
        /// every pane in the project.
        #[serde(default)]
        kind: Option<String>,
    },
    /// `tbmsg emit --project P --topic T [--json '{...}'] [--retain]` —
    /// publish a message onto the widget↔widget bus from the shell (or a
    /// `proc_spawn`ed child). Delivered to every widget in project `P` as
    /// `on_message` / `HostEvent::Message` with `sender = "tbmsg"`.
    /// Fire-and-forget; no response body.
    WidgetMessage {
        /// Resolved on the GUI side against current `Projects`. A name
        /// (`"datalog-db"`) or `"active"` for the current one.
        #[serde(default)]
        project: Option<String>,
        topic: String,
        /// Parsed JSON payload (object/array/scalar). Defaults to null.
        #[serde(default)]
        payload: serde_json::Value,
        /// Retain as the topic's last value for late-joining widgets.
        #[serde(default)]
        retain: bool,
    },
    /// `tbissue --title "…" [--body "…"] [--project NAME]` — file an
    /// issue into a project's Issues pane from the shell. The app appends
    /// it to `~/.jim/issues/<id>.json` (single-writer, no clobber) and
    /// any open Issues pane for that project shows it live. When
    /// `project` is unset the app maps the caller's `from_cwd` to its
    /// owning project, falling back to the active one.
    AddIssue {
        title: String,
        #[serde(default)]
        body: Option<String>,
        #[serde(default)]
        project: Option<String>,
        #[serde(default)]
        from_cwd: Option<PathBuf>,
    },
    /// Open the command palette overlay, optionally pre-filling the search
    /// query. Used for scripting / verification and (later) as the entry
    /// point for the DeepSeek "Ask" flow. Fire-and-forget.
    OpenPalette {
        #[serde(default)]
        query: Option<String>,
        /// Immediately route the query to DeepSeek (the "Ask" flow)
        /// instead of just opening the action search.
        #[serde(default)]
        ask: bool,
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

/// Dispatch a request to *this* app's own IPC socket — i.e. drive the
/// app the same way the `tb*` CLIs do, over the same wire path. Used by
/// the DeepSeek tool executor so its actions go through the identical
/// `listener → drain_ipc_open_requests` path rather than a parallel
/// in-process dispatch that could drift. Blocking, but the write is a
/// tiny local unix-socket send; the request lands on the next frame.
pub fn dispatch_local(req: &IpcRequest) -> std::io::Result<()> {
    use std::io::Write as _;
    let path = socket_path()
        .ok_or_else(|| std::io::Error::other("no socket path ($HOME unset)"))?;
    let mut stream = UnixStream::connect(path)?;
    let bytes = serde_json::to_vec(req)
        .map_err(|e| std::io::Error::other(format!("serialize ipc request: {e}")))?;
    stream.write_all(&bytes)?;
    let _ = stream.shutdown(std::net::Shutdown::Write);
    Ok(())
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
