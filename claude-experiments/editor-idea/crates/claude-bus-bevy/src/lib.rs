//! Bevy integration for `claude-bus`.
//!
//! Add [`BusEventPlugin`] to your `App` and any system can react to
//! Claude Code hook events by reading `MessageReader<ClaudeBusEvent>`
//! — no socket handling at the call site, no per-pane wiring.
//!
//! Example:
//! ```ignore
//! app.add_plugins(claude_bus_bevy::BusEventPlugin::default())
//!    .add_systems(Update, react_to_tool_use);
//!
//! fn react_to_tool_use(mut ev: MessageReader<ClaudeBusEvent>) {
//!     for e in ev.read() {
//!         if e.kind == "pre_tool_use" { /* ... */ }
//!     }
//! }
//! ```
//!
//! Bevy 0.18 renamed `Event`/`EventReader` to `Message`/`MessageReader`;
//! we follow that convention.
//!
//! The plugin owns a single background thread (via
//! `claude_bus::client::Subscriber`) that pumps the unix socket; a
//! per-frame system in `PreUpdate` drains its channel into Bevy's
//! message queue. The worker reconnects automatically across bus
//! restarts.

use std::sync::Mutex;

use bevy::prelude::*;

use claude_bus::client::{BusItem, Subscriber};

/// One Claude Code hook event, mirrored from the bus.
///
/// `payload_json` is left as a string because most consumers only care
/// about a handful of kinds — parse on demand with `serde_json::from_str`
/// rather than paying the cost on every event.
#[derive(Message, Debug, Clone)]
pub struct ClaudeBusEvent {
    pub seq: u64,
    pub kind: String,
    pub ts: u64,
    pub terminal_session_id: String,
    pub claude_pid: u32,
    pub payload_json: String,
}

/// Emitted when the worker had to skip ahead (bus ring evicted events
/// we hadn't received yet). Indicates a gap between
/// `last_delivered_seq` and `replay_from`; consumers that care about
/// completeness should read the JSONL log for that range.
#[derive(Message, Debug, Clone)]
pub struct ClaudeBusGap {
    pub last_delivered_seq: Option<u64>,
    pub replay_from: u64,
}

/// Emitted when the socket dropped (bus restarting / crashed).
#[derive(Message, Debug, Clone, Copy)]
pub struct ClaudeBusDisconnected;

/// Emitted when the socket came back after a `ClaudeBusDisconnected`.
#[derive(Message, Debug, Clone, Copy)]
pub struct ClaudeBusReconnected;

/// Plugin handle. Configure with `since_seq` if you want a startup
/// replay; default is live-only.
#[derive(Default, Clone)]
pub struct BusEventPlugin {
    pub since_seq: Option<u64>,
}

impl BusEventPlugin {
    pub fn since(mut self, seq: u64) -> Self {
        self.since_seq = Some(seq);
        self
    }
}

/// Resource wrapping the subscriber. `Subscriber` itself holds an
/// `mpsc::Receiver` which is `!Sync`, so the resource needs a `Mutex`
/// to satisfy Bevy's `Send + Sync` bound. Contention is trivial — one
/// lock per frame per app.
#[derive(Resource)]
struct BusSubscription(Mutex<Subscriber>);

impl Plugin for BusEventPlugin {
    fn build(&self, app: &mut App) {
        let Some(socket) = claude_bus::socket_path() else {
            warn!("claude-bus-bevy: HOME not set, no events will be delivered");
            return;
        };
        let sub = Subscriber::spawn(socket, self.since_seq);
        app.insert_resource(BusSubscription(Mutex::new(sub)))
            .add_message::<ClaudeBusEvent>()
            .add_message::<ClaudeBusGap>()
            .add_message::<ClaudeBusDisconnected>()
            .add_message::<ClaudeBusReconnected>()
            .add_systems(PreUpdate, pump_bus_events);
    }
}

/// Per-frame: drain everything the worker has queued and fan out into
/// the matching Bevy messages.
fn pump_bus_events(
    sub: Res<BusSubscription>,
    mut events: MessageWriter<ClaudeBusEvent>,
    mut gaps: MessageWriter<ClaudeBusGap>,
    mut disc: MessageWriter<ClaudeBusDisconnected>,
    mut rec: MessageWriter<ClaudeBusReconnected>,
) {
    let items = {
        let s = sub.0.lock().expect("bus subscription poisoned");
        s.drain()
    };
    for item in items {
        match item {
            BusItem::Event(ev) => {
                events.write(ClaudeBusEvent {
                    seq: ev.seq,
                    kind: ev.kind,
                    ts: ev.ts,
                    terminal_session_id: ev.terminal_session_id,
                    claude_pid: ev.claude_pid,
                    payload_json: ev.payload_json,
                });
            }
            BusItem::Gap {
                last_delivered_seq,
                replay_from,
            } => {
                gaps.write(ClaudeBusGap {
                    last_delivered_seq,
                    replay_from,
                });
            }
            BusItem::Disconnected => {
                disc.write(ClaudeBusDisconnected);
            }
            BusItem::Reconnected => {
                rec.write(ClaudeBusReconnected);
            }
        }
    }
}
