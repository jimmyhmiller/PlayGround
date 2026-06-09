//! Widget↔widget message bus — a general signalling channel that lets
//! several widget panes coordinate as one app (e.g. an editor pane, a
//! results pane and a schema browser making up a "SQL IDE").
//!
//! This is deliberately **separate** from the Claude Code event bus
//! (`ClaudeEvent` / `on_bus`). The Claude bus carries hook events the
//! host receives from Claude Code; this carries control messages widgets
//! send each other ("run this query", "query finished", "table
//! selected"). They never share a channel.
//!
//! ## Model
//!
//! - A widget **publishes** with `emit(topic, payload)` (Rhai host fn) or
//!   `WidgetMsg::Emit` (subprocess) or the `tbmsg` CLI (via IPC). The host
//!   serializes the payload; scripts pass native values, never JSON.
//! - Every widget in the **same editor project** receives the message as
//!   a pushed `on_message(topic, payload, sender)` (Rhai) /
//!   `HostEvent::Message` (subprocess). Delivery wakes the receiver —
//!   there is no polling and no `set_animating` requirement.
//! - `sender` is the publishing widget's id, so a widget can ignore
//!   echoes of its own emits and address targeted replies.
//! - `emit_retained` keeps a message as the topic's last value. A widget
//!   that spawns later receives the retained value for every topic in its
//!   project on init (MQTT-style retain), so late joiners learn current
//!   state without asking.
//!
//! ## Flow
//!
//! `pump_widget_messages` runs every frame:
//!   1. Drain each Rhai widget's outbox + the `external` queue (CLI/IPC).
//!   2. Update the retained store for any `retain` messages.
//!   3. Deliver this frame's messages to every same-project widget, and
//!      deliver the retained backlog to any widget seen for the first time.
//!
//! Project scoping uses `pane_bevy::PaneProject` (a `u64` id). Widgets
//! with no project share the `None` channel. Nothing crosses projects.

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::path::PathBuf;

use bevy::prelude::*;
use serde_json::Value;

use pane_bevy::{PaneKindMarker, PaneProject};

use crate::protocol::HostEvent;
use crate::rhai_widget::{self, RhaiWidget};
use crate::{WidgetIO, WidgetRender};

/// One message awaiting delivery on the widget↔widget bus. Produced by
/// draining widget outboxes and the external (CLI/IPC) queue.
pub struct PendingMsg {
    /// Project channel. `None` = the project-less channel.
    pub project: Option<u64>,
    pub topic: String,
    pub payload: Value,
    /// Publishing widget's id (`"tbmsg"` for the CLI).
    pub sender: String,
    /// Keep as the topic's retained last value for late joiners.
    pub retain: bool,
}

/// Central state for the widget↔widget bus. Ephemeral: retained values
/// live only in memory (the debug log on disk is for `tbmsg tail`, not a
/// persistence layer — it is truncated on app start).
#[derive(Resource, Default)]
pub struct WidgetMsgBus {
    /// (project, topic) → last retained (payload, sender).
    retained: HashMap<(Option<u64>, String), (Value, String)>,
    /// Widget ids that have already received the retained backlog, so a
    /// late joiner gets it exactly once. Pruned to live widgets each pump.
    seen: HashSet<String>,
    /// Messages injected from outside the ECS (the `tbmsg` CLI via IPC),
    /// drained next pump. The host pushes here from its IPC handler.
    external: Vec<PendingMsg>,
}

impl WidgetMsgBus {
    /// Inject a message from outside the ECS (the `tbmsg` CLI / IPC).
    /// Delivered on the next `pump_widget_messages` tick.
    pub fn push_external(&mut self, msg: PendingMsg) {
        self.external.push(msg);
    }
}

/// Best-effort NDJSON debug log of every delivered message, for
/// `tbmsg tail`. Truncated when the app starts (this fn is called once at
/// resource init) so it never grows without bound across restarts.
fn bus_log_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".jim");
    p.push("widget-bus.log");
    Some(p)
}

fn truncate_bus_log() {
    if let Some(p) = bus_log_path() {
        let _ = std::fs::write(&p, b"");
    }
}

fn append_bus_log(m: &PendingMsg) {
    let Some(p) = bus_log_path() else { return };
    let line = serde_json::json!({
        "project": m.project,
        "topic": m.topic,
        "sender": m.sender,
        "retain": m.retain,
        "payload": m.payload,
    });
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&p)
    {
        let _ = writeln!(f, "{}", line);
    }
}

/// Drives the bus once per frame. See module docs for the three phases.
fn pump_widget_messages(
    mut bus: ResMut<WidgetMsgBus>,
    rhai_widgets: Query<(&PaneKindMarker, &RhaiWidget, Option<&PaneProject>)>,
    sub_widgets: Query<(
        Entity,
        &PaneKindMarker,
        &WidgetIO,
        &WidgetRender,
        Option<&PaneProject>,
    )>,
) {
    // ---- Phase 1: collect this frame's outbound messages ----
    // External (CLI/IPC) messages first so their relative order is kept.
    let mut pending: Vec<PendingMsg> = std::mem::take(&mut bus.external);
    let mut live_ids: HashSet<String> = HashSet::new();

    for (kind, w, proj) in &rhai_widgets {
        if kind.0 != rhai_widget::PANE_KIND {
            continue;
        }
        live_ids.insert(w.widget_id.clone());
        let project = proj.map(|p| p.0);
        for out in w.drain_bus_outbox() {
            pending.push(PendingMsg {
                project,
                topic: out.topic,
                payload: out.payload,
                sender: w.widget_id.clone(),
                retain: out.retain,
            });
        }
    }
    // Subprocess widgets only join `live_ids` here; their *emits* are
    // collected in `tick_widget_io` (which owns the stdout channel) and
    // pushed onto `bus.external`.
    for (entity, kind, _io, _render, _proj) in &sub_widgets {
        if kind.0 != crate::PANE_KIND {
            continue;
        }
        live_ids.insert(subprocess_widget_id(entity));
    }

    // ---- Phase 2: update the retained store ----
    for m in &pending {
        if m.retain {
            bus.retained.insert(
                (m.project, m.topic.clone()),
                (m.payload.clone(), m.sender.clone()),
            );
        }
        append_bus_log(m);
    }
    // Drop a widget id from `seen` once it's gone so a future widget that
    // happens to reuse the id (entity bits recycle) still gets a backlog.
    bus.seen.retain(|id| live_ids.contains(id));

    // ---- Phase 3: deliver ----
    // Collect ids that need the retained backlog this pass, then mark them
    // seen after the immutable delivery loops (can't mutate `bus` mid-read).
    let mut newly_seen: Vec<String> = Vec::new();

    for (kind, w, proj) in &rhai_widgets {
        if kind.0 != rhai_widget::PANE_KIND {
            continue;
        }
        let pk = proj.map(|p| p.0);
        if !bus.seen.contains(&w.widget_id) {
            for ((rpk, topic), (payload, sender)) in &bus.retained {
                if *rpk == pk {
                    w.deliver_bus_message(topic.clone(), payload.clone(), sender.clone());
                }
            }
            newly_seen.push(w.widget_id.clone());
        }
        for m in &pending {
            if m.project == pk {
                w.deliver_bus_message(m.topic.clone(), m.payload.clone(), m.sender.clone());
            }
        }
    }

    for (entity, kind, io, render, proj) in &sub_widgets {
        if kind.0 != crate::PANE_KIND || !render.init_sent {
            // Not initialized yet: it'll pick up the retained backlog on
            // the first pump after its `init` line goes out.
            continue;
        }
        let pk = proj.map(|p| p.0);
        let id = subprocess_widget_id(entity);
        if !bus.seen.contains(&id) {
            for ((rpk, topic), (payload, sender)) in &bus.retained {
                if *rpk == pk {
                    send_sub_message(io, topic.clone(), payload.clone(), sender.clone());
                }
            }
            newly_seen.push(id);
        }
        for m in &pending {
            if m.project == pk {
                send_sub_message(io, m.topic.clone(), m.payload.clone(), m.sender.clone());
            }
        }
    }

    bus.seen.extend(newly_seen);
}

/// Stable bus id for a subprocess widget pane. Mirrors the Rhai side's
/// `rw{bits}` scheme so the two id namespaces never collide.
pub(crate) fn subprocess_widget_id(entity: Entity) -> String {
    format!("sw{:x}", entity.to_bits())
}

fn send_sub_message(io: &WidgetIO, topic: String, payload: Value, sender: String) {
    let ev = HostEvent::Message {
        topic,
        payload,
        sender,
    };
    if let Ok(json) = serde_json::to_string(&ev) {
        let _ = io.tx.send(json);
    }
}

pub struct WidgetMsgBusPlugin;

impl Plugin for WidgetMsgBusPlugin {
    fn build(&self, app: &mut App) {
        truncate_bus_log();
        app.init_resource::<WidgetMsgBus>()
            .add_systems(Update, pump_widget_messages);
    }
}
