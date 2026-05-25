//! Bridge between Rhai workers and the main thread.
//!
//! Rhai workers run on background threads with no ECS access. They
//! issue host calls (`uniform_set("name", v)`, `mask_paint(...)`,
//! `emit(...)`) which are encoded as [`DynamicMsg`] variants and
//! pushed through an mpsc channel. A main-thread system
//! ([`drain_script_msgs`]) consumes the queue each frame and applies
//! the effects via the ECS.
//!
//! For *reads* (e.g. `pane_rects()`, `uniform_get(...)`) the worker
//! reads from a shared snapshot that the main thread refreshes each
//! frame — no blocking, no waiting on a reply.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Mutex, OnceLock, RwLock, Arc};

use bevy::prelude::*;
use rhai::{Dynamic, Engine, Map};
use serde_json::Value;

/// All host calls a worker can issue. Each variant maps 1:1 to a
/// Rhai-callable host function. Adding a new variant + a matching
/// `register_fn` block in [`register_script_host_fns`] is how a new
/// primitive gets added — but the goal is to keep this enum *small
/// and frozen*: every "new behavior" should be doable via the
/// existing variants composed in scripts.
#[derive(Debug)]
pub enum DynamicMsg {
    /// `uniform_set("name", scalar | [f32;2] | [f32;4])` — writes the
    /// value into the active material's uniform buffer at the offset
    /// the schema records for `name`. Silently no-ops if the name
    /// doesn't exist in the current shader.
    SetUniformF32(String, f32),
    SetUniformVec2(String, [f32; 2]),
    SetUniformVec4(String, [f32; 4]),
    /// `mask_paint("name", x, y, radius, value)` — stamp a soft brush
    /// (cosine falloff) at `(x, y)` in window-logical pixels into the
    /// named texture. Texture is auto-allocated on first reference.
    MaskPaint {
        name: String,
        x: f32,
        y: f32,
        radius: f32,
        value: f32,
    },
    /// `mask_fill("name", value)` — set every R-channel pixel to
    /// `value * 255`. Sugar: `mask_clear(name)` = `mask_fill(name, 0)`.
    MaskFill(String, f32),
    /// `emit("kind", payload)` — push an event onto the global bus
    /// that every script worker can listen to via the `events` array.
    Emit(String, Value),
    /// `schedule(delay_secs, "kind", payload)` — fire an emit after
    /// `delay_secs`. Bookkeeping lives on the main thread.
    Schedule {
        delay_secs: f32,
        kind: String,
        payload: Value,
    },
    /// `state_set(key, value)` — write per-project script state.
    StateSet(String, Value),
}

#[derive(Resource)]
pub struct ScriptReceiver(Mutex<Receiver<DynamicMsg>>);

/// Events waiting to be delivered to scripts on their next tick.
/// Drained by [`tick_system_script`] each frame and fed into the
/// script's `events` scope variable.
///
/// Anyone — Rust systems or scripts via `emit(...)` — can push here.
/// Rust producers use [`EventBus::push`]; the rhai bridge routes
/// `emit`/`schedule` host calls through the dynamic-msg channel which
/// drain into this bus.
#[derive(Resource, Default)]
pub struct EventBus {
    pub pending: Vec<(String, Value)>,
}

impl EventBus {
    pub fn push(&mut self, kind: impl Into<String>, payload: Value) {
        self.pending.push((kind.into(), payload));
    }
}

/// Timed events that will fire later. Maintained by the dynamic-msg
/// drain (which moves `Schedule` variants here) and flushed each
/// frame into the bus once their delay has elapsed.
#[derive(Resource, Default)]
pub struct ScheduledEvents {
    pub items: Vec<ScheduledEvent>,
}

#[derive(Clone, Debug)]
pub struct ScheduledEvent {
    pub fire_at: f32,
    pub kind: String,
    pub payload: Value,
}

/// Shared snapshot the main thread refreshes each frame, accessible
/// to workers without blocking. Holds whatever's needed for read-side
/// host fns (uniform_get, pane_rects, state_get).
#[derive(Resource, Default, Clone)]
pub struct ScriptSnapshot {
    inner: Arc<RwLock<SnapshotData>>,
}

#[derive(Default, Clone)]
pub struct SnapshotData {
    /// Per-project state, serialized for cheap reads from any thread.
    pub state: serde_json::Map<String, Value>,
    /// Pane rects in window-local logical pixels.
    pub pane_rects: Vec<PaneRectSnap>,
    /// Uniform values from the last main-thread tick, keyed by name.
    pub uniforms: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug)]
pub struct PaneRectSnap {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub kind: String,
}

impl ScriptSnapshot {
    pub fn read<R>(&self, f: impl FnOnce(&SnapshotData) -> R) -> R {
        let g = self.inner.read().unwrap_or_else(|p| p.into_inner());
        f(&g)
    }
    pub fn write(&self, f: impl FnOnce(&mut SnapshotData)) {
        if let Ok(mut g) = self.inner.write() {
            f(&mut g);
        }
    }
}

static SCRIPT_SENDER: OnceLock<Sender<DynamicMsg>> = OnceLock::new();
static SNAPSHOT: OnceLock<ScriptSnapshot> = OnceLock::new();

pub fn script_sender() -> Option<Sender<DynamicMsg>> {
    SCRIPT_SENDER.get().cloned()
}

pub fn snapshot() -> Option<ScriptSnapshot> {
    SNAPSHOT.get().cloned()
}

pub struct ScriptBridgePlugin;

impl Plugin for ScriptBridgePlugin {
    fn build(&self, app: &mut App) {
        let (tx, rx) = mpsc::channel::<DynamicMsg>();
        let _ = SCRIPT_SENDER.set(tx);
        let snap = ScriptSnapshot::default();
        let _ = SNAPSHOT.set(snap.clone());
        app.insert_resource(ScriptReceiver(Mutex::new(rx)))
            .insert_resource(snap)
            .init_resource::<EventBus>()
            .init_resource::<ScheduledEvents>();
    }
}

/// Convert a `serde_json::Value` payload back to a Rhai map (or
/// scalar) for delivery to scripts. Used when building the `events`
/// array that gets pushed into a script's scope.
pub fn value_to_rhai_public(v: Value) -> Option<Dynamic> {
    value_to_rhai(v)
}

/// Register every host fn a script can call on the Rhai engine. Idempotent
/// across multiple engines (each worker calls this on its own engine).
/// All fns are no-ops if `ScriptBridgePlugin` hasn't been added.
pub fn register_script_host_fns(engine: &mut Engine) {
    let Some(tx) = script_sender() else { return };

    // ---- uniform_set ----
    let tx_f = tx.clone();
    engine.register_fn("uniform_set", move |name: &str, value: f64| {
        let _ = tx_f.send(DynamicMsg::SetUniformF32(name.to_string(), value as f32));
    });
    let tx_i = tx.clone();
    engine.register_fn("uniform_set", move |name: &str, value: i64| {
        let _ = tx_i.send(DynamicMsg::SetUniformF32(name.to_string(), value as f32));
    });
    let tx_arr = tx.clone();
    engine.register_fn("uniform_set", move |name: &str, value: rhai::Array| {
        match value.len() {
            2 => {
                let a = dynamic_to_f32(&value[0]);
                let b = dynamic_to_f32(&value[1]);
                let _ = tx_arr.send(DynamicMsg::SetUniformVec2(name.to_string(), [a, b]));
            }
            4 => {
                let a = dynamic_to_f32(&value[0]);
                let b = dynamic_to_f32(&value[1]);
                let c = dynamic_to_f32(&value[2]);
                let d = dynamic_to_f32(&value[3]);
                let _ = tx_arr.send(DynamicMsg::SetUniformVec4(
                    name.to_string(),
                    [a, b, c, d],
                ));
            }
            _ => {
                eprintln!("[bridge] uniform_set(name, [...]) expects array of len 2 or 4");
            }
        }
    });

    // ---- uniform_get (read from snapshot) ----
    engine.register_fn("uniform_get", |name: &str| -> Dynamic {
        let Some(snap) = snapshot() else { return Dynamic::UNIT };
        snap.read(|d| {
            d.uniforms
                .get(name)
                .cloned()
                .and_then(|v| value_to_rhai(v))
                .unwrap_or(Dynamic::UNIT)
        })
    });

    // ---- mask_paint / fill / clear ----
    let tx_paint = tx.clone();
    engine.register_fn(
        "mask_paint",
        move |name: &str, x: f64, y: f64, radius: f64, value: f64| {
            let _ = tx_paint.send(DynamicMsg::MaskPaint {
                name: name.to_string(),
                x: x as f32,
                y: y as f32,
                radius: radius as f32,
                value: value as f32,
            });
        },
    );
    let tx_fill = tx.clone();
    engine.register_fn("mask_fill", move |name: &str, value: f64| {
        let _ = tx_fill.send(DynamicMsg::MaskFill(name.to_string(), value as f32));
    });
    let tx_clear = tx.clone();
    engine.register_fn("mask_clear", move |name: &str| {
        let _ = tx_clear.send(DynamicMsg::MaskFill(name.to_string(), 0.0));
    });

    // ---- emit / schedule ----
    let tx_emit = tx.clone();
    engine.register_fn("emit", move |kind: &str, payload: Map| {
        let v = rhai_map_to_value(payload);
        let _ = tx_emit.send(DynamicMsg::Emit(kind.to_string(), v));
    });
    // Convenience: emit with no payload.
    let tx_emit_empty = tx.clone();
    engine.register_fn("emit", move |kind: &str| {
        let _ = tx_emit_empty.send(DynamicMsg::Emit(kind.to_string(), Value::Null));
    });
    let tx_sched = tx.clone();
    engine.register_fn("schedule", move |delay: f64, kind: &str, payload: Map| {
        let v = rhai_map_to_value(payload);
        let _ = tx_sched.send(DynamicMsg::Schedule {
            delay_secs: delay as f32,
            kind: kind.to_string(),
            payload: v,
        });
    });

    // ---- pane_rects (read from snapshot) ----
    engine.register_fn("pane_rects", || -> rhai::Array {
        let Some(snap) = snapshot() else { return rhai::Array::new() };
        snap.read(|d| {
            d.pane_rects
                .iter()
                .map(|r| {
                    let mut m = Map::new();
                    m.insert("x".into(), Dynamic::from(r.x as f64));
                    m.insert("y".into(), Dynamic::from(r.y as f64));
                    m.insert("w".into(), Dynamic::from(r.w as f64));
                    m.insert("h".into(), Dynamic::from(r.h as f64));
                    m.insert("kind".into(), Dynamic::from(r.kind.clone()));
                    Dynamic::from(m)
                })
                .collect()
        })
    });

    // ---- state_get / state_set ----
    engine.register_fn("state_get", |key: &str, default: Dynamic| -> Dynamic {
        let Some(snap) = snapshot() else { return default };
        snap.read(|d| {
            d.state
                .get(key)
                .cloned()
                .and_then(|v| value_to_rhai(v))
                .unwrap_or(default)
        })
    });
    let tx_state = tx.clone();
    engine.register_fn("state_set", move |key: &str, value: Dynamic| {
        let Ok(v) = rhai_dyn_to_value(&value) else { return };
        let _ = tx_state.send(DynamicMsg::StateSet(key.to_string(), v));
    });

    // Drain `tx` so the borrow checker doesn't grumble (keeps last
    // clone alive too).
    let _keep = tx;
}

fn dynamic_to_f32(d: &Dynamic) -> f32 {
    if let Some(f) = d.clone().try_cast::<f64>() {
        return f as f32;
    }
    if let Some(i) = d.clone().try_cast::<i64>() {
        return i as f32;
    }
    0.0
}

fn rhai_map_to_value(m: Map) -> Value {
    let mut out = serde_json::Map::new();
    for (k, v) in m.into_iter() {
        if let Ok(jv) = rhai_dyn_to_value(&v) {
            out.insert(k.to_string(), jv);
        }
    }
    Value::Object(out)
}

fn rhai_dyn_to_value(d: &Dynamic) -> Result<Value, ()> {
    rhai::serde::to_dynamic::<Value>(serde_json::Value::Null).ok();
    if let Some(f) = d.clone().try_cast::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return Ok(Value::Number(n));
        }
    }
    if let Some(i) = d.clone().try_cast::<i64>() {
        return Ok(Value::Number(i.into()));
    }
    if let Some(s) = d.clone().try_cast::<String>() {
        return Ok(Value::String(s));
    }
    if let Some(s) = d.clone().try_cast::<rhai::ImmutableString>() {
        return Ok(Value::String(s.to_string()));
    }
    if let Some(b) = d.clone().try_cast::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Some(arr) = d.clone().try_cast::<rhai::Array>() {
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            if let Ok(v) = rhai_dyn_to_value(&item) {
                out.push(v);
            }
        }
        return Ok(Value::Array(out));
    }
    if let Some(map) = d.clone().try_cast::<Map>() {
        return Ok(rhai_map_to_value(map));
    }
    if d.is_unit() {
        return Ok(Value::Null);
    }
    Err(())
}

fn value_to_rhai(v: Value) -> Option<Dynamic> {
    match v {
        Value::Null => Some(Dynamic::UNIT),
        Value::Bool(b) => Some(Dynamic::from(b)),
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                Some(Dynamic::from(f))
            } else if let Some(i) = n.as_i64() {
                Some(Dynamic::from(i))
            } else {
                None
            }
        }
        Value::String(s) => Some(Dynamic::from(s)),
        Value::Array(a) => {
            let mut out = rhai::Array::new();
            for item in a {
                if let Some(d) = value_to_rhai(item) {
                    out.push(d);
                }
            }
            Some(Dynamic::from(out))
        }
        Value::Object(o) => {
            let mut m = Map::new();
            for (k, v) in o {
                if let Some(d) = value_to_rhai(v) {
                    m.insert(k.into(), d);
                }
            }
            Some(Dynamic::from(m))
        }
    }
}

/// Drain queued messages on the main thread. The actual side-effects
/// (writing to material, painting masks, emitting events) are routed
/// to specific systems via small intermediate resources so this file
/// doesn't depend on every sibling module's internals.
#[derive(Resource, Default)]
pub struct PendingScriptOps {
    pub uniform_writes: Vec<UniformWrite>,
    pub mask_ops: Vec<MaskOp>,
    pub emits: Vec<(String, Value)>,
    pub schedules: Vec<(f32, String, Value)>,
    pub state_writes: Vec<(String, Value)>,
}

#[derive(Debug)]
pub enum UniformWrite {
    F32(String, f32),
    Vec2(String, [f32; 2]),
    Vec4(String, [f32; 4]),
}

#[derive(Debug)]
pub enum MaskOp {
    Paint {
        name: String,
        x: f32,
        y: f32,
        radius: f32,
        value: f32,
    },
    Fill(String, f32),
}

pub fn drain_script_msgs(
    rx: Res<ScriptReceiver>,
    mut pending: ResMut<PendingScriptOps>,
    mut bus: ResMut<EventBus>,
    mut scheduled: ResMut<ScheduledEvents>,
    time: Res<Time>,
) {
    let Ok(rx) = rx.0.lock() else { return };
    while let Ok(msg) = rx.try_recv() {
        match msg {
            DynamicMsg::SetUniformF32(n, v) => pending.uniform_writes.push(UniformWrite::F32(n, v)),
            DynamicMsg::SetUniformVec2(n, v) => pending.uniform_writes.push(UniformWrite::Vec2(n, v)),
            DynamicMsg::SetUniformVec4(n, v) => pending.uniform_writes.push(UniformWrite::Vec4(n, v)),
            DynamicMsg::MaskPaint { name, x, y, radius, value } => {
                pending.mask_ops.push(MaskOp::Paint { name, x, y, radius, value });
            }
            DynamicMsg::MaskFill(n, v) => pending.mask_ops.push(MaskOp::Fill(n, v)),
            DynamicMsg::Emit(k, p) => bus.push(k, p),
            DynamicMsg::Schedule { delay_secs, kind, payload } => {
                scheduled.items.push(ScheduledEvent {
                    fire_at: time.elapsed_secs() + delay_secs.max(0.0),
                    kind,
                    payload,
                });
            }
            DynamicMsg::StateSet(k, v) => pending.state_writes.push((k, v)),
        }
    }
}

/// Move any scheduled events whose `fire_at` has passed into the
/// pending event bus. Runs each frame after `drain_script_msgs`.
pub fn fire_scheduled_events(
    time: Res<Time>,
    mut scheduled: ResMut<ScheduledEvents>,
    mut bus: ResMut<EventBus>,
) {
    let now = time.elapsed_secs();
    let mut still_pending = Vec::with_capacity(scheduled.items.len());
    for ev in scheduled.items.drain(..) {
        if ev.fire_at <= now {
            bus.push(ev.kind, ev.payload);
        } else {
            still_pending.push(ev);
        }
    }
    scheduled.items = still_pending;
}
