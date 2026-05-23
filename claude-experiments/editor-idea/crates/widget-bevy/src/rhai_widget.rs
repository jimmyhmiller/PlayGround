//! In-process Rhai-scripted widgets — script runs on a **worker
//! thread**, so a slow / busy / pathological script can never tank
//! the editor's framerate.
//!
//! # Architecture
//!
//! Each `rhai_widget` pane owns a `WorkerHandle` whose internals are:
//!   - A worker `JoinHandle` running the Rhai engine.
//!   - An mpsc channel `HostToWorker` for events sent from main →
//!     worker (Tick, Resize, ClaudeEvent, Reload, Shutdown).
//!   - A shared `Mutex<Option<Element>>` slot — the latest frame the
//!     worker has produced. Main thread reads it whenever it wants.
//!   - A shared `AtomicU64` `frame_gen` — bumped each time the worker
//!     writes a new frame. Main checks this to avoid relocking the
//!     mutex when nothing has changed.
//!   - A shared `Mutex<Value>` snapshot slot — what the worker last
//!     persisted; main reads from this when the host asks for a
//!     `PaneSnapshot.config`.
//!
//! The main thread never executes Rhai code. It just shuffles events
//! over a channel and reads frames out of a mutex. Worst case the
//! main thread sees a stale frame for one extra tick — it never
//! blocks waiting on the script.
//!
//! # Hot reload
//!
//! Parse on the main thread (cheap, microseconds for typical
//! scripts), then send the new AST over the channel. The worker
//! swaps it in on its next message dispatch and re-initializes its
//! scope from the last known snapshot. Same pattern as `Shutdown`.
//!
//! # Cleanup
//!
//! The pane's `on_close` callback sends `Shutdown` and despawns all
//! sprite entities the widget has been tracking. `Drop` on
//! `WorkerHandle` also sends `Shutdown` as a safety net so a panic-
//! despawned pane doesn't leak the worker thread.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

use bevy::prelude::*;
use bevy::sprite::Anchor;
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use rhai::{Dynamic, Engine, Scope, AST};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use claude_bus_bevy::ClaudeBusEvent;
use pane_bevy::{
    PaneChrome, PaneContentPressed, PaneFont, PaneKindMarker, PaneKindSpec, PaneRect,
    PaneRegistry, PaneTitle, MARGIN, TITLE_H,
};

use crate::WidgetTargets;

use crate::protocol::{CanvasAnchor, CanvasItem, Element, ImageRef};
use crate::{
    canvas_anchor_to_bevy, load_image_for_ref, parse_canvas_color, WidgetClipDirty,
    WidgetImageCache,
};

pub const PANE_KIND: &str = "rhai_widget";

/// Per-tick worker step cadence. The worker self-paces — main can
/// send Tick faster, but the worker drops ticks shorter than this.
const WORKER_MIN_TICK_SECS: f32 = 1.0 / 30.0;

pub fn widgets_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".terminal-bevy");
    p.push("widgets");
    Some(p)
}

// ============================================================
// Worker protocol
// ============================================================

enum HostToWorker {
    /// Per-frame tick from the main thread. Worker decides whether to
    /// actually rerun the script (self-throttles at WORKER_MIN_TICK_SECS).
    Tick {
        dt_secs: f32,
        canvas_w: f32,
        canvas_h: f32,
    },
    /// Mouse press in the pane's content area. Delivered to the
    /// script as a `kind == "click"` entry in the `events` array on
    /// the next tick — same channel events use, so scripts handle
    /// clicks with the same `events` loop they use for Claude events.
    ///
    /// `button_id` is `Some(id)` when the click landed inside a
    /// `Button` element rendered by the previous frame; the host hit-
    /// tests against `WidgetTargets` (populated by `render::render`).
    /// Scripts that just want "which button did the user press" can
    /// read `ev.payload.id` directly instead of doing their own
    /// y-range routing.
    Click {
        local_x: f32,
        local_y: f32,
        shift: bool,
        cmd: bool,
        button_id: Option<String>,
    },
    /// Pane size changed. Worker stores the new size for the next
    /// script run.
    Resize {
        canvas_w: f32,
        canvas_h: f32,
    },
    /// A Claude bus event the script should see on its next tick.
    ClaudeEvent {
        kind: String,
        payload: Value,
    },
    /// Hot reload — main parsed a new AST, worker should swap in and
    /// re-init scope from the last snapshot.
    Reload {
        ast: AST,
    },
    /// Exit the worker loop. Sent by `on_close` and by `Drop`.
    Shutdown,
}

/// What main reads from the worker: the latest frame the script
/// produced, plus diagnostic state.
#[derive(Clone)]
struct WorkerSlots {
    /// Latest fully-rendered frame. Worker overwrites; main clones.
    latest_frame: Arc<Mutex<Option<Element>>>,
    /// Latest snapshot the script published (for persistence).
    snapshot: Arc<Mutex<Value>>,
    /// Bumped each time `latest_frame` is replaced. Main compares
    /// against its last-applied value to skip redundant diffing.
    frame_gen: Arc<AtomicU64>,
    /// Last runtime error the worker encountered. Cleared on next
    /// successful run.
    last_error: Arc<Mutex<Option<String>>>,
}

impl WorkerSlots {
    fn new() -> Self {
        Self {
            latest_frame: Arc::new(Mutex::new(None)),
            snapshot: Arc::new(Mutex::new(Value::Null)),
            frame_gen: Arc::new(AtomicU64::new(0)),
            last_error: Arc::new(Mutex::new(None)),
        }
    }
}

/// Owned by the `RhaiWidget` component. Dropping it sends `Shutdown`
/// to the worker as a backstop in case `on_close` didn't run (e.g.
/// pane despawn took an unusual path).
pub struct WorkerHandle {
    tx: Sender<HostToWorker>,
    slots: WorkerSlots,
    _join: Option<JoinHandle<()>>,
}

impl WorkerHandle {
    fn send(&self, msg: HostToWorker) {
        let _ = self.tx.send(msg);
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(HostToWorker::Shutdown);
    }
}

/// Spawn a worker thread that runs the Rhai engine. `initial_ast` is
/// optional — first launch with a parse error has None and we install
/// later via Reload. `initial_state` carries the snapshot blob from
/// `PaneSnapshot.config.state` so widget state survives restarts.
fn spawn_worker(
    initial_ast: Option<AST>,
    initial_state: Value,
    script_name: String,
) -> WorkerHandle {
    let (tx, rx) = mpsc::channel::<HostToWorker>();
    let slots = WorkerSlots::new();
    let slots_for_thread = slots.clone();
    let join = thread::Builder::new()
        .name(format!("rhai-widget:{}", script_name))
        .spawn(move || worker_main(rx, slots_for_thread, initial_ast, initial_state))
        .expect("spawn rhai-widget worker thread");
    WorkerHandle {
        tx,
        slots,
        _join: Some(join),
    }
}

fn worker_main(
    rx: Receiver<HostToWorker>,
    slots: WorkerSlots,
    mut ast: Option<AST>,
    mut last_state_json: Value,
) {
    let mut engine = Engine::new();
    engine.set_max_expr_depths(256, 128);
    register_host_functions(&mut engine);

    let mut scope = Scope::new();
    let mut canvas_w: f32 = 0.0;
    let mut canvas_h: f32 = 0.0;
    let mut pending_events: VecDeque<(String, Value)> = VecDeque::new();
    let mut init_done = false;
    let mut last_tick_at: Option<std::time::Instant> = None;
    let now = std::time::Instant::now;

    let init_scope = |scope: &mut Scope<'static>, last_state_json: &Value| {
        let initial_state = if last_state_json.is_null() {
            Dynamic::from(rhai::Map::new())
        } else {
            rhai::serde::to_dynamic(last_state_json).unwrap_or(Dynamic::from(rhai::Map::new()))
        };
        scope.push("state", initial_state);
        scope.push("canvas_w", 0.0_f64);
        scope.push("canvas_h", 0.0_f64);
        scope.push("dt", 0.0_f64);
        scope.push("events", rhai::Array::new());
        scope.push("frame", Dynamic::UNIT);
    };

    for msg in rx {
        match msg {
            HostToWorker::Shutdown => break,
            HostToWorker::Reload { ast: new_ast } => {
                ast = Some(new_ast);
                scope.clear();
                init_done = false;
                // last_state_json stays as-is so the new script comes
                // up with the same persisted plant state.
            }
            HostToWorker::Resize { canvas_w: w, canvas_h: h } => {
                canvas_w = w;
                canvas_h = h;
            }
            HostToWorker::ClaudeEvent { kind, payload } => {
                pending_events.push_back((kind, payload));
            }
            HostToWorker::Click {
                local_x,
                local_y,
                shift,
                cmd,
                button_id,
            } => {
                let payload = serde_json::json!({
                    "x": local_x,
                    "y": local_y,
                    "shift": shift,
                    "cmd": cmd,
                    "id": button_id.unwrap_or_default(),
                });
                pending_events.push_back(("click".to_string(), payload));
            }
            HostToWorker::Tick {
                dt_secs,
                canvas_w: w,
                canvas_h: h,
            } => {
                canvas_w = w;
                canvas_h = h;
                let Some(ref ast) = ast else { continue };

                // Self-throttle.
                let now_inst = now();
                if let Some(prev) = last_tick_at {
                    if (now_inst - prev).as_secs_f32() < WORKER_MIN_TICK_SECS {
                        continue;
                    }
                }
                last_tick_at = Some(now_inst);

                if !init_done {
                    init_scope(&mut scope, &last_state_json);
                    init_done = true;
                }

                let _ = scope.set_value("canvas_w", canvas_w as f64);
                let _ = scope.set_value("canvas_h", canvas_h as f64);
                let _ = scope.set_value("dt", dt_secs as f64);
                let events_rhai: rhai::Array = pending_events
                    .drain(..)
                    .map(|(kind, payload)| {
                        let mut m = rhai::Map::new();
                        m.insert("kind".into(), Dynamic::from(kind));
                        m.insert(
                            "payload".into(),
                            rhai::serde::to_dynamic(&payload).unwrap_or(Dynamic::UNIT),
                        );
                        Dynamic::from(m)
                    })
                    .collect();
                let _ = scope.set_value("events", events_rhai);
                let _ = scope.set_value("frame", Dynamic::UNIT);

                if let Err(e) = engine.run_ast_with_scope(&mut scope, ast) {
                    let msg = format!("{}", e);
                    eprintln!("[rhai] runtime error: {}", msg);
                    if let Ok(mut slot) = slots.last_error.lock() {
                        *slot = Some(msg);
                    }
                    continue;
                }
                if let Ok(mut slot) = slots.last_error.lock() {
                    *slot = None;
                }

                let frame_dyn = scope.get_value::<Dynamic>("frame").unwrap_or(Dynamic::UNIT);
                let element = if frame_dyn.is_unit() {
                    None
                } else {
                    match rhai::serde::from_dynamic::<Element>(&frame_dyn) {
                        Ok(el) => Some(el),
                        Err(e) => {
                            eprintln!("[rhai] frame deserialize: {}", e);
                            if let Ok(mut slot) = slots.last_error.lock() {
                                *slot = Some(format!("frame: {}", e));
                            }
                            None
                        }
                    }
                };

                if let Some(state_dyn) = scope.get_value::<Dynamic>("state") {
                    if let Ok(v) = rhai::serde::from_dynamic::<Value>(&state_dyn) {
                        last_state_json = v.clone();
                        if let Ok(mut slot) = slots.snapshot.lock() {
                            *slot = v;
                        }
                    }
                    // Re-push state so the *next* tick definitely sees
                    // the mutations from this tick. Rhai's CoW
                    // semantics on Map values held in Scope can make
                    // inner mutations (`state.foo = ...`) not write
                    // back to the scope slot — this round-trips
                    // through the just-read Dynamic to guarantee they
                    // stick.
                    let _ = scope.set_value("state", state_dyn);
                }

                if let Ok(mut slot) = slots.latest_frame.lock() {
                    *slot = element;
                }
                slots.frame_gen.fetch_add(1, Ordering::Release);
            }
        }
    }
}

// ============================================================
// Component / per-pane state on the main thread
// ============================================================

#[derive(Component)]
pub struct RhaiWidget {
    pub script: String,
    pub script_path: PathBuf,
    pub handle: WorkerHandle,
    /// Last frame generation we applied to the scene. Compared against
    /// `handle.slots.frame_gen` to skip diffing when nothing changed.
    pub applied_frame_gen: u64,
    /// Snapshot mirror used to populate `PaneSnapshot.config.state`.
    /// Updated whenever a new frame_gen comes in.
    pub last_state: Value,
    pub last_size: Vec2,
    pub last_tick_at: Option<std::time::Instant>,
    pub reload_gen: u32,
    pub applied_reload_gen: u32,
    /// Sprite id → entity. Lets us diff frames instead of
    /// despawn+respawn.
    pub sprite_entities: HashMap<String, Entity>,
}

#[derive(Resource)]
struct ScriptWatcher {
    rx: Mutex<Receiver<PathBuf>>,
    _watcher: RecommendedWatcher,
}

pub struct RhaiWidgetPlugin;

impl Plugin for RhaiWidgetPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (register_kind, setup_watcher))
            .add_systems(
                Update,
                (
                    poll_watcher,
                    apply_reloads,
                    forward_clicks_to_workers,
                    forward_inputs_to_workers,
                    apply_latest_frames,
                )
                    .chain(),
            );
    }
}

fn register_kind(mut registry: ResMut<PaneRegistry>) {
    registry.register(PaneKindSpec {
        kind: PANE_KIND,
        display_name: "Rhai Widget",
        radial_icon: None,
        default_size: Vec2::new(720.0, 360.0),
        spawn: rhai_widget_spawn,
        snapshot: rhai_widget_snapshot,
        on_close: Some(rhai_widget_close),
    });
}

fn setup_watcher(world: &mut World) {
    let Some(dir) = widgets_dir() else {
        warn!("rhai_widget: HOME not set, no script hot reload");
        return;
    };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        warn!(
            "rhai_widget: couldn't create {}: {} — no hot reload",
            dir.display(),
            e
        );
        return;
    }
    let garden_path = dir.join("garden.rhai");
    if !garden_path.exists() {
        let _ = std::fs::write(&garden_path, DEFAULT_GARDEN_SCRIPT);
    }
    // dev_panel.rhai is intentionally NOT auto-bootstrapped from an
    // embedded constant. If we shipped a Rust fallback, every edit
    // would tempt me into changing the Rust source instead of the
    // disk file, which requires a rebuild. By having no fallback,
    // the script HAS to live on disk and HAS to be edited live.

    let (tx, rx) = mpsc::channel::<PathBuf>();
    let watcher = match notify::recommended_watcher(
        move |res: notify::Result<notify::Event>| {
            let Ok(ev) = res else { return };
            if !matches!(
                ev.kind,
                EventKind::Modify(_) | EventKind::Create(_) | EventKind::Any
            ) {
                return;
            }
            for path in ev.paths {
                let _ = tx.send(path);
            }
        },
    ) {
        Ok(w) => w,
        Err(e) => {
            warn!("rhai_widget: file watcher failed to start: {}", e);
            return;
        }
    };
    let mut watcher = watcher;
    if let Err(e) = watcher.watch(&dir, RecursiveMode::NonRecursive) {
        warn!("rhai_widget: failed to watch {}: {}", dir.display(), e);
        return;
    }
    world.insert_resource(ScriptWatcher {
        rx: Mutex::new(rx),
        _watcher: watcher,
    });
}

#[derive(Serialize, Deserialize)]
struct RhaiWidgetConfig {
    script: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    state: Value,
}

fn rhai_widget_spawn(world: &mut World, entity: Entity, _content_root: Entity, config: &Value) {
    let cfg: RhaiWidgetConfig = serde_json::from_value(config.clone()).unwrap_or_else(|_| {
        RhaiWidgetConfig {
            script: "garden.rhai".to_string(),
            title: None,
            state: Value::Null,
        }
    });
    if let Some(title) = cfg.title.clone() {
        if let Some(mut t) = world.get_mut::<PaneTitle>(entity) {
            t.0 = title;
        }
    } else if let Some(mut t) = world.get_mut::<PaneTitle>(entity) {
        t.0 = cfg.script.trim_end_matches(".rhai").to_string();
    }

    let script_path = widgets_dir()
        .map(|d| d.join(&cfg.script))
        .unwrap_or_else(|| PathBuf::from(&cfg.script));

    // Parse the script on the main thread now so the worker has an
    // AST from frame 1. Parse failure means worker starts with None
    // and we'll send a Reload after the watcher catches the file.
    let initial_ast = match std::fs::read_to_string(&script_path) {
        Ok(body) => {
            let mut engine = Engine::new();
            engine.set_max_expr_depths(256, 128);
            register_host_functions(&mut engine);
            match engine.compile(&body) {
                Ok(ast) => Some(ast),
                Err(e) => {
                    eprintln!("[rhai] initial parse {}: {}", script_path.display(), e);
                    None
                }
            }
        }
        Err(e) => {
            eprintln!("[rhai] failed to read {}: {}", script_path.display(), e);
            None
        }
    };

    let handle = spawn_worker(initial_ast, cfg.state.clone(), cfg.script.clone());

    world.entity_mut(entity).insert((
        RhaiWidget {
            script: cfg.script.clone(),
            script_path,
            handle,
            applied_frame_gen: 0,
            last_state: cfg.state,
            last_size: Vec2::ZERO,
            last_tick_at: None,
            reload_gen: 0,
            applied_reload_gen: 0,
            sprite_entities: HashMap::new(),
        },
        WidgetTargets::default(),
    ));
}

fn rhai_widget_snapshot(world: &World, entity: Entity) -> Value {
    let Some(w) = world.get::<RhaiWidget>(entity) else {
        return Value::Null;
    };
    // Prefer the live worker-published snapshot; fall back to the
    // last value the host already mirrored from it.
    let state = w
        .handle
        .slots
        .snapshot
        .lock()
        .ok()
        .map(|s| s.clone())
        .unwrap_or_else(|| w.last_state.clone());
    let title = world.get::<PaneTitle>(entity).map(|t| t.0.clone());
    serde_json::json!({
        "script": w.script,
        "title": title,
        "state": state,
    })
}

/// Pane close: tell the worker to stop, then explicitly clear any
/// sprite entities we created so they don't linger as ghosts on the
/// canvas after the pane disappears.
fn rhai_widget_close(world: &mut World, entity: Entity) {
    let entities_to_despawn: Vec<Entity> = world
        .get::<RhaiWidget>(entity)
        .map(|w| {
            // Tell the worker thread to exit promptly.
            w.handle.send(HostToWorker::Shutdown);
            w.sprite_entities.values().copied().collect()
        })
        .unwrap_or_default();
    for e in entities_to_despawn {
        if world.get_entity(e).is_ok() {
            world.entity_mut(e).despawn();
        }
    }
}

// ============================================================
// File watcher → reload
// ============================================================

fn poll_watcher(
    watcher: Option<Res<ScriptWatcher>>,
    mut widgets: Query<&mut RhaiWidget>,
) {
    let Some(watcher) = watcher else { return };
    let paths: Vec<PathBuf> = {
        let rx = watcher.rx.lock().expect("rhai watcher channel poisoned");
        rx.try_iter().collect()
    };
    if paths.is_empty() {
        return;
    }
    let unique: HashSet<PathBuf> = paths.into_iter().collect();
    for mut w in &mut widgets {
        if unique.contains(&w.script_path) {
            w.reload_gen = w.reload_gen.wrapping_add(1);
        }
    }
}

fn apply_reloads(mut widgets: Query<&mut RhaiWidget>) {
    for mut w in &mut widgets {
        if w.applied_reload_gen == w.reload_gen {
            continue;
        }
        w.applied_reload_gen = w.reload_gen;
        let path = w.script_path.clone();
        let Ok(body) = std::fs::read_to_string(&path) else {
            eprintln!("[rhai] reload: couldn't read {}", path.display());
            continue;
        };
        let mut engine = Engine::new();
        engine.set_max_expr_depths(256, 128);
        register_host_functions(&mut engine);
        match engine.compile(&body) {
            Ok(ast) => {
                w.handle.send(HostToWorker::Reload { ast });
                eprintln!("[rhai] reloaded {}", path.display());
            }
            Err(e) => {
                eprintln!("[rhai] reload parse error in {}: {}", path.display(), e);
            }
        }
    }
}

// ============================================================
// Main thread: feed worker, drain claude events, send size + dt
// ============================================================

/// Translate pane-bevy's `PaneContentPressed` into `HostToWorker::Click`
/// for rhai widgets. The script sees the click in its `events` array
/// on the next tick as `{ kind: "click", payload: { x, y, shift, cmd } }`.
fn forward_clicks_to_workers(
    mut presses: MessageReader<PaneContentPressed>,
    keys: Res<ButtonInput<KeyCode>>,
    widgets: Query<(&PaneKindMarker, &RhaiWidget, Option<&WidgetTargets>)>,
) {
    let cmd = keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight);
    for ev in presses.read() {
        let Ok((kind, w, targets)) = widgets.get(ev.pane) else {
            continue;
        };
        if kind.0 != PANE_KIND {
            continue;
        }
        // Hit-test against last-rendered button rects. The script gets
        // the matched id (if any) in `ev.payload.id` and can route
        // directly without estimating row y-ranges.
        let button_id = targets.and_then(|t| {
            t.clicks
                .iter()
                .find(|ct| ct.rect.contains(ev.local_pt))
                .map(|ct| ct.id.clone())
        });
        w.handle.send(HostToWorker::Click {
            local_x: ev.local_pt.x,
            local_y: ev.local_pt.y,
            shift: ev.shift,
            cmd,
            button_id,
        });
    }
}

fn forward_inputs_to_workers(
    time: Res<Time>,
    mut events: MessageReader<ClaudeBusEvent>,
    mut widgets: Query<(&PaneKindMarker, &PaneRect, &mut RhaiWidget)>,
) {
    let new_events: Vec<(String, Value)> = events
        .read()
        .map(|ev| {
            let payload: Value =
                serde_json::from_str(&ev.payload_json).unwrap_or(Value::Null);
            (ev.kind.clone(), payload)
        })
        .collect();

    let now = std::time::Instant::now();
    for (kind, rect, mut w) in &mut widgets {
        if kind.0 != PANE_KIND {
            continue;
        }
        let content_size = Vec2::new(
            (rect.size.x - 2.0 * MARGIN).max(0.0),
            (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
        );
        if w.last_size != content_size && w.last_size != Vec2::ZERO {
            w.handle.send(HostToWorker::Resize {
                canvas_w: content_size.x,
                canvas_h: content_size.y,
            });
        }
        w.last_size = content_size;

        for (k, p) in &new_events {
            w.handle.send(HostToWorker::ClaudeEvent {
                kind: k.clone(),
                payload: p.clone(),
            });
        }

        // Tick the worker. It self-throttles so a 240Hz host doesn't
        // run scripts 240 times per second.
        let dt = match w.last_tick_at {
            Some(prev) => (now - prev).as_secs_f32(),
            None => 0.0,
        };
        w.last_tick_at = Some(now);
        w.handle.send(HostToWorker::Tick {
            dt_secs: dt,
            canvas_w: content_size.x,
            canvas_h: content_size.y,
        });
        let _ = time; // suppress warning; Time is here for future use
    }
}

// ============================================================
// Main thread: read latest frame, diff entities
// ============================================================

fn apply_latest_frames(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut image_cache: ResMut<WidgetImageCache>,
    mut clip_dirty: ResMut<WidgetClipDirty>,
    pane_font: Res<PaneFont>,
    pane_metrics: Res<pane_bevy::PaneFontMetrics>,
    mut q: Query<(
        &PaneKindMarker,
        &PaneChrome,
        &PaneRect,
        &mut RhaiWidget,
        &mut WidgetTargets,
    )>,
    children_q: Query<&Children>,
) {
    for (kind, chrome, rect, mut w, mut targets) in &mut q {
        if kind.0 != PANE_KIND {
            continue;
        }
        let current_gen = w.handle.slots.frame_gen.load(Ordering::Acquire);
        if current_gen == w.applied_frame_gen {
            continue;
        }
        w.applied_frame_gen = current_gen;

        // Grab the frame the worker last produced.
        let frame = w
            .handle
            .slots
            .latest_frame
            .lock()
            .ok()
            .and_then(|s| s.clone());
        // Also mirror snapshot for persistence. Done in two steps so
        // we don't hold the snapshot lock across a borrow of `w`.
        let new_state = w
            .handle
            .slots
            .snapshot
            .lock()
            .ok()
            .map(|s| s.clone());
        if let Some(s) = new_state {
            w.last_state = s;
        }

        let Some(frame) = frame else { continue };

        match frame {
            // Absolute-positioned sprite tree: garden + similar
            // visualizers. Diffs against sprite_entities for cheap
            // per-frame mutation.
            Element::Canvas { children } => {
                diff_render(
                    &mut commands,
                    &mut images,
                    &mut image_cache,
                    chrome.content_root,
                    &children,
                    &mut w.sprite_entities,
                );
            }
            // Flow layout (vstack / hstack / text / button / divider /
            // bar / spacer / etc.). Rebuild from scratch each frame —
            // tree is small enough that diffing isn't worth the code.
            // The script handles its own hit-testing because the
            // protocol's Button id is not echoed back through clicks,
            // so we don't need the click-target Vec we'd populate via
            // WidgetTargets.
            other => {
                // Clear previously-rendered flow children but keep
                // sprite entities tracked separately (in case a widget
                // ever mixes both, which the protocol doesn't currently
                // allow but might in the future).
                if let Ok(children) = children_q.get(chrome.content_root) {
                    for c in children.iter() {
                        if !w.sprite_entities.values().any(|e| *e == c) {
                            commands.entity(c).despawn();
                        }
                    }
                }
                let content_size = Vec2::new(
                    (rect.size.x - 2.0 * MARGIN).max(0.0),
                    (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
                );
                let ctx = crate::render::LayoutCtx {
                    font: pane_font.0.clone(),
                    metrics: *pane_metrics,
                    content_root: chrome.content_root,
                    content_size,
                };
                // Wipe last frame's click targets so a stale button
                // rect from before a script reload doesn't keep
                // matching clicks.
                targets.clicks.clear();
                targets.links.clear();
                crate::render::render(
                    &mut commands,
                    &ctx,
                    &mut targets,
                    &other,
                    Vec2::ZERO,
                    content_size.x,
                    0.0,
                );
            }
        }
        clip_dirty.0 = true;
    }
}

/// Reconcile `items` against `sprite_entities`: reuse entities whose
/// id appears in both old + new, spawn new entities for ids only in
/// new, despawn entities for ids only in old.
///
/// Compared to despawn-everything-then-respawn this saves the ECS
/// from churning hundreds of entities every frame in a busy garden.
fn diff_render(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    image_cache: &mut WidgetImageCache,
    content_root: Entity,
    items: &[CanvasItem],
    sprite_entities: &mut HashMap<String, Entity>,
) {
    let mut seen: HashSet<String> = HashSet::with_capacity(items.len());
    for item in items {
        let id = match item {
            CanvasItem::Sprite { id, .. } => id.clone(),
            CanvasItem::Rect { id, .. } => id.clone(),
        };
        seen.insert(id.clone());
        let existing = sprite_entities.get(&id).copied();
        match item {
            CanvasItem::Sprite {
                x,
                y,
                w,
                h,
                image,
                hue_shift,
                anchor,
                z,
                ..
            } => {
                let Some(handle) = load_image_for_ref(images, image_cache, image, *hue_shift)
                else {
                    continue;
                };
                let sprite = Sprite {
                    image: handle,
                    custom_size: Some(Vec2::new(*w, *h)),
                    ..default()
                };
                let transform = Transform::from_xyz(*x, -*y, *z);
                let anchor_cmp = canvas_anchor_to_bevy(*anchor);
                match existing {
                    Some(e) => {
                        // Reuse — overwrite the components we own.
                        commands.entity(e).insert((sprite, anchor_cmp, transform));
                    }
                    None => {
                        let e = commands
                            .spawn((
                                ChildOf(content_root),
                                sprite,
                                anchor_cmp,
                                transform,
                                Visibility::Inherited,
                            ))
                            .id();
                        sprite_entities.insert(id, e);
                    }
                }
            }
            CanvasItem::Rect {
                x,
                y,
                w,
                h,
                color,
                anchor,
                z,
                ..
            } => {
                let bevy_color =
                    parse_canvas_color(color).unwrap_or(Color::srgb(0.20, 0.22, 0.26));
                let sprite = Sprite {
                    color: bevy_color,
                    custom_size: Some(Vec2::new(*w, *h)),
                    ..default()
                };
                let transform = Transform::from_xyz(*x, -*y, *z);
                let anchor_cmp = canvas_anchor_to_bevy(*anchor);
                match existing {
                    Some(e) => {
                        commands.entity(e).insert((sprite, anchor_cmp, transform));
                    }
                    None => {
                        let e = commands
                            .spawn((
                                ChildOf(content_root),
                                sprite,
                                anchor_cmp,
                                transform,
                                Visibility::Inherited,
                            ))
                            .id();
                        sprite_entities.insert(id, e);
                    }
                }
            }
        }
    }

    // Despawn entities whose id wasn't seen this frame.
    let stale: Vec<String> = sprite_entities
        .keys()
        .filter(|id| !seen.contains(id.as_str()))
        .cloned()
        .collect();
    for id in stale {
        if let Some(e) = sprite_entities.remove(&id) {
            commands.entity(e).despawn();
        }
    }
    let _ = CanvasAnchor::TopLeft; // suppress unused-import warning
    let _: ImageRef = ImageRef::Path {
        path: String::new(),
    };
}

// ============================================================
// Host helper functions registered with the engine
// ============================================================

fn register_host_functions(engine: &mut Engine) {
    // Dev-overrides API (dev_dust / dev_edit / dev_age /
    // dev_time_scale / dev_clear). Lets a scripted dev panel scrub
    // shader inputs in real time so you don't have to wait a real day
    // to see what dust looks like at 24h. No-op if StylePlugin isn't
    // active (e.g. headless tests).
    style_bevy::register_dev_rhai_fns(engine);

    // Named `host_log` (not just `log`) because Rhai already has a
    // `log()` math fn for natural log; same name collides at the
    // overload resolver and the string call silently doesn't reach
    // our handler.
    engine.register_fn("host_log", |msg: &str| {
        // Write to BOTH stderr and a dedicated file so we can verify
        // whether the issue is "script doesn't run" vs "stderr from
        // worker thread doesn't reach the redirected log".
        eprintln!("[rhai] {}", msg);
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/rhai-worker.log")
        {
            let _ = writeln!(f, "{}", msg);
        }
    });

    engine.register_fn("rand", || -> f64 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        (rand_state(nanos as u64) as f64) / (u32::MAX as f64)
    });

    engine.register_fn("rand_int", |lo: i64, hi: i64| -> i64 {
        if hi <= lo {
            return lo;
        }
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let r = rand_state(nanos as u64) as i64;
        lo + r.rem_euclid(hi - lo + 1)
    });

    engine.register_fn("time", || -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    });

    engine.register_fn("hash_str", |s: &str| -> i64 {
        let mut h: u64 = 14695981039346656037;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        h as i64
    });
}

fn rand_state(seed: u64) -> u32 {
    let mut x = (seed as u32) | 1;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

// ============================================================
// Default garden script (written if missing)
// ============================================================

const DEFAULT_GARDEN_SCRIPT: &str = r###"// garden.rhai — Claude garden widget.
//
// This script is RUN AS PROCEDURAL TOP-LEVEL STATEMENTS once per tick
// by a dedicated worker thread (~30Hz). The worker pre-populates these
// scope variables before each run:
//
//   state      — Map. Persistent across runs. Mutate in place.
//   events     — Array of `#{ kind, payload }` Claude bus events that
//                arrived since the last tick.
//   dt         — f64 seconds since last tick.
//   canvas_w   — f64 pane content width in px.
//   canvas_h   — f64 pane content height in px.
//
// Write the rendered frame to `frame` (a `#{ type: "canvas", children: [...] }`).

if !("plants" in state) { state.plants = []; }
if !("next_id" in state) { state.next_id = 1; }

const PLANTS_SHEET = "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/editor-idea/crates/terminal-bevy/assets/garden/plants.png";
const TILE_W = 12;
const TILE_H = 24;
const NUM_TILES = 78;
const SPRITE_W = 200.0;
const SPRITE_H = 400.0;
const GROW_SECS = 8.0;
const MAX_PLANTS = 120;
const GROUND_INSET = 18.0;
const GROUND_H = 5.0;

// ----- handle events -----
for ev in events {
    let kind = ev.kind;
    let payload = ev.payload;
    let payload_is_map = type_of(payload) == "map";

    if kind == "pre_tool_use" {
        let cwd = if payload_is_map && "cwd" in payload { payload.cwd } else { "" };
        let tool_name = if payload_is_map && "tool_name" in payload { payload.tool_name } else { "" };
        let species = if tool_name == "Edit" || tool_name == "Write" || tool_name == "MultiEdit" || tool_name == "NotebookEdit" {
            "flower"
        } else if tool_name == "Bash" {
            "vine"
        } else if tool_name == "Read" || tool_name == "Grep" || tool_name == "Glob" {
            "grass"
        } else if tool_name == "Task" || tool_name == "Agent" {
            "tree"
        } else {
            "seed"
        };

        let plot_key = hash_str(cwd);
        let cx_frac = ((plot_key & 0xFFFF).to_float() / 65535.0) * 0.84 + 0.08;
        let hue = ((plot_key.abs() / 65536) % 360).to_float();
        let spread_px = (canvas_w * 0.15).max(40.0).min(180.0);
        let center_x = cx_frac * canvas_w;
        let offset = (rand() - 0.5) * 2.0 * spread_px;
        let margin = SPRITE_W * 0.5;
        let x = (center_x + offset).max(margin).min(canvas_w - margin);

        // Wider scale ranges so heights actually vary.
        let target_scale = if species == "tree" {
            0.90 + rand() * 1.10
        } else if species == "flower" {
            0.45 + rand() * 1.20
        } else if species == "vine" {
            0.40 + rand() * 1.00
        } else if species == "grass" {
            0.35 + rand() * 0.95
        } else {
            0.20 + rand() * 0.80
        };

        state.plants.push(#{
            id: state.next_id,
            species: species,
            age: 0.0,
            target_scale: target_scale,
            x: x,
            hue: (hue + (rand() - 0.5) * 30.0) % 360.0,
            tile: (plot_key.abs() % NUM_TILES).to_int(),
        });
        state.next_id = state.next_id + 1;
    } else if kind == "user_prompt_submit" {
        let cwd = if payload_is_map && "cwd" in payload { payload.cwd } else { "" };
        let plot_key = hash_str(cwd);
        let cx_frac = ((plot_key & 0xFFFF).to_float() / 65535.0) * 0.84 + 0.08;
        let center_x = cx_frac * canvas_w;
        let margin = SPRITE_W * 0.5;
        let i = 0;
        while i < 3 {
            let offset = (rand() - 0.5) * 80.0;
            let x = (center_x + offset).max(margin).min(canvas_w - margin);
            state.plants.push(#{
                id: state.next_id,
                species: "seed",
                age: 0.0,
                target_scale: 0.30 + rand() * 0.20,
                x: x,
                hue: rand() * 360.0,
                tile: (plot_key.abs() % NUM_TILES).to_int(),
            });
            state.next_id = state.next_id + 1;
            i = i + 1;
        }
    } else if kind == "stop" {
        for plant in state.plants {
            if plant.age < GROW_SECS {
                plant.age = GROW_SECS;
            }
        }
    }
}

for plant in state.plants {
    if plant.age < GROW_SECS {
        plant.age = plant.age + dt;
        if plant.age > GROW_SECS { plant.age = GROW_SECS; }
    }
}

while state.plants.len() > MAX_PLANTS {
    state.plants.shift();
}

// ----- build frame -----
let items = [];
items.push(#{
    type: "rect",
    id: "sky",
    x: 0.0, y: 0.0,
    w: canvas_w, h: canvas_h,
    color: "#11192a",
    anchor: "top-left",
    z: 0.0,
});
items.push(#{
    type: "rect",
    id: "ground",
    x: 0.0, y: canvas_h - GROUND_INSET - GROUND_H,
    w: canvas_w, h: GROUND_H,
    color: "#1a2e1a",
    anchor: "top-left",
    z: 0.05,
});
for plant in state.plants {
    let t = plant.age / GROW_SECS;
    if t > 1.0 { t = 1.0; }
    let scale = plant.target_scale * (0.25 + 0.75 * t);
    items.push(#{
        type: "sprite",
        id: `plant-${plant.id}`,
        x: plant.x,
        y: canvas_h - GROUND_INSET,
        w: SPRITE_W * scale,
        h: SPRITE_H * scale,
        image: "tile",
        path: PLANTS_SHEET,
        tile_w: TILE_W,
        tile_h: TILE_H,
        col: plant.tile,
        row: 0,
        hue_shift: plant.hue,
        anchor: "bottom-center",
        z: 1.0,
    });
}

frame = #{
    type: "canvas",
    children: items,
};
"###;
