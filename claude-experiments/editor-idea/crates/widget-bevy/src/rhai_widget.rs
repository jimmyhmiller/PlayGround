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

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

use bevy::prelude::*;
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use rhai::{Dynamic, Engine, EvalAltResult, Scope, AST};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use claude_bus_bevy::ClaudeBusEvent;
use pane_bevy::{
    PaneChrome, PaneContentDragged, PaneContentHovered, PaneContentPressed,
    PaneContentReleased, PaneFont, PaneKindMarker, PaneKindSpec, PaneRect, PaneRegistry,
    PaneTitle, MARGIN, TITLE_H,
};

use crate::WidgetTargets;

use crate::protocol::{CanvasAnchor, CanvasItem, Element, ImageRef};
use crate::{
    canvas_anchor_to_bevy, load_image_for_ref, parse_canvas_color, WidgetClipDirty,
    WidgetImageCache,
};

pub const PANE_KIND: &str = "rhai_widget";

/// Frame cadence used **only when a widget has opted into animation**
/// via `set_animating(true)`. Idle widgets receive no Tick at all; the
/// main thread checks `WorkerSlots::animating` before sending one. So
/// this isn't a polling cadence, it's a max frame rate for the small
/// subset of widgets that are actively in motion.
const ANIMATION_MIN_FRAME_SECS: f32 = 1.0 / 30.0;

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
    /// Animation frame. Only sent while the worker has set its
    /// `animating` flag — idle widgets get zero ticks. Drives
    /// `on_frame(dt)` in the script.
    Tick {
        dt_secs: f32,
    },
    /// Mouse press in the pane's content area. Drives `on_click(x, y,
    /// shift, cmd, id)` in the script.
    ///
    /// `button_id` is `Some(id)` when the click landed inside a
    /// `Button` element rendered by the previous frame; the host hit-
    /// tests against `WidgetTargets` (populated by `render::render`).
    /// Scripts that just want "which button did the user press" can
    /// read the `id` argument directly instead of doing their own
    /// y-range routing.
    Click {
        local_x: f32,
        local_y: f32,
        shift: bool,
        cmd: bool,
        button_id: Option<String>,
    },
    /// Cursor moved while the left button is held after a content
    /// press. Drives `on_drag(x, y)` in the script. Coords may sit
    /// outside the content rect — handlers like chess use that to
    /// know the user has dragged past the board edge.
    Drag { local_x: f32, local_y: f32 },
    /// Left button released after a content press. Drives
    /// `on_release(x, y)` in the script. Drag-and-drop widgets commit
    /// here; click-style widgets typically ignore (they've already
    /// acted on Click at press time).
    Release { local_x: f32, local_y: f32 },
    /// Cursor moved over the pane content area with no button held.
    /// Drives `on_hover(x, y)` in the script. `x = f32::INFINITY`
    /// signals the cursor LEFT the pane — widgets should clear any
    /// hover indicator on receipt.
    Hover { local_x: f32, local_y: f32 },
    /// Pane size changed. Drives `on_resize(w, h)` in the script and
    /// updates `canvas_w` / `canvas_h` in scope so `render` sees the
    /// new size.
    Resize {
        canvas_w: f32,
        canvas_h: f32,
    },
    /// A navigation key press routed to the focused widget. Drives
    /// `on_key(key)` in the script. `key` is a stable name like
    /// "ArrowLeft" / "ArrowRight" / "Home" / "End".
    Key {
        key: String,
    },
    /// A Claude bus event. Drives `on_event(kind, payload)`.
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
/// produced, plus diagnostic state and the animation flag main checks
/// before deciding whether to send Tick.
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
    /// Set by the script via `set_animating(true)`. Main reads this
    /// each frame and only sends `Tick` if true. Idle widgets =
    /// zero Rhai eval and zero CPU.
    animating: Arc<AtomicBool>,
    /// Set by the script via `request_render()`. Worker calls
    /// `render(canvas_w, canvas_h)` and publishes a frame whenever
    /// this is set after a handler completes, then clears it.
    render_dirty: Arc<AtomicBool>,
}

impl WorkerSlots {
    fn new() -> Self {
        Self {
            latest_frame: Arc::new(Mutex::new(None)),
            snapshot: Arc::new(Mutex::new(Value::Null)),
            frame_gen: Arc::new(AtomicU64::new(0)),
            last_error: Arc::new(Mutex::new(None)),
            animating: Arc::new(AtomicBool::new(false)),
            render_dirty: Arc::new(AtomicBool::new(true)),
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

/// Per-worker state that the worker thread fully owns. Held in a
/// struct so handler dispatch can be split into methods without a
/// giant closure capture list.
struct Worker {
    engine: Engine,
    scope: Scope<'static>,
    slots: WorkerSlots,
    ast: Option<AST>,
    /// Last persisted state JSON. Re-seeded on Reload so the script
    /// comes back up with the same garden state after a hot reload.
    last_state_json: Value,
    canvas_w: f32,
    canvas_h: f32,
    /// True between Reload and the next successful top-level eval.
    /// Top-level runs once per AST (declares state shape, defines
    /// handler functions); handlers run on every event after.
    needs_top_level: bool,
}

impl Worker {
    fn init_scope_state(&mut self) {
        let initial_state = if self.last_state_json.is_null() {
            Dynamic::from(rhai::Map::new())
        } else {
            rhai::serde::to_dynamic(&self.last_state_json)
                .unwrap_or_else(|_| Dynamic::from(rhai::Map::new()))
        };
        self.scope.clear();
        self.scope.push("state", initial_state);
        self.scope.push("canvas_w", self.canvas_w as f64);
        self.scope.push("canvas_h", self.canvas_h as f64);
    }

    /// Run the script's top-level statements: var initialization,
    /// function definitions, one-shot migrations. Only happens once
    /// per AST load — every subsequent event runs handler functions.
    fn run_top_level(&mut self) -> bool {
        let Some(ref ast) = self.ast else { return false };
        if let Err(e) = self.engine.run_ast_with_scope(&mut self.scope, ast) {
            self.set_error(format!("top-level: {}", e));
            return false;
        }
        self.clear_error();
        true
    }

    /// Try to call a script-defined handler. Returns Ok(()) if the
    /// handler ran (or doesn't exist — that's not an error). Re-pushes
    /// `state` afterwards to make sure inner mutations persist past
    /// Rhai's CoW Map semantics.
    fn call_handler<A: rhai::FuncArgs>(&mut self, name: &str, args: A) {
        let Some(ref ast) = self.ast else { return };
        match self
            .engine
            .call_fn::<Dynamic>(&mut self.scope, ast, name, args)
        {
            Ok(_) => self.clear_error(),
            Err(e) => match *e {
                EvalAltResult::ErrorFunctionNotFound(ref n, _) if n == name => {
                    // Handler is optional; missing one isn't an error.
                }
                _ => self.set_error(format!("{}: {}", name, e)),
            },
        }
        // Round-trip state to defeat CoW so inner `state.foo = ...`
        // mutations are visible to later handlers.
        if let Some(state_dyn) = self.scope.get_value::<Dynamic>("state") {
            let _ = self.scope.set_value("state", state_dyn);
        }
    }

    /// If this is the first event since spawn or reload, evaluate the
    /// script's top-level statements (state migrations, fn defs) and
    /// call `on_init`. Returns false if top-level eval errored — the
    /// caller should bail without dispatching the triggering event.
    fn ensure_initialized(&mut self) -> bool {
        if !self.needs_top_level {
            return true;
        }
        self.init_scope_state();
        if !self.run_top_level() {
            return false;
        }
        self.needs_top_level = false;
        self.call_handler("on_init", ());
        true
    }

    fn maybe_render_and_persist(&mut self) {
        // Persist state to snapshot slot every cycle so closes /
        // restarts don't lose recent changes.
        if let Some(state_dyn) = self.scope.get_value::<Dynamic>("state") {
            if let Ok(v) = rhai::serde::from_dynamic::<Value>(&state_dyn) {
                self.last_state_json = v.clone();
                if let Ok(mut slot) = self.slots.snapshot.lock() {
                    *slot = v;
                }
            }
        }

        if !self.slots.render_dirty.swap(false, Ordering::AcqRel) {
            return;
        }
        let Some(ref ast) = self.ast else { return };
        let frame_dyn = match self.engine.call_fn::<Dynamic>(
            &mut self.scope,
            ast,
            "render",
            (self.canvas_w as f64, self.canvas_h as f64),
        ) {
            Ok(v) => v,
            Err(e) => {
                if let EvalAltResult::ErrorFunctionNotFound(ref n, _) = *e {
                    if n == "render" {
                        // No render fn defined — widget produces no
                        // visual. Valid (a script could be purely a
                        // bus → state_set bridge).
                        return;
                    }
                }
                self.set_error(format!("render: {}", e));
                return;
            }
        };

        let element = if frame_dyn.is_unit() {
            None
        } else {
            match rhai::serde::from_dynamic::<Element>(&frame_dyn) {
                Ok(el) => Some(el),
                Err(e) => {
                    self.set_error(format!("render: frame deserialize: {}", e));
                    return;
                }
            }
        };
        if let Ok(mut slot) = self.slots.latest_frame.lock() {
            *slot = element;
        }
        self.slots.frame_gen.fetch_add(1, Ordering::Release);
    }

    fn set_error(&self, msg: String) {
        eprintln!("[rhai] {}", msg);
        if let Ok(mut slot) = self.slots.last_error.lock() {
            *slot = Some(msg);
        }
    }
    fn clear_error(&self) {
        if let Ok(mut slot) = self.slots.last_error.lock() {
            *slot = None;
        }
    }
}

fn worker_main(
    rx: Receiver<HostToWorker>,
    slots: WorkerSlots,
    initial_ast: Option<AST>,
    initial_state: Value,
) {
    let mut engine = Engine::new();
    engine.set_max_expr_depths(256, 128);
    register_host_functions(&mut engine, &slots);

    let mut worker = Worker {
        engine,
        scope: Scope::new(),
        slots,
        ast: initial_ast,
        last_state_json: initial_state,
        canvas_w: 0.0,
        canvas_h: 0.0,
        needs_top_level: true,
    };

    for msg in rx {
        match msg {
            HostToWorker::Shutdown => break,
            HostToWorker::Reload { ast: new_ast } => {
                worker.ast = Some(new_ast);
                worker.needs_top_level = true;
                // last_state_json stays so the new script comes up
                // with the same persisted state.
            }
            HostToWorker::Resize { canvas_w: w, canvas_h: h } => {
                worker.canvas_w = w;
                worker.canvas_h = h;
                if !worker.ensure_initialized() { continue; }
                let _ = worker.scope.set_value("canvas_w", w as f64);
                let _ = worker.scope.set_value("canvas_h", h as f64);
                worker.call_handler("on_resize", (w as f64, h as f64));
                worker.maybe_render_and_persist();
            }
            HostToWorker::Key { key } => {
                if !worker.ensure_initialized() { continue; }
                worker.call_handler("on_key", (key,));
                worker.maybe_render_and_persist();
            }
            HostToWorker::ClaudeEvent { kind, payload } => {
                if !worker.ensure_initialized() { continue; }
                let payload_dyn =
                    rhai::serde::to_dynamic(&payload).unwrap_or(Dynamic::UNIT);
                worker.call_handler("on_event", (kind, payload_dyn));
                worker.maybe_render_and_persist();
            }
            HostToWorker::Click {
                local_x,
                local_y,
                shift,
                cmd,
                button_id,
            } => {
                if !worker.ensure_initialized() { continue; }
                let id = button_id.unwrap_or_default();
                worker.call_handler(
                    "on_click",
                    (local_x as f64, local_y as f64, shift, cmd, id),
                );
                worker.maybe_render_and_persist();
            }
            HostToWorker::Drag { local_x, local_y } => {
                if !worker.ensure_initialized() { continue; }
                worker.call_handler("on_drag", (local_x as f64, local_y as f64));
                worker.maybe_render_and_persist();
            }
            HostToWorker::Release { local_x, local_y } => {
                if !worker.ensure_initialized() { continue; }
                worker.call_handler("on_release", (local_x as f64, local_y as f64));
                worker.maybe_render_and_persist();
            }
            HostToWorker::Hover { local_x, local_y } => {
                if !worker.ensure_initialized() { continue; }
                worker.call_handler("on_hover", (local_x as f64, local_y as f64));
                worker.maybe_render_and_persist();
            }
            HostToWorker::Tick { dt_secs } => {
                if !worker.ensure_initialized() { continue; }
                // Tick only arrives while animating; on_frame is the
                // only handler that wants a per-frame heartbeat.
                worker.call_handler("on_frame", (dt_secs as f64,));
                worker.maybe_render_and_persist();
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
    /// True when the script defines an `on_click` handler — i.e. it's
    /// an interactive widget rather than ambient decoration. Used to
    /// treat a canvas widget's whole content as a hot-zone so its
    /// clicks route while pinned (canvas widgets self-route and publish
    /// no per-element `WidgetTargets`, so they'd otherwise be
    /// click-through when pinned). Recomputed on reload.
    pub wants_clicks: bool,
}

/// Does this AST define an `on_click` handler? (Cheap metadata scan.)
fn ast_wants_clicks(ast: &AST) -> bool {
    ast.iter_functions().any(|f| f.name == "on_click")
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
                    forward_drags_to_workers,
                    forward_releases_to_workers,
                    forward_hovers_to_workers,
                    forward_keys_to_workers,
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
    let picker_path = dir.join("style_picker.rhai");
    if !picker_path.exists() {
        let _ = std::fs::write(&picker_path, DEFAULT_STYLE_PICKER_SCRIPT);
    }
    let editor_path = dir.join("theme_editor.rhai");
    if !editor_path.exists() {
        let _ = std::fs::write(&editor_path, DEFAULT_THEME_EDITOR_SCRIPT);
    }
    let chess_path = dir.join("chess.rhai");
    if !chess_path.exists() {
        let _ = std::fs::write(&chess_path, DEFAULT_CHESS_SCRIPT);
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
            // Parse-only engine; the worker thread builds its own with
            // slot-aware host fns. Parse doesn't need any registration
            // since Rhai resolves identifiers at evaluation, not parse.
            let mut engine = Engine::new();
            engine.set_max_expr_depths(256, 128);
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

    let wants_clicks = initial_ast.as_ref().map(ast_wants_clicks).unwrap_or(false);
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
            wants_clicks,
        },
        WidgetTargets::default(),
        crate::WidgetScroll::default(),
        crate::WidgetHover::default(),
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
        match engine.compile(&body) {
            Ok(ast) => {
                w.wants_clicks = ast_wants_clicks(&ast);
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
    widgets: Query<(
        &PaneKindMarker,
        &RhaiWidget,
        Option<&WidgetTargets>,
        Option<&crate::WidgetScroll>,
    )>,
) {
    let cmd = keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight);
    for ev in presses.read() {
        let Ok((kind, w, targets, scroll)) = widgets.get(ev.pane) else {
            continue;
        };
        if kind.0 != PANE_KIND {
            continue;
        }
        // `ev.local_pt` is pane-content coords with scroll=0 baked in.
        // Click rects in `targets` are stored relative to content_root's
        // local frame, which slides up by `scroll.y` when the user
        // scrolls. Add the scroll offset so the hit-test matches the
        // visually-rendered position of each rect.
        let scroll_y = scroll.map(|s| s.y).unwrap_or(0.0);
        let hit_pt = ev.local_pt + Vec2::new(0.0, scroll_y);
        let button_id = targets.and_then(|t| {
            t.clicks
                .iter()
                .find(|ct| ct.rect.contains(hit_pt))
                .map(|ct| ct.id.clone())
        });
        w.handle.send(HostToWorker::Click {
            local_x: hit_pt.x,
            local_y: hit_pt.y,
            shift: ev.shift,
            cmd,
            button_id,
        });
    }
}

fn forward_drags_to_workers(
    mut events: MessageReader<PaneContentDragged>,
    widgets: Query<
        (&PaneKindMarker, &RhaiWidget, Option<&crate::WidgetScroll>),
    >,
) {
    for ev in events.read() {
        let Ok((kind, w, scroll)) = widgets.get(ev.pane) else { continue };
        if kind.0 != PANE_KIND { continue; }
        let scroll_y = scroll.map(|s| s.y).unwrap_or(0.0);
        let pt = ev.local_pt + Vec2::new(0.0, scroll_y);
        w.handle.send(HostToWorker::Drag { local_x: pt.x, local_y: pt.y });
    }
}

fn forward_releases_to_workers(
    mut events: MessageReader<PaneContentReleased>,
    widgets: Query<
        (&PaneKindMarker, &RhaiWidget, Option<&crate::WidgetScroll>),
    >,
) {
    for ev in events.read() {
        let Ok((kind, w, scroll)) = widgets.get(ev.pane) else { continue };
        if kind.0 != PANE_KIND { continue; }
        let scroll_y = scroll.map(|s| s.y).unwrap_or(0.0);
        let pt = ev.local_pt + Vec2::new(0.0, scroll_y);
        w.handle.send(HostToWorker::Release { local_x: pt.x, local_y: pt.y });
    }
}

fn forward_hovers_to_workers(
    mut events: MessageReader<PaneContentHovered>,
    widgets: Query<
        (&PaneKindMarker, &RhaiWidget, Option<&crate::WidgetScroll>),
    >,
) {
    for ev in events.read() {
        let Ok((kind, w, scroll)) = widgets.get(ev.pane) else { continue };
        if kind.0 != PANE_KIND { continue; }
        // INFINITY is the "cursor left" sentinel — pass through
        // untouched so the script can detect it.
        let pt = if ev.local_pt.x.is_finite() {
            let scroll_y = scroll.map(|s| s.y).unwrap_or(0.0);
            ev.local_pt + Vec2::new(0.0, scroll_y)
        } else {
            ev.local_pt
        };
        w.handle.send(HostToWorker::Hover { local_x: pt.x, local_y: pt.y });
    }
}

/// Route navigation keys (arrows / Home / End) to the focused rhai
/// widget as `on_key`. Terminals consume these themselves when focused,
/// so there's no conflict; we only fire when a rhai widget holds focus
/// and isn't in text-edit mode (which owns the keyboard).
fn forward_keys_to_workers(
    keys: Res<ButtonInput<KeyCode>>,
    focused: Res<pane_bevy::FocusedPane>,
    widgets: Query<(&PaneKindMarker, &RhaiWidget, Option<&crate::WidgetEditMode>)>,
) {
    let Some(pane) = focused.0 else { return };
    let Ok((kind, w, edit)) = widgets.get(pane) else { return };
    if kind.0 != PANE_KIND || edit.is_some() {
        return;
    }
    for (code, name) in [
        (KeyCode::ArrowLeft, "ArrowLeft"),
        (KeyCode::ArrowRight, "ArrowRight"),
        (KeyCode::ArrowUp, "ArrowUp"),
        (KeyCode::ArrowDown, "ArrowDown"),
        (KeyCode::Home, "Home"),
        (KeyCode::End, "End"),
    ] {
        if keys.just_pressed(code) {
            w.handle.send(HostToWorker::Key { key: name.to_string() });
        }
    }
}

fn forward_inputs_to_workers(
    time: Res<Time>,
    pane_zoom: Res<pane_bevy::PaneZoom>,
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
        // PaneRect is canvas-units now; pane Transform handles zoom.
        let content_size = Vec2::new(
            (rect.size.x - 2.0 * MARGIN).max(0.0),
            (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
        );
        // Send Resize whenever content_size changes, including the
        // very first non-zero size after spawn. The previous guard
        // (`w.last_size != Vec2::ZERO`) suppressed exactly that case,
        // so the worker stayed at canvas_w=canvas_h=0 until the user
        // manually dragged a corner — visible as garden plants
        // rendering at the top of the pane (y = canvas_h - inset).
        if w.last_size != content_size && content_size != Vec2::ZERO {
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

        // Tick only fires while the widget has opted into animation
        // (`set_animating(true)` in the script). Most widgets stay
        // event-driven and never receive a Tick — that's the whole
        // point of the event-driven worker contract.
        if !w.handle.slots.animating.load(Ordering::Acquire) {
            w.last_tick_at = None;
            continue;
        }
        let dt = match w.last_tick_at {
            Some(prev) => (now - prev).as_secs_f32(),
            None => 0.0,
        };
        if dt > 0.0 && dt < ANIMATION_MIN_FRAME_SECS {
            // Cap animation frame rate; drop sub-frame ticks.
            continue;
        }
        w.last_tick_at = Some(now);
        w.handle.send(HostToWorker::Tick { dt_secs: dt });
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
    theme: Res<style_bevy::Theme>,
    fonts: Res<style_bevy::FontRegistry>,
    pane_zoom: Res<pane_bevy::PaneZoom>,
    mut q: Query<(
        &PaneKindMarker,
        &PaneChrome,
        &PaneRect,
        &mut RhaiWidget,
        &mut WidgetTargets,
        &mut crate::WidgetScroll,
        Option<&crate::WidgetHover>,
    )>,
    children_q: Query<&Children>,
) {
    let theme_changed = theme.is_changed();
    let zoom = pane_zoom.0.max(0.0001);
    for (kind, chrome, rect, mut w, mut targets, mut scroll, hover) in &mut q {
        if kind.0 != PANE_KIND {
            continue;
        }
        let current_gen = w.handle.slots.frame_gen.load(Ordering::Acquire);
        // Theme changes also need to re-emit the frame so widgets
        // pick up new palette colors; without this, switching presets
        // wouldn't retone existing widget buttons/text/dividers until
        // the next event triggered a worker re-render.
        if current_gen == w.applied_frame_gen && !theme_changed {
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
                    &pane_font.0,
                    &fonts,
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
                    palette: crate::render::WidgetPalette::from_theme(&theme),
                    theme: theme.clone(),
                    fonts: fonts.clone(),
                    focused_input: None,
                    caret_visible: true,
                    hovered_click_id: hover.and_then(|h| h.click_id.clone()),
                };
                // Wipe last frame's click targets so a stale button
                // rect from before a script reload doesn't keep
                // matching clicks.
                targets.clicks.clear();
                targets.links.clear();
                let consumed = crate::render::render(
                    &mut commands,
                    &ctx,
                    &mut targets,
                    &other,
                    Vec2::ZERO,
                    content_size.x,
                    0.0,
                );
                // Update scroll bounds based on what the render
                // actually consumed. Clamp current scroll to new max
                // so resizing the pane shorter doesn't strand the
                // user past the new bottom.
                let new_max = (consumed.y - content_size.y).max(0.0);
                if (scroll.max_y - new_max).abs() > 0.1 {
                    scroll.max_y = new_max;
                }
                if scroll.y > new_max {
                    scroll.y = new_max;
                }
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
    default_font: &Handle<Font>,
    fonts: &style_bevy::FontRegistry,
) {
    let mut seen: HashSet<String> = HashSet::with_capacity(items.len());
    for item in items {
        let id = match item {
            CanvasItem::Sprite { id, .. } => id.clone(),
            CanvasItem::Rect { id, .. } => id.clone(),
            CanvasItem::Text { id, .. } => id.clone(),
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
                rotation,
                ..
            } => {
                let bevy_color =
                    parse_canvas_color(color).unwrap_or(Color::srgb(0.20, 0.22, 0.26));
                let sprite = Sprite {
                    color: bevy_color,
                    custom_size: Some(Vec2::new(*w, *h)),
                    ..default()
                };
                // Canvas is y-down but the world is y-up (we render at
                // -y), so a clockwise canvas rotation is a negative
                // world rotation about z.
                let mut transform = Transform::from_xyz(*x, -*y, *z);
                if *rotation != 0.0 {
                    transform.rotation = Quat::from_rotation_z(-rotation.to_radians());
                }
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
            CanvasItem::Text {
                x,
                y,
                value,
                color,
                size,
                family,
                anchor,
                z,
                ..
            } => {
                let font_size = size.unwrap_or(14.0).max(1.0);
                let col = color
                    .as_deref()
                    .and_then(parse_canvas_color)
                    .unwrap_or(Color::WHITE);
                let font = match family.as_deref() {
                    Some(f) => fonts.resolve(f),
                    None => default_font.clone(),
                };
                let anchor_cmp = canvas_anchor_to_bevy(*anchor);
                let transform = Transform::from_xyz(*x, -*y, *z);
                let text = Text2d::new(value.clone());
                let text_font = TextFont {
                    font,
                    font_size,
                    ..default()
                };
                let text_color = TextColor(col);
                // No-wrap: short labels (button text, status lines) must
                // never break mid-word. Without this, "New game" wraps
                // to "New\ngame" inside a narrow canvas because Bevy's
                // default TextLayout still inserts soft breaks.
                let layout = bevy::text::TextLayout::new_with_no_wrap();
                match existing {
                    Some(e) => {
                        commands.entity(e).insert((
                            text, text_font, text_color, anchor_cmp, transform, layout,
                        ));
                    }
                    None => {
                        let e = commands
                            .spawn((
                                ChildOf(content_root),
                                text,
                                text_font,
                                text_color,
                                anchor_cmp,
                                transform,
                                layout,
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

fn register_host_functions(engine: &mut Engine, slots: &WorkerSlots) {
    // Generic style-bevy host primitives: uniform_set/get, mask_paint,
    // emit/schedule, state_set/get, pane_rects. Same registration the
    // system-script side uses, so any rhai widget can drive the
    // dynamic shader pipeline.
    style_bevy::register_script_host_fns(engine);
    // Preset switching: `list_styles()` and `set_active_style(name)`
    // for widgets like the style picker. Empty-string clears, falling
    // back to the per-project theme.
    style_bevy::register_preset_host_fns(engine);
    // Live theme editing: `theme_tokens()`, `theme_get`, `theme_get_oklch`,
    // `theme_set_oklch`, `theme_set_color`, `theme_set_number`, `theme_reset`.
    style_bevy::register_theme_host_fns(engine);

    // Animation opt-in. Defaults to false. Main thread checks this
    // before sending Tick — idle widgets cost zero CPU.
    let animating = slots.animating.clone();
    engine.register_fn("set_animating", move |on: bool| {
        animating.store(on, Ordering::Release);
    });

    // Mark that the widget's visual needs to be re-rendered. After the
    // current handler returns, the worker calls `render(canvas_w,
    // canvas_h)` and publishes the resulting frame. Handlers that
    // don't change visible state can skip the call and save the
    // render cost entirely.
    let dirty = slots.render_dirty.clone();
    engine.register_fn("request_render", move || {
        dirty.store(true, Ordering::Release);
    });

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

    // `widget_asset("chess/wK.png")` → absolute path to a file under
    // crates/widget-bevy/assets/. Resolves at build time via
    // CARGO_MANIFEST_DIR so scripts don't hardcode the user's home.
    // Production deploys that move the binary out of the source tree
    // will need a runtime override (env var) before this works there;
    // matches the existing garden-script absolute-path practice.
    engine.register_fn("widget_asset", |rel: &str| -> String {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("assets");
        path.push(rel);
        path.to_string_lossy().into_owned()
    });

    engine.register_fn("hash_str", |s: &str| -> i64 {
        let mut h: u64 = 14695981039346656037;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        h as i64
    });

    // Generic subprocess bridge — lets any widget spawn a child
    // process and pipe lines to/from it (the chess widget drives a UCI
    // engine with it). One registry per worker, captured by these
    // closures; when the worker's Rhai `Engine` drops at pane close,
    // the registry drops and kills every child it owns, so panes can't
    // leak processes. Reads are non-blocking (poll from on_frame).
    let procs = Arc::new(Mutex::new(crate::subprocess::ProcRegistry::new()));
    {
        let procs = procs.clone();
        engine.register_fn("proc_spawn", move |cmd: &str| -> i64 {
            procs.lock().map(|mut r| r.spawn(cmd, &[])).unwrap_or(-1)
        });
    }
    {
        let procs = procs.clone();
        engine.register_fn("proc_spawn", move |cmd: &str, args: rhai::Array| -> i64 {
            let args: Vec<String> = args
                .into_iter()
                .map(|a| a.into_string().unwrap_or_default())
                .collect();
            procs.lock().map(|mut r| r.spawn(cmd, &args)).unwrap_or(-1)
        });
    }
    {
        let procs = procs.clone();
        engine.register_fn("proc_write", move |id: i64, line: &str| -> bool {
            procs.lock().map(|mut r| r.write_line(id, line)).unwrap_or(false)
        });
    }
    {
        let procs = procs.clone();
        engine.register_fn("proc_read", move |id: i64| -> String {
            procs.lock().map(|mut r| r.read_line(id)).unwrap_or_default()
        });
    }
    {
        let procs = procs.clone();
        engine.register_fn("proc_alive", move |id: i64| -> bool {
            procs.lock().map(|r| r.alive(id)).unwrap_or(false)
        });
    }
    {
        let procs = procs.clone();
        engine.register_fn("proc_kill", move |id: i64| {
            if let Ok(mut r) = procs.lock() {
                r.kill(id);
            }
        });
    }

    engine.register_fn("host_env", |name: &str| -> String {
        std::env::var(name).unwrap_or_default()
    });

    engine.register_fn("clipboard_set", |text: &str| -> bool {
        crate::subprocess::clipboard_set(text)
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
// EVENT-DRIVEN. The top-level body runs ONCE per AST load (initialize
// state, declare constants, define handlers). After that, the host
// calls specific handler functions:
//
//   on_init()                                  — once after top-level
//   on_event(kind, payload)                    — Claude bus event
//   on_click(x, y, shift, cmd, id)             — pane click
//   on_resize(w, h)                            — pane resized
//   on_frame(dt)                               — only while animating
//   render(canvas_w, canvas_h) -> Element      — produces the frame
//
// Two host fns control the worker:
//   set_animating(bool)   — opts into per-frame on_frame() calls. Off
//                           by default. Idle widgets cost 0 CPU.
//   request_render()      — call from a handler to schedule render()
//                           after the handler returns.
//
// `state` is a Map in scope, persistent across runs (round-tripped to
// JSON for snapshot). Mutate in place.

// Rhai pure-function note: user-defined `fn` blocks DO NOT see
// top-level `const` declarations. Literals are inlined inside the
// fns that need them.
//
//   PLANTS_SHEET = path below
//   TILE_W = 12, TILE_H = 24, NUM_TILES = 78
//   SPRITE_W = 200.0, SPRITE_H = 400.0
//   GROW_SECS = 8.0, MAX_PLANTS = 120
//   GROUND_INSET = 18.0, GROUND_H = 5.0

if !("plants" in state) { state.plants = []; }
if !("next_id" in state) { state.next_id = 1; }

fn any_growing(plants) {
    for plant in plants { if plant.age < 8.0 { return true; } }
    false
}

fn on_init() {
    // Resume animation if there are still-growing plants from the
    // previous session. If everything is mature we stay idle and
    // burn zero CPU until the next event.
    set_animating(any_growing(state.plants));
    request_render();
}

fn on_event(kind, payload) {
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
        let margin = 100.0;  // SPRITE_W * 0.5
        let x = (center_x + offset).max(margin).min(canvas_w - margin);

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
            tile: (plot_key.abs() % 78).to_int(),  // NUM_TILES
        });
        state.next_id = state.next_id + 1;
        set_animating(true);
        request_render();
    } else if kind == "user_prompt_submit" {
        let cwd = if payload_is_map && "cwd" in payload { payload.cwd } else { "" };
        let plot_key = hash_str(cwd);
        let cx_frac = ((plot_key & 0xFFFF).to_float() / 65535.0) * 0.84 + 0.08;
        let center_x = cx_frac * canvas_w;
        let margin = 100.0;  // SPRITE_W * 0.5
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
                tile: (plot_key.abs() % 78).to_int(),  // NUM_TILES
            });
            state.next_id = state.next_id + 1;
            i = i + 1;
        }
        set_animating(true);
        request_render();
    } else if kind == "stop" {
        for plant in state.plants {
            if plant.age < 8.0 { plant.age = 8.0; }  // GROW_SECS
        }
        set_animating(false);
        request_render();
    }
}

fn on_frame(dt) {
    let still_growing = false;
    for plant in state.plants {
        if plant.age < 8.0 {  // GROW_SECS
            plant.age = plant.age + dt;
            if plant.age >= 8.0 { plant.age = 8.0; }
            else { still_growing = true; }
        }
    }
    while state.plants.len() > 120 {  // MAX_PLANTS
        state.plants.shift();
    }
    if !still_growing { set_animating(false); }
    request_render();
}

fn on_resize(w, h) { request_render(); }

fn render(canvas_w, canvas_h) {
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
        x: 0.0, y: canvas_h - 18.0 - 5.0,  // GROUND_INSET + GROUND_H
        w: canvas_w, h: 5.0,
        color: "#1a2e1a",
        anchor: "top-left",
        z: 0.05,
    });
    for plant in state.plants {
        let t = plant.age / 8.0;  // GROW_SECS
        if t > 1.0 { t = 1.0; }
        let scale = plant.target_scale * (0.25 + 0.75 * t);
        items.push(#{
            type: "sprite",
            id: `plant-${plant.id}`,
            x: plant.x,
            y: canvas_h - 18.0,  // GROUND_INSET
            w: 200.0 * scale,    // SPRITE_W
            h: 400.0 * scale,    // SPRITE_H
            image: "tile",
            path: "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/editor-idea/crates/terminal-bevy/assets/garden/plants.png",
            tile_w: 12,
            tile_h: 24,
            col: plant.tile,
            row: 0,
            hue_shift: plant.hue,
            anchor: "bottom-center",
            z: 1.0,
        });
    }
    #{ type: "canvas", children: items }
}
"###;

// ============================================================
// Default style-picker widget (written if missing)
// ============================================================

const DEFAULT_STYLE_PICKER_SCRIPT: &str = r###"// style_picker.rhai — switch between visual presets.
//
// EVENT-DRIVEN. Renders one button per discovered preset (plus a
// "(project)" button to clear back to the per-project theme).
// Clicking a button calls set_active_style(name) — the host swaps
// the active theme path, theme.rhai reloads, and every pane's
// chrome material updates the same frame.
//
// Host fns:
//   list_styles()              -> array of preset names
//   set_active_style(name)     -> switch to that preset
//                                  (empty string = clear, project)

if !("selected" in state) { state.selected = ""; }

fn on_init() { request_render(); }

fn on_click(x, y, shift, cmd, id) {
    if id == "" { return; }
    if id == "_project" {
        state.selected = "";
        set_active_style("");
    } else if id.starts_with("style_") {
        let name = id.sub_string(6);
        state.selected = name;
        set_active_style(name);
    }
    request_render();
}

fn render(canvas_w, canvas_h) {
    let presets = list_styles();
    let rows = [];
    rows.push(#{ type: "text", value: "Style preset", size: 14.0, color: "#cfd2d8" });

    let mark = if state.selected == "" { ">  " } else { "   " };
    rows.push(#{
        type: "button",
        id: "_project",
        label: mark + "(per-project theme)",
    });

    for name in presets {
        let mark = if state.selected == name { ">  " } else { "   " };
        rows.push(#{
            type: "button",
            id: "style_" + name,
            label: mark + name,
        });
    }

    #{
        type: "vstack",
        pad: 10.0,
        gap: 4.0,
        children: rows,
    }
}
"###;

// ============================================================
// Default theme-editor widget (written if missing)
// ============================================================

const DEFAULT_THEME_EDITOR_SCRIPT: &str = r###"// theme_editor.rhai — live OkLCh theme editor with a real color
// picker.
//
// EVENT-DRIVEN. Click a token to focus it; the focused token reveals
// an LxH (lightness × hue) swatch grid at the token's current chroma,
// plus chroma steppers. Clicking a swatch in the grid commits that
// (L, C, h) directly. Writes go through theme_set_oklch — the bridge
// rewrites the active preset's theme.rhai and the notify watcher
// retones the rest of the app in the same frame.

// NOTE on Rhai scoping: top-level `const` declarations are NOT visible
// inside user-defined `fn` bodies. So every list below is duplicated
// locally inside `render()` / `color_row()` where it's used.

if !("focus" in state) { state.focus = "fg"; }

fn on_init() { request_render(); }

fn on_click(x, y, shift, cmd, id) {
    if id == "" { return; }
    if id.starts_with("focus_") {
        state.focus = id.sub_string(6);
        request_render();
        return;
    }
    if id.starts_with("pick_") {
        // pick_<token>_<L>_<h> — chroma stays at current value
        let rest = id.sub_string(5);
        let parts = rest.split("_");
        let l = parts[parts.len() - 2].parse_float();
        let h = parts[parts.len() - 1].parse_float();
        let token = "";
        let i = 0;
        while i < parts.len() - 2 {
            if i > 0 { token = token + "_"; }
            token = token + parts[i];
            i = i + 1;
        }
        let cur = theme_get_oklch(token);
        let c = if type_of(cur) == "array" { cur[1] } else { 0.1 };
        theme_set_oklch(token, l, c, h);
        request_render();
        return;
    }
    if id.starts_with("chr_") {
        // chr_<token>_<sign> — sign is p (more chroma) / m (less)
        let rest = id.sub_string(4);
        let last_us = rest.index_of("_");
        let token = rest.sub_string(0, last_us);
        let sign = rest.sub_string(last_us + 1);
        let cur = theme_get_oklch(token);
        if type_of(cur) != "array" { return; }
        let l = cur[0]; let c = cur[1]; let h = cur[2];
        let delta = if sign == "p" { 0.02 } else { -0.02 };
        c = c + delta;
        if c < 0.0 { c = 0.0; }
        if c > 0.4 { c = 0.4; }
        theme_set_oklch(token, l, c, h);
        request_render();
        return;
    }
    if id.starts_with("num_") {
        let rest = id.sub_string(4);
        let last_us = rest.index_of("_");
        let token = rest.sub_string(0, last_us);
        let sign = rest.sub_string(last_us + 1);
        let cur = theme_get(token);
        if type_of(cur) != "f64" && type_of(cur) != "i64" { return; }
        let v = if type_of(cur) == "i64" { cur.to_float() } else { cur };
        let step = if sign == "p" { 1.0 }
            else if sign == "P" { 5.0 }
            else if sign == "m" { -1.0 }
            else if sign == "M" { -5.0 }
            else { 0.0 };
        v = v + step;
        if v < 0.0 { v = 0.0; }
        theme_set_number(token, v);
        request_render();
        return;
    }
    if id == "reset_all" {
        theme_reset_all();
        request_render();
        return;
    }
    if id.starts_with("reset_") {
        let token = id.sub_string(6);
        theme_reset(token);
        request_render();
        return;
    }
}

fn on_event(kind, payload) {
    if kind == "theme_changed" { request_render(); }
}

fn fmt_oklch(arr) {
    "L " + arr[0].to_int() + " C " + (arr[1] * 100.0).to_int() + " h " + arr[2].to_int()
}

fn fmt_num(v) {
    if type_of(v) == "i64" { v.to_string() }
    else { ((v * 10.0).to_int().to_float() / 10.0).to_string() }
}

fn color_row(token, is_focused) {
    let cur = theme_get(token);
    let lch = theme_get_oklch(token);
    let lch_text = if type_of(lch) == "array" { fmt_oklch(lch) } else { "" };
    let prefix = if is_focused { "> " } else { "  " };
    let swatch_color = if type_of(cur) == "string" { cur } else { "#000000" };
    let header = #{ type: "hstack", gap: 6.0, align: "center", children: [
        #{ type: "swatch", color: swatch_color, size: 16.0 },
        #{ type: "button", id: "focus_" + token, label: prefix + token + "    " + lch_text },
    ]};
    if !is_focused { return [header]; }
    let rows = [header];
    let chroma = if type_of(lch) == "array" { lch[1] } else { 0.1 };
    rows.push(#{ type: "text", value: "  pick L (rows) x h (cols) at C=" + (chroma * 100.0).to_int(), size: 10.0, color: "#888c98" });
    // Lightest at the top — first-row clicks default to bright,
    // not "near-black", which has wrecked themes by setting fg to
    // L=12 and making the whole app unreadable.
    let ls = [95, 85, 70, 55, 40, 25, 12];
    let hs = [0, 25, 50, 80, 110, 140, 170, 210, 240, 270, 300, 330];
    for l_pct in ls {
        let cells = [];
        for h in hs {
            let col = oklch(l_pct.to_float(), chroma, h.to_float());
            cells.push(#{
                type: "swatch-button",
                id: "pick_" + token + "_" + l_pct + "_" + h,
                color: col,
                size: 18.0,
            });
        }
        rows.push(#{ type: "hstack", gap: 2.0, children: cells });
    }
    rows.push(#{ type: "hstack", gap: 4.0, align: "center", children: [
        #{ type: "text", value: "  chroma:", size: 11.0, color: "#bdc1c9" },
        #{ type: "button", id: "chr_" + token + "_m", label: "-" },
        #{ type: "button", id: "chr_" + token + "_p", label: "+" },
        #{ type: "button", id: "reset_" + token, label: "reset" },
    ]});
    rows
}

fn num_row(token) {
    let cur = theme_get(token);
    let v = if type_of(cur) == "i64" { cur.to_float() }
        else if type_of(cur) == "f64" { cur }
        else { 0.0 };
    #{ type: "hstack", gap: 4.0, align: "center", children: [
        #{ type: "text", value: token + "  " + fmt_num(v), size: 11.0, color: "#bdc1c9" },
        #{ type: "button", id: "num_" + token + "_M", label: "-5" },
        #{ type: "button", id: "num_" + token + "_m", label: "-1" },
        #{ type: "button", id: "num_" + token + "_p", label: "+1" },
        #{ type: "button", id: "num_" + token + "_P", label: "+5" },
        #{ type: "button", id: "reset_" + token, label: "reset" },
    ]}
}

fn contrast_row(pair) {
    let fg_v = theme_get(pair.fg);
    let bg_v = theme_get(pair.bg);
    let score = if type_of(fg_v) == "string" && type_of(bg_v) == "string" {
        theme_contrast(fg_v, bg_v)
    } else { 0.0 };
    let pass = if score >= 25.0 { "✓" } else { "!" };
    let txt_color = if score >= 25.0 { "#7bd58a" } else { "#ff7a7a" };
    let bg_color = if type_of(bg_v) == "string" { bg_v } else { "#000000" };
    let fg_color = if type_of(fg_v) == "string" { fg_v } else { "#ffffff" };
    #{ type: "hstack", gap: 6.0, align: "center", children: [
        #{ type: "swatch", color: bg_color, size: 14.0 },
        #{ type: "swatch", color: fg_color, size: 14.0 },
        #{ type: "text", value: pair.label + "   ΔL " + score.to_int() + "  " + pass, size: 11.0, color: txt_color },
    ]}
}

fn render(canvas_w, canvas_h) {
    let color_tokens = [
        "bg", "fg", "fg_muted", "accent", "caret",
        "pane_bg", "pane_border", "pane_border_focused", "pane_focus_glow",
        "chrome_title", "chrome_title_focused",
        "syntax_keyword", "syntax_string", "syntax_comment", "syntax_function",
        "syntax_type", "syntax_constant",
        "button_bg", "button_label", "button_primary_bg",
        "input_bg", "input_text",
        "radial_wedge", "radial_wedge_hover",
        "canvas_bg", "sidebar_bg",
    ];
    let num_tokens = [
        "pane_corner_radius",
        "pane_border_width",
        "pane_border_width_focused",
        "pane_focus_width",
        "pane_focus_strength",
        "pane_shadow_blur",
        "pane_shadow_offset_y",
        "widget_button_corner_radius",
        "widget_button_border_width",
        "widget_button_shadow_blur",
        "widget_button_shadow_offset_y",
    ];
    let contrast_pairs = [
        #{ label: "body",    fg: "fg",                    bg: "bg" },
        #{ label: "comment", fg: "syntax_comment",        bg: "bg" },
        #{ label: "string",  fg: "syntax_string",         bg: "bg" },
        #{ label: "keyword", fg: "syntax_keyword",        bg: "bg" },
        #{ label: "input",   fg: "input_text",            bg: "input_bg" },
        #{ label: "button",  fg: "button_label",          bg: "button_bg" },
        #{ label: "title",   fg: "chrome_title_focused",  bg: "pane_bg" },
    ];

    let rows = [];
    rows.push(#{ type: "text", value: "Theme editor", size: 14.0, color: "#dde1ea" });
    rows.push(#{ type: "text", value: "Click a token to pick its color from the L x h grid.", size: 10.0, color: "#888c98" });
    rows.push(#{ type: "hstack", gap: 6.0, align: "center", children: [
        #{ type: "button", id: "reset_all", label: "RESET ALL OVERRIDES" },
        #{ type: "text", value: "(emergency escape — clears every override on the active theme)", size: 10.0, color: "#888c98" },
    ]});
    rows.push(#{ type: "divider" });

    rows.push(#{ type: "text", value: "Contrast (OkLab dL; 25 = pass)", size: 11.0, color: "#bdc1c9" });
    for pair in contrast_pairs {
        rows.push(contrast_row(pair));
    }
    rows.push(#{ type: "divider" });

    rows.push(#{ type: "text", value: "Colors", size: 11.0, color: "#bdc1c9" });
    for token in color_tokens {
        let is_focused = state.focus == token;
        for r in color_row(token, is_focused) {
            rows.push(r);
        }
    }

    rows.push(#{ type: "divider" });
    rows.push(#{ type: "text", value: "Numbers", size: 11.0, color: "#bdc1c9" });
    for token in num_tokens {
        rows.push(num_row(token));
    }

    #{
        type: "scroll",
        pad: 10.0,
        gap: 3.0,
        children: rows,
    }
}
"###;

// ============================================================
// Default chess widget (written if missing). The source lives in
// crates/widget-bevy/widgets/chess.rhai and is pulled in at build
// time so the script is real-editable in the repo, while still
// shipping in the binary.
// ============================================================

const DEFAULT_CHESS_SCRIPT: &str =
    include_str!("../widgets/chess.rhai");
