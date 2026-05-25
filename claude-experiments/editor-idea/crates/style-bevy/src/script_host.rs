//! Headless Rhai system scripts — script-as-behavior (no pane, no UI).
//!
//! Where `rhai_widget` is "a Rhai script + a pane to render into",
//! this module is "a Rhai script + nothing visible." It exists because
//! the dust effect's behavior (animation ticks, mouse painting, reset
//! logic) needs to run continuously regardless of whether any pane is
//! open. The script doesn't render anything — it issues host calls
//! (`uniform_set`, `mask_paint`, `emit`) which are routed through the
//! [`crate::script_bridge`] channel to the renderer.
//!
//! Run model: a single ticker system invokes the AST inline on the
//! main thread each frame. Rhai is fast enough at our script sizes
//! (< 1 ms per tick for the dust script) that worker-thread isolation
//! isn't worth the complexity here.
//!
//! Hot-reload: a `notify` watcher mirrors `theme.rs` — file changes
//! re-parse the AST without restarting the runtime.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use rhai::{Dynamic, Engine, Scope, AST};

use crate::script_bridge::{register_script_host_fns, value_to_rhai_public, EventBus};

/// A running script. Owns its Engine + last-parsed AST + a persistent
/// Scope (the `state` map survives across ticks and reloads).
#[derive(Resource)]
pub struct SystemScript {
    pub path: PathBuf,
    engine: Engine,
    ast: Option<AST>,
    scope: Scope<'static>,
    last_error: Option<String>,
}

impl SystemScript {
    /// Construct + try-load. If the file can't be read or parses with
    /// errors, the script is still installed (so hot-reload can fix
    /// it) but `last_error` is set.
    pub fn new(path: PathBuf) -> Self {
        let mut engine = Engine::new();
        register_script_host_fns(&mut engine);
        // Lightweight host_log for debugging the script.
        engine.register_fn("host_log", |msg: &str| {
            eprintln!("[script] {}", msg);
        });

        let mut scope = Scope::new();
        scope.push("state", rhai::Map::new());
        scope.push("events", rhai::Array::new());
        scope.push("dt", 0.0_f64);
        scope.push("time", 0.0_f64);

        let mut me = Self {
            path,
            engine,
            ast: None,
            scope,
            last_error: None,
        };
        me.reload();
        me
    }

    pub fn reload(&mut self) {
        match std::fs::read_to_string(&self.path) {
            Ok(src) => match self.engine.compile(&src) {
                Ok(ast) => {
                    self.ast = Some(ast);
                    self.last_error = None;
                    eprintln!("[script] loaded {}", self.path.display());
                }
                Err(e) => {
                    self.last_error = Some(format!("parse: {}", e));
                    eprintln!("[script] parse {}: {}", self.path.display(), e);
                }
            },
            Err(e) => {
                self.last_error = Some(format!("read: {}", e));
                eprintln!("[script] read {}: {}", self.path.display(), e);
            }
        }
    }

    pub fn tick(&mut self, time: f32, dt: f32, events: rhai::Array) {
        let Some(ast) = &self.ast else { return };
        let _ = self.scope.set_value("time", time as f64);
        let _ = self.scope.set_value("dt", dt as f64);
        let _ = self.scope.set_value("events", events);
        match self.engine.run_ast_with_scope(&mut self.scope, ast) {
            Ok(()) => {}
            Err(e) => {
                let msg = format!("runtime: {}", e);
                if self.last_error.as_deref() != Some(msg.as_str()) {
                    eprintln!("[script] {}: {}", self.path.display(), msg);
                }
                self.last_error = Some(msg);
            }
        }
        // Round-trip `state` so internal mutations stick.
        if let Some(state) = self.scope.get_value::<Dynamic>("state") {
            let _ = self.scope.set_value("state", state);
        }
    }
}

/// Optional file watcher so the script hot-reloads on save.
#[derive(Resource)]
struct ScriptWatcher {
    rx: Mutex<Receiver<PathBuf>>,
    _watcher: RecommendedWatcher,
    last_reload: Instant,
}

pub struct SystemScriptPlugin {
    pub path: PathBuf,
    /// Embedded fallback source. Written to `path` on first launch
    /// if no file exists there; never overwrites a user-edited copy.
    /// `None` = don't bootstrap (path must already exist).
    pub bootstrap_source: Option<&'static str>,
}

impl Plugin for SystemScriptPlugin {
    fn build(&self, app: &mut App) {
        if let Some(src) = self.bootstrap_source
            && !self.path.exists()
        {
            if let Some(parent) = self.path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = std::fs::write(&self.path, src) {
                eprintln!("[script] bootstrap write {:?}: {}", self.path, e);
            } else {
                eprintln!("[script] wrote bootstrap {:?}", self.path);
            }
        }
        app.insert_resource(SystemScript::new(self.path.clone()));
        if let Some(watcher) = build_watcher(&self.path) {
            app.insert_resource(watcher);
        }
        app.add_systems(Update, (drain_reloads, tick_system_script).chain());
    }
}

fn build_watcher(path: &PathBuf) -> Option<ScriptWatcher> {
    let dir = path.parent()?.to_path_buf();
    let watched = path.clone();
    let (tx, rx) = mpsc::channel::<PathBuf>();
    let mut w = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        let Ok(ev) = res else { return };
        for p in ev.paths {
            if p == watched {
                let _ = tx.send(p);
            }
        }
    })
    .ok()?;
    w.watch(&dir, RecursiveMode::NonRecursive).ok()?;
    Some(ScriptWatcher {
        rx: Mutex::new(rx),
        _watcher: w,
        last_reload: Instant::now() - Duration::from_secs(1),
    })
}

fn drain_reloads(
    watcher: Option<ResMut<ScriptWatcher>>,
    mut script: ResMut<SystemScript>,
) {
    let Some(mut watcher) = watcher else { return };
    let mut got = false;
    if let Ok(rx) = watcher.rx.lock() {
        while rx.try_recv().is_ok() {
            got = true;
        }
    }
    if !got {
        return;
    }
    let now = Instant::now();
    if now.duration_since(watcher.last_reload) < Duration::from_millis(150) {
        return;
    }
    watcher.last_reload = now;
    script.reload();
}

/// Minimum seconds between script ticks. Bevy may render at 60+ FPS
/// (the dust system's own writes mark the material modified, which
/// in reactive mode counts as an input → another render). Without a
/// floor here, every render → another script tick → another set of
/// uniform writes → another render, in a tight feedback loop that
/// pegged CPU at ~300%. 30 Hz is plenty for animation.
const SCRIPT_TICK_FLOOR_SECS: f32 = 1.0 / 30.0;

fn tick_system_script(
    time: Res<Time>,
    mut script: ResMut<SystemScript>,
    mut bus: ResMut<EventBus>,
    mut last_tick: Local<f32>,
) {
    let now = time.elapsed_secs();
    if now - *last_tick < SCRIPT_TICK_FLOOR_SECS {
        return;
    }
    *last_tick = now;

    // Drain anything emitted (or fired from `schedule`) since the
    // last tick. Each entry becomes `#{ kind, payload }` — same
    // shape rhai widget scripts expect, so the same script can run
    // either as a system script or in a widget without changes.
    let drained: Vec<(String, serde_json::Value)> = std::mem::take(&mut bus.pending);
    let mut events = rhai::Array::with_capacity(drained.len());
    for (kind, payload) in drained {
        let mut m = rhai::Map::new();
        m.insert("kind".into(), rhai::Dynamic::from(kind));
        m.insert(
            "payload".into(),
            value_to_rhai_public(payload).unwrap_or(rhai::Dynamic::UNIT),
        );
        events.push(rhai::Dynamic::from(m));
    }

    script.tick(now, time.delta_secs(), events);
}
