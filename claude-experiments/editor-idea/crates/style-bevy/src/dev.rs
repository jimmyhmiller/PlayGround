//! Developer overrides for shader inputs.
//!
//! Lets a Rhai dev panel (or any host code) say "pretend dust_seconds
//! is 36000 right now" so you can see what your shader looks like at
//! 10 hours of dust without waiting 10 hours. Each override is an
//! `Option<f32>`: `None` = let the normal provider value through;
//! `Some(v)` = force this value.
//!
//! ## Cross-thread bridge
//!
//! The Rhai widget worker runs on its own thread with no ECS access.
//! Worker calls (`dev_dust(7200.0)` from a script) push `DevMsg`
//! variants into an mpsc channel; a Bevy system on the main thread
//! drains the channel into the [`DevOverrides`] resource each frame.
//!
//! Workers obtain a `Sender<DevMsg>` via the [`dev_sender()`] global,
//! which the [`StylePlugin`] populates on startup. The global is a
//! `OnceLock` — fine for this single-app use case; would be replaced
//! with proper resource passing if style-bevy ever shipped as part of
//! a multi-instance setup.

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Mutex, OnceLock};

use bevy::prelude::*;

use crate::shader::ActiveProject;

/// Per-project overrides. Set values replace the corresponding provider
/// field at tick-tail for **that specific project**. Switching to a
/// different project naturally shows that project's own overrides
/// (or its live values, if none are set).
#[derive(Default, Clone, Debug)]
pub struct ProjectOverride {
    pub dust_seconds: Option<f32>,
    pub last_edit_seconds: Option<f32>,
    pub age_seconds: Option<f32>,
}

#[derive(Resource, Default, Clone, Debug)]
pub struct DevOverrides {
    /// Overrides keyed by project id. Looked up by the active project
    /// in `tick_project`.
    pub per_project: HashMap<u64, ProjectOverride>,
    /// `world.time` multiplier. World-scoped, not per-project — a
    /// time-scale change affects every shader on the canvas.
    pub time_scale: Option<f32>,
    /// One-shot signal: when set to `true`, the next wipe tick fills
    /// the active project's wipe mask to fully cleared, then clears
    /// the flag. Triggered from scripts via `dev_windex()` — the
    /// "spray bottle" effect that wipes the whole canvas at once.
    pub pending_windex: bool,
}

impl DevOverrides {
    pub fn project(&self, pid: u64) -> ProjectOverride {
        self.per_project.get(&pid).cloned().unwrap_or_default()
    }
}

/// Messages sent from the Rhai worker thread into the main thread's
/// override receiver. One variant per scrubbable field.
#[derive(Clone, Debug)]
pub enum DevMsg {
    SetDust(Option<f32>),
    SetEdit(Option<f32>),
    SetAge(Option<f32>),
    SetTimeScale(Option<f32>),
    ClearAll,
    /// "Windex" — fill the active project's wipe mask completely.
    /// Equivalent to spraying the whole canvas with cleaner.
    Windex,
}

#[derive(Resource)]
struct DevReceiver(Mutex<Receiver<DevMsg>>);

static DEV_SENDER: OnceLock<Sender<DevMsg>> = OnceLock::new();

/// Returns a cloned sender for the dev override channel, if
/// [`StylePlugin`] has been added to the app. Used by widget-bevy
/// (or any other host code) to register Rhai/script-side fns that
/// push overrides without touching ECS directly.
pub fn dev_sender() -> Option<Sender<DevMsg>> {
    DEV_SENDER.get().cloned()
}

pub struct DevPlugin;

impl Plugin for DevPlugin {
    fn build(&self, app: &mut App) {
        let (tx, rx) = mpsc::channel::<DevMsg>();
        // First plugin to set wins; subsequent App rebuilds (e.g. in
        // tests) silently reuse the existing sender.
        let _ = DEV_SENDER.set(tx);
        app.init_resource::<DevOverrides>()
            .insert_resource(DevReceiver(Mutex::new(rx)))
            .add_systems(Update, drain_dev_msgs);
    }
}

/// Register the script-callable dev fns on a Rhai engine. Called by
/// widget-bevy for every Rhai worker engine. The fns just push
/// [`DevMsg`]s through [`dev_sender`]; the actual state mutation
/// happens on the main thread in [`drain_dev_msgs`].
///
/// Fns exposed to scripts:
/// - `dev_dust(seconds)` — force dust_seconds; `dev_dust(-1)` clears.
/// - `dev_edit(seconds)` — force last_edit_seconds; -1 clears.
/// - `dev_age(seconds)` — force age_seconds; -1 clears.
/// - `dev_time_scale(scale)` — multiplier for world.time; -1 clears.
/// - `dev_clear()` — clear all overrides.
pub fn register_dev_rhai_fns(engine: &mut rhai::Engine) {
    let Some(tx) = dev_sender() else { return };

    let tx_dust = tx.clone();
    engine.register_fn("dev_dust", move |s: f64| {
        let v = if s < 0.0 { None } else { Some(s as f32) };
        let _ = tx_dust.send(DevMsg::SetDust(v));
    });
    let tx_edit = tx.clone();
    engine.register_fn("dev_edit", move |s: f64| {
        let v = if s < 0.0 { None } else { Some(s as f32) };
        let _ = tx_edit.send(DevMsg::SetEdit(v));
    });
    let tx_age = tx.clone();
    engine.register_fn("dev_age", move |s: f64| {
        let v = if s < 0.0 { None } else { Some(s as f32) };
        let _ = tx_age.send(DevMsg::SetAge(v));
    });
    let tx_ts = tx.clone();
    engine.register_fn("dev_time_scale", move |s: f64| {
        let v = if s < 0.0 { None } else { Some(s as f32) };
        let _ = tx_ts.send(DevMsg::SetTimeScale(v));
    });
    let tx_clear = tx.clone();
    engine.register_fn("dev_clear", move || {
        let _ = tx_clear.send(DevMsg::ClearAll);
    });

    let tx_windex = tx.clone();
    engine.register_fn("dev_windex", move || {
        let _ = tx_windex.send(DevMsg::Windex);
    });

    // Integer overloads — Rhai prefers i64 for literals like `7200`
    // and won't auto-coerce to f64 for an `f64` host fn.
    let tx_dust_i = tx.clone();
    engine.register_fn("dev_dust", move |s: i64| {
        let v = if s < 0 { None } else { Some(s as f32) };
        let _ = tx_dust_i.send(DevMsg::SetDust(v));
    });
    let tx_edit_i = tx.clone();
    engine.register_fn("dev_edit", move |s: i64| {
        let v = if s < 0 { None } else { Some(s as f32) };
        let _ = tx_edit_i.send(DevMsg::SetEdit(v));
    });
    let tx_age_i = tx.clone();
    engine.register_fn("dev_age", move |s: i64| {
        let v = if s < 0 { None } else { Some(s as f32) };
        let _ = tx_age_i.send(DevMsg::SetAge(v));
    });
    let tx_ts_i = tx;
    engine.register_fn("dev_time_scale", move |s: i64| {
        let v = if s < 0 { None } else { Some(s as f32) };
        let _ = tx_ts_i.send(DevMsg::SetTimeScale(v));
    });
}

/// Drain whatever DevMsgs arrived since the last frame and apply them
/// to the active project's [`ProjectOverride`] (or, for `SetTimeScale`,
/// to the world-scoped slot).
///
/// Dev-panel buttons target "whoever is currently active" — there's no
/// project-aware UI on the script side, just `dev_dust(s)` etc. We
/// route to the active project here so switching projects in the
/// sidebar gives each its own scrubbed state.
fn drain_dev_msgs(
    rx: Res<DevReceiver>,
    mut overrides: ResMut<DevOverrides>,
    active: Res<ActiveProject>,
) {
    let Ok(rx) = rx.0.lock() else { return };
    while let Ok(msg) = rx.try_recv() {
        match msg {
            DevMsg::SetTimeScale(v) => overrides.time_scale = v,
            DevMsg::ClearAll => *overrides = DevOverrides::default(),
            DevMsg::Windex => overrides.pending_windex = true,
            other => {
                let Some(pid) = active.0 else { continue };
                let entry = overrides.per_project.entry(pid).or_default();
                match other {
                    DevMsg::SetDust(v) => entry.dust_seconds = v,
                    DevMsg::SetEdit(v) => entry.last_edit_seconds = v,
                    DevMsg::SetAge(v) => entry.age_seconds = v,
                    DevMsg::SetTimeScale(_) | DevMsg::ClearAll | DevMsg::Windex => unreachable!(),
                }
            }
        }
    }
}
