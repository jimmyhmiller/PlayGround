//! Per-project temporal state — the data behind Tier-2 shader uniforms.
//!
//! The "dust gathering" effect needs to know how long it's been since
//! the user touched the project; that's what this file records. Each
//! known project gets a small JSON blob persisted under
//! `<base>/<project_id>/state.json`.
//!
//! Timestamps are stored as **wall-clock seconds since the Unix epoch
//! (f64)** so they're meaningful across restarts. Shaders see derived
//! quantities like `dust_seconds = now - last_focus_at`, computed each
//! frame, not the absolute timestamps.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Where per-project asset directories live. Host inserts this at
/// startup. E.g. `~/.jim/projects/`.
#[derive(Resource, Clone, Debug)]
pub struct StyleDataDir(pub PathBuf);

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProjectStyleStateEntry {
    /// Seconds since Unix epoch when this project was first seen.
    #[serde(default)]
    pub created_at: f64,
    /// Seconds since Unix epoch when the user last focused any pane in
    /// this project.
    #[serde(default)]
    pub last_focus_at: f64,
    /// Seconds since Unix epoch of the most recent edit. Updated by
    /// host signals (text-editor edits, file watcher, …) — style-bevy
    /// itself only consumes this value.
    #[serde(default)]
    pub last_edit_at: f64,
    /// Named style preset chosen for this project, e.g. `"forest"`.
    /// `None` means "no preset — use the project's own theme.rhai (or the
    /// built-in default)." This is what makes theming per-project: the
    /// active preset is loaded from here on every project switch.
    #[serde(default)]
    pub preset: Option<String>,
}

/// In-memory cache of every known project's state. The Tier-2 providers
/// read from here; the host calls [`Self::note_focus`] /
/// [`Self::note_edit`] to advance the timestamps.
#[derive(Resource, Default, Debug, Clone)]
pub struct ProjectStyleState {
    by_project: HashMap<u64, ProjectStyleStateEntry>,
    /// Projects whose state has been mutated since the last save, used
    /// to throttle disk writes. A simple sentinel — we save the entire
    /// dirty project on the next save tick, not an incremental delta.
    dirty: HashMap<u64, ()>,
}

impl ProjectStyleState {
    pub fn entry(&self, project_id: u64) -> ProjectStyleStateEntry {
        self.by_project
            .get(&project_id)
            .cloned()
            .unwrap_or_default()
    }

    /// The named preset chosen for this project, if any.
    pub fn preset_of(&self, project_id: u64) -> Option<String> {
        self.by_project
            .get(&project_id)
            .and_then(|e| e.preset.clone())
    }

    /// Set (or clear) the project's chosen preset. Marks it dirty so the
    /// next `save_dirty_tick` persists it to the project's state.json.
    /// No-op if unchanged.
    pub fn set_preset(&mut self, project_id: u64, preset: Option<String>) {
        let e = self.ensure(project_id);
        if e.preset != preset {
            e.preset = preset;
            self.dirty.insert(project_id, ());
        }
    }

    /// Mark "user is engaging with this project right now."
    pub fn note_focus(&mut self, project_id: u64) {
        let now = unix_now();
        let e = self.ensure(project_id);
        e.last_focus_at = now;
        if e.created_at == 0.0 {
            e.created_at = now;
        }
        self.dirty.insert(project_id, ());
    }

    /// Mark "the user edited content tied to this project just now."
    /// `host` signal: editor save, file watcher fire, terminal cd, …
    pub fn note_edit(&mut self, project_id: u64) {
        let now = unix_now();
        let e = self.ensure(project_id);
        e.last_edit_at = now;
        if e.created_at == 0.0 {
            e.created_at = now;
        }
        self.dirty.insert(project_id, ());
    }

    /// Make sure an entry exists (used when a project is first seen so
    /// `created_at` is populated even before any user gesture).
    pub fn touch(&mut self, project_id: u64) {
        let now = unix_now();
        let e = self.ensure(project_id);
        if e.created_at == 0.0 {
            e.created_at = now;
            self.dirty.insert(project_id, ());
        }
    }

    fn ensure(&mut self, project_id: u64) -> &mut ProjectStyleStateEntry {
        self.by_project.entry(project_id).or_default()
    }
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Path to a project's `state.json`.
pub fn state_path(data_dir: &StyleDataDir, project_id: u64) -> PathBuf {
    data_dir.0.join(project_id.to_string()).join("state.json")
}

/// Plugin: loads on first observation of a project, saves dirty entries
/// at most once per ~2 seconds, and on app exit.
pub struct ProjectStatePlugin;

impl Plugin for ProjectStatePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, save_dirty_tick);
    }
}

/// Save dirty entries every couple of seconds. Keeps disk I/O light
/// even if focus is bouncing around — dust changes are smooth at the
/// hour scale, no need for sub-second persistence.
fn save_dirty_tick(
    mut state: ResMut<ProjectStyleState>,
    data_dir: Option<Res<StyleDataDir>>,
    time: Res<Time>,
    mut accum: Local<f32>,
) {
    *accum += time.delta_secs();
    if *accum < 2.0 {
        return;
    }
    *accum = 0.0;
    let Some(data_dir) = data_dir else { return };
    let dirty: Vec<u64> = state.dirty.keys().copied().collect();
    state.dirty.clear();
    for pid in dirty {
        let entry = state.entry(pid);
        let path = state_path(&data_dir, pid);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string_pretty(&entry) {
            Ok(s) => {
                if let Err(e) = std::fs::write(&path, s) {
                    eprintln!("[style] save state {:?}: {}", path, e);
                }
            }
            Err(e) => eprintln!("[style] serialize state {}: {}", pid, e),
        }
    }
}

/// Ensure a project's state is loaded from disk into [`ProjectStyleState`].
/// Host calls this when it first becomes aware of a project (e.g. when
/// projects.json is read on startup, or when a new project is created).
pub fn load_project_state(
    data_dir: &StyleDataDir,
    state: &mut ProjectStyleState,
    project_id: u64,
) {
    if state.by_project.contains_key(&project_id) {
        return;
    }
    let path = state_path(data_dir, project_id);
    let entry = match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str::<ProjectStyleStateEntry>(&s).unwrap_or_default(),
        Err(_) => ProjectStyleStateEntry::default(),
    };
    state.by_project.insert(project_id, entry);
    state.touch(project_id);
}
