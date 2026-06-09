//! Persist + restore the OS-window size and position across runs.
//!
//! On startup, [`load`] reads `~/.jim/window.json` (if any)
//! and returns the saved geometry so `main.rs` can seed the initial
//! `Window` resolution + position.
//!
//! At runtime, [`save_on_change`] listens for `WindowResized` and
//! `WindowMoved` events; when one fires it writes the current geometry
//! back to disk. Writes are debounced so a continuous drag doesn't
//! hammer the filesystem.

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::window::{PrimaryWindow, WindowMoved, WindowResized};
use serde::{Deserialize, Serialize};

const FILE_NAME: &str = "window.json";
const WRITE_DEBOUNCE: Duration = Duration::from_millis(400);

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct WindowGeometry {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

fn path() -> Option<std::path::PathBuf> {
    crate::data_dir().map(|d| d.join(FILE_NAME))
}

/// Read the saved geometry, if any. Returns `None` on first run, on
/// IO error, or if the file is malformed (so the caller falls back to
/// hard-coded defaults).
pub fn load() -> Option<WindowGeometry> {
    let p = path()?;
    let body = std::fs::read_to_string(&p).ok()?;
    let g: WindowGeometry = serde_json::from_str(&body).ok()?;
    // Drop obviously-degenerate values (window minimized to 0, etc.)
    // — falling back to defaults is friendlier than restoring a
    // window the user can't see.
    if g.w < 200 || g.h < 150 {
        return None;
    }
    Some(g)
}

fn write(g: &WindowGeometry) {
    let Some(p) = path() else { return };
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(body) = serde_json::to_string(g) {
        let _ = std::fs::write(&p, body);
    }
}

/// Local state for the debounce. `pending` carries the latest geometry
/// the user has nudged toward; `last_write_at` gates how often we
/// actually flush to disk.
#[derive(Default)]
pub struct SaveState {
    pending: Option<WindowGeometry>,
    last_write_at: Option<Instant>,
}

pub fn save_on_change(
    mut resized: MessageReader<WindowResized>,
    mut moved: MessageReader<WindowMoved>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut state: Local<SaveState>,
) {
    let mut dirty = false;
    for _ in resized.read() {
        dirty = true;
    }
    for _ in moved.read() {
        dirty = true;
    }

    if dirty {
        if let Ok(window) = windows.single() {
            let pos = match window.position {
                WindowPosition::At(p) => p,
                _ => IVec2::ZERO,
            };
            // Save PHYSICAL pixels — Bevy 0.18's
            // `WindowResolution::from((u32, u32))` calls
            // `WindowResolution::new(physical_width, physical_height)`,
            // so the restore path treats these as physical.
            state.pending = Some(WindowGeometry {
                x: pos.x,
                y: pos.y,
                w: window.resolution.physical_width(),
                h: window.resolution.physical_height(),
            });
        }
    }

    let Some(g) = state.pending else { return };
    let now = Instant::now();
    let should_write = state
        .last_write_at
        .is_none_or(|t| now.duration_since(t) >= WRITE_DEBOUNCE);
    if !should_write {
        return;
    }
    write(&g);
    state.pending = None;
    state.last_write_at = Some(now);
}
