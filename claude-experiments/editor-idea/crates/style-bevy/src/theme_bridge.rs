//! Script-facing bridge for live theme editing.
//!
//! Theme editor widgets read the current token values via
//! `theme_get(name)` and push edits via `theme_set_color` /
//! `theme_set_number`. Writes go through an mpsc channel; the main
//! thread drains it, rewrites the active preset's `theme.rhai`, and
//! the existing notify watcher hot-reloads the rest of the app.
//!
//! Reads use a shared `Arc<RwLock<Snapshot>>` that mirrors the
//! current `Theme` resource; updated on every `ThemeChanged` so
//! widgets never read stale values.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use bevy::prelude::*;
use rhai::{Array, Dynamic, Engine};

use crate::oklab;
use crate::theme::{parse_color_string, ActiveThemePath, Theme, ThemeChanged, TokenValue};

/// Snapshot of the live theme tokens, keyed by name.
type Snapshot = HashMap<String, TokenValue>;

static SNAPSHOT: OnceLock<Arc<RwLock<Snapshot>>> = OnceLock::new();
fn snapshot() -> &'static Arc<RwLock<Snapshot>> {
    SNAPSHOT.get_or_init(|| Arc::new(RwLock::new(HashMap::new())))
}

/// Messages from script worker threads to the main thread.
pub enum ThemeWrite {
    /// Set a token to a hex (or oklch/oklab/rgb) color string.
    SetColor(String, String),
    /// Set a numeric token.
    SetNumber(String, f32),
    /// Remove a token override (falls back to default on next reload).
    Reset(String),
    /// Wipe every override — rewrites the active theme.rhai to an
    /// empty `#{}`. Escape hatch when the user has accidentally
    /// thrashed a token to an unreadable color.
    ResetAll,
}

static TX: OnceLock<Mutex<Sender<ThemeWrite>>> = OnceLock::new();
static RX: OnceLock<Mutex<Receiver<ThemeWrite>>> = OnceLock::new();

fn ensure_channel() {
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<ThemeWrite>();
        let _ = RX.set(Mutex::new(rx));
        Mutex::new(tx)
    });
}

pub struct ThemeBridgePlugin;

impl Plugin for ThemeBridgePlugin {
    fn build(&self, app: &mut App) {
        ensure_channel();
        app.add_systems(Startup, publish_initial_snapshot)
            .add_systems(
                Update,
                (publish_snapshot_on_change, drain_theme_writes).chain(),
            );
    }
}

fn publish_initial_snapshot(theme: Res<Theme>) {
    publish(&theme);
}

fn publish_snapshot_on_change(
    mut events: MessageReader<ThemeChanged>,
    theme: Res<Theme>,
) {
    if events.read().last().is_none() {
        return;
    }
    publish(&theme);
}

fn publish(theme: &Theme) {
    if let Ok(mut w) = snapshot().write() {
        w.clear();
        for name in theme.token_names() {
            if let Some(v) = theme.get_by_name(&name) {
                w.insert(name, v);
            }
        }
    }
}

fn drain_theme_writes(active_path: Res<ActiveThemePath>) {
    let Some(rx) = RX.get() else { return };
    let Ok(rx) = rx.lock() else { return };
    let mut updates: HashMap<String, Option<String>> = HashMap::new();
    let mut reset_all = false;
    while let Ok(msg) = rx.try_recv() {
        match msg {
            ThemeWrite::SetColor(name, color_str) => {
                if parse_color_string(&color_str).is_ok() {
                    updates.insert(name, Some(format!("\"{}\"", color_str)));
                } else {
                    warn!("[theme-bridge] rejected unparseable color: {}", color_str);
                }
            }
            ThemeWrite::SetNumber(name, value) => {
                updates.insert(name, Some(format!("{}", value)));
            }
            ThemeWrite::Reset(name) => {
                updates.insert(name, None);
            }
            ThemeWrite::ResetAll => {
                reset_all = true;
                updates.clear();
            }
        }
    }
    if !reset_all && updates.is_empty() {
        return;
    }
    let Some(path) = active_path.0.clone() else {
        warn!("[theme-bridge] no active theme path; can't persist writes");
        return;
    };
    if reset_all {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Err(e) =
            std::fs::write(&path, "// Reset by theme editor.\n#{\n}\n")
        {
            warn!("[theme-bridge] reset_all write {:?}: {}", path, e);
        }
        return;
    }
    if let Err(e) = patch_theme_file(&path, updates) {
        warn!("[theme-bridge] couldn't patch {:?}: {}", path, e);
    }
}

/// Minimal line-based patcher: for each `key: value,` line in the
/// theme.rhai, if the key is in `updates`, replace its value (or
/// remove the line entirely if the override is `None`). Tokens not
/// present in the file get appended just before the closing `}`.
///
/// Comments, blank lines, and section labels are preserved.
fn patch_theme_file(
    path: &std::path::Path,
    mut updates: HashMap<String, Option<String>>,
) -> std::io::Result<()> {
    // If the file doesn't exist (e.g. a project hasn't been themed
    // yet), create a minimal `#{}` skeleton so the patcher has
    // somewhere to inject. Parent dir is created too.
    if !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            path,
            "// Auto-created by theme editor.\n#{\n}\n",
        )?;
    }
    let src = std::fs::read_to_string(path)?;
    let mut out = String::with_capacity(src.len() + 64);
    let mut close_brace_idx: Option<usize> = None;

    for (i, line) in src.lines().enumerate() {
        // Detect a `    name: value,` line. Use a permissive split.
        let trimmed = line.trim_start();
        if let Some(colon) = trimmed.find(':') {
            // Token name = everything before colon if it's an identifier.
            let key = trimmed[..colon].trim();
            let key_clean = key.trim_end_matches(',').trim();
            let valid_ident = !key_clean.is_empty()
                && key_clean
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_');
            if valid_ident {
                if let Some(new_value) = updates.remove(key_clean) {
                    match new_value {
                        Some(v) => {
                            // Preserve the original indentation.
                            let indent_len = line.len() - trimmed.len();
                            let indent = &line[..indent_len];
                            out.push_str(&format!("{}{}: {},\n", indent, key_clean, v));
                        }
                        None => {
                            // Drop the line entirely — token will revert
                            // to default on reload.
                        }
                    }
                    continue;
                }
            }
        }
        // Track the closing brace so we know where to insert tokens
        // that weren't present in the file already.
        if trimmed == "}" {
            close_brace_idx = Some(out.len());
        }
        out.push_str(line);
        out.push('\n');
        let _ = i;
    }

    // Any updates left over → new tokens not previously in the file.
    if !updates.is_empty() {
        if let Some(idx) = close_brace_idx {
            let mut insertion = String::new();
            insertion.push_str("    // --- editor additions ---\n");
            for (k, v) in updates {
                if let Some(v) = v {
                    insertion.push_str(&format!("    {}: {},\n", k, v));
                }
            }
            out.insert_str(idx, &insertion);
        }
    }

    std::fs::write(path, out)
}

// ----------------- Rhai host fns -----------------

pub fn register_theme_host_fns(engine: &mut Engine) {
    ensure_channel();

    engine.register_fn("theme_tokens", || -> Array {
        let Ok(snap) = snapshot().read() else { return Array::new() };
        let mut names: Vec<String> = snap.keys().cloned().collect();
        names.sort();
        names.into_iter().map(Dynamic::from).collect()
    });

    engine.register_fn("theme_get", |name: &str| -> Dynamic {
        let Ok(snap) = snapshot().read() else { return Dynamic::UNIT };
        match snap.get(name) {
            Some(TokenValue::Color(c)) => {
                Dynamic::from(linear_rgba_to_hex(*c))
            }
            Some(TokenValue::F32(v)) => Dynamic::from(*v as f64),
            Some(TokenValue::Bool(b)) => Dynamic::from(*b),
            Some(TokenValue::Str(s)) => Dynamic::from(s.clone()),
            None => Dynamic::UNIT,
        }
    });

    // theme_get_oklch returns [L, C, h] for color tokens, or () for
    // non-color tokens. Lets editor widgets present perceptual sliders
    // directly.
    engine.register_fn("theme_get_oklch", |name: &str| -> Dynamic {
        let Ok(snap) = snapshot().read() else { return Dynamic::UNIT };
        let Some(TokenValue::Color(c)) = snap.get(name) else {
            return Dynamic::UNIT;
        };
        let (l, ch, h) = oklab::linear_srgb_to_oklch(*c);
        let mut arr = Array::new();
        arr.push(Dynamic::from(l as f64));
        arr.push(Dynamic::from(ch as f64));
        arr.push(Dynamic::from(h as f64));
        Dynamic::from(arr)
    });

    let tx_color = TX.get().unwrap().lock().unwrap().clone();
    engine.register_fn("theme_set_color", move |name: &str, value: &str| {
        let _ = tx_color.send(ThemeWrite::SetColor(name.into(), value.into()));
    });

    let tx_color_lch = TX.get().unwrap().lock().unwrap().clone();
    engine.register_fn(
        "theme_set_oklch",
        move |name: &str, l: f64, c: f64, h: f64| {
            let s = format!("oklch({}, {}, {})", l, c, h);
            let _ = tx_color_lch.send(ThemeWrite::SetColor(name.into(), s));
        },
    );

    let tx_num = TX.get().unwrap().lock().unwrap().clone();
    engine.register_fn("theme_set_number", move |name: &str, value: f64| {
        let _ = tx_num.send(ThemeWrite::SetNumber(name.into(), value as f32));
    });

    let tx_reset = TX.get().unwrap().lock().unwrap().clone();
    engine.register_fn("theme_reset", move |name: &str| {
        let _ = tx_reset.send(ThemeWrite::Reset(name.into()));
    });

    let tx_reset_all = TX.get().unwrap().lock().unwrap().clone();
    engine.register_fn("theme_reset_all", move || {
        let _ = tx_reset_all.send(ThemeWrite::ResetAll);
    });
}

fn linear_rgba_to_hex(c: bevy::color::LinearRgba) -> String {
    let srgb = Color::LinearRgba(c).to_srgba();
    let r = (srgb.red.clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (srgb.green.clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (srgb.blue.clamp(0.0, 1.0) * 255.0).round() as u8;
    let a = (srgb.alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
    if a == 255 {
        format!("#{:02x}{:02x}{:02x}", r, g, b)
    } else {
        format!("#{:02x}{:02x}{:02x}{:02x}", r, g, b, a)
    }
}

/// Allow other modules to read the active path for diagnostics.
pub fn active_theme_path(active: &ActiveThemePath) -> Option<PathBuf> {
    active.0.clone()
}
