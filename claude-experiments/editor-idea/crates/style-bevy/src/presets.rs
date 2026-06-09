//! Style **presets** — named bundles of theme tokens (and, in future,
//! shader overrides) that live under `~/.jim/styles/<name>/`.
//!
//! Each preset is a directory:
//!
//!     ~/.jim/styles/
//!         default/
//!             theme.rhai      # required
//!         linear/
//!             theme.rhai
//!
//! Scripts switch presets by calling `set_active_style("linear")`. The
//! choice is persisted to `~/.jim/active_style` and survives
//! restarts. A preset's `theme.rhai` *overrides* the per-project
//! theme.rhai whenever a preset is active — switch the preset to
//! `None` (empty file) to fall back to per-project theming.
//!
//! Future-extension surface (manifest.toml + chrome.wgsl + dust.wgsl
//! per preset) isn't built yet; today a preset is just one theme file.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use bevy::prelude::*;
use pane_bevy::ActiveChromeShader;
use rhai::{Array, Dynamic, Engine};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Mutex;
use std::sync::OnceLock;

use crate::active::ActiveProject;
use crate::material::PRESET_SOURCE;
use crate::state::{ProjectStyleState, StyleDataDir};
use crate::theme::{load_theme, theme_path_for_project, ActiveThemePath, Theme};

/// One discovered preset on disk.
#[derive(Clone, Debug)]
pub struct StylePreset {
    pub name: String,
    pub theme_path: PathBuf,
    /// `Some` if the preset ships its own pane chrome WGSL. Loaded
    /// via the `preset://` asset source so AssetServer file-watching
    /// hot-reloads the shader without restarting.
    pub chrome_shader: Option<String>,
    /// Heuristic — does the chrome.wgsl reference `params.time`? If
    /// not, it's a static custom shader and the app can stay in
    /// reactive mode (no CPU cost when nothing else is happening).
    /// Theme-only presets (no chrome.wgsl) are always considered
    /// non-animated for this purpose.
    pub chrome_animates: bool,
}

#[derive(Resource, Default, Debug, Clone)]
pub struct StylePresetRegistry {
    pub presets: Vec<StylePreset>,
}

/// Preset of the *currently active project*. `None` means "no preset —
/// use that project's theme.rhai (or the default)." This is a derived
/// mirror: it's reloaded from the active project's [`ProjectStyleState`]
/// on every project switch (`sync_active_preset_from_project`) and
/// written back there when the user picks a style (`drain_preset_messages`).
/// The selection persists per-project via `state.json`, not globally.
#[derive(Resource, Default, Debug, Clone)]
pub struct ActiveStylePreset(pub Option<String>);

pub struct PresetsPlugin;

impl Plugin for PresetsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StylePresetRegistry>()
            .init_resource::<ActiveStylePreset>()
            .add_systems(
                Startup,
                (seed_default_presets, discover_presets).chain(),
            )
            // `sync_active_preset_from_project` loads the active project's
            // saved preset on every switch; `drain_preset_messages` writes
            // the user's pick back to that project. Both feed
            // `ActiveStylePreset`, which `sync_active_theme_from_preset`
            // then turns into the `ActiveThemePath` (the SOLE owner of it —
            // terminal-bevy no longer writes that resource). Chained so the
            // ordering within the frame is deterministic.
            .add_systems(
                Update,
                (
                    sync_active_preset_from_project,
                    drain_preset_messages,
                    sync_active_theme_from_preset,
                    sync_active_chrome_shader_from_preset,
                )
                    .chain(),
            );
    }
}

fn styles_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".jim");
    p.push("styles");
    Some(p)
}

const SEED_DEFAULT_THEME: &str = r###"// default style preset — built-in cool dark.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#1a1c20",
    pane_border: "#2e3036",
    pane_border_focused: "#4d6688",
    pane_focus_glow: "#6b9eeb",
    pane_corner_radius: 6.0,
    pane_border_width: 1.0,
    pane_border_width_focused: 1.5,
    pane_focus_width: 8.0,
    pane_focus_strength: 0.35,
    pane_shadow_color: "#00000073",
    pane_shadow_blur: 24.0,
    pane_shadow_offset_y: 6.0,

    // --- core text ---
    bg: "#16181c",
    fg: "#dcdde1",
    fg_muted: "#80848c",
    accent: "#6b9eeb",
    caret: "#6b9eeb",
    selection: "#6b9eeb47",
    warn: "#f5c168",
    err: "#ff7a7a",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#a0a4ac",
    chrome_title_focused: "#f0f1f5",
    chrome_divider: "#2e3036",
    chrome_close: "#80848c",
    chrome_handle: "#383b42",

    // --- syntax highlighting (editor) ---
    syntax_default: "#dcdde1",
    syntax_keyword: "#c78cdc",
    syntax_string: "#a6e09b",
    syntax_comment: "#7a8090",
    syntax_function: "#8cc7ff",
    syntax_type: "#f0d18c",
    syntax_attribute: "#d9b380",
    syntax_constant: "#f29e7b",
    syntax_operator: "#b3bbc6",
    syntax_punctuation: "#b3bbc6",
    syntax_variable: "#dcdde1",
    syntax_property: "#d9d2f0",
    syntax_label: "#f29e7b",
    syntax_escape: "#f2bf66",
    syntax_constructor: "#f0d18c",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#13151a",
    input_text: "#b8bcc4",
    input_text_focused: "#f0f1f5",

    // --- buttons (widget / run-button save) ---
    button_bg: "#2a2e35",
    button_bg_hover: "#373c45",
    button_label: "#e0e2e8",
    button_primary_bg: "#3a7050",
    button_primary_label: "#eff7f0",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 4.0,
    widget_button_border: "#00000000",
    widget_button_border_width: 0.0,
    widget_button_shadow_color: "#00000059",
    widget_button_shadow_blur: 6.0,
    widget_button_shadow_offset_y: 2.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#8cc7ff",
    status_running: "#f5c168",
    status_success: "#7fd98a",
    status_failed: "#ff7a7a",

    // --- radial menu ---
    radial_wedge: "#22252c",
    radial_wedge_hover: "#3a5a8a",
    radial_deadzone: "#15171b",
    radial_deadzone_ring: "#3c4048",
    radial_label: "#d4d7dc",
    radial_label_hover: "#f8f9fb",
    radial_icon: "#f0f1f5",
    radial_backdrop: "#00000052",

    // --- widget protocol extras ---
    widget_bar_track: "#22252c",
    widget_bar_fill: "#6b9eeb",
    widget_badge_bg: "#3a5a8a",
    widget_badge_label: "#eef2fa",
    widget_link: "#8cc7ff",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#0e1014",
    sidebar_bg: "#13161c",
    sidebar_row_active_bg: "#243049",
    sidebar_row_renaming_bg: "#1f2330",
    sidebar_text_faint: "#5a5e68",

}
"###;

const SEED_LINEAR_THEME: &str = r###"// linear style preset — punchier modern dark.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#0d0e12",
    pane_border: "#1f242e",
    pane_border_focused: "#5b8def",
    pane_focus_glow: "#5b8def",
    pane_corner_radius: 10.0,
    pane_border_width: 1.0,
    pane_border_width_focused: 2.0,
    pane_focus_width: 14.0,
    pane_focus_strength: 0.55,
    pane_shadow_color: "#000000a6",
    pane_shadow_blur: 36.0,
    pane_shadow_offset_y: 10.0,

    // --- core text ---
    bg: "#0a0b0e",
    fg: "#e3e5ea",
    fg_muted: "#82869a",
    accent: "#5b8def",
    caret: "#5b8def",
    selection: "#5b8def52",
    warn: "#f6c054",
    err: "#ff7080",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#9aa0b3",
    chrome_title_focused: "#f4f6fb",
    chrome_divider: "#1f242e",
    chrome_close: "#82869a",
    chrome_handle: "#2a2f3a",

    // --- syntax highlighting (editor) ---
    syntax_default: "#e3e5ea",
    syntax_keyword: "#b8a0ff",
    syntax_string: "#a4e09a",
    syntax_comment: "#5d6478",
    syntax_function: "#74b4ff",
    syntax_type: "#f0c97a",
    syntax_attribute: "#d4a87a",
    syntax_constant: "#f08e6a",
    syntax_operator: "#a6adba",
    syntax_punctuation: "#a6adba",
    syntax_variable: "#e3e5ea",
    syntax_property: "#d8caf0",
    syntax_label: "#f08e6a",
    syntax_escape: "#f6c054",
    syntax_constructor: "#f0c97a",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#0a0c10",
    input_text: "#b6bbc8",
    input_text_focused: "#f4f6fb",

    // --- buttons (widget / run-button save) ---
    button_bg: "#1c2230",
    button_bg_hover: "#283042",
    button_label: "#e3e5ea",
    button_primary_bg: "#5b8def",
    button_primary_label: "#ffffff",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 6.0,
    widget_button_border: "#00000000",
    widget_button_border_width: 0.0,
    widget_button_shadow_color: "#00000080",
    widget_button_shadow_blur: 10.0,
    widget_button_shadow_offset_y: 3.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#74b4ff",
    status_running: "#f6c054",
    status_success: "#7fd98a",
    status_failed: "#ff7080",

    // --- radial menu ---
    radial_wedge: "#181c26",
    radial_wedge_hover: "#2c4a82",
    radial_deadzone: "#0a0c10",
    radial_deadzone_ring: "#2a2f3a",
    radial_label: "#d2d6dd",
    radial_label_hover: "#ffffff",
    radial_icon: "#f4f6fb",
    radial_backdrop: "#00000080",

    // --- widget protocol extras ---
    widget_bar_track: "#181c26",
    widget_bar_fill: "#5b8def",
    widget_badge_bg: "#2c4a82",
    widget_badge_label: "#f4f6fb",
    widget_link: "#74b4ff",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#06070a",
    sidebar_bg: "#0a0c12",
    sidebar_row_active_bg: "#1f2a40",
    sidebar_row_renaming_bg: "#15192a",
    sidebar_text_faint: "#5a5e6c",

}
"###;

const SEED_PAPER_THEME: &str = r###"// paper style preset — warm cream + sepia ink.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#f3eddf",
    pane_border: "#c9b78f",
    pane_border_focused: "#7a5a2c",
    pane_focus_glow: "#a37d3a",
    pane_corner_radius: 3.0,
    pane_border_width: 1.0,
    pane_border_width_focused: 2.0,
    pane_focus_width: 0.0,
    pane_focus_strength: 0.0,
    pane_shadow_color: "#3c2a107a",
    pane_shadow_blur: 18.0,
    pane_shadow_offset_y: 8.0,

    // --- core text ---
    bg: "#fef8e8",
    fg: "#2a2620",
    fg_muted: "#8a7f6e",
    accent: "#7a5a2c",
    caret: "#2a2620",
    selection: "#7a5a2c40",
    warn: "#a35a1a",
    err: "#a83a3a",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#5a513f",
    chrome_title_focused: "#1f1b14",
    chrome_divider: "#c9b78f",
    chrome_close: "#8a7f6e",
    chrome_handle: "#b8a888",

    // --- syntax highlighting (editor) ---
    syntax_default: "#2a2620",
    syntax_keyword: "#7a3a5a",
    syntax_string: "#5a6a2c",
    syntax_comment: "#a89878",
    syntax_function: "#2a4a7a",
    syntax_type: "#7a4a1a",
    syntax_attribute: "#7a5a2c",
    syntax_constant: "#8a3a1a",
    syntax_operator: "#5a513f",
    syntax_punctuation: "#5a513f",
    syntax_variable: "#2a2620",
    syntax_property: "#4a3a5a",
    syntax_label: "#8a3a1a",
    syntax_escape: "#a35a1a",
    syntax_constructor: "#7a4a1a",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#faf6ec",
    input_text: "#2a2620",
    input_text_focused: "#1f1b14",

    // --- buttons (widget / run-button save) ---
    button_bg: "#e8dbc1",
    button_bg_hover: "#dac9a8",
    button_label: "#2a2620",
    button_primary_bg: "#7a5a2c",
    button_primary_label: "#faf6ec",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 8.0,
    widget_button_border: "#a08868",
    widget_button_border_width: 1.0,
    widget_button_shadow_color: "#3c2a1066",
    widget_button_shadow_blur: 8.0,
    widget_button_shadow_offset_y: 2.5,

    // --- status (run-button play state, badges) ---
    status_idle: "#2a4a7a",
    status_running: "#a35a1a",
    status_success: "#5a6a2c",
    status_failed: "#a83a3a",

    // --- radial menu ---
    radial_wedge: "#e8dbc1",
    radial_wedge_hover: "#7a5a2c",
    radial_deadzone: "#dac9a8",
    radial_deadzone_ring: "#a8987c",
    radial_label: "#2a2620",
    radial_label_hover: "#faf6ec",
    radial_icon: "#1f1b14",
    radial_backdrop: "#2a262040",

    // --- widget protocol extras ---
    widget_bar_track: "#dac9a8",
    widget_bar_fill: "#7a5a2c",
    widget_badge_bg: "#7a5a2c",
    widget_badge_label: "#faf6ec",
    widget_link: "#2a4a7a",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#e9e0c8",
    sidebar_bg: "#ece4d0",
    sidebar_row_active_bg: "#dcc89e",
    sidebar_row_renaming_bg: "#d8c8a8",
    sidebar_text_faint: "#a89878",

}
"###;

fn seed_default_presets() {
    let Some(dir) = styles_dir() else { return };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        warn!("style presets: couldn't create {}: {}", dir.display(), e);
        return;
    }
    // Per-preset seeding: only writes if the preset's theme.rhai
    // doesn't exist yet. New built-in presets show up automatically
    // for users who already had earlier ones; user edits never get
    // clobbered because we never overwrite an existing theme.rhai.
    //
    // `atelier` shares its bytes with `theme.rs::ATELIER_DEFAULT_THEME`
    // — the engine's default palette and the preset are the same file,
    // so tuning happens in one place.
    write_seed(&dir.join("atelier"), crate::theme::ATELIER_DEFAULT_THEME);
    write_seed(&dir.join("default"), SEED_DEFAULT_THEME);
    write_seed(&dir.join("linear"), SEED_LINEAR_THEME);
    write_seed(&dir.join("paper"), SEED_PAPER_THEME);
    write_seed_full(&dir.join("neon"), SEED_NEON_THEME, Some(seed_neon_chrome_wgsl()));
    write_seed_full(&dir.join("terminal"), SEED_TERMINAL_THEME, Some(seed_terminal_chrome_wgsl()));
    write_seed_full(&dir.join("wireframe"), SEED_WIREFRAME_THEME, Some(seed_wireframe_chrome_wgsl()));
    write_seed_full(&dir.join("glass"), SEED_GLASS_THEME, Some(seed_glass_chrome_wgsl()));
    // Practical presets — distinct *moods* rather than showy effects.
    // Theme-only:
    write_seed(&dir.join("ink"), SEED_INK_THEME);
    write_seed(&dir.join("amber"), SEED_AMBER_THEME);
    write_seed(&dir.join("slate"), SEED_SLATE_THEME);
    write_seed(&dir.join("blueprint"), SEED_BLUEPRINT_THEME);
    write_seed(&dir.join("forest"), SEED_FOREST_THEME);
    write_seed(&dir.join("kindle"), SEED_KINDLE_THEME);
    // Static custom shaders (no params.time → app stays on Reactive):
    write_seed_full(
        &dir.join("sketch"),
        SEED_SKETCH_THEME,
        Some(seed_sketch_chrome_wgsl()),
    );
    write_seed_full(
        &dir.join("mesh"),
        SEED_MESH_THEME,
        Some(seed_mesh_chrome_wgsl()),
    );
}

/// Seed a preset with a `theme.rhai` (always) and optionally a
/// `chrome.wgsl`. Both writes are idempotent: only writes if the file
/// doesn't already exist, so user edits never get clobbered.
fn write_seed(preset_dir: &std::path::Path, theme_body: &str) {
    write_seed_full(preset_dir, theme_body, None);
}

fn write_seed_full(
    preset_dir: &std::path::Path,
    theme_body: &str,
    chrome_wgsl: Option<String>,
) {
    if let Err(e) = std::fs::create_dir_all(preset_dir) {
        warn!("style presets: mkdir {}: {}", preset_dir.display(), e);
        return;
    }
    let theme = preset_dir.join("theme.rhai");
    if !theme.exists() {
        if let Err(e) = std::fs::write(&theme, theme_body) {
            warn!("style presets: write {}: {}", theme.display(), e);
        }
    }
    if let Some(wgsl) = chrome_wgsl {
        let chrome = preset_dir.join("chrome.wgsl");
        if !chrome.exists() {
            if let Err(e) = std::fs::write(&chrome, &wgsl) {
                warn!("style presets: write {}: {}", chrome.display(), e);
            }
        }
    }
}

fn discover_presets(mut registry: ResMut<StylePresetRegistry>) {
    let Some(dir) = styles_dir() else { return };
    let mut found: Vec<StylePreset> = Vec::new();
    let read = match std::fs::read_dir(&dir) {
        Ok(r) => r,
        Err(_) => {
            registry.presets.clear();
            publish_names(&registry.presets);
            return;
        }
    };
    for entry in read.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let theme = path.join("theme.rhai");
        if !theme.exists() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        // Optional chrome.wgsl makes the preset a "full" preset (its
        // own pane chrome rendering); theme-only presets just leave
        // the active chrome shader unchanged.
        let chrome_path = path.join("chrome.wgsl");
        let (chrome_shader, chrome_animates) = if chrome_path.exists() {
            let animates = std::fs::read_to_string(&chrome_path)
                .map(|s| s.contains("params.time"))
                .unwrap_or(false);
            (
                Some(format!("{}://{}/chrome.wgsl", PRESET_SOURCE, name)),
                animates,
            )
        } else {
            (None, false)
        };
        found.push(StylePreset {
            name,
            theme_path: theme,
            chrome_shader,
            chrome_animates,
        });
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    registry.presets = found;
    publish_names(&registry.presets);
}

/// Resolve a project's effective theme path: its preset's `theme.rhai`
/// if it has a preset, else its own `theme.rhai`. `None` if neither is
/// determinable (no data dir).
pub fn theme_path_for(
    project_id: u64,
    style_state: &ProjectStyleState,
    registry: &StylePresetRegistry,
    data_dir: Option<&StyleDataDir>,
) -> Option<std::path::PathBuf> {
    style_state
        .preset_of(project_id)
        .and_then(|name| {
            registry
                .presets
                .iter()
                .find(|p| p.name == name)
                .map(|p| p.theme_path.clone())
        })
        .or_else(|| data_dir.map(|d| theme_path_for_project(d, project_id)))
}

/// Resolve and load a project's effective theme (preset → its theme.rhai
/// → default). Loads from disk, so callers should cache the result (see
/// [`crate::theme::ProjectThemes`]).
pub fn resolve_project_theme(
    project_id: u64,
    style_state: &ProjectStyleState,
    registry: &StylePresetRegistry,
    data_dir: Option<&StyleDataDir>,
) -> Theme {
    theme_path_for(project_id, style_state, registry, data_dir)
        .filter(|p| p.exists())
        .and_then(|p| load_theme(&p).ok())
        .unwrap_or_default()
}

/// Keep [`ActiveStylePreset`] equal to the active project's saved preset.
/// Evaluated every frame rather than only on project-change: the host's
/// switch hook (`load_project_state`) and this system live in different
/// crates with no ordering guarantee, so a one-shot would permanently
/// miss the preset on the frame it loses that race. The `!=` guard keeps
/// change-detection quiet, and `drain_preset_messages` runs right after
/// in the chain so a user's pick (which writes the project's state) is
/// never clobbered.
fn sync_active_preset_from_project(
    project: Res<ActiveProject>,
    state: Res<ProjectStyleState>,
    mut active: ResMut<ActiveStylePreset>,
) {
    let want = project.0.and_then(|pid| state.preset_of(pid));
    if active.0 != want {
        active.0 = want;
    }
}

/// Owns ActiveThemePath. Preset wins over project when set; when
/// preset is cleared, we restore the project's theme path so the
/// fallback is automatic. Both `ActiveStylePreset` and `ActiveProject`
/// are inputs.
fn sync_active_theme_from_preset(
    preset: Res<ActiveStylePreset>,
    project: Res<ActiveProject>,
    registry: Res<StylePresetRegistry>,
    data_dir: Option<Res<StyleDataDir>>,
    mut active_theme: ResMut<ActiveThemePath>,
) {
    if !preset.is_changed() && !project.is_changed() && !registry.is_changed() {
        return;
    }
    if let Some(name) = preset.0.as_deref() {
        if let Some(p) = registry.presets.iter().find(|p| p.name == name) {
            active_theme.0 = Some(p.theme_path.clone());
            return;
        }
        // Preset name set but not found on disk — log and fall through.
        warn!("style presets: active='{}' not found in registry", name);
    }
    active_theme.0 = match project.0 {
        Some(pid) => data_dir.as_ref().map(|d| theme_path_for_project(d, pid)),
        None => None,
    };
}

/// When the active preset changes (or the registry updates), load the
/// preset's `chrome.wgsl` via the `preset://` asset source and stamp
/// the handle into `ActiveChromeShader`. Presets that don't ship a
/// chrome shader leave it alone — switching from a custom-chrome
/// preset to a theme-only preset keeps the previous chrome shader
/// active. Switching to "(per-project theme)" (preset = None) reverts
/// to the embedded default.
fn sync_active_chrome_shader_from_preset(
    preset: Res<ActiveStylePreset>,
    registry: Res<StylePresetRegistry>,
    asset_server: Res<AssetServer>,
    mut active_chrome: ResMut<ActiveChromeShader>,
    mut last_applied: Local<Option<String>>,
) {
    if !preset.is_changed() && !registry.is_changed() {
        return;
    }
    // What would the new shader path be?
    let new_url: Option<String> = preset.0.as_deref().and_then(|name| {
        registry
            .presets
            .iter()
            .find(|p| p.name == name)
            .and_then(|p| p.chrome_shader.clone())
    });

    // Same as last frame? Don't re-load (which would create an
    // identical weak handle but still trigger downstream change
    // detection).
    if new_url == *last_applied {
        return;
    }

    let to_load = new_url
        .clone()
        .unwrap_or_else(|| "embedded://pane_bevy/chrome_material.wgsl".to_string());
    active_chrome.0 = asset_server.load::<Shader>(to_load);
    *last_applied = new_url;
}

// ============================================================
// Script bridge — `list_styles()` and `set_active_style(name)`
// ============================================================

/// Snapshot of preset names, refreshed every time `StylePresetRegistry`
/// changes. Engine-side `list_styles()` reads from here without
/// touching the Bevy world.
static PRESET_NAMES: OnceLock<Arc<RwLock<Vec<String>>>> = OnceLock::new();

fn names_handle() -> &'static Arc<RwLock<Vec<String>>> {
    PRESET_NAMES.get_or_init(|| Arc::new(RwLock::new(Vec::new())))
}

fn publish_names(presets: &[StylePreset]) {
    if let Ok(mut w) = names_handle().write() {
        *w = presets.iter().map(|p| p.name.clone()).collect();
    }
}

/// Channel for `set_active_style` calls. Main-thread system drains it.
struct PresetMsg(pub Option<String>);
static PRESET_TX: OnceLock<Mutex<Sender<PresetMsg>>> = OnceLock::new();
static PRESET_RX: OnceLock<Mutex<Receiver<PresetMsg>>> = OnceLock::new();

fn ensure_channel() {
    PRESET_TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<PresetMsg>();
        let _ = PRESET_RX.set(Mutex::new(rx));
        Mutex::new(tx)
    });
}

/// Register `list_styles()` + `set_active_style(name)` on a Rhai
/// engine. Called by each widget worker on its own engine. Idempotent.
pub fn register_preset_host_fns(engine: &mut Engine) {
    ensure_channel();
    engine.register_fn("list_styles", || -> Array {
        let lock = names_handle().read();
        match lock {
            Ok(names) => names
                .iter()
                .map(|n| Dynamic::from(n.clone()))
                .collect(),
            Err(_) => Array::new(),
        }
    });
    engine.register_fn("set_active_style", move |name: &str| {
        let n = if name.is_empty() { None } else { Some(name.to_string()) };
        if let Some(tx) = PRESET_TX.get() {
            if let Ok(tx) = tx.lock() {
                let _ = tx.send(PresetMsg(n));
            }
        }
    });
    engine.register_fn("active_style", || -> Dynamic {
        // We don't have a global mirror of the active preset (it
        // could be stale across the wakeups). Reading from a snapshot
        // would be one more moving part; for now, return UNIT and let
        // scripts track it via state_set if they want UI highlight.
        Dynamic::UNIT
    });
}

fn drain_preset_messages(
    mut active: ResMut<ActiveStylePreset>,
    project: Res<ActiveProject>,
    mut state: ResMut<ProjectStyleState>,
) {
    let Some(rx) = PRESET_RX.get() else { return };
    let Ok(rx) = rx.lock() else { return };
    let mut applied: Option<Option<String>> = None;
    while let Ok(msg) = rx.try_recv() {
        applied = Some(msg.0);
    }
    if let Some(new_value) = applied {
        if active.0 != new_value {
            active.0 = new_value.clone();
            // Persist per-project: `set_active_style` applies to whichever
            // project is active, and only that project. `save_dirty_tick`
            // flushes it to the project's state.json.
            if let Some(pid) = project.0 {
                state.set_preset(pid, new_value);
            }
        }
    }
}

// ============================================================
// Seed shaders + themes for the "show off the range" presets.
// Each chrome.wgsl uses the same ChromeParams uniform layout that
// pane-bevy's default shader declares (see crates/pane-bevy/src/
// chrome_material.wgsl) so any preset can substitute its own
// fragment shader without coordinating with the host.
// ============================================================

const CHROME_PARAMS_PREAMBLE: &str = r###"#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct ChromeParams {
    size: vec2<f32>,
    corner_radius: f32,
    border_width: f32,
    bg: vec4<f32>,
    border: vec4<f32>,
    focus: vec4<f32>,
    focus_width: f32,
    time: f32,
    // `cover_mode` and `title_h` are set on the title-cover material
    // (an extra pass that paints the title region above content_root
    // so scrolled content doesn't bleed under the title bar). The
    // host's default shader uses them to cut out the content area in
    // cover mode. Preset shaders can ignore them — if your shader
    // doesn't punch out the content area, the cover will still draw
    // opaquely there and hide everything underneath, so prefer the
    // default chrome shader unless you re-implement the same cutout.
    cover_mode: f32,
    title_h: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> params: ChromeParams;

fn rounded_rect_sdf(p: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - half_size + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(5.0, 3.0, 1.0) + h * 6.0;
    let f = abs(((k - floor(k / 6.0) * 6.0) - 3.0)) - 1.0;
    return v * mix(vec3<f32>(1.0), clamp(f, vec3<f32>(0.0), vec3<f32>(1.0)), s);
}
"###;

// --- NEON ---------------------------------------------------------------
//
// Hue-cycling neon border with a bright outer halo and a darker
// inner body that pulls the eye to the glow. The focus state pumps
// the halo up roughly 2x.

const SEED_NEON_THEME: &str = r###"// neon style preset — cyberpunk rim cycle (animated chrome shader).
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#080612",
    pane_border: "#1a0830",
    pane_border_focused: "#ff00aa",
    pane_focus_glow: "#00f0ff",
    pane_corner_radius: 14.0,
    pane_border_width: 2.0,
    pane_border_width_focused: 3.0,
    pane_focus_width: 24.0,
    pane_focus_strength: 1.0,
    pane_shadow_color: "#9000ff66",
    pane_shadow_blur: 40.0,
    pane_shadow_offset_y: 0.0,

    // --- core text ---
    bg: "#050410",
    fg: "#d4d4ff",
    fg_muted: "#7c7ca5",
    accent: "#ff00aa",
    caret: "#00f0ff",
    selection: "#ff00aa3d",
    warn: "#ffd400",
    err: "#ff2a6d",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#9a9ad4",
    chrome_title_focused: "#00f0ff",
    chrome_divider: "#1a0830",
    chrome_close: "#ff00aa",
    chrome_handle: "#3a1a55",

    // --- syntax highlighting (editor) ---
    syntax_default: "#d4d4ff",
    syntax_keyword: "#00f0ff",
    syntax_string: "#ff66d4",
    syntax_comment: "#4a4470",
    syntax_function: "#aaff66",
    syntax_type: "#ffd400",
    syntax_attribute: "#ff8aff",
    syntax_constant: "#ff5c8a",
    syntax_operator: "#b0a0ff",
    syntax_punctuation: "#9a9ad4",
    syntax_variable: "#d4d4ff",
    syntax_property: "#ffaae0",
    syntax_label: "#ff5c8a",
    syntax_escape: "#ffd400",
    syntax_constructor: "#ffd400",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#04030a",
    input_text: "#aaaad4",
    input_text_focused: "#00f0ff",

    // --- buttons (widget / run-button save) ---
    button_bg: "#180a30",
    button_bg_hover: "#2a1050",
    button_label: "#00f0ff",
    button_primary_bg: "#ff00aa",
    button_primary_label: "#080612",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 0.0,
    widget_button_border: "#ff00aa",
    widget_button_border_width: 1.5,
    widget_button_shadow_color: "#ff00aa66",
    widget_button_shadow_blur: 16.0,
    widget_button_shadow_offset_y: 0.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#00f0ff",
    status_running: "#ffd400",
    status_success: "#aaff66",
    status_failed: "#ff2a6d",

    // --- radial menu ---
    radial_wedge: "#180a30",
    radial_wedge_hover: "#ff00aa",
    radial_deadzone: "#04030a",
    radial_deadzone_ring: "#ff00aa",
    radial_label: "#d4d4ff",
    radial_label_hover: "#00f0ff",
    radial_icon: "#00f0ff",
    radial_backdrop: "#000010a6",

    // --- widget protocol extras ---
    widget_bar_track: "#180a30",
    widget_bar_fill: "#ff00aa",
    widget_badge_bg: "#ff00aa",
    widget_badge_label: "#080612",
    widget_link: "#00f0ff",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#040308",
    sidebar_bg: "#0a0418",
    sidebar_row_active_bg: "#3a0a55",
    sidebar_row_renaming_bg: "#1a0830",
    sidebar_text_faint: "#5a4a7a",

}
"###;

// --- TERMINAL -----------------------------------------------------------
//
// Phosphor green on black. CRT scanlines, slight horizontal stripe
// noise, sharp 1px border, no rounding, no shadow body — looks like
// a tube monitor.

const SEED_TERMINAL_THEME: &str = r###"// terminal style preset — VT100 monochrome phosphor green.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#020a02",
    pane_border: "#19c618",
    pane_border_focused: "#9bff7a",
    pane_focus_glow: "#33ff33",
    pane_corner_radius: 0.0,
    pane_border_width: 1.0,
    pane_border_width_focused: 1.0,
    pane_focus_width: 0.0,
    pane_focus_strength: 0.0,
    pane_shadow_color: "#00ff0033",
    pane_shadow_blur: 20.0,
    pane_shadow_offset_y: 0.0,

    // --- core text ---
    bg: "#020a02",
    fg: "#33ff33",
    fg_muted: "#1a8a1a",
    accent: "#9bff7a",
    caret: "#9bff7a",
    selection: "#33ff3340",
    warn: "#e0c050",
    err: "#ff6633",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#19c618",
    chrome_title_focused: "#9bff7a",
    chrome_divider: "#0a3a0a",
    chrome_close: "#19c618",
    chrome_handle: "#0a3a0a",

    // --- syntax highlighting (editor) ---
    syntax_default: "#33ff33",
    syntax_keyword: "#aaff7a",
    syntax_string: "#33ff33",
    syntax_comment: "#1a8a1a",
    syntax_function: "#aaff7a",
    syntax_type: "#80ff80",
    syntax_attribute: "#60c060",
    syntax_constant: "#aaff7a",
    syntax_operator: "#33ff33",
    syntax_punctuation: "#1a8a1a",
    syntax_variable: "#33ff33",
    syntax_property: "#60c060",
    syntax_label: "#aaff7a",
    syntax_escape: "#e0ff80",
    syntax_constructor: "#aaff7a",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#000000",
    input_text: "#33ff33",
    input_text_focused: "#9bff7a",

    // --- buttons (widget / run-button save) ---
    button_bg: "#0a3a0a",
    button_bg_hover: "#155a15",
    button_label: "#9bff7a",
    button_primary_bg: "#19c618",
    button_primary_label: "#020a02",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 0.0,
    widget_button_border: "#33ff33",
    widget_button_border_width: 1.0,
    widget_button_shadow_color: "#00ff0033",
    widget_button_shadow_blur: 4.0,
    widget_button_shadow_offset_y: 0.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#9bff7a",
    status_running: "#e0c050",
    status_success: "#33ff33",
    status_failed: "#ff6633",

    // --- radial menu ---
    radial_wedge: "#0a3a0a",
    radial_wedge_hover: "#155a15",
    radial_deadzone: "#020a02",
    radial_deadzone_ring: "#19c618",
    radial_label: "#9bff7a",
    radial_label_hover: "#33ff33",
    radial_icon: "#33ff33",
    radial_backdrop: "#000a0099",

    // --- widget protocol extras ---
    widget_bar_track: "#0a3a0a",
    widget_bar_fill: "#33ff33",
    widget_badge_bg: "#0a3a0a",
    widget_badge_label: "#9bff7a",
    widget_link: "#aaff7a",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#010401",
    sidebar_bg: "#020a02",
    sidebar_row_active_bg: "#0a3a0a",
    sidebar_row_renaming_bg: "#051805",
    sidebar_text_faint: "#19c618",

}
"###;

// --- WIREFRAME ----------------------------------------------------------
//
// Schematic drawing: nearly-transparent body so the canvas shows
// through, animated dashed border marching around the perimeter,
// a faint grid inside. Focus turns the dashes solid and bright.

const SEED_WIREFRAME_THEME: &str = r###"// wireframe style preset — schematic blueprint with marching dashes.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#0a0e1a",
    pane_border: "#5588ff",
    pane_border_focused: "#aaccff",
    pane_focus_glow: "#aaccff",
    pane_corner_radius: 2.0,
    pane_border_width: 1.5,
    pane_border_width_focused: 2.0,
    pane_focus_width: 0.0,
    pane_focus_strength: 0.0,
    pane_shadow_color: "#00000000",
    pane_shadow_blur: 1.0,
    pane_shadow_offset_y: 0.0,

    // --- core text ---
    bg: "#060a14",
    fg: "#aaccff",
    fg_muted: "#5577aa",
    accent: "#aaccff",
    caret: "#aaccff",
    selection: "#5588ff4d",
    warn: "#ffcc55",
    err: "#ff7799",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#88aae0",
    chrome_title_focused: "#eaf2ff",
    chrome_divider: "#22335a",
    chrome_close: "#5588ff",
    chrome_handle: "#22335a",

    // --- syntax highlighting (editor) ---
    syntax_default: "#aaccff",
    syntax_keyword: "#ffffff",
    syntax_string: "#9ad4ff",
    syntax_comment: "#3a4a6a",
    syntax_function: "#ffffff",
    syntax_type: "#7aaaff",
    syntax_attribute: "#5588ff",
    syntax_constant: "#aaccff",
    syntax_operator: "#88aae0",
    syntax_punctuation: "#5577aa",
    syntax_variable: "#aaccff",
    syntax_property: "#7aaaff",
    syntax_label: "#ffffff",
    syntax_escape: "#ffcc55",
    syntax_constructor: "#7aaaff",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#050810",
    input_text: "#aaccff",
    input_text_focused: "#ffffff",

    // --- buttons (widget / run-button save) ---
    button_bg: "#0e1422",
    button_bg_hover: "#1a2438",
    button_label: "#aaccff",
    button_primary_bg: "#5588ff",
    button_primary_label: "#0a0e1a",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 1.0,
    widget_button_border: "#5588ff",
    widget_button_border_width: 1.0,
    widget_button_shadow_color: "#00000000",
    widget_button_shadow_blur: 0.0,
    widget_button_shadow_offset_y: 0.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#aaccff",
    status_running: "#ffcc55",
    status_success: "#5588ff",
    status_failed: "#ff7799",

    // --- radial menu ---
    radial_wedge: "#0e1422",
    radial_wedge_hover: "#22335a",
    radial_deadzone: "#050810",
    radial_deadzone_ring: "#5588ff",
    radial_label: "#aaccff",
    radial_label_hover: "#ffffff",
    radial_icon: "#aaccff",
    radial_backdrop: "#0008108c",

    // --- widget protocol extras ---
    widget_bar_track: "#0e1422",
    widget_bar_fill: "#5588ff",
    widget_badge_bg: "#22335a",
    widget_badge_label: "#aaccff",
    widget_link: "#7aaaff",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#04060c",
    sidebar_bg: "#06091a",
    sidebar_row_active_bg: "#1a2240",
    sidebar_row_renaming_bg: "#0e1422",
    sidebar_text_faint: "#3a4d7a",

}
"###;

// --- GLASS --------------------------------------------------------------
//
// Frosted glass: semi-transparent noisy body with a bright top
// highlight band and a soft inner gradient. No real backdrop blur
// (would require a render-graph pass capturing canvas underneath),
// just procedural noise that reads as "frosted." Focus brightens
// the top highlight.

const SEED_GLASS_THEME: &str = r###"// glass style preset — frosted translucency, top highlight.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#cfdce840",
    pane_border: "#ffffff35",
    pane_border_focused: "#ffffff8c",
    pane_focus_glow: "#ffffff",
    pane_corner_radius: 14.0,
    pane_border_width: 1.5,
    pane_border_width_focused: 2.5,
    pane_focus_width: 18.0,
    pane_focus_strength: 0.20,
    pane_shadow_color: "#0a1a2870",
    pane_shadow_blur: 32.0,
    pane_shadow_offset_y: 8.0,

    // --- core text ---
    bg: "#0a1422",
    fg: "#1a2030",
    fg_muted: "#5a6878",
    accent: "#2060c0",
    caret: "#102040",
    selection: "#2060c04d",
    warn: "#a06010",
    err: "#a02040",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#3a4a60",
    chrome_title_focused: "#080c18",
    chrome_divider: "#7080a040",
    chrome_close: "#3a4a60",
    chrome_handle: "#5a6878",

    // --- syntax highlighting (editor) ---
    syntax_default: "#1a2030",
    syntax_keyword: "#2060c0",
    syntax_string: "#106060",
    syntax_comment: "#88a0b8",
    syntax_function: "#1a40a0",
    syntax_type: "#603090",
    syntax_attribute: "#205080",
    syntax_constant: "#a04060",
    syntax_operator: "#3a4a60",
    syntax_punctuation: "#5a6878",
    syntax_variable: "#1a2030",
    syntax_property: "#404a78",
    syntax_label: "#a04060",
    syntax_escape: "#a06010",
    syntax_constructor: "#603090",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#dceaf833",
    input_text: "#1a2030",
    input_text_focused: "#080c18",

    // --- buttons (widget / run-button save) ---
    button_bg: "#a0b8d04d",
    button_bg_hover: "#b8c8e066",
    button_label: "#1a2030",
    button_primary_bg: "#2060c0",
    button_primary_label: "#ffffff",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 12.0,
    widget_button_border: "#ffffff80",
    widget_button_border_width: 1.0,
    widget_button_shadow_color: "#0a1a2880",
    widget_button_shadow_blur: 12.0,
    widget_button_shadow_offset_y: 4.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#2060c0",
    status_running: "#c08020",
    status_success: "#208060",
    status_failed: "#a02040",

    // --- radial menu ---
    radial_wedge: "#a0b8d04d",
    radial_wedge_hover: "#2060c0",
    radial_deadzone: "#80a0c04d",
    radial_deadzone_ring: "#ffffff80",
    radial_label: "#1a2030",
    radial_label_hover: "#ffffff",
    radial_icon: "#080c18",
    radial_backdrop: "#1a304040",

    // --- widget protocol extras ---
    widget_bar_track: "#80a0c04d",
    widget_bar_fill: "#2060c0",
    widget_badge_bg: "#2060c0",
    widget_badge_label: "#ffffff",
    widget_link: "#1a40a0",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#1a2632",
    sidebar_bg: "#142030",
    sidebar_row_active_bg: "#264064",
    sidebar_row_renaming_bg: "#1a2840",
    sidebar_text_faint: "#7090b0",

}
"###;

// Each *_CHROME_WGSL prepends `CHROME_PARAMS_PREAMBLE` so we don't
// repeat the param/SDF/helpers block in every file. Concatenated by
// `write_seed_full` at write time below.

const SEED_NEON_CHROME_FRAGMENT: &str = r###"
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = (in.uv - vec2<f32>(0.5)) * params.size;
    let half_size = params.size * 0.5;
    let r = min(params.corner_radius, min(half_size.x, half_size.y));
    let d = rounded_rect_sdf(p, half_size, r);

    let coverage = 1.0 - smoothstep(-0.5, 0.5, d);
    if (coverage <= 0.0) {
        return vec4<f32>(0.0);
    }

    // Hue cycles through the spectrum slowly + scrolls around the rim
    // so the border feels alive.
    let rim_pos = atan2(p.y, p.x) / 6.2832 + 0.5;
    let hue = fract(params.time * 0.07 + rim_pos * 0.5);
    let rim_color = hsv_to_rgb(hue, 0.95, 1.0);

    // Body is dark with a slight vignette toward center.
    let center_pull = length(p) / max(length(half_size), 1.0);
    var color = params.bg.rgb * (1.0 - 0.35 * center_pull);

    // Animated border: thicker pulse on focus.
    let pulse = 0.85 + 0.15 * sin(params.time * 3.0);
    let bw = params.border_width * (1.0 + params.focus.a * 0.5) * pulse;
    let band = 1.0 - smoothstep(-bw, -bw + 1.5, d);
    color = mix(color, rim_color, band);

    // Inner halo: bright bloom feathered inward from the border.
    let focus_a = max(params.focus.a, 0.5);  // always SOME inner glow
    let inset = clamp((-d - bw) / max(params.focus_width, 0.001), 0.0, 1.0);
    let glow = (1.0 - inset);
    color = color + rim_color * glow * glow * 0.85 * focus_a;

    return vec4<f32>(color, coverage);
}
"###;

const SEED_TERMINAL_CHROME_FRAGMENT: &str = r###"
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = (in.uv - vec2<f32>(0.5)) * params.size;
    let half_size = params.size * 0.5;
    let r = min(params.corner_radius, min(half_size.x, half_size.y));
    let d = rounded_rect_sdf(p, half_size, r);

    let coverage = 1.0 - smoothstep(-0.5, 0.5, d);
    if (coverage <= 0.0) {
        return vec4<f32>(0.0);
    }

    var color = params.bg.rgb;

    // Scanlines: alternating 4px bright/4px dark bands. Earlier I had
    // 2px sinusoidal which aliased into a constant green tint at
    // typical display resolution — invisible. 4px hard bands at this
    // amplitude are unmissable on the near-black body.
    let scan = step(0.0, sin(in.uv.y * params.size.y * 3.1415 / 4.0));
    color = color + vec3<f32>(0.0, 0.18, 0.0) * scan;

    // Slow vertical roll: one bright band drifts down every ~20s.
    let roll_y = fract(in.uv.y - params.time * 0.05);
    let roll_band = smoothstep(0.985, 1.0, roll_y);
    color = color + vec3<f32>(0.0, 0.22, 0.0) * roll_band;

    // 1px sharp border at the rect edge.
    let band = 1.0 - smoothstep(-params.border_width, -params.border_width + 1.0, d);
    color = mix(color, params.border.rgb, band);

    return vec4<f32>(color, coverage);
}
"###;

const SEED_WIREFRAME_CHROME_FRAGMENT: &str = r###"
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = (in.uv - vec2<f32>(0.5)) * params.size;
    let half_size = params.size * 0.5;
    let r = min(params.corner_radius, min(half_size.x, half_size.y));
    let d = rounded_rect_sdf(p, half_size, r);

    let coverage = 1.0 - smoothstep(-0.5, 0.5, d);
    if (coverage <= 0.0) {
        return vec4<f32>(0.0);
    }

    // Mostly-transparent body: just enough fill to read against the
    // canvas, leaving the dust shader visible underneath.
    var color = params.bg.rgb;
    var alpha = params.bg.a * coverage;

    // Faint internal 32 px grid.
    let grid = step(0.92, max(
        abs(sin(in.uv.x * params.size.x * 3.1415 / 32.0)),
        abs(sin(in.uv.y * params.size.y * 3.1415 / 32.0))
    ));
    alpha = max(alpha, 0.06 * grid);
    color = mix(color, params.border.rgb * 0.6, grid * 0.4);

    // Solid border whose hue cycles slowly through the spectrum —
    // gives the "schematic alive" feel without needing perimeter
    // arc-length math (which is what made the earlier marching-ants
    // approach paint diagonal triangles at the corners).
    let bw = params.border_width;
    let on_border = 1.0 - smoothstep(-bw - 0.5, -bw + 0.5, d);
    let hue = fract(params.time * 0.10);
    let rim = hsv_to_rgb(hue, 0.55, 1.0);
    color = mix(color, rim, on_border);
    alpha = max(alpha, on_border * params.border.a);

    return vec4<f32>(color, alpha);
}
"###;

const SEED_GLASS_CHROME_FRAGMENT: &str = r###"
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = (in.uv - vec2<f32>(0.5)) * params.size;
    let half_size = params.size * 0.5;
    let r = min(params.corner_radius, min(half_size.x, half_size.y));
    let d = rounded_rect_sdf(p, half_size, r);

    let coverage = 1.0 - smoothstep(-0.5, 0.5, d);
    if (coverage <= 0.0) {
        return vec4<f32>(0.0);
    }

    // Cool diffuse body with frost noise. The noise breaks up flat
    // color so it reads as a textured surface, not paint.
    let noise = hash21(floor(in.uv * params.size * 0.5));
    let vertical = mix(1.18, 0.92, in.uv.y);
    var color = params.bg.rgb * vertical + vec3<f32>(noise * 0.03);
    var alpha = params.bg.a + 0.10 * (1.0 - in.uv.y);  // slightly more opaque near top

    // Bright highlight band along the top edge (~10% of height) —
    // reads as a curved glass meniscus.
    let top_t = clamp(1.0 - in.uv.y * 8.0, 0.0, 1.0);
    color = color + vec3<f32>(top_t * 0.18);
    alpha = alpha + top_t * 0.10;

    // Crisp 1.5px border using bg color border field.
    let bw = params.border_width;
    let on_border = 1.0 - smoothstep(-bw - 0.5, -bw + 0.5, d);
    color = mix(color, params.border.rgb, on_border * params.border.a);
    alpha = max(alpha, on_border * params.border.a);

    // Focus state: brighten the top highlight + add a subtle inner
    // glow ring just past the border.
    let inset = clamp((-d - bw) / max(params.focus_width, 0.001), 0.0, 1.0);
    let glow_t = (1.0 - inset);
    let focus_add = params.focus.a * glow_t * glow_t;
    color = color + params.focus.rgb * focus_add;

    return vec4<f32>(color, clamp(alpha, 0.0, 1.0) * coverage);
}
"###;

// Helpers that concatenate the shared preamble with each fragment
// body — keeps the on-disk WGSL files self-contained even though we
// don't want to duplicate ChromeParams in source.
fn join_chrome(fragment_src: &str) -> String {
    let mut s = String::with_capacity(CHROME_PARAMS_PREAMBLE.len() + fragment_src.len());
    s.push_str(CHROME_PARAMS_PREAMBLE);
    s.push_str(fragment_src);
    s
}

// Lazy-eval so we only build the joined strings when seeding fires
// (once on startup, after disk check).
fn seed_neon_chrome_wgsl() -> String { join_chrome(SEED_NEON_CHROME_FRAGMENT) }
fn seed_terminal_chrome_wgsl() -> String { join_chrome(SEED_TERMINAL_CHROME_FRAGMENT) }
fn seed_wireframe_chrome_wgsl() -> String { join_chrome(SEED_WIREFRAME_CHROME_FRAGMENT) }
fn seed_glass_chrome_wgsl() -> String { join_chrome(SEED_GLASS_CHROME_FRAGMENT) }
fn seed_sketch_chrome_wgsl() -> String { join_chrome(SEED_SKETCH_CHROME_FRAGMENT) }
fn seed_mesh_chrome_wgsl() -> String { join_chrome(SEED_MESH_CHROME_FRAGMENT) }

// ============================================================
// Practical presets — distinct *moods* of work, not showy effects.
// Theme-only seeds reuse the default chrome shader; the look comes
// entirely from token values.
// ============================================================

const SEED_INK_THEME: &str = r###"// ink style preset — high-contrast paper for typography.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#fafaf5",
    pane_border: "#2a2a28",
    pane_border_focused: "#000000",
    pane_focus_glow: "#2a2a28",
    pane_corner_radius: 1.0,
    pane_border_width: 0.5,
    pane_border_width_focused: 1.0,
    pane_focus_width: 0.0,
    pane_focus_strength: 0.0,
    pane_shadow_color: "#00000000",
    pane_shadow_blur: 1.0,
    pane_shadow_offset_y: 0.0,

    // --- core text ---
    bg: "#fafaf5",
    fg: "#1a1a18",
    fg_muted: "#6e6e68",
    accent: "#303030",
    caret: "#1a1a18",
    selection: "#00000020",
    warn: "#665020",
    err: "#802020",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#4a4a48",
    chrome_title_focused: "#000000",
    chrome_divider: "#c8c8c0",
    chrome_close: "#6e6e68",
    chrome_handle: "#b8b8b0",

    // --- syntax highlighting (editor) ---
    syntax_default: "#1a1a18",
    syntax_keyword: "#1a1a18",
    syntax_string: "#3a4a3a",
    syntax_comment: "#9a9a92",
    syntax_function: "#1a1a18",
    syntax_type: "#2a3a4a",
    syntax_attribute: "#5a4a2a",
    syntax_constant: "#4a2a2a",
    syntax_operator: "#4a4a48",
    syntax_punctuation: "#6e6e68",
    syntax_variable: "#1a1a18",
    syntax_property: "#3a3a4a",
    syntax_label: "#4a2a2a",
    syntax_escape: "#5a4a2a",
    syntax_constructor: "#2a3a4a",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#ffffff",
    input_text: "#1a1a18",
    input_text_focused: "#000000",

    // --- buttons (widget / run-button save) ---
    button_bg: "#f0f0e8",
    button_bg_hover: "#e0e0d8",
    button_label: "#1a1a18",
    button_primary_bg: "#1a1a18",
    button_primary_label: "#fafaf5",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 0.0,
    widget_button_border: "#1a1a18",
    widget_button_border_width: 1.0,
    widget_button_shadow_color: "#00000000",
    widget_button_shadow_blur: 0.0,
    widget_button_shadow_offset_y: 0.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#4a4a48",
    status_running: "#665020",
    status_success: "#3a4a3a",
    status_failed: "#802020",

    // --- radial menu ---
    radial_wedge: "#f0f0e8",
    radial_wedge_hover: "#1a1a18",
    radial_deadzone: "#fafaf5",
    radial_deadzone_ring: "#9a9a92",
    radial_label: "#1a1a18",
    radial_label_hover: "#fafaf5",
    radial_icon: "#000000",
    radial_backdrop: "#1a1a1840",

    // --- widget protocol extras ---
    widget_bar_track: "#e0e0d8",
    widget_bar_fill: "#1a1a18",
    widget_badge_bg: "#1a1a18",
    widget_badge_label: "#fafaf5",
    widget_link: "#2a3a4a",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#f0f0e6",
    sidebar_bg: "#f3f3e9",
    sidebar_row_active_bg: "#e0e0d4",
    sidebar_row_renaming_bg: "#e8e8dc",
    sidebar_text_faint: "#a6a6a0",

}
"###;

const SEED_AMBER_THEME: &str = r###"// amber style preset — warm dark for evening sessions.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#1a1310",
    pane_border: "#5e3b1a",
    pane_border_focused: "#c87a2c",
    pane_focus_glow: "#ffa64a",
    pane_corner_radius: 6.0,
    pane_border_width: 1.0,
    pane_border_width_focused: 1.5,
    pane_focus_width: 10.0,
    pane_focus_strength: 0.30,
    pane_shadow_color: "#1a05007a",
    pane_shadow_blur: 24.0,
    pane_shadow_offset_y: 6.0,

    // --- core text ---
    bg: "#100a08",
    fg: "#f5e8d0",
    fg_muted: "#8a7060",
    accent: "#ffa64a",
    caret: "#ffa64a",
    selection: "#c87a2c4d",
    warn: "#ffc060",
    err: "#ff7050",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#b0967a",
    chrome_title_focused: "#fff0d0",
    chrome_divider: "#3a2a18",
    chrome_close: "#a07a55",
    chrome_handle: "#42301c",

    // --- syntax highlighting (editor) ---
    syntax_default: "#f5e8d0",
    syntax_keyword: "#ffa64a",
    syntax_string: "#f0d090",
    syntax_comment: "#705a48",
    syntax_function: "#ffce80",
    syntax_type: "#e0b070",
    syntax_attribute: "#d09060",
    syntax_constant: "#ff8060",
    syntax_operator: "#c0a888",
    syntax_punctuation: "#8a7060",
    syntax_variable: "#f5e8d0",
    syntax_property: "#e0a878",
    syntax_label: "#ff8060",
    syntax_escape: "#ffc060",
    syntax_constructor: "#e0b070",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#0e0907",
    input_text: "#d0bca0",
    input_text_focused: "#ffe0b0",

    // --- buttons (widget / run-button save) ---
    button_bg: "#2e1f15",
    button_bg_hover: "#403028",
    button_label: "#f5e8d0",
    button_primary_bg: "#c87a2c",
    button_primary_label: "#1a1310",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 6.0,
    widget_button_border: "#5e3b1a",
    widget_button_border_width: 1.0,
    widget_button_shadow_color: "#1a050099",
    widget_button_shadow_blur: 8.0,
    widget_button_shadow_offset_y: 3.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#ffa64a",
    status_running: "#ffc060",
    status_success: "#cab070",
    status_failed: "#ff7050",

    // --- radial menu ---
    radial_wedge: "#2e1f15",
    radial_wedge_hover: "#c87a2c",
    radial_deadzone: "#180e08",
    radial_deadzone_ring: "#5e3b1a",
    radial_label: "#d0bca0",
    radial_label_hover: "#fff0d0",
    radial_icon: "#ffe0b0",
    radial_backdrop: "#1a050099",

    // --- widget protocol extras ---
    widget_bar_track: "#2e1f15",
    widget_bar_fill: "#ffa64a",
    widget_badge_bg: "#c87a2c",
    widget_badge_label: "#fff0d0",
    widget_link: "#ffce80",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#0a0704",
    sidebar_bg: "#14100a",
    sidebar_row_active_bg: "#3c2810",
    sidebar_row_renaming_bg: "#1f1810",
    sidebar_text_faint: "#705a48",

}
"###;

const SEED_SLATE_THEME: &str = r###"// slate style preset — flat brutalist gray, no curves.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#232628",
    pane_border: "#585d62",
    pane_border_focused: "#aabac5",
    pane_focus_glow: "#aabac5",
    pane_corner_radius: 0.0,
    pane_border_width: 2.0,
    pane_border_width_focused: 2.0,
    pane_focus_width: 0.0,
    pane_focus_strength: 0.0,
    pane_shadow_color: "#00000000",
    pane_shadow_blur: 1.0,
    pane_shadow_offset_y: 0.0,

    // --- core text ---
    bg: "#1a1d20",
    fg: "#d4d8dc",
    fg_muted: "#787c80",
    accent: "#aabac5",
    caret: "#d4d8dc",
    selection: "#aabac540",
    warn: "#d4b070",
    err: "#d47080",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#a0a4a8",
    chrome_title_focused: "#ffffff",
    chrome_divider: "#3a3e42",
    chrome_close: "#787c80",
    chrome_handle: "#3a3e42",

    // --- syntax highlighting (editor) ---
    syntax_default: "#d4d8dc",
    syntax_keyword: "#ffffff",
    syntax_string: "#c0c0b0",
    syntax_comment: "#5a5e62",
    syntax_function: "#e0e8f0",
    syntax_type: "#c0b8a8",
    syntax_attribute: "#a0a4a8",
    syntax_constant: "#d4b070",
    syntax_operator: "#a0a4a8",
    syntax_punctuation: "#787c80",
    syntax_variable: "#d4d8dc",
    syntax_property: "#b0c0d0",
    syntax_label: "#d4b070",
    syntax_escape: "#e0c890",
    syntax_constructor: "#c0b8a8",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#1a1d20",
    input_text: "#d4d8dc",
    input_text_focused: "#ffffff",

    // --- buttons (widget / run-button save) ---
    button_bg: "#2e3236",
    button_bg_hover: "#3e4348",
    button_label: "#d4d8dc",
    button_primary_bg: "#aabac5",
    button_primary_label: "#232628",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 0.0,
    widget_button_border: "#585d62",
    widget_button_border_width: 2.0,
    widget_button_shadow_color: "#00000000",
    widget_button_shadow_blur: 0.0,
    widget_button_shadow_offset_y: 0.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#aabac5",
    status_running: "#d4b070",
    status_success: "#90b090",
    status_failed: "#d47080",

    // --- radial menu ---
    radial_wedge: "#2e3236",
    radial_wedge_hover: "#585d62",
    radial_deadzone: "#1a1d20",
    radial_deadzone_ring: "#585d62",
    radial_label: "#d4d8dc",
    radial_label_hover: "#ffffff",
    radial_icon: "#ffffff",
    radial_backdrop: "#1a1d2099",

    // --- widget protocol extras ---
    widget_bar_track: "#2e3236",
    widget_bar_fill: "#aabac5",
    widget_badge_bg: "#585d62",
    widget_badge_label: "#d4d8dc",
    widget_link: "#c0d0e0",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#15171a",
    sidebar_bg: "#1a1d20",
    sidebar_row_active_bg: "#2e3236",
    sidebar_row_renaming_bg: "#22262a",
    sidebar_text_faint: "#5a5e62",

}
"###;

const SEED_BLUEPRINT_THEME: &str = r###"// blueprint style preset — deep navy with cyan hairlines.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#0d2030",
    pane_border: "#4ec3f0",
    pane_border_focused: "#a5e3ff",
    pane_focus_glow: "#4ec3f0",
    pane_corner_radius: 0.0,
    pane_border_width: 1.0,
    pane_border_width_focused: 1.5,
    pane_focus_width: 0.0,
    pane_focus_strength: 0.0,
    pane_shadow_color: "#00000000",
    pane_shadow_blur: 1.0,
    pane_shadow_offset_y: 0.0,

    // --- core text ---
    bg: "#0a1828",
    fg: "#e0f0ff",
    fg_muted: "#6090b0",
    accent: "#4ec3f0",
    caret: "#a5e3ff",
    selection: "#4ec3f04d",
    warn: "#ffd070",
    err: "#ff8090",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#a0c8e0",
    chrome_title_focused: "#eaf6ff",
    chrome_divider: "#1a3a52",
    chrome_close: "#4ec3f0",
    chrome_handle: "#1a3a52",

    // --- syntax highlighting (editor) ---
    syntax_default: "#e0f0ff",
    syntax_keyword: "#4ec3f0",
    syntax_string: "#a5e3ff",
    syntax_comment: "#4a6a82",
    syntax_function: "#ffffff",
    syntax_type: "#80d8ff",
    syntax_attribute: "#80aae0",
    syntax_constant: "#ffd070",
    syntax_operator: "#a0c8e0",
    syntax_punctuation: "#6090b0",
    syntax_variable: "#e0f0ff",
    syntax_property: "#a0c8ff",
    syntax_label: "#ffd070",
    syntax_escape: "#ffe890",
    syntax_constructor: "#80d8ff",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#061525",
    input_text: "#e0f0ff",
    input_text_focused: "#ffffff",

    // --- buttons (widget / run-button save) ---
    button_bg: "#102e44",
    button_bg_hover: "#1a4258",
    button_label: "#e0f0ff",
    button_primary_bg: "#4ec3f0",
    button_primary_label: "#0d2030",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 0.0,
    widget_button_border: "#4ec3f0",
    widget_button_border_width: 1.0,
    widget_button_shadow_color: "#00000000",
    widget_button_shadow_blur: 0.0,
    widget_button_shadow_offset_y: 0.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#4ec3f0",
    status_running: "#ffd070",
    status_success: "#80d8ff",
    status_failed: "#ff8090",

    // --- radial menu ---
    radial_wedge: "#102e44",
    radial_wedge_hover: "#4ec3f0",
    radial_deadzone: "#061525",
    radial_deadzone_ring: "#4ec3f0",
    radial_label: "#e0f0ff",
    radial_label_hover: "#ffffff",
    radial_icon: "#a5e3ff",
    radial_backdrop: "#061525a6",

    // --- widget protocol extras ---
    widget_bar_track: "#102e44",
    widget_bar_fill: "#4ec3f0",
    widget_badge_bg: "#4ec3f0",
    widget_badge_label: "#0d2030",
    widget_link: "#a5e3ff",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#06121e",
    sidebar_bg: "#0a1828",
    sidebar_row_active_bg: "#1a3a5a",
    sidebar_row_renaming_bg: "#11253a",
    sidebar_text_faint: "#3a6080",

}
"###;

const SEED_FOREST_THEME: &str = r###"// forest style preset — moss-green nature tones.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#1a2218",
    pane_border: "#4a5e3e",
    pane_border_focused: "#8caa6c",
    pane_focus_glow: "#b8d090",
    pane_corner_radius: 6.0,
    pane_border_width: 1.0,
    pane_border_width_focused: 1.5,
    pane_focus_width: 8.0,
    pane_focus_strength: 0.25,
    pane_shadow_color: "#1a100066",
    pane_shadow_blur: 20.0,
    pane_shadow_offset_y: 6.0,

    // --- core text ---
    bg: "#141a14",
    fg: "#d8e4cc",
    fg_muted: "#6e8060",
    accent: "#8caa6c",
    caret: "#b8d090",
    selection: "#8caa6c4d",
    warn: "#d4a050",
    err: "#c47060",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#9aa888",
    chrome_title_focused: "#eaf2dc",
    chrome_divider: "#2e3a26",
    chrome_close: "#8caa6c",
    chrome_handle: "#2e3a26",

    // --- syntax highlighting (editor) ---
    syntax_default: "#d8e4cc",
    syntax_keyword: "#b8d090",
    syntax_string: "#c0d4a0",
    syntax_comment: "#5a6a52",
    syntax_function: "#e0e8ce",
    syntax_type: "#d0c890",
    syntax_attribute: "#a0b88c",
    syntax_constant: "#d4a050",
    syntax_operator: "#a0b090",
    syntax_punctuation: "#6e8060",
    syntax_variable: "#d8e4cc",
    syntax_property: "#c0bcd0",
    syntax_label: "#d4a050",
    syntax_escape: "#e0c050",
    syntax_constructor: "#d0c890",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#101810",
    input_text: "#c8d8b8",
    input_text_focused: "#eaf2dc",

    // --- buttons (widget / run-button save) ---
    button_bg: "#283022",
    button_bg_hover: "#384230",
    button_label: "#d8e4cc",
    button_primary_bg: "#8caa6c",
    button_primary_label: "#1a2218",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 4.0,
    widget_button_border: "#4a5e3e",
    widget_button_border_width: 1.0,
    widget_button_shadow_color: "#1a100066",
    widget_button_shadow_blur: 6.0,
    widget_button_shadow_offset_y: 2.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#8caa6c",
    status_running: "#d4a050",
    status_success: "#b8d090",
    status_failed: "#c47060",

    // --- radial menu ---
    radial_wedge: "#283022",
    radial_wedge_hover: "#8caa6c",
    radial_deadzone: "#141a12",
    radial_deadzone_ring: "#4a5e3e",
    radial_label: "#d8e4cc",
    radial_label_hover: "#eaf2dc",
    radial_icon: "#b8d090",
    radial_backdrop: "#0a100899",

    // --- widget protocol extras ---
    widget_bar_track: "#283022",
    widget_bar_fill: "#8caa6c",
    widget_badge_bg: "#4a5e3e",
    widget_badge_label: "#eaf2dc",
    widget_link: "#b8d090",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#0d130c",
    sidebar_bg: "#141a12",
    sidebar_row_active_bg: "#283022",
    sidebar_row_renaming_bg: "#1c2218",
    sidebar_text_faint: "#4e6240",

}
"###;

const SEED_KINDLE_THEME: &str = r###"// kindle style preset — minimal e-reader paper warm.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#faf6ed",
    pane_border: "#b5ab9a",
    pane_border_focused: "#5a5448",
    pane_focus_glow: "#5a5448",
    pane_corner_radius: 1.0,
    pane_border_width: 0.5,
    pane_border_width_focused: 0.8,
    pane_focus_width: 0.0,
    pane_focus_strength: 0.0,
    pane_shadow_color: "#00000000",
    pane_shadow_blur: 1.0,
    pane_shadow_offset_y: 0.0,

    // --- core text ---
    bg: "#faf6ed",
    fg: "#2e2820",
    fg_muted: "#80766a",
    accent: "#5a5448",
    caret: "#5a5448",
    selection: "#5a544830",
    warn: "#7a6028",
    err: "#8a3a2a",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#6a6258",
    chrome_title_focused: "#1a1610",
    chrome_divider: "#d0c8b8",
    chrome_close: "#80766a",
    chrome_handle: "#b8aea0",

    // --- syntax highlighting (editor) ---
    syntax_default: "#2e2820",
    syntax_keyword: "#1a1610",
    syntax_string: "#5a5448",
    syntax_comment: "#9c9080",
    syntax_function: "#1a1610",
    syntax_type: "#4a4030",
    syntax_attribute: "#6a5c40",
    syntax_constant: "#5a4030",
    syntax_operator: "#5a5448",
    syntax_punctuation: "#80766a",
    syntax_variable: "#2e2820",
    syntax_property: "#4a4448",
    syntax_label: "#5a4030",
    syntax_escape: "#7a6028",
    syntax_constructor: "#4a4030",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#ffffff",
    input_text: "#2e2820",
    input_text_focused: "#1a1610",

    // --- buttons (widget / run-button save) ---
    button_bg: "#f0e8d8",
    button_bg_hover: "#e0d6c0",
    button_label: "#2e2820",
    button_primary_bg: "#2e2820",
    button_primary_label: "#faf6ed",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 0.0,
    widget_button_border: "#a89880",
    widget_button_border_width: 0.5,
    widget_button_shadow_color: "#00000000",
    widget_button_shadow_blur: 0.0,
    widget_button_shadow_offset_y: 0.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#5a5448",
    status_running: "#7a6028",
    status_success: "#4a5040",
    status_failed: "#8a3a2a",

    // --- radial menu ---
    radial_wedge: "#f0e8d8",
    radial_wedge_hover: "#2e2820",
    radial_deadzone: "#faf6ed",
    radial_deadzone_ring: "#a89c88",
    radial_label: "#2e2820",
    radial_label_hover: "#faf6ed",
    radial_icon: "#1a1610",
    radial_backdrop: "#2e282040",

    // --- widget protocol extras ---
    widget_bar_track: "#e0d6c0",
    widget_bar_fill: "#2e2820",
    widget_badge_bg: "#2e2820",
    widget_badge_label: "#faf6ed",
    widget_link: "#4a4030",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#f5efe0",
    sidebar_bg: "#f7f2e4",
    sidebar_row_active_bg: "#e8dec0",
    sidebar_row_renaming_bg: "#ede4c8",
    sidebar_text_faint: "#aaa090",

}
"###;

const SEED_SKETCH_THEME: &str = r###"// sketch style preset — cream notebook page with sepia ink.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#f7f1de",
    pane_border: "#5c5346",
    pane_border_focused: "#1d1812",
    pane_focus_glow: "#1d1812",
    pane_corner_radius: 2.0,
    pane_border_width: 0.75,
    pane_border_width_focused: 1.5,
    pane_focus_width: 0.0,
    pane_focus_strength: 0.0,
    pane_shadow_color: "#2b1a0840",
    pane_shadow_blur: 14.0,
    pane_shadow_offset_y: 4.0,

    // --- core text ---
    bg: "#faf3e0",
    fg: "#2c1f10",
    fg_muted: "#806c50",
    accent: "#5c5346",
    caret: "#2c1f10",
    selection: "#5c534633",
    warn: "#8a5a18",
    err: "#902a2a",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#5a4a30",
    chrome_title_focused: "#1d1812",
    chrome_divider: "#d0c0a0",
    chrome_close: "#806c50",
    chrome_handle: "#b8a888",

    // --- syntax highlighting (editor) ---
    syntax_default: "#2c1f10",
    syntax_keyword: "#702048",
    syntax_string: "#5a6028",
    syntax_comment: "#a89070",
    syntax_function: "#2a4080",
    syntax_type: "#7a3a18",
    syntax_attribute: "#6a5028",
    syntax_constant: "#8a2a18",
    syntax_operator: "#5c5346",
    syntax_punctuation: "#806c50",
    syntax_variable: "#2c1f10",
    syntax_property: "#4a3060",
    syntax_label: "#8a2a18",
    syntax_escape: "#8a5a18",
    syntax_constructor: "#7a3a18",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#fcf6e2",
    input_text: "#2c1f10",
    input_text_focused: "#1d1812",

    // --- buttons (widget / run-button save) ---
    button_bg: "#e8dec0",
    button_bg_hover: "#d8c8a0",
    button_label: "#2c1f10",
    button_primary_bg: "#5c5346",
    button_primary_label: "#f7f1de",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 2.0,
    widget_button_border: "#5c5346",
    widget_button_border_width: 1.0,
    widget_button_shadow_color: "#3c2a1066",
    widget_button_shadow_blur: 6.0,
    widget_button_shadow_offset_y: 2.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#2a4080",
    status_running: "#8a5a18",
    status_success: "#5a6028",
    status_failed: "#902a2a",

    // --- radial menu ---
    radial_wedge: "#e8dec0",
    radial_wedge_hover: "#5c5346",
    radial_deadzone: "#dac9a8",
    radial_deadzone_ring: "#a08868",
    radial_label: "#2c1f10",
    radial_label_hover: "#f7f1de",
    radial_icon: "#1d1812",
    radial_backdrop: "#2c1f1040",

    // --- widget protocol extras ---
    widget_bar_track: "#dac9a8",
    widget_bar_fill: "#5c5346",
    widget_badge_bg: "#5c5346",
    widget_badge_label: "#f7f1de",
    widget_link: "#2a4080",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#efe7d0",
    sidebar_bg: "#f2ead4",
    sidebar_row_active_bg: "#dac9a8",
    sidebar_row_renaming_bg: "#e0d4b8",
    sidebar_text_faint: "#9a8868",

}
"###;

const SEED_MESH_THEME: &str = r###"// mesh style preset — quiet multi-stop gradient body.
#{
    // --- pane chrome material (rounded rect SDF) ---
    pane_bg: "#161a26",
    pane_border: "#2b324a",
    pane_border_focused: "#6f7fa8",
    pane_focus_glow: "#3a1e50",
    pane_corner_radius: 8.0,
    pane_border_width: 1.0,
    pane_border_width_focused: 1.5,
    pane_focus_width: 0.0,
    pane_focus_strength: 0.25,
    pane_shadow_color: "#00000066",
    pane_shadow_blur: 22.0,
    pane_shadow_offset_y: 6.0,

    // --- core text ---
    bg: "#0e1220",
    fg: "#d4d8e8",
    fg_muted: "#70748c",
    accent: "#6f7fa8",
    caret: "#b0c0e0",
    selection: "#6f7fa84d",
    warn: "#d4a060",
    err: "#c47080",

    // --- pane chrome text (title / close / handle / divider) ---
    chrome_title: "#8c92ac",
    chrome_title_focused: "#e8eaf6",
    chrome_divider: "#262c44",
    chrome_close: "#6f7fa8",
    chrome_handle: "#262c44",

    // --- syntax highlighting (editor) ---
    syntax_default: "#d4d8e8",
    syntax_keyword: "#b08aff",
    syntax_string: "#80c0c0",
    syntax_comment: "#4a5070",
    syntax_function: "#90a8e0",
    syntax_type: "#d0b0a0",
    syntax_attribute: "#8a9ac0",
    syntax_constant: "#c08aa0",
    syntax_operator: "#8c92ac",
    syntax_punctuation: "#70748c",
    syntax_variable: "#d4d8e8",
    syntax_property: "#a0a0d0",
    syntax_label: "#c08aa0",
    syntax_escape: "#d4a060",
    syntax_constructor: "#d0b0a0",

    // --- form fields (run-button input rows, generally) ---
    input_bg: "#0c1018",
    input_text: "#b0b8d0",
    input_text_focused: "#e8eaf6",

    // --- buttons (widget / run-button save) ---
    button_bg: "#1e2436",
    button_bg_hover: "#2a324a",
    button_label: "#d4d8e8",
    button_primary_bg: "#6f7fa8",
    button_primary_label: "#161a26",

    // --- widget Button shape + drop shadow (SDF material) ---
    widget_button_corner_radius: 6.0,
    widget_button_border: "#2b324a",
    widget_button_border_width: 1.0,
    widget_button_shadow_color: "#00000066",
    widget_button_shadow_blur: 8.0,
    widget_button_shadow_offset_y: 3.0,

    // --- status (run-button play state, badges) ---
    status_idle: "#90a8e0",
    status_running: "#d4a060",
    status_success: "#80c0c0",
    status_failed: "#c47080",

    // --- radial menu ---
    radial_wedge: "#1e2436",
    radial_wedge_hover: "#6f7fa8",
    radial_deadzone: "#0c1018",
    radial_deadzone_ring: "#2b324a",
    radial_label: "#d4d8e8",
    radial_label_hover: "#e8eaf6",
    radial_icon: "#b0c0e0",
    radial_backdrop: "#0a0e1899",

    // --- widget protocol extras ---
    widget_bar_track: "#1e2436",
    widget_bar_fill: "#6f7fa8",
    widget_badge_bg: "#3a1e50",
    widget_badge_label: "#e8eaf6",
    widget_link: "#90a8e0",

    // --- shell (canvas + sidebar) ---
    canvas_bg: "#0a0e16",
    sidebar_bg: "#0f1320",
    sidebar_row_active_bg: "#262c44",
    sidebar_row_renaming_bg: "#1a1f30",
    sidebar_text_faint: "#4a4e68",

}
"###;

// --- SKETCH CHROME (static) -----------------------------------------
//
// Body = cream + low-frequency noise + subtle horizontal lines that
// suggest notebook ruling. No time references → app stays Reactive.

const SEED_SKETCH_CHROME_FRAGMENT: &str = r###"
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = (in.uv - vec2<f32>(0.5)) * params.size;
    let half_size = params.size * 0.5;
    let r = min(params.corner_radius, min(half_size.x, half_size.y));
    let d = rounded_rect_sdf(p, half_size, r);

    let coverage = 1.0 - smoothstep(-0.5, 0.5, d);
    if (coverage <= 0.0) {
        return vec4<f32>(0.0);
    }

    // Two octaves of value noise; tile coords keep edges crisp.
    let pixel = floor(in.uv * params.size);
    let n1 = hash21(pixel);
    let n2 = hash21(floor(in.uv * params.size * 0.5));
    let grain = (n1 * 0.7 + n2 * 0.3) - 0.5;
    var color = params.bg.rgb + vec3<f32>(grain * 0.025);

    // Faint horizontal pencil rule every ~22 px. Slightly darker
    // band so the body reads like ruled paper without dominating.
    let line_y = sin(in.uv.y * params.size.y * 3.1415 / 22.0);
    let rule = smoothstep(0.985, 1.0, abs(line_y));
    color = color - vec3<f32>(0.018) * rule;

    // 0.75-1.5 px hand-drawn-ish border.
    let bw = params.border_width;
    let band = 1.0 - smoothstep(-bw - 0.5, -bw + 0.5, d);
    color = mix(color, params.border.rgb, band * params.border.a);

    return vec4<f32>(color, coverage);
}
"###;

// --- MESH CHROME (static) -------------------------------------------
//
// Multi-stop gradient body. params.bg sits at the top, params.focus
// (used here as "second gradient stop") sits at the bottom. Subtle
// 1-stop highlight at the very top edge for depth.

const SEED_MESH_CHROME_FRAGMENT: &str = r###"
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = (in.uv - vec2<f32>(0.5)) * params.size;
    let half_size = params.size * 0.5;
    let r = min(params.corner_radius, min(half_size.x, half_size.y));
    let d = rounded_rect_sdf(p, half_size, r);

    let coverage = 1.0 - smoothstep(-0.5, 0.5, d);
    if (coverage <= 0.0) {
        return vec4<f32>(0.0);
    }

    // Top → bottom gradient between bg and focus colors. We use
    // smoothstep on uv.y so the transition reads as a soft mesh
    // rather than a hard linear ramp.
    let t = smoothstep(0.0, 1.0, in.uv.y);
    var color = mix(params.bg.rgb, params.focus.rgb, t);

    // Faint top highlight (~5% of height).
    let top_t = clamp(1.0 - in.uv.y * 18.0, 0.0, 1.0);
    color = color + vec3<f32>(top_t * 0.04);

    // Border.
    let bw = params.border_width;
    let band = 1.0 - smoothstep(-bw - 0.5, -bw + 0.5, d);
    color = mix(color, params.border.rgb, band * params.border.a);

    return vec4<f32>(color, coverage);
}
"###;
