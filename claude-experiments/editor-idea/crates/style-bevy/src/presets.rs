//! Style **presets** — named bundles of theme tokens (and, in future,
//! shader overrides) that live under `~/.terminal-bevy/styles/<name>/`.
//!
//! Each preset is a directory:
//!
//!     ~/.terminal-bevy/styles/
//!         default/
//!             theme.rhai      # required
//!         linear/
//!             theme.rhai
//!
//! Scripts switch presets by calling `set_active_style("linear")`. The
//! choice is persisted to `~/.terminal-bevy/active_style` and survives
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
use crate::state::StyleDataDir;
use crate::theme::{theme_path_for_project, ActiveThemePath};

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

/// Active preset name. `None` means "use the per-project theme."
/// Persisted to disk so it survives restarts.
#[derive(Resource, Default, Debug, Clone)]
pub struct ActiveStylePreset(pub Option<String>);

pub struct PresetsPlugin;

impl Plugin for PresetsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<StylePresetRegistry>()
            .init_resource::<ActiveStylePreset>()
            .add_systems(
                Startup,
                (seed_default_presets, discover_presets, load_active_preset).chain(),
            )
            // Runs AFTER terminal-bevy's `mirror_active_project_to_style`
            // so we can override its choice of ActiveThemePath when a
            // preset is set. When preset is cleared, we restore the
            // per-project theme path ourselves.
            .add_systems(
                Update,
                (
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
    p.push(".terminal-bevy");
    p.push("styles");
    Some(p)
}

fn active_preset_file() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".terminal-bevy");
    p.push("active_style");
    Some(p)
}

const SEED_DEFAULT_THEME: &str = r###"// default style preset — matches the editor's built-in look.
// Per-project theme.rhai files can override individual tokens further.
#{
    pane_bg:                   "#1a1c20",
    pane_border:               "#2e3036",
    pane_border_focused:       "#4d6688",
    pane_focus_glow:           "#6b9eeb",
    pane_corner_radius:        6.0,
    pane_border_width:         1.0,
    pane_border_width_focused: 1.5,
    pane_focus_width:          8.0,
    pane_focus_strength:       0.35,
    pane_shadow_color:         "#00000073",  // 45% black
    pane_shadow_blur:          24.0,
    pane_shadow_offset_y:      6.0,
}
"###;

const SEED_LINEAR_THEME: &str = r###"// "linear" style preset — punchier borders, bigger radius, brighter
// focus glow, deeper shadow. Mimics modern-product chrome.
#{
    pane_bg:                   "#0d0e12",
    pane_border:               "#1f242e",
    pane_border_focused:       "#5b8def",
    pane_focus_glow:           "#5b8def",
    pane_corner_radius:        10.0,
    pane_border_width:         1.0,
    pane_border_width_focused: 2.0,
    pane_focus_width:          14.0,
    pane_focus_strength:       0.55,
    pane_shadow_color:         "#000000a6",  // 65% black
    pane_shadow_blur:          36.0,
    pane_shadow_offset_y:      10.0,
}
"###;

const SEED_PAPER_THEME: &str = r###"// "paper" style preset — warm cream body, brown borders, no focus
// glow at all, soft warm shadow. Reads like physical paper sitting on
// a wooden desk; intentionally the opposite of the dark/glassy looks.
#{
    pane_bg:                   "#f3eddf",
    pane_border:               "#c9b78f",
    pane_border_focused:       "#7a5a2c",
    pane_focus_glow:           "#a37d3a",
    pane_corner_radius:        3.0,
    pane_border_width:         1.0,
    pane_border_width_focused: 2.0,
    pane_focus_width:          0.0,   // no inner glow on paper
    pane_focus_strength:       0.0,
    pane_shadow_color:         "#3c2a107a",  // warm brown, ~48% alpha
    pane_shadow_blur:          18.0,
    pane_shadow_offset_y:      8.0,
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

fn load_active_preset(mut active: ResMut<ActiveStylePreset>) {
    let Some(path) = active_preset_file() else { return };
    let Ok(body) = std::fs::read_to_string(&path) else { return };
    let trimmed = body.trim();
    if trimmed.is_empty() {
        active.0 = None;
    } else {
        active.0 = Some(trimmed.to_string());
    }
}

fn persist_active_preset(name: Option<&str>) {
    let Some(path) = active_preset_file() else { return };
    let body = name.unwrap_or("");
    if let Err(e) = std::fs::write(&path, body) {
        warn!("style presets: persist {}: {}", path.display(), e);
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

fn drain_preset_messages(mut active: ResMut<ActiveStylePreset>) {
    let Some(rx) = PRESET_RX.get() else { return };
    let Ok(rx) = rx.lock() else { return };
    let mut applied: Option<Option<String>> = None;
    while let Ok(msg) = rx.try_recv() {
        applied = Some(msg.0);
    }
    if let Some(new_value) = applied {
        if active.0 != new_value {
            active.0 = new_value.clone();
            persist_active_preset(new_value.as_deref());
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
    _pad0: f32,
    _pad1: f32,
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

const SEED_NEON_THEME: &str = r###"// neon style preset — cyberpunk chrome with a rainbow border halo.
// The chrome.wgsl drives the visible look; this file just biases
// uniforms (bg color, glow strength, etc.) the shader reads.
#{
    pane_bg:                   "#080612",
    pane_border:               "#1a0830",
    pane_border_focused:       "#ff00aa",
    pane_focus_glow:           "#00f0ff",
    pane_corner_radius:        14.0,
    pane_border_width:         2.0,
    pane_border_width_focused: 3.0,
    pane_focus_width:          24.0,
    pane_focus_strength:       1.0,
    pane_shadow_color:         "#9000ff66",
    pane_shadow_blur:          40.0,
    pane_shadow_offset_y:      0.0,
}
"###;

// --- TERMINAL -----------------------------------------------------------
//
// Phosphor green on black. CRT scanlines, slight horizontal stripe
// noise, sharp 1px border, no rounding, no shadow body — looks like
// a tube monitor.

const SEED_TERMINAL_THEME: &str = r###"// terminal style preset — VT100 phosphor look.
#{
    pane_bg:                   "#020a02",
    pane_border:               "#19c618",
    pane_border_focused:       "#9bff7a",
    pane_focus_glow:           "#33ff33",
    pane_corner_radius:        0.0,
    pane_border_width:         1.0,
    pane_border_width_focused: 1.0,
    pane_focus_width:          0.0,
    pane_focus_strength:       0.0,
    pane_shadow_color:         "#00ff0033",
    pane_shadow_blur:          20.0,
    pane_shadow_offset_y:      0.0,
}
"###;

// --- WIREFRAME ----------------------------------------------------------
//
// Schematic drawing: nearly-transparent body so the canvas shows
// through, animated dashed border marching around the perimeter,
// a faint grid inside. Focus turns the dashes solid and bright.

const SEED_WIREFRAME_THEME: &str = r###"// wireframe style preset — schematic / blueprint look.
// chrome.wgsl draws marching-ants dashes; the body is mostly
// transparent so the canvas reads through.
#{
    pane_bg:                   "#0a0e1a",
    pane_border:               "#5588ff",
    pane_border_focused:       "#aaccff",
    pane_focus_glow:           "#aaccff",
    pane_corner_radius:        2.0,
    pane_border_width:         1.5,
    pane_border_width_focused: 2.0,
    pane_focus_width:          0.0,
    pane_focus_strength:       0.0,
    pane_shadow_color:         "#00000000",
    pane_shadow_blur:          1.0,
    pane_shadow_offset_y:      0.0,
}
"###;

// --- GLASS --------------------------------------------------------------
//
// Frosted glass: semi-transparent noisy body with a bright top
// highlight band and a soft inner gradient. No real backdrop blur
// (would require a render-graph pass capturing canvas underneath),
// just procedural noise that reads as "frosted." Focus brightens
// the top highlight.

const SEED_GLASS_THEME: &str = r###"// glass style preset — frosted translucency, top-edge highlight.
#{
    pane_bg:                   "#cfdce840",  // mostly transparent cool cyan
    pane_border:               "#ffffff35",
    pane_border_focused:       "#ffffff8c",
    pane_focus_glow:           "#ffffff",
    pane_corner_radius:        14.0,
    pane_border_width:         1.5,
    pane_border_width_focused: 2.5,
    pane_focus_width:          18.0,
    pane_focus_strength:       0.20,
    pane_shadow_color:         "#0a1a2870",
    pane_shadow_blur:          32.0,
    pane_shadow_offset_y:      8.0,
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

const SEED_INK_THEME: &str = r###"// ink — paper-like contrast for typography. No glow, no shadow,
// hairline border. The chrome shader's vertical body gradient still
// applies, but it's so subtle on near-white that it reads as flat.
#{
    pane_bg:                   "#fafaf5",
    pane_border:               "#2a2a28",
    pane_border_focused:       "#000000",
    pane_focus_glow:           "#2a2a28",
    pane_corner_radius:        1.0,
    pane_border_width:         0.5,
    pane_border_width_focused: 1.0,
    pane_focus_width:          0.0,
    pane_focus_strength:       0.0,
    pane_shadow_color:         "#00000000",
    pane_shadow_blur:          1.0,
    pane_shadow_offset_y:      0.0,
}
"###;

const SEED_AMBER_THEME: &str = r###"// amber — warm dark for evening sessions. Espresso body, amber
// border and glow. Reduces the cool-blue feel of the default dark.
#{
    pane_bg:                   "#1a1310",
    pane_border:               "#5e3b1a",
    pane_border_focused:       "#c87a2c",
    pane_focus_glow:           "#ffa64a",
    pane_corner_radius:        6.0,
    pane_border_width:         1.0,
    pane_border_width_focused: 1.5,
    pane_focus_width:          10.0,
    pane_focus_strength:       0.30,
    pane_shadow_color:         "#1a05007a",
    pane_shadow_blur:          24.0,
    pane_shadow_offset_y:      6.0,
}
"###;

const SEED_SLATE_THEME: &str = r###"// slate — brutalist no-nonsense. Hard corners, flat blocks, sharp
// monochrome borders. No shadow, no glow. For deep coding where
// chrome should be invisible.
#{
    pane_bg:                   "#232628",
    pane_border:               "#585d62",
    pane_border_focused:       "#aabac5",
    pane_focus_glow:           "#aabac5",
    pane_corner_radius:        0.0,
    pane_border_width:         2.0,
    pane_border_width_focused: 2.0,
    pane_focus_width:          0.0,
    pane_focus_strength:       0.0,
    pane_shadow_color:         "#00000000",
    pane_shadow_blur:          1.0,
    pane_shadow_offset_y:      0.0,
}
"###;

const SEED_BLUEPRINT_THEME: &str = r###"// blueprint — architect's drawing. Deep navy body, crisp cyan
// hairlines, sharp corners. Reads as technical diagram.
#{
    pane_bg:                   "#0d2030",
    pane_border:               "#4ec3f0",
    pane_border_focused:       "#a5e3ff",
    pane_focus_glow:           "#4ec3f0",
    pane_corner_radius:        0.0,
    pane_border_width:         1.0,
    pane_border_width_focused: 1.5,
    pane_focus_width:          0.0,
    pane_focus_strength:       0.0,
    pane_shadow_color:         "#00000000",
    pane_shadow_blur:          1.0,
    pane_shadow_offset_y:      0.0,
}
"###;

const SEED_FOREST_THEME: &str = r###"// forest — nature-toned alternative to the cool dark default.
// Deep moss-grey body, olive border, warm brown shadow.
#{
    pane_bg:                   "#1a2218",
    pane_border:               "#4a5e3e",
    pane_border_focused:       "#8caa6c",
    pane_focus_glow:           "#b8d090",
    pane_corner_radius:        6.0,
    pane_border_width:         1.0,
    pane_border_width_focused: 1.5,
    pane_focus_width:          8.0,
    pane_focus_strength:       0.25,
    pane_shadow_color:         "#1a100066",
    pane_shadow_blur:          20.0,
    pane_shadow_offset_y:      6.0,
}
"###;

const SEED_KINDLE_THEME: &str = r###"// kindle — minimal e-reader vibe for long reading sessions.
// Paper-warm body, super thin border, no glow at all.
#{
    pane_bg:                   "#faf6ed",
    pane_border:               "#b5ab9a",
    pane_border_focused:       "#5a5448",
    pane_focus_glow:           "#5a5448",
    pane_corner_radius:        1.0,
    pane_border_width:         0.5,
    pane_border_width_focused: 0.8,
    pane_focus_width:          0.0,
    pane_focus_strength:       0.0,
    pane_shadow_color:         "#00000000",
    pane_shadow_blur:          1.0,
    pane_shadow_offset_y:      0.0,
}
"###;

const SEED_SKETCH_THEME: &str = r###"// sketch — notebook page with character. Cream body with subtle
// pencil-noise texture (custom static shader; no per-frame cost),
// thin dark border, soft shadow.
#{
    pane_bg:                   "#f7f1de",
    pane_border:               "#5c5346",
    pane_border_focused:       "#1d1812",
    pane_focus_glow:           "#1d1812",
    pane_corner_radius:        2.0,
    pane_border_width:         0.75,
    pane_border_width_focused: 1.5,
    pane_focus_width:          0.0,
    pane_focus_strength:       0.0,
    pane_shadow_color:         "#2b1a0840",
    pane_shadow_blur:          14.0,
    pane_shadow_offset_y:      4.0,
}
"###;

const SEED_MESH_THEME: &str = r###"// mesh — quiet multi-stop gradient body. Bottom edge picks up the
// focus color, top edge stays bg. Body isn't flat but isn't busy.
// The chrome shader interpolates between params.bg and params.focus.
#{
    pane_bg:                   "#161a26",   // top
    pane_border:               "#2b324a",
    pane_border_focused:       "#6f7fa8",
    pane_focus_glow:           "#3a1e50",   // bottom of the gradient
    pane_corner_radius:        8.0,
    pane_border_width:         1.0,
    pane_border_width_focused: 1.5,
    pane_focus_width:          0.0,
    pane_focus_strength:       0.25,
    pane_shadow_color:         "#00000066",
    pane_shadow_blur:          22.0,
    pane_shadow_offset_y:      6.0,
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
