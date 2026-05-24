//! Design tokens loaded from `<project>/theme.rhai`.
//!
//! A token is a named value: a color, an f32, or a bool. Widget code
//! looks tokens up by [`TokenId`] (a `&'static str` newtype) so a typo
//! is a compile error.  The Rhai file may also define ad-hoc tokens
//! beyond the engine-known set — those are kept in the [`Theme`] map and
//! can be referenced by name (e.g. a custom shader reads
//! `theme.get_by_name("aurora_speed")`).
//!
//! ## Hot reload
//!
//! A background thread watches the project's theme.rhai via `notify`.
//! On any change we re-evaluate; on success the [`Theme`] resource is
//! replaced and [`ThemeChanged`] fires. On failure the last good theme
//! stays in place and the error string is stashed in [`StyleErrors`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};

use crate::state::StyleDataDir;
use crate::StyleErrors;

/// Stable identifier for an engine-known token. Constants live in the
/// [`tokens`] module so widget code references them as
/// `theme.color(tokens::ACCENT)` rather than stringly-typed lookups.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TokenId(pub &'static str);

#[derive(Clone, Copy, Debug)]
pub enum TokenValue {
    Color(LinearRgba),
    F32(f32),
    Bool(bool),
}

/// Active theme. Always populated — the default theme provides every
/// engine-known token, and project theme.rhai overrides selected ones.
#[derive(Resource, Clone, Debug)]
pub struct Theme {
    tokens: HashMap<String, TokenValue>,
}

impl Default for Theme {
    fn default() -> Self {
        Theme {
            tokens: default_tokens(),
        }
    }
}

impl Theme {
    pub fn get(&self, id: TokenId) -> Option<TokenValue> {
        self.tokens.get(id.0).copied()
    }

    pub fn get_by_name(&self, name: &str) -> Option<TokenValue> {
        self.tokens.get(name).copied()
    }

    pub fn color(&self, id: TokenId) -> LinearRgba {
        match self.tokens.get(id.0) {
            Some(TokenValue::Color(c)) => *c,
            Some(other) => panic!("theme token {:?} is not a Color: {:?}", id.0, other),
            None => panic!("theme token {:?} is missing (default theme is incomplete)", id.0),
        }
    }

    pub fn f32(&self, id: TokenId) -> f32 {
        match self.tokens.get(id.0) {
            Some(TokenValue::F32(v)) => *v,
            Some(other) => panic!("theme token {:?} is not an F32: {:?}", id.0, other),
            None => panic!("theme token {:?} is missing (default theme is incomplete)", id.0),
        }
    }

    pub fn bool(&self, id: TokenId) -> bool {
        match self.tokens.get(id.0) {
            Some(TokenValue::Bool(v)) => *v,
            Some(other) => panic!("theme token {:?} is not a Bool: {:?}", id.0, other),
            None => panic!("theme token {:?} is missing (default theme is incomplete)", id.0),
        }
    }

    pub fn set(&mut self, name: impl Into<String>, value: TokenValue) {
        self.tokens.insert(name.into(), value);
    }
}

/// Fired whenever the theme resource is replaced from a successful
/// theme.rhai load (or, in future, an inspector edit). Bevy 0.18
/// renamed `Event` → `Message`; we follow.
#[derive(Message, Clone, Copy, Debug)]
pub struct ThemeChanged;

/// Engine-known token IDs. Any shader's theme UBO reads from exactly
/// this set, in this order.
pub mod tokens {
    use super::TokenId;
    pub const BG: TokenId = TokenId("bg");
    pub const FG: TokenId = TokenId("fg");
    pub const FG_MUTED: TokenId = TokenId("fg_muted");
    pub const ACCENT: TokenId = TokenId("accent");
    pub const CARET: TokenId = TokenId("caret");
    pub const SELECTION: TokenId = TokenId("selection");
    pub const WARN: TokenId = TokenId("warn");
    pub const ERR: TokenId = TokenId("err");
    pub const FONT_SIZE: TokenId = TokenId("font_size");
    pub const LINE_HEIGHT_RATIO: TokenId = TokenId("line_height_ratio");
    /// Wipe-mask tuning: minimum dust_seconds for mouse motion to
    /// register as a wipe stroke. Below this the mask is not painted
    /// (i.e. "no visible dust to wipe yet"). Default ~10 min.
    pub const WIPE_DUST_GATE_SECS: TokenId = TokenId("wipe_dust_gate_secs");
    /// Wipe-mask tuning: brush radius in mask pixels (the mask is
    /// 1024×1024 UV-mapped over the canvas, so this is also roughly
    /// "fraction of canvas wiped per stamp" × 1024).
    pub const WIPE_BRUSH_RADIUS_PX: TokenId = TokenId("wipe_brush_radius_px");
    /// Multiplier on dust output for this project. Default 1.0; set
    /// to 0 in a project's `theme.rhai` to completely opt out of
    /// dust (e.g. "website" projects where the effect is distracting).
    /// 0.5 = half-strength dust; >1 amplifies but the visible result
    /// saturates fast.
    pub const DUST_INTENSITY: TokenId = TokenId("dust_intensity");

    // --- pane chrome (rounded-rect SDF material in pane-bevy) ---
    /// Body fill color of a pane.
    pub const PANE_BG: TokenId = TokenId("pane_bg");
    /// Border color, non-focused pane.
    pub const PANE_BORDER: TokenId = TokenId("pane_border");
    /// Border color, focused pane.
    pub const PANE_BORDER_FOCUSED: TokenId = TokenId("pane_border_focused");
    /// Focus-glow color (alpha replaced by `pane_focus_strength`).
    pub const PANE_FOCUS_GLOW: TokenId = TokenId("pane_focus_glow");
    /// Corner radius in pixels.
    pub const PANE_CORNER_RADIUS: TokenId = TokenId("pane_corner_radius");
    pub const PANE_BORDER_WIDTH: TokenId = TokenId("pane_border_width");
    pub const PANE_BORDER_WIDTH_FOCUSED: TokenId = TokenId("pane_border_width_focused");
    /// How far the focus glow fades into the body, in pixels.
    pub const PANE_FOCUS_WIDTH: TokenId = TokenId("pane_focus_width");
    /// Peak alpha of the focus glow at the inside edge of the border.
    pub const PANE_FOCUS_STRENGTH: TokenId = TokenId("pane_focus_strength");
    /// Drop-shadow color (rgb + base alpha).
    pub const PANE_SHADOW_COLOR: TokenId = TokenId("pane_shadow_color");
    /// How far the shadow fades, pixels. Also doubles as the shadow
    /// mesh padding around the pane, so it's how far the shadow
    /// visibly extends past the pane edge.
    pub const PANE_SHADOW_BLUR: TokenId = TokenId("pane_shadow_blur");
    /// Push the shadow down so it sits below the pane (positive).
    pub const PANE_SHADOW_OFFSET_Y: TokenId = TokenId("pane_shadow_offset_y");
}

fn default_tokens() -> HashMap<String, TokenValue> {
    use tokens::*;
    let mut m = HashMap::new();
    // Colors stored as linear-RGB; theme.rhai uses sRGB hex, converted
    // on parse.
    let srgb = |r, g, b, a| {
        let c = Color::srgba(r, g, b, a);
        TokenValue::Color(c.to_linear())
    };
    m.insert(BG.0.into(), srgb(0.072, 0.075, 0.085, 1.0));
    m.insert(FG.0.into(), srgb(0.860, 0.870, 0.900, 1.0));
    m.insert(FG_MUTED.0.into(), srgb(0.500, 0.520, 0.560, 1.0));
    m.insert(ACCENT.0.into(), srgb(0.420, 0.620, 0.920, 1.0));
    m.insert(CARET.0.into(), srgb(1.000, 0.800, 0.400, 1.0));
    m.insert(SELECTION.0.into(), srgb(0.149, 0.310, 0.471, 0.667));
    m.insert(WARN.0.into(), srgb(0.878, 0.686, 0.408, 1.0));
    m.insert(ERR.0.into(), srgb(0.969, 0.463, 0.557, 1.0));
    m.insert(FONT_SIZE.0.into(), TokenValue::F32(14.0));
    m.insert(LINE_HEIGHT_RATIO.0.into(), TokenValue::F32(1.3));
    // Lower default gate: natural dust starts being faintly visible
    // around 60s of focus on the default `sqrt(hours/24)` curve. Per
    // project: bump up in theme.rhai if you find the wipe too eager.
    m.insert(WIPE_DUST_GATE_SECS.0.into(), TokenValue::F32(60.0));
    m.insert(WIPE_BRUSH_RADIUS_PX.0.into(), TokenValue::F32(80.0));
    m.insert(DUST_INTENSITY.0.into(), TokenValue::F32(1.0));
    // Pane chrome defaults — match `pane_bevy::ChromeStyle::default()`.
    m.insert(PANE_BG.0.into(), srgb(0.105, 0.110, 0.122, 1.0));
    m.insert(PANE_BORDER.0.into(), srgb(0.18, 0.19, 0.22, 1.0));
    m.insert(PANE_BORDER_FOCUSED.0.into(), srgb(0.30, 0.40, 0.55, 1.0));
    m.insert(PANE_FOCUS_GLOW.0.into(), srgb(0.42, 0.62, 0.92, 1.0));
    m.insert(PANE_CORNER_RADIUS.0.into(), TokenValue::F32(6.0));
    m.insert(PANE_BORDER_WIDTH.0.into(), TokenValue::F32(1.0));
    m.insert(PANE_BORDER_WIDTH_FOCUSED.0.into(), TokenValue::F32(1.5));
    m.insert(PANE_FOCUS_WIDTH.0.into(), TokenValue::F32(8.0));
    m.insert(PANE_FOCUS_STRENGTH.0.into(), TokenValue::F32(0.35));
    m.insert(PANE_SHADOW_COLOR.0.into(), srgb(0.0, 0.0, 0.0, 0.45));
    m.insert(PANE_SHADOW_BLUR.0.into(), TokenValue::F32(24.0));
    m.insert(PANE_SHADOW_OFFSET_Y.0.into(), TokenValue::F32(6.0));
    m
}

// ---------- Rhai loader ----------

#[derive(Debug)]
pub enum ThemeLoadError {
    Rhai(String),
    BadValue { key: String, reason: String },
}

impl std::fmt::Display for ThemeLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rhai(e) => write!(f, "rhai: {}", e),
            Self::BadValue { key, reason } => write!(f, "token {:?}: {}", key, reason),
        }
    }
}

/// Evaluate `theme.rhai` and overlay its tokens onto a default theme.
pub fn load_theme(path: &Path) -> Result<Theme, ThemeLoadError> {
    let engine = rhai::Engine::new();
    let result: rhai::Dynamic = engine
        .eval_file::<rhai::Dynamic>(path.to_path_buf())
        .map_err(|e| ThemeLoadError::Rhai(e.to_string()))?;
    let map = result.try_cast::<rhai::Map>().ok_or_else(|| {
        ThemeLoadError::Rhai("theme.rhai must return a #{} map".to_string())
    })?;

    let mut theme = Theme::default();
    for (k, v) in map.into_iter() {
        let key = k.to_string();
        let value = parse_token(&key, &v)?;
        theme.set(key, value);
    }
    Ok(theme)
}

fn parse_token(key: &str, v: &rhai::Dynamic) -> Result<TokenValue, ThemeLoadError> {
    // Hex string "#rrggbb" or "#rrggbbaa" -> Color
    if let Some(s) = v.clone().try_cast::<String>() {
        return parse_hex_color(&s).map(TokenValue::Color).map_err(|reason| {
            ThemeLoadError::BadValue {
                key: key.into(),
                reason,
            }
        });
    }
    if let Some(s) = v.clone().try_cast::<rhai::ImmutableString>() {
        return parse_hex_color(s.as_str())
            .map(TokenValue::Color)
            .map_err(|reason| ThemeLoadError::BadValue {
                key: key.into(),
                reason,
            });
    }
    if let Some(b) = v.clone().try_cast::<bool>() {
        return Ok(TokenValue::Bool(b));
    }
    if let Some(i) = v.clone().try_cast::<i64>() {
        return Ok(TokenValue::F32(i as f32));
    }
    if let Some(f) = v.clone().try_cast::<f64>() {
        return Ok(TokenValue::F32(f as f32));
    }
    Err(ThemeLoadError::BadValue {
        key: key.into(),
        reason: format!("unsupported value type ({})", v.type_name()),
    })
}

fn parse_hex_color(s: &str) -> Result<LinearRgba, String> {
    let s = s.trim().trim_start_matches('#');
    let bytes = (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("bad hex: {}", e))?;
    let (r, g, b, a) = match bytes.as_slice() {
        [r, g, b] => (*r, *g, *b, 255),
        [r, g, b, a] => (*r, *g, *b, *a),
        _ => return Err(format!("expected #rrggbb or #rrggbbaa, got {:?}", s)),
    };
    Ok(Color::srgba_u8(r, g, b, a).to_linear())
}

// ---------- Hot reload plugin ----------

/// Resolves to the current `<base>/<active>/theme.rhai` path. Updated
/// whenever the active project changes.
#[derive(Resource, Default, Debug, Clone)]
pub struct ActiveThemePath(pub Option<PathBuf>);

/// Holds the running notify watcher plus the channel it pushes paths
/// into. Wrapped in `Mutex` so the resource is `Sync` (mpsc::Receiver
/// is `Send` but not `Sync`). Pattern lifted from `rhai_widget.rs` and
/// `garden_pane.rs` which do the same dance.
#[derive(Resource)]
struct ThemeWatcher {
    rx: Mutex<Receiver<PathBuf>>,
    _watcher: RecommendedWatcher,
    watched: PathBuf,
    last_reload: Instant,
}

pub struct ThemePlugin;

impl Plugin for ThemePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveThemePath>().add_systems(
            Update,
            (sync_watch_path, drain_watch_events).chain(),
        );
    }
}

/// When `ActiveThemePath` changes, point the notify watcher at the new
/// file. Also performs the initial load so a fresh project gets its
/// theme applied without any user gesture.
fn sync_watch_path(
    active: Res<ActiveThemePath>,
    mut commands: Commands,
    existing: Option<Res<ThemeWatcher>>,
    mut theme: ResMut<Theme>,
    mut errors: ResMut<StyleErrors>,
    mut ev: MessageWriter<ThemeChanged>,
) {
    if !active.is_changed() {
        return;
    }
    let new_path = active.0.clone();
    if let Some(w) = existing.as_ref()
        && Some(&w.watched) == new_path.as_ref()
    {
        return;
    }

    // Drop any old watcher.
    commands.remove_resource::<ThemeWatcher>();

    let Some(path) = new_path else {
        // No active project — fall back to default theme.
        *theme = Theme::default();
        ev.write(ThemeChanged);
        return;
    };

    // Initial load.
    reload_theme(&path, &mut theme, &mut errors, &mut ev);

    // Spin up a watcher on the parent directory; the watcher filters
    // for the exact file in its callback.
    let parent = match path.parent() {
        Some(p) => p.to_path_buf(),
        None => return,
    };
    let (tx, rx) = mpsc::channel::<PathBuf>();
    let watched = path.clone();
    let mut watcher = match notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        let Ok(event) = res else { return };
        for p in event.paths {
            if p == watched {
                let _ = tx.send(p);
            }
        }
    }) {
        Ok(w) => w,
        Err(e) => {
            errors.theme_error = Some(format!("watcher: {}", e));
            return;
        }
    };
    if let Err(e) = watcher.watch(&parent, RecursiveMode::NonRecursive) {
        errors.theme_error = Some(format!("watch {:?}: {}", parent, e));
        return;
    }
    commands.insert_resource(ThemeWatcher {
        rx: Mutex::new(rx),
        _watcher: watcher,
        watched: path,
        last_reload: Instant::now() - Duration::from_secs(1),
    });
}

fn drain_watch_events(
    watcher: Option<ResMut<ThemeWatcher>>,
    active: Res<ActiveThemePath>,
    mut theme: ResMut<Theme>,
    mut errors: ResMut<StyleErrors>,
    mut ev: MessageWriter<ThemeChanged>,
) {
    let Some(mut watcher) = watcher else { return };

    let mut got = false;
    {
        let rx = match watcher.rx.lock() {
            Ok(rx) => rx,
            Err(_) => return,
        };
        while rx.try_recv().is_ok() {
            got = true;
        }
    }
    if !got {
        return;
    }
    // Debounce: collapse a burst of events into one reload.
    let now = Instant::now();
    if now.duration_since(watcher.last_reload) < Duration::from_millis(150) {
        return;
    }
    watcher.last_reload = now;

    let Some(path) = active.0.clone() else { return };
    reload_theme(&path, &mut theme, &mut errors, &mut ev);
}

fn reload_theme(
    path: &Path,
    theme: &mut Theme,
    errors: &mut StyleErrors,
    ev: &mut MessageWriter<ThemeChanged>,
) {
    if !path.exists() {
        // No project-specific theme — keep default, clear any prior
        // error (file genuinely doesn't exist; not an error).
        *theme = Theme::default();
        errors.theme_error = None;
        ev.write(ThemeChanged);
        return;
    }
    match load_theme(path) {
        Ok(new_theme) => {
            *theme = new_theme;
            errors.theme_error = None;
            ev.write(ThemeChanged);
        }
        Err(e) => {
            errors.theme_error = Some(e.to_string());
            // Keep existing theme — don't blank everything on a typo.
        }
    }
}

/// Convenience: set [`ActiveThemePath`] from a project id + the data
/// dir. The host typically calls this from a system that watches its
/// own active-project resource.
pub fn theme_path_for_project(data_dir: &StyleDataDir, project_id: u64) -> PathBuf {
    data_dir.0.join(project_id.to_string()).join("theme.rhai")
}
