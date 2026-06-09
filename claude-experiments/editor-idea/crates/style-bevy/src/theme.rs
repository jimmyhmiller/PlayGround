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

#[derive(Clone, Debug)]
pub enum TokenValue {
    Color(LinearRgba),
    F32(f32),
    Bool(bool),
    /// Free-form string. Used by font-family tokens that name a bundled
    /// font (resolved through [`crate::FontRegistry`]) and by any future
    /// token that wants a symbolic value.
    Str(String),
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
        self.tokens.get(id.0).cloned()
    }

    pub fn get_by_name(&self, name: &str) -> Option<TokenValue> {
        self.tokens.get(name).cloned()
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

    pub fn str_value(&self, id: TokenId) -> &str {
        match self.tokens.get(id.0) {
            Some(TokenValue::Str(s)) => s.as_str(),
            Some(other) => panic!("theme token {:?} is not a Str: {:?}", id.0, other),
            None => panic!("theme token {:?} is missing (default theme is incomplete)", id.0),
        }
    }

    pub fn set(&mut self, name: impl Into<String>, value: TokenValue) {
        self.tokens.insert(name.into(), value);
    }

    /// All currently-known token names (engine defaults + theme.rhai
    /// additions). Returned unsorted; callers sort if they need to.
    pub fn token_names(&self) -> Vec<String> {
        self.tokens.keys().cloned().collect()
    }
}

/// Fired whenever the theme resource is replaced from a successful
/// theme.rhai load (or, in future, an inspector edit). Bevy 0.18
/// renamed `Event` → `Message`; we follow.
#[derive(Message, Clone, Copy, Debug)]
pub struct ThemeChanged;

/// Per-project resolved themes. The global [`Theme`] only ever holds the
/// *active* project's look; this cache holds one [`Theme`] per project so
/// each pane can render in its OWN project's theme (needed for the cube
/// overview, where every project is on screen at once, and to make
/// flat-view per-project theming correct rather than coincidental).
///
/// Populated by the host (which owns the project list + each project's
/// preset) via [`Self::set`]; read by every pane-theming system through
/// [`Self::get`], which falls back to `None` for unknown projects (the
/// caller then uses the global theme).
#[derive(Resource, Default, Debug, Clone)]
pub struct ProjectThemes {
    by_project: HashMap<u64, Theme>,
}

impl ProjectThemes {
    pub fn set(&mut self, project_id: u64, theme: Theme) {
        self.by_project.insert(project_id, theme);
    }
    pub fn get(&self, project_id: u64) -> Option<&Theme> {
        self.by_project.get(&project_id)
    }
    pub fn contains(&self, project_id: u64) -> bool {
        self.by_project.contains_key(&project_id)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&u64, &Theme)> {
        self.by_project.iter()
    }
    /// Drop entries for projects no longer present, so deleted projects
    /// don't leak. `keep` is the current set of live project ids.
    pub fn retain_projects(&mut self, keep: &std::collections::HashSet<u64>) {
        self.by_project.retain(|id, _| keep.contains(id));
    }
}

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

    // --- pane chrome text (title bar / close × / resize handle) ---
    pub const CHROME_TITLE: TokenId = TokenId("chrome_title");
    pub const CHROME_TITLE_FOCUSED: TokenId = TokenId("chrome_title_focused");
    /// Title-bar background fill (separate from pane body so focus can
    /// change the title strip without touching the body).
    pub const CHROME_TITLE_BG: TokenId = TokenId("chrome_title_bg");
    pub const CHROME_TITLE_BG_FOCUSED: TokenId = TokenId("chrome_title_bg_focused");
    pub const CHROME_DIVIDER: TokenId = TokenId("chrome_divider");
    pub const CHROME_CLOSE: TokenId = TokenId("chrome_close");
    pub const CHROME_HANDLE: TokenId = TokenId("chrome_handle");

    // --- code syntax highlighting (editor) ---
    pub const SYNTAX_DEFAULT: TokenId = TokenId("syntax_default");
    pub const SYNTAX_KEYWORD: TokenId = TokenId("syntax_keyword");
    pub const SYNTAX_STRING: TokenId = TokenId("syntax_string");
    pub const SYNTAX_COMMENT: TokenId = TokenId("syntax_comment");
    pub const SYNTAX_FUNCTION: TokenId = TokenId("syntax_function");
    pub const SYNTAX_TYPE: TokenId = TokenId("syntax_type");
    pub const SYNTAX_ATTRIBUTE: TokenId = TokenId("syntax_attribute");
    pub const SYNTAX_CONSTANT: TokenId = TokenId("syntax_constant");
    pub const SYNTAX_OPERATOR: TokenId = TokenId("syntax_operator");
    pub const SYNTAX_PUNCTUATION: TokenId = TokenId("syntax_punctuation");
    pub const SYNTAX_VARIABLE: TokenId = TokenId("syntax_variable");
    pub const SYNTAX_PROPERTY: TokenId = TokenId("syntax_property");
    pub const SYNTAX_LABEL: TokenId = TokenId("syntax_label");
    pub const SYNTAX_ESCAPE: TokenId = TokenId("syntax_escape");
    pub const SYNTAX_CONSTRUCTOR: TokenId = TokenId("syntax_constructor");

    // --- form fields (used by run-button and any widget text input) ---
    pub const INPUT_BG: TokenId = TokenId("input_bg");
    pub const INPUT_TEXT: TokenId = TokenId("input_text");
    pub const INPUT_TEXT_FOCUSED: TokenId = TokenId("input_text_focused");

    // --- buttons (run-button save/play + widget protocol Button) ---
    pub const BUTTON_BG: TokenId = TokenId("button_bg");
    pub const BUTTON_BG_HOVER: TokenId = TokenId("button_bg_hover");
    pub const BUTTON_LABEL: TokenId = TokenId("button_label");
    /// Subdued button color for "save" / "confirm" actions.
    pub const BUTTON_PRIMARY_BG: TokenId = TokenId("button_primary_bg");
    pub const BUTTON_PRIMARY_LABEL: TokenId = TokenId("button_primary_label");

    // --- status colors (run-button play state, badge, etc.) ---
    pub const STATUS_IDLE: TokenId = TokenId("status_idle");
    pub const STATUS_RUNNING: TokenId = TokenId("status_running");
    pub const STATUS_SUCCESS: TokenId = TokenId("status_success");
    pub const STATUS_FAILED: TokenId = TokenId("status_failed");

    // --- radial menu ---
    pub const RADIAL_WEDGE: TokenId = TokenId("radial_wedge");
    pub const RADIAL_WEDGE_HOVER: TokenId = TokenId("radial_wedge_hover");
    pub const RADIAL_DEADZONE: TokenId = TokenId("radial_deadzone");
    pub const RADIAL_DEADZONE_RING: TokenId = TokenId("radial_deadzone_ring");
    pub const RADIAL_LABEL: TokenId = TokenId("radial_label");
    pub const RADIAL_LABEL_HOVER: TokenId = TokenId("radial_label_hover");
    pub const RADIAL_ICON: TokenId = TokenId("radial_icon");
    pub const RADIAL_BACKDROP: TokenId = TokenId("radial_backdrop");

    // --- widget protocol bits not covered above ---
    pub const WIDGET_BAR_TRACK: TokenId = TokenId("widget_bar_track");
    pub const WIDGET_BAR_FILL: TokenId = TokenId("widget_bar_fill");
    pub const WIDGET_BADGE_BG: TokenId = TokenId("widget_badge_bg");
    pub const WIDGET_BADGE_LABEL: TokenId = TokenId("widget_badge_label");
    pub const WIDGET_LINK: TokenId = TokenId("widget_link");
    // --- widget Button shape / shadow (SDF material) ---
    pub const WIDGET_BUTTON_CORNER_RADIUS: TokenId = TokenId("widget_button_corner_radius");
    pub const WIDGET_BUTTON_BORDER: TokenId = TokenId("widget_button_border");
    pub const WIDGET_BUTTON_BORDER_WIDTH: TokenId = TokenId("widget_button_border_width");
    pub const WIDGET_BUTTON_SHADOW_COLOR: TokenId = TokenId("widget_button_shadow_color");
    pub const WIDGET_BUTTON_SHADOW_BLUR: TokenId = TokenId("widget_button_shadow_blur");
    pub const WIDGET_BUTTON_SHADOW_OFFSET_Y: TokenId = TokenId("widget_button_shadow_offset_y");

    // --- canvas (main camera ClearColor — the "void" behind panes) ---
    pub const CANVAS_BG: TokenId = TokenId("canvas_bg");

    // --- sidebar (project list, left edge) ---
    pub const SIDEBAR_BG: TokenId = TokenId("sidebar_bg");
    pub const SIDEBAR_ROW_ACTIVE_BG: TokenId = TokenId("sidebar_row_active_bg");
    pub const SIDEBAR_ROW_RENAMING_BG: TokenId = TokenId("sidebar_row_renaming_bg");
    pub const SIDEBAR_TEXT_FAINT: TokenId = TokenId("sidebar_text_faint");

    // ---------- Design-token scales ----------
    //
    // Five-step ramps for layout primitives. Widget code that takes a
    // style override is free to plug a literal pixel value, but
    // theme.rhai authors should reach for these tokens so the whole UI
    // moves coherently when the scale is tuned.

    // --- spacing scale (px) ---
    pub const SPACE_XS: TokenId = TokenId("space_xs");
    pub const SPACE_SM: TokenId = TokenId("space_sm");
    pub const SPACE_MD: TokenId = TokenId("space_md");
    pub const SPACE_LG: TokenId = TokenId("space_lg");
    pub const SPACE_XL: TokenId = TokenId("space_xl");

    // --- radius scale (px) ---
    pub const RADIUS_XS: TokenId = TokenId("radius_xs");
    pub const RADIUS_SM: TokenId = TokenId("radius_sm");
    pub const RADIUS_MD: TokenId = TokenId("radius_md");
    pub const RADIUS_LG: TokenId = TokenId("radius_lg");
    /// Sentinel for fully-rounded ends — render code clamps to half the
    /// shorter side so the value itself can be any "large" number.
    pub const RADIUS_PILL: TokenId = TokenId("radius_pill");

    // --- shadow ladder ---
    // Each step is three coupled tokens: color (rgba), blur (px), and
    // vertical offset (px, positive = shadow below).
    pub const SHADOW_SM_COLOR: TokenId = TokenId("shadow_sm_color");
    pub const SHADOW_SM_BLUR: TokenId = TokenId("shadow_sm_blur");
    pub const SHADOW_SM_OFFSET_Y: TokenId = TokenId("shadow_sm_offset_y");
    pub const SHADOW_MD_COLOR: TokenId = TokenId("shadow_md_color");
    pub const SHADOW_MD_BLUR: TokenId = TokenId("shadow_md_blur");
    pub const SHADOW_MD_OFFSET_Y: TokenId = TokenId("shadow_md_offset_y");
    pub const SHADOW_LG_COLOR: TokenId = TokenId("shadow_lg_color");
    pub const SHADOW_LG_BLUR: TokenId = TokenId("shadow_lg_blur");
    pub const SHADOW_LG_OFFSET_Y: TokenId = TokenId("shadow_lg_offset_y");

    // --- surface ladder (card-elevation backgrounds) ---
    /// Same as BG/PANE_BG conceptually; explicit so cards inside cards
    /// have a coherent stack.
    pub const SURFACE_1: TokenId = TokenId("surface_1");
    pub const SURFACE_2: TokenId = TokenId("surface_2");
    pub const SURFACE_3: TokenId = TokenId("surface_3");

    // --- accent ramp ---
    // Tailwind-style 50..900 tint scale derived from ACCENT. Default
    // theme generates them by mixing ACCENT toward white (low numbers)
    // and toward black (high numbers).
    pub const ACCENT_50: TokenId = TokenId("accent_50");
    pub const ACCENT_100: TokenId = TokenId("accent_100");
    pub const ACCENT_200: TokenId = TokenId("accent_200");
    pub const ACCENT_300: TokenId = TokenId("accent_300");
    pub const ACCENT_400: TokenId = TokenId("accent_400");
    pub const ACCENT_500: TokenId = TokenId("accent_500");
    pub const ACCENT_600: TokenId = TokenId("accent_600");
    pub const ACCENT_700: TokenId = TokenId("accent_700");
    pub const ACCENT_800: TokenId = TokenId("accent_800");
    pub const ACCENT_900: TokenId = TokenId("accent_900");

    // --- typography ---
    /// Resolved against the [`crate::FontRegistry`]. Names map to
    /// bundled fonts; unknown names fall back to "mono".
    pub const FONT_FAMILY_HEADING: TokenId = TokenId("font_family_heading");
    pub const FONT_FAMILY_BODY: TokenId = TokenId("font_family_body");
    pub const FONT_FAMILY_MONO: TokenId = TokenId("font_family_mono");
}

/// Atelier dark palette, embedded so the engine has a sensible
/// default look without depending on any on-disk preset. The same
/// bytes are also seeded to `~/.jim/styles/atelier/theme.rhai`
/// at first launch so the preset picker has an entry for it.
pub const ATELIER_DEFAULT_THEME: &str =
    include_str!("../assets/themes/atelier.rhai");

fn default_tokens() -> HashMap<String, TokenValue> {
    parse_theme_str(ATELIER_DEFAULT_THEME)
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

/// Parse a rhai theme document from a string, returning the
/// `(name, TokenValue)` pairs. Used by [`default_tokens`] to load the
/// embedded Atelier preset at startup. Errors panic — the embedded
/// file is shipped with the binary, so a parse failure is a build-time
/// bug, not a runtime issue.
fn parse_theme_str(src: &str) -> HashMap<String, TokenValue> {
    let engine = rhai::Engine::new();
    let result: rhai::Dynamic = engine
        .eval::<rhai::Dynamic>(src)
        .expect("embedded theme rhai must parse");
    let map = result
        .try_cast::<rhai::Map>()
        .expect("embedded theme must return a #{} map");
    let mut out = HashMap::with_capacity(map.len());
    for (k, v) in map.into_iter() {
        let key = k.to_string();
        let value = parse_token(&key, &v).expect("embedded theme token must parse");
        out.insert(key, value);
    }
    out
}

fn parse_token(key: &str, v: &rhai::Dynamic) -> Result<TokenValue, ThemeLoadError> {
    // Strings: try color forms first ("#rrggbb[aa]", "oklch(...)",
    // "oklab(...)", "rgb(...)"). If none match, keep as a free-form Str
    // — used for font-family names like "serif" / "mono".
    if let Some(s) = v.clone().try_cast::<String>() {
        return Ok(match parse_color_string(&s) {
            Ok(c) => TokenValue::Color(c),
            Err(_) => TokenValue::Str(s),
        });
    }
    if let Some(s) = v.clone().try_cast::<rhai::ImmutableString>() {
        let owned = s.to_string();
        return Ok(match parse_color_string(&owned) {
            Ok(c) => TokenValue::Color(c),
            Err(_) => TokenValue::Str(owned),
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

/// Parse any of the accepted color string forms. Hex (`#rrggbb` or
/// `#rrggbbaa`), `oklch(L, C, h[, a])`, `oklab(L, a, b[, alpha])`,
/// or `rgb(r, g, b[, a])` (each 0-255). Whitespace is ignored.
pub fn parse_color_string(s: &str) -> Result<LinearRgba, String> {
    let trimmed = s.trim();
    if trimmed.starts_with('#') || trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
        return parse_hex_color(trimmed);
    }
    if let Some(args) = parse_func_call(trimmed, "oklch") {
        return parse_oklch_args(&args);
    }
    if let Some(args) = parse_func_call(trimmed, "oklab") {
        return parse_oklab_args(&args);
    }
    if let Some(args) = parse_func_call(trimmed, "rgb") {
        return parse_rgb_args(&args);
    }
    if let Some(args) = parse_func_call(trimmed, "rgba") {
        return parse_rgb_args(&args);
    }
    Err(format!("unrecognized color form: {:?}", s))
}

/// If `s` looks like `name(a, b, c)`, return `Some("a, b, c")`. Else None.
fn parse_func_call(s: &str, name: &str) -> Option<String> {
    let s = s.trim();
    let lower = s.to_ascii_lowercase();
    let prefix = format!("{}(", name);
    if !lower.starts_with(&prefix) || !lower.ends_with(')') {
        return None;
    }
    let inside = &s[prefix.len()..s.len() - 1];
    Some(inside.to_string())
}

fn split_args(s: &str) -> Vec<&str> {
    s.split(',').map(str::trim).filter(|p| !p.is_empty()).collect()
}

fn parse_oklch_args(s: &str) -> Result<LinearRgba, String> {
    let parts = split_args(s);
    if !(parts.len() == 3 || parts.len() == 4) {
        return Err(format!("oklch expects 3 or 4 args, got {}", parts.len()));
    }
    let l: f32 = parts[0].parse().map_err(|e| format!("oklch L: {}", e))?;
    let c: f32 = parts[1].parse().map_err(|e| format!("oklch C: {}", e))?;
    let h: f32 = parts[2].parse().map_err(|e| format!("oklch h: {}", e))?;
    let alpha: f32 = if parts.len() == 4 {
        parts[3].parse().map_err(|e| format!("oklch alpha: {}", e))?
    } else {
        1.0
    };
    let mut out = crate::oklab::oklch_to_linear_srgb(l, c, h);
    out.alpha = alpha;
    Ok(out)
}

fn parse_oklab_args(s: &str) -> Result<LinearRgba, String> {
    let parts = split_args(s);
    if !(parts.len() == 3 || parts.len() == 4) {
        return Err(format!("oklab expects 3 or 4 args, got {}", parts.len()));
    }
    let l: f32 = parts[0].parse().map_err(|e| format!("oklab L: {}", e))?;
    let a: f32 = parts[1].parse().map_err(|e| format!("oklab a: {}", e))?;
    let b: f32 = parts[2].parse().map_err(|e| format!("oklab b: {}", e))?;
    let alpha: f32 = if parts.len() == 4 {
        parts[3].parse().map_err(|e| format!("oklab alpha: {}", e))?
    } else {
        1.0
    };
    let mut out = crate::oklab::oklab_to_linear_srgb(l, a, b);
    out.alpha = alpha;
    Ok(out)
}

fn parse_rgb_args(s: &str) -> Result<LinearRgba, String> {
    let parts = split_args(s);
    if !(parts.len() == 3 || parts.len() == 4) {
        return Err(format!("rgb expects 3 or 4 args, got {}", parts.len()));
    }
    let r: u8 = parts[0].parse().map_err(|e| format!("rgb r: {}", e))?;
    let g: u8 = parts[1].parse().map_err(|e| format!("rgb g: {}", e))?;
    let b: u8 = parts[2].parse().map_err(|e| format!("rgb b: {}", e))?;
    let a: u8 = if parts.len() == 4 {
        let v: f32 = parts[3].parse().map_err(|e| format!("rgb alpha: {}", e))?;
        (v.clamp(0.0, 1.0) * 255.0).round() as u8
    } else {
        255
    };
    Ok(Color::srgba_u8(r, g, b, a).to_linear())
}

pub fn parse_hex_color(s: &str) -> Result<LinearRgba, String> {
    let s = s.trim().trim_start_matches('#');
    // Expand CSS-style shorthand: #rgb → #rrggbb, #rgba → #rrggbbaa.
    let expanded;
    let hex: &str = match s.len() {
        3 | 4 => {
            expanded = s.chars().flat_map(|c| [c, c]).collect::<String>();
            &expanded
        }
        6 | 8 => s,
        n => return Err(format!("expected #rgb, #rgba, #rrggbb, or #rrggbbaa, got {} chars ({:?})", n, s)),
    };
    let bytes = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("bad hex: {}", e))?;
    let (r, g, b, a) = match bytes.as_slice() {
        [r, g, b] => (*r, *g, *b, 255),
        [r, g, b, a] => (*r, *g, *b, *a),
        _ => unreachable!(),
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
            audit_contrast(theme, path);
            errors.theme_error = None;
            ev.write(ThemeChanged);
        }
        Err(e) => {
            errors.theme_error = Some(e.to_string());
            // Keep existing theme — don't blank everything on a typo.
        }
    }
}

/// Critical foreground/background pairs that should have legible
/// contrast in every theme. Audited on each theme load; failures get
/// logged with the OkLab ΔL between the two so the author knows how
/// bad and on which surface.
const AUDIT_PAIRS: &[(&str, TokenId, TokenId)] = &[
    ("body text", tokens::FG, tokens::BG),
    ("syntax comment", tokens::SYNTAX_COMMENT, tokens::BG),
    ("syntax string", tokens::SYNTAX_STRING, tokens::BG),
    ("syntax keyword", tokens::SYNTAX_KEYWORD, tokens::BG),
    ("input text", tokens::INPUT_TEXT, tokens::INPUT_BG),
    ("button label", tokens::BUTTON_LABEL, tokens::BUTTON_BG),
    ("button primary", tokens::BUTTON_PRIMARY_LABEL, tokens::BUTTON_PRIMARY_BG),
    ("chrome title", tokens::CHROME_TITLE_FOCUSED, tokens::PANE_BG),
    ("radial label", tokens::RADIAL_LABEL, tokens::RADIAL_WEDGE),
];

/// OkLab ΔL threshold below which a pair is flagged. 25 is a rough
/// proxy for WCAG AA on small body text — colors with less ΔL than
/// that tend to read as the same brightness and won't separate.
const AUDIT_DELTA_L_THRESHOLD: f32 = 25.0;

fn audit_contrast(theme: &Theme, path: &Path) {
    use crate::oklab::lightness_delta;
    let preset = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("?");
    for (label, fg_id, bg_id) in AUDIT_PAIRS {
        let fg = theme.color(*fg_id);
        let bg = theme.color(*bg_id);
        let delta = lightness_delta(fg, bg);
        if delta < AUDIT_DELTA_L_THRESHOLD {
            warn!(
                "[theme/{}] low contrast on {}: ΔL = {:.1} (threshold {:.0}) — \
                 {} on {} may be hard to read",
                preset, label, delta, AUDIT_DELTA_L_THRESHOLD, fg_id.0, bg_id.0,
            );
        }
    }
}

/// Convenience: set [`ActiveThemePath`] from a project id + the data
/// dir. The host typically calls this from a system that watches its
/// own active-project resource.
pub fn theme_path_for_project(data_dir: &StyleDataDir, project_id: u64) -> PathBuf {
    data_dir.0.join(project_id.to_string()).join("theme.rhai")
}
