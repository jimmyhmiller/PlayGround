//! Newline-delimited JSON protocol between the host and a widget process.
//!
//! Widget → host (`{"type": …, …}`):
//!   - `frame` : full retained UI tree to render
//!   - `state` : opaque blob host persists in `PaneSnapshot.config`
//!   - `title` : title-bar text
//!
//! Host → widget (`{"event": …, …}`):
//!   - `init`    : sent once after spawn (size + saved state)
//!   - `resize`  : pane content area changed
//!   - `click`   : user pressed a button with this id
//!   - `refresh` : user-requested reload
//!   - `close`   : pane is being closed; widget should exit promptly

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A message produced by the widget process (one NDJSON line on stdout).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum WidgetMsg {
    /// Full retained UI tree to render. Sending a new `frame` fully
    /// replaces the previous one — there is no diffing protocol.
    Frame { root: Element },
    /// Opaque blob the host persists in `PaneSnapshot.config` and feeds
    /// back via `init.state` on the next launch. Use it to checkpoint
    /// scroll position, form values, etc.
    State { value: serde_json::Value },
    /// Title-bar text for the hosting pane.
    Title { value: String },
}

/// An event the host delivers to the widget (one NDJSON line on stdin).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "event", rename_all = "kebab-case")]
pub enum HostEvent {
    /// Sent once after spawn. Carries the initial content size and the
    /// last `state` blob the previous run published (or `null`).
    Init {
        width: f32,
        height: f32,
        #[serde(default)]
        state: serde_json::Value,
    },
    /// The pane's content area changed. Width/height are in logical px.
    Resize { width: f32, height: f32 },
    /// User pressed a `button` with the matching `id`.
    Click { id: String },
    /// User-requested reload (e.g. via the pane's refresh affordance).
    Refresh,
    /// The pane is closing; flush and exit promptly.
    Close,
    /// Per-frame heartbeat. `dt` is seconds since the previous tick.
    /// Animated widgets advance their state on each tick and emit a
    /// new `frame`. Static widgets can ignore. Rate-limited to ~30Hz
    /// in the host so subprocesses on a slow link don't get flooded.
    Tick {
        dt: f32,
    },
    /// A Claude Code hook event mirrored from the central bus. `kind`
    /// matches the bus event kind (`pre_tool_use`, `user_prompt_submit`,
    /// `stop`, etc.); `payload` is the raw event payload parsed as
    /// JSON. Every running widget receives every event — filter by
    /// `kind` and `payload.cwd` if you only care about a subset.
    ClaudeEvent {
        kind: String,
        payload: serde_json::Value,
    },
}

/// One node in the widget's UI tree. Every frame is a single root
/// `Element` that the host walks top-down to spawn Bevy entities.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Element {
    /// Vertical stack — lays children top-to-bottom with `gap` px between
    /// each and `pad` px around the group.
    Vstack {
        #[serde(default)]
        gap: f32,
        #[serde(default)]
        pad: f32,
        #[serde(default)]
        children: Vec<Element>,
    },
    /// Horizontal stack — children laid out left-to-right. `align`
    /// controls cross-axis (vertical) placement of children.
    Hstack {
        #[serde(default)]
        gap: f32,
        #[serde(default)]
        pad: f32,
        #[serde(default = "default_align")]
        align: Align,
        #[serde(default)]
        children: Vec<Element>,
    },
    /// A run of text. `color` accepts `#rrggbb` / `#rgb`. `size` is in
    /// logical pixels; defaults to the pane's body size. Overflow is
    /// truncated with an ellipsis.
    Text {
        value: String,
        #[serde(default)]
        color: Option<String>,
        #[serde(default)]
        size: Option<f32>,
        #[serde(default)]
        weight: Option<Weight>,
    },
    /// Clickable button. The host sends `{"event":"click","id":"<id>"}`
    /// back to the widget on press; ids must be unique within a frame.
    Button { id: String, label: String },
    /// Hyperlink. Click opens `url` in the system browser; no event is
    /// delivered back to the widget.
    Link { url: String, label: String },
    /// Horizontal hairline that fills the available width.
    Divider,
    /// Inserts `size` px of empty space along the parent's main axis.
    Spacer {
        #[serde(default = "default_spacer")]
        size: f32,
    },
    /// Small colored pill, typically used for status labels.
    Badge {
        value: String,
        #[serde(default)]
        color: Option<String>,
    },
    /// Vertical scroll region. v0 lays children out as a vstack and
    /// clips to the available height (no actual scrolling yet — clipping
    /// is enough to validate the protocol; wheel handling lands later).
    Scroll {
        #[serde(default)]
        gap: f32,
        #[serde(default)]
        pad: f32,
        children: Vec<Element>,
    },
    /// Filled horizontal bar with a background track. `value/max` sets
    /// the fill ratio (clamped to 0..1). Use it for progress bars,
    /// gauges, or — stacked in an hstack with `width` shrunk — bar-graph
    /// columns and sparkline-ish strips.
    Bar {
        value: f32,
        #[serde(default = "default_bar_max")]
        max: f32,
        /// Fill color. Defaults to a neutral accent.
        #[serde(default)]
        color: Option<String>,
        /// Background-track color. Defaults to a subtle dark fill.
        #[serde(default)]
        track: Option<String>,
        #[serde(default = "default_bar_width")]
        width: f32,
        #[serde(default = "default_bar_height")]
        height: f32,
    },
    /// Absolute-positioned drawing region. Children are placed by `x,y`
    /// (pixels from the canvas top-left, y down) regardless of any
    /// parent stack layout. Use this for games / gardens / visualizers
    /// where you don't want flow layout. Canvas children are
    /// CanvasItems (sprites, rects), not full Elements.
    Canvas {
        #[serde(default)]
        children: Vec<CanvasItem>,
    },
}

/// One item inside an absolute-positioned `Canvas`. Position is
/// pixel-space relative to the canvas top-left, y-down.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum CanvasItem {
    /// A textured sprite, optionally hue-shifted in HSV space.
    Sprite {
        /// Stable identity for the host to diff frames. Must be
        /// unique within a frame. Reusing the same id across frames
        /// lets the host smoothly update position/scale of an
        /// existing sprite instead of despawn+respawn.
        id: String,
        x: f32,
        y: f32,
        /// On-screen width. The image is upscaled to fit.
        w: f32,
        h: f32,
        /// `image_ref` is either a filesystem path or a host-known
        /// builtin (sky/ground sprite shorthands, etc.).
        #[serde(flatten)]
        image: ImageRef,
        /// HSV hue rotation in degrees, applied to every non-transparent
        /// pixel. 0 = source colors.
        #[serde(default)]
        hue_shift: f32,
        /// `bottom-center` is the default — useful for plants that
        /// grow from a ground line. Set to `center` for free-floating
        /// objects (butterflies, particles).
        #[serde(default = "default_canvas_anchor")]
        anchor: CanvasAnchor,
        /// Stacking order within the canvas; higher draws on top.
        #[serde(default)]
        z: f32,
    },
    /// A solid-color rectangle. Use for backgrounds, ground strips,
    /// status fills, etc.
    Rect {
        id: String,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        /// `#rrggbb` or `#rrggbbaa` (preserved alpha).
        color: String,
        #[serde(default = "default_canvas_anchor_topleft")]
        anchor: CanvasAnchor,
        #[serde(default)]
        z: f32,
    },
}

/// Where the (x,y) coordinate is on the item — i.e., what point of the
/// sprite gets pinned at the given position.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum CanvasAnchor {
    TopLeft,
    TopCenter,
    Center,
    BottomCenter,
    BottomLeft,
}

/// Reference to an image. Today only filesystem paths are supported;
/// future variants might add a built-in registry of host-provided
/// sprites or inline pixel buffers.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "image", rename_all = "kebab-case")]
pub enum ImageRef {
    /// Filesystem path to a PNG/JPEG file. The host loads + caches the
    /// image and shares one `Handle<Image>` across every reference.
    /// `~` is NOT expanded; pass absolute paths or paths relative to
    /// the widget script's directory.
    Path { path: String },
    /// One tile of a host-managed sprite sheet referenced by file path
    /// + (col, row) into a `tile_w × tile_h` grid. Lets a widget pick
    /// individual tiles out of a strip like the OGA "Flowers" sheet
    /// without slicing it itself.
    Tile {
        path: String,
        tile_w: u32,
        tile_h: u32,
        col: u32,
        #[serde(default)]
        row: u32,
    },
}

/// Cross-axis alignment for hstack children.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Align {
    Start,
    Center,
    End,
}

/// Font weight for text runs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Weight {
    Normal,
    Bold,
}

fn default_align() -> Align {
    Align::Center
}

fn default_spacer() -> f32 {
    8.0
}

fn default_bar_max() -> f32 {
    1.0
}

fn default_bar_width() -> f32 {
    80.0
}

fn default_bar_height() -> f32 {
    8.0
}

fn default_canvas_anchor() -> CanvasAnchor {
    CanvasAnchor::BottomCenter
}

fn default_canvas_anchor_topleft() -> CanvasAnchor {
    CanvasAnchor::TopLeft
}

/// Parse `#rrggbb` or `#rgb` into 0..1 sRGB components.
pub fn parse_hex_color(s: &str) -> Option<[f32; 3]> {
    let s = s.strip_prefix('#')?;
    let (r, g, b) = match s.len() {
        6 => (
            u8::from_str_radix(&s[0..2], 16).ok()?,
            u8::from_str_radix(&s[2..4], 16).ok()?,
            u8::from_str_radix(&s[4..6], 16).ok()?,
        ),
        3 => {
            let r = u8::from_str_radix(&s[0..1], 16).ok()?;
            let g = u8::from_str_radix(&s[1..2], 16).ok()?;
            let b = u8::from_str_radix(&s[2..3], 16).ok()?;
            (r * 17, g * 17, b * 17)
        }
        _ => return None,
    };
    Some([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0])
}
