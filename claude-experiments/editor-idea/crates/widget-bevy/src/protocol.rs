//! Newline-delimited JSON protocol between the host and a widget process.
//!
//! This file is the single source of truth for the widget UI vocabulary:
//! `Element` (the UI tree), `Style`, `HostEvent` (host → widget), and
//! `WidgetMsg` (widget → host). Both hosting paths (in-process Rhai and
//! subprocess) speak this vocabulary. For the authoring guide — handlers,
//! the event model, examples — see `crates/widget-bevy/AUTHORING.md`.
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
    /// Publish a message onto the widget↔widget bus (the general
    /// signalling channel, separate from the Claude Code event bus).
    /// The host broadcasts it to every widget in the SAME editor project
    /// — each one receives a `HostEvent::Message` and filters by `topic`.
    /// `retain = true` keeps it as the topic's last value so a widget
    /// that spawns later receives it on init (MQTT-style retain).
    Emit {
        topic: String,
        #[serde(default)]
        payload: serde_json::Value,
        #[serde(default)]
        retain: bool,
    },
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
    Tick { dt: f32 },
    /// A Claude Code hook event mirrored from the central bus. `kind`
    /// matches the bus event kind (`pre_tool_use`, `user_prompt_submit`,
    /// `stop`, etc.); `payload` is the raw event payload parsed as
    /// JSON. Every running widget receives every event — filter by
    /// `kind` and `payload.cwd` if you only care about a subset.
    ClaudeEvent {
        kind: String,
        payload: serde_json::Value,
    },
    /// User clicked a tab in an `Element::Tabs`. `id` is the tabs
    /// group id; `tab` is the selected `TabItem.id`.
    TabSelect { id: String, tab: String },
    /// User toggled an `Element::Toggle`. `checked` is the new value.
    Toggle { id: String, checked: bool },
    /// User edited an `Element::Input`. `value` is the new full
    /// string; widget echoes it back in the next frame.
    InputChange { id: String, value: String },
    /// User focused/blurred an `Element::Input`. Widget mirrors
    /// `focused` in the next frame to reflect the new state.
    InputFocus { id: String, focused: bool },
    /// User submitted an `Element::Input` (typically Enter key).
    InputSubmit { id: String, value: String },
    /// A widget↔widget bus message addressed to this widget's project.
    /// Delivered (pushed, not polled) whenever another widget — or the
    /// `tbmsg` CLI — publishes via `emit` / `WidgetMsg::Emit`. `sender`
    /// is the originating widget's id (or `"tbmsg"` for the CLI) so the
    /// widget can ignore its own messages and route targeted replies.
    /// This is NOT the Claude Code bus (`ClaudeEvent`); it carries
    /// widget-app control signals.
    Message {
        topic: String,
        #[serde(default)]
        payload: serde_json::Value,
        sender: String,
    },
}

/// Optional visual override applied on top of theme defaults. Every
/// flow-layout element accepts a `style: Style` to set its own
/// background / border / radius / shadow / padding / margin.
///
/// String values for color / radius / shadow accept either a literal
/// value (e.g. `"#1a1d24"`, `"oklch(0.65, 0.04, 250)"`, `"12"`) or a
/// theme token name (e.g. `"surface_2"`, `"radius_md"`, `"shadow_sm"`).
/// The renderer resolves token names against the active theme; literal
/// values short-circuit the lookup.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct Style {
    /// Solid background color (token name or literal). Painted as a
    /// rounded rect when `radius` is also set; otherwise a sharp rect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<String>,
    /// Optional background image painted under the children. Anchored
    /// to the element's top-left and stretched to fill. Use this for
    /// the "Textures & motifs" thumbnails in the Atelier mockup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background_image: Option<String>,
    /// Corner radius. Accepts a token name (e.g. `"radius_md"`,
    /// `"radius_pill"`) or a literal pixel value as a string
    /// (e.g. `"8"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<String>,
    /// Border drawn just inside the element's bounds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub border: Option<Border>,
    /// Drop shadow rendered outside the element bounds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow: Option<Shadow>,
    /// Inner padding. Stacks already accept `pad`; setting both means
    /// `style.padding` wins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub padding: Option<Edges>,
    /// Outer margin around the element.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub margin: Option<Edges>,

    // ---------- Flex-layout knobs (consumed by Taffy) ----------
    /// How much of the parent's remaining main-axis space this element
    /// should grow into. `1.0` on every sibling in a row → equal
    /// distribution. Default 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flex_grow: Option<f32>,
    /// How much this element should shrink when there isn't enough
    /// room. Default 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flex_shrink: Option<f32>,
    /// Explicit width. Number → pixels. `"100%"` → percent of parent.
    /// `"auto"` (default) → intrinsic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<String>,
    /// Explicit height.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_width: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_width: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_height: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_height: Option<String>,
    /// Override the parent's cross-axis alignment for this child only.
    /// Useful e.g. when most cards in a row should stretch but one
    /// should center.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub align_self: Option<Align>,
    /// Override a container's main-axis direction (`"row"` / `"column"`). Lets
    /// a Glaze `when` breakpoint flip a stack between row and column for
    /// responsive layout, independent of the Element variant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flex_direction: Option<String>,

    /// Ordered paint plan produced by Glaze. When non-empty, these layers
    /// replace the scalar `background` / `border` / `shadow` / `shader`
    /// paint fields while leaving layout fields unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub glaze_layers: Vec<GlazeLayer>,

    /// A compiled Glaze shader layer painted on this element's box. The
    /// `body` is WGSL fragment-shader source (the `glaze` compiler's output
    /// for an `overlay shader {}` block); the host wraps it in the canonical
    /// `GlazeUniforms` block and runs it on a quad at the element's rect.
    ///
    /// Kept for protocol compatibility. New Glaze output uses
    /// [`Style::glaze_layers`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shader: Option<ShaderSpec>,
}

/// One entry in Glaze's ordered paint plan.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum GlazeLayer {
    Fill {
        color: String,
    },
    Border {
        color: String,
        width: f32,
    },
    Shadow {
        color: String,
        blur: f32,
        offset_y: f32,
    },
    Shader {
        body: String,
        #[serde(default)]
        overlay: bool,
    },
}

/// A compiled shader layer: WGSL fragment body produced by `glaze`, plus
/// its source-level `overlay` intent. The current widget renderer keeps
/// the full paint plan beneath child content.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ShaderSpec {
    /// WGSL fragment body — `let`s + `return <vec4>;`, referencing `u.*`
    /// uniforms and `in.uv`. Wrapped by the host into a full module.
    pub body: String,
    #[serde(default)]
    pub overlay: bool,
}

/// Border specification. `color` is a token name or literal; `width`
/// is in pixels.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Border {
    pub color: String,
    #[serde(default = "default_border_width")]
    pub width: f32,
}

fn default_border_width() -> f32 {
    1.0
}

/// Drop shadow. Either set `token: "shadow_md"` to pull a coupled
/// triple from the theme, or supply explicit `color` / `blur` /
/// `offset_y`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct Shadow {
    /// Convenience: name of a `shadow_*` token triple. When present,
    /// `color/blur/offset_y` below are ignored unless they explicitly
    /// override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// Shadow color (token name or literal). Defaults to `shadow_md_color`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// Blur radius in px.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blur: Option<f32>,
    /// Positive pushes shadow downward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_y: Option<f32>,
}

/// Four-sided edge values. Use named fields for asymmetric padding/
/// margin. Missing sides default to 0.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema)]
pub struct Edges {
    #[serde(default)]
    pub top: f32,
    #[serde(default)]
    pub right: f32,
    #[serde(default)]
    pub bottom: f32,
    #[serde(default)]
    pub left: f32,
}

impl Edges {
    pub fn all(v: f32) -> Self {
        Self {
            top: v,
            right: v,
            bottom: v,
            left: v,
        }
    }
    pub fn symmetric(h: f32, v: f32) -> Self {
        Self {
            top: v,
            right: h,
            bottom: v,
            left: h,
        }
    }
    pub fn horizontal(&self) -> f32 {
        self.left + self.right
    }
    pub fn vertical(&self) -> f32 {
        self.top + self.bottom
    }
}

/// Visual kind for an Element::Button. Filled draws the standard
/// solid-bg button; Outline draws transparent bg + accent border;
/// Ghost draws label only with a subtle hover fill.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ButtonKind {
    #[default]
    Filled,
    Outline,
    Ghost,
}

/// One tab in an Element::Tabs strip.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TabItem {
    pub id: String,
    pub label: String,
}

/// One column definition in an Element::Table.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TableColumn {
    /// Header label shown in the top row.
    pub header: String,
    /// Fixed column width in px. `None` shares the remaining width
    /// equally with other auto columns (CSS grid `1fr`).
    #[serde(default)]
    pub width: Option<f32>,
    /// Horizontal text alignment for this column's cells (and header).
    #[serde(default)]
    pub align: Align,
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        style: Option<Style>,
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        style: Option<Style>,
    },
    /// Generic styled container. Functionally a Vstack, but exists so
    /// `style` is the primary affordance (the name reads as "give me a
    /// card / section / panel" at a call site).
    Frame {
        #[serde(default)]
        gap: f32,
        #[serde(default)]
        pad: f32,
        #[serde(default)]
        children: Vec<Element>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        style: Option<Style>,
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
        /// Font family name resolved through the host's
        /// `style_bevy::FontRegistry`. Unknown names fall back to mono.
        /// Typical values: `"serif"`, `"sans"`, `"mono"`, or pull a
        /// token name from theme.rhai (`"font_family_heading"`).
        #[serde(default)]
        family: Option<String>,
        /// Drag-selectable: the user can drag across this text to
        /// highlight a range and Cmd/Ctrl+C the selection. ON by default
        /// for read-only labels; set `selectable: false` to opt a label
        /// out (e.g. text that's part of a custom drag gesture).
        #[serde(default = "default_true")]
        selectable: bool,
    },
    /// Clickable button. The host sends `{"event":"click","id":"<id>"}`
    /// back to the widget on press; ids must be unique within a frame.
    Button {
        id: String,
        label: String,
        #[serde(default)]
        kind: ButtonKind,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        style: Option<Style>,
    },
    /// Tab strip. Renders one `Text` per item with an indicator under
    /// the selected one. Click sends
    /// `{"event":"tab-select","id":"<tabs-id>","tab":"<tab-id>"}` back
    /// to the widget.
    Tabs {
        /// Identity for the tab group; used as the `id` in the
        /// outbound event.
        id: String,
        items: Vec<TabItem>,
        /// Id of the currently-selected tab. Empty string = none.
        #[serde(default)]
        selected: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        style: Option<Style>,
    },
    /// On/off pill toggle. Renders an iOS-style track with a sliding
    /// knob. Click sends `{"event":"toggle","id":"<id>","checked":<bool>}`
    /// — `checked` is the new value (already inverted).
    Toggle {
        id: String,
        #[serde(default)]
        label: String,
        #[serde(default)]
        checked: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        style: Option<Style>,
    },
    /// Selectable row. Functionally a Frame that's click-targetable,
    /// with `selected` driving a visual highlight (the accent border /
    /// background pulled from the active theme).
    ListItem {
        id: String,
        #[serde(default)]
        children: Vec<Element>,
        #[serde(default)]
        gap: f32,
        #[serde(default)]
        pad: f32,
        #[serde(default)]
        selected: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        style: Option<Style>,
    },
    /// Single-line text input. Click to focus; typing emits
    /// `{"event":"input-change","id":"<id>","value":"<new>"}` back to
    /// the widget on every change. Maintains a blinking caret while
    /// focused; supports left/right arrows, Home/End, backspace,
    /// delete. No IME.
    Input {
        id: String,
        #[serde(default)]
        value: String,
        #[serde(default)]
        placeholder: String,
        /// Mirrored from the widget. Set this to drive focus
        /// programmatically; otherwise the host updates it on click.
        #[serde(default)]
        focused: bool,
        #[serde(default = "default_input_width")]
        width: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        style: Option<Style>,
    },
    /// Multi-line text input — like `Input`, but `Enter` inserts a
    /// newline and the box is `rows` text-lines tall. Submit (the
    /// `input-submit` event) is **Cmd/Ctrl+Enter**, not plain Enter, so
    /// authors can write multi-line queries. Click to focus; emits the
    /// same `input-change` / `input-submit` / `input-focus` events as
    /// `Input`, carrying the full multi-line `value`. Hard newlines only
    /// (no soft wrap). Caret supports arrows (incl. up/down across
    /// lines), Home/End (line-aware), backspace, delete. No IME.
    #[serde(rename = "textarea", alias = "text-area")]
    TextArea {
        id: String,
        #[serde(default)]
        value: String,
        #[serde(default)]
        placeholder: String,
        /// Mirrored from the widget; set to drive focus programmatically.
        #[serde(default)]
        focused: bool,
        /// Visible height in text lines. Defaults to 4.
        #[serde(default = "default_textarea_rows")]
        rows: u32,
        #[serde(default = "default_input_width")]
        width: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        style: Option<Style>,
    },
    /// Data table: a header row plus data rows laid out on a CSS grid.
    /// `columns` defines the header text, per-column width (fixed px, or
    /// `null` to share remaining space equally), and text alignment.
    /// `rows` is row-major; each inner vector is one row's cells (missing
    /// cells render empty, extras are ignored). Long cell text wraps
    /// within its column. Set `zebra` for alternating row backgrounds.
    Table {
        #[serde(default)]
        columns: Vec<TableColumn>,
        #[serde(default)]
        rows: Vec<Vec<String>>,
        #[serde(default)]
        zebra: bool,
        /// Cells are drag-selectable: drag across a cell to highlight a
        /// range and Cmd/Ctrl+C it — the "grab one value out of the
        /// results" workflow. ON by default; set `selectable: false` to
        /// disable.
        #[serde(default = "default_true")]
        selectable: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        style: Option<Style>,
    },
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
        /// Drag-selectable like `Text`. ON by default; set
        /// `selectable: false` to disable.
        #[serde(default = "default_true")]
        selectable: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        style: Option<Style>,
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
    /// Colored block, fixed size. Useful for color swatches inside
    /// theme/picker UIs — render a solid square of the given color
    /// at a known size. Hex / oklch() / oklab() / rgb() all accepted.
    /// Optional `id` makes it click-targetable like a Button.
    Swatch {
        color: String,
        #[serde(default = "default_swatch_size")]
        size: f32,
        #[serde(default)]
        id: Option<String>,
    },
    /// Clickable colored swatch tile. Same as Swatch but always emits
    /// a click target so the widget receives `id` on press. Kept as
    /// a separate variant so non-interactive swatches don't take a
    /// hit-test slot.
    SwatchButton {
        id: String,
        color: String,
        #[serde(default = "default_swatch_size")]
        size: f32,
    },
}

fn default_swatch_size() -> f32 {
    14.0
}

fn default_input_width() -> f32 {
    160.0
}

fn default_textarea_rows() -> u32 {
    4
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
        /// Clockwise rotation in degrees about the item's anchor point,
        /// measured in canvas (y-down) space. 0 = axis-aligned. Lets a
        /// widget draw diagonal strokes (e.g. a move arrow) out of plain
        /// rects without a dedicated line primitive.
        #[serde(default)]
        rotation: f32,
    },
    /// A run of text positioned absolutely in the canvas. `anchor` is
    /// applied to the text-bounds box; e.g. `center` places (x, y) at
    /// the text's center. Use this for on-board labels (rank/file marks,
    /// piece counts, status overlays) without exiting Canvas mode.
    Text {
        id: String,
        x: f32,
        y: f32,
        value: String,
        /// `#rrggbb` or `#rrggbbaa`. Defaults to white.
        #[serde(default)]
        color: Option<String>,
        /// Font size in logical pixels. Defaults to 14.
        #[serde(default)]
        size: Option<f32>,
        /// Font family name resolved through the host's FontRegistry.
        /// Unknown → mono.
        #[serde(default)]
        family: Option<String>,
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
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Align {
    #[default]
    Start,
    Center,
    End,
    /// Make every child consume the full cross-axis size. The typical
    /// use: cards in a row should be the same height — `Stretch` on
    /// the row's `align` does it declaratively.
    Stretch,
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

/// Read-only text displays (`Text`, `Table`, `Badge`) are drag-selectable
/// by default; a widget opts a specific element out with
/// `selectable: false`.
fn default_true() -> bool {
    true
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

#[cfg(test)]
mod style_tests {
    use super::*;

    #[test]
    fn glaze_layers_round_trip_in_order() {
        let style = Style {
            glaze_layers: vec![
                GlazeLayer::Fill {
                    color: "#112233".into(),
                },
                GlazeLayer::Shader {
                    body: "return vec4<f32>(1.0);".into(),
                    overlay: true,
                },
            ],
            ..Default::default()
        };

        let json = serde_json::to_string(&style).unwrap();
        let decoded: Style = serde_json::from_str(&json).unwrap();
        assert!(matches!(decoded.glaze_layers[0], GlazeLayer::Fill { .. }));
        assert!(matches!(
            decoded.glaze_layers[1],
            GlazeLayer::Shader { overlay: true, .. }
        ));
    }
}
