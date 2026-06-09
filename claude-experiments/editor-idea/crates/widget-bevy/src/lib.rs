//! Widget panes: a UI-as-data tree (`Element`) rendered by the host,
//! driven by events.
//!
//! ## Read this first: `../AUTHORING.md`
//!
//! `crates/widget-bevy/AUTHORING.md` is the authoring guide — the two
//! hosting paths, the full handler / event model, and worked examples.
//! Start there if you're writing a widget. Common gotcha it heads off:
//! **UI events and the Claude Code bus are separate channels** (the bus
//! handler is `on_bus`, *not* UI events).
//!
//! ## Two hosting paths
//!
//! - **In-process Rhai** — `rhai_widget.rs`. A `.rhai` script runs on a
//!   worker thread and the host calls named handlers (`on_click`,
//!   `on_toggle`, `on_input_change`, `on_bus`, …). Hot-reloaded from
//!   `~/.terminal-bevy/widgets/`. This is the default for new widgets.
//! - **Subprocess** — *this file*. Each pane spawns a child process and
//!   pipes NDJSON: the host sends one `HostEvent` per line, the child
//!   replies with `frame` / `state` / `title` messages. See
//!   `protocol.rs` for the message shapes.
//!
//! Both paths share the same `Element` vocabulary (`protocol.rs`) and
//! the same rendering / hit-testing / scroll / focused-input typing,
//! which live in this file and `render.rs` / `layout.rs`.
//!
//! Config keys consumed by the subprocess `spawn`:
//!   - `command` (string)  — shell command line to run; falls back to
//!                            `WIDGET_BEVY_DEFAULT_CMD` env var, then a
//!                            placeholder frame if neither is set.
//!   - `args`    (string[]) — when present, runs `command args…` directly
//!                            (no shell). When absent, `command` is fed
//!                            to `sh -c`.
//!   - `title`   (string)  — initial pane title bar text.
//!   - `cwd`     (string)  — working directory for the child.
//!   - `state`   (any)     — last `state` blob the widget published;
//!                            sent back as `init.state` next launch.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, Sender};

use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::LineHeight;
use claude_bus_bevy::ClaudeBusEvent;
use pane_bevy::{
    MARGIN, PaneContentPressed, PaneFont, PaneFontMetrics, PaneHotZones, PaneKindMarker,
    PaneKindSpec, PaneRect, PaneRegistry, PaneTitle, TITLE_H,
};
use serde_json::Value;

pub mod button_material;
pub mod glaze_material;
pub mod glaze_style;
pub mod layout;
pub mod msgbus;
pub mod protocol;
pub mod render;
pub mod rhai_widget;
pub mod subprocess;

pub use msgbus::{PendingMsg, WidgetMsgBus};

pub use button_material::{
    ButtonParams as WidgetButtonParams, WidgetButtonMaterial, WidgetButtonMaterialPlugin,
};

/// Process-global callback that wakes the host's main loop. The host
/// (terminal-bevy) runs winit in a reactive update mode, so when it's
/// idle the main loop only wakes on input or the reactive timeout
/// (~5s). Widget worker threads run OFF the main thread; when a worker
/// produces something the main thread must observe while the window is
/// idle — it `set_animating(true)` (needs the mode-maintainer to flip to
/// `Continuous`), publishes a frame, or emits on the bus — there's
/// nothing to wake the loop, so the change can stall ~5s (or forever, if
/// the work depends on `on_frame` ticking). This hook lets a worker
/// nudge the loop awake. widget-bevy deliberately doesn't depend on
/// winit, so the host installs a closure that fires its `EventLoopProxy`.
static WAKEUP_HOOK: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>> = std::sync::OnceLock::new();

/// Install the main-loop wakeup callback. Called once by the host at
/// startup. Subsequent calls are ignored (first hook wins).
pub fn set_wakeup_hook(f: impl Fn() + Send + Sync + 'static) {
    let _ = WAKEUP_HOOK.set(Box::new(f));
}

/// Wake the host's main loop if a hook is installed. No-op otherwise
/// (e.g. headless tests). Cheap and thread-safe; safe to call from
/// worker threads.
pub fn request_main_loop_wakeup() {
    if let Some(f) = WAKEUP_HOOK.get() {
        f();
    }
}

/// Per-widget-pane vertical scroll state. Updated by
/// `handle_widget_wheel` when the user scrolls over the pane; applied
/// to `PaneChrome.content_root.transform.y` by
/// `apply_widget_scroll`. The two render paths (`rerender_widgets`
/// here and `apply_latest_frames` in rhai_widget) write `max_y`
/// after they know how tall the content drew.
#[derive(Component, Default, Debug)]
pub struct WidgetScroll {
    /// Pixels scrolled down from the top. Always non-negative.
    pub y: f32,
    /// `content_height - viewport_height`, clamped ≥ 0. When 0 the
    /// content fits and no scrolling is allowed.
    pub max_y: f32,
}

use protocol::{CanvasAnchor, CanvasItem, Element, HostEvent, ImageRef, Weight, WidgetMsg};

/// Stable identifier for widget panes. Used in `PaneKindMarker` and
/// snapshots.
pub const PANE_KIND: &str = "widget";

/// Env var consulted when a pane is spawned without an explicit command
/// (i.e. from the radial menu rather than a saved snapshot).
pub const DEFAULT_CMD_ENV: &str = "WIDGET_BEVY_DEFAULT_CMD";

// ---------- Components ----------

/// Widget config + per-pane state that persists across process exits.
#[derive(Component)]
pub struct Widget {
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    /// Most recent `state` message the child published. Mirrored into
    /// snapshots so the next launch can resume.
    pub last_state: Value,
}

impl Widget {
    pub fn new(command: impl Into<String>, args: Vec<String>, cwd: Option<PathBuf>) -> Self {
        Self {
            command: command.into(),
            args,
            cwd,
            last_state: Value::Null,
        }
    }
}

/// Handle to the running child process. Absent when the widget has no
/// command configured (placeholder mode) or after the child exits.
#[derive(Component)]
pub struct WidgetProcess {
    pub child: Child,
}

/// Channels to/from the IO threads. `rx` carries parsed widget→host
/// messages; `tx` carries host→widget JSON lines (already serialized).
#[derive(Component)]
pub struct WidgetIO {
    pub rx: Mutex<Receiver<WidgetMsg>>,
    pub tx: Sender<String>,
}

/// What the host knows about the visual: the active frame, an optional
/// pending frame waiting to be rendered, and the size we last rendered
/// at so we can detect resizes.
#[derive(Component, Default)]
pub struct WidgetRender {
    pub pending_frame: Option<Element>,
    pub current_frame: Option<Element>,
    pub last_size: Vec2,
    pub init_sent: bool,
    /// Last time (in `Time::elapsed_secs`) we sent a `Tick` host event.
    /// Used by `forward_ticks` to rate-limit ticks to ~30Hz regardless
    /// of the host's frame rate.
    pub last_tick_secs: f32,
    /// Set when transient render state (hover, etc.) changes so the
    /// next pass through `rerender_widgets` redraws even if the element
    /// tree itself didn't change. The hover system flips this.
    pub force_render: bool,
}

/// Which clickable target (if any) the mouse is currently over inside
/// this widget pane. Updated each frame from `WidgetTargets.clicks` and
/// the cursor position. The id string matches the `id` carried on the
/// element + `ClickTarget`. Cleared to `None` when the mouse leaves the
/// pane or moves off all clickables.
#[derive(Component, Default, Debug, Clone)]
pub struct WidgetHover {
    pub click_id: Option<String>,
}

/// Hit-test geometry collected while rendering the current frame.
#[derive(Component, Default)]
pub struct WidgetTargets {
    pub clicks: Vec<ClickTarget>,
    pub links: Vec<LinkTarget>,
    /// Selectable text runs (from `Text`/`Table` with `selectable: true`).
    /// The drag-select systems map a drag onto one of these and copy the
    /// covered substring on Cmd/Ctrl+C.
    pub spans: Vec<TextSpan>,
    /// Draggable slider hit-regions. A press/drag inside one maps the cursor
    /// x to a value and emits `SliderChange`.
    pub sliders: Vec<SliderTarget>,
    /// `Select` triggers collected this frame (anchor + data for the overlay
    /// renderer when one is open).
    pub selects: Vec<SelectTarget>,
    /// `Tooltip` hover-regions collected this frame.
    pub tooltips: Vec<TooltipTarget>,
    /// Open `Dialog`s collected this frame (rendered centered on the overlay).
    pub dialogs: Vec<DialogTarget>,
    /// `Popover` triggers collected this frame.
    pub popovers: Vec<PopoverTarget>,
    /// `Toast` notifications collected this frame.
    pub toasts: Vec<ToastTarget>,
}

/// A draggable `Element::Slider` hit-region collected during render. `rect` is
/// the full element box (content_root-local); value maps over the thumb-centre
/// travel `[value_x0, value_x0 + value_span]`.
#[derive(Clone, Debug)]
pub struct SliderTarget {
    pub id: String,
    pub rect: Rect,
    pub value_x0: f32,
    pub value_span: f32,
    pub min: f32,
    pub max: f32,
    pub step: f32,
}

impl SliderTarget {
    /// Map a content-local cursor x to this slider's clamped, step-snapped value.
    pub fn value_at(&self, x: f32) -> f32 {
        let ratio = ((x - self.value_x0) / self.value_span.max(1.0)).clamp(0.0, 1.0);
        let mut v = self.min + ratio * (self.max - self.min);
        if self.step > 0.0 {
            v = self.min + ((v - self.min) / self.step).round() * self.step;
        }
        v.clamp(self.min.min(self.max), self.min.max(self.max))
    }
}

/// Marker on a widget pane while a slider drag is in progress. Holds the
/// resolved value mapping so drag events don't need to re-find the target.
#[derive(Component, Clone)]
pub struct WidgetSliderDrag {
    target: SliderTarget,
    last_value: f32,
}

// ============================================================
// Floating overlay layer (Select dropdowns, menus, …)
// ============================================================

/// The `RenderLayers` id the floating overlay content renders on. Must be a
/// layer the host (a) reserves in `PaneLayerAllocator` and (b) renders with a
/// high-order camera (`clear_color: None`). Defaults to 32, which terminal-bevy
/// already reserves + cameras for (`MENU_OVERLAY_LAYER`).
#[derive(Resource, Clone, Copy)]
pub struct WidgetOverlayLayer(pub usize);

impl Default for WidgetOverlayLayer {
    fn default() -> Self {
        WidgetOverlayLayer(32)
    }
}

/// Which `Select` (if any) currently has its dropdown open. Host-owned (the open
/// state is transient UI, not widget data), one at a time app-wide.
#[derive(Resource, Default)]
pub struct WidgetOpenSelect(pub Option<OpenSelect>);

#[derive(Clone)]
pub struct OpenSelect {
    pub pane: Entity,
    pub id: String,
}

/// Marks the per-frame floating overlay root entity (rebuilt each frame). The
/// menu Element subtree renders under it; `stamp_overlay_layers` propagates the
/// overlay `RenderLayers` to its descendants.
#[derive(Component)]
pub struct WidgetOverlayRoot;

/// Marks the per-frame floating tooltip root (separate from `WidgetOverlayRoot`
/// so the select and tooltip renderers despawn only their own content).
#[derive(Component)]
pub struct WidgetTooltipRoot;

/// Marks the per-frame modal-dialog root (scrim + panel) on the overlay layer.
#[derive(Component)]
pub struct WidgetDialogRoot;

/// Marks the per-frame floating popover root on the overlay layer.
#[derive(Component)]
pub struct WidgetPopoverRoot;

/// Marks the per-frame toast root (corner notifications) on the overlay layer.
#[derive(Component)]
pub struct WidgetToastRoot;

/// A `Toast` collected during render — shown stacked at the window corner.
#[derive(Clone)]
pub struct ToastTarget {
    pub id: String,
    pub text: String,
    pub style: Option<crate::protocol::ToastStyle>,
}

/// Window-space rects for visible toasts (clicking one dismisses it). Rebuilt
/// each frame by `render_toast_overlay`.
#[derive(Resource, Default)]
pub struct ToastHits {
    /// `(window rect, toast id, owning pane)`
    pub items: Vec<(Rect, String, Entity)>,
}

/// Which `Popover` (if any) is open. Host-owned (transient UI), one at a time.
#[derive(Resource, Default)]
pub struct WidgetOpenPopover(pub Option<OpenSelect>);

/// A `Popover` trigger collected during render: anchor + the arbitrary content
/// to float when open.
#[derive(Clone)]
pub struct PopoverTarget {
    pub id: String,
    pub anchor: Rect,
    pub content: Option<Box<crate::protocol::Element>>,
    pub width: f32,
    pub style: Option<crate::protocol::PopoverStyle>,
}

/// Window-space routing for the open popover: content click hits + the surface
/// and trigger rects (a click outside both dismisses).
#[derive(Resource, Default)]
pub struct PopoverHits {
    pub pane: Option<Entity>,
    pub popover_id: String,
    pub surface_rect: Rect,
    pub trigger_rect: Rect,
    pub clicks: Vec<OverlayClickHit>,
}

/// An open `Dialog` collected during render (it has no in-pane visual — the
/// overlay renderer draws it centered when `open`).
#[derive(Clone)]
pub struct DialogTarget {
    pub id: String,
    pub title: String,
    pub body: Option<Box<crate::protocol::Element>>,
    pub width: f32,
    pub style: Option<crate::protocol::DialogStyle>,
}

/// One routed click-region for arbitrary overlay content (a dialog body's
/// buttons). Window-space rect + the event it fires.
#[derive(Clone)]
pub struct OverlayClickHit {
    pub rect: Rect,
    pub kind: ClickKind,
    pub id: String,
}

/// Window-space click routing for the open dialog: its body's click hits plus
/// the panel rect (so a click outside the panel = scrim dismiss). Rebuilt each
/// frame by `render_dialog_overlay`.
#[derive(Resource, Default)]
pub struct WidgetOverlayHits {
    pub pane: Option<Entity>,
    pub dialog_id: String,
    pub panel_rect: Rect,
    pub clicks: Vec<OverlayClickHit>,
}

/// A `Select` trigger collected during render: its content-local anchor rect plus
/// the data the overlay renderer needs to draw the open dropdown.
#[derive(Clone)]
pub struct SelectTarget {
    pub id: String,
    pub anchor: Rect,
    pub options: Vec<crate::protocol::TabItem>,
    pub value: String,
    pub width: f32,
    pub style: Option<crate::protocol::SelectStyle>,
}

/// One open-dropdown item's window-space rect, recorded for the overlay input
/// hit-test (which runs outside the pane framework, like the context menu).
#[derive(Clone)]
pub struct SelectMenuHit {
    pub option_id: String,
    pub rect: Rect,
}

/// A `Tooltip` collected during render: its content-local anchor rect plus the
/// hint text + style. Hovered → shown on the overlay layer.
#[derive(Clone)]
pub struct TooltipTarget {
    pub anchor: Rect,
    pub text: String,
    pub style: Option<crate::protocol::TooltipStyle>,
}

/// Which tooltip (if any) the cursor is hovering, resolved each frame.
#[derive(Resource, Default)]
pub struct ActiveTooltip(pub Option<ActiveTip>);

#[derive(Clone)]
pub struct ActiveTip {
    pub pane: Entity,
    pub anchor: Rect,
    pub text: String,
    pub style: Option<crate::protocol::TooltipStyle>,
}

/// Window-space hit-regions for the currently-open dropdown's items, plus the
/// owning select id. Rebuilt each frame by the overlay renderer.
#[derive(Resource, Default)]
pub struct SelectMenuHits {
    pub select_id: String,
    pub pane: Option<Entity>,
    pub items: Vec<SelectMenuHit>,
    /// The trigger's window-space rect — the dismiss handler ignores the click
    /// that lands here (it's the toggle, handled by the pane press handler).
    pub trigger_rect: Rect,
}

/// One run of rendered, selectable text, collected during render so a
/// drag can be mapped onto a character range. `rect` is content_root-
/// local (y-down, px from the content top-left) — the same frame as
/// `ClickTarget.rect` and `PaneContentPressed.local_pt`. `rect.min` is
/// the text's top-left (the glyph origin), so character offsets measure
/// left-to-right from there.
pub struct TextSpan {
    pub text: String,
    pub rect: Rect,
    pub font_size: f32,
}

pub struct ClickTarget {
    pub id: String,
    /// What HostEvent to emit when this rect is clicked. Plain buttons
    /// emit `Click { id }`; Tabs emit `TabSelect { id, tab }`; Toggle
    /// emits `Toggle { id, checked }`; Input emits `InputFocus { id,
    /// focused: true }`.
    pub kind: ClickKind,
    /// Local to the content_root (y-down, pixels from top-left of the
    /// content area). Same frame as `PaneContentPressed.local_pt`.
    pub rect: Rect,
}

/// Discriminator on a [`ClickTarget`] that picks the outbound HostEvent.
#[derive(Clone, Debug)]
pub enum ClickKind {
    /// Plain `Click { id }`.
    Button,
    /// Tabs strip: send `TabSelect { id, tab }` with the carried tab.
    TabSelect { tab: String },
    /// Radio option: send `RadioSelect { id, option }` with the carried option.
    RadioSelect { option: String },
    /// Stepper button: send `NumberChange { id, value }` with the carried value.
    NumberChange { value: f32 },
    /// Select trigger: toggle the host-owned open-dropdown state (no widget
    /// event until an option is picked).
    SelectTrigger,
    /// Popover trigger: toggle the host-owned open-popover state.
    PopoverTrigger,
    /// Toggle: send `Toggle { id, checked: <new value> }`.
    Toggle { new_checked: bool },
    /// Input: send `InputFocus { id, focused: true }`.
    InputFocus,
}

pub struct LinkTarget {
    pub url: String,
    pub rect: Rect,
}

/// Marker attached to a widget pane while an `Element::Input` is
/// focused. Holds the input's id plus locally-edited value + caret
/// position; the host owns these so the caret can blink and typing
/// echoes immediately without waiting for the widget to round-trip a
/// new frame.
#[derive(Component, Clone, Debug)]
pub struct WidgetInputFocus {
    pub id: String,
    pub value: String,
    pub caret: usize,
    /// Per-pane time accumulator for the caret blink. Reset to 0 every
    /// keystroke so the caret stays solid while typing.
    pub blink: f32,
    /// True when the focused element is an `Element::TextArea`. Changes
    /// keyboard handling: Enter inserts a newline (submit is
    /// Cmd/Ctrl+Enter) and up/down arrows move between lines.
    pub multiline: bool,
}

impl WidgetInputFocus {
    pub fn new(id: String) -> Self {
        Self {
            id,
            value: String::new(),
            caret: 0,
            blink: 0.0,
            multiline: false,
        }
    }
}

// ============================================================
// Drag-to-select over rendered widget text (+ Cmd/Ctrl+C copy)
// ============================================================

/// Active drag-selection over a widget's rendered text. Covers a SINGLE
/// text run (the "grab this value / cell" case); cross-run or
/// rectangular selection isn't modeled. Stored fully resolved (text +
/// geometry) so the highlight survives the frame rebuilds that
/// re-render the widget.
#[derive(Component, Clone)]
pub struct WidgetTextSelection {
    text: String,
    /// The run's rect, content_root-local (y-down). `rect.min` is the
    /// glyph origin; char offsets measure left-to-right from there.
    rect: Rect,
    font_size: f32,
    /// Char indices into `text`; the selection is the half-open range
    /// [min(anchor, focus), max(anchor, focus)).
    anchor: usize,
    focus: usize,
    /// True while the mouse button is held (drag in progress).
    dragging: bool,
}

/// Marker for the transient selection-highlight sprite (rebuilt each
/// frame from the active `WidgetTextSelection`).
#[derive(Component)]
struct WidgetSelectionHighlight;

/// Nearest character boundary to local-x `x` within a run that starts at
/// `origin_x`. Measures growing prefixes through the same metrics the
/// renderer uses, so it's correct for proportional fonts too.
fn char_index_at_x(
    text: &str,
    origin_x: f32,
    font_size: f32,
    metrics: &PaneFontMetrics,
    x: f32,
) -> usize {
    let target = (x - origin_x).max(0.0);
    let chars: Vec<char> = text.chars().collect();
    let mut prefix = String::new();
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for i in 0..=chars.len() {
        let w = metrics.measure(&prefix, font_size);
        let d = (w - target).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
        if i < chars.len() {
            prefix.push(chars[i]);
        }
    }
    best
}

/// Press: begin a selection if the press landed on a selectable span.
/// Enforces a single active selection app-wide (clears the others), and
/// clears the selection when a press on a widget misses every span.
fn begin_text_selection(
    mut commands: Commands,
    mut presses: MessageReader<PaneContentPressed>,
    metrics: Res<PaneFontMetrics>,
    widgets: Query<(&WidgetTargets, Option<&WidgetScroll>)>,
    existing: Query<Entity, With<WidgetTextSelection>>,
) {
    for ev in presses.read() {
        let Ok((targets, scroll)) = widgets.get(ev.pane) else {
            continue;
        };
        let scroll_y = scroll.map(|s| s.y).unwrap_or(0.0);
        let hit = ev.local_pt + Vec2::new(0.0, scroll_y);
        let span = targets.spans.iter().find(|s| s.rect.contains(hit));
        // One selection at a time, app-wide.
        for e in &existing {
            commands.entity(e).remove::<WidgetTextSelection>();
        }
        if let Some(s) = span {
            let ci = char_index_at_x(&s.text, s.rect.min.x, s.font_size, &metrics, hit.x);
            commands.entity(ev.pane).insert(WidgetTextSelection {
                text: s.text.clone(),
                rect: s.rect,
                font_size: s.font_size,
                anchor: ci,
                focus: ci,
                dragging: true,
            });
        }
    }
}

/// Drag: extend the active selection's focus to the cursor's character.
fn update_text_selection(
    mut drags: MessageReader<pane_bevy::PaneContentDragged>,
    metrics: Res<PaneFontMetrics>,
    mut q: Query<&mut WidgetTextSelection>,
) {
    for ev in drags.read() {
        let Ok(mut sel) = q.get_mut(ev.pane) else {
            continue;
        };
        if !sel.dragging {
            continue;
        }
        sel.focus = char_index_at_x(
            &sel.text,
            sel.rect.min.x,
            sel.font_size,
            &metrics,
            ev.local_pt.x,
        );
    }
}

/// Release: end the drag. A zero-width selection (a plain click) clears.
fn end_text_selection(
    mut commands: Commands,
    mut releases: MessageReader<pane_bevy::PaneContentReleased>,
    mut q: Query<&mut WidgetTextSelection>,
) {
    for ev in releases.read() {
        if let Ok(mut sel) = q.get_mut(ev.pane) {
            sel.dragging = false;
            if sel.anchor == sel.focus {
                commands.entity(ev.pane).remove::<WidgetTextSelection>();
            }
        }
    }
}

// ============================================================
// Slider drag → SliderChange
// ============================================================

fn send_slider_change(io: &WidgetIO, id: &str, value: f32) {
    let evt = HostEvent::SliderChange {
        id: id.to_string(),
        value,
    };
    if let Ok(json) = serde_json::to_string(&evt) {
        let _ = io.tx.send(json);
    }
}

/// Press: if it landed on a slider track, set the value immediately and start
/// a drag (records the value-mapping on the pane so drag events are cheap).
fn begin_slider_drag(
    mut commands: Commands,
    mut presses: MessageReader<PaneContentPressed>,
    kinds: Query<&PaneKindMarker>,
    widgets: Query<(
        &WidgetTargets,
        Option<&WidgetIO>,
        Option<&crate::rhai_widget::RhaiWidget>,
        Option<&WidgetScroll>,
    )>,
) {
    for ev in presses.read() {
        let Ok(kind) = kinds.get(ev.pane) else { continue };
        if kind.0 != PANE_KIND {
            continue;
        }
        let Ok((targets, io, rhai, scroll)) = widgets.get(ev.pane) else {
            continue;
        };
        let scroll_y = scroll.map(|s| s.y).unwrap_or(0.0);
        let hit = ev.local_pt + Vec2::new(0.0, scroll_y);
        let Some(t) = targets.sliders.iter().find(|s| s.rect.contains(hit)) else {
            continue;
        };
        let value = t.value_at(hit.x);
        if let Some(io) = io {
            send_slider_change(io, &t.id, value);
        }
        if let Some(rhai) = rhai {
            rhai.send_slider_change(t.id.clone(), value);
        }
        commands.entity(ev.pane).insert(WidgetSliderDrag {
            target: t.clone(),
            last_value: value,
        });
    }
}

/// Drag: re-map the cursor x to a value and emit `SliderChange` on change.
/// Slider value maps over x only, so the vertical scroll offset is irrelevant.
fn update_slider_drag(
    mut drags: MessageReader<pane_bevy::PaneContentDragged>,
    mut q: Query<(
        &mut WidgetSliderDrag,
        Option<&WidgetIO>,
        Option<&crate::rhai_widget::RhaiWidget>,
    )>,
) {
    for ev in drags.read() {
        let Ok((mut drag, io, rhai)) = q.get_mut(ev.pane) else {
            continue;
        };
        let value = drag.target.value_at(ev.local_pt.x);
        if (value - drag.last_value).abs() > f32::EPSILON {
            drag.last_value = value;
            if let Some(io) = io {
                send_slider_change(io, &drag.target.id, value);
            }
            if let Some(rhai) = rhai {
                rhai.send_slider_change(drag.target.id.clone(), value);
            }
        }
    }
}

/// Release: end the drag.
fn end_slider_drag(
    mut commands: Commands,
    mut releases: MessageReader<pane_bevy::PaneContentReleased>,
    q: Query<Entity, With<WidgetSliderDrag>>,
) {
    for ev in releases.read() {
        if q.get(ev.pane).is_ok() {
            commands.entity(ev.pane).remove::<WidgetSliderDrag>();
        }
    }
}

/// Base z for floating overlay content (above any pane child z; the overlay
/// camera's order is what actually composites it above panes).
const OVERLAY_BASE_Z: f32 = 1.0;

/// Render the open `Select`'s dropdown on the floating overlay layer, anchored
/// below its trigger. Rebuilt every frame so it tracks pan/zoom/scroll. Records
/// per-item + trigger window-rects into `SelectMenuHits` for the dismiss handler.
#[allow(clippy::too_many_arguments)]
fn render_select_overlay(
    mut commands: Commands,
    open: Res<WidgetOpenSelect>,
    overlay_layer: Res<WidgetOverlayLayer>,
    pane_font: Res<PaneFont>,
    metrics: Res<PaneFontMetrics>,
    theme: Res<style_bevy::Theme>,
    fonts: Res<style_bevy::FontRegistry>,
    windows: Query<&Window>,
    viewport: Option<Res<pane_bevy::PaneViewport>>,
    panes: Query<(&PaneRect, &WidgetTargets, Option<&WidgetScroll>)>,
    existing: Query<Entity, With<WidgetOverlayRoot>>,
    mut hits: ResMut<SelectMenuHits>,
) {
    // Tear down last frame's overlay (cheap: a handful of entities).
    for e in &existing {
        commands.entity(e).try_despawn();
    }
    hits.items.clear();
    hits.select_id.clear();
    hits.pane = None;
    hits.trigger_rect = Rect::default();

    let Some(os) = open.0.as_ref() else {
        return;
    };
    let Ok((rect, targets, scroll)) = panes.get(os.pane) else {
        return;
    };
    let Some(target) = targets.selects.iter().find(|s| s.id == os.id) else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(viewport) = viewport.as_deref() else {
        return;
    };
    let (win_w, win_h) = (window.width(), window.height());
    let cursor_pos = window.cursor_position();
    let scroll_y = scroll.map(|s| s.y).unwrap_or(0.0);
    let to_world = |p: Vec2| Vec2::new(p.x - win_w * 0.5, win_h * 0.5 - p.y);

    // content-local (unscrolled) → visual canvas → window.
    let content_origin = rect.pos + Vec2::new(pane_bevy::MARGIN, pane_bevy::TITLE_H + pane_bevy::MARGIN);
    let local_to_window = |local: Vec2| {
        viewport.canvas_to_window(content_origin + Vec2::new(local.x, local.y - scroll_y))
    };
    // trigger window rect (so the dismiss handler can ignore the toggling click)
    let tr_min = local_to_window(target.anchor.min);
    let tr_max = local_to_window(target.anchor.max);
    hits.trigger_rect = Rect::from_corners(tr_min, tr_max);

    // Menu anchor: just below the trigger.
    let menu_top_window = local_to_window(Vec2::new(target.anchor.min.x, target.anchor.max.y + 4.0));
    let anchor_world = to_world(menu_top_window);

    let pad = render::SELECT_MENU_PAD;
    let item_h = render::SELECT_ITEM_H;
    let menu_w = target.width;
    let n = target.options.len();
    let menu_h = pad * 2.0 + n as f32 * item_h;
    let layer = bevy::camera::visibility::RenderLayers::layer(overlay_layer.0);

    // Overlay root at the anchor; children render in content-local (y-down,
    // negated) under it, dropping downward from the trigger.
    let root = commands
        .spawn((
            WidgetOverlayRoot,
            Transform::from_xyz(anchor_world.x, anchor_world.y, OVERLAY_BASE_Z),
            Visibility::Inherited,
            layer.clone(),
        ))
        .id();

    let octx = render::LayoutCtx {
        font: pane_font.0.clone(),
        metrics: *metrics,
        owner_pane: os.pane,
        content_root: root,
        content_size: Vec2::new(menu_w, menu_h),
        palette: render::WidgetPalette::from_theme(&theme),
        theme: theme.clone(),
        fonts: fonts.clone(),
        focused_input: None,
        caret_visible: false,
        hovered_click_id: None,
    };
    let transparent = Color::srgba(0.0, 0.0, 0.0, 0.0);

    // Menu panel: `menu` slot plan, else a default surface + border.
    if let Some(plan) = target.style.as_ref().and_then(|s| s.menu.as_ref()) {
        render::paint_style_background(&mut commands, &octx, Some(plan), Vec2::ZERO, Vec2::new(menu_w, menu_h), 0.0);
    } else {
        render::paint_rounded_panel(
            &mut commands,
            &octx,
            Vec2::ZERO,
            Vec2::new(menu_w, menu_h),
            8.0,
            octx.palette.bar_track,
            octx.palette.divider,
            1.0,
            Color::srgba(0.0, 0.0, 0.0, 0.4),
            8.0,
            2.0,
            0.0,
        );
    }

    let accent = octx
        .resolve_color("accent")
        .unwrap_or(Color::srgb(0.42, 0.62, 0.92));
    for (i, opt) in target.options.iter().enumerate() {
        let y = pad + i as f32 * item_h;
        let item_origin = Vec2::new(pad, y);
        let item_size = Vec2::new((menu_w - pad * 2.0).max(0.0), item_h);
        let is_sel = opt.id == target.value;
        // Is the cursor over this row? (item window rects match menu_top_window.)
        let item_win_top = menu_top_window.y + y;
        let is_hovered = cursor_pos.is_some_and(|c| {
            c.x >= menu_top_window.x
                && c.x <= menu_top_window.x + menu_w
                && c.y >= item_win_top
                && c.y <= item_win_top + item_h
        });
        let plan = if is_sel {
            target
                .style
                .as_ref()
                .and_then(|s| s.item_selected.as_ref().or(s.item.as_ref()))
        } else {
            target.style.as_ref().and_then(|s| s.item.as_ref())
        };
        if let Some(plan) = plan {
            render::paint_style_background(&mut commands, &octx, Some(plan), item_origin, item_size, 0.01);
        } else if is_sel {
            render::paint_rounded_panel(
                &mut commands,
                &octx,
                item_origin,
                item_size,
                4.0,
                octx.palette.divider,
                transparent,
                0.0,
                transparent,
                0.0,
                0.0,
                0.01,
            );
        }
        // Hover highlight: a soft accent wash over the hovered row (painted
        // above the base/selected fill, below the label).
        if is_hovered {
            let h = accent.to_linear();
            let wash = Color::LinearRgba(LinearRgba {
                red: h.red,
                green: h.green,
                blue: h.blue,
                alpha: 0.20,
            });
            render::paint_rounded_panel(
                &mut commands,
                &octx,
                item_origin,
                item_size,
                4.0,
                wash,
                transparent,
                0.0,
                transparent,
                0.0,
                0.0,
                0.015,
            );
        }
        let color = if is_sel || is_hovered {
            octx.palette.text
        } else {
            octx.palette.text_muted
        };
        commands.spawn((
            ChildOf(root),
            Text2d::new(opt.label.clone()),
            TextFont {
                font: pane_font.0.clone(),
                font_size: render::DEFAULT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(render::line_height(render::DEFAULT_FONT_SIZE)),
            TextColor(color),
            bevy::sprite::Anchor::CENTER_LEFT,
            bevy::text::TextLayout::new_with_no_wrap(),
            Transform::from_xyz(pad + render::SELECT_PAD_X, -(y + item_h * 0.5), 0.02),
        ));

        // window-space hit rect (overlay camera is 1:1 with the window).
        hits.items.push(SelectMenuHit {
            option_id: opt.id.clone(),
            rect: Rect::new(
                menu_top_window.x,
                menu_top_window.y + y,
                menu_top_window.x + menu_w,
                menu_top_window.y + y + item_h,
            ),
        });
    }
    hits.select_id = os.id.clone();
    hits.pane = Some(os.pane);
}

/// Stamp the overlay `RenderLayers` onto every descendant of each overlay root,
/// so paint helpers (which don't set layers themselves) render on the overlay
/// camera. Mirrors pane-bevy's content-layer reconciliation.
fn stamp_overlay_layers(
    mut commands: Commands,
    roots: Query<
        Entity,
        Or<(
            With<WidgetOverlayRoot>,
            With<WidgetTooltipRoot>,
            With<WidgetDialogRoot>,
            With<WidgetPopoverRoot>,
            With<WidgetToastRoot>,
        )>,
    >,
    children_q: Query<&Children>,
    overlay_layer: Res<WidgetOverlayLayer>,
) {
    let layer = bevy::camera::visibility::RenderLayers::layer(overlay_layer.0);
    for root in &roots {
        let mut stack = vec![root];
        while let Some(e) = stack.pop() {
            commands.entity(e).insert(layer.clone());
            if let Ok(ch) = children_q.get(e) {
                stack.extend(ch.iter());
            }
        }
    }
}

/// Dismiss / select for the open dropdown: left-click on an item picks it
/// (emits `SelectChange` + closes); a click outside the menu (and not on the
/// trigger) closes; Escape closes.
fn handle_overlay_input(
    buttons: Res<ButtonInput<bevy::input::mouse::MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    hits: Res<SelectMenuHits>,
    mut open: ResMut<WidgetOpenSelect>,
    widgets: Query<(Option<&WidgetIO>, Option<&crate::rhai_widget::RhaiWidget>)>,
) {
    if open.0.is_none() {
        return;
    }
    if keys.just_pressed(KeyCode::Escape) {
        open.0 = None;
        return;
    }
    if !buttons.just_pressed(bevy::input::mouse::MouseButton::Left) {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };
    // The click that opened the dropdown lands on the trigger — ignore it here.
    if hits.trigger_rect.contains(pt) {
        return;
    }
    if let Some(hit) = hits.items.iter().find(|h| h.rect.contains(pt)) {
        if let Some(pane) = hits.pane {
            if let Ok((io, rhai)) = widgets.get(pane) {
                let evt = HostEvent::SelectChange {
                    id: hits.select_id.clone(),
                    value: hit.option_id.clone(),
                };
                if let Some(io) = io {
                    if let Ok(json) = serde_json::to_string(&evt) {
                        let _ = io.tx.send(json);
                    }
                }
                if let Some(rhai) = rhai {
                    rhai.send_select_change(hits.select_id.clone(), hit.option_id.clone());
                }
            }
        }
        open.0 = None;
    } else {
        // Outside click → just close.
        open.0 = None;
    }
}

/// Map a click target to the host event it fires (shared by the pane press
/// handler and the overlay/dialog router). `None` for `SelectTrigger` (handled
/// specially by the press handler).
fn click_to_host_event(kind: &ClickKind, id: &str) -> Option<HostEvent> {
    Some(match kind {
        ClickKind::Button => HostEvent::Click { id: id.to_string() },
        ClickKind::TabSelect { tab } => HostEvent::TabSelect {
            id: id.to_string(),
            tab: tab.clone(),
        },
        ClickKind::RadioSelect { option } => HostEvent::RadioSelect {
            id: id.to_string(),
            option: option.clone(),
        },
        ClickKind::NumberChange { value } => HostEvent::NumberChange {
            id: id.to_string(),
            value: *value,
        },
        ClickKind::Toggle { new_checked } => HostEvent::Toggle {
            id: id.to_string(),
            checked: *new_checked,
        },
        ClickKind::InputFocus => HostEvent::InputFocus {
            id: id.to_string(),
            focused: true,
        },
        ClickKind::SelectTrigger | ClickKind::PopoverTrigger => return None,
    })
}

fn send_host_event(io: Option<&WidgetIO>, rhai: Option<&crate::rhai_widget::RhaiWidget>, evt: &HostEvent) {
    if let Some(io) = io {
        if let Ok(json) = serde_json::to_string(evt) {
            let _ = io.tx.send(json);
        }
    }
    if let Some(rhai) = rhai {
        rhai.send_host_event(evt);
    }
}

/// Render the first open `Dialog` centered on the overlay layer: a full-window
/// scrim behind a `panel` (title + arbitrary `body`). The body renders through
/// the normal `render::render` path, so its buttons are real click targets —
/// translated to window space in `WidgetOverlayHits` for `handle_dialog_input`.
#[allow(clippy::too_many_arguments)]
fn render_dialog_overlay(
    mut commands: Commands,
    overlay_layer: Res<WidgetOverlayLayer>,
    pane_font: Res<PaneFont>,
    metrics: Res<PaneFontMetrics>,
    theme: Res<style_bevy::Theme>,
    fonts: Res<style_bevy::FontRegistry>,
    windows: Query<&Window>,
    panes: Query<(Entity, &WidgetTargets), With<pane_bevy::PaneTag>>,
    existing: Query<Entity, With<WidgetDialogRoot>>,
    mut hits: ResMut<WidgetOverlayHits>,
) {
    for e in &existing {
        commands.entity(e).try_despawn();
    }
    hits.clicks.clear();
    hits.dialog_id.clear();
    hits.pane = None;
    hits.panel_rect = Rect::default();

    let Some((pane, dialog)) = panes
        .iter()
        .find_map(|(p, t)| t.dialogs.first().map(|d| (p, d.clone())))
    else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };
    let (win_w, win_h) = (window.width(), window.height());
    let to_world = |p: Vec2| Vec2::new(p.x - win_w * 0.5, win_h * 0.5 - p.y);
    let layer = bevy::camera::visibility::RenderLayers::layer(overlay_layer.0);
    let pad = 16.0;
    let panel_w = dialog.width;

    // Scrim: full-window dim, its own root, behind the panel.
    let scrim_root = commands
        .spawn((
            WidgetDialogRoot,
            Transform::from_xyz(0.0, 0.0, OVERLAY_BASE_Z),
            Visibility::Inherited,
            layer.clone(),
        ))
        .id();
    if let Some(plan) = dialog.style.as_ref().and_then(|s| s.scrim.as_ref()) {
        let sctx = render::LayoutCtx {
            font: pane_font.0.clone(),
            metrics: *metrics,
            owner_pane: pane,
            content_root: scrim_root,
            content_size: Vec2::new(win_w, win_h),
            palette: render::WidgetPalette::from_theme(&theme),
            theme: theme.clone(),
            fonts: fonts.clone(),
            focused_input: None,
            caret_visible: false,
            hovered_click_id: None,
        };
        render::paint_style_background(
            &mut commands,
            &sctx,
            Some(plan),
            Vec2::new(-win_w * 0.5, -win_h * 0.5),
            Vec2::new(win_w, win_h),
            0.0,
        );
    } else {
        commands.spawn((
            ChildOf(scrim_root),
            Sprite {
                color: Color::srgba(0.0, 0.0, 0.0, 0.55),
                custom_size: Some(Vec2::new(win_w, win_h)),
                ..default()
            },
            bevy::sprite::Anchor::CENTER,
            Transform::from_xyz(0.0, 0.0, 0.0),
            layer.clone(),
        ));
    }

    // Panel root — positioned after we measure the content.
    let panel_root = commands
        .spawn((
            WidgetDialogRoot,
            Transform::from_xyz(0.0, 0.0, OVERLAY_BASE_Z + 0.5),
            Visibility::Inherited,
            layer.clone(),
        ))
        .id();
    let pctx = render::LayoutCtx {
        font: pane_font.0.clone(),
        metrics: *metrics,
        owner_pane: pane,
        content_root: panel_root,
        content_size: Vec2::new(panel_w, win_h),
        palette: render::WidgetPalette::from_theme(&theme),
        theme: theme.clone(),
        fonts: fonts.clone(),
        focused_input: None,
        caret_visible: false,
        hovered_click_id: None,
    };

    // Build the content element: title + arbitrary body.
    use crate::protocol::{Element as E, Weight};
    let mut kids: Vec<E> = Vec::new();
    if !dialog.title.is_empty() {
        kids.push(E::Text {
            value: dialog.title.clone(),
            color: None,
            size: Some(16.0),
            weight: Some(Weight::Bold),
            family: None,
            selectable: false,
        });
    }
    if let Some(body) = &dialog.body {
        kids.push((**body).clone());
    }
    let content = E::Vstack {
        gap: 12.0,
        pad: 0.0,
        children: kids,
        style: None,
    };

    // Render the body through the normal path (so its buttons are real click
    // targets), inset by `pad`. Its targets are panel-root-local.
    let mut body_targets = WidgetTargets::default();
    let consumed = render::render(
        &mut commands,
        &pctx,
        &mut body_targets,
        &content,
        Vec2::splat(pad),
        panel_w - 2.0 * pad,
        0.02,
    );
    let panel_h = consumed.y + 2.0 * pad;

    // Center the panel; the children move with the root transform.
    let panel_tl_window = Vec2::new((win_w - panel_w) * 0.5, (win_h - panel_h) * 0.5);
    let panel_world = to_world(panel_tl_window);
    commands.entity(panel_root).insert(Transform::from_xyz(
        panel_world.x,
        panel_world.y,
        OVERLAY_BASE_Z + 0.5,
    ));

    // Panel background (behind the content, z 0.0 within the root).
    if let Some(plan) = dialog.style.as_ref().and_then(|s| s.panel.as_ref()) {
        render::paint_style_background(&mut commands, &pctx, Some(plan), Vec2::ZERO, Vec2::new(panel_w, panel_h), 0.0);
    } else {
        render::paint_rounded_panel(
            &mut commands,
            &pctx,
            Vec2::ZERO,
            Vec2::new(panel_w, panel_h),
            12.0,
            pctx.palette.bar_track,
            pctx.palette.divider,
            1.0,
            Color::srgba(0.0, 0.0, 0.0, 0.5),
            12.0,
            4.0,
            0.0,
        );
    }

    // Route the body's clicks (window-space) for handle_dialog_input.
    for ct in &body_targets.clicks {
        hits.clicks.push(OverlayClickHit {
            rect: Rect::new(
                panel_tl_window.x + ct.rect.min.x,
                panel_tl_window.y + ct.rect.min.y,
                panel_tl_window.x + ct.rect.max.x,
                panel_tl_window.y + ct.rect.max.y,
            ),
            kind: ct.kind.clone(),
            id: ct.id.clone(),
        });
    }
    hits.panel_rect = Rect::from_corners(panel_tl_window, panel_tl_window + Vec2::new(panel_w, panel_h));
    hits.dialog_id = dialog.id.clone();
    hits.pane = Some(pane);
}

/// Modal input for the open dialog: a body button fires its event; a click
/// outside the panel (scrim) or Escape sends `DialogClose`.
fn handle_dialog_input(
    buttons: Res<ButtonInput<bevy::input::mouse::MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    hits: Res<WidgetOverlayHits>,
    widgets: Query<(Option<&WidgetIO>, Option<&crate::rhai_widget::RhaiWidget>)>,
) {
    if hits.dialog_id.is_empty() {
        return;
    }
    let Some(pane) = hits.pane else {
        return;
    };
    let Ok((io, rhai)) = widgets.get(pane) else {
        return;
    };
    if keys.just_pressed(KeyCode::Escape) {
        send_host_event(io, rhai, &HostEvent::DialogClose { id: hits.dialog_id.clone() });
        return;
    }
    if !buttons.just_pressed(bevy::input::mouse::MouseButton::Left) {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };
    if let Some(hit) = hits.clicks.iter().find(|h| h.rect.contains(pt)) {
        if let Some(evt) = click_to_host_event(&hit.kind, &hit.id) {
            send_host_event(io, rhai, &evt);
        }
    } else if !hits.panel_rect.contains(pt) {
        send_host_event(io, rhai, &HostEvent::DialogClose { id: hits.dialog_id.clone() });
    }
}

/// Render the open `Popover`'s floating card on the overlay, anchored below its
/// trigger. Composes the anchored positioning (like Select) with arbitrary
/// content + click routing (like Dialog).
#[allow(clippy::too_many_arguments)]
fn render_popover_overlay(
    mut commands: Commands,
    open: Res<WidgetOpenPopover>,
    overlay_layer: Res<WidgetOverlayLayer>,
    pane_font: Res<PaneFont>,
    metrics: Res<PaneFontMetrics>,
    theme: Res<style_bevy::Theme>,
    fonts: Res<style_bevy::FontRegistry>,
    windows: Query<&Window>,
    viewport: Option<Res<pane_bevy::PaneViewport>>,
    panes: Query<(&PaneRect, &WidgetTargets, Option<&WidgetScroll>)>,
    existing: Query<Entity, With<WidgetPopoverRoot>>,
    mut hits: ResMut<PopoverHits>,
) {
    for e in &existing {
        commands.entity(e).try_despawn();
    }
    // Which content button is hovered? Resolve from last frame's hit rects (the
    // overlay rebuilds every frame, so a 1-frame lag is invisible) — the content
    // renders through `render::render`, which honors `hovered_click_id`.
    let cursor = windows.single().ok().and_then(|w| w.cursor_position());
    let hovered_id = cursor.and_then(|pt| {
        hits.clicks
            .iter()
            .find(|h| h.rect.contains(pt))
            .map(|h| h.id.clone())
    });
    hits.clicks.clear();
    hits.popover_id.clear();
    hits.pane = None;
    hits.surface_rect = Rect::default();
    hits.trigger_rect = Rect::default();

    let Some(op) = open.0.as_ref() else {
        return;
    };
    let Ok((rect, targets, scroll)) = panes.get(op.pane) else {
        return;
    };
    let Some(target) = targets.popovers.iter().find(|p| p.id == op.id) else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(viewport) = viewport.as_deref() else {
        return;
    };
    let (win_w, win_h) = (window.width(), window.height());
    let scroll_y = scroll.map(|s| s.y).unwrap_or(0.0);
    let to_world = |p: Vec2| Vec2::new(p.x - win_w * 0.5, win_h * 0.5 - p.y);
    let content_origin =
        rect.pos + Vec2::new(pane_bevy::MARGIN, pane_bevy::TITLE_H + pane_bevy::MARGIN);
    let local_to_window =
        |l: Vec2| viewport.canvas_to_window(content_origin + Vec2::new(l.x, l.y - scroll_y));
    hits.trigger_rect =
        Rect::from_corners(local_to_window(target.anchor.min), local_to_window(target.anchor.max));
    let surface_top_window =
        local_to_window(Vec2::new(target.anchor.min.x, target.anchor.max.y + 4.0));
    let anchor_world = to_world(surface_top_window);

    let pad = 12.0;
    let width = target.width;
    let layer = bevy::camera::visibility::RenderLayers::layer(overlay_layer.0);
    let root = commands
        .spawn((
            WidgetPopoverRoot,
            Transform::from_xyz(anchor_world.x, anchor_world.y, OVERLAY_BASE_Z),
            Visibility::Inherited,
            layer.clone(),
        ))
        .id();
    let pctx = render::LayoutCtx {
        font: pane_font.0.clone(),
        metrics: *metrics,
        owner_pane: op.pane,
        content_root: root,
        content_size: Vec2::new(width, win_h),
        palette: render::WidgetPalette::from_theme(&theme),
        theme: theme.clone(),
        fonts: fonts.clone(),
        focused_input: None,
        caret_visible: false,
        hovered_click_id: hovered_id,
    };
    let content = target
        .content
        .as_ref()
        .map(|c| (**c).clone())
        .unwrap_or(crate::protocol::Element::Spacer { size: 0.0 });
    let mut body_targets = WidgetTargets::default();
    let consumed = render::render(
        &mut commands,
        &pctx,
        &mut body_targets,
        &content,
        Vec2::splat(pad),
        width - 2.0 * pad,
        0.02,
    );
    let surf_h = consumed.y + 2.0 * pad;
    if let Some(plan) = target.style.as_ref().and_then(|s| s.surface.as_ref()) {
        render::paint_style_background(&mut commands, &pctx, Some(plan), Vec2::ZERO, Vec2::new(width, surf_h), 0.0);
    } else {
        render::paint_rounded_panel(
            &mut commands,
            &pctx,
            Vec2::ZERO,
            Vec2::new(width, surf_h),
            10.0,
            pctx.palette.bar_track,
            pctx.palette.divider,
            1.0,
            Color::srgba(0.0, 0.0, 0.0, 0.45),
            10.0,
            3.0,
            0.0,
        );
    }
    for ct in &body_targets.clicks {
        hits.clicks.push(OverlayClickHit {
            rect: Rect::new(
                surface_top_window.x + ct.rect.min.x,
                surface_top_window.y + ct.rect.min.y,
                surface_top_window.x + ct.rect.max.x,
                surface_top_window.y + ct.rect.max.y,
            ),
            kind: ct.kind.clone(),
            id: ct.id.clone(),
        });
    }
    hits.surface_rect =
        Rect::from_corners(surface_top_window, surface_top_window + Vec2::new(width, surf_h));
    hits.popover_id = op.id.clone();
    hits.pane = Some(op.pane);
}

/// Input for the open popover: a content button fires its event (and closes);
/// a click outside the surface (and not on the trigger) or Escape dismisses.
fn handle_popover_input(
    buttons: Res<ButtonInput<bevy::input::mouse::MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    hits: Res<PopoverHits>,
    mut open: ResMut<WidgetOpenPopover>,
    widgets: Query<(Option<&WidgetIO>, Option<&crate::rhai_widget::RhaiWidget>)>,
) {
    if open.0.is_none() {
        return;
    }
    if keys.just_pressed(KeyCode::Escape) {
        open.0 = None;
        return;
    }
    if !buttons.just_pressed(bevy::input::mouse::MouseButton::Left) {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };
    if hits.trigger_rect.contains(pt) {
        return;
    }
    if let Some(hit) = hits.clicks.iter().find(|h| h.rect.contains(pt)) {
        if let Some(pane) = hits.pane {
            if let Ok((io, rhai)) = widgets.get(pane) {
                if let Some(evt) = click_to_host_event(&hit.kind, &hit.id) {
                    send_host_event(io, rhai, &evt);
                }
            }
        }
        open.0 = None;
    } else if !hits.surface_rect.contains(pt) {
        open.0 = None;
    }
}

/// Render all `Toast`s stacked at the bottom-right window corner on the overlay
/// layer. Passive (no input); the widget owns each toast's lifecycle.
fn render_toast_overlay(
    mut commands: Commands,
    overlay_layer: Res<WidgetOverlayLayer>,
    pane_font: Res<PaneFont>,
    metrics: Res<PaneFontMetrics>,
    theme: Res<style_bevy::Theme>,
    fonts: Res<style_bevy::FontRegistry>,
    windows: Query<&Window>,
    panes: Query<(Entity, &WidgetTargets), With<pane_bevy::PaneTag>>,
    existing: Query<Entity, With<WidgetToastRoot>>,
    mut hits: ResMut<ToastHits>,
) {
    for e in &existing {
        commands.entity(e).try_despawn();
    }
    hits.items.clear();
    let toasts: Vec<(Entity, ToastTarget)> = panes
        .iter()
        .flat_map(|(p, t)| t.toasts.iter().cloned().map(move |toast| (p, toast)))
        .collect();
    if toasts.is_empty() {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let (win_w, win_h) = (window.width(), window.height());
    let to_world = |p: Vec2| Vec2::new(p.x - win_w * 0.5, win_h * 0.5 - p.y);
    let layer = bevy::camera::visibility::RenderLayers::layer(overlay_layer.0);

    let margin = 16.0;
    let pad = 12.0;
    let x_space = 22.0; // reserved on the right for the × dismiss glyph
    let th = render::line_height(render::DEFAULT_FONT_SIZE) + 2.0 * pad;
    let gap = 8.0;
    let mut y_from_bottom = margin;
    for (pane, toast) in toasts {
        // Size the toast to its text (so it never overflows), clamped to a sane
        // range; the text is *also* hard-bounded below as a belt-and-suspenders.
        let tw = metrics.measure(&toast.text, render::DEFAULT_FONT_SIZE);
        let toast_w = (pad + tw + x_space + pad).clamp(140.0, 360.0);
        let text_region = (toast_w - pad - x_space).max(0.0);
        let tl_window = Vec2::new(win_w - margin - toast_w, win_h - y_from_bottom - th);
        // The whole toast is a dismiss target.
        hits.items.push((
            Rect::from_corners(tl_window, tl_window + Vec2::new(toast_w, th)),
            toast.id.clone(),
            pane,
        ));
        let world = to_world(tl_window);
        let root = commands
            .spawn((
                WidgetToastRoot,
                Transform::from_xyz(world.x, world.y, OVERLAY_BASE_Z + 1.0),
                Visibility::Inherited,
                layer.clone(),
            ))
            .id();
        let octx = render::LayoutCtx {
            font: pane_font.0.clone(),
            metrics: *metrics,
            owner_pane: pane,
            content_root: root,
            content_size: Vec2::new(toast_w, th),
            palette: render::WidgetPalette::from_theme(&theme),
            theme: theme.clone(),
            fonts: fonts.clone(),
            focused_input: None,
            caret_visible: false,
            hovered_click_id: None,
        };
        if let Some(plan) = toast.style.as_ref().and_then(|s| s.surface.as_ref()) {
            render::paint_style_background(&mut commands, &octx, Some(plan), Vec2::ZERO, Vec2::new(toast_w, th), 0.0);
        } else {
            render::paint_rounded_panel(
                &mut commands,
                &octx,
                Vec2::ZERO,
                Vec2::new(toast_w, th),
                8.0,
                octx.palette.bar_track,
                octx.palette.divider,
                1.0,
                Color::srgba(0.0, 0.0, 0.0, 0.45),
                8.0,
                3.0,
                0.0,
            );
        }
        commands.spawn((
            ChildOf(root),
            Text2d::new(toast.text.clone()),
            TextFont {
                font: pane_font.0.clone(),
                font_size: render::DEFAULT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(render::line_height(render::DEFAULT_FONT_SIZE)),
            TextColor(octx.palette.text),
            bevy::sprite::Anchor::CENTER_LEFT,
            bevy::text::TextLayout::new_with_no_wrap(),
            // Hard cap on width so the label can never spill past the toast
            // (minus the × area). Width-only keeps the CENTER_LEFT vertical
            // centering intact (a height bound would top-align the text).
            bevy::text::TextBounds {
                width: Some(text_region),
                height: None,
            },
            Transform::from_xyz(pad, -(th * 0.5), 0.02),
        ));
        // Dismiss hint (the whole toast is clickable).
        commands.spawn((
            ChildOf(root),
            Text2d::new("\u{00d7}"),
            TextFont {
                font: pane_font.0.clone(),
                font_size: render::DEFAULT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(render::line_height(render::DEFAULT_FONT_SIZE)),
            TextColor(octx.palette.text_muted),
            bevy::sprite::Anchor::CENTER_RIGHT,
            bevy::text::TextLayout::new_with_no_wrap(),
            Transform::from_xyz(toast_w - pad, -(th * 0.5), 0.02),
        ));
        y_from_bottom += th + gap;
    }
}

/// While a floating overlay is open, own the cursor over it: a pointer over a
/// clickable region, the default arrow over a non-clickable surface. Runs in
/// PostUpdate so it wins over pane-bevy's resize-edge cursor (which doesn't know
/// the overlay is floating above the pane).
fn override_overlay_cursor(
    mut commands: Commands,
    windows: Query<(Entity, &Window)>,
    popover: Res<PopoverHits>,
    select: Res<SelectMenuHits>,
    dialog: Res<WidgetOverlayHits>,
    toast: Res<ToastHits>,
) {
    let Ok((win, window)) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };
    use bevy::window::{CursorIcon, SystemCursorIcon as I};
    let icon = if !dialog.dialog_id.is_empty() {
        // Modal: pointer over a body button, plain arrow everywhere else.
        Some(if dialog.clicks.iter().any(|h| h.rect.contains(pt)) {
            I::Pointer
        } else {
            I::Default
        })
    } else if popover.pane.is_some()
        && (popover.surface_rect.contains(pt) || popover.trigger_rect.contains(pt))
    {
        Some(if popover.clicks.iter().any(|h| h.rect.contains(pt)) {
            I::Pointer
        } else {
            I::Default
        })
    } else if !select.select_id.is_empty() && select.items.iter().any(|h| h.rect.contains(pt)) {
        Some(I::Pointer)
    } else if toast.items.iter().any(|(r, _, _)| r.contains(pt)) {
        Some(I::Pointer)
    } else {
        None
    };
    if let Some(ic) = icon {
        commands.entity(win).insert(CursorIcon::System(ic));
    }
}

/// Click a toast to dismiss it (raw mouse, like the other overlay handlers).
fn handle_toast_input(
    buttons: Res<ButtonInput<bevy::input::mouse::MouseButton>>,
    windows: Query<&Window>,
    hits: Res<ToastHits>,
    widgets: Query<(Option<&WidgetIO>, Option<&crate::rhai_widget::RhaiWidget>)>,
) {
    if hits.items.is_empty() || !buttons.just_pressed(bevy::input::mouse::MouseButton::Left) {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };
    if let Some((_, id, pane)) = hits.items.iter().find(|(rect, _, _)| rect.contains(pt)) {
        if let Ok((io, rhai)) = widgets.get(*pane) {
            send_host_event(io, rhai, &HostEvent::ToastDismiss { id: id.clone() });
        }
    }
}

/// Resolve which `Tooltip` the cursor is over (per-frame; the topmost pane under
/// the cursor wins). The result drives `render_tooltip_overlay`.
fn update_tooltip_hover(
    windows: Query<&Window>,
    viewport: Option<Res<pane_bevy::PaneViewport>>,
    panes: Query<
        (
            Entity,
            &PaneRect,
            Option<&Visibility>,
            &WidgetTargets,
            Option<&WidgetScroll>,
        ),
        With<pane_bevy::PaneTag>,
    >,
    mut active: ResMut<ActiveTooltip>,
) {
    // Only recompute when there's a real cursor (so a programmatically-forced
    // tooltip in a headless/no-cursor host isn't clobbered).
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };
    let Some(viewport) = viewport.as_deref() else {
        return;
    };
    active.0 = None;
    let pt_canvas = viewport.window_to_canvas(pt);
    let candidates: Vec<(Entity, PaneRect)> = panes
        .iter()
        .filter(|(_, _, vis, _, _)| !matches!(vis, Some(Visibility::Hidden)))
        .map(|(e, r, _, _, _)| (e, *r))
        .collect();
    let Some(pane) = pane_bevy::topmost_pane_at(pt_canvas, &candidates) else {
        return;
    };
    let Ok((_, rect, _, targets, scroll)) = panes.get(pane) else {
        return;
    };
    let scroll_y = scroll.map(|s| s.y).unwrap_or(0.0);
    let local = pane_bevy::pt_to_content_local(pt_canvas, rect) + Vec2::new(0.0, scroll_y);
    if let Some(t) = targets.tooltips.iter().find(|t| t.anchor.contains(local)) {
        active.0 = Some(ActiveTip {
            pane,
            anchor: t.anchor,
            text: t.text.clone(),
            style: t.style.clone(),
        });
    }
}

/// Render the active tooltip's bubble on the overlay layer, just below its
/// anchor. Reuses the overlay layer + stamp; no input (tooltips are passive).
#[allow(clippy::too_many_arguments)]
fn render_tooltip_overlay(
    mut commands: Commands,
    active: Res<ActiveTooltip>,
    overlay_layer: Res<WidgetOverlayLayer>,
    pane_font: Res<PaneFont>,
    metrics: Res<PaneFontMetrics>,
    theme: Res<style_bevy::Theme>,
    fonts: Res<style_bevy::FontRegistry>,
    windows: Query<&Window>,
    viewport: Option<Res<pane_bevy::PaneViewport>>,
    panes: Query<(&PaneRect, Option<&WidgetScroll>)>,
    existing: Query<Entity, With<WidgetTooltipRoot>>,
) {
    for e in &existing {
        commands.entity(e).try_despawn();
    }
    let Some(tip) = active.0.as_ref() else {
        return;
    };
    let Ok((rect, scroll)) = panes.get(tip.pane) else {
        return;
    };
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(viewport) = viewport.as_deref() else {
        return;
    };
    let (win_w, win_h) = (window.width(), window.height());
    let scroll_y = scroll.map(|s| s.y).unwrap_or(0.0);
    let to_world = |p: Vec2| Vec2::new(p.x - win_w * 0.5, win_h * 0.5 - p.y);
    let content_origin =
        rect.pos + Vec2::new(pane_bevy::MARGIN, pane_bevy::TITLE_H + pane_bevy::MARGIN);
    let bubble_top_window = viewport.canvas_to_window(
        content_origin + Vec2::new(tip.anchor.min.x, tip.anchor.max.y - scroll_y + 6.0),
    );
    let anchor_world = to_world(bubble_top_window);

    let pad = 8.0;
    let tw = metrics.measure(&tip.text, render::DEFAULT_FONT_SIZE);
    let bubble_w = tw + pad * 2.0;
    let bubble_h = render::line_height(render::DEFAULT_FONT_SIZE) + pad * 2.0;
    let layer = bevy::camera::visibility::RenderLayers::layer(overlay_layer.0);

    let root = commands
        .spawn((
            WidgetTooltipRoot,
            Transform::from_xyz(anchor_world.x, anchor_world.y, OVERLAY_BASE_Z),
            Visibility::Inherited,
            layer.clone(),
        ))
        .id();
    let octx = render::LayoutCtx {
        font: pane_font.0.clone(),
        metrics: *metrics,
        owner_pane: tip.pane,
        content_root: root,
        content_size: Vec2::new(bubble_w, bubble_h),
        palette: render::WidgetPalette::from_theme(&theme),
        theme: theme.clone(),
        fonts: fonts.clone(),
        focused_input: None,
        caret_visible: false,
        hovered_click_id: None,
    };
    if let Some(plan) = tip.style.as_ref().and_then(|s| s.bubble.as_ref()) {
        render::paint_style_background(&mut commands, &octx, Some(plan), Vec2::ZERO, Vec2::new(bubble_w, bubble_h), 0.0);
    } else {
        render::paint_rounded_panel(
            &mut commands,
            &octx,
            Vec2::ZERO,
            Vec2::new(bubble_w, bubble_h),
            6.0,
            Color::srgb(0.08, 0.09, 0.11),
            octx.palette.divider,
            1.0,
            Color::srgba(0.0, 0.0, 0.0, 0.4),
            6.0,
            2.0,
            0.0,
        );
    }
    commands.spawn((
        ChildOf(root),
        Text2d::new(tip.text.clone()),
        TextFont {
            font: pane_font.0.clone(),
            font_size: render::DEFAULT_FONT_SIZE,
            ..default()
        },
        LineHeight::Px(render::line_height(render::DEFAULT_FONT_SIZE)),
        TextColor(octx.palette.text),
        bevy::sprite::Anchor::CENTER_LEFT,
        bevy::text::TextLayout::new_with_no_wrap(),
        Transform::from_xyz(pad, -(bubble_h * 0.5), 0.02),
    ));
}

/// Repaint the selection band each frame (cheap: one run, one sprite).
/// Drawn as a translucent accent overlay on top of the glyphs — flow
/// text z varies by nesting depth, so a reliable "behind" layer isn't
/// available; a low-alpha tint reads as a selection and never z-fights
/// the backgrounds.
fn render_text_selection_highlight(
    mut commands: Commands,
    metrics: Res<PaneFontMetrics>,
    theme: Res<style_bevy::Theme>,
    sels: Query<(&WidgetTextSelection, &pane_bevy::PaneChrome)>,
    old: Query<Entity, With<WidgetSelectionHighlight>>,
) {
    for e in &old {
        commands.entity(e).try_despawn();
    }
    let base = theme.color(style_bevy::tokens::ACCENT);
    let color = Color::LinearRgba(LinearRgba {
        alpha: 0.32,
        ..base
    });
    for (sel, chrome) in &sels {
        let (a, b) = (sel.anchor.min(sel.focus), sel.anchor.max(sel.focus));
        if a == b {
            continue;
        }
        let chars: Vec<char> = sel.text.chars().collect();
        let pre: String = chars[..a.min(chars.len())].iter().collect();
        let mid: String = chars[a.min(chars.len())..b.min(chars.len())]
            .iter()
            .collect();
        let x0 = sel.rect.min.x + metrics.measure(&pre, sel.font_size);
        let w = metrics.measure(&mid, sel.font_size);
        if w <= 0.0 {
            continue;
        }
        commands.spawn((
            WidgetSelectionHighlight,
            ChildOf(chrome.content_root),
            Sprite {
                color,
                custom_size: Some(Vec2::new(w, sel.rect.height())),
                ..default()
            },
            Anchor::TOP_LEFT,
            pane_bevy::PaneContentNoClip,
            Transform::from_xyz(x0, -sel.rect.min.y, 30.0),
        ));
    }
}

/// Cmd/Ctrl+C copies the active selection's substring to the clipboard.
fn copy_text_selection(keys: Res<ButtonInput<KeyCode>>, sels: Query<&WidgetTextSelection>) {
    let mod_down = keys.pressed(KeyCode::SuperLeft)
        || keys.pressed(KeyCode::SuperRight)
        || keys.pressed(KeyCode::ControlLeft)
        || keys.pressed(KeyCode::ControlRight);
    if !(mod_down && keys.just_pressed(KeyCode::KeyC)) {
        return;
    }
    for sel in &sels {
        let (a, b) = (sel.anchor.min(sel.focus), sel.anchor.max(sel.focus));
        if a == b {
            continue;
        }
        let s: String = sel.text.chars().skip(a).take(b - a).collect();
        crate::subprocess::clipboard_set(&s);
        return;
    }
}

/// Cached `content_root` entity so render systems don't have to walk
/// the pane chrome to find it.
#[derive(Component)]
pub struct WidgetContentRoot(pub Entity);

/// Set by any system that just spawned new sprites under a widget's
/// content_root (e.g. `rerender_widgets`). Consumed by
/// `clip_widget_sprites` so it knows to do a sweep this frame even
/// when no `PaneRect` changed.
#[derive(Resource, Default)]
pub struct WidgetClipDirty(pub bool);

/// Process-wide cache of images loaded from filesystem paths. Keyed by
/// `(absolute path, optional tile coords)` so the same PNG referenced
/// by N widgets only pays the decode + GPU upload once. Slicing a
/// sheet into tiles is also cached per (path, tile_w, tile_h, col, row).
#[derive(Resource, Default)]
pub struct WidgetImageCache {
    pub by_path: HashMap<PathBuf, Handle<Image>>,
    pub tiles: HashMap<TileKey, Handle<Image>>,
}

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct TileKey {
    pub path: PathBuf,
    pub tile_w: u32,
    pub tile_h: u32,
    pub col: u32,
    pub row: u32,
}

// ---------- Plugin / registry ----------

pub struct WidgetPlugin;

impl Plugin for WidgetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WidgetClipDirty>()
            .init_resource::<WidgetImageCache>()
            .init_resource::<WidgetOverlayLayer>()
            .init_resource::<WidgetOpenSelect>()
            .init_resource::<SelectMenuHits>()
            .init_resource::<ActiveTooltip>()
            .init_resource::<WidgetOverlayHits>()
            .init_resource::<WidgetOpenPopover>()
            .init_resource::<PopoverHits>()
            .init_resource::<ToastHits>()
            .add_plugins(WidgetButtonMaterialPlugin)
            .add_plugins(glaze_material::GlazeMaterialPlugin)
            .add_plugins(msgbus::WidgetMsgBusPlugin)
            .add_systems(Startup, register_kind)
            .add_systems(
                Update,
                (
                    tick_widget_io,
                    forward_claude_events,
                    forward_ticks,
                    // Hover runs BEFORE rerender so a hover-change this
                    // frame is reflected in the same-frame redraw.
                    update_widget_hover,
                    rerender_widgets,
                    blur_inputs_on_focus_change,
                    handle_widget_press,
                    begin_text_selection,
                    update_text_selection,
                    end_text_selection,
                    (begin_slider_drag, update_slider_drag, end_slider_drag).chain(),
                    // Floating overlays: render the open dropdown + handle its
                    // input, then resolve + render the hovered tooltip. After the
                    // press handler that toggles the open state.
                    (
                        render_select_overlay,
                        handle_overlay_input,
                        update_tooltip_hover,
                        render_tooltip_overlay,
                        render_dialog_overlay,
                        handle_dialog_input,
                        render_popover_overlay,
                        handle_popover_input,
                        render_toast_overlay,
                        handle_toast_input,
                    )
                        .chain(),
                    render_text_selection_highlight,
                    copy_text_selection,
                    handle_widget_input_typing,
                    poll_widget_children,
                    handle_widget_wheel,
                    apply_widget_scroll,
                    update_widget_hot_zones,
                )
                    .chain(),
            )
            .add_systems(
                PostUpdate,
                stamp_overlay_layers
                    .before(bevy::camera::visibility::VisibilitySystems::CheckVisibility),
            )
            .add_systems(PostUpdate, (clip_widget_sprites, override_overlay_cursor));
    }
}

/// Update each widget pane's `WidgetHover` based on the cursor position
/// + that pane's `ClickTarget` rects. Event-driven: only when the
/// hovered id actually changes do we flip `WidgetRender.force_render`
/// so `rerender_widgets` redraws with the new hover state. Also nudges
/// the system cursor to a pointer when over any clickable.
fn update_widget_hover(
    mut commands: Commands,
    windows: Query<(Entity, &Window)>,
    viewport: Res<pane_bevy::PaneViewport>,
    all_panes: Query<(Entity, &pane_bevy::PaneRect, Option<&Visibility>), With<pane_bevy::PaneTag>>,
    targets: Query<&WidgetTargets>,
    mut widgets: Query<
        (
            Entity,
            &pane_bevy::PaneKindMarker,
            &mut WidgetHover,
            &mut WidgetRender,
            Option<&WidgetScroll>,
        ),
        With<pane_bevy::PaneTag>,
    >,
) {
    let Ok((win_entity, window)) = windows.single() else {
        return;
    };
    let cursor = window.cursor_position();

    // Topmost pane under the cursor (any kind). Hover is only valid
    // when the topmost pane is a widget; otherwise something else
    // (terminal, editor) is on top and the widget isn't really hovered.
    let candidates: Vec<(Entity, pane_bevy::PaneRect)> = all_panes
        .iter()
        .filter(|(_, _, vis)| !matches!(vis, Some(Visibility::Hidden)))
        .map(|(e, r, _)| (e, *r))
        .collect();
    // PaneRect is canvas-space; project the cursor before hit-testing.
    let topmost = cursor.and_then(|pt| {
        let pt_canvas = viewport.window_to_canvas(pt);
        pane_bevy::topmost_pane_at(pt_canvas, &candidates).map(|e| (pt_canvas, e))
    });

    let mut want_pointer = false;
    for (pane, kind, mut hover, mut render_state, scroll) in &mut widgets {
        let is_widget_kind = kind.0 == PANE_KIND || kind.0 == rhai_widget::PANE_KIND;
        if !is_widget_kind {
            continue;
        }
        // Find this pane's rect (we already have it in `candidates`,
        // but cheaper to re-query than threading it through).
        let pane_rect = candidates.iter().find(|(e, _)| *e == pane).map(|(_, r)| *r);
        let scroll_y = scroll.map(|s| s.y).unwrap_or(0.0);
        let new_id: Option<String> = match (topmost, pane_rect, cursor) {
            (Some((pt, top)), Some(rect), Some(_)) if top == pane => {
                // Click rects are content-local (unscrolled); add the scroll
                // offset to the cursor so hover matches the visible target —
                // same correction the press handler applies.
                let local = pane_bevy::pt_to_content_local(pt, &rect) + Vec2::new(0.0, scroll_y);
                targets.get(pane).ok().and_then(|t| {
                    t.clicks
                        .iter()
                        .find(|c| c.rect.contains(local))
                        .map(|c| c.id.clone())
                })
            }
            _ => None,
        };
        if new_id.is_some() && topmost.map(|(_, top)| top == pane).unwrap_or(false) {
            want_pointer = true;
        }
        if hover.click_id != new_id {
            hover.click_id = new_id;
            render_state.force_render = true;
        }
    }
    use bevy::window::{CursorIcon, SystemCursorIcon};
    if want_pointer {
        commands
            .entity(win_entity)
            .insert(CursorIcon::System(SystemCursorIcon::Pointer));
    }
    // Note: we don't clear the cursor here on no-hover; the pane-bevy
    // `update_pane_cursor` system already restores the default cursor
    // when nothing else owns it.
}

/// Read mouse wheel events, route to whichever widget pane (any kind:
/// the protocol-driven `widget` or the `rhai_widget`) is topmost
/// under the cursor, and update its `WidgetScroll.y` clamped to
/// `[0, max_y]`. Lines are converted to pixels via `LINE_PX` since we
/// don't carry per-widget font metrics here.
fn handle_widget_wheel(
    mut wheel: MessageReader<bevy::input::mouse::MouseWheel>,
    windows: Query<&Window>,
    keys: Res<ButtonInput<KeyCode>>,
    viewport: Res<pane_bevy::PaneViewport>,
    all_panes: Query<(Entity, &pane_bevy::PaneRect, Option<&Visibility>), With<pane_bevy::PaneTag>>,
    mut widgets: Query<
        (Entity, &mut WidgetScroll, &pane_bevy::PaneKindMarker),
        With<pane_bevy::PaneTag>,
    >,
) {
    // Cmd+scroll is canvas pan in the host; don't double-scroll widgets.
    if keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight) {
        wheel.clear();
        return;
    }
    use bevy::input::mouse::MouseScrollUnit;
    const LINE_PX: f32 = 16.0;
    let mut dy_px = 0.0_f32;
    for ev in wheel.read() {
        let uy = match ev.unit {
            MouseScrollUnit::Pixel => ev.y,
            MouseScrollUnit::Line => ev.y * LINE_PX,
        };
        dy_px += uy;
    }
    if dy_px == 0.0 {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Some(pt) = win.cursor_position() else {
        return;
    };

    // Topmost pane of ANY kind under the cursor. If the topmost isn't
    // a widget, the wheel belongs to whatever's on top (editor scroll,
    // terminal, etc.) — don't blindly scroll a widget pane underneath.
    let candidates: Vec<(Entity, pane_bevy::PaneRect)> = all_panes
        .iter()
        .filter(|(_, _, vis)| !matches!(vis, Some(Visibility::Hidden)))
        .map(|(e, r, _)| (e, *r))
        .collect();
    let Some(target) = pane_bevy::topmost_pane_at(viewport.window_to_canvas(pt), &candidates)
    else {
        return;
    };

    if let Ok((_, mut scroll, kind)) = widgets.get_mut(target) {
        if kind.0 != PANE_KIND && kind.0 != rhai_widget::PANE_KIND {
            return;
        }
        let new_y = (scroll.y - dy_px).clamp(0.0, scroll.max_y);
        if (new_y - scroll.y).abs() > 0.001 {
            scroll.y = new_y;
        }
    }
}

/// Apply the per-pane scroll offset to `content_root.transform.y` and
/// hide any direct children whose translation falls into the title-bar
/// region after the shift. Without the visibility pass, scrolled
/// content paints over the pane title; the per-pane camera viewport
/// covers the full pane rect so there's no GPU-side clip line.
fn apply_widget_scroll(
    widgets: Query<
        (
            &WidgetScroll,
            &pane_bevy::PaneChrome,
            &pane_bevy::PaneKindMarker,
        ),
        (
            With<pane_bevy::PaneTag>,
            Or<(Changed<WidgetScroll>, Changed<pane_bevy::PaneChrome>)>,
        ),
    >,
    mut t_q: Query<&mut Transform>,
) {
    use pane_bevy::{MARGIN, TITLE_H};
    for (scroll, chrome, kind) in &widgets {
        if kind.0 != PANE_KIND && kind.0 != rhai_widget::PANE_KIND {
            continue;
        }
        if let Ok(mut t) = t_q.get_mut(chrome.content_root) {
            let want = -(TITLE_H + MARGIN) + scroll.y;
            if t.translation.y != want {
                t.translation.y = want;
            }
        }
        // Clipping (hiding children that scroll into the title bar)
        // was setting Visibility::Hidden on entities it shouldn't —
        // disabled until I have a precise repro. Scrolled content may
        // visually overlap the title bar, but that's a much smaller
        // problem than panes vanishing on click.
    }
}

/// Mirror every Claude bus event onto every initialized widget's
/// stdin. Widgets that don't care can ignore by `kind`; the wire cost
/// is negligible (one short NDJSON line per event per widget).
///
/// Gated on `WidgetRender.init_sent` so a freshly-spawned widget
/// doesn't receive bus events before it has read its `init` line.
/// Events that arrive during that startup window are dropped — they're
/// available via `~/.claude/events.jsonl` if the widget needs history.
fn forward_claude_events(
    mut events: MessageReader<ClaudeBusEvent>,
    widgets: Query<(&PaneKindMarker, &WidgetRender, &WidgetIO)>,
) {
    // Materialize once so every widget sees every event (MessageReader
    // hands each event out exactly once per read site).
    let mut lines: Vec<String> = Vec::new();
    for ev in events.read() {
        let payload: serde_json::Value =
            serde_json::from_str(&ev.payload_json).unwrap_or(serde_json::Value::Null);
        let host_ev = HostEvent::ClaudeEvent {
            kind: ev.kind.clone(),
            payload,
        };
        if let Ok(json) = serde_json::to_string(&host_ev) {
            lines.push(json);
        }
    }
    if lines.is_empty() {
        return;
    }
    for (kind, render_state, io) in &widgets {
        if kind.0 != PANE_KIND || !render_state.init_sent {
            continue;
        }
        for line in &lines {
            let _ = io.tx.send(line.clone());
        }
    }
}

/// Per-frame heartbeat forwarded to every initialized widget. Rate-
/// limited to ~30Hz so a 120fps host doesn't flood subprocess stdin.
/// Each widget tracks its own `last_tick_secs` so a widget that
/// initialized later still gets the first tick at a sensible boundary.
fn forward_ticks(
    time: Res<Time>,
    mut widgets: Query<(&PaneKindMarker, &mut WidgetRender, &WidgetIO)>,
) {
    const TICK_INTERVAL_SECS: f32 = 1.0 / 30.0;
    let now = time.elapsed_secs();
    for (kind, mut render_state, io) in &mut widgets {
        if kind.0 != PANE_KIND || !render_state.init_sent {
            continue;
        }
        let last = render_state.last_tick_secs;
        let dt = now - last;
        if last > 0.0 && dt < TICK_INTERVAL_SECS {
            continue;
        }
        render_state.last_tick_secs = now;
        // First-ever tick: dt=0 so widgets don't see a huge initial dt
        // (init happened on frame N; this tick is frame N+1 at most).
        let dt_send = if last == 0.0 { 0.0 } else { dt };
        let tick = HostEvent::Tick { dt: dt_send };
        if let Ok(json) = serde_json::to_string(&tick) {
            let _ = io.tx.send(json);
        }
    }
}

/// Walks every widget pane's content_root subtree and clamps every
/// Sprite's `custom_size` so it can't escape the pane's content area.
///
/// Note: visible clipping is now handled at the renderer by
/// pane-bevy's per-pane camera viewports (see pane-bevy's top-of-file
/// docs). This system still runs because it bounds the sprites'
/// LAYOUT size — widget sprites are used for backgrounds, borders,
/// and click targets, and a sprite whose `custom_size` extends past
/// the pane edge would have its click target leak across pane
/// boundaries even though the visible portion is clipped. Keeping
/// custom_size honest avoids that mismatch.
fn clip_widget_sprites(
    panes: Query<(&PaneKindMarker, &PaneRect, &WidgetContentRoot)>,
    changed_panes: Query<(), (With<PaneKindMarker>, Changed<PaneRect>)>,
    mut needs_clip: ResMut<WidgetClipDirty>,
    children_q: Query<&Children>,
    transforms: Query<&Transform>,
    mut sprites: Query<&mut Sprite>,
) {
    // Same idle-fast-path as pane_bevy::enforce_pane_content_bounds:
    // walking every widget subtree every frame to clamp sizes that
    // haven't changed is pure waste. Re-walk only when a pane just
    // resized or `rerender_widgets` (or edit-mode) just spawned new
    // sprites under a content_root. Use a `ResMut<WidgetClipDirty>`
    // signal for the latter — we can't ask for `Added<Sprite>` here
    // because that conflicts with `&mut Sprite`.
    let new_content = needs_clip.0;
    needs_clip.0 = false;
    if changed_panes.is_empty() && !new_content {
        return;
    }
    for (kind, rect, root) in &panes {
        if kind.0 != PANE_KIND {
            continue;
        }
        let content_w = (rect.size.x - 2.0 * MARGIN).max(0.0);
        let content_h = (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0);

        // Walk subtree depth-first. `offset` accumulates Bevy local
        // translations from `content_root` outward: x is right, y is up
        // (negative y = down inside the pane).
        let mut stack: Vec<(Entity, Vec2)> = Vec::with_capacity(16);
        if let Ok(children) = children_q.get(root.0) {
            for c in children.iter() {
                let t = transforms
                    .get(c)
                    .map(|t| Vec2::new(t.translation.x, t.translation.y))
                    .unwrap_or(Vec2::ZERO);
                stack.push((c, t));
            }
        }

        while let Some((entity, offset)) = stack.pop() {
            // top_offset is "distance below content_root top, in px".
            let top_offset = (-offset.y).max(0.0);
            let left_offset = offset.x.max(0.0);
            let avail_w = (content_w - left_offset).max(0.0);
            let avail_h = (content_h - top_offset).max(0.0);

            if let Ok(mut sprite) = sprites.get_mut(entity) {
                if let Some(want) = sprite.custom_size {
                    let new = Vec2::new(want.x.min(avail_w), want.y.min(avail_h));
                    if (new.x - want.x).abs() > f32::EPSILON
                        || (new.y - want.y).abs() > f32::EPSILON
                    {
                        sprite.custom_size = Some(new);
                    }
                }
            }

            if let Ok(children) = children_q.get(entity) {
                for child in children.iter() {
                    let ct = transforms
                        .get(child)
                        .map(|t| Vec2::new(t.translation.x, t.translation.y))
                        .unwrap_or(Vec2::ZERO);
                    stack.push((child, offset + ct));
                }
            }
        }
    }
}

fn register_kind(mut registry: ResMut<PaneRegistry>) {
    registry.register(PaneKindSpec {
        kind: PANE_KIND,
        display_name: "Widget",
        radial_icon: Some("◫"),
        default_size: Vec2::new(360.0, 240.0),
        spawn: widget_spawn,
        snapshot: widget_snapshot,
        on_close: Some(widget_on_close),
    });
}

// ---------- Spawn ----------

fn widget_spawn(world: &mut World, entity: Entity, content_root: Entity, config: &Value) {
    let title = config
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Widget")
        .to_string();
    let command = config
        .get("command")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| std::env::var(DEFAULT_CMD_ENV).ok())
        .unwrap_or_default();
    let args: Vec<String> = config
        .get("args")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let cwd = config
        .get("cwd")
        .and_then(|v| v.as_str())
        .map(PathBuf::from);
    let state = config.get("state").cloned().unwrap_or(Value::Null);

    if let Some(mut t) = world.get_mut::<PaneTitle>(entity) {
        t.0 = title.clone();
    } else {
        world.entity_mut(entity).insert(PaneTitle(title));
    }

    let widget = Widget {
        command: command.clone(),
        args: args.clone(),
        cwd: cwd.clone(),
        last_state: state.clone(),
    };
    let mut render_state = WidgetRender::default();
    let targets = WidgetTargets::default();

    if command.trim().is_empty() {
        render_state.current_frame = Some(placeholder_frame());
        world.entity_mut(entity).insert((
            widget,
            render_state,
            targets,
            WidgetContentRoot(content_root),
            WidgetScroll::default(),
            WidgetHover::default(),
        ));
        return;
    }

    match spawn_widget_process(&command, &args, cwd.as_deref()) {
        Ok((process, io)) => {
            world.entity_mut(entity).insert((
                widget,
                render_state,
                targets,
                WidgetContentRoot(content_root),
                WidgetScroll::default(),
                WidgetHover::default(),
                process,
                io,
            ));
        }
        Err(e) => {
            eprintln!("[widget] spawn failed: {}", e);
            render_state.current_frame = Some(error_frame(&format!("spawn failed: {}", e)));
            world.entity_mut(entity).insert((
                widget,
                render_state,
                targets,
                WidgetContentRoot(content_root),
                WidgetScroll::default(),
                WidgetHover::default(),
            ));
        }
    }
}

pub fn spawn_widget_process(
    cmd: &str,
    args: &[String],
    cwd: Option<&Path>,
) -> std::io::Result<(WidgetProcess, WidgetIO)> {
    let mut command = if args.is_empty() {
        let mut c = Command::new("sh");
        c.arg("-c").arg(cmd);
        c
    } else {
        let mut c = Command::new(cmd);
        c.args(args);
        c
    };
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(p) = cwd {
        command.current_dir(p);
    }
    let mut child = command.spawn()?;
    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");
    let stdin = child.stdin.take().expect("stdin was piped");

    let (msg_tx, msg_rx) = mpsc::channel::<WidgetMsg>();
    let (line_tx, line_rx) = mpsc::channel::<String>();

    // Stdout: parse NDJSON, forward parsed messages.
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(s) = line else { return };
            if s.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<WidgetMsg>(&s) {
                Ok(m) => {
                    if msg_tx.send(m).is_err() {
                        return;
                    }
                }
                Err(e) => eprintln!("[widget] parse error: {} | line: {}", e, s),
            }
        }
    });

    // Stderr: log only — useful for `set -x` style debugging.
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(s) = line {
                eprintln!("[widget stderr] {}", s);
            }
        }
    });

    // Stdin writer thread — receives already-serialized JSON lines.
    let mut stdin_writer = stdin;
    std::thread::spawn(move || {
        while let Ok(line) = line_rx.recv() {
            if stdin_writer.write_all(line.as_bytes()).is_err() {
                return;
            }
            if stdin_writer.write_all(b"\n").is_err() {
                return;
            }
            if stdin_writer.flush().is_err() {
                return;
            }
        }
    });

    Ok((
        WidgetProcess { child },
        WidgetIO {
            rx: Mutex::new(msg_rx),
            tx: line_tx,
        },
    ))
}

// ---------- Snapshot / close ----------

fn widget_snapshot(world: &World, entity: Entity) -> Value {
    let Some(w) = world.get::<Widget>(entity) else {
        return Value::Null;
    };
    let title = world
        .get::<PaneTitle>(entity)
        .map(|t| t.0.clone())
        .unwrap_or_default();
    let mut out = serde_json::Map::new();
    out.insert("title".into(), Value::String(title));
    out.insert("command".into(), Value::String(w.command.clone()));
    if !w.args.is_empty() {
        out.insert(
            "args".into(),
            Value::Array(w.args.iter().cloned().map(Value::String).collect()),
        );
    }
    if let Some(p) = &w.cwd {
        out.insert(
            "cwd".into(),
            Value::String(p.to_string_lossy().into_owned()),
        );
    }
    if !w.last_state.is_null() {
        out.insert("state".into(), w.last_state.clone());
    }
    Value::Object(out)
}

fn widget_on_close(world: &mut World, entity: Entity) {
    // Best-effort graceful shutdown: send a "close" event, then kill.
    // The stdin writer thread may not get a chance to flush before drop,
    // but the kill guarantees the child goes away.
    if let Some(io) = world.get::<WidgetIO>(entity)
        && let Ok(json) = serde_json::to_string(&HostEvent::Close)
    {
        let _ = io.tx.send(json);
    }
    if let Some(mut wp) = world.get_mut::<WidgetProcess>(entity) {
        let _ = wp.child.kill();
    }
}

// ---------- Systems ----------

/// Drain inbound messages, send `init` once, and send `resize` when the
/// content area changes.
fn tick_widget_io(
    mut bus: ResMut<WidgetMsgBus>,
    mut q: Query<(
        &PaneKindMarker,
        &PaneRect,
        &mut Widget,
        &mut WidgetRender,
        Option<&WidgetIO>,
        Option<&pane_bevy::PaneProject>,
    )>,
    mut titles: Query<&mut PaneTitle>,
    pane_q: Query<Entity, With<PaneKindMarker>>,
) {
    // Walk pane entities so we can update PaneTitle by entity. Bevy 0.18
    // doesn't allow mixing &mut Widget and &mut PaneTitle on the same
    // entity from the same Query when they coexist, so we look up titles
    // separately by entity id.
    let entities: Vec<Entity> = pane_q.iter().collect();
    for entity in entities {
        let Ok((kind, rect, mut w, mut render_state, io_opt, project)) = q.get_mut(entity) else {
            continue;
        };
        if kind.0 != PANE_KIND {
            continue;
        }
        let project_id = project.map(|p| p.0);

        let content_size = Vec2::new(
            (rect.size.x - 2.0 * MARGIN).max(0.0),
            (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
        );

        if let Some(io) = io_opt {
            // Init once we know the pane size.
            if !render_state.init_sent {
                let init = HostEvent::Init {
                    width: content_size.x,
                    height: content_size.y,
                    state: w.last_state.clone(),
                };
                if let Ok(json) = serde_json::to_string(&init) {
                    let _ = io.tx.send(json);
                }
                render_state.init_sent = true;
            }

            // Resize on size change (after init has gone out).
            if render_state.last_size != content_size && render_state.last_size != Vec2::ZERO {
                let resize = HostEvent::Resize {
                    width: content_size.x,
                    height: content_size.y,
                };
                if let Ok(json) = serde_json::to_string(&resize) {
                    let _ = io.tx.send(json);
                }
            }

            // Drain everything available without blocking.
            if let Ok(rx) = io.rx.lock() {
                loop {
                    match rx.try_recv() {
                        Ok(WidgetMsg::Frame { root }) => {
                            render_state.pending_frame = Some(root);
                        }
                        Ok(WidgetMsg::State { value }) => {
                            w.last_state = value;
                        }
                        Ok(WidgetMsg::Title { value }) => {
                            if let Ok(mut t) = titles.get_mut(entity) {
                                if t.0 != value {
                                    t.0 = value;
                                }
                            }
                        }
                        Ok(WidgetMsg::Emit {
                            topic,
                            payload,
                            retain,
                        }) => {
                            // Publish onto the widget↔widget bus; the
                            // pump delivers it to same-project widgets
                            // next frame (see `crate::msgbus`).
                            bus.push_external(crate::msgbus::PendingMsg {
                                project: project_id,
                                topic,
                                payload,
                                sender: crate::msgbus::subprocess_widget_id(entity),
                                retain,
                            });
                        }
                        Err(_) => break,
                    }
                }
            }
        }
    }
}

/// Despawn old content children and rebuild the tree whenever the
/// frame changes or the content area resizes. Skipped while a widget is
/// in edit mode (the overlay owns the content_root subtree).
fn rerender_widgets(
    mut commands: Commands,
    pane_font: Res<PaneFont>,
    metrics: Res<PaneFontMetrics>,
    theme: Res<style_bevy::Theme>,
    themes: Res<style_bevy::ProjectThemes>,
    fonts: Res<style_bevy::FontRegistry>,
    time: Res<Time>,
    pane_zoom: Res<pane_bevy::PaneZoom>,
    mut clip_dirty: ResMut<WidgetClipDirty>,
    mut images: ResMut<Assets<Image>>,
    mut image_cache: ResMut<WidgetImageCache>,
    mut q: Query<(
        Entity,
        &PaneKindMarker,
        &PaneRect,
        &WidgetContentRoot,
        &mut WidgetRender,
        &mut WidgetTargets,
        &mut WidgetScroll,
        Option<&WidgetInputFocus>,
        Option<&WidgetHover>,
        Option<&pane_bevy::PaneProject>,
    )>,
    children_q: Query<&Children>,
) {
    // Per-project theming: each widget renders in its OWN project's theme
    // (so the cube overview shows every project faithfully), falling back
    // to the global/active theme when its project isn't cached.
    let theme_changed = theme.is_changed() || themes.is_changed();
    // Caret blink: visible during the first half of each 1s cycle.
    let blink_phase = time.elapsed_secs().rem_euclid(1.0);
    let caret_visible = blink_phase < 0.5;
    for (
        pane,
        kind,
        rect,
        root,
        mut render_state,
        mut targets,
        mut scroll,
        input_focus,
        hover,
        proj,
    ) in &mut q
    {
        if kind.0 != PANE_KIND {
            continue;
        }

        // This widget's project theme (falls back to the active theme).
        let w_theme: &style_bevy::Theme = proj.and_then(|p| themes.get(p.0)).unwrap_or(&theme);
        let palette = render::WidgetPalette::from_theme(w_theme);

        // PaneRect is now canvas-units; the pane entity Transform
        // applies zoom. Lay out at (rect.size - chrome) canvas-units.
        let content_size = Vec2::new(
            (rect.size.x - 2.0 * MARGIN).max(0.0),
            (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
        );

        let frame_came_in = render_state.pending_frame.is_some();
        if let Some(p) = render_state.pending_frame.take() {
            render_state.current_frame = Some(p);
        }
        let size_changed = render_state.last_size != content_size;
        let forced = render_state.force_render;
        let needs_render = (frame_came_in || size_changed || theme_changed || forced)
            && render_state.current_frame.is_some();
        render_state.force_render = false;

        if !needs_render {
            // Track size even when we don't render so we don't fire a
            // spurious "size changed" the first time a frame arrives.
            if size_changed {
                render_state.last_size = content_size;
            }
            continue;
        }

        let _prof = pane_bevy::prof::pane_span(pane.to_bits(), "widget");

        if let Ok(children) = children_q.get(root.0) {
            for c in children.iter() {
                // `try_despawn`: this per-frame rebuild can race a pane
                // teardown (an exclusive system in another plugin) that
                // recursively despawns this content. A plain `despawn` on
                // a stale child panics the app. Same fix as the rhai
                // widget render path (`apply_latest_frames`/`diff_render`).
                commands.entity(c).try_despawn();
            }
        }
        targets.clicks.clear();
        targets.links.clear();
        targets.spans.clear();
        targets.sliders.clear();
        targets.selects.clear();
        targets.tooltips.clear();
        targets.dialogs.clear();
        targets.popovers.clear();
        targets.toasts.clear();

        let frame_clone = render_state.current_frame.clone().unwrap();

        // Top-level Canvas frames bypass the text/layout renderer
        // entirely — they're absolute-positioned sprite trees, not flow
        // layouts, so trying to measure them through the same pipeline
        // is just wasted work.
        if let Element::Canvas { children } = &frame_clone {
            render_canvas_items(
                &mut commands,
                &mut images,
                &mut image_cache,
                root.0,
                children,
                content_size,
                &pane_font.0,
                &fonts,
            );
        } else {
            let ctx = render::LayoutCtx {
                font: pane_font.0.clone(),
                metrics: *metrics,
                owner_pane: pane,
                content_root: root.0,
                content_size,
                palette: palette.clone(),
                theme: w_theme.clone(),
                fonts: fonts.clone(),
                focused_input: input_focus.cloned(),
                caret_visible,
                hovered_click_id: hover.and_then(|h| h.click_id.clone()),
            };
            let consumed = render::render(
                &mut commands,
                &ctx,
                &mut targets,
                &frame_clone,
                Vec2::ZERO,
                content_size.x,
                0.0,
            );
            let new_max = (consumed.y - content_size.y).max(0.0);
            if (scroll.max_y - new_max).abs() > 0.1 {
                scroll.max_y = new_max;
            }
            if scroll.y > new_max {
                scroll.y = new_max;
            }
        }

        render_state.last_size = content_size;
        clip_dirty.0 = true;
    }
}

/// Spawn each `CanvasItem` as a child of `content_root` at its
/// absolute (x, y), loading sprites through `WidgetImageCache` so the
/// same path resolves to the same `Handle<Image>` across panes.
///
/// Coordinate convention: y grows downward (top-left origin), matching
/// the `Resize` width/height the widget already sees. Internally we
/// flip to Bevy's y-up before assigning Transform.
fn render_canvas_items(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    cache: &mut WidgetImageCache,
    content_root: Entity,
    items: &[CanvasItem],
    _content_size: Vec2,
    default_font: &Handle<Font>,
    fonts: &style_bevy::FontRegistry,
) {
    for item in items {
        match item {
            CanvasItem::Sprite {
                id: _,
                x,
                y,
                w,
                h,
                image,
                hue_shift,
                anchor,
                z,
            } => {
                let Some(handle) = load_image_for_ref(images, cache, image, *hue_shift) else {
                    continue;
                };
                let anchor_cmp = canvas_anchor_to_bevy(*anchor);
                commands.spawn((
                    ChildOf(content_root),
                    Sprite {
                        image: handle,
                        custom_size: Some(Vec2::new(*w, *h)),
                        ..default()
                    },
                    anchor_cmp,
                    Transform::from_xyz(*x, -*y, *z),
                    Visibility::Inherited,
                ));
            }
            CanvasItem::Rect {
                id: _,
                x,
                y,
                w,
                h,
                color,
                anchor,
                z,
                rotation,
            } => {
                let bevy_color = parse_canvas_color(color).unwrap_or(Color::srgb(0.20, 0.22, 0.26));
                let anchor_cmp = canvas_anchor_to_bevy(*anchor);
                let mut transform = Transform::from_xyz(*x, -*y, *z);
                if *rotation != 0.0 {
                    transform.rotation = Quat::from_rotation_z(-rotation.to_radians());
                }
                commands.spawn((
                    ChildOf(content_root),
                    Sprite {
                        color: bevy_color,
                        custom_size: Some(Vec2::new(*w, *h)),
                        ..default()
                    },
                    anchor_cmp,
                    transform,
                    Visibility::Inherited,
                ));
            }
            CanvasItem::Text {
                id: _,
                x,
                y,
                value,
                color,
                size,
                family,
                anchor,
                z,
            } => {
                let font_size = size.unwrap_or(14.0).max(1.0);
                let col = color
                    .as_deref()
                    .and_then(parse_canvas_color)
                    .unwrap_or(Color::WHITE);
                let font = match family.as_deref() {
                    Some(f) => fonts.resolve(f),
                    None => default_font.clone(),
                };
                let anchor_cmp = canvas_anchor_to_bevy(*anchor);
                commands.spawn((
                    ChildOf(content_root),
                    Text2d::new(value.clone()),
                    TextFont {
                        font,
                        font_size,
                        ..default()
                    },
                    TextColor(col),
                    anchor_cmp,
                    bevy::text::TextLayout::new_with_no_wrap(),
                    Transform::from_xyz(*x, -*y, *z),
                    Visibility::Inherited,
                ));
            }
        }
    }
}

pub(crate) fn canvas_anchor_to_bevy(a: CanvasAnchor) -> bevy::sprite::Anchor {
    match a {
        CanvasAnchor::TopLeft => bevy::sprite::Anchor::TOP_LEFT,
        CanvasAnchor::TopCenter => bevy::sprite::Anchor::TOP_CENTER,
        CanvasAnchor::Center => bevy::sprite::Anchor::CENTER,
        CanvasAnchor::BottomCenter => bevy::sprite::Anchor::BOTTOM_CENTER,
        CanvasAnchor::BottomLeft => bevy::sprite::Anchor::BOTTOM_LEFT,
    }
}

/// Parse a `#rrggbb` or `#rrggbbaa` color into a Bevy `Color`. Accepts
/// the same syntax as `protocol::parse_hex_color` plus an optional
/// alpha byte. Returns None on malformed input so callers can fall
/// back to a default.
pub(crate) fn parse_canvas_color(s: &str) -> Option<Color> {
    let s = s.strip_prefix('#').unwrap_or(s);
    match s.len() {
        6 => {
            let r = u8::from_str_radix(&s[0..2], 16).ok()?;
            let g = u8::from_str_radix(&s[2..4], 16).ok()?;
            let b = u8::from_str_radix(&s[4..6], 16).ok()?;
            Some(Color::srgb(
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
            ))
        }
        8 => {
            let r = u8::from_str_radix(&s[0..2], 16).ok()?;
            let g = u8::from_str_radix(&s[2..4], 16).ok()?;
            let b = u8::from_str_radix(&s[4..6], 16).ok()?;
            let a = u8::from_str_radix(&s[6..8], 16).ok()?;
            Some(Color::srgba(
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                a as f32 / 255.0,
            ))
        }
        _ => None,
    }
}

/// Resolve an `ImageRef` to a Bevy `Handle<Image>`, going through the
/// cache so repeated references to the same path don't re-decode.
/// `hue_shift_deg != 0` bypasses the cache because the resulting
/// image depends on the rotation — at the moment we just log and
/// return the un-shifted image; a proper per-plant tint pass lands
/// once a widget actually exercises it.
pub(crate) fn load_image_for_ref(
    images: &mut Assets<Image>,
    cache: &mut WidgetImageCache,
    image_ref: &ImageRef,
    _hue_shift_deg: f32,
) -> Option<Handle<Image>> {
    match image_ref {
        ImageRef::Path { path } => {
            let path = PathBuf::from(path);
            if let Some(handle) = cache.by_path.get(&path) {
                return Some(handle.clone());
            }
            let image = load_image_from_disk(&path)?;
            let handle = images.add(image);
            cache.by_path.insert(path, handle.clone());
            Some(handle)
        }
        ImageRef::Tile {
            path,
            tile_w,
            tile_h,
            col,
            row,
        } => {
            let key = TileKey {
                path: PathBuf::from(path),
                tile_w: *tile_w,
                tile_h: *tile_h,
                col: *col,
                row: *row,
            };
            if let Some(handle) = cache.tiles.get(&key) {
                return Some(handle.clone());
            }
            let image = load_tile_from_disk(&key)?;
            let handle = images.add(image);
            cache.tiles.insert(key, handle.clone());
            Some(handle)
        }
    }
}

fn load_image_from_disk(path: &Path) -> Option<Image> {
    let bytes = std::fs::read(path)
        .map_err(|e| eprintln!("widget: failed to read {}: {}", path.display(), e))
        .ok()?;
    let img = image::load_from_memory(&bytes)
        .map_err(|e| eprintln!("widget: failed to decode {}: {}", path.display(), e))
        .ok()?
        .to_rgba8();
    let (w, h) = (img.width(), img.height());
    let data = img.into_raw();
    Some(make_nearest_image(data, w, h))
}

fn load_tile_from_disk(key: &TileKey) -> Option<Image> {
    let bytes = std::fs::read(&key.path)
        .map_err(|e| eprintln!("widget: failed to read {}: {}", key.path.display(), e))
        .ok()?;
    let sheet = image::load_from_memory(&bytes)
        .map_err(|e| eprintln!("widget: failed to decode {}: {}", key.path.display(), e))
        .ok()?
        .to_rgba8();
    let bg_px = sheet.get_pixel(0, 0).0;
    let (bg_r, bg_g, bg_b) = (bg_px[0], bg_px[1], bg_px[2]);
    let mut data: Vec<u8> = Vec::with_capacity((key.tile_w * key.tile_h * 4) as usize);
    let x0 = key.col * key.tile_w;
    let y0 = key.row * key.tile_h;
    for y in 0..key.tile_h {
        for x in 0..key.tile_w {
            let px_x = x0 + x;
            let px_y = y0 + y;
            if px_x >= sheet.width() || px_y >= sheet.height() {
                data.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            let p = sheet.get_pixel(px_x, px_y).0;
            if p[0] == bg_r && p[1] == bg_g && p[2] == bg_b {
                data.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                data.extend_from_slice(&[p[0], p[1], p[2], 255]);
            }
        }
    }
    Some(make_nearest_image(data, key.tile_w, key.tile_h))
}

pub fn make_nearest_image(data: Vec<u8>, w: u32, h: u32) -> Image {
    use bevy::asset::RenderAssetUsages;
    use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
    let mut img = Image::new(
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    img.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        mag_filter: ImageFilterMode::Nearest,
        min_filter: ImageFilterMode::Nearest,
        mipmap_filter: ImageFilterMode::Nearest,
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        ..ImageSamplerDescriptor::nearest()
    });
    img
}

/// Mirror each widget pane's `WidgetTargets` (clicks + links) into
/// `PaneHotZones` in **visual content-local coords** (i.e. with the
/// pane's scroll offset already subtracted and clipped to the visible
/// content area). pane-bevy's pinned-pane hit-tester consults these so
/// buttons / links / inputs on pinned widgets keep working while
/// empty space falls through to whatever is underneath.
///
/// Covers both `widget` and `rhai_widget` kinds because they share the
/// `WidgetTargets` component — no kind check required.
fn update_widget_hot_zones(
    mut q: Query<(
        &PaneRect,
        &WidgetTargets,
        Option<&WidgetScroll>,
        Option<&crate::rhai_widget::RhaiWidget>,
        &mut PaneHotZones,
    )>,
) {
    for (rect, targets, scroll, rhai, mut zones) in &mut q {
        zones.clear();
        let scroll_y = scroll.map(|s| s.y).unwrap_or(0.0);
        let content_size = Vec2::new(
            (rect.size.x - 2.0 * MARGIN).max(0.0),
            (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
        );
        let visible = Rect::from_corners(Vec2::ZERO, content_size);
        for c in &targets.clicks {
            let visual = Rect::from_corners(
                Vec2::new(c.rect.min.x, c.rect.min.y - scroll_y),
                Vec2::new(c.rect.max.x, c.rect.max.y - scroll_y),
            );
            let clipped = visual.intersect(visible);
            if !clipped.is_empty() {
                zones.push(clipped);
            }
        }
        for l in &targets.links {
            let visual = Rect::from_corners(
                Vec2::new(l.rect.min.x, l.rect.min.y - scroll_y),
                Vec2::new(l.rect.max.x, l.rect.max.y - scroll_y),
            );
            let clipped = visual.intersect(visible);
            if !clipped.is_empty() {
                zones.push(clipped);
            }
        }
        // Canvas-based rhai widgets (e.g. chess) self-route clicks and
        // publish no per-element targets, so the loops above leave them
        // with no hot-zones — meaning they'd be entirely click-through
        // when pinned. If such a widget handles clicks at all (defines
        // `on_click`), treat its whole content area as one hot-zone so
        // pinned clicks reach it. Decorative widgets without `on_click`
        // (dust) stay click-through.
        if zones.0.is_empty() && rhai.map_or(false, |r| r.wants_clicks) && !visible.is_empty() {
            zones.push(visible);
        }
    }
}

fn handle_widget_press(
    mut commands: Commands,
    mut open_select: ResMut<WidgetOpenSelect>,
    mut open_popover: ResMut<WidgetOpenPopover>,
    overlay_hits: Res<WidgetOverlayHits>,
    mut presses: MessageReader<PaneContentPressed>,
    kinds: Query<&PaneKindMarker>,
    widgets: Query<(
        &Widget,
        &WidgetTargets,
        Option<&WidgetIO>,
        &WidgetContentRoot,
        Option<&WidgetScroll>,
        &WidgetRender,
    )>,
) {
    for ev in presses.read() {
        let Ok(kind) = kinds.get(ev.pane) else {
            continue;
        };
        if kind.0 != PANE_KIND {
            continue;
        }
        // A modal dialog is open (anywhere) — it owns all input; the overlay
        // handler routes the click (body button or scrim dismiss).
        if !overlay_hits.dialog_id.is_empty() {
            continue;
        }
        let Ok((_widget, targets, io, _root, scroll, render_state)) = widgets.get(ev.pane) else {
            continue;
        };

        // Click rects are stored in content_root local coords; once the
        // user scrolls, content_root has slid up by `scroll.y` so the
        // visual position of each rect is `rect.y - scroll.y`. Add the
        // scroll offset to the hit-test point so it lands on the rect
        // that's currently under the cursor, not the one that USED to
        // be there at scroll=0.
        let scroll_y = scroll.map(|s| s.y).unwrap_or(0.0);
        let hit_pt = ev.local_pt + Vec2::new(0.0, scroll_y);

        // First: normal widget-frame click handling (buttons, links).
        let click = targets.clicks.iter().find(|t| t.rect.contains(hit_pt));
        let link = targets.links.iter().find(|t| t.rect.contains(hit_pt));

        // While this pane has an open dropdown / popover, route every non-trigger
        // press to the overlay (select / dismiss) instead of elements beneath it.
        if open_select.0.as_ref().is_some_and(|o| o.pane == ev.pane) {
            let is_trigger = click.is_some_and(|c| matches!(c.kind, ClickKind::SelectTrigger));
            if !is_trigger {
                continue;
            }
        }
        if open_popover.0.as_ref().is_some_and(|o| o.pane == ev.pane) {
            let is_trigger = click.is_some_and(|c| matches!(c.kind, ClickKind::PopoverTrigger));
            if !is_trigger {
                continue;
            }
        }

        if let Some(ct) = click {
            // A select trigger toggles the host-owned open-dropdown state — no
            // widget event until an option is actually picked.
            if matches!(ct.kind, ClickKind::SelectTrigger) {
                let already = open_select
                    .0
                    .as_ref()
                    .is_some_and(|o| o.pane == ev.pane && o.id == ct.id);
                open_select.0 = if already {
                    None
                } else {
                    Some(OpenSelect {
                        pane: ev.pane,
                        id: ct.id.clone(),
                    })
                };
                commands.entity(ev.pane).remove::<WidgetInputFocus>();
                continue;
            }
            if matches!(ct.kind, ClickKind::PopoverTrigger) {
                let already = open_popover
                    .0
                    .as_ref()
                    .is_some_and(|o| o.pane == ev.pane && o.id == ct.id);
                open_popover.0 = if already {
                    None
                } else {
                    Some(OpenSelect {
                        pane: ev.pane,
                        id: ct.id.clone(),
                    })
                };
                commands.entity(ev.pane).remove::<WidgetInputFocus>();
                continue;
            }
            if let Some(io) = io {
                let evt = match &ct.kind {
                    ClickKind::Button => HostEvent::Click { id: ct.id.clone() },
                    ClickKind::TabSelect { tab } => HostEvent::TabSelect {
                        id: ct.id.clone(),
                        tab: tab.clone(),
                    },
                    ClickKind::RadioSelect { option } => HostEvent::RadioSelect {
                        id: ct.id.clone(),
                        option: option.clone(),
                    },
                    ClickKind::NumberChange { value } => HostEvent::NumberChange {
                        id: ct.id.clone(),
                        value: *value,
                    },
                    ClickKind::Toggle { new_checked } => HostEvent::Toggle {
                        id: ct.id.clone(),
                        checked: *new_checked,
                    },
                    ClickKind::InputFocus => HostEvent::InputFocus {
                        id: ct.id.clone(),
                        focused: true,
                    },
                    ClickKind::SelectTrigger | ClickKind::PopoverTrigger => {
                        unreachable!("handled above")
                    }
                };
                if let Ok(json) = serde_json::to_string(&evt) {
                    let _ = io.tx.send(json);
                }
            }
            // Update host-side input focus marker when clicking an Input.
            if matches!(ct.kind, ClickKind::InputFocus) {
                // Seed the local typing buffer from the Input element's
                // current value so the first keystroke appends instead
                // of clearing.
                let mut focus = WidgetInputFocus::new(ct.id.clone());
                if let Some(frame) = render_state.current_frame.as_ref() {
                    if let Some((value, _, multiline)) = find_input_value(frame, &ct.id) {
                        focus.caret = value.chars().count();
                        focus.value = value;
                        focus.multiline = multiline;
                    }
                }
                commands.entity(ev.pane).insert(focus);
            } else {
                commands.entity(ev.pane).remove::<WidgetInputFocus>();
            }
            continue;
        }
        if let Some(lt) = link {
            open_url(&lt.url);
            continue;
        }

        // Pinned panes are background decoration: a press only reaches us via
        // a hot-zone hit; if click/link missed (e.g. scroll changed mid-frame),
        // swallow it rather than treating it as a first-class focus event.
        if ev.pinned {
            continue;
        }
        // Empty-space press on an unpinned widget: nothing to do.
    }
}


fn poll_widget_children(
    mut commands: Commands,
    mut q: Query<(
        Entity,
        &PaneKindMarker,
        &mut WidgetProcess,
        &mut WidgetRender,
        &Widget,
    )>,
) {
    for (entity, kind, mut wp, mut render_state, widget) in &mut q {
        if kind.0 != PANE_KIND {
            continue;
        }
        match wp.child.try_wait() {
            Ok(Some(status)) => {
                let code = status.code();
                eprintln!("[widget] exited code={:?}; respawning", code);
                commands
                    .entity(entity)
                    .remove::<WidgetProcess>()
                    .remove::<WidgetIO>();

                if widget.command.trim().is_empty() {
                    if render_state.current_frame.is_none() {
                        render_state.current_frame = Some(placeholder_frame());
                        render_state.pending_frame = None;
                        render_state.last_size = Vec2::ZERO;
                    }
                    continue;
                }

                // Force a fresh init/resize handshake so the new child
                // gets the current pane width on its first frame.
                render_state.init_sent = false;
                render_state.pending_frame = None;

                match spawn_widget_process(&widget.command, &widget.args, widget.cwd.as_deref()) {
                    Ok((process, io)) => {
                        commands.entity(entity).insert((process, io));
                    }
                    Err(e) => {
                        eprintln!("[widget] respawn failed: {}", e);
                        render_state.current_frame =
                            Some(error_frame(&format!("respawn failed: {}", e)));
                        render_state.last_size = Vec2::ZERO;
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("[widget] try_wait failed: {}", e);
            }
        }
    }
}

// ---------- Helpers ----------

fn open_url(url: &str) {
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(url).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("xdg-open").arg(url).spawn();
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        eprintln!("[widget] open_url unsupported on this OS: {}", url);
    }
}

/// Find the first `Element::Input` with id == `target` anywhere in the
/// tree. Returns `(value, placeholder)`. Used to seed
/// [`WidgetInputFocus`] when a click first focuses the input.
/// Char index of the first character of each line (line N starts at
/// `starts[N]`). Always has at least one entry (`0`).
fn line_starts(chars: &[char]) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (i, &c) in chars.iter().enumerate() {
        if c == '\n' {
            starts.push(i + 1);
        }
    }
    starts
}

/// `(line, column)` of a caret char-index within `chars`.
fn caret_line_col(chars: &[char], caret: usize) -> (usize, usize) {
    let mut line = 0;
    let mut col = 0;
    for &c in chars.iter().take(caret) {
        if c == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Number of characters on `line` (excluding its trailing newline).
fn line_len(chars: &[char], starts: &[usize], line: usize) -> usize {
    let start = starts[line];
    let end = starts.get(line + 1).map(|n| n - 1).unwrap_or(chars.len());
    end.saturating_sub(start)
}

/// Seed info for a focusable text element: `(value, placeholder,
/// multiline)`. Returns the matching `Input` or `TextArea` by id.
pub(crate) fn find_input_value(el: &Element, target: &str) -> Option<(String, String, bool)> {
    match el {
        Element::Input {
            id,
            value,
            placeholder,
            ..
        } if id == target => Some((value.clone(), placeholder.clone(), false)),
        Element::TextArea {
            id,
            value,
            placeholder,
            ..
        } if id == target => Some((value.clone(), placeholder.clone(), true)),
        Element::Vstack { children, .. }
        | Element::Hstack { children, .. }
        | Element::Frame { children, .. }
        | Element::Scroll { children, .. }
        | Element::ListItem { children, .. } => {
            children.iter().find_map(|c| find_input_value(c, target))
        }
        _ => None,
    }
}

/// Blur a widget's focused input when keyboard focus moves to another
/// pane — clicking a different pane, switching projects, focusing the
/// terminal, etc. Without this an `Input` / `TextArea` would keep its
/// caret and keep capturing keystrokes after its pane lost focus.
///
/// Keyed on `FocusedPane` changing: we only ever blur the *previously*
/// focused pane, so this never races with the same-frame click that
/// focuses a new input.
fn blur_inputs_on_focus_change(
    mut commands: Commands,
    focused: Res<pane_bevy::FocusedPane>,
    mut prev: Local<Option<Entity>>,
    q: Query<(
        &WidgetInputFocus,
        Option<&WidgetIO>,
        Option<&crate::rhai_widget::RhaiWidget>,
    )>,
) {
    if !focused.is_changed() {
        return;
    }
    let now = focused.0;
    if let Some(old) = *prev {
        if Some(old) != now {
            if let Ok((focus, io, rhai)) = q.get(old) {
                if let Some(io) = io {
                    let evt = HostEvent::InputFocus {
                        id: focus.id.clone(),
                        focused: false,
                    };
                    if let Ok(json) = serde_json::to_string(&evt) {
                        let _ = io.tx.send(json);
                    }
                }
                if let Some(rhai) = rhai {
                    rhai.send_input_focus(focus.id.clone(), false);
                }
                commands.entity(old).remove::<WidgetInputFocus>();
            }
        }
    }
    *prev = now;
}

/// Process keyboard events for whichever widget pane currently has an
/// `Element::Input` focused. Mutates the pane's [`WidgetInputFocus`]
/// in place and forwards [`HostEvent::InputChange`] / [`HostEvent::
/// InputSubmit`] / [`HostEvent::InputFocus`] to the widget.
fn handle_widget_input_typing(
    mut commands: Commands,
    mut keys: MessageReader<bevy::input::keyboard::KeyboardInput>,
    modifiers: Res<ButtonInput<KeyCode>>,
    owner: Res<pane_bevy::KeyboardOwner>,
    mut q: Query<(
        Entity,
        &mut WidgetInputFocus,
        Option<&WidgetIO>,
        Option<&crate::rhai_widget::RhaiWidget>,
    )>,
) {
    use bevy::input::keyboard::Key;
    let submit_modifier = modifiers.pressed(KeyCode::SuperLeft)
        || modifiers.pressed(KeyCode::SuperRight)
        || modifiers.pressed(KeyCode::ControlLeft)
        || modifiers.pressed(KeyCode::ControlRight);
    for (pane, mut focus, io, rhai) in &mut q {
        // A widget input can keep `WidgetInputFocus` while another pane is
        // focused; only consume keys when the keyboard owner allows this
        // pane (and never while a text modal owns input).
        if !owner.allows_pane(pane) {
            continue;
        }
        let mut value_changed = false;
        let mut submitted = false;
        let mut blurred = false;
        for ev in keys.read() {
            if !ev.state.is_pressed() {
                continue;
            }
            match &ev.logical_key {
                Key::Character(s) => {
                    // Skip control combos; only insert the literal char.
                    let prev = focus.value.chars().take(focus.caret).collect::<String>();
                    let after: String = focus.value.chars().skip(focus.caret).collect();
                    focus.value = format!("{}{}{}", prev, s.as_str(), after);
                    focus.caret += s.chars().count();
                    focus.blink = 0.0;
                    value_changed = true;
                }
                Key::Space => {
                    let prev = focus.value.chars().take(focus.caret).collect::<String>();
                    let after: String = focus.value.chars().skip(focus.caret).collect();
                    focus.value = format!("{} {}", prev, after);
                    focus.caret += 1;
                    focus.blink = 0.0;
                    value_changed = true;
                }
                Key::Backspace => {
                    if focus.caret > 0 {
                        let mut chars: Vec<char> = focus.value.chars().collect();
                        chars.remove(focus.caret - 1);
                        focus.value = chars.into_iter().collect();
                        focus.caret -= 1;
                        focus.blink = 0.0;
                        value_changed = true;
                    }
                }
                Key::Delete => {
                    let mut chars: Vec<char> = focus.value.chars().collect();
                    if focus.caret < chars.len() {
                        chars.remove(focus.caret);
                        focus.value = chars.into_iter().collect();
                        focus.blink = 0.0;
                        value_changed = true;
                    }
                }
                Key::ArrowLeft => {
                    focus.caret = focus.caret.saturating_sub(1);
                    focus.blink = 0.0;
                }
                Key::ArrowRight => {
                    let n = focus.value.chars().count();
                    focus.caret = (focus.caret + 1).min(n);
                    focus.blink = 0.0;
                }
                Key::ArrowUp => {
                    let chars: Vec<char> = focus.value.chars().collect();
                    let starts = line_starts(&chars);
                    let (line, col) = caret_line_col(&chars, focus.caret);
                    if line > 0 {
                        let target = line - 1;
                        let c = col.min(line_len(&chars, &starts, target));
                        focus.caret = starts[target] + c;
                    }
                    focus.blink = 0.0;
                }
                Key::ArrowDown => {
                    let chars: Vec<char> = focus.value.chars().collect();
                    let starts = line_starts(&chars);
                    let (line, col) = caret_line_col(&chars, focus.caret);
                    if line + 1 < starts.len() {
                        let target = line + 1;
                        let c = col.min(line_len(&chars, &starts, target));
                        focus.caret = starts[target] + c;
                    }
                    focus.blink = 0.0;
                }
                Key::Home => {
                    // Line-aware: jump to the start of the current line
                    // (same as start-of-value for single-line inputs).
                    let chars: Vec<char> = focus.value.chars().collect();
                    let starts = line_starts(&chars);
                    let (line, _) = caret_line_col(&chars, focus.caret);
                    focus.caret = starts[line];
                    focus.blink = 0.0;
                }
                Key::End => {
                    // Line-aware: jump to the end of the current line.
                    let chars: Vec<char> = focus.value.chars().collect();
                    let starts = line_starts(&chars);
                    let (line, _) = caret_line_col(&chars, focus.caret);
                    focus.caret = starts[line] + line_len(&chars, &starts, line);
                    focus.blink = 0.0;
                }
                Key::Enter => {
                    // In a TextArea, plain Enter inserts a newline and
                    // submit is Cmd/Ctrl+Enter. Single-line inputs submit
                    // on plain Enter.
                    if focus.multiline && !submit_modifier {
                        let prev: String = focus.value.chars().take(focus.caret).collect();
                        let after: String = focus.value.chars().skip(focus.caret).collect();
                        focus.value = format!("{}\n{}", prev, after);
                        focus.caret += 1;
                        focus.blink = 0.0;
                        value_changed = true;
                    } else {
                        submitted = true;
                    }
                }
                Key::Escape => {
                    blurred = true;
                }
                _ => {}
            }
        }
        if value_changed {
            if let Some(io) = io {
                let evt = HostEvent::InputChange {
                    id: focus.id.clone(),
                    value: focus.value.clone(),
                };
                if let Ok(json) = serde_json::to_string(&evt) {
                    let _ = io.tx.send(json);
                }
            }
            if let Some(rhai) = rhai {
                rhai.send_input_change(focus.id.clone(), focus.value.clone());
            }
        }
        if submitted {
            if let Some(io) = io {
                let evt = HostEvent::InputSubmit {
                    id: focus.id.clone(),
                    value: focus.value.clone(),
                };
                if let Ok(json) = serde_json::to_string(&evt) {
                    let _ = io.tx.send(json);
                }
            }
            if let Some(rhai) = rhai {
                rhai.send_input_submit(focus.id.clone(), focus.value.clone());
            }
        }
        if blurred {
            if let Some(io) = io {
                let evt = HostEvent::InputFocus {
                    id: focus.id.clone(),
                    focused: false,
                };
                if let Ok(json) = serde_json::to_string(&evt) {
                    let _ = io.tx.send(json);
                }
            }
            if let Some(rhai) = rhai {
                rhai.send_input_focus(focus.id.clone(), false);
            }
            commands.entity(pane).remove::<WidgetInputFocus>();
        }
    }
}

#[cfg(test)]
mod slider_tests {
    use super::*;

    fn target(x0: f32, span: f32, min: f32, max: f32, step: f32) -> SliderTarget {
        SliderTarget {
            id: "s".into(),
            rect: Rect::new(0.0, 0.0, 100.0, 20.0),
            value_x0: x0,
            value_span: span,
            min,
            max,
            step,
        }
    }

    #[test]
    fn value_at_maps_and_clamps() {
        let t = target(10.0, 80.0, 0.0, 100.0, 0.0);
        assert_eq!(t.value_at(10.0), 0.0); // left edge
        assert_eq!(t.value_at(90.0), 100.0); // right edge
        assert!((t.value_at(50.0) - 50.0).abs() < 1e-3); // middle
        assert_eq!(t.value_at(-50.0), 0.0); // clamp low
        assert_eq!(t.value_at(500.0), 100.0); // clamp high
    }

    #[test]
    fn value_at_snaps_to_step() {
        let t = target(0.0, 100.0, 0.0, 10.0, 1.0);
        // 37% → 3.7 → snaps to 4
        assert_eq!(t.value_at(37.0), 4.0);
        // 34% → 3.4 → snaps to 3
        assert_eq!(t.value_at(34.0), 3.0);
    }

    #[test]
    fn value_at_handles_nonzero_min() {
        let t = target(0.0, 100.0, -50.0, 50.0, 0.0);
        assert!((t.value_at(0.0) - -50.0).abs() < 1e-3);
        assert!((t.value_at(50.0) - 0.0).abs() < 1e-3);
        assert!((t.value_at(100.0) - 50.0).abs() < 1e-3);
    }
}

#[cfg(test)]
mod line_math_tests {
    use super::*;

    #[test]
    fn line_col_and_lens() {
        let chars: Vec<char> = "ab\ncde\nf".chars().collect();
        let starts = line_starts(&chars);
        assert_eq!(starts, vec![0, 3, 7]);
        // caret after "ab\ncd" → line 1, col 2
        assert_eq!(caret_line_col(&chars, 5), (1, 2));
        // caret at very start
        assert_eq!(caret_line_col(&chars, 0), (0, 0));
        assert_eq!(line_len(&chars, &starts, 0), 2); // "ab"
        assert_eq!(line_len(&chars, &starts, 1), 3); // "cde"
        assert_eq!(line_len(&chars, &starts, 2), 1); // "f"
    }

    #[test]
    fn single_line_is_one_line() {
        let chars: Vec<char> = "hello".chars().collect();
        let starts = line_starts(&chars);
        assert_eq!(starts, vec![0]);
        assert_eq!(caret_line_col(&chars, 3), (0, 3));
        assert_eq!(line_len(&chars, &starts, 0), 5);
    }

    #[test]
    fn finds_textarea_as_multiline() {
        let frame = Element::TextArea {
            id: "q".into(),
            value: "x".into(),
            placeholder: String::new(),
            focused: false,
            rows: 4,
            width: 200.0,
            style: None,
        };
        let (value, _, multiline) = find_input_value(&frame, "q").unwrap();
        assert_eq!(value, "x");
        assert!(multiline);
    }

    #[test]
    fn finds_input_as_single_line() {
        let frame = Element::Input {
            id: "q".into(),
            value: "y".into(),
            placeholder: String::new(),
            focused: false,
            width: 160.0,
            style: None,
        };
        let (_, _, multiline) = find_input_value(&frame, "q").unwrap();
        assert!(!multiline);
    }

    #[test]
    fn char_index_maps_x_to_nearest_boundary() {
        // Monospace: cell_width 10 at font_size 10 → each char is 10px.
        let m = PaneFontMetrics {
            cell_width: 10.0,
            font_size: 10.0,
        };
        // Boundaries land at 0,10,20,30,40,50 for "hello".
        assert_eq!(char_index_at_x("hello", 0.0, 10.0, &m, 0.0), 0);
        assert_eq!(char_index_at_x("hello", 0.0, 10.0, &m, 4.0), 0); // nearer 0 than 10
        assert_eq!(char_index_at_x("hello", 0.0, 10.0, &m, 23.0), 2); // nearer 20 than 30
        assert_eq!(char_index_at_x("hello", 0.0, 10.0, &m, 27.0), 3); // nearer 30 than 20
        // Past the end clamps to the final boundary (whole string).
        assert_eq!(char_index_at_x("hello", 0.0, 10.0, &m, 1000.0), 5);
        // A left-clipped point clamps to the start.
        assert_eq!(char_index_at_x("hello", 0.0, 10.0, &m, -50.0), 0);
        // Honors a non-zero run origin (e.g. a right-aligned table cell):
        // x is relative to origin_x, so 127 → rel 27 → boundary 3.
        assert_eq!(char_index_at_x("hello", 100.0, 10.0, &m, 127.0), 3);
    }
}

fn placeholder_frame() -> Element {
    Element::Vstack {
        gap: 8.0,
        pad: 12.0,
        children: vec![
            Element::Text {
                value: "Widget not configured".into(),
                color: Some("#cc8".into()),
                size: Some(14.0),
                weight: Some(Weight::Bold),
                family: None,
                selectable: false,
            },
            Element::Text {
                value: format!("Set {} or save a snapshot with a command.", DEFAULT_CMD_ENV),
                color: Some("#888".into()),
                size: None,
                weight: None,
                family: None,
                selectable: false,
            },
        ],
        style: None,
    }
}

fn error_frame(msg: &str) -> Element {
    Element::Vstack {
        gap: 6.0,
        pad: 12.0,
        children: vec![
            Element::Text {
                value: "Widget error".into(),
                color: Some("#e55".into()),
                size: Some(14.0),
                weight: Some(Weight::Bold),
                family: None,
                selectable: false,
            },
            Element::Text {
                value: msg.into(),
                color: Some("#aaa".into()),
                size: None,
                weight: None,
                family: None,
                selectable: false,
            },
        ],
        style: None,
    }
}
