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
    PaneKindSpec, PaneRect, PaneRegistry, PaneTitle, TITLE_H, FocusedTextInput, TextInput,
    TextInputEvent, TextInputStyle, focus_text_input, spawn_text_input,
};
use serde_json::Value;

pub mod button_material;
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
static WAKEUP_HOOK: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>> =
    std::sync::OnceLock::new();

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
    /// Timestamp of most recent content press, for double-click detection.
    last_press_time: Option<f64>,
}

impl Widget {
    pub fn new(command: impl Into<String>, args: Vec<String>, cwd: Option<PathBuf>) -> Self {
        Self {
            command: command.into(),
            args,
            cwd,
            last_state: Value::Null,
            last_press_time: None,
        }
    }
}

const DOUBLE_CLICK_SECS: f64 = 0.35;
const EDIT_FONT_SIZE: f32 = 13.0;
const EDIT_LINE_HEIGHT: f32 = 16.0;
const EDIT_PAD: f32 = 8.0;
const EDIT_INPUT_H: f32 = 22.0;
const EDIT_HINT_FONT_SIZE: f32 = 11.0;
const EDIT_LABEL_FONT_SIZE: f32 = 11.0;

/// Marker + entity refs for the "edit command" overlay shown when the
/// user double-clicks a widget. While present, `rerender_widgets`
/// suppresses normal frame rendering.
#[derive(Component)]
pub struct WidgetEditMode {
    pub command_input: Entity,
    pub label: Entity,
    pub hint: Entity,
    pub bg: Entity,
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
        Self { id, value: String::new(), caret: 0, blink: 0.0, multiline: false }
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
        sel.focus = char_index_at_x(&sel.text, sel.rect.min.x, sel.font_size, &metrics, ev.local_pt.x);
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
    let color = Color::LinearRgba(LinearRgba { alpha: 0.32, ..base });
    for (sel, chrome) in &sels {
        let (a, b) = (sel.anchor.min(sel.focus), sel.anchor.max(sel.focus));
        if a == b {
            continue;
        }
        let chars: Vec<char> = sel.text.chars().collect();
        let pre: String = chars[..a.min(chars.len())].iter().collect();
        let mid: String = chars[a.min(chars.len())..b.min(chars.len())].iter().collect();
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
/// content_root (`rerender_widgets`, `enter_edit_mode`). Consumed by
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
            .add_plugins(WidgetButtonMaterialPlugin)
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
                    render_text_selection_highlight,
                    copy_text_selection,
                    handle_widget_input_typing,
                    handle_widget_edit_events,
                    poll_widget_children,
                    handle_widget_wheel,
                    apply_widget_scroll,
                    update_widget_hot_zones,
                )
                    .chain(),
            )
            .add_systems(PostUpdate, clip_widget_sprites);
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
    all_panes: Query<
        (Entity, &pane_bevy::PaneRect, Option<&Visibility>),
        With<pane_bevy::PaneTag>,
    >,
    targets: Query<&WidgetTargets>,
    mut widgets: Query<
        (
            Entity,
            &pane_bevy::PaneKindMarker,
            &mut WidgetHover,
            &mut WidgetRender,
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
    for (pane, kind, mut hover, mut render_state) in &mut widgets {
        let is_widget_kind = kind.0 == PANE_KIND || kind.0 == rhai_widget::PANE_KIND;
        if !is_widget_kind {
            continue;
        }
        // Find this pane's rect (we already have it in `candidates`,
        // but cheaper to re-query than threading it through).
        let pane_rect = candidates.iter().find(|(e, _)| *e == pane).map(|(_, r)| *r);
        let new_id: Option<String> = match (topmost, pane_rect, cursor) {
            (Some((pt, top)), Some(rect), Some(_)) if top == pane => {
                let local = pane_bevy::pt_to_content_local(pt, &rect);
                targets
                    .get(pane)
                    .ok()
                    .and_then(|t| {
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
    all_panes: Query<
        (Entity, &pane_bevy::PaneRect, Option<&Visibility>),
        With<pane_bevy::PaneTag>,
    >,
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
    let Some(pt) = win.cursor_position() else { return };

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
        (&WidgetScroll, &pane_bevy::PaneChrome, &pane_bevy::PaneKindMarker),
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
        let payload: serde_json::Value = serde_json::from_str(&ev.payload_json)
            .unwrap_or(serde_json::Value::Null);
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
    for (kind, mut render_state, io) in &mut widgets {        if kind.0 != PANE_KIND || !render_state.init_sent {
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
        last_press_time: None,
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
        out.insert("cwd".into(), Value::String(p.to_string_lossy().into_owned()));
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
                        Ok(WidgetMsg::Emit { topic, payload, retain }) => {
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
        Option<&WidgetEditMode>,
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
        _pane,
        kind,
        rect,
        root,
        mut render_state,
        mut targets,
        mut scroll,
        edit,
        input_focus,
        hover,
        proj,
    ) in &mut q
    {
        if kind.0 != PANE_KIND {
            continue;
        }
        if edit.is_some() {
            continue;
        }

        // This widget's project theme (falls back to the active theme).
        let w_theme: &style_bevy::Theme =
            proj.and_then(|p| themes.get(p.0)).unwrap_or(&theme);
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
                let bevy_color = parse_canvas_color(color)
                    .unwrap_or(Color::srgb(0.20, 0.22, 0.26));
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
            Some(Color::srgb(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0))
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
        Option<&WidgetEditMode>,
        Option<&crate::rhai_widget::RhaiWidget>,
        &mut PaneHotZones,
    )>,
) {
    for (rect, targets, scroll, edit, rhai, mut zones) in &mut q {
        zones.clear();
        // In edit mode the overlay covers the pane and the underlying
        // targets are stale; leave hot-zones empty so a pinned widget
        // in edit mode passes clicks through (the user would have to
        // unpin to interact with the overlay anyway).
        if edit.is_some() {
            continue;
        }
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
    time: Res<Time>,
    pane_font: Res<PaneFont>,
    metrics: Res<PaneFontMetrics>,
    mut focused: ResMut<FocusedTextInput>,
    mut clip_dirty: ResMut<WidgetClipDirty>,
    mut presses: MessageReader<PaneContentPressed>,
    kinds: Query<&PaneKindMarker>,
    pane_rects: Query<&PaneRect>,
    mut widgets: Query<(
        &mut Widget,
        &WidgetTargets,
        Option<&WidgetIO>,
        &WidgetContentRoot,
        Option<&WidgetEditMode>,
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
        let Ok((mut widget, targets, io, root, edit, scroll, render_state)) =
            widgets.get_mut(ev.pane)
        else {
            continue;
        };

        // In edit mode, every press is consumed by the overlay. Blur on
        // clicks outside the input rect; focus on clicks inside.
        if let Some(edit) = edit {
            let input_rect = Rect::from_corners(
                Vec2::new(EDIT_PAD, EDIT_PAD + EDIT_LINE_HEIGHT + 4.0),
                Vec2::new(f32::MAX, EDIT_PAD + EDIT_LINE_HEIGHT + 4.0 + EDIT_INPUT_H),
            );
            let target = if input_rect.contains(ev.local_pt) {
                Some(edit.command_input)
            } else {
                None
            };
            focus_text_input(&mut commands, &mut focused, [], target);
            continue;
        }

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

        if let Some(ct) = click {
            if let Some(io) = io {
                let evt = match &ct.kind {
                    ClickKind::Button => HostEvent::Click { id: ct.id.clone() },
                    ClickKind::TabSelect { tab } => HostEvent::TabSelect {
                        id: ct.id.clone(),
                        tab: tab.clone(),
                    },
                    ClickKind::Toggle { new_checked } => HostEvent::Toggle {
                        id: ct.id.clone(),
                        checked: *new_checked,
                    },
                    ClickKind::InputFocus => HostEvent::InputFocus {
                        id: ct.id.clone(),
                        focused: true,
                    },
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
            widget.last_press_time = Some(time.elapsed_secs_f64());
            continue;
        }
        if let Some(lt) = link {
            open_url(&lt.url);
            widget.last_press_time = Some(time.elapsed_secs_f64());
            continue;
        }

        // Pinned panes are background decoration: the only way the
        // press reached us is because pane-bevy thought we had a
        // hot-zone hit. If a transient mismatch (e.g. scroll changed
        // mid-frame) made click/link miss anyway, swallow it rather
        // than entering edit mode or otherwise treating it as a
        // first-class focus event.
        if ev.pinned {
            continue;
        }

        // Empty-space press: double-click → enter edit mode.
        let now = time.elapsed_secs_f64();
        let is_double = widget
            .last_press_time
            .is_some_and(|t| now - t < DOUBLE_CLICK_SECS);
        widget.last_press_time = Some(now);
        if is_double {
            enter_edit_mode(
                &mut commands,
                &mut focused,
                ev.pane,
                root.0,
                &widget,
                pane_font.0.clone(),
                content_size_of(&pane_rects, ev.pane),
                &metrics,
            );
            clip_dirty.0 = true;
        }
    }
}

fn content_size_of(pane_rects: &Query<&PaneRect>, pane: Entity) -> Vec2 {
    let Ok(rect) = pane_rects.get(pane) else {
        return Vec2::ZERO;
    };
    Vec2::new(
        (rect.size.x - 2.0 * MARGIN).max(0.0),
        (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
    )
}

fn enter_edit_mode(
    commands: &mut Commands,
    focused: &mut FocusedTextInput,
    pane: Entity,
    content_root: Entity,
    widget: &Widget,
    font: Handle<Font>,
    content_size: Vec2,
    metrics: &PaneFontMetrics,
) {
    // Clear out any frame children — the overlay owns the content_root
    // subtree while we're in edit mode. We use Commands::despawn rather
    // than walking Children explicitly; the next rerender (after exit)
    // will rebuild the frame from current_frame.
    commands.entity(content_root).despawn_related::<Children>();

    // Background fills the content area. clip_widget_sprites would
    // clamp us anyway, but starting at the right size avoids the
    // one-frame flash of an oversize sprite.
    let bg = commands
        .spawn((
            ChildOf(content_root),
            Sprite {
                color: Color::srgba(0.08, 0.085, 0.10, 0.92),
                custom_size: Some(content_size),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 0.05),
        ))
        .id();

    let label = commands
        .spawn((
            ChildOf(content_root),
            Text2d::new("Command"),
            TextFont {
                font: font.clone(),
                font_size: EDIT_LABEL_FONT_SIZE,
                ..default()
            },
            TextColor(Color::srgb(0.65, 0.68, 0.74)),
            LineHeight::Px(EDIT_LINE_HEIGHT),
            Anchor::TOP_LEFT,
            Transform::from_xyz(EDIT_PAD, -EDIT_PAD, 0.1),
        ))
        .id();

    let input_y = -(EDIT_PAD + EDIT_LINE_HEIGHT + 4.0);
    let style = TextInputStyle {
        font: font.clone(),
        font_size: EDIT_FONT_SIZE,
        line_height: EDIT_INPUT_H,
        cell_width: metrics.char_width(EDIT_FONT_SIZE),
        color_idle: Color::srgb(0.85, 0.86, 0.90),
        color_focused: Color::srgb(0.97, 0.98, 1.00),
        color_caret: Color::srgb(0.55, 0.85, 1.0),
        color_selection: Color::srgba(0.42, 0.62, 0.92, 0.35),
    };
    let input_width = (content_size.x - 2.0 * EDIT_PAD).max(40.0);
    let command_input = spawn_text_input(
        commands,
        content_root,
        &widget.command,
        style,
        input_width,
        Transform::from_xyz(EDIT_PAD, input_y, 0.2),
    );

    let hint_y = input_y - EDIT_INPUT_H - 4.0;
    let hint = commands
        .spawn((
            ChildOf(content_root),
            Text2d::new("Enter to save, Esc to cancel"),
            TextFont {
                font,
                font_size: EDIT_HINT_FONT_SIZE,
                ..default()
            },
            TextColor(Color::srgb(0.50, 0.52, 0.58)),
            LineHeight::Px(EDIT_LINE_HEIGHT),
            Anchor::TOP_LEFT,
            Transform::from_xyz(EDIT_PAD, hint_y, 0.1),
        ))
        .id();

    commands.entity(pane).insert(WidgetEditMode {
        command_input,
        label,
        hint,
        bg,
    });

    focus_text_input(commands, focused, [], Some(command_input));
}

/// React to Submit/Cancel from the command-edit TextInput. Submit
/// applies the new command and respawns the subprocess; Cancel just
/// tears down the overlay.
fn handle_widget_edit_events(
    mut commands: Commands,
    mut events: MessageReader<TextInputEvent>,
    mut focused: ResMut<FocusedTextInput>,
    text_inputs: Query<&TextInput>,
    mut state_q: Query<(&mut Widget, &mut WidgetRender)>,
    io_q: Query<&WidgetIO>,
    pane_for_input: Query<(Entity, &WidgetEditMode)>,
) {
    for ev in events.read() {
        let (submit, entity_input) = match *ev {
            TextInputEvent::Submit { entity } => (true, entity),
            TextInputEvent::Cancel { entity } => (false, entity),
            TextInputEvent::Changed { .. } => continue,
        };
        let Some((pane, _)) = pane_for_input
            .iter()
            .find(|(_, e)| e.command_input == entity_input)
        else {
            continue;
        };
        // Re-fetch the edit-mode component so we can call exit_edit_mode
        // with a stable reference (the borrow from pane_for_input would
        // conflict with state_q's mutable borrow).
        let edit_snapshot = pane_for_input
            .get(pane)
            .ok()
            .map(|(_, e)| (e.command_input, e.label, e.hint, e.bg));

        if submit {
            let new_cmd = text_inputs
                .get(entity_input)
                .map(|ti| ti.text())
                .unwrap_or_default();
            apply_command_change(&mut commands, pane, new_cmd, &mut state_q, &io_q);
        }

        let Some((ci, lbl, hint, bg)) = edit_snapshot else {
            continue;
        };
        let Ok((_, mut render_state)) = state_q.get_mut(pane) else {
            continue;
        };
        if focused.0 == Some(ci) {
            focus_text_input(&mut commands, &mut focused, [], None);
        }
        commands.entity(ci).despawn();
        commands.entity(lbl).despawn();
        commands.entity(hint).despawn();
        commands.entity(bg).despawn();
        commands.entity(pane).remove::<WidgetEditMode>();
        render_state.last_size = Vec2::ZERO;
    }
}

/// Replace the pane's command with `new_cmd`, kill any running child,
/// and spawn a fresh one. Empty command → placeholder frame.
fn apply_command_change(
    commands: &mut Commands,
    pane: Entity,
    new_cmd: String,
    state_q: &mut Query<(&mut Widget, &mut WidgetRender)>,
    io_q: &Query<&WidgetIO>,
) {
    if let Ok(io) = io_q.get(pane) {
        if let Ok(json) = serde_json::to_string(&HostEvent::Close) {
            let _ = io.tx.send(json);
        }
    }
    // Kill + remove the old child via an exclusive-world hop, since the
    // outer system can't get &mut WidgetProcess (would alias with the
    // other mut queries).
    commands.queue(move |world: &mut World| {
        if let Some(mut wp) = world.get_mut::<WidgetProcess>(pane) {
            let _ = wp.child.kill();
        }
        world
            .entity_mut(pane)
            .remove::<WidgetProcess>()
            .remove::<WidgetIO>();
    });

    let (cmd_str, args_vec, cwd_opt) = {
        let Ok((mut widget, mut render_state)) = state_q.get_mut(pane) else {
            return;
        };
        widget.command = new_cmd.clone();
        widget.last_state = Value::Null;
        render_state.init_sent = false;
        render_state.pending_frame = None;
        if new_cmd.trim().is_empty() {
            render_state.current_frame = Some(placeholder_frame());
            return;
        }
        render_state.current_frame = None;
        (widget.command.clone(), widget.args.clone(), widget.cwd.clone())
    };

    match spawn_widget_process(&cmd_str, &args_vec, cwd_opt.as_deref()) {
        Ok((process, io)) => {
            commands.entity(pane).insert((process, io));
        }
        Err(e) => {
            eprintln!("[widget] respawn failed: {}", e);
            if let Ok((_, mut render_state)) = state_q.get_mut(pane) {
                render_state.current_frame = Some(error_frame(&format!("spawn failed: {}", e)));
            }
        }
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

                match spawn_widget_process(
                    &widget.command,
                    &widget.args,
                    widget.cwd.as_deref(),
                ) {
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
        | Element::ListItem { children, .. } => children
            .iter()
            .find_map(|c| find_input_value(c, target)),
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
        let m = PaneFontMetrics { cell_width: 10.0, font_size: 10.0 };
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
