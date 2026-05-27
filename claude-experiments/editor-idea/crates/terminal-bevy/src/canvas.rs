//! Scrolling canvas — per-project pan + zoom.
//!
//! ## Model
//!
//! - `CanvasPos` is the source-of-truth position/size of every pane in
//!   **canvas-space** (the abstract infinite-plane coords the user thinks
//!   in). At zoom=1, pan=(0,0) canvas-space equals window-space.
//! - `PaneRect` (owned by pane-bevy) is the **screen-space** projection.
//!   All existing chrome / per-pane-camera / hit-test code expects
//!   `PaneRect` in window pixels, so we keep it that way and derive it
//!   from `CanvasPos` through the active project's `CanvasView`.
//! - Sync:
//!   - `PreUpdate::project_canvas_to_pane`: `PaneRect = view.project(CanvasPos)`
//!     for every pane in the active project.
//!   - `PostUpdate::write_back_pane_to_canvas`: if `PaneRect` differs from
//!     `view.project(CanvasPos)` by more than a small tolerance, the user
//!     just dragged / resized — invert-project back into `CanvasPos`.
//!
//! ## Zoom and content
//!
//! Pane chrome (title bar, close button, padding) stays at constant
//! pixel size — chrome is positioned in pixel-space inside the pane,
//! and we leave it alone. Content inside the pane *does* zoom: a
//! per-pane system sets `content_root.scale = view.zoom`. Combined with
//! the per-pane camera viewport scaling with `PaneRect.size`
//! (= `canvas_size * zoom`), content laid out at canvas-units renders
//! at `canvas_units * zoom` screen pixels.
//!
//! ## Per-project state
//!
//! Each project keeps its own `CanvasViewState { pan, zoom }`. Switching
//! projects flips `CanvasView` to the new project's state; panes don't
//! move in canvas-space, but their screen-projection changes because
//! the active view did. Persisted alongside projects.json.
//!
//! ## Configurable input
//!
//! `CanvasConfig` is a runtime-mutable resource listing which gestures
//! are enabled (trackpad scroll, left-drag on empty canvas,
//! middle-mouse drag, space+drag) and whether zoom is on. Cycling
//! through configs lets us A/B different "feels" without rebuilding.

use std::collections::HashMap;

use bevy::input::gestures::PinchGesture;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::sprite::Anchor;
use serde::{Deserialize, Serialize};

use pane_bevy::{
    InputConsumed, PaneCanvasRegion, PaneChrome, PaneInputBlockZones, PaneProject, PaneRect,
    PaneTag,
};

use crate::projects::{Projects, Sidebar};

// ---------- Per-project view state ----------

/// One project's pan/zoom snapshot. Stored both at runtime (under
/// [`CanvasView`]) and serialized into projects.json (see
/// `projects.rs::PersistedCanvas`).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CanvasViewState {
    /// Canvas-space point that should appear at `screen_origin`.
    /// `pan = (0, 0)` means "the canvas origin is at the sidebar's
    /// right edge / top of window".
    pub pan: Vec2,
    /// 1.0 = native pixel-per-canvas-unit. >1 zooms in, <1 zooms out.
    pub zoom: f32,
}

impl Default for CanvasViewState {
    fn default() -> Self {
        Self {
            pan: Vec2::ZERO,
            zoom: 1.0,
        }
    }
}

impl CanvasViewState {
    pub fn clamp_zoom(&mut self) {
        self.zoom = self.zoom.clamp(MIN_ZOOM, MAX_ZOOM);
    }
}

pub const MIN_ZOOM: f32 = 0.2;
pub const MAX_ZOOM: f32 = 4.0;

/// Live per-project view. The active project's state is read by every
/// projection / inverse-projection system; other projects' states sit
/// here until the user switches.
#[derive(Resource, Default)]
pub struct CanvasView {
    pub per_project: HashMap<u64, CanvasViewState>,
}

impl CanvasView {
    pub fn state_for(&self, project_id: u64) -> CanvasViewState {
        self.per_project
            .get(&project_id)
            .copied()
            .unwrap_or_default()
    }
    pub fn state_mut(&mut self, project_id: u64) -> &mut CanvasViewState {
        self.per_project.entry(project_id).or_default()
    }
}

// ---------- Configurable input ----------

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PanGestures {
    /// Two-finger scroll / mouse wheel (no modifier) pans the canvas.
    /// On macOS trackpads this is the most natural gesture.
    pub trackpad_scroll: bool,
    /// Left-click drag on empty canvas (not on a pane) pans.
    pub left_drag_empty: bool,
    /// Middle-mouse drag pans anywhere.
    pub middle_drag: bool,
    /// Hold Space then left-drag anywhere to pan (Photoshop-style).
    pub space_drag: bool,
}

impl Default for PanGestures {
    fn default() -> Self {
        Self {
            trackpad_scroll: true,
            left_drag_empty: false,
            middle_drag: true,
            space_drag: true,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CanvasConfig {
    pub pan: PanGestures,
    /// If false, zoom input is ignored and content_root.scale stays 1.
    pub zoom_enabled: bool,
    /// `cmd` (Super) + scroll wheel zooms. The only zoom input wired
    /// for now; pinch isn't on the menu for v0.
    pub zoom_with_cmd_scroll: bool,
    /// Scale factor per scroll-line unit at zoom_with_cmd_scroll. 1.05
    /// means each "tick" of the wheel multiplies zoom by 1.05.
    pub zoom_per_line: f32,
    /// Trackpad pan sensitivity multiplier (canvas units per scroll px).
    pub trackpad_sensitivity: f32,
    /// Two-finger pinch on macOS trackpad zooms around the cursor.
    #[serde(default = "default_true")]
    pub pinch_to_zoom: bool,
    /// Multiplier on each PinchGesture event's delta. Bevy/winit emits
    /// PinchGesture(f32) where the f32 is a per-event delta in scale
    /// units; sensible amplification keeps the feel responsive without
    /// overshooting on macOS's high-frequency event stream.
    #[serde(default = "default_pinch_gain")]
    pub pinch_gain: f32,
}

fn default_true() -> bool {
    true
}
fn default_pinch_gain() -> f32 {
    3.5
}

impl Default for CanvasConfig {
    fn default() -> Self {
        Self {
            pan: PanGestures::default(),
            zoom_enabled: true,
            zoom_with_cmd_scroll: true,
            zoom_per_line: 1.08,
            trackpad_sensitivity: 1.0,
            pinch_to_zoom: true,
            pinch_gain: default_pinch_gain(),
        }
    }
}

// ---------- Components ----------

/// Source-of-truth canvas-space rect for a pane. PaneRect (screen
/// space) is recomputed from this every PreUpdate.
#[derive(Component, Copy, Clone, Debug)]
pub struct CanvasPos {
    pub pos: Vec2,
    pub size: Vec2,
}

// ---------- Mouse drag state ----------

#[derive(Resource, Default)]
struct PanDragState {
    /// Active pan-drag (button held + dragging). None when idle.
    active: Option<PanDragKind>,
    /// Mouse position when drag started (window coords).
    last_pt: Vec2,
}

#[derive(Clone, Copy, Debug)]
enum PanDragKind {
    LeftEmpty,
    Middle,
    SpaceLeft,
}

// ---------- Plugin ----------

pub struct CanvasPlugin;

impl Plugin for CanvasPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CanvasView::default())
            .insert_resource(CanvasConfig::default())
            .insert_resource(PanDragState::default())
            // Mouse / wheel pan + zoom. Runs in Update before pane-bevy's
            // chrome drag handler so we can claim wheel events and
            // (optionally) empty-canvas clicks.
            .add_systems(
                Update,
                (
                    init_canvas_pos_for_new_panes,
                    handle_pan_zoom_input,
                    cycle_config_hotkey,
                )
                    .chain(),
            )
            // Project canvas → PaneRect in PreUpdate so all downstream
            // chrome / camera / hit-test consumers see screen-space rects
            // this frame. Also publish the canvas region (= window minus
            // sidebar) so pane-bevy clips per-pane camera viewports to
            // it; that's what makes the sidebar visually sit on top of
            // the panes regardless of pan/zoom.
            .add_systems(PreUpdate, (publish_canvas_region, project_canvas_to_pane).chain())
            // After drag/resize lands in PaneRect, invert back to canvas.
            // After all chrome/camera work, set content_root.scale.
            .add_systems(
                PostUpdate,
                (
                    write_back_pane_to_canvas,
                    sync_content_root_scale,
                    sync_origin_indicators,
                ),
            );
    }
}

// ---------- Projection helpers ----------

/// World/canvas coordinate to window/screen coordinate.
///
/// `screen_origin` is the canvas-window pixel where canvas (0,0) maps
/// to when pan=(0,0). For us that's (sidebar_width, 0) so panes don't
/// overlap the sidebar at the default view.
fn project_pos(canvas_pos: Vec2, view: &CanvasViewState, screen_origin: Vec2) -> Vec2 {
    (canvas_pos - view.pan) * view.zoom + screen_origin
}

fn project_size(canvas_size: Vec2, view: &CanvasViewState) -> Vec2 {
    canvas_size * view.zoom
}

fn unproject_pos(screen_pos: Vec2, view: &CanvasViewState, screen_origin: Vec2) -> Vec2 {
    (screen_pos - screen_origin) / view.zoom.max(0.0001) + view.pan
}

fn unproject_size(screen_size: Vec2, view: &CanvasViewState) -> Vec2 {
    screen_size / view.zoom.max(0.0001)
}

fn screen_origin(sidebar: &Sidebar) -> Vec2 {
    Vec2::new(sidebar.width, 0.0)
}

// ---------- Systems ----------

/// Any new pane that has `PaneRect` but no `CanvasPos` gets initialized
/// with `CanvasPos = unproject(PaneRect)`. New panes spawned by the
/// host (cascade_pos, restore) supply PaneRect in window/screen coords;
/// at zoom=1, pan=(0,0) that equals canvas coords minus screen_origin.
fn init_canvas_pos_for_new_panes(
    mut commands: Commands,
    panes: Query<(Entity, &PaneRect, Option<&PaneProject>), (With<PaneTag>, Without<CanvasPos>)>,
    view: Res<CanvasView>,
    sidebar: Res<Sidebar>,
) {
    if panes.is_empty() {
        return;
    }
    let origin = screen_origin(&sidebar);
    for (e, rect, proj) in &panes {
        let state = proj
            .map(|p| view.state_for(p.0))
            .unwrap_or_default();
        let canvas_pos = unproject_pos(rect.pos, &state, origin);
        let canvas_size = unproject_size(rect.size, &state);
        commands.entity(e).insert(CanvasPos {
            pos: canvas_pos,
            size: canvas_size,
        });
    }
}

/// PreUpdate: write `PaneRect = project(CanvasPos)` for every pane in
/// the active project. Skip panes belonging to other projects — their
/// PaneRect would jump to a stale projection that nobody renders
/// anyway (they're Visibility::Hidden), and skipping avoids spurious
/// Changed<PaneRect> ticks.
/// Tell pane-bevy where the canvas is, so per-pane cameras clip their
/// viewports to that region. Anything drawn by the main camera (the
/// sidebar) outside this region sits visually on top of the panes.
fn publish_canvas_region(
    windows: Query<&Window>,
    sidebar: Res<Sidebar>,
    mut region: ResMut<PaneCanvasRegion>,
    mut block_zones: ResMut<PaneInputBlockZones>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let want = PaneCanvasRegion {
        min: Vec2::new(sidebar.width, 0.0),
        max: Vec2::new(window.width(), window.height()),
        active: true,
    };
    if region.min != want.min || region.max != want.max || region.active != want.active {
        *region = want;
    }
    // Also block pane-bevy's mouse hit-testing from reaching whatever
    // PaneRect happens to extend under the sidebar after a pan — without
    // this the sidebar would *look* on top (cameras clipped) but a click
    // in it could still drag a hidden pane.
    let sidebar_rect = Rect {
        min: Vec2::ZERO,
        max: Vec2::new(sidebar.width, window.height()),
    };
    block_zones.0.clear();
    block_zones.0.push(sidebar_rect);
}

fn project_canvas_to_pane(
    mut panes: Query<(&CanvasPos, &mut PaneRect, Option<&PaneProject>), With<PaneTag>>,
    view: Res<CanvasView>,
    sidebar: Res<Sidebar>,
    projects: Res<Projects>,
) {
    let Some(active) = projects.active else {
        return;
    };
    let origin = screen_origin(&sidebar);
    let state = view.state_for(active);
    for (cpos, mut rect, proj) in &mut panes {
        let pid = match proj {
            Some(p) => p.0,
            None => continue,
        };
        if pid != active {
            continue;
        }
        let want_pos = project_pos(cpos.pos, &state, origin);
        let want_size = project_size(cpos.size, &state);
        if (rect.pos - want_pos).length_squared() > 0.01
            || (rect.size - want_size).length_squared() > 0.01
        {
            rect.pos = want_pos;
            rect.size = want_size;
        }
    }
}

/// PostUpdate: any pane whose PaneRect drifted from project(CanvasPos)
/// got moved by pane-bevy's drag/resize handler — capture that into
/// CanvasPos via inverse projection. Skips panes outside the active
/// project (they can't be dragged, their PaneRect is stale anyway).
fn write_back_pane_to_canvas(
    mut panes: Query<(&PaneRect, &mut CanvasPos, Option<&PaneProject>), With<PaneTag>>,
    view: Res<CanvasView>,
    sidebar: Res<Sidebar>,
    projects: Res<Projects>,
) {
    // CRITICAL: if the view changed this frame (pan/zoom input), PaneRect
    // still reflects the OLD projection — PreUpdate ran before the input.
    // The "drift" we'd see here isn't a user drag, it's just stale rect.
    // Don't write back; next PreUpdate will re-project with the new view.
    if view.is_changed() || projects.is_changed() {
        return;
    }
    let Some(active) = projects.active else {
        return;
    };
    let origin = screen_origin(&sidebar);
    let state = view.state_for(active);
    for (rect, mut cpos, proj) in &mut panes {
        let pid = match proj {
            Some(p) => p.0,
            None => continue,
        };
        if pid != active {
            continue;
        }
        let want_pos = project_pos(cpos.pos, &state, origin);
        let want_size = project_size(cpos.size, &state);
        let pos_drift = (rect.pos - want_pos).length_squared();
        let size_drift = (rect.size - want_size).length_squared();
        // Drift > 0.25 px² = user dragged/resized; inverse-project.
        // Smaller drift is float noise from the project step.
        if pos_drift > 0.25 {
            cpos.pos = unproject_pos(rect.pos, &state, origin);
        }
        if size_drift > 0.25 {
            cpos.size = unproject_size(rect.size, &state);
        }
    }
}

/// Set each pane's `content_root` Transform.scale to the active
/// project's zoom. Content laid out at canvas-units therefore renders
/// at `canvas_units * zoom` world units, which (with the per-pane
/// camera's default 1:1 projection over the viewport sized to
/// `canvas_size * zoom` pixels) means content visually scales with the
/// pane.
fn sync_content_root_scale(
    panes: Query<(&PaneChrome, Option<&PaneProject>), With<PaneTag>>,
    mut t_q: Query<&mut Transform>,
    view: Res<CanvasView>,
    projects: Res<Projects>,
    config: Res<CanvasConfig>,
) {
    let Some(active) = projects.active else {
        return;
    };
    let zoom = if config.zoom_enabled {
        view.state_for(active).zoom
    } else {
        1.0
    };
    for (chrome, proj) in &panes {
        // Other projects' panes are hidden; their content_root scale is
        // a don't-care, but we'll set it to their project's zoom anyway
        // so a future "show all projects" doesn't surprise.
        let pid = proj.map(|p| p.0).unwrap_or(active);
        let pane_zoom = if pid == active {
            zoom
        } else if config.zoom_enabled {
            view.state_for(pid).zoom
        } else {
            1.0
        };
        if let Ok(mut t) = t_q.get_mut(chrome.content_root) {
            if (t.scale.x - pane_zoom).abs() > 0.001 || (t.scale.y - pane_zoom).abs() > 0.001 {
                t.scale.x = pane_zoom;
                t.scale.y = pane_zoom;
            }
            // content_root's translation is (MARGIN, -(TITLE_H + MARGIN)) in
            // pixel-space — i.e. the chrome inset. Those stay unscaled
            // (chrome inset is constant pixels). The scale only affects
            // descendants' positions/sizes inside the content area.
        }
    }
}

// ---------- Input ----------

/// All pan / zoom input in one system so we can route a single
/// MouseWheel event to either pan or zoom based on modifiers without
/// double-handling.
#[allow(clippy::too_many_arguments)]
fn handle_pan_zoom_input(
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: MessageReader<MouseWheel>,
    mut pinch: MessageReader<PinchGesture>,
    mut consumed: ResMut<InputConsumed>,
    block_zones: Res<PaneInputBlockZones>,
    sidebar: Res<Sidebar>,
    config: Res<CanvasConfig>,
    projects: Res<Projects>,
    mut view: ResMut<CanvasView>,
    mut drag: ResMut<PanDragState>,
    panes: Query<(&PaneRect, Option<&Visibility>), With<PaneTag>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        // Still consume any wheel events even with no cursor so they
        // don't bleed into the terminal scroll handler.
        wheel.clear();
        pinch.clear();
        return;
    };
    let Some(active) = projects.active else {
        return;
    };

    // Sidebar (or any host block-zone) eats canvas input.
    let in_block_zone = pt.x < sidebar.width
        || block_zones
            .0
            .iter()
            .any(|r| pt.x >= r.min.x && pt.x <= r.max.x && pt.y >= r.min.y && pt.y <= r.max.y);

    // ----- Wheel: cmd+scroll = canvas pan, cmd+opt+scroll = canvas zoom -----
    //
    // Plain wheel is left for the pane under the cursor (terminal
    // scrollback, run-button output, etc.) — `handle_scroll` and
    // `scroll_run_button_output` early-return when cmd is held, so the
    // same wheel event never both pans and scrolls a pane.

    let cmd_held = keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight);
    let alt_held = keys.pressed(KeyCode::AltLeft) || keys.pressed(KeyCode::AltRight);
    let mut wheel_total = Vec2::ZERO;
    let mut had_wheel = false;
    for ev in wheel.read() {
        let scale = match ev.unit {
            MouseScrollUnit::Line => 16.0,
            MouseScrollUnit::Pixel => 1.0,
        };
        wheel_total.x += ev.x * scale;
        wheel_total.y += ev.y * scale;
        had_wheel = true;
    }
    if had_wheel && cmd_held && !in_block_zone {
        if alt_held && config.zoom_enabled && config.zoom_with_cmd_scroll {
            let origin = screen_origin(&sidebar);
            let state = view.state_mut(active);
            // Zoom factor from total wheel y. Each "line" (16 px after
            // our scale) multiplies by zoom_per_line.
            let lines = wheel_total.y / 16.0;
            let factor = config.zoom_per_line.powf(lines);
            let canvas_pt_before = unproject_pos(pt, state, origin);
            state.zoom = (state.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
            let canvas_pt_after = unproject_pos(pt, state, origin);
            // Anchor zoom on cursor: keep the canvas point under the
            // cursor fixed by adjusting pan.
            state.pan += canvas_pt_before - canvas_pt_after;
            consumed.0 = true;
        } else if config.pan.trackpad_scroll {
            let state = view.state_mut(active);
            // Viewport-pan vertical: two-finger swipe up = view moves up
            // (panes appear to scroll down on screen). Horizontal keeps
            // the "drag canvas with fingers" sign.
            let pan_delta = Vec2::new(-wheel_total.x, -wheel_total.y)
                * config.trackpad_sensitivity
                / state.zoom.max(0.0001);
            state.pan += pan_delta;
            consumed.0 = true;
        }
    }

    // ----- Drag-pan: start -----

    if drag.active.is_none() && !in_block_zone {
        if config.pan.middle_drag && buttons.just_pressed(MouseButton::Middle) {
            drag.active = Some(PanDragKind::Middle);
            drag.last_pt = pt;
            consumed.0 = true;
        } else if config.pan.space_drag
            && keys.pressed(KeyCode::Space)
            && buttons.just_pressed(MouseButton::Left)
        {
            drag.active = Some(PanDragKind::SpaceLeft);
            drag.last_pt = pt;
            consumed.0 = true;
        } else if config.pan.left_drag_empty
            && buttons.just_pressed(MouseButton::Left)
            && !consumed.0
        {
            // Only when the click DIDN'T land on a visible pane. Walk
            // the panes to check.
            let on_pane = panes.iter().any(|(r, vis)| {
                !matches!(vis, Some(Visibility::Hidden))
                    && pt.x >= r.pos.x
                    && pt.x <= r.pos.x + r.size.x
                    && pt.y >= r.pos.y
                    && pt.y <= r.pos.y + r.size.y
            });
            if !on_pane {
                drag.active = Some(PanDragKind::LeftEmpty);
                drag.last_pt = pt;
                consumed.0 = true;
            }
        }
    }

    // ----- Drag-pan: continue / end -----

    if let Some(kind) = drag.active {
        let still_held = match kind {
            PanDragKind::LeftEmpty | PanDragKind::SpaceLeft => {
                buttons.pressed(MouseButton::Left)
            }
            PanDragKind::Middle => buttons.pressed(MouseButton::Middle),
        };
        if !still_held {
            drag.active = None;
        } else {
            let delta_screen = pt - drag.last_pt;
            drag.last_pt = pt;
            let state = view.state_mut(active);
            // Drag canvas-with-cursor: moving the cursor right pans the
            // canvas right (pan decreases, since pan is "canvas point
            // at screen_origin").
            state.pan -= delta_screen / state.zoom.max(0.0001);
            consumed.0 = true;
        }
    }
}

/// Dev hotkeys to cycle through canvas configs at runtime so we can
/// A/B different setups without rebuilding. Bindings (all require Cmd
/// + Shift to avoid colliding with terminal input):
///
/// - **Cmd+Shift+P** — cycle pan-gesture preset
/// - **Cmd+Shift+Z** — toggle zoom on/off
/// - **Cmd+Shift+0** — reset active project's view (pan=0, zoom=1)
fn cycle_config_hotkey(
    keys: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<CanvasConfig>,
    mut events: MessageReader<KeyboardInput>,
    mut view: ResMut<CanvasView>,
    projects: Res<Projects>,
    mut step: Local<usize>,
) {
    let cmd = keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight);
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let mut fire_pan = false;
    let mut fire_zoom = false;
    let mut fire_reset = false;
    for ev in events.read() {
        if !ev.state.is_pressed() {
            continue;
        }
        if !(cmd && shift) {
            continue;
        }
        match &ev.logical_key {
            Key::Character(s) if s.eq_ignore_ascii_case("p") => fire_pan = true,
            Key::Character(s) if s.eq_ignore_ascii_case("z") => fire_zoom = true,
            Key::Character(s) if s.as_str() == "0" || s.as_str() == ")" => fire_reset = true,
            _ => {}
        }
    }
    if fire_pan {
        *step = (*step + 1) % PAN_PRESETS.len();
        config.pan = PAN_PRESETS[*step].1;
        eprintln!(
            "[canvas] pan preset → {} ({:?})",
            PAN_PRESETS[*step].0, config.pan
        );
    }
    if fire_zoom {
        config.zoom_enabled = !config.zoom_enabled;
        eprintln!("[canvas] zoom_enabled = {}", config.zoom_enabled);
    }
    if fire_reset {
        if let Some(id) = projects.active {
            let state = view.state_mut(id);
            state.pan = Vec2::ZERO;
            state.zoom = 1.0;
            eprintln!("[canvas] view reset for project {}", id);
        }
    }
}

// ---------- Origin indicators ----------
//
// Four edge strips that fade in as the user pans away from canvas origin.
// `pan.x > 0` means the user is looking right of origin → origin is OFF
// to the LEFT, so the LEFT strip darkens. Symmetric for the other three.
// Each strip is a stack of thin sprites with linearly-decreasing alpha
// from edge inward, faking a soft gradient using only flat-color sprites.

const ORIGIN_STRIP_LAYERS: usize = 10;
const ORIGIN_STRIP_THICKNESS: f32 = 140.0;
/// Canvas units past the viewport edge at which an edge's vignette
/// saturates. Smaller = more responsive feedback as you pan.
const ORIGIN_SATURATE_AT: f32 = 600.0;
/// Peak alpha of an edge strip's outermost layer at full saturation.
const ORIGIN_PEAK_INTENSITY: f32 = 1.1;
/// Z above the canvas clear color but below the per-pane cameras (which
/// have order ≥ 1, drawn on top regardless).
const ORIGIN_STRIP_Z: f32 = 0.5;

#[derive(Component, Copy, Clone, Debug)]
enum OriginEdge {
    Top,
    Bottom,
    Left,
    Right,
}

#[derive(Component)]
struct OriginStripLayer {
    edge: OriginEdge,
    /// 0 = at the edge (darkest), ORIGIN_STRIP_LAYERS-1 = innermost (lightest).
    rank: usize,
}

#[derive(Resource, Default)]
struct OriginIndicatorsInit(bool);

/// Each frame, recompute alpha + geometry for the four edge strips so
/// they track sidebar width, window resize, and pan magnitude. Spawns
/// the strip entities lazily on the first call. Color comes from the
/// active theme's `PANE_SHADOW_COLOR` — themes that ship a dark shadow
/// give the canvas a dark vignette; light themes get a soft one.
fn sync_origin_indicators(
    mut commands: Commands,
    windows: Query<&Window>,
    sidebar: Res<Sidebar>,
    view: Res<CanvasView>,
    projects: Res<Projects>,
    theme: Res<style_bevy::Theme>,
    mut init: Local<OriginIndicatorsInit>,
    mut strips: Query<(&OriginStripLayer, &mut Sprite, &mut Transform)>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let win_w = window.width();
    let win_h = window.height();
    let canvas_left = sidebar.width;
    let canvas_right = win_w;
    let canvas_w = (canvas_right - canvas_left).max(1.0);
    let world_canvas_left = canvas_left - win_w * 0.5;
    let world_top = win_h * 0.5;

    if !init.0 {
        init.0 = true;
        let dark = Color::srgba(0.0, 0.0, 0.0, 0.0);
        for edge in [OriginEdge::Top, OriginEdge::Bottom, OriginEdge::Left, OriginEdge::Right] {
            for rank in 0..ORIGIN_STRIP_LAYERS {
                commands.spawn((
                    OriginStripLayer { edge, rank },
                    Sprite {
                        color: dark,
                        custom_size: Some(Vec2::new(1.0, 1.0)),
                        ..default()
                    },
                    Anchor::TOP_LEFT,
                    Transform::from_xyz(0.0, 0.0, ORIGIN_STRIP_Z),
                    bevy::camera::visibility::RenderLayers::layer(0),
                ));
            }
        }
        // Bail this frame — the spawned entities aren't queryable yet.
        return;
    }

    let active = projects.active;
    let state = active.map(|a| view.state_for(a)).unwrap_or_default();
    let pan = state.pan;
    let zoom = state.zoom.max(0.0001);

    // Canvas-space viewport: top-left = pan, size = (canvas_w / zoom,
    // win_h / zoom). When canvas-origin (0,0) sits inside that rect the
    // user can see it on screen — no vignette in that case. Each edge's
    // depth is how far origin is past that edge, in canvas units, mapped
    // to [0, 1] via ORIGIN_SATURATE_AT.
    let view_w = canvas_w / zoom;
    let view_h = win_h / zoom;
    let depth_left = (pan.x.max(0.0) / ORIGIN_SATURATE_AT).min(1.0);
    let depth_right =
        ((-(pan.x + view_w)).max(0.0) / ORIGIN_SATURATE_AT).min(1.0);
    let depth_top = (pan.y.max(0.0) / ORIGIN_SATURATE_AT).min(1.0);
    let depth_bottom =
        ((-(pan.y + view_h)).max(0.0) / ORIGIN_SATURATE_AT).min(1.0);

    // Theme-driven base color. We pull from `PANE_SHADOW_COLOR` because
    // it's already a soft dark across every preset (and inverts for
    // light themes), so the canvas vignette stays coherent with the
    // panes' drop shadows. Alpha is ours; the theme just supplies a hue.
    let theme_rgba = theme.color(style_bevy::tokens::PANE_SHADOW_COLOR);
    let base_rgb = (theme_rgba.red, theme_rgba.green, theme_rgba.blue);

    let layer_thickness = ORIGIN_STRIP_THICKNESS / ORIGIN_STRIP_LAYERS as f32;

    for (info, mut sprite, mut t) in &mut strips {
        let directional = match info.edge {
            OriginEdge::Top => depth_top,
            OriginEdge::Bottom => depth_bottom,
            OriginEdge::Left => depth_left,
            OriginEdge::Right => depth_right,
        };
        // Edge layer (rank 0) is darkest; innermost layer fades to 0.
        let rank_t = info.rank as f32 / (ORIGIN_STRIP_LAYERS - 1).max(1) as f32;
        let falloff = (1.0 - rank_t).powf(1.5); // ease-out so the inner
        // layers fade faster than linear; gives a softer gradient feel.
        let alpha = (directional * ORIGIN_PEAK_INTENSITY * falloff).clamp(0.0, 1.0);
        let want_color =
            Color::srgba(base_rgb.0, base_rgb.1, base_rgb.2, alpha);
        if sprite.color != want_color {
            sprite.color = want_color;
        }
        // Geometry: each rank-th layer sits one `layer_thickness` further
        // from its edge, layer i covers [i*lt, (i+1)*lt] along the
        // perpendicular axis.
        let inset = info.rank as f32 * layer_thickness;
        let (size, world_pos) = match info.edge {
            OriginEdge::Top => (
                Vec2::new(canvas_w, layer_thickness),
                Vec2::new(world_canvas_left, world_top - inset),
            ),
            OriginEdge::Bottom => (
                Vec2::new(canvas_w, layer_thickness),
                Vec2::new(world_canvas_left, -world_top + inset + layer_thickness),
            ),
            OriginEdge::Left => (
                Vec2::new(layer_thickness, win_h),
                Vec2::new(world_canvas_left + inset, world_top),
            ),
            OriginEdge::Right => (
                Vec2::new(layer_thickness, win_h),
                Vec2::new(world_canvas_left + canvas_w - inset - layer_thickness, world_top),
            ),
        };
        if sprite.custom_size != Some(size) {
            sprite.custom_size = Some(size);
        }
        let want_t = Vec3::new(world_pos.x, world_pos.y, ORIGIN_STRIP_Z);
        if t.translation != want_t {
            t.translation = want_t;
        }
    }
}

const PAN_PRESETS: &[(&str, PanGestures)] = &[
    (
        "trackpad+middle+space",
        PanGestures {
            trackpad_scroll: true,
            left_drag_empty: false,
            middle_drag: true,
            space_drag: true,
        },
    ),
    (
        "trackpad-only",
        PanGestures {
            trackpad_scroll: true,
            left_drag_empty: false,
            middle_drag: false,
            space_drag: false,
        },
    ),
    (
        "left-drag-empty",
        PanGestures {
            trackpad_scroll: true,
            left_drag_empty: true,
            middle_drag: false,
            space_drag: false,
        },
    ),
    (
        "middle-only",
        PanGestures {
            trackpad_scroll: false,
            left_drag_empty: false,
            middle_drag: true,
            space_drag: false,
        },
    ),
    (
        "space-drag-only",
        PanGestures {
            trackpad_scroll: false,
            left_drag_empty: false,
            middle_drag: false,
            space_drag: true,
        },
    ),
    (
        "everything",
        PanGestures {
            trackpad_scroll: true,
            left_drag_empty: true,
            middle_drag: true,
            space_drag: true,
        },
    ),
];
