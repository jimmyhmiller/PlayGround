//! Shared chrome + lifecycle for floating widget panes.
//!
//! A "pane" here is the window-like rectangle that hosts a widget on
//! the canvas: terminal panes, editor panes, run-button panes, and any
//! future widget type. Pane-bevy owns everything that's the same across
//! kinds — drag-by-title-bar, drag-corner-to-resize, close button,
//! z-ordering, focus, hit-testing — so a new widget type only writes
//! its content rendering and its content-area input handler.
//!
//! # Adding a new kind
//!
//! 1. Implement two functions and a struct:
//!    - `fn spawn(world, entity, content_root, &Value)` — populate the
//!      pane entity with your kind's components and add content children.
//!    - `fn snapshot(world, entity) -> Value` — serialize whatever your
//!      kind needs to restore (file path, command string, session id, …).
//!    - `PaneKindSpec { kind, display_name, radial_icon, default_size,
//!       spawn, snapshot, on_close }`.
//! 2. In your plugin's `build`, push the spec into `PaneRegistry`.
//! 3. Add systems for content rendering + per-kind input. Read
//!    `PaneContentPressed`/`PaneContentDragged` events scoped to your
//!    kind via the `PaneKindMarker` component on the pane entity.
//!
//! That's it — chrome, drag/resize, focus, persistence, radial menu
//! entry are all wired up by `PanePlugin`.
//!
//! # Clipping
//!
//! Pane content is clipped at pane edges by the *renderer*, not by
//! per-kind discipline. Each pane gets its own [`Camera2d`] with a
//! viewport restricted to the pane's screen rect and a unique
//! [`bevy::camera::visibility::RenderLayers`] id (allocated by
//! [`PaneLayerAllocator`]). The main camera renders layer 0 (canvas
//! background, sidebar, anything non-pane); each pane camera renders
//! only its own pane's layer. Chrome (bg sprite, title text, close
//! button, resize handle) AND content are all on the pane's layer, so
//! the pane camera draws the pane's entire visible rectangle and the
//! GPU refuses to write fragments outside the viewport.
//!
//! Pane z-order is realised through `Camera.order` — the higher-z
//! pane's camera runs later and overdraws lower-z panes wherever they
//! overlap.
//!
//! The propagation system in [`mod@camera`] stamps the right
//! `RenderLayers` on every descendant of `content_root` (and on every
//! newly-added child anywhere under the pane), so kinds add children
//! exactly as they always did — no clip-awareness required at the
//! kind layer.
//!
//! [`enforce_pane_content_bounds`] is no longer the clipping
//! mechanism; it survives only to set `TextBounds` for kinds that use
//! wrap-mode text and need the wrap width.

use std::collections::HashMap;

use bevy::input::mouse::MouseButton;
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::{LineHeight, TextBounds};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub mod camera;
pub mod chrome_material;
pub mod layers;
pub mod text_input;

pub use camera::{PaneCameraOf, PaneCanvasRegion};
pub use chrome_material::{
    ActiveChromeShader, ChromeMaterialPlugin, ChromeParams, ChromeStyle, ChromeTextStyle,
    PaneChromeMaterial, PaneShadowMaterial, ShadowParams,
};
pub use layers::{PaneLayer, PaneLayerAllocator};

use bevy::camera::CameraUpdateSystems;
use bevy::transform::TransformSystems;
pub use text_input::{
    col_at_x, click_to_caret, focus_text_input, spawn_text_input, spawn_text_input_multiline,
    FocusedTextInput, TextInput, TextInputEvent, TextInputFocused, TextInputPlugin,
    TextInputStyle, TextInputView,
};

pub const TITLE_H: f32 = 22.0;
pub const MARGIN: f32 = 8.0;
pub const HANDLE_SIZE: f32 = 14.0;
pub const CLOSE_BTN_SIZE: f32 = 14.0;
pub const CLOSE_BTN_INSET: f32 = 4.0;
pub const MIN_PANE_SIZE: Vec2 = Vec2::new(160.0, 120.0);

// Pane body color lives in `chrome_material::ChromeStyle` (SDF
// uniform). Text-side chrome colors (title / close / handle /
// divider) live in `chrome_material::ChromeTextStyle`, also theme-
// driven via the style-bevy → pane-bevy bridge. The spawn site reads
// the resource for initial colors; `sync_chrome_text_colors` reapplies
// them whenever the resource OR focus state changes.
const TITLE_FONT_SIZE: f32 = 12.0;
const TITLE_DIVIDER_H_FOCUSED: f32 = 2.0;

// ---------- Components ----------

/// Marker that an entity is a pane (any kind). Use this to query "all
/// panes regardless of kind" — e.g., for hit-testing, persistence,
/// visibility-by-project. Combine with `PaneKindMarker` to filter to a
/// specific kind.
#[derive(Component)]
pub struct PaneTag;

/// What kind this pane is. The string matches a `PaneKindSpec.kind` in
/// the registry. Stored as a string so kinds can be registered by
/// crates that don't know about each other.
#[derive(Component, Clone, Debug)]
pub struct PaneKindMarker(pub &'static str);

/// Position, size, z in window-space coords (top-left origin, y-down).
/// `position_root` converts this to the pane's world `Transform` each
/// frame.
#[derive(Component, Copy, Clone, Debug)]
pub struct PaneRect {
    pub pos: Vec2,
    pub size: Vec2,
    pub z: f32,
}

/// Optional human-readable title shown in the title bar. Update by
/// `commands.entity(pane).insert(PaneTitle("…".into()))`; the title
/// system rebuilds the text entity each time it `Changed`.
#[derive(Component, Clone, Debug, Default)]
pub struct PaneTitle(pub String);

/// References to the chrome child entities so systems don't have to
/// re-walk the scene graph each frame.
#[derive(Component)]
pub struct PaneChrome {
    pub bg: Entity,
    /// Drop-shadow quad. Lives on `RenderLayers::layer(0)` (main
    /// camera) so it can extend past the per-pane camera's viewport
    /// clip. Sized larger than the pane by `shadow_blur` on each side.
    pub shadow: Entity,
    pub title_bar: Entity,
    pub title_text: Entity,
    /// Opaque sprite that sits at `z > content_root` so any pane
    /// content scrolled up into the title area gets painted over.
    /// Inset by `corner_radius` on the x axis so it doesn't square
    /// off the chrome's rounded top corners; content is also inset
    /// by `MARGIN > corner_radius`, so no widget pixels can leak
    /// through the uncovered corner regions.
    pub title_cover: Entity,
    pub content_root: Entity,
    pub close_button: Entity,
}

/// Membership in a host-defined "project" / workspace bucket. Pane-bevy
/// itself doesn't interpret this — the host queries it for visibility,
/// persistence partitioning, etc. Optional so standalone demos can
/// spawn panes without a project system.
#[derive(Component, Copy, Clone, Debug)]
pub struct PaneProject(pub u64);

/// Marker: this pane is "pinned to the background". Drag/resize/focus
/// are suppressed, `bring_to_front` never bumps its z, its chrome
/// (title bar, close button, resize handle) is hidden, and its z is
/// forced to 0 so it always sits below any unpinned pane. Empty-space
/// left-clicks fall through to whatever pane (or canvas) is under it.
///
/// Interactive elements (buttons, links, list rows, etc.) remain
/// clickable: each kind publishes hit rects to [`PaneHotZones`] every
/// frame, and pinned-pane hit-testing only fires `PaneContentPressed`
/// when the click lands inside one of those zones. Right-click still
/// goes to the host's context menu (which handles unpinning).
#[derive(Component, Copy, Clone, Debug, Default)]
pub struct PanePinned;

/// Per-pane list of interactive rects in **content-local visual coords**
/// (same frame as [`PaneContentPressed::local_pt`] — i.e. with the
/// kind's scroll offset already subtracted). Any kind that has
/// interactive elements (buttons, links, list rows, input fields)
/// MUST mirror those rects here each frame so pinned-pane hit-testing
/// can route clicks through to them while letting empty space fall
/// through.
///
/// Unpinned panes are unaffected by this list: the chrome hit-tester
/// uses the full pane bounds for them. Kinds with no interactive
/// elements (terminal, editor) can leave this absent / empty — when
/// pinned they will simply pass all clicks through.
#[derive(Component, Default, Debug)]
pub struct PaneHotZones(pub Vec<Rect>);

impl PaneHotZones {
    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn push(&mut self, rect: Rect) {
        self.0.push(rect);
    }

    pub fn contains(&self, pt: Vec2) -> bool {
        self.0.iter().any(|r| r.contains(pt))
    }
}

// ---------- Resources ----------

/// Font used for chrome text (title bar, close button glyph). Host
/// must insert this before any pane is spawned.
#[derive(Resource)]
pub struct PaneFont(pub Handle<Font>);

/// Real per-character advance for `PaneFont`, measured by the host
/// (typically via `skrifa` on the actual font bytes). Kinds that need
/// to size text inline — for truncation, hit-testing, etc. — should
/// read this rather than approximating with a `chars * size * ratio`
/// constant. `cell_width` is measured at `font_size` px; scale linearly
/// for other sizes (the font is monospace).
#[derive(Resource, Clone, Copy, Debug)]
pub struct PaneFontMetrics {
    pub cell_width: f32,
    pub font_size: f32,
}

impl PaneFontMetrics {
    /// Approximate width of `s` rendered at `size` px. Linear scaling
    /// from the measured `(cell_width, font_size)` pair — exact for the
    /// monospace fonts the host ships.
    pub fn measure(&self, s: &str, size: f32) -> f32 {
        let scale = size / self.font_size.max(1.0);
        s.chars().count() as f32 * self.cell_width * scale
    }

    pub fn char_width(&self, size: f32) -> f32 {
        self.cell_width * (size / self.font_size.max(1.0))
    }
}

/// Currently-focused pane (any kind). Replaces per-kind FocusedFoo
/// resources. Kind plugins read this from their keyboard handlers and
/// early-return if the focused entity isn't of their kind.
#[derive(Resource, Default)]
pub struct FocusedPane(pub Option<Entity>);

/// Set by an input handler that just claimed the current frame's mouse
/// click so other handlers don't double-process it. Reset to false in
/// PostUpdate.
#[derive(Resource, Default)]
pub struct InputConsumed(pub bool);

/// What the left mouse button is doing right now, at the pane-chrome
/// level. Per-kind content interactions (text-select, cell-select,
/// button-press) keep their own state on the pane entity — this enum
/// only covers chrome-owned modes.
#[derive(Resource, Default)]
pub enum PaneMouseMode {
    #[default]
    Idle,
    WindowDrag {
        pane: Entity,
        grab_offset: Vec2,
    },
    WindowResize {
        pane: Entity,
        /// Which edges of the rect are being dragged. Any subset of
        /// {N, S, E, W} — corners set two adjacent edges.
        edges: ResizeDir,
        /// Mouse position at press time.
        start_pt: Vec2,
        /// Pane position at press time.
        start_pos: Vec2,
        /// Pane size at press time.
        start_size: Vec2,
    },
    /// Mouse is held after a content-area press. Drives
    /// [`PaneContentDragged`] each frame and a [`PaneContentReleased`]
    /// on button release. Per-kind systems own the actual drag effect
    /// (lift a chess piece, paint a stroke, etc.); pane-bevy only
    /// emits the events.
    ContentDrag {
        pane: Entity,
        /// True iff the originating press came via the pinned-pane
        /// path. Forwarded on every drag/release event so handlers
        /// keep the same pinned semantics.
        pinned: bool,
    },
}

/// Which edges of the pane are being dragged during a resize. Corners
/// set two adjacent edges (e.g. SE → south + east).
#[derive(Copy, Clone, Default, Debug)]
pub struct ResizeDir {
    pub north: bool,
    pub south: bool,
    pub east: bool,
    pub west: bool,
}

impl ResizeDir {
    pub fn any(&self) -> bool {
        self.north || self.south || self.east || self.west
    }
}

/// Side-channel for actions that need exclusive World access (close
/// runs the kind's on_close callback then despawns the entity;
/// pin/unpin tweaks z and toggles the [`PanePinned`] marker, which
/// needs `&mut World` so the z update and component flip land in the
/// same frame).
#[derive(Resource, Default)]
pub struct PendingPaneActions {
    pub close: Vec<Entity>,
    /// Pin to background: insert `PanePinned`, force z = 0.
    pub pin: Vec<Entity>,
    /// Unpin: remove `PanePinned`, bump z above all unpinned panes.
    pub unpin: Vec<Entity>,
}

/// Host-published canvas viewport. `PaneRect.pos/size` lives in
/// canvas-units (zoom + pan invariant) — this resource carries the
/// transform pane-bevy applies to map between canvas-space and window
/// pixels (for hit-testing, per-pane camera viewports, and the pane
/// entity Transform's pan+zoom). The host (terminal-bevy's
/// `publish_canvas_region`) updates it once per frame from the active
/// project's `CanvasView`.
///
/// At defaults (`origin = (0,0)`, `pan = (0,0)`, `zoom = 1.0`) the
/// transform is the identity and pane-bevy behaves exactly like the
/// pre-canvas era.
#[derive(Resource, Copy, Clone, Debug)]
pub struct PaneViewport {
    /// Where canvas (0, 0) sits on screen (in window pixels) when `pan = 0`.
    pub origin: Vec2,
    /// Canvas-space point that should appear at `origin` on screen.
    pub pan: Vec2,
    /// Canvas-units → screen-pixels multiplier.
    pub zoom: f32,
}

impl Default for PaneViewport {
    fn default() -> Self {
        Self {
            origin: Vec2::ZERO,
            pan: Vec2::ZERO,
            zoom: 1.0,
        }
    }
}

/// SystemSet covering every pane-bevy system that reads `PaneViewport`
/// inside Update. The host's `publish_canvas_region` schedules itself
/// `.before(PaneViewportReaders)` so that a mid-Update project switch
/// (e.g. a sidebar click) updates `PaneViewport` before pane positions
/// are recomputed — otherwise switching projects shows one frame of
/// the *previous* project's pan/zoom while the panes catch up.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct PaneViewportReaders;

impl PaneViewport {
    /// Window-pixel position of a canvas-space point.
    pub fn canvas_to_window(&self, canvas: Vec2) -> Vec2 {
        self.origin + (canvas - self.pan) * self.zoom
    }
    /// Canvas-space point under a window-pixel cursor position.
    pub fn window_to_canvas(&self, window_pt: Vec2) -> Vec2 {
        (window_pt - self.origin) / self.zoom.max(0.0001) + self.pan
    }
    /// Project a canvas-space [`PaneRect`] into a screen-space rect
    /// (used by per-pane camera viewport math).
    pub fn projected_rect(&self, rect: &PaneRect) -> PaneRect {
        PaneRect {
            pos: self.canvas_to_window(rect.pos),
            size: rect.size * self.zoom,
            z: rect.z,
        }
    }
}

/// Backwards-compatible alias: a few kinds still consume zoom in
/// isolation. Reads `PaneViewport.zoom`. Newly-written code should
/// take `Res<PaneViewport>` directly.
#[derive(Resource, Copy, Clone, Debug)]
pub struct PaneZoom(pub f32);

impl Default for PaneZoom {
    fn default() -> Self {
        Self(1.0)
    }
}

/// Optional rectangular zones in window-space the host wants pane-bevy
/// to ignore for hit-testing (sidebar, top menu, etc.). The host
/// repopulates this each frame; pane-bevy clears nothing on its own.
/// Consumers should think of this as "advisory" — sidebar's own input
/// handler still owns its clicks; this just stops a click that lands
/// in the zone from focusing a pane underneath.
#[derive(Resource, Default)]
pub struct PaneInputBlockZones(pub Vec<Rect>);

// ---------- Registry ----------

pub type PaneSpawnFn = fn(&mut World, Entity, Entity, &Value);
pub type PaneSnapshotFn = fn(&World, Entity) -> Value;
pub type PaneCloseFn = fn(&mut World, Entity);

#[derive(Clone, Copy)]
pub struct PaneKindSpec {
    /// Stable identifier — used as the dispatch key in the registry and
    /// stored in serialized snapshots.
    pub kind: &'static str,
    /// Title shown in the radial menu.
    pub display_name: &'static str,
    /// Optional 2-3 char glyph for the radial menu icon. None hides
    /// this kind from the radial.
    pub radial_icon: Option<&'static str>,
    /// Default rect size when spawned from the radial menu.
    pub default_size: Vec2,
    /// Populate the pane entity with kind-specific components and add
    /// content children under `content_root`. The chrome (bg, title
    /// bar, resize handle, close button) is already spawned by
    /// `spawn_pane` before this runs.
    pub spawn: PaneSpawnFn,
    /// Serialize the kind-specific state to JSON for persistence. Must
    /// pair with a `spawn` that accepts the same JSON shape.
    pub snapshot: PaneSnapshotFn,
    /// Optional teardown — kill workers, delete temp files, etc. Run
    /// before the entity is despawned. None = nothing extra to do.
    pub on_close: Option<PaneCloseFn>,
}

#[derive(Resource, Default)]
pub struct PaneRegistry {
    by_kind: HashMap<&'static str, PaneKindSpec>,
}

impl PaneRegistry {
    pub fn register(&mut self, spec: PaneKindSpec) {
        if self.by_kind.insert(spec.kind, spec).is_some() {
            panic!("pane kind {:?} registered twice", spec.kind);
        }
    }

    pub fn get(&self, kind: &str) -> Option<&PaneKindSpec> {
        self.by_kind.get(kind)
    }

    pub fn iter(&self) -> impl Iterator<Item = &PaneKindSpec> {
        self.by_kind.values()
    }
}

// ---------- Persistence shape ----------

/// One pane's serialized state. The host writes a `Vec<PaneSnapshot>`
/// to disk; on restart it iterates and calls `spawn_pane` for each
/// (looking the kind up in the registry). The `config` blob is whatever
/// the kind's `snapshot` function returned.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaneSnapshot {
    pub kind: String,
    pub project_id: Option<u64>,
    pub pos: [f32; 2],
    pub size: [f32; 2],
    pub z: f32,
    #[serde(default = "default_value_null")]
    pub config: Value,
    /// True iff the pane was pinned-to-background at snapshot time.
    /// `serde(default)` keeps old saves loadable.
    #[serde(default)]
    pub pinned: bool,
}

fn default_value_null() -> Value {
    Value::Null
}

// ---------- Events ----------

/// Fired when the user mouse-down's inside a pane's content area.
/// Per-kind systems read this to start text selection, button click,
/// etc. The pane-bevy handler has already set focus + InputConsumed
/// before firing.
#[derive(Message, Clone, Copy, Debug)]
pub struct PaneContentPressed {
    pub pane: Entity,
    /// Cursor position in window-space.
    pub window_pt: Vec2,
    /// Cursor position relative to the content_root's top-left. Same
    /// frame of reference as content_root child transforms (y-down).
    pub local_pt: Vec2,
    /// True if shift was held — kinds use this to extend a selection
    /// rather than start a new one.
    pub shift: bool,
    /// True iff this press was routed via the pinned-pane path
    /// (i.e. the pane has [`PanePinned`] and the click landed inside
    /// one of its [`PaneHotZones`]). When set, focus + raise + drag
    /// have all been suppressed; kinds should ONLY handle the actual
    /// interactive element under the cursor (button, link, list row)
    /// and skip "empty-space" behaviors like entering edit mode or
    /// placing a caret.
    pub pinned: bool,
}

/// Fired every frame the mouse moves while the left button is held
/// after a content-area press. `local_pt` may be outside the content
/// rect (and even negative) — drag handlers commonly need the cursor
/// position past the pane edge to compute snap-back, scroll edges, etc.
#[derive(Message, Clone, Copy, Debug)]
pub struct PaneContentDragged {
    pub pane: Entity,
    pub window_pt: Vec2,
    /// Cursor in content-root local coords. May be outside `[0, size]`.
    pub local_pt: Vec2,
    pub pinned: bool,
}

/// Fired once on left-button release after a content-area press. Drag
/// handlers use this to commit (drop a piece on the square under the
/// cursor, finalize a brush stroke, etc.). `local_pt` may be outside
/// the content rect — same semantics as [`PaneContentDragged`].
#[derive(Message, Clone, Copy, Debug)]
pub struct PaneContentReleased {
    pub pane: Entity,
    pub window_pt: Vec2,
    pub local_pt: Vec2,
    pub pinned: bool,
}

/// Fired when the cursor moves over a pane's content area with no
/// button held. Used for hover state — canvas widgets like chess use
/// it to light up buttons under the cursor before a click.
///
/// Skipped while any mouse button is down (drag/release events cover
/// that case). Fires at most once per frame the cursor moves; sitting
/// motionless produces no events.
#[derive(Message, Clone, Copy, Debug)]
pub struct PaneContentHovered {
    pub pane: Entity,
    pub window_pt: Vec2,
    pub local_pt: Vec2,
}

/// Fired on the second consecutive left press on the same pane within
/// a short window (≤500ms, ≤8px between presses). Hit-tests against
/// any pane region — title bar, content, resize edge — because the
/// gesture's intent is "I want to focus on this pane", not "interact
/// with its sub-region". Canvas viewport uses this as a quick way to
/// jump-to-pane: pan + zoom-to-1 so the pane sits in the top-left.
///
/// The press that triggered the event is still emitted normally
/// (content press, focus, raise, etc.) — the double-click is an
/// additional signal layered on top, not a replacement.
#[derive(Message, Clone, Copy, Debug)]
pub struct PaneDoubleClicked {
    pub pane: Entity,
}

// ---------- Plugin ----------

pub struct PanePlugin;

impl Plugin for PanePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FocusedPane>()
            .init_resource::<InputConsumed>()
            .init_resource::<PaneMouseMode>()
            .init_resource::<PendingPaneActions>()
            .init_resource::<PaneRegistry>()
            .init_resource::<PaneInputBlockZones>()
            .init_resource::<PaneCanvasRegion>()
            .init_resource::<PaneViewport>()
            .init_resource::<PaneZoom>()
            .insert_resource(PaneLayerAllocator::new())
            .add_message::<PaneContentPressed>()
            .add_message::<PaneContentDragged>()
            .add_message::<PaneContentReleased>()
            .add_message::<PaneContentHovered>()
            .add_message::<PaneDoubleClicked>()
            .add_plugins((TextInputPlugin, ChromeMaterialPlugin))
            .add_systems(
                Update,
                (
                    handle_pane_mouse,
                    update_pane_titles,
                    position_panes,
                    apply_pending_pane_actions,
                    sync_pinned_chrome,
                    sync_chrome_uniforms,
                    push_chrome_time,
                    (update_pane_cursor, emit_pane_hover),
                )
                    .chain()
                    // The whole chain ends up after PaneViewportReaders'
                    // ordering constraints — that's fine because the
                    // only systems in the set are members of this chain.
                    .in_set(PaneViewportReaders),
            )
            .add_systems(
                PreUpdate,
                camera::spawn_pane_cameras,
            )
            // Propagation runs in PostUpdate, not PreUpdate, so
            // children spawned by kinds during Update (the widget-bevy
            // `rerender_widgets` system despawns + respawns its tree
            // every resize frame, for one) get their RenderLayers
            // stamped in the same frame they're created — otherwise
            // they render on the default layer for one frame each,
            // bypassing the pane camera's viewport clip.
            .add_systems(
                PostUpdate,
                (
                    reset_input_consumed,
                    camera::propagate_render_layers,
                    enforce_pane_content_bounds,
                    // Run BEFORE Bevy's CameraUpdateSystems so that
                    // when Bevy's `camera_system` runs later in
                    // PostUpdate, it sees our just-updated viewport
                    // and recomputes the projection in the same
                    // frame. Otherwise resize introduces a one-frame
                    // lag where the projection still reflects the
                    // previous viewport size and content visibly
                    // drifts inside the pane.
                    camera::sync_pane_cameras
                        .before(CameraUpdateSystems)
                        .before(TransformSystems::Propagate),
                )
                    .chain(),
            );
    }
}

/// Emit [`PaneContentHovered`] for the topmost pane whose content area
/// is under the cursor, when no mouse button is held. Debounced so a
/// stationary cursor doesn't refire every frame. Cursor-leaves-pane
/// also emits one event with `local_pt` clamped to the last pane —
/// canvases that show a hover indicator can clear it on receipt of an
/// out-of-bounds local_pt.
#[derive(Resource, Default)]
struct LastHover {
    pane: Option<Entity>,
    window_pt: Vec2,
}

fn emit_pane_hover(
    windows: Query<&Window>,
    viewport: Res<PaneViewport>,
    buttons: Res<ButtonInput<MouseButton>>,
    mode: Res<PaneMouseMode>,
    mut last: Local<LastHover>,
    mut writer: MessageWriter<PaneContentHovered>,
    panes: Query<
        (Entity, &PaneRect, Option<&Visibility>, Has<PanePinned>),
        With<PaneTag>,
    >,
) {
    // Any button held → drag/release flow owns motion. Don't double-emit.
    if buttons.pressed(MouseButton::Left)
        || buttons.pressed(MouseButton::Right)
        || buttons.pressed(MouseButton::Middle)
        || !matches!(*mode, PaneMouseMode::Idle)
    {
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Some(pt) = window.cursor_position() else { return };
    if last.window_pt == pt {
        return; // cursor parked — no new event.
    }
    last.window_pt = pt;
    let pt_canvas = viewport.window_to_canvas(pt);
    let rects: Vec<(Entity, PaneRect)> = panes
        .iter()
        .filter(|(_, _, vis, pinned)| !matches!(vis, Some(Visibility::Hidden)) && !pinned)
        .map(|(e, r, _, _)| (e, *r))
        .collect();
    let target = topmost_pane_at(pt_canvas, &rects).and_then(|e| {
        let rect = rects.iter().find(|(x, _)| *x == e).map(|(_, r)| r)?;
        if matches!(region_at(pt_canvas, rect), Some(PaneRegion::Content)) {
            Some((e, pt_to_content_local(pt_canvas, rect)))
        } else {
            None
        }
    });
    if let Some((pane, local_pt)) = target {
        writer.write(PaneContentHovered {
            pane,
            window_pt: pt,
            local_pt,
        });
        last.pane = Some(pane);
    } else if let Some(prev) = last.pane.take() {
        // Cursor left the previously-hovered pane. Emit one synthetic
        // event with a sentinel local_pt so widgets can clear their
        // hover state.
        writer.write(PaneContentHovered {
            pane: prev,
            window_pt: pt,
            local_pt: Vec2::new(f32::INFINITY, f32::INFINITY),
        });
    }
}

/// Update the window cursor icon based on what's under the mouse.
/// Resize edges + corners get the matching directional resize cursor;
/// title bar gets the move cursor while idle; everywhere else stays
/// default. Skips when a drag/resize is already in progress so the
/// cursor doesn't flicker mid-gesture.
fn update_pane_cursor(
    mut commands: Commands,
    mode: Res<PaneMouseMode>,
    viewport: Res<PaneViewport>,
    windows: Query<(Entity, &Window)>,
    panes: Query<(&PaneRect, Option<&Visibility>, Has<PanePinned>), With<PaneTag>>,
) {
    use bevy::window::SystemCursorIcon;
    let Ok((win_entity, window)) = windows.single() else {
        return;
    };
    let icon: SystemCursorIcon = match *mode {
        PaneMouseMode::WindowResize { edges, .. } => cursor_for_edges(edges),
        // Drag in progress: leave the cursor alone — a grabbing cursor
        // over the title bar was distracting.
        PaneMouseMode::WindowDrag { .. } | PaneMouseMode::ContentDrag { .. } => {
            commands.entity(win_entity).remove::<bevy::window::CursorIcon>();
            return;
        }
        PaneMouseMode::Idle => {
            let Some(pt) = window.cursor_position() else {
                commands.entity(win_entity).remove::<bevy::window::CursorIcon>();
                return;
            };
            let pt_canvas = viewport.window_to_canvas(pt);
            let mut best: Option<(SystemCursorIcon, f32)> = None;
            for (rect, vis, pinned) in &panes {
                if matches!(vis, Some(Visibility::Hidden)) || pinned {
                    continue;
                }
                // Only resize edges get a special cursor. The title
                // bar stays default — no hand / grab indicator there.
                let icon = match region_at(pt_canvas, rect) {
                    Some(PaneRegion::ResizeEdge(e)) => Some(cursor_for_edges(e)),
                    _ => None,
                };
                if let Some(i) = icon
                    && best.map_or(true, |(_, z)| rect.z > z)
                {
                    best = Some((i, rect.z));
                }
            }
            match best {
                Some((i, _)) => i,
                None => {
                    commands.entity(win_entity).remove::<bevy::window::CursorIcon>();
                    return;
                }
            }
        }
    };
    commands
        .entity(win_entity)
        .insert(bevy::window::CursorIcon::System(icon));
}

fn cursor_for_edges(e: ResizeDir) -> bevy::window::SystemCursorIcon {
    use bevy::window::SystemCursorIcon;
    // macOS uses double-headed resize cursors — both edges and corners
    // are bidirectional. Pick the bidirectional system cursors rather
    // than the single-headed N/S/E/W variants so the cursor matches
    // what native macOS apps draw at the same hover point.
    match (e.north, e.south, e.east, e.west) {
        (true, false, true, false) | (false, true, false, true) => SystemCursorIcon::NeswResize,
        (true, false, false, true) | (false, true, true, false) => SystemCursorIcon::NwseResize,
        (true, false, false, false) | (false, true, false, false) => SystemCursorIcon::NsResize,
        (false, false, true, false) | (false, false, false, true) => SystemCursorIcon::EwResize,
        _ => SystemCursorIcon::Default,
    }
}

fn reset_input_consumed(mut consumed: ResMut<InputConsumed>) {
    consumed.0 = false;
}

/// Drip wall time into every chrome material so the focus glow can
/// pulse. Cheap: one f32 write per material per frame. Marks the
/// material Changed regardless of focus state, but Bevy already
/// detects "uniform actually differs" before re-uploading to GPU.
fn push_chrome_time(
    time: Res<Time>,
    mut materials: ResMut<Assets<PaneChromeMaterial>>,
) {
    let t = time.elapsed_secs();
    for (_id, mat) in materials.iter_mut() {
        mat.params.time = t;
    }
}

/// Keep each pane's chrome material in sync with its `PaneRect` and
/// focus state. The bg mesh is a unit square scaled via Transform so
/// we never need to recreate the Mesh asset on resize — only the
/// transform and the material's uniform parameters change.
fn sync_chrome_uniforms(
    panes: Query<(Entity, Ref<PaneRect>, &PaneChrome)>,
    mut bgs: Query<(&MeshMaterial2d<PaneChromeMaterial>, &mut Transform), Without<MeshMaterial2d<PaneShadowMaterial>>>,
    mut shadows: Query<(&MeshMaterial2d<PaneShadowMaterial>, &mut Transform), Without<MeshMaterial2d<PaneChromeMaterial>>>,
    focused: Res<FocusedPane>,
    style: Res<chrome_material::ChromeStyle>,
    active_shader: Res<ActiveChromeShader>,
    mut chrome_mats: ResMut<Assets<PaneChromeMaterial>>,
    mut shadow_mats: ResMut<Assets<PaneShadowMaterial>>,
) {
    let focus_changed = focused.is_changed();
    let style_changed = style.is_changed();
    let shader_changed = active_shader.is_changed();
    for (pane_entity, rect, chrome) in &panes {
        let needs_chrome_update =
            rect.is_changed() || focus_changed || style_changed || shader_changed;
        if !needs_chrome_update {
            continue;
        }
        let is_focused = focused.0 == Some(pane_entity);

        // Title cover: identical params to the body except for the
        // cover_mode + title_h flags, so it tracks resize / focus /
        // theme just like the body. Always uses the embedded default
        // shader (the cover_mode cutout lives there); active_shader
        // swaps don't touch its fragment handle.
        if let Ok((handle, mut transform)) = bgs.get_mut(chrome.title_cover) {
            transform.translation.x = rect.size.x * 0.5;
            transform.translation.y = -rect.size.y * 0.5;
            transform.scale.x = rect.size.x.max(1.0);
            transform.scale.y = rect.size.y.max(1.0);
            if let Some(mat) = chrome_mats.get_mut(&handle.0) {
                mat.params = style.params_for_title_cover(rect.size, is_focused, TITLE_H);
            }
        }

        // Chrome (body): unit mesh scaled to pane size.
        if let Ok((handle, mut transform)) = bgs.get_mut(chrome.bg) {
            transform.translation.x = rect.size.x * 0.5;
            transform.translation.y = -rect.size.y * 0.5;
            transform.scale.x = rect.size.x.max(1.0);
            transform.scale.y = rect.size.y.max(1.0);
            if let Some(mat) = chrome_mats.get_mut(&handle.0) {
                mat.params = style.params_for(rect.size, is_focused);
                if shader_changed {
                    mat.fragment = active_shader.0.clone();
                }
            }
        }

        // Shadow: unit mesh scaled to (pane size + 2*blur), centered
        // on the pane. Don't bother updating on focus changes since
        // shadow look doesn't depend on focus.
        if rect.is_changed() || style_changed {
            if let Ok((handle, mut transform)) = shadows.get_mut(chrome.shadow) {
                let sp = style.shadow_params_for(rect.size);
                transform.translation.x = rect.size.x * 0.5;
                transform.translation.y = -rect.size.y * 0.5;
                transform.scale.x = sp.mesh_size.x;
                transform.scale.y = sp.mesh_size.y;
                if let Some(mat) = shadow_mats.get_mut(&handle.0) {
                    mat.params = sp;
                }
            }
        }
    }
}

// ---------- Spawn ----------

/// Returned by `spawn_pane` so callers can hand the content_root to
/// per-kind setup code that wants to add child entities under it.
pub struct SpawnedPane {
    pub entity: Entity,
    pub content_root: Entity,
}

/// Create a pane entity + chrome. Does NOT call the registry's `spawn`
/// — callers do that themselves so they have direct access to the
/// returned SpawnedPane. (The registry-based path lives in
/// `spawn_pane_from_registry` for the host's restore loop.)
///
/// Z is taken as-is from `rect.z` — if you want it stacked above all
/// other panes, call `next_pane_z(world)` first.
pub fn spawn_pane(
    world: &mut World,
    kind: &'static str,
    title: impl Into<String>,
    rect: PaneRect,
    project_id: Option<u64>,
) -> SpawnedPane {
    let font = world
        .get_resource::<PaneFont>()
        .expect("PaneFont resource missing — host must insert it before spawning panes")
        .0
        .clone();

    let title_str: String = title.into();

    // Compute the initial translation the same way `position_panes` does
    // so children inherit the right GlobalTransform on the very first
    // frame. Without this, the pane spawns at (0,0,0); children render
    // there for one frame and then jump to the real position once
    // `position_panes` runs and transform propagation catches up.
    let viewport: PaneViewport = world.get_resource::<PaneViewport>().copied().unwrap_or_default();
    let initial_translation = world
        .query::<&Window>()
        .iter(world)
        .next()
        .map(|w| {
            let win_size = Vec2::new(w.width(), w.height());
            let screen_pos = viewport.canvas_to_window(rect.pos);
            bevy::math::Vec3::new(
                screen_pos.x - win_size.x * 0.5,
                win_size.y * 0.5 - screen_pos.y,
                rect.z,
            )
        })
        .unwrap_or(bevy::math::Vec3::ZERO);

    let pane = world
        .spawn((
            PaneTag,
            PaneKindMarker(kind),
            rect,
            PaneTitle(title_str.clone()),
            Transform::from_translation(initial_translation),
            Visibility::default(),
            PaneHotZones::default(),
        ))
        .id();
    if let Some(pid) = project_id {
        world.entity_mut(pane).insert(PaneProject(pid));
    }

    // Pane body: rounded-rect SDF material on a unit quad scaled via
    // Transform to the pane size. The SDF math in the shader works in
    // pixel units off `params.size`, so the mesh is always 1×1 and
    // resize is just a Transform tweak — no Mesh-asset re-upload on
    // drag. Anchor::TopLeft doesn't apply to Mesh2d, so we center the
    // unit square at (size/2, -size/2) to put its top-left at the
    // pane's origin. `sync_chrome_uniforms` keeps transform and
    // `params.size` in sync with PaneRect.
    let style = world
        .resource::<chrome_material::ChromeStyle>()
        .clone();
    let initial_params = style.params_for(rect.size, false);
    let shadow_params = style.shadow_params_for(rect.size);
    let active_shader = world.resource::<ActiveChromeShader>().0.clone();
    let unit_mesh = world
        .resource_mut::<Assets<Mesh>>()
        .add(Rectangle::new(1.0, 1.0));
    let bg_material = world
        .resource_mut::<Assets<PaneChromeMaterial>>()
        .add(PaneChromeMaterial {
            params: initial_params,
            fragment: active_shader,
        });
    let bg = world
        .spawn((
            ChildOf(pane),
            Mesh2d(unit_mesh.clone()),
            MeshMaterial2d(bg_material),
            Transform {
                translation: Vec3::new(rect.size.x * 0.5, -rect.size.y * 0.5, 0.0),
                scale: Vec3::new(rect.size.x.max(1.0), rect.size.y.max(1.0), 1.0),
                ..default()
            },
        ))
        .id();

    // Drop shadow: same unit mesh + scale-via-transform trick, but
    // sized to pane + 2×blur on each side and explicitly stamped onto
    // layer 0 so the main camera renders it (no per-pane viewport
    // clip). Local z is just below the bg's so the chrome sits in
    // front in the unusual case the same camera ever sees both.
    let shadow_material_handle = world
        .resource_mut::<Assets<PaneShadowMaterial>>()
        .add(PaneShadowMaterial { params: shadow_params });
    let shadow = world
        .spawn((
            ChildOf(pane),
            Mesh2d(unit_mesh.clone()),
            MeshMaterial2d(shadow_material_handle),
            Transform {
                translation: Vec3::new(
                    rect.size.x * 0.5,
                    -rect.size.y * 0.5,
                    -0.05,
                ),
                scale: Vec3::new(
                    shadow_params.mesh_size.x,
                    shadow_params.mesh_size.y,
                    1.0,
                ),
                ..default()
            },
            bevy::camera::visibility::RenderLayers::layer(0),
        ))
        .id();

    let text_style = world.resource::<ChromeTextStyle>().clone();

    // Title cover: a second chrome-material quad sized identically
    // to the body (same rounded corners, same gradient/glow, same
    // bg color) but with `cover_mode=1` so the shader cuts out the
    // content area. It sits at z > content_root so any pane content
    // scrolled up past the top of the content area gets masked by
    // the cover instead of bleeding over the title.
    //
    // Pinned to the embedded default shader rather than the active
    // chrome shader: the cutout logic lives in the default shader.
    // Preset shaders not handling cover_mode would render the cover
    // opaquely in the content area too, hiding everything. The
    // default's title region matches the active shader's title
    // region for the bundled presets; user-installed presets that
    // diverge can re-implement the cutout to make them match.
    let default_chrome_shader = world
        .resource::<AssetServer>()
        .load::<Shader>("embedded://pane_bevy/chrome_material.wgsl");
    let cover_params = style.params_for_title_cover(rect.size, false, TITLE_H);
    let cover_material = world
        .resource_mut::<Assets<PaneChromeMaterial>>()
        .add(PaneChromeMaterial {
            params: cover_params,
            fragment: default_chrome_shader,
        });
    let title_cover = world
        .spawn((
            ChildOf(pane),
            Mesh2d(unit_mesh),
            MeshMaterial2d(cover_material),
            Transform {
                translation: Vec3::new(
                    rect.size.x * 0.5,
                    -rect.size.y * 0.5,
                    0.25,
                ),
                scale: Vec3::new(rect.size.x.max(1.0), rect.size.y.max(1.0), 1.0),
                ..default()
            },
        ))
        .id();

    // Title-bar divider (1 px hairline at the bottom of the title region).
    // z bumped above title_cover (0.25) so it stays visible.
    let title_bar = world
        .spawn((
            ChildOf(pane),
            Sprite {
                color: text_style.divider,
                custom_size: Some(Vec2::new(rect.size.x, 1.0)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, -(TITLE_H - 1.0), 0.26),
        ))
        .id();

    // Title text — always created, even if empty, so we don't have to
    // conditionally spawn/despawn on PaneTitle changes.
    // z bumped above title_cover (0.25) so it stays visible.
    let title_text = world
        .spawn((
            ChildOf(pane),
            Text2d::new(title_str),
            TextFont {
                font: font.clone(),
                font_size: TITLE_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(TITLE_H),
            TextColor(text_style.title),
            Anchor::TOP_LEFT,
            Transform::from_xyz(MARGIN, -2.0, 0.27),
        ))
        .id();

    let content_root = world
        .spawn((
            ChildOf(pane),
            Transform::from_xyz(MARGIN, -(TITLE_H + MARGIN), 0.2),
            Visibility::default(),
        ))
        .id();

    let close_button = world
        .spawn((
            ChildOf(pane),
            Text2d::new("\u{00D7}"),
            TextFont {
                font,
                font_size: 16.0,
                ..default()
            },
            LineHeight::Px(CLOSE_BTN_SIZE),
            TextColor(text_style.close),
            Anchor::TOP_LEFT,
            Transform::from_xyz(
                rect.size.x - CLOSE_BTN_SIZE - CLOSE_BTN_INSET,
                -CLOSE_BTN_INSET,
                0.4,
            ),
        ))
        .id();

    world.entity_mut(pane).insert(PaneChrome {
        bg,
        shadow,
        title_bar,
        title_text,
        title_cover,
        content_root,
        close_button,
    });

    // Allocate this pane's RenderLayer id. Chrome AND content all
    // render on this layer through the per-pane camera. The main
    // camera (layer 0) draws nothing pane-related; it owns the
    // canvas/sidebar background. When two panes overlap on screen,
    // the higher-z pane's camera order is higher, so its full
    // viewport (bg included) draws over the lower-z pane's content
    // — that's what stops one pane's text from bleeding through
    // another pane that's stacked on top of it.
    let layer_id = world
        .resource_mut::<PaneLayerAllocator>()
        .allocate();
    let pane_layer_component = bevy::camera::visibility::RenderLayers::layer(layer_id);
    for chrome_entity in [bg, title_bar, title_text, title_cover, close_button] {
        world
            .entity_mut(chrome_entity)
            .insert(pane_layer_component.clone());
    }
    // Content root + descendants get the same layer via the propagate
    // system in `camera.rs` (it walks content_root's subtree on
    // Added<PaneLayer>). Chrome is stamped explicitly above because
    // propagation only descends from content_root, not from the pane
    // entity itself.
    world.entity_mut(pane).insert(PaneLayer(layer_id));

    SpawnedPane {
        entity: pane,
        content_root,
    }
}

/// Lookup-and-spawn: used by the host's restore loop and the radial
/// menu. Spawns the pane chrome, then runs the kind's `spawn` callback
/// to populate it.
pub fn spawn_pane_from_registry(
    world: &mut World,
    kind: &'static str,
    title: impl Into<String>,
    rect: PaneRect,
    project_id: Option<u64>,
    config: &Value,
) -> Option<Entity> {
    let spec = world.resource::<PaneRegistry>().get(kind).copied();
    let Some(spec) = spec else {
        eprintln!("[pane] no kind registered for {:?}; dropping spawn", kind);
        return None;
    };
    let SpawnedPane {
        entity,
        content_root,
    } = spawn_pane(world, spec.kind, title, rect, project_id);
    (spec.spawn)(world, entity, content_root, config);
    Some(entity)
}

/// Highest pane z + 1, so callers can stack new panes on top.
pub fn next_pane_z(world: &mut World) -> f32 {
    let mut q = world.query::<&PaneRect>();
    q.iter(world).map(|r| r.z).fold(0.0_f32, f32::max) + 1.0
}

// ---------- Hit testing / regions ----------

#[derive(Copy, Clone)]
pub enum PaneRegion {
    CloseButton,
    TitleBar,
    /// Mouse is over an edge or corner of the pane — directions tell
    /// the caller which edges to drag. Replaces the old single-corner
    /// `ResizeHandle` so panes can be resized from any edge / corner,
    /// not just bottom-right.
    ResizeEdge(ResizeDir),
    Content,
}

/// Pixel-thickness of the edge-resize hit zone (along each side).
pub const RESIZE_EDGE_PX: f32 = 6.0;

pub fn region_at(pt: Vec2, rect: &PaneRect) -> Option<PaneRegion> {
    if pt.x < rect.pos.x || pt.x > rect.pos.x + rect.size.x {
        return None;
    }
    if pt.y < rect.pos.y || pt.y > rect.pos.y + rect.size.y {
        return None;
    }
    // Close button takes priority over the edge band so the top-right
    // corner stays a close affordance, not a resize.
    let close_x0 = rect.pos.x + rect.size.x - CLOSE_BTN_SIZE - CLOSE_BTN_INSET;
    let close_x1 = close_x0 + CLOSE_BTN_SIZE;
    let close_y0 = rect.pos.y + CLOSE_BTN_INSET;
    let close_y1 = close_y0 + CLOSE_BTN_SIZE;
    if pt.x >= close_x0 && pt.x <= close_x1 && pt.y >= close_y0 && pt.y <= close_y1 {
        return Some(PaneRegion::CloseButton);
    }
    // Edge-band hit test. Any of the four outer rings — plus corners
    // (combined N+E, N+W, S+E, S+W) — become a resize gesture.
    let near_n = pt.y - rect.pos.y < RESIZE_EDGE_PX;
    let near_s = (rect.pos.y + rect.size.y) - pt.y < RESIZE_EDGE_PX;
    let near_w = pt.x - rect.pos.x < RESIZE_EDGE_PX;
    let near_e = (rect.pos.x + rect.size.x) - pt.x < RESIZE_EDGE_PX;
    if near_n || near_s || near_w || near_e {
        return Some(PaneRegion::ResizeEdge(ResizeDir {
            north: near_n,
            south: near_s,
            east: near_e,
            west: near_w,
        }));
    }
    if pt.y < rect.pos.y + TITLE_H {
        return Some(PaneRegion::TitleBar);
    }
    Some(PaneRegion::Content)
}

pub fn topmost_pane_at(pt: Vec2, panes: &[(Entity, PaneRect)]) -> Option<Entity> {
    let mut best: Option<(Entity, f32)> = None;
    for &(e, r) in panes {
        if pt.x >= r.pos.x
            && pt.x <= r.pos.x + r.size.x
            && pt.y >= r.pos.y
            && pt.y <= r.pos.y + r.size.y
            && best.map_or(true, |(_, z)| r.z > z)
        {
            best = Some((e, r.z));
        }
    }
    best.map(|(e, _)| e)
}

/// Window-space → content-area-local coords for a pane. The result has
/// y-down; (0,0) is the top-left of the content area (just inside
/// MARGIN below the title bar).
pub fn pt_to_content_local(pt: Vec2, rect: &PaneRect) -> Vec2 {
    let origin = rect.pos + Vec2::new(MARGIN, TITLE_H + MARGIN);
    pt - origin
}

/// Top-left + size of the content area (not including chrome).
pub fn content_area(rect: &PaneRect) -> (Vec2, Vec2) {
    let origin = Vec2::new(MARGIN, -(TITLE_H + MARGIN));
    let size = Vec2::new(
        (rect.size.x - 2.0 * MARGIN).max(0.0),
        (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
    );
    (origin, size)
}

/// Legacy alias — `PaneRect` is now canvas-space directly, so the
/// zoom parameter is ignored. Kept so call sites compile during the
/// transition; new code should use [`content_area`] directly.
pub fn content_area_canvas(rect: &PaneRect, _zoom: f32) -> (Vec2, Vec2) {
    content_area(rect)
}

// ---------- Mouse handling ----------

#[allow(clippy::too_many_arguments)]
#[derive(Default)]
struct PressTracker {
    pane: Option<Entity>,
    count: u32,
    last_time: f64,
    last_pt: Vec2,
}

/// Bundled SystemParam for double-click detection. Kept as one param
/// so `handle_pane_mouse` stays under Bevy's 16-tuple limit.
#[derive(bevy::ecs::system::SystemParam)]
struct DoublePress<'w, 's> {
    time: Res<'w, Time>,
    tracker: Local<'s, PressTracker>,
    writer: MessageWriter<'w, PaneDoubleClicked>,
}

impl<'w, 's> DoublePress<'w, 's> {
    /// Update press history. Fire a double-click when the second
    /// consecutive press lands on the same pane within 500ms and 8px
    /// of the prior press. Counter resets after firing so a 3rd press
    /// starts a fresh sequence.
    fn note(&mut self, target: Entity, pt: Vec2) {
        let now = self.time.elapsed_secs_f64();
        let same_pane = self.tracker.pane == Some(target);
        let close_in_time = now - self.tracker.last_time < 0.5;
        let close_in_space = (self.tracker.last_pt - pt).length() < 8.0;
        if same_pane && close_in_time && close_in_space {
            self.tracker.count += 1;
        } else {
            self.tracker.pane = Some(target);
            self.tracker.count = 1;
        }
        self.tracker.last_time = now;
        self.tracker.last_pt = pt;
        if self.tracker.count >= 2 {
            self.writer.write(PaneDoubleClicked { pane: target });
            self.tracker.count = 0;
        }
    }
}

fn handle_pane_mouse(
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    mods: Res<ButtonInput<KeyCode>>,
    viewport: Res<PaneViewport>,
    mut consumed: ResMut<InputConsumed>,
    mut mode: ResMut<PaneMouseMode>,
    mut focused: ResMut<FocusedPane>,
    mut close_actions: ResMut<PendingPaneActions>,
    block_zones: Res<PaneInputBlockZones>,
    mut content_press: MessageWriter<PaneContentPressed>,
    mut content_drag: MessageWriter<PaneContentDragged>,
    mut content_release: MessageWriter<PaneContentReleased>,
    mut dbl: DoublePress,
    mut panes: Query<
        (Entity, &mut PaneRect, Option<&Visibility>, Has<PanePinned>),
        With<PaneTag>,
    >,
    hot_zones: Query<&PaneHotZones>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };
    // PaneRect now lives in canvas-units. Convert cursor to the same
    // frame once; every hit-test below operates in canvas-space.
    let pt_canvas = viewport.window_to_canvas(pt);

    if buttons.just_released(MouseButton::Left) {
        if let PaneMouseMode::ContentDrag { pane, pinned } = *mode {
            if let Ok((_, rect, _, _)) = panes.get(pane) {
                content_release.write(PaneContentReleased {
                    pane,
                    window_pt: pt,
                    local_pt: pt_to_content_local(pt_canvas, &rect),
                    pinned,
                });
            }
        }
        *mode = PaneMouseMode::Idle;
    }

    // Block zones (sidebar etc.) stay in window-pixel coords — they're
    // chrome that's NOT under the canvas transform.
    let in_block_zone = block_zones
        .0
        .iter()
        .any(|r| pt.x >= r.min.x && pt.x <= r.max.x && pt.y >= r.min.y && pt.y <= r.max.y);

    if buttons.just_pressed(MouseButton::Left) && !consumed.0 && !in_block_zone {
        // Stage 1: unpinned panes get the normal chrome/content hit-test.
        // They always sit above pinned panes (pinned are forced to z=0,
        // unpinned start at z=1), so any unpinned hit wins outright.
        let unpinned_rects: Vec<(Entity, PaneRect)> = panes
            .iter()
            .filter(|(_, _, vis, pinned)| {
                !matches!(vis, Some(Visibility::Hidden)) && !pinned
            })
            .map(|(e, r, _, _)| (e, *r))
            .collect();
        if let Some(target) = topmost_pane_at(pt_canvas, &unpinned_rects) {
            let rect = *panes.get(target).unwrap().1;
            let region = region_at(pt_canvas, &rect);
            // Double-click on a pane is a "zoom/jump to this pane"
            // gesture — but only when it isn't landing on an interactive
            // element. A content click that hits one of the kind's
            // registered hot-zones (a button, link, input) belongs to
            // the widget, not the canvas, so don't count it.
            let on_click_target = matches!(region, Some(PaneRegion::Content))
                && hot_zones
                    .get(target)
                    .map_or(false, |z| z.contains(pt_to_content_local(pt_canvas, &rect)));
            if region.is_some() && !on_click_target {
                dbl.note(target, pt);
            }
            match region {
                Some(PaneRegion::CloseButton) => {
                    close_actions.close.push(target);
                    consumed.0 = true;
                    return;
                }
                Some(PaneRegion::TitleBar) => {
                    focused.0 = Some(target);
                    consumed.0 = true;
                    bring_to_front(target, &mut panes);
                    *mode = PaneMouseMode::WindowDrag {
                        pane: target,
                        grab_offset: pt_canvas - rect.pos,
                    };
                }
                Some(PaneRegion::ResizeEdge(edges)) => {
                    focused.0 = Some(target);
                    consumed.0 = true;
                    bring_to_front(target, &mut panes);
                    *mode = PaneMouseMode::WindowResize {
                        pane: target,
                        edges,
                        start_pt: pt_canvas,
                        start_pos: rect.pos,
                        start_size: rect.size,
                    };
                }
                Some(PaneRegion::Content) => {
                    focused.0 = Some(target);
                    consumed.0 = true;
                    bring_to_front(target, &mut panes);
                    let shift = mods.pressed(KeyCode::ShiftLeft)
                        || mods.pressed(KeyCode::ShiftRight);
                    content_press.write(PaneContentPressed {
                        pane: target,
                        window_pt: pt,
                        local_pt: pt_to_content_local(pt_canvas, &rect),
                        shift,
                        pinned: false,
                    });
                    *mode = PaneMouseMode::ContentDrag {
                        pane: target,
                        pinned: false,
                    };
                }
                None => {}
            }
            return;
        }

        // Stage 2: no unpinned pane covered the click. Walk pinned panes
        // and route the press if it lands inside a registered hot-zone.
        // Empty space on a pinned pane is intentionally NOT consumed so
        // it falls through to the canvas (matches the "background
        // decoration" promise). Chrome (drag/resize/close/focus) stays
        // suppressed for pinned panes regardless.
        let mut best: Option<(Entity, f32, Vec2)> = None;
        for (e, r, vis, pinned) in panes.iter() {
            if matches!(vis, Some(Visibility::Hidden)) || !pinned {
                continue;
            }
            if !matches!(region_at(pt_canvas, &r), Some(PaneRegion::Content)) {
                continue;
            }
            let local = pt_to_content_local(pt_canvas, &r);
            let Ok(zones) = hot_zones.get(e) else { continue };
            if !zones.contains(local) {
                continue;
            }
            if best.map_or(true, |(_, z, _)| r.z >= z) {
                best = Some((e, r.z, local));
            }
        }
        if let Some((target, _, local)) = best {
            consumed.0 = true;
            // Grant keyboard focus so hot-zone elements that edit text
            // (issues titles/fields, inline inputs) can hold focus.
            // Without this the pinned pane is never `FocusedPane`, so
            // `commit_edit_on_focus_change` tears the edit down the same
            // frame it starts. We deliberately do NOT `bring_to_front`
            // here: z stays 0 so the pane remains background decoration.
            focused.0 = Some(target);
            // No double-click jump here: a pinned-pane press only routes
            // when it lands inside a hot-zone, i.e. always a click target.
            let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);
            content_press.write(PaneContentPressed {
                pane: target,
                window_pt: pt,
                local_pt: local,
                shift,
                pinned: true,
            });
            *mode = PaneMouseMode::ContentDrag {
                pane: target,
                pinned: true,
            };
        } else {
            // Click landed on empty canvas (no unpinned pane in Stage 1,
            // no pinned hot-zone here). Blur whatever was focused so an
            // in-progress edit on a pinned pane commits + closes via
            // `commit_edit_on_focus_change`, matching the "click away to
            // deselect" expectation.
            focused.0 = None;
        }
        return;
    }

    if !buttons.pressed(MouseButton::Left) {
        return;
    }

    match *mode {
        PaneMouseMode::WindowDrag { pane, grab_offset } => {
            if let Ok((_, mut rect, _, _)) = panes.get_mut(pane) {
                // Both grab_offset and pt_canvas are in canvas-units.
                rect.pos = pt_canvas - grab_offset;
            }
        }
        PaneMouseMode::WindowResize {
            pane,
            edges,
            start_pt,
            start_pos,
            start_size,
        } => {
            if let Ok((_, mut rect, _, _)) = panes.get_mut(pane) {
                let delta = pt_canvas - start_pt;
                let mut new_pos = start_pos;
                let mut new_size = start_size;
                if edges.east {
                    new_size.x = (start_size.x + delta.x).max(MIN_PANE_SIZE.x);
                }
                if edges.west {
                    // Dragging the west edge: position moves with the
                    // cursor; size shrinks/grows so the east edge stays
                    // put. Clamp so the size never goes below the min
                    // (and accordingly the position never crosses past
                    // the fixed east edge).
                    let max_dx = start_size.x - MIN_PANE_SIZE.x;
                    let dx = delta.x.min(max_dx);
                    new_pos.x = start_pos.x + dx;
                    new_size.x = start_size.x - dx;
                }
                if edges.south {
                    new_size.y = (start_size.y + delta.y).max(MIN_PANE_SIZE.y);
                }
                if edges.north {
                    let max_dy = start_size.y - MIN_PANE_SIZE.y;
                    let dy = delta.y.min(max_dy);
                    new_pos.y = start_pos.y + dy;
                    new_size.y = start_size.y - dy;
                }
                rect.pos = new_pos;
                rect.size = new_size;
            }
        }
        PaneMouseMode::ContentDrag { pane, pinned } => {
            if let Ok((_, rect, _, _)) = panes.get(pane) {
                content_drag.write(PaneContentDragged {
                    pane,
                    window_pt: pt,
                    local_pt: pt_to_content_local(pt_canvas, &rect),
                    pinned,
                });
            }
        }
        PaneMouseMode::Idle => {}
    }
}

/// Max pane z. Above this we renormalize the whole z-stack down to
/// [1, N] so we never bump past Bevy's default 2D camera ortho
/// frustum (z = [-1000, 1000]). When that happens the pane silently
/// vanishes — looked just like "clicking made the pane disappear,"
/// which is exactly the bug this guards against.
const MAX_PANE_Z: f32 = 500.0;

fn bring_to_front(
    target: Entity,
    panes: &mut Query<
        (Entity, &mut PaneRect, Option<&Visibility>, Has<PanePinned>),
        With<PaneTag>,
    >,
) {
    // Pinned panes are background decoration — never bump their z.
    if let Ok((_, _, _, true)) = panes.get(target) {
        return;
    }
    // Compute max-z over UNPINNED panes only so pinned panes (kept at
    // z=0 by `pin_pane`) don't pull the floor up and don't get touched
    // by renormalization.
    let max_z = panes
        .iter()
        .filter(|(_, _, _, pinned)| !pinned)
        .map(|(_, r, _, _)| r.z)
        .fold(0.0_f32, f32::max);
    if let Ok((_, mut rect, _, _)) = panes.get_mut(target) {
        if rect.z < max_z {
            rect.z = max_z + 1.0;
        }
    }
    // Renormalize if the stack is approaching the camera frustum.
    // Only renumber unpinned panes; pinned panes stay at z=0.
    if max_z + 1.0 > MAX_PANE_Z {
        let mut entries: Vec<(Entity, f32)> = panes
            .iter()
            .filter(|(_, _, _, pinned)| !pinned)
            .map(|(e, r, _, _)| (e, r.z))
            .collect();
        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (i, (e, _)) in entries.into_iter().enumerate() {
            if let Ok((_, mut rect, _, _)) = panes.get_mut(e) {
                rect.z = (i as f32) + 1.0;
            }
        }
    }
}

// ---------- Layout ----------

fn position_panes(
    windows: Query<&Window>,
    focused: Res<FocusedPane>,
    text_style: Res<ChromeTextStyle>,
    chrome_style: Res<chrome_material::ChromeStyle>,
    viewport: Res<PaneViewport>,
    panes: Query<(&PaneRect, &PaneChrome), With<PaneTag>>,
    mut t_q: Query<&mut Transform>,
    mut sprite_q: Query<&mut Sprite>,
    mut color_q: Query<&mut TextColor>,
    parents: Query<Entity, With<PaneTag>>,
) {
    // Focused divider color = the same accent the SDF chrome material
    // uses for border_focused, so the two cues read as one design
    // choice rather than two unrelated highlights.
    let focused_divider_color = Color::LinearRgba(LinearRgba::new(
        chrome_style.border_focused.x,
        chrome_style.border_focused.y,
        chrome_style.border_focused.z,
        chrome_style.border_focused.w,
    ));
    let Ok(win) = windows.single() else {
        return;
    };
    let win_size = Vec2::new(win.width(), win.height());

    for entity in &parents {
        let Ok((rect, chrome)) = panes.get(entity) else {
            continue;
        };
        let is_focused = focused.0 == Some(entity);

        // Every write below goes through a "compare first, write only on
        // change" guard. Unconditional `*x = y` deref-mut on a Mut<T>
        // marks the entity Changed, which keeps Bevy's reactive loop
        // active and re-extracts the sprite/text every frame — even when
        // the value hasn't moved. With ~6 chrome entities per pane and
        // several terminal/widget panes alive, the unguarded version
        // burns >300% CPU just keeping itself awake.
        if let Ok(mut t) = t_q.get_mut(entity) {
            // PaneRect is canvas-space; project to a screen-pixel
            // position via the canvas viewport, then convert to
            // Bevy's world coords (centered, y-up). Pane scale = zoom
            // makes all children (chrome + content) magnify uniformly.
            let screen_pos = viewport.canvas_to_window(rect.pos);
            let want = bevy::math::Vec3::new(
                screen_pos.x - win_size.x * 0.5,
                win_size.y * 0.5 - screen_pos.y,
                rect.z,
            );
            if t.translation != want {
                t.translation = want;
            }
            let want_scale = Vec3::new(viewport.zoom, viewport.zoom, 1.0);
            if t.scale != want_scale {
                t.scale = want_scale;
            }
        }
        if let Ok(mut s) = sprite_q.get_mut(chrome.bg) {
            let want = Some(rect.size);
            if s.custom_size != want {
                s.custom_size = want;
            }
        }
        // Focused panes get a thicker, accent-colored title divider so
        // it's obvious which one will receive keys.
        let (bar_h, bar_color) = if is_focused {
            (TITLE_DIVIDER_H_FOCUSED, focused_divider_color)
        } else {
            (1.0, text_style.divider)
        };
        if let Ok(mut s) = sprite_q.get_mut(chrome.title_bar) {
            let want_size = Some(Vec2::new(rect.size.x, bar_h));
            if s.custom_size != want_size {
                s.custom_size = want_size;
            }
            if s.color != bar_color {
                s.color = bar_color;
            }
        }
        if let Ok(mut t) = t_q.get_mut(chrome.title_bar) {
            let want_y = -(TITLE_H - bar_h);
            if t.translation.y != want_y {
                t.translation.y = want_y;
            }
        }
        if let Ok(mut c) = color_q.get_mut(chrome.title_text) {
            let want = if is_focused {
                text_style.title_focused
            } else {
                text_style.title
            };
            if c.0 != want {
                c.0 = want;
            }
        }
        // Close + handle colors track ChromeTextStyle. They don't
        // change on focus today, but pulling them through this same
        // loop keeps the data path uniform (one resource → one writer).
        if let Ok(mut c) = color_q.get_mut(chrome.close_button) {
            if c.0 != text_style.close {
                c.0 = text_style.close;
            }
        }
        if let Ok(mut t) = t_q.get_mut(chrome.close_button) {
            let wx = rect.size.x - CLOSE_BTN_SIZE - CLOSE_BTN_INSET;
            let wy = -CLOSE_BTN_INSET;
            if t.translation.x != wx || t.translation.y != wy {
                t.translation.x = wx;
                t.translation.y = wy;
            }
        }
    }
}

/// Hide chrome (title bar, title text, title cover, close button,
/// resize handle) on pinned panes; restore visibility when unpinned.
/// Driven by `Added<PanePinned>` and `RemovedComponents<PanePinned>`
/// so it only walks chrome when pin state actually flips.
fn sync_pinned_chrome(
    added: Query<&PaneChrome, Added<PanePinned>>,
    mut removed: RemovedComponents<PanePinned>,
    chromes: Query<&PaneChrome>,
    mut vis_q: Query<&mut Visibility>,
) {
    let set_vis = |chrome: &PaneChrome, vis: Visibility, vis_q: &mut Query<&mut Visibility>| {
        for e in [
            chrome.bg,
            chrome.shadow,
            chrome.title_bar,
            chrome.title_text,
            chrome.title_cover,
            chrome.close_button,
        ] {
            if let Ok(mut v) = vis_q.get_mut(e) {
                if *v != vis {
                    *v = vis;
                }
            }
        }
    };
    for chrome in &added {
        set_vis(chrome, Visibility::Hidden, &mut vis_q);
    }
    for entity in removed.read() {
        if let Ok(chrome) = chromes.get(entity) {
            set_vis(chrome, Visibility::Inherited, &mut vis_q);
        }
    }
}

fn update_pane_titles(
    titles: Query<(&PaneTitle, &PaneChrome), Changed<PaneTitle>>,
    mut text_q: Query<&mut Text2d>,
) {
    for (title, chrome) in &titles {
        if let Ok(mut text) = text_q.get_mut(chrome.title_text) {
            if text.0 != title.0 {
                text.0 = title.0.clone();
            }
        }
    }
}

// ---------- Close ----------

/// Run the registered on_close (if any) for every pane queued in
/// PendingPaneActions.close, then despawn the entity. Exclusive system
/// because on_close may need &mut World (worker shutdown, file
/// removal, etc.). Also applies queued pin/unpin actions in the same
/// pass.
fn apply_pending_pane_actions(world: &mut World) {
    // Pin / unpin first so the z update lands before any layout
    // consumer reads `PaneRect` this frame.
    let to_pin = std::mem::take(&mut world.resource_mut::<PendingPaneActions>().pin);
    for entity in to_pin {
        if world.get_entity(entity).is_err() {
            continue;
        }
        if let Some(mut rect) = world.get_mut::<PaneRect>(entity) {
            // z=0 keeps pinned panes strictly below any unpinned pane
            // (unpinned start at z=1 via `next_pane_z`).
            rect.z = 0.0;
        }
        world.entity_mut(entity).insert(PanePinned);
    }
    let to_unpin = std::mem::take(&mut world.resource_mut::<PendingPaneActions>().unpin);
    for entity in to_unpin {
        if world.get_entity(entity).is_err() {
            continue;
        }
        world.entity_mut(entity).remove::<PanePinned>();
        let new_z = next_pane_z(world);
        if let Some(mut rect) = world.get_mut::<PaneRect>(entity) {
            rect.z = new_z;
        }
    }

    let to_close = std::mem::take(&mut world.resource_mut::<PendingPaneActions>().close);
    if to_close.is_empty() {
        return;
    }
    for entity in to_close {
        if world.get_entity(entity).is_err() {
            continue;
        }
        // Look up the kind, then the spec's on_close callback.
        let on_close = world
            .get::<PaneKindMarker>(entity)
            .and_then(|k| world.resource::<PaneRegistry>().get(k.0).copied())
            .and_then(|s| s.on_close);
        if let Some(cb) = on_close {
            cb(world, entity);
        }
        // Reclaim this pane's RenderLayer id + despawn its per-pane
        // camera. Read the layer BEFORE the pane is despawned (after
        // despawn the component is gone and we can't recover the id).
        let layer = world.get::<PaneLayer>(entity).copied();
        if let Some(PaneLayer(id)) = layer {
            // Find the camera linked back to this pane and despawn it.
            let mut camera_q = world.query::<(Entity, &PaneCameraOf)>();
            let cam_entity = camera_q
                .iter(world)
                .find_map(|(cam, owner)| (owner.0 == entity).then_some(cam));
            if let Some(cam) = cam_entity
                && world.get_entity(cam).is_ok()
            {
                world.entity_mut(cam).despawn();
            }
            world.resource_mut::<PaneLayerAllocator>().free(id);
        }
        if world.get_entity(entity).is_ok() {
            world.entity_mut(entity).despawn();
        }
        let mut focused = world.resource_mut::<FocusedPane>();
        if focused.0 == Some(entity) {
            focused.0 = None;
        }
    }
}

// ---------- Helpers exposed for kind plugins ----------

/// True iff the entity is a pane of the given kind. Kind plugins use
/// this in their input/render systems to filter to their own panes.
pub fn pane_is_kind(world: &World, entity: Entity, kind: &str) -> bool {
    world
        .get::<PaneKindMarker>(entity)
        .is_some_and(|m| m.0 == kind)
}

// ---------- Content bounds enforcement ----------

/// Marks a Text2d descendant of a content_root as "do not auto-resize
/// its `TextBounds` to the pane". The enforcement system skips these.
///
/// Note: visible clipping no longer depends on this — every pane has
/// its own viewport-clipped camera (see [`crate::camera`]), so content
/// outside the pane simply isn't rendered regardless of what we do
/// here. This component remains useful only for kinds that want to
/// manage their own `TextBounds` for layout reasons (run-button's
/// output panel snaps bounds to whole-line increments to avoid
/// half-line bleed at the bottom).
#[derive(Component, Copy, Clone, Debug, Default)]
pub struct PaneContentNoClip;

/// Walks each pane's content_root subtree and writes a `TextBounds`
/// onto every Text2d to match the pane's content area. This is
/// **layout** assistance for kinds that use wrapping (the wrap
/// width comes from `TextBounds.width`) — it is *not* the clipping
/// mechanism: that's the per-pane camera viewport in
/// [`crate::camera`]. Kinds doing their own bounds management can
/// opt out with [`PaneContentNoClip`].
///
/// Sprites are deliberately not touched here: textured sprites
/// (terminal glyph cells, atlas tiles) treat `custom_size` as a
/// render size and shrinking it would distort the glyph. Kinds size
/// their own sprites; the terminal cell pool, editor caret, and
/// run-button divider all already do.
fn enforce_pane_content_bounds(
    panes: Query<(&PaneRect, &PaneChrome), With<PaneTag>>,
    changed_panes: Query<(), (With<PaneTag>, Changed<PaneRect>)>,
    new_text: Query<(), Added<Text2d>>,
    children_q: Query<&Children>,
    transforms: Query<&Transform>,
    text_q: Query<(), With<Text2d>>,
    no_clip_q: Query<(), With<PaneContentNoClip>>,
    mut bounds_q: Query<&mut TextBounds>,
    mut commands: Commands,
) {
    // Hot path: nothing relevant changed this frame, skip the subtree
    // walk. Without this, we re-walk every pane's content_root every
    // frame — for terminal panes that's thousands of cell sprites,
    // dominating CPU even while truly idle.
    if changed_panes.is_empty() && new_text.is_empty() {
        return;
    }
    for (rect, chrome) in &panes {
        // PaneRect is canvas-units. TextBounds lives in the Text2d's
        // local frame, which is the pane's local frame (since Text2d
        // is a descendant of content_root, child of the pane). The
        // pane entity's Transform.scale = zoom handles the visual
        // magnification, so TextBounds should be set in pane-local
        // (canvas-units) — same as rect.size.
        let content_w = (rect.size.x - 2.0 * MARGIN).max(0.0);

        let Ok(root_children) = children_q.get(chrome.content_root) else {
            continue;
        };

        // (entity, accumulated local offset from content_root in Bevy
        // world coords — x is right, y is up; children of content_root
        // typically have negative y to render below the origin.)
        let mut stack: Vec<(Entity, Vec2)> = Vec::with_capacity(16);
        for c in root_children.iter() {
            let t = transforms
                .get(c)
                .map(|t| Vec2::new(t.translation.x, t.translation.y))
                .unwrap_or(Vec2::ZERO);
            stack.push((c, t));
        }

        while let Some((entity, offset)) = stack.pop() {
            let skip = no_clip_q.get(entity).is_ok();

            if !skip && text_q.get(entity).is_ok() {
                let avail_w = (content_w - offset.x).max(0.0);
                // Height is intentionally `None`: visible clipping is
                // done by the per-pane camera viewport. Setting a
                // bounded height here breaks scrolling — anything whose
                // initial position is below the content area gets
                // avail_h=0 and Bevy renders no glyphs inside a
                // zero-height TextBounds, so the text stays invisible
                // even after the user scrolls it into view.
                let new_bounds = TextBounds {
                    width: Some(avail_w),
                    height: None,
                };
                if let Ok(mut existing) = bounds_q.get_mut(entity) {
                    if existing.width != new_bounds.width
                        || existing.height != new_bounds.height
                    {
                        *existing = new_bounds;
                    }
                } else {
                    commands.entity(entity).insert(new_bounds);
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
