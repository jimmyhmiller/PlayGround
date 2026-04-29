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

use std::collections::HashMap;

use bevy::input::mouse::MouseButton;
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::{LineHeight, TextBounds};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub mod text_input;
pub use text_input::{
    col_at_x, click_to_caret, focus_text_input, spawn_text_input, FocusedTextInput, TextInput,
    TextInputEvent, TextInputFocused, TextInputPlugin, TextInputStyle, TextInputView,
};

pub const TITLE_H: f32 = 22.0;
pub const MARGIN: f32 = 8.0;
pub const HANDLE_SIZE: f32 = 14.0;
pub const CLOSE_BTN_SIZE: f32 = 14.0;
pub const CLOSE_BTN_INSET: f32 = 4.0;
pub const MIN_PANE_SIZE: Vec2 = Vec2::new(160.0, 120.0);

const PANEL_BG: Color = Color::srgb(0.105, 0.110, 0.122);
const TITLE_DIVIDER: Color = Color::srgb(0.165, 0.170, 0.188);
const HANDLE_COLOR: Color = Color::srgb(0.22, 0.23, 0.26);
const CLOSE_COLOR: Color = Color::srgb(0.50, 0.52, 0.56);
const TITLE_TEXT_COLOR: Color = Color::srgb(0.78, 0.80, 0.84);
const TITLE_FONT_SIZE: f32 = 12.0;

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
    pub title_bar: Entity,
    pub title_text: Entity,
    pub content_root: Entity,
    pub resize_handle: Entity,
    pub close_button: Entity,
}

/// Membership in a host-defined "project" / workspace bucket. Pane-bevy
/// itself doesn't interpret this — the host queries it for visibility,
/// persistence partitioning, etc. Optional so standalone demos can
/// spawn panes without a project system.
#[derive(Component, Copy, Clone, Debug)]
pub struct PaneProject(pub u64);

// ---------- Resources ----------

/// Font used for chrome text (title bar, close button glyph). Host
/// must insert this before any pane is spawned.
#[derive(Resource)]
pub struct PaneFont(pub Handle<Font>);

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
        anchor_pos: Vec2,
    },
}

/// Side-channel for actions that need exclusive World access (close
/// runs the kind's on_close callback then despawns the entity).
#[derive(Resource, Default)]
pub struct PendingPaneActions {
    pub close: Vec<Entity>,
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
            .add_message::<PaneContentPressed>()
            .add_plugins(TextInputPlugin)
            .add_systems(
                Update,
                (
                    handle_pane_mouse,
                    update_pane_titles,
                    position_panes,
                    apply_pending_pane_actions,
                )
                    .chain(),
            )
            .add_systems(PostUpdate, (reset_input_consumed, enforce_pane_content_bounds));
    }
}

fn reset_input_consumed(mut consumed: ResMut<InputConsumed>) {
    consumed.0 = false;
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

    let pane = world
        .spawn((
            PaneTag,
            PaneKindMarker(kind),
            rect,
            PaneTitle(title_str.clone()),
            Transform::default(),
            Visibility::default(),
        ))
        .id();
    if let Some(pid) = project_id {
        world.entity_mut(pane).insert(PaneProject(pid));
    }

    let bg = world
        .spawn((
            ChildOf(pane),
            Sprite {
                color: PANEL_BG,
                custom_size: Some(rect.size),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 0.0),
        ))
        .id();

    // Title-bar divider (1 px hairline at the bottom of the title region).
    let title_bar = world
        .spawn((
            ChildOf(pane),
            Sprite {
                color: TITLE_DIVIDER,
                custom_size: Some(Vec2::new(rect.size.x, 1.0)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, -(TITLE_H - 1.0), 0.1),
        ))
        .id();

    // Title text — always created, even if empty, so we don't have to
    // conditionally spawn/despawn on PaneTitle changes.
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
            TextColor(TITLE_TEXT_COLOR),
            Anchor::TOP_LEFT,
            Transform::from_xyz(MARGIN, -2.0, 0.15),
        ))
        .id();

    let content_root = world
        .spawn((
            ChildOf(pane),
            Transform::from_xyz(MARGIN, -(TITLE_H + MARGIN), 0.2),
            Visibility::default(),
        ))
        .id();

    let resize_handle = world
        .spawn((
            ChildOf(pane),
            Sprite {
                color: HANDLE_COLOR,
                custom_size: Some(Vec2::splat(HANDLE_SIZE)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(
                rect.size.x - HANDLE_SIZE,
                -(rect.size.y - HANDLE_SIZE),
                0.3,
            ),
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
            TextColor(CLOSE_COLOR),
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
        title_bar,
        title_text,
        content_root,
        resize_handle,
        close_button,
    });

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
    ResizeHandle,
    Content,
}

pub fn region_at(pt: Vec2, rect: &PaneRect) -> Option<PaneRegion> {
    if pt.x < rect.pos.x || pt.x > rect.pos.x + rect.size.x {
        return None;
    }
    if pt.y < rect.pos.y || pt.y > rect.pos.y + rect.size.y {
        return None;
    }
    let close_x0 = rect.pos.x + rect.size.x - CLOSE_BTN_SIZE - CLOSE_BTN_INSET;
    let close_x1 = close_x0 + CLOSE_BTN_SIZE;
    let close_y0 = rect.pos.y + CLOSE_BTN_INSET;
    let close_y1 = close_y0 + CLOSE_BTN_SIZE;
    if pt.x >= close_x0 && pt.x <= close_x1 && pt.y >= close_y0 && pt.y <= close_y1 {
        return Some(PaneRegion::CloseButton);
    }
    let handle_x0 = rect.pos.x + rect.size.x - HANDLE_SIZE;
    let handle_y0 = rect.pos.y + rect.size.y - HANDLE_SIZE;
    if pt.x >= handle_x0 && pt.y >= handle_y0 {
        return Some(PaneRegion::ResizeHandle);
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

// ---------- Mouse handling ----------

#[allow(clippy::too_many_arguments)]
fn handle_pane_mouse(
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    mods: Res<ButtonInput<KeyCode>>,
    mut consumed: ResMut<InputConsumed>,
    mut mode: ResMut<PaneMouseMode>,
    mut focused: ResMut<FocusedPane>,
    mut close_actions: ResMut<PendingPaneActions>,
    block_zones: Res<PaneInputBlockZones>,
    mut content_press: MessageWriter<PaneContentPressed>,
    mut panes: Query<(Entity, &mut PaneRect, Option<&Visibility>), With<PaneTag>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };

    if buttons.just_released(MouseButton::Left) {
        *mode = PaneMouseMode::Idle;
    }

    let in_block_zone = block_zones
        .0
        .iter()
        .any(|r| pt.x >= r.min.x && pt.x <= r.max.x && pt.y >= r.min.y && pt.y <= r.max.y);

    if buttons.just_pressed(MouseButton::Left) && !consumed.0 && !in_block_zone {
        let visible_rects: Vec<(Entity, PaneRect)> = panes
            .iter()
            .filter(|(_, _, vis)| !matches!(vis, Some(Visibility::Hidden)))
            .map(|(e, r, _)| (e, *r))
            .collect();
        let Some(target) = topmost_pane_at(pt, &visible_rects) else {
            return;
        };

        let rect = *panes.get(target).unwrap().1;
        match region_at(pt, &rect) {
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
                    grab_offset: pt - rect.pos,
                };
            }
            Some(PaneRegion::ResizeHandle) => {
                focused.0 = Some(target);
                consumed.0 = true;
                bring_to_front(target, &mut panes);
                *mode = PaneMouseMode::WindowResize {
                    pane: target,
                    anchor_pos: rect.pos,
                };
            }
            Some(PaneRegion::Content) => {
                focused.0 = Some(target);
                consumed.0 = true;
                bring_to_front(target, &mut panes);
                let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);
                content_press.write(PaneContentPressed {
                    pane: target,
                    window_pt: pt,
                    local_pt: pt_to_content_local(pt, &rect),
                    shift,
                });
            }
            None => {}
        }
        return;
    }

    if !buttons.pressed(MouseButton::Left) {
        return;
    }

    match *mode {
        PaneMouseMode::WindowDrag { pane, grab_offset } => {
            if let Ok((_, mut rect, _)) = panes.get_mut(pane) {
                rect.pos = pt - grab_offset;
            }
        }
        PaneMouseMode::WindowResize { pane, anchor_pos } => {
            if let Ok((_, mut rect, _)) = panes.get_mut(pane) {
                let raw = pt - anchor_pos;
                rect.size = Vec2::new(raw.x.max(MIN_PANE_SIZE.x), raw.y.max(MIN_PANE_SIZE.y));
            }
        }
        PaneMouseMode::Idle => {}
    }
}

fn bring_to_front(
    target: Entity,
    panes: &mut Query<(Entity, &mut PaneRect, Option<&Visibility>), With<PaneTag>>,
) {
    let max_z = panes.iter().map(|(_, r, _)| r.z).fold(0.0_f32, f32::max);
    if let Ok((_, mut rect, _)) = panes.get_mut(target) {
        if rect.z < max_z {
            rect.z = max_z + 1.0;
        }
    }
}

// ---------- Layout ----------

fn position_panes(
    windows: Query<&Window>,
    panes: Query<(&PaneRect, &PaneChrome), With<PaneTag>>,
    mut t_q: Query<&mut Transform>,
    mut sprite_q: Query<&mut Sprite>,
    parents: Query<Entity, With<PaneTag>>,
) {
    let Ok(win) = windows.single() else {
        return;
    };
    let win_size = Vec2::new(win.width(), win.height());

    for entity in &parents {
        let Ok((rect, chrome)) = panes.get(entity) else {
            continue;
        };
        if let Ok(mut t) = t_q.get_mut(entity) {
            t.translation.x = rect.pos.x - win_size.x * 0.5;
            t.translation.y = win_size.y * 0.5 - rect.pos.y;
            t.translation.z = rect.z;
        }
        if let Ok(mut s) = sprite_q.get_mut(chrome.bg) {
            s.custom_size = Some(rect.size);
        }
        if let Ok(mut s) = sprite_q.get_mut(chrome.title_bar) {
            s.custom_size = Some(Vec2::new(rect.size.x, 1.0));
        }
        if let Ok(mut t) = t_q.get_mut(chrome.resize_handle) {
            t.translation.x = rect.size.x - HANDLE_SIZE;
            t.translation.y = -(rect.size.y - HANDLE_SIZE);
        }
        if let Ok(mut t) = t_q.get_mut(chrome.close_button) {
            t.translation.x = rect.size.x - CLOSE_BTN_SIZE - CLOSE_BTN_INSET;
            t.translation.y = -CLOSE_BTN_INSET;
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
/// removal, etc.).
fn apply_pending_pane_actions(world: &mut World) {
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

/// Marks a Text2d descendant of a content_root as "do not auto-clip
/// to pane bounds". The enforcement system skips these. Use sparingly
/// — by default every Text2d under a pane is clipped, and that is
/// the intended invariant.
#[derive(Component, Copy, Clone, Debug, Default)]
pub struct PaneContentNoClip;

/// Walks each pane's content_root subtree and forces every Text2d's
/// `TextBounds` to fit within the remaining content area. This is the
/// single source of truth for "text cannot escape the pane" — kinds
/// don't have to remember to clip; if they do their own sizing this
/// just clamps it.
///
/// Sprites are deliberately not clamped here: textured sprites
/// (terminal glyph cells, atlas tiles) treat `custom_size` as a render
/// size and shrinking it would distort the glyph rather than clip it.
/// Kinds that want sprite clipping should size their sprites to the
/// pane themselves; the run-button divider, terminal cell pool, and
/// editor caret all already do.
fn enforce_pane_content_bounds(
    panes: Query<(&PaneRect, &PaneChrome), With<PaneTag>>,
    children_q: Query<&Children>,
    transforms: Query<&Transform>,
    text_q: Query<(), With<Text2d>>,
    no_clip_q: Query<(), With<PaneContentNoClip>>,
    mut bounds_q: Query<&mut TextBounds>,
    mut commands: Commands,
) {
    for (rect, chrome) in &panes {
        let content_w = (rect.size.x - 2.0 * MARGIN).max(0.0);
        let content_h = (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0);

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
                // offset.y is <= 0 for typical content (negative goes
                // down); content_h + offset.y = remaining height below
                // the text's top edge.
                let avail_h = (content_h + offset.y).max(0.0);
                let new_bounds = TextBounds {
                    width: Some(avail_w),
                    height: Some(avail_h),
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
