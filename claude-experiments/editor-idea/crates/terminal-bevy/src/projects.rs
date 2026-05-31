//! Projects + sidebar.
//!
//! A project is a named bucket that owns one or more terminal entities.
//! At any moment exactly one project is "active" — terminals belonging
//! to other projects keep running (their worker threads are unaffected),
//! they're just hidden via `Visibility::Hidden` and skipped by the
//! mouse picker. Switching projects flips which set is visible.
//!
//! ## Persistence
//!
//! Project list + active id + the next id to hand out are serialized to
//! `~/.terminal-bevy/projects.json`. Terminal *layouts* (position, size,
//! z, project membership, session id) are also stored there; their raw
//! pty byte streams live alongside in `~/.terminal-bevy/scrollback/<id>.bytes`
//! so the visible scrollback survives restarts. The shell process itself
//! can't survive — restored terminals always get a fresh shell, the
//! prior session's bytes just paint the screen behind it.
//!
//! ## Sidebar UI
//!
//! Drawn as flat sprites + Text2d on the LEFT edge of the window. No
//! `bevy_ui`, matching the rest of this crate. The whole tree is
//! rebuilt whenever `Projects::layout_dirty` is set or the window
//! dimensions change — cheap because it's a few dozen entities, and
//! avoids a separate "follow the window" system that would have to
//! shadow every layout decision.

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::LineHeight;
use serde::{Deserialize, Serialize};

use pane_bevy::{
    spawn_pane_from_registry, FocusedPane, PaneKindMarker, PanePinned, PaneProject, PaneRect,
    PaneRegistry, PaneSnapshot, PaneTag, PaneTitle, MIN_PANE_SIZE,
};

use crate::TerminalSession;

use editor_bevy::EditorFilePath;

use crate::{MonoFont, MonoMetrics, FONT_SIZE};

pub const SIDEBAR_DEFAULT_WIDTH: f32 = 220.0;
pub const SIDEBAR_MIN_WIDTH: f32 = 160.0;
pub const SIDEBAR_MAX_WIDTH: f32 = 480.0;
/// Hit area for the resize handle is `2 * SIDEBAR_RESIZE_HALF` pixels
/// wide, centered on the sidebar's right edge — wider than the visible
/// divider so the user doesn't have to be pixel-perfect.
const SIDEBAR_RESIZE_HALF: f32 = 4.0;
/// Far enough above any terminal `rect.z` (terminals start at 1.0 and
/// only step up by 1.0 per focus-bring-to-front) but well inside the
/// default Bevy 2D camera's far plane (1000) — z values past that get
/// silently clipped, which manifests as a black screen on first run.
const SIDEBAR_Z: f32 = 500.0;

// Muted, modern palette — no saturated reds/greens/blues. Active state
// uses a thin accent stripe instead of a full-row colour wash.
/// Theme-derived sidebar colors, resolved once per `sidebar_layout`
/// call. Preset switches retone the whole sidebar in the same frame
/// because layout rebuilds on `theme.is_changed()`.
struct SidebarPalette {
    bg: Color,
    divider: Color,
    row_active_bg: Color,
    row_renaming_bg: Color,
    active_stripe: Color,
    edit_underline: Color,
    text: Color,
    text_dim: Color,
    text_faint: Color,
}

fn sidebar_palette(theme: &style_bevy::Theme) -> SidebarPalette {
    use style_bevy::tokens as t;
    let c = |id| Color::LinearRgba(theme.color(id));
    SidebarPalette {
        bg: c(t::SIDEBAR_BG),
        divider: c(t::CHROME_DIVIDER),
        row_active_bg: c(t::SIDEBAR_ROW_ACTIVE_BG),
        row_renaming_bg: c(t::SIDEBAR_ROW_RENAMING_BG),
        active_stripe: c(t::ACCENT),
        edit_underline: c(t::ACCENT),
        text: c(t::FG),
        text_dim: c(t::FG_MUTED),
        text_faint: c(t::SIDEBAR_TEXT_FAINT),
    }
}

const HEADER_H: f32 = 36.0;
const ROW_H: f32 = 28.0;
const ROW_PAD_X: f32 = 14.0;
const STRIPE_W: f32 = 3.0;
const DELETE_W: f32 = 22.0;
const DIVIDER_H: f32 = 1.0;
const TEXT_FONT_SIZE: f32 = 13.0;
const HEADER_FONT_SIZE: f32 = 12.0;

const NEW_TERMINAL_OFFSET: f32 = 28.0;

// ---------- Persistence ----------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectData {
    pub id: u64,
    pub name: String,
    /// Remembered working directory for terminals spawned in this
    /// project. Populated by the inference layer the first time the
    /// user `cd`s into a plausible project root (the classifier in
    /// [`crate::inferences_pane`]'s feed decides); used at terminal
    /// spawn time as the initial cwd. `None` means "no remembered
    /// default" → fall back to `$HOME` like every other terminal.
    /// `serde(default)` keeps old projects.json files loadable.
    #[serde(default)]
    pub default_cwd: Option<String>,
}

/// Legacy terminal-only snapshot from before the pane unification.
/// Kept for `serde(default)` deserialization so old projects.json files
/// still load; on save we always write the new `panes` field instead.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TerminalSnapshot {
    pub session_id: u64,
    pub project_id: u64,
    pub pos: [f32; 2],
    pub size: [f32; 2],
    pub z: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PersistedState {
    #[serde(default)]
    projects: Vec<ProjectData>,
    #[serde(default)]
    active: Option<u64>,
    #[serde(default)]
    next_id: u64,
    #[serde(default)]
    sidebar_width: Option<f32>,
    /// Legacy field — populated when reading old saves; never written.
    #[serde(default, skip_serializing)]
    terminals: Vec<TerminalSnapshot>,
    /// All panes (any kind) with their kind-specific config blob.
    #[serde(default)]
    panes: Vec<PaneSnapshot>,
    #[serde(default)]
    next_terminal_id: u64,
    /// Per-project canvas view (pan + zoom). Keyed by project id as a
    /// string for JSON friendliness. `serde(default)` keeps old saves
    /// loadable; missing projects use the default view.
    #[serde(default)]
    canvas_views: std::collections::HashMap<String, crate::canvas::CanvasViewState>,
}

fn save_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".terminal-bevy");
    Some(p)
}

fn load_persisted() -> PersistedState {
    let Some(dir) = save_path() else {
        return PersistedState::default();
    };
    let file = dir.join("projects.json");
    let Ok(bytes) = fs::read(&file) else {
        return PersistedState::default();
    };
    serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        eprintln!(
            "[projects] failed to parse {}: {} — starting empty",
            file.display(),
            e
        );
        PersistedState::default()
    })
}

fn save_persisted(state: &PersistedState) {
    let Some(dir) = save_path() else {
        return;
    };
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("[projects] mkdir {}: {}", dir.display(), e);
        return;
    }
    let file = dir.join("projects.json");
    let tmp = dir.join("projects.json.tmp");
    let bytes = match serde_json::to_vec_pretty(state) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[projects] serialize failed: {}", e);
            return;
        }
    };
    let write_result = (|| -> std::io::Result<()> {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
        fs::rename(&tmp, &file)
    })();
    if let Err(e) = write_result {
        eprintln!("[projects] write {}: {}", file.display(), e);
    }
}

// ---------- Resource state ----------

#[derive(Resource, Default)]
pub struct Projects {
    pub list: Vec<ProjectData>,
    pub active: Option<u64>,
    pub next_id: u64,
    /// Counter for `TerminalSession` ids. Bumped on every spawn (new or
    /// restored) so we never collide with an existing scrollback file.
    pub next_terminal_id: u64,
    /// Set when the on-disk file is out of date.
    pub dirty: bool,
    /// Set when the sidebar entity tree needs rebuilding (rows added /
    /// removed / renamed / active-project changed).
    pub layout_dirty: bool,
    /// Set when terminal layouts (positions, sizes, membership) have
    /// changed and need flushing to disk. Saved separately from `dirty`
    /// because we want to debounce drag/resize bursts to mouse-up; the
    /// project-state save is fine to fire immediately.
    pub terminals_dirty: bool,
    /// Per-project unread BEL counter. Bumped when a terminal in a
    /// hidden context (window unfocused, or project not active) rings
    /// the bell; cleared when the user is actually looking at that
    /// project. Session-only — not persisted to projects.json.
    pub unread_bells: std::collections::HashMap<u64, u64>,
}

impl Projects {
    fn from_persisted(p: PersistedState) -> Self {
        let next_id = p.next_id.max(p.projects.iter().map(|p| p.id + 1).max().unwrap_or(1));
        let legacy_max_session = p
            .terminals
            .iter()
            .map(|t| t.session_id + 1)
            .max()
            .unwrap_or(0);
        let panes_max_session = p
            .panes
            .iter()
            .filter(|p| p.kind == "terminal")
            .filter_map(|p| p.config.get("session_id").and_then(|v| v.as_u64()))
            .map(|id| id + 1)
            .max()
            .unwrap_or(0);
        let next_terminal_id = p
            .next_terminal_id
            .max(legacy_max_session.max(panes_max_session))
            .max(1);
        Self {
            list: p.projects,
            active: p.active,
            next_id,
            next_terminal_id,
            dirty: false,
            layout_dirty: true,
            terminals_dirty: false,
            unread_bells: std::collections::HashMap::new(),
        }
    }
    pub fn allocate_terminal_id(&mut self) -> u64 {
        let id = self.next_terminal_id.max(1);
        self.next_terminal_id = id + 1;
        self.dirty = true;
        id
    }
    pub fn create(&mut self) -> u64 {
        let id = self.next_id.max(1);
        self.next_id = id + 1;
        self.list.push(ProjectData {
            id,
            name: format!("Project {}", id),
            default_cwd: None,
        });
        if self.active.is_none() {
            self.active = Some(id);
        }
        self.dirty = true;
        self.layout_dirty = true;
        id
    }
    pub fn delete(&mut self, id: u64) {
        let before = self.list.len();
        self.list.retain(|p| p.id != id);
        if self.list.len() == before {
            return;
        }
        if self.active == Some(id) {
            self.active = self.list.first().map(|p| p.id);
        }
        self.dirty = true;
        self.layout_dirty = true;
    }
    pub fn set_active(&mut self, id: u64) {
        if self.active != Some(id) {
            self.active = Some(id);
            self.dirty = true;
            self.layout_dirty = true;
        }
    }
    pub fn rename(&mut self, id: u64, new_name: String) {
        for p in &mut self.list {
            if p.id == id {
                if p.name != new_name {
                    p.name = new_name;
                    self.dirty = true;
                    self.layout_dirty = true;
                }
                return;
            }
        }
    }
    pub fn name_of(&self, id: u64) -> Option<&str> {
        self.list
            .iter()
            .find(|p| p.id == id)
            .map(|p| p.name.as_str())
    }
    pub fn default_cwd_of(&self, id: u64) -> Option<&str> {
        self.list
            .iter()
            .find(|p| p.id == id)
            .and_then(|p| p.default_cwd.as_deref())
    }
    /// Set or clear a project's remembered default cwd. Marks the
    /// projects.json dirty so it flushes to disk on the next save tick.
    /// Returns true if the value actually changed.
    pub fn set_default_cwd(&mut self, id: u64, cwd: Option<String>) -> bool {
        for p in &mut self.list {
            if p.id == id {
                if p.default_cwd != cwd {
                    p.default_cwd = cwd;
                    self.dirty = true;
                    return true;
                }
                return false;
            }
        }
        false
    }
    /// Bump the unread bell counter for one project. Marks layout
    /// dirty so the sidebar redraws with the new badge.
    pub fn bump_unread(&mut self, project_id: u64) {
        *self.unread_bells.entry(project_id).or_insert(0) += 1;
        self.layout_dirty = true;
    }
    /// Clear a project's unread count. No-op if it's already zero.
    /// Returns true if anything actually changed (so the caller can
    /// decide whether to mark layout dirty).
    pub fn clear_unread(&mut self, project_id: u64) -> bool {
        match self.unread_bells.remove(&project_id) {
            Some(n) if n > 0 => {
                self.layout_dirty = true;
                true
            }
            _ => false,
        }
    }
    pub fn unread_total(&self) -> u64 {
        self.unread_bells.values().copied().sum()
    }
}

#[derive(Resource, Default)]
pub struct Renaming {
    pub id: Option<u64>,
    pub buffer: String,
}

/// Side-channel for spawn / restore / open-file actions that need
/// exclusive World access (worker setup is `!Send`, spawn registration
/// goes through `PaneRegistry`). Pane closes are owned by pane-bevy's
/// own `PendingPaneActions`.
#[derive(Resource, Default)]
pub struct PendingActions {
    /// Spawn a new pane of any registered kind.
    pub new_panes: Vec<NewPaneRequest>,
    /// Restore persisted panes at startup. Each is dispatched to the
    /// registered kind's `spawn` callback with its saved config blob.
    pub restore_panes: Vec<PaneSnapshot>,
    /// Files to open into a new editor pane (Cmd+O picker, `tbopen`
    /// CLI, etc.).
    pub open_files: Vec<OpenFileRequest>,
}

/// Request to spawn one new pane of a given kind.
#[derive(Debug, Clone)]
pub struct NewPaneRequest {
    pub kind: &'static str,
    pub project_id: u64,
    /// Optional window-space top-left for the new pane. `None` cascades
    /// from a default position based on how many panes the project has.
    pub origin: Option<Vec2>,
    /// Optional pixel size. `None` uses the kind's `default_size` from
    /// `PaneRegistry` (clamped to `MIN_PANE_SIZE`).
    pub size: Option<Vec2>,
    pub config: serde_json::Value,
}

/// Request to load a file into a new editor pane. Project is resolved
/// when the request is consumed — by then `Projects` is up to date.
#[derive(Debug, Clone)]
pub struct OpenFileRequest {
    pub path: PathBuf,
    pub project: OpenProjectTarget,
    /// Optional window-space top-left for the new pane. `None` means
    /// cascade from the default canvas position.
    pub origin: Option<Vec2>,
}

#[derive(Debug, Clone)]
pub enum OpenProjectTarget {
    /// Whichever project is currently active.
    Active,
    /// Project with the given id (no-op if it's been deleted).
    ById(u64),
    /// First project whose name matches case-insensitively (no-op if
    /// nothing matches). Used by the `tbopen --project NAME` flag.
    ByName(String),
}

/// Re-exported from `pane_bevy` so call sites in this crate keep their
/// existing import paths.
pub use pane_bevy::InputConsumed;

/// Sidebar geometry. Only the width is mutable; height + position are
/// driven by the window. Persisted as part of the projects file.
#[derive(Resource)]
pub struct Sidebar {
    pub width: f32,
}

impl Default for Sidebar {
    fn default() -> Self {
        Self {
            width: SIDEBAR_DEFAULT_WIDTH,
        }
    }
}

/// Live drag state for the sidebar resize handle. `active` flips on
/// when the user mouse-downs in the handle hit area; `grab_offset_x`
/// stores `pt.x - sidebar.width` so the handle stays glued under the
/// cursor across drags. `dirty_pending` lets us batch the disk-save to
/// mouse-up instead of every drag tick.
#[derive(Resource, Default)]
struct SidebarResize {
    active: bool,
    grab_offset_x: f32,
    dirty_pending: bool,
}

// ---------- Components ----------

/// Project membership component. Aliased to `pane_bevy::PaneProject` so
/// pane-bevy's visibility/persistence systems can read it directly.
pub type ProjectMembership = PaneProject;

#[derive(Component)]
pub struct SidebarEntity;

#[derive(Component, Copy, Clone, Debug)]
pub enum SidebarHit {
    Project(u64),
    DeleteProject(u64),
    NewProject,
}

/// Bounds in window coords (top-left origin). Recomputed each frame so
/// resizing the window doesn't desync hit-tests from the visuals.
#[derive(Component, Copy, Clone, Debug)]
pub struct SidebarBounds {
    pub min: Vec2,
    pub max: Vec2,
}

// ---------- Plugin ----------

pub struct ProjectsPlugin;

impl Plugin for ProjectsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Projects::default())
            .insert_resource(Sidebar::default())
            .insert_resource(SidebarResize::default())
            .insert_resource(Renaming::default())
            .insert_resource(PendingActions::default())
            .add_systems(
                Startup,
                load_or_seed_projects.after(crate::setup_camera_and_font),
            )
            .add_systems(
                Update,
                (
                    sidebar_resize_drag,
                    sidebar_layout,
                    sidebar_input,
                    rename_keyboard,
                    apply_pending_actions,
                    sync_visibility,
                    refocus_on_project_change,
                    mark_terminals_dirty_on_change,
                    publish_live_terminals,
                    apply_inference_suggestions,
                    save_if_dirty,
                )
                    .chain(),
            );
        // PostUpdate reset for `InputConsumed` is owned by
        // `editor_bevy::EditorEmbedPlugin`.
    }
}

fn load_or_seed_projects(
    mut commands: Commands,
    mut pending: ResMut<PendingActions>,
) {
    let persisted = load_persisted();
    let sidebar_width = persisted
        .sidebar_width
        .unwrap_or(SIDEBAR_DEFAULT_WIDTH)
        .clamp(SIDEBAR_MIN_WIDTH, SIDEBAR_MAX_WIDTH);
    // Restore per-project canvas views (pan + zoom).
    let mut canvas_view = crate::canvas::CanvasView::default();
    for (k, v) in &persisted.canvas_views {
        if let Ok(id) = k.parse::<u64>() {
            let mut state = *v;
            state.clamp_zoom();
            canvas_view.per_project.insert(id, state);
        }
    }
    commands.insert_resource(canvas_view);
    let mut projects = Projects::from_persisted(persisted.clone());
    if projects.list.is_empty() {
        projects.create();
        projects.dirty = true;
    }
    if projects.active.is_none() {
        projects.active = projects.list.first().map(|p| p.id);
    }

    // Queue restore for any pane whose project still exists. The
    // exclusive `apply_pending_actions` system spawns them on the
    // first Update tick. Old saves only had `terminals`; convert them
    // into PaneSnapshot form so the unified restore path handles them.
    let known_projects: std::collections::HashSet<u64> =
        projects.list.iter().map(|p| p.id).collect();
    for snap in persisted.panes {
        let belongs = snap
            .project_id
            .map(|p| known_projects.contains(&p))
            .unwrap_or(true);
        if belongs {
            pending.restore_panes.push(snap);
        }
    }
    for legacy in persisted.terminals {
        if !known_projects.contains(&legacy.project_id) {
            continue;
        }
        pending.restore_panes.push(PaneSnapshot {
            kind: "terminal".into(),
            project_id: Some(legacy.project_id),
            pos: legacy.pos,
            size: legacy.size,
            z: legacy.z,
            config: serde_json::json!({ "session_id": legacy.session_id }),
            pinned: false,
        });
    }

    commands.insert_resource(projects);
    commands.insert_resource(Sidebar {
        width: sidebar_width,
    });
}

// ---------- Sidebar layout ----------

/// Window dims at the time of the last sidebar rebuild — when the
/// window resizes we must rebuild so the bg sprite + hit-test bounds
/// follow it.
#[derive(Default)]
struct LastWindowDims(Option<Vec2>);

/// Rebuild the sidebar entity tree when project state, rename state, or
/// window size changes. Otherwise this system early-returns.
fn sidebar_layout(
    mut commands: Commands,
    windows: Query<&Window>,
    sidebar: Res<Sidebar>,
    theme: Res<style_bevy::Theme>,
    mut projects: ResMut<Projects>,
    renaming: Res<Renaming>,
    font: Res<MonoFont>,
    metrics: Res<MonoMetrics>,
    existing: Query<Entity, With<SidebarEntity>>,
    mut last_dims: Local<LastWindowDims>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let win_w = window.width();
    let win_h = window.height();
    let width = sidebar.width;
    let dims = Vec2::new(win_w, win_h);
    let palette = sidebar_palette(&theme);

    let dims_changed = last_dims.0 != Some(dims);
    let mut needs_rebuild =
        projects.layout_dirty || dims_changed || theme.is_changed();
    if existing.iter().next().is_none() {
        needs_rebuild = true;
    }
    if !needs_rebuild {
        return;
    }
    last_dims.0 = Some(dims);
    for e in &existing {
        commands.entity(e).despawn();
    }

    // Sidebar lives at the LEFT edge. Window coords are top-left origin
    // (matching cursor_position), world coords have the camera at (0,0)
    // with y-up — so x=0 in window is x=-win_w/2 in world, y=0 in window
    // is y=win_h/2 in world.
    let sidebar_origin_x_window = 0.0;
    let world_left_edge = -win_w * 0.5;
    let world_top_edge = win_h * 0.5;

    // Container bg — full height.
    commands.spawn((
        SidebarEntity,
        Sprite {
            color: palette.bg,
            custom_size: Some(Vec2::new(width, win_h)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(world_left_edge, world_top_edge, SIDEBAR_Z),
    ));

    // Right-edge divider (1px) so the sidebar has a clean shoulder
    // against the canvas without needing a contrasting bg.
    commands.spawn((
        SidebarEntity,
        Sprite {
            color: palette.divider,
            custom_size: Some(Vec2::new(DIVIDER_H, win_h)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(
            world_left_edge + width - DIVIDER_H,
            world_top_edge,
            SIDEBAR_Z + 0.05,
        ),
    ));

    // Header label — uppercase, dim, like a section caption in a modern
    // sidebar. No header-bar bg; just text on the sidebar bg with a
    // divider underneath.
    {
        let line_h = HEADER_FONT_SIZE * 1.4;
        let pad_y = ((HEADER_H - line_h) * 0.5).max(0.0);
        commands.spawn((
            SidebarEntity,
            Text2d::new("PROJECTS"),
            TextFont {
                font: font.0.clone(),
                font_size: HEADER_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(line_h),
            TextColor(palette.text_faint),
            Anchor::TOP_LEFT,
            Transform::from_xyz(
                world_left_edge + ROW_PAD_X,
                world_top_edge - pad_y,
                SIDEBAR_Z + 0.2,
            ),
        ));
    }
    // Header divider.
    commands.spawn((
        SidebarEntity,
        Sprite {
            color: palette.divider,
            custom_size: Some(Vec2::new(width - DIVIDER_H, DIVIDER_H)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(
            world_left_edge,
            world_top_edge - HEADER_H,
            SIDEBAR_Z + 0.05,
        ),
    ));

    // Project rows.
    let rows_top_window = HEADER_H;
    for (idx, proj) in projects.list.iter().enumerate() {
        let row_top_window = rows_top_window + idx as f32 * ROW_H;
        let row_top_world = world_top_edge - row_top_window;
        let active = projects.active == Some(proj.id);
        let renaming_this = renaming.id == Some(proj.id);

        // Row bg — only painted when active or renaming. Inactive rows
        // sit on the sidebar bg with no separator: the spacing from
        // ROW_H + the indent is enough visual structure.
        if active || renaming_this {
            let bg_color = if renaming_this {
                palette.row_renaming_bg
            } else {
                palette.row_active_bg
            };
            commands.spawn((
                SidebarEntity,
                Sprite {
                    color: bg_color,
                    custom_size: Some(Vec2::new(width - DIVIDER_H, ROW_H)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(world_left_edge, row_top_world, SIDEBAR_Z + 0.1),
            ));
        }

        // Active accent stripe (thin coloured bar on the left edge).
        if active {
            commands.spawn((
                SidebarEntity,
                Sprite {
                    color: palette.active_stripe,
                    custom_size: Some(Vec2::new(STRIPE_W, ROW_H)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(world_left_edge, row_top_world, SIDEBAR_Z + 0.15),
            ));
        }

        // Renaming underline — a 2px accent strip along the bottom of
        // the row that reads as the cursor of a text input field.
        if renaming_this {
            commands.spawn((
                SidebarEntity,
                Sprite {
                    color: palette.edit_underline,
                    custom_size: Some(Vec2::new(width - DIVIDER_H, 2.0)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(
                    world_left_edge,
                    row_top_world - (ROW_H - 2.0),
                    SIDEBAR_Z + 0.15,
                ),
            ));
        }

        // Project pick hit-region — covers the row minus the delete glyph.
        commands.spawn((
            SidebarEntity,
            Transform::from_xyz(world_left_edge, row_top_world, SIDEBAR_Z + 0.05),
            SidebarHit::Project(proj.id),
            SidebarBounds {
                min: Vec2::new(sidebar_origin_x_window, row_top_window),
                max: Vec2::new(
                    sidebar_origin_x_window + width - DELETE_W - DIVIDER_H,
                    row_top_window + ROW_H,
                ),
            },
        ));

        // Label.
        let label = if renaming_this {
            renaming.buffer.clone()
        } else {
            proj.name.clone()
        };
        let label_color = if active || renaming_this {
            palette.text
        } else {
            palette.text_dim
        };
        {
            let line_h = TEXT_FONT_SIZE * 1.4;
            let pad_y = ((ROW_H - line_h) * 0.5).max(0.0);
            commands.spawn((
                SidebarEntity,
                Text2d::new(label),
                TextFont {
                    font: font.0.clone(),
                    font_size: TEXT_FONT_SIZE,
                    ..default()
                },
                LineHeight::Px(line_h),
                TextColor(label_color),
                Anchor::TOP_LEFT,
                Transform::from_xyz(
                    world_left_edge + ROW_PAD_X,
                    row_top_world - pad_y,
                    SIDEBAR_Z + 0.2,
                ),
            ));
        }

        // Caret — real sprite (not a U+2502 glyph, which renders too
        // low because box-drawing chars don't share the letter baseline).
        // Positioned via monospace cell-width scaled from FONT_SIZE→TEXT_FONT_SIZE,
        // and vertically centred in the row so it doesn't sit at the descender.
        if renaming_this {
            let char_advance = metrics.cell_width * (TEXT_FONT_SIZE / FONT_SIZE);
            let caret_w = 2.0;
            let caret_h = 16.0;
            let caret_x = world_left_edge
                + ROW_PAD_X
                + renaming.buffer.chars().count() as f32 * char_advance;
            let caret_top_y = row_top_world - (ROW_H - caret_h) * 0.5;
            commands.spawn((
                SidebarEntity,
                Sprite {
                    color: palette.edit_underline,
                    custom_size: Some(Vec2::new(caret_w, caret_h)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(caret_x, caret_top_y, SIDEBAR_Z + 0.25),
            ));
        }

        // Unread bell badge — right-aligned just before the delete X
        // when the project has any unseen bells. Uses the active-stripe
        // colour so it reads as a "this needs attention" cue regardless
        // of which project is currently selected.
        if let Some(&n) = projects.unread_bells.get(&proj.id)
            && n > 0
        {
            let badge_text = if n > 99 {
                "99+".to_string()
            } else {
                n.to_string()
            };
            let badge_anchor_x_world = world_left_edge + width - DELETE_W - 4.0;
            {
                let line_h = TEXT_FONT_SIZE * 1.4;
                let pad_y = ((ROW_H - line_h) * 0.5).max(0.0);
                commands.spawn((
                    SidebarEntity,
                    Text2d::new(badge_text),
                    TextFont {
                        font: font.0.clone(),
                        font_size: TEXT_FONT_SIZE,
                        ..default()
                    },
                    LineHeight::Px(line_h),
                    TextColor(palette.active_stripe),
                    Anchor::TOP_RIGHT,
                    Transform::from_xyz(
                        badge_anchor_x_world,
                        row_top_world - pad_y,
                        SIDEBAR_Z + 0.2,
                    ),
                ));
            }
        }

        // Delete glyph — just a dim × at the right edge of the row.
        // No filled background; the bounds are still a tappable rect.
        let delete_x_window = sidebar_origin_x_window + width - DELETE_W;
        let delete_x_world = world_left_edge + width - DELETE_W;
        commands.spawn((
            SidebarEntity,
            Transform::from_xyz(delete_x_world, row_top_world, SIDEBAR_Z + 0.05),
            SidebarHit::DeleteProject(proj.id),
            SidebarBounds {
                min: Vec2::new(delete_x_window, row_top_window),
                max: Vec2::new(delete_x_window + DELETE_W, row_top_window + ROW_H),
            },
        ));
        {
            let glyph_size = TEXT_FONT_SIZE + 1.0;
            let line_h = glyph_size * 1.4;
            let pad_y = ((ROW_H - line_h) * 0.5).max(0.0);
            commands.spawn((
                SidebarEntity,
                Text2d::new("\u{00D7}"), // multiplication sign — looks better than ASCII 'x'
                TextFont {
                    font: font.0.clone(),
                    font_size: glyph_size,
                    ..default()
                },
                LineHeight::Px(line_h),
                TextColor(palette.text_faint),
                Anchor::TOP_LEFT,
                Transform::from_xyz(
                    delete_x_world + 6.0,
                    row_top_world - pad_y,
                    SIDEBAR_Z + 0.2,
                ),
            ));
        }
    }

    // Divider before the "+ New Project" row.
    let after_rows_window = rows_top_window + projects.list.len() as f32 * ROW_H;
    commands.spawn((
        SidebarEntity,
        Sprite {
            color: palette.divider,
            custom_size: Some(Vec2::new(width - DIVIDER_H, DIVIDER_H)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(
            world_left_edge,
            world_top_edge - after_rows_window,
            SIDEBAR_Z + 0.05,
        ),
    ));

    let new_proj_top_window = after_rows_window + DIVIDER_H;
    let new_proj_top_world = world_top_edge - new_proj_top_window;
    // "+ New Project" — same row style as project rows. Hit area only,
    // no painted bg until you hover (we skip hover for now).
    commands.spawn((
        SidebarEntity,
        Transform::from_xyz(world_left_edge, new_proj_top_world, SIDEBAR_Z + 0.05),
        SidebarHit::NewProject,
        SidebarBounds {
            min: Vec2::new(sidebar_origin_x_window, new_proj_top_window),
            max: Vec2::new(
                sidebar_origin_x_window + width - DIVIDER_H,
                new_proj_top_window + ROW_H,
            ),
        },
    ));
    {
        let line_h = TEXT_FONT_SIZE * 1.4;
        let pad_y = ((ROW_H - line_h) * 0.5).max(0.0);
        commands.spawn((
            SidebarEntity,
            Text2d::new("+  New Project"),
            TextFont {
                font: font.0.clone(),
                font_size: TEXT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(line_h),
            TextColor(palette.text_dim),
            Anchor::TOP_LEFT,
            Transform::from_xyz(
                world_left_edge + ROW_PAD_X,
                new_proj_top_world - pad_y,
                SIDEBAR_Z + 0.2,
            ),
        ));
    }

    projects.layout_dirty = false;
}

// ---------- Sidebar input ----------

/// Tracks click state for double-click detection on project rows.
#[derive(Resource, Default)]
pub struct ClickTracker {
    last_project: Option<u64>,
    last_time: f64,
}

pub fn sidebar_input(
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    consumed: Res<InputConsumed>,
    sidebar: Res<Sidebar>,
    time: Res<Time>,
    hits: Query<(&SidebarHit, &SidebarBounds)>,
    mut projects: ResMut<Projects>,
    mut renaming: ResMut<Renaming>,
    mut tracker: Local<ClickTracker>,
) {
    if consumed.0 {
        return;
    }
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };
    // Sidebar is on the LEFT now — anything past `sidebar.width` is canvas.
    if pt.x >= sidebar.width {
        // Click hit the canvas — if we were renaming, commit on click-out.
        if renaming.id.is_some() {
            commit_rename(&mut projects, &mut renaming);
        }
        return;
    }

    // Pick the topmost hit. Only the row bgs / buttons carry SidebarHit
    // so we don't need z-sorting — they're disjoint by construction.
    let mut chosen: Option<SidebarHit> = None;
    for (hit, b) in &hits {
        if pt.x >= b.min.x && pt.x <= b.max.x && pt.y >= b.min.y && pt.y <= b.max.y {
            chosen = Some(*hit);
            break;
        }
    }
    let Some(hit) = chosen else {
        return;
    };

    match hit {
        SidebarHit::Project(id) => {
            // Double-click on the already-active project starts rename mode.
            let now = time.elapsed_secs_f64();
            let is_double = tracker.last_project == Some(id) && now - tracker.last_time < 0.4;
            tracker.last_project = Some(id);
            tracker.last_time = now;

            if is_double {
                let current = projects.name_of(id).unwrap_or("").to_string();
                renaming.id = Some(id);
                renaming.buffer = current;
            } else {
                if renaming.id.is_some() {
                    commit_rename(&mut projects, &mut renaming);
                }
                projects.set_active(id);
            }
        }
        SidebarHit::DeleteProject(id) => {
            // Drop the project from state. apply_pending_actions sweeps
            // any terminals whose ProjectMembership now points at a
            // missing project and shuts them down.
            projects.delete(id);
            if renaming.id == Some(id) {
                renaming.id = None;
                renaming.buffer.clear();
            }
        }
        SidebarHit::NewProject => {
            if renaming.id.is_some() {
                commit_rename(&mut projects, &mut renaming);
            }
            let id = projects.create();
            projects.set_active(id);
            // Open rename immediately so the user can name it. Start
            // empty so typing replaces the auto-generated "Project N"
            // instead of appending to it.
            renaming.id = Some(id);
            renaming.buffer.clear();
        }
    }
}

fn commit_rename(projects: &mut Projects, renaming: &mut Renaming) {
    if let Some(id) = renaming.id.take() {
        let mut name = std::mem::take(&mut renaming.buffer);
        let trimmed = name.trim();
        if trimmed.is_empty() {
            // Reject empty — keep the old name. Still need a redraw to
            // shed the caret + edit-state styling on the row.
            projects.layout_dirty = true;
        } else {
            name = trimmed.to_string();
            projects.rename(id, name);
        }
    }
}

// ---------- Rename keyboard ----------

fn rename_keyboard(
    mut events: MessageReader<KeyboardInput>,
    mut renaming: ResMut<Renaming>,
    mut projects: ResMut<Projects>,
) {
    if renaming.id.is_none() {
        return;
    }
    let mut changed = false;
    for ev in events.read() {
        if !ev.state.is_pressed() {
            continue;
        }
        match (&ev.key_code, &ev.logical_key) {
            (KeyCode::Enter, _) | (KeyCode::NumpadEnter, _) => {
                commit_rename(&mut projects, &mut renaming);
                return;
            }
            (KeyCode::Escape, _) => {
                renaming.id = None;
                renaming.buffer.clear();
                projects.layout_dirty = true;
                return;
            }
            (KeyCode::Backspace, _) => {
                renaming.buffer.pop();
                changed = true;
            }
            (_, Key::Character(s)) => {
                for c in s.chars() {
                    if !c.is_control() {
                        renaming.buffer.push(c);
                        changed = true;
                    }
                }
            }
            (_, Key::Space) => {
                renaming.buffer.push(' ');
                changed = true;
            }
            _ => {}
        }
    }
    if changed {
        projects.layout_dirty = true;
    }
}

// ---------- Apply pending actions ----------

/// Exclusive system — pane spawning and registry callbacks both need
/// `&mut World`. Restores first, then handles project-deletion sweeps,
/// then new-pane requests, then open-file requests.
fn apply_pending_actions(world: &mut World) {
    let actions = std::mem::take(&mut *world.resource_mut::<PendingActions>());
    let sidebar_width = world.resource::<Sidebar>().width;

    // Restore persisted panes first so they appear before any new ones
    // queued in the same frame.
    for snap in actions.restore_panes {
        restore_pane(world, snap);
        world.resource_mut::<Projects>().layout_dirty = true;
    }

    // Project deletion sweep: any pane whose project no longer exists
    // is queued for close (pane-bevy's apply_pending_pane_actions runs
    // the kind's on_close + despawns).
    let active_ids: std::collections::HashSet<u64> = world
        .resource::<Projects>()
        .list
        .iter()
        .map(|p| p.id)
        .collect();
    let orphans: Vec<Entity> = {
        let mut q = world.query::<(Entity, &PaneProject, &PaneTag)>();
        q.iter(world)
            .filter_map(|(e, m, _)| (!active_ids.contains(&m.0)).then_some(e))
            .collect()
    };
    if !orphans.is_empty() {
        let mut close_q = world.resource_mut::<pane_bevy::PendingPaneActions>();
        for e in orphans {
            close_q.close.push(e);
        }
    }

    // Spawn requested panes (radial menu, RunButton creation, etc.).
    for req in actions.new_panes {
        let pos = req.origin.unwrap_or_else(|| {
            let count_in_project = pane_count_in_project(world, &req.kind, req.project_id);
            cascade_pos(sidebar_width, count_in_project)
        });
        let size = req
            .size
            .unwrap_or_else(|| {
                world
                    .resource::<PaneRegistry>()
                    .get(req.kind)
                    .map(|s| s.default_size)
                    .unwrap_or(Vec2::new(560.0, 360.0))
            })
            .max(MIN_PANE_SIZE);
        let next_z = pane_bevy::next_pane_z(world);
        let rect = PaneRect {
            pos,
            size,
            z: next_z,
        };
        if let Some(entity) = spawn_pane_from_registry(
            world,
            kind_to_static(req.kind),
            kind_display_name(world, req.kind),
            rect,
            Some(req.project_id),
            &req.config,
        ) {
            world.resource_mut::<FocusedPane>().0 = Some(entity);
            let mut projects = world.resource_mut::<Projects>();
            projects.layout_dirty = true;
            projects.terminals_dirty = true;
        }
    }

    // Open files into editor panes.
    for req in actions.open_files {
        let project_id = match resolve_project(&req.project, world.resource::<Projects>()) {
            Some(id) => id,
            None => {
                eprintln!(
                    "[open_file] no matching project for {:?}; dropping request for {}",
                    req.project,
                    req.path.display()
                );
                continue;
            }
        };
        let text = match std::fs::read_to_string(&req.path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!(
                    "[open_file] read {} failed: {}",
                    req.path.display(),
                    e
                );
                continue;
            }
        };
        let pos = req.origin.unwrap_or_else(|| {
            let count_in_project = pane_count_in_project(world, "editor", project_id);
            cascade_pos(sidebar_width, count_in_project)
        });
        let next_z = pane_bevy::next_pane_z(world);
        let rect = PaneRect {
            pos,
            size: Vec2::new(720.0, 480.0),
            z: next_z,
        };
        let config = serde_json::json!({
            "text": text,
            "path": req.path.to_string_lossy(),
        });
        if let Some(entity) =
            spawn_pane_from_registry(world, "editor", "Editor", rect, Some(project_id), &config)
        {
            // EditorFilePath is also added by the editor's spawn callback
            // when the config carries `path`; we add it here too in case
            // future kinds want a different file-tagging convention.
            world
                .entity_mut(entity)
                .insert(EditorFilePath(req.path.clone()));
            if world.resource::<Projects>().active == Some(project_id) {
                world.resource_mut::<FocusedPane>().0 = Some(entity);
            }
            world.resource_mut::<Projects>().layout_dirty = true;
        }
    }
}

/// Restore one persisted pane via the registry. Reserves any embedded
/// session id in the allocator so subsequent spawns don't collide.
fn restore_pane(world: &mut World, snap: PaneSnapshot) {
    if snap.kind == "terminal" {
        if let Some(id) = snap.config.get("session_id").and_then(|v| v.as_u64()) {
            let mut projects = world.resource_mut::<Projects>();
            if projects.next_terminal_id <= id {
                projects.next_terminal_id = id + 1;
            }
        }
    }
    let kind_static = kind_to_static(&snap.kind);
    let display = kind_display_name(world, &snap.kind);
    // PaneRect is canvas-space now — restore directly from the snapshot.
    let rect = PaneRect {
        pos: Vec2::new(snap.pos[0], snap.pos[1]),
        size: Vec2::new(snap.size[0], snap.size[1]),
        z: snap.z,
    };
    let entity = spawn_pane_from_registry(
        world,
        kind_static,
        display,
        rect,
        snap.project_id,
        &snap.config,
    );
    if let Some(e) = entity {
        // Reapply the pin marker if this pane was pinned at save time.
        if snap.pinned {
            world.entity_mut(e).insert(PanePinned);
        }
    }
}

/// Look the kind up in the registry to get its registered `kind`
/// `&'static str` (so callers can pass owned `String` and we still hand
/// pane-bevy a static slice). Falls back to leaking the input if the
/// kind isn't registered, so the spawn-from-registry call still finds
/// it stored on the entity for diagnostics.
pub(crate) fn kind_to_static(kind: &str) -> &'static str {
    match kind {
        "terminal" => "terminal",
        "editor" => "editor",
        "run-button" => "run-button",
        other => Box::leak(other.to_string().into_boxed_str()),
    }
}

fn kind_display_name(world: &World, kind: &str) -> String {
    world
        .resource::<PaneRegistry>()
        .get(kind)
        .map(|s| s.display_name.to_string())
        .unwrap_or_else(|| kind.to_string())
}

fn pane_count_in_project(world: &mut World, kind: &str, project_id: u64) -> usize {
    let mut q = world.query::<(&PaneProject, &PaneKindMarker)>();
    q.iter(world)
        .filter(|(m, k)| k.0 == kind && m.0 == project_id)
        .count()
}

fn cascade_pos(sidebar_width: f32, n: usize) -> Vec2 {
    Vec2::new(
        sidebar_width + 60.0 + (n as f32) * NEW_TERMINAL_OFFSET,
        60.0 + (n as f32) * NEW_TERMINAL_OFFSET,
    )
}

pub fn resolve_project(target: &OpenProjectTarget, projects: &Projects) -> Option<u64> {
    match target {
        OpenProjectTarget::Active => projects.active,
        OpenProjectTarget::ById(id) => {
            projects.list.iter().any(|p| p.id == *id).then_some(*id)
        }
        OpenProjectTarget::ByName(name) => projects
            .list
            .iter()
            .find(|p| p.name.eq_ignore_ascii_case(name))
            .map(|p| p.id),
    }
}

/// Map a working directory to the project that owns it: the project
/// whose `default_cwd` is `cwd` or a parent of it. When several match
/// (nested roots), the longest `default_cwd` wins. `None` if no project
/// has a `default_cwd` that contains `cwd` — callers treat that as
/// "unscoped / global". Mirrors the "this project = cwd's project" rule.
pub fn project_for_cwd(cwd: &std::path::Path, projects: &Projects) -> Option<u64> {
    let cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    let mut best: Option<(usize, u64)> = None;
    for p in &projects.list {
        let Some(dc) = p.default_cwd.as_deref() else {
            continue;
        };
        let root = std::path::Path::new(dc);
        let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        if cwd.starts_with(&root) {
            let depth = root.components().count();
            if best.map_or(true, |(d, _)| depth > d) {
                best = Some((depth, p.id));
            }
        }
    }
    best.map(|(_, id)| id)
}

// ---------- Visibility sync ----------

/// Hide panes whose project is not the active one.
fn sync_visibility(
    projects: Res<Projects>,
    mut panes: Query<(&PaneProject, &mut Visibility), With<PaneTag>>,
) {
    let active = projects.active;
    for (m, mut vis) in &mut panes {
        let want = if Some(m.0) == active {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != want {
            *vis = want;
        }
    }
}

/// When the active project changes, move keyboard focus into the new
/// project — preferring a terminal at the top of its z-stack. Without
/// this, `FocusedPane` keeps pointing at a now-hidden pane in the old
/// project, so typing goes nowhere (or worse, into a hidden widget).
/// `handle_pane_mouse` already filters out hidden panes, so it's only
/// the residual state we have to fix here.
fn refocus_on_project_change(
    projects: Res<Projects>,
    mut last_active: Local<Option<u64>>,
    mut focused: ResMut<FocusedPane>,
    panes: Query<(Entity, &PaneProject, &PaneKindMarker, &PaneRect), With<PaneTag>>,
) {
    if *last_active == projects.active {
        return;
    }
    *last_active = projects.active;

    let Some(active) = projects.active else {
        focused.0 = None;
        return;
    };

    // If the current focus is already in the active project, leave it.
    if let Some(cur) = focused.0 {
        if let Ok((_, proj, _, _)) = panes.get(cur) {
            if proj.0 == active {
                return;
            }
        }
    }

    // Pick a candidate from the active project: prefer terminals, break
    // ties by topmost z so the visually-frontmost pane gets focus.
    let pick = panes
        .iter()
        .filter(|(_, p, _, _)| p.0 == active)
        .max_by(|a, b| {
            let a_term = a.2.0 == crate::PANE_KIND;
            let b_term = b.2.0 == crate::PANE_KIND;
            a_term
                .cmp(&b_term)
                .then(a.3.z.partial_cmp(&b.3.z).unwrap_or(std::cmp::Ordering::Equal))
        })
        .map(|(e, _, _, _)| e);

    focused.0 = pick;
}

// ---------- Live terminals export ----------
//
// `~/.terminal-bevy/terminals.json` is a small "what's open right now"
// snapshot that out-of-process widgets (e.g. claude-context-bars) can
// poll to learn which terminal sessions are open, what project they
// belong to, and their pane title. The full `projects.json` is too
// big and only persisted on mouse-up; this file is updated within a
// frame of any relevant change.

#[derive(Serialize)]
struct LiveTerminalEntry {
    session_id: u64,
    project_id: u64,
    project_name: String,
    title: String,
}

#[derive(Serialize)]
struct LiveTerminals {
    terminals: Vec<LiveTerminalEntry>,
}

fn live_terminals_path() -> Option<PathBuf> {
    save_path().map(|d| d.join("terminals.json"))
}

fn write_live_terminals(state: &LiveTerminals) -> std::io::Result<()> {
    let Some(file) = live_terminals_path() else {
        return Err(std::io::Error::other("no HOME"));
    };
    let Some(dir) = file.parent() else {
        return Err(std::io::Error::other("no parent"));
    };
    fs::create_dir_all(dir)?;
    let bytes = serde_json::to_vec(state)?;
    let tmp = file.with_extension("json.tmp");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, &file)
}

/// Emit `~/.terminal-bevy/terminals.json` whenever the set of open
/// terminals, their project assignment, their title, or the project
/// catalog changes. Hashes the snapshot to suppress redundant writes.
fn publish_live_terminals(
    projects: Res<Projects>,
    terminals: Query<(&TerminalSession, &PaneProject, &PaneTitle), With<PaneTag>>,
    mut last_hash: Local<u64>,
) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let name_of = |id: u64| -> String {
        projects
            .list
            .iter()
            .find(|p| p.id == id)
            .map(|p| p.name.clone())
            .unwrap_or_default()
    };

    let mut entries: Vec<LiveTerminalEntry> = terminals
        .iter()
        .map(|(s, p, t)| LiveTerminalEntry {
            session_id: s.0,
            project_id: p.0,
            project_name: name_of(p.0),
            title: t.0.clone(),
        })
        .collect();
    entries.sort_by_key(|e| e.session_id);

    let mut hasher = DefaultHasher::new();
    for e in &entries {
        e.session_id.hash(&mut hasher);
        e.project_id.hash(&mut hasher);
        e.project_name.hash(&mut hasher);
        e.title.hash(&mut hasher);
    }
    let h = hasher.finish();
    if h == *last_hash {
        return;
    }
    *last_hash = h;

    if let Err(e) = write_live_terminals(&LiveTerminals { terminals: entries }) {
        eprintln!("[projects] write terminals.json: {}", e);
    }
}

// ---------- Inference consumer ----------

/// Confidence below which we don't auto-write the suggestion onto a
/// project. Picked conservatively — the model is small and the cost
/// of a wrong default (new terminals open in a stale directory) is
/// felt every spawn. Above the threshold we still only apply when
/// `good_default = true`.
const INFERENCE_AUTO_APPLY_THRESHOLD: f32 = 0.7;

/// Subscribes to `inference.project_default_cwd_suggested` events
/// from the bus and writes the verdict onto the owning project's
/// `default_cwd`. The owning project is resolved by:
///
///   `terminal_session_id` → `TerminalSession` component
///                         → `ProjectMembership` (== `PaneProject`)
///                         → project id
///
/// If the matching pane has no `ProjectMembership` (standalone test
/// pane) we drop the suggestion silently.
///
/// Only writes when `good_default = true` AND `confidence >=
/// INFERENCE_AUTO_APPLY_THRESHOLD`; the inferences pane still shows
/// every verdict so the user can see what was filtered out.
fn apply_inference_suggestions(
    mut events: MessageReader<claude_bus_bevy::ClaudeBusEvent>,
    panes: Query<(&crate::TerminalSession, Option<&ProjectMembership>)>,
    mut projects: ResMut<Projects>,
) {
    for ev in events.read() {
        if ev.kind != "inference.project_default_cwd_suggested" {
            continue;
        }
        let Ok(payload) = serde_json::from_str::<InferenceSuggestionPayload>(&ev.payload_json)
        else {
            continue;
        };
        if !payload.good_default || payload.confidence < INFERENCE_AUTO_APPLY_THRESHOLD {
            continue;
        }
        let Ok(sid) = ev.terminal_session_id.parse::<u64>() else {
            continue;
        };
        // Find the pane whose TerminalSession matches; read its project
        // membership. There's at most one matching pane (session ids
        // are unique).
        let project_id = panes
            .iter()
            .find(|(ts, _)| ts.0 == sid)
            .and_then(|(_, pm)| pm.map(|p| p.0));
        let Some(project_id) = project_id else {
            continue;
        };
        if projects.set_default_cwd(project_id, Some(payload.cwd.clone())) {
            info!(
                "[projects] project {} default_cwd ← {} (confidence {:.2})",
                project_id, payload.cwd, payload.confidence
            );
        }
    }
}

#[derive(serde::Deserialize)]
struct InferenceSuggestionPayload {
    good_default: bool,
    confidence: f32,
    cwd: String,
}

// ---------- Persistence flush ----------

fn save_if_dirty(
    world: &mut World,
) {
    let mouse_down = world
        .resource::<ButtonInput<MouseButton>>()
        .pressed(MouseButton::Left);
    if mouse_down {
        return;
    }
    {
        let projects = world.resource::<Projects>();
        if !projects.dirty && !projects.terminals_dirty {
            return;
        }
    }
    let panes = collect_pane_snapshots(world);
    let projects = world.resource::<Projects>();
    let sidebar_width = world.resource::<Sidebar>().width;
    let canvas_views: std::collections::HashMap<String, crate::canvas::CanvasViewState> = world
        .resource::<crate::canvas::CanvasView>()
        .per_project
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
    let snapshot = PersistedState {
        projects: projects.list.clone(),
        active: projects.active,
        next_id: projects.next_id,
        sidebar_width: Some(sidebar_width),
        terminals: Vec::new(),
        panes,
        next_terminal_id: projects.next_terminal_id,
        canvas_views,
    };
    save_persisted(&snapshot);
    let mut projects = world.resource_mut::<Projects>();
    projects.dirty = false;
    projects.terminals_dirty = false;
}

/// Walk every PaneTag entity, ask the registered kind for a snapshot,
/// and bundle them into a Vec<PaneSnapshot>. `PaneRect` is canvas-space
/// in the new model, so we just write its values directly.
fn collect_pane_snapshots(world: &mut World) -> Vec<PaneSnapshot> {
    let entries: Vec<(Entity, String, Option<u64>, PaneRect, bool)> = {
        let mut q = world.query::<(
            Entity,
            &PaneKindMarker,
            Option<&PaneProject>,
            &PaneRect,
            Has<PanePinned>,
        )>();
        q.iter(world)
            .map(|(e, k, p, r, pinned)| {
                (e, k.0.to_string(), p.map(|p| p.0), *r, pinned)
            })
            .collect()
    };
    let snapshots: Vec<PaneSnapshot> = entries
        .into_iter()
        .filter_map(|(entity, kind, project_id, rect, pinned)| {
            let snap_fn = world.resource::<PaneRegistry>().get(&kind).map(|s| s.snapshot)?;
            let config = (snap_fn)(world, entity);
            Some(PaneSnapshot {
                kind,
                project_id,
                pos: [rect.pos.x, rect.pos.y],
                size: [rect.size.x, rect.size.y],
                z: rect.z,
                config,
                pinned,
            })
        })
        .collect();
    snapshots
}

/// Mark the persisted layout dirty whenever any pane's rect or
/// project membership changes. Save itself is debounced to mouse-up by
/// `save_if_dirty`.
fn mark_terminals_dirty_on_change(
    rect_changed: Query<
        (),
        (
            With<PaneTag>,
            Or<(Changed<PaneRect>, Changed<PaneProject>)>,
        ),
    >,
    pin_added: Query<(), Added<PanePinned>>,
    mut pin_removed: RemovedComponents<PanePinned>,
    mut projects: ResMut<Projects>,
) {
    if !rect_changed.is_empty() || !pin_added.is_empty() || pin_removed.read().next().is_some() {
        projects.terminals_dirty = true;
    }
}

/// Click + drag inside the resize hit-strip on the sidebar's right
/// edge to resize. Live-updates `Sidebar.width` and triggers a layout
/// rebuild; defers the disk save until mouse-up so we don't write a
/// hundred files during one drag.
fn sidebar_resize_drag(
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut sidebar: ResMut<Sidebar>,
    mut resize: ResMut<SidebarResize>,
    mut consumed: ResMut<InputConsumed>,
    mut projects: ResMut<Projects>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };

    if buttons.just_released(MouseButton::Left) && resize.active {
        resize.active = false;
        if resize.dirty_pending {
            resize.dirty_pending = false;
            // Reuse the project save channel — both project state and
            // sidebar width live in the same JSON file.
            projects.dirty = true;
        }
    }

    let in_handle =
        pt.x >= sidebar.width - SIDEBAR_RESIZE_HALF && pt.x <= sidebar.width + SIDEBAR_RESIZE_HALF;

    if buttons.just_pressed(MouseButton::Left) && in_handle && !resize.active {
        resize.active = true;
        resize.grab_offset_x = pt.x - sidebar.width;
        consumed.0 = true;
        return;
    }

    if resize.active && buttons.pressed(MouseButton::Left) {
        let new_width =
            (pt.x - resize.grab_offset_x).clamp(SIDEBAR_MIN_WIDTH, SIDEBAR_MAX_WIDTH);
        if (new_width - sidebar.width).abs() > 0.5 {
            sidebar.width = new_width;
            projects.layout_dirty = true;
            resize.dirty_pending = true;
        }
        consumed.0 = true;
    }
}
