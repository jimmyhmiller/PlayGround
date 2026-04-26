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

use crate::worker::WorkerMsg;
use crate::{
    scrollback_path, spawn_terminal, FocusedTerminal, MonoFont, TerminalRect, TerminalSession,
    TerminalStore, TerminalTag, MIN_TERMINAL_SIZE,
};

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
const COLOR_SIDEBAR_BG: Color = Color::srgb(0.086, 0.090, 0.102);
const COLOR_DIVIDER: Color = Color::srgb(0.165, 0.170, 0.188);
const COLOR_ROW_ACTIVE_BG: Color = Color::srgb(0.125, 0.130, 0.149);
const COLOR_ROW_RENAMING_BG: Color = Color::srgb(0.140, 0.146, 0.165);
const COLOR_ACTIVE_STRIPE: Color = Color::srgb(0.42, 0.62, 0.92);
const COLOR_EDIT_UNDERLINE: Color = Color::srgb(0.42, 0.62, 0.92);
const COLOR_TEXT: Color = Color::srgb(0.86, 0.87, 0.90);
const COLOR_TEXT_DIM: Color = Color::srgb(0.50, 0.52, 0.56);
const COLOR_TEXT_FAINT: Color = Color::srgb(0.36, 0.38, 0.42);

const HEADER_H: f32 = 36.0;
const ROW_H: f32 = 28.0;
const ROW_PAD_X: f32 = 14.0;
const STRIPE_W: f32 = 2.0;
const DELETE_W: f32 = 22.0;
const DIVIDER_H: f32 = 1.0;
const TEXT_FONT_SIZE: f32 = 13.0;
const HEADER_FONT_SIZE: f32 = 12.0;

const NEW_TERMINAL_OFFSET: f32 = 28.0;
const DEFAULT_TERMINAL_SIZE: Vec2 = Vec2::new(640.0, 400.0);

// ---------- Persistence ----------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectData {
    pub id: u64,
    pub name: String,
}

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
    #[serde(default)]
    terminals: Vec<TerminalSnapshot>,
    #[serde(default)]
    next_terminal_id: u64,
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
}

impl Projects {
    fn from_persisted(p: PersistedState) -> Self {
        let next_id = p.next_id.max(p.projects.iter().map(|p| p.id + 1).max().unwrap_or(1));
        let next_terminal_id = p
            .next_terminal_id
            .max(p.terminals.iter().map(|t| t.session_id + 1).max().unwrap_or(1));
        Self {
            list: p.projects,
            active: p.active,
            next_id,
            next_terminal_id,
            dirty: false,
            layout_dirty: true,
            terminals_dirty: false,
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
}

#[derive(Resource, Default)]
pub struct Renaming {
    pub id: Option<u64>,
    pub buffer: String,
}

/// Side-channel for actions the regular (non-exclusive) input systems
/// can't perform themselves — spawning terminals needs `&mut World` for
/// the `!Send` worker setup, despawning is done in the same exclusive
/// system to keep ordering simple.
#[derive(Resource, Default)]
pub struct PendingActions {
    /// Each entry: (project id to attach to, optional window-space
    /// top-left for the new terminal; `None` means cascade from the
    /// default canvas position).
    pub new_terminals: Vec<(u64, Option<Vec2>)>,
    pub close_terminals: Vec<Entity>,
    /// Terminals to restore at startup from the persisted snapshot.
    /// Each carries the rect + session id so the worker reopens the
    /// matching scrollback file and replays it on spawn.
    pub restore_terminals: Vec<TerminalSnapshot>,
    /// Editor panes to spawn. Same shape as `new_terminals`.
    pub new_editors: Vec<(u64, Option<Vec2>)>,
    /// Files to open into a new editor pane (Cmd+O picker, `tbopen`
    /// CLI, etc.).
    pub open_files: Vec<OpenFileRequest>,
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

/// File path the editor pane was loaded from. Tagged at spawn so future
/// save / "reopen at the same path" features can find it without a
/// separate side-table. No path = a scratch buffer (radial-menu spawn).
#[derive(Component, Debug, Clone)]
pub struct EditorFilePath(pub PathBuf);

/// Re-exported from `editor_bevy` so terminal-bevy and editor-bevy can
/// coordinate which handler claims a click in a given frame. Both
/// crates' mouse handlers check it on press and set it on a successful
/// hit; the editor embed plugin owns the PostUpdate reset.
pub use editor_bevy::InputConsumed;

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

#[derive(Component, Copy, Clone)]
pub struct ProjectMembership(pub u64);

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
                    mark_terminals_dirty_on_change,
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
    let mut projects = Projects::from_persisted(persisted.clone());
    if projects.list.is_empty() {
        projects.create();
        projects.dirty = true;
    }
    if projects.active.is_none() {
        projects.active = projects.list.first().map(|p| p.id);
    }

    // Queue restore for any terminal whose project still exists. The
    // exclusive `apply_pending_actions` system will spawn them on the
    // first Update tick.
    let known_projects: std::collections::HashSet<u64> =
        projects.list.iter().map(|p| p.id).collect();
    for snap in persisted.terminals {
        if known_projects.contains(&snap.project_id) {
            pending.restore_terminals.push(snap);
        }
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
    mut projects: ResMut<Projects>,
    renaming: Res<Renaming>,
    font: Res<MonoFont>,
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

    let dims_changed = last_dims.0 != Some(dims);
    let mut needs_rebuild = projects.layout_dirty || dims_changed;
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
            color: COLOR_SIDEBAR_BG,
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
            color: COLOR_DIVIDER,
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
    commands.spawn((
        SidebarEntity,
        Text2d::new("PROJECTS"),
        TextFont {
            font: font.0.clone(),
            font_size: HEADER_FONT_SIZE,
            ..default()
        },
        LineHeight::Px(HEADER_H),
        TextColor(COLOR_TEXT_FAINT),
        Anchor::TOP_LEFT,
        Transform::from_xyz(
            world_left_edge + ROW_PAD_X,
            world_top_edge - 4.0,
            SIDEBAR_Z + 0.2,
        ),
    ));
    // Header divider.
    commands.spawn((
        SidebarEntity,
        Sprite {
            color: COLOR_DIVIDER,
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
                COLOR_ROW_RENAMING_BG
            } else {
                COLOR_ROW_ACTIVE_BG
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
                    color: COLOR_ACTIVE_STRIPE,
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
                    color: COLOR_EDIT_UNDERLINE,
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

        // Label. While renaming, append a thin caret so the field
        // reads as an editable input. (U+2502 BOX DRAWINGS LIGHT
        // VERTICAL — looks like a real cursor; ASCII `|` is too thick.)
        let label = if renaming_this {
            format!("{}\u{2502}", renaming.buffer)
        } else {
            proj.name.clone()
        };
        let label_color = if active || renaming_this {
            COLOR_TEXT
        } else {
            COLOR_TEXT_DIM
        };
        commands.spawn((
            SidebarEntity,
            Text2d::new(label),
            TextFont {
                font: font.0.clone(),
                font_size: TEXT_FONT_SIZE,
                ..default()
            },
            LineHeight::Px(ROW_H),
            TextColor(label_color),
            Anchor::TOP_LEFT,
            Transform::from_xyz(
                world_left_edge + ROW_PAD_X,
                row_top_world - 5.0,
                SIDEBAR_Z + 0.2,
            ),
        ));

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
        commands.spawn((
            SidebarEntity,
            Text2d::new("\u{00D7}"), // multiplication sign — looks better than ASCII 'x'
            TextFont {
                font: font.0.clone(),
                font_size: TEXT_FONT_SIZE + 1.0,
                ..default()
            },
            LineHeight::Px(ROW_H),
            TextColor(COLOR_TEXT_FAINT),
            Anchor::TOP_LEFT,
            Transform::from_xyz(delete_x_world + 6.0, row_top_world - 5.0, SIDEBAR_Z + 0.2),
        ));
    }

    // Divider before the "+ New Project" row.
    let after_rows_window = rows_top_window + projects.list.len() as f32 * ROW_H;
    commands.spawn((
        SidebarEntity,
        Sprite {
            color: COLOR_DIVIDER,
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
    commands.spawn((
        SidebarEntity,
        Text2d::new("+  New Project"),
        TextFont {
            font: font.0.clone(),
            font_size: TEXT_FONT_SIZE,
            ..default()
        },
        LineHeight::Px(ROW_H),
        TextColor(COLOR_TEXT_DIM),
        Anchor::TOP_LEFT,
        Transform::from_xyz(
            world_left_edge + ROW_PAD_X,
            new_proj_top_world - 5.0,
            SIDEBAR_Z + 0.2,
        ),
    ));

    projects.layout_dirty = false;
}

// ---------- Sidebar input ----------

/// Tracks click state for double-click detection on project rows.
#[derive(Resource, Default)]
struct ClickTracker {
    last_project: Option<u64>,
    last_time: f64,
}

fn sidebar_input(
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

/// Exclusive — needs `&mut World` to call `spawn_terminal` (worker setup
/// touches `!Send` resources) and to despawn entities + flush their
/// worker shutdown.
fn apply_pending_actions(world: &mut World) {
    let actions = std::mem::take(&mut *world.resource_mut::<PendingActions>());

    // Restore persisted terminals first so they appear before any new
    // ones the user might have queued during the same frame.
    for snap in actions.restore_terminals {
        // Reserve the saved id in the allocator so a future create
        // never collides with it.
        {
            let mut projects = world.resource_mut::<Projects>();
            if projects.next_terminal_id <= snap.session_id {
                projects.next_terminal_id = snap.session_id + 1;
            }
        }
        let replay = scrollback_path(snap.session_id)
            .and_then(|p| std::fs::read(&p).ok());
        let rect = TerminalRect {
            pos: Vec2::new(snap.pos[0], snap.pos[1]),
            size: Vec2::new(snap.size[0], snap.size[1]),
            z: snap.z,
        };
        spawn_terminal(world, rect, Some(snap.project_id), snap.session_id, replay);
        world.resource_mut::<Projects>().layout_dirty = true;
    }

    // Close requested terminals (from the close button on each window).
    for entity in &actions.close_terminals {
        close_terminal_entity(world, *entity);
    }

    // When a project was just deleted, close all of its panes too —
    // sidebar_input only knows the project id, not which entities belong
    // to it, so we sweep both terminal and editor panes here.
    let active_ids: std::collections::HashSet<u64> = world
        .resource::<Projects>()
        .list
        .iter()
        .map(|p| p.id)
        .collect();
    let term_orphans: Vec<Entity> = {
        let mut q = world.query::<(Entity, &ProjectMembership, &TerminalTag)>();
        q.iter(world)
            .filter_map(|(e, m, _)| (!active_ids.contains(&m.0)).then_some(e))
            .collect()
    };
    for entity in term_orphans {
        close_terminal_entity(world, entity);
    }
    let editor_orphans: Vec<Entity> = {
        let mut q = world
            .query::<(Entity, &ProjectMembership, &editor_bevy::Editor)>();
        q.iter(world)
            .filter_map(|(e, m, _)| (!active_ids.contains(&m.0)).then_some(e))
            .collect()
    };
    for entity in editor_orphans {
        if world.get_entity(entity).is_ok() {
            world.entity_mut(entity).despawn();
        }
        let mut focused = world.resource_mut::<editor_bevy::FocusedEditor>();
        if focused.0 == Some(entity) {
            focused.0 = None;
        }
    }

    // Spawn requested terminals.
    let sidebar_width = world.resource::<Sidebar>().width;
    for (project_id, origin_opt) in actions.new_terminals {
        let pos = match origin_opt {
            Some(p) => p,
            None => {
                // Cascade from a starting point inside the canvas area
                // (right of the sidebar) based on how many terminals
                // already belong to the project.
                let count_in_project = {
                    let mut q = world.query::<(&ProjectMembership, &TerminalTag)>();
                    q.iter(world).filter(|(m, _)| m.0 == project_id).count()
                };
                Vec2::new(
                    sidebar_width + 40.0 + (count_in_project as f32) * NEW_TERMINAL_OFFSET,
                    40.0 + (count_in_project as f32) * NEW_TERMINAL_OFFSET,
                )
            }
        };
        let size = DEFAULT_TERMINAL_SIZE.max(MIN_TERMINAL_SIZE);
        let next_z = {
            let mut q = world.query::<(&TerminalRect, &TerminalTag)>();
            q.iter(world)
                .map(|(r, _)| r.z)
                .fold(0.0_f32, f32::max)
                + 1.0
        };
        let session_id = world.resource_mut::<Projects>().allocate_terminal_id();
        let entity = spawn_terminal(
            world,
            TerminalRect { pos, size, z: next_z },
            Some(project_id),
            session_id,
            None,
        );
        world.resource_mut::<FocusedTerminal>().0 = Some(entity);
        // Stealing focus from any editor pane so keyboard goes to the
        // freshly-spawned terminal.
        world.resource_mut::<editor_bevy::FocusedEditor>().0 = None;
        {
            let mut projects = world.resource_mut::<Projects>();
            projects.layout_dirty = true;
            projects.terminals_dirty = true;
        }
    }

    // Spawn requested editor panes. Same canvas placement model as
    // terminals — explicit origin from the radial menu, otherwise a
    // cascade from a default position based on how many panes the
    // project already has.
    for (project_id, origin_opt) in actions.new_editors {
        let pos = match origin_opt {
            Some(p) => p,
            None => {
                let count_in_project = {
                    let mut q =
                        world.query::<(&ProjectMembership, &editor_bevy::Editor)>();
                    q.iter(world).filter(|(m, _)| m.0 == project_id).count()
                };
                Vec2::new(
                    sidebar_width + 60.0 + (count_in_project as f32) * NEW_TERMINAL_OFFSET,
                    60.0 + (count_in_project as f32) * NEW_TERMINAL_OFFSET,
                )
            }
        };
        let size = Vec2::new(640.0, 420.0);
        let next_z = {
            // Reuse terminal z-axis so editors and terminals stack in a
            // single global front-to-back order.
            let mut tq = world.query::<&TerminalRect>();
            let mut eq = world.query::<&editor_bevy::EditorRect>();
            let term_max = tq.iter(world).map(|r| r.z).fold(0.0_f32, f32::max);
            let edit_max = eq.iter(world).map(|r| r.z).fold(0.0_f32, f32::max);
            term_max.max(edit_max) + 1.0
        };
        let entity = editor_bevy::spawn_editor_world(
            world,
            "",
            editor_bevy::EditorRect {
                pos,
                size,
                z: next_z,
            },
        );
        world.entity_mut(entity).insert(ProjectMembership(project_id));
        world.resource_mut::<editor_bevy::FocusedEditor>().0 = Some(entity);
        // Stealing focus from any terminal so keyboard goes to the new
        // editor.
        world.resource_mut::<FocusedTerminal>().0 = None;
        world.resource_mut::<Projects>().layout_dirty = true;
    }

    // Open files into editor panes (from Cmd+O dialog or `tbopen` CLI).
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
        // If the requested project isn't active, switching to it
        // matches user intent ("open this file" implies "show me the
        // pane that holds it").
        if world.resource::<Projects>().active != Some(project_id) {
            world.resource_mut::<Projects>().set_active(project_id);
        }
        let pos = req.origin.unwrap_or_else(|| {
            let count_in_project = {
                let mut q =
                    world.query::<(&ProjectMembership, &editor_bevy::Editor)>();
                q.iter(world).filter(|(m, _)| m.0 == project_id).count()
            };
            Vec2::new(
                sidebar_width + 60.0 + (count_in_project as f32) * NEW_TERMINAL_OFFSET,
                60.0 + (count_in_project as f32) * NEW_TERMINAL_OFFSET,
            )
        });
        let next_z = {
            let mut tq = world.query::<&TerminalRect>();
            let mut eq = world.query::<&editor_bevy::EditorRect>();
            let term_max = tq.iter(world).map(|r| r.z).fold(0.0_f32, f32::max);
            let edit_max = eq.iter(world).map(|r| r.z).fold(0.0_f32, f32::max);
            term_max.max(edit_max) + 1.0
        };
        let entity = editor_bevy::spawn_editor_world(
            world,
            &text,
            editor_bevy::EditorRect {
                pos,
                size: Vec2::new(720.0, 480.0),
                z: next_z,
            },
        );
        world.entity_mut(entity).insert((
            ProjectMembership(project_id),
            EditorFilePath(req.path),
        ));
        world.resource_mut::<editor_bevy::FocusedEditor>().0 = Some(entity);
        world.resource_mut::<FocusedTerminal>().0 = None;
        world.resource_mut::<Projects>().layout_dirty = true;
    }
}

fn resolve_project(target: &OpenProjectTarget, projects: &Projects) -> Option<u64> {
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

fn close_terminal_entity(world: &mut World, entity: Entity) {
    // Tell the worker to exit; the channel send is best-effort because
    // the worker might already be gone (child EOF + thread exit).
    if let Some(store) = world.get_resource::<TerminalStore>()
        && let Some(data) = store.map.get(&entity)
    {
        data.worker.send(WorkerMsg::Shutdown);
    }
    if let Some(mut store) = world.get_resource_mut::<TerminalStore>() {
        store.map.remove(&entity);
    }
    // Drop the terminal's persisted scrollback file so closed terminals
    // don't accumulate on disk forever.
    let session_id = world
        .get_entity(entity)
        .ok()
        .and_then(|e| e.get::<TerminalSession>())
        .map(|s| s.0);
    if let Some(id) = session_id
        && let Some(p) = scrollback_path(id)
    {
        let _ = std::fs::remove_file(&p);
    }
    if world.get_entity(entity).is_ok() {
        world.entity_mut(entity).despawn();
    }
    let mut focused = world.resource_mut::<FocusedTerminal>();
    if focused.0 == Some(entity) {
        focused.0 = None;
    }
    world.resource_mut::<Projects>().terminals_dirty = true;
}

// ---------- Visibility sync ----------

/// Hide panes whose project is not the active one. Their workers /
/// background state keep running — we only flip the parent entity's
/// `Visibility`, which cascades to the chrome + sprite pools.
/// Covers both terminal panes and editor panes.
fn sync_visibility(
    projects: Res<Projects>,
    mut terms: Query<
        (&ProjectMembership, &mut Visibility),
        (With<TerminalTag>, Without<editor_bevy::Editor>),
    >,
    mut editors: Query<
        (&ProjectMembership, &mut Visibility),
        (With<editor_bevy::Editor>, Without<TerminalTag>),
    >,
) {
    let active = projects.active;
    let apply = |m: &ProjectMembership, vis: &mut Visibility| {
        let want = if Some(m.0) == active {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != want {
            *vis = want;
        }
    };
    for (m, mut vis) in &mut terms {
        apply(m, &mut vis);
    }
    for (m, mut vis) in &mut editors {
        apply(m, &mut vis);
    }
}

// ---------- Persistence flush ----------

fn save_if_dirty(
    buttons: Res<ButtonInput<MouseButton>>,
    mut projects: ResMut<Projects>,
    sidebar: Res<Sidebar>,
    terminals: Query<(&TerminalRect, &TerminalSession, &ProjectMembership), With<TerminalTag>>,
) {
    // Defer writes while a drag is in progress: we'd otherwise flush a
    // file per frame for the duration of the drag. Project-state changes
    // (rename, create, delete) don't go through mouse drag, so they
    // still flush promptly on next idle frame.
    let mouse_down = buttons.pressed(MouseButton::Left);
    if mouse_down {
        return;
    }
    if !projects.dirty && !projects.terminals_dirty {
        return;
    }
    let snapshot = PersistedState {
        projects: projects.list.clone(),
        active: projects.active,
        next_id: projects.next_id,
        sidebar_width: Some(sidebar.width),
        terminals: terminals
            .iter()
            .map(|(rect, sess, mem)| TerminalSnapshot {
                session_id: sess.0,
                project_id: mem.0,
                pos: [rect.pos.x, rect.pos.y],
                size: [rect.size.x, rect.size.y],
                z: rect.z,
            })
            .collect(),
        next_terminal_id: projects.next_terminal_id,
    };
    save_persisted(&snapshot);
    projects.dirty = false;
    projects.terminals_dirty = false;
}

/// Mark the persisted layout dirty whenever any terminal's rect or
/// project membership changes (drag, resize, bring-to-front, project
/// reassignment). Save itself is debounced to mouse-up by `save_if_dirty`.
fn mark_terminals_dirty_on_change(
    rect_changed: Query<
        (),
        (
            With<TerminalTag>,
            Or<(Changed<TerminalRect>, Changed<ProjectMembership>)>,
        ),
    >,
    mut projects: ResMut<Projects>,
) {
    if !rect_changed.is_empty() {
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
