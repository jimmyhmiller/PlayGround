//! Minimal Bevy-native terminal emulator built on libghostty-vt.
//!
//! Multi-instance: each terminal is an Entity with chrome (bg, title bar,
//! content root, cursor, resize handle), draggable and resizable like the
//! editors in `editor-bevy`. Focus + mouse logic mirror that crate so the
//! two will later share a `text-surface-bevy` extraction (step 2 of the
//! plan). For now this is a fork, not a refactor.
//!
//! ## Threading
//!
//! `libghostty_vt::Terminal` is `!Send + !Sync`, so we can't store it as a
//! Bevy `Component`. Instead a single `NonSend<TerminalStore>` resource
//! owns a `HashMap<Entity, TerminalData>`. Entities still carry Send
//! components (rect, chrome, rev counter, row-entity pool); the
//! non-Send runtime lives in the store keyed by the same entity. Systems
//! that need both iterate entities and look up the store by id.
//!
//! ## Rendering
//!
//! Each cell is a textured sprite that samples its glyph from a shared
//! `GlyphAtlas`. Two sprite pools per terminal (`bg` for solid quads,
//! `fg` for glyph quads) sized exactly `cols * rows`; resized only on
//! grid resize. Per-frame work for an unchanged grid is one
//! `render_state.update` + cursor position; for a partial-dirty grid it's
//! a few hundred component mutations. No cosmic-text shaping in the hot
//! path.
//!
//! ## v0 scope
//!
//! - Direct key encoding (no libghostty key encoder / Kitty kb protocol):
//!   printable chars, Enter/Tab/Backspace/Escape, arrows (xterm style),
//!   Home/End/PageUp/PageDown, Delete, ctrl+letter → control codes.
//! - no wide-char handling (1 cell per char assumed; CJK/emoji get
//!   rasterized into a normal-width slot and look squished)
//! - no mouse reporting to the child, no selection / scrollback panning

use std::collections::HashMap;
use std::path::PathBuf;

use bevy::image::{Image, TextureAtlasLayout};
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::LineHeight;

use libghostty_vt::style::RgbColor;

pub mod atlas;
pub mod ipc;
pub mod projects;
pub mod pty;
pub mod radial;
pub mod selection;
pub mod vt;
pub mod worker;
use atlas::GlyphAtlas;
use projects::{
    InputConsumed, OpenFileRequest, OpenProjectTarget, PendingActions, ProjectMembership,
    Projects, Renaming, Sidebar,
};
use pty::PtySize;
use worker::{SnapCell, WorkerHandle, WorkerMsg};

pub const FONT_SIZE: f32 = 14.0;
pub const LINE_HEIGHT: f32 = 18.0;
pub const MARGIN: f32 = 8.0;
pub const TITLE_H: f32 = 22.0;
pub const HANDLE_SIZE: f32 = 14.0;
pub const CLOSE_BTN_SIZE: f32 = 14.0;
pub const CLOSE_BTN_INSET: f32 = 4.0;
pub const MIN_TERMINAL_SIZE: Vec2 = Vec2::new(240.0, 160.0);
pub const SCROLLBACK_LINES: usize = 100_000;

/// Path to SF Mono — Apple ships it with Terminal.app, so any Mac that
/// has launched Terminal.app once has this file. Loaded at startup and
/// leaked into a `'static` slice so the atlas (which holds a borrow of
/// the font bytes for `swash`) sees a stable address for the program's
/// lifetime.
const PRIMARY_FONT_PATH: &str = "/Library/Fonts/SF-Mono-Regular.otf";

fn load_primary_font() -> &'static [u8] {
    let bytes = std::fs::read(PRIMARY_FONT_PATH)
        .unwrap_or_else(|e| panic!("read {}: {}", PRIMARY_FONT_PATH, e));
    Box::leak(bytes.into_boxed_slice())
}

/// Root for all on-disk persistence (projects + per-terminal scrollback).
/// `~/.terminal-bevy/` on every supported platform.
pub fn data_dir() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = PathBuf::from(home);
    p.push(".terminal-bevy");
    Some(p)
}

/// Per-terminal scrollback log. Raw pty bytes are appended here as they
/// flow from the child; on restore the bytes are replayed into the new
/// libghostty Terminal so the visible scrollback persists across runs.
pub fn scrollback_path(session_id: u64) -> Option<PathBuf> {
    let mut p = data_dir()?;
    p.push("scrollback");
    Some(p.join(format!("{}.bytes", session_id)))
}

// ---------- Per-entity runtime ----------

/// Per-entity handle to the worker thread. Plain `Send` data — the
/// `!Send` libghostty `Terminal` lives entirely on the worker, so the
/// main Bevy thread sees only the snapshot mutex + a message channel.
pub struct TerminalData {
    pub worker: WorkerHandle,
}

#[derive(Default, Resource)]
pub struct TerminalStore {
    pub map: HashMap<Entity, TerminalData>,
}

// ---------- Components (Send) ----------

#[derive(Component)]
pub struct TerminalTag;

/// Stable id used to key per-terminal on-disk state (scrollback log,
/// layout snapshot in `projects.json`). Allocated by `Projects` and
/// preserved across restarts so a restored terminal finds its old
/// scrollback file.
#[derive(Component, Copy, Clone, Debug)]
pub struct TerminalSession(pub u64);

/// Position + size + z in window space, top-left origin, y-down. Same
/// layout as `editor-bevy::EditorRect`.
#[derive(Component, Copy, Clone, Debug)]
pub struct TerminalRect {
    pub pos: Vec2,
    pub size: Vec2,
    pub z: f32,
}

/// Child entities making up the visible chrome for one terminal.
#[derive(Component)]
pub struct TerminalChrome {
    pub bg: Entity,
    pub title_bar: Entity,
    pub content_root: Entity,
    pub cursor: Entity,
    pub resize_handle: Entity,
    pub close_button: Entity,
}

/// Per-terminal sprite pools — one solid-color quad per cell for the
/// background, one atlas-sampled quad per cell for the glyph. Both
/// pools are sized exactly `cols * rows` and stay that size between
/// frames; only resized on grid resize. Mutating `Sprite.color` and
/// `TextureAtlas.index` per cell (no spawn/despawn in the hot path) is
/// what makes throughput-heavy workloads like `cat` not melt.
///
/// `last_rendered_generation` is compared against the worker's snapshot
/// generation to skip whole frames when the grid hasn't changed.
#[derive(Component, Default)]
pub struct CellSprites {
    pub bg: Vec<Entity>,
    pub fg: Vec<Entity>,
    pub cols: u16,
    pub rows: u16,
    pub last_rendered_generation: u64,
}

/// Bumped whenever the Terminal for this entity is mutated (vt bytes
/// processed, resize). `sync_grid` rebuilds row spans when it differs
/// from the value we last rendered.
#[derive(Component, Default)]
pub struct TerminalRev(pub u64);

#[derive(Resource)]
pub struct MonoFont(pub Handle<Font>);

#[derive(Resource, Copy, Clone)]
pub struct MonoMetrics {
    pub cell_width: f32,
}

#[derive(Resource, Default)]
pub struct FocusedTerminal(pub Option<Entity>);

#[derive(Resource, Default)]
pub enum MouseMode {
    #[default]
    Idle,
    WindowDrag {
        terminal: Entity,
        grab_offset: Vec2,
    },
    WindowResize {
        terminal: Entity,
        anchor_pos: Vec2,
    },
    /// User is click-dragging inside a terminal's content area to select
    /// text. The selection's anchor + head live on `TerminalSelection`.
    TerminalSelectionDrag {
        terminal: Entity,
    },
}

/// Per-terminal text selection. `anchor` and `head` are cell coords
/// (col, row) in the terminal's grid — kept as `i32` so we can stash
/// out-of-bounds drag positions without losing direction information.
#[derive(Component, Default, Debug)]
pub struct TerminalSelection {
    pub anchor: Option<(i32, i32)>,
    pub head: Option<(i32, i32)>,
    /// Pool of overlay sprite entities visualising the selection
    /// (children of the terminal's `content_root`). Rebuilt by the
    /// selection-render system as the selection changes.
    pub overlays: Vec<Entity>,
}

impl TerminalSelection {
    pub fn clear(&mut self) {
        self.anchor = None;
        self.head = None;
    }
    pub fn is_active(&self) -> bool {
        match (self.anchor, self.head) {
            (Some(a), Some(h)) => a != h,
            _ => false,
        }
    }
    /// Return (start, end) normalised so start ≤ end in line-flow order.
    pub fn normalised(&self) -> Option<((i32, i32), (i32, i32))> {
        let (a, h) = (self.anchor?, self.head?);
        let order = (a.1, a.0) <= (h.1, h.0);
        if order {
            Some((a, h))
        } else {
            Some((h, a))
        }
    }
}

// ---------- Plugin ----------

pub struct TerminalPlugin;

impl Plugin for TerminalPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(Color::srgb(0.072, 0.075, 0.085)))
            .insert_resource(FocusedTerminal::default())
            .insert_resource(MouseMode::default())
            // Reactive: only run the schedule + render in response to
            // events. Input from winit fires this directly; pty output
            // fires it via `EventLoopProxy::send_event(WakeUp)` from
            // each terminal worker (see `WorkerHandle::spawn`). The
            // `wait` is just a safety-net fallback — under steady state
            // every redraw is event-driven and an idle terminal sits
            // at literally zero CPU between events.
            .insert_resource(bevy::winit::WinitSettings {
                focused_mode: bevy::winit::UpdateMode::reactive(
                    std::time::Duration::from_secs(5),
                ),
                unfocused_mode: bevy::winit::UpdateMode::reactive_low_power(
                    std::time::Duration::from_secs(60),
                ),
            })
            .add_plugins(projects::ProjectsPlugin)
            .add_plugins(radial::RadialPlugin)
            .add_plugins(selection::SelectionPlugin)
            // Editor panes live alongside terminal panes in the same
            // window. The embed plugin brings in the editor's input,
            // rendering, and the shared `InputConsumed` resource — but
            // leaves camera, ClearColor, WinitSettings, and font setup
            // to us (we're the host).
            .add_plugins(editor_bevy::EditorEmbedPlugin)
            .add_systems(
                Startup,
                (
                    setup_camera_and_font,
                    editor_bevy::setup_editor_font,
                    setup_ipc_listener,
                ),
            )
            .add_systems(PostStartup, release_os_focus)
            .add_systems(
                Update,
                (
                    arbitrate_pane_focus,
                    handle_open_file_shortcut,
                    drain_ipc_open_requests,
                ),
            )
            .add_systems(
                Update,
                (
                    handle_mouse,
                    handle_scroll,
                    handle_resize,
                    handle_keyboard,
                    sync_grid,
                    position_root,
                )
                    .chain(),
            );
    }
}

#[cfg(target_os = "macos")]
fn release_os_focus() {
    use objc2_app_kit::NSApplication;
    use objc2_foundation::MainThreadMarker;
    if let Some(mtm) = MainThreadMarker::new() {
        let app = NSApplication::sharedApplication(mtm);
        unsafe { app.deactivate() };
    }
}

#[cfg(not(target_os = "macos"))]
fn release_os_focus() {}

/// Holds the receiver half of the IPC channel. `mpsc::Receiver` is
/// `Send` but `!Sync`, so we install it as a `NonSend` resource and
/// drain it from a system that always runs on the main thread.
pub struct IpcInbox(pub std::sync::mpsc::Receiver<ipc::OpenRequest>);

fn setup_ipc_listener(world: &mut World) {
    if let Some(rx) = ipc::spawn_listener() {
        world.insert_non_send_resource(IpcInbox(rx));
    }
}

/// Drain any IPC open requests received this frame and queue them as
/// `PendingActions::open_files`. The actual file-read + editor spawn
/// happens in `apply_pending_actions` on the next frame, so the IPC
/// thread doesn't touch the World.
fn drain_ipc_open_requests(
    inbox: Option<NonSend<IpcInbox>>,
    mut pending: ResMut<PendingActions>,
) {
    let Some(inbox) = inbox else { return };
    while let Ok(req) = inbox.0.try_recv() {
        let project = match req.project {
            Some(name) => OpenProjectTarget::ByName(name),
            None => OpenProjectTarget::Active,
        };
        pending.open_files.push(OpenFileRequest {
            path: req.path,
            project,
            origin: None,
        });
    }
}

/// Cmd+O opens a native file picker and queues the chosen file as a
/// new editor pane in the active project. The picker is synchronous —
/// it blocks the main thread until the user picks or cancels, which
/// matches how every other macOS app handles file dialogs.
///
/// We swallow Cmd+O ourselves so the focused pane (terminal or editor)
/// never sees a stray "o" insert. Both pane keyboard handlers already
/// skip Cmd-modified keys, but we still bail explicitly here in case
/// that contract loosens.
fn handle_open_file_shortcut(
    mut events: MessageReader<bevy::input::keyboard::KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    mut pending: ResMut<PendingActions>,
) {
    let cmd = mods.pressed(KeyCode::SuperLeft) || mods.pressed(KeyCode::SuperRight);
    let mut triggered = false;
    for ev in events.read() {
        if ev.state.is_pressed() && cmd && matches!(ev.key_code, KeyCode::KeyO) {
            triggered = true;
        }
    }
    if !triggered {
        return;
    }
    let dialog = rfd::FileDialog::new()
        .set_directory(std::env::current_dir().unwrap_or_else(|_| ".".into()))
        .set_title("Open file");
    let Some(path) = dialog.pick_file() else {
        return;
    };
    pending.open_files.push(OpenFileRequest {
        path,
        project: OpenProjectTarget::Active,
        origin: None,
    });
}

/// Keep terminal-pane focus and editor-pane focus mutually exclusive:
/// at most one pane should receive keyboard input at any time. Each
/// crate's mouse handler claims clicks through `InputConsumed`, so in
/// practice only one of `FocusedTerminal`/`FocusedEditor` flips to
/// `Some` per click — this system clears the *other* when it sees the
/// change so subsequent keyboard handlers don't both receive events.
fn arbitrate_pane_focus(
    focused_editor: Res<editor_bevy::FocusedEditor>,
    mut focused_terminal: ResMut<FocusedTerminal>,
) {
    if focused_editor.is_changed() && focused_editor.0.is_some() && focused_terminal.0.is_some() {
        focused_terminal.0 = None;
    }
}

fn measure_cell_width(font_bytes: &[u8], font_size: f32) -> f32 {
    use skrifa::instance::{LocationRef, Size};
    use skrifa::{FontRef, MetadataProvider};
    let font = FontRef::from_index(font_bytes, 0).expect("embedded font must parse");
    let metrics = font.glyph_metrics(Size::new(font_size), LocationRef::default());
    let gid = font.charmap().map('M').expect("font must contain 'M'");
    metrics
        .advance_width(gid)
        .expect("'M' must have an advance width")
}

/// Exposed so callers can `.after(setup_camera_and_font)` their own
/// startup systems that spawn terminals.
///
/// Note we deliberately do *not* call `CosmicFontSystem::load_system_fonts`
/// — Text2d/cosmic-text isn't on the rendering path anymore (we draw
/// glyphs from our own atlas), and loading every font on the system
/// adds ~100ms of cold-start cost for nothing.
pub fn setup_camera_and_font(world: &mut World) {
    world.spawn(Camera2d);

    let font_bytes: &'static [u8] = load_primary_font();

    let font_handle = world
        .resource_mut::<Assets<Font>>()
        .add(Font::try_from_bytes(font_bytes.to_vec()).expect("SFMono must parse"));
    world.insert_resource(MonoFont(font_handle));

    let cell_width = measure_cell_width(font_bytes, FONT_SIZE);
    world.insert_resource(MonoMetrics { cell_width });

    // Build the glyph atlas now — pre-rasterizes printable ASCII so
    // first-frame rendering doesn't pay for it. Needs mutable access
    // to both `Assets<Image>` and `Assets<TextureAtlasLayout>` at the
    // same time; `resource_scope` lifts one out so we can grab the other.
    let atlas = world.resource_scope::<Assets<Image>, _>(|world, mut images| {
        let mut layouts = world.resource_mut::<Assets<TextureAtlasLayout>>();
        GlyphAtlas::new(
            font_bytes,
            FONT_SIZE,
            cell_width,
            LINE_HEIGHT,
            &mut images,
            &mut layouts,
        )
    });
    world.insert_resource(atlas);

    // Init the NonSend store once — spawners populate it per entity.
    world.insert_resource(TerminalStore::default());
}

// ---------- Spawn ----------

/// Create one terminal entity with its chrome + spawn a shell on its pty.
/// Returns the entity so the caller can set focus, tweak z, etc.
///
/// `project_id` tags the terminal with `ProjectMembership` so the sidebar
/// can group + show/hide it. Pass `None` only in tests / standalone demos.
///
/// `session_id` is the persistence key — the worker logs raw pty bytes
/// to `scrollback_path(session_id)` and on restart loads the same file
/// back into a fresh Terminal via `replay_bytes`.
pub fn spawn_terminal(
    world: &mut World,
    rect: TerminalRect,
    project_id: Option<u64>,
    session_id: u64,
    replay_bytes: Option<Vec<u8>>,
) -> Entity {
    let cell_width = world.resource::<MonoMetrics>().cell_width;
    let font_handle = world.resource::<MonoFont>().0.clone();

    let (cols, rows) = grid_size_for_rect(rect.size, cell_width);

    // Spawn the worker thread up front so the libghostty Terminal +
    // Pty + render iterators all live on the worker side. The main
    // thread holds only the WorkerHandle (snapshot Arc + message
    // channel), keeping the renderer fully decoupled from the parser.
    //
    // Hand the worker a clone of winit's EventLoopProxy so it can
    // wake the renderer when a new snapshot is published. Without
    // this, `WinitSettings::reactive` would never refresh on pty
    // output — the only events that wake the loop are user input.
    let wakeup = world
        .get_resource::<bevy::winit::EventLoopProxyWrapper>()
        .map(|w| bevy::winit::EventLoopProxy::clone(w));
    let worker = WorkerHandle::spawn(
        PtySize {
            cols,
            rows,
            cell_width_px: cell_width as u16,
            cell_height_px: LINE_HEIGHT as u16,
        },
        SCROLLBACK_LINES,
        scrollback_path(session_id),
        replay_bytes,
        wakeup,
    )
    .expect("WorkerHandle::spawn");
    let data = TerminalData { worker };

    // Parent entity + chrome children.
    let terminal_entity = world
        .spawn((
            TerminalTag,
            rect,
            TerminalRev::default(),
            CellSprites::default(),
            TerminalSelection::default(),
            Transform::default(),
            Visibility::default(),
        ))
        .id();

    // Panel background — single muted color, no separate title-bar
    // band. The "title bar" is just the top TITLE_H pixels of the same
    // panel; it's distinguished from content by a 1-pixel divider line.
    let bg = world
        .spawn((
            ChildOf(terminal_entity),
            Sprite {
                color: Color::srgb(0.105, 0.110, 0.122),
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
            ChildOf(terminal_entity),
            Sprite {
                color: Color::srgb(0.165, 0.170, 0.188),
                custom_size: Some(Vec2::new(rect.size.x, 1.0)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, -(TITLE_H - 1.0), 0.1),
        ))
        .id();
    let content_root = world
        .spawn((
            ChildOf(terminal_entity),
            Transform::from_xyz(MARGIN, -(TITLE_H + MARGIN), 0.2),
            Visibility::default(),
        ))
        .id();

    let cursor = world
        .spawn((
            ChildOf(content_root),
            Sprite {
                color: Color::srgba(0.55, 0.75, 0.95, 0.50),
                custom_size: Some(Vec2::new(cell_width, LINE_HEIGHT)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 1.0),
        ))
        .id();
    // Resize handle — a small muted square in the bottom-right corner.
    // Same value as the divider so it reads as a UI affordance, not a
    // bright corner sticker.
    let resize_handle = world
        .spawn((
            ChildOf(terminal_entity),
            Sprite {
                color: Color::srgb(0.22, 0.23, 0.26),
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

    // Close button — a dim × glyph in the title region. The hit area is
    // computed from CLOSE_BTN_SIZE/INSET in `region_at`, so we don't
    // need a backing sprite for click detection.
    let close_button = world
        .spawn((
            ChildOf(terminal_entity),
            Text2d::new("\u{00D7}"),
            TextFont {
                font: font_handle.clone(),
                font_size: 16.0,
                ..default()
            },
            LineHeight::Px(CLOSE_BTN_SIZE),
            TextColor(Color::srgb(0.50, 0.52, 0.56)),
            Anchor::TOP_LEFT,
            Transform::from_xyz(
                rect.size.x - CLOSE_BTN_SIZE - CLOSE_BTN_INSET,
                -CLOSE_BTN_INSET,
                0.4,
            ),
        ))
        .id();

    world.entity_mut(terminal_entity).insert(TerminalChrome {
        bg,
        title_bar,
        content_root,
        cursor,
        resize_handle,
        close_button,
    });
    world
        .entity_mut(terminal_entity)
        .insert(TerminalSession(session_id));
    if let Some(pid) = project_id {
        world
            .entity_mut(terminal_entity)
            .insert(ProjectMembership(pid));
    }

    world
        .get_resource_mut::<TerminalStore>()
        .expect("TerminalStore resource (did setup_camera_and_font run?)")
        .map
        .insert(terminal_entity, data);

    terminal_entity
}

fn grid_size_for_rect(size: Vec2, cell_width: f32) -> (u16, u16) {
    let content_w = (size.x - 2.0 * MARGIN).max(0.0);
    let content_h = (size.y - TITLE_H - 2.0 * MARGIN).max(0.0);
    let cols = ((content_w / cell_width).floor() as u16).max(1);
    let rows = ((content_h / LINE_HEIGHT).floor() as u16).max(1);
    (cols, rows)
}

// ---------- Resize ----------

/// When a terminal's rect resolves to a different grid dimension than
/// the worker's snapshot reports, send a `Resize` message. The worker
/// applies it, the next snapshot reflects the new dims, and `sync_grid`
/// resizes its sprite pools accordingly.
fn handle_resize(
    metrics: Res<MonoMetrics>,
    store: Res<TerminalStore>,
    rect_q: Query<(Entity, &TerminalRect)>,
) {
    for (entity, rect) in &rect_q {
        let Some(data) = store.map.get(&entity) else {
            continue;
        };
        let (cols, rows) = grid_size_for_rect(rect.size, metrics.cell_width);
        let (snap_cols, snap_rows) = {
            let g = data.worker.snapshot.lock().expect("snapshot lock");
            (g.cols, g.rows)
        };
        if cols == snap_cols && rows == snap_rows {
            continue;
        }
        data.worker.send(WorkerMsg::Resize {
            cols,
            rows,
            cell_w_px: metrics.cell_width as u32,
            cell_h_px: LINE_HEIGHT as u32,
        });
    }
}

// ---------- Keyboard ----------

/// Translate Bevy key events to VT bytes for the focused terminal.
///
/// Direct mapping (not libghostty's key encoder) for v0 simplicity and
/// to fix space/printable keys landing as `Key::Space` / `Key::Character`
/// rather than going through an encoder path that requires a separate
/// text stream.
fn handle_keyboard(
    mut events: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    focused: Res<FocusedTerminal>,
    store: Res<TerminalStore>,
    renaming: Res<Renaming>,
) {
    // While the user is renaming a project the sidebar's keyboard handler
    // owns input. Drain the events so they don't get re-read next frame
    // — we explicitly don't want them landing in the focused terminal.
    if renaming.id.is_some() {
        events.read().for_each(|_| {});
        return;
    }
    let Some(target) = focused.0 else {
        events.read().for_each(|_| {});
        return;
    };
    let Some(data) = store.map.get(&target) else {
        events.read().for_each(|_| {});
        return;
    };
    let child_alive = {
        let g = data.worker.snapshot.lock().expect("snapshot lock");
        g.child_alive
    };
    if !child_alive {
        events.read().for_each(|_| {});
        return;
    }

    let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);
    let ctrl = mods.pressed(KeyCode::ControlLeft) || mods.pressed(KeyCode::ControlRight);
    let alt = mods.pressed(KeyCode::AltLeft) || mods.pressed(KeyCode::AltRight);
    let cmd = mods.pressed(KeyCode::SuperLeft) || mods.pressed(KeyCode::SuperRight);

    // Cmd-modified keys are owned by app-level handlers (copy/paste,
    // future shortcuts). Skip routing them to the pty so Cmd+C doesn't
    // also send "c" to the shell.
    if cmd {
        events.read().for_each(|_| {});
        return;
    }

    // For v0 we always emit xterm-style cursor-key escapes (CSI A/B/C/D);
    // we don't have main-thread visibility into the worker's DECCKM mode.
    // Most shells/readline work fine with these — apps that need SS3
    // form (like vim in normal mode) will be addressed when we route
    // mode bits through the snapshot.
    let app_cursor = false;

    let mut out: Vec<u8> = Vec::with_capacity(16);

    for ev in events.read() {
        if !ev.state.is_pressed() {
            continue;
        }

        // Ctrl + printable letter → control byte (Ctrl+A = 0x01, etc.).
        if ctrl && !alt {
            if let KeyCode::KeyA
            | KeyCode::KeyB
            | KeyCode::KeyC
            | KeyCode::KeyD
            | KeyCode::KeyE
            | KeyCode::KeyF
            | KeyCode::KeyG
            | KeyCode::KeyH
            | KeyCode::KeyI
            | KeyCode::KeyJ
            | KeyCode::KeyK
            | KeyCode::KeyL
            | KeyCode::KeyM
            | KeyCode::KeyN
            | KeyCode::KeyO
            | KeyCode::KeyP
            | KeyCode::KeyQ
            | KeyCode::KeyR
            | KeyCode::KeyS
            | KeyCode::KeyT
            | KeyCode::KeyU
            | KeyCode::KeyV
            | KeyCode::KeyW
            | KeyCode::KeyX
            | KeyCode::KeyY
            | KeyCode::KeyZ = ev.key_code
            {
                let b = keycode_to_ctrl_byte(ev.key_code);
                out.push(b);
                continue;
            }
        }

        // Named keys we know the VT encoding for. Arrows / Home / End
        // honor DECCKM.
        if let Some(bytes) = named_key_bytes(&ev.key_code, app_cursor) {
            out.extend_from_slice(bytes);
            continue;
        }

        // Printable text via Bevy's logical_key.
        match &ev.logical_key {
            Key::Character(s) => {
                let mut bytes: Vec<u8> = s.as_str().as_bytes().to_vec();
                // Alt+letter sends ESC-prefixed byte (meta convention).
                if alt && !ctrl {
                    out.push(0x1b);
                }
                out.append(&mut bytes);
            }
            Key::Space => {
                if alt && !ctrl {
                    out.push(0x1b);
                }
                out.push(b' ');
            }
            _ => {
                let _ = shift; // informational — most shifting already baked into Character.
            }
        }
    }

    if !out.is_empty() {
        data.worker.send(WorkerMsg::Input(out));
        // Real terminals snap the viewport back to the active region
        // the moment you type — otherwise hitting Enter while scrolled
        // up leaves you staring at history while your shell scrolls
        // past below. Match that behavior.
        data.worker.send(WorkerMsg::ScrollToBottom);
    }
}

fn keycode_to_ctrl_byte(code: KeyCode) -> u8 {
    // Ctrl+A = 0x01 ... Ctrl+Z = 0x1a.
    let base = match code {
        KeyCode::KeyA => 1,
        KeyCode::KeyB => 2,
        KeyCode::KeyC => 3,
        KeyCode::KeyD => 4,
        KeyCode::KeyE => 5,
        KeyCode::KeyF => 6,
        KeyCode::KeyG => 7,
        KeyCode::KeyH => 8,
        KeyCode::KeyI => 9,
        KeyCode::KeyJ => 10,
        KeyCode::KeyK => 11,
        KeyCode::KeyL => 12,
        KeyCode::KeyM => 13,
        KeyCode::KeyN => 14,
        KeyCode::KeyO => 15,
        KeyCode::KeyP => 16,
        KeyCode::KeyQ => 17,
        KeyCode::KeyR => 18,
        KeyCode::KeyS => 19,
        KeyCode::KeyT => 20,
        KeyCode::KeyU => 21,
        KeyCode::KeyV => 22,
        KeyCode::KeyW => 23,
        KeyCode::KeyX => 24,
        KeyCode::KeyY => 25,
        KeyCode::KeyZ => 26,
        _ => 0,
    };
    base
}

fn named_key_bytes(code: &KeyCode, app_cursor: bool) -> Option<&'static [u8]> {
    Some(match code {
        KeyCode::Enter | KeyCode::NumpadEnter => b"\r",
        KeyCode::Tab => b"\t",
        KeyCode::Backspace => b"\x7f",
        KeyCode::Escape => b"\x1b",
        KeyCode::Delete => b"\x1b[3~",
        KeyCode::Insert => b"\x1b[2~",
        KeyCode::PageUp => b"\x1b[5~",
        KeyCode::PageDown => b"\x1b[6~",
        KeyCode::ArrowUp => {
            if app_cursor {
                b"\x1bOA"
            } else {
                b"\x1b[A"
            }
        }
        KeyCode::ArrowDown => {
            if app_cursor {
                b"\x1bOB"
            } else {
                b"\x1b[B"
            }
        }
        KeyCode::ArrowRight => {
            if app_cursor {
                b"\x1bOC"
            } else {
                b"\x1b[C"
            }
        }
        KeyCode::ArrowLeft => {
            if app_cursor {
                b"\x1bOD"
            } else {
                b"\x1b[D"
            }
        }
        KeyCode::Home => {
            if app_cursor {
                b"\x1bOH"
            } else {
                b"\x1b[H"
            }
        }
        KeyCode::End => {
            if app_cursor {
                b"\x1bOF"
            } else {
                b"\x1b[F"
            }
        }
        _ => return None,
    })
}

// ---------- Mouse / chrome ----------

#[derive(Copy, Clone)]
enum Region {
    CloseButton,
    TitleBar,
    ResizeHandle,
    Content,
}

fn region_at(pt: Vec2, rect: &TerminalRect) -> Option<Region> {
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
        return Some(Region::CloseButton);
    }
    let handle_x0 = rect.pos.x + rect.size.x - HANDLE_SIZE;
    let handle_y0 = rect.pos.y + rect.size.y - HANDLE_SIZE;
    if pt.x >= handle_x0 && pt.y >= handle_y0 {
        return Some(Region::ResizeHandle);
    }
    if pt.y < rect.pos.y + TITLE_H {
        return Some(Region::TitleBar);
    }
    Some(Region::Content)
}

/// Convert a window-space cursor position to a cell coord (col, row)
/// inside the terminal at `rect`. The result is intentionally not
/// clamped — the caller owns clipping to the actual grid bounds, and
/// keeping out-of-grid drags signed lets the selection extend off-edge
/// without losing direction.
pub fn pt_to_cell(pt: Vec2, rect: &TerminalRect, cell_width: f32) -> (i32, i32) {
    let local = pt - rect.pos;
    let content_x = local.x - MARGIN;
    let content_y = local.y - (TITLE_H + MARGIN);
    let col = (content_x / cell_width).floor() as i32;
    let row = (content_y / LINE_HEIGHT).floor() as i32;
    (col, row)
}

fn topmost_terminal_at(pt: Vec2, rects: &[(Entity, TerminalRect)]) -> Option<Entity> {
    let mut best: Option<(Entity, f32)> = None;
    for &(e, r) in rects {
        if pt.x >= r.pos.x
            && pt.x <= r.pos.x + r.size.x
            && pt.y >= r.pos.y
            && pt.y <= r.pos.y + r.size.y
        {
            if best.map_or(true, |(_, z)| r.z > z) {
                best = Some((e, r.z));
            }
        }
    }
    best.map(|(e, _)| e)
}

fn handle_mouse(
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut consumed: ResMut<InputConsumed>,
    metrics: Res<MonoMetrics>,
    sidebar: Res<Sidebar>,
    mut mode: ResMut<MouseMode>,
    mut focused: ResMut<FocusedTerminal>,
    mut focused_editor: ResMut<editor_bevy::FocusedEditor>,
    projects: Res<Projects>,
    mut close_requests: ResMut<projects::PendingActions>,
    mut terminals: Query<
        (
            Entity,
            &mut TerminalRect,
            Option<&ProjectMembership>,
            Option<&mut TerminalSelection>,
        ),
        With<TerminalTag>,
    >,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };

    if buttons.just_released(MouseButton::Left) {
        *mode = MouseMode::Idle;
    }

    // Sidebar owns the left edge of the window. Don't let drags
    // initiated there start a window-drag, and don't pull focus to a
    // terminal underneath. Sidebar's own click handler runs separately.
    let in_sidebar = pt.x < sidebar.width;

    // Radial menu / other input system already swallowed this click.
    let click_eaten = consumed.0;

    if buttons.just_pressed(MouseButton::Left) && !in_sidebar && !click_eaten {
        let rects: Vec<(Entity, TerminalRect)> = terminals
            .iter()
            .filter(|(_, _, m, _)| match (projects.active, m) {
                (Some(a), Some(ProjectMembership(p))) => a == *p,
                _ => false,
            })
            .map(|(e, r, _, _)| (e, *r))
            .collect();
        let Some(target) = topmost_terminal_at(pt, &rects) else {
            // Click missed every terminal — clear any existing selection.
            for (_, _, _, sel) in &mut terminals {
                if let Some(mut sel) = sel {
                    sel.clear();
                }
            }
            return;
        };

        let rect = *terminals.get(target).unwrap().1;
        if matches!(region_at(pt, &rect), Some(Region::CloseButton)) {
            // Defer the actual teardown so we don't despawn while iterating.
            close_requests.close_terminals.push(target);
            return;
        }

        focused.0 = Some(target);
        // Focusing a terminal yields editor focus so keyboard input
        // doesn't get delivered to both panes at once.
        if focused_editor.0.is_some() {
            focused_editor.0 = None;
        }
        // Tell other handlers (radial, editor) we own this click.
        consumed.0 = true;
        bring_to_front(target, &mut terminals);

        // Any new click clears every other terminal's selection so only
        // one terminal at a time has a visible highlight.
        for (e, _, _, sel) in &mut terminals {
            if e != target
                && let Some(mut s) = sel
            {
                s.clear();
            }
        }

        match region_at(pt, &rect) {
            Some(Region::TitleBar) => {
                *mode = MouseMode::WindowDrag {
                    terminal: target,
                    grab_offset: pt - rect.pos,
                };
            }
            Some(Region::ResizeHandle) => {
                *mode = MouseMode::WindowResize {
                    terminal: target,
                    anchor_pos: rect.pos,
                };
            }
            Some(Region::Content) => {
                let cell = pt_to_cell(pt, &rect, metrics.cell_width);
                if let Ok((_, _, _, Some(mut sel))) = terminals.get_mut(target) {
                    sel.anchor = Some(cell);
                    sel.head = Some(cell);
                }
                *mode = MouseMode::TerminalSelectionDrag { terminal: target };
            }
            Some(Region::CloseButton) | None => {}
        }
        return;
    }

    if !buttons.pressed(MouseButton::Left) {
        return;
    }

    match *mode {
        MouseMode::WindowDrag {
            terminal,
            grab_offset,
        } => {
            if let Ok((_, mut rect, _, _)) = terminals.get_mut(terminal) {
                let mut new_pos = pt - grab_offset;
                // Don't let the title bar slide under the sidebar — once
                // it does, you can't grab the terminal back without
                // resizing the window.
                if new_pos.x < sidebar.width {
                    new_pos.x = sidebar.width;
                }
                rect.pos = new_pos;
            }
        }
        MouseMode::WindowResize {
            terminal,
            anchor_pos,
        } => {
            if let Ok((_, mut rect, _, _)) = terminals.get_mut(terminal) {
                let raw = pt - anchor_pos;
                rect.size = Vec2::new(
                    raw.x.max(MIN_TERMINAL_SIZE.x),
                    raw.y.max(MIN_TERMINAL_SIZE.y),
                );
            }
        }
        MouseMode::TerminalSelectionDrag { terminal } => {
            if let Ok((_, rect, _, Some(mut sel))) = terminals.get_mut(terminal) {
                let cell = pt_to_cell(pt, &rect, metrics.cell_width);
                sel.head = Some(cell);
            }
        }
        MouseMode::Idle => {}
    }
}

/// Mouse-wheel scrolls the terminal under the cursor (in the active
/// project). Pixel-mode events (trackpads) accumulate a fractional line
/// counter so small swipes still register; line-mode events go through
/// at face value.
fn handle_scroll(
    mut wheel: MessageReader<MouseWheel>,
    mut accum: Local<f32>,
    windows: Query<&Window>,
    sidebar: Res<Sidebar>,
    projects: Res<Projects>,
    store: Res<TerminalStore>,
    terminals: Query<
        (Entity, &TerminalRect, Option<&ProjectMembership>),
        With<TerminalTag>,
    >,
) {
    // Sum the frame's wheel events into our accumulator regardless of
    // whether we end up dispatching, so a slow scroll across multiple
    // ticks doesn't lose precision.
    let mut delta_lines: f32 = 0.0;
    for ev in wheel.read() {
        let lines = match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y / LINE_HEIGHT,
        };
        delta_lines += lines;
    }
    if delta_lines == 0.0 {
        return;
    }
    *accum += delta_lines;
    let whole_lines = accum.trunc() as isize;
    if whole_lines == 0 {
        return;
    }
    *accum -= whole_lines as f32;

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(pt) = window.cursor_position() else {
        return;
    };
    if pt.x < sidebar.width {
        return;
    }

    // Pick the topmost terminal under the cursor (active project only).
    let rects: Vec<(Entity, TerminalRect)> = terminals
        .iter()
        .filter(|(_, _, m)| match (projects.active, m) {
            (Some(a), Some(ProjectMembership(p))) => a == *p,
            _ => false,
        })
        .map(|(e, r, _)| (e, *r))
        .collect();
    let Some(target) = topmost_terminal_at(pt, &rects) else {
        return;
    };
    let Some(data) = store.map.get(&target) else {
        return;
    };

    // Bevy: wheel.y > 0 = scroll-up gesture = reveal older content.
    // libghostty: ScrollViewport::Delta is positive toward the active
    // area, negative back into history. So mirror the sign.
    let scroll_delta = -whole_lines;
    data.worker.send(WorkerMsg::ScrollDelta(scroll_delta));
}

fn bring_to_front(
    target: Entity,
    terminals: &mut Query<
        (
            Entity,
            &mut TerminalRect,
            Option<&ProjectMembership>,
            Option<&mut TerminalSelection>,
        ),
        With<TerminalTag>,
    >,
) {
    let max_z = terminals
        .iter()
        .map(|(_, r, _, _)| r.z)
        .fold(0.0_f32, f32::max);
    if let Ok((_, mut rect, _, _)) = terminals.get_mut(target) {
        if rect.z < max_z {
            rect.z = max_z + 1.0;
        }
    }
}

fn position_root(
    windows: Query<&Window>,
    terminals: Query<(&TerminalRect, &TerminalChrome), With<TerminalTag>>,
    parents: Query<Entity, With<TerminalTag>>,
    mut t_q: Query<&mut Transform>,
    mut sprite_q: Query<&mut Sprite>,
) {
    let Ok(win) = windows.single() else {
        return;
    };
    let win_size = Vec2::new(win.width(), win.height());

    for entity in &parents {
        let Ok((rect, chrome)) = terminals.get(entity) else {
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

// ---------- Rendering ----------

/// Render the visible grid into per-cell sprites that sample glyphs
/// from a shared atlas. The atlas pre-rasterized printable ASCII at
/// startup; non-ASCII chars get rasterized lazily on first sight.
///
/// Pool sizes (`bg`, `fg`) are exactly `cols * rows` and only change
/// on grid resize — every other frame just mutates `Sprite.color` and
/// `TextureAtlas.index` on the dirty rows. No cosmic-text, no Text2d,
/// no spawn/despawn churn.
fn sync_grid(
    metrics: Res<MonoMetrics>,
    mut atlas: ResMut<GlyphAtlas>,
    mut images: ResMut<Assets<Image>>,
    mut layouts: ResMut<Assets<TextureAtlasLayout>>,
    store: Res<TerminalStore>,
    mut terminals: Query<(Entity, &TerminalChrome, &mut CellSprites)>,
    mut sprite_q: Query<&mut Sprite>,
    mut transform_q: Query<&mut Transform>,
    mut vis_q: Query<&mut Visibility>,
    mut commands: Commands,
    mut prof: Local<SyncGridProfile>,
) {
    use std::time::Instant;
    let frame_start = Instant::now();

    // Local reusable scratch — copied OUT of the snapshot mutex so we
    // never hold the lock while mutating thousands of sprites.
    let mut local_cells: Vec<SnapCell> = Vec::new();
    let mut local_dirty_rows: Vec<bool> = Vec::new();

    let mut work_done = false;
    let mut lock_ns: u128 = 0;
    let mut mutate_ns: u128 = 0;
    let mut cells_touched = 0u64;

    for (entity, chrome, mut pools) in &mut terminals {
        let Some(data) = store.map.get(&entity) else {
            continue;
        };

        // Lock briefly, copy snapshot fields into locals, drop lock.
        let lock_t = Instant::now();
        let (cols, rows, default_fg, default_bg, cursor, generation) = {
            let g = data.worker.snapshot.lock().expect("snapshot lock");
            local_cells.clear();
            local_cells.extend_from_slice(&g.cells);
            local_dirty_rows.clear();
            local_dirty_rows.extend_from_slice(&g.dirty_rows);
            (
                g.cols,
                g.rows,
                g.default_fg,
                g.default_bg,
                g.cursor,
                g.generation,
            )
        };
        lock_ns += lock_t.elapsed().as_nanos();

        // Cursor (always update — cursor moves don't always set row-dirty).
        if let Ok(mut v) = vis_q.get_mut(chrome.cursor) {
            *v = if cursor.is_some() {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
        }
        if let Some((cx, cy)) = cursor {
            if let Ok(mut t) = transform_q.get_mut(chrome.cursor) {
                t.translation.x = cx as f32 * metrics.cell_width;
                t.translation.y = -(cy as f32) * LINE_HEIGHT;
                t.translation.z = 1.0;
            }
        }

        let pool_changed = pools.cols != cols || pools.rows != rows;
        let nothing_changed = !pool_changed && pools.last_rendered_generation == generation;
        if nothing_changed {
            continue;
        }
        work_done = true;
        let mutate_t = Instant::now();

        // Resize sprite pools to match the worker's current grid.
        if pool_changed {
            let needed = cols as usize * rows as usize;
            for &e in pools.bg.iter().skip(needed) {
                commands.entity(e).despawn();
            }
            for &e in pools.fg.iter().skip(needed) {
                commands.entity(e).despawn();
            }
            pools.bg.truncate(needed);
            pools.fg.truncate(needed);

            // Spawn new sprites with their actual content from local_cells.
            // commands.spawn is queued — these entities don't exist in the
            // World during this system run, so a follow-up sprite_q.get_mut
            // silently fails. By next frame the snapshot generation hasn't
            // bumped (we already consumed it), so the dirty-row loop skips
            // them and the sprites stay at spawn-time defaults forever.
            // Computing content at spawn time avoids the deferred mutation.
            let cell_w = metrics.cell_width;
            for i in pools.bg.len()..needed {
                let row = (i / cols as usize) as f32;
                let col = (i % cols as usize) as f32;
                let cell = local_cells.get(i).copied().unwrap_or_default();
                let (_fg, init_bg) = if cell.inverse {
                    (cell.bg, cell.fg)
                } else {
                    (cell.fg, cell.bg)
                };
                let bg = commands
                    .spawn((
                        ChildOf(chrome.content_root),
                        Sprite {
                            color: rgb_to_color(init_bg),
                            custom_size: Some(Vec2::new(cell_w, LINE_HEIGHT)),
                            ..default()
                        },
                        Anchor::TOP_LEFT,
                        Transform::from_xyz(col * cell_w, -row * LINE_HEIGHT, 0.0),
                    ))
                    .id();
                pools.bg.push(bg);
            }
            for i in pools.fg.len()..needed {
                let row = (i / cols as usize) as f32;
                let col = (i % cols as usize) as f32;
                let cell = local_cells.get(i).copied().unwrap_or_default();
                let (init_fg, _bg) = if cell.inverse {
                    (cell.bg, cell.fg)
                } else {
                    (cell.fg, cell.bg)
                };
                let glyph_index =
                    atlas.lookup_or_insert(cell.ch, &mut images, &mut layouts);
                let fg = commands
                    .spawn((
                        ChildOf(chrome.content_root),
                        Sprite {
                            image: atlas.image.clone(),
                            texture_atlas: Some(TextureAtlas {
                                layout: atlas.layout.clone(),
                                index: glyph_index as usize,
                            }),
                            color: rgb_to_color(init_fg),
                            custom_size: Some(Vec2::new(cell_w, LINE_HEIGHT)),
                            ..default()
                        },
                        Anchor::TOP_LEFT,
                        Transform::from_xyz(col * cell_w, -row * LINE_HEIGHT, 0.5),
                    ))
                    .id();
                pools.fg.push(fg);
            }

            // Reposition the existing cells too in case cell_w/line_h changed.
            for (i, &e) in pools.bg.iter().enumerate() {
                if let Ok(mut t) = transform_q.get_mut(e) {
                    let row = (i / cols as usize) as f32;
                    let col = (i % cols as usize) as f32;
                    t.translation.x = col * cell_w;
                    t.translation.y = -row * LINE_HEIGHT;
                }
            }
            for (i, &e) in pools.fg.iter().enumerate() {
                if let Ok(mut t) = transform_q.get_mut(e) {
                    let row = (i / cols as usize) as f32;
                    let col = (i % cols as usize) as f32;
                    t.translation.x = col * cell_w;
                    t.translation.y = -row * LINE_HEIGHT;
                }
            }

            pools.cols = cols;
            pools.rows = rows;
        }

        // Walk dirty rows and mutate the cells. Pool resize forced a
        // full repaint above; otherwise only rows the worker flagged
        // as dirty get touched.
        let force_all = pool_changed;
        for r in 0..rows as usize {
            let row_dirty = force_all
                || local_dirty_rows.get(r).copied().unwrap_or(true);
            if !row_dirty {
                continue;
            }
            let row_base = r * cols as usize;
            for c in 0..cols as usize {
                let idx = row_base + c;
                let cell = match local_cells.get(idx) {
                    Some(c) => *c,
                    None => continue,
                };
                let (final_fg, final_bg) = if cell.inverse {
                    (cell.bg, cell.fg)
                } else {
                    (cell.fg, cell.bg)
                };

                if let Some(&bg_entity) = pools.bg.get(idx) {
                    if let Ok(mut s) = sprite_q.get_mut(bg_entity) {
                        s.color = rgb_to_color(final_bg);
                    }
                }

                let glyph_index =
                    atlas.lookup_or_insert(cell.ch, &mut images, &mut layouts);
                if let Some(&fg_entity) = pools.fg.get(idx) {
                    if let Ok(mut s) = sprite_q.get_mut(fg_entity) {
                        s.color = rgb_to_color(final_fg);
                        if let Some(ref mut ta) = s.texture_atlas {
                            ta.index = glyph_index as usize;
                        }
                    }
                }
                cells_touched += 1;
            }
        }

        pools.last_rendered_generation = generation;
        mutate_ns += mutate_t.elapsed().as_nanos();
    }

    if work_done && std::env::var("TERMINAL_PROFILE").is_ok() {
        prof.frames += 1;
        prof.frame_ns += frame_start.elapsed().as_nanos();
        prof.lock_ns += lock_ns;
        prof.mutate_ns += mutate_ns;
        prof.cells += cells_touched;
        if prof.frames >= 30 {
            eprintln!(
                "[render] {} frames | avg frame {:>5.2} ms | lock {:>4.2} ms | mutate {:>4.2} ms | {:>5.0} cells/frame",
                prof.frames,
                (prof.frame_ns as f64 / 1_000_000.0) / prof.frames as f64,
                (prof.lock_ns as f64 / 1_000_000.0) / prof.frames as f64,
                (prof.mutate_ns as f64 / 1_000_000.0) / prof.frames as f64,
                prof.cells as f64 / prof.frames as f64,
            );
            *prof = SyncGridProfile::default();
        }
    }
}

#[derive(Default)]
struct SyncGridProfile {
    frames: u64,
    frame_ns: u128,
    lock_ns: u128,
    mutate_ns: u128,
    cells: u64,
}

// ---------- helpers ----------

fn rgb_to_color(c: RgbColor) -> Color {
    Color::srgb(c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0)
}
