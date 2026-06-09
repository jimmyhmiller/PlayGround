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

use libghostty_vt::style::RgbColor;
use pane_bevy::{
    spawn_pane, FocusedPane, PaneChrome, PaneContentPressed, PaneFont, PaneKindMarker,
    PanePlugin, PaneRect, PaneRegistry, PaneTag, SpawnedPane, MARGIN, TITLE_H,
};
use serde_json::Value;

pub mod actions;
pub mod atlas;
pub mod canvas;
pub mod command_palette;
pub mod claude_events_pane;
pub mod command_watch;
pub mod context_menu;
pub mod cube;
pub mod daemon_client;
pub mod debug_bar;
pub mod diagnostics;
pub mod drawer;
pub mod fps;
pub mod graph_view;
pub mod inference_dispatch;
pub mod inbox;
pub mod inferences_pane;
pub mod issues_pane;
pub mod osc7;
/// Re-export of the daemon protocol from the headless crate so existing
/// callers can continue to write `terminal_bevy::daemon_proto::*`.
pub use terminal_daemon::proto as daemon_proto;
pub mod ipc;
pub mod projects;
pub mod pty;
pub mod radial;
pub mod run_button;
pub mod selection;
pub mod term_material;
pub mod tools;
pub mod vt;
pub mod window_geometry;
pub mod worker;
pub mod workflow_graph;
use atlas::GlyphAtlas;
use term_material::{
    make_cells_image, pack_rgb, GpuCell, TermMaterial, TermMaterialPlugin, TermParams,
};
use projects::{
    NewPaneRequest, OpenFileRequest, OpenProjectTarget, PendingActions, ProjectMembership,
    Projects, Sidebar,
};
use pty::PtySize;
use worker::{SnapCell, WorkerHandle, WorkerMsg};

pub const FONT_SIZE: f32 = 14.0;
pub const LINE_HEIGHT: f32 = 18.0;
pub const SCROLLBACK_LINES: usize = 100_000;

/// Stable identifier for terminal panes. Stored on every terminal pane
/// in `PaneKindMarker` and referenced by the registry.
pub const PANE_KIND: &str = "terminal";

/// Candidate monospace fonts tried in order, first readable one wins.
/// SF Mono is preferred — Apple ships it with Terminal.app, so any Mac
/// that has launched Terminal.app once has it — but we fall back to other
/// common monospace faces so the terminal still starts on a machine that
/// lacks it. Override the whole search with the `TERMINAL_BEVY_FONT` env
/// var (absolute path to a `.otf`/`.ttf`).
const PRIMARY_FONT_CANDIDATES: &[&str] = &[
    "/Library/Fonts/SF-Mono-Regular.otf",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/Library/Fonts/Andale Mono.ttf",
];

/// Loads the primary monospace font and leaks it into a `'static` slice
/// so the atlas (which holds a borrow of the font bytes for `swash`) sees
/// a stable address for the program's lifetime. Tries `TERMINAL_BEVY_FONT`
/// first, then [`PRIMARY_FONT_CANDIDATES`]; panics with the full list if
/// none can be read (the terminal cannot render without a font).
fn load_primary_font() -> &'static [u8] {
    let mut tried: Vec<String> = Vec::new();
    let override_path = std::env::var_os("TERMINAL_BEVY_FONT");
    let candidates = override_path
        .as_ref()
        .map(|p| std::path::Path::new(p))
        .into_iter()
        .map(std::borrow::Cow::Borrowed)
        .chain(
            PRIMARY_FONT_CANDIDATES
                .iter()
                .map(|p| std::borrow::Cow::Owned(std::path::PathBuf::from(p))),
        );
    for path in candidates {
        match std::fs::read(path.as_ref()) {
            Ok(bytes) => return Box::leak(bytes.into_boxed_slice()),
            Err(e) => tried.push(format!("  {}: {}", path.as_ref().display(), e)),
        }
    }
    panic!(
        "no usable monospace font found — set TERMINAL_BEVY_FONT to an \
         absolute .otf/.ttf path. Tried:\n{}",
        tried.join("\n")
    );
}

/// Root for all on-disk persistence (projects + per-terminal scrollback).
/// `~/.terminal-bevy/` on every supported platform.
///
/// Delegates to `terminal_daemon::data_dir` so the daemon process and
/// the editor process agree on the location of socket / pid files.
pub fn data_dir() -> Option<PathBuf> {
    terminal_daemon::data_dir()
}

/// Per-terminal scrollback log. Raw pty bytes are appended here as they
/// flow from the child; on restore the bytes are replayed into the new
/// libghostty Terminal so the visible scrollback persists across runs.
pub fn scrollback_path(session_id: u64) -> Option<PathBuf> {
    let mut p = data_dir()?;
    p.push("scrollback");
    Some(p.join(format!("{}.bytes", session_id)))
}

/// Unix socket the per-session daemon listens on. Forwards to the
/// daemon crate so client and daemon share a single source of truth.
pub fn socket_path(session_id: u64) -> Option<PathBuf> {
    terminal_daemon::socket_path(session_id)
}

/// PID file the daemon writes on startup. Same delegation as
/// [`socket_path`].
pub fn pid_path(session_id: u64) -> Option<PathBuf> {
    terminal_daemon::pid_path(session_id)
}

/// Pick the shell to launch in a new daemon. Resolution: `$SHELL` →
/// passwd entry → `/bin/sh`. Returns a `Vec<String>` so it slots into
/// `Command::new(args[0]).args(&args[1..])` cleanly. Matches what
/// `Pty::spawn` used to do before we moved PTY ownership into the daemon.
///
/// `-l` makes it a login shell so macOS's `/etc/zprofile` runs
/// `/usr/libexec/path_helper` — without it `PATH` is missing
/// `/opt/homebrew/bin` etc., breaking tools like autojump.
pub fn default_shell_command() -> Vec<String> {
    use std::path::PathBuf as PB;
    let shell = match std::env::var_os("SHELL") {
        Some(s) if !s.is_empty() => PB::from(s),
        _ => match nix::unistd::User::from_uid(nix::unistd::getuid()) {
            Ok(Some(user)) => user.shell,
            _ => PB::from("/bin/sh"),
        },
    };
    vec![shell.to_string_lossy().into_owned(), "-l".to_string()]
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

/// Stable id used to key per-terminal on-disk state (scrollback log,
/// layout snapshot in `projects.json`). Allocated by `Projects` and
/// preserved across restarts so a restored terminal finds its old
/// scrollback file.
#[derive(Component, Copy, Clone, Debug)]
pub struct TerminalSession(pub u64);

/// Cursor sprite child of a terminal pane's content_root. Held on the
/// pane entity so `sync_grid` can position/show-hide the cursor without
/// looking it up by traversal.
#[derive(Component, Copy, Clone)]
pub struct TerminalCursor(pub Entity);

/// Per-terminal GPU grid state. One `TermMaterial` + cells texture per
/// pane; the worker → sync_grid pipeline rewrites texels of the cells
/// texture and Bevy re-uploads it. Replaces the previous "one Sprite
/// entity per cell" model — for a busy TUI we used to walk thousands
/// of `Mut<Sprite>` per frame, marking each one Changed; now it's a
/// single mutated Image asset per frame regardless of how many cells
/// changed.
///
/// `last_rendered_generation` is compared against the worker's snapshot
/// generation to skip whole frames when the grid hasn't changed.
#[derive(Component)]
pub struct TermGrid {
    pub material: Handle<term_material::TermMaterial>,
    pub cells_image: Handle<Image>,
    pub mesh: Handle<Mesh>,
    /// Entity (child of `content_root`) carrying the `Mesh2d` +
    /// `MeshMaterial2d<TermMaterial>`.
    pub render_entity: Entity,
    pub cols: u16,
    pub rows: u16,
    pub last_rendered_generation: u64,
    /// Was this pane visible the last time sync_grid touched it? Used
    /// to detect the hidden→visible transition so we can force a full
    /// repaint of the cells texture (the worker has been processing pty
    /// bytes the whole time but not pushing snapshots into the GPU).
    pub was_visible: bool,
}

/// Bumped whenever the Terminal for this entity is mutated (vt bytes
/// processed, resize). `sync_grid` rebuilds row spans when it differs
/// from the value we last rendered.
#[derive(Component, Default)]
pub struct TerminalRev(pub u64);

/// Per-terminal bell-tracking state. `last_seen` mirrors the worker's
/// `bell_count` so we only react to *new* bells — incrementing the
/// project counter once each, never every frame the counter is non-zero.
#[derive(Component, Default)]
pub struct BellPulse {
    pub last_seen: u64,
}

#[derive(Resource)]
pub struct MonoFont(pub Handle<Font>);

/// Dedicated RenderLayer for menu overlays (radial menu, per-pane
/// context menu) so they draw on top of every per-pane camera. Pane
/// cameras have order `(rect.z * 100) + 1`, which can climb past 600
/// as panes are focused — anything drawn on layer 0 ends up *under*
/// those pane cameras inside their viewports, which made the radial
/// vanish behind panes. The overlay camera (see [`setup_camera_and_font`])
/// runs at order [`MENU_OVERLAY_CAMERA_ORDER`] (well above any pane)
/// and renders only this layer, so menu items are guaranteed on top.
pub const MENU_OVERLAY_LAYER: usize = 32;
/// Camera order for the menu-overlay camera. Sized so it stays above
/// any plausible pane-camera order: pane cameras max out around
/// `(MAX_PANE_Z * 100) + 1` ≈ 50_001, so 100_000 leaves headroom.
pub const MENU_OVERLAY_CAMERA_ORDER: isize = 100_000;

#[derive(Resource, Copy, Clone)]
pub struct MonoMetrics {
    pub cell_width: f32,
}

/// Whether our OS window currently has keyboard focus. Mirrors the
/// `WindowFocused` events winit dispatches; we maintain it ourselves
/// rather than polling `Window::focused` because (at least on
/// macOS / Bevy 0.18) the field doesn't always reflect app-level
/// activation changes when the user Cmd+Tabs to another app.
///
/// Defaults to true — first frame the user is presumably looking at
/// us; a `WindowFocused(false)` will arrive if not.
#[derive(Resource)]
pub struct AppFocused(pub bool);

impl Default for AppFocused {
    fn default() -> Self {
        Self(true)
    }
}

/// Per-terminal text selection.
///
/// `anchor` and `head` are `(col, absolute_row)`:
/// - `col` is a grid column, `i32` so out-of-bounds drag positions
///   don't lose direction.
/// - `absolute_row` is `i64`, indexing into libghostty's *total*
///   scrollable area (scrollback + active) — i.e.,
///   `snapshot.viewport_offset + viewport_row` at the moment of the
///   click. Anchoring against the absolute row makes a selection
///   follow its content while the user scrolls the viewport.
///
/// Limitation: when libghostty's bounded scrollback wraps (oldest line
/// pushed out), all absolute rows shift down by one. We don't
/// compensate; selections older than the wrap point will drift. In
/// practice selections are short-lived enough that this is fine.
#[derive(Component, Default, Debug)]
pub struct TerminalSelection {
    pub anchor: Option<(i32, i64)>,
    pub head: Option<(i32, i64)>,
    /// True while the user is mid-drag selecting. Cleared on mouse-up.
    /// Per-frame drag-update checks this instead of consulting a global
    /// mouse-mode enum.
    pub dragging: bool,
    /// Pool of overlay sprite entities visualising the selection
    /// (children of the terminal's `content_root`). Rebuilt by the
    /// selection-render system as the selection changes.
    pub overlays: Vec<Entity>,
}

impl TerminalSelection {
    pub fn clear(&mut self) {
        self.anchor = None;
        self.head = None;
        self.dragging = false;
    }
    pub fn is_active(&self) -> bool {
        match (self.anchor, self.head) {
            (Some(a), Some(h)) => a != h,
            _ => false,
        }
    }
    /// Return (start, end) normalised so start ≤ end in line-flow order.
    pub fn normalised(&self) -> Option<((i32, i64), (i32, i64))> {
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
            .insert_resource(AppFocused::default())
            .insert_resource(bevy::winit::WinitSettings {
                focused_mode: bevy::winit::UpdateMode::reactive(
                    std::time::Duration::from_secs(5),
                ),
                unfocused_mode: bevy::winit::UpdateMode::reactive_low_power(
                    std::time::Duration::from_secs(60),
                ),
            })
            // Pane-bevy owns chrome (drag, resize, close, focus,
            // hit-test). Terminal-specific systems below register the
            // "terminal" kind, render the grid, and handle keyboard +
            // mouse-driven selection inside the content area.
            // Reserve EVERY layer that a non-pane, non-project-scoped
            // camera renders, so no pane is ever allocated one. A collision
            // draws that pane's content across every project (and over the
            // cube), because that global camera isn't gated by project:
            //   - MENU_OVERLAY_LAYER (32): menus / FPS / status bar.
            //   - cube::CUBE_LAYER (4096): the prism's structural geometry.
            //   - style_bevy::dynamic::OVERLAY_LAYER (30): the dust/shader
            //     canvas overlay, drawn at order 1_000_001 above everything.
            // This is the single registry of global layers; anyone adding a
            // global overlay camera MUST add its layer here. See
            // `PaneLayerAllocator`.
            .add_plugins(PanePlugin {
                reserved_layers: vec![
                    MENU_OVERLAY_LAYER,
                    cube::CUBE_LAYER,
                    style_bevy::dynamic::OVERLAY_LAYER,
                ],
            })
            .add_plugins(TermMaterialPlugin)
            .add_plugins(diagnostics::DiagnosticsPlugin)
            .add_plugins(projects::ProjectsPlugin)
            .add_plugins(actions::ActionsPlugin)
            .add_plugins(canvas::CanvasPlugin)
            .add_plugins(context_menu::ContextMenuPlugin)
            .add_plugins(cube::CubePlugin)
            .add_plugins(radial::RadialPlugin)
            .add_plugins(command_palette::CommandPalettePlugin)
            .add_plugins(drawer::DrawerPlugin)
            .add_plugins(selection::SelectionPlugin)
            .add_plugins(run_button::RunButtonPlugin)
            .add_plugins(workflow_graph::WorkflowGraphPlugin)
            .add_plugins(fps::FpsOverlayPlugin)
            .add_plugins(debug_bar::DebugBarPlugin)
            .add_plugins(claude_events_pane::ClaudeEventsPanePlugin)
            .add_plugins(inferences_pane::InferencesPanePlugin)
            .add_plugins(issues_pane::IssuesPanePlugin)
            .add_plugins(inbox::InboxPanePlugin)
            .add_plugins(inference_dispatch::InferenceDispatchPlugin)
            .add_plugins(widget_bevy::WidgetPlugin)
            .add_plugins(widget_bevy::rhai_widget::RhaiWidgetPlugin)
            .add_plugins(editor_bevy::EditorEmbedPlugin)
            .add_plugins(style_bevy::StylePlugin)
            .add_plugins(style_bevy::state::ProjectStatePlugin);
        // Bespoke (non-pane) actions. Pane-spawn actions are auto-
        // generated from the `PaneRegistry` at PostStartup; these are the
        // capabilities that aren't "spawn a pane kind". Each was formerly
        // a hand-rolled keyboard-shortcut system.
        use actions::{Action, ActionRun, AppActionsExt, KeyChord};
        app.add_action(Action {
            id: "file.open",
            title: "Open File…",
            category: "File",
            keywords: &["edit", "buffer"],
            radial_icon: None,
            default_keys: const { &[KeyChord::cmd(KeyCode::KeyO)] },
            run: ActionRun::Custom(action_open_file),
        })
        .add_action(Action {
            id: "view.dev_panel",
            title: "Style Dev Panel",
            category: "View",
            keywords: &["debug", "tokens"],
            radial_icon: None,
            default_keys: const { &[KeyChord::cmd_shift(KeyCode::KeyD)] },
            run: ActionRun::Custom(action_open_dev_panel),
        })
        .add_action(Action {
            id: "view.theme_editor",
            title: "Theme Editor",
            category: "View",
            keywords: &["color", "oklch", "palette"],
            radial_icon: None,
            default_keys: const { &[KeyChord::cmd_shift(KeyCode::KeyT)] },
            run: ActionRun::Custom(action_open_theme_editor),
        })
        .add_action(Action {
            id: "view.style_picker",
            title: "Styles",
            category: "View",
            keywords: &["preset", "theme", "skin"],
            radial_icon: None,
            default_keys: const { &[KeyChord::cmd_shift(KeyCode::KeyS)] },
            run: ActionRun::Custom(action_open_style_picker),
        })
        .add_action(Action {
            id: "view.toggle_cube",
            title: "Toggle Project Cube",
            category: "View",
            keywords: &["prism", "3d", "overview", "switch"],
            // Eligible for the radial ring — proof the ring now hosts
            // non-pane actions. `cube.rs` keeps its own ⌘⇧\ single-chord
            // toggle; here we add a *sequence* binding (⌘K then C) — a
            // different chord, so no double-toggle — to exercise the
            // chord-sequence matcher in-tree.
            radial_icon: Some("◧"),
            default_keys: const { &[KeyChord::cmd(KeyCode::KeyK), KeyChord::plain(KeyCode::KeyC)] },
            run: ActionRun::Custom(action_toggle_cube),
        })
        // ----- Pane / view control (formerly nothing — new global chords) -----
        .add_action(Action {
            id: "pane.close_focused",
            title: "Close Focused Pane",
            category: "Pane",
            keywords: &["kill", "dismiss"],
            radial_icon: None,
            default_keys: const { &[KeyChord::cmd(KeyCode::KeyW)] },
            run: ActionRun::Custom(action_close_focused),
        })
        .add_action(Action {
            id: "pane.focus_next",
            title: "Focus Next Pane",
            category: "Pane",
            keywords: &["cycle", "switch"],
            radial_icon: None,
            default_keys: const { &[KeyChord::cmd(KeyCode::BracketRight)] },
            run: ActionRun::Custom(action_focus_next),
        })
        .add_action(Action {
            id: "pane.focus_prev",
            title: "Focus Previous Pane",
            category: "Pane",
            keywords: &["cycle", "switch"],
            radial_icon: None,
            default_keys: const { &[KeyChord::cmd(KeyCode::BracketLeft)] },
            run: ActionRun::Custom(action_focus_prev),
        })
        .add_action(Action {
            id: "view.zoom_in",
            title: "Zoom In",
            category: "View",
            keywords: &["scale", "magnify"],
            radial_icon: None,
            default_keys: const { &[KeyChord::cmd(KeyCode::Equal)] },
            run: ActionRun::Custom(|ctx| canvas::zoom_active(ctx.world, 1.1)),
        })
        .add_action(Action {
            id: "view.zoom_out",
            title: "Zoom Out",
            category: "View",
            keywords: &["scale", "shrink"],
            radial_icon: None,
            default_keys: const { &[KeyChord::cmd(KeyCode::Minus)] },
            run: ActionRun::Custom(|ctx| canvas::zoom_active(ctx.world, 1.0 / 1.1)),
        })
        .add_action(Action {
            id: "view.zoom_reset",
            title: "Reset Zoom",
            category: "View",
            keywords: &["scale", "100%", "actual size"],
            radial_icon: None,
            default_keys: const { &[KeyChord::cmd(KeyCode::Digit0)] },
            run: ActionRun::Custom(|ctx| canvas::zoom_reset_active(ctx.world)),
        })
        .add_action(Action {
            id: "keybinds.reload",
            title: "Reload Keybindings",
            category: "View",
            keywords: &["hotkey", "shortcut", "rebind", "config"],
            radial_icon: None,
            default_keys: &[],
            run: ActionRun::Custom(|ctx| actions::rebuild_keymap(ctx.world)),
        });
        app
            .add_systems(
                Startup,
                (
                    setup_camera_and_font,
                    register_terminal_kind,
                    // Runs after `setup_camera_and_font` so its `PaneFont` /
                    // `PaneFontMetrics` (the themed JetBrains mono used by
                    // every cosmic-text pane) deterministically replace the
                    // terminal's SF Mono defaults as a matched pair. Without
                    // the ordering, only one of the two resources might win
                    // and the caret grid would drift from the rendered text.
                    editor_bevy::setup_editor_font.after(setup_camera_and_font),
                    setup_ipc_listener,
                    request_microphone_access,
                ),
            )
            .add_systems(
                Update,
                (
                    mirror_active_project_to_style,
                    maintain_project_themes,
                    mirror_focus_to_style,
                    maintain_winit_mode_for_animation,
                    sync_canvas_clear_color,
                    window_geometry::save_on_change,
                ),
            )
            .add_systems(PostStartup, release_os_focus)
            // Single keyboard-ownership authority, before every Update
            // consumer reads it.
            .add_systems(PreUpdate, compute_keyboard_owner)
            .add_systems(Update, debug_fps_log)
            .add_systems(
                Update,
                (
                    drain_ipc_open_requests,
                ),
            )
            .add_systems(
                Update,
                (
                    // Focus-state + modifier reconciliation MUST run before
                    // handle_keyboard so a stuck Cmd (e.g. swallowed Cmd-up
                    // from a system shortcut) doesn't drop this frame's keys.
                    track_app_focus,
                    reconcile_macos_modifiers,
                    handle_terminal_content_press,
                    handle_terminal_selection_drag,
                    handle_scroll,
                    handle_resize,
                    handle_keyboard,
                    handle_file_drop,
                    sync_grid,
                    apply_bell_pulse,
                    apply_claude_notification_pulse,
                    clear_active_unread,
                    sync_dock_badge,
                )
                    .chain(),
            );
    }
}

fn register_terminal_kind(mut registry: ResMut<PaneRegistry>) {
    registry.register(pane_bevy::PaneKindSpec {
        kind: PANE_KIND,
        display_name: "Terminal",
        radial_icon: Some(">_"),
        default_size: Vec2::new(640.0, 400.0),
        spawn: terminal_spawn_from_config,
        snapshot: terminal_snapshot,
        on_close: Some(terminal_on_close),
    });
}

fn terminal_spawn_from_config(
    world: &mut World,
    entity: Entity,
    content_root: Entity,
    config: &Value,
) {
    let session_id = config
        .get("session_id")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| {
            let mut p = world.resource_mut::<Projects>();
            p.allocate_terminal_id()
        });
    let replay_bytes = scrollback_path(session_id).and_then(|p| std::fs::read(&p).ok());
    let initial_cwd = world
        .get::<ProjectMembership>(entity)
        .map(|m| m.0)
        .and_then(|pid| {
            world
                .get_resource::<Projects>()
                .and_then(|p| p.default_cwd_of(pid).map(str::to_string))
        });
    populate_terminal_pane(
        world,
        entity,
        content_root,
        session_id,
        initial_cwd,
        replay_bytes,
    );
}

fn terminal_snapshot(world: &World, entity: Entity) -> Value {
    let session_id = world
        .get::<TerminalSession>(entity)
        .map(|s| s.0)
        .unwrap_or(0);
    serde_json::json!({ "session_id": session_id })
}

fn terminal_on_close(world: &mut World, entity: Entity) {
    if let Some(store) = world.get_resource::<TerminalStore>()
        && let Some(data) = store.map.get(&entity)
    {
        data.worker.send(WorkerMsg::Shutdown);
    }
    if let Some(mut store) = world.get_resource_mut::<TerminalStore>() {
        store.map.remove(&entity);
    }
    let session_id = world.get::<TerminalSession>(entity).map(|s| s.0);
    if let Some(id) = session_id
        && let Some(p) = scrollback_path(id)
    {
        let _ = std::fs::remove_file(&p);
    }
    world.resource_mut::<Projects>().terminals_dirty = true;
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

/// Trigger the macOS microphone permission prompt at startup.
///
/// Why this is needed: `claude` (and any other CLI a user runs) records
/// audio through whichever process the OS deems *responsible* for it.
/// Our shells run under a launchd-detached daemon (double-fork +
/// `setsid`, PPID 1), so that responsible process is this app's code
/// identity — but a headless background daemon can't present the TCC
/// permission dialog. The foreground GUI can. Calling
/// `requestAccessForMediaType:` here pops the prompt once while we're
/// frontmost; the resulting grant is keyed on our code identity, which
/// the daemon shares (same signed binary), so `claude`'s voice dictation
/// can capture audio. Requires `NSMicrophoneUsageDescription` in
/// Info.plist (added by make-bundle.sh) — without it the request is
/// denied outright. Already-granted launches resolve to a no-op.
#[cfg(target_os = "macos")]
fn request_microphone_access() {
    use objc2::runtime::Bool;
    use objc2::{class, msg_send};
    use objc2_foundation::NSString;

    // winit/Bevy don't load AVFoundation, so the AVCaptureDevice class
    // wouldn't resolve at runtime. This empty extern forces a framework
    // load command into the binary.
    #[link(name = "AVFoundation", kind = "framework")]
    unsafe extern "C" {}

    // AVMediaTypeAudio's documented constant value is the FourCC "soun";
    // using the literal avoids linking the Obj-C string symbol.
    let media_type = NSString::from_str("soun");
    // Heap block: AVFoundation invokes the completion handler
    // asynchronously, after this function returns, so it must outlive the
    // stack frame.
    let handler = block2::RcBlock::new(|granted: Bool| {
        eprintln!(
            "[mic] microphone access request resolved: granted={}",
            granted.as_bool()
        );
    });
    let cls = class!(AVCaptureDevice);
    unsafe {
        let _: () = msg_send![
            cls,
            requestAccessForMediaType: &*media_type,
            completionHandler: &*handler,
        ];
    }
}

#[cfg(not(target_os = "macos"))]
fn request_microphone_access() {}

/// Holds the receiver half of the IPC channel. `mpsc::Receiver` is
/// `Send` but `!Sync`, so we install it as a `NonSend` resource and
/// drain it from a system that always runs on the main thread.
pub struct IpcInbox(pub std::sync::mpsc::Receiver<ipc::IpcMessage>);

fn setup_ipc_listener(world: &mut World) {
    let wakeup = world
        .get_resource::<bevy::winit::EventLoopProxyWrapper>()
        .map(|w| bevy::winit::EventLoopProxy::clone(w));
    // Let widget worker threads wake the reactive main loop (so
    // `set_animating(true)`, async frame publishes, and bus emits aren't
    // stalled until the next input / ~5s timeout). widget-bevy doesn't
    // depend on winit, so we hand it a closure over the proxy.
    if let Some(proxy) = wakeup.clone() {
        widget_bevy::set_wakeup_hook(move || {
            let _ = proxy.send_event(bevy::winit::WinitUserEvent::WakeUp);
        });
    }
    if let Some(rx) = ipc::spawn_listener(wakeup) {
        world.insert_non_send_resource(IpcInbox(rx));
    }
}

/// Drain any IPC requests received this frame and queue them as
/// entries in `PendingActions`. The actual world-mutating work
/// (file-read + editor spawn, widget spawn) happens in
/// `apply_pending_actions` next frame, so the IPC thread never touches
/// the World.
fn drain_ipc_open_requests(
    inbox: Option<NonSend<IpcInbox>>,
    mut pending: ResMut<PendingActions>,
    mut projects: ResMut<Projects>,
    mut drawer: ResMut<drawer::Drawer>,
    mut prism: ResMut<cube::Prism>,
    mut msg_bus: ResMut<widget_bevy::WidgetMsgBus>,
    mut palette_open: ResMut<command_palette::PaletteOpenRequest>,
    mut commands: Commands,
) {
    let Some(inbox) = inbox else { return };
    while let Ok(msg) = inbox.0.try_recv() {
        let ipc::IpcMessage {
            req,
            stream: mut _stream,
        } = msg;
        match req {
            ipc::IpcRequest::OpenFile { path, project } => {
                let target = match project {
                    Some(name) => OpenProjectTarget::ByName(name),
                    None => OpenProjectTarget::Active,
                };
                pending.open_files.push(OpenFileRequest {
                    path,
                    project: target,
                    origin: None,
                });
            }
            ipc::IpcRequest::SpawnWidget {
                command,
                args,
                title,
                cwd,
                project,
                position,
                size,
                kind,
            } => {
                let target = match project {
                    Some(name) => OpenProjectTarget::ByName(name),
                    None => OpenProjectTarget::Active,
                };
                let Some(project_id) = projects::resolve_project(&target, &projects) else {
                    eprintln!("[ipc] spawn_widget: no matching project");
                    continue;
                };

                // Route by `kind`. Rhai widgets get a different config
                // shape (`script` field, not `command`) and a different
                // pane kind in the registry.
                let pane_kind = kind.as_deref().unwrap_or(widget_bevy::PANE_KIND);
                let mut cfg = serde_json::Map::new();
                if pane_kind == widget_bevy::rhai_widget::PANE_KIND {
                    // For rhai_widget, `command` is the script filename.
                    cfg.insert("script".into(), Value::String(command));
                    if let Some(t) = title {
                        cfg.insert("title".into(), Value::String(t));
                    }
                } else {
                    cfg.insert("command".into(), Value::String(command));
                    if !args.is_empty() {
                        cfg.insert(
                            "args".into(),
                            Value::Array(args.into_iter().map(Value::String).collect()),
                        );
                    }
                    if let Some(t) = title {
                        cfg.insert("title".into(), Value::String(t));
                    }
                    if let Some(p) = cwd {
                        cfg.insert(
                            "cwd".into(),
                            Value::String(p.to_string_lossy().into_owned()),
                        );
                    }
                }
                let kind_static: &'static str = match pane_kind {
                    "widget" => widget_bevy::PANE_KIND,
                    "rhai_widget" => widget_bevy::rhai_widget::PANE_KIND,
                    other => Box::leak(other.to_string().into_boxed_str()),
                };
                pending.new_panes.push(NewPaneRequest {
                    kind: kind_static,
                    project_id,
                    origin: position.map(|[x, y]| Vec2::new(x, y)),
                    size: size.map(|[w, h]| Vec2::new(w, h)),
                    config: Value::Object(cfg),
                });
            }
            ipc::IpcRequest::ToggleCube => {
                prism.pending_toggle = true;
            }
            ipc::IpcRequest::OpenPalette { query, ask } => {
                palette_open.requested = true;
                palette_open.seed = query;
                palette_open.ask = ask;
            }
            ipc::IpcRequest::ListProjects => {
                use std::io::Write as _;
                let active = projects.active;
                let entries: Vec<Value> = projects
                    .list
                    .iter()
                    .map(|p| {
                        serde_json::json!({
                            "id": p.id,
                            "name": p.name,
                            "active": Some(p.id) == active,
                        })
                    })
                    .collect();
                let body = serde_json::json!({ "projects": entries });
                let bytes = match serde_json::to_vec(&body) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("[ipc] list_projects: serialize: {}", e);
                        continue;
                    }
                };
                if let Err(e) = _stream.write_all(&bytes) {
                    eprintln!("[ipc] list_projects: write: {}", e);
                }
                let _ = _stream.shutdown(std::net::Shutdown::Write);
            }
            ipc::IpcRequest::SetProjectDefaultCwd { project, cwd } => {
                let target = match project.as_deref() {
                    Some("active") | None => OpenProjectTarget::Active,
                    Some(name) => OpenProjectTarget::ByName(name.to_string()),
                };
                let Some(project_id) = projects::resolve_project(&target, &projects) else {
                    eprintln!("[ipc] set_project_default_cwd: no matching project");
                    continue;
                };
                let cwd_str = cwd.map(|p| p.to_string_lossy().into_owned());
                let changed = projects.set_default_cwd(project_id, cwd_str.clone());
                eprintln!(
                    "[ipc] set_project_default_cwd: project={} cwd={:?} changed={}",
                    project_id, cwd_str, changed
                );
            }
            ipc::IpcRequest::SendInbox {
                project,
                sender,
                subject,
                body,
            } => {
                // Resolve project: name → id, or "active" / None → active.
                let target = match project.as_deref() {
                    Some("active") | None => OpenProjectTarget::Active,
                    Some(name) => OpenProjectTarget::ByName(name.to_string()),
                };
                let Some(project_id) = projects::resolve_project(&target, &projects) else {
                    eprintln!("[ipc] send_inbox: no matching project");
                    continue;
                };
                let sender = sender.unwrap_or_else(|| "external".to_string());
                if let Err(e) = inbox::append_message(project_id, sender, subject, body) {
                    eprintln!("[ipc] send_inbox: append: {}", e);
                }
            }
            ipc::IpcRequest::SuggestPane {
                kind,
                title,
                command,
                cwd,
                reason,
                config,
                project,
                from_cwd,
            } => {
                // Resolve the pane kind. Explicit `kind` wins; otherwise
                // a bare `command` implies the run-button "command pane".
                let kind = match kind {
                    Some(k) => k,
                    None if command.is_some() => "run-button".to_string(),
                    None => {
                        eprintln!(
                            "[ipc] suggest_pane: need --kind or --command; dropping"
                        );
                        continue;
                    }
                };

                // Build the config blob. Explicit `config` is stored
                // verbatim; otherwise synthesize a run-button config from
                // command/title/cwd (matching `run_button_snapshot`).
                let config = match config {
                    Some(c) => c,
                    None => {
                        let mut cfg = serde_json::Map::new();
                        if let Some(cmd) = &command {
                            cfg.insert("command".into(), Value::String(cmd.clone()));
                        }
                        if let Some(t) = &title {
                            cfg.insert("title".into(), Value::String(t.clone()));
                        }
                        if let Some(p) = &cwd {
                            cfg.insert(
                                "cwd".into(),
                                Value::String(p.to_string_lossy().into_owned()),
                            );
                        }
                        Value::Object(cfg)
                    }
                };

                // Row title: explicit, else the command, else the kind.
                let row_title = title
                    .or_else(|| command.clone())
                    .unwrap_or_else(|| kind.clone());

                // Scope the suggestion to a project at arrival: an
                // explicit name wins; otherwise map the caller's cwd to
                // its owning project; otherwise leave it unscoped
                // (global — shows in every project's drawer).
                let project_id = match &project {
                    Some(name) => {
                        projects::resolve_project(
                            &OpenProjectTarget::ByName(name.clone()),
                            &projects,
                        )
                    }
                    None => from_cwd
                        .as_deref()
                        .and_then(|c| projects::project_for_cwd(c, &projects)),
                };

                drawer.push(kind, row_title, reason, config, project_id);
            }
            ipc::IpcRequest::Screenshot { path } => {
                // Render-side capture: spawn a one-shot Screenshot entity
                // and save it to disk when the GPU readback lands. Works
                // headlessly and never grabs the user's screen.
                use bevy::render::view::screenshot::{save_to_disk, Screenshot};
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path));
            }
            ipc::IpcRequest::CloseProjectPanes { project, kind } => {
                let target = match project.as_deref() {
                    Some("active") | None => OpenProjectTarget::Active,
                    Some(name) => OpenProjectTarget::ByName(name.to_string()),
                };
                let Some(project_id) = projects::resolve_project(&target, &projects) else {
                    eprintln!("[ipc] close_project_panes: no matching project");
                    continue;
                };
                pending.close_panes.push((project_id, kind));
            }
            ipc::IpcRequest::WidgetMessage { project, topic, payload, retain } => {
                let target = match project.as_deref() {
                    Some("active") | None => OpenProjectTarget::Active,
                    Some(name) => OpenProjectTarget::ByName(name.to_string()),
                };
                let Some(project_id) = projects::resolve_project(&target, &projects) else {
                    eprintln!("[ipc] widget_message: no matching project");
                    continue;
                };
                msg_bus.push_external(widget_bevy::PendingMsg {
                    project: Some(project_id),
                    topic,
                    payload,
                    sender: "tbmsg".to_string(),
                    retain,
                });
            }
        }
    }
}

/// Cmd+O opens a native file picker and queues the chosen file as a
/// new editor pane in the active project. The picker is synchronous —
/// it blocks the calling thread until the user picks or cancels, which
/// matches how every other macOS app handles file dialogs.
///
/// The `NonSendMarker` is load-bearing: `rfd::FileDialog::pick_file` on
/// macOS does `dispatch_sync(main_queue, ...)` internally. If this system
/// ran on a Compute Task Pool thread (Bevy's default), the worker would
/// `dispatch_sync` to main while the main thread sat parked in the
/// executor waiting for the worker to finish — instant deadlock. The
/// marker pins us to the main thread so the dispatch is a no-op.
///
/// We swallow Cmd+O ourselves so the focused pane (terminal or editor)
/// never sees a stray "o" insert. Both pane keyboard handlers already
/// skip Cmd-modified keys, but we still bail explicitly here in case
/// that contract loosens.
/// `file.open` action (Cmd+O). Opens a native file picker and routes
/// the chosen file to an editor pane in the active project. Runs on the
/// main thread via the exclusive action dispatcher, so the blocking
/// `rfd` dialog is fine. Cmd+O is swallowed by the keybind matcher so
/// the focused pane never sees a stray "o" insert.
fn action_open_file(ctx: &mut actions::ActionCtx) {
    let dialog = rfd::FileDialog::new()
        .set_directory(std::env::current_dir().unwrap_or_else(|_| ".".into()))
        .set_title("Open file");
    let Some(path) = dialog.pick_file() else {
        return;
    };
    ctx.world
        .resource_mut::<PendingActions>()
        .open_files
        .push(OpenFileRequest {
            path,
            project: OpenProjectTarget::Active,
            origin: None,
        });
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
    // Main camera explicitly on layer 0 — pane-bevy reserves layer 0
    // for pane chrome + non-pane scene content, and uses layers 1.. for
    // each per-pane camera. Making the main camera's layer explicit
    // matches the contract documented in `pane-bevy/src/camera.rs`.
    world.spawn((
        Camera2d,
        bevy::camera::visibility::RenderLayers::layer(0),
    ));

    // Menu overlay camera — renders only `MENU_OVERLAY_LAYER` at a
    // camera order far above any per-pane camera, so radial / context
    // menus draw on top of every pane even when many panes are
    // focused. `clear_color: None` keeps the underlying scene visible
    // wherever the overlay has no geometry.
    world.spawn((
        Camera2d,
        bevy::camera::Camera {
            order: MENU_OVERLAY_CAMERA_ORDER,
            clear_color: bevy::camera::ClearColorConfig::None,
            ..default()
        },
        bevy::camera::visibility::RenderLayers::layer(MENU_OVERLAY_LAYER),
    ));

    let font_bytes: &'static [u8] = load_primary_font();

    let font_handle = world
        .resource_mut::<Assets<Font>>()
        .add(Font::try_from_bytes(font_bytes.to_vec()).expect("SFMono must parse"));
    world.insert_resource(MonoFont(font_handle.clone()));
    // pane-bevy uses this for chrome glyphs (close button, title text).
    world.insert_resource(PaneFont(font_handle));

    let cell_width = measure_cell_width(font_bytes, FONT_SIZE);
    world.insert_resource(MonoMetrics { cell_width });
    world.insert_resource(pane_bevy::PaneFontMetrics {
        cell_width,
        font_size: FONT_SIZE,
    });

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
/// can group + show/hide it. It is REQUIRED: a terminal with no project
/// leaks across every project (see `assert_pane_project_invariant`).
///
/// `session_id` is the persistence key — the worker logs raw pty bytes
/// to `scrollback_path(session_id)` and on restart loads the same file
/// back into a fresh Terminal via `replay_bytes`.
pub fn spawn_terminal(
    world: &mut World,
    rect: PaneRect,
    project_id: u64,
    session_id: u64,
    replay_bytes: Option<Vec<u8>>,
) -> Entity {
    let SpawnedPane {
        entity: terminal_entity,
        content_root,
    } = spawn_pane(world, PANE_KIND, "Terminal", rect, Some(project_id));
    let initial_cwd = world
        .get_resource::<Projects>()
        .and_then(|p| p.default_cwd_of(project_id).map(str::to_string));
    populate_terminal_pane(
        world,
        terminal_entity,
        content_root,
        session_id,
        initial_cwd,
        replay_bytes,
    );
    terminal_entity
}

/// Insert terminal-specific components on an already-spawned pane, spawn
/// its worker, and add the cursor child under `content_root`. Shared
/// between `spawn_terminal` and the registry restore path.
///
/// `initial_cwd` overrides the daemon's default-to-$HOME behavior for
/// this pane's shell. Used to honor a project's remembered
/// `default_cwd` (populated by the inference layer). Only consulted
/// when the daemon has to fork a fresh shell — attaching to an
/// already-running daemon ignores it.
fn populate_terminal_pane(
    world: &mut World,
    terminal_entity: Entity,
    content_root: Entity,
    session_id: u64,
    initial_cwd: Option<String>,
    replay_bytes: Option<Vec<u8>>,
) {
    let cell_width = world.resource::<MonoMetrics>().cell_width;
    let rect = *world
        .get::<PaneRect>(terminal_entity)
        .expect("pane entity must already have PaneRect");
    let (cols, rows) = grid_size_for_rect(rect.size, cell_width);

    // Spawn the worker thread up front so the libghostty Terminal +
    // Pty + render iterators all live on the worker side.
    let wakeup = world
        .get_resource::<bevy::winit::EventLoopProxyWrapper>()
        .map(|w| bevy::winit::EventLoopProxy::clone(w));
    let worker = WorkerHandle::spawn(
        session_id,
        default_shell_command(),
        initial_cwd,
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

    // Build the GPU grid for this terminal: one mesh + one cells texture +
    // one material instance. `sync_grid` updates the cells texture
    // texel-by-texel when the worker publishes a new snapshot.
    let term_grid = build_term_grid(world, content_root, cell_width, cols, rows);

    world.entity_mut(terminal_entity).insert((
        TerminalRev::default(),
        term_grid,
        TerminalSelection::default(),
        BellPulse::default(),
        TerminalSession(session_id),
        TerminalCursor(cursor),
    ));

    world
        .get_resource_mut::<TerminalStore>()
        .expect("TerminalStore resource (did setup_camera_and_font run?)")
        .map
        .insert(terminal_entity, data);
}

fn grid_size_for_rect(size: Vec2, cell_width: f32) -> (u16, u16) {
    let content_w = (size.x - 2.0 * MARGIN).max(0.0);
    let content_h = (size.y - TITLE_H - 2.0 * MARGIN).max(0.0);
    let cols = ((content_w / cell_width).floor() as u16).max(1);
    let rows = ((content_h / LINE_HEIGHT).floor() as u16).max(1);
    (cols, rows)
}

/// Spawn the single quad+material that renders an entire terminal grid
/// on the GPU. Returns the `TermGrid` component that goes on the pane
/// entity; the render entity itself becomes a child of `content_root`.
fn build_term_grid(
    world: &mut World,
    content_root: Entity,
    cell_width: f32,
    cols: u16,
    rows: u16,
) -> TermGrid {
    // Snapshot atlas geometry up-front so we don't hold the resource
    // borrow across the asset writes below.
    let (atlas_cols, atlas_slot_w, atlas_slot_h, atlas_stride_w, atlas_stride_h, atlas_dim, atlas_image) = {
        let atlas = world.resource::<GlyphAtlas>();
        (
            atlas.cols_per_row(),
            atlas.slot_w(),
            atlas.slot_h(),
            atlas.stride_w(),
            atlas.stride_h(),
            atlas.dim(),
            atlas.image.clone(),
        )
    };

    // Initial background is the worker's default bg (matches what a
    // freshly-spawned libghostty Terminal reports for unwritten cells).
    let default_bg = pack_rgb(13, 15, 20);
    let cells_image = make_cells_image(cols as u32, rows as u32, default_bg);
    let cells_handle = world.resource_mut::<Assets<Image>>().add(cells_image);

    let grid_w = cols as f32 * cell_width;
    let grid_h = rows as f32 * LINE_HEIGHT;
    let mesh_handle = world
        .resource_mut::<Assets<Mesh>>()
        .add(Mesh::from(Rectangle::new(grid_w, grid_h)));

    let params = TermParams {
        cols: cols as u32,
        rows: rows as u32,
        atlas_cols,
        atlas_slot_w,
        atlas_slot_h,
        atlas_dim,
        atlas_stride_w,
        atlas_stride_h,
    };
    let material_handle = world.resource_mut::<Assets<TermMaterial>>().add(TermMaterial {
        params,
        atlas: atlas_image,
        cells: cells_handle.clone(),
    });

    // `Rectangle` mesh is centered on its origin; shift it so top-left
    // lands at the content_root origin (matches where the cursor sprite
    // and the previous per-cell sprites lived).
    //
    // `Visibility::Inherited` is load-bearing: Bevy's 2D extract path
    // queries `&ViewVisibility`, which only exists if `Visibility` (and
    // its required `InheritedVisibility`/`ViewVisibility` companions)
    // is on the entity. Without it the mesh silently never reaches the
    // render world, the shader never runs, and the pane shows the chrome
    // background through what looks like a blank quad.
    let render_entity = world
        .spawn((
            ChildOf(content_root),
            bevy::mesh::Mesh2d(mesh_handle.clone()),
            bevy::sprite_render::MeshMaterial2d(material_handle.clone()),
            Transform::from_xyz(grid_w * 0.5, -(grid_h * 0.5), 0.0),
            Visibility::Inherited,
        ))
        .id();

    TermGrid {
        material: material_handle,
        cells_image: cells_handle,
        mesh: mesh_handle,
        render_entity,
        cols,
        rows,
        last_rendered_generation: 0,
        was_visible: true,
    }
}

// ---------- Resize ----------

/// When a terminal's rect resolves to a different grid dimension than
/// the worker's snapshot reports, send a `Resize` message. The worker
/// applies it, the next snapshot reflects the new dims, and `sync_grid`
/// resizes its sprite pools accordingly.
fn handle_resize(
    metrics: Res<MonoMetrics>,
    store: Res<TerminalStore>,
    rect_q: Query<(Entity, &PaneRect, &PaneKindMarker)>,
) {
    for (entity, rect, kind) in &rect_q {
        if kind.0 != PANE_KIND {
            continue;
        }
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
    focused: Res<FocusedPane>,
    owner: Res<pane_bevy::KeyboardOwner>,
    store: Res<TerminalStore>,
    kinds: Query<&PaneKindMarker>,
    mut last_drop_reason: Local<Option<&'static str>>,
) {
    // Diagnostic: log on the *transition* into a drop reason whenever
    // a real press event is being dropped. Logs once per reason-change
    // (not per event) so a stuck-Cmd or dead-shell scenario surfaces
    // exactly one stderr line, not a flood.
    let buffered: Vec<KeyboardInput> = events.read().cloned().collect();
    let any_press = buffered.iter().any(|e| e.state.is_pressed());
    let mut report = |reason: &'static str| {
        if any_press && last_drop_reason.as_deref() != Some(reason) {
            eprintln!("[handle_keyboard] dropping key press: {reason}");
            *last_drop_reason = Some(reason);
        }
    };
    // Re-emit so the rest of the function can iterate buffered events
    // without re-reading the channel.
    let mut events_iter = buffered.iter();

    // Skip unless the focused pane is a terminal.
    let target_kind = focused.0.and_then(|e| kinds.get(e).ok());
    if !matches!(target_kind, Some(k) if k.0 == PANE_KIND) {
        report("focused pane is not a terminal");
        return;
    }
    // Central keyboard ownership: a text modal (command palette, project
    // rename) holds `KeyboardOwner::Modal`, so even though this terminal is
    // still the focused pane, it must not consume keystrokes. (Subsumes the
    // old explicit `Renaming` check.)
    if matches!(focused.0, Some(t) if !owner.allows_pane(t)) {
        report("keyboard owned by a text modal");
        return;
    }
    let Some(target) = focused.0 else {
        report("no focused pane");
        return;
    };
    let Some(data) = store.map.get(&target) else {
        report("focused pane has no terminal data");
        return;
    };
    let child_alive = {
        let g = data.worker.snapshot.lock().expect("snapshot lock");
        g.child_alive
    };
    if !child_alive {
        report("shell process has exited (child_alive=false)");
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
        report("Cmd modifier held — see stuck-modifier note");
        return;
    }
    // We made it past every gate. Clear any stale drop reason so the
    // next drop transition logs again.
    *last_drop_reason = None;

    // For v0 we always emit xterm-style cursor-key escapes (CSI A/B/C/D);
    // we don't have main-thread visibility into the worker's DECCKM mode.
    // Most shells/readline work fine with these — apps that need SS3
    // form (like vim in normal mode) will be addressed when we route
    // mode bits through the snapshot.
    let app_cursor = false;

    let mut out: Vec<u8> = Vec::with_capacity(16);

    for ev in events_iter.by_ref() {
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

        // Option+Left / Option+Right: send the readline word-jump
        // bytes (ESC+b, ESC+f) instead of the regular arrow CSI. zsh,
        // bash, fish, and friends all bind these to backward-word /
        // forward-word, matching what macOS Terminal.app and iTerm2 do
        // for Option+arrow. We do this *before* the named-key branch
        // so the regular arrow encoding doesn't fire.
        if alt
            && matches!(ev.key_code, KeyCode::ArrowLeft | KeyCode::ArrowRight)
        {
            out.push(0x1b);
            out.push(if matches!(ev.key_code, KeyCode::ArrowLeft) {
                b'b'
            } else {
                b'f'
            });
            continue;
        }

        // Named keys we know the VT encoding for. Arrows / Home / End
        // honor DECCKM.
        if let Some(bytes) = named_key_bytes(&ev.key_code, app_cursor) {
            // Option+Enter sends ESC+CR (the iTerm2-compatible "meta
            // newline" convention). Shells / readline bind \e\r to
            // self-insert-newline, so the user gets a literal LF in
            // their command line instead of submitting it. Same trick
            // Terminal.app's "Option-Enter inserts newline" uses.
            //
            // Option+Backspace sends ESC+0x7f, which readline/zle bind
            // to `backward-kill-word`. Same meta-prefix convention as
            // Option+Enter.
            if alt
                && matches!(
                    ev.key_code,
                    KeyCode::Enter | KeyCode::NumpadEnter | KeyCode::Backspace
                )
            {
                out.push(0x1b);
            }
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

/// Route Finder/Files-app drag-drops onto a terminal pane: insert the
/// dropped file's absolute path (POSIX single-quoted) into the pty,
/// followed by a trailing space. Mirrors what Terminal.app and iTerm2
/// do when you drag a file onto their window — Claude Code's prompt
/// then sees the path as plain text and can read the image from disk.
///
/// Bevy fires one `DroppedFile` event per file, so multi-file drops
/// land as space-separated tokens for free.
fn handle_file_drop(
    mut drops: MessageReader<bevy::window::FileDragAndDrop>,
    windows: Query<&Window>,
    viewport: Res<pane_bevy::PaneViewport>,
    store: Res<TerminalStore>,
    panes: Query<(Entity, &PaneRect, &PaneKindMarker, Option<&Visibility>), With<PaneTag>>,
    mut focused: ResMut<FocusedPane>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let cursor = window.cursor_position();

    for ev in drops.read() {
        let bevy::window::FileDragAndDrop::DroppedFile { path_buf, .. } = ev else {
            continue;
        };
        let Some(pt) = cursor else {
            // Drop arrived without a cursor sample (window not focused
            // yet, or pointer left between drop start + finish). Without
            // a cursor we can't pick a pane — skip rather than guess.
            eprintln!(
                "[file-drop] no cursor position — ignoring drop of {}",
                path_buf.display()
            );
            continue;
        };
        let visible: Vec<(Entity, PaneRect)> = panes
            .iter()
            .filter(|(_, _, kind, vis)| {
                kind.0 == PANE_KIND && !matches!(vis, Some(Visibility::Hidden))
            })
            .map(|(e, r, _, _)| (e, *r))
            .collect();
        let Some(target) = pane_bevy::topmost_pane_at(viewport.window_to_canvas(pt), &visible)
        else {
            eprintln!(
                "[file-drop] no terminal under cursor — ignoring drop of {}",
                path_buf.display()
            );
            continue;
        };
        let Some(data) = store.map.get(&target) else {
            continue;
        };

        // Canonicalize to an absolute path so the receiving shell /
        // Claude Code resolves the file regardless of its cwd. Fall
        // back to the raw path if canonicalize fails (e.g., the source
        // is a symlink the user wants preserved as-typed).
        let abs = std::fs::canonicalize(path_buf).unwrap_or_else(|_| path_buf.clone());
        let quoted = posix_single_quote(&abs.to_string_lossy());
        let mut bytes = quoted.into_bytes();
        bytes.push(b' ');
        data.worker.send(WorkerMsg::Input(bytes));
        data.worker.send(WorkerMsg::ScrollToBottom);
        focused.0 = Some(target);
    }
}

/// POSIX-safe shell quoting: wrap in single quotes; embed any literal
/// `'` as `'\''` (close-quote, escaped quote, reopen-quote). Always
/// safe regardless of the path's contents — spaces, $, *, !, newlines
/// are all preserved literally by the shell.
fn posix_single_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
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

/// Convert a window-space cursor position to a cell coord (col, row)
/// inside the terminal at `rect`. The result is intentionally not
/// clamped — the caller owns clipping to the actual grid bounds.
pub fn pt_to_cell(pt: Vec2, rect: &PaneRect, cell_width: f32) -> (i32, i32) {
    let local = pane_bevy::pt_to_content_local(pt, rect);
    let col = (local.x / cell_width).floor() as i32;
    let row = (local.y / LINE_HEIGHT).floor() as i32;
    (col, row)
}

/// Snapshot's `viewport_offset` for `entity`'s terminal, or 0 if the
/// store doesn't know about it (e.g. mid-spawn).
fn viewport_offset_of(store: &TerminalStore, entity: Entity) -> u64 {
    let Some(data) = store.map.get(&entity) else {
        return 0;
    };
    let g = data.worker.snapshot.lock().expect("snapshot lock");
    g.viewport_offset
}

/// Promote a `(col, viewport_row)` cell to a `(col, absolute_row)`
/// selection cell using a terminal's current `viewport_offset`. Done at
/// click/drag time so the selection stays pinned to its content while
/// the user scrolls (see [`TerminalSelection`]).
fn promote_to_absolute(cell: (i32, i32), viewport_offset: u64) -> (i32, i64) {
    (cell.0, viewport_offset as i64 + cell.1 as i64)
}

/// Start a selection drag inside a terminal pane in response to a
/// pane-bevy content press event.
fn handle_terminal_content_press(
    mut presses: MessageReader<PaneContentPressed>,
    metrics: Res<MonoMetrics>,
    viewport: Res<pane_bevy::PaneViewport>,
    store: Res<TerminalStore>,
    rects: Query<&PaneRect>,
    kinds: Query<&PaneKindMarker>,
    mut selections: Query<&mut TerminalSelection>,
) {
    for ev in presses.read() {
        let Ok(kind) = kinds.get(ev.pane) else {
            continue;
        };
        if kind.0 != PANE_KIND {
            continue;
        }
        // Clear any other terminal's selection.
        for mut sel in &mut selections {
            sel.clear();
        }
        let Ok(rect) = rects.get(ev.pane) else { continue };
        let viewport_cell =
            pt_to_cell(viewport.window_to_canvas(ev.window_pt), rect, metrics.cell_width);
        let cell = promote_to_absolute(viewport_cell, viewport_offset_of(&store, ev.pane));
        if let Ok(mut sel) = selections.get_mut(ev.pane) {
            sel.anchor = Some(cell);
            sel.head = Some(cell);
            sel.dragging = true;
        }
    }
}

/// Update the selection head while LMB is held; clear `dragging` on
/// release. Mirrors editor-bevy's `handle_text_select_drag` shape.
fn handle_terminal_selection_drag(
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    metrics: Res<MonoMetrics>,
    viewport: Res<pane_bevy::PaneViewport>,
    store: Res<TerminalStore>,
    mut selections: Query<(Entity, &PaneRect, &PaneKindMarker, &mut TerminalSelection)>,
) {
    if buttons.just_released(MouseButton::Left) {
        for (_, _, kind, mut sel) in &mut selections {
            if kind.0 == PANE_KIND {
                sel.dragging = false;
            }
        }
        return;
    }
    if !buttons.pressed(MouseButton::Left) {
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Some(pt) = window.cursor_position() else { return };
    let pt_canvas = viewport.window_to_canvas(pt);

    for (entity, rect, kind, mut sel) in &mut selections {
        if kind.0 != PANE_KIND || !sel.dragging {
            continue;
        }
        let viewport_cell = pt_to_cell(pt_canvas, rect, metrics.cell_width);
        let cell = promote_to_absolute(viewport_cell, viewport_offset_of(&store, entity));
        sel.head = Some(cell);
    }
}

/// Mouse-wheel scrolls the terminal under the cursor (in the active
/// project). Pixel-mode events (trackpads) accumulate a fractional line
/// counter so small swipes still register.
fn handle_scroll(
    mut wheel: MessageReader<MouseWheel>,
    mut accum: Local<f32>,
    windows: Query<&Window>,
    sidebar: Res<Sidebar>,
    viewport: Res<pane_bevy::PaneViewport>,
    projects: Res<Projects>,
    store: Res<TerminalStore>,
    keys: Res<ButtonInput<KeyCode>>,
    all_panes: Query<(Entity, &PaneRect, Option<&Visibility>), With<PaneTag>>,
    terminals: Query<
        (Entity, Option<&ProjectMembership>, &PaneKindMarker),
        With<PaneTag>,
    >,
) {
    // Cmd+scroll is reserved for canvas pan (see canvas.rs). Drain the
    // events so they don't accumulate, but don't act on them.
    if keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight) {
        wheel.clear();
        *accum = 0.0;
        return;
    }
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

    // Topmost pane of ANY kind under the cursor. If something is
    // sitting over the terminal (e.g. a widget pane), the wheel
    // belongs to that pane — don't steal it for the terminal
    // underneath.
    let all_rects: Vec<(Entity, PaneRect)> = all_panes
        .iter()
        .filter(|(_, _, vis)| !matches!(vis, Some(Visibility::Hidden)))
        .map(|(e, r, _)| (e, *r))
        .collect();
    let Some(target) = pane_bevy::topmost_pane_at(viewport.window_to_canvas(pt), &all_rects)
    else {
        return;
    };
    // Only consume the wheel if that topmost pane is a terminal in
    // the active project.
    let Ok((_, membership, kind)) = terminals.get(target) else {
        return;
    };
    if kind.0 != PANE_KIND {
        return;
    }
    let in_active_project = match (projects.active, membership) {
        (Some(a), Some(p)) => a == p.0,
        _ => false,
    };
    if !in_active_project {
        return;
    }
    let Some(data) = store.map.get(&target) else {
        return;
    };

    // Bevy: wheel.y > 0 = scroll-up gesture = reveal older content.
    // libghostty: ScrollViewport::Delta is positive toward the active
    // area, negative back into history. So mirror the sign.
    let scroll_delta = -whole_lines;
    data.worker.send(WorkerMsg::ScrollDelta(scroll_delta));
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
/// Maintain `AppFocused` from app-level activation state, NOT winit's
/// `WindowFocused` events: on macOS those fire on per-window key focus
/// transitions and have been observed flipping back to `true`
/// spuriously even while the app is backgrounded. Polling
/// `NSApplication.isActive` each frame matches what the user actually
/// perceives as "looking at us". Logs every transition while diagnosing.
fn debug_fps_log(
    time: Res<Time>,
    mut frames: Local<u64>,
    mut last: Local<f64>,
) {
    if std::env::var("FPS_LOG").is_err() {
        return;
    }
    *frames += 1;
    let now = time.elapsed_secs_f64();
    if *last == 0.0 {
        *last = now;
        *frames = 0;
        return;
    }
    if now - *last >= 1.0 {
        eprintln!("[fps] {:.1}", *frames as f64 / (now - *last));
        *frames = 0;
        *last = now;
    }
}

fn track_app_focus(
    mut focused: ResMut<AppFocused>,
    mut keys: ResMut<ButtonInput<KeyCode>>,
) {
    let now = current_app_active();
    if focused.0 != now {
        eprintln!("[focus] {} → {}", focused.0, now);
        focused.0 = now;
        // Cmd+Tab (and any other modal app switch) eats the key-release
        // events for whatever was held — most commonly Cmd itself. Without
        // this reset, ButtonInput<KeyCode> stays "pressed" on Super* and
        // every subsequent keystroke gets dropped by handle_keyboard's
        // `if cmd { return; }` gate.
        keys.release_all();
    }
}

/// Reconcile Bevy's modifier state with the OS's real-time view each
/// frame. The focus-transition reset in `track_app_focus` catches the
/// common Cmd+Tab case, but system shortcuts (Spotlight, Mission
/// Control, screenshots) can swallow a Cmd-up without changing
/// `frontmostApplication`, leaving `ButtonInput<KeyCode>::pressed(Super*)`
/// stuck true. Polling NSEvent.modifierFlags is the authoritative
/// signal: if Bevy thinks a modifier is held but the OS says it isn't,
/// release it — otherwise every terminal keystroke after the stuck
/// modifier gets silently dropped by handle_keyboard's gate.
#[cfg(target_os = "macos")]
fn reconcile_macos_modifiers(mut keys: ResMut<ButtonInput<KeyCode>>) {
    use objc2_app_kit::{NSEvent, NSEventModifierFlags};

    let flags = unsafe { NSEvent::modifierFlags_class() };
    let want = |mask: NSEventModifierFlags| flags.contains(mask);

    let cmd = want(NSEventModifierFlags::NSEventModifierFlagCommand);
    let shift = want(NSEventModifierFlags::NSEventModifierFlagShift);
    let ctrl = want(NSEventModifierFlags::NSEventModifierFlagControl);
    let alt = want(NSEventModifierFlags::NSEventModifierFlagOption);

    let pairs = [
        (cmd, KeyCode::SuperLeft),
        (cmd, KeyCode::SuperRight),
        (shift, KeyCode::ShiftLeft),
        (shift, KeyCode::ShiftRight),
        (ctrl, KeyCode::ControlLeft),
        (ctrl, KeyCode::ControlRight),
        (alt, KeyCode::AltLeft),
        (alt, KeyCode::AltRight),
    ];
    for (os_held, code) in pairs {
        if !os_held && keys.pressed(code) {
            keys.release(code);
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn reconcile_macos_modifiers() {}

#[cfg(target_os = "macos")]
fn current_app_active() -> bool {
    // `NSApplication.isActive` doesn't reliably flip for our app under
    // winit / Bevy on macOS — we've observed it staying `true` even
    // when the user has Cmd+Tab'd to another app. The authoritative
    // signal is "are we the frontmost app, system-wide": ask
    // NSWorkspace and compare its `frontmostApplication.pid` to ours.
    use objc2_app_kit::NSWorkspace;
    let workspace = unsafe { NSWorkspace::sharedWorkspace() };
    let Some(front) = (unsafe { workspace.frontmostApplication() }) else {
        return true;
    };
    let front_pid = unsafe { front.processIdentifier() };
    let our_pid = unsafe { nix::libc::getpid() };
    front_pid == our_pid
}

#[cfg(not(target_os = "macos"))]
fn current_app_active() -> bool {
    true
}

/// Bell counter. Polls each terminal's worker `bell_count` and bumps
/// the per-project unread counter for every fresh BEL the user can't
/// currently see (window unfocused, or its project not active). No
/// in-pane visual — only the sidebar badge + dock-tile badge react.
fn apply_bell_pulse(
    store: Res<TerminalStore>,
    app_focused: Res<AppFocused>,
    mut projects: ResMut<Projects>,
    mut terms: Query<(Entity, Option<&ProjectMembership>, &mut BellPulse)>,
) {
    let window_focused = app_focused.0;
    let active_project = projects.active;
    for (entity, membership, mut pulse) in &mut terms {
        let Some(data) = store.map.get(&entity) else {
            continue;
        };
        let cur = data
            .worker
            .bell_count
            .load(std::sync::atomic::Ordering::Relaxed);
        if cur <= pulse.last_seen {
            continue;
        }
        let new_bells = cur - pulse.last_seen;
        pulse.last_seen = cur;
        let Some(membership) = membership else {
            eprintln!(
                "[bell] new={} on terminal {:?} but no ProjectMembership — skipping",
                new_bells, entity
            );
            continue;
        };
        let pid = membership.0;
        let visible = window_focused && active_project == Some(pid);
        eprintln!(
            "[bell] new={} pid={} window_focused={} active={:?} visible={}",
            new_bells, pid, window_focused, active_project, visible
        );
        if visible {
            continue;
        }
        for _ in 0..new_bells {
            projects.bump_unread(pid);
        }
        eprintln!(
            "[bell] bumped pid={} → {} (total {})",
            pid,
            projects.unread_bells.get(&pid).copied().unwrap_or(0),
            projects.unread_total()
        );
    }
}

/// Bumps the per-project unread counter when Claude's Notification hook
/// fires ("Claude is waiting for your input" / "needs your permission").
///
/// This is the *authoritative* "Claude wants attention" signal. It
/// arrives on the bus (via `claude-event-logger notification`) every
/// time, independent of whether Claude emits a terminal BEL — recent
/// Claude builds frequently don't, which is why the BEL-only
/// `apply_bell_pulse` path stopped lighting up project badges. We route
/// the event to a project by `terminal_session_id` → the pane's
/// `TerminalSession` → its `ProjectMembership`, then apply the same
/// visibility gate as the bell path: skip when the user is already
/// looking at that project.
fn apply_claude_notification_pulse(
    mut events: MessageReader<claude_bus_bevy::ClaudeBusEvent>,
    app_focused: Res<AppFocused>,
    mut projects: ResMut<Projects>,
    panes: Query<(&TerminalSession, Option<&ProjectMembership>)>,
) {
    let window_focused = app_focused.0;
    let active_project = projects.active;
    for ev in events.read() {
        if ev.kind != "notification" {
            continue;
        }
        // Standalone Claude sessions (not running inside one of our
        // panes) carry an empty / non-numeric session id — ignore them.
        let Ok(sid) = ev.terminal_session_id.parse::<u64>() else {
            continue;
        };
        let pid = panes
            .iter()
            .find(|(ts, _)| ts.0 == sid)
            .and_then(|(_, pm)| pm.map(|p| p.0));
        let Some(pid) = pid else {
            eprintln!(
                "[notify] notification for session {} but no project pane — skipping",
                sid
            );
            continue;
        };
        let visible = window_focused && active_project == Some(pid);
        eprintln!(
            "[notify] notification sid={} pid={} window_focused={} active={:?} visible={}",
            sid, pid, window_focused, active_project, visible
        );
        if visible {
            continue;
        }
        projects.bump_unread(pid);
        eprintln!(
            "[notify] bumped pid={} → {} (total {})",
            pid,
            projects.unread_bells.get(&pid).copied().unwrap_or(0),
            projects.unread_total()
        );
    }
}

/// Clears the active project's unread count whenever the OS window is
/// focused — that's the moment "the user is looking at it" becomes
/// true. Runs every frame; the no-op fast path inside `clear_unread`
/// (returns false when count was already zero) keeps the cost free.
fn clear_active_unread(
    app_focused: Res<AppFocused>,
    mut projects: ResMut<Projects>,
) {
    if !app_focused.0 {
        return;
    }
    let Some(active) = projects.active else {
        return;
    };
    projects.clear_unread(active);
}

/// Push the sum of unread bell counts to the macOS Dock icon as a
/// badge label. Tracked via a `Local<u64>` so we only hit the FFI when
/// the value actually changes — `setBadgeLabel` is cheap but it's not
/// free, and most frames have no change.
#[cfg(target_os = "macos")]
fn sync_dock_badge(
    // NonSendMarker forces this system onto the main thread, which is
    // mandatory for NSDockTile / NSApplication AppKit calls. Without
    // it Bevy may schedule us on a worker thread and `MainThreadMarker`
    // refuses to construct → the badge never updates.
    _main: bevy::ecs::system::NonSendMarker,
    projects: Res<Projects>,
    mut last: Local<u64>,
) {
    let total = projects.unread_total();
    if total == *last {
        return;
    }
    eprintln!("[dock] total {} → {}", *last, total);
    *last = total;
    use objc2_app_kit::NSApplication;
    use objc2_foundation::{MainThreadMarker, NSString};
    let Some(mtm) = MainThreadMarker::new() else {
        eprintln!("[dock] MainThreadMarker::new() returned None — not main thread?");
        return;
    };
    let app = NSApplication::sharedApplication(mtm);
    let tile = unsafe { app.dockTile() };
    let label = if total == 0 {
        None
    } else {
        Some(NSString::from_str(&total.to_string()))
    };
    unsafe { tile.setBadgeLabel(label.as_deref()) };
    eprintln!("[dock] setBadgeLabel({:?})", total);
}

#[cfg(not(target_os = "macos"))]
fn sync_dock_badge(_projects: Res<Projects>, _last: Local<u64>) {}

fn sync_grid(
    metrics: Res<MonoMetrics>,
    theme: Res<style_bevy::Theme>,
    themes: Res<style_bevy::ProjectThemes>,
    mut atlas: ResMut<GlyphAtlas>,
    mut images: ResMut<Assets<Image>>,
    mut layouts: ResMut<Assets<TextureAtlasLayout>>,
    mut materials: ResMut<Assets<TermMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    store: Res<TerminalStore>,
    mut terminals: Query<
        (
            Entity,
            &TerminalCursor,
            &mut TermGrid,
            &PaneKindMarker,
            &Visibility,
            Option<&pane_bevy::PaneProject>,
        ),
        With<PaneTag>,
    >,
    mut transform_q: Query<&mut Transform>,
    mut vis_q: Query<&mut Visibility, Without<TermGrid>>,
    mut prof: Local<SyncGridProfile>,
) {
    use std::time::Instant;
    let frame_start = Instant::now();

    // Theme-driven defaults, substituted in below for any cell whose fg
    // or bg matches libghostty's reported `default_fg/default_bg` (plain
    // text the shell didn't color). Resolved PER PANE from its project's
    // theme so each terminal reads in its own project's palette (a paper
    // project gives ink-on-cream, a terminal project phosphor-on-black),
    // including all faces in the cube overview. These globals are the
    // fallback for panes with no project theme cached.
    let global_default_fg = lin_to_rgb_bytes(theme.color(style_bevy::tokens::FG));
    let global_default_bg = lin_to_rgb_bytes(theme.color(style_bevy::tokens::BG));
    let theme_changed = theme.is_changed() || themes.is_changed();

    let mut local_cells: Vec<SnapCell> = Vec::new();
    let mut local_dirty_rows: Vec<bool> = Vec::new();

    let mut work_done = false;
    let mut lock_ns: u128 = 0;
    let mut mutate_ns: u128 = 0;
    let mut cells_touched = 0u64;

    // Scratch reused across terminals: avoids per-frame allocation in
    // the dirty-row hot path.
    let mut pending_writes: Vec<(usize, GpuCell)> = Vec::new();

    for (entity, cursor_marker, mut grid, kind, vis, proj) in &mut terminals {
        // Per-pane theme defaults: this terminal's project theme if known,
        // else the global (active) theme.
        let (theme_default_fg, theme_default_bg) = proj
            .and_then(|p| themes.get(p.0))
            .map(|t| {
                (
                    lin_to_rgb_bytes(t.color(style_bevy::tokens::FG)),
                    lin_to_rgb_bytes(t.color(style_bevy::tokens::BG)),
                )
            })
            .unwrap_or((global_default_fg, global_default_bg));
        if kind.0 != PANE_KIND {
            continue;
        }
        let _prof = pane_bevy::prof::pane_span(entity.to_bits(), "terminal");
        let Some(data) = store.map.get(&entity) else {
            continue;
        };

        // Propagate this pane's visibility to the worker so it skips the
        // 60Hz Bevy wake while the user isn't looking. The worker keeps
        // processing pty bytes either way — the libghostty terminal
        // state stays correct — but inactive-project panes contribute
        // zero to per-frame schedule cost.
        let is_hidden = matches!(vis, Visibility::Hidden);
        data.worker
            .visible
            .store(!is_hidden, std::sync::atomic::Ordering::Relaxed);
        if is_hidden {
            grid.was_visible = false;
            continue;
        }
        // First frame after un-hide: libghostty's dirty_rows only reflect
        // changes SINCE the last publish, but the worker has been
        // publishing all along without us reading. Force a full repaint
        // to bring the cells texture up to the current grid state.
        let just_shown = !grid.was_visible;
        grid.was_visible = true;

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

        // Cursor — compare-before-write to avoid spurious Changed.
        let cursor_entity = cursor_marker.0;
        if let Ok(mut v) = vis_q.get_mut(cursor_entity) {
            let want = if cursor.is_some() {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
            if *v != want {
                *v = want;
            }
        }
        if let Some((cx, cy)) = cursor {
            if let Ok(mut t) = transform_q.get_mut(cursor_entity) {
                let wx = cx as f32 * metrics.cell_width;
                let wy = -(cy as f32) * LINE_HEIGHT;
                let wz = 1.0;
                if t.translation.x != wx
                    || t.translation.y != wy
                    || t.translation.z != wz
                {
                    t.translation.x = wx;
                    t.translation.y = wy;
                    t.translation.z = wz;
                }
            }
        }

        let pool_changed = grid.cols != cols || grid.rows != rows;
        let nothing_changed = !pool_changed
            && !just_shown
            && !theme_changed
            && grid.last_rendered_generation == generation;
        if nothing_changed {
            continue;
        }
        work_done = true;
        let mutate_t = Instant::now();

        // Resize the GPU grid: replace the cells image + mesh + uniform
        // params. Cheap because there's only one of each per terminal.
        if pool_changed {
            let bg_packed = pack_rgb(
                theme_default_bg.0,
                theme_default_bg.1,
                theme_default_bg.2,
            );
            // Replace cells image in place — keep the same Handle so the
            // material doesn't need rebinding.
            if let Some(img) = images.get_mut(&grid.cells_image) {
                *img = make_cells_image(cols as u32, rows as u32, bg_packed);
            }
            // Replace mesh contents (same handle stays bound).
            let grid_w = cols as f32 * metrics.cell_width;
            let grid_h = rows as f32 * LINE_HEIGHT;
            if let Some(mesh) = meshes.get_mut(&grid.mesh) {
                *mesh = Mesh::from(Rectangle::new(grid_w, grid_h));
            }
            if let Some(mat) = materials.get_mut(&grid.material) {
                mat.params.cols = cols as u32;
                mat.params.rows = rows as u32;
            }
            if let Ok(mut t) = transform_q.get_mut(grid.render_entity) {
                t.translation.x = grid_w * 0.5;
                t.translation.y = -(grid_h * 0.5);
            }
            grid.cols = cols;
            grid.rows = rows;
        }

        // Pass 1: resolve glyph indices and collect (idx, GpuCell) for
        // every dirty cell. Atlas lookups borrow `images` mutably (atlas
        // may insert a new glyph and re-upload), so we can't hold a
        // `Image::get_mut(cells_image)` borrow at the same time. Two
        // passes keeps the borrow checker happy and lets us compare
        // existing-vs-new in pass 2.
        pending_writes.clear();
        let force_all = pool_changed || just_shown || theme_changed;
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
                // Substitute theme defaults for cells the shell didn't
                // color explicitly. libghostty has already filled in
                // its own palette default at the worker; we recognize
                // it by exact-equal byte match and swap in the theme
                // color instead. False positives (shell explicitly set
                // a color that happens to equal libghostty's default)
                // are visually identical to the user's intent, since
                // the theme picks "the same color the shell would've
                // shown" anyway.
                let theme_fg =
                    if final_fg.r == default_fg.r
                        && final_fg.g == default_fg.g
                        && final_fg.b == default_fg.b
                    {
                        theme_default_fg
                    } else {
                        (final_fg.r, final_fg.g, final_fg.b)
                    };
                let theme_bg =
                    if final_bg.r == default_bg.r
                        && final_bg.g == default_bg.g
                        && final_bg.b == default_bg.b
                    {
                        theme_default_bg
                    } else {
                        (final_bg.r, final_bg.g, final_bg.b)
                    };
                let glyph_index =
                    atlas.lookup_or_insert(cell.ch, &mut images, &mut layouts);
                let gpu = GpuCell {
                    glyph_index,
                    fg_packed: pack_rgb(theme_fg.0, theme_fg.1, theme_fg.2),
                    bg_packed: pack_rgb(theme_bg.0, theme_bg.1, theme_bg.2),
                    flags: 0,
                };
                pending_writes.push((idx, gpu));
                cells_touched += 1;
            }
        }

        // Pass 2: filter no-op writes (cells whose state didn't change)
        // by reading from the current cells image, then mutate it in one
        // go. Reading via `Assets::get` doesn't mark the asset Changed —
        // important so we don't re-upload the texture every frame when
        // libghostty flagged a row dirty but no visible state moved.
        if pending_writes.is_empty() {
            grid.last_rendered_generation = generation;
            mutate_ns += mutate_t.elapsed().as_nanos();
            continue;
        }
        let mut needs_upload = false;
        {
            let current = images
                .get(&grid.cells_image)
                .expect("cells image must be alive");
            let current_cells: &[GpuCell] = bytemuck::cast_slice(
                current
                    .data
                    .as_ref()
                    .expect("cells image must have CPU data")
                    .as_slice(),
            );
            for (idx, new_cell) in &pending_writes {
                if current_cells
                    .get(*idx)
                    .map_or(true, |existing| existing != new_cell)
                {
                    needs_upload = true;
                    break;
                }
            }
        }
        if needs_upload {
            if let Some(img) = images.get_mut(&grid.cells_image) {
                let dst: &mut [GpuCell] = bytemuck::cast_slice_mut(
                    img.data
                        .as_mut()
                        .expect("cells image must have CPU data"),
                );
                for (idx, new_cell) in &pending_writes {
                    if let Some(slot) = dst.get_mut(*idx) {
                        if slot != new_cell {
                            *slot = *new_cell;
                        }
                    }
                }
            }
            // Mutating the cells image alone re-extracts the GpuImage with
            // a fresh wgpu texture, but the material's bind group still
            // caches the OLD texture view — so without poking the material
            // too, the shader keeps sampling the pre-modification upload.
            // (Bevy's own tilemap_chunk material follows this same pattern;
            // see `bevy_sprite_render::tilemap_chunk::update_chunk` —
            // `materials.get_mut(material.id())` is the load-bearing line.)
            let _ = materials.get_mut(&grid.material);
        }

        grid.last_rendered_generation = generation;
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

/// Linear-RGB theme color → (r, g, b) byte triple in sRGB space, the
/// format libghostty + `pack_rgb` use. Used by `sync_grid` to convert
/// theme tokens before stuffing them into per-cell colors.
fn lin_to_rgb_bytes(c: bevy::color::LinearRgba) -> (u8, u8, u8) {
    let srgb = Color::LinearRgba(c).to_srgba();
    (
        (srgb.red.clamp(0.0, 1.0) * 255.0).round() as u8,
        (srgb.green.clamp(0.0, 1.0) * 255.0).round() as u8,
        (srgb.blue.clamp(0.0, 1.0) * 255.0).round() as u8,
    )
}

// ---------- style-bevy glue ----------

/// Mirror `Projects.active` into style-bevy's `ActiveProject`. Also
/// ensures each newly-observed project has its state.json loaded into
/// memory so dust timers + the per-project preset are available.
///
/// Note: this no longer touches `ActiveThemePath`. `presets.rs` is the
/// sole owner of that resource — it derives it from `ActiveStylePreset`
/// (which is itself loaded per-project from `ProjectStyleState`), so
/// theming follows the active project's saved preset automatically.
fn mirror_active_project_to_style(
    projects: Res<Projects>,
    mut active_proj: ResMut<style_bevy::shader::ActiveProject>,
    mut state: ResMut<style_bevy::ProjectStyleState>,
    data_dir: Option<Res<style_bevy::StyleDataDir>>,
) {
    if !projects.is_changed() {
        return;
    }
    if active_proj.0 == projects.active {
        return;
    }
    active_proj.0 = projects.active;
    if let Some(pid) = projects.active {
        if let Some(d) = data_dir.as_ref() {
            style_bevy::state::load_project_state(d, &mut state, pid);
        }
        // Intentionally NOT calling note_focus here — switching to a
        // project on startup or via the sidebar shouldn't blow away
        // accumulated dust. The mirror_focus_to_style hook records
        // actual focus gestures.
    }
}

/// Keep `ProjectThemes` (style-bevy's per-project theme cache) populated
/// for every project, so each pane can render in its OWN project's theme
/// — in the cube overview (all projects on screen) and in flat view.
///
/// Full reload when the project set or any project's preset changes;
/// targeted reload of just the active project when its theme file is
/// live-edited (only the active project's file is watched, so a
/// `ThemeChanged` means *its* tokens moved). A `(pid, preset)` signature
/// hash keeps this from re-reading 14 theme files every frame.
fn maintain_project_themes(
    projects: Res<Projects>,
    mut style_state: ResMut<style_bevy::ProjectStyleState>,
    registry: Res<style_bevy::StylePresetRegistry>,
    data_dir: Option<Res<style_bevy::StyleDataDir>>,
    mut themes: ResMut<style_bevy::ProjectThemes>,
    mut theme_changed: MessageReader<style_bevy::ThemeChanged>,
    mut last_sig: Local<u64>,
) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let theme_edited = theme_changed.read().last().is_some();

    // Make sure EVERY project's saved style state (its preset) is in
    // memory, not just the ones visited this session. `load_project_state`
    // is idempotent (loads once, then a cheap map check), so this is fine
    // to run each frame. Without it, an unvisited project resolves with
    // `preset_of() == None` → default theme until you switch to it.
    if let Some(d) = data_dir.as_ref() {
        for p in &projects.list {
            style_bevy::state::load_project_state(d, &mut style_state, p.id);
        }
    }

    let mut hasher = DefaultHasher::new();
    for p in &projects.list {
        p.id.hash(&mut hasher);
        style_state.preset_of(p.id).hash(&mut hasher);
    }
    let sig = hasher.finish();

    let dd = data_dir.as_deref();
    if sig != *last_sig {
        // Project set or a preset changed — rebuild the whole cache and
        // drop entries for projects that no longer exist.
        *last_sig = sig;
        let keep: std::collections::HashSet<u64> =
            projects.list.iter().map(|p| p.id).collect();
        themes.retain_projects(&keep);
        for p in &projects.list {
            themes.set(
                p.id,
                style_bevy::resolve_project_theme(p.id, &style_state, &registry, dd),
            );
        }
    } else if theme_edited {
        // Live edit of the active project's theme.rhai — reload just it.
        if let Some(pid) = projects.active {
            themes.set(
                pid,
                style_bevy::resolve_project_theme(pid, &style_state, &registry, dd),
            );
        }
    }
}

/// Cmd+Shift+D opens the style dev panel (a Rhai widget). Lets you
/// scrub dust / edit / age / time_scale without waiting for real time
/// to pass. Spawning goes through the same `PendingActions.new_panes`
/// channel the radial menu uses, so all the usual pane-bevy chrome
/// applies.
/// `view.dev_panel` action (Cmd+Shift+D). Opens the style dev panel
/// (a Rhai widget). Dedups: each spawn leaves a fresh Rhai worker thread
/// ticking the script at 30 Hz (~50% CPU per duplicate), so if a dev
/// panel already exists anywhere on the canvas, silently do nothing.
fn action_open_dev_panel(ctx: &mut actions::ActionCtx) {
    let exists = {
        let mut q = ctx
            .world
            .query::<&widget_bevy::rhai_widget::RhaiWidget>();
        q.iter(ctx.world).any(|w| w.script == "dev_panel.rhai")
    };
    if exists {
        return;
    }
    let Some(active) = ctx.world.resource::<projects::Projects>().active else {
        return;
    };
    ctx.world
        .resource_mut::<projects::PendingActions>()
        .new_panes
        .push(projects::NewPaneRequest {
            kind: widget_bevy::rhai_widget::PANE_KIND,
            project_id: active,
            origin: None,
            size: Some(Vec2::new(420.0, 280.0)),
            config: serde_json::json!({
                "script": "dev_panel.rhai",
                "title": "Style dev panel",
            }),
        });
}

/// `view.toggle_cube` action. Mirrors the `IpcRequest::ToggleCube` path
/// (and `cube.rs`'s own Cmd+Shift+\ keybind) by flipping the prism's
/// pending-toggle flag.
fn action_toggle_cube(ctx: &mut actions::ActionCtx) {
    ctx.world.resource_mut::<cube::Prism>().pending_toggle = true;
}

/// `pane.close_focused` — route the focused pane through the normal close
/// path (runs the kind's `on_close`, then despawns). No-op when nothing is
/// focused.
fn action_close_focused(ctx: &mut actions::ActionCtx) {
    if let Some(e) = ctx.world.resource::<pane_bevy::FocusedPane>().0 {
        ctx.world
            .resource_mut::<pane_bevy::PendingPaneActions>()
            .close
            .push(e);
    }
}

/// `pane.focus_next` / `pane.focus_prev` — move keyboard focus to the next
/// / previous pane in the active project, ordered back-to-front by z and
/// wrapping around.
fn action_focus_next(ctx: &mut actions::ActionCtx) {
    cycle_focus(ctx.world, 1);
}

fn action_focus_prev(ctx: &mut actions::ActionCtx) {
    cycle_focus(ctx.world, -1);
}

fn cycle_focus(world: &mut World, dir: i32) {
    let Some(active) = world.resource::<projects::Projects>().active else {
        return;
    };
    // Active-project panes, ordered back-to-front by z so the cycle order
    // matches the visual stack.
    let mut panes: Vec<(Entity, f32)> = world
        .query_filtered::<(Entity, &pane_bevy::PaneRect, &pane_bevy::PaneProject), With<pane_bevy::PaneTag>>()
        .iter(world)
        .filter(|(_, _, proj)| proj.0 == active)
        .map(|(e, rect, _)| (e, rect.z))
        .collect();
    if panes.is_empty() {
        return;
    }
    panes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let order: Vec<Entity> = panes.into_iter().map(|(e, _)| e).collect();

    let cur = world.resource::<pane_bevy::FocusedPane>().0;
    let next = match cur.and_then(|c| order.iter().position(|&e| e == c)) {
        Some(i) => {
            let n = order.len() as i32;
            order[(((i as i32 + dir) % n + n) % n) as usize]
        }
        // Nothing (or an off-project pane) focused: enter the stack from
        // the front when going forward, the back when going backward.
        None if dir >= 0 => *order.last().unwrap(),
        None => *order.first().unwrap(),
    };
    world.resource_mut::<pane_bevy::FocusedPane>().0 = Some(next);
}

/// The single authority for `pane_bevy::KeyboardOwner` — runs in
/// `PreUpdate`, before any keyboard consumer in `Update`, so every
/// handler sees a consistent owner for the frame. Precedence: a text-
/// entry modal (command palette or project rename) owns everything; else
/// the focused pane owns typing; else nobody. See the type docs in
/// `pane-bevy` for why this replaces the old per-handler focus gating.
fn compute_keyboard_owner(
    palette: Res<command_palette::CommandPalette>,
    renaming: Res<projects::Renaming>,
    pending_seq: Res<actions::PendingSequence>,
    focused: Res<pane_bevy::FocusedPane>,
    mut owner: ResMut<pane_bevy::KeyboardOwner>,
) {
    // A pending chord sequence also claims the keyboard: the continuation
    // key (e.g. the `C` in `⌘K C`) must reach the action matcher, not the
    // focused pane. The matcher itself special-cases `Modal` while a
    // sequence is in progress, so it keeps reading.
    let next = if renaming.id.is_some() || palette.open || !pending_seq.chords.is_empty() {
        pane_bevy::KeyboardOwner::Modal
    } else if let Some(e) = focused.0 {
        pane_bevy::KeyboardOwner::Pane(e)
    } else {
        pane_bevy::KeyboardOwner::None
    };
    if *owner != next {
        *owner = next;
    }
}

/// Track the active theme's `canvas_bg` token in `ClearColor` so a
/// preset switch retones the void around the dust shader (visible at
/// pane rounded-corners + during the windex sweep).
fn sync_canvas_clear_color(
    theme: Res<style_bevy::Theme>,
    mut clear: ResMut<ClearColor>,
) {
    if !theme.is_changed() {
        return;
    }
    let c = Color::LinearRgba(theme.color(style_bevy::tokens::CANVAS_BG));
    if clear.0 != c {
        clear.0 = c;
    }
}

/// Switch the winit update mode between Continuous (every frame) and
/// Reactive (only on input + a 5s heartbeat) depending on whether the
/// active visual preset needs to animate. Continuous burns ~1.5 cores
/// at 60fps because the dust shader and chrome materials all redraw
/// every frame; Reactive is battery-friendly. The transition itself
/// is event-driven (preset switch), so reactive mode reliably wakes
/// up to handle it.
///
/// Rule: a preset that ships a chrome.wgsl that references
/// `params.time` is assumed to animate. Static custom shaders
/// (sketch, mesh, blueprint) are *not* animated and stay on
/// Reactive — they paint once per Reactive frame and that's fine.
fn maintain_winit_mode_for_animation(
    preset: Res<style_bevy::ActiveStylePreset>,
    registry: Res<style_bevy::StylePresetRegistry>,
    drawer: Res<drawer::Drawer>,
    prism: Res<cube::Prism>,
    palette: Res<command_palette::CommandPalette>,
    rhai_widgets: Query<&widget_bevy::rhai_widget::RhaiWidget>,
    mut settings: ResMut<bevy::winit::WinitSettings>,
) {
    let preset_animates = preset.0.as_deref().map_or(false, |name| {
        registry
            .presets
            .iter()
            .find(|p| p.name == name)
            .map_or(false, |p| p.chrome_animates)
    });
    // A Rhai widget that opted into animation via `set_animating(true)`
    // (e.g. the datalog IDE results pane draining a `datalog` subprocess in
    // `on_frame`, or chess polling Stockfish) also needs every frame. Without
    // this term, the reactive loop only wakes ~every 5s while the window is
    // idle, so the widget's `on_frame` tick — and thus its proc-drain —
    // arrives ~5s late even though the underlying work finished in ms.
    let widget_animating = rhai_widgets.iter().any(|w| w.is_animating());
    // The drawer's slide and the 3D project prism (live textures + camera
    // animation) are the other sources of "needs every frame". The cooldown
    // keeps redrawing briefly after the prism closes so the flat panes
    // repaint instead of staying black.
    let want_continuous = preset_animates
        || widget_animating
        || drawer.animating()
        || prism.active
        || prism.continuous_cooldown > 0
        // Keep ticking while the palette is open so its DeepSeek worker
        // result is polled promptly (reactive mode would wake only every
        // 5s otherwise) and keystrokes feel instant.
        || palette.open;

    let target = if want_continuous {
        bevy::winit::UpdateMode::Continuous
    } else {
        bevy::winit::UpdateMode::reactive(std::time::Duration::from_secs(5))
    };
    if settings.focused_mode != target {
        settings.focused_mode = target;
    }
    // A proc-polling widget (datalog query drain, chess vs Stockfish)
    // must keep ticking even when the window loses focus, or its work
    // hangs the moment the user clicks away. The other continuous
    // sources are decorative and don't need unfocused frames, so only an
    // animating widget escalates the unfocused mode.
    let unfocused_target = if widget_animating {
        bevy::winit::UpdateMode::Continuous
    } else {
        bevy::winit::UpdateMode::reactive_low_power(std::time::Duration::from_secs(60))
    };
    if settings.unfocused_mode != unfocused_target {
        settings.unfocused_mode = unfocused_target;
    }
}

/// Cmd+Shift+T opens the live theme editor (a Rhai widget). OkLCh
/// steppers per color token; click to focus a token, then ± each
/// of L / C / h. Writes propagate to the active preset's `theme.rhai`
/// via the bridge; notify watcher reloads it and the rest of the
/// app retones the same frame.
/// `view.theme_editor` action (Cmd+Shift+T). Opens the live theme editor
/// (a Rhai widget): OkLCh steppers per color token that write back to the
/// active preset's `theme.rhai`. Dedups like the dev panel.
fn action_open_theme_editor(ctx: &mut actions::ActionCtx) {
    let exists = {
        let mut q = ctx
            .world
            .query::<&widget_bevy::rhai_widget::RhaiWidget>();
        q.iter(ctx.world).any(|w| w.script == "theme_editor.rhai")
    };
    if exists {
        return;
    }
    let Some(active) = ctx.world.resource::<projects::Projects>().active else {
        return;
    };
    ctx.world
        .resource_mut::<projects::PendingActions>()
        .new_panes
        .push(projects::NewPaneRequest {
            kind: widget_bevy::rhai_widget::PANE_KIND,
            project_id: active,
            origin: None,
            size: Some(Vec2::new(420.0, 600.0)),
            config: serde_json::json!({
                "script": "theme_editor.rhai",
                "title": "Theme editor",
            }),
        });
}

/// Cmd+Shift+S opens the style preset picker (a Rhai widget). Lists
/// every preset under `~/.terminal-bevy/styles/` plus a `(per-project
/// theme)` entry; clicking switches the active style and persists the
/// choice. Same dedup logic as the dev panel.
/// `view.style_picker` action (Cmd+Shift+S). Opens the style preset
/// picker (a Rhai widget). No dedup: each instance is a parked, event-
/// driven worker (~zero idle CPU), so stacking a few is cheap.
fn action_open_style_picker(ctx: &mut actions::ActionCtx) {
    let Some(active) = ctx.world.resource::<projects::Projects>().active else {
        return;
    };
    ctx.world
        .resource_mut::<projects::PendingActions>()
        .new_panes
        .push(projects::NewPaneRequest {
            kind: widget_bevy::rhai_widget::PANE_KIND,
            project_id: active,
            origin: None,
            size: Some(Vec2::new(280.0, 240.0)),
            config: serde_json::json!({
                "script": "style_picker.rhai",
                "title": "Styles",
            }),
        });
}

/// When the user focuses any pane, mark that pane's project as
/// recently-active so its dust timer resets. Skips the very first
/// observation after startup — that one fires when the persistence
/// layer restores focus, and counting "we restored your focus state"
/// as engagement would zero out dust across restarts.
fn mirror_focus_to_style(
    focused: Res<pane_bevy::FocusedPane>,
    pane_projects: Query<&pane_bevy::PaneProject>,
    mut state: ResMut<style_bevy::ProjectStyleState>,
    mut last: Local<Option<Entity>>,
    mut warmed_up: Local<bool>,
) {
    if !focused.is_changed() {
        return;
    }
    let Some(entity) = focused.0 else {
        *last = None;
        return;
    };
    if *last == Some(entity) {
        return;
    }
    *last = Some(entity);
    if !*warmed_up {
        *warmed_up = true;
        return;
    }
    if let Ok(pp) = pane_projects.get(entity) {
        state.note_focus(pp.0);
    }
}
