//! The suggestion **drawer** — a Quake-style panel that slides down from
//! the top of the canvas and holds *suggested* panes the AI has parked
//! for us.
//!
//! ## Why this exists
//!
//! One workflow pattern: you (or the AI) open a side terminal and run a
//! command. From that we can *infer* a pane might be useful — e.g. a
//! `run-button` pre-loaded with that command. We don't want to spew
//! panes onto the canvas uninvited, so instead they land here, in a
//! staging area you pull down when you want it.
//!
//! ## How items arrive
//!
//! Today the only feeder is the `tbsuggest` CLI → `IpcRequest::SuggestPane`
//! → [`Drawer::push`] (see `lib.rs`'s IPC drain). The push API is kept
//! deliberately source-agnostic: a future bus-driven feeder (the
//! `inference_dispatch` classifiers already publish verdicts on
//! `claude-bus`) can call the same [`Drawer::push`] without touching any
//! of the UI below.
//!
//! ## What an item is
//!
//! A [`Suggestion`] is a generic `PaneSnapshot`-shaped record: a
//! registered pane `kind` plus its JSON `config`. Materializing one just
//! feeds that pair to the normal spawn path (`PendingActions.new_panes`
//! → `spawn_pane_from_registry`), so the drawer works for *any* pane
//! kind, not just run-buttons.
//!
//! Picking an item pulls it onto the canvas in the active (or hinted)
//! project and removes it from the drawer. The `×` on a row dismisses
//! without spawning.
//!
//! Toggle with **Cmd+J**; **Esc** closes. (Quake's backtick key is
//! swallowed by macOS for window cycling before winit ever sees it, so
//! we use Cmd+J.) The panel is non-modal: it floats over the top of the
//! canvas but doesn't dim or block the rest of the surface.

use std::path::PathBuf;

use bevy::camera::visibility::RenderLayers;
use bevy::input::keyboard::KeyboardInput;
use bevy::math::Rect;
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::LineHeight;

use pane_bevy::{InputConsumed, PaneRegistry};
use serde::{Deserialize, Serialize};

use crate::projects::{kind_to_static, NewPaneRequest, PendingActions, Projects, Sidebar};
use crate::MonoFont;

/// Drawer sits above the sidebar but below the radial / context menus,
/// all on `MENU_OVERLAY_LAYER` so it composites over canvas + panes.
const DRAWER_Z: f32 = 550.0;

const HEADER_H: f32 = 34.0;
const ROW_H: f32 = 46.0;
const PAD: f32 = 10.0;
const DISMISS_W: f32 = 30.0;
/// No scrollback in v0 — rows past this are summarized in the footer and
/// logged (never silently dropped).
const MAX_VISIBLE_ROWS: usize = 8;
/// Seconds for a full open / close slide.
const SLIDE_SECS: f32 = 0.16;
/// Approx glyph advance as a fraction of font size for the mono font;
/// used only for cheap text truncation, not layout.
const CHAR_W_RATIO: f32 = 0.6;

// ---------- Data model ----------

/// One parked pane suggestion. Mirrors the shape of a `PaneSnapshot`
/// (kind + config) plus drawer-only presentation fields.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Suggestion {
    pub id: u64,
    /// Registered pane kind, e.g. `"run-button"`, `"terminal"`.
    pub kind: String,
    /// Row title.
    pub title: String,
    /// One-line "why this is here", shown muted under the title.
    #[serde(default)]
    pub reason: Option<String>,
    /// Kind-specific config handed verbatim to the kind's `spawn`
    /// callback when materialized.
    pub config: serde_json::Value,
    /// Owning project, resolved at arrival (explicit `--project` or the
    /// caller's cwd → project). `None` = unscoped/global: shows in every
    /// project's drawer and materializes into whatever's active.
    #[serde(default)]
    pub project_id: Option<u64>,
}

/// On-disk shape for `~/.jim/suggestions.json`.
#[derive(Default, Serialize, Deserialize)]
struct PersistedDrawer {
    #[serde(default)]
    items: Vec<Suggestion>,
    #[serde(default)]
    next_id: u64,
}

#[derive(Resource, Default)]
pub struct Drawer {
    pub items: Vec<Suggestion>,
    /// User intent: down or up. The visible position eases toward this.
    pub open: bool,
    /// Slide progress, 0 (fully hidden above the top) → 1 (fully down).
    pub anim: f32,
    /// Hovered row index (into the *visible* slice).
    pub hovered: Option<usize>,
    next_id: u64,
    /// Set when `items`/`next_id` change and need flushing to disk.
    dirty: bool,
}

impl Drawer {
    /// Park a new suggestion. Source-agnostic entry point — IPC today,
    /// a bus feeder tomorrow. Assigns an id, marks the drawer dirty so
    /// it persists, and (gently) does not auto-open: the badge/peek is
    /// the user's cue, not a popup that steals focus.
    pub fn push(
        &mut self,
        kind: String,
        title: String,
        reason: Option<String>,
        config: serde_json::Value,
        project_id: Option<u64>,
    ) {
        let id = self.next_id;
        self.next_id += 1;
        self.items.push(Suggestion {
            id,
            kind,
            title,
            reason,
            config,
            project_id,
        });
        self.dirty = true;
    }

    /// True while the slide is mid-transition (used to keep winit in
    /// Continuous mode so the animation doesn't stutter).
    pub fn animating(&self) -> bool {
        (self.open && self.anim < 1.0) || (!self.open && self.anim > 0.0)
    }

    /// Whether a run-button suggestion with this exact command already
    /// exists in the same scope (so the inference feeder doesn't stack
    /// duplicates of a repeated command). Items differing only in scope
    /// are treated as distinct.
    pub fn has_command(&self, project_id: Option<u64>, command: &str) -> bool {
        self.items.iter().any(|s| {
            s.project_id == project_id
                && s.config.get("command").and_then(|c| c.as_str()) == Some(command)
        })
    }
}

// ---------- Plugin ----------

pub struct DrawerPlugin;

impl Plugin for DrawerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Drawer>()
            .add_systems(Startup, drawer_load)
            .add_systems(
                Update,
                (
                    drawer_input,
                    drawer_hover,
                    drawer_animate,
                    drawer_render,
                    drawer_persist,
                )
                    .chain(),
            );
    }
}

// ---------- Persistence ----------

fn drawer_file() -> Option<PathBuf> {
    Some(crate::data_dir()?.join("suggestions.json"))
}

fn drawer_load(mut drawer: ResMut<Drawer>) {
    let Some(path) = drawer_file() else { return };
    let Ok(bytes) = std::fs::read(&path) else {
        return; // first run / no file yet
    };
    match serde_json::from_slice::<PersistedDrawer>(&bytes) {
        Ok(p) => {
            // Guard next_id against hand-edited / stale files so a fresh
            // push can never collide with a loaded id.
            let max_id = p.items.iter().map(|s| s.id).max().map(|m| m + 1).unwrap_or(0);
            drawer.next_id = p.next_id.max(max_id);
            drawer.items = p.items;
        }
        Err(e) => eprintln!("[drawer] parse {}: {}", path.display(), e),
    }
}

fn drawer_persist(mut drawer: ResMut<Drawer>) {
    if !drawer.dirty {
        return;
    }
    drawer.dirty = false;
    let Some(path) = drawer_file() else { return };
    let state = PersistedDrawer {
        items: drawer.items.clone(),
        next_id: drawer.next_id,
    };
    let bytes = match serde_json::to_vec_pretty(&state) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[drawer] serialize: {}", e);
            return;
        }
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = path.with_extension("json.tmp");
    if let Err(e) = std::fs::write(&tmp, &bytes).and_then(|_| std::fs::rename(&tmp, &path)) {
        eprintln!("[drawer] write {}: {}", path.display(), e);
    }
}

// ---------- Layout ----------

/// One laid-out row in window coords (y-down, top-left origin). `dismiss`
/// is the `×` hit area at the right edge; `body` is the rest (click to
/// materialize). `index` is the *visible-row position* (0..shown); map
/// it through `View::rows` to get the real `Drawer::items` index.
struct RowLayout {
    body: Rect,
    dismiss: Rect,
    index: usize,
}

struct DrawerLayout {
    /// Whole panel rect (window coords).
    panel: Rect,
    rows: Vec<RowLayout>,
    /// Window-space top y of the panel (negative while sliding in).
    top_y: f32,
    panel_h: f32,
    width: f32,
    left: f32,
}

/// The per-project slice of the drawer for the current active project.
/// `rows` are *real* indices into `Drawer::items`, already capped to
/// `MAX_VISIBLE_ROWS`. `overflow` is how many matching items were cut by
/// that cap; `elsewhere` is how many belong to *other* live projects
/// (hidden here, surfaced as a header note so nothing seems lost).
struct View {
    rows: Vec<usize>,
    overflow: usize,
    elsewhere: usize,
}

/// An item shows in the active project's drawer when it's unscoped
/// (`None`), scoped to the active project, or an orphan whose project no
/// longer exists (treated as global so it never vanishes silently).
fn project_view(items: &[Suggestion], projects: &Projects) -> View {
    let active = projects.active;
    let exists = |id: u64| projects.list.iter().any(|p| p.id == id);
    let mut rows = Vec::new();
    let mut elsewhere = 0;
    for (i, s) in items.iter().enumerate() {
        let show = match s.project_id {
            None => true,
            Some(id) if Some(id) == active => true,
            Some(id) if !exists(id) => true,
            Some(_) => {
                elsewhere += 1;
                false
            }
        };
        if show {
            rows.push(i);
        }
    }
    let overflow = rows.len().saturating_sub(MAX_VISIBLE_ROWS);
    rows.truncate(MAX_VISIBLE_ROWS);
    View {
        rows,
        overflow,
        elsewhere,
    }
}

/// Pure function of window size, sidebar width, *shown* row count and
/// slide progress — shared by render and hit-testing so they never
/// desync. `shown` is already project-filtered and capped.
fn compute_layout(
    win_w: f32,
    sidebar_width: f32,
    shown: usize,
    overflow: bool,
    anim: f32,
) -> DrawerLayout {
    let left = sidebar_width;
    let width = (win_w - left).max(0.0);
    let content_h = if shown == 0 {
        44.0 // empty-state line
    } else {
        shown as f32 * ROW_H + if overflow { 22.0 } else { 0.0 }
    };
    let panel_h = HEADER_H + content_h + PAD * 2.0;
    // Slide: anim 0 → fully above the top (top_y = -panel_h), 1 → docked.
    let top_y = -(1.0 - anim) * panel_h;

    let mut rows = Vec::with_capacity(shown);
    let rows_top = top_y + HEADER_H + PAD;
    for i in 0..shown {
        let y0 = rows_top + i as f32 * ROW_H;
        let y1 = y0 + ROW_H;
        let body = Rect::new(left + PAD, y0, left + width - PAD - DISMISS_W, y1);
        let dismiss = Rect::new(left + width - PAD - DISMISS_W, y0, left + width - PAD, y1);
        rows.push(RowLayout {
            body,
            dismiss,
            index: i,
        });
    }

    DrawerLayout {
        panel: Rect::new(left, top_y, left + width, top_y + panel_h),
        rows,
        top_y,
        panel_h,
        width,
        left,
    }
}

// ---------- Input ----------

fn drawer_input(
    mut keys: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    sidebar: Res<Sidebar>,
    projects: Res<Projects>,
    registry: Res<PaneRegistry>,
    mut drawer: ResMut<Drawer>,
    mut pending: ResMut<PendingActions>,
    mut consumed: ResMut<InputConsumed>,
) {
    let cmd = mods.pressed(KeyCode::SuperLeft) || mods.pressed(KeyCode::SuperRight);
    let mut toggle = false;
    let mut esc = false;
    for ev in keys.read() {
        if !ev.state.is_pressed() {
            continue;
        }
        match ev.key_code {
            // Cmd+J toggles. (The Quake-authentic Cmd+` is swallowed by
            // macOS for window cycling and never reaches winit.)
            KeyCode::KeyJ if cmd => toggle = true,
            KeyCode::Escape => esc = true,
            _ => {}
        }
    }
    if toggle {
        drawer.open = !drawer.open;
        drawer.hovered = None;
    } else if esc && drawer.open {
        drawer.open = false;
        drawer.hovered = None;
    }

    // Clicks only matter while the panel is actually down.
    if !drawer.open || !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Some(pt) = window.cursor_position() else { return };

    let view = project_view(&drawer.items, &projects);
    let layout = compute_layout(
        window.width(),
        sidebar.width,
        view.rows.len(),
        view.overflow > 0,
        drawer.anim,
    );

    // Click outside the docked panel closes it (and is left for whatever
    // is underneath to handle).
    if !layout.panel.contains(pt) {
        drawer.open = false;
        drawer.hovered = None;
        return;
    }

    // Inside the panel — this click is ours, don't let it fall through to
    // the canvas / a pane behind the panel.
    consumed.0 = true;

    for row in &layout.rows {
        let Some(&real) = view.rows.get(row.index) else {
            continue;
        };
        if row.dismiss.contains(pt) {
            if real < drawer.items.len() {
                drawer.items.remove(real);
                drawer.dirty = true;
                drawer.hovered = None;
            }
            return;
        }
        if row.body.contains(pt) {
            materialize(real, &mut drawer, &projects, &registry, &mut pending);
            return;
        }
    }
}

/// Pull suggestion `index` onto the canvas and remove it from the drawer.
fn materialize(
    index: usize,
    drawer: &mut Drawer,
    projects: &Projects,
    registry: &PaneRegistry,
    pending: &mut PendingActions,
) {
    if index >= drawer.items.len() {
        return;
    }
    let suggestion = &drawer.items[index];

    // Spawn into the suggestion's project if it still exists; otherwise
    // (unscoped, or its project was deleted) fall back to the active one.
    let project_id = suggestion
        .project_id
        .filter(|id| projects.list.iter().any(|p| p.id == *id))
        .or(projects.active);
    let Some(project_id) = project_id else {
        eprintln!(
            "[drawer] no project to materialize {:?} into; leaving it parked",
            suggestion.title
        );
        return;
    };

    if registry.get(&suggestion.kind).is_none() {
        // Hard, visible failure instead of a silent no-op: the kind must
        // be registered for spawn to do anything.
        eprintln!(
            "[drawer] unknown pane kind {:?}; can't materialize {:?}",
            suggestion.kind, suggestion.title
        );
        return;
    }

    let suggestion = drawer.items.remove(index);
    drawer.dirty = true;
    drawer.hovered = None;
    pending.new_panes.push(NewPaneRequest {
        kind: kind_to_static(&suggestion.kind),
        project_id,
        origin: None, // cascade into the project like any new pane
        size: None,
        config: suggestion.config,
    });
}

fn drawer_hover(
    windows: Query<&Window>,
    sidebar: Res<Sidebar>,
    projects: Res<Projects>,
    mut drawer: ResMut<Drawer>,
) {
    if !drawer.open {
        if drawer.hovered.is_some() {
            drawer.hovered = None;
        }
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Some(pt) = window.cursor_position() else {
        if drawer.hovered.is_some() {
            drawer.hovered = None;
        }
        return;
    };
    let view = project_view(&drawer.items, &projects);
    let layout = compute_layout(
        window.width(),
        sidebar.width,
        view.rows.len(),
        view.overflow > 0,
        drawer.anim,
    );
    let mut new_hover = None;
    for row in &layout.rows {
        if row.body.contains(pt) || row.dismiss.contains(pt) {
            new_hover = Some(row.index);
            break;
        }
    }
    if drawer.hovered != new_hover {
        drawer.hovered = new_hover;
    }
}

fn drawer_animate(time: Res<Time>, mut drawer: ResMut<Drawer>) {
    let target = if drawer.open { 1.0 } else { 0.0 };
    if drawer.anim == target {
        return;
    }
    let step = time.delta_secs() / SLIDE_SECS;
    if drawer.anim < target {
        drawer.anim = (drawer.anim + step).min(target);
    } else {
        drawer.anim = (drawer.anim - step).max(target);
    }
}

// ---------- Render ----------

#[derive(Component)]
struct DrawerEntity;

/// Ease-out so the panel decelerates as it docks.
fn ease_out_cubic(t: f32) -> f32 {
    let u = 1.0 - t;
    1.0 - u * u * u
}

fn truncate_to_width(s: &str, font_size: f32, max_w: f32) -> String {
    let max_chars = (max_w / (font_size * CHAR_W_RATIO)).floor() as usize;
    if max_chars == 0 {
        return String::new();
    }
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let keep = max_chars.saturating_sub(1).max(1);
    let mut out: String = s.chars().take(keep).collect();
    out.push('…');
    out
}

fn drawer_render(
    mut commands: Commands,
    drawer: Res<Drawer>,
    windows: Query<&Window>,
    sidebar: Res<Sidebar>,
    projects: Res<Projects>,
    registry: Res<PaneRegistry>,
    font: Res<MonoFont>,
    theme: Res<style_bevy::Theme>,
    existing: Query<Entity, With<DrawerEntity>>,
    mut last_overflow_logged: Local<usize>,
) {
    let Ok(window) = windows.single() else { return };

    // Nothing to draw when fully retracted.
    if !drawer.open && drawer.anim <= 0.0 {
        for e in &existing {
            commands.entity(e).despawn();
        }
        return;
    }

    // Rebuild every frame while the panel is active. It's a dozen-ish
    // sprites/texts; cheaper than diffing and always correct mid-slide.
    for e in &existing {
        commands.entity(e).despawn();
    }

    let win_w = window.width();
    let win_h = window.height();
    let overlay = RenderLayers::layer(crate::MENU_OVERLAY_LAYER);

    // Palette (re-resolved per frame; a handful of hash lookups).
    use style_bevy::tokens;
    let panel_bg = Color::LinearRgba(theme.color(tokens::PANE_BG));
    let border = Color::LinearRgba(theme.color(tokens::PANE_BORDER));
    let fg = Color::LinearRgba(theme.color(tokens::FG));
    let fg_muted = Color::LinearRgba(theme.color(tokens::FG_MUTED));
    let accent = Color::LinearRgba(theme.color(tokens::ACCENT));
    let row_hover = Color::LinearRgba(theme.color(tokens::SELECTION));

    let view = project_view(&drawer.items, &projects);
    let eased = ease_out_cubic(drawer.anim.clamp(0.0, 1.0));
    let layout = compute_layout(
        win_w,
        sidebar.width,
        view.rows.len(),
        view.overflow > 0,
        eased,
    );

    // Window point → world translation (y flips). Mirrors radial.rs.
    let w2w = |x: f32, y: f32| Vec2::new(x - win_w * 0.5, win_h * 0.5 - y);

    // Panel body.
    let tl = w2w(layout.left, layout.top_y);
    commands.spawn((
        DrawerEntity,
        Sprite {
            color: panel_bg,
            custom_size: Some(Vec2::new(layout.width, layout.panel_h)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(tl.x, tl.y, DRAWER_Z),
        overlay.clone(),
    ));
    // Bottom border hairline so the panel reads as a separate surface.
    let bl = w2w(layout.left, layout.top_y + layout.panel_h);
    commands.spawn((
        DrawerEntity,
        Sprite {
            color: border,
            custom_size: Some(Vec2::new(layout.width, 1.5)),
            ..default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(bl.x, bl.y, DRAWER_Z + 0.05),
        overlay.clone(),
    ));

    // Header. Scope it to the active project so it's obvious the list is
    // per-project; the "+N elsewhere" note tells you items exist in other
    // projects (so a suggestion never seems to have vanished).
    let proj_name = projects
        .active
        .and_then(|id| projects.list.iter().find(|p| p.id == id))
        .map(|p| p.name.as_str());
    let header_text = match (proj_name, view.elsewhere) {
        (Some(name), 0) => format!("SUGGESTIONS · {} · {}", name, view.rows.len()),
        (Some(name), n) => format!("SUGGESTIONS · {} · {} (+{} elsewhere)", name, view.rows.len(), n),
        (None, 0) => format!("SUGGESTIONS · {}", view.rows.len()),
        (None, n) => format!("SUGGESTIONS · {} (+{} elsewhere)", view.rows.len(), n),
    };
    let header_p = w2w(layout.left + PAD, layout.top_y + HEADER_H * 0.5);
    commands.spawn((
        DrawerEntity,
        Text2d::new(header_text),
        TextFont {
            font: font.0.clone(),
            font_size: 12.0,
            ..default()
        },
        LineHeight::Px(12.0),
        TextColor(accent),
        Anchor::CENTER_LEFT,
        Transform::from_xyz(header_p.x, header_p.y, DRAWER_Z + 0.2),
        overlay.clone(),
    ));
    let hint = "cmd+J  ·  esc to close";
    let hint_p = w2w(layout.left + layout.width - PAD, layout.top_y + HEADER_H * 0.5);
    commands.spawn((
        DrawerEntity,
        Text2d::new(hint),
        TextFont {
            font: font.0.clone(),
            font_size: 10.0,
            ..default()
        },
        LineHeight::Px(10.0),
        TextColor(fg_muted),
        Anchor::CENTER_RIGHT,
        Transform::from_xyz(hint_p.x, hint_p.y, DRAWER_Z + 0.2),
        overlay.clone(),
    ));

    // Empty state — distinguish "drawer truly empty" from "nothing for
    // this project, but items live elsewhere".
    if view.rows.is_empty() {
        let msg = if view.elsewhere > 0 {
            "nothing parked for this project"
        } else {
            "nothing parked here yet"
        };
        let p = w2w(layout.left + layout.width * 0.5, layout.top_y + HEADER_H + PAD + 20.0);
        commands.spawn((
            DrawerEntity,
            Text2d::new(msg),
            TextFont {
                font: font.0.clone(),
                font_size: 12.0,
                ..default()
            },
            LineHeight::Px(12.0),
            TextColor(fg_muted),
            Anchor::CENTER,
            Transform::from_xyz(p.x, p.y, DRAWER_Z + 0.2),
            overlay.clone(),
        ));
        return;
    }

    // Rows. `row.index` is the visible position; map it to the real
    // items index through the filtered view.
    let text_w = layout.width - PAD * 2.0 - DISMISS_W - 34.0; // minus icon gutter
    for row in &layout.rows {
        let Some(&real) = view.rows.get(row.index) else {
            continue;
        };
        let item = &drawer.items[real];
        let hovered = drawer.hovered == Some(row.index);
        let row_h = row.body.height();
        let row_w = row.dismiss.max.x - row.body.min.x;

        // Row background (only when hovered — keeps the list quiet).
        if hovered {
            let rp = w2w(row.body.min.x, row.body.min.y);
            commands.spawn((
                DrawerEntity,
                Sprite {
                    color: row_hover,
                    custom_size: Some(Vec2::new(row_w, row_h - 2.0)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(rp.x, rp.y, DRAWER_Z + 0.1),
                overlay.clone(),
            ));
        }

        // Kind icon (reuse the registry's radial glyph).
        let icon = registry
            .get(&item.kind)
            .and_then(|s| s.radial_icon)
            .unwrap_or("◆");
        let icon_p = w2w(row.body.min.x + 14.0, row.body.min.y + row_h * 0.5);
        commands.spawn((
            DrawerEntity,
            Text2d::new(icon),
            TextFont {
                font: font.0.clone(),
                font_size: 15.0,
                ..default()
            },
            LineHeight::Px(15.0),
            TextColor(accent),
            Anchor::CENTER,
            Transform::from_xyz(icon_p.x, icon_p.y, DRAWER_Z + 0.2),
            overlay.clone(),
        ));

        let text_x = row.body.min.x + 34.0;
        let has_reason = item.reason.as_deref().map_or(false, |r| !r.is_empty());
        // Title (vertically centered if no reason, else upper line).
        let title_y = if has_reason {
            row.body.min.y + row_h * 0.5 - 9.0
        } else {
            row.body.min.y + row_h * 0.5
        };
        let title_p = w2w(text_x, title_y);
        commands.spawn((
            DrawerEntity,
            Text2d::new(truncate_to_width(&item.title, 13.0, text_w)),
            TextFont {
                font: font.0.clone(),
                font_size: 13.0,
                ..default()
            },
            LineHeight::Px(13.0),
            TextColor(fg),
            Anchor::CENTER_LEFT,
            Transform::from_xyz(title_p.x, title_p.y, DRAWER_Z + 0.2),
            overlay.clone(),
        ));
        if let Some(reason) = item.reason.as_deref().filter(|r| !r.is_empty()) {
            let reason_p = w2w(text_x, row.body.min.y + row_h * 0.5 + 9.0);
            commands.spawn((
                DrawerEntity,
                Text2d::new(truncate_to_width(reason, 11.0, text_w)),
                TextFont {
                    font: font.0.clone(),
                    font_size: 11.0,
                    ..default()
                },
                LineHeight::Px(11.0),
                TextColor(fg_muted),
                Anchor::CENTER_LEFT,
                Transform::from_xyz(reason_p.x, reason_p.y, DRAWER_Z + 0.2),
                overlay.clone(),
            ));
        }

        // Dismiss ×.
        let x_p = w2w(
            (row.dismiss.min.x + row.dismiss.max.x) * 0.5,
            row.dismiss.min.y + row_h * 0.5,
        );
        commands.spawn((
            DrawerEntity,
            Text2d::new("×"),
            TextFont {
                font: font.0.clone(),
                font_size: 16.0,
                ..default()
            },
            LineHeight::Px(16.0),
            TextColor(if hovered { fg } else { fg_muted }),
            Anchor::CENTER,
            Transform::from_xyz(x_p.x, x_p.y, DRAWER_Z + 0.2),
            overlay.clone(),
        ));
    }

    // Overflow footer (never silently drop rows). Counts only this
    // project's items past the visible cap, not other-project ones.
    if view.overflow > 0 {
        let hidden = view.overflow;
        if *last_overflow_logged != hidden {
            bevy::log::warn!(
                "[drawer] {} suggestion(s) past the visible {} not shown (no scroll yet)",
                hidden,
                MAX_VISIBLE_ROWS
            );
            *last_overflow_logged = hidden;
        }
        let foot_y = layout.top_y + layout.panel_h - PAD - 11.0;
        let p = w2w(layout.left + layout.width * 0.5, foot_y);
        commands.spawn((
            DrawerEntity,
            Text2d::new(format!("+{} more (dismiss some to see them)", hidden)),
            TextFont {
                font: font.0.clone(),
                font_size: 11.0,
                ..default()
            },
            LineHeight::Px(11.0),
            TextColor(fg_muted),
            Anchor::CENTER,
            Transform::from_xyz(p.x, p.y, DRAWER_Z + 0.2),
            overlay.clone(),
        ));
    } else if *last_overflow_logged != 0 {
        *last_overflow_logged = 0;
    }
}
