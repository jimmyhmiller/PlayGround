//! Editor view layer built on Bevy's 2D/text pipeline.
//!
//! Structure:
//! - `EditorPlugin` wires resources and systems into an `App`.
//! - Pure helpers (`caret_local`, `char_to_line_byte`, `mouse_to_char_local`)
//!   are exposed so they can be unit-tested without starting Bevy.
//! - `build_app(initial)` constructs a full `App` with `DefaultPlugins` +
//!   our plugin — used by the binary. Tests can build their own with
//!   `MinimalPlugins`-style setups and just add `EditorPlugin`.

use std::collections::HashMap;

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::{CosmicFontSystem, LineHeight, PositionedGlyph, TextLayoutInfo};
use editor_core::commands::{
    cursor_char_left, cursor_char_right, cursor_doc_end, cursor_doc_start, cursor_line_down,
    cursor_line_end, cursor_line_start, cursor_line_up, cursor_word_left, cursor_word_right,
    delete_char_backward, delete_char_forward, indent_more, insert_newline_and_indent, select_all,
    select_char_left, select_char_right, select_doc_end, select_doc_start, select_line_down,
    select_line_end, select_line_start, select_line_up, select_word_left, select_word_right,
};
use editor_core::history::{redo, undo};
use editor_core::selection::{Range, Selection};
use editor_core::state::EditorState;
use editor_core::transaction::{Change, Transaction};

pub const FONT_SIZE: f32 = 16.0;
pub const LINE_HEIGHT: f32 = 20.0;
pub const MARGIN: f32 = 16.0;

#[derive(Resource)]
pub struct EditorRes(pub EditorState);

/// Top-level render entities for the editor. `lines` is a pool of
/// `Text2d` entities — one per visible doc line, indexed by line
/// number. We own the vertical positioning (`y = line * LINE_HEIGHT`)
/// because Bevy's `Text2d` layout doesn't apply the per-span line
/// height to empty lines, so a single shared buffer collapses blank
/// lines to the buffer's unscaled default (20px even on 2x HiDPI).
/// With one `Text2d` per line there is no layout interaction between
/// lines at all.
#[derive(Resource)]
pub struct Doc {
    pub root: Entity,
    pub caret: Entity,
}

/// Pool of line-row entities, keyed by doc line number. Only lines
/// currently in (or near) the viewport have entries — everything else
/// is despawned so the per-frame work is `O(viewport)` not `O(doc)`.
/// A missing key means either the line is empty, scrolled off, or
/// past the doc end; callers that care about the difference consult
/// the rope directly.
#[derive(Resource, Default)]
pub struct LineRows(pub HashMap<usize, Entity>);

/// Lines of overdraw kept on either side of the visible window so that
/// small scroll deltas don't trigger a spawn/despawn every frame.
const VIEWPORT_OVERDRAW: usize = 8;

/// Handle to the editor font, cached so `sync_text` doesn't re-load
/// from disk when spawning new line entities.
#[derive(Resource)]
pub struct MonoFont(pub Handle<Font>);

#[derive(Resource, Default)]
pub struct Scroll(pub f32);

#[derive(Component)]
pub struct SelRect;

#[derive(Resource, Default)]
pub struct MouseDrag {
    pub anchor: Option<usize>,
}

/// Adds the editor's resources (except `EditorRes`, which callers
/// insert themselves) and the full Update-schedule system chain.
/// The caller is responsible for choosing plugins — this plugin does
/// not touch `DefaultPlugins`, so tests can skip the window layer.
pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(MouseDrag::default())
            .insert_resource(Scroll::default())
            .insert_resource(LineRows::default())
            .insert_resource(ClearColor(Color::srgb(0.10, 0.11, 0.13)))
            .add_systems(Startup, (setup, load_fallback_fonts))
            .add_systems(
                Update,
                (
                    handle_input,
                    handle_scroll,
                    handle_mouse,
                    sync_text,
                    position_root,
                    sync_caret,
                    sync_selection,
                )
                    .chain(),
            );
    }
}

/// Minimal plugin for headless state-transition tests: only the
/// resources + `handle_input` system. No rendering, no window, no
/// asset loading. `handle_mouse`, `handle_scroll`, and the sync
/// systems are omitted because they'd bail out on missing resources
/// anyway — registering them adds noise without catching bugs.
pub struct HeadlessEditorPlugin;

impl Plugin for HeadlessEditorPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(MouseDrag::default())
            .insert_resource(Scroll::default())
            .add_systems(Update, handle_input);
    }
}

/// Build a ready-to-run App with the real window and default plugins.
/// Binaries use this; tests generally don't.
pub fn build_app(initial: &str) -> App {
    let mut app = App::new();
    app.insert_resource(EditorRes(
        EditorState::new(ropey::Rope::from_str(initial), Selection::cursor(0))
            .with_indent_unit("    "),
    ));
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "editor".into(),
            resolution: (900u32, 600u32).into(),
            ..default()
        }),
        ..default()
    }));
    app.add_plugins(EditorPlugin);
    app
}

/// Bevy initializes `CosmicFontSystem` with an empty `fontdb`, so any
/// glyph not present in the primary editor font (JetBrains Mono —
/// Latin/Greek only) renders as tofu. Populating the db with the
/// user's installed system fonts lets cosmic-text fall back for CJK,
/// emoji, and other scripts automatically.
fn load_fallback_fonts(mut fonts: ResMut<CosmicFontSystem>) {
    fonts.0.db_mut().load_system_fonts();
}

/// JetBrains Mono Regular bundled into the binary so the release exe
/// runs from any cwd. Using `include_bytes!` (not `AssetServer::load`)
/// sidesteps Bevy's cwd-relative asset resolution, which fails for
/// `./target/release/editor` invoked outside the crate dir.
const EMBEDDED_FONT: &[u8] =
    include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf");

fn setup(mut commands: Commands, mut fonts: ResMut<Assets<Font>>) {
    commands.spawn(Camera2d);
    let font = fonts.add(
        Font::try_from_bytes(EMBEDDED_FONT.to_vec())
            .expect("embedded JetBrainsMono-Regular.ttf must parse"),
    );

    let root = commands
        .spawn((Transform::default(), Visibility::default()))
        .id();

    let caret = commands
        .spawn((
            ChildOf(root),
            Sprite {
                color: Color::srgb(0.55, 0.85, 1.0),
                custom_size: Some(Vec2::new(2.0, LINE_HEIGHT)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 1.0),
        ))
        .id();

    commands.insert_resource(MonoFont(font));
    commands.insert_resource(Doc { root, caret });
}

fn line_text(doc: &ropey::Rope, idx: usize) -> String {
    let s = doc.line(idx).to_string();
    s.strip_suffix('\n').map(str::to_string).unwrap_or(s)
}

/// Reconcile the line-row pool with the document. One `Text2d` per
/// non-empty line, positioned at `y = idx * LINE_HEIGHT` in doc-local
/// coords. Empty lines get a `None` slot — they contribute no glyphs,
/// so there's nothing to draw, but the index still advances the grid.
///
/// This keeps vertical positioning under our control: Bevy never
/// stacks multiple lines in a single buffer, so its buggy empty-line
/// line-height fallback (where HiDPI span metrics are scaled but the
/// buffer default is not) has no effect.
fn sync_text(
    state: Res<EditorRes>,
    doc: Res<Doc>,
    font: Res<MonoFont>,
    scroll: Res<Scroll>,
    windows: Query<&Window>,
    mut pool: ResMut<LineRows>,
    mut text_q: Query<&mut Text2d>,
    mut commands: Commands,
) {
    let rope = &state.0.doc;
    let effective = effective_line_count(rope);
    let (first, last) = viewport_line_range(windows.single().ok(), scroll.0, effective);

    // Despawn entities that are out of the doc or out of the viewport
    // window. `retain` iterates only existing pool entries — O(entities
    // currently alive), which is bounded by `viewport + 2 * overdraw`.
    pool.0.retain(|&idx, entity| {
        let keep = idx < effective && idx >= first && idx <= last;
        if !keep {
            commands.entity(*entity).despawn();
        }
        keep
    });

    // Spawn or refresh the entities that belong in the viewport.
    for idx in first..=last.min(effective.saturating_sub(1)) {
        let wanted = line_text(rope, idx);
        if wanted.is_empty() {
            // Empty line — no glyphs to draw; drop any stale entity.
            if let Some(entity) = pool.0.remove(&idx) {
                commands.entity(entity).despawn();
            }
            continue;
        }
        match pool.0.get(&idx) {
            Some(&e) => {
                if let Ok(mut t) = text_q.get_mut(e) {
                    if t.0 != wanted {
                        t.0 = wanted;
                    }
                }
            }
            None => {
                let entity = commands
                    .spawn((
                        ChildOf(doc.root),
                        Text2d::new(wanted),
                        TextFont {
                            font: font.0.clone(),
                            font_size: FONT_SIZE,
                            ..default()
                        },
                        LineHeight::Px(LINE_HEIGHT),
                        TextColor(Color::srgb(0.92, 0.92, 0.94)),
                        Anchor::TOP_LEFT,
                        Transform::from_xyz(0.0, -(idx as f32) * LINE_HEIGHT, 0.0),
                    ))
                    .id();
                pool.0.insert(idx, entity);
            }
        }
    }
}

fn effective_line_count(rope: &ropey::Rope) -> usize {
    let n = rope.len_lines();
    if n == 0 {
        1
    } else if n > 1 && rope.line(n - 1).len_chars() == 0 {
        // Trailing-newline phantom line: drop the empty slot past EOF.
        n - 1
    } else {
        n
    }
}

fn viewport_line_range(
    window: Option<&Window>,
    scroll: f32,
    effective_lines: usize,
) -> (usize, usize) {
    let height = window.map(|w| w.height()).unwrap_or(600.0);
    let top = (scroll - MARGIN).max(0.0);
    let bottom = (scroll + height + MARGIN).max(0.0);
    let first_f = (top / LINE_HEIGHT).floor();
    let last_f = (bottom / LINE_HEIGHT).ceil();
    let mut first = first_f.max(0.0) as usize;
    let mut last = last_f.max(0.0) as usize;
    first = first.saturating_sub(VIEWPORT_OVERDRAW);
    last = last.saturating_add(VIEWPORT_OVERDRAW);
    if effective_lines > 0 {
        last = last.min(effective_lines - 1);
    } else {
        last = 0;
    }
    (first, last)
}

fn position_root(
    windows: Query<&Window>,
    doc: Res<Doc>,
    scroll: Res<Scroll>,
    mut t_q: Query<&mut Transform>,
) {
    let Ok(win) = windows.single() else { return };
    let Ok(mut t) = t_q.get_mut(doc.root) else { return };
    t.translation.x = -win.width() * 0.5 + MARGIN;
    t.translation.y = win.height() * 0.5 - MARGIN + scroll.0;
}

fn sync_caret(
    state: Res<EditorRes>,
    doc: Res<Doc>,
    pool: Res<LineRows>,
    layout_q: Query<&TextLayoutInfo>,
    mut t_q: Query<&mut Transform>,
) {
    let head = state.0.selection.primary_range().head;
    let (line, byte_in_line) = char_to_line_byte(&state.0.doc, head);

    let x = line_entity(&pool, line)
        .and_then(|e| layout_q.get(e).ok())
        .map(|layout| caret_x_in_line(&layout.glyphs, layout.scale_factor, byte_in_line))
        .unwrap_or(0.0);
    let y = line as f32 * LINE_HEIGHT;

    if let Ok(mut t) = t_q.get_mut(doc.caret) {
        t.translation.x = x;
        t.translation.y = -y;
        t.translation.z = 1.0;
    }
}

/// Entity for a given doc line, if the pool has one.
fn line_entity(pool: &LineRows, line: usize) -> Option<Entity> {
    pool.0.get(&line).copied()
}

/// Doc-local caret X within a single line's `TextLayoutInfo.glyphs`.
/// Every glyph in these inputs belongs to line 0 of its own `Text2d`,
/// so we don't filter by `line_index` here — only by `byte_index`.
/// Y is not part of the result because callers own it (line number
/// times `LINE_HEIGHT`, per our grid).
pub fn caret_x_in_line(
    glyphs: &[PositionedGlyph],
    scale_factor: f32,
    byte_in_line: usize,
) -> f32 {
    let sf = if scale_factor > 0.0 { scale_factor } else { 1.0 };
    let mut last: Option<&PositionedGlyph> = None;
    for g in glyphs {
        if byte_in_line <= g.byte_index {
            return (g.position.x - g.size.x * 0.5) / sf;
        }
        last = Some(g);
    }
    if let Some(g) = last {
        (g.position.x + g.size.x * 0.5) / sf
    } else {
        0.0
    }
}

/// Convert a rope char offset to (line_index, byte_in_line). Exposed
/// for unit tests.
pub fn char_to_line_byte(doc: &ropey::Rope, char_idx: usize) -> (usize, usize) {
    let line = doc.char_to_line(char_idx);
    let line_start = doc.line_to_char(line);
    let chars_in = char_idx - line_start;
    let line_slice = doc.line(line);
    let byte_in_line = line_slice.char_to_byte(chars_in);
    (line, byte_in_line)
}

fn sync_selection(
    state: Res<EditorRes>,
    doc: Res<Doc>,
    pool: Res<LineRows>,
    layout_q: Query<&TextLayoutInfo>,
    existing: Query<Entity, With<SelRect>>,
    mut commands: Commands,
) {
    for e in &existing {
        commands.entity(e).despawn();
    }

    let range = state.0.selection.primary_range();
    let (from, to) = (range.from(), range.to());
    if from == to {
        return;
    }

    let (start_line, start_byte) = char_to_line_byte(&state.0.doc, from);
    let (end_line, end_byte) = char_to_line_byte(&state.0.doc, to);

    for line in start_line..=end_line {
        let lo = if line == start_line { start_byte } else { 0 };
        let hi = if line == end_line { end_byte } else { usize::MAX };

        let span = line_entity(&pool, line)
            .and_then(|e| layout_q.get(e).ok())
            .and_then(|layout| line_selection_span(&layout.glyphs, layout.scale_factor, lo, hi));

        let ends_mid_doc = line < end_line;
        let (x0, x1) = match span {
            Some((a, b)) => {
                let extra = if ends_mid_doc { LINE_HEIGHT * 0.3 } else { 0.0 };
                (a, b + extra)
            }
            None if ends_mid_doc => (0.0, LINE_HEIGHT * 0.3),
            None => continue,
        };

        let y_top = line as f32 * LINE_HEIGHT;
        commands.spawn((
            SelRect,
            ChildOf(doc.root),
            Sprite {
                color: Color::srgba(0.35, 0.55, 0.9, 0.35),
                custom_size: Some(Vec2::new((x1 - x0).max(1.0), LINE_HEIGHT)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(x0, -y_top, 0.5),
        ));
    }
}

/// X-range `[min, max]` of glyphs whose byte range intersects `[lo, hi)`
/// on a single line. Returns `None` if no glyph overlaps (e.g. the
/// selection covers only a newline).
pub fn line_selection_span(
    glyphs: &[PositionedGlyph],
    scale_factor: f32,
    lo: usize,
    hi: usize,
) -> Option<(f32, f32)> {
    let sf = if scale_factor > 0.0 { scale_factor } else { 1.0 };
    let mut x_min: Option<f32> = None;
    let mut x_max: Option<f32> = None;
    for g in glyphs {
        let g_start = g.byte_index;
        let g_end = g.byte_index + g.byte_length;
        if g_end <= lo || g_start >= hi {
            continue;
        }
        let gl = (g.position.x - g.size.x * 0.5) / sf;
        let gr = (g.position.x + g.size.x * 0.5) / sf;
        x_min = Some(x_min.map_or(gl, |v| v.min(gl)));
        x_max = Some(x_max.map_or(gr, |v| v.max(gr)));
    }
    match (x_min, x_max) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    }
}

fn handle_scroll(
    mut wheel: MessageReader<MouseWheel>,
    mut scroll: ResMut<Scroll>,
    state: Res<EditorRes>,
    windows: Query<&Window>,
) {
    // Trackpad events come in pixels (use delta directly); mouse-wheel
    // events come in notches of `Line` (convert 1 notch = 1 line height).
    let mut delta_px = 0.0;
    for ev in wheel.read() {
        delta_px += match ev.unit {
            MouseScrollUnit::Pixel => ev.y,
            MouseScrollUnit::Line => ev.y * LINE_HEIGHT,
        };
    }
    if delta_px == 0.0 {
        return;
    }
    scroll.0 = (scroll.0 - delta_px).max(0.0);

    let Ok(win) = windows.single() else { return };
    let doc_height = state.0.doc.len_lines() as f32 * LINE_HEIGHT;
    let viewport = (win.height() - 2.0 * MARGIN).max(0.0);
    let max = (doc_height - viewport).max(0.0);
    if scroll.0 > max {
        scroll.0 = max;
    }
}

fn handle_input(
    mut keys: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<EditorRes>,
) {
    let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);
    let ctrl = mods.pressed(KeyCode::ControlLeft) || mods.pressed(KeyCode::ControlRight);
    let alt = mods.pressed(KeyCode::AltLeft) || mods.pressed(KeyCode::AltRight);
    let meta = mods.pressed(KeyCode::SuperLeft) || mods.pressed(KeyCode::SuperRight);
    let mod_word = alt || ctrl;
    let mod_doc = meta || ctrl;

    for ev in keys.read() {
        if !ev.state.is_pressed() {
            continue;
        }

        let cmd_result = match ev.key_code {
            KeyCode::ArrowLeft => Some(if shift {
                if mod_word {
                    run(&state.0, select_word_left)
                } else {
                    run(&state.0, select_char_left)
                }
            } else if mod_word {
                run(&state.0, cursor_word_left)
            } else {
                run(&state.0, cursor_char_left)
            }),
            KeyCode::ArrowRight => Some(if shift {
                if mod_word {
                    run(&state.0, select_word_right)
                } else {
                    run(&state.0, select_char_right)
                }
            } else if mod_word {
                run(&state.0, cursor_word_right)
            } else {
                run(&state.0, cursor_char_right)
            }),
            KeyCode::ArrowUp => Some(if shift {
                run(&state.0, select_line_up)
            } else {
                run(&state.0, cursor_line_up)
            }),
            KeyCode::ArrowDown => Some(if shift {
                run(&state.0, select_line_down)
            } else {
                run(&state.0, cursor_line_down)
            }),
            KeyCode::Home => Some(if shift {
                if mod_doc {
                    run(&state.0, select_doc_start)
                } else {
                    run(&state.0, select_line_start)
                }
            } else if mod_doc {
                run(&state.0, cursor_doc_start)
            } else {
                run(&state.0, cursor_line_start)
            }),
            KeyCode::End => Some(if shift {
                if mod_doc {
                    run(&state.0, select_doc_end)
                } else {
                    run(&state.0, select_line_end)
                }
            } else if mod_doc {
                run(&state.0, cursor_doc_end)
            } else {
                run(&state.0, cursor_line_end)
            }),
            KeyCode::Backspace => Some(run_history(&state.0, delete_char_backward)),
            KeyCode::Delete => Some(run_history(&state.0, delete_char_forward)),
            KeyCode::Enter | KeyCode::NumpadEnter => {
                Some(run_history(&state.0, insert_newline_and_indent))
            }
            KeyCode::Tab => Some(run_history(&state.0, indent_more)),
            KeyCode::KeyA if mod_doc => Some(run(&state.0, select_all)),
            KeyCode::KeyZ if mod_doc => Some(if shift {
                redo(&state.0).map(|new| (new, true))
            } else {
                undo(&state.0).map(|new| (new, true))
            }),
            KeyCode::KeyC if mod_doc => {
                copy_selection(&state.0);
                Some(None)
            }
            KeyCode::KeyX if mod_doc => {
                copy_selection(&state.0);
                Some(delete_selection(&state.0))
            }
            KeyCode::KeyV if mod_doc => Some(paste_from_clipboard(&state.0)),
            _ => None,
        };

        if let Some(Some((new_state, _))) = cmd_result {
            state.0 = new_state;
            continue;
        }
        if let Some(None) = cmd_result {
            continue;
        }

        if mod_doc || alt {
            continue;
        }
        let text: Option<String> = match &ev.logical_key {
            Key::Character(s) => Some(s.chars().take(1).collect()),
            Key::Space => Some(" ".into()),
            _ => None,
        };
        if let Some(text) = text.filter(|t| !t.is_empty()) {
            let tr = Transaction::new()
                .change(Change::new(
                    state.0.selection.primary_range().from(),
                    state.0.selection.primary_range().to(),
                    text.clone(),
                ))
                .select(Selection::cursor(
                    state.0.selection.primary_range().from() + text.chars().count(),
                ));
            state.0 = state.0.apply_with_history(&tr);
        }
    }
}

/// Pure hit-test: given a doc-local pointer position (logical pixels,
/// y-down, origin at doc top-left) and the target line's glyphs,
/// return the byte-in-line the caret should go to. Exposed for tests.
pub fn mouse_byte_in_line(
    local_x: f32,
    glyphs: &[PositionedGlyph],
    scale_factor: f32,
) -> usize {
    let sf = if scale_factor > 0.0 { scale_factor } else { 1.0 };
    let mut last: Option<&PositionedGlyph> = None;
    for g in glyphs {
        let midpoint = g.position.x / sf;
        if local_x < midpoint {
            return g.byte_index;
        }
        last = Some(g);
    }
    match last {
        Some(g) => g.byte_index + g.byte_length,
        None => 0,
    }
}

/// Resolve a line + byte-in-line to a rope char offset, clamped to
/// the line's end-of-line char. Extracted so tests can verify the
/// "don't spill onto the next line" clamp without building a full App.
pub fn char_from_line_byte(state: &EditorState, line: usize, byte_in_line: usize) -> usize {
    let n_lines = state.doc.len_lines().max(1);
    let last_line = n_lines - 1;
    let line = line.min(last_line);

    let line_slice = state.doc.line(line);
    let byte_clamped = byte_in_line.min(line_slice.len_bytes());
    let char_in_line = line_slice.byte_to_char(byte_clamped);
    let line_start = state.doc.line_to_char(line);
    let line_end = if line + 1 < state.doc.len_lines() {
        state.doc.line_to_char(line + 1).saturating_sub(1)
    } else {
        state.doc.len_chars()
    };
    (line_start + char_in_line).min(line_end)
}

fn handle_mouse(
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    mods: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<EditorRes>,
    mut drag: ResMut<MouseDrag>,
    scroll: Res<Scroll>,
    pool: Res<LineRows>,
    layout_q: Query<&TextLayoutInfo>,
) {
    let Ok(window) = windows.single() else { return };
    let Some(cursor_pos) = window.cursor_position() else {
        return;
    };

    let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);
    let local = Vec2::new(cursor_pos.x - MARGIN, cursor_pos.y - MARGIN + scroll.0);

    let resolve = |s: &EditorState| -> usize {
        let n_lines = s.doc.len_lines().max(1);
        let line = (local.y / LINE_HEIGHT)
            .floor()
            .max(0.0) as usize;
        let line = line.min(n_lines - 1);
        let byte_in_line = match line_entity(&pool, line) {
            Some(e) => layout_q
                .get(e)
                .map(|l| mouse_byte_in_line(local.x, &l.glyphs, l.scale_factor))
                .unwrap_or(0),
            None => 0,
        };
        char_from_line_byte(s, line, byte_in_line)
    };

    if buttons.just_pressed(MouseButton::Left) {
        let pos = resolve(&state.0);
        if shift {
            let anchor = state.0.selection.primary_range().anchor;
            drag.anchor = Some(anchor);
            state.0 = apply_selection(&state.0, anchor, pos);
        } else {
            drag.anchor = Some(pos);
            state.0 = apply_selection(&state.0, pos, pos);
        }
        return;
    }

    if buttons.just_released(MouseButton::Left) {
        drag.anchor = None;
        return;
    }

    if buttons.pressed(MouseButton::Left) {
        if let Some(anchor) = drag.anchor {
            let head = resolve(&state.0);
            let current = state.0.selection.primary_range();
            if current.anchor != anchor || current.head != head {
                state.0 = apply_selection(&state.0, anchor, head);
            }
        }
    }
}

fn apply_selection(state: &EditorState, anchor: usize, head: usize) -> EditorState {
    let tr = Transaction::new().select(Selection::single(Range::new(anchor, head)));
    state.apply_with_history(&tr)
}

/// Copy the primary selection's text to the OS clipboard. No-op if the
/// selection is empty or if the clipboard is unavailable (headless /
/// CI). Never touches editor state.
fn copy_selection(state: &EditorState) {
    let range = state.selection.primary_range();
    if range.from() == range.to() {
        return;
    }
    let text = state.doc.slice(range.from()..range.to()).to_string();
    if let Ok(mut cb) = arboard::Clipboard::new() {
        let _ = cb.set_text(text);
    }
}

/// Delete the primary selection. Returns `None` if there's nothing to
/// delete so the key is consumed without re-running history.
fn delete_selection(state: &EditorState) -> Option<(EditorState, bool)> {
    let range = state.selection.primary_range();
    if range.from() == range.to() {
        return None;
    }
    let tr = Transaction::new()
        .change(Change::new(range.from(), range.to(), String::new()))
        .select(Selection::cursor(range.from()));
    Some((state.apply_with_history(&tr), true))
}

/// Insert clipboard text at the cursor, replacing any selection.
/// Forces a new history group — pastes never coalesce with adjacent
/// typing, which matches user expectation and sidesteps the pending
/// undo-granularity gap documented in CM_PARITY.md.
fn paste_from_clipboard(state: &EditorState) -> Option<(EditorState, bool)> {
    let mut cb = arboard::Clipboard::new().ok()?;
    let text = cb.get_text().ok()?;
    if text.is_empty() {
        return None;
    }
    let range = state.selection.primary_range();
    let end = range.from() + text.chars().count();
    let tr = Transaction::new()
        .change(Change::new(range.from(), range.to(), text))
        .select(Selection::cursor(end));
    Some((state.apply_with_history_isolated(&tr), true))
}

fn run(
    state: &EditorState,
    cmd: fn(&EditorState) -> Option<Transaction>,
) -> Option<(EditorState, bool)> {
    cmd(state).map(|tr| (state.apply(&tr), true))
}

fn run_history(
    state: &EditorState,
    cmd: fn(&EditorState) -> Option<Transaction>,
) -> Option<(EditorState, bool)> {
    cmd(state).map(|tr| (state.apply_with_history(&tr), true))
}
