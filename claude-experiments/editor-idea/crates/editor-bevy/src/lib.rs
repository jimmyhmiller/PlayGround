//! Editor view layer built on Bevy's 2D/text pipeline.
//!
//! # Multi-editor architecture
//!
//! Each editor instance is a Bevy entity that carries its own state
//! components (`EditorStateComp`, `LineRows`, `EditorScroll`,
//! `EditorHighlighter`, `EditorRect`, `EditorChrome`, `TextDragAnchor`).
//! Systems iterate these entities in a `Query` instead of reading
//! singleton resources. This makes N editors in one window trivial —
//! they just all exist as entities.
//!
//! Editors are draggable via their title bar and resizable via the
//! bottom-right handle. Focus is tracked with a `FocusedEditor` resource
//! so keyboard input routes to a single editor, and a `MouseMode` state
//! machine keeps drag / resize / text-selection distinct.
//!
//! Horizontal overflow is not clipped: `sync_text` truncates each line's
//! rendered text to the editor's content-area column count, so
//! characters past the edge simply don't get spawned.

use std::collections::HashMap;

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::text::{CosmicFontSystem, LineHeight, TextSpan};
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

pub mod highlight;
use highlight::{color_for, Highlighter};

pub const FONT_SIZE: f32 = 16.0;
pub const LINE_HEIGHT: f32 = 20.0;
pub const MARGIN: f32 = 16.0;
/// Height of the draggable title bar on each editor.
pub const TITLE_H: f32 = 22.0;
/// Side length of the square resize handle in the bottom-right corner.
pub const HANDLE_SIZE: f32 = 14.0;
pub const MIN_EDITOR_SIZE: Vec2 = Vec2::new(160.0, 120.0);


// ---------- Components: one editor = one entity with these ----------

#[derive(Component)]
pub struct Editor;

/// The editor's document + selection + history. Own component so
/// multiple editors can diverge independently.
#[derive(Component)]
pub struct EditorStateComp(pub EditorState);

/// Per-editor pool of line-row entities, keyed by doc line number.
/// Only lines currently in (or near) the editor's content viewport
/// have entries.
#[derive(Component, Default)]
pub struct LineRows(pub HashMap<usize, Entity>);

/// Scroll offset in pixels from the top-left of the doc's logical
/// coordinate space. `x` tracks horizontal column offset (multiples of
/// cell_width when snapped to cell boundaries); `y` tracks vertical
/// line offset.
#[derive(Component, Copy, Clone, Default)]
pub struct EditorScroll {
    pub x: f32,
    pub y: f32,
}

/// Anchor char offset of an in-progress text-selection drag inside
/// this editor, or `None` if no drag is active.
#[derive(Component, Default)]
pub struct TextDragAnchor(pub Option<usize>);

/// Per-editor tree-sitter parser + highlight spans.
#[derive(Component)]
pub struct EditorHighlighter(pub Highlighter);

/// Position, size, and Z-order of the editor in window-space coords
/// (top-left origin, y-down). `position_root` converts this to the
/// editor entity's world-space `Transform`.
#[derive(Component, Copy, Clone, Debug)]
pub struct EditorRect {
    pub pos: Vec2,
    pub size: Vec2,
    pub z: f32,
}

/// References to the child entities that make up an editor's visible
/// chrome. Held on the editor entity so systems can grab them without
/// re-querying the scene graph.
#[derive(Component)]
pub struct EditorChrome {
    pub bg: Entity,
    pub title_bar: Entity,
    pub content_root: Entity,
    pub caret: Entity,
    pub resize_handle: Entity,
}

/// Per-line marker: the text + highlighter revision last rendered into
/// this line entity. Rebuild spans only when either changes so Bevy's
/// text layout stays cached.
#[derive(Component)]
struct LineRender {
    text: String,
    rev: u64,
}

/// Marks a sprite spawned for one line of a selection highlight.
/// Carries the owning editor so we despawn only that editor's rects
/// when its selection changes.
#[derive(Component)]
pub struct SelRect {
    pub editor: Entity,
}

// ---------- Shared resources ----------

/// Handle to the editor font, shared by all editors.
#[derive(Resource)]
pub struct MonoFont(pub Handle<Font>);

/// Advance width of a single character cell in logical pixels, measured
/// once from the embedded font file. Caret, selection, mouse hit-test,
/// and horizontal truncation all do plain arithmetic against this.
#[derive(Resource, Copy, Clone)]
pub struct MonoMetrics {
    pub cell_width: f32,
}

/// Currently keyboard-focused editor. `None` if no editor has been
/// clicked yet or the last focused editor was despawned.
#[derive(Resource, Default)]
pub struct FocusedEditor(pub Option<Entity>);

/// What the left mouse button is doing right now. Set on press, held
/// while dragged, cleared on release. Keeps drag / resize / text
/// selection from interfering with each other.
#[derive(Resource, Default)]
pub enum MouseMode {
    #[default]
    Idle,
    TextSelect {
        editor: Entity,
    },
    WindowDrag {
        editor: Entity,
        grab_offset: Vec2,
    },
    WindowResize {
        editor: Entity,
        anchor_pos: Vec2,
    },
}

// ---------- Plugin ----------

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(FocusedEditor::default())
            .insert_resource(MouseMode::default())
            // Keep the scheduler in Continuous mode always — the
            // default `reactive_low_power` mode can drop or coalesce
            // wheel events on macOS when the window is briefly
            // unfocused, which made scroll feel like it "didn't work".
            .insert_resource(bevy::winit::WinitSettings {
                focused_mode: bevy::winit::UpdateMode::Continuous,
                unfocused_mode: bevy::winit::UpdateMode::Continuous,
            })
            .insert_resource(ClearColor(Color::srgb(0.10, 0.11, 0.13)))
            .add_systems(Startup, (setup_camera_and_font, load_fallback_fonts))
            // Run once, after the window exists, so we can immediately
            // give focus back to whatever app the user was in. The
            // PostStartup schedule is after Startup and after the
            // winit window has been created.
            .add_systems(PostStartup, release_os_focus)
            .add_systems(
                Update,
                (
                    handle_mouse,
                    handle_scroll,
                    handle_input,
                    update_highlight,
                    sync_text,
                    position_root,
                    sync_caret,
                    sync_selection,
                )
                    .chain(),
            );
    }
}

/// Tell the OS to deactivate our app so the user's previous app keeps
/// keyboard focus. Without this, macOS's `NSApplication.activate` call
/// (which winit issues on first window) pulls the user away from
/// whatever they were doing.
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

/// Minimal plugin for headless state-transition tests: just `handle_input`.
/// No rendering, no window, no asset loading. Tests spawn editor entities
/// directly; input routes via `FocusedEditor`.
pub struct HeadlessEditorPlugin;

impl Plugin for HeadlessEditorPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(FocusedEditor::default())
            .insert_resource(MouseMode::default())
            .add_systems(Update, handle_input);
    }
}

/// Bevy initializes `CosmicFontSystem` with an empty `fontdb`, so any
/// glyph not present in the primary editor font renders as tofu.
/// Populating the db with system fonts enables CJK/emoji fallback.
fn load_fallback_fonts(mut fonts: ResMut<CosmicFontSystem>) {
    fonts.0.db_mut().load_system_fonts();
}

/// JetBrains Mono bundled into the binary so release builds run from
/// any cwd without asset-path lookups.
const EMBEDDED_FONT: &[u8] = include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf");

/// Read the horizontal advance of `'M'` at `font_size` from the font
/// bytes. JetBrains Mono reports the same advance for every glyph so
/// `'M'` is an arbitrary choice.
fn measure_cell_width(font_bytes: &[u8], font_size: f32) -> f32 {
    use skrifa::instance::{LocationRef, Size};
    use skrifa::{FontRef, MetadataProvider};

    let font = FontRef::from_index(font_bytes, 0).expect("embedded font must parse");
    let metrics = font.glyph_metrics(Size::new(font_size), LocationRef::default());
    let gid = font
        .charmap()
        .map('M')
        .expect("embedded font must contain 'M'");
    metrics
        .advance_width(gid)
        .expect("'M' must have an advance width")
}

/// Spawns the shared camera and loads the embedded font. Exposed so
/// external startup systems that call `spawn_editor` can order
/// themselves via `.after(setup_camera_and_font)`.
pub fn setup_camera_and_font(mut commands: Commands, mut fonts: ResMut<Assets<Font>>) {
    commands.spawn(Camera2d);
    let font = fonts.add(
        Font::try_from_bytes(EMBEDDED_FONT.to_vec())
            .expect("embedded JetBrainsMono-Regular.ttf must parse"),
    );
    commands.insert_resource(MonoFont(font));
    commands.insert_resource(MonoMetrics {
        cell_width: measure_cell_width(EMBEDDED_FONT, FONT_SIZE),
    });
}

/// Spawn a new editor entity with initial doc text at the given rect.
/// Returns the editor entity so callers can set focus, add debug
/// components, etc.
pub fn spawn_editor(
    commands: &mut Commands,
    font: &MonoFont,
    initial_text: &str,
    rect: EditorRect,
) -> Entity {
    // Parent entity owns state + Transform; children form the visible
    // chrome. Positioning happens in `position_root` from `EditorRect`.
    let editor = commands
        .spawn((
            Editor,
            EditorStateComp(
                EditorState::new(
                    ropey::Rope::from_str(initial_text),
                    Selection::cursor(0),
                )
                .with_indent_unit("    "),
            ),
            EditorHighlighter(Highlighter::new()),
            LineRows::default(),
            EditorScroll::default(),
            TextDragAnchor::default(),
            rect,
            Transform::default(),
            Visibility::default(),
        ))
        .id();

    let bg = commands
        .spawn((
            ChildOf(editor),
            Sprite {
                color: Color::srgb(0.14, 0.16, 0.19),
                custom_size: Some(rect.size),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 0.0),
        ))
        .id();

    let title_bar = commands
        .spawn((
            ChildOf(editor),
            Sprite {
                color: Color::srgb(0.22, 0.24, 0.28),
                custom_size: Some(Vec2::new(rect.size.x, TITLE_H)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 0.1),
        ))
        .id();

    // Content root holds line text, caret, selection. Its local
    // transform carries the scroll offset so the chrome stays fixed
    // while the doc pans underneath it.
    let content_root = commands
        .spawn((
            ChildOf(editor),
            Transform::from_xyz(MARGIN, -(TITLE_H + MARGIN) * 0.0 - TITLE_H, 0.2),
            Visibility::default(),
        ))
        .id();

    let caret = commands
        .spawn((
            ChildOf(content_root),
            Sprite {
                color: Color::srgb(0.55, 0.85, 1.0),
                custom_size: Some(Vec2::new(2.0, LINE_HEIGHT)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 1.0),
        ))
        .id();

    let resize_handle = commands
        .spawn((
            ChildOf(editor),
            Sprite {
                color: Color::srgb(0.40, 0.44, 0.50),
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

    commands.entity(editor).insert(EditorChrome {
        bg,
        title_bar,
        content_root,
        caret,
        resize_handle,
    });
    let _ = font; // held by callers; kept in signature so future spawns can set a custom font
    editor
}

/// Build a ready-to-run App with a single editor loaded from `initial`.
pub fn build_app(initial: &str) -> App {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "editor".into(),
            resolution: (900u32, 600u32).into(),
            ..default()
        }),
        ..default()
    }));
    app.add_plugins(EditorPlugin);

    let initial = initial.to_string();
    app.add_systems(
        Startup,
        (move |mut commands: Commands, font: Res<MonoFont>| {
            let e = spawn_editor(
                &mut commands,
                &font,
                &initial,
                EditorRect {
                    pos: Vec2::new(40.0, 40.0),
                    size: Vec2::new(820.0, 500.0),
                    z: 1.0,
                },
            );
            commands.insert_resource(FocusedEditor(Some(e)));
        })
            .after(setup_camera_and_font),
    );
    app
}

// ---------- Content-area geometry helpers ----------

/// Top-left / size of the editable content area relative to the editor
/// entity's own origin (editor's origin = rect top-left). Scroll is
/// applied separately to the content root's transform.
fn content_area(rect: &EditorRect) -> (Vec2, Vec2) {
    let origin = Vec2::new(MARGIN, -(TITLE_H + MARGIN));
    let size = Vec2::new(
        (rect.size.x - 2.0 * MARGIN).max(0.0),
        (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
    );
    (origin, size)
}

/// Maximum visible character columns on a single line given the
/// editor's content width. `max(1)` so truncation never emits zero
/// length (which would hide the caret too).
fn max_cols(content_width: f32, cell_width: f32) -> usize {
    if cell_width <= 0.0 {
        return 0;
    }
    ((content_width / cell_width).floor() as usize).max(1)
}

/// Return the byte offset of the `n`-th character in `s`, or `s.len()`
/// if the string has fewer than `n` characters.
fn byte_offset_for_col(s: &str, col: usize) -> usize {
    s.char_indices().nth(col).map(|(b, _)| b).unwrap_or(s.len())
}

/// Slice `line_text` to the visible character window `[start_col,
/// start_col + max_cols)`. Returns `(byte_offset_in_line,
/// rendered_text)` — the offset lets the highlighter paint the right
/// spans for a horizontally-scrolled view. Characters outside the
/// window never become TextSpans, so nothing spills past the editor's
/// left or right edge.
fn slice_visible_cols(line_text: &str, start_col: usize, max_cols: usize) -> (usize, &str) {
    if max_cols == 0 {
        return (line_text.len(), "");
    }
    let start_byte = byte_offset_for_col(line_text, start_col);
    let end_byte = byte_offset_for_col(line_text, start_col + max_cols);
    (start_byte, &line_text[start_byte..end_byte])
}

// ---------- Systems ----------

fn line_text(doc: &ropey::Rope, idx: usize) -> String {
    let s = doc.line(idx).to_string();
    s.strip_suffix('\n').map(str::to_string).unwrap_or(s)
}

fn update_highlight(mut editors: Query<(&EditorStateComp, &mut EditorHighlighter)>) {
    for (state, mut hl) in &mut editors {
        hl.0.maybe_reparse(&state.0.doc);
    }
}

/// Nudge `scroll` so the caret at `(line, col)` sits inside the
/// editor's content area. Called from the input path after edits or
/// caret movement — NOT run as a Changed-filtered system, because
/// that fires on mouse clicks too, which would clobber manual scrolls.
fn ensure_caret_visible(
    state: &EditorState,
    rect: &EditorRect,
    scroll: &mut EditorScroll,
    cell_width: f32,
) {
    let head = state.selection.primary_range().head;
    let (line, col) = char_to_line_col(&state.doc, head);
    let (_, content) = content_area(rect);
    if content.x <= 0.0 || content.y <= 0.0 {
        return;
    }

    let line_top = line as f32 * LINE_HEIGHT;
    let line_bottom = line_top + LINE_HEIGHT;
    if line_top < scroll.y {
        scroll.y = line_top;
    } else if line_bottom > scroll.y + content.y {
        scroll.y = line_bottom - content.y;
    }

    let cell_left = col as f32 * cell_width;
    let cell_right = cell_left + cell_width;
    if cell_left < scroll.x {
        scroll.x = cell_left;
    } else if cell_right > scroll.x + content.x {
        scroll.x = cell_right - content.x;
    }
    scroll.x = scroll.x.max(0.0);
    scroll.y = scroll.y.max(0.0);
}

fn sync_text(
    font: Res<MonoFont>,
    metrics: Res<MonoMetrics>,
    mut editors: Query<(
        &EditorStateComp,
        &EditorRect,
        &EditorChrome,
        &EditorScroll,
        &EditorHighlighter,
        &mut LineRows,
    )>,
    mut line_q: Query<&mut LineRender>,
    children_q: Query<&Children>,
    mut commands: Commands,
) {
    for (state, rect, chrome, scroll, hl, mut pool) in &mut editors {
        sync_editor_lines(
            &state.0,
            rect,
            chrome,
            *scroll,
            &hl.0,
            &mut pool,
            &font,
            &metrics,
            &mut line_q,
            &children_q,
            &mut commands,
        );
    }
}

fn sync_editor_lines(
    state: &EditorState,
    rect: &EditorRect,
    chrome: &EditorChrome,
    scroll: EditorScroll,
    hl: &Highlighter,
    pool: &mut LineRows,
    font: &MonoFont,
    metrics: &MonoMetrics,
    line_q: &mut Query<&mut LineRender>,
    children_q: &Query<&Children>,
    commands: &mut Commands,
) {
    let rope = &state.doc;
    let effective = effective_line_count(rope);
    let (_, content_size) = content_area(rect);
    let (first, last) = viewport_line_range(content_size.y, scroll.y, effective);
    let cols = max_cols(content_size.x, metrics.cell_width);
    let scroll_cols = (scroll.x / metrics.cell_width).max(0.0) as usize;

    // Despawn entities out of the doc or out of the viewport window.
    pool.0.retain(|&idx, entity| {
        let keep = idx < effective && idx >= first && idx <= last;
        if !keep {
            commands.entity(*entity).despawn();
        }
        keep
    });

    for idx in first..=last.min(effective.saturating_sub(1)) {
        let full = line_text(rope, idx);
        let (byte_offset, slice) = slice_visible_cols(&full, scroll_cols, cols);
        let truncated = slice.to_string();
        if truncated.is_empty() {
            if let Some(entity) = pool.0.remove(&idx) {
                commands.entity(entity).despawn();
            }
            continue;
        }
        match pool.0.get(&idx).copied() {
            Some(entity) => {
                let needs_rebuild = line_q
                    .get(entity)
                    .map(|lr| lr.text != truncated || lr.rev != hl.rev)
                    .unwrap_or(true);
                if needs_rebuild {
                    rebuild_line_spans(
                        commands,
                        entity,
                        children_q,
                        hl,
                        rope,
                        idx,
                        byte_offset,
                        &truncated,
                        font,
                    );
                    if let Ok(mut lr) = line_q.get_mut(entity) {
                        lr.text = truncated;
                        lr.rev = hl.rev;
                    } else {
                        commands.entity(entity).insert(LineRender {
                            text: truncated,
                            rev: hl.rev,
                        });
                    }
                }
            }
            None => {
                let entity = commands
                    .spawn((
                        ChildOf(chrome.content_root),
                        Text2d::new(String::new()),
                        TextFont {
                            font: font.0.clone(),
                            font_size: FONT_SIZE,
                            ..default()
                        },
                        LineHeight::Px(LINE_HEIGHT),
                        TextColor(color_for(highlight::HighlightKind::Default)),
                        Anchor::TOP_LEFT,
                        Transform::from_xyz(0.0, -(idx as f32) * LINE_HEIGHT, 0.0),
                        LineRender {
                            text: truncated.clone(),
                            rev: hl.rev,
                        },
                    ))
                    .id();
                rebuild_line_spans(
                    commands,
                    entity,
                    children_q,
                    hl,
                    rope,
                    idx,
                    byte_offset,
                    &truncated,
                    font,
                );
                pool.0.insert(idx, entity);
            }
        }
    }
}

fn rebuild_line_spans(
    commands: &mut Commands,
    parent: Entity,
    children_q: &Query<&Children>,
    hl: &Highlighter,
    rope: &ropey::Rope,
    line_idx: usize,
    byte_offset: usize,
    line_text: &str,
    font: &MonoFont,
) {
    if let Ok(children) = children_q.get(parent) {
        for child in children.iter() {
            commands.entity(child).despawn();
        }
    }
    for (text, kind) in hl.line_chunks(rope, line_idx, byte_offset, line_text) {
        commands.spawn((
            ChildOf(parent),
            TextSpan::new(text),
            TextFont {
                font: font.0.clone(),
                font_size: FONT_SIZE,
                ..default()
            },
            LineHeight::Px(LINE_HEIGHT),
            TextColor(color_for(kind)),
        ));
    }
}

fn effective_line_count(rope: &ropey::Rope) -> usize {
    let n = rope.len_lines();
    if n == 0 {
        1
    } else if n > 1 && rope.line(n - 1).len_chars() == 0 {
        n - 1
    } else {
        n
    }
}

fn viewport_line_range(
    content_height: f32,
    scroll: f32,
    effective_lines: usize,
) -> (usize, usize) {
    if content_height <= 0.0 || effective_lines == 0 {
        return (0, 0);
    }
    let top = scroll.max(0.0);
    let bottom = (scroll + content_height).max(0.0);
    let first = (top / LINE_HEIGHT).floor().max(0.0) as usize;
    // Line `i` occupies y ∈ [i*LH, (i+1)*LH). It's visible iff it
    // starts strictly before `bottom`. `ceil(bottom/LH)` is the first
    // line past the viewport; subtract one to get the last visible.
    let last_excl = (bottom / LINE_HEIGHT).ceil().max(0.0) as usize;
    let last = last_excl.saturating_sub(1).min(effective_lines - 1);
    (first, last)
}

/// Place each editor's root transform + resize/chrome sprites based on
/// its `EditorRect`. Also syncs the content_root scroll offset and
/// resize-handle position, which change when the editor is dragged or
/// resized. Runs after state mutations so position is always current.
fn position_root(
    windows: Query<&Window>,
    editors: Query<(&EditorRect, &EditorScroll, &EditorChrome), With<Editor>>,
    mut t_q: Query<&mut Transform>,
    mut sprite_q: Query<&mut Sprite>,
    parents: Query<Entity, With<Editor>>,
) {
    let Ok(win) = windows.single() else { return };
    let win_size = Vec2::new(win.width(), win.height());

    for editor in &parents {
        let Ok((rect, scroll, chrome)) = editors.get(editor) else {
            continue;
        };

        if let Ok(mut t) = t_q.get_mut(editor) {
            t.translation.x = rect.pos.x - win_size.x * 0.5;
            t.translation.y = win_size.y * 0.5 - rect.pos.y;
            t.translation.z = rect.z;
        }

        // Background resizes with the editor.
        if let Ok(mut s) = sprite_q.get_mut(chrome.bg) {
            s.custom_size = Some(rect.size);
        }
        if let Ok(mut s) = sprite_q.get_mut(chrome.title_bar) {
            s.custom_size = Some(Vec2::new(rect.size.x, TITLE_H));
        }

        // Content root sits below the title bar and pans with vertical
        // scroll. Horizontal scroll is *not* applied here — we instead
        // slice each rendered line to the visible column range, so
        // lines always draw starting at `x = MARGIN` (no content ever
        // extends past the editor's left edge).
        if let Ok(mut t) = t_q.get_mut(chrome.content_root) {
            t.translation.x = MARGIN;
            t.translation.y = -(TITLE_H + MARGIN) + scroll.y;
        }

        // Handle in the bottom-right, anchored to the chrome rect.
        if let Ok(mut t) = t_q.get_mut(chrome.resize_handle) {
            t.translation.x = rect.size.x - HANDLE_SIZE;
            t.translation.y = -(rect.size.y - HANDLE_SIZE);
        }
    }
}

fn sync_caret(
    metrics: Res<MonoMetrics>,
    editors: Query<(&EditorStateComp, &EditorScroll, &EditorChrome)>,
    mut t_q: Query<&mut Transform>,
) {
    for (state, scroll, chrome) in &editors {
        let head = state.0.selection.primary_range().head;
        let (line, col) = char_to_line_col(&state.0.doc, head);
        // Caret's content-area-local X is the doc-col X minus any
        // horizontal scroll. Y is grid-aligned so the line's scroll.y
        // offset applied to content_root already shifts the caret.
        let x = caret_x_in_line(col, metrics.cell_width) - scroll.x;
        let y = line as f32 * LINE_HEIGHT;
        if let Ok(mut t) = t_q.get_mut(chrome.caret) {
            t.translation.x = x;
            t.translation.y = -y;
            t.translation.z = 1.0;
        }
    }
}

/// Doc-local caret X for a character column on a monospace line.
pub fn caret_x_in_line(col: usize, cell_width: f32) -> f32 {
    col as f32 * cell_width
}

/// Convert a rope char offset to (line_index, column_in_chars).
pub fn char_to_line_col(doc: &ropey::Rope, char_idx: usize) -> (usize, usize) {
    let line = doc.char_to_line(char_idx);
    let line_start = doc.line_to_char(line);
    (line, char_idx - line_start)
}

fn sync_selection(
    metrics: Res<MonoMetrics>,
    editors: Query<(Entity, &EditorStateComp, &EditorRect, &EditorScroll, &EditorChrome)>,
    existing: Query<(Entity, &SelRect)>,
    mut commands: Commands,
) {
    // Despawn all selection rects; we'll respawn the live ones below.
    for (entity, _) in &existing {
        commands.entity(entity).despawn();
    }

    for (editor_entity, state, rect, scroll, chrome) in &editors {
        let range = state.0.selection.primary_range();
        let (from, to) = (range.from(), range.to());
        if from == to {
            continue;
        }

        let (start_line, start_col) = char_to_line_col(&state.0.doc, from);
        let (end_line, end_col) = char_to_line_col(&state.0.doc, to);

        let (_, content) = content_area(rect);
        let rope = &state.0.doc;
        for line in start_line..=end_line {
            let line_chars = line_char_len(rope, line);
            let lo = if line == start_line { start_col } else { 0 };
            let hi = if line == end_line { end_col } else { line_chars };
            let ends_mid_doc = line < end_line;

            let (x0_raw, x1_raw) = line_selection_span(lo, hi, metrics.cell_width);
            let extra = if ends_mid_doc { LINE_HEIGHT * 0.3 } else { 0.0 };
            // Clip to the horizontally visible window.
            let x0 = (x0_raw - scroll.x).max(0.0);
            let x1 = (x1_raw + extra - scroll.x).min(content.x);
            if x1 <= x0 {
                continue;
            }

            let y_top = line as f32 * LINE_HEIGHT;
            commands.spawn((
                SelRect {
                    editor: editor_entity,
                },
                ChildOf(chrome.content_root),
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
}

/// X-range `[lo_col, hi_col]` in logical pixels on a monospace line.
pub fn line_selection_span(lo_col: usize, hi_col: usize, cell_width: f32) -> (f32, f32) {
    (lo_col as f32 * cell_width, hi_col as f32 * cell_width)
}

/// Character length of a line excluding its trailing newline, or 0 for
/// a past-EOF index.
fn line_char_len(rope: &ropey::Rope, line: usize) -> usize {
    if line >= rope.len_lines() {
        return 0;
    }
    let slice = rope.line(line);
    let n = slice.len_chars();
    if slice
        .chars_at(n)
        .reversed()
        .next()
        .is_some_and(|c| c == '\n')
    {
        n - 1
    } else {
        n
    }
}

/// Scroll the editor whose rect the pointer is over, in both axes. The
/// Y max clamp prevents scrolling past EOF; the X max grows with the
/// longest visible line. If the pointer is outside all editors, the
/// event is dropped.
fn handle_scroll(
    mut wheel: MessageReader<MouseWheel>,
    metrics: Res<MonoMetrics>,
    windows: Query<&Window>,
    mut editors: Query<(Entity, &EditorRect, &EditorStateComp, &mut EditorScroll)>,
) {
    // Accumulate X + Y deltas in pixels, converting notch-based mouse
    // wheel events the same way for both axes.
    let mut dx_px = 0.0;
    let mut dy_px = 0.0;
    for ev in wheel.read() {
        let (ux, uy) = match ev.unit {
            MouseScrollUnit::Pixel => (ev.x, ev.y),
            MouseScrollUnit::Line => (ev.x * metrics.cell_width, ev.y * LINE_HEIGHT),
        };
        dx_px += ux;
        dy_px += uy;
    }
    if dx_px == 0.0 && dy_px == 0.0 {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Some(pt) = win.cursor_position() else { return };

    let target = topmost_editor_at(
        pt,
        &editors.iter().map(|(e, r, _, _)| (e, *r)).collect::<Vec<_>>(),
    );
    let Some(editor) = target else { return };

    if let Ok((_, rect, state, mut scroll)) = editors.get_mut(editor) {
        let (_, content_size) = content_area(rect);
        let doc_height = state.0.doc.len_lines() as f32 * LINE_HEIGHT;
        let y_max = (doc_height - content_size.y).max(0.0);
        // Natural-scroll convention: dragging content up (positive
        // wheel Y) scrolls the viewport down. Same for X.
        scroll.y = (scroll.y - dy_px).clamp(0.0, y_max);

        // Horizontal max = the widest line minus the visible width.
        let widest_cols = widest_line_cols(&state.0.doc);
        let doc_width = widest_cols as f32 * metrics.cell_width;
        let x_max = (doc_width - content_size.x).max(0.0);
        scroll.x = (scroll.x - dx_px).clamp(0.0, x_max);
    }
}

/// Scan the rope for the longest line (character count, not bytes).
/// Used only to clamp horizontal scroll — called on wheel events, not
/// every frame.
fn widest_line_cols(rope: &ropey::Rope) -> usize {
    let n = rope.len_lines();
    let mut widest = 0;
    for i in 0..n {
        let w = line_char_len(rope, i);
        if w > widest {
            widest = w;
        }
    }
    widest
}

/// Routes keyboard input to the `FocusedEditor`.
fn handle_input(
    mut keys: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    focused: Res<FocusedEditor>,
    metrics: Res<MonoMetrics>,
    mut editors: Query<(&mut EditorStateComp, &EditorRect, &mut EditorScroll)>,
) {
    let Some(target) = focused.0 else {
        keys.read().for_each(|_| {});
        return;
    };
    let Ok((mut state_comp, rect, mut scroll)) = editors.get_mut(target) else {
        keys.read().for_each(|_| {});
        return;
    };
    let state = &mut state_comp.0;
    let mut state_mutated = false;

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
                    run(state, select_word_left)
                } else {
                    run(state, select_char_left)
                }
            } else if mod_word {
                run(state, cursor_word_left)
            } else {
                run(state, cursor_char_left)
            }),
            KeyCode::ArrowRight => Some(if shift {
                if mod_word {
                    run(state, select_word_right)
                } else {
                    run(state, select_char_right)
                }
            } else if mod_word {
                run(state, cursor_word_right)
            } else {
                run(state, cursor_char_right)
            }),
            KeyCode::ArrowUp => Some(if shift {
                run(state, select_line_up)
            } else {
                run(state, cursor_line_up)
            }),
            KeyCode::ArrowDown => Some(if shift {
                run(state, select_line_down)
            } else {
                run(state, cursor_line_down)
            }),
            KeyCode::Home => Some(if shift {
                if mod_doc {
                    run(state, select_doc_start)
                } else {
                    run(state, select_line_start)
                }
            } else if mod_doc {
                run(state, cursor_doc_start)
            } else {
                run(state, cursor_line_start)
            }),
            KeyCode::End => Some(if shift {
                if mod_doc {
                    run(state, select_doc_end)
                } else {
                    run(state, select_line_end)
                }
            } else if mod_doc {
                run(state, cursor_doc_end)
            } else {
                run(state, cursor_line_end)
            }),
            KeyCode::Backspace => Some(run_history(state, delete_char_backward)),
            KeyCode::Delete => Some(run_history(state, delete_char_forward)),
            KeyCode::Enter | KeyCode::NumpadEnter => {
                Some(run_history(state, insert_newline_and_indent))
            }
            KeyCode::Tab => Some(run_history(state, indent_more)),
            KeyCode::KeyA if mod_doc => Some(run(state, select_all)),
            KeyCode::KeyZ if mod_doc => Some(if shift {
                redo(state).map(|new| (new, true))
            } else {
                undo(state).map(|new| (new, true))
            }),
            KeyCode::KeyC if mod_doc => {
                copy_selection(state);
                Some(None)
            }
            KeyCode::KeyX if mod_doc => {
                copy_selection(state);
                Some(delete_selection(state))
            }
            KeyCode::KeyV if mod_doc => Some(paste_from_clipboard(state)),
            _ => None,
        };

        if let Some(Some((new_state, _))) = cmd_result {
            *state = new_state;
            state_mutated = true;
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
                    state.selection.primary_range().from(),
                    state.selection.primary_range().to(),
                    text.clone(),
                ))
                .select(Selection::cursor(
                    state.selection.primary_range().from() + text.chars().count(),
                ));
            *state = state.apply_with_history(&tr);
            state_mutated = true;
        }
    }

    if state_mutated {
        ensure_caret_visible(state, rect, &mut scroll, metrics.cell_width);
    }
}

/// Pure hit-test: logical x (content-area-local) → character column.
pub fn mouse_col_at_x(local_x: f32, cell_width: f32) -> usize {
    if cell_width <= 0.0 {
        return 0;
    }
    let col = (local_x / cell_width + 0.5).floor();
    if col <= 0.0 { 0 } else { col as usize }
}

/// Resolve a line + column (in chars) to a rope char offset, clamped
/// to the line's end-of-line char.
pub fn char_from_line_col(state: &EditorState, line: usize, col: usize) -> usize {
    let n_lines = state.doc.len_lines().max(1);
    let last_line = n_lines - 1;
    let line = line.min(last_line);
    let line_start = state.doc.line_to_char(line);
    let line_chars = line_char_len(&state.doc, line);
    line_start + col.min(line_chars)
}

/// Which editor's chrome contains `pt`, ordered by `z` descending.
/// Returns the topmost hit or `None`. Shared between scroll and mouse
/// handling.
fn topmost_editor_at(pt: Vec2, editors: &[(Entity, EditorRect)]) -> Option<Entity> {
    let mut best: Option<(Entity, f32)> = None;
    for &(e, r) in editors {
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

/// What part of an editor the pointer hit (pre-computed in window
/// space). Used by `handle_mouse` to pick between drag / resize /
/// text-select on click.
#[derive(Copy, Clone)]
enum Region {
    TitleBar,
    ResizeHandle,
    Content,
}

fn region_at(pt: Vec2, rect: &EditorRect) -> Option<Region> {
    if pt.x < rect.pos.x || pt.x > rect.pos.x + rect.size.x {
        return None;
    }
    if pt.y < rect.pos.y || pt.y > rect.pos.y + rect.size.y {
        return None;
    }
    // Resize handle wins at the bottom-right corner.
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

/// Convert a window-space point to content-area-local coords for the
/// given editor, including both-axis scroll offsets.
fn pt_to_content_local(pt: Vec2, rect: &EditorRect, scroll: EditorScroll) -> Vec2 {
    let origin = rect.pos + Vec2::new(MARGIN, TITLE_H + MARGIN);
    Vec2::new(pt.x - origin.x + scroll.x, pt.y - origin.y + scroll.y)
}

fn handle_mouse(
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    mods: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<MouseMode>,
    mut focused: ResMut<FocusedEditor>,
    metrics: Res<MonoMetrics>,
    mut editors: Query<(
        Entity,
        &mut EditorRect,
        &mut EditorStateComp,
        &EditorScroll,
        &mut TextDragAnchor,
    )>,
) {
    let Ok(window) = windows.single() else { return };
    let Some(pt) = window.cursor_position() else { return };
    let shift = mods.pressed(KeyCode::ShiftLeft) || mods.pressed(KeyCode::ShiftRight);

    // Release always clears the mode, regardless of which button type
    // started it.
    if buttons.just_released(MouseButton::Left) {
        if let MouseMode::TextSelect { editor } = *mode {
            if let Ok((_, _, _, _, mut drag)) = editors.get_mut(editor) {
                drag.0 = None;
            }
        }
        *mode = MouseMode::Idle;
    }

    if buttons.just_pressed(MouseButton::Left) {
        // Hit-test from top-most editor.
        let editor_rects: Vec<(Entity, EditorRect)> =
            editors.iter().map(|(e, r, _, _, _)| (e, *r)).collect();
        let Some(editor) = topmost_editor_at(pt, &editor_rects) else {
            return;
        };
        focused.0 = Some(editor);
        bring_to_front(editor, &mut editors);

        let rect = *editors.get(editor).unwrap().1;
        match region_at(pt, &rect) {
            Some(Region::TitleBar) => {
                *mode = MouseMode::WindowDrag {
                    editor,
                    grab_offset: pt - rect.pos,
                };
            }
            Some(Region::ResizeHandle) => {
                *mode = MouseMode::WindowResize {
                    editor,
                    anchor_pos: rect.pos,
                };
            }
            Some(Region::Content) => {
                if let Ok((_, _, mut state_comp, scroll, mut drag)) = editors.get_mut(editor) {
                    let state = &mut state_comp.0;
                    let local = pt_to_content_local(pt, &rect, *scroll);
                    let line = (local.y / LINE_HEIGHT).floor().max(0.0) as usize;
                    let col = mouse_col_at_x(local.x, metrics.cell_width);
                    let pos = char_from_line_col(state, line, col);
                    if shift {
                        let anchor = state.selection.primary_range().anchor;
                        drag.0 = Some(anchor);
                        *state = apply_selection(state, anchor, pos);
                    } else {
                        drag.0 = Some(pos);
                        *state = apply_selection(state, pos, pos);
                    }
                    *mode = MouseMode::TextSelect { editor };
                }
            }
            None => {}
        }
        return;
    }

    if !buttons.pressed(MouseButton::Left) {
        return;
    }

    match *mode {
        MouseMode::WindowDrag { editor, grab_offset } => {
            if let Ok((_, mut rect, _, _, _)) = editors.get_mut(editor) {
                rect.pos = pt - grab_offset;
            }
        }
        MouseMode::WindowResize { editor, anchor_pos } => {
            if let Ok((_, mut rect, _, _, _)) = editors.get_mut(editor) {
                let raw = pt - anchor_pos;
                rect.size = Vec2::new(raw.x.max(MIN_EDITOR_SIZE.x), raw.y.max(MIN_EDITOR_SIZE.y));
            }
        }
        MouseMode::TextSelect { editor } => {
            if let Ok((_, rect, mut state_comp, scroll, drag)) = editors.get_mut(editor) {
                let Some(anchor) = drag.0 else { return };
                let state = &mut state_comp.0;
                let local = pt_to_content_local(pt, &rect, *scroll);
                let line = (local.y / LINE_HEIGHT).floor().max(0.0) as usize;
                let col = mouse_col_at_x(local.x, metrics.cell_width);
                let head = char_from_line_col(state, line, col);
                let cur = state.selection.primary_range();
                if cur.anchor != anchor || cur.head != head {
                    *state = apply_selection(state, anchor, head);
                }
            }
        }
        MouseMode::Idle => {}
    }
}

/// Bump the editor's z above all other editors so subsequent z-sorts
/// draw it on top. Called on focus.
fn bring_to_front(
    editor: Entity,
    editors: &mut Query<(
        Entity,
        &mut EditorRect,
        &mut EditorStateComp,
        &EditorScroll,
        &mut TextDragAnchor,
    )>,
) {
    let max_z = editors
        .iter()
        .map(|(_, r, _, _, _)| r.z)
        .fold(0.0_f32, f32::max);
    if let Ok((_, mut rect, _, _, _)) = editors.get_mut(editor) {
        if rect.z < max_z {
            rect.z = max_z + 1.0;
        }
    }
}

fn apply_selection(state: &EditorState, anchor: usize, head: usize) -> EditorState {
    let tr = Transaction::new().select(Selection::single(Range::new(anchor, head)));
    state.apply_with_history(&tr)
}

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
