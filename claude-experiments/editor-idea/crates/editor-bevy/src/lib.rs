//! Editor view layer built on Bevy's 2D/text pipeline.
//!
//! Pane chrome (drag, resize, close, focus, z-order) is owned by
//! `pane-bevy`. This crate only owns editor-specific concerns:
//! rendering text spans into the pane's content_root, caret/selection
//! visuals, scroll, keyboard input, and the highlighter.
//!
//! # Embedding
//!
//! - `EditorPlugin` — standalone editor app: camera, font, clear color,
//!   winit settings, plus `PanePlugin` and `EditorEmbedPlugin`.
//! - `EditorEmbedPlugin` — for hosts (like `terminal-bevy`) that already
//!   own the camera, clear color, etc. Registers the "editor" pane kind
//!   in `PaneRegistry` and adds editor-specific systems. Hosts must
//!   also add `PanePlugin` and provide `EditorFont` + `EditorMetrics`
//!   via `setup_editor_font`.

use std::collections::HashMap;
use std::path::PathBuf;

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
use pane_bevy::{
    spawn_pane, FocusedPane, PaneChrome, PaneContentPressed, PaneFont, PaneKindMarker,
    PanePlugin, PaneRect, PaneRegistry, PaneTag, SpawnedPane, MARGIN, TITLE_H,
};
use serde_json::Value;

pub mod highlight;
use highlight::{color_for, Highlighter, SyntaxPalette};

pub const FONT_SIZE: f32 = 16.0;
pub const LINE_HEIGHT: f32 = 20.0;

pub const PANE_KIND: &str = "editor";

// ---------- Components: editor-specific ----------

/// Filename a pane was opened from. Hosts (terminal-bevy) attach this on
/// open-file requests; the editor crate doesn't read it itself, but
/// pane-bevy persistence routes through `editor_snapshot` which inspects it.
#[derive(Component, Clone, Debug)]
pub struct EditorFilePath(pub PathBuf);

#[derive(Component)]
pub struct EditorStateComp(pub EditorState);

#[derive(Component, Default)]
pub struct LineRows(pub HashMap<usize, Entity>);

#[derive(Component, Copy, Clone, Default)]
pub struct EditorScroll {
    pub x: f32,
    pub y: f32,
}

/// Anchor char offset of an in-progress text-selection drag.
#[derive(Component, Default)]
pub struct TextDragAnchor(pub Option<usize>);

#[derive(Component)]
pub struct EditorHighlighter(pub Highlighter);

#[derive(Component)]
struct LineRender {
    text: String,
    rev: u64,
    /// Snapshot of `SyntaxPalette.rev` at the time we built the
    /// colored spans. A theme switch bumps the palette rev; we rebuild
    /// the line so the new colors apply.
    palette_rev: u64,
}

#[derive(Component)]
pub struct SelRect {
    pub editor: Entity,
}

// ---------- Shared resources ----------

#[derive(Resource)]
pub struct EditorFont(pub Handle<Font>);

#[derive(Resource, Copy, Clone)]
pub struct EditorMetrics {
    pub cell_width: f32,
}

// ---------- Plugins ----------

pub struct EditorEmbedPlugin;

impl Plugin for EditorEmbedPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(highlight::HighlightPlugin)
            .add_systems(Startup, register_editor_kind)
            .add_systems(
                Update,
                (
                    handle_pane_content_press,
                    handle_text_select_drag,
                    handle_scroll,
                    handle_input,
                    update_highlight,
                    sync_text,
                    sync_content_root,
                    sync_caret,
                    sync_selection,
                )
                    .chain(),
            );
    }
}

fn register_editor_kind(mut registry: ResMut<PaneRegistry>) {
    registry.register(pane_bevy::PaneKindSpec {
        kind: PANE_KIND,
        display_name: "Editor",
        radial_icon: Some("{}"),
        default_size: Vec2::new(640.0, 420.0),
        spawn: editor_spawn_from_config,
        snapshot: editor_snapshot,
        on_close: None,
    });
}

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(bevy::winit::WinitSettings {
            focused_mode: bevy::winit::UpdateMode::Continuous,
            unfocused_mode: bevy::winit::UpdateMode::Continuous,
        })
        .insert_resource(ClearColor(Color::srgb(0.10, 0.11, 0.13)))
        .add_systems(
            Startup,
            (
                setup_editor_camera,
                setup_editor_font,
                load_fallback_fonts,
            ),
        )
        .add_systems(PostStartup, release_os_focus)
        .add_plugins(PanePlugin::default())
        .add_plugins(EditorEmbedPlugin);
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

/// Headless plugin for keybinding tests. No PanePlugin dependency —
/// tests spawn a bare editor entity and drive `handle_input` directly.
pub struct HeadlessEditorPlugin;

impl Plugin for HeadlessEditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FocusedPane>()
            .add_systems(Update, handle_input);
    }
}

fn load_fallback_fonts(mut fonts: ResMut<CosmicFontSystem>) {
    fonts.0.db_mut().load_system_fonts();
}

const EMBEDDED_FONT: &[u8] = include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf");

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

pub fn setup_editor_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// Load the editor font and insert both `EditorFont` and `PaneFont`
/// (the latter is what pane-bevy uses for chrome). Standalone hosts
/// can call this; embedders that already have a font may skip it and
/// insert their own.
///
/// Resolves the font through [`style_bevy::FontRegistry`] using
/// `FONT_FAMILY_MONO` from the active theme, so adding a real mono
/// family later (or pointing the token at a different bundled name)
/// retones the editor without code changes. `measure_cell_width` still
/// reads from the bundled bytes since skrifa needs raw bytes.
pub fn setup_editor_font(
    mut commands: Commands,
    mut fonts: ResMut<Assets<Font>>,
    mut registry: ResMut<style_bevy::FontRegistry>,
    theme: Res<style_bevy::Theme>,
) {
    // Make sure the registry has its bundled families — Startup-system
    // ordering is unreliable across plugins, so push entries from here
    // too. `ensure_initialized` is idempotent.
    style_bevy::fonts::ensure_initialized(&mut registry, &mut fonts);
    let mono_name = theme.str_value(style_bevy::tokens::FONT_FAMILY_MONO).to_string();
    let font = registry.resolve(&mono_name);
    let measure_bytes = registry.bytes(&mono_name).unwrap_or(EMBEDDED_FONT);
    commands.insert_resource(EditorFont(font.clone()));
    commands.insert_resource(EditorMetrics {
        cell_width: measure_cell_width(measure_bytes, FONT_SIZE),
    });
    commands.insert_resource(PaneFont(font));
    // `PaneFont` is what pane-bevy and every cosmic-text-backed pane
    // (Issues, text inputs) actually *render* with, so `PaneFontMetrics`
    // — which those panes use to place carets/selection on a
    // `col * cell_width` grid and to compute word-wrap — must describe
    // THIS font. We set both here (overwriting any host default) so the
    // measured advance always matches the rendered glyphs; otherwise the
    // caret drifts right of the text by the per-glyph advance difference,
    // growing with column. Measured at `FONT_SIZE`; callers scale linearly
    // for other sizes (the mono family is fixed-pitch).
    commands.insert_resource(pane_bevy::PaneFontMetrics {
        cell_width: measure_cell_width(measure_bytes, FONT_SIZE),
        font_size: FONT_SIZE,
    });
}

// ---------- Spawn ----------

/// Spawn an editor pane with the given initial text. Returns the pane
/// entity. Uses pane-bevy chrome under the hood.
pub fn spawn_editor_pane(
    world: &mut World,
    initial_text: &str,
    rect: PaneRect,
    project_id: Option<u64>,
) -> Entity {
    let SpawnedPane {
        entity,
        content_root,
    } = spawn_pane(world, PANE_KIND, "Editor", rect, project_id);
    populate_editor_pane(world, entity, content_root, initial_text);
    entity
}

/// Insert editor-specific components on an already-spawned pane and
/// add the caret child under `content_root`. Shared between
/// `spawn_editor_pane` and the registry restore path.
fn populate_editor_pane(
    world: &mut World,
    entity: Entity,
    content_root: Entity,
    initial_text: &str,
) {
    world.entity_mut(entity).insert((
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
    ));
    let caret_color = world
        .get_resource::<style_bevy::Theme>()
        .map(|t| Color::LinearRgba(t.color(style_bevy::tokens::CARET)))
        .unwrap_or_else(|| Color::srgb(0.55, 0.85, 1.0));
    let _caret = world
        .spawn((
            ChildOf(content_root),
            EditorCaret(entity),
            Sprite {
                color: caret_color,
                custom_size: Some(Vec2::new(2.0, LINE_HEIGHT)),
                ..default()
            },
            Anchor::TOP_LEFT,
            Transform::from_xyz(0.0, 0.0, 1.0),
        ))
        .id();
}

/// Marker for the caret child of an editor pane. Holds the parent
/// pane entity so the caret-sync system can join its position back to
/// the pane's editor state without needing the pane chrome to track
/// caret separately.
#[derive(Component)]
struct EditorCaret(Entity);

/// Registry callback — invoked by pane-bevy on restore. The config
/// blob is whatever `editor_snapshot` produced (currently `{ text,
/// path }`).
fn editor_spawn_from_config(world: &mut World, entity: Entity, content_root: Entity, config: &Value) {
    let text = config
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    populate_editor_pane(world, entity, content_root, &text);
    if let Some(path) = config.get("path").and_then(|v| v.as_str()) {
        world
            .entity_mut(entity)
            .insert(EditorFilePath(PathBuf::from(path)));
    }
}

fn editor_snapshot(world: &World, entity: Entity) -> Value {
    let text = world
        .get::<EditorStateComp>(entity)
        .map(|s| s.0.doc.to_string())
        .unwrap_or_default();
    let mut obj = serde_json::Map::new();
    obj.insert("text".into(), Value::String(text));
    if let Some(path) = world.get::<EditorFilePath>(entity) {
        obj.insert(
            "path".into(),
            Value::String(path.0.to_string_lossy().into()),
        );
    }
    Value::Object(obj)
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
        (move |world: &mut World| {
            let initial = initial.clone();
            let e = spawn_editor_pane(
                world,
                &initial,
                PaneRect {
                    pos: Vec2::new(40.0, 40.0),
                    size: Vec2::new(820.0, 500.0),
                    z: 1.0,
                },
                None,
            );
            world.resource_mut::<FocusedPane>().0 = Some(e);
        })
            .after(setup_editor_font),
    );
    app
}

// ---------- Content-area geometry ----------

/// Legacy: `PaneRect` is now canvas-space directly (the pane entity's
/// Transform handles zoom). Zoom is ignored; kept so existing call
/// sites compile.
fn content_area_size_zoomed(rect: &PaneRect, _zoom: f32) -> Vec2 {
    content_area_size(rect)
}

fn content_area_size(rect: &PaneRect) -> Vec2 {
    Vec2::new(
        (rect.size.x - 2.0 * MARGIN).max(0.0),
        (rect.size.y - TITLE_H - 2.0 * MARGIN).max(0.0),
    )
}

fn max_cols(content_width: f32, cell_width: f32) -> usize {
    if cell_width <= 0.0 {
        return 0;
    }
    ((content_width / cell_width).floor() as usize).max(1)
}

fn byte_offset_for_col(s: &str, col: usize) -> usize {
    s.char_indices().nth(col).map(|(b, _)| b).unwrap_or(s.len())
}

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

fn ensure_caret_visible(
    state: &EditorState,
    rect: &PaneRect,
    scroll: &mut EditorScroll,
    cell_width: f32,
    zoom: f32,
) {
    let head = state.selection.primary_range().head;
    let (line, col) = char_to_line_col(&state.doc, head);
    let content = content_area_size_zoomed(rect, zoom);
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
    font: Res<EditorFont>,
    metrics: Res<EditorMetrics>,
    palette: Res<SyntaxPalette>,
    project_palettes: Res<highlight::ProjectSyntaxPalettes>,
    pane_zoom: Res<pane_bevy::PaneZoom>,
    mut editors: Query<
        (
            Entity,
            &EditorStateComp,
            &PaneRect,
            &PaneChrome,
            &EditorScroll,
            &EditorHighlighter,
            &mut LineRows,
            &PaneKindMarker,
            Option<&pane_bevy::PaneProject>,
        ),
        With<PaneTag>,
    >,
    mut line_q: Query<&mut LineRender>,
    children_q: Query<&Children>,
    mut commands: Commands,
) {
    let zoom = pane_zoom.0;
    for (entity, state, rect, chrome, scroll, hl, mut pool, kind, proj) in &mut editors {
        if kind.0 != PANE_KIND {
            continue;
        }
        let _prof = pane_bevy::prof::pane_span(entity.to_bits(), "editor");
        // This editor's project palette if cached, else the global one.
        let pal = proj
            .and_then(|p| project_palettes.get(p.0))
            .unwrap_or(&palette);
        sync_editor_lines(
            &state.0,
            rect,
            chrome,
            *scroll,
            &hl.0,
            pal,
            &mut pool,
            &font,
            &metrics,
            &mut line_q,
            &children_q,
            &mut commands,
            zoom,
        );
    }
}

fn sync_editor_lines(
    state: &EditorState,
    rect: &PaneRect,
    chrome: &PaneChrome,
    scroll: EditorScroll,
    hl: &Highlighter,
    palette: &SyntaxPalette,
    pool: &mut LineRows,
    font: &EditorFont,
    metrics: &EditorMetrics,
    line_q: &mut Query<&mut LineRender>,
    children_q: &Query<&Children>,
    commands: &mut Commands,
    zoom: f32,
) {
    let rope = &state.doc;
    let effective = effective_line_count(rope);
    let content_size = content_area_size_zoomed(rect, zoom);
    let (first, last) = viewport_line_range(content_size.y, scroll.y, effective);
    let cols = max_cols(content_size.x, metrics.cell_width);
    let scroll_cols = (scroll.x / metrics.cell_width).max(0.0) as usize;

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
                    .map(|lr| {
                        lr.text != truncated
                            || lr.rev != hl.rev
                            || lr.palette_rev != palette.rev
                    })
                    .unwrap_or(true);
                if needs_rebuild {
                    rebuild_line_spans(
                        commands,
                        entity,
                        children_q,
                        hl,
                        palette,
                        rope,
                        idx,
                        byte_offset,
                        &truncated,
                        font,
                    );
                    if let Ok(mut lr) = line_q.get_mut(entity) {
                        lr.text = truncated;
                        lr.rev = hl.rev;
                        lr.palette_rev = palette.rev;
                    } else {
                        commands.entity(entity).insert(LineRender {
                            text: truncated,
                            rev: hl.rev,
                            palette_rev: palette.rev,
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
                        TextColor(palette.color_for(highlight::HighlightKind::Default)),
                        Anchor::TOP_LEFT,
                        Transform::from_xyz(0.0, -(idx as f32) * LINE_HEIGHT, 0.0),
                        LineRender {
                            text: truncated.clone(),
                            rev: hl.rev,
                            palette_rev: palette.rev,
                        },
                    ))
                    .id();
                rebuild_line_spans(
                    commands,
                    entity,
                    children_q,
                    hl,
                    palette,
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
    palette: &SyntaxPalette,
    rope: &ropey::Rope,
    line_idx: usize,
    byte_offset: usize,
    line_text: &str,
    font: &EditorFont,
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
            TextColor(color_for(palette, kind)),
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
    let last_excl = (bottom / LINE_HEIGHT).ceil().max(0.0) as usize;
    let last = last_excl.saturating_sub(1).min(effective_lines - 1);
    (first, last)
}

/// Apply scroll Y to the content_root transform. (X scroll is handled
/// by per-line slicing inside sync_editor_lines.)
fn sync_content_root(
    editors: Query<(&EditorScroll, &PaneChrome, &PaneKindMarker), With<PaneTag>>,
    mut t_q: Query<&mut Transform>,
) {
    for (scroll, chrome, kind) in &editors {
        if kind.0 != PANE_KIND {
            continue;
        }
        if let Ok(mut t) = t_q.get_mut(chrome.content_root) {
            t.translation.x = MARGIN;
            t.translation.y = -(TITLE_H + MARGIN) + scroll.y;
        }
    }
}

fn sync_caret(
    metrics: Res<EditorMetrics>,
    theme: Res<style_bevy::Theme>,
    pane_zoom: Res<pane_bevy::PaneZoom>,
    editors: Query<
        (&EditorStateComp, &PaneRect, &EditorScroll, &PaneKindMarker),
        With<PaneTag>,
    >,
    carets: Query<(Entity, &EditorCaret)>,
    mut t_q: Query<&mut Transform>,
    mut vis_q: Query<&mut Visibility>,
    mut sprite_q: Query<&mut Sprite>,
) {
    let caret_color = Color::LinearRgba(theme.color(style_bevy::tokens::CARET));
    let theme_changed = theme.is_changed();
    let zoom = pane_zoom.0;
    for (caret_entity, parent) in &carets {
        let Ok((state, rect, scroll, kind)) = editors.get(parent.0) else {
            continue;
        };
        if kind.0 != PANE_KIND {
            continue;
        }
        let head = state.0.selection.primary_range().head;
        let (line, col) = char_to_line_col(&state.0.doc, head);
        let caret_x = caret_x_in_line(col, metrics.cell_width);
        let content = content_area_size_zoomed(rect, zoom);

        let x = caret_x - scroll.x;
        let y = line as f32 * LINE_HEIGHT;
        let visible = x >= 0.0
            && x <= content.x
            && (line as f32 + 1.0) * LINE_HEIGHT > scroll.y
            && (line as f32) * LINE_HEIGHT < scroll.y + content.y;

        if let Ok(mut t) = t_q.get_mut(caret_entity) {
            t.translation.x = x;
            t.translation.y = -y;
            t.translation.z = 1.0;
        }
        if let Ok(mut v) = vis_q.get_mut(caret_entity) {
            *v = if visible {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
        }
        if theme_changed {
            if let Ok(mut s) = sprite_q.get_mut(caret_entity) {
                s.color = caret_color;
            }
        }
    }
}

pub fn caret_x_in_line(col: usize, cell_width: f32) -> f32 {
    col as f32 * cell_width
}

pub fn char_to_line_col(doc: &ropey::Rope, char_idx: usize) -> (usize, usize) {
    let line = doc.char_to_line(char_idx);
    let line_start = doc.line_to_char(line);
    (line, char_idx - line_start)
}

fn sync_selection(
    metrics: Res<EditorMetrics>,
    theme: Res<style_bevy::Theme>,
    pane_zoom: Res<pane_bevy::PaneZoom>,
    editors: Query<
        (
            Entity,
            &EditorStateComp,
            &PaneRect,
            &EditorScroll,
            &PaneChrome,
            &PaneKindMarker,
        ),
        With<PaneTag>,
    >,
    existing: Query<(Entity, &SelRect)>,
    mut commands: Commands,
) {
    let zoom = pane_zoom.0;
    for (entity, _) in &existing {
        commands.entity(entity).despawn();
    }

    for (editor_entity, state, rect, scroll, chrome, kind) in &editors {
        if kind.0 != PANE_KIND {
            continue;
        }
        let range = state.0.selection.primary_range();
        let (from, to) = (range.from(), range.to());
        if from == to {
            continue;
        }

        let (start_line, start_col) = char_to_line_col(&state.0.doc, from);
        let (end_line, end_col) = char_to_line_col(&state.0.doc, to);

        let content = content_area_size_zoomed(rect, zoom);
        let rope = &state.0.doc;
        for line in start_line..=end_line {
            let line_chars = line_char_len(rope, line);
            let lo = if line == start_line { start_col } else { 0 };
            let hi = if line == end_line { end_col } else { line_chars };
            let ends_mid_doc = line < end_line;

            let (x0_raw, x1_raw) = line_selection_span(lo, hi, metrics.cell_width);
            let extra = if ends_mid_doc { LINE_HEIGHT * 0.3 } else { 0.0 };
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
                    color: Color::LinearRgba(theme.color(style_bevy::tokens::SELECTION)),
                    custom_size: Some(Vec2::new((x1 - x0).max(1.0), LINE_HEIGHT)),
                    ..default()
                },
                Anchor::TOP_LEFT,
                Transform::from_xyz(x0, -y_top, 0.5),
            ));
        }
    }
}

pub fn line_selection_span(lo_col: usize, hi_col: usize, cell_width: f32) -> (f32, f32) {
    (lo_col as f32 * cell_width, hi_col as f32 * cell_width)
}

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

fn handle_scroll(
    mut wheel: MessageReader<MouseWheel>,
    metrics: Res<EditorMetrics>,
    windows: Query<&Window>,
    keys: Res<ButtonInput<KeyCode>>,
    pane_zoom: Res<pane_bevy::PaneZoom>,
    viewport: Res<pane_bevy::PaneViewport>,
    all_panes: Query<(Entity, &PaneRect, Option<&Visibility>), With<PaneTag>>,
    mut editors: Query<
        (Entity, &PaneRect, &EditorStateComp, &mut EditorScroll, &PaneKindMarker),
        With<PaneTag>,
    >,
) {
    let zoom = pane_zoom.0;
    // Cmd+scroll is the host's canvas pan gesture; don't double-scroll.
    if keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight) {
        wheel.clear();
        return;
    }
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

    // Topmost pane of ANY kind under the cursor. Bail if it isn't an
    // editor — otherwise scrolling on a widget/terminal that happens
    // to overlap an editor underneath would also scroll the editor.
    let all_rects: Vec<(Entity, PaneRect)> = all_panes
        .iter()
        .filter(|(_, _, vis)| !matches!(vis, Some(Visibility::Hidden)))
        .map(|(e, r, _)| (e, *r))
        .collect();
    let Some(editor) = pane_bevy::topmost_pane_at(viewport.window_to_canvas(pt), &all_rects)
    else {
        return;
    };
    let Ok((_, _, _, _, kind)) = editors.get(editor) else {
        return;
    };
    if kind.0 != PANE_KIND {
        return;
    }

    if let Ok((_, rect, state, mut scroll, _)) = editors.get_mut(editor) {
        let content_size = content_area_size_zoomed(rect, zoom);
        let doc_height = state.0.doc.len_lines() as f32 * LINE_HEIGHT;
        let y_max = (doc_height - content_size.y).max(0.0);
        scroll.y = (scroll.y - dy_px).clamp(0.0, y_max);

        let widest_cols = widest_line_cols(&state.0.doc);
        let doc_width = widest_cols as f32 * metrics.cell_width;
        let x_max = (doc_width - content_size.x).max(0.0);
        scroll.x = (scroll.x - dx_px).clamp(0.0, x_max);
    }
}

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

fn handle_input(
    mut keys: MessageReader<KeyboardInput>,
    mods: Res<ButtonInput<KeyCode>>,
    focused: Res<FocusedPane>,
    metrics: Option<Res<EditorMetrics>>,
    pane_zoom: Res<pane_bevy::PaneZoom>,
    mut editors: Query<
        (
            &mut EditorStateComp,
            &PaneRect,
            &mut EditorScroll,
            &PaneKindMarker,
            Option<&EditorFilePath>,
        ),
        With<PaneTag>,
    >,
) {
    let zoom = pane_zoom.0;
    let Some(target) = focused.0 else {
        keys.read().for_each(|_| {});
        return;
    };
    let Ok((mut state_comp, rect, mut scroll, kind, file_path)) = editors.get_mut(target) else {
        keys.read().for_each(|_| {});
        return;
    };
    if kind.0 != PANE_KIND {
        keys.read().for_each(|_| {});
        return;
    }
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
            // Paste is Cmd/Ctrl+V *without* Shift — Cmd+Shift+V is reserved
            // for app-global shortcuts (e.g. the profiler's vsync toggle) so
            // it must not leak into a focused editor as a paste.
            KeyCode::KeyV if mod_doc && !shift => Some(paste_from_clipboard(state)),
            // Cmd/Ctrl+S — save the buffer to its file. Side effect only
            // (no document change), so it returns `Some(None)`.
            KeyCode::KeyS if mod_doc => {
                match file_path {
                    Some(path) => {
                        let text = state.doc.to_string();
                        match std::fs::write(&path.0, text) {
                            Ok(()) => eprintln!("[editor] saved {}", path.0.display()),
                            Err(e) => {
                                eprintln!("[editor] save failed {}: {}", path.0.display(), e)
                            }
                        }
                    }
                    None => eprintln!("[editor] save: no file path for this pane"),
                }
                Some(None)
            }
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
        if let Some(metrics) = metrics {
            ensure_caret_visible(state, rect, &mut scroll, metrics.cell_width, zoom);
        }
    }
}

pub fn mouse_col_at_x(local_x: f32, cell_width: f32) -> usize {
    if cell_width <= 0.0 {
        return 0;
    }
    let col = (local_x / cell_width + 0.5).floor();
    if col <= 0.0 { 0 } else { col as usize }
}

pub fn char_from_line_col(state: &EditorState, line: usize, col: usize) -> usize {
    let n_lines = state.doc.len_lines().max(1);
    let last_line = n_lines - 1;
    let line = line.min(last_line);
    let line_start = state.doc.line_to_char(line);
    let line_chars = line_char_len(&state.doc, line);
    line_start + col.min(line_chars)
}

// ---------- Content-area input (pane-bevy events) ----------

/// React to `PaneContentPressed` for editor panes: place caret or
/// extend selection from the previous anchor (shift held).
fn handle_pane_content_press(
    mut presses: MessageReader<PaneContentPressed>,
    metrics: Option<Res<EditorMetrics>>,
    mut editors: Query<
        (
            &mut EditorStateComp,
            &EditorScroll,
            &mut TextDragAnchor,
            &PaneKindMarker,
        ),
        With<PaneTag>,
    >,
) {
    let Some(metrics) = metrics else {
        presses.read().for_each(|_| {});
        return;
    };
    for ev in presses.read() {
        let Ok((mut state_comp, scroll, mut drag, kind)) = editors.get_mut(ev.pane) else {
            continue;
        };
        if kind.0 != PANE_KIND {
            continue;
        }
        let state = &mut state_comp.0;
        let local_with_scroll = ev.local_pt + Vec2::new(scroll.x, scroll.y);
        let line = (local_with_scroll.y / LINE_HEIGHT).floor().max(0.0) as usize;
        let col = mouse_col_at_x(local_with_scroll.x, metrics.cell_width);
        let pos = char_from_line_col(state, line, col);
        if ev.shift {
            let anchor = state.selection.primary_range().anchor;
            drag.0 = Some(anchor);
            *state = apply_selection(state, anchor, pos);
        } else {
            drag.0 = Some(pos);
            *state = apply_selection(state, pos, pos);
        }
    }
}

/// Each frame while LMB is held, drag the head of any in-progress
/// text selection. Cleared on release.
fn handle_text_select_drag(
    windows: Query<&Window>,
    buttons: Res<ButtonInput<MouseButton>>,
    metrics: Option<Res<EditorMetrics>>,
    viewport: Res<pane_bevy::PaneViewport>,
    mut editors: Query<
        (
            &mut EditorStateComp,
            &PaneRect,
            &EditorScroll,
            &mut TextDragAnchor,
            &PaneKindMarker,
        ),
        With<PaneTag>,
    >,
) {
    let Some(metrics) = metrics else {
        return;
    };
    if buttons.just_released(MouseButton::Left) {
        for (_, _, _, mut drag, kind) in &mut editors {
            if kind.0 == PANE_KIND {
                drag.0 = None;
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

    for (mut state_comp, rect, scroll, drag, kind) in &mut editors {
        if kind.0 != PANE_KIND {
            continue;
        }
        let Some(anchor) = drag.0 else { continue };
        let local = pane_bevy::pt_to_content_local(pt_canvas, rect) + Vec2::new(scroll.x, scroll.y);
        let line = (local.y / LINE_HEIGHT).floor().max(0.0) as usize;
        let col = mouse_col_at_x(local.x, metrics.cell_width);
        let head = char_from_line_col(&state_comp.0, line, col);
        let cur = state_comp.0.selection.primary_range();
        if cur.anchor != anchor || cur.head != head {
            state_comp.0 = apply_selection(&state_comp.0, anchor, head);
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
