//! Headless integration tests for the KeyboardInput -> editor-command
//! glue in `handle_input`. These exercise the *translation* between
//! Bevy key events and editor-core commands — a layer that isn't
//! covered by editor-core's own tests.
//!
//! The test App uses `MinimalPlugins + InputPlugin + HeadlessEditorPlugin`
//! and spawns a single editor entity wired up as the focused target.
//! No window, no rendering, no fonts.

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::ButtonState;
use bevy::input::InputPlugin;
use bevy::prelude::*;
use editor_bevy::{
    Editor, EditorHighlighter, EditorRect, EditorScroll, EditorStateComp, FocusedEditor,
    HeadlessEditorPlugin, LineRows, MonoMetrics, TextDragAnchor,
};
use editor_bevy::highlight::Highlighter;
use editor_core::selection::Selection;
use editor_core::state::EditorState;
use ropey::Rope;

fn make_app(initial: &str) -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(InputPlugin);
    app.add_plugins(HeadlessEditorPlugin);
    // handle_input consults MonoMetrics for the caret-visible nudge;
    // tests never render, so a dummy cell width is fine.
    app.insert_resource(MonoMetrics { cell_width: 9.6 });
    let initial = initial.to_string();
    app.add_systems(Startup, move |mut commands: Commands| {
        let e = commands
            .spawn((
                Editor,
                EditorStateComp(
                    EditorState::new(Rope::from_str(&initial), Selection::cursor(0))
                        .with_indent_unit("    "),
                ),
                EditorHighlighter(Highlighter::new()),
                LineRows::default(),
                EditorScroll::default(),
                TextDragAnchor::default(),
                EditorRect {
                    pos: Vec2::ZERO,
                    size: Vec2::new(800.0, 600.0),
                    z: 0.0,
                },
            ))
            .id();
        commands.insert_resource(FocusedEditor(Some(e)));
    });
    // Run startup once so the entity exists before tests drive events.
    app.update();
    app
}

fn press(app: &mut App, key_code: KeyCode, logical: Key) {
    app.world_mut().write_message(KeyboardInput {
        key_code,
        logical_key: logical,
        state: ButtonState::Pressed,
        text: None,
        repeat: false,
        window: Entity::PLACEHOLDER,
    });
}

fn press_char(app: &mut App, c: char) {
    let mut buf = [0u8; 4];
    let s = c.encode_utf8(&mut buf);
    press(
        app,
        KeyCode::KeyA,
        Key::Character(s.into()),
    );
}

fn read_state(app: &mut App) -> EditorState {
    let mut q = app.world_mut().query::<&EditorStateComp>();
    q.single(app.world()).expect("one editor entity").0.clone()
}

#[test]
fn typing_a_char_inserts_at_caret() {
    let mut app = make_app("");
    press_char(&mut app, 'h');
    app.update();
    press_char(&mut app, 'i');
    app.update();
    let s = read_state(&mut app);
    assert_eq!(s.doc.to_string(), "hi");
    assert_eq!(s.selection.primary_range().head, 2);
}

#[test]
fn arrow_right_moves_caret() {
    let mut app = make_app("abc");
    press(&mut app, KeyCode::ArrowRight, Key::ArrowRight);
    app.update();
    assert_eq!(read_state(&mut app).selection.primary_range().head, 1);
}

#[test]
fn end_key_jumps_to_line_end() {
    let mut app = make_app("hello\nworld");
    press(&mut app, KeyCode::End, Key::End);
    app.update();
    assert_eq!(read_state(&mut app).selection.primary_range().head, 5);
}

#[test]
fn backspace_at_caret_deletes_prior_char() {
    let mut app = make_app("abc");
    press(&mut app, KeyCode::End, Key::End);
    app.update();
    press(&mut app, KeyCode::Backspace, Key::Backspace);
    app.update();
    let s = read_state(&mut app);
    assert_eq!(s.doc.to_string(), "ab");
    assert_eq!(s.selection.primary_range().head, 2);
}

#[test]
fn enter_inserts_newline_and_indents() {
    let mut app = make_app("    fn foo {");
    press(&mut app, KeyCode::End, Key::End);
    app.update();
    press(&mut app, KeyCode::Enter, Key::Enter);
    app.update();
    let s = read_state(&mut app);
    assert!(s.doc.to_string().contains('\n'));
    assert!(s.selection.primary_range().head > 12);
}

#[test]
fn arrow_keys_do_not_insert_text() {
    let mut app = make_app("");
    press(&mut app, KeyCode::ArrowRight, Key::ArrowRight);
    app.update();
    assert_eq!(read_state(&mut app).doc.to_string(), "");
}

#[test]
fn key_release_does_not_trigger_command() {
    let mut app = make_app("abc");
    app.world_mut().write_message(KeyboardInput {
        key_code: KeyCode::ArrowRight,
        logical_key: Key::ArrowRight,
        state: ButtonState::Released,
        text: None,
        repeat: false,
        window: Entity::PLACEHOLDER,
    });
    app.update();
    assert_eq!(read_state(&mut app).selection.primary_range().head, 0);
}
