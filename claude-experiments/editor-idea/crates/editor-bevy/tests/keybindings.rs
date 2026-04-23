//! Headless integration tests for the KeyboardInput -> editor-command
//! glue in `handle_input`. These exercise the *translation* between
//! Bevy key events and editor-core commands — a layer that isn't
//! covered by editor-core's own tests.
//!
//! The test App uses `MinimalPlugins + InputPlugin + HeadlessEditorPlugin`.
//! No window, no rendering, no fonts. Events are written directly to the
//! `KeyboardInput` message queue, then `app.update()` steps the schedule.

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::ButtonState;
use bevy::input::InputPlugin;
use bevy::prelude::*;
use editor_bevy::{EditorRes, HeadlessEditorPlugin};
use editor_core::selection::Selection;
use editor_core::state::EditorState;
use ropey::Rope;

fn make_app(initial: &str) -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(InputPlugin);
    app.insert_resource(EditorRes(
        EditorState::new(Rope::from_str(initial), Selection::cursor(0))
            .with_indent_unit("    "),
    ));
    app.add_plugins(HeadlessEditorPlugin);
    app
}

/// Push a key press event as if winit produced it. For the modifier
/// state used by `handle_input`, we also have to press the matching
/// `ButtonInput<KeyCode>` entries — `InputPlugin`'s keyboard system
/// populates those from `KeyboardInput` events in `PreUpdate`, so we
/// drive the same path.
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
        KeyCode::KeyA, // key_code isn't used for character fallback; any value works
        Key::Character(s.into()),
    );
}

fn state(app: &App) -> &EditorState {
    &app.world().resource::<EditorRes>().0
}

#[test]
fn typing_a_char_inserts_at_caret() {
    let mut app = make_app("");
    press_char(&mut app, 'h');
    app.update();
    press_char(&mut app, 'i');
    app.update();
    assert_eq!(state(&app).doc.to_string(), "hi");
    assert_eq!(state(&app).selection.primary_range().head, 2);
}

#[test]
fn arrow_right_moves_caret() {
    let mut app = make_app("abc");
    press(&mut app, KeyCode::ArrowRight, Key::ArrowRight);
    app.update();
    assert_eq!(state(&app).selection.primary_range().head, 1);
}

#[test]
fn end_key_jumps_to_line_end() {
    let mut app = make_app("hello\nworld");
    press(&mut app, KeyCode::End, Key::End);
    app.update();
    assert_eq!(state(&app).selection.primary_range().head, 5);
}

#[test]
fn backspace_at_caret_deletes_prior_char() {
    let mut app = make_app("abc");
    // Move to end first
    press(&mut app, KeyCode::End, Key::End);
    app.update();
    press(&mut app, KeyCode::Backspace, Key::Backspace);
    app.update();
    assert_eq!(state(&app).doc.to_string(), "ab");
    assert_eq!(state(&app).selection.primary_range().head, 2);
}

#[test]
fn enter_inserts_newline_and_indents() {
    // "    fn foo {" - after Enter past the `{`, should insert newline
    // and one more indent level (the indent-aware enter).
    let mut app = make_app("    fn foo {");
    press(&mut app, KeyCode::End, Key::End);
    app.update();
    press(&mut app, KeyCode::Enter, Key::Enter);
    app.update();
    let s = state(&app).doc.to_string();
    // Just assert the newline got inserted and caret advanced; exact
    // indent amount is editor-core's concern (we're testing the glue).
    assert!(s.contains('\n'));
    assert!(state(&app).selection.primary_range().head > 12);
}

#[test]
fn arrow_keys_do_not_insert_text() {
    // A regression-guard for an easy bug: if the fall-through to
    // character insertion doesn't filter out non-Character logical
    // keys, pressing ArrowRight on empty doc would insert text.
    let mut app = make_app("");
    press(&mut app, KeyCode::ArrowRight, Key::ArrowRight);
    app.update();
    assert_eq!(state(&app).doc.to_string(), "");
}

#[test]
fn key_release_does_not_trigger_command() {
    // handle_input ignores !ev.state.is_pressed() events.
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
    assert_eq!(state(&app).selection.primary_range().head, 0);
}
