//! Text *entry* end to end: place a text element with the Text tool, type into
//! it via keyboard events, watch the box grow, commit, and confirm it persists.

use whiteboard_core::editor::Editor;
use whiteboard_core::element::ElementKind;
use whiteboard_core::geometry::Point;
use whiteboard_core::interaction::{InputEvent, Key, Modifiers, PointerButton, Tool};
use whiteboard_core::text::MonospaceMeasurer;

fn editor() -> Editor<MonospaceMeasurer> {
    Editor::new(MonospaceMeasurer::default())
}
fn key(c: char) -> InputEvent {
    InputEvent::KeyDown {
        key: Key::Char(c),
        mods: Modifiers::default(),
    }
}
fn special(k: Key) -> InputEvent {
    InputEvent::KeyDown {
        key: k,
        mods: Modifiers::default(),
    }
}
fn click(x: f64, y: f64) -> [InputEvent; 2] {
    [
        InputEvent::PointerDown {
            pos: Point::new(x, y),
            button: PointerButton::Primary,
            mods: Modifiers::default(),
        },
        InputEvent::PointerUp {
            pos: Point::new(x, y),
            button: PointerButton::Primary,
            mods: Modifiers::default(),
        },
    ]
}

fn text_of(ed: &Editor<MonospaceMeasurer>) -> Option<String> {
    ed.scene().iter_live().find_map(|e| match &e.kind {
        ElementKind::Text(t) => Some(t.text.clone()),
        _ => None,
    })
}

#[test]
fn place_type_and_commit_text() {
    let mut ed = editor();
    ed.set_tool(Tool::Text);
    for ev in click(40.0, 40.0) {
        ed.handle(ev);
    }
    // Placing text auto-enters edit mode.
    assert!(ed.is_editing_text(), "text placement starts editing");

    // Type "Hi".
    ed.handle(key('H'));
    ed.handle(key('i'));
    assert_eq!(text_of(&ed).as_deref(), Some("Hi"));

    // The box grew to fit the text (width > 0 once there are glyphs).
    let w = ed
        .scene()
        .iter_live()
        .find_map(|e| match &e.kind {
            ElementKind::Text(_) => Some(e.width),
            _ => None,
        })
        .unwrap();
    assert!(w > 0.0, "text box width tracks content: {w}");

    // Backspace removes the last char.
    ed.handle(special(Key::Backspace));
    assert_eq!(text_of(&ed).as_deref(), Some("H"));

    // Enter inserts a newline; more typing continues on the next line.
    ed.handle(special(Key::Enter));
    ed.handle(key('2'));
    assert_eq!(text_of(&ed).as_deref(), Some("H\n2"));

    // Clicking elsewhere commits the edit.
    for ev in click(300.0, 300.0) {
        ed.handle(ev);
    }
    assert!(!ed.is_editing_text(), "click elsewhere commits the edit");
    assert_eq!(
        text_of(&ed).as_deref(),
        Some("H\n2"),
        "committed text persists"
    );
}

#[test]
fn empty_text_is_discarded_on_commit() {
    let mut ed = editor();
    ed.set_tool(Tool::Text);
    for ev in click(10.0, 10.0) {
        ed.handle(ev);
    }
    assert!(ed.is_editing_text());
    // Commit without typing anything (Escape).
    ed.handle(special(Key::Escape));
    assert!(!ed.is_editing_text());
    assert_eq!(
        ed.scene().iter_live().count(),
        0,
        "empty text element is discarded (Excalidraw behavior)"
    );
}

#[test]
fn whole_text_edit_is_one_undo_step() {
    let mut ed = editor();
    ed.set_tool(Tool::Text);
    for ev in click(40.0, 40.0) {
        ed.handle(ev);
    }
    for c in "hello".chars() {
        ed.handle(key(c));
    }
    // Commit, then a single undo removes the whole text element.
    ed.handle(special(Key::Escape));
    assert_eq!(text_of(&ed).as_deref(), Some("hello"));
    assert!(ed.undo());
    assert_eq!(
        ed.scene().iter_live().count(),
        0,
        "the entire typing session undoes in one step"
    );
}
