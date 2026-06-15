//! Regression tests for interactive Text and Frame creation — gaps found during
//! the feature-completeness audit (these tools previously created nothing).

use whiteboard_core::editor::Editor;
use whiteboard_core::element::ElementKind;
use whiteboard_core::geometry::Point;
use whiteboard_core::interaction::{InputEvent, Modifiers, PointerButton, Tool};
use whiteboard_core::text::MonospaceMeasurer;

fn editor() -> Editor<MonospaceMeasurer> {
    Editor::new(MonospaceMeasurer::default())
}
fn down(x: f64, y: f64) -> InputEvent {
    InputEvent::PointerDown {
        pos: Point::new(x, y),
        button: PointerButton::Primary,
        mods: Modifiers::default(),
    }
}
fn mv(x: f64, y: f64) -> InputEvent {
    InputEvent::PointerMove {
        pos: Point::new(x, y),
        mods: Modifiers::default(),
    }
}
fn up(x: f64, y: f64) -> InputEvent {
    InputEvent::PointerUp {
        pos: Point::new(x, y),
        button: PointerButton::Primary,
        mods: Modifiers::default(),
    }
}

#[test]
fn text_tool_click_places_one_text_element() {
    let mut ed = editor();
    ed.set_tool(Tool::Text);
    ed.handle(down(60.0, 40.0));
    ed.handle(up(60.0, 40.0));

    let texts: Vec<_> = ed
        .scene()
        .iter_live()
        .filter(|e| matches!(e.kind, ElementKind::Text(_)))
        .collect();
    assert_eq!(texts.len(), 1, "one text element placed");
    let t = texts[0];
    assert!(
        (t.x - 60.0).abs() < 1e-6 && (t.y - 40.0).abs() < 1e-6,
        "at the click point"
    );
    assert!(ed.selection().contains(&t.id), "the new text is selected");

    // Placement is a single undo step.
    assert!(ed.undo());
    assert_eq!(ed.scene().iter_live().count(), 0);
}

#[test]
fn frame_tool_drag_creates_a_frame() {
    let mut ed = editor();
    ed.set_tool(Tool::Frame);
    ed.handle(down(20.0, 20.0));
    ed.handle(mv(120.0, 90.0));
    ed.handle(mv(220.0, 160.0));
    ed.handle(up(220.0, 160.0));

    let frames: Vec<_> = ed
        .scene()
        .iter_live()
        .filter(|e| matches!(e.kind, ElementKind::Frame(_)))
        .collect();
    assert_eq!(frames.len(), 1, "one frame created");
    let f = frames[0];
    assert!((f.width - 200.0).abs() < 1e-6, "frame width = drag extent");
    assert!(
        (f.height - 140.0).abs() < 1e-6,
        "frame height = drag extent"
    );

    assert!(ed.undo());
    assert_eq!(
        ed.scene().iter_live().count(),
        0,
        "frame creation is undoable"
    );
}

#[test]
fn frame_click_without_drag_creates_nothing() {
    // A bare click with the frame tool (no drag) should not leave a zero-size frame.
    let mut ed = editor();
    ed.set_tool(Tool::Frame);
    ed.handle(down(50.0, 50.0));
    ed.handle(up(50.0, 50.0));
    assert_eq!(
        ed.scene().iter_live().count(),
        0,
        "bare frame click creates nothing"
    );
}
