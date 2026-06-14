//! End-to-end integration tests driving the full pipeline:
//! raw `InputEvent`s → `Editor` → `InteractionState` → `Scene` mutations →
//! `RenderScene` draw commands. These exercise the cross-module seams wired up
//! in Phase 2 (interaction ↔ editor ↔ history ↔ render), not just unit behavior.

use whiteboard_core::editor::Editor;
use whiteboard_core::geometry::Point;
use whiteboard_core::interaction::{InputEvent, Modifiers, PointerButton, Tool};
use whiteboard_core::text::MonospaceMeasurer;

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

/// Drag with the Rectangle tool to create an element, then confirm it renders
/// and is undoable — the whole lifecycle through one public API.
#[test]
fn draw_rectangle_by_dragging() {
    let mut editor = Editor::new(MonospaceMeasurer::default());
    editor.set_tool(Tool::Rectangle);

    assert_eq!(editor.scene().iter_live().count(), 0);

    editor.handle(down(10.0, 10.0));
    editor.handle(mv(50.0, 40.0));
    editor.handle(mv(110.0, 60.0));
    editor.handle(up(110.0, 60.0));

    // One rectangle now exists, spanning the drag.
    let live: Vec<_> = editor.scene().iter_live().collect();
    assert_eq!(live.len(), 1, "drag should create exactly one element");
    let el = live[0];
    assert!((el.width - 100.0).abs() < 1e-6, "width={}", el.width);
    assert!((el.height - 50.0).abs() < 1e-6, "height={}", el.height);

    // It renders to draw commands.
    let scene = editor.render();
    assert!(!scene.is_empty(), "created rectangle must render");

    // The whole create gesture is a single undo step.
    assert!(editor.can_undo());
    assert!(editor.undo());
    assert_eq!(
        editor.scene().iter_live().count(),
        0,
        "undo removes the created rectangle"
    );
    assert!(editor.redo());
    assert_eq!(editor.scene().iter_live().count(), 1, "redo restores it");
}

/// A bare click with a creation tool must not leave a zero-size element, and
/// must not push an undo entry.
#[test]
fn bare_click_creates_nothing() {
    let mut editor = Editor::new(MonospaceMeasurer::default());
    editor.set_tool(Tool::Ellipse);

    editor.handle(down(20.0, 20.0));
    editor.handle(up(20.0, 20.0));

    assert_eq!(editor.scene().iter_live().count(), 0);
    assert!(!editor.can_undo(), "no-op click must not record undo");
}

/// The sketchy (rough) generator yields different geometry than the clean one
/// for the same scene, proving the rough path is actually wired into render.
#[test]
fn rough_and_clean_render_differently() {
    use whiteboard_core::element::{Element, ElementId, ElementKind};

    let mut clean = Editor::new(MonospaceMeasurer::default());
    let mut rough = Editor::new_rough(MonospaceMeasurer::default());

    let make = || {
        let mut e = Element::new(
            ElementId::from("r"),
            12345,
            0.0,
            0.0,
            100.0,
            60.0,
            ElementKind::Ellipse,
        );
        e.roughness = 1.5;
        e
    };
    clean.add_element(make());
    rough.add_element(make());

    let clean_cmds = clean.render().commands;
    let rough_cmds = rough.render().commands;
    assert_ne!(
        clean_cmds, rough_cmds,
        "rough generator must produce a different (sketchy) ellipse"
    );
}
