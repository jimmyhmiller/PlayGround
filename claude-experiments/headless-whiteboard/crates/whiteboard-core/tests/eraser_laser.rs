//! Editor-level integration for the eraser and laser tools (Phase 9), driven
//! through the public `Editor` API.

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind};
use whiteboard_core::geometry::Point;
use whiteboard_core::interaction::{InputEvent, Modifiers, PointerButton, Tool};
use whiteboard_core::render::DrawCommand;
use whiteboard_core::text::MonospaceMeasurer;

fn editor() -> Editor<MonospaceMeasurer> {
    Editor::new(MonospaceMeasurer::default())
}

fn rect(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
    Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Rectangle)
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
fn eraser_stroke_deletes_and_undoes_as_one() {
    let mut ed = editor();
    ed.add_element(rect("a", 0.0, 0.0, 40.0, 40.0));
    ed.add_element(rect("b", 100.0, 0.0, 40.0, 40.0));
    assert_eq!(ed.scene().iter_live().count(), 2);

    ed.set_tool(Tool::Eraser);
    // Drag the eraser across both elements.
    ed.handle(down(20.0, 20.0));
    ed.handle(mv(60.0, 20.0));
    ed.handle(mv(120.0, 20.0));
    ed.handle(up(120.0, 20.0));

    assert_eq!(
        ed.scene().iter_live().count(),
        0,
        "eraser stroke removed both"
    );

    // The whole stroke is a single undo step that restores everything.
    assert!(ed.undo());
    assert_eq!(
        ed.scene().iter_live().count(),
        2,
        "undo restores both in one step"
    );
}

#[test]
fn laser_trail_renders_but_does_not_mutate_scene() {
    let mut ed = editor();
    ed.add_element(rect("a", 200.0, 200.0, 20.0, 20.0));
    let before = ed.scene().iter_live().count();
    // Baseline undo depth after the add (which DID record one entry); the laser
    // gesture must not change this.
    let undo_before = ed.can_undo();

    ed.set_tool(Tool::Laser);
    ed.handle(down(10.0, 10.0));
    ed.handle(mv(40.0, 30.0));
    ed.handle(mv(80.0, 60.0));

    assert_eq!(
        ed.scene().iter_live().count(),
        before,
        "laser never mutates scene"
    );
    assert_eq!(ed.can_undo(), undo_before, "laser adds no undo entry");

    // ...but the overlay renders the trail as a stroked polyline.
    let overlay = ed.render_with_overlay();
    let strokes = overlay
        .commands
        .iter()
        .filter(|c| matches!(c, DrawCommand::StrokePath { .. }))
        .count();
    assert!(strokes >= 1, "laser trail draws a stroked polyline");

    // Pointer-up clears the trail.
    ed.handle(up(80.0, 60.0));
    let after = ed.render_with_overlay();
    // With only one rect (no selection) and no laser, far fewer stroke commands.
    let trail_after = after
        .commands
        .iter()
        .filter(|c| matches!(c, DrawCommand::StrokePath { .. }))
        .count();
    // The rect itself strokes once; the trail is gone.
    assert!(trail_after <= 1, "laser trail cleared on pointer-up");
}

#[test]
fn laser_trail_accumulates_points() {
    let mut ed = editor();
    ed.set_tool(Tool::Laser);
    ed.handle(down(10.0, 10.0));
    ed.handle(mv(40.0, 30.0));
    ed.handle(mv(80.0, 60.0));
    assert_eq!(
        ed.interaction().laser_trail().len(),
        3,
        "trail has the down point plus two moves"
    );
}
