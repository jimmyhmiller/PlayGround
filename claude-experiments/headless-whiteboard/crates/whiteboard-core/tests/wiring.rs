//! Integration tests for the Phase-7 features wired into the `Editor`:
//! selection-overlay rendering and arrow-binding follow-on-move.

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{compute_binding, Element, ElementId, ElementKind, LinearData};
use whiteboard_core::geometry::Point;
use whiteboard_core::interaction::{InputEvent, Modifiers, PointerButton, Tool};
use whiteboard_core::render::DrawCommand;
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

fn rect(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
    Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Rectangle)
}

#[test]
fn overlay_appears_only_with_a_selection() {
    let mut ed = editor();
    let id = ed.add_element(rect("r", 10.0, 10.0, 60.0, 60.0));

    // No selection: render_with_overlay == plain render.
    assert_eq!(
        ed.render_with_overlay().commands.len(),
        ed.render().commands.len(),
        "no overlay without a selection"
    );

    // Select: the overlay adds handle commands on top.
    ed.select([id]);
    let plain = ed.render().commands.len();
    let with_overlay = ed.render_with_overlay().commands.len();
    assert!(
        with_overlay > plain,
        "overlay adds commands: {with_overlay} > {plain}"
    );

    // The overlay includes filled handle squares (FillPath) beyond the scene.
    let fills = ed
        .render_with_overlay()
        .commands
        .iter()
        .filter(|c| matches!(c, DrawCommand::FillPath { .. }))
        .count();
    assert!(fills >= 8, "expected >=8 handle fills, got {fills}");
}

#[test]
fn marquee_drag_renders_a_marquee_overlay() {
    let mut ed = editor();
    ed.add_element(rect("r", 200.0, 200.0, 20.0, 20.0));
    ed.set_tool(Tool::Select);

    // Begin a marquee drag from empty space and hold (no pointer-up yet).
    ed.handle(down(5.0, 5.0));
    ed.handle(mv(80.0, 80.0));

    // Mid-drag, an overlay with the marquee rect should be present.
    let overlay = ed.render_with_overlay();
    let strokes = overlay
        .commands
        .iter()
        .filter(|c| matches!(c, DrawCommand::StrokePath { .. }))
        .count();
    assert!(strokes >= 1, "marquee draws a stroked rect");

    ed.handle(up(80.0, 80.0));
}

#[test]
fn bound_arrow_follows_moved_shape() {
    let mut ed = editor();

    // A target rectangle.
    let target = ed.add_element(rect("box", 200.0, 100.0, 80.0, 60.0));

    // An arrow whose END binds to the rectangle. Build the binding from the
    // arrow's current end point toward the target, then store it.
    let arrow_origin = Point::new(20.0, 120.0);
    let end_scene = Point::new(190.0, 130.0); // just left of the box
    let mut data = LinearData::arrow(vec![
        Point::new(0.0, 0.0),
        Point::new(end_scene.x - arrow_origin.x, end_scene.y - arrow_origin.y),
    ]);
    let target_el = ed.scene().get(&target).unwrap();
    data.end_binding = Some({
        let mut b = compute_binding(end_scene, target_el);
        b.element_id = target.clone();
        b
    });
    let arrow_id = ed.add_element(Element::new(
        ElementId::from("arrow"),
        2,
        arrow_origin.x,
        arrow_origin.y,
        end_scene.x - arrow_origin.x,
        10.0,
        ElementKind::Arrow(data),
    ));

    // Record the arrow's bound end in scene space before the move.
    let end_before = {
        let a = ed.scene().get(&arrow_id).unwrap();
        let ElementKind::Arrow(d) = &a.kind else {
            unreachable!()
        };
        let last = *d.points.last().unwrap();
        Point::new(a.x + last.x, a.y + last.y)
    };

    // Select the rectangle and drag it +100 in x.
    ed.set_tool(Tool::Select);
    ed.handle(down(240.0, 130.0)); // inside the box
    ed.handle(up(240.0, 130.0));
    ed.handle(down(240.0, 130.0));
    ed.handle(mv(290.0, 130.0));
    ed.handle(mv(340.0, 130.0));
    ed.handle(up(340.0, 130.0));

    // The box moved +100; the arrow's bound end should have tracked rightward.
    let end_after = {
        let a = ed.scene().get(&arrow_id).unwrap();
        let ElementKind::Arrow(d) = &a.kind else {
            unreachable!()
        };
        let last = *d.points.last().unwrap();
        Point::new(a.x + last.x, a.y + last.y)
    };

    assert!(
        end_after.x > end_before.x + 50.0,
        "bound arrow end followed the moved shape: {} -> {}",
        end_before.x,
        end_after.x
    );
}
