//! Editor-level integration for Phase-11: filled arrowheads render a stroke-color
//! fill, and object snapping surfaces alignment guides in the overlay.

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Arrowhead, Element, ElementId, ElementKind, LinearData};
use whiteboard_core::geometry::Point;
use whiteboard_core::interaction::{InputEvent, Modifiers, PointerButton, Tool};
use whiteboard_core::render::{Color, DrawCommand, Paint};
use whiteboard_core::text::MonospaceMeasurer;

fn editor() -> Editor<MonospaceMeasurer> {
    Editor::new(MonospaceMeasurer::default())
}

fn rect(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
    Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Rectangle)
}

#[test]
fn filled_triangle_arrowhead_emits_stroke_colored_fill() {
    let mut ed = editor();
    let mut data = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(100.0, 0.0)]);
    data.end_arrowhead = Some(Arrowhead::Triangle); // filled
    let stroke = Color::rgb(200, 30, 30);
    let mut e = Element::new(
        ElementId::from("arr"),
        1,
        10.0,
        10.0,
        100.0,
        0.0,
        ElementKind::Arrow(data),
    );
    e.stroke_color = stroke;
    e.roughness = 0.0;
    ed.add_element(e);

    let scene = ed.render();
    // There must be a FillPath painted with the STROKE color (the filled head),
    // even though the arrow has no background.
    let has_stroke_fill = scene.commands.iter().any(
        |c| matches!(c, DrawCommand::FillPath { paint: Paint::Solid(col), .. } if *col == stroke),
    );
    assert!(
        has_stroke_fill,
        "filled triangle head fills with the stroke color"
    );
}

#[test]
fn outline_triangle_arrowhead_has_no_stroke_fill() {
    let mut ed = editor();
    let mut data = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(100.0, 0.0)]);
    data.end_arrowhead = Some(Arrowhead::TriangleOutline); // outline only
    let stroke = Color::rgb(10, 60, 200);
    let mut e = Element::new(
        ElementId::from("arr"),
        1,
        10.0,
        10.0,
        100.0,
        0.0,
        ElementKind::Arrow(data),
    );
    e.stroke_color = stroke;
    e.roughness = 0.0;
    ed.add_element(e);

    let scene = ed.render();
    let has_stroke_fill = scene.commands.iter().any(
        |c| matches!(c, DrawCommand::FillPath { paint: Paint::Solid(col), .. } if *col == stroke),
    );
    assert!(!has_stroke_fill, "outline head is stroked, not filled");
}

#[test]
fn moving_near_another_element_snaps_and_shows_a_guide() {
    let mut ed = editor();
    // A stationary reference rect with left edge at x = 100.
    ed.add_element(rect("ref", 100.0, 200.0, 60.0, 60.0));
    // A rect whose left edge (x = 97) is 3px from the reference's left edge.
    let mover = ed.add_element(rect("m", 0.0, 0.0, 40.0, 40.0));

    ed.set_tool(Tool::Select);
    let down = |x: f64, y: f64| InputEvent::PointerDown {
        pos: Point::new(x, y),
        button: PointerButton::Primary,
        mods: Modifiers::default(),
    };
    let mv = |x: f64, y: f64| InputEvent::PointerMove {
        pos: Point::new(x, y),
        mods: Modifiers::default(),
    };

    // Select the mover, then drag it so its left edge lands at x≈97 (3px from 100).
    ed.handle(down(20.0, 20.0)); // inside mover at (0,0)-(40,40)
    ed.handle(InputEvent::PointerUp {
        pos: Point::new(20.0, 20.0),
        button: PointerButton::Primary,
        mods: Modifiers::default(),
    });
    ed.handle(down(20.0, 20.0));
    // Pointer is 20px inside the box, so box-left = pointer_x - 20. Drag the
    // pointer to x=123 ⇒ box-left ≈ 103, within the 5px snap threshold of the
    // reference's left edge at x=100, so it snaps to exactly 100.
    ed.handle(mv(123.0, 220.0));

    // Mid-drag, with the left edge within snap threshold of x=100, a guide shows
    // and the box snaps to exact alignment.
    let moved = ed.scene().get(&mover).unwrap();
    assert!(
        (moved.x - 100.0).abs() < 1e-6,
        "snapped to the reference left edge (x=100), got {}",
        moved.x
    );

    let overlay = ed.render_with_overlay();
    // A thin pink guide stroke is present.
    let pink = Color::rgb(255, 0, 200);
    let has_guide = overlay.commands.iter().any(
        |c| matches!(c, DrawCommand::StrokePath { paint: Paint::Solid(col), .. } if *col == pink),
    );
    assert!(has_guide, "an alignment guide is rendered in the overlay");
}
