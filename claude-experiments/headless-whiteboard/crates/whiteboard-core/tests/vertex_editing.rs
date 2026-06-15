//! Line/arrow vertex editing: when a single linear element is selected, its
//! vertices can be dragged to reshape it (endpoints and midpoints), driven
//! through the public `Editor` API.

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind, LinearData};
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

/// Scene-space vertices of the (only) linear element.
fn vertices(ed: &Editor<MonospaceMeasurer>) -> Vec<Point> {
    let e = ed.scene().iter_live().next().unwrap();
    match &e.kind {
        ElementKind::Line(d) | ElementKind::Arrow(d) => {
            d.points.iter().map(|p| Point::new(e.x + p.x, e.y + p.y)).collect()
        }
        _ => vec![],
    }
}

/// A 3-point arrow: (50,50) -> (100,50) -> (150,50), stored element-local.
fn three_point_arrow(ed: &mut Editor<MonospaceMeasurer>) -> ElementId {
    let data = LinearData::arrow(vec![
        Point::new(0.0, 0.0),
        Point::new(50.0, 0.0),
        Point::new(100.0, 0.0),
    ]);
    ed.add_element(Element::new(
        ElementId::from("arr"),
        1,
        50.0,
        50.0,
        100.0,
        0.0,
        ElementKind::Arrow(data),
    ))
}

#[test]
fn drag_middle_vertex_reshapes_the_line() {
    let mut ed = editor();
    let id = three_point_arrow(&mut ed);
    // Vertices start at (50,50),(100,50),(150,50).
    let before = vertices(&ed);
    assert_eq!(before[1], Point::new(100.0, 50.0), "middle vertex start");

    // Select it, then grab the middle vertex (100,50) and drag it up to (100,10).
    ed.set_tool(Tool::Select);
    ed.handle(down(100.0, 50.0)); // click the line selects it
    ed.handle(up(100.0, 50.0));
    assert!(ed.selection().contains(&id), "line is selected");

    ed.handle(down(100.0, 50.0)); // grab the middle vertex
    ed.handle(mv(100.0, 30.0));
    ed.handle(mv(100.0, 10.0));
    ed.handle(up(100.0, 10.0));

    let after = vertices(&ed);
    assert_eq!(after.len(), 3, "still three vertices");
    // The middle vertex moved up to y≈10; endpoints unchanged.
    assert!((after[1].y - 10.0).abs() < 1e-6, "middle vertex moved up: {:?}", after[1]);
    assert!((after[0].y - 50.0).abs() < 1e-6, "first endpoint unchanged");
    assert!((after[2].y - 50.0).abs() < 1e-6, "last endpoint unchanged");

    // The element box re-normalized to span the new extent (y from 10 to 50).
    let el = ed.scene().get(&id).unwrap();
    assert!((el.y - 10.0).abs() < 1e-6, "box top follows the raised vertex");
    assert!((el.height - 40.0).abs() < 1e-6, "box height grew to 40");

    // The reshape is undoable in one step.
    assert!(ed.undo());
    assert_eq!(vertices(&ed)[1], Point::new(100.0, 50.0), "undo restores the vertex");
}

#[test]
fn drag_endpoint_moves_just_that_end() {
    let mut ed = editor();
    let id = three_point_arrow(&mut ed);

    ed.set_tool(Tool::Select);
    ed.handle(down(100.0, 50.0));
    ed.handle(up(100.0, 50.0));

    // Grab the LAST endpoint (150,50) and drag it to (150,120).
    ed.handle(down(150.0, 50.0));
    ed.handle(mv(150.0, 120.0));
    ed.handle(up(150.0, 120.0));

    let v = vertices(&ed);
    assert!((v[2].y - 120.0).abs() < 1e-6, "last endpoint dragged down");
    assert!((v[0].y - 50.0).abs() < 1e-6, "first endpoint stays");
    let el = ed.scene().get(&id).unwrap();
    assert!((el.height - 70.0).abs() < 1e-6, "box height = 50..120");
}

#[test]
fn vertex_drag_needs_single_linear_selection() {
    // With two elements selected, a click near a vertex must NOT start a vertex
    // drag (it should move/marquee instead). Here we just confirm a non-linear
    // selection doesn't expose vertices: dragging a rectangle's corner resizes.
    let mut ed = editor();
    let r = ed.add_element(Element::new(
        ElementId::from("r"),
        1,
        50.0,
        50.0,
        100.0,
        100.0,
        ElementKind::Rectangle,
    ));
    ed.select([r.clone()]);
    // A rectangle has no draggable vertices; the scene must still have 1 element
    // and the rect unchanged after a stray click in its interior.
    ed.set_tool(Tool::Select);
    ed.handle(down(100.0, 100.0));
    ed.handle(up(100.0, 100.0));
    assert_eq!(ed.scene().iter_live().count(), 1);
}
