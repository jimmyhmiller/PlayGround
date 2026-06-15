//! Interaction lifecycle integration tests, driven through the public `Editor`
//! API with synthetic input-event sequences. These complement the per-module
//! unit tests by exercising the full pipeline an application uses:
//! events → `InteractionState` → `Scene` mutations → selection / undo.

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind};
use whiteboard_core::geometry::Point;
use whiteboard_core::interaction::{InputEvent, Modifiers, PointerButton, Tool};
use whiteboard_core::text::MonospaceMeasurer;

type Ed = Editor<MonospaceMeasurer>;

fn editor() -> Ed {
    Editor::new(MonospaceMeasurer::default())
}

fn down(x: f64, y: f64) -> InputEvent {
    InputEvent::PointerDown {
        pos: Point::new(x, y),
        button: PointerButton::Primary,
        mods: Modifiers::default(),
    }
}
fn down_mods(x: f64, y: f64, mods: Modifiers) -> InputEvent {
    InputEvent::PointerDown {
        pos: Point::new(x, y),
        button: PointerButton::Primary,
        mods,
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

/// Draw an element of `tool` by dragging from `a` to `b`. Returns the editor's
/// only live element id afterward.
fn drag_create(ed: &mut Ed, tool: Tool, a: (f64, f64), b: (f64, f64)) -> ElementId {
    ed.set_tool(tool);
    ed.handle(down(a.0, a.1));
    ed.handle(mv((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0));
    ed.handle(mv(b.0, b.1));
    ed.handle(up(b.0, b.1));
    ed.scene()
        .iter_live()
        .last()
        .expect("an element was created")
        .id
        .clone()
}

#[test]
fn create_each_shape_tool() {
    for tool in [Tool::Rectangle, Tool::Ellipse, Tool::Diamond] {
        let mut ed = editor();
        drag_create(&mut ed, tool, (10.0, 10.0), (60.0, 50.0));
        assert_eq!(
            ed.scene().iter_live().count(),
            1,
            "{tool:?} should create one element"
        );
        let el = ed.scene().iter_live().next().unwrap();
        assert!((el.width - 50.0).abs() < 1e-6);
        assert!((el.height - 40.0).abs() < 1e-6);
    }
}

#[test]
fn click_selects_topmost_and_empty_click_deselects() {
    let mut ed = editor();
    let id = drag_create(&mut ed, Tool::Rectangle, (10.0, 10.0), (110.0, 110.0));

    ed.set_tool(Tool::Select);
    // Click inside the rectangle selects it.
    ed.handle(down(50.0, 50.0));
    ed.handle(up(50.0, 50.0));
    assert!(
        ed.selection().contains(&id),
        "click inside selects the element"
    );

    // Click far outside clears the selection.
    ed.handle(down(400.0, 400.0));
    ed.handle(up(400.0, 400.0));
    assert!(ed.selection().is_empty(), "empty-space click deselects");
}

#[test]
fn drag_moves_selected_element() {
    let mut ed = editor();
    let id = drag_create(&mut ed, Tool::Rectangle, (10.0, 10.0), (60.0, 60.0));
    let before = ed.scene().get(&id).unwrap().clone();

    ed.set_tool(Tool::Select);
    // Select, then drag from inside the element by (+40, +30).
    ed.handle(down(35.0, 35.0));
    ed.handle(up(35.0, 35.0));
    ed.handle(down(35.0, 35.0));
    ed.handle(mv(55.0, 50.0));
    ed.handle(mv(75.0, 65.0));
    ed.handle(up(75.0, 65.0));

    let after = ed.scene().get(&id).unwrap();
    assert!(
        (after.x - (before.x + 40.0)).abs() < 1e-6,
        "x moved by +40: {} -> {}",
        before.x,
        after.x
    );
    assert!(
        (after.y - (before.y + 30.0)).abs() < 1e-6,
        "y moved by +30: {} -> {}",
        before.y,
        after.y
    );
    // Size unchanged by a move.
    assert!((after.width - before.width).abs() < 1e-6);

    // The move is a single undo step.
    assert!(ed.undo());
    let undone = ed.scene().get(&id).unwrap();
    assert!((undone.x - before.x).abs() < 1e-6, "undo restores position");
}

#[test]
fn marquee_selects_enclosed_elements() {
    let mut ed = editor();
    let a = drag_create(&mut ed, Tool::Rectangle, (10.0, 10.0), (40.0, 40.0));
    let b = drag_create(&mut ed, Tool::Rectangle, (60.0, 60.0), (90.0, 90.0));
    let far = drag_create(&mut ed, Tool::Rectangle, (300.0, 300.0), (330.0, 330.0));

    ed.set_tool(Tool::Select);
    // Marquee from empty space enclosing a and b but not `far`.
    ed.handle(down(5.0, 5.0));
    ed.handle(mv(50.0, 50.0));
    ed.handle(mv(100.0, 100.0));
    ed.handle(up(100.0, 100.0));

    let sel = ed.selection();
    assert!(sel.contains(&a), "a enclosed by marquee");
    assert!(sel.contains(&b), "b enclosed by marquee");
    assert!(!sel.contains(&far), "far element not enclosed");
}

#[test]
fn shift_click_toggles_multi_selection() {
    let mut ed = editor();
    let a = drag_create(&mut ed, Tool::Rectangle, (10.0, 10.0), (40.0, 40.0));
    let b = drag_create(&mut ed, Tool::Rectangle, (60.0, 60.0), (90.0, 90.0));

    ed.set_tool(Tool::Select);
    ed.handle(down(25.0, 25.0));
    ed.handle(up(25.0, 25.0));
    assert!(ed.selection().contains(&a));

    let shift = Modifiers {
        shift: true,
        ..Default::default()
    };
    ed.handle(down_mods(75.0, 75.0, shift));
    ed.handle(InputEvent::PointerUp {
        pos: Point::new(75.0, 75.0),
        button: PointerButton::Primary,
        mods: shift,
    });
    assert!(ed.selection().contains(&a), "a stays selected");
    assert!(ed.selection().contains(&b), "shift-click adds b");
    assert_eq!(ed.selection().len(), 2);
}

#[test]
fn resize_via_handle_changes_size() {
    let mut ed = editor();
    let id = drag_create(&mut ed, Tool::Rectangle, (50.0, 50.0), (150.0, 150.0));
    let before = ed.scene().get(&id).unwrap().clone();

    ed.set_tool(Tool::Select);
    ed.handle(down(100.0, 100.0));
    ed.handle(up(100.0, 100.0));
    assert!(ed.selection().contains(&id));

    // Find the south-east handle's screen position and drag it outward.
    let vp = ed.viewport();
    let layout = ed
        .interaction()
        .handle_layout(ed.scene(), &vp)
        .expect("selection has a handle layout");
    use whiteboard_core::interaction::Handle;
    let se = layout.center(Handle::SouthEast);

    ed.handle(down(se.x, se.y));
    ed.handle(mv(se.x + 40.0, se.y + 40.0));
    ed.handle(up(se.x + 40.0, se.y + 40.0));

    let after = ed.scene().get(&id).unwrap();
    assert!(
        after.width > before.width + 30.0,
        "SE-handle drag grows width: {} -> {}",
        before.width,
        after.width
    );
    assert!(after.height > before.height + 30.0, "and height");

    // Resize is undoable in one step.
    assert!(ed.undo());
    let undone = ed.scene().get(&id).unwrap();
    assert!((undone.width - before.width).abs() < 1e-6);
}

#[test]
fn rotate_via_handle_sets_angle() {
    let mut ed = editor();
    let id = drag_create(&mut ed, Tool::Rectangle, (50.0, 50.0), (150.0, 150.0));
    assert_eq!(ed.scene().get(&id).unwrap().angle, 0.0);

    ed.set_tool(Tool::Select);
    ed.handle(down(100.0, 100.0));
    ed.handle(up(100.0, 100.0));

    let vp = ed.viewport();
    let layout = ed.interaction().handle_layout(ed.scene(), &vp).unwrap();
    use whiteboard_core::interaction::Handle;
    let rot = layout.center(Handle::Rotation);
    let pivot = layout.pivot();

    // Grab the rotation handle and swing it 90° around the pivot.
    ed.handle(down(rot.x, rot.y));
    // Move to a point rotated ~90° clockwise from the handle around the pivot.
    let dx = rot.x - pivot.x;
    let dy = rot.y - pivot.y;
    // 90° CW: (x,y) -> (-y, x)
    let target = Point::new(pivot.x - dy, pivot.y + dx);
    ed.handle(mv(target.x, target.y));
    ed.handle(up(target.x, target.y));

    let angle = ed.scene().get(&id).unwrap().angle;
    assert!(
        angle.abs() > 0.1,
        "rotation handle changed the angle: {angle}"
    );
}

#[test]
fn delete_then_undo_restores_multiple() {
    let mut ed = editor();
    let a = drag_create(&mut ed, Tool::Rectangle, (10.0, 10.0), (40.0, 40.0));
    let b = drag_create(&mut ed, Tool::Rectangle, (60.0, 60.0), (90.0, 90.0));
    ed.select([a.clone(), b.clone()]);
    assert!(ed.delete_selection());
    assert_eq!(ed.scene().iter_live().count(), 0);

    assert!(ed.undo());
    assert_eq!(ed.scene().iter_live().count(), 2, "undo restores both");
}

#[test]
fn pan_does_not_move_elements_in_scene_space() {
    let mut ed = editor();
    let id = drag_create(&mut ed, Tool::Rectangle, (10.0, 10.0), (60.0, 60.0));
    let scene_x = ed.scene().get(&id).unwrap().x;

    // Pan with the Pan tool.
    ed.set_tool(Tool::Pan);
    ed.handle(down(200.0, 200.0));
    ed.handle(mv(240.0, 230.0));
    ed.handle(up(240.0, 230.0));

    // The element's scene coordinates are unchanged; only the viewport moved.
    assert!((ed.scene().get(&id).unwrap().x - scene_x).abs() < 1e-6);
}

/// Programmatically added elements remain renderable after an event round-trip.
#[test]
fn programmatic_add_survives_event_loop() {
    let mut ed = editor();
    let e = Element::new(
        ElementId::from("p"),
        1,
        0.0,
        0.0,
        20.0,
        20.0,
        ElementKind::Rectangle,
    );
    ed.add_element(e);
    // A stray pointer move should not disturb the scene.
    ed.set_tool(Tool::Select);
    ed.handle(mv(500.0, 500.0));
    assert_eq!(ed.scene().iter_live().count(), 1);
    assert!(!ed.render().is_empty());
}
