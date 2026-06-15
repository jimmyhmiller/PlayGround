//! Integration tests for Phase-8 features wired into the `Editor`:
//! clipboard (copy/cut/paste), duplicate, group/ungroup, and frame membership
//! reassignment when an element is dragged into a frame.

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind, FrameData};
use whiteboard_core::geometry::{Point, Vec2};
use whiteboard_core::interaction::{InputEvent, Modifiers, PointerButton, Tool};
use whiteboard_core::text::MonospaceMeasurer;

fn editor() -> Editor<MonospaceMeasurer> {
    Editor::new(MonospaceMeasurer::default())
}

fn rect(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
    Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Rectangle)
}

#[test]
fn copy_paste_creates_offset_duplicates() {
    let mut ed = editor();
    let a = ed.add_element(rect("a", 10.0, 10.0, 40.0, 40.0));
    ed.select([a]);
    ed.copy();

    let pasted = ed.paste(Vec2::new(25.0, 15.0));
    assert_eq!(pasted.len(), 1, "one element pasted");
    assert_eq!(ed.scene().iter_live().count(), 2);

    // Pasted element is offset and has a fresh id; the paste becomes selected.
    let new = ed.scene().get(&pasted[0]).unwrap();
    assert!((new.x - 35.0).abs() < 1e-6, "x offset by +25");
    assert!((new.y - 25.0).abs() < 1e-6, "y offset by +15");
    assert!(ed.selection().contains(&pasted[0]));

    // Paste is undoable in one step.
    assert!(ed.undo());
    assert_eq!(ed.scene().iter_live().count(), 1);
}

#[test]
fn cut_removes_then_paste_restores() {
    let mut ed = editor();
    let a = ed.add_element(rect("a", 0.0, 0.0, 20.0, 20.0));
    ed.select([a]);
    assert!(ed.cut());
    assert_eq!(ed.scene().iter_live().count(), 0, "cut removes the element");

    let pasted = ed.paste(Vec2::new(5.0, 5.0));
    assert_eq!(pasted.len(), 1, "cut content can be pasted back");
}

#[test]
fn duplicate_selection_nudges_and_selects() {
    let mut ed = editor();
    let a = ed.add_element(rect("a", 100.0, 100.0, 30.0, 30.0));
    ed.select([a.clone()]);

    let dups = ed.duplicate_selection();
    assert_eq!(dups.len(), 1);
    assert_eq!(ed.scene().iter_live().count(), 2);
    let dup = ed.scene().get(&dups[0]).unwrap();
    assert!((dup.x - 110.0).abs() < 1e-6, "nudged +10 in x");
    assert!(ed.selection().contains(&dups[0]));
    assert!(
        !ed.selection().contains(&a),
        "selection moves to the duplicate"
    );
}

#[test]
fn group_and_ungroup_round_trip() {
    let mut ed = editor();
    let a = ed.add_element(rect("a", 0.0, 0.0, 10.0, 10.0));
    let b = ed.add_element(rect("b", 20.0, 0.0, 10.0, 10.0));
    ed.select([a.clone(), b.clone()]);

    assert!(ed.group_selection(), "two elements form a group");
    let ga = ed.scene().get(&a).unwrap().group_ids.clone();
    let gb = ed.scene().get(&b).unwrap().group_ids.clone();
    assert!(!ga.is_empty() && ga == gb, "both share the new group id");

    // Clicking one member expands selection to the whole group.
    let expanded = ed.expand_to_groups(std::slice::from_ref(&a));
    assert!(expanded.contains(&a) && expanded.contains(&b));

    // Ungroup removes the shared group.
    ed.select([a.clone(), b.clone()]);
    assert!(ed.ungroup_selection());
    assert!(ed.scene().get(&a).unwrap().group_ids.is_empty());
    assert!(ed.scene().get(&b).unwrap().group_ids.is_empty());
}

#[test]
fn group_needs_two_elements() {
    let mut ed = editor();
    let a = ed.add_element(rect("a", 0.0, 0.0, 10.0, 10.0));
    ed.select([a]);
    assert!(!ed.group_selection(), "a single element cannot be grouped");
}

#[test]
fn dragging_into_a_frame_assigns_membership() {
    let mut ed = editor();
    // A frame at (200,200) 200x200.
    ed.add_element(Element::new(
        ElementId::from("frame"),
        1,
        200.0,
        200.0,
        200.0,
        200.0,
        ElementKind::Frame(FrameData { name: None }),
    ));
    // A small rectangle starting OUTSIDE the frame.
    let r = ed.add_element(rect("r", 20.0, 20.0, 30.0, 30.0));
    assert!(ed.scene().get(&r).unwrap().frame_id.is_none());

    // Select and drag it fully inside the frame.
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
    let up = |x: f64, y: f64| InputEvent::PointerUp {
        pos: Point::new(x, y),
        button: PointerButton::Primary,
        mods: Modifiers::default(),
    };
    // Click to select, then drag from inside the rect (35,35) to (285,285):
    // delta (+250,+250) puts the 30x30 rect at (270,270)-(300,300), inside frame.
    ed.handle(down(35.0, 35.0));
    ed.handle(up(35.0, 35.0));
    ed.handle(down(35.0, 35.0));
    ed.handle(mv(160.0, 160.0));
    ed.handle(mv(285.0, 285.0));
    ed.handle(up(285.0, 285.0));

    let moved = ed.scene().get(&r).unwrap();
    assert!(
        moved.x > 200.0,
        "rect moved inside the frame: x={}",
        moved.x
    );
    assert_eq!(
        moved.frame_id,
        Some(ElementId::from("frame")),
        "dragged into the frame ⇒ frame membership assigned"
    );
}
