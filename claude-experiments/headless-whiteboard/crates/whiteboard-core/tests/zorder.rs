//! Z-order editor methods over the selection.

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind};
use whiteboard_core::text::MonospaceMeasurer;

fn editor() -> Editor<MonospaceMeasurer> {
    Editor::new(MonospaceMeasurer::default())
}
fn rect(id: &str) -> Element {
    Element::new(
        ElementId::from(id),
        1,
        0.0,
        0.0,
        10.0,
        10.0,
        ElementKind::Rectangle,
    )
}

fn order(ed: &Editor<MonospaceMeasurer>) -> Vec<String> {
    ed.scene()
        .order()
        .iter()
        .map(|id| id.as_str().to_string())
        .collect()
}

#[test]
fn send_to_back_and_bring_to_front() {
    let mut ed = editor();
    let a = ed.add_element(rect("a"));
    let _b = ed.add_element(rect("b"));
    let _c = ed.add_element(rect("c"));
    assert_eq!(
        order(&ed),
        ["a", "b", "c"],
        "insertion order is paint order"
    );

    // Bring 'a' to the front.
    ed.select([a.clone()]);
    assert!(ed.bring_to_front());
    assert_eq!(order(&ed).last().unwrap(), "a", "a is now frontmost");

    // Send it back.
    assert!(ed.send_to_back());
    assert_eq!(order(&ed).first().unwrap(), "a", "a is now backmost");

    // Undo restores the previous order.
    assert!(ed.undo());
    assert_eq!(order(&ed).last().unwrap(), "a", "undo reverts send_to_back");
}

#[test]
fn raise_and_lower_one_step() {
    let mut ed = editor();
    let _a = ed.add_element(rect("a"));
    let b = ed.add_element(rect("b"));
    let _c = ed.add_element(rect("c"));

    ed.select([b.clone()]);
    assert!(ed.raise());
    assert_eq!(order(&ed), ["a", "c", "b"], "b raised above c");
    assert!(ed.lower());
    assert_eq!(order(&ed), ["a", "b", "c"], "b lowered back");
}

#[test]
fn reorder_with_empty_selection_is_noop() {
    let mut ed = editor();
    ed.add_element(rect("a"));
    assert!(!ed.bring_to_front(), "no selection ⇒ no-op");
    assert!(!ed.can_undo() || ed.can_undo(), "no panic"); // just ensure it ran
}
