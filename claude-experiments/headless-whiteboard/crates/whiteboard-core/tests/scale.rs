//! Scale / scaling-behavior tests: the headless core should handle large scenes
//! correctly (paint order, render command counts, hit-testing) without
//! pathological blowups. Not a benchmark — a correctness-at-scale guard.

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind};
use whiteboard_core::geometry::{element_bounds, hit_test, Point};
use whiteboard_core::text::MonospaceMeasurer;

fn grid_scene(n: usize) -> Editor<MonospaceMeasurer> {
    let mut editor = Editor::new(MonospaceMeasurer::default());
    let cols = 40;
    for i in 0..n {
        let col = (i % cols) as f64;
        let row = (i / cols) as f64;
        let mut e = Element::new(
            ElementId::from(format!("e{i}")),
            (i as u32).wrapping_mul(2654435761),
            col * 25.0,
            row * 25.0,
            20.0,
            20.0,
            ElementKind::Rectangle,
        );
        // Alternate clean/rough roughness to exercise both shape paths.
        e.roughness = if i % 2 == 0 { 0.0 } else { 1.0 };
        editor.add_element(e);
    }
    editor
}

#[test]
fn renders_a_thousand_elements() {
    let editor = grid_scene(1000);
    assert_eq!(editor.scene().iter_live().count(), 1000);

    let scene = editor.render();
    // Every rectangle emits at least an outline stroke; with transforms the
    // command count should be on the order of (and at least) the element count.
    assert!(
        scene.commands.len() >= 1000,
        "expected >=1000 commands, got {}",
        scene.commands.len()
    );
    // The scene bounds should cover the whole grid.
    assert!(scene.bounds.width > 0.0 && scene.bounds.height > 0.0);
}

#[test]
fn paint_order_is_stable_at_scale() {
    let editor = grid_scene(500);
    let order: Vec<_> = editor
        .scene()
        .iter_live()
        .map(|e| e.id.as_str().to_string())
        .collect();
    // Insertion order is paint order: e0, e1, ... e499.
    assert_eq!(order.first().unwrap(), "e0");
    assert_eq!(order.last().unwrap(), "e499");
    assert_eq!(order.len(), 500);
}

#[test]
fn hit_test_finds_the_right_element_in_a_grid() {
    let editor = grid_scene(200);
    // Element e0 sits at (0,0)-(20,20); its center (10,10) must hit e0 and not
    // its neighbor e1 at (25,0)-(45,20).
    let e0 = editor.scene().get(&ElementId::from("e0")).unwrap();
    let e1 = editor.scene().get(&ElementId::from("e1")).unwrap();
    assert!(hit_test(e0, Point::new(10.0, 10.0), 1.0));
    assert!(!hit_test(e1, Point::new(10.0, 10.0), 1.0));

    // Bounds of e1 start at x=25.
    let b1 = element_bounds(e1);
    assert!((b1.min_x() - 25.0).abs() < 1e-6, "e1 min_x={}", b1.min_x());
}

#[test]
fn deleting_half_leaves_the_rest_renderable() {
    let mut editor = grid_scene(100);
    // Select and delete every even-indexed element.
    let evens: Vec<_> = (0..100)
        .step_by(2)
        .map(|i| ElementId::from(format!("e{i}")))
        .collect();
    editor.select(evens);
    assert!(editor.delete_selection());
    assert_eq!(editor.scene().iter_live().count(), 50);
    assert!(!editor.render().is_empty());

    // Undo brings them all back.
    assert!(editor.undo());
    assert_eq!(editor.scene().iter_live().count(), 100);
}
