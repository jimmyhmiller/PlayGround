//! Integration tests that load the hand-authored real-format `.excalidraw`
//! fixtures from disk and assert their parsed shape, then confirm they render
//! to a non-empty [`RenderScene`].
//!
//! Fixtures live under `tests/fixtures/` and are located at runtime via
//! `CARGO_MANIFEST_DIR` so the test is independent of the working directory.

use std::path::PathBuf;

use whiteboard_core::editor::Editor;
use whiteboard_core::element::ElementKind;
use whiteboard_core::io::excalidraw::load_excalidraw_str;
use whiteboard_core::render::Color;
use whiteboard_core::text::{FontSpec, TextMeasurer, TextMetrics};
use whiteboard_core::ElementId;

/// A trivial measurer so the core tests don't need a font backend. Width is
/// proportional to character count; this is enough to exercise text layout.
struct StubMeasurer;

impl TextMeasurer for StubMeasurer {
    fn measure(&self, text: &str, font: &FontSpec) -> TextMetrics {
        let cols = text.chars().count() as f64;
        TextMetrics {
            width: cols * font.size * 0.5,
            ascent: font.size * 0.8,
            descent: font.size * 0.2,
        }
    }
}

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

fn load(name: &str) -> Vec<whiteboard_core::Element> {
    let path = fixture_path(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading fixture {}: {e}", path.display()));
    load_excalidraw_str(&text).unwrap_or_else(|e| panic!("parsing fixture {}: {e}", path.display()))
}

fn render_count(elements: &[whiteboard_core::Element]) -> usize {
    let mut editor = Editor::new_rough(StubMeasurer);
    for el in elements {
        editor.add_element(el.clone());
    }
    editor.render().commands.len()
}

#[test]
fn shapes_fixture_parses_and_renders() {
    let els = load("shapes.excalidraw");
    assert_eq!(els.len(), 5, "shapes.excalidraw element count");

    // Two rectangles, one ellipse, one arrow, one text.
    let kinds: Vec<&str> = els.iter().map(|e| e.type_name()).collect();
    assert_eq!(
        kinds,
        vec!["rectangle", "rectangle", "ellipse", "arrow", "text"]
    );

    // First rectangle: position, size, colors.
    let rect_a = &els[0];
    assert_eq!(rect_a.id, ElementId::new("rect-a"));
    assert_eq!(rect_a.x, 40.0);
    assert_eq!(rect_a.y, 40.0);
    assert_eq!(rect_a.width, 160.0);
    assert_eq!(rect_a.height, 90.0);
    assert_eq!(
        rect_a.background_color,
        Color::parse_hex("#a5d8ff").unwrap()
    );

    // Ellipse kind.
    assert!(matches!(els[2].kind, ElementKind::Ellipse));

    // Arrow: two points, bound at both ends.
    match &els[3].kind {
        ElementKind::Arrow(l) => {
            assert_eq!(l.points.len(), 2);
            assert_eq!(
                l.start_binding.as_ref().unwrap().element_id,
                ElementId::new("rect-a")
            );
            assert_eq!(
                l.end_binding.as_ref().unwrap().element_id,
                ElementId::new("rect-b")
            );
        }
        other => panic!("expected arrow, got {other:?}"),
    }

    // Text content.
    match &els[4].kind {
        ElementKind::Text(t) => assert_eq!(t.text, "Box A"),
        other => panic!("expected text, got {other:?}"),
    }

    assert!(
        render_count(&els) > 0,
        "shapes fixture should render a non-empty scene"
    );
}

#[test]
fn grouped_fixture_parses_and_renders() {
    let els = load("grouped.excalidraw");
    assert_eq!(els.len(), 2, "grouped.excalidraw element count");

    // Both elements share the same single group id.
    assert_eq!(els[0].group_ids.len(), 1);
    assert_eq!(els[1].group_ids.len(), 1);
    assert_eq!(
        els[0].group_ids, els[1].group_ids,
        "grouped elements share a group id"
    );

    assert_eq!(els[0].id, ElementId::new("g-rect"));
    assert!(matches!(els[0].kind, ElementKind::Rectangle));
    assert!(matches!(els[1].kind, ElementKind::Ellipse));

    assert!(
        render_count(&els) > 0,
        "grouped fixture should render a non-empty scene"
    );
}
