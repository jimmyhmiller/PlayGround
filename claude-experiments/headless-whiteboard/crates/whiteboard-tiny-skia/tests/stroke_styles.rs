//! Stroke-style rendering tests: a dashed/dotted stroke must leave gaps along
//! the line that a solid stroke would fill. Drives the full pipeline (element
//! stroke_style → `Stroke` dash pattern → `DrawCommand::StrokePath` → tiny-skia
//! dashed rasterization).

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind, LinearData};
use whiteboard_core::geometry::Point;
use whiteboard_core::render::{Backend, Color, StrokeStyle};
use whiteboard_core::text::MonospaceMeasurer;
use whiteboard_tiny_skia::TinySkiaBackend;

/// Render a horizontal line element with the given stroke style and count the
/// dark pixels along its row.
fn dark_pixels_along_line(style: StrokeStyle) -> usize {
    let mut editor = Editor::new(MonospaceMeasurer::default());
    // A straight horizontal line from (10,50) to (190,50), drawn cleanly
    // (roughness 0) so the only variable is the dash pattern.
    let data = LinearData::line(vec![Point::new(0.0, 0.0), Point::new(180.0, 0.0)]);
    let mut e = Element::new(
        ElementId::from("line"),
        1,
        10.0,
        50.0,
        180.0,
        0.0,
        ElementKind::Line(data),
    );
    e.stroke_color = Color::BLACK;
    e.stroke_width = 3.0;
    e.stroke_style = style;
    e.roughness = 0.0;
    editor.add_element(e);

    let mut backend = TinySkiaBackend::new(220, 100);
    backend.render(&editor.render());

    let px = backend.pixmap();
    // Scan a band of rows around y=50 and count dark pixels.
    let mut count = 0;
    for y in 47..=53 {
        for x in 0..220 {
            if let Some(p) = px.pixel(x, y) {
                if p.red() < 120 && p.green() < 120 && p.blue() < 120 {
                    count += 1;
                }
            }
        }
    }
    count
}

#[test]
fn solid_draws_more_ink_than_dashed() {
    let solid = dark_pixels_along_line(StrokeStyle::Solid);
    let dashed = dark_pixels_along_line(StrokeStyle::Dashed);
    assert!(solid > 0, "solid line must draw ink");
    assert!(dashed > 0, "dashed line must draw some ink");
    assert!(
        dashed < solid,
        "dashed line leaves gaps: dashed={dashed} should be < solid={solid}"
    );
}

#[test]
fn dotted_draws_less_ink_than_dashed() {
    // Dotted has a larger gap-to-on ratio than dashed, so even less ink.
    let dashed = dark_pixels_along_line(StrokeStyle::Dashed);
    let dotted = dark_pixels_along_line(StrokeStyle::Dotted);
    assert!(dotted > 0);
    assert!(
        dotted <= dashed,
        "dotted is at most as inky as dashed: dotted={dotted} dashed={dashed}"
    );
}
