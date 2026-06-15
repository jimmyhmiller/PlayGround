//! Deterministic SVG output tests: a fixed command list must produce stable,
//! well-formed SVG covering every `DrawCommand` variant. Guards the SVG backend
//! against regressions in path-data, transforms, clipping, text, and images.

use whiteboard_core::geometry::{Path, Point, Rect, Transform};
use whiteboard_core::render::{
    Color, DrawCommand, ImageId, Paint, RenderScene, Stroke, StrokeStyle,
};
use whiteboard_core::text::{FontFamily, FontSpec, TextAlign, TextRun};
use whiteboard_svg::{to_svg, IMAGE_HREF_SCHEME};

fn tri(x: f64, y: f64) -> Path {
    Path::polygon(&[
        Point::new(x, y),
        Point::new(x + 20.0, y),
        Point::new(x + 10.0, y + 20.0),
    ])
}

/// A scene exercising every command kind.
fn full_scene() -> RenderScene {
    let mut s = RenderScene::new();
    s.push(DrawCommand::PushTransform(Transform::translate(5.0, 7.0)));
    s.push(DrawCommand::FillPath {
        path: tri(0.0, 0.0),
        paint: Paint::solid(Color::rgb(255, 0, 0)),
    });
    s.push(DrawCommand::StrokePath {
        path: tri(30.0, 0.0),
        stroke: Stroke::with_style(2.0, StrokeStyle::Dashed),
        paint: Paint::solid(Color::rgb(0, 0, 255)),
    });
    s.push(DrawCommand::PopTransform);
    s.push(DrawCommand::PushClip(Rect::new(0.0, 0.0, 50.0, 50.0)));
    s.push(DrawCommand::FillPath {
        path: tri(0.0, 30.0),
        paint: Paint::solid(Color::rgb(0, 200, 0)),
    });
    s.push(DrawCommand::PopClip);
    s.push(DrawCommand::DrawText {
        run: TextRun {
            text: "Hi <there>".to_string(),
            font: FontSpec::new(FontFamily::Normal, 16.0),
            origin: Point::new(10.0, 80.0),
            align: TextAlign::Left,
        },
        paint: Paint::solid(Color::rgb(20, 20, 20)),
    });
    s.push(DrawCommand::DrawImage {
        id: ImageId("img-1".to_string()),
        dst: Rect::new(60.0, 60.0, 40.0, 40.0),
        opacity: 0.5,
    });
    s
}

#[test]
fn svg_is_deterministic() {
    let scene = full_scene();
    let a = to_svg(&scene, 200, 120);
    let b = to_svg(&scene, 200, 120);
    assert_eq!(a, b, "same scene must produce identical SVG");
}

#[test]
fn svg_covers_every_command_kind() {
    let svg = to_svg(&full_scene(), 200, 120);

    // Envelope.
    assert!(svg.starts_with("<?xml"), "has XML prolog");
    assert!(svg.contains("<svg"), "has svg root");
    assert!(svg.trim_end().ends_with("</svg>"), "closed svg");
    assert!(svg.contains("width=\"200\"") && svg.contains("height=\"120\""));

    // Fill + stroke paths.
    assert!(svg.contains("fill=\"#ff0000\""), "red fill present");
    assert!(svg.contains("stroke=\"#0000ff\""), "blue stroke present");
    assert!(
        svg.contains("stroke-dasharray"),
        "dashed stroke emits dasharray"
    );
    assert!(svg.contains("fill=\"none\""), "stroke path has no fill");

    // Transform group.
    assert!(
        svg.contains("matrix(1,0,0,1,5,7)"),
        "translate transform group"
    );

    // Clip group.
    assert!(
        svg.contains("clipPath") && svg.contains("clip-path"),
        "clip applied"
    );

    // Text — XML-escaped.
    assert!(svg.contains("<text"), "text element");
    assert!(
        svg.contains("Hi &lt;there&gt;"),
        "text content is XML-escaped: {svg}"
    );

    // Image placeholder href.
    assert!(
        svg.contains(&format!("{IMAGE_HREF_SCHEME}img-1")),
        "image href placeholder present"
    );
}

#[test]
fn transform_groups_are_balanced() {
    let svg = to_svg(&full_scene(), 200, 120);
    let opens = svg.matches("<g").count();
    let closes = svg.matches("</g>").count();
    assert_eq!(
        opens, closes,
        "every <g> is closed ({opens} open, {closes} close)"
    );
}

#[test]
fn empty_scene_is_a_valid_empty_svg() {
    let svg = to_svg(&RenderScene::new(), 10, 10);
    assert!(svg.contains("<svg"));
    assert!(svg.trim_end().ends_with("</svg>"));
    assert!(!svg.contains("<path"), "no geometry for an empty scene");
}
