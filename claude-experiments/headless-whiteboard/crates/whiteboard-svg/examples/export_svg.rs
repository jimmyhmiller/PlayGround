//! Export a headless-whiteboard scene to an SVG file — proves the same
//! `DrawCommand` list that the raster backends consume also drives a vector
//! format. No window, no GPU.
//!
//! Run: `cargo run -p whiteboard-svg --example export_svg -- out.svg`

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind, LinearData, TextData};
use whiteboard_core::geometry::Point;
use whiteboard_core::render::Color;
use whiteboard_core::text::MonospaceMeasurer;
use whiteboard_svg::to_svg;

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "whiteboard.svg".into());

    // Use the clean generator for crisp SVG vectors.
    let mut editor = Editor::new(MonospaceMeasurer::default());

    let mut rect = Element::new(
        ElementId::from("rect"),
        11,
        40.0,
        40.0,
        200.0,
        120.0,
        ElementKind::Rectangle,
    );
    rect.background_color = Color::rgb(255, 224, 178);
    rect.stroke_color = Color::rgb(230, 81, 0);
    rect.stroke_width = 2.0;
    rect.roughness = 0.0;
    editor.add_element(rect);

    let mut ell = Element::new(
        ElementId::from("ell"),
        29,
        300.0,
        50.0,
        180.0,
        110.0,
        ElementKind::Ellipse,
    );
    ell.stroke_color = Color::rgb(13, 71, 161);
    ell.stroke_width = 2.0;
    ell.roughness = 0.0;
    editor.add_element(ell);

    let arrow = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(160.0, 40.0)]);
    let mut arr = Element::new(
        ElementId::from("arr"),
        41,
        60.0,
        220.0,
        160.0,
        40.0,
        ElementKind::Arrow(arrow),
    );
    arr.stroke_color = Color::rgb(74, 20, 140);
    arr.stroke_width = 2.5;
    arr.roughness = 0.0;
    editor.add_element(arr);

    let mut text = TextData::new("vector export");
    text.font_size = 24.0;
    let mut t = Element::new(
        ElementId::from("txt"),
        53,
        300.0,
        210.0,
        0.0,
        30.0,
        ElementKind::Text(text),
    );
    t.stroke_color = Color::rgb(33, 33, 33);
    editor.add_element(t);

    let scene = editor.render();
    let svg = to_svg(&scene, 540, 320);
    std::fs::write(&out, &svg).expect("write svg");
    println!(
        "Wrote {} bytes of SVG ({} draw commands) to {out}",
        svg.len(),
        scene.commands.len()
    );
}
