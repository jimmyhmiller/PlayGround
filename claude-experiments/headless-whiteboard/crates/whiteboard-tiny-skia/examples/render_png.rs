//! Headless render smoke test: build a scene with several sketchy elements, run
//! it through the editor's render pass, rasterize via the tiny-skia backend, and
//! write a PNG. No window required — proves the headless → backend → pixels path.
//!
//! Run: `cargo run -p whiteboard-tiny-skia --example render_png -- out.png`

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{
    Element, ElementId, ElementKind, FreedrawData, LinearData, TextData,
};
use whiteboard_core::geometry::Point;
use whiteboard_core::render::{Backend, Color};
use whiteboard_tiny_skia::{FontMeasurer, TinySkiaBackend};

fn text_el(id: &str, x: f64, y: f64, s: &str, size: f64, color: Color) -> Element {
    let mut data = TextData::new(s);
    data.font_size = size;
    let mut e = Element::new(
        ElementId::from(id),
        7,
        x,
        y,
        s.len() as f64 * size * 0.6,
        size * 1.25,
        ElementKind::Text(data),
    );
    e.stroke_color = color;
    e
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "whiteboard.png".into());

    // Use the real font-backed measurer so layout matches what is rasterized.
    let mut editor = Editor::new_rough(FontMeasurer::new());

    // A filled rough rectangle.
    let mut rect = Element::new(
        ElementId::from("rect"),
        11,
        60.0,
        60.0,
        220.0,
        140.0,
        ElementKind::Rectangle,
    );
    rect.background_color = Color::rgb(255, 224, 178);
    rect.stroke_color = Color::rgb(230, 81, 0);
    rect.roughness = 1.5;
    rect.stroke_width = 2.0;
    editor.add_element(rect);

    // A hachure-filled ellipse.
    let mut ellipse = Element::new(
        ElementId::from("ell"),
        29,
        330.0,
        70.0,
        200.0,
        120.0,
        ElementKind::Ellipse,
    );
    ellipse.background_color = Color::rgb(187, 222, 251);
    ellipse.stroke_color = Color::rgb(13, 71, 161);
    ellipse.roughness = 1.0;
    ellipse.stroke_width = 2.0;
    editor.add_element(ellipse);

    // A diamond.
    let mut diamond = Element::new(
        ElementId::from("dia"),
        47,
        120.0,
        260.0,
        160.0,
        120.0,
        ElementKind::Diamond,
    );
    diamond.stroke_color = Color::rgb(56, 142, 60);
    diamond.roughness = 2.0;
    diamond.stroke_width = 2.0;
    editor.add_element(diamond);

    // An arrow.
    let arrow_data = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(180.0, 60.0)]);
    let mut arrow = Element::new(
        ElementId::from("arr"),
        53,
        360.0,
        260.0,
        180.0,
        60.0,
        ElementKind::Arrow(arrow_data),
    );
    arrow.stroke_color = Color::rgb(74, 20, 140);
    arrow.roughness = 1.0;
    arrow.stroke_width = 2.5;
    editor.add_element(arrow);

    // A freedraw squiggle.
    let pts: Vec<Point> = (0..40)
        .map(|i| {
            let t = i as f64 / 6.0;
            Point::new(t * 14.0, (t).sin() * 30.0)
        })
        .collect();
    let mut free = Element::new(
        ElementId::from("free"),
        61,
        90.0,
        430.0,
        560.0,
        80.0,
        ElementKind::Freedraw(FreedrawData::new(pts)),
    );
    free.stroke_color = Color::rgb(33, 33, 33);
    free.stroke_width = 2.0;
    editor.add_element(free);

    // Text labels — now rasterized with real glyphs.
    editor.add_element(text_el(
        "title",
        60.0,
        18.0,
        "headless-whiteboard",
        26.0,
        Color::rgb(33, 33, 33),
    ));
    editor.add_element(text_el(
        "label-rect",
        95.0,
        118.0,
        "rectangle",
        18.0,
        Color::rgb(230, 81, 0),
    ));
    editor.add_element(text_el(
        "label-ell",
        365.0,
        120.0,
        "ellipse",
        18.0,
        Color::rgb(13, 71, 161),
    ));

    // Render headlessly and rasterize.
    let scene = editor.render();
    let mut backend = TinySkiaBackend::new(680, 560).with_background(Color::WHITE);
    backend.render(&scene);

    backend.save_png(&out).expect("write png");
    println!("Rendered {} draw commands to {out}", scene.commands.len());
}
