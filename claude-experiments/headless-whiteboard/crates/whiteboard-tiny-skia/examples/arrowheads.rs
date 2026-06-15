//! Showcase of arrowhead variants (filled vs outline), rendered headlessly.
//! Run: `cargo run -p whiteboard-tiny-skia --example arrowheads -- out.png`

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Arrowhead, Element, ElementId, ElementKind, LinearData};
use whiteboard_core::geometry::Point;
use whiteboard_core::render::{Backend, Color};
use whiteboard_core::text::MonospaceMeasurer;
use whiteboard_tiny_skia::TinySkiaBackend;

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "arrowheads.png".into());
    let mut editor = Editor::new(MonospaceMeasurer::default());

    let heads = [
        Arrowhead::Arrow,
        Arrowhead::Triangle,
        Arrowhead::TriangleOutline,
        Arrowhead::Bar,
        Arrowhead::Dot,
        Arrowhead::CircleOutline,
        Arrowhead::Diamond,
        Arrowhead::DiamondOutline,
        Arrowhead::Crowfoot,
    ];

    for (i, head) in heads.iter().enumerate() {
        let y = 30.0 + i as f64 * 40.0;
        let mut data = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(180.0, 0.0)]);
        data.end_arrowhead = Some(*head);
        let mut e = Element::new(
            ElementId::from(format!("a{i}")),
            1,
            40.0,
            y,
            180.0,
            0.0,
            ElementKind::Arrow(data),
        );
        e.stroke_color = Color::rgb(30, 30, 30);
        e.stroke_width = 2.0;
        e.roughness = 0.0;
        editor.add_element(e);
    }

    let scene = editor.render();
    let mut backend = TinySkiaBackend::new(280, 400).with_background(Color::WHITE);
    backend.render(&scene);
    backend.save_png(&out).expect("write png");
    println!("Rendered {} arrowhead variants to {out}", heads.len());
}
