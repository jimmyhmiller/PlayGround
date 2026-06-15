//! Visual demo of frame clipping + the selection overlay, rendered headlessly to
//! a PNG. A frame contains two child rectangles, one of which extends past the
//! frame edge (and is clipped); a third element outside the frame is selected, so
//! the selection bounding box + resize/rotation handles draw over the scene.
//!
//! Run: `cargo run -p whiteboard-tiny-skia --example frames_overlay -- out.png`

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind, FrameData};
use whiteboard_core::render::{Backend, Color};
use whiteboard_core::text::MonospaceMeasurer;
use whiteboard_tiny_skia::TinySkiaBackend;

fn child(id: &str, x: f64, y: f64, w: f64, h: f64, frame: &str, fill: Color) -> Element {
    let mut e = Element::new(ElementId::from(id), 7, x, y, w, h, ElementKind::Rectangle);
    e.frame_id = Some(ElementId::from(frame));
    e.background_color = fill;
    e.fill_style = whiteboard_core::render::FillStyle::Solid;
    e.stroke_color = Color::rgb(40, 40, 40);
    e.roughness = 0.0;
    e
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "frames_overlay.png".into());

    let mut editor = Editor::new(MonospaceMeasurer::default());

    // The frame (drawn first so children paint on top, then clipped to it).
    let mut frame = Element::new(
        ElementId::from("frame"),
        1,
        40.0,
        40.0,
        260.0,
        200.0,
        ElementKind::Frame(FrameData {
            name: Some("Frame".into()),
        }),
    );
    frame.stroke_color = Color::rgb(120, 120, 120);
    frame.roughness = 0.0;
    editor.add_element(frame);

    // Child fully inside the frame.
    editor.add_element(child(
        "c1",
        70.0,
        70.0,
        90.0,
        70.0,
        "frame",
        Color::rgb(187, 222, 251),
    ));
    // Child extending past the right + bottom edges — should be CLIPPED to frame.
    editor.add_element(child(
        "c2",
        200.0,
        150.0,
        160.0,
        140.0,
        "frame",
        Color::rgb(255, 205, 210),
    ));

    // An element OUTSIDE the frame, which we select to show the overlay.
    let mut outside = Element::new(
        ElementId::from("sel"),
        3,
        360.0,
        80.0,
        120.0,
        90.0,
        ElementKind::Rectangle,
    );
    outside.background_color = Color::rgb(200, 230, 201);
    outside.fill_style = whiteboard_core::render::FillStyle::Solid;
    outside.stroke_color = Color::rgb(56, 142, 60);
    outside.roughness = 0.0;
    let sel_id = editor.add_element(outside);
    editor.select([sel_id]);

    // Render with the selection overlay on top.
    let scene = editor.render_with_overlay();
    let mut backend = TinySkiaBackend::new(520, 320).with_background(Color::WHITE);
    backend.render(&scene);
    backend.save_png(&out).expect("write png");
    println!("Rendered {} commands to {out}", scene.commands.len());
}
