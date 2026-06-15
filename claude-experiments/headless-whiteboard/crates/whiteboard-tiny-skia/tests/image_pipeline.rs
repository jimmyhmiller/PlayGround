//! End-to-end image pipeline: Editor::add_image places an image element, the
//! tessellator emits DrawImage, and the tiny-skia backend draws the registered
//! pixels into the destination rect.

use tiny_skia::Pixmap;
use whiteboard_core::editor::Editor;
use whiteboard_core::render::{Backend, Color};
use whiteboard_core::text::MonospaceMeasurer;
use whiteboard_tiny_skia::TinySkiaBackend;

#[test]
fn add_image_renders_registered_pixels() {
    let mut editor = Editor::new(MonospaceMeasurer::default());
    // Place an image at (20,20), 60x60, keyed "photo".
    let id = editor.add_image("photo", 20.0, 20.0, 60.0, 60.0);
    assert_eq!(id.as_str(), "img-photo");
    assert_eq!(editor.scene().iter_live().count(), 1);

    // A 4x4 solid-blue source image.
    let mut src = Pixmap::new(4, 4).unwrap();
    src.fill(tiny_skia::Color::from_rgba8(0, 0, 220, 255));

    let mut backend = TinySkiaBackend::new(100, 100).with_background(Color::WHITE);
    backend.register_image("photo", src);
    backend.render(&editor.render());

    // Center of the image rect (≈ 50,50) is blue.
    let px = backend.pixmap().pixel(50, 50).unwrap();
    assert!(
        px.blue() > 150 && px.red() < 80,
        "image drew blue pixels, got rgba({},{},{},{})",
        px.red(),
        px.green(),
        px.blue(),
        px.alpha()
    );

    // A corner outside the image stays background white.
    let corner = backend.pixmap().pixel(95, 95).unwrap();
    assert_eq!(
        (corner.red(), corner.green(), corner.blue()),
        (255, 255, 255)
    );
}

#[test]
fn add_image_is_undoable() {
    let mut editor = Editor::new(MonospaceMeasurer::default());
    editor.add_image("x", 0.0, 0.0, 10.0, 10.0);
    assert_eq!(editor.scene().iter_live().count(), 1);
    assert!(editor.undo());
    assert_eq!(editor.scene().iter_live().count(), 0);
}
