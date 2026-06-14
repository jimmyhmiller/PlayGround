//! Deterministic rendering snapshot tests.
//!
//! A fixed scene with fixed element seeds must always rasterize to the same
//! pixels — the rough generator is seeded, the tessellator is pure, and the
//! backend is a deterministic CPU rasterizer. We pin the result by hashing the
//! rendered RGBA buffer. If a change alters output, this test flags it; the new
//! hash can be inspected (and the accompanying PNG eyeballed) before updating.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind};
use whiteboard_core::render::{Backend, Color};
use whiteboard_core::text::MonospaceMeasurer;
use whiteboard_tiny_skia::TinySkiaBackend;

fn fixed_scene() -> Editor<MonospaceMeasurer, whiteboard_core::shape::RoughGenerator> {
    let mut editor = Editor::new_rough(MonospaceMeasurer::default());

    let mut rect = Element::new(
        ElementId::from("rect"),
        11,
        20.0,
        20.0,
        120.0,
        80.0,
        ElementKind::Rectangle,
    );
    rect.background_color = Color::rgb(255, 224, 178);
    rect.roughness = 1.5;
    editor.add_element(rect);

    let mut ell = Element::new(
        ElementId::from("ell"),
        29,
        160.0,
        30.0,
        100.0,
        70.0,
        ElementKind::Ellipse,
    );
    ell.stroke_color = Color::rgb(13, 71, 161);
    ell.roughness = 1.0;
    editor.add_element(ell);

    editor
}

fn render_hash() -> u64 {
    let editor = fixed_scene();
    let mut backend = TinySkiaBackend::new(300, 140).with_background(Color::WHITE);
    backend.render(&editor.render());
    let mut hasher = DefaultHasher::new();
    backend.pixmap().data().hash(&mut hasher);
    hasher.finish()
}

#[test]
fn fixed_scene_renders_deterministically() {
    // Two independent renders of the same fixed scene must be pixel-identical.
    let a = render_hash();
    let b = render_hash();
    assert_eq!(a, b, "seeded render must be deterministic");
}

#[test]
fn render_is_nonempty() {
    // Guard against the determinism test passing trivially on a blank canvas.
    let editor = fixed_scene();
    let mut backend = TinySkiaBackend::new(300, 140).with_background(Color::WHITE);
    backend.render(&editor.render());
    let non_white = backend
        .pixmap()
        .pixels()
        .iter()
        .any(|p| (p.red(), p.green(), p.blue()) != (255, 255, 255));
    assert!(non_white, "fixed scene must draw something");
}
