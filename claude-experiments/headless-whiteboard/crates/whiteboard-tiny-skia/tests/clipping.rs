//! The tiny-skia backend must honor `PushClip`/`PopClip`: geometry drawn while a
//! clip is active is masked to the clip rect. This guards the frame-clipping path
//! end to end (a fill that overflows a clip is cut off at the clip boundary).

use whiteboard_core::geometry::{Path, Point, Rect};
use whiteboard_core::render::{Backend, Color, DrawCommand, Paint, RenderScene};
use whiteboard_tiny_skia::TinySkiaBackend;

fn filled_rect_cmd(x: f64, y: f64, w: f64, h: f64, color: Color) -> DrawCommand {
    DrawCommand::FillPath {
        path: Path::polygon(&[
            Point::new(x, y),
            Point::new(x + w, y),
            Point::new(x + w, y + h),
            Point::new(x, y + h),
        ]),
        paint: Paint::solid(color),
    }
}

#[test]
fn fill_is_clipped_to_push_clip_rect() {
    // Clip to the left half, then fill the whole canvas red.
    let mut scene = RenderScene::new();
    scene.push(DrawCommand::PushClip(Rect::new(0.0, 0.0, 50.0, 100.0)));
    scene.push(filled_rect_cmd(
        0.0,
        0.0,
        100.0,
        100.0,
        Color::rgb(220, 20, 20),
    ));
    scene.push(DrawCommand::PopClip);

    let mut backend = TinySkiaBackend::new(100, 100).with_background(Color::WHITE);
    backend.render(&scene);
    let px = backend.pixmap();

    // Inside the clip (left): red.
    let left = px.pixel(20, 50).unwrap();
    assert!(
        left.red() > 150 && left.green() < 80,
        "left half is filled red"
    );

    // Outside the clip (right): untouched white background.
    let right = px.pixel(80, 50).unwrap();
    assert_eq!(
        (right.red(), right.green(), right.blue()),
        (255, 255, 255),
        "right half is clipped away (stays background)"
    );
}

#[test]
fn pop_clip_restores_unclipped_drawing() {
    // Fill after PopClip should NOT be clipped.
    let mut scene = RenderScene::new();
    scene.push(DrawCommand::PushClip(Rect::new(0.0, 0.0, 10.0, 10.0)));
    scene.push(DrawCommand::PopClip);
    scene.push(filled_rect_cmd(
        0.0,
        0.0,
        100.0,
        100.0,
        Color::rgb(0, 0, 220),
    ));

    let mut backend = TinySkiaBackend::new(100, 100).with_background(Color::WHITE);
    backend.render(&scene);

    // A far corner outside the popped clip is still filled blue.
    let px = backend.pixmap().pixel(90, 90).unwrap();
    assert!(px.blue() > 150, "unclipped fill covers the whole canvas");
}

#[test]
fn nested_clips_intersect() {
    // Two overlapping clips: only their intersection is drawable.
    let mut scene = RenderScene::new();
    scene.push(DrawCommand::PushClip(Rect::new(0.0, 0.0, 60.0, 100.0))); // left 60
    scene.push(DrawCommand::PushClip(Rect::new(40.0, 0.0, 60.0, 100.0))); // right 60
    scene.push(filled_rect_cmd(
        0.0,
        0.0,
        100.0,
        100.0,
        Color::rgb(20, 160, 20),
    ));
    scene.push(DrawCommand::PopClip);
    scene.push(DrawCommand::PopClip);

    let mut backend = TinySkiaBackend::new(100, 100).with_background(Color::WHITE);
    backend.render(&scene);
    let px = backend.pixmap();

    // x in [40,60] is the intersection: green.
    assert!(
        px.pixel(50, 50).unwrap().green() > 120,
        "intersection is filled"
    );
    // x = 20 is in the first clip but not the second: clipped.
    assert_eq!(
        {
            let p = px.pixel(20, 50).unwrap();
            (p.red(), p.green(), p.blue())
        },
        (255, 255, 255),
        "outside the intersection stays background"
    );
    // x = 80 is in the second clip but not the first: clipped.
    assert_eq!(
        {
            let p = px.pixel(80, 50).unwrap();
            (p.red(), p.green(), p.blue())
        },
        (255, 255, 255)
    );
}
