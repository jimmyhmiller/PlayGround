//! The Vello backend must faithfully encode a real `Editor`-produced
//! `RenderScene`: every fill/stroke command becomes a Vello draw, and
//! text/image/clip commands are deferred and *counted* (never silently dropped).

use whiteboard_core::editor::Editor;
use whiteboard_core::element::{Element, ElementId, ElementKind, TextData};
use whiteboard_core::render::{Color, DrawCommand};
use whiteboard_core::text::MonospaceMeasurer;
use whiteboard_vello::{build_vello_scene, build_vello_scene_report};

fn scene_with_shapes_and_text() -> whiteboard_core::render::RenderScene {
    let mut editor = Editor::new(MonospaceMeasurer::default());

    let mut rect = Element::new(
        ElementId::from("r"),
        1,
        10.0,
        10.0,
        80.0,
        60.0,
        ElementKind::Rectangle,
    );
    rect.background_color = Color::rgb(200, 200, 255);
    rect.roughness = 0.0;
    editor.add_element(rect);

    let mut text = TextData::new("hello");
    text.font_size = 20.0;
    editor.add_element(Element::new(
        ElementId::from("t"),
        2,
        20.0,
        90.0,
        0.0,
        25.0,
        ElementKind::Text(text),
    ));

    editor.render()
}

#[test]
fn report_counts_match_command_stream() {
    let scene = scene_with_shapes_and_text();

    // Count command kinds independently.
    let mut fills = 0;
    let mut strokes = 0;
    let mut texts = 0;
    for c in &scene.commands {
        match c {
            DrawCommand::FillPath { .. } => fills += 1,
            DrawCommand::StrokePath { .. } => strokes += 1,
            DrawCommand::DrawText { .. } => texts += 1,
            _ => {}
        }
    }
    assert!(fills >= 1, "rect background is a fill");
    assert!(strokes >= 1, "rect outline is a stroke");
    assert!(texts >= 1, "text element emits DrawText");

    let (_scene, report) = build_vello_scene_report(&scene);
    assert_eq!(report.fills, fills, "all fills encoded");
    assert_eq!(report.strokes, strokes, "all strokes encoded");
    assert_eq!(
        report.skipped_text, texts,
        "text deferred and counted (font atlas pending)"
    );
}

#[test]
fn build_vello_scene_is_pure_and_deterministic() {
    // Building a vello::Scene needs no GPU; it must succeed headlessly and be
    // stable for the same input (we can't compare Scene values directly, but the
    // report is a deterministic fingerprint of what was encoded).
    let scene = scene_with_shapes_and_text();
    let (_a, ra) = build_vello_scene_report(&scene);
    let (_b, rb) = build_vello_scene_report(&scene);
    assert_eq!(ra, rb, "encoding is deterministic");

    // The plain entry point also runs without panicking.
    let _ = build_vello_scene(&scene);
}

#[test]
fn empty_scene_encodes_nothing() {
    let (_s, report) = build_vello_scene_report(&whiteboard_core::render::RenderScene::new());
    assert_eq!(report.fills, 0);
    assert_eq!(report.strokes, 0);
    assert_eq!(report.skipped_text, 0);
}
