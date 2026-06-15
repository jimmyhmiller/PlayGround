//! Scene → draw commands.
//!
//! The tessellator walks the scene in paint order, asks the [`ShapeGenerator`]
//! for each element's geometry, applies the element's rotation/translation and
//! opacity, and emits [`DrawCommand`]s. Phase 1 extends it with text, images,
//! frame clipping, dashed strokes, and the selection/handle overlay. This
//! baseline already produces correct fill+stroke commands for the element types
//! the clean generator supports, so the headless→backend path is live.

use super::{clip, DrawCommand, ImageId, Paint, RenderScene, Stroke};
use crate::element::{Element, ElementKind};
use crate::geometry::{Point, Transform};
use crate::render::Color;
use crate::scene::Scene;
use crate::shape::{ShapeGenerator, ShapeGeometry};
use crate::text::{layout_text, FontSpec, TextMeasurer};

/// Configuration for a render pass.
#[derive(Debug, Clone, Copy)]
pub struct RenderOptions {
    /// Viewport transform (pan/zoom) applied to the whole scene.
    pub viewport: Transform,
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions {
            viewport: Transform::IDENTITY,
        }
    }
}

/// Turn a scene into draw commands using the given shape generator and text
/// measurer. The measurer lays out text elements into positioned glyph runs; it
/// is the same one the editor injects, so text rendering is consistent with text
/// layout used elsewhere.
pub fn tessellate<G: ShapeGenerator>(
    scene: &Scene,
    generator: &G,
    measurer: &dyn TextMeasurer,
    opts: &RenderOptions,
) -> RenderScene {
    let mut out = RenderScene::new();

    let viewport_is_identity = opts.viewport == Transform::IDENTITY;
    if !viewport_is_identity {
        out.push(DrawCommand::PushTransform(opts.viewport));
    }

    // Plan frame clips before walking the scene. Clip edges are expressed against
    // the same paint-ordered live element list we iterate below, so index `i`
    // lines up. Clips live *inside* the viewport transform: the clip rect is in
    // scene space and the backend's clip stack composes it with the active
    // transform, matching how every other command is placed.
    let live: Vec<&Element> = scene.iter_live().collect();
    let edges = clip::plan_clips(scene, &live);

    for (i, element) in live.iter().enumerate() {
        let edge = edges[i];
        // Close the previous frame's clip run before drawing an element that
        // starts a new run (or leaves all runs).
        if edge.close_before_open {
            out.push(DrawCommand::PopClip);
        }
        // Open this frame's clip before drawing its first contiguous child.
        if let Some(rect) = edge.open {
            out.push(DrawCommand::PushClip(rect));
        }
        emit_element(element, generator, measurer, &mut out);
        out.bounds = out.bounds.union(&element.raw_box());
    }
    // A trailing run of framed children leaves one clip open; close it so
    // push/pop stay balanced.
    if clip::trailing_clip_open(scene, &live) {
        out.push(DrawCommand::PopClip);
    }

    if !viewport_is_identity {
        out.push(DrawCommand::PopTransform);
    }

    out
}

fn emit_element<G: ShapeGenerator>(
    element: &Element,
    generator: &G,
    measurer: &dyn TextMeasurer,
    out: &mut RenderScene,
) {
    // Text and image elements carry no vector outline; they emit their own
    // command kinds. Handle them first so the empty-geometry guard below doesn't
    // skip them.
    match &element.kind {
        ElementKind::Text(_) => {
            emit_text(element, measurer, out);
            return;
        }
        ElementKind::Image(data) => {
            emit_image(element, &data.file_id, out);
            return;
        }
        _ => {}
    }

    let ShapeGeometry {
        outline,
        fill,
        fill_strokes,
    } = generator.geometry(element);
    if outline.is_empty() && fill.is_empty() && fill_strokes.is_empty() {
        return;
    }

    // Element-local geometry is in unrotated space with origin at (0,0); place
    // it at (x, y) and rotate about the box center.
    let place = element_transform(element);
    let needs_transform = place != Transform::IDENTITY;
    if needs_transform {
        out.push(DrawCommand::PushTransform(place));
    }

    let opacity = element.opacity_unit();

    if !element.background_color.is_transparent() {
        let paint = Paint::solid(element.background_color).with_opacity(opacity);
        // Solid fill regions: flood-fill.
        for path in &fill {
            out.push(DrawCommand::FillPath {
                path: path.clone(),
                paint: paint.clone(),
            });
        }
        // Hachure / cross-hatch / zigzag fill lines: stroke with the background
        // color at a thin, fixed width (independent of the outline width).
        if !fill_strokes.is_empty() {
            let fill_stroke = Stroke::solid(element.stroke_width.max(1.0));
            for path in &fill_strokes {
                out.push(DrawCommand::StrokePath {
                    path: path.clone(),
                    stroke: fill_stroke.clone(),
                    paint: paint.clone(),
                });
            }
        }
    }

    if !element.stroke_color.is_transparent() && element.stroke_color != Color::TRANSPARENT {
        let stroke = Stroke::with_style(element.stroke_width, element.stroke_style);
        let paint = Paint::solid(element.stroke_color).with_opacity(opacity);
        for path in &outline {
            out.push(DrawCommand::StrokePath {
                path: path.clone(),
                stroke: stroke.clone(),
                paint: paint.clone(),
            });
        }
    }

    if needs_transform {
        out.push(DrawCommand::PopTransform);
    }
}

/// Transform placing element-local geometry into scene space: translate to
/// `(x, y)`, then rotate about the (now scene-space) box center.
fn element_transform(element: &Element) -> Transform {
    let translate = Transform::translate(element.x, element.y);
    if element.angle == 0.0 {
        return translate;
    }
    translate.then(&Transform::rotate_around(element.angle, element.center()))
}

/// Lay out a text element and emit a [`DrawCommand::DrawText`] per line.
///
/// The runs come back positioned in scene space (top-left at the element's
/// `(x, y)`); for rotated text we push a rotation about the element center so the
/// backend draws each line rotated.
fn emit_text(element: &Element, measurer: &dyn TextMeasurer, out: &mut RenderScene) {
    let ElementKind::Text(data) = &element.kind else {
        return;
    };
    if data.text.is_empty() {
        return;
    }

    let font = FontSpec {
        family: data.font_family.clone(),
        size: data.font_size,
        line_height: data.line_height,
    };
    // Wrap to the element's box width when it is a real (positive) width; a
    // zero/negative width means "auto" — split on explicit newlines only.
    let max_width = (element.width > 0.0).then_some(element.width);
    let laid = layout_text(measurer, &data.text, &font, data.text_align, max_width);
    let runs = laid.runs_at(
        Point::new(element.x, element.y),
        data.vertical_align,
        Some(element.height),
    );

    let opacity = element.opacity_unit();
    let paint = Paint::solid(element.stroke_color).with_opacity(opacity);

    let rotated = element.angle != 0.0;
    if rotated {
        out.push(DrawCommand::PushTransform(Transform::rotate_around(
            element.angle,
            element.center(),
        )));
    }
    for run in runs {
        if run.text.is_empty() {
            continue;
        }
        out.push(DrawCommand::DrawText {
            run,
            paint: paint.clone(),
        });
    }
    if rotated {
        out.push(DrawCommand::PopTransform);
    }
}

/// Emit a [`DrawCommand::DrawImage`] covering the element's box.
fn emit_image(element: &Element, file_id: &str, out: &mut RenderScene) {
    let opacity = element.opacity_unit();
    let dst = element.raw_box();
    let rotated = element.angle != 0.0;
    if rotated {
        out.push(DrawCommand::PushTransform(Transform::rotate_around(
            element.angle,
            element.center(),
        )));
    }
    out.push(DrawCommand::DrawImage {
        id: ImageId(file_id.to_string()),
        dst,
        opacity,
    });
    if rotated {
        out.push(DrawCommand::PopTransform);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Element, ElementId, ElementKind};
    use crate::render::Color;
    use crate::shape::CleanGenerator;
    use crate::text::MonospaceMeasurer;

    fn filled_rect() -> Element {
        let mut e = Element::new(
            ElementId::from("r"),
            1,
            10.0,
            10.0,
            20.0,
            20.0,
            ElementKind::Rectangle,
        );
        e.background_color = Color::rgb(255, 0, 0);
        e
    }

    #[test]
    fn emits_fill_then_stroke() {
        let mut scene = Scene::new();
        scene.insert(filled_rect());
        let rs = tessellate(
            &scene,
            &CleanGenerator,
            &MonospaceMeasurer::default(),
            &RenderOptions::default(),
        );
        // PushTransform, FillPath, StrokePath, PopTransform
        let kinds: Vec<_> = rs
            .commands
            .iter()
            .map(|c| match c {
                DrawCommand::PushTransform(_) => "pushT",
                DrawCommand::PopTransform => "popT",
                DrawCommand::FillPath { .. } => "fill",
                DrawCommand::StrokePath { .. } => "stroke",
                _ => "other",
            })
            .collect();
        assert_eq!(kinds, ["pushT", "fill", "stroke", "popT"]);
    }

    #[test]
    fn viewport_wraps_scene() {
        let mut scene = Scene::new();
        scene.insert(filled_rect());
        let opts = RenderOptions {
            viewport: Transform::translate(5.0, 5.0),
        };
        let rs = tessellate(
            &scene,
            &CleanGenerator,
            &MonospaceMeasurer::default(),
            &opts,
        );
        assert!(matches!(
            rs.commands.first(),
            Some(DrawCommand::PushTransform(_))
        ));
        assert!(matches!(
            rs.commands.last(),
            Some(DrawCommand::PopTransform)
        ));
    }

    #[test]
    fn deleted_elements_are_skipped() {
        let mut scene = Scene::new();
        let mut e = filled_rect();
        e.is_deleted = true;
        scene.insert(e);
        let rs = tessellate(
            &scene,
            &CleanGenerator,
            &MonospaceMeasurer::default(),
            &RenderOptions::default(),
        );
        assert!(rs.is_empty());
    }

    #[test]
    fn text_element_emits_draw_text() {
        use crate::element::TextData;
        let mut scene = Scene::new();
        let e = Element::new(
            ElementId::from("t"),
            1,
            10.0,
            10.0,
            0.0, // auto width => no wrap
            25.0,
            ElementKind::Text(TextData::new("hello\nworld")),
        );
        scene.insert(e);
        let rs = tessellate(
            &scene,
            &CleanGenerator,
            &MonospaceMeasurer::default(),
            &RenderOptions::default(),
        );
        let text_runs = rs
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::DrawText { .. }))
            .count();
        assert_eq!(text_runs, 2, "two lines => two DrawText commands");
    }

    fn frame_kind() -> ElementKind {
        serde_json::from_value(serde_json::json!({ "type": "frame", "name": null }))
            .expect("frame kind deserializes")
    }

    fn frame(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from(id), 1, x, y, w, h, frame_kind())
    }

    fn framed_rect(id: &str, frame_id: &str) -> Element {
        let mut e = filled_rect();
        e.id = ElementId::from(id);
        e.frame_id = Some(ElementId::from(frame_id));
        e
    }

    /// Indices of every PushClip/PopClip in the command stream, as a kind list.
    fn clip_kinds(rs: &RenderScene) -> Vec<&'static str> {
        rs.commands
            .iter()
            .filter_map(|c| match c {
                DrawCommand::PushClip(_) => Some("push"),
                DrawCommand::PopClip => Some("pop"),
                _ => None,
            })
            .collect()
    }

    /// Assert push/pop clips are balanced and never pop below zero depth.
    fn assert_clips_balanced(rs: &RenderScene) {
        let mut depth = 0i32;
        for c in &rs.commands {
            match c {
                DrawCommand::PushClip(_) => depth += 1,
                DrawCommand::PopClip => depth -= 1,
                _ => {}
            }
            assert!(depth >= 0, "PopClip without matching PushClip");
        }
        assert_eq!(depth, 0, "unbalanced clips: ended at depth {depth}");
    }

    #[test]
    fn frame_children_are_clipped() {
        let mut scene = Scene::new();
        scene.insert(frame("f", 0.0, 0.0, 100.0, 100.0));
        scene.insert(framed_rect("c1", "f"));
        scene.insert(framed_rect("c2", "f"));
        let rs = tessellate(
            &scene,
            &CleanGenerator,
            &MonospaceMeasurer::default(),
            &RenderOptions::default(),
        );
        // Exactly one push and one pop wrapping the two children.
        assert_eq!(clip_kinds(&rs), ["push", "pop"]);
        assert_clips_balanced(&rs);

        // The push carries the frame's clip rect.
        let push = rs
            .commands
            .iter()
            .find(|c| matches!(c, DrawCommand::PushClip(_)))
            .unwrap();
        assert!(matches!(
            push,
            DrawCommand::PushClip(r) if *r == crate::geometry::Rect::new(0.0, 0.0, 100.0, 100.0)
        ));

        // The PushClip comes before any child paint command, and the PopClip
        // comes after the last one (no draw command sits outside the clip pair
        // for the framed children).
        let push_idx = rs
            .commands
            .iter()
            .position(|c| matches!(c, DrawCommand::PushClip(_)))
            .unwrap();
        let pop_idx = rs
            .commands
            .iter()
            .position(|c| matches!(c, DrawCommand::PopClip))
            .unwrap();
        let first_fill = rs
            .commands
            .iter()
            .position(|c| matches!(c, DrawCommand::FillPath { .. }))
            .unwrap();
        // The frame outline itself draws unclipped, so a stroke for the frame
        // exists before the push; the children's fills come after it.
        assert!(push_idx < first_fill, "clip opens before children draw");
        assert!(pop_idx > push_idx, "pop after push");
    }

    #[test]
    fn frame_outline_draws_unclipped() {
        // The frame element's own stroke must be emitted outside any clip.
        let mut scene = Scene::new();
        scene.insert(frame("f", 0.0, 0.0, 100.0, 100.0));
        scene.insert(framed_rect("c1", "f"));
        let rs = tessellate(
            &scene,
            &CleanGenerator,
            &MonospaceMeasurer::default(),
            &RenderOptions::default(),
        );
        let push_idx = rs
            .commands
            .iter()
            .position(|c| matches!(c, DrawCommand::PushClip(_)))
            .unwrap();
        // The first stroke (the frame outline) precedes the clip push.
        let first_stroke = rs
            .commands
            .iter()
            .position(|c| matches!(c, DrawCommand::StrokePath { .. }))
            .unwrap();
        assert!(
            first_stroke < push_idx,
            "frame outline draws before the children's clip is pushed"
        );
        assert_clips_balanced(&rs);
    }

    #[test]
    fn frame_with_no_children_emits_no_clip() {
        let mut scene = Scene::new();
        scene.insert(frame("f", 0.0, 0.0, 100.0, 100.0));
        let rs = tessellate(
            &scene,
            &CleanGenerator,
            &MonospaceMeasurer::default(),
            &RenderOptions::default(),
        );
        assert!(clip_kinds(&rs).is_empty(), "no children => no clip");
        assert_clips_balanced(&rs);
    }

    #[test]
    fn unframed_elements_are_not_clipped() {
        let mut scene = Scene::new();
        scene.insert(filled_rect());
        let rs = tessellate(
            &scene,
            &CleanGenerator,
            &MonospaceMeasurer::default(),
            &RenderOptions::default(),
        );
        assert!(clip_kinds(&rs).is_empty());
        assert_clips_balanced(&rs);
    }

    #[test]
    fn two_frames_each_get_their_own_clip() {
        let mut scene = Scene::new();
        scene.insert(frame("f", 0.0, 0.0, 50.0, 50.0));
        scene.insert(framed_rect("c1", "f"));
        scene.insert(frame("g", 100.0, 0.0, 50.0, 50.0));
        scene.insert(framed_rect("c2", "g"));
        let rs = tessellate(
            &scene,
            &CleanGenerator,
            &MonospaceMeasurer::default(),
            &RenderOptions::default(),
        );
        // Two balanced push/pop pairs, one per frame run.
        assert_eq!(clip_kinds(&rs), ["push", "pop", "push", "pop"]);
        assert_clips_balanced(&rs);
    }

    #[test]
    fn trailing_framed_child_clip_is_closed() {
        // The framed child is the very last element in paint order; the trailing
        // clip must still be popped.
        let mut scene = Scene::new();
        scene.insert(frame("f", 0.0, 0.0, 100.0, 100.0));
        scene.insert(framed_rect("c1", "f"));
        let rs = tessellate(
            &scene,
            &CleanGenerator,
            &MonospaceMeasurer::default(),
            &RenderOptions::default(),
        );
        assert_eq!(clip_kinds(&rs), ["push", "pop"]);
        assert_clips_balanced(&rs);
        // The PopClip is inside the (identity) viewport — with identity viewport
        // there is no PopTransform, so PopClip is last.
        assert!(matches!(rs.commands.last(), Some(DrawCommand::PopClip)));
    }

    #[test]
    fn clips_nested_inside_viewport_transform() {
        let mut scene = Scene::new();
        scene.insert(frame("f", 0.0, 0.0, 100.0, 100.0));
        scene.insert(framed_rect("c1", "f"));
        let opts = RenderOptions {
            viewport: Transform::translate(5.0, 5.0),
        };
        let rs = tessellate(
            &scene,
            &CleanGenerator,
            &MonospaceMeasurer::default(),
            &opts,
        );
        // Outermost is the viewport push/pop; clips live inside it.
        assert!(matches!(
            rs.commands.first(),
            Some(DrawCommand::PushTransform(_))
        ));
        assert!(matches!(
            rs.commands.last(),
            Some(DrawCommand::PopTransform)
        ));
        let push_clip = rs
            .commands
            .iter()
            .position(|c| matches!(c, DrawCommand::PushClip(_)))
            .unwrap();
        let pop_clip = rs
            .commands
            .iter()
            .position(|c| matches!(c, DrawCommand::PopClip))
            .unwrap();
        assert!(push_clip > 0 && pop_clip < rs.commands.len() - 1);
        assert_clips_balanced(&rs);
    }

    #[test]
    fn empty_text_emits_nothing() {
        use crate::element::TextData;
        let mut scene = Scene::new();
        scene.insert(Element::new(
            ElementId::from("t"),
            1,
            0.0,
            0.0,
            0.0,
            25.0,
            ElementKind::Text(TextData::new("")),
        ));
        let rs = tessellate(
            &scene,
            &CleanGenerator,
            &MonospaceMeasurer::default(),
            &RenderOptions::default(),
        );
        assert!(rs.is_empty());
    }
}
