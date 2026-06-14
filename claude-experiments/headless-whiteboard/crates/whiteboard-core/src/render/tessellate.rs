//! Scene → draw commands.
//!
//! The tessellator walks the scene in paint order, asks the [`ShapeGenerator`]
//! for each element's geometry, applies the element's rotation/translation and
//! opacity, and emits [`DrawCommand`]s. Phase 1 extends it with text, images,
//! frame clipping, dashed strokes, and the selection/handle overlay. This
//! baseline already produces correct fill+stroke commands for the element types
//! the clean generator supports, so the headless→backend path is live.

use super::{DrawCommand, ImageId, Paint, RenderScene, Stroke};
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

    for element in scene.iter_live() {
        emit_element(element, generator, measurer, &mut out);
        out.bounds = out.bounds.union(&element.raw_box());
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
