//! Scene → draw commands.
//!
//! The tessellator walks the scene in paint order, asks the [`ShapeGenerator`]
//! for each element's geometry, applies the element's rotation/translation and
//! opacity, and emits [`DrawCommand`]s. Phase 1 extends it with text, images,
//! frame clipping, dashed strokes, and the selection/handle overlay. This
//! baseline already produces correct fill+stroke commands for the element types
//! the clean generator supports, so the headless→backend path is live.

use super::{DrawCommand, Paint, RenderScene, Stroke};
use crate::element::Element;
use crate::geometry::Transform;
use crate::render::Color;
use crate::scene::Scene;
use crate::shape::{ShapeGenerator, ShapeGeometry};

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

/// Turn a scene into draw commands using the given shape generator.
pub fn tessellate<G: ShapeGenerator>(
    scene: &Scene,
    generator: &G,
    opts: &RenderOptions,
) -> RenderScene {
    let mut out = RenderScene::new();

    let viewport_is_identity = opts.viewport == Transform::IDENTITY;
    if !viewport_is_identity {
        out.push(DrawCommand::PushTransform(opts.viewport));
    }

    for element in scene.iter_live() {
        emit_element(element, generator, &mut out);
        out.bounds = out.bounds.union(&element.raw_box());
    }

    if !viewport_is_identity {
        out.push(DrawCommand::PopTransform);
    }

    out
}

fn emit_element<G: ShapeGenerator>(element: &Element, generator: &G, out: &mut RenderScene) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Element, ElementId, ElementKind};
    use crate::render::Color;
    use crate::shape::CleanGenerator;

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
        let rs = tessellate(&scene, &CleanGenerator, &RenderOptions::default());
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
        let rs = tessellate(&scene, &CleanGenerator, &opts);
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
        let rs = tessellate(&scene, &CleanGenerator, &RenderOptions::default());
        assert!(rs.is_empty());
    }
}
