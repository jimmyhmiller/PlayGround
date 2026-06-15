//! GPU backend for `whiteboard-core` targeting [Vello].
//!
//! This crate consumes our backend-neutral [`DrawCommand`] vocabulary in two
//! layers:
//!
//! 1. [`to_gpu_ops`] — a thin, dependency-free flattening of the command list
//!    into [`GpuOp`]s. It proves the command list is renderer-agnostic and gives
//!    any retained-mode GPU backend a typed seam to build from.
//! 2. [`build_vello_scene`] — the real Vello path. It walks a [`RenderScene`] and
//!    assembles a [`vello::Scene`] by translating fills/strokes into
//!    [`kurbo::BezPath`]s painted with [`peniko`] brushes, applying a
//!    [`kurbo::Affine`] transform stack as it goes.
//!
//! Building a `vello::Scene` is pure CPU-side work: no GPU, surface, or `wgpu`
//! device is required, so the whole conversion is unit-testable headlessly. The
//! crate depends on `vello` with `default-features = false`, which drops the
//! `wgpu`/`vello_shaders` stack and keeps only the scene-encoding machinery. The
//! device/surface wiring that turns a `Scene` into pixels lands in a later phase
//! behind vello's `default` feature.
//!
//! Text and image commands are intentionally **not** rendered yet:
//! [`DrawCommand::DrawText`] needs a font atlas / glyph shaping pipeline and
//! [`DrawCommand::DrawImage`] needs an image registry mapping `ImageId` to pixel
//! data. Rather than emit fake geometry, those commands are skipped and counted
//! so callers can see what was deferred (see [`build_vello_scene_report`]).
//!
//! [Vello]: https://github.com/linebender/vello

use vello::kurbo::{self, Affine, BezPath, Cap, Join, Stroke as KStroke};
use vello::peniko::{self, Brush, Fill};
use vello::Scene;

use whiteboard_core::geometry::{Path, PathSegment, Point, Transform};
use whiteboard_core::render::{Color, DrawCommand, LineCap, LineJoin, Paint, RenderScene, Stroke};

/// A flattened, GPU-friendly view of one draw command. This is the seam a Vello
/// (or any retained-mode GPU) backend builds its scene from. It deliberately
/// carries the same information as [`DrawCommand`]; the value is in proving the
/// command list is renderer-agnostic and giving the GPU backend a typed target.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuOp<'a> {
    PushLayer,
    PopLayer,
    Fill(&'a DrawCommand),
    Stroke(&'a DrawCommand),
    Text(&'a DrawCommand),
    Image(&'a DrawCommand),
}

/// Walk a render scene and yield GPU ops in paint order. Pure and allocation-
/// free over the borrowed scene, so it is trivially testable without a GPU.
pub fn to_gpu_ops(scene: &RenderScene) -> Vec<GpuOp<'_>> {
    scene
        .commands
        .iter()
        .map(|cmd| match cmd {
            DrawCommand::PushClip(_) | DrawCommand::PushTransform(_) => GpuOp::PushLayer,
            DrawCommand::PopClip | DrawCommand::PopTransform => GpuOp::PopLayer,
            DrawCommand::FillPath { .. } => GpuOp::Fill(cmd),
            DrawCommand::StrokePath { .. } => GpuOp::Stroke(cmd),
            DrawCommand::DrawText { .. } => GpuOp::Text(cmd),
            DrawCommand::DrawImage { .. } => GpuOp::Image(cmd),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Type conversions (pure, GPU-free, individually testable)
// ---------------------------------------------------------------------------

/// Convert our [`Point`] into a [`kurbo::Point`].
#[inline]
fn kpoint(p: Point) -> kurbo::Point {
    kurbo::Point::new(p.x, p.y)
}

/// Build a [`kurbo::BezPath`] from our backend-neutral [`Path`].
///
/// The mapping is one-to-one: `MoveTo`/`LineTo`/`CubicTo`/`Close` map onto
/// kurbo's `move_to`/`line_to`/`curve_to`/`close_path`. Cursor / subpath
/// bookkeeping is kurbo's job; we only forward segments in order.
///
/// Our own [`Path`] builder always emits a `MoveTo` before any draw segment, so
/// we faithfully forward what we are given rather than inventing an origin.
pub fn path_to_bez(path: &Path) -> BezPath {
    let mut bez = BezPath::new();
    for seg in &path.segments {
        match *seg {
            PathSegment::MoveTo(p) => bez.move_to(kpoint(p)),
            PathSegment::LineTo(p) => bez.line_to(kpoint(p)),
            PathSegment::CubicTo { c1, c2, to } => bez.curve_to(kpoint(c1), kpoint(c2), kpoint(to)),
            PathSegment::Close => bez.close_path(),
        }
    }
    bez
}

/// Convert our SVG-matrix [`Transform`] into a [`kurbo::Affine`].
///
/// Both store the affine as `[a, b, c, d, e, f]` in the same convention
/// (`matrix(a, b, c, d, e, f)` from SVG/Canvas), so this is a direct coefficient
/// copy with no re-ordering.
#[inline]
pub fn affine_of(t: &Transform) -> Affine {
    Affine::new([t.a, t.b, t.c, t.d, t.e, t.f])
}

/// Convert our straight-alpha sRGB [`Color`] into a [`peniko::Color`].
#[inline]
pub fn peniko_color(c: Color) -> peniko::Color {
    peniko::Color::from_rgba8(c.r, c.g, c.b, c.a)
}

/// Convert a [`Paint`] into a peniko [`Brush`]. Only solid paint exists today.
#[inline]
fn brush_of(paint: &Paint) -> Brush {
    match paint {
        Paint::Solid(c) => Brush::Solid(peniko_color(*c)),
    }
}

/// Convert our [`Stroke`] into a [`kurbo::Stroke`], including cap, join and
/// resolved dash pattern.
///
/// Our [`Stroke::dash`] is already a flat list of explicit on/off lengths (the
/// core resolves `StrokeStyle` to concrete lengths), which is exactly the shape
/// kurbo's `with_dashes` wants. An empty list means a solid stroke.
pub fn kurbo_stroke(stroke: &Stroke) -> KStroke {
    let cap = match stroke.cap {
        LineCap::Round => Cap::Round,
        LineCap::Butt => Cap::Butt,
        LineCap::Square => Cap::Square,
    };
    let join = match stroke.join {
        LineJoin::Round => Join::Round,
        LineJoin::Miter => Join::Miter,
        LineJoin::Bevel => Join::Bevel,
    };
    let s = KStroke::new(stroke.width).with_caps(cap).with_join(join);
    if stroke.dash.is_empty() {
        s
    } else {
        s.with_dashes(0.0, stroke.dash.iter().copied())
    }
}

// ---------------------------------------------------------------------------
// Scene assembly
// ---------------------------------------------------------------------------

/// What [`build_vello_scene`] did with a render scene, beyond the `Scene` itself.
///
/// Returned alongside the scene by [`build_vello_scene_report`] so callers can
/// observe which commands were honored and which were deferred (text/image/clip),
/// without that information being silently swallowed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BuildReport {
    /// Number of `FillPath` commands encoded.
    pub fills: usize,
    /// Number of `StrokePath` commands encoded.
    pub strokes: usize,
    /// `DrawText` commands skipped pending a font atlas.
    pub skipped_text: usize,
    /// `DrawImage` commands skipped pending an image registry.
    pub skipped_images: usize,
    /// `PushClip` commands skipped — clip support lands with layer push/pop.
    pub skipped_clips: usize,
}

/// Build a [`vello::Scene`] from a [`RenderScene`].
///
/// Walks the command list in paint order, maintaining a [`kurbo::Affine`]
/// transform stack (`PushTransform`/`PopTransform`). Each fill/stroke is encoded
/// with the current top-of-stack affine applied, so geometry authored in
/// pre-transform space lands in the right place.
///
/// Text and image commands are skipped (see the module docs); clip commands are
/// skipped for now too — Vello expresses clipping via layer push/pop with a clip
/// shape, which is a separate piece of work from the geometry path and is left
/// for the GPU-surface phase to avoid emitting a half-correct clip.
///
/// This function is pure: it allocates and returns a `Scene` value and never
/// touches a GPU. Use [`build_vello_scene_report`] if you also want a
/// [`BuildReport`] of what was deferred.
pub fn build_vello_scene(scene: &RenderScene) -> Scene {
    build_vello_scene_report(scene).0
}

/// Like [`build_vello_scene`] but also returns a [`BuildReport`] describing which
/// commands were encoded versus deferred.
pub fn build_vello_scene_report(scene: &RenderScene) -> (Scene, BuildReport) {
    let mut vscene = Scene::new();
    let mut report = BuildReport::default();

    // Transform stack. The top is the affine applied to current geometry. We seed
    // it with identity so an unbalanced/empty scene still encodes sensibly.
    let mut stack: Vec<Affine> = vec![Affine::IDENTITY];
    let top = |stack: &[Affine]| *stack.last().unwrap_or(&Affine::IDENTITY);

    for cmd in &scene.commands {
        match cmd {
            DrawCommand::PushTransform(t) => {
                // Compose parent * child: the pushed transform applies first
                // (innermost), matching `Transform::then` / SVG nesting.
                let next = top(&stack) * affine_of(t);
                stack.push(next);
            }
            DrawCommand::PopTransform => {
                // Never pop the identity sentinel; an unbalanced PopTransform is
                // tolerated rather than panicking on malformed input.
                if stack.len() > 1 {
                    stack.pop();
                }
            }
            DrawCommand::FillPath { path, paint } => {
                let bez = path_to_bez(path);
                let brush = brush_of(paint);
                vscene.fill(Fill::NonZero, top(&stack), &brush, None, &bez);
                report.fills += 1;
            }
            DrawCommand::StrokePath {
                path,
                stroke,
                paint,
            } => {
                let bez = path_to_bez(path);
                let brush = brush_of(paint);
                let kstroke = kurbo_stroke(stroke);
                vscene.stroke(&kstroke, top(&stack), &brush, None, &bez);
                report.strokes += 1;
            }
            // Clip support is intentionally deferred: Vello clips via
            // push_layer(blend, alpha, transform, clip_shape) / pop_layer, which
            // is a distinct concern from geometry. Skip rather than emit a
            // partial clip that would silently mis-render.
            DrawCommand::PushClip(_) => report.skipped_clips += 1,
            DrawCommand::PopClip => {}
            // TODO(font-atlas): glyph runs need shaping + a glyph atlas before we
            // can encode them. Skip rather than draw placeholder boxes.
            DrawCommand::DrawText { .. } => report.skipped_text += 1,
            // TODO(image-registry): images need an ImageId -> pixels registry and
            // a peniko::Image. Skip rather than draw a fake rectangle.
            DrawCommand::DrawImage { .. } => report.skipped_images += 1,
        }
    }

    (vscene, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use whiteboard_core::geometry::Path;
    use whiteboard_core::render::{Color, Paint, Stroke, StrokeStyle};

    // ---- to_gpu_ops ----

    #[test]
    fn maps_fill_command() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::FillPath {
            path: Path::polygon(&[
                Point::new(0.0, 0.0),
                Point::new(1.0, 0.0),
                Point::new(0.0, 1.0),
            ]),
            paint: Paint::solid(Color::BLACK),
        });
        let ops = to_gpu_ops(&scene);
        assert!(matches!(ops.as_slice(), [GpuOp::Fill(_)]));
    }

    // ---- path_to_bez ----

    #[test]
    fn path_to_bez_maps_each_segment_kind() {
        let mut b = Path::builder();
        b.move_to(Point::new(0.0, 0.0))
            .line_to(Point::new(10.0, 0.0))
            .cubic_to(
                Point::new(10.0, 5.0),
                Point::new(5.0, 10.0),
                Point::new(0.0, 10.0),
            )
            .close();
        let path = b.build();
        let bez = path_to_bez(&path);

        use kurbo::PathEl;
        let els: Vec<PathEl> = bez.elements().to_vec();
        assert_eq!(els.len(), 4, "one element per segment");
        assert!(matches!(els[0], PathEl::MoveTo(p) if p == kurbo::Point::new(0.0, 0.0)));
        assert!(matches!(els[1], PathEl::LineTo(p) if p == kurbo::Point::new(10.0, 0.0)));
        assert!(matches!(
            els[2],
            PathEl::CurveTo(c1, c2, to)
                if c1 == kurbo::Point::new(10.0, 5.0)
                    && c2 == kurbo::Point::new(5.0, 10.0)
                    && to == kurbo::Point::new(0.0, 10.0)
        ));
        assert!(matches!(els[3], PathEl::ClosePath));
    }

    #[test]
    fn path_to_bez_empty_is_empty() {
        let bez = path_to_bez(&Path::new());
        assert!(bez.elements().is_empty());
    }

    #[test]
    fn path_to_bez_bbox_matches_geometry() {
        // A filled triangle's bezpath bounds should match the input extent.
        let path = Path::polygon(&[
            Point::new(2.0, 3.0),
            Point::new(12.0, 3.0),
            Point::new(2.0, 9.0),
        ]);
        let bez = path_to_bez(&path);
        let bbox = kurbo::Shape::bounding_box(&bez);
        assert_eq!(bbox.x0, 2.0);
        assert_eq!(bbox.y0, 3.0);
        assert_eq!(bbox.x1, 12.0);
        assert_eq!(bbox.y1, 9.0);
    }

    // ---- affine_of ----

    #[test]
    fn affine_of_copies_coeffs_in_order() {
        let t = Transform::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let a = affine_of(&t);
        assert_eq!(a.as_coeffs(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn affine_of_translate_matches_transform_apply() {
        let t = Transform::translate(5.0, -2.0);
        let a = affine_of(&t);
        let p = Point::new(1.0, 1.0);
        let via_core = t.apply(p);
        let via_kurbo = a * kpoint(p);
        assert_eq!(via_kurbo.x, via_core.x);
        assert_eq!(via_kurbo.y, via_core.y);
    }

    #[test]
    fn affine_of_rotate_matches_transform_apply() {
        use std::f64::consts::PI;
        let t = Transform::rotate_around(PI / 2.0, Point::new(10.0, 10.0));
        let a = affine_of(&t);
        let p = Point::new(11.0, 10.0);
        let via_core = t.apply(p);
        let via_kurbo = a * kpoint(p);
        assert!((via_kurbo.x - via_core.x).abs() < 1e-9);
        assert!((via_kurbo.y - via_core.y).abs() < 1e-9);
    }

    // ---- color / stroke conversion ----

    #[test]
    fn peniko_color_preserves_channels() {
        let c = Color::rgba(18, 52, 86, 200);
        let pc = peniko_color(c);
        let rgba8 = pc.to_rgba8();
        assert_eq!([rgba8.r, rgba8.g, rgba8.b, rgba8.a], [18, 52, 86, 200]);
    }

    #[test]
    fn kurbo_stroke_carries_width_and_caps() {
        let stroke = Stroke {
            width: 3.5,
            cap: LineCap::Square,
            join: LineJoin::Miter,
            dash: Vec::new(),
        };
        let ks = kurbo_stroke(&stroke);
        assert_eq!(ks.width, 3.5);
        assert!(matches!(ks.start_cap, Cap::Square));
        assert!(matches!(ks.end_cap, Cap::Square));
        assert!(matches!(ks.join, Join::Miter));
        assert!(ks.dash_pattern.is_empty());
    }

    #[test]
    fn kurbo_stroke_carries_dashes() {
        let stroke = Stroke::with_style(2.0, StrokeStyle::Dashed);
        assert!(
            !stroke.dash.is_empty(),
            "precondition: dashed has a pattern"
        );
        let ks = kurbo_stroke(&stroke);
        let pattern: Vec<f64> = ks.dash_pattern.iter().copied().collect();
        assert_eq!(pattern, stroke.dash);
    }

    #[test]
    fn kurbo_stroke_round_is_default_caps() {
        let ks = kurbo_stroke(&Stroke::solid(1.0));
        assert!(matches!(ks.start_cap, Cap::Round));
        assert!(matches!(ks.join, Join::Round));
    }

    // ---- build_vello_scene ----

    fn triangle() -> Path {
        Path::polygon(&[
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(0.0, 10.0),
        ])
    }

    #[test]
    fn build_vello_scene_is_pure_and_returns_scene() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::FillPath {
            path: triangle(),
            paint: Paint::solid(Color::BLACK),
        });
        // Must not panic / require a GPU; just produces a Scene value.
        let vscene = build_vello_scene(&scene);
        // A filled scene must have encoded something.
        assert!(
            !vscene.encoding().is_empty(),
            "fill should produce encoding data"
        );
    }

    #[test]
    fn empty_scene_encodes_empty() {
        let vscene = build_vello_scene(&RenderScene::new());
        assert!(vscene.encoding().is_empty());
    }

    #[test]
    fn report_counts_fills_and_strokes() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::FillPath {
            path: triangle(),
            paint: Paint::solid(Color::BLACK),
        });
        scene.push(DrawCommand::StrokePath {
            path: triangle(),
            stroke: Stroke::solid(2.0),
            paint: Paint::solid(Color::WHITE),
        });
        let (_scene, report) = build_vello_scene_report(&scene);
        assert_eq!(report.fills, 1);
        assert_eq!(report.strokes, 1);
        assert_eq!(report.skipped_text, 0);
        assert_eq!(report.skipped_images, 0);
    }

    #[test]
    fn report_tracks_deferred_text_image_clip() {
        use whiteboard_core::geometry::Rect;
        use whiteboard_core::render::ImageId;
        use whiteboard_core::text::{FontSpec, TextAlign, TextRun};

        let run = TextRun {
            text: "hi".to_string(),
            font: FontSpec::default(),
            origin: Point::new(0.0, 0.0),
            align: TextAlign::default(),
        };

        let mut scene = RenderScene::new();
        scene.push(DrawCommand::PushClip(Rect::new(0.0, 0.0, 10.0, 10.0)));
        scene.push(DrawCommand::DrawText {
            run,
            paint: Paint::solid(Color::BLACK),
        });
        scene.push(DrawCommand::DrawImage {
            id: ImageId("img-1".to_string()),
            dst: Rect::new(0.0, 0.0, 4.0, 4.0),
            opacity: 1.0,
        });
        scene.push(DrawCommand::PopClip);
        let (_scene, report) = build_vello_scene_report(&scene);
        assert_eq!(report.skipped_text, 1);
        assert_eq!(report.skipped_images, 1);
        assert_eq!(report.skipped_clips, 1);
        assert_eq!(report.fills, 0);
    }

    #[test]
    fn transform_stack_balances_and_tolerates_unbalanced_pop() {
        // A push/pop pair around a fill, plus a stray pop that must not panic.
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::PushTransform(Transform::translate(5.0, 5.0)));
        scene.push(DrawCommand::FillPath {
            path: triangle(),
            paint: Paint::solid(Color::BLACK),
        });
        scene.push(DrawCommand::PopTransform);
        scene.push(DrawCommand::PopTransform); // stray / unbalanced
        scene.push(DrawCommand::FillPath {
            path: triangle(),
            paint: Paint::solid(Color::WHITE),
        });
        let (_scene, report) = build_vello_scene_report(&scene);
        assert_eq!(report.fills, 2);
    }

    #[test]
    fn nested_transforms_compose_innermost_first() {
        // Build the scene's effective top-of-stack affine the way the encoder
        // does for `PushTransform(outer); PushTransform(inner)`, and confirm a
        // point in innermost space is mapped by inner *then* outer. We re-derive
        // the affine here because the encoded Scene bytes are opaque; the affine
        // math is what we assert on.
        let outer = Transform::translate(1.0, 1.0);
        let inner = Transform::scale(2.0, 2.0);

        // Encoder convention: top = parent_affine * child_affine, so the
        // innermost (most recently pushed) transform is applied first.
        let encoded_top = (Affine::IDENTITY * affine_of(&outer)) * affine_of(&inner);

        // Core equivalent: inner.then(outer) applies inner first, then outer —
        // exactly the nesting the encoder produces.
        let core = inner.then(&outer);

        let p = Point::new(3.0, 4.0);
        let via_encoder = encoded_top * kpoint(p);
        let via_core = core.apply(p);
        assert!((via_encoder.x - via_core.x).abs() < 1e-9);
        assert!((via_encoder.y - via_core.y).abs() < 1e-9);

        // Concretely: (3,4) scaled by 2 -> (6,8), then translated +1 -> (7,9).
        assert!((via_encoder.x - 7.0).abs() < 1e-9);
        assert!((via_encoder.y - 9.0).abs() < 1e-9);
    }
}
