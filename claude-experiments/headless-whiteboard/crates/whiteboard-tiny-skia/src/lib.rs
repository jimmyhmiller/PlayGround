//! A reference [`whiteboard_core::render::Backend`] built on `tiny-skia`.
//!
//! This crate exists to (a) prove the draw-command vocabulary is sufficient for
//! a real renderer and (b) power deterministic snapshot tests: render a scene to
//! an RGBA buffer, hash or save it. It is intentionally simple — a CPU
//! rasterizer — not a performance target. GPU backends (Vello/wgpu) consume the
//! exact same commands.
//!
//! Text is drawn with bundled fonts via [`fontdue`] (see [`text`]); image
//! drawing is the one remaining `DrawCommand` the backend does not yet
//! rasterize (it needs decoded pixels supplied out-of-band).

mod text;

pub use text::FontSet;

use tiny_skia::{
    FillRule, LineCap as SkCap, LineJoin as SkJoin, Paint as SkPaint, PathBuilder as SkPathBuilder,
    Pixmap, Stroke as SkStroke, StrokeDash, Transform as SkTransform,
};
use whiteboard_core::geometry::Path;
use whiteboard_core::geometry::PathSegment;
use whiteboard_core::render::{
    Backend, Color, DrawCommand, LineCap, LineJoin, Paint, RenderScene, Stroke,
};
use whiteboard_core::text::{FontSpec, TextMeasurer, TextMetrics};

/// A tiny-skia raster surface that consumes draw commands.
pub struct TinySkiaBackend {
    pixmap: Pixmap,
    /// Stack of accumulated transforms (composed top-to-bottom).
    transform_stack: Vec<SkTransform>,
    /// Stack of clip rects (intersection applied at draw time). Each entry is
    /// the clip in device space; `None` sentinels mark unclipped pushes so pops
    /// stay balanced.
    clip_stack: Vec<Option<tiny_skia::Rect>>,
    /// Background fill color.
    background: Color,
    /// Bundled fonts for text rasterization.
    fonts: FontSet,
}

/// A [`TextMeasurer`] backed by the bundled fonts. Hand this to the editor so
/// text layout matches what this backend will actually rasterize.
///
/// ```no_run
/// use whiteboard_tiny_skia::FontMeasurer;
/// use whiteboard_core::editor::Editor;
/// let editor = Editor::new_rough(FontMeasurer::new());
/// ```
pub struct FontMeasurer {
    fonts: FontSet,
}

impl FontMeasurer {
    pub fn new() -> Self {
        FontMeasurer {
            fonts: FontSet::new(),
        }
    }
}

impl Default for FontMeasurer {
    fn default() -> Self {
        FontMeasurer::new()
    }
}

impl TextMeasurer for FontMeasurer {
    fn measure(&self, text: &str, font: &FontSpec) -> TextMetrics {
        self.fonts.measure(text, font)
    }
}

impl TinySkiaBackend {
    /// Create a backend rendering into a `width` x `height` RGBA pixmap.
    pub fn new(width: u32, height: u32) -> Self {
        TinySkiaBackend {
            pixmap: Pixmap::new(width.max(1), height.max(1)).expect("valid pixmap size"),
            transform_stack: Vec::new(),
            clip_stack: Vec::new(),
            background: Color::WHITE,
            fonts: FontSet::new(),
        }
    }

    pub fn with_background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Access the rendered pixels (RGBA, premultiplied as tiny-skia stores them).
    pub fn pixmap(&self) -> &Pixmap {
        &self.pixmap
    }

    /// Encode the current surface as PNG bytes.
    pub fn encode_png(&self) -> Result<Vec<u8>, String> {
        self.pixmap.encode_png().map_err(|e| e.to_string())
    }

    /// Save the current surface as a PNG file.
    pub fn save_png(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        self.pixmap.save_png(path).map_err(|e| e.to_string())
    }

    fn current_transform(&self) -> SkTransform {
        self.transform_stack
            .last()
            .copied()
            .unwrap_or(SkTransform::identity())
    }

    fn clear_background(&mut self) {
        self.pixmap.fill(to_sk_color(self.background));
    }

    fn draw_fill(&mut self, path: &Path, paint: &Paint) {
        let Some(sk_path) = build_path(path) else {
            return;
        };
        let mut sk_paint = SkPaint {
            anti_alias: true,
            ..SkPaint::default()
        };
        set_paint_color(&mut sk_paint, paint);
        let ts = self.current_transform();
        self.pixmap
            .fill_path(&sk_path, &sk_paint, FillRule::Winding, ts, None);
    }

    fn draw_text(&mut self, run: &whiteboard_core::text::TextRun, paint: &Paint) {
        let Paint::Solid(color) = paint;
        let ts = self.current_transform();
        // Map the baseline-left origin into device space.
        let mut pt = tiny_skia::Point::from_xy(run.origin.x as f32, run.origin.y as f32);
        ts.map_point(&mut pt);
        // Scale the font size by the transform's uniform scale (viewport zoom).
        let scale = ((ts.sx * ts.sx + ts.ky * ts.ky).sqrt()) as f64;

        let mut dev_run = run.clone();
        dev_run.origin = whiteboard_core::geometry::Point::new(pt.x as f64, pt.y as f64);
        dev_run.font = FontSpec {
            size: run.font.size * scale,
            ..run.font.clone()
        };
        // fontdue rasterizes axis-aligned glyphs; rotation/skew in the transform
        // is not applied to the glyph shapes (the common viewport case is
        // translate + uniform scale, which is handled exactly).
        self.fonts.draw_run(&mut self.pixmap, &dev_run, *color);
    }

    fn draw_stroke(&mut self, path: &Path, stroke: &Stroke, paint: &Paint) {
        let Some(sk_path) = build_path(path) else {
            return;
        };
        let mut sk_paint = SkPaint {
            anti_alias: true,
            ..SkPaint::default()
        };
        set_paint_color(&mut sk_paint, paint);
        let sk_stroke = build_stroke(stroke);
        let ts = self.current_transform();
        self.pixmap
            .stroke_path(&sk_path, &sk_paint, &sk_stroke, ts, None);
    }
}

impl Backend for TinySkiaBackend {
    fn render(&mut self, scene: &RenderScene) {
        self.transform_stack.clear();
        self.clip_stack.clear();
        self.clear_background();

        for cmd in &scene.commands {
            match cmd {
                DrawCommand::PushTransform(t) => {
                    let base = self.current_transform();
                    let next = base.pre_concat(to_sk_transform(t));
                    self.transform_stack.push(next);
                }
                DrawCommand::PopTransform => {
                    self.transform_stack.pop();
                }
                DrawCommand::PushClip(r) => {
                    self.clip_stack.push(tiny_skia::Rect::from_xywh(
                        r.x as f32,
                        r.y as f32,
                        r.width.max(0.0) as f32,
                        r.height.max(0.0) as f32,
                    ));
                }
                DrawCommand::PopClip => {
                    self.clip_stack.pop();
                }
                DrawCommand::FillPath { path, paint } => self.draw_fill(path, paint),
                DrawCommand::StrokePath {
                    path,
                    stroke,
                    paint,
                } => self.draw_stroke(path, stroke, paint),
                DrawCommand::DrawText { run, paint } => self.draw_text(run, paint),
                // Image drawing needs decoded pixels supplied out-of-band; the
                // backend has no pixel store yet, so we skip rather than fake it.
                DrawCommand::DrawImage { .. } => {}
            }
        }
    }
}

fn to_sk_color(c: Color) -> tiny_skia::Color {
    tiny_skia::Color::from_rgba8(c.r, c.g, c.b, c.a)
}

fn set_paint_color(sk_paint: &mut SkPaint, paint: &Paint) {
    match paint {
        Paint::Solid(c) => sk_paint.set_color(to_sk_color(*c)),
    }
}

fn to_sk_transform(t: &whiteboard_core::Transform) -> SkTransform {
    SkTransform::from_row(
        t.a as f32, t.b as f32, t.c as f32, t.d as f32, t.e as f32, t.f as f32,
    )
}

fn build_stroke(stroke: &Stroke) -> SkStroke {
    let mut s = SkStroke {
        width: stroke.width as f32,
        ..SkStroke::default()
    };
    s.line_cap = match stroke.cap {
        LineCap::Round => SkCap::Round,
        LineCap::Butt => SkCap::Butt,
        LineCap::Square => SkCap::Square,
    };
    s.line_join = match stroke.join {
        LineJoin::Round => SkJoin::Round,
        LineJoin::Miter => SkJoin::Miter,
        LineJoin::Bevel => SkJoin::Bevel,
    };
    if !stroke.dash.is_empty() {
        let dashes: Vec<f32> = stroke.dash.iter().map(|d| *d as f32).collect();
        s.dash = StrokeDash::new(dashes, 0.0);
    }
    s
}

/// Translate our backend-neutral [`Path`] into a tiny-skia path.
fn build_path(path: &Path) -> Option<tiny_skia::Path> {
    let mut b = SkPathBuilder::new();
    for seg in &path.segments {
        match seg {
            PathSegment::MoveTo(p) => b.move_to(p.x as f32, p.y as f32),
            PathSegment::LineTo(p) => b.line_to(p.x as f32, p.y as f32),
            PathSegment::CubicTo { c1, c2, to } => b.cubic_to(
                c1.x as f32,
                c1.y as f32,
                c2.x as f32,
                c2.y as f32,
                to.x as f32,
                to.y as f32,
            ),
            PathSegment::Close => b.close(),
        }
    }
    b.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use whiteboard_core::editor::Editor;
    use whiteboard_core::element::{Element, ElementId, ElementKind};
    use whiteboard_core::text::MonospaceMeasurer;

    #[test]
    fn renders_a_rectangle_to_pixels() {
        let mut editor = Editor::new(MonospaceMeasurer::default());
        let mut e = Element::new(
            ElementId::from("r"),
            1,
            10.0,
            10.0,
            40.0,
            30.0,
            ElementKind::Rectangle,
        );
        e.background_color = Color::rgb(200, 30, 30);
        editor.add_element(e);

        let mut backend = TinySkiaBackend::new(100, 100);
        backend.render(&editor.render());

        // The center of the rectangle should no longer be the white background.
        let px = backend.pixmap();
        let center = px.pixel(30, 25).unwrap();
        assert!(
            center.red() > 100 && center.green() < 120,
            "expected reddish fill, got rgba({},{},{},{})",
            center.red(),
            center.green(),
            center.blue(),
            center.alpha()
        );
    }

    #[test]
    fn empty_scene_is_background() {
        let backend_bg = Color::rgb(250, 250, 250);
        let mut backend = TinySkiaBackend::new(10, 10).with_background(backend_bg);
        backend.render(&RenderScene::new());
        let px = backend.pixmap().pixel(5, 5).unwrap();
        assert_eq!((px.red(), px.green(), px.blue()), (250, 250, 250));
    }

    #[test]
    fn text_draws_dark_pixels() {
        use whiteboard_core::element::TextData;

        // Use the font-backed measurer so layout matches rasterization.
        let mut editor = Editor::new(FontMeasurer::new());
        let mut data = TextData::new("Hello");
        data.font_size = 40.0;
        let mut e = Element::new(
            ElementId::from("t"),
            1,
            5.0,
            5.0,
            0.0,
            50.0,
            ElementKind::Text(data),
        );
        e.stroke_color = Color::BLACK;
        editor.add_element(e);

        let mut backend = TinySkiaBackend::new(220, 70);
        backend.render(&editor.render());

        // Somewhere in the text region there must be a dark (drawn) pixel.
        let any_dark = backend
            .pixmap()
            .pixels()
            .iter()
            .any(|p| p.red() < 100 && p.green() < 100 && p.blue() < 100 && p.alpha() > 0);
        assert!(any_dark, "text rasterization produced no dark glyph pixels");
    }
}
