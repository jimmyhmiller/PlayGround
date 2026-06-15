//! An SVG export [`whiteboard_core::render::Backend`].
//!
//! This crate proves the draw-command vocabulary drives a *vector* format, not
//! just a raster one: the very same [`RenderScene`] that the tiny-skia backend
//! rasterizes is here serialized into a standalone, well-formed SVG document.
//!
//! There is no external SVG library — the document is hand-rolled as a string —
//! and the only dependency is `whiteboard-core` for the command vocabulary.
//!
//! ## Command mapping
//!
//! | [`DrawCommand`]            | SVG                                                     |
//! |---------------------------|--------------------------------------------------------|
//! | [`DrawCommand::FillPath`]   | `<path d="…" fill="…"/>`                                |
//! | [`DrawCommand::StrokePath`] | `<path d="…" fill="none" stroke="…" stroke-width="…"/>` |
//! | [`DrawCommand::DrawText`]    | `<text x y font-size font-family fill>…</text>`         |
//! | [`DrawCommand::DrawImage`]   | `<image href="…"/>` (a *placeholder*, see below)        |
//! | [`DrawCommand::PushTransform`]/[`DrawCommand::PopTransform`] | nested `<g transform="matrix(…)">…</g>` |
//! | [`DrawCommand::PushClip`]/[`DrawCommand::PopClip`]           | a `<clipPath>` def + `<g clip-path="url(#…)">…</g>` |
//!
//! ### Images are placeholders
//!
//! The headless core never holds pixel data, so the SVG cannot embed the actual
//! image bytes. [`DrawCommand::DrawImage`] is emitted as an `<image>` element
//! whose `href` is `whiteboard-image:<id>` — a sentinel URI naming the
//! [`ImageId`]. A consumer that has the pixels can post-process the document to
//! swap in a real `data:`/`http:` URL.

use std::fmt::Write as _;

use whiteboard_core::geometry::{Path, PathSegment, Transform};
use whiteboard_core::render::{Backend, Color, DrawCommand, ImageId, Paint, RenderScene, Stroke};
use whiteboard_core::text::{FontFamily, TextAlign, TextRun};

/// The sentinel URI scheme used for [`DrawCommand::DrawImage`] placeholders.
/// The full `href` is `whiteboard-image:<image-id>`.
pub const IMAGE_HREF_SCHEME: &str = "whiteboard-image:";

/// An SVG-export backend. Feed it a [`RenderScene`] and read back an SVG
/// document `String`.
///
/// Two equivalent entry points exist:
///
/// * the free function [`to_svg`], for a one-shot scene -> string conversion, and
/// * the [`Backend`] impl, which accumulates the rendered document so generic
///   code that only knows about `&mut dyn Backend` can target SVG too.
///
/// ```
/// use whiteboard_svg::{SvgBackend, to_svg};
/// use whiteboard_core::render::RenderScene;
///
/// let scene = RenderScene::new();
/// let svg = to_svg(&scene, 100, 100);
/// assert!(svg.contains("<svg"));
///
/// // Or via the Backend trait:
/// use whiteboard_core::render::Backend;
/// let mut backend = SvgBackend::new(100, 100);
/// backend.render(&scene);
/// assert_eq!(backend.document(), &svg);
/// ```
#[derive(Debug, Clone)]
pub struct SvgBackend {
    width: u32,
    height: u32,
    document: String,
}

impl SvgBackend {
    /// Create a backend that will produce a `width` x `height` viewport.
    pub fn new(width: u32, height: u32) -> Self {
        SvgBackend {
            width,
            height,
            document: String::new(),
        }
    }

    /// The rendered SVG document. Empty until [`SvgBackend::render`] is called.
    pub fn document(&self) -> &str {
        &self.document
    }

    /// Consume the backend and take ownership of the rendered document string.
    pub fn into_document(self) -> String {
        self.document
    }
}

impl Backend for SvgBackend {
    fn render(&mut self, scene: &RenderScene) {
        self.document = to_svg(scene, self.width, self.height);
    }
}

/// Serialize a [`RenderScene`] to a standalone SVG document of the given pixel
/// dimensions. This is the canonical conversion; [`SvgBackend`] wraps it.
pub fn to_svg(scene: &RenderScene, width: u32, height: u32) -> String {
    let mut w = Writer::new();
    w.body
        .push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    let _ = writeln!(
        w.body,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    );

    for cmd in &scene.commands {
        w.command(cmd);
    }

    // Close any groups left open by unbalanced push/pop. Commands from the core
    // are always balanced, but a defensive close keeps the document well-formed
    // even for hand-built scenes.
    while w.open_groups > 0 {
        w.close_group();
    }

    // Stitch <defs> (clip-paths) ahead of the drawing body.
    let mut out = String::with_capacity(w.body.len() + w.defs.len() + 64);
    out.push_str(&w.body_header());
    if !w.defs.is_empty() {
        out.push_str("<defs>\n");
        out.push_str(&w.defs);
        out.push_str("</defs>\n");
    }
    out.push_str(&w.body_rest());
    out.push_str("</svg>\n");
    out
}

/// Internal accumulator that tracks open `<g>` groups and emitted `<clipPath>`
/// definitions while walking the command list.
struct Writer {
    /// The XML declaration + opening `<svg>` tag plus all drawing markup.
    body: String,
    /// `<clipPath>` definitions, hoisted into a `<defs>` block at assembly time.
    defs: String,
    /// Number of `<g>` elements currently open (transform or clip groups).
    open_groups: u32,
    /// Monotonic id source for generated `clipPath` ids.
    next_clip_id: u32,
    /// Byte offset in `body` where the post-header content begins.
    header_end: usize,
    /// Whether `header_end` has been recorded yet.
    header_marked: bool,
}

impl Writer {
    fn new() -> Self {
        Writer {
            body: String::new(),
            defs: String::new(),
            open_groups: 0,
            next_clip_id: 0,
            header_end: 0,
            header_marked: false,
        }
    }

    /// Record where the `<svg>` header ends so `<defs>` can be injected between
    /// the header and the body. Called lazily the first time real content or a
    /// def is produced.
    fn mark_header(&mut self) {
        if !self.header_marked {
            self.header_end = self.body.len();
            self.header_marked = true;
        }
    }

    fn body_header(&self) -> String {
        let end = if self.header_marked {
            self.header_end
        } else {
            self.body.len()
        };
        self.body[..end].to_string()
    }

    fn body_rest(&self) -> String {
        let end = if self.header_marked {
            self.header_end
        } else {
            self.body.len()
        };
        self.body[end..].to_string()
    }

    fn command(&mut self, cmd: &DrawCommand) {
        self.mark_header();
        match cmd {
            DrawCommand::PushTransform(t) => self.push_transform(t),
            DrawCommand::PopTransform => self.close_group(),
            DrawCommand::PushClip(r) => self.push_clip(r),
            DrawCommand::PopClip => self.close_group(),
            DrawCommand::FillPath { path, paint } => self.fill_path(path, paint),
            DrawCommand::StrokePath {
                path,
                stroke,
                paint,
            } => self.stroke_path(path, stroke, paint),
            DrawCommand::DrawText { run, paint } => self.draw_text(run, paint),
            DrawCommand::DrawImage { id, dst, opacity } => self.draw_image(id, dst, *opacity),
        }
    }

    fn push_transform(&mut self, t: &Transform) {
        let _ = writeln!(
            self.body,
            "<g transform=\"matrix({},{},{},{},{},{})\">",
            fmt_num(t.a),
            fmt_num(t.b),
            fmt_num(t.c),
            fmt_num(t.d),
            fmt_num(t.e),
            fmt_num(t.f),
        );
        self.open_groups += 1;
    }

    fn push_clip(&mut self, r: &whiteboard_core::geometry::Rect) {
        let id = self.next_clip_id;
        self.next_clip_id += 1;
        let _ = writeln!(
            self.defs,
            "<clipPath id=\"clip{id}\"><rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/></clipPath>",
            fmt_num(r.x),
            fmt_num(r.y),
            fmt_num(r.width.max(0.0)),
            fmt_num(r.height.max(0.0)),
        );
        let _ = writeln!(self.body, "<g clip-path=\"url(#clip{id})\">");
        self.open_groups += 1;
    }

    fn close_group(&mut self) {
        if self.open_groups == 0 {
            return;
        }
        self.open_groups -= 1;
        self.body.push_str("</g>\n");
    }

    fn fill_path(&mut self, path: &Path, paint: &Paint) {
        let Paint::Solid(color) = paint;
        if color.is_transparent() {
            return;
        }
        let d = path_d(path);
        if d.is_empty() {
            return;
        }
        let _ = write!(
            self.body,
            "<path d=\"{d}\" fill=\"{}\"",
            color_value(*color)
        );
        write_opacity(&mut self.body, "fill-opacity", *color);
        self.body.push_str("/>\n");
    }

    fn stroke_path(&mut self, path: &Path, stroke: &Stroke, paint: &Paint) {
        let Paint::Solid(color) = paint;
        let d = path_d(path);
        if d.is_empty() {
            return;
        }
        let _ = write!(
            self.body,
            "<path d=\"{d}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\"",
            color_value(*color),
            fmt_num(stroke.width),
        );
        write_opacity(&mut self.body, "stroke-opacity", *color);
        self.body
            .push_str(&format!(" stroke-linecap=\"{}\"", cap_name(stroke)));
        self.body
            .push_str(&format!(" stroke-linejoin=\"{}\"", join_name(stroke)));
        if !stroke.dash.is_empty() {
            let dashes: Vec<String> = stroke.dash.iter().map(|d| fmt_num(*d)).collect();
            let _ = write!(self.body, " stroke-dasharray=\"{}\"", dashes.join(","));
        }
        self.body.push_str("/>\n");
    }

    fn draw_text(&mut self, run: &TextRun, paint: &Paint) {
        let Paint::Solid(color) = paint;
        let anchor = match run.align {
            TextAlign::Left => "start",
            TextAlign::Center => "middle",
            TextAlign::Right => "end",
        };
        let _ = write!(
            self.body,
            "<text x=\"{}\" y=\"{}\" font-size=\"{}\" font-family=\"{}\" text-anchor=\"{}\" fill=\"{}\"",
            fmt_num(run.origin.x),
            fmt_num(run.origin.y),
            fmt_num(run.font.size),
            xml_escape(&font_family_name(&run.font.family)),
            anchor,
            color_value(*color),
        );
        write_opacity(&mut self.body, "fill-opacity", *color);
        let _ = writeln!(self.body, ">{}</text>", xml_escape(&run.text));
    }

    fn draw_image(&mut self, id: &ImageId, dst: &whiteboard_core::geometry::Rect, opacity: f32) {
        // The headless core holds no pixels, so the image cannot be embedded.
        // Emit a placeholder <image> naming the ImageId via a sentinel href; a
        // consumer with the bytes can rewrite it.
        let href = format!("{IMAGE_HREF_SCHEME}{}", id.0);
        let _ = write!(
            self.body,
            "<image href=\"{}\" x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"",
            xml_escape(&href),
            fmt_num(dst.x),
            fmt_num(dst.y),
            fmt_num(dst.width.max(0.0)),
            fmt_num(dst.height.max(0.0)),
        );
        let op = opacity.clamp(0.0, 1.0);
        if op < 1.0 {
            let _ = write!(self.body, " opacity=\"{}\"", fmt_num(op as f64));
        }
        self.body.push_str("/>\n");
    }
}

/// Build the SVG `d` attribute string from a path's segments.
fn path_d(path: &Path) -> String {
    let mut d = String::with_capacity(path.segments.len() * 12);
    for seg in &path.segments {
        if !d.is_empty() {
            d.push(' ');
        }
        match seg {
            PathSegment::MoveTo(p) => {
                let _ = write!(d, "M{} {}", fmt_num(p.x), fmt_num(p.y));
            }
            PathSegment::LineTo(p) => {
                let _ = write!(d, "L{} {}", fmt_num(p.x), fmt_num(p.y));
            }
            PathSegment::CubicTo { c1, c2, to } => {
                let _ = write!(
                    d,
                    "C{} {} {} {} {} {}",
                    fmt_num(c1.x),
                    fmt_num(c1.y),
                    fmt_num(c2.x),
                    fmt_num(c2.y),
                    fmt_num(to.x),
                    fmt_num(to.y),
                );
            }
            PathSegment::Close => d.push('Z'),
        }
    }
    d
}

/// A color for a `fill`/`stroke` attribute: `#rrggbb` (alpha is carried in a
/// separate `*-opacity` attribute, the most broadly compatible encoding).
fn color_value(c: Color) -> String {
    format!("#{:02x}{:02x}{:02x}", c.r, c.g, c.b)
}

/// Append a `fill-opacity`/`stroke-opacity` attribute when alpha is below full.
fn write_opacity(out: &mut String, attr: &str, c: Color) {
    if c.a < 255 {
        let frac = c.a as f64 / 255.0;
        let _ = write!(out, " {attr}=\"{}\"", fmt_num(frac));
    }
}

fn cap_name(stroke: &Stroke) -> &'static str {
    use whiteboard_core::render::LineCap;
    match stroke.cap {
        LineCap::Round => "round",
        LineCap::Butt => "butt",
        LineCap::Square => "square",
    }
}

fn join_name(stroke: &Stroke) -> &'static str {
    use whiteboard_core::render::LineJoin;
    match stroke.join {
        LineJoin::Round => "round",
        LineJoin::Miter => "miter",
        LineJoin::Bevel => "bevel",
    }
}

/// Map a core [`FontFamily`] to a CSS font-family list for the SVG `font-family`
/// attribute. The generic families mirror Excalidraw's three slots.
fn font_family_name(family: &FontFamily) -> String {
    match family {
        FontFamily::HandDrawn => "Excalifont, Virgil, cursive".to_string(),
        FontFamily::Normal => "Nunito, Helvetica, sans-serif".to_string(),
        FontFamily::Code => "Cascadia Code, monospace".to_string(),
        FontFamily::Custom(name) => name.clone(),
    }
}

/// Format an `f64` for SVG output: trims a trailing `.0` and avoids scientific
/// notation, keeping documents compact and stable across platforms.
fn fmt_num(v: f64) -> String {
    if v == 0.0 {
        // Normalize -0.0 to 0.
        return "0".to_string();
    }
    if v.fract() == 0.0 && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    // Six decimals is plenty for screen coordinates; strip trailing zeros.
    let mut s = format!("{v:.6}");
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    s
}

/// Escape the five XML predefined entities so text and attribute values are
/// always well-formed.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use whiteboard_core::geometry::{Point, Rect};
    use whiteboard_core::render::{DrawCommand, Paint, RenderScene, Stroke};
    use whiteboard_core::text::{FontSpec, TextRun};

    fn fill_cmd(color: Color) -> DrawCommand {
        DrawCommand::FillPath {
            path: Path::polygon(&[
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
                Point::new(10.0, 10.0),
            ]),
            paint: Paint::solid(color),
        }
    }

    #[test]
    fn well_formed_envelope() {
        let svg = to_svg(&RenderScene::new(), 200, 100);
        assert!(svg.contains("<?xml"));
        assert!(svg.contains("xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(svg.contains("width=\"200\""));
        assert!(svg.contains("height=\"100\""));
        assert!(svg.trim_end().ends_with("</svg>"));
    }

    #[test]
    fn fill_stroke_and_text_emit_expected_elements() {
        let mut scene = RenderScene::new();
        scene.push(fill_cmd(Color::rgb(200, 30, 30)));
        scene.push(DrawCommand::StrokePath {
            path: Path::polyline(&[Point::new(0.0, 0.0), Point::new(20.0, 20.0)]),
            stroke: Stroke::solid(3.0),
            paint: Paint::solid(Color::BLACK),
        });
        scene.push(DrawCommand::DrawText {
            run: TextRun {
                text: "Hi".to_string(),
                font: FontSpec::default(),
                origin: Point::new(5.0, 15.0),
                align: TextAlign::Left,
            },
            paint: Paint::solid(Color::BLACK),
        });
        let svg = to_svg(&scene, 100, 100);

        // Fill path.
        assert!(svg.contains("<path d=\"M0 0 L10 0 L10 10 Z\" fill=\"#c81e1e\""));
        // Stroke path: fill=none, stroke color + width.
        assert!(svg.contains("fill=\"none\""));
        assert!(svg.contains("stroke=\"#000000\""));
        assert!(svg.contains("stroke-width=\"3\""));
        // Text element with the content.
        assert!(svg.contains("<text "));
        assert!(svg.contains(">Hi</text>"));
        assert!(svg.contains("font-size=\"20\""));
    }

    #[test]
    fn transform_nests_a_group() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::PushTransform(Transform::new(
            1.0, 0.0, 0.0, 1.0, 5.0, 7.0,
        )));
        scene.push(fill_cmd(Color::BLACK));
        scene.push(DrawCommand::PopTransform);
        let svg = to_svg(&scene, 50, 50);

        assert!(svg.contains("<g transform=\"matrix(1,0,0,1,5,7)\">"));
        // The fill must appear between the open and close <g>.
        let open = svg.find("<g transform").unwrap();
        let fill = svg.find("<path").unwrap();
        let close = svg.find("</g>").unwrap();
        assert!(
            open < fill && fill < close,
            "fill must nest inside the group"
        );
        // Balanced <g> tags.
        assert_eq!(svg.matches("<g ").count(), svg.matches("</g>").count());
    }

    #[test]
    fn balanced_groups_for_nested_transforms() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::PushTransform(Transform::scale(2.0, 2.0)));
        scene.push(DrawCommand::PushTransform(Transform::translate(1.0, 1.0)));
        scene.push(fill_cmd(Color::BLACK));
        scene.push(DrawCommand::PopTransform);
        scene.push(DrawCommand::PopTransform);
        let svg = to_svg(&scene, 10, 10);
        assert_eq!(svg.matches("<g ").count(), 2);
        assert_eq!(svg.matches("</g>").count(), 2);
    }

    #[test]
    fn clip_emits_defs_and_group_reference() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::PushClip(Rect::new(1.0, 2.0, 30.0, 40.0)));
        scene.push(fill_cmd(Color::BLACK));
        scene.push(DrawCommand::PopClip);
        let svg = to_svg(&scene, 50, 50);

        assert!(svg.contains("<defs>"));
        assert!(svg.contains("<clipPath id=\"clip0\">"));
        assert!(svg.contains("<rect x=\"1\" y=\"2\" width=\"30\" height=\"40\"/>"));
        assert!(svg.contains("<g clip-path=\"url(#clip0)\">"));
        // defs must come before the body group that references it.
        let defs = svg.find("<defs>").unwrap();
        let group = svg.find("clip-path=\"url(#clip0)\"").unwrap();
        assert!(defs < group);
    }

    #[test]
    fn alpha_becomes_opacity_attribute() {
        let mut scene = RenderScene::new();
        scene.push(fill_cmd(Color::rgba(10, 20, 30, 128)));
        let svg = to_svg(&scene, 10, 10);
        assert!(svg.contains("fill=\"#0a141e\""));
        assert!(svg.contains("fill-opacity=\""));
    }

    #[test]
    fn transparent_fill_is_skipped() {
        let mut scene = RenderScene::new();
        scene.push(fill_cmd(Color::TRANSPARENT));
        let svg = to_svg(&scene, 10, 10);
        assert!(!svg.contains("<path"));
    }

    #[test]
    fn dashed_stroke_emits_dasharray() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::StrokePath {
            path: Path::polyline(&[Point::new(0.0, 0.0), Point::new(10.0, 0.0)]),
            stroke: Stroke {
                width: 2.0,
                dash: vec![4.0, 2.0],
                ..Stroke::default()
            },
            paint: Paint::solid(Color::BLACK),
        });
        let svg = to_svg(&scene, 10, 10);
        assert!(svg.contains("stroke-dasharray=\"4,2\""));
        assert!(svg.contains("stroke-linecap=\"round\""));
    }

    #[test]
    fn cubic_segment_in_path_d() {
        let mut b = Path::builder();
        b.move_to(Point::new(0.0, 0.0)).cubic_to(
            Point::new(0.0, 5.0),
            Point::new(5.0, 5.0),
            Point::new(5.0, 0.0),
        );
        let path = b.build();
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::FillPath {
            path,
            paint: Paint::solid(Color::BLACK),
        });
        let svg = to_svg(&scene, 10, 10);
        assert!(svg.contains("C0 5 5 5 5 0"));
    }

    #[test]
    fn text_xml_is_escaped() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::DrawText {
            run: TextRun {
                text: "a < b & \"c\"".to_string(),
                font: FontSpec::default(),
                origin: Point::new(0.0, 0.0),
                align: TextAlign::Center,
            },
            paint: Paint::solid(Color::BLACK),
        });
        let svg = to_svg(&scene, 10, 10);
        assert!(svg.contains("a &lt; b &amp; &quot;c&quot;"));
        assert!(!svg.contains("a < b"));
        assert!(svg.contains("text-anchor=\"middle\""));
    }

    #[test]
    fn image_emits_placeholder_href() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::DrawImage {
            id: ImageId("photo1".to_string()),
            dst: Rect::new(2.0, 3.0, 40.0, 50.0),
            opacity: 0.5,
        });
        let svg = to_svg(&scene, 60, 60);
        assert!(svg.contains("<image href=\"whiteboard-image:photo1\""));
        assert!(svg.contains("width=\"40\""));
        assert!(svg.contains("opacity=\"0.5\""));
    }

    #[test]
    fn backend_trait_matches_free_function() {
        let mut scene = RenderScene::new();
        scene.push(fill_cmd(Color::rgb(1, 2, 3)));
        let direct = to_svg(&scene, 33, 44);
        let mut backend = SvgBackend::new(33, 44);
        backend.render(&scene);
        assert_eq!(backend.document(), direct);
        assert_eq!(backend.into_document(), direct);
    }

    #[test]
    fn fmt_num_trims_trailing_zeros() {
        assert_eq!(fmt_num(1.0), "1");
        assert_eq!(fmt_num(1.5), "1.5");
        assert_eq!(fmt_num(0.0), "0");
        assert_eq!(fmt_num(-0.0), "0");
        assert_eq!(fmt_num(2.250000), "2.25");
    }

    #[test]
    fn unbalanced_pop_does_not_break_document() {
        // A stray PopTransform with no matching push must not underflow nor
        // emit a dangling </g>.
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::PopTransform);
        scene.push(fill_cmd(Color::BLACK));
        let svg = to_svg(&scene, 10, 10);
        assert_eq!(svg.matches("</g>").count(), 0);
        assert!(svg.contains("<path"));
    }
}
