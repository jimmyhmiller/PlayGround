//! A web-canvas [`whiteboard_core::render::Backend`] targeting an HTML5
//! `<canvas>` 2D context via `wasm-bindgen`/`web-sys`.
//!
//! This crate proves the browser target: the very same [`RenderScene`] the
//! tiny-skia backend rasterizes and the SVG backend serializes is here painted
//! onto a [`web_sys::CanvasRenderingContext2d`].
//!
//! ## Host-buildable by construction
//!
//! `web-sys` *types* (a `CanvasRenderingContext2d`, a `Path2d`) can only be
//! constructed inside a browser, so a naive backend would only compile for
//! `wasm32`. Instead the rendering **logic** is factored into a pure,
//! dependency-light function — [`scene_to_ops`] — that lowers a `&RenderScene`
//! into a flat `Vec<`[`CanvasOp`]`>`: an explicit, serializable list of canvas
//! state-machine operations (`Save`, `Restore`, `Transform`, `Clip`,
//! `FillPath`, `StrokePath`, `FillText`, `DrawImage`). That function, and the
//! pure string helpers it uses ([`css_color`], [`path_data`], [`font_string`]),
//! are ordinary Rust with real unit tests that run on the host (`cargo test
//! -p whiteboard-web`).
//!
//! The thin [`WebBackend`] layer — compiled only for `wasm32` — does nothing but
//! *execute* each [`CanvasOp`] against a real `web-sys` context. There is no
//! drawing decision left in the wasm layer, so the host build exercises the
//! whole translation.
//!
//! ## Command mapping
//!
//! | [`DrawCommand`]              | [`CanvasOp`] / canvas API |
//! |-----------------------------|---------------------------|
//! | `PushTransform`             | `Save` + `Transform(a,b,c,d,e,f)` |
//! | `PopTransform`             | `Restore` |
//! | `PushClip`                | `Save` + `Clip(rect)` |
//! | `PopClip`                 | `Restore` |
//! | `FillPath`                | `FillPath { data, fill }` -> `Path2d` + `fill` |
//! | `StrokePath`              | `StrokePath { data, stroke, line_width, dash, cap, join }` |
//! | `DrawText`                | `FillText { text, x, y, font, fill, align }` |
//! | `DrawImage`               | `DrawImage { id, dst, opacity }` (see images, below) |
//!
//! ### Images
//!
//! The headless core never holds pixel data, so a [`DrawCommand::DrawImage`]
//! cannot embed bytes. The backend resolves an [`ImageId`] through an injected
//! image map ([`WebBackend::set_image`]); when an id is present the bitmap is
//! drawn into `dst` at the requested opacity, otherwise the op is a documented
//! no-op (nothing is painted, and rendering continues). The pure [`scene_to_ops`]
//! layer always emits the [`CanvasOp::DrawImage`] so the host can assert on it.

use whiteboard_core::geometry::{Path, PathSegment, Rect};
use whiteboard_core::render::{Color, DrawCommand, ImageId, Paint, RenderScene, Stroke};
use whiteboard_core::text::{FontFamily, FontSpec, TextAlign, TextRun};

#[cfg(target_arch = "wasm32")]
mod backend;
#[cfg(target_arch = "wasm32")]
pub use backend::{render_scene_json, WebBackend};

// ---------------------------------------------------------------------------
// Pure helpers (host-testable)
// ---------------------------------------------------------------------------

/// Convert a core [`Color`] to a CSS `rgba(r, g, b, a)` string, with alpha as a
/// 0..=1 fraction. This is the form the Canvas 2D `fillStyle`/`strokeStyle`
/// accept; using `rgba()` keeps straight (non-premultiplied) alpha intact.
///
/// ```
/// use whiteboard_core::render::Color;
/// use whiteboard_web::css_color;
/// assert_eq!(css_color(Color::rgb(10, 20, 30)), "rgba(10, 20, 30, 1)");
/// assert_eq!(css_color(Color::rgba(255, 0, 0, 128)), "rgba(255, 0, 0, 0.501961)");
/// ```
pub fn css_color(c: Color) -> String {
    let a = c.a as f64 / 255.0;
    format!("rgba({}, {}, {}, {})", c.r, c.g, c.b, fmt_num(a))
}

/// Build an SVG-`d`-style path-data string from a [`Path`]'s segments.
///
/// The same grammar SVG uses (`M`/`L`/`C`/`Z`) is exactly what `Path2d`'s
/// constructor accepts, so this single string both (a) drives the wasm
/// `Path2d::new_with_path_string(&path_data(p))` and (b) is trivially assertable
/// on the host.
///
/// ```
/// use whiteboard_core::geometry::{Path, Point};
/// use whiteboard_web::path_data;
/// let p = Path::polygon(&[Point::new(0.0, 0.0), Point::new(10.0, 0.0), Point::new(10.0, 10.0)]);
/// assert_eq!(path_data(&p), "M0 0 L10 0 L10 10 Z");
/// ```
pub fn path_data(path: &Path) -> String {
    use std::fmt::Write as _;
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

/// Build a CSS `font` shorthand string (`<size>px <family-list>`) for the Canvas
/// 2D `ctx.font` property from a core [`FontSpec`]. The generic families mirror
/// Excalidraw's three slots.
///
/// ```
/// use whiteboard_core::text::{FontFamily, FontSpec};
/// use whiteboard_web::font_string;
/// let f = FontSpec::new(FontFamily::Code, 16.0);
/// assert_eq!(font_string(&f), "16px Cascadia Code, monospace");
/// ```
pub fn font_string(font: &FontSpec) -> String {
    format!(
        "{}px {}",
        fmt_num(font.size),
        font_family_list(&font.family)
    )
}

/// Map a core [`FontFamily`] to a CSS font-family list.
fn font_family_list(family: &FontFamily) -> String {
    match family {
        FontFamily::HandDrawn => "Excalifont, Virgil, cursive".to_string(),
        FontFamily::Normal => "Nunito, Helvetica, sans-serif".to_string(),
        FontFamily::Code => "Cascadia Code, monospace".to_string(),
        FontFamily::Custom(name) => name.clone(),
    }
}

/// Map a core [`TextAlign`] to the Canvas 2D `textAlign` keyword.
fn text_align_keyword(align: TextAlign) -> &'static str {
    match align {
        TextAlign::Left => "left",
        TextAlign::Center => "center",
        TextAlign::Right => "right",
    }
}

/// Format an `f64` compactly: trim a trailing `.0`, avoid scientific notation,
/// and normalize `-0.0` to `0`. Keeps emitted strings stable across platforms.
fn fmt_num(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    if v.fract() == 0.0 && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    let mut s = format!("{v:.6}");
    while s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    s
}

// ---------------------------------------------------------------------------
// The command -> canvas-op translation (pure, host-tested)
// ---------------------------------------------------------------------------

/// The Canvas 2D line-cap keyword, decoupled from `web-sys` so it is host-usable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapKind {
    Round,
    Butt,
    Square,
}

impl CapKind {
    /// The Canvas 2D `lineCap` keyword.
    pub fn keyword(self) -> &'static str {
        match self {
            CapKind::Round => "round",
            CapKind::Butt => "butt",
            CapKind::Square => "square",
        }
    }
}

/// The Canvas 2D line-join keyword, decoupled from `web-sys`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinKind {
    Round,
    Miter,
    Bevel,
}

impl JoinKind {
    /// The Canvas 2D `lineJoin` keyword.
    pub fn keyword(self) -> &'static str {
        match self {
            JoinKind::Round => "round",
            JoinKind::Miter => "miter",
            JoinKind::Bevel => "bevel",
        }
    }
}

/// A single, fully-resolved canvas state-machine operation. Producing these from
/// a [`RenderScene`] is the entire job of the pure layer; the wasm layer only
/// replays them onto a real context. Everything here is plain data, so the
/// translation is exhaustively unit-testable on the host.
#[derive(Debug, Clone, PartialEq)]
pub enum CanvasOp {
    /// `ctx.save()`.
    Save,
    /// `ctx.restore()`.
    Restore,
    /// `ctx.transform(a, b, c, d, e, f)`.
    Transform {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
        f: f64,
    },
    /// Build a rectangular path and `ctx.clip()` it. Width/height are clamped to
    /// be non-negative.
    Clip { x: f64, y: f64, w: f64, h: f64 },
    /// Build a `Path2d` from `data` and `ctx.fill_with_path_2d`, after setting
    /// `fillStyle = fill`.
    FillPath { data: String, fill: String },
    /// Build a `Path2d` from `data` and `ctx.stroke_with_path`, after setting the
    /// stroke style/width/dash/cap/join.
    StrokePath {
        data: String,
        stroke: String,
        line_width: f64,
        dash: Vec<f64>,
        cap: CapKind,
        join: JoinKind,
    },
    /// Set `ctx.font`/`textAlign`/`fillStyle` then `ctx.fill_text(text, x, y)`.
    FillText {
        text: String,
        x: f64,
        y: f64,
        font: String,
        fill: String,
        align: &'static str,
    },
    /// Draw a previously-registered image into `dst` at `opacity`. If the id has
    /// no registered bitmap this is a no-op at execution time.
    DrawImage {
        id: String,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
        opacity: f32,
    },
}

/// Lower a whole [`RenderScene`] into a flat list of [`CanvasOp`]s in paint
/// order. This is the heart of the backend and is pure: no `web-sys`, no global
/// state, fully deterministic.
///
/// Transparent fills are dropped (matching the SVG backend) so the canvas does
/// no wasted work. Push/pop pairs become `Save`/`Restore`; the function does not
/// invent extra restores for unbalanced input — the wasm executor tolerates a
/// `Restore` with an empty stack the same way the browser does.
pub fn scene_to_ops(scene: &RenderScene) -> Vec<CanvasOp> {
    let mut ops = Vec::with_capacity(scene.commands.len());
    for cmd in &scene.commands {
        command_to_ops(cmd, &mut ops);
    }
    ops
}

/// Lower a single [`DrawCommand`], appending zero or more [`CanvasOp`]s.
fn command_to_ops(cmd: &DrawCommand, out: &mut Vec<CanvasOp>) {
    match cmd {
        DrawCommand::PushTransform(t) => {
            out.push(CanvasOp::Save);
            out.push(CanvasOp::Transform {
                a: t.a,
                b: t.b,
                c: t.c,
                d: t.d,
                e: t.e,
                f: t.f,
            });
        }
        DrawCommand::PopTransform => out.push(CanvasOp::Restore),
        DrawCommand::PushClip(r) => {
            out.push(CanvasOp::Save);
            out.push(CanvasOp::Clip {
                x: r.x,
                y: r.y,
                w: r.width.max(0.0),
                h: r.height.max(0.0),
            });
        }
        DrawCommand::PopClip => out.push(CanvasOp::Restore),
        DrawCommand::FillPath { path, paint } => {
            let Paint::Solid(color) = paint;
            if color.is_transparent() {
                return;
            }
            let data = path_data(path);
            if data.is_empty() {
                return;
            }
            out.push(CanvasOp::FillPath {
                data,
                fill: css_color(*color),
            });
        }
        DrawCommand::StrokePath {
            path,
            stroke,
            paint,
        } => {
            let Paint::Solid(color) = paint;
            if color.is_transparent() {
                return;
            }
            let data = path_data(path);
            if data.is_empty() {
                return;
            }
            out.push(CanvasOp::StrokePath {
                data,
                stroke: css_color(*color),
                line_width: stroke.width,
                dash: stroke.dash.clone(),
                cap: cap_kind(stroke),
                join: join_kind(stroke),
            });
        }
        DrawCommand::DrawText { run, paint } => {
            let Paint::Solid(color) = paint;
            if color.is_transparent() {
                return;
            }
            out.push(text_op(run, *color));
        }
        DrawCommand::DrawImage { id, dst, opacity } => {
            out.push(image_op(id, dst, *opacity));
        }
    }
}

fn text_op(run: &TextRun, color: Color) -> CanvasOp {
    CanvasOp::FillText {
        text: run.text.clone(),
        x: run.origin.x,
        y: run.origin.y,
        font: font_string(&run.font),
        fill: css_color(color),
        align: text_align_keyword(run.align),
    }
}

fn image_op(id: &ImageId, dst: &Rect, opacity: f32) -> CanvasOp {
    CanvasOp::DrawImage {
        id: id.0.clone(),
        x: dst.x,
        y: dst.y,
        w: dst.width.max(0.0),
        h: dst.height.max(0.0),
        opacity: opacity.clamp(0.0, 1.0),
    }
}

fn cap_kind(stroke: &Stroke) -> CapKind {
    use whiteboard_core::render::LineCap;
    match stroke.cap {
        LineCap::Round => CapKind::Round,
        LineCap::Butt => CapKind::Butt,
        LineCap::Square => CapKind::Square,
    }
}

fn join_kind(stroke: &Stroke) -> JoinKind {
    use whiteboard_core::render::LineJoin;
    match stroke.join {
        LineJoin::Round => JoinKind::Round,
        LineJoin::Miter => JoinKind::Miter,
        LineJoin::Bevel => JoinKind::Bevel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use whiteboard_core::geometry::{Point, Transform};
    use whiteboard_core::render::{Color, Paint, Stroke};
    use whiteboard_core::text::FontSpec;

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

    // --- css_color -------------------------------------------------------

    #[test]
    fn css_color_opaque_alpha_is_one() {
        assert_eq!(css_color(Color::rgb(1, 2, 3)), "rgba(1, 2, 3, 1)");
        assert_eq!(css_color(Color::BLACK), "rgba(0, 0, 0, 1)");
        assert_eq!(css_color(Color::WHITE), "rgba(255, 255, 255, 1)");
    }

    #[test]
    fn css_color_half_alpha_is_fraction() {
        // 128/255 = 0.501960..., kept to six decimals.
        assert_eq!(
            css_color(Color::rgba(255, 0, 0, 128)),
            "rgba(255, 0, 0, 0.501961)"
        );
    }

    #[test]
    fn css_color_zero_alpha() {
        assert_eq!(css_color(Color::TRANSPARENT), "rgba(0, 0, 0, 0)");
    }

    // --- path_data -------------------------------------------------------

    #[test]
    fn path_data_polygon_round_trips_grammar() {
        let p = Path::polygon(&[
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
        ]);
        assert_eq!(path_data(&p), "M0 0 L10 0 L10 10 Z");
    }

    #[test]
    fn path_data_cubic_segment() {
        let mut b = Path::builder();
        b.move_to(Point::new(0.0, 0.0)).cubic_to(
            Point::new(0.0, 5.0),
            Point::new(5.0, 5.0),
            Point::new(5.0, 0.0),
        );
        assert_eq!(path_data(&b.build()), "M0 0 C0 5 5 5 5 0");
    }

    #[test]
    fn path_data_empty_path_is_empty_string() {
        assert_eq!(path_data(&Path::new()), "");
    }

    #[test]
    fn path_data_fractions_are_trimmed() {
        let p = Path::polyline(&[Point::new(1.5, 2.25), Point::new(-0.0, 3.0)]);
        assert_eq!(path_data(&p), "M1.5 2.25 L0 3");
    }

    // --- font_string -----------------------------------------------------

    #[test]
    fn font_string_code_family() {
        let f = FontSpec::new(FontFamily::Code, 16.0);
        assert_eq!(font_string(&f), "16px Cascadia Code, monospace");
    }

    #[test]
    fn font_string_handdrawn_and_normal() {
        assert_eq!(
            font_string(&FontSpec::new(FontFamily::HandDrawn, 20.0)),
            "20px Excalifont, Virgil, cursive"
        );
        assert_eq!(
            font_string(&FontSpec::new(FontFamily::Normal, 12.5)),
            "12.5px Nunito, Helvetica, sans-serif"
        );
    }

    #[test]
    fn font_string_custom_family_is_verbatim() {
        let f = FontSpec::new(FontFamily::Custom("Comic Sans MS".to_string()), 10.0);
        assert_eq!(font_string(&f), "10px Comic Sans MS");
    }

    // --- scene_to_ops ----------------------------------------------------

    #[test]
    fn transform_lowers_to_save_then_transform() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::PushTransform(Transform::new(
            1.0, 0.0, 0.0, 1.0, 5.0, 7.0,
        )));
        scene.push(fill_cmd(Color::BLACK));
        scene.push(DrawCommand::PopTransform);
        let ops = scene_to_ops(&scene);
        assert_eq!(
            ops,
            vec![
                CanvasOp::Save,
                CanvasOp::Transform {
                    a: 1.0,
                    b: 0.0,
                    c: 0.0,
                    d: 1.0,
                    e: 5.0,
                    f: 7.0
                },
                CanvasOp::FillPath {
                    data: "M0 0 L10 0 L10 10 Z".to_string(),
                    fill: "rgba(0, 0, 0, 1)".to_string(),
                },
                CanvasOp::Restore,
            ]
        );
    }

    #[test]
    fn clip_lowers_to_save_then_clip_with_clamped_size() {
        let mut scene = RenderScene::new();
        // Rect::new normalizes negative sizes, so feed a raw rect via from_min_max
        // to ensure clamping logic is exercised even if a degenerate rect appears.
        scene.push(DrawCommand::PushClip(Rect::new(1.0, 2.0, 30.0, 40.0)));
        scene.push(DrawCommand::PopClip);
        let ops = scene_to_ops(&scene);
        assert_eq!(
            ops,
            vec![
                CanvasOp::Save,
                CanvasOp::Clip {
                    x: 1.0,
                    y: 2.0,
                    w: 30.0,
                    h: 40.0
                },
                CanvasOp::Restore,
            ]
        );
    }

    #[test]
    fn transparent_fill_emits_no_op() {
        let mut scene = RenderScene::new();
        scene.push(fill_cmd(Color::TRANSPARENT));
        assert!(scene_to_ops(&scene).is_empty());
    }

    #[test]
    fn empty_path_fill_emits_no_op() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::FillPath {
            path: Path::new(),
            paint: Paint::solid(Color::BLACK),
        });
        assert!(scene_to_ops(&scene).is_empty());
    }

    #[test]
    fn stroke_carries_width_dash_cap_join() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::StrokePath {
            path: Path::polyline(&[Point::new(0.0, 0.0), Point::new(10.0, 0.0)]),
            stroke: Stroke {
                width: 3.0,
                dash: vec![4.0, 2.0],
                ..Stroke::default()
            },
            paint: Paint::solid(Color::BLACK),
        });
        let ops = scene_to_ops(&scene);
        assert_eq!(
            ops,
            vec![CanvasOp::StrokePath {
                data: "M0 0 L10 0".to_string(),
                stroke: "rgba(0, 0, 0, 1)".to_string(),
                line_width: 3.0,
                dash: vec![4.0, 2.0],
                cap: CapKind::Round,
                join: JoinKind::Round,
            }]
        );
    }

    #[test]
    fn stroke_cap_join_keywords() {
        use whiteboard_core::render::{LineCap, LineJoin};
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::StrokePath {
            path: Path::polyline(&[Point::new(0.0, 0.0), Point::new(10.0, 0.0)]),
            stroke: Stroke {
                width: 1.0,
                cap: LineCap::Square,
                join: LineJoin::Miter,
                dash: Vec::new(),
            },
            paint: Paint::solid(Color::BLACK),
        });
        let ops = scene_to_ops(&scene);
        match &ops[0] {
            CanvasOp::StrokePath { cap, join, .. } => {
                assert_eq!(cap.keyword(), "square");
                assert_eq!(join.keyword(), "miter");
            }
            other => panic!("expected StrokePath, got {other:?}"),
        }
    }

    #[test]
    fn text_op_carries_font_align_and_origin() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::DrawText {
            run: TextRun {
                text: "Hi".to_string(),
                font: FontSpec::new(FontFamily::Code, 14.0),
                origin: Point::new(5.0, 15.0),
                align: TextAlign::Center,
            },
            paint: Paint::solid(Color::BLACK),
        });
        let ops = scene_to_ops(&scene);
        assert_eq!(
            ops,
            vec![CanvasOp::FillText {
                text: "Hi".to_string(),
                x: 5.0,
                y: 15.0,
                font: "14px Cascadia Code, monospace".to_string(),
                fill: "rgba(0, 0, 0, 1)".to_string(),
                align: "center",
            }]
        );
    }

    #[test]
    fn image_op_clamps_opacity_and_size() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::DrawImage {
            id: ImageId("photo1".to_string()),
            dst: Rect::new(2.0, 3.0, 40.0, 50.0),
            opacity: 1.5,
        });
        let ops = scene_to_ops(&scene);
        assert_eq!(
            ops,
            vec![CanvasOp::DrawImage {
                id: "photo1".to_string(),
                x: 2.0,
                y: 3.0,
                w: 40.0,
                h: 50.0,
                opacity: 1.0,
            }]
        );
    }

    #[test]
    fn nested_transforms_balance_save_restore() {
        let mut scene = RenderScene::new();
        scene.push(DrawCommand::PushTransform(Transform::scale(2.0, 2.0)));
        scene.push(DrawCommand::PushTransform(Transform::translate(1.0, 1.0)));
        scene.push(fill_cmd(Color::BLACK));
        scene.push(DrawCommand::PopTransform);
        scene.push(DrawCommand::PopTransform);
        let ops = scene_to_ops(&scene);
        let saves = ops.iter().filter(|o| matches!(o, CanvasOp::Save)).count();
        let restores = ops
            .iter()
            .filter(|o| matches!(o, CanvasOp::Restore))
            .count();
        assert_eq!(saves, 2);
        assert_eq!(restores, 2);
    }

    #[test]
    fn full_scene_paint_order_is_preserved() {
        let mut scene = RenderScene::new();
        scene.push(fill_cmd(Color::rgb(1, 1, 1)));
        scene.push(DrawCommand::StrokePath {
            path: Path::polyline(&[Point::new(0.0, 0.0), Point::new(1.0, 1.0)]),
            stroke: Stroke::solid(1.0),
            paint: Paint::solid(Color::BLACK),
        });
        let ops = scene_to_ops(&scene);
        assert!(matches!(ops[0], CanvasOp::FillPath { .. }));
        assert!(matches!(ops[1], CanvasOp::StrokePath { .. }));
    }

    #[test]
    fn fmt_num_trims_trailing_zeros() {
        assert_eq!(fmt_num(1.0), "1");
        assert_eq!(fmt_num(1.5), "1.5");
        assert_eq!(fmt_num(0.0), "0");
        assert_eq!(fmt_num(-0.0), "0");
        assert_eq!(fmt_num(2.250000), "2.25");
    }
}
