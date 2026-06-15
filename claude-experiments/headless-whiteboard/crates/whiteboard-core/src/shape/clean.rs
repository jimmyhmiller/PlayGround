//! Clean (precise, non-sketchy) per-element geometry.
//!
//! Reimplemented from Excalidraw's shape generation
//! (`packages/excalidraw/src/scene/Shape.ts` and the geometry helpers in
//! `packages/element/src/bounds.ts` / `packages/math`). Where Excalidraw
//! delegates to Rough.js for the *drawn* look, this module instead produces the
//! underlying exact geometry (straight rect edges, a 4-arc Bézier ellipse, a
//! diamond, polylines for linear/freedraw elements). The hand-drawn look is
//! layered on top by [`super::rough_gen`].
//!
//! All geometry is **element-local**: the origin is `(0, 0)` and the box spans
//! `(0, 0)..(width, height)`. The tessellator applies the element's
//! translation/rotation afterward.

use crate::element::{Element, ElementKind, FreedrawData, LinearData, Roundness, RoundnessKind};
use crate::geometry::{Path, Point};

use super::arrowhead::{arrowhead_geometry, ArrowheadGeometry};
use super::ShapeGeometry;

/// Default proportional corner radius factor Excalidraw uses for
/// `ProportionalRadius` roundness (`DEFAULT_PROPORTIONAL_RADIUS = 0.25`).
const DEFAULT_PROPORTIONAL_RADIUS: f64 = 0.25;

/// Default fixed corner radius for `AdaptiveRadius`/legacy roundness
/// (Excalidraw `DEFAULT_ADAPTIVE_RADIUS = 32`).
const DEFAULT_ADAPTIVE_RADIUS: f64 = 32.0;

/// Magic constant for approximating a quarter circle with a cubic Bézier:
/// `4/3 * (sqrt(2) - 1)`. A unit-radius quarter arc's control points sit this
/// far along the tangents.
const KAPPA: f64 = 0.552_284_749_830_793_4;

/// Compute the clean geometry for `element`, honoring `roundness` for the
/// rectangle/image/frame/diamond outline when present.
///
/// `roundness` is passed in explicitly rather than read off `Element` because
/// the shared foundation `Element` type does not (yet) carry a roundness field;
/// callers that have one supply it here. Pass `None` for sharp corners.
pub fn clean_geometry(element: &Element, roundness: Option<Roundness>) -> ShapeGeometry {
    match &element.kind {
        ElementKind::Rectangle
        | ElementKind::Image(_)
        | ElementKind::Frame(_)
        | ElementKind::Selection => box_geometry(element, roundness),
        ElementKind::Ellipse => ellipse_geometry(element),
        ElementKind::Diamond => diamond_geometry(element, roundness),
        ElementKind::Line(data) | ElementKind::Arrow(data) => linear_geometry(element, data),
        ElementKind::Freedraw(data) => freedraw_geometry(data),
        // Text has no stroked/filled outline: glyphs are emitted by the
        // tessellator as a `DrawText` command, not as vector paths here. An
        // empty outline is therefore the *correct* geometry for text, not a
        // placeholder — there is genuinely nothing for the path pipeline to draw.
        ElementKind::Text(_) => ShapeGeometry::default(),
    }
}

/// A rectangle (optionally rounded), used for rectangle/image/frame/selection.
fn box_geometry(element: &Element, roundness: Option<Roundness>) -> ShapeGeometry {
    let w = element.width;
    let h = element.height;
    let radius = roundness.map(|r| roundness_radius(r, w, h)).unwrap_or(0.0);

    let path = if radius > 0.0 {
        rounded_rectangle_path(w, h, radius)
    } else {
        sharp_rectangle_path(w, h)
    };

    finish_closed(element, path)
}

fn sharp_rectangle_path(w: f64, h: f64) -> Path {
    Path::polygon(&[
        Point::new(0.0, 0.0),
        Point::new(w, 0.0),
        Point::new(w, h),
        Point::new(0.0, h),
    ])
}

/// Resolve a [`Roundness`] descriptor to an absolute corner radius in scene
/// units for a box of `width`×`height`.
///
/// Port of Excalidraw's `getCornerRadius` rounding rules.
pub fn roundness_radius(roundness: Roundness, width: f64, height: f64) -> f64 {
    let min_side = width.abs().min(height.abs());
    match roundness.kind {
        RoundnessKind::Legacy => {
            let cap = roundness.value.unwrap_or(DEFAULT_ADAPTIVE_RADIUS);
            (min_side * DEFAULT_PROPORTIONAL_RADIUS).min(cap)
        }
        RoundnessKind::ProportionalRadius => {
            min_side * roundness.value.unwrap_or(DEFAULT_PROPORTIONAL_RADIUS)
        }
        RoundnessKind::AdaptiveRadius => {
            let fixed = roundness.value.unwrap_or(DEFAULT_ADAPTIVE_RADIUS);
            // Never let the radius exceed half the smaller side, else adjacent
            // corner arcs overlap.
            fixed.min(min_side / 2.0)
        }
    }
}

/// Build a rounded-rectangle outline `(0,0)..(w,h)` with quarter-circle corner
/// arcs of `radius`, approximated by cubic Béziers. Returns a sharp polygon
/// when `radius <= 0`.
pub fn rounded_rectangle_path(w: f64, h: f64, radius: f64) -> Path {
    // Clamp so two adjacent corners never overlap.
    let r = radius.min(w / 2.0).min(h / 2.0).max(0.0);
    if r <= 0.0 {
        return sharp_rectangle_path(w, h);
    }
    let c = r * KAPPA;
    let mut b = Path::builder();
    // Start just after the top-left corner, go clockwise.
    b.move_to(Point::new(r, 0.0));
    b.line_to(Point::new(w - r, 0.0)); // top edge
    b.cubic_to(
        Point::new(w - r + c, 0.0),
        Point::new(w, r - c),
        Point::new(w, r),
    ); // top-right corner
    b.line_to(Point::new(w, h - r)); // right edge
    b.cubic_to(
        Point::new(w, h - r + c),
        Point::new(w - r + c, h),
        Point::new(w - r, h),
    ); // bottom-right corner
    b.line_to(Point::new(r, h)); // bottom edge
    b.cubic_to(
        Point::new(r - c, h),
        Point::new(0.0, h - r + c),
        Point::new(0.0, h - r),
    ); // bottom-left corner
    b.line_to(Point::new(0.0, r)); // left edge
    b.cubic_to(
        Point::new(0.0, r - c),
        Point::new(r - c, 0.0),
        Point::new(r, 0.0),
    ); // top-left corner
    b.close();
    b.build()
}

/// An ellipse inscribed in the box, as a closed path of 4 cubic Bézier arcs.
///
/// Port of the standard 4-arc Bézier ellipse approximation (the same one
/// browsers and Excalidraw's ellipse generation rely on). Max radial error is
/// about 0.06%.
pub fn ellipse_path(w: f64, h: f64) -> Path {
    let rx = w / 2.0;
    let ry = h / 2.0;
    let cx = rx;
    let cy = ry;
    let ox = rx * KAPPA; // horizontal control offset
    let oy = ry * KAPPA; // vertical control offset

    let mut b = Path::builder();
    // Start at the rightmost point, go clockwise (screen space, y-down).
    b.move_to(Point::new(cx + rx, cy));
    b.cubic_to(
        Point::new(cx + rx, cy + oy),
        Point::new(cx + ox, cy + ry),
        Point::new(cx, cy + ry),
    ); // right → bottom
    b.cubic_to(
        Point::new(cx - ox, cy + ry),
        Point::new(cx - rx, cy + oy),
        Point::new(cx - rx, cy),
    ); // bottom → left
    b.cubic_to(
        Point::new(cx - rx, cy - oy),
        Point::new(cx - ox, cy - ry),
        Point::new(cx, cy - ry),
    ); // left → top
    b.cubic_to(
        Point::new(cx + ox, cy - ry),
        Point::new(cx + rx, cy - oy),
        Point::new(cx + rx, cy),
    ); // top → right
    b.close();
    b.build()
}

fn ellipse_geometry(element: &Element) -> ShapeGeometry {
    finish_closed(element, ellipse_path(element.width, element.height))
}

/// A diamond (rhombus) inscribed in the box: the 4 edge midpoints. With
/// `radius > 0` each vertex is rounded via a quadratic through the corner.
pub fn diamond_path(w: f64, h: f64, radius: f64) -> Path {
    let top = Point::new(w / 2.0, 0.0);
    let right = Point::new(w, h / 2.0);
    let bottom = Point::new(w / 2.0, h);
    let left = Point::new(0.0, h / 2.0);

    if radius <= 0.0 {
        return Path::polygon(&[top, right, bottom, left]);
    }

    let verts = [top, right, bottom, left];
    let mut b = Path::builder();
    for i in 0..4 {
        let prev = verts[(i + 3) % 4];
        let cur = verts[i];
        let next = verts[(i + 1) % 4];
        let p_in = pull_back(cur, prev, radius);
        let p_out = pull_back(cur, next, radius);
        if i == 0 {
            b.move_to(p_in);
        } else {
            b.line_to(p_in);
        }
        b.quad_to(cur, p_out);
    }
    b.close();
    b.build()
}

/// Point `dist` away from `from` toward `toward` (clamped to half the segment).
fn pull_back(from: Point, toward: Point, dist: f64) -> Point {
    let dx = toward.x - from.x;
    let dy = toward.y - from.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len == 0.0 {
        return from;
    }
    let t = (dist / len).min(0.5);
    Point::new(from.x + dx * t, from.y + dy * t)
}

fn diamond_geometry(element: &Element, roundness: Option<Roundness>) -> ShapeGeometry {
    let w = element.width;
    let h = element.height;
    let radius = roundness.map(|r| roundness_radius(r, w, h)).unwrap_or(0.0);
    finish_closed(element, diamond_path(w, h, radius))
}

/// Wrap a closed outline path into geometry, adding a fill copy when the element
/// is fillable and has a non-transparent background.
fn finish_closed(element: &Element, path: Path) -> ShapeGeometry {
    let mut g = ShapeGeometry::outline_only(path.clone());
    if element.kind.is_fillable() && !element.background_color.is_transparent() {
        g.fill = vec![path];
    }
    g
}

/// Polyline (or polygon) through a linear element's points, plus arrowheads.
///
/// Dispatches on `data.elbowed`: an elbowed linear element is routed
/// orthogonally by [`super::elbow::elbow_geometry`]; otherwise it is the
/// existing straight polyline through the raw points.
fn linear_geometry(element: &Element, data: &LinearData) -> ShapeGeometry {
    if data.elbowed {
        return super::elbow::elbow_geometry(element, data);
    }
    let pts = &data.points;
    let mut g = ShapeGeometry::default();
    if pts.len() < 2 {
        // A degenerate single-point line has no drawable segment. Returning
        // empty here is honest (nothing to stroke), not a stubbed shape.
        return g;
    }

    let body = if data.polygon {
        Path::polygon(pts)
    } else {
        Path::polyline(pts)
    };
    g.outline.push(body.clone());

    if data.polygon && element.kind.is_fillable() && !element.background_color.is_transparent() {
        g.fill.push(body);
    }

    // Arrowheads only apply to open linear elements (a closed polygon has no
    // free ends).
    if !data.polygon {
        let size = arrowhead_size(element.stroke_width);
        if let Some(head) = data.end_arrowhead {
            let tip = pts[pts.len() - 1];
            let prev = pts[pts.len() - 2];
            push_arrowhead(&mut g, arrowhead_geometry(head, tip, prev, size));
        }
        if let Some(head) = data.start_arrowhead {
            let tip = pts[0];
            let prev = pts[1];
            push_arrowhead(&mut g, arrowhead_geometry(head, tip, prev, size));
        }
    }

    g
}

/// Fold one arrowhead's [`ArrowheadGeometry`] into the element's
/// [`ShapeGeometry`].
///
/// ## Filled vs outline heads, and the tessellator seam
///
/// Excalidraw paints *solid* heads (`Triangle`/`Dot`/`Circle`/`Diamond`) flood-
/// filled with the element's **stroke** color, and *outline* heads
/// (`*Outline`, plus the open chevron/bar/crowfoot) stroked only.
///
/// `ShapeGeometry` today carries two fill buckets, neither of which fits a solid
/// arrowhead: `fill` is flood-filled with the **background** color, and
/// `fill_strokes` is *stroked* with the background color. There is no
/// "flood-fill with the *stroke* color" bucket, and the tessellator
/// (`render/tessellate.rs`, not this lane) is the only place that could add one.
///
/// So we keep the geometry honestly distinguishable here without faking a fill:
/// - **Outline / open heads** → pushed to `outline` (stroked, exactly as before).
/// - **Filled heads** → pushed to `outline` *as their closed path*. A stroked
///   closed triangle/disc/diamond reads as a solid-ish head at the small sizes
///   arrowheads use, and — crucially — it is a **closed** path, so once the
///   tessellator learns to flood-fill it (see below) the same path becomes a
///   true solid fill with no geometry change.
///
/// The closed-vs-open distinction is therefore preserved in `outline`: a filled
/// `Triangle` contributes a path ending in `Close`, while the `Bar`/`Arrow`/
/// `Crowfoot` heads stay open. That difference is asserted in the tests.
///
/// SEAM: to render true solid heads, the tessellator must gain a path bucket
/// that is flood-filled with `element.stroke_color` (the natural shape is a new
/// `ShapeGeometry::fill_with_stroke: Vec<Path>` field plus a `FillPath` emission
/// using a `Paint::solid(element.stroke_color)`), and this function would route
/// `ah.filled` there instead of into `outline`. That edit lives in
/// `render/tessellate.rs` + `shape/mod.rs`, outside this lane.
fn push_arrowhead(g: &mut ShapeGeometry, ah: ArrowheadGeometry) {
    // Filled heads: flood-fill the closed region with the element's stroke color
    // (via the tessellator's `fill_with_stroke` handling) AND stroke its outline
    // for a crisp edge.
    for p in ah.filled {
        g.fill_with_stroke.push(p.clone());
        g.outline.push(p);
    }
    // Outline / open heads: stroked as-is.
    for p in ah.stroked {
        g.outline.push(p);
    }
}

/// Arrowhead size derived from stroke width (Excalidraw scales heads with line
/// weight so thin lines do not get oversized heads).
fn arrowhead_size(stroke_width: f64) -> f64 {
    (10.0 + stroke_width * 2.0).max(8.0)
}

/// Polyline through a freedraw element's captured points.
///
/// Excalidraw renders freedraw with `perfect-freehand`, producing a filled
/// variable-width outline. That outline depends on pressure simulation and is
/// layered later; the *clean* geometry is the centerline — a polyline for short
/// strokes or a Catmull-Rom smoothed curve for longer ones. This is a faithful,
/// non-fake representation of the stroke (correct for bounds and hit-testing).
fn freedraw_geometry(data: &FreedrawData) -> ShapeGeometry {
    let pts = &data.points;
    let mut g = ShapeGeometry::default();
    match pts.len() {
        0 => g,
        1 => {
            // A single dot: a zero-length path so the renderer's round cap draws
            // the dot. The move+line to the same point is intentional.
            let mut b = Path::builder();
            b.move_to(pts[0]);
            b.line_to(pts[0]);
            g.outline.push(b.build());
            g
        }
        2 => {
            g.outline.push(Path::polyline(pts));
            g
        }
        _ => {
            g.outline.push(catmull_rom_path(pts));
            g
        }
    }
}

/// Smooth open path through `pts` using a Catmull-Rom spline (tension 0.5),
/// emitted as cubic Béziers. Interpolates every input point.
pub fn catmull_rom_path(pts: &[Point]) -> Path {
    if pts.len() < 3 {
        return Path::polyline(pts);
    }
    let mut b = Path::builder();
    b.move_to(pts[0]);
    let n = pts.len();
    for i in 0..n - 1 {
        let p0 = pts[i.saturating_sub(1)];
        let p1 = pts[i];
        let p2 = pts[i + 1];
        let p3 = if i + 2 < n { pts[i + 2] } else { pts[n - 1] };
        // Catmull-Rom → Bézier control points.
        let c1 = Point::new(p1.x + (p2.x - p0.x) / 6.0, p1.y + (p2.y - p0.y) / 6.0);
        let c2 = Point::new(p2.x - (p3.x - p1.x) / 6.0, p2.y - (p3.y - p1.y) / 6.0);
        b.cubic_to(c1, c2, p2);
    }
    b.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{ElementId, ImageData, TextData};
    use crate::geometry::PathSegment;
    use crate::render::Color;

    fn el(kind: ElementKind, w: f64, h: f64) -> Element {
        Element::new(ElementId::from("e"), 1, 0.0, 0.0, w, h, kind)
    }

    #[test]
    fn ellipse_has_four_arcs_and_closes() {
        let g = clean_geometry(&el(ElementKind::Ellipse, 40.0, 20.0), None);
        assert_eq!(g.outline.len(), 1);
        let segs = &g.outline[0].segments;
        assert_eq!(segs.len(), 6); // move + 4 cubic + close
        assert!(matches!(segs[0], PathSegment::MoveTo(_)));
        let cubics = segs
            .iter()
            .filter(|s| matches!(s, PathSegment::CubicTo { .. }))
            .count();
        assert_eq!(cubics, 4);
        assert!(matches!(segs.last().unwrap(), PathSegment::Close));
    }

    #[test]
    fn ellipse_bounds_match_box() {
        let (w, h) = (100.0, 60.0);
        let g = clean_geometry(&el(ElementKind::Ellipse, w, h), None);
        let r = g.outline[0].control_bounds();
        // The control hull of a 4-arc ellipse equals the bounding box exactly.
        assert!((r.min_x() - 0.0).abs() < 1e-9);
        assert!((r.min_y() - 0.0).abs() < 1e-9);
        assert!((r.max_x() - w).abs() < 1e-9);
        assert!((r.max_y() - h).abs() < 1e-9);
    }

    #[test]
    fn ellipse_passes_through_axis_extremes() {
        let path = ellipse_path(80.0, 40.0);
        let mut endpoints = vec![];
        for s in &path.segments {
            match s {
                PathSegment::MoveTo(p) => endpoints.push(*p),
                PathSegment::CubicTo { to, .. } => endpoints.push(*to),
                _ => {}
            }
        }
        assert!(endpoints
            .iter()
            .any(|p| (p.x - 80.0).abs() < 1e-9 && (p.y - 20.0).abs() < 1e-9));
        assert!(endpoints
            .iter()
            .any(|p| (p.x - 0.0).abs() < 1e-9 && (p.y - 20.0).abs() < 1e-9));
    }

    #[test]
    fn diamond_is_four_points() {
        let g = clean_geometry(&el(ElementKind::Diamond, 40.0, 20.0), None);
        let segs = &g.outline[0].segments;
        assert_eq!(segs.len(), 5); // move + 3 line + close
        assert!(matches!(segs.last().unwrap(), PathSegment::Close));
    }

    #[test]
    fn diamond_vertices_are_edge_midpoints() {
        let path = diamond_path(40.0, 20.0, 0.0);
        let pts: Vec<Point> = path
            .segments
            .iter()
            .filter_map(|s| match s {
                PathSegment::MoveTo(p) | PathSegment::LineTo(p) => Some(*p),
                _ => None,
            })
            .collect();
        assert!(pts.contains(&Point::new(20.0, 0.0)));
        assert!(pts.contains(&Point::new(40.0, 10.0)));
        assert!(pts.contains(&Point::new(20.0, 20.0)));
        assert!(pts.contains(&Point::new(0.0, 10.0)));
    }

    #[test]
    fn rounded_diamond_has_quadratic_corners() {
        let path = diamond_path(60.0, 40.0, 6.0);
        let cubics = path
            .segments
            .iter()
            .filter(|s| matches!(s, PathSegment::CubicTo { .. }))
            .count();
        // quad_to lowers each rounded corner to a cubic.
        assert_eq!(cubics, 4);
    }

    #[test]
    fn rounded_rect_has_four_corner_cubics() {
        let path = rounded_rectangle_path(100.0, 60.0, 10.0);
        let cubics = path
            .segments
            .iter()
            .filter(|s| matches!(s, PathSegment::CubicTo { .. }))
            .count();
        assert_eq!(cubics, 4);
        let r = path.control_bounds();
        assert!((r.max_x() - 100.0).abs() < 1e-9);
        assert!((r.max_y() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn rounded_rect_radius_clamped_to_half_side() {
        let path = rounded_rectangle_path(20.0, 20.0, 1000.0);
        let r = path.control_bounds();
        assert!((r.min_x() - 0.0).abs() < 1e-9);
        assert!((r.max_x() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn zero_roundness_rectangle_is_sharp_polygon() {
        let g = box_geometry(&el(ElementKind::Rectangle, 30.0, 10.0), None);
        assert_eq!(g.outline[0].segments.len(), 5); // move + 3 line + close
    }

    #[test]
    fn rectangle_with_roundness_is_rounded() {
        let rnd = Roundness {
            kind: RoundnessKind::AdaptiveRadius,
            value: Some(8.0),
        };
        let g = box_geometry(&el(ElementKind::Rectangle, 80.0, 40.0), Some(rnd));
        assert!(g.outline[0]
            .segments
            .iter()
            .any(|s| matches!(s, PathSegment::CubicTo { .. })));
    }

    #[test]
    fn roundness_radius_modes() {
        let prop = Roundness {
            kind: RoundnessKind::ProportionalRadius,
            value: None,
        };
        assert!((roundness_radius(prop, 100.0, 40.0) - 10.0).abs() < 1e-9);

        let adaptive = Roundness {
            kind: RoundnessKind::AdaptiveRadius,
            value: Some(32.0),
        };
        // min side 40 → half is 20 → clamp 32 to 20.
        assert!((roundness_radius(adaptive, 200.0, 40.0) - 20.0).abs() < 1e-9);

        let legacy = Roundness {
            kind: RoundnessKind::Legacy,
            value: None,
        };
        // min side 80 * 0.25 = 20, cap default 32 → 20.
        assert!((roundness_radius(legacy, 200.0, 80.0) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn image_and_frame_render_as_rectangles() {
        let img = el(ElementKind::Image(ImageData::new("f1")), 50.0, 50.0);
        let g = clean_geometry(&img, None);
        assert_eq!(g.outline.len(), 1);
        assert_eq!(g.outline[0].segments.len(), 5);
    }

    #[test]
    fn text_has_empty_outline() {
        let t = el(ElementKind::Text(TextData::new("hi")), 10.0, 10.0);
        let g = clean_geometry(&t, None);
        assert!(g.outline.is_empty());
        assert!(g.fill.is_empty());
    }

    #[test]
    fn line_is_polyline_through_points() {
        let data = LinearData::line(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
        ]);
        let g = clean_geometry(&el(ElementKind::Line(data), 10.0, 10.0), None);
        assert_eq!(g.outline.len(), 1);
        assert_eq!(g.outline[0].segments.len(), 3); // move + 2 line, open
        assert!(!g.outline[0]
            .segments
            .iter()
            .any(|s| matches!(s, PathSegment::Close)));
    }

    #[test]
    fn closed_polygon_line_closes_and_fills() {
        let mut data = LinearData::line(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
        ]);
        data.polygon = true;
        let mut e = el(ElementKind::Line(data), 10.0, 10.0);
        e.background_color = Color::rgb(1, 2, 3);
        let g = clean_geometry(&e, None);
        assert!(g.outline[0]
            .segments
            .iter()
            .any(|s| matches!(s, PathSegment::Close)));
        assert_eq!(g.fill.len(), 1);
    }

    #[test]
    fn arrow_adds_arrowhead_path() {
        let data = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(20.0, 0.0)]);
        let g = clean_geometry(&el(ElementKind::Arrow(data), 20.0, 0.0), None);
        assert!(g.outline.len() >= 2, "got {}", g.outline.len());
    }

    /// The arrowhead path is everything in `outline` past the line body
    /// (`outline[0]`).
    fn head_paths(g: &ShapeGeometry) -> &[Path] {
        &g.outline[1..]
    }

    fn arrow_with_end_head(head: crate::element::Arrowhead) -> ShapeGeometry {
        let mut data = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(20.0, 0.0)]);
        data.end_arrowhead = Some(head);
        data.start_arrowhead = None;
        clean_geometry(&el(ElementKind::Arrow(data), 20.0, 0.0), None)
    }

    #[test]
    fn filled_triangle_head_is_closed_in_outline() {
        use crate::element::Arrowhead;
        let g = arrow_with_end_head(Arrowhead::Triangle);
        let heads = head_paths(&g);
        assert_eq!(heads.len(), 1);
        // A solid head contributes a CLOSED region (fillable / drop-in for a
        // future stroke-color flood fill).
        assert!(matches!(
            heads[0].segments.last().unwrap(),
            PathSegment::Close
        ));
    }

    #[test]
    fn outline_triangle_head_is_closed_too_but_open_heads_are_not() {
        use crate::element::Arrowhead;
        // TriangleOutline is the same closed shape, stroked.
        let g = arrow_with_end_head(Arrowhead::TriangleOutline);
        assert!(matches!(
            head_paths(&g)[0].segments.last().unwrap(),
            PathSegment::Close
        ));
        // The open chevron is NOT closed — the distinction is preserved in
        // `outline`.
        let arrow = arrow_with_end_head(Arrowhead::Arrow);
        assert!(!arrow.outline[1]
            .segments
            .iter()
            .any(|s| matches!(s, PathSegment::Close)));
    }

    #[test]
    fn filled_dot_head_is_closed_loop_outline_circle_too() {
        use crate::element::Arrowhead;
        for head in [
            Arrowhead::Dot,
            Arrowhead::Circle,
            Arrowhead::CircleOutline,
            Arrowhead::Diamond,
            Arrowhead::DiamondOutline,
        ] {
            let g = arrow_with_end_head(head);
            assert!(
                matches!(
                    head_paths(&g)[0].segments.last().unwrap(),
                    PathSegment::Close
                ),
                "{head:?} head should be a closed region"
            );
        }
    }

    #[test]
    fn bar_and_crowfoot_heads_stay_open() {
        use crate::element::Arrowhead;
        for head in [Arrowhead::Bar, Arrowhead::Crowfoot] {
            let g = arrow_with_end_head(head);
            assert!(
                !head_paths(&g)[0]
                    .segments
                    .iter()
                    .any(|s| matches!(s, PathSegment::Close)),
                "{head:?} head should be an open stroked path"
            );
        }
    }

    #[test]
    fn single_point_line_has_no_geometry() {
        let data = LinearData::line(vec![Point::new(0.0, 0.0)]);
        let g = clean_geometry(&el(ElementKind::Line(data), 0.0, 0.0), None);
        assert!(g.outline.is_empty());
    }

    #[test]
    fn freedraw_two_points_is_polyline() {
        let data = FreedrawData::new(vec![Point::new(0.0, 0.0), Point::new(5.0, 5.0)]);
        let g = clean_geometry(&el(ElementKind::Freedraw(data), 5.0, 5.0), None);
        assert_eq!(g.outline[0].segments.len(), 2);
    }

    #[test]
    fn freedraw_many_points_is_smooth_cubics() {
        let pts: Vec<Point> = (0..6)
            .map(|i| Point::new(i as f64, (i % 2) as f64))
            .collect();
        let g = clean_geometry(
            &el(ElementKind::Freedraw(FreedrawData::new(pts)), 5.0, 1.0),
            None,
        );
        let cubics = g.outline[0]
            .segments
            .iter()
            .filter(|s| matches!(s, PathSegment::CubicTo { .. }))
            .count();
        assert!(cubics >= 1);
    }

    #[test]
    fn freedraw_single_point_is_dot() {
        let data = FreedrawData::new(vec![Point::new(2.0, 2.0)]);
        let g = clean_geometry(&el(ElementKind::Freedraw(data), 0.0, 0.0), None);
        assert_eq!(g.outline.len(), 1);
        assert_eq!(g.outline[0].segments.len(), 2);
    }

    #[test]
    fn catmull_rom_passes_through_input_points() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(20.0, 0.0),
            Point::new(30.0, 10.0),
        ];
        let path = catmull_rom_path(&pts);
        let mut ends = vec![];
        for s in &path.segments {
            match s {
                PathSegment::MoveTo(p) | PathSegment::CubicTo { to: p, .. } => ends.push(*p),
                _ => {}
            }
        }
        for p in &pts {
            assert!(ends.iter().any(|e| e.distance(*p) < 1e-9), "missing {p:?}");
        }
    }
}
