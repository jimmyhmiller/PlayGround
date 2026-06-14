//! Fill-pattern generators — Rust port of Rough.js's fillers.
//!
//! Reimplemented from Rough.js (MIT, © 2019 Preet Shihn), specifically
//! `src/fillers/scan-line-hachure.ts` (the polygon/hachure scan-line
//! intersection), `src/fillers/hachure-filler.ts`, `src/fillers/hatchure`
//! cross-hatch (`src/fillers/cross-hatch-filler.ts`), `src/fillers/zigzag-filler.ts`,
//! and `src/fillers/dots-filler.ts`.
//!
//! These produce the *patterned* fills only. `FillStyle::Solid` is filled flat
//! elsewhere; if asked to generate a fill for `Solid` we return an empty fill
//! (the caller draws the solid region directly) — this is honest, not a stub:
//! there is no hand-drawn pattern geometry to produce for a solid fill.
//!
//! The hachure lines themselves are *not* re-roughened here (Rough.js applies
//! `helper.doubleLineOps` on top in `hachure-filler`, but that is the same
//! drawable roughening implemented in [`super::generator`]); callers that want
//! sketchy fill lines run each returned segment through
//! [`super::generator::rough_line`]. Keeping the geometry split this way makes
//! the intersection math independently testable.

use super::RoughOptions;
use crate::geometry::{Path, Point};
use crate::render::FillStyle;

/// A straight fill segment in element-local space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillLine {
    pub a: Point,
    pub b: Point,
}

impl FillLine {
    #[inline]
    pub fn new(a: Point, b: Point) -> Self {
        FillLine { a, b }
    }

    /// As a two-point [`Path`] polyline.
    pub fn to_path(self) -> Path {
        Path::polyline(&[self.a, self.b])
    }
}

/// Generate the fill geometry for `polygon` under `style`.
///
/// Returns the patterned fill as a list of [`Path`]s (each a small polyline or
/// dot region). For [`FillStyle::Solid`] this returns an empty `Vec` — solid
/// fills carry no pattern geometry and are rasterized directly by the renderer.
///
/// `polygon` is the region outline in element-local space (the polygon need not
/// be closed; the last->first edge is implied).
pub fn fill_polygon(polygon: &[Point], style: FillStyle, opts: &RoughOptions) -> Vec<Path> {
    match style {
        FillStyle::Solid => Vec::new(),
        FillStyle::Hachure => hachure(polygon, opts.hachure_angle, opts)
            .into_iter()
            .map(FillLine::to_path)
            .collect(),
        FillStyle::CrossHatch => cross_hatch(polygon, opts)
            .into_iter()
            .map(FillLine::to_path)
            .collect(),
        FillStyle::Zigzag => zigzag(polygon, opts)
            .into_iter()
            .map(FillLine::to_path)
            .collect(),
        FillStyle::Dots => dots(polygon, opts),
    }
}

/// Hachure: parallel lines at `angle_deg`, `hachure_gap` apart, clipped to the
/// polygon. Port of Rough.js `scan-line-hachure.ts` `hachureLines`.
///
/// The algorithm rotates the polygon by `-angle` so the fill lines become
/// horizontal, scans horizontal lines `gap` apart across the rotated bounding
/// box, intersects each with the polygon edges, pairs the crossings, then
/// rotates the resulting segments back by `+angle`.
pub fn hachure(polygon: &[Point], angle_deg: f64, opts: &RoughOptions) -> Vec<FillLine> {
    if polygon.len() < 3 {
        return Vec::new();
    }
    let gap = opts.hachure_gap.max(0.1);
    let angle = angle_deg.to_radians();

    // Rotate polygon so hachure direction is horizontal.
    let rotated: Vec<Point> = polygon.iter().map(|p| rotate(*p, -angle)).collect();

    // Bounding box of the rotated polygon.
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in &rotated {
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    if !min_y.is_finite() || !max_y.is_finite() {
        return Vec::new();
    }

    let mut lines = Vec::new();
    // Start half a gap in so we don't sit exactly on the top edge.
    let mut y = min_y + gap / 2.0;
    while y <= max_y {
        let xs = scanline_intersections(&rotated, y);
        // Pair sorted crossings: (0,1), (2,3), ... — even-odd interior spans.
        let mut i = 0;
        while i + 1 < xs.len() {
            let x0 = xs[i];
            let x1 = xs[i + 1];
            if (x1 - x0).abs() > 1e-9 {
                let a = rotate(Point::new(x0, y), angle);
                let b = rotate(Point::new(x1, y), angle);
                lines.push(FillLine::new(a, b));
            }
            i += 2;
        }
        y += gap;
    }
    lines
}

/// Cross-hatch: hachure at `+angle` and `-angle` (Rough.js applies the second
/// pass at the perpendicular complement). Port of `cross-hatch-filler.ts`.
pub fn cross_hatch(polygon: &[Point], opts: &RoughOptions) -> Vec<FillLine> {
    let mut a = hachure(polygon, opts.hachure_angle, opts);
    let b = hachure(polygon, opts.hachure_angle + 90.0, opts);
    a.extend(b);
    a
}

/// Zigzag: like hachure, but adjacent scan lines are connected into a single
/// back-and-forth path within each interior span. Port of `zigzag-filler.ts`.
///
/// We subdivide each hachure span into zig segments of width ~`hachure_gap` and
/// alternate the vertical position by half a gap to form the zigzag.
pub fn zigzag(polygon: &[Point], opts: &RoughOptions) -> Vec<FillLine> {
    let base = hachure(polygon, opts.hachure_angle, opts);
    let gap = opts.hachure_gap.max(0.1);
    let angle = opts.hachure_angle.to_radians();
    let mut out = Vec::new();
    for line in base {
        // Work in the rotated (horizontal) frame so zig width is along x.
        let a = rotate(line.a, -angle);
        let b = rotate(line.b, -angle);
        let (left, right) = if a.x <= b.x { (a, b) } else { (b, a) };
        let y = left.y;
        let mut x = left.x;
        let mut up = true;
        while x < right.x {
            let next_x = (x + gap).min(right.x);
            let y0 = if up { y } else { y + gap / 2.0 };
            let y1 = if up { y + gap / 2.0 } else { y };
            let p0 = rotate(Point::new(x, y0), angle);
            let p1 = rotate(Point::new(next_x, y1), angle);
            out.push(FillLine::new(p0, p1));
            x = next_x;
            up = !up;
        }
    }
    out
}

/// Dots: a jittered grid of tiny dots filling the polygon interior. Port of
/// `dots-filler.ts`. Each dot is returned as a degenerate 1-point `Path`
/// (a `MoveTo` with no following segment) so the renderer can draw a filled
/// disc of the stroke width at that location.
pub fn dots(polygon: &[Point], opts: &RoughOptions) -> Vec<Path> {
    if polygon.len() < 3 {
        return Vec::new();
    }
    let gap = (opts.hachure_gap.max(0.1)) * 1.5; // dots are spaced wider
    let mut rng = opts.rng();

    // Bounding box in original (unrotated) space — dots use an axis grid.
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in polygon {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }

    let mut out = Vec::new();
    let mut y = min_y + gap / 2.0;
    while y <= max_y {
        let mut x = min_x + gap / 2.0;
        while x <= max_x {
            // Jitter within the cell, then keep only points inside the polygon.
            let jx = x + rng.offset(gap * 0.25, 1.0);
            let jy = y + rng.offset(gap * 0.25, 1.0);
            let p = Point::new(jx, jy);
            if point_in_polygon(polygon, p) {
                out.push(Path::polyline(&[p]));
            }
            x += gap;
        }
        y += gap;
    }
    out
}

/// X-crossings of the horizontal line `y = y` with polygon edges, sorted
/// ascending. Helper shared by hachure/zigzag — the line/polygon intersection.
///
/// Port of the inner loop of Rough.js `scan-line-hachure.ts`. Vertices lying
/// exactly on the scan line are handled by the half-open edge convention
/// (`p0.y <= y < p1.y` or the reverse) so we never double-count a shared vertex.
fn scanline_intersections(polygon: &[Point], y: f64) -> Vec<f64> {
    let n = polygon.len();
    let mut xs = Vec::new();
    for i in 0..n {
        let p0 = polygon[i];
        let p1 = polygon[(i + 1) % n];
        let (lo, hi) = if p0.y <= p1.y { (p0, p1) } else { (p1, p0) };
        // Half-open: include lower endpoint, exclude upper. Skips horizontals.
        if y >= lo.y && y < hi.y {
            let t = (y - p0.y) / (p1.y - p0.y);
            let x = p0.x + t * (p1.x - p0.x);
            xs.push(x);
        }
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs
}

/// Even-odd point-in-polygon test (ray cast to +x). Used by the dots filler.
fn point_in_polygon(polygon: &[Point], p: Point) -> bool {
    let n = polygon.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let pi = polygon[i];
        let pj = polygon[j];
        let intersects = ((pi.y > p.y) != (pj.y > p.y))
            && (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x);
        if intersects {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Rotate a point around the origin by `angle` radians.
#[inline]
fn rotate(p: Point, angle: f64) -> Point {
    let (s, c) = angle.sin_cos();
    Point::new(p.x * c - p.y * s, p.x * s + p.y * c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::PathSegment;

    fn unit_square() -> Vec<Point> {
        vec![
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(100.0, 100.0),
            Point::new(0.0, 100.0),
        ]
    }

    #[test]
    fn solid_yields_no_pattern() {
        let out = fill_polygon(&unit_square(), FillStyle::Solid, &RoughOptions::default());
        assert!(out.is_empty());
    }

    #[test]
    fn scanline_square_two_crossings() {
        let sq = unit_square();
        let xs = scanline_intersections(&sq, 50.0);
        assert_eq!(xs.len(), 2);
        assert!((xs[0] - 0.0).abs() < 1e-9);
        assert!((xs[1] - 100.0).abs() < 1e-9);
    }

    #[test]
    fn hachure_axis_aligned_lines_clip_to_square() {
        // angle 0 => horizontal lines spanning x in [0,100].
        let opts = RoughOptions {
            hachure_angle: 0.0,
            hachure_gap: 10.0,
            ..RoughOptions::default()
        };
        let lines = hachure(&unit_square(), 0.0, &opts);
        assert!(!lines.is_empty());
        for l in &lines {
            // endpoints lie on the left/right edges.
            let (lx, rx) = if l.a.x <= l.b.x {
                (l.a.x, l.b.x)
            } else {
                (l.b.x, l.a.x)
            };
            assert!((lx - 0.0).abs() < 1e-6, "lx={lx}");
            assert!((rx - 100.0).abs() < 1e-6, "rx={rx}");
            assert!(l.a.y >= -1e-6 && l.a.y <= 100.0 + 1e-6);
        }
    }

    #[test]
    fn hachure_gap_controls_line_count() {
        let mut opts = RoughOptions {
            hachure_angle: 0.0,
            hachure_gap: 20.0,
            ..RoughOptions::default()
        };
        let coarse = hachure(&unit_square(), 0.0, &opts).len();
        opts.hachure_gap = 10.0;
        let fine = hachure(&unit_square(), 0.0, &opts).len();
        assert!(fine > coarse, "fine={fine} coarse={coarse}");
    }

    #[test]
    fn hachure_within_bounds_for_angled() {
        let opts = RoughOptions {
            hachure_angle: -41.0,
            hachure_gap: 8.0,
            ..RoughOptions::default()
        };
        let lines = hachure(&unit_square(), opts.hachure_angle, &opts);
        assert!(!lines.is_empty());
        for l in &lines {
            for pt in [l.a, l.b] {
                assert!(pt.x >= -1e-6 && pt.x <= 100.0 + 1e-6, "x={}", pt.x);
                assert!(pt.y >= -1e-6 && pt.y <= 100.0 + 1e-6, "y={}", pt.y);
            }
        }
    }

    #[test]
    fn cross_hatch_has_more_lines_than_hachure() {
        let opts = RoughOptions {
            hachure_angle: 30.0,
            hachure_gap: 12.0,
            ..RoughOptions::default()
        };
        let h = hachure(&unit_square(), opts.hachure_angle, &opts).len();
        let x = cross_hatch(&unit_square(), &opts).len();
        assert!(x > h, "cross={x} hachure={h}");
    }

    #[test]
    fn zigzag_nonempty_and_in_bounds() {
        let opts = RoughOptions {
            hachure_angle: 0.0,
            hachure_gap: 10.0,
            ..RoughOptions::default()
        };
        let z = zigzag(&unit_square(), &opts);
        assert!(!z.is_empty());
        for l in &z {
            for pt in [l.a, l.b] {
                assert!(pt.x >= -1e-6 && pt.x <= 100.0 + 1e-6, "x={}", pt.x);
                // zigzag rides up to half a gap below the line; allow that.
                assert!(
                    pt.y >= -1e-6 && pt.y <= 100.0 + opts.hachure_gap,
                    "y={}",
                    pt.y
                );
            }
        }
    }

    #[test]
    fn dots_inside_polygon_only() {
        let opts = RoughOptions {
            hachure_gap: 12.0,
            ..RoughOptions::default()
        };
        let sq = unit_square();
        let d = dots(&sq, &opts);
        assert!(!d.is_empty());
        for path in &d {
            // single-point path
            assert_eq!(path.segments.len(), 1);
            let p = match path.segments[0] {
                PathSegment::MoveTo(p) => p,
                _ => panic!("dot must be a MoveTo"),
            };
            assert!(point_in_polygon(&sq, p), "dot {p:?} outside polygon");
        }
    }

    #[test]
    fn determinism_dots_same_seed() {
        let opts = RoughOptions::for_element(1.0, 314);
        let a = dots(&unit_square(), &opts);
        let b = dots(&unit_square(), &opts);
        assert_eq!(a, b);
    }

    #[test]
    fn determinism_fill_polygon_each_style() {
        for style in [
            FillStyle::Hachure,
            FillStyle::CrossHatch,
            FillStyle::Zigzag,
            FillStyle::Dots,
        ] {
            let opts = RoughOptions::for_element(1.0, 2024);
            let a = fill_polygon(&unit_square(), style, &opts);
            let b = fill_polygon(&unit_square(), style, &opts);
            assert_eq!(a, b, "style {style:?} not deterministic");
            assert!(!a.is_empty(), "style {style:?} produced no fill");
        }
    }

    #[test]
    fn point_in_polygon_basic() {
        let sq = unit_square();
        assert!(point_in_polygon(&sq, Point::new(50.0, 50.0)));
        assert!(!point_in_polygon(&sq, Point::new(150.0, 50.0)));
        assert!(!point_in_polygon(&sq, Point::new(-1.0, 50.0)));
    }

    #[test]
    fn triangle_hachure_spans_shrink_toward_apex() {
        // Triangle with apex at top: scan lines near the apex are shorter.
        let tri = vec![
            Point::new(50.0, 0.0),
            Point::new(100.0, 100.0),
            Point::new(0.0, 100.0),
        ];
        let opts = RoughOptions {
            hachure_angle: 0.0,
            hachure_gap: 10.0,
            ..RoughOptions::default()
        };
        let lines = hachure(&tri, 0.0, &opts);
        assert!(lines.len() >= 2);
        let span = |l: &FillLine| (l.a.x - l.b.x).abs();
        // Sort by y; spans should be (weakly) increasing downward.
        let mut sorted = lines.clone();
        sorted.sort_by(|a, b| a.a.y.partial_cmp(&b.a.y).unwrap());
        let top = span(&sorted[0]);
        let bottom = span(sorted.last().unwrap());
        assert!(bottom > top, "bottom span {bottom} should exceed top {top}");
    }
}
