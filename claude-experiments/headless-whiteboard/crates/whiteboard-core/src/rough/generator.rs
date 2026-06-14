//! Drawable geometry generators — Rust port of Rough.js's sketch core.
//!
//! Reimplemented from Rough.js (MIT, © 2019 Preet Shihn), specifically
//! `src/renderer.ts` (the `_line`, `line`, `linearPath`, `polygon`,
//! `rectangle`, `ellipse`, `_curve` / `_bezierTo` helpers) and the
//! `_doubleLine` double-stroke logic in `src/core.ts`. We emit
//! [`crate::geometry::Path`] values in element-local space instead of an SVG
//! path string, but the roughening math (per-point random offsets scaled by
//! `roughness`, control-point bowing, the two-pass `_doubleLine`) follows the
//! upstream algorithm so a given seed reproduces a comparable sketch.
//!
//! The seeded RNG is [`super::RoughRng`]; this module never introduces its own
//! randomness source.

use super::{RoughOptions, RoughRng};
use crate::geometry::{Path, PathBuilder, Point};

/// Result of generating a drawable: the roughened outline as a single [`Path`].
///
/// Multiple stroke passes are concatenated as independent subpaths (each begun
/// by its own `MoveTo`), which is exactly how Rough.js layers its double
/// stroke.
pub type Drawable = Path;

/// Per-point maximum random offset, following Rough.js's `_offset`:
/// the offset magnitude grows with `roughness` but is clamped against
/// `max_randomness_offset` so large `roughness` values stay bounded.
#[inline]
fn roughness_offset(opts: &RoughOptions) -> f64 {
    // Rough.js scales the base offset by roughness and caps it.
    (opts.max_randomness_offset * opts.roughness).min(opts.max_randomness_offset.max(1.0) * 2.0)
}

/// Port of Rough.js `_line` (the single roughened segment) folded into the
/// `_doubleLine` driver. Pushes `move`-prefixed roughened polyline(s)
/// approximating the straight segment `a -> b` into `b_out`.
///
/// `move_only_first` controls whether this pass emits its own `MoveTo`
/// (Rough.js's second pass starts a fresh subpath without re-issuing the move
/// in the same way; we always issue a `MoveTo` per pass to keep subpaths
/// independent, matching how the renderer concatenates passes).
fn rough_segment(
    a: Point,
    b: Point,
    opts: &RoughOptions,
    rng: &mut RoughRng,
    out: &mut PathBuilder,
) {
    let length = a.distance(b);
    // Rough.js scales the random offset down for very short lines and up to a
    // cap for long ones (see `_line`: `roughnessGain`).
    let mut offset = roughness_offset(opts);
    if length < 200.0 {
        // keep offset as-is for short lines
    } else if length > 500.0 {
        offset *= 0.4;
    } else {
        // linearly damp between 200 and 500
        offset *= 1.0 - ((length - 200.0) / 300.0) * 0.6;
    }

    // Bowing: Rough.js displaces the mid control points perpendicular-ish using
    // `bowing * maxRandomnessOffset * (b - a) / 200`.
    let half_offset = offset / 2.0;
    let bow = opts.bowing * opts.max_randomness_offset * (b.distance(a)) / 200.0;

    // Two diverging mid control points, each jittered, as in Rough.js's cubic.
    let mid1 = jittered_mid(a, b, 0.5, bow, half_offset, opts, rng);
    let mid2 = jittered_mid(a, b, 0.5, bow, half_offset, opts, rng);

    let start = jitter(a, offset, opts, rng);
    let end = jitter(b, offset, opts, rng);

    out.move_to(start);
    out.cubic_to(mid1, mid2, end);
}

/// A point on segment `a->b` at parameter `t`, displaced by bowing along the
/// segment normal plus an independent random jitter — Rough.js's control-point
/// construction.
fn jittered_mid(
    a: Point,
    b: Point,
    t: f64,
    bow: f64,
    offset: f64,
    opts: &RoughOptions,
    rng: &mut RoughRng,
) -> Point {
    let base = Point::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
    // Perpendicular to the segment direction.
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len = (dx * dx + dy * dy).sqrt().max(1e-9);
    let nx = -dy / len;
    let ny = dx / len;
    let bow_sign = rng.range(-1.0, 1.0);
    Point::new(
        base.x + nx * bow * bow_sign + rng.offset(offset, opts.roughness),
        base.y + ny * bow * bow_sign + rng.offset(offset, opts.roughness),
    )
}

/// Jitter a single endpoint by up to `offset`, scaled by roughness.
#[inline]
fn jitter(p: Point, offset: f64, opts: &RoughOptions, rng: &mut RoughRng) -> Point {
    Point::new(
        p.x + rng.offset(offset, opts.roughness),
        p.y + rng.offset(offset, opts.roughness),
    )
}

/// Double-stroke a single straight segment: emit `stroke_passes` roughened
/// copies as independent subpaths (Rough.js's `_doubleLine`).
fn double_line(a: Point, b: Point, opts: &RoughOptions, rng: &mut RoughRng, out: &mut PathBuilder) {
    let passes = opts.stroke_passes.max(1);
    for _ in 0..passes {
        rough_segment(a, b, opts, rng, out);
    }
}

/// Roughened line from `a` to `b`. Port of Rough.js `line`.
pub fn rough_line(a: Point, b: Point, opts: &RoughOptions) -> Drawable {
    let mut rng = opts.rng();
    let mut out = Path::builder();
    double_line(a, b, opts, &mut rng, &mut out);
    out.build()
}

/// Roughened open polyline through `points`. Port of Rough.js `linearPath`
/// with `close = false`.
pub fn rough_linear_path(points: &[Point], opts: &RoughOptions) -> Drawable {
    linear_path_impl(points, false, opts)
}

/// Roughened closed polygon through `points`. Port of Rough.js `polygon`
/// (`linearPath` with `close = true`).
pub fn rough_polygon(points: &[Point], opts: &RoughOptions) -> Drawable {
    linear_path_impl(points, true, opts)
}

fn linear_path_impl(points: &[Point], close: bool, opts: &RoughOptions) -> Drawable {
    let mut rng = opts.rng();
    let mut out = Path::builder();
    if points.len() < 2 {
        return out.build();
    }
    for w in points.windows(2) {
        double_line(w[0], w[1], opts, &mut rng, &mut out);
    }
    if close && points.len() > 2 {
        let last = *points.last().unwrap();
        let first = points[0];
        double_line(last, first, opts, &mut rng, &mut out);
    }
    out.build()
}

/// Roughened axis-aligned rectangle of size `w` x `h` with its top-left corner
/// at the local origin. Port of Rough.js `rectangle`.
pub fn rough_rectangle(w: f64, h: f64, opts: &RoughOptions) -> Drawable {
    let p0 = Point::new(0.0, 0.0);
    let p1 = Point::new(w, 0.0);
    let p2 = Point::new(w, h);
    let p3 = Point::new(0.0, h);
    // Rough.js draws the four edges as separate double-lines.
    let mut rng = opts.rng();
    let mut out = Path::builder();
    double_line(p0, p1, opts, &mut rng, &mut out);
    double_line(p1, p2, opts, &mut rng, &mut out);
    double_line(p2, p3, opts, &mut rng, &mut out);
    double_line(p3, p0, opts, &mut rng, &mut out);
    out.build()
}

/// Roughened ellipse of size `w` x `h` centered at the local origin's box
/// (so it spans `[0, w] x [0, h]`, centered at `(w/2, h/2)`). Port of Rough.js
/// `ellipse` / `_computeEllipsePoints` + `_curve`.
pub fn rough_ellipse(w: f64, h: f64, opts: &RoughOptions) -> Drawable {
    let cx = w / 2.0;
    let cy = h / 2.0;
    let rx = (w / 2.0).abs();
    let ry = (h / 2.0).abs();

    // Rough.js increases the number of sample points and the per-step angle
    // jitter with the ellipse size and roughness. We follow the same
    // `_computeEllipsePoints` structure.
    let mut rng = opts.rng();

    // `ellipseInc` and step count from Rough.js (curveStepCount default 9).
    let curve_step_count = 9.0_f64;
    let increment = (std::f64::consts::PI * 2.0) / curve_step_count;
    let rx_off = roughness_offset(opts);
    let ry_off = rx_off;

    let mut points: Vec<Point> = Vec::new();
    // Two overlapping loops, as Rough.js does, to close the curve cleanly.
    let offset = increment;
    let mut angle = rng.range(-offset, offset);
    let end = std::f64::consts::PI * 2.0 + angle - 1e-9;
    while angle < end {
        let cos = angle.cos();
        let sin = angle.sin();
        points.push(Point::new(
            cx + cos * rx + rng.offset(rx_off, opts.roughness),
            cy + sin * ry + rng.offset(ry_off, opts.roughness),
        ));
        angle += increment;
    }
    // Close the loop back onto the first sampled angle.
    if let Some(&first) = points.first() {
        points.push(first);
    }

    let mut out = Path::builder();
    curve_through(&points, &mut out);
    out.build()
}

/// Catmull-Rom-ish smooth curve through `points`, lowered to cubic Béziers —
/// Rough.js's `_curve` / `_bezierTo`. Used by the ellipse generator.
fn curve_through(points: &[Point], out: &mut PathBuilder) {
    let n = points.len();
    if n < 3 {
        if n >= 1 {
            out.move_to(points[0]);
            for p in &points[1..] {
                out.line_to(*p);
            }
        }
        return;
    }

    out.move_to(points[1]);
    // Rough.js uses a tension of 0; cubic control points derived from
    // neighboring samples (the classic Catmull-Rom -> Bézier conversion).
    let mut i = 1;
    while i + 2 < n {
        let p0 = points[i - 1];
        let p1 = points[i];
        let p2 = points[i + 1];
        let p3 = points[i + 2];
        let c1 = Point::new(p1.x + (p2.x - p0.x) / 6.0, p1.y + (p2.y - p0.y) / 6.0);
        let c2 = Point::new(p2.x - (p3.x - p1.x) / 6.0, p2.y - (p3.y - p1.y) / 6.0);
        out.cubic_to(c1, c2, p2);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::PathSegment;

    fn clean_opts(seed: u32) -> RoughOptions {
        // roughness 0 and no bowing => essentially clean output.
        RoughOptions {
            roughness: 0.0,
            bowing: 0.0,
            ..RoughOptions::for_element(0.0, seed)
        }
    }

    fn first_point(path: &Path) -> Point {
        match path.segments[0] {
            PathSegment::MoveTo(p) => p,
            _ => panic!("path must start with MoveTo"),
        }
    }

    #[test]
    fn line_is_nonempty() {
        let p = rough_line(
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            &RoughOptions::default(),
        );
        assert!(!p.is_empty());
        assert!(matches!(p.segments[0], PathSegment::MoveTo(_)));
    }

    #[test]
    fn line_double_stroke_has_two_subpaths() {
        let opts = RoughOptions::default();
        assert_eq!(opts.stroke_passes, 2);
        let p = rough_line(Point::new(0.0, 0.0), Point::new(100.0, 0.0), &opts);
        let moves = p
            .segments
            .iter()
            .filter(|s| matches!(s, PathSegment::MoveTo(_)))
            .count();
        assert_eq!(moves, 2, "two passes => two subpaths");
    }

    #[test]
    fn determinism_same_seed() {
        let opts = RoughOptions::for_element(1.5, 777);
        let a = rough_rectangle(120.0, 80.0, &opts);
        let b = rough_rectangle(120.0, 80.0, &opts);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seed_differs() {
        let a = rough_ellipse(100.0, 60.0, &RoughOptions::for_element(1.0, 1));
        let b = rough_ellipse(100.0, 60.0, &RoughOptions::for_element(1.0, 2));
        assert_ne!(a, b);
    }

    #[test]
    fn clean_line_endpoints_close_to_exact() {
        let a = Point::new(10.0, 20.0);
        let b = Point::new(110.0, 20.0);
        let opts = clean_opts(5);
        let path = rough_line(a, b, &opts);
        // With roughness 0, the first subpath should start at ~a and the cubic
        // should end at ~b.
        let start = first_point(&path);
        assert!((start.x - a.x).abs() < 1e-6 && (start.y - a.y).abs() < 1e-6);
        let end = match path.segments[1] {
            PathSegment::CubicTo { to, .. } => to,
            _ => panic!("expected cubic"),
        };
        assert!((end.x - b.x).abs() < 1e-6 && (end.y - b.y).abs() < 1e-6);
    }

    #[test]
    fn rectangle_stays_in_bounds_when_clean() {
        let opts = clean_opts(9);
        let w = 200.0;
        let h = 100.0;
        let path = rough_rectangle(w, h, &opts);
        let bounds = path.control_bounds();
        // Clean rectangle's control points should stay within the box (+/- tiny).
        assert!(bounds.min_x() >= -1e-6);
        assert!(bounds.min_y() >= -1e-6);
        assert!(bounds.max_x() <= w + 1e-6);
        assert!(bounds.max_y() <= h + 1e-6);
    }

    #[test]
    fn rough_rectangle_bounds_within_offset_envelope() {
        let opts = RoughOptions::for_element(1.0, 3);
        let w = 200.0;
        let h = 100.0;
        let path = rough_rectangle(w, h, &opts);
        let bounds = path.control_bounds();
        // Roughened output may exceed the box but only by a bounded envelope.
        let env = opts.max_randomness_offset * 4.0 + opts.bowing * opts.max_randomness_offset;
        assert!(bounds.min_x() >= -env, "min_x={}", bounds.min_x());
        assert!(bounds.max_x() <= w + env, "max_x={}", bounds.max_x());
        assert!(bounds.min_y() >= -env, "min_y={}", bounds.min_y());
        assert!(bounds.max_y() <= h + env, "max_y={}", bounds.max_y());
    }

    #[test]
    fn polygon_closes_back() {
        let pts = [
            Point::new(0.0, 0.0),
            Point::new(50.0, 0.0),
            Point::new(25.0, 40.0),
        ];
        let opts = RoughOptions::for_element(1.0, 11);
        let path = rough_polygon(&pts, &opts);
        // 3 edges * 2 passes = 6 subpaths.
        let moves = path
            .segments
            .iter()
            .filter(|s| matches!(s, PathSegment::MoveTo(_)))
            .count();
        assert_eq!(moves, 6);
    }

    #[test]
    fn linear_path_open_has_no_closing_edge() {
        let pts = [
            Point::new(0.0, 0.0),
            Point::new(50.0, 0.0),
            Point::new(25.0, 40.0),
        ];
        let opts = RoughOptions::for_element(1.0, 11);
        let path = rough_linear_path(&pts, &opts);
        // 2 edges * 2 passes = 4 subpaths (no closing edge).
        let moves = path
            .segments
            .iter()
            .filter(|s| matches!(s, PathSegment::MoveTo(_)))
            .count();
        assert_eq!(moves, 4);
    }

    #[test]
    fn ellipse_nonempty_and_centered() {
        let opts = RoughOptions::for_element(1.0, 42);
        let path = rough_ellipse(100.0, 60.0, &opts);
        assert!(!path.is_empty());
        let b = path.control_bounds();
        // Center of the control bounds should be near (50, 30).
        let cx = (b.min_x() + b.max_x()) / 2.0;
        let cy = (b.min_y() + b.max_y()) / 2.0;
        assert!((cx - 50.0).abs() < 20.0, "cx={cx}");
        assert!((cy - 30.0).abs() < 20.0, "cy={cy}");
    }

    #[test]
    fn degenerate_inputs_dont_panic() {
        let opts = RoughOptions::default();
        assert!(rough_linear_path(&[], &opts).is_empty());
        assert!(rough_linear_path(&[Point::new(1.0, 1.0)], &opts).is_empty());
        // single-edge polygon (2 points) won't add a closing edge.
        let two = [Point::new(0.0, 0.0), Point::new(10.0, 0.0)];
        let p = rough_polygon(&two, &opts);
        let moves = p
            .segments
            .iter()
            .filter(|s| matches!(s, PathSegment::MoveTo(_)))
            .count();
        assert_eq!(moves, 2);
    }
}
