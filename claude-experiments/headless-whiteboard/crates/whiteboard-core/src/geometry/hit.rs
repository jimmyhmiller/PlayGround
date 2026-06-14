//! Hit-testing: is a scene point on / inside an element?
//!
//! Reimplemented from Excalidraw's collision logic
//! (`packages/element/src/collision.ts` — `hitElementItself`,
//! `isPointInShape`/`distanceToBindableElement`, point-to-segment distance).
//! No JavaScript is vendored; this is a Rust reimplementation.
//!
//! Strategy (matching upstream):
//! - For **fillable closed** shapes (rectangle, ellipse, diamond, closed
//!   polygons) a point hits if it is *inside* the shape, OR within `tolerance`
//!   of the outline.
//! - For **open / non-fillable** shapes (lines, arrows, freedraw, text/image
//!   outlines treated as strokes) a point hits if it is within `tolerance` of
//!   the outline.
//!
//! Rotation is honored by transforming the test point into element-local space
//! (inverse-rotating it about [`Element::center`]) before the unrotated inside
//! test, which is cheaper and exactly equivalent to rotating the shape.

use super::{element_line_segments, point_rotate_rads, Point};
use crate::element::{Element, ElementKind};

/// Distance from point `p` to the segment `a`–`b`.
///
/// Projects `p` onto the segment, clamps the projection parameter to `[0, 1]`,
/// and returns the Euclidean distance to that clamped closest point. A zero
/// length segment degenerates to the distance to the shared endpoint.
pub fn point_distance_to_segment(p: Point, a: Point, b: Point) -> f64 {
    let abx = b.x - a.x;
    let aby = b.y - a.y;
    let len_sq = abx * abx + aby * aby;
    if len_sq == 0.0 {
        return p.distance(a);
    }
    let apx = p.x - a.x;
    let apy = p.y - a.y;
    let t = ((apx * abx + apy * aby) / len_sq).clamp(0.0, 1.0);
    let closest = Point::new(a.x + t * abx, a.y + t * aby);
    p.distance(closest)
}

/// Minimum distance from `p` to any of an element's outline segments (scene
/// space). Returns `f64::INFINITY` for an element with no outline.
pub fn distance_to_outline(element: &Element, p: Point) -> f64 {
    element_line_segments(element)
        .iter()
        .map(|(a, b)| point_distance_to_segment(p, *a, *b))
        .fold(f64::INFINITY, f64::min)
}

/// Whether `point` (scene space) hits `element` within `tolerance` scene units.
///
/// `tolerance` is typically derived from the stroke width / a few device pixels
/// by the interaction layer.
pub fn hit_test(element: &Element, point: Point, tolerance: f64) -> bool {
    // Transform the test point into the element's *unrotated* local frame by
    // inverse-rotating about the center. Then inside/outline tests can ignore
    // rotation. (distance is rotation-invariant, so tolerance carries over.)
    let center = element.center();
    let local = point_rotate_rads(point, center, -element.angle);

    if element.kind.is_fillable() && point_inside_filled(element, local) {
        return true;
    }

    // Outline proximity. We measure against the *unrotated* outline using the
    // inverse-rotated point, which is equidistant to measuring the rotated
    // outline against the original point.
    let mut unrotated = element.clone();
    unrotated.angle = 0.0;
    distance_to_outline(&unrotated, local) <= tolerance
}

/// Inside test for fillable closed shapes, in the element's unrotated local
/// frame (so `local` is the inverse-rotated test point).
fn point_inside_filled(element: &Element, local: Point) -> bool {
    match &element.kind {
        ElementKind::Rectangle => element.raw_box().contains(local),
        ElementKind::Ellipse => point_in_ellipse(element, local),
        ElementKind::Diamond => point_in_diamond(element, local),
        ElementKind::Line(data) if data.polygon => {
            // Closed polygon: scene-space relative points (origin added). Since
            // we pass the already-inverse-rotated `local`, build the polygon
            // unrotated too.
            let poly: Vec<Point> = data
                .points
                .iter()
                .map(|p| Point::new(element.x + p.x, element.y + p.y))
                .collect();
            point_in_polygon(local, &poly)
        }
        _ => false,
    }
}

/// Standard implicit-equation ellipse interior test.
fn point_in_ellipse(element: &Element, p: Point) -> bool {
    let cx = element.x + element.width / 2.0;
    let cy = element.y + element.height / 2.0;
    let rx = element.width / 2.0;
    let ry = element.height / 2.0;
    if rx == 0.0 || ry == 0.0 {
        return false;
    }
    let nx = (p.x - cx) / rx;
    let ny = (p.y - cy) / ry;
    nx * nx + ny * ny <= 1.0
}

/// Diamond interior test via the L1-style implicit inequality
/// `|x|/a + |y|/b <= 1` relative to the center.
fn point_in_diamond(element: &Element, p: Point) -> bool {
    let cx = element.x + element.width / 2.0;
    let cy = element.y + element.height / 2.0;
    let a = element.width / 2.0;
    let b = element.height / 2.0;
    if a == 0.0 || b == 0.0 {
        return false;
    }
    (p.x - cx).abs() / a + (p.y - cy).abs() / b <= 1.0
}

/// Even-odd ray-casting point-in-polygon test.
fn point_in_polygon(p: Point, poly: &[Point]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let pi = poly[i];
        let pj = poly[j];
        let intersect = ((pi.y > p.y) != (pj.y > p.y))
            && (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{ElementId, FreedrawData, LinearData};
    use std::f64::consts::PI;

    fn el(kind: ElementKind, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from("t"), 1, x, y, w, h, kind)
    }

    #[test]
    fn distance_to_segment_3_4_5() {
        // Segment along the x-axis from (0,0) to (10,0); point (3,4) projects to
        // (3,0) → distance 4. Point (-? ) etc.
        let a = Point::new(0.0, 0.0);
        let b = Point::new(10.0, 0.0);
        assert!((point_distance_to_segment(Point::new(3.0, 4.0), a, b) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn distance_to_segment_clamps_to_endpoint() {
        // (3,4) relative to a vertical segment, but past the far endpoint so the
        // closest point is the endpoint (0,0): classic 3-4-5 → 5.
        let a = Point::new(0.0, 0.0);
        let b = Point::new(0.0, -100.0); // segment goes the other way
        let d = point_distance_to_segment(Point::new(3.0, 4.0), a, b);
        assert!((d - 5.0).abs() < 1e-9, "d={d}");
    }

    #[test]
    fn distance_to_zero_length_segment_is_point_distance() {
        let a = Point::new(2.0, 2.0);
        let d = point_distance_to_segment(Point::new(5.0, 6.0), a, a);
        assert!((d - 5.0).abs() < 1e-9, "d={d}");
    }

    #[test]
    fn rectangle_inside_hits() {
        let e = el(ElementKind::Rectangle, 0.0, 0.0, 100.0, 50.0);
        assert!(hit_test(&e, Point::new(50.0, 25.0), 1.0));
        assert!(!hit_test(&e, Point::new(200.0, 200.0), 1.0));
    }

    #[test]
    fn rectangle_edge_within_tolerance_hits() {
        let e = el(ElementKind::Rectangle, 0.0, 0.0, 100.0, 50.0);
        // 3 units outside the right edge, tolerance 4 → hit; tolerance 2 → miss.
        assert!(hit_test(&e, Point::new(103.0, 25.0), 4.0));
        assert!(!hit_test(&e, Point::new(103.0, 25.0), 2.0));
    }

    #[test]
    fn rotated_rectangle_inside_hits() {
        // 100x40 box at (10,20), center (60,40), rotated 90°. A point that is
        // inside the rotated box but OUTSIDE the unrotated box must still hit.
        let mut e = el(ElementKind::Rectangle, 10.0, 20.0, 100.0, 40.0);
        e.angle = PI / 2.0;
        // The rotated box spans x∈[40,80], y∈[-10,90]. (50, 85) is inside that
        // but outside the unrotated box (which is y∈[20,60]).
        assert!(hit_test(&e, Point::new(50.0, 85.0), 1.0));
        // A point outside the rotated box.
        assert!(!hit_test(&e, Point::new(100.0, 40.0), 1.0));
    }

    #[test]
    fn ellipse_inside_and_outside() {
        let e = el(ElementKind::Ellipse, 0.0, 0.0, 200.0, 100.0);
        assert!(hit_test(&e, Point::new(100.0, 50.0), 1.0)); // center
        assert!(hit_test(&e, Point::new(190.0, 50.0), 1.0)); // inside near edge
                                                             // Corner of the box is outside the ellipse and far from the outline.
        assert!(!hit_test(&e, Point::new(5.0, 5.0), 1.0));
    }

    #[test]
    fn ellipse_outline_only_when_unfilled_far_from_outline_misses() {
        // A transparent-but-fillable ellipse still uses inside test (Excalidraw
        // treats closed shapes as hittable in their interior regardless of fill
        // for selection). Point just inside near center hits.
        let e = el(ElementKind::Ellipse, 0.0, 0.0, 100.0, 100.0);
        assert!(hit_test(&e, Point::new(50.0, 50.0), 0.5));
    }

    #[test]
    fn diamond_inside_and_outside() {
        let e = el(ElementKind::Diamond, 0.0, 0.0, 100.0, 100.0);
        assert!(hit_test(&e, Point::new(50.0, 50.0), 1.0)); // center
                                                            // Box corner (0,0): |−50|/50 + |−50|/50 = 2 > 1 → outside, and far from
                                                            // the diamond edges.
        assert!(!hit_test(&e, Point::new(0.0, 0.0), 1.0));
        // Point on the upper-left edge midpoint region, inside.
        assert!(hit_test(&e, Point::new(30.0, 30.0), 1.0));
    }

    #[test]
    fn line_hit_near_outline() {
        // Open line from (0,0) to (100,0) relative; element origin (0,0).
        let data = LinearData::line(vec![Point::new(0.0, 0.0), Point::new(100.0, 0.0)]);
        let e = el(ElementKind::Line(data), 0.0, 0.0, 100.0, 0.0);
        assert!(hit_test(&e, Point::new(50.0, 3.0), 4.0));
        assert!(!hit_test(&e, Point::new(50.0, 3.0), 2.0));
        // Not filled: a point "inside" a region bounded by the chain does not
        // count — only proximity to the stroke.
        assert!(!hit_test(&e, Point::new(50.0, 50.0), 4.0));
    }

    #[test]
    fn line_hit_respects_element_origin() {
        // Same line shifted to origin (100, 100).
        let data = LinearData::line(vec![Point::new(0.0, 0.0), Point::new(100.0, 0.0)]);
        let e = el(ElementKind::Line(data), 100.0, 100.0, 100.0, 0.0);
        assert!(hit_test(&e, Point::new(150.0, 101.0), 2.0));
        assert!(!hit_test(&e, Point::new(50.0, 1.0), 2.0));
    }

    #[test]
    fn closed_polygon_inside_hits() {
        // Triangle polygon.
        let mut data = LinearData::line(vec![
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(50.0, 100.0),
        ]);
        data.polygon = true;
        let e = el(ElementKind::Line(data), 0.0, 0.0, 100.0, 100.0);
        assert!(hit_test(&e, Point::new(50.0, 30.0), 0.5)); // inside
        assert!(!hit_test(&e, Point::new(5.0, 90.0), 0.5)); // outside, near base corner
    }

    #[test]
    fn freedraw_hit_near_stroke() {
        let data = FreedrawData::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(20.0, 0.0),
        ]);
        let e = el(ElementKind::Freedraw(data), 0.0, 0.0, 20.0, 10.0);
        assert!(hit_test(&e, Point::new(5.0, 5.0), 1.5)); // near the up-stroke
        assert!(!hit_test(&e, Point::new(10.0, 0.0), 1.5)); // below the V, far from both strokes
    }

    #[test]
    fn rotated_line_hit() {
        // Horizontal line, then rotate 90° about its center so it becomes
        // vertical. A point near the rotated stroke must hit.
        let data = LinearData::line(vec![Point::new(0.0, 0.0), Point::new(100.0, 0.0)]);
        let mut e = el(ElementKind::Line(data), 0.0, 0.0, 100.0, 0.0);
        e.angle = PI / 2.0; // center is (50, 0)
                            // After rotating the horizontal segment 90° about (50,0) it runs
                            // vertically through x=50. A point at (51, 30) is ~1 unit away.
        assert!(hit_test(&e, Point::new(51.0, 30.0), 2.0));
        assert!(!hit_test(&e, Point::new(80.0, 30.0), 2.0));
    }
}
