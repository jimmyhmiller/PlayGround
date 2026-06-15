//! Elbow (orthogonal) arrow routing.
//!
//! Reimplemented — in a deliberately *simplified but correct* form — from
//! Excalidraw's elbow-arrow router (`packages/element/src/elbowArrow.ts`). The
//! upstream router is a full grid-based A*/heuristic path-finder that accounts
//! for bound-element avoidance, dynamic gaps, and heading-aware dog-legs. This
//! port keeps the *observable* property that matters for headless geometry: the
//! produced polyline is **axis-aligned** (every segment is purely horizontal or
//! vertical), runs from the original first point to the original last point, and
//! threads through any intermediate user points orthogonally.
//!
//! ## Routing rule
//!
//! Given the element-local `points`, we build an orthogonal polyline pairwise:
//!
//! * If two consecutive points already share an `x` or a `y` (the segment is
//!   already axis-aligned), it is emitted unchanged — a single straight
//!   horizontal/vertical run.
//! * Otherwise we insert **one** intermediate corner, turning the diagonal into
//!   an L of a horizontal and a vertical leg. The corner is chosen on the
//!   **dominant axis**: when the horizontal span is at least as large as the
//!   vertical span we go *horizontal-first* (corner at `(b.x, a.y)`); otherwise
//!   *vertical-first* (corner at `(a.x, b.y)`). Picking the dominant axis keeps
//!   the longer leg leading, which matches the natural "mostly travels in the
//!   primary direction, then steps over" feel of Excalidraw elbow arrows and
//!   keeps a 2-point arrow a clean single-bend L (or a straight line when the
//!   endpoints are already aligned).
//!
//! After assembling the corners we drop exact duplicates and collapse any three
//! collinear points, so a straight elbow arrow stays a 2-point polyline and an
//! L stays a 3-point polyline.
//!
//! All geometry is **element-local**, like the rest of [`super::clean`].

use crate::element::{Element, LinearData};
use crate::geometry::{Path, Point};

use super::arrowhead::arrowhead_paths;
use super::ShapeGeometry;

/// Tolerance under which two coordinates are treated as equal (already aligned).
const EPS: f64 = 1e-9;

/// Route `points` into an orthogonal (axis-aligned) polyline from the first to
/// the last point, threading through every intermediate point.
///
/// See the module docs for the routing rule. The returned vector always starts
/// at `points[0]` and ends at `points[last]` (endpoints preserved), and every
/// consecutive pair is either horizontal or vertical. Inputs of fewer than two
/// points are returned unchanged (nothing to route).
pub fn elbow_route(points: &[Point]) -> Vec<Point> {
    if points.len() < 2 {
        return points.to_vec();
    }

    let mut out: Vec<Point> = Vec::with_capacity(points.len() * 2);
    out.push(points[0]);

    for pair in points.windows(2) {
        let a = pair[0];
        let b = pair[1];
        let aligned_x = (a.x - b.x).abs() <= EPS;
        let aligned_y = (a.y - b.y).abs() <= EPS;

        if aligned_x || aligned_y {
            // Already a single horizontal or vertical run (or a zero-length
            // step): just continue to `b`.
            push_point(&mut out, b);
            continue;
        }

        // Diagonal: insert one corner, leading with the dominant axis.
        let horizontal_first = (b.x - a.x).abs() >= (b.y - a.y).abs();
        let corner = if horizontal_first {
            Point::new(b.x, a.y)
        } else {
            Point::new(a.x, b.y)
        };
        push_point(&mut out, corner);
        push_point(&mut out, b);
    }

    collapse_collinear(out)
}

/// Push `p` unless it duplicates the current tail (avoids zero-length segments).
fn push_point(out: &mut Vec<Point>, p: Point) {
    if let Some(last) = out.last() {
        if (last.x - p.x).abs() <= EPS && (last.y - p.y).abs() <= EPS {
            return;
        }
    }
    out.push(p);
}

/// Remove the middle of any three points that are collinear on an axis, so a
/// straight elbow stays minimal (e.g. a horizontal-then-horizontal run becomes a
/// single segment). Only axis-aligned collinearity is collapsed, which is all
/// `elbow_route` can produce.
fn collapse_collinear(pts: Vec<Point>) -> Vec<Point> {
    if pts.len() <= 2 {
        return pts;
    }
    let mut out: Vec<Point> = Vec::with_capacity(pts.len());
    out.push(pts[0]);
    for i in 1..pts.len() - 1 {
        let prev = *out.last().unwrap();
        let cur = pts[i];
        let next = pts[i + 1];
        let collinear_h = (prev.y - cur.y).abs() <= EPS && (cur.y - next.y).abs() <= EPS;
        let collinear_v = (prev.x - cur.x).abs() <= EPS && (cur.x - next.x).abs() <= EPS;
        if collinear_h || collinear_v {
            // `cur` lies on a straight run between `prev` and `next`; skip it.
            continue;
        }
        out.push(cur);
    }
    out.push(*pts.last().unwrap());
    out
}

/// Build the elbow [`ShapeGeometry`] for a linear element: an orthogonal outline
/// polyline (closed when `data.polygon`) plus start/end arrowheads oriented along
/// the first/final orthogonal segment.
pub fn elbow_geometry(element: &Element, data: &LinearData) -> ShapeGeometry {
    let mut g = ShapeGeometry::default();
    if data.points.len() < 2 {
        // No drawable segment — honest empty geometry (mirrors `linear_geometry`).
        return g;
    }

    let route = elbow_route(&data.points);
    if route.len() < 2 {
        return g;
    }

    let body = if data.polygon {
        Path::polygon(&route)
    } else {
        Path::polyline(&route)
    };
    g.outline.push(body.clone());

    if data.polygon && element.kind.is_fillable() && !element.background_color.is_transparent() {
        g.fill.push(body);
    }

    // Arrowheads only apply to open elbow elements (a closed polygon has no free
    // ends). Orient each head along its incident orthogonal segment so it points
    // straight along the final/first leg of the route.
    if !data.polygon {
        let size = arrowhead_size(element.stroke_width);
        let n = route.len();
        if let Some(head) = data.end_arrowhead {
            let tip = route[n - 1];
            let prev = route[n - 2];
            for p in arrowhead_paths(head, tip, prev, size) {
                g.outline.push(p);
            }
        }
        if let Some(head) = data.start_arrowhead {
            let tip = route[0];
            let prev = route[1];
            for p in arrowhead_paths(head, tip, prev, size) {
                g.outline.push(p);
            }
        }
    }

    g
}

/// Arrowhead size derived from stroke width (kept in step with
/// [`super::clean`]'s straight-line arrowheads).
fn arrowhead_size(stroke_width: f64) -> f64 {
    (10.0 + stroke_width * 2.0).max(8.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{ElementId, ElementKind};
    use crate::geometry::PathSegment;

    fn arrow_el(data: LinearData) -> Element {
        Element::new(
            ElementId::from("e"),
            1,
            0.0,
            0.0,
            100.0,
            100.0,
            ElementKind::Arrow(data),
        )
    }

    /// Collect the vertex points (move/line endpoints) of a path's main body.
    fn vertices(path: &Path) -> Vec<Point> {
        path.segments
            .iter()
            .filter_map(|s| match s {
                PathSegment::MoveTo(p) | PathSegment::LineTo(p) => Some(*p),
                _ => None,
            })
            .collect()
    }

    fn all_segments_axis_aligned(pts: &[Point]) -> bool {
        pts.windows(2)
            .all(|w| (w[0].x - w[1].x).abs() <= EPS || (w[0].y - w[1].y).abs() <= EPS)
    }

    #[test]
    fn two_point_diagonal_is_single_l() {
        let pts = vec![Point::new(0.0, 0.0), Point::new(30.0, 10.0)];
        let route = elbow_route(&pts);
        // One corner ⇒ 3 points.
        assert_eq!(route.len(), 3);
        assert!(all_segments_axis_aligned(&route));
        assert_eq!(route[0], pts[0]);
        assert_eq!(route[2], pts[1]);
        // Horizontal span (30) > vertical (10) ⇒ horizontal-first corner.
        assert_eq!(route[1], Point::new(30.0, 0.0));
    }

    #[test]
    fn vertical_dominant_diagonal_is_vertical_first() {
        let pts = vec![Point::new(0.0, 0.0), Point::new(10.0, 40.0)];
        let route = elbow_route(&pts);
        assert_eq!(route.len(), 3);
        assert!(all_segments_axis_aligned(&route));
        // Vertical span dominates ⇒ corner at (a.x, b.y).
        assert_eq!(route[1], Point::new(0.0, 40.0));
    }

    #[test]
    fn already_horizontal_stays_straight() {
        let pts = vec![Point::new(0.0, 5.0), Point::new(40.0, 5.0)];
        let route = elbow_route(&pts);
        assert_eq!(route, pts);
        assert!(all_segments_axis_aligned(&route));
    }

    #[test]
    fn already_vertical_stays_straight() {
        let pts = vec![Point::new(7.0, 0.0), Point::new(7.0, 25.0)];
        let route = elbow_route(&pts);
        assert_eq!(route, pts);
    }

    #[test]
    fn intermediate_points_are_routed_orthogonally() {
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(20.0, 15.0),
            Point::new(40.0, 5.0),
        ];
        let route = elbow_route(&pts);
        assert!(all_segments_axis_aligned(&route));
        // Endpoints preserved; intermediate point present.
        assert_eq!(*route.first().unwrap(), pts[0]);
        assert_eq!(*route.last().unwrap(), pts[2]);
        assert!(route.iter().any(|p| *p == pts[1]));
    }

    #[test]
    fn collinear_runs_collapse() {
        // Two horizontal-first L's that happen to share the same y at the seam
        // should not leave a redundant midpoint.
        let pts = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(20.0, 0.0),
        ];
        let route = elbow_route(&pts);
        // All on y=0 ⇒ collapses to the two endpoints.
        assert_eq!(route, vec![pts[0], pts[2]]);
    }

    #[test]
    fn degenerate_inputs_unchanged() {
        assert!(elbow_route(&[]).is_empty());
        let one = vec![Point::new(3.0, 4.0)];
        assert_eq!(elbow_route(&one), one);
    }

    #[test]
    fn geometry_outline_is_axis_aligned_open_polyline() {
        let mut data = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(30.0, 12.0)]);
        data.elbowed = true;
        let el = arrow_el(data.clone());
        let g = elbow_geometry(&el, &data);
        assert!(!g.outline.is_empty());
        let body = &g.outline[0];
        // Open polyline ⇒ no Close segment.
        assert!(!body
            .segments
            .iter()
            .any(|s| matches!(s, PathSegment::Close)));
        let verts = vertices(body);
        assert!(
            all_segments_axis_aligned(&verts),
            "outline not orthogonal: {verts:?}"
        );
        // Endpoints preserved.
        assert_eq!(*verts.first().unwrap(), Point::new(0.0, 0.0));
        assert_eq!(*verts.last().unwrap(), Point::new(30.0, 12.0));
    }

    #[test]
    fn geometry_has_end_arrowhead() {
        let mut data = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(30.0, 12.0)]);
        data.elbowed = true;
        let el = arrow_el(data.clone());
        let g = elbow_geometry(&el, &data);
        // Body + at least one arrowhead path.
        assert!(
            g.outline.len() >= 2,
            "expected arrowhead path, got {}",
            g.outline.len()
        );
    }

    #[test]
    fn arrowhead_oriented_along_final_orthogonal_leg() {
        // Horizontal-dominant ⇒ final leg is vertical (the step-over). The end
        // arrowhead must orient along that final vertical segment.
        let mut data = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(30.0, 12.0)]);
        data.elbowed = true;
        let el = arrow_el(data.clone());
        let route = elbow_route(&data.points);
        // Final leg: (30,0) -> (30,12): vertical, pointing +y.
        let n = route.len();
        assert!((route[n - 1].x - route[n - 2].x).abs() <= EPS);
        let g = elbow_geometry(&el, &data);
        // The triangle arrowhead's apex is the tip; its base points sit behind
        // the tip along -y (above it) for a +y final leg.
        let head = &g.outline[1];
        for s in &head.segments {
            if let PathSegment::LineTo(p) = s {
                assert!(
                    p.y < route[n - 1].y + EPS,
                    "head base not behind tip: {p:?}"
                );
            }
        }
    }

    #[test]
    fn closed_polygon_elbow_closes_and_fills() {
        use crate::element::ElementKind;
        use crate::render::Color;
        let mut data = LinearData::line(vec![
            Point::new(0.0, 0.0),
            Point::new(20.0, 8.0),
            Point::new(0.0, 16.0),
        ]);
        data.elbowed = true;
        data.polygon = true;
        let mut el = Element::new(
            ElementId::from("e"),
            1,
            0.0,
            0.0,
            20.0,
            16.0,
            ElementKind::Line(data.clone()),
        );
        el.background_color = Color::rgb(1, 2, 3);
        let g = elbow_geometry(&el, &data);
        assert!(g.outline[0]
            .segments
            .iter()
            .any(|s| matches!(s, PathSegment::Close)));
        assert_eq!(g.fill.len(), 1);
        // Polygon outline still fully axis-aligned.
        let verts = vertices(&g.outline[0]);
        assert!(all_segments_axis_aligned(&verts));
    }
}
