//! Arrowhead outline geometry for linear elements.
//!
//! Reimplemented from Excalidraw's `getArrowheadPoints`
//! (`packages/element/src/shape.ts`). Each arrowhead is produced in the same
//! element-local space as the line body, oriented along the incident segment so
//! it points outward at the endpoint.
//!
//! The head shapes here cover the geometric primitives Excalidraw builds every
//! arrowhead from: a two-barb open chevron (`Arrow`), a solid/outline triangle,
//! a perpendicular bar, a dot (small circle), and a small diamond. The
//! `*Outline` and filled variants share the same outline path; whether it is
//! filled is decided by the renderer from the element's stroke color.

use crate::element::Arrowhead;
use crate::geometry::{Path, Point, Vec2};

/// Produce the outline path(s) for `head` placed at `tip`, oriented so it points
/// away from `prev` (the previous vertex on the line). `size` is the nominal
/// head length in scene units.
///
/// Returns an empty vec when the segment is degenerate (`tip == prev`), which is
/// honest: there is no direction to orient the head along, so nothing is drawn.
pub fn arrowhead_paths(head: Arrowhead, tip: Point, prev: Point, size: f64) -> Vec<Path> {
    let dir = Vec2::new(tip.x - prev.x, tip.y - prev.y);
    if dir.length_sq() == 0.0 || size <= 0.0 {
        return Vec::new();
    }
    let dir = dir.normalized();
    // Perpendicular (rotate dir +90°).
    let perp = Vec2::new(-dir.y, dir.x);

    match head {
        Arrowhead::Arrow => vec![chevron(tip, dir, perp, size)],
        Arrowhead::Triangle | Arrowhead::TriangleOutline => {
            vec![triangle(tip, dir, perp, size)]
        }
        Arrowhead::Bar => vec![bar(tip, perp, size)],
        Arrowhead::Dot | Arrowhead::Circle | Arrowhead::CircleOutline => {
            vec![dot(tip, dir, size)]
        }
        Arrowhead::Diamond | Arrowhead::DiamondOutline => vec![diamond(tip, dir, perp, size)],
        // Crowfoot is a "many" cardinality fork (three splayed prongs).
        Arrowhead::Crowfoot => vec![crowfoot(tip, dir, perp, size)],
    }
}

/// Point at `tip - dir*back + perp*side`.
#[inline]
fn at(tip: Point, dir: Vec2, perp: Vec2, back: f64, side: f64) -> Point {
    Point::new(
        tip.x - dir.x * back + perp.x * side,
        tip.y - dir.y * back + perp.y * side,
    )
}

/// Open two-barb chevron: `\` and `/` meeting at the tip.
fn chevron(tip: Point, dir: Vec2, perp: Vec2, size: f64) -> Path {
    let back = size;
    let half = size * 0.5;
    let left = at(tip, dir, perp, back, half);
    let right = at(tip, dir, perp, back, -half);
    Path::polyline(&[left, tip, right])
}

/// Closed isosceles triangle with the apex at the tip.
fn triangle(tip: Point, dir: Vec2, perp: Vec2, size: f64) -> Path {
    let back = size;
    let half = size * 0.5;
    let base_left = at(tip, dir, perp, back, half);
    let base_right = at(tip, dir, perp, back, -half);
    Path::polygon(&[tip, base_left, base_right])
}

/// A perpendicular bar straddling the tip.
fn bar(tip: Point, perp: Vec2, size: f64) -> Path {
    let half = size * 0.5;
    let a = Point::new(tip.x + perp.x * half, tip.y + perp.y * half);
    let b = Point::new(tip.x - perp.x * half, tip.y - perp.y * half);
    Path::polyline(&[a, b])
}

/// A small circle centered just behind the tip, as a 4-arc Bézier loop.
fn dot(tip: Point, dir: Vec2, size: f64) -> Path {
    let r = size * 0.35;
    // Center the dot one radius behind the tip so it sits on the line end.
    let cx = tip.x - dir.x * r;
    let cy = tip.y - dir.y * r;
    const KAPPA: f64 = 0.552_284_749_830_793_4;
    let k = r * KAPPA;
    let mut b = Path::builder();
    b.move_to(Point::new(cx + r, cy));
    b.cubic_to(
        Point::new(cx + r, cy + k),
        Point::new(cx + k, cy + r),
        Point::new(cx, cy + r),
    );
    b.cubic_to(
        Point::new(cx - k, cy + r),
        Point::new(cx - r, cy + k),
        Point::new(cx - r, cy),
    );
    b.cubic_to(
        Point::new(cx - r, cy - k),
        Point::new(cx - k, cy - r),
        Point::new(cx, cy - r),
    );
    b.cubic_to(
        Point::new(cx + k, cy - r),
        Point::new(cx + r, cy - k),
        Point::new(cx + r, cy),
    );
    b.close();
    b.build()
}

/// A small diamond centered just behind the tip.
fn diamond(tip: Point, dir: Vec2, perp: Vec2, size: f64) -> Path {
    let half = size * 0.5;
    let front = tip;
    let back = at(tip, dir, perp, size, 0.0);
    let mid = at(tip, dir, perp, half, 0.0);
    let left = Point::new(mid.x + perp.x * half, mid.y + perp.y * half);
    let right = Point::new(mid.x - perp.x * half, mid.y - perp.y * half);
    Path::polygon(&[front, left, back, right])
}

/// Crowfoot ("many" cardinality): three prongs splaying from a point set back
/// from the tip out to the tip's left, center, and right.
fn crowfoot(tip: Point, dir: Vec2, perp: Vec2, size: f64) -> Path {
    let half = size * 0.5;
    let root = at(tip, dir, perp, size, 0.0);
    let left = Point::new(tip.x + perp.x * half, tip.y + perp.y * half);
    let right = Point::new(tip.x - perp.x * half, tip.y - perp.y * half);
    // root→left, root→tip, root→right as a single open path.
    let mut b = Path::builder();
    b.move_to(left);
    b.line_to(root);
    b.line_to(tip);
    b.line_to(root);
    b.line_to(right);
    b.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::PathSegment;

    fn tip_right() -> (Point, Point) {
        // Line going +x; tip at (10,0), previous at (0,0).
        (Point::new(10.0, 0.0), Point::new(0.0, 0.0))
    }

    #[test]
    fn degenerate_segment_yields_nothing() {
        let p = Point::new(1.0, 1.0);
        assert!(arrowhead_paths(Arrowhead::Triangle, p, p, 10.0).is_empty());
    }

    #[test]
    fn triangle_is_closed_three_point_polygon() {
        let (tip, prev) = tip_right();
        let paths = arrowhead_paths(Arrowhead::Triangle, tip, prev, 10.0);
        assert_eq!(paths.len(), 1);
        let segs = &paths[0].segments;
        // move + 2 line + close
        assert_eq!(segs.len(), 4);
        assert!(matches!(segs.last().unwrap(), PathSegment::Close));
        // Apex is the tip.
        assert!(matches!(segs[0], PathSegment::MoveTo(p) if p == tip));
    }

    #[test]
    fn triangle_base_is_behind_tip() {
        let (tip, prev) = tip_right();
        let paths = arrowhead_paths(Arrowhead::Triangle, tip, prev, 10.0);
        // For a +x line, the base points have x < tip.x (behind the tip).
        for s in &paths[0].segments {
            if let PathSegment::LineTo(p) = s {
                assert!(p.x < tip.x, "base point not behind tip: {p:?}");
            }
        }
    }

    #[test]
    fn chevron_is_open_three_points() {
        let (tip, prev) = tip_right();
        let paths = arrowhead_paths(Arrowhead::Arrow, tip, prev, 10.0);
        let segs = &paths[0].segments;
        assert_eq!(segs.len(), 3); // move + 2 line
        assert!(!segs.iter().any(|s| matches!(s, PathSegment::Close)));
    }

    #[test]
    fn bar_is_two_points_perpendicular() {
        let (tip, prev) = tip_right();
        let paths = arrowhead_paths(Arrowhead::Bar, tip, prev, 10.0);
        let pts: Vec<Point> = paths[0]
            .segments
            .iter()
            .filter_map(|s| match s {
                PathSegment::MoveTo(p) | PathSegment::LineTo(p) => Some(*p),
                _ => None,
            })
            .collect();
        assert_eq!(pts.len(), 2);
        // Perpendicular to +x line ⇒ both endpoints share tip.x.
        assert!((pts[0].x - tip.x).abs() < 1e-9);
        assert!((pts[1].x - tip.x).abs() < 1e-9);
        // and straddle y=0 symmetrically.
        assert!((pts[0].y + pts[1].y).abs() < 1e-9);
    }

    #[test]
    fn dot_is_closed_four_arc_loop() {
        let (tip, prev) = tip_right();
        let paths = arrowhead_paths(Arrowhead::Dot, tip, prev, 10.0);
        let cubics = paths[0]
            .segments
            .iter()
            .filter(|s| matches!(s, PathSegment::CubicTo { .. }))
            .count();
        assert_eq!(cubics, 4);
        assert!(matches!(
            paths[0].segments.last().unwrap(),
            PathSegment::Close
        ));
    }

    #[test]
    fn diamond_is_closed_quad() {
        let (tip, prev) = tip_right();
        let paths = arrowhead_paths(Arrowhead::Diamond, tip, prev, 10.0);
        let segs = &paths[0].segments;
        assert_eq!(segs.len(), 5); // move + 3 line + close
    }

    #[test]
    fn crowfoot_has_two_prong_lines() {
        let (tip, prev) = tip_right();
        let paths = arrowhead_paths(Arrowhead::Crowfoot, tip, prev, 10.0);
        let lines = paths[0]
            .segments
            .iter()
            .filter(|s| matches!(s, PathSegment::LineTo(_)))
            .count();
        assert_eq!(lines, 4);
    }

    #[test]
    fn orientation_follows_direction() {
        // Line going +y (downward); triangle apex must be the tip and base above.
        let tip = Point::new(0.0, 10.0);
        let prev = Point::new(0.0, 0.0);
        let paths = arrowhead_paths(Arrowhead::Triangle, tip, prev, 10.0);
        for s in &paths[0].segments {
            if let PathSegment::LineTo(p) = s {
                assert!(p.y < tip.y, "base not behind tip for +y line: {p:?}");
            }
        }
    }
}
