//! Arrowhead outline geometry for linear elements.
//!
//! Reimplemented from Excalidraw's `getArrowheadPoints`
//! (`packages/element/src/shape.ts`). Each arrowhead is produced in the same
//! element-local space as the line body, oriented along the incident segment so
//! it points outward at the endpoint.
//!
//! The head shapes here cover the geometric primitives Excalidraw builds every
//! arrowhead from: a two-barb open chevron (`Arrow`), a solid/outline triangle,
//! a perpendicular bar, a dot/disc (small circle), and a small diamond.
//!
//! ## Filled vs outline heads
//!
//! Excalidraw distinguishes *solid* (flood-filled) heads from *outline* (stroked
//! only) heads. A filled `Triangle`/`Dot`/`Circle`/`Diamond` is painted solid in
//! the element's **stroke** color; the `*Outline` variants and the open
//! chevron/bar/crowfoot are merely stroked. The two cases are returned in
//! separate buckets by [`arrowhead_geometry`] so the caller (and ultimately the
//! tessellator) can flood-fill the former and stroke the latter. The legacy
//! [`arrowhead_paths`] helper flattens both buckets back to a single
//! stroke-everything `Vec<Path>` for callers that do not yet distinguish them.

use crate::element::Arrowhead;
use crate::geometry::{Path, Point, Vec2};

/// Arrowhead geometry split by paint mode.
///
/// `filled` paths are **closed** regions to be flood-filled with the element's
/// *stroke* color (a solid head). `stroked` paths are stroked with the stroke
/// color only (open chevrons/bars, or the explicit `*Outline` variants).
///
/// A given head populates exactly one bucket: e.g. `Triangle` → `filled`,
/// `TriangleOutline` → `stroked`. Both buckets are never both non-empty for a
/// single head today, but the struct allows it for future composite heads.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ArrowheadGeometry {
    /// Closed regions to flood-fill with the element's stroke color.
    pub filled: Vec<Path>,
    /// Paths to stroke (outline-only / open heads).
    pub stroked: Vec<Path>,
}

impl ArrowheadGeometry {
    fn filled(path: Path) -> Self {
        ArrowheadGeometry {
            filled: vec![path],
            stroked: Vec::new(),
        }
    }

    fn stroked(path: Path) -> Self {
        ArrowheadGeometry {
            filled: Vec::new(),
            stroked: vec![path],
        }
    }

    /// `true` when there is nothing to draw.
    pub fn is_empty(&self) -> bool {
        self.filled.is_empty() && self.stroked.is_empty()
    }
}

/// Produce the paint-mode-aware geometry for `head` placed at `tip`, oriented so
/// it points away from `prev` (the previous vertex on the line). `size` is the
/// nominal head length in scene units.
///
/// Returns empty geometry when the segment is degenerate (`tip == prev`) or the
/// size is non-positive, which is honest: there is no direction to orient the
/// head along, so nothing is drawn.
pub fn arrowhead_geometry(
    head: Arrowhead,
    tip: Point,
    prev: Point,
    size: f64,
) -> ArrowheadGeometry {
    let dir = Vec2::new(tip.x - prev.x, tip.y - prev.y);
    if dir.length_sq() == 0.0 || size <= 0.0 {
        return ArrowheadGeometry::default();
    }
    let dir = dir.normalized();
    // Perpendicular (rotate dir +90°).
    let perp = Vec2::new(-dir.y, dir.x);

    match head {
        // Open two-barb chevron: stroked only (it is not a closed region).
        Arrowhead::Arrow => ArrowheadGeometry::stroked(chevron(tip, dir, perp, size)),
        // Solid triangle, painted in the stroke color.
        Arrowhead::Triangle => ArrowheadGeometry::filled(triangle(tip, dir, perp, size)),
        // Same closed outline, but stroked only.
        Arrowhead::TriangleOutline => ArrowheadGeometry::stroked(triangle(tip, dir, perp, size)),
        // Perpendicular bar: an open segment, stroked.
        Arrowhead::Bar => ArrowheadGeometry::stroked(bar(tip, perp, size)),
        // Excalidraw "dot"/"circle" are solid discs in the stroke color.
        Arrowhead::Dot | Arrowhead::Circle => ArrowheadGeometry::filled(dot(tip, dir, size)),
        // Hollow circle: same loop, stroked only.
        Arrowhead::CircleOutline => ArrowheadGeometry::stroked(dot(tip, dir, size)),
        // Solid diamond vs. its hollow outline.
        Arrowhead::Diamond => ArrowheadGeometry::filled(diamond(tip, dir, perp, size)),
        Arrowhead::DiamondOutline => ArrowheadGeometry::stroked(diamond(tip, dir, perp, size)),
        // Crowfoot is a "many" cardinality fork (three splayed prongs), stroked.
        Arrowhead::Crowfoot => ArrowheadGeometry::stroked(crowfoot(tip, dir, perp, size)),
    }
}

/// Back-compat: produce the outline path(s) for `head`, flattening filled and
/// stroked heads into a single stroke-everything list.
///
/// This loses the filled/outline distinction (a filled triangle comes back as a
/// closed path that the caller will merely stroke). Prefer
/// [`arrowhead_geometry`] when the fill vs. stroke distinction matters. Retained
/// so callers that only need outline paths (bounds, hit-testing previews) keep
/// working unchanged.
pub fn arrowhead_paths(head: Arrowhead, tip: Point, prev: Point, size: f64) -> Vec<Path> {
    let g = arrowhead_geometry(head, tip, prev, size);
    let mut paths = g.filled;
    paths.extend(g.stroked);
    paths
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

    // --- Filled vs outline geometry --------------------------------------

    fn is_closed(path: &Path) -> bool {
        matches!(path.segments.last(), Some(PathSegment::Close))
    }

    #[test]
    fn degenerate_geometry_is_empty() {
        let p = Point::new(2.0, 3.0);
        assert!(arrowhead_geometry(Arrowhead::Triangle, p, p, 10.0).is_empty());
        assert!(arrowhead_geometry(Arrowhead::Diamond, p, p, 10.0).is_empty());
        // Non-positive size is also nothing.
        let (tip, prev) = tip_right();
        assert!(arrowhead_geometry(Arrowhead::Dot, tip, prev, 0.0).is_empty());
    }

    #[test]
    fn triangle_is_filled_and_closed() {
        let (tip, prev) = tip_right();
        let g = arrowhead_geometry(Arrowhead::Triangle, tip, prev, 10.0);
        assert_eq!(g.filled.len(), 1);
        assert!(g.stroked.is_empty());
        assert!(is_closed(&g.filled[0]), "filled triangle must be closed");
    }

    #[test]
    fn triangle_outline_is_stroked_closed_path() {
        let (tip, prev) = tip_right();
        let g = arrowhead_geometry(Arrowhead::TriangleOutline, tip, prev, 10.0);
        assert!(g.filled.is_empty());
        assert_eq!(g.stroked.len(), 1);
        // It is the same closed shape, just stroked rather than filled.
        assert!(is_closed(&g.stroked[0]));
    }

    #[test]
    fn triangle_vs_outline_share_geometry_differ_in_bucket() {
        let (tip, prev) = tip_right();
        let filled = arrowhead_geometry(Arrowhead::Triangle, tip, prev, 10.0);
        let outline = arrowhead_geometry(Arrowhead::TriangleOutline, tip, prev, 10.0);
        // Same underlying path, opposite paint buckets.
        assert_eq!(filled.filled, outline.stroked);
        assert!(filled.stroked.is_empty() && outline.filled.is_empty());
        assert_ne!(filled, outline);
    }

    #[test]
    fn dot_and_circle_are_filled_outline_is_stroked() {
        let (tip, prev) = tip_right();
        let dot = arrowhead_geometry(Arrowhead::Dot, tip, prev, 10.0);
        let circle = arrowhead_geometry(Arrowhead::Circle, tip, prev, 10.0);
        let hollow = arrowhead_geometry(Arrowhead::CircleOutline, tip, prev, 10.0);
        assert_eq!(dot.filled.len(), 1);
        assert!(dot.stroked.is_empty());
        // Dot and Circle are both solid discs ⇒ identical geometry.
        assert_eq!(dot, circle);
        // The outline variant carries the same loop in the stroked bucket.
        assert_eq!(hollow.stroked, dot.filled);
        assert!(hollow.filled.is_empty());
        assert_ne!(dot, hollow);
    }

    #[test]
    fn diamond_filled_vs_outline_differ() {
        let (tip, prev) = tip_right();
        let solid = arrowhead_geometry(Arrowhead::Diamond, tip, prev, 10.0);
        let outline = arrowhead_geometry(Arrowhead::DiamondOutline, tip, prev, 10.0);
        assert_eq!(solid.filled.len(), 1);
        assert!(is_closed(&solid.filled[0]));
        assert_eq!(outline.stroked.len(), 1);
        assert_eq!(solid.filled, outline.stroked);
        assert_ne!(solid, outline);
    }

    #[test]
    fn arrow_and_bar_and_crowfoot_are_stroked_open() {
        let (tip, prev) = tip_right();
        for head in [Arrowhead::Arrow, Arrowhead::Bar, Arrowhead::Crowfoot] {
            let g = arrowhead_geometry(head, tip, prev, 10.0);
            assert!(g.filled.is_empty(), "{head:?} must not be filled");
            assert_eq!(g.stroked.len(), 1, "{head:?} should stroke one path");
            assert!(!is_closed(&g.stroked[0]), "{head:?} should be an open path");
        }
    }

    #[test]
    fn filled_heads_orient_along_segment() {
        // +y line: every filled-triangle base vertex sits behind the tip.
        let tip = Point::new(0.0, 10.0);
        let prev = Point::new(0.0, 0.0);
        let g = arrowhead_geometry(Arrowhead::Triangle, tip, prev, 10.0);
        for s in &g.filled[0].segments {
            if let PathSegment::LineTo(p) = s {
                assert!(p.y < tip.y, "filled base not behind tip: {p:?}");
            }
        }
    }

    #[test]
    fn back_compat_paths_flatten_both_buckets() {
        let (tip, prev) = tip_right();
        // Filled head still yields its (closed) path via the legacy API.
        let filled = arrowhead_paths(Arrowhead::Triangle, tip, prev, 10.0);
        assert_eq!(filled.len(), 1);
        assert!(is_closed(&filled[0]));
        // Stroked head likewise.
        let stroked = arrowhead_paths(Arrowhead::Arrow, tip, prev, 10.0);
        assert_eq!(stroked.len(), 1);
        assert!(!is_closed(&stroked[0]));
    }
}
