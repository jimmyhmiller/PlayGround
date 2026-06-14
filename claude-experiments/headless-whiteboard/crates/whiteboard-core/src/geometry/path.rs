use super::{Point, Rect};
use serde::{Deserialize, Serialize};

/// A single path segment. Paths are sequences of segments that share endpoints
/// implicitly: each segment starts where the previous one ended, beginning from
/// the most recent `MoveTo`.
///
/// This is intentionally a small, backend-neutral vocabulary (move / line /
/// cubic / close). Quadratic curves are represented as cubics by callers; arcs
/// are flattened to cubics by the shape generators. Backends only ever need to
/// understand these four cases.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PathSegment {
    MoveTo(Point),
    LineTo(Point),
    /// Cubic Bézier: two control points and the endpoint.
    CubicTo {
        c1: Point,
        c2: Point,
        to: Point,
    },
    Close,
}

/// A vector path: an ordered list of segments. May contain multiple subpaths
/// (each begun by a `MoveTo`). Filling uses the non-zero winding rule.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Path {
    pub segments: Vec<PathSegment>,
}

impl Path {
    pub fn new() -> Self {
        Path {
            segments: Vec::new(),
        }
    }

    pub fn builder() -> PathBuilder {
        PathBuilder {
            path: Path::new(),
            cursor: None,
            start: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn push(&mut self, seg: PathSegment) {
        self.segments.push(seg);
    }

    pub fn extend(&mut self, other: &Path) {
        self.segments.extend_from_slice(&other.segments);
    }

    /// Build an open polyline through the given points.
    pub fn polyline(points: &[Point]) -> Path {
        let mut b = Path::builder();
        if let Some((first, rest)) = points.split_first() {
            b.move_to(*first);
            for p in rest {
                b.line_to(*p);
            }
        }
        b.build()
    }

    /// Build a closed polygon through the given points.
    pub fn polygon(points: &[Point]) -> Path {
        let mut b = Path::builder();
        if let Some((first, rest)) = points.split_first() {
            b.move_to(*first);
            for p in rest {
                b.line_to(*p);
            }
            b.close();
        }
        b.build()
    }

    /// Conservative bounds: the bounding box of every point referenced by the
    /// path, including Bézier control points. This over-approximates curve
    /// extents but never under-approximates — exact curve bounds live in the
    /// bounds module (Phase 1).
    pub fn control_bounds(&self) -> Rect {
        let mut pts: Vec<Point> = Vec::with_capacity(self.segments.len() * 2);
        for seg in &self.segments {
            match seg {
                PathSegment::MoveTo(p) | PathSegment::LineTo(p) => pts.push(*p),
                PathSegment::CubicTo { c1, c2, to } => {
                    pts.push(*c1);
                    pts.push(*c2);
                    pts.push(*to);
                }
                PathSegment::Close => {}
            }
        }
        Rect::bounding(pts)
    }
}

/// Fluent builder that tracks the current point and subpath start.
pub struct PathBuilder {
    path: Path,
    cursor: Option<Point>,
    start: Option<Point>,
}

impl PathBuilder {
    pub fn move_to(&mut self, p: Point) -> &mut Self {
        self.cursor = Some(p);
        self.start = Some(p);
        self.path.push(PathSegment::MoveTo(p));
        self
    }

    pub fn line_to(&mut self, p: Point) -> &mut Self {
        if self.cursor.is_none() {
            return self.move_to(p);
        }
        self.cursor = Some(p);
        self.path.push(PathSegment::LineTo(p));
        self
    }

    pub fn cubic_to(&mut self, c1: Point, c2: Point, to: Point) -> &mut Self {
        if self.cursor.is_none() {
            self.move_to(c1);
        }
        self.cursor = Some(to);
        self.path.push(PathSegment::CubicTo { c1, c2, to });
        self
    }

    /// Quadratic curve, emitted as an equivalent cubic.
    pub fn quad_to(&mut self, ctrl: Point, to: Point) -> &mut Self {
        let from = self.cursor.unwrap_or(ctrl);
        let c1 = Point::new(
            from.x + 2.0 / 3.0 * (ctrl.x - from.x),
            from.y + 2.0 / 3.0 * (ctrl.y - from.y),
        );
        let c2 = Point::new(
            to.x + 2.0 / 3.0 * (ctrl.x - to.x),
            to.y + 2.0 / 3.0 * (ctrl.y - to.y),
        );
        self.cubic_to(c1, c2, to)
    }

    pub fn close(&mut self) -> &mut Self {
        self.path.push(PathSegment::Close);
        self.cursor = self.start;
        self
    }

    pub fn current(&self) -> Option<Point> {
        self.cursor
    }

    pub fn build(&mut self) -> Path {
        std::mem::take(&mut self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polyline_segments() {
        let p = Path::polyline(&[
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
        ]);
        assert_eq!(p.segments.len(), 3);
        assert!(matches!(p.segments[0], PathSegment::MoveTo(_)));
        assert!(matches!(p.segments[2], PathSegment::LineTo(_)));
    }

    #[test]
    fn polygon_closes() {
        let p = Path::polygon(&[
            Point::new(0.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(1.0, 2.0),
        ]);
        assert_eq!(*p.segments.last().unwrap(), PathSegment::Close);
    }

    #[test]
    fn quad_lowered_to_cubic() {
        let mut b = Path::builder();
        b.move_to(Point::new(0.0, 0.0))
            .quad_to(Point::new(1.0, 1.0), Point::new(2.0, 0.0));
        let p = b.build();
        assert!(matches!(p.segments[1], PathSegment::CubicTo { .. }));
    }

    #[test]
    fn control_bounds_includes_control_points() {
        let mut b = Path::builder();
        b.move_to(Point::new(0.0, 0.0)).cubic_to(
            Point::new(0.0, 10.0),
            Point::new(10.0, 10.0),
            Point::new(10.0, 0.0),
        );
        let r = b.build().control_bounds();
        assert_eq!(r.max_y(), 10.0);
    }
}
