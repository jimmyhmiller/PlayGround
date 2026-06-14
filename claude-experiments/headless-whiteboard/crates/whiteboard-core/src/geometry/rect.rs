use super::Point;
use serde::{Deserialize, Serialize};

/// An axis-aligned rectangle in scene coordinates.
///
/// Stored as a min corner plus width/height. Width/height are normalized to be
/// non-negative by [`Rect::new`]; construct degenerate/empty rects with
/// [`Rect::EMPTY`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Default for Rect {
    /// The empty rect — a sensible identity for accumulation via `union`.
    fn default() -> Self {
        Rect::EMPTY
    }
}

impl Rect {
    /// A rect that contains nothing; `union` with it is the identity.
    pub const EMPTY: Rect = Rect {
        x: f64::INFINITY,
        y: f64::INFINITY,
        width: f64::NEG_INFINITY,
        height: f64::NEG_INFINITY,
    };

    /// Construct from a corner and size. Negative sizes are normalized so the
    /// rect always has its min corner at `(x, y)` after normalization.
    #[inline]
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        let (x, width) = if width < 0.0 {
            (x + width, -width)
        } else {
            (x, width)
        };
        let (y, height) = if height < 0.0 {
            (y + height, -height)
        } else {
            (y, height)
        };
        Rect {
            x,
            y,
            width,
            height,
        }
    }

    /// Construct from two opposite corners.
    #[inline]
    pub fn from_corners(a: Point, b: Point) -> Self {
        Rect::new(a.x, a.y, b.x - a.x, b.y - a.y)
    }

    /// Construct from min/max bounds.
    #[inline]
    pub fn from_min_max(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Rect {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }

    #[inline]
    pub fn min_x(&self) -> f64 {
        self.x
    }
    #[inline]
    pub fn min_y(&self) -> f64 {
        self.y
    }
    #[inline]
    pub fn max_x(&self) -> f64 {
        self.x + self.width
    }
    #[inline]
    pub fn max_y(&self) -> f64 {
        self.y + self.height
    }

    #[inline]
    pub fn center(&self) -> Point {
        Point::new(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    #[inline]
    pub fn top_left(&self) -> Point {
        Point::new(self.min_x(), self.min_y())
    }
    #[inline]
    pub fn top_right(&self) -> Point {
        Point::new(self.max_x(), self.min_y())
    }
    #[inline]
    pub fn bottom_left(&self) -> Point {
        Point::new(self.min_x(), self.max_y())
    }
    #[inline]
    pub fn bottom_right(&self) -> Point {
        Point::new(self.max_x(), self.max_y())
    }

    /// The four corners, clockwise from top-left.
    #[inline]
    pub fn corners(&self) -> [Point; 4] {
        [
            self.top_left(),
            self.top_right(),
            self.bottom_right(),
            self.bottom_left(),
        ]
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.width < 0.0 || self.height < 0.0
    }

    /// Whether the point lies inside (inclusive of the border).
    #[inline]
    pub fn contains(&self, p: Point) -> bool {
        p.x >= self.min_x() && p.x <= self.max_x() && p.y >= self.min_y() && p.y <= self.max_y()
    }

    /// Whether this rect fully contains `other`.
    #[inline]
    pub fn contains_rect(&self, other: &Rect) -> bool {
        other.min_x() >= self.min_x()
            && other.max_x() <= self.max_x()
            && other.min_y() >= self.min_y()
            && other.max_y() <= self.max_y()
    }

    /// Whether the two rects overlap (touching edges count as intersecting).
    #[inline]
    pub fn intersects(&self, other: &Rect) -> bool {
        self.min_x() <= other.max_x()
            && self.max_x() >= other.min_x()
            && self.min_y() <= other.max_y()
            && self.max_y() >= other.min_y()
    }

    /// Smallest rect containing both. `union` with [`Rect::EMPTY`] is identity.
    #[inline]
    pub fn union(&self, other: &Rect) -> Rect {
        Rect::from_min_max(
            self.min_x().min(other.min_x()),
            self.min_y().min(other.min_y()),
            self.max_x().max(other.max_x()),
            self.max_y().max(other.max_y()),
        )
    }

    /// Smallest rect containing all `points`. Returns [`Rect::EMPTY`] if empty.
    pub fn bounding(points: impl IntoIterator<Item = Point>) -> Rect {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        let mut any = false;
        for p in points {
            any = true;
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
        if !any {
            return Rect::EMPTY;
        }
        Rect::from_min_max(min_x, min_y, max_x, max_y)
    }

    /// Grow (or shrink, with negative `amount`) the rect on all sides.
    #[inline]
    pub fn inflate(&self, amount: f64) -> Rect {
        Rect::from_min_max(
            self.min_x() - amount,
            self.min_y() - amount,
            self.max_x() + amount,
            self.max_y() + amount,
        )
    }

    #[inline]
    pub fn translate(&self, dx: f64, dy: f64) -> Rect {
        Rect {
            x: self.x + dx,
            y: self.y + dy,
            width: self.width,
            height: self.height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_size_normalizes() {
        let r = Rect::new(10.0, 10.0, -5.0, -8.0);
        assert_eq!(r.x, 5.0);
        assert_eq!(r.y, 2.0);
        assert_eq!(r.width, 5.0);
        assert_eq!(r.height, 8.0);
    }

    #[test]
    fn union_with_empty_is_identity() {
        let r = Rect::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(Rect::EMPTY.union(&r), r);
        assert_eq!(r.union(&Rect::EMPTY), r);
    }

    #[test]
    fn contains_and_intersects() {
        let a = Rect::new(0.0, 0.0, 10.0, 10.0);
        let b = Rect::new(5.0, 5.0, 2.0, 2.0);
        assert!(a.contains_rect(&b));
        assert!(a.intersects(&b));
        assert!(a.contains(Point::new(5.0, 5.0)));
        assert!(!a.contains(Point::new(11.0, 5.0)));
    }

    #[test]
    fn bounding_of_points() {
        let r = Rect::bounding([
            Point::new(1.0, 5.0),
            Point::new(-2.0, 3.0),
            Point::new(4.0, -1.0),
        ]);
        assert_eq!(r.min_x(), -2.0);
        assert_eq!(r.min_y(), -1.0);
        assert_eq!(r.max_x(), 4.0);
        assert_eq!(r.max_y(), 5.0);
    }
}
