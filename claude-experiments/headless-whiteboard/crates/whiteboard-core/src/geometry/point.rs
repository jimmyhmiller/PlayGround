use serde::{Deserialize, Serialize};

/// A 2D point in scene coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const ORIGIN: Point = Point { x: 0.0, y: 0.0 };

    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    /// Euclidean distance to another point.
    #[inline]
    pub fn distance(self, other: Point) -> f64 {
        ((other.x - self.x).powi(2) + (other.y - self.y).powi(2)).sqrt()
    }

    /// Squared distance — cheaper when you only need to compare.
    #[inline]
    pub fn distance_sq(self, other: Point) -> f64 {
        (other.x - self.x).powi(2) + (other.y - self.y).powi(2)
    }

    /// Treat this point as a displacement from the origin.
    #[inline]
    pub fn to_vec(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Midpoint between two points.
    #[inline]
    pub fn midpoint(self, other: Point) -> Point {
        Point::new((self.x + other.x) / 2.0, (self.y + other.y) / 2.0)
    }

    #[inline]
    pub fn translate(self, v: Vec2) -> Point {
        Point::new(self.x + v.x, self.y + v.y)
    }
}

impl From<(f64, f64)> for Point {
    fn from((x, y): (f64, f64)) -> Self {
        Point::new(x, y)
    }
}

/// A 2D vector / displacement.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Vec2 { x, y }
    }

    #[inline]
    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    #[inline]
    pub fn length_sq(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    #[inline]
    pub fn dot(self, other: Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// 2D cross product (returns the z-component).
    #[inline]
    pub fn cross(self, other: Vec2) -> f64 {
        self.x * other.y - self.y * other.x
    }

    #[inline]
    pub fn scale(self, s: f64) -> Vec2 {
        Vec2::new(self.x * s, self.y * s)
    }

    /// Unit vector in the same direction; zero vector stays zero.
    #[inline]
    pub fn normalized(self) -> Vec2 {
        let len = self.length();
        if len == 0.0 {
            Vec2::ZERO
        } else {
            self.scale(1.0 / len)
        }
    }

    #[inline]
    pub fn to_point(self) -> Point {
        Point::new(self.x, self.y)
    }
}

impl std::ops::Add for Vec2 {
    type Output = Vec2;
    #[inline]
    fn add(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x + other.x, self.y + other.y)
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Vec2;
    #[inline]
    fn sub(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x - other.x, self.y - other.y)
    }
}

impl std::ops::Mul<f64> for Vec2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, s: f64) -> Vec2 {
        self.scale(s)
    }
}

/// The displacement that takes `a` to `b`.
// Used by the bounds/hit-test and interaction modules (Phase 1).
#[allow(dead_code)]
#[inline]
pub fn displacement(a: Point, b: Point) -> Vec2 {
    Vec2::new(b.x - a.x, b.y - a.y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_3_4_5() {
        assert_eq!(Point::ORIGIN.distance(Point::new(3.0, 4.0)), 5.0);
    }

    #[test]
    fn normalize_unit() {
        let v = Vec2::new(3.0, 4.0).normalized();
        assert!((v.length() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cross_and_dot() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert_eq!(a.dot(b), 0.0);
        assert_eq!(a.cross(b), 1.0);
    }
}
