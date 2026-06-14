use super::{Point, Rect, Vec2};
use serde::{Deserialize, Serialize};

/// A 2D affine transform stored as a 2x3 matrix (column-major-ish, row form):
///
/// ```text
/// | a c e |   | x |
/// | b d f | * | y |
/// | 0 0 1 |   | 1 |
/// ```
///
/// This is the same layout as the SVG/Canvas `matrix(a, b, c, d, e, f)` and
/// makes it trivial for backends to forward to their own transform stacks.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl Default for Transform {
    fn default() -> Self {
        Transform::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: 0.0,
        f: 0.0,
    };

    #[inline]
    pub const fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Transform { a, b, c, d, e, f }
    }

    #[inline]
    pub const fn translate(dx: f64, dy: f64) -> Self {
        Transform::new(1.0, 0.0, 0.0, 1.0, dx, dy)
    }

    #[inline]
    pub const fn scale(sx: f64, sy: f64) -> Self {
        Transform::new(sx, 0.0, 0.0, sy, 0.0, 0.0)
    }

    /// Rotation by `angle` radians around the origin (clockwise in screen space).
    #[inline]
    pub fn rotate(angle: f64) -> Self {
        let (sin, cos) = angle.sin_cos();
        Transform::new(cos, sin, -sin, cos, 0.0, 0.0)
    }

    /// Rotation by `angle` radians around `center`.
    #[inline]
    pub fn rotate_around(angle: f64, center: Point) -> Self {
        // Apply, in order: move center to origin, rotate, move back.
        Transform::translate(-center.x, -center.y)
            .then(&Transform::rotate(angle))
            .then(&Transform::translate(center.x, center.y))
    }

    /// Compose: `self.then(other)` applies `self` first, then `other`.
    #[inline]
    pub fn then(&self, other: &Transform) -> Transform {
        // other * self
        Transform {
            a: other.a * self.a + other.c * self.b,
            b: other.b * self.a + other.d * self.b,
            c: other.a * self.c + other.c * self.d,
            d: other.b * self.c + other.d * self.d,
            e: other.a * self.e + other.c * self.f + other.e,
            f: other.b * self.e + other.d * self.f + other.f,
        }
    }

    #[inline]
    pub fn apply(&self, p: Point) -> Point {
        Point::new(
            self.a * p.x + self.c * p.y + self.e,
            self.b * p.x + self.d * p.y + self.f,
        )
    }

    /// Apply to a vector (ignores translation).
    #[inline]
    pub fn apply_vec(&self, v: Vec2) -> Vec2 {
        Vec2::new(self.a * v.x + self.c * v.y, self.b * v.x + self.d * v.y)
    }

    pub fn determinant(&self) -> f64 {
        self.a * self.d - self.b * self.c
    }

    /// Inverse transform, or `None` if singular.
    pub fn inverse(&self) -> Option<Transform> {
        let det = self.determinant();
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Transform {
            a: self.d * inv_det,
            b: -self.b * inv_det,
            c: -self.c * inv_det,
            d: self.a * inv_det,
            e: (self.c * self.f - self.d * self.e) * inv_det,
            f: (self.b * self.e - self.a * self.f) * inv_det,
        })
    }

    /// Transform a rect by mapping its four corners and taking the bounds.
    pub fn apply_rect_bounds(&self, r: &Rect) -> Rect {
        Rect::bounding(r.corners().into_iter().map(|p| self.apply(p)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn identity_apply() {
        let p = Point::new(3.0, 4.0);
        assert_eq!(Transform::IDENTITY.apply(p), p);
    }

    #[test]
    fn translate_then_inverse_round_trips() {
        let t = Transform::translate(5.0, -2.0);
        let p = Point::new(1.0, 1.0);
        let moved = t.apply(p);
        let back = t.inverse().unwrap().apply(moved);
        assert!((back.x - p.x).abs() < 1e-9 && (back.y - p.y).abs() < 1e-9);
    }

    #[test]
    fn rotate_around_center() {
        let c = Point::new(10.0, 10.0);
        let t = Transform::rotate_around(PI / 2.0, c);
        let p = Point::new(11.0, 10.0);
        let r = t.apply(p);
        assert!((r.x - 10.0).abs() < 1e-9, "x={}", r.x);
        assert!((r.y - 11.0).abs() < 1e-9, "y={}", r.y);
    }

    #[test]
    fn compose_translate_scale() {
        // scale first, then translate
        let t = Transform::scale(2.0, 2.0).then(&Transform::translate(1.0, 1.0));
        let r = t.apply(Point::new(3.0, 4.0));
        assert_eq!(r, Point::new(7.0, 9.0));
    }
}
