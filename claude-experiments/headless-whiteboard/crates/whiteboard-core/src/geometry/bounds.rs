//! Tight element bounds + outline segmentation.
//!
//! Reimplemented from Excalidraw's `packages/element/src/bounds.ts`
//! (`getElementBounds`, `getElementAbsoluteCoords`, `getCurvePathOps`/curve
//! extent handling) and the linear/freedraw point-extent logic. We do not vendor
//! any JavaScript; the algorithms below are reimplementations in Rust.
//!
//! Coordinate conventions (matching the foundation `Element`):
//! - `x`/`y`/`width`/`height` describe the element's *unrotated* axis-aligned
//!   box in scene coordinates.
//! - `angle` rotates the element clockwise (screen-space y-down) about the box
//!   center, via [`crate::geometry::point_rotate_rads`].
//! - Linear/freedraw `points` are *element-relative*: scene point = element
//!   `(x, y)` + the relative point.

use super::{point_rotate_rads, Point, Rect};
use crate::element::{Element, ElementKind};

/// Number of segments used to discretize an ellipse perimeter for both tight
/// rotated bounds and outline segmentation. A few dozen keeps the
/// approximation tight without being wasteful.
const ELLIPSE_SEGMENTS: usize = 48;

/// Tight bounds of an element in scene space, accounting for rotation, curve
/// extents (ellipse), the diamond's cardinal vertices, and the actual point
/// extents of linear/freedraw elements.
///
/// Port of Excalidraw `getElementBounds`. For closed generic shapes this rotates
/// the relevant cardinal/corner points about [`Element::center`] and takes
/// min/max — exactly matching upstream's approach of bounding the rotated
/// extreme points rather than the rotated axis-aligned box.
pub fn element_bounds(element: &Element) -> Rect {
    let center = element.center();
    let angle = element.angle;

    match &element.kind {
        ElementKind::Ellipse => ellipse_bounds(element, center, angle),
        ElementKind::Diamond => diamond_bounds(element, center, angle),
        ElementKind::Line(data) | ElementKind::Arrow(data) => {
            linear_bounds(element, &data.points, center, angle)
        }
        ElementKind::Freedraw(data) => linear_bounds(element, &data.points, center, angle),
        // Rectangle, Text, Image, Frame, Selection: rotate the four box corners.
        _ => rect_corner_bounds(element, center, angle),
    }
}

/// Bound the four corners of the unrotated box after rotating each about
/// `center`. With `angle == 0` this is exactly the raw box.
fn rect_corner_bounds(element: &Element, center: Point, angle: f64) -> Rect {
    let box_ = element.raw_box();
    Rect::bounding(
        box_.corners()
            .into_iter()
            .map(|p| point_rotate_rads(p, center, angle)),
    )
}

/// Tight bounds of a (possibly rotated) ellipse.
///
/// Excalidraw computes ellipse bounds from the curve's actual extents rather
/// than its bounding box corners. We sample the parametric perimeter densely,
/// rotate each sample about `center`, and take min/max. For the axis-aligned
/// case this recovers the exact box; for the rotated case it matches the true
/// rotated-ellipse extent to within the sampling resolution.
fn ellipse_bounds(element: &Element, center: Point, angle: f64) -> Rect {
    Rect::bounding(
        ellipse_perimeter_points(element, ELLIPSE_SEGMENTS)
            .into_iter()
            .map(|p| point_rotate_rads(p, center, angle)),
    )
}

/// Tight bounds of a (possibly rotated) diamond. The diamond's vertices are the
/// four edge midpoints of the box; we rotate those about `center`.
fn diamond_bounds(element: &Element, center: Point, angle: f64) -> Rect {
    Rect::bounding(
        diamond_vertices(element)
            .into_iter()
            .map(|p| point_rotate_rads(p, center, angle)),
    )
}

/// Tight bounds of a linear/freedraw element: scene-space points (element origin
/// added) rotated about `center`.
///
/// Falls back to the raw box corners when there are no points so the result is
/// never empty for a sized element.
fn linear_bounds(element: &Element, rel_points: &[Point], center: Point, angle: f64) -> Rect {
    if rel_points.is_empty() {
        return rect_corner_bounds(element, center, angle);
    }
    Rect::bounding(
        rel_points
            .iter()
            .map(|p| Point::new(element.x + p.x, element.y + p.y))
            .map(|p| point_rotate_rads(p, center, angle)),
    )
}

/// The four diamond vertices (edge midpoints of the unrotated box) in scene
/// space, clockwise from top.
fn diamond_vertices(element: &Element) -> [Point; 4] {
    let cx = element.x + element.width / 2.0;
    let cy = element.y + element.height / 2.0;
    [
        Point::new(cx, element.y),                  // top
        Point::new(element.x + element.width, cy),  // right
        Point::new(cx, element.y + element.height), // bottom
        Point::new(element.x, cy),                  // left
    ]
}

/// Sample `n` points around the (unrotated) ellipse perimeter in scene space.
fn ellipse_perimeter_points(element: &Element, n: usize) -> Vec<Point> {
    use std::f64::consts::TAU;
    let cx = element.x + element.width / 2.0;
    let cy = element.y + element.height / 2.0;
    let rx = element.width / 2.0;
    let ry = element.height / 2.0;
    (0..n)
        .map(|i| {
            let t = TAU * (i as f64) / (n as f64);
            Point::new(cx + rx * t.cos(), cy + ry * t.sin())
        })
        .collect()
}

/// The element's outline, approximated as line segments in **scene** space.
///
/// - rectangle/text/image/frame/selection: the four box sides (rotated).
/// - diamond: the four sides between its cardinal vertices (rotated).
/// - ellipse: a `~ELLIPSE_SEGMENTS`-gon approximating the perimeter (rotated).
/// - line/arrow/freedraw: consecutive points (element origin added, rotated),
///   plus the closing edge when the linear element is a polygon.
///
/// Used by hit-testing (distance-to-outline) and by collision queries in other
/// modules.
pub fn element_line_segments(element: &Element) -> Vec<(Point, Point)> {
    let center = element.center();
    let angle = element.angle;
    let rot = |p: Point| point_rotate_rads(p, center, angle);

    match &element.kind {
        ElementKind::Ellipse => {
            let pts: Vec<Point> = ellipse_perimeter_points(element, ELLIPSE_SEGMENTS)
                .into_iter()
                .map(rot)
                .collect();
            closed_loop_segments(&pts)
        }
        ElementKind::Diamond => {
            let pts: Vec<Point> = diamond_vertices(element).into_iter().map(rot).collect();
            closed_loop_segments(&pts)
        }
        ElementKind::Line(data) | ElementKind::Arrow(data) => {
            let pts: Vec<Point> = data
                .points
                .iter()
                .map(|p| rot(Point::new(element.x + p.x, element.y + p.y)))
                .collect();
            if data.polygon {
                closed_loop_segments(&pts)
            } else {
                open_chain_segments(&pts)
            }
        }
        ElementKind::Freedraw(data) => {
            let pts: Vec<Point> = data
                .points
                .iter()
                .map(|p| rot(Point::new(element.x + p.x, element.y + p.y)))
                .collect();
            open_chain_segments(&pts)
        }
        // Rectangle, Text, Image, Frame, Selection: box outline.
        _ => {
            let pts: Vec<Point> = element.raw_box().corners().into_iter().map(rot).collect();
            closed_loop_segments(&pts)
        }
    }
}

/// Segments connecting consecutive points, not closing the loop.
fn open_chain_segments(pts: &[Point]) -> Vec<(Point, Point)> {
    pts.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Segments connecting consecutive points and closing back to the first.
fn closed_loop_segments(pts: &[Point]) -> Vec<(Point, Point)> {
    let mut segs = open_chain_segments(pts);
    if pts.len() >= 2 {
        segs.push((pts[pts.len() - 1], pts[0]));
    }
    segs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{ElementId, FreedrawData, LinearData};
    use std::f64::consts::PI;

    fn el(kind: ElementKind, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from("t"), 1, x, y, w, h, kind)
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-6, "expected {b}, got {a}");
    }

    #[test]
    fn rect_unrotated_bounds_is_raw_box() {
        let e = el(ElementKind::Rectangle, 10.0, 20.0, 100.0, 40.0);
        let b = element_bounds(&e);
        approx(b.min_x(), 10.0);
        approx(b.min_y(), 20.0);
        approx(b.max_x(), 110.0);
        approx(b.max_y(), 60.0);
    }

    #[test]
    fn rect_rotated_90_swaps_extents_about_center() {
        // Box (10,20) 100x40, center (60,40). Rotating 90° clockwise about the
        // center turns a 100-wide, 40-tall box into a 40-wide, 100-tall box.
        let mut e = el(ElementKind::Rectangle, 10.0, 20.0, 100.0, 40.0);
        e.angle = PI / 2.0;
        let b = element_bounds(&e);
        approx(b.min_x(), 40.0);
        approx(b.max_x(), 80.0);
        approx(b.min_y(), -10.0);
        approx(b.max_y(), 90.0);
    }

    #[test]
    fn ellipse_unrotated_bounds_equals_box() {
        let e = el(ElementKind::Ellipse, 0.0, 0.0, 200.0, 100.0);
        let b = element_bounds(&e);
        // Perimeter sampling includes the axis extremes exactly (t=0, π/2, …).
        approx(b.min_x(), 0.0);
        approx(b.min_y(), 0.0);
        approx(b.max_x(), 200.0);
        approx(b.max_y(), 100.0);
    }

    #[test]
    fn ellipse_rotated_45_extent_matches_analytic() {
        // For an axis-aligned ellipse rx, ry rotated by θ, the rotated bounding
        // half-width is sqrt((rx cosθ)^2 + (ry sinθ)^2). Center at origin here.
        let mut e = el(ElementKind::Ellipse, -100.0, -50.0, 200.0, 100.0);
        e.angle = PI / 4.0;
        let (rx, ry) = (100.0_f64, 50.0_f64);
        let theta = PI / 4.0;
        let half_w = ((rx * theta.cos()).powi(2) + (ry * theta.sin()).powi(2)).sqrt();
        let half_h = ((rx * theta.sin()).powi(2) + (ry * theta.cos()).powi(2)).sqrt();
        let b = element_bounds(&e);
        // Sampling at 48 segments lands within a small tolerance of the analytic
        // extent.
        assert!((b.max_x() - half_w).abs() < 0.2, "max_x={}", b.max_x());
        assert!((b.max_y() - half_h).abs() < 0.2, "max_y={}", b.max_y());
        assert!((b.min_x() + half_w).abs() < 0.2);
        assert!((b.min_y() + half_h).abs() < 0.2);
    }

    #[test]
    fn diamond_unrotated_bounds_equals_box() {
        let e = el(ElementKind::Diamond, 5.0, 7.0, 40.0, 60.0);
        let b = element_bounds(&e);
        approx(b.min_x(), 5.0);
        approx(b.min_y(), 7.0);
        approx(b.max_x(), 45.0);
        approx(b.max_y(), 67.0);
    }

    #[test]
    fn line_bounds_use_point_extents_in_scene_space() {
        // Points are element-relative; element origin (100, 200) is added.
        let data = LinearData::line(vec![
            Point::new(0.0, 0.0),
            Point::new(50.0, -30.0),
            Point::new(20.0, 80.0),
        ]);
        let e = el(ElementKind::Line(data), 100.0, 200.0, 50.0, 110.0);
        let b = element_bounds(&e);
        approx(b.min_x(), 100.0);
        approx(b.max_x(), 150.0);
        approx(b.min_y(), 170.0);
        approx(b.max_y(), 280.0);
    }

    #[test]
    fn freedraw_bounds_use_point_extents() {
        let data = FreedrawData::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 5.0),
            Point::new(-4.0, 12.0),
        ]);
        let e = el(ElementKind::Freedraw(data), 0.0, 0.0, 14.0, 12.0);
        let b = element_bounds(&e);
        approx(b.min_x(), -4.0);
        approx(b.max_x(), 10.0);
        approx(b.min_y(), 0.0);
        approx(b.max_y(), 12.0);
    }

    #[test]
    fn rect_line_segments_are_four_sides() {
        let e = el(ElementKind::Rectangle, 0.0, 0.0, 10.0, 10.0);
        let segs = element_line_segments(&e);
        assert_eq!(segs.len(), 4);
        // First side runs along the top edge from top-left to top-right.
        approx(segs[0].0.x, 0.0);
        approx(segs[0].0.y, 0.0);
        approx(segs[0].1.x, 10.0);
        approx(segs[0].1.y, 0.0);
        // The loop closes back to the start.
        approx(segs[3].1.x, segs[0].0.x);
        approx(segs[3].1.y, segs[0].0.y);
    }

    #[test]
    fn diamond_line_segments_are_four_sides() {
        let e = el(ElementKind::Diamond, 0.0, 0.0, 20.0, 20.0);
        let segs = element_line_segments(&e);
        assert_eq!(segs.len(), 4);
    }

    #[test]
    fn open_line_segments_do_not_close() {
        let data = LinearData::line(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
        ]);
        let e = el(ElementKind::Line(data), 0.0, 0.0, 10.0, 10.0);
        let segs = element_line_segments(&e);
        assert_eq!(segs.len(), 2);
    }

    #[test]
    fn closed_polygon_line_segments_close() {
        let mut data = LinearData::line(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
        ]);
        data.polygon = true;
        let e = el(ElementKind::Line(data), 0.0, 0.0, 10.0, 10.0);
        let segs = element_line_segments(&e);
        assert_eq!(segs.len(), 3);
    }

    #[test]
    fn ellipse_segment_count_matches_resolution() {
        let e = el(ElementKind::Ellipse, 0.0, 0.0, 10.0, 10.0);
        let segs = element_line_segments(&e);
        assert_eq!(segs.len(), ELLIPSE_SEGMENTS);
    }
}
