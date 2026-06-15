//! Arrow-to-shape binding: the model/geometry, not the UI wiring.
//!
//! Reimplemented from Excalidraw's `packages/element/src/binding.ts`
//! (`bindingBorderTest` / `maxBindingGap`, `determineFocusDistance` /
//! `determineFocusPoint`, `calculateFocusAndGap`, `updateBoundPoint`). We do not
//! vendor any JavaScript; this is a Rust reimplementation that keeps the same
//! field meaning for [`PointBinding`] (`focus` in `-1..=1`, `gap >= 0`).
//!
//! # Edge model / approximation
//!
//! Upstream resolves the focus point against the *exact* shape outline (rounded
//! rectangles, diamonds, real ellipse arcs, linear hulls). To stay simple and
//! fully self-consistent here we use a deliberately reduced model, documented so
//! callers know the contract:
//!
//! - The bindable target is modelled as its axis-aligned [`element_bounds`] rect,
//!   *except* ellipses, which are modelled as the inscribed ellipse of that rect.
//! - The binding ray runs from the target center toward the endpoint. We encode
//!   that ray's **direction** in `focus` as the normalized angle
//!   `focus = atan2(dy, dx) / PI ∈ -1..=1`. `focus == 0` means the endpoint is
//!   directly to the right of (and aimed straight through) the center;
//!   `focus == ±1` means directly to the left; intermediate values sweep the
//!   circle. This is a single-scalar, fully reversible encoding of the binding
//!   direction — the property the model needs to keep an endpoint attached as the
//!   shape moves. (Excalidraw's `focus` is instead a perpendicular *offset*; we
//!   deliberately use the equivalent-purpose angular form so `bound_point` is a
//!   self-contained exact inverse without needing the arrow's opposite endpoint.)
//! - `gap` is the distance from the endpoint to that modelled edge, clamped to be
//!   non-negative.
//!
//! [`compute_binding`] and [`bound_point`] are **exact inverses** for any endpoint
//! at or outside the target edge, for every direction:
//! `bound_point(t, compute_binding(p, t))` reproduces `p`. This is what lets the
//! interaction layer move a shape and have its bound arrow endpoints follow while
//! preserving both `focus` (direction) and `gap` (clearance).
//!
//! Rotation note: like the upstream "simple" path and the rest of this crate's
//! binding-adjacent geometry, the focus/gap math operates on the *axis-aligned*
//! [`element_bounds`]. A rotated target therefore binds against its upright
//! bounding box; this matches what callers get from [`element_bounds`].

use crate::geometry::{element_bounds, Point, Rect, Vec2};
use crate::scene::Scene;

use super::{Element, ElementId, ElementKind, LinearData, PointBinding};

/// The modelled outline shape of a bindable target.
#[derive(Debug, Clone, Copy, PartialEq)]
enum EdgeModel {
    /// Axis-aligned rectangle (rectangles, diamonds, text, images, frames).
    Rect,
    /// Inscribed ellipse of the bounding rect.
    Ellipse,
}

impl EdgeModel {
    fn of(element: &Element) -> EdgeModel {
        match element.kind {
            ElementKind::Ellipse => EdgeModel::Ellipse,
            _ => EdgeModel::Rect,
        }
    }
}

/// Whether an element can be an arrow-binding target.
///
/// Mirrors upstream `isBindableElement`: any non-linear, non-deleted shape with a
/// fillable/closed silhouette. Linear elements (lines/arrows/freedraw), text and
/// selection marquees are excluded.
pub fn is_bindable_element(element: &Element) -> bool {
    if element.is_deleted {
        return false;
    }
    matches!(
        element.kind,
        ElementKind::Rectangle
            | ElementKind::Ellipse
            | ElementKind::Diamond
            | ElementKind::Image(_)
            | ElementKind::Frame(_)
    )
}

/// The topmost bindable element whose modelled edge/area is within `tolerance`
/// scene units of `point`.
///
/// Port of upstream `getHoveredElementForBinding` + `bindingBorderTest`: a point
/// inside the shape, or within the binding gap of its border, binds. We iterate
/// in reverse paint order so the visually-topmost candidate wins.
pub fn bindable_element_at(scene: &Scene, point: Point, tolerance: f64) -> Option<ElementId> {
    // `iter_live` yields in paint order (bottom first) but is not double-ended,
    // so we keep the last matching candidate to land on the topmost one.
    scene
        .iter_live()
        .filter(|e| is_bindable_element(e))
        .filter(|e| within_binding_distance(e, point, tolerance))
        .map(|e| e.id.clone())
        .last()
}

/// Whether `point` is inside the modelled target or within `tolerance` of its
/// edge. `tolerance` is the binding gap the interaction layer supplies.
fn within_binding_distance(element: &Element, point: Point, tolerance: f64) -> bool {
    let rect = element_bounds(element);
    match EdgeModel::of(element) {
        EdgeModel::Rect => signed_rect_distance(rect, point) <= tolerance,
        EdgeModel::Ellipse => signed_ellipse_distance(rect, point) <= tolerance,
    }
}

/// Compute the binding for an arrow endpoint aimed at `target`.
///
/// `focus` is the normalized angle of the center→endpoint binding ray
/// (`atan2(dy, dx) / PI`, in `-1..=1`); `gap` is the colinear distance from the
/// endpoint to the modelled target edge along that ray (clamped `>= 0`). See the
/// module docs for the exact edge model. This is the inverse of [`bound_point`].
pub fn compute_binding(arrow_endpoint: Point, target: &Element) -> PointBinding {
    let rect = element_bounds(target);
    let center = rect.center();
    let (focus, gap) = match EdgeModel::of(target) {
        EdgeModel::Rect => rect_focus_and_gap(rect, center, arrow_endpoint),
        EdgeModel::Ellipse => ellipse_focus_and_gap(rect, center, arrow_endpoint),
    };
    PointBinding {
        element_id: target.id.clone(),
        focus,
        gap: gap.max(0.0),
    }
}

/// The scene-space point an arrow endpoint should sit at for the given binding:
/// the modelled `target` edge along the `focus` direction, pushed outward by
/// `gap`. Inverse of [`compute_binding`].
pub fn bound_point(target: &Element, focus: f64, gap: f64) -> Point {
    let rect = element_bounds(target);
    let center = rect.center();
    let gap = gap.max(0.0);
    match EdgeModel::of(target) {
        EdgeModel::Rect => rect_bound_point(rect, center, focus, gap),
        EdgeModel::Ellipse => ellipse_bound_point(rect, center, focus, gap),
    }
}

/// New endpoints for a bound arrow after its target(s) moved.
///
/// Returns `(start, end)` in scene space. `None` for an endpoint means that end
/// is unbound and the caller should leave it untouched. This function does **not**
/// mutate the scene (the arrow and its targets are borrowed immutably); the caller
/// applies the returned points, preserving the arrow's intermediate points and
/// only moving the bound ends.
///
/// Port of the relevant half of upstream `updateBoundPoint` /
/// `bindOrUnbindLinearElement`: for each bound end we read the *current* target
/// bounds and recompute the endpoint from the stored `focus`/`gap`.
pub fn update_bound_arrow(scene: &Scene, arrow_id: &ElementId) -> Option<BoundEndpoints> {
    let arrow = scene.get(arrow_id)?;
    let data = match &arrow.kind {
        ElementKind::Arrow(d) | ElementKind::Line(d) => d,
        _ => return None,
    };

    let start = data
        .start_binding
        .as_ref()
        .and_then(|b| resolve_endpoint(scene, b));
    let end = data
        .end_binding
        .as_ref()
        .and_then(|b| resolve_endpoint(scene, b));

    Some(BoundEndpoints { start, end })
}

/// The recomputed scene-space endpoints for a bound arrow. Each is `None` when
/// that end is unbound (or its target is missing) and must be left as-is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundEndpoints {
    pub start: Option<Point>,
    pub end: Option<Point>,
}

/// Apply recomputed endpoints to a linear element's relative `points`, keeping
/// intermediate points intact and only moving the bound first/last vertices.
///
/// Endpoints are in **scene** space; the element's `points` are relative to its
/// `(x, y)` origin, so each endpoint is converted back to element-relative space.
/// Returns whether anything changed. (Convenience for the caller's apply step; it
/// takes `&mut LinearData` plus the element origin so it stays free of `Scene`
/// borrow juggling.)
pub fn apply_bound_endpoints(
    data: &mut LinearData,
    origin: Point,
    endpoints: BoundEndpoints,
) -> bool {
    let mut changed = false;
    if let (Some(p), Some(first)) = (endpoints.start, data.points.first().copied()) {
        let rel = Point::new(p.x - origin.x, p.y - origin.y);
        if rel != first {
            data.points[0] = rel;
            changed = true;
        }
    }
    if let (Some(p), Some(last)) = (endpoints.end, data.points.last().copied()) {
        let rel = Point::new(p.x - origin.x, p.y - origin.y);
        if rel != last {
            let idx = data.points.len() - 1;
            data.points[idx] = rel;
            changed = true;
        }
    }
    changed
}

fn resolve_endpoint(scene: &Scene, binding: &PointBinding) -> Option<Point> {
    let target = scene.get(&binding.element_id)?;
    if !is_bindable_element(target) {
        return None;
    }
    Some(bound_point(target, binding.focus, binding.gap))
}

// ----- Rect edge model -------------------------------------------------------

/// Signed distance from `point` to the rect border: negative inside, `0` on the
/// border, positive outside (true Euclidean distance when outside).
fn signed_rect_distance(rect: Rect, point: Point) -> f64 {
    let dx = (rect.min_x() - point.x).max(point.x - rect.max_x());
    let dy = (rect.min_y() - point.y).max(point.y - rect.max_y());
    if dx <= 0.0 && dy <= 0.0 {
        // Inside: distance to the nearest edge, reported as negative.
        dx.max(dy)
    } else {
        // Outside: standard AABB exterior distance.
        let ox = dx.max(0.0);
        let oy = dy.max(0.0);
        (ox * ox + oy * oy).sqrt()
    }
}

/// Where the ray from `center` along unit `dir` exits the rect. Returns
/// `(edge_point, outward_unit)`.
fn rect_edge_hit(rect: Rect, center: Point, dir: Vec2) -> (Point, Vec2) {
    let hw = rect.width / 2.0;
    let hh = rect.height / 2.0;
    let (dx, dy) = (dir.x, dir.y);
    if dx == 0.0 && dy == 0.0 {
        return (Point::new(center.x + hw, center.y), Vec2::new(1.0, 0.0));
    }
    // Scale the ray so it lands exactly on the rect border (slab method).
    let tx = if dx != 0.0 {
        hw / dx.abs()
    } else {
        f64::INFINITY
    };
    let ty = if dy != 0.0 {
        hh / dy.abs()
    } else {
        f64::INFINITY
    };
    let t = tx.min(ty);
    let edge = Point::new(center.x + dx * t, center.y + dy * t);
    let outward = Vec2::new(edge.x - center.x, edge.y - center.y).normalized();
    (edge, outward)
}

fn rect_focus_and_gap(rect: Rect, center: Point, endpoint: Point) -> (f64, f64) {
    let to_end = Vec2::new(endpoint.x - center.x, endpoint.y - center.y);
    if to_end.length_sq() == 0.0 {
        // Endpoint at the center: aim straight out the right edge, no gap to add.
        let (edge, _) = rect_edge_hit(rect, center, Vec2::new(1.0, 0.0));
        return (0.0, -endpoint.distance(edge));
    }
    let focus = to_end.y.atan2(to_end.x) / std::f64::consts::PI;
    // Gap measured **colinearly along the binding ray**: the signed distance from
    // the ray/edge crossing to the endpoint. This is what makes bound_point an
    // exact inverse (it pushes outward along the same ray). Negative when the
    // endpoint is inside the edge; the public wrapper clamps to >= 0.
    let (edge, _) = rect_edge_hit(rect, center, to_end.normalized());
    let gap = colinear_gap(center, edge, endpoint);
    (focus, gap)
}

/// Signed distance along the center→endpoint ray from the edge crossing to the
/// endpoint. Positive when the endpoint is beyond the edge (outside), negative
/// when short of it (inside). `center`, `edge`, `endpoint` are colinear by
/// construction (`edge` is the ray/shape crossing).
fn colinear_gap(center: Point, edge: Point, endpoint: Point) -> f64 {
    endpoint.distance(center) - edge.distance(center)
}

fn rect_bound_point(rect: Rect, center: Point, focus: f64, gap: f64) -> Point {
    let angle = focus.clamp(-1.0, 1.0) * std::f64::consts::PI;
    let dir = Vec2::new(angle.cos(), angle.sin());
    let (edge, outward) = rect_edge_hit(rect, center, dir);
    Point::new(edge.x + outward.x * gap, edge.y + outward.y * gap)
}

// ----- Ellipse edge model ----------------------------------------------------

/// Signed distance from `point` to the inscribed ellipse of `rect`: negative
/// inside, positive outside. Uses the gradient-normalized implicit value, which
/// is the standard cheap first-order distance estimate and is exact along the
/// axes (sufficient for the gap term given the documented approximation).
fn signed_ellipse_distance(rect: Rect, point: Point) -> f64 {
    let center = rect.center();
    let rx = rect.width / 2.0;
    let ry = rect.height / 2.0;
    if rx == 0.0 || ry == 0.0 {
        return point.distance(center);
    }
    let dx = point.x - center.x;
    let dy = point.y - center.y;
    let f = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) - 1.0;
    // Gradient magnitude of the implicit function, for a first-order distance.
    let gx = 2.0 * dx / (rx * rx);
    let gy = 2.0 * dy / (ry * ry);
    let grad = (gx * gx + gy * gy).sqrt();
    if grad == 0.0 {
        // At the center.
        -rx.min(ry)
    } else {
        f / grad
    }
}

/// The point where the ray from `center` along `dir` crosses the inscribed
/// ellipse of `rect`. Returns just the crossing point.
fn ellipse_edge_hit(rect: Rect, center: Point, dir: Vec2) -> Point {
    let rx = rect.width / 2.0;
    let ry = rect.height / 2.0;
    let (dx, dy) = (dir.x, dir.y);
    if (dx == 0.0 && dy == 0.0) || rx == 0.0 || ry == 0.0 {
        return Point::new(center.x + rx, center.y);
    }
    // Solve ((t*dx)/rx)^2 + ((t*dy)/ry)^2 = 1 for t > 0.
    let a = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry);
    let t = (1.0 / a).sqrt();
    Point::new(center.x + dx * t, center.y + dy * t)
}

fn ellipse_focus_and_gap(rect: Rect, center: Point, endpoint: Point) -> (f64, f64) {
    let to_end = Vec2::new(endpoint.x - center.x, endpoint.y - center.y);
    if to_end.length_sq() == 0.0 {
        let edge = ellipse_edge_hit(rect, center, Vec2::new(1.0, 0.0));
        return (0.0, -endpoint.distance(edge));
    }
    // Same angular focus encoding as the rect model.
    let focus = to_end.y.atan2(to_end.x) / std::f64::consts::PI;
    // Colinear gap along the binding ray, for an exact inverse.
    let edge = ellipse_edge_hit(rect, center, to_end.normalized());
    let gap = colinear_gap(center, edge, endpoint);
    (focus, gap)
}

fn ellipse_bound_point(rect: Rect, center: Point, focus: f64, gap: f64) -> Point {
    let angle = focus.clamp(-1.0, 1.0) * std::f64::consts::PI;
    let dir = Vec2::new(angle.cos(), angle.sin());
    let edge = ellipse_edge_hit(rect, center, dir);
    // Push the endpoint outward **along the same binding ray** so this exactly
    // inverts ellipse_focus_and_gap's colinear gap.
    Point::new(edge.x + dir.x * gap, edge.y + dir.y * gap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Arrowhead, LinearData};
    use crate::geometry::Point;

    fn rect_el(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Rectangle)
    }

    fn ellipse_el(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Ellipse)
    }

    fn arrow_el(id: &str, points: Vec<Point>) -> Element {
        // The element origin is the first point; width/height from extents are
        // not load-bearing for these tests.
        let data = LinearData::arrow(points.clone());
        Element::new(
            ElementId::from(id),
            1,
            0.0,
            0.0,
            0.0,
            0.0,
            ElementKind::Arrow(data),
        )
    }

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "expected {b}, got {a}");
    }

    fn approx_pt(p: Point, q: Point, eps: f64) {
        approx(p.x, q.x, eps);
        approx(p.y, q.y, eps);
    }

    // ---- is_bindable / bindable_element_at ----

    #[test]
    fn bindable_classification() {
        assert!(is_bindable_element(&rect_el("r", 0.0, 0.0, 10.0, 10.0)));
        assert!(is_bindable_element(&ellipse_el("e", 0.0, 0.0, 10.0, 10.0)));
        assert!(is_bindable_element(&Element::new(
            ElementId::from("d"),
            1,
            0.0,
            0.0,
            10.0,
            10.0,
            ElementKind::Diamond,
        )));
        // Arrows / lines are not bindable targets.
        assert!(!is_bindable_element(&arrow_el(
            "a",
            vec![Point::new(0.0, 0.0), Point::new(10.0, 0.0)]
        )));
        // Deleted shapes are excluded.
        let mut del = rect_el("x", 0.0, 0.0, 10.0, 10.0);
        del.is_deleted = true;
        assert!(!is_bindable_element(&del));
    }

    #[test]
    fn bindable_element_at_inside_and_near_edge() {
        let mut scene = Scene::new();
        scene.insert(rect_el("r", 0.0, 0.0, 100.0, 50.0));

        // Inside.
        assert_eq!(
            bindable_element_at(&scene, Point::new(50.0, 25.0), 5.0),
            Some(ElementId::from("r"))
        );
        // 3 units outside the right edge, tolerance 5 → binds; tolerance 2 → none.
        assert_eq!(
            bindable_element_at(&scene, Point::new(103.0, 25.0), 5.0),
            Some(ElementId::from("r"))
        );
        assert_eq!(
            bindable_element_at(&scene, Point::new(103.0, 25.0), 2.0),
            None
        );
        // Far away.
        assert_eq!(
            bindable_element_at(&scene, Point::new(500.0, 500.0), 5.0),
            None
        );
    }

    #[test]
    fn bindable_element_at_picks_topmost() {
        let mut scene = Scene::new();
        scene.insert(rect_el("bottom", 0.0, 0.0, 100.0, 100.0));
        scene.insert(rect_el("top", 0.0, 0.0, 100.0, 100.0));
        // Both overlap the point; the later-inserted (topmost) wins.
        assert_eq!(
            bindable_element_at(&scene, Point::new(50.0, 50.0), 5.0),
            Some(ElementId::from("top"))
        );
    }

    #[test]
    fn bindable_element_at_skips_arrows() {
        let mut scene = Scene::new();
        scene.insert(arrow_el(
            "a",
            vec![Point::new(0.0, 0.0), Point::new(100.0, 0.0)],
        ));
        assert_eq!(
            bindable_element_at(&scene, Point::new(50.0, 0.0), 5.0),
            None
        );
    }

    // ---- compute_binding / bound_point round-trips (rect) ----

    #[test]
    fn rect_focus_zero_on_axis_with_gap() {
        // Rect (0,0)-(100,50), center (50,25). Endpoint straight out the right
        // edge at (110,25): focus 0, gap 10 (edge at x=100).
        let r = rect_el("r", 0.0, 0.0, 100.0, 50.0);
        let b = compute_binding(Point::new(110.0, 25.0), &r);
        approx(b.focus, 0.0, 1e-9);
        approx(b.gap, 10.0, 1e-9);
        // Inverse reproduces the endpoint exactly.
        approx_pt(
            bound_point(&r, b.focus, b.gap),
            Point::new(110.0, 25.0),
            1e-9,
        );
    }

    #[test]
    fn rect_focus_encodes_ray_direction() {
        // focus = atan2(dy, dx) / PI. Cardinal directions are clean:
        let r = rect_el("r", 0.0, 0.0, 100.0, 50.0); // center (50,25)
                                                     // Straight right → 0.
        approx(
            compute_binding(Point::new(200.0, 25.0), &r).focus,
            0.0,
            1e-9,
        );
        // Straight left → ±1 (atan2(0,-x) = π).
        approx(
            compute_binding(Point::new(-200.0, 25.0), &r).focus.abs(),
            1.0,
            1e-9,
        );
        // Straight down (screen y grows downward) → +0.5.
        approx(
            compute_binding(Point::new(50.0, 200.0), &r).focus,
            0.5,
            1e-9,
        );
        // Straight up → -0.5.
        approx(
            compute_binding(Point::new(50.0, -200.0), &r).focus,
            -0.5,
            1e-9,
        );
    }

    #[test]
    fn rect_roundtrip_every_octant() {
        // The angle encoding makes compute/bound exact inverses in all directions.
        let r = rect_el("r", 10.0, 20.0, 80.0, 40.0); // center (50,40)
        let center = Point::new(50.0, 40.0);
        for k in 0..16 {
            let theta = std::f64::consts::TAU * (k as f64) / 16.0;
            // A point well outside the shape in direction theta.
            let endpoint = Point::new(
                center.x + 200.0 * theta.cos(),
                center.y + 200.0 * theta.sin(),
            );
            let b = compute_binding(endpoint, &r);
            let back = bound_point(&r, b.focus, b.gap);
            approx_pt(back, endpoint, 1e-6);
            assert!(b.gap > 0.0);
            assert!(b.focus.abs() <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn rect_roundtrip_off_center() {
        let r = rect_el("r", 10.0, 20.0, 80.0, 40.0); // center (50,40)
                                                      // An off-center endpoint outside the right edge.
        let endpoint = Point::new(120.0, 55.0);
        let b = compute_binding(endpoint, &r);
        let back = bound_point(&r, b.focus, b.gap);
        approx_pt(back, endpoint, 1e-6);
        assert!(b.gap > 0.0);
        assert!(b.focus.abs() <= 1.0);
    }

    #[test]
    fn rect_gap_is_negative_when_inside_clamped_to_zero() {
        let r = rect_el("r", 0.0, 0.0, 100.0, 100.0);
        // Endpoint inside the rect → raw signed distance negative, gap clamped 0.
        let b = compute_binding(Point::new(50.0, 60.0), &r);
        approx(b.gap, 0.0, 1e-9);
    }

    // ---- ellipse ----

    #[test]
    fn ellipse_focus_zero_on_axis_with_gap() {
        // Ellipse (0,0)-(200,100): center (100,50), rx 100, right vertex x=200.
        // Endpoint (230,50) is 30 past the vertex → focus 0, gap 30.
        let e = ellipse_el("e", 0.0, 0.0, 200.0, 100.0);
        let b = compute_binding(Point::new(230.0, 50.0), &e);
        approx(b.focus, 0.0, 1e-9);
        approx(b.gap, 30.0, 1e-6);
        approx_pt(
            bound_point(&e, b.focus, b.gap),
            Point::new(230.0, 50.0),
            1e-6,
        );
    }

    #[test]
    fn ellipse_roundtrip_off_center() {
        let e = ellipse_el("e", 0.0, 0.0, 200.0, 100.0); // center (100,50)
                                                         // Well outside the ellipse along a diagonal.
        let endpoint = Point::new(280.0, 130.0);
        let b = compute_binding(endpoint, &e);
        let back = bound_point(&e, b.focus, b.gap);
        approx_pt(back, endpoint, 1e-6);
        assert!(b.gap > 0.0);
    }

    // ---- update_bound_arrow: the headline behavior ----

    fn bind_arrow_to_target(
        scene: &mut Scene,
        arrow_id: &str,
        end_target: &Element,
        endpoint: Point,
    ) {
        let binding = compute_binding(endpoint, end_target);
        let arrow = scene.get_mut(&ElementId::from(arrow_id)).unwrap();
        if let ElementKind::Arrow(d) = &mut arrow.kind {
            d.end_binding = Some(binding);
        }
    }

    #[test]
    fn bound_endpoint_tracks_moved_rectangle_with_gap_preserved() {
        let mut scene = Scene::new();

        // Target rect centered at (150,25): (100,0)-(200,50).
        scene.insert(rect_el("r", 100.0, 0.0, 100.0, 50.0));

        // Arrow from (0,25) to (90,25): its end is 10 left of the rect's left
        // edge (x=100), aiming straight at the center horizontally. The ray points
        // in -x, so focus = atan2(0,-60)/PI = 1; colinear gap = 10.
        scene.insert(arrow_el(
            "arrow",
            vec![Point::new(0.0, 25.0), Point::new(90.0, 25.0)],
        ));
        let target = scene.get(&ElementId::from("r")).unwrap().clone();
        bind_arrow_to_target(&mut scene, "arrow", &target, Point::new(90.0, 25.0));

        let binding = match &scene.get(&ElementId::from("arrow")).unwrap().kind {
            ElementKind::Arrow(d) => d.end_binding.clone().unwrap(),
            _ => unreachable!(),
        };
        approx(binding.focus, 1.0, 1e-9);
        approx(binding.gap, 10.0, 1e-9);

        // Recompute against the current target: endpoint sits at left edge - gap.
        let ep = update_bound_arrow(&scene, &ElementId::from("arrow")).unwrap();
        approx_pt(ep.end.unwrap(), Point::new(90.0, 25.0), 1e-9);

        // Now move the rectangle right by 40 and down by 10.
        {
            let r = scene.get_mut(&ElementId::from("r")).unwrap();
            r.x += 40.0;
            r.y += 10.0;
        }
        // New rect (140,10)-(240,60), center (190,35), left edge x=140.
        let ep = update_bound_arrow(&scene, &ElementId::from("arrow")).unwrap();
        // focus 1 → ray straight left through center y=35; gap 10 → x = 140-10 = 130.
        approx_pt(ep.end.unwrap(), Point::new(130.0, 35.0), 1e-9);

        // The gap to the (new) left edge is preserved at 10.
        let new_r = scene.get(&ElementId::from("r")).unwrap();
        let gap = signed_rect_distance(element_bounds(new_r), ep.end.unwrap());
        approx(gap, 10.0, 1e-9);

        // focus stays stable across the move.
        let new_binding = compute_binding(ep.end.unwrap(), new_r);
        approx(new_binding.focus, binding.focus, 1e-6);
        approx(new_binding.gap, binding.gap, 1e-6);
    }

    #[test]
    fn update_keeps_intermediate_points_and_only_moves_bound_end() {
        let mut scene = Scene::new();
        scene.insert(rect_el("r", 100.0, 0.0, 100.0, 50.0));
        // 3-point arrow; only the end is bound.
        scene.insert(arrow_el(
            "arrow",
            vec![
                Point::new(0.0, 25.0),
                Point::new(45.0, 60.0),
                Point::new(90.0, 25.0),
            ],
        ));
        let target = scene.get(&ElementId::from("r")).unwrap().clone();
        bind_arrow_to_target(&mut scene, "arrow", &target, Point::new(90.0, 25.0));

        // Move target.
        scene.get_mut(&ElementId::from("r")).unwrap().x += 40.0;

        let ep = update_bound_arrow(&scene, &ElementId::from("arrow")).unwrap();
        assert!(ep.start.is_none()); // start unbound
        let new_end = ep.end.unwrap();

        // Apply and check the middle point is untouched.
        let arrow = scene.get_mut(&ElementId::from("arrow")).unwrap();
        let origin = Point::new(arrow.x, arrow.y);
        if let ElementKind::Arrow(d) = &mut arrow.kind {
            let changed = apply_bound_endpoints(d, origin, ep);
            assert!(changed);
            assert_eq!(d.points[0], Point::new(0.0, 25.0)); // start unchanged
            assert_eq!(d.points[1], Point::new(45.0, 60.0)); // middle unchanged
            approx_pt(d.points[2], Point::new(new_end.x, new_end.y), 1e-9);
        }
    }

    #[test]
    fn update_unbound_arrow_returns_none_ends() {
        let mut scene = Scene::new();
        scene.insert(arrow_el(
            "arrow",
            vec![Point::new(0.0, 0.0), Point::new(10.0, 0.0)],
        ));
        let ep = update_bound_arrow(&scene, &ElementId::from("arrow")).unwrap();
        assert!(ep.start.is_none());
        assert!(ep.end.is_none());
    }

    #[test]
    fn update_missing_target_yields_none_for_that_end() {
        let mut scene = Scene::new();
        let mut data = LinearData::arrow(vec![Point::new(0.0, 0.0), Point::new(10.0, 0.0)]);
        data.end_arrowhead = Some(Arrowhead::Triangle);
        data.end_binding = Some(PointBinding {
            element_id: ElementId::from("ghost"),
            focus: 0.0,
            gap: 5.0,
        });
        let el = Element::new(
            ElementId::from("arrow"),
            1,
            0.0,
            0.0,
            10.0,
            0.0,
            ElementKind::Arrow(data),
        );
        scene.insert(el);
        let ep = update_bound_arrow(&scene, &ElementId::from("arrow")).unwrap();
        assert!(ep.end.is_none());
    }

    #[test]
    fn update_non_linear_element_returns_none() {
        let mut scene = Scene::new();
        scene.insert(rect_el("r", 0.0, 0.0, 10.0, 10.0));
        assert!(update_bound_arrow(&scene, &ElementId::from("r")).is_none());
    }

    #[test]
    fn signed_rect_distance_inside_outside() {
        let r = Rect::new(0.0, 0.0, 100.0, 50.0);
        // Outside right by 10.
        approx(signed_rect_distance(r, Point::new(110.0, 25.0)), 10.0, 1e-9);
        // Outside corner: (103,54) → (3,4) → 5.
        approx(signed_rect_distance(r, Point::new(103.0, 54.0)), 5.0, 1e-9);
        // Inside near right edge by 5 → negative.
        approx(signed_rect_distance(r, Point::new(95.0, 25.0)), -5.0, 1e-9);
    }
}
