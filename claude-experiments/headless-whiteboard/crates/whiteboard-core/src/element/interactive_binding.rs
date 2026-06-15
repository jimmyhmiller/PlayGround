//! Interactive arrow-binding creation: the pure helper the interaction layer
//! calls while dragging an arrow endpoint.
//!
//! Reimplemented from the binding-creation half of Excalidraw's
//! `packages/element/src/binding.ts` — specifically the work `bindOrUnbindLinearElement`
//! / `maybeBindLinearElement` do while dragging an endpoint: hit-test for a
//! bindable target under the pointer (`getHoveredElementForBinding`), and if one
//! exists, compute the [`PointBinding`] to persist. We do not vendor any
//! JavaScript; this is a thin, side-effect-free composition over the existing
//! [`bindable_element_at`] (hit-test) and [`compute_binding`] (focus/gap math)
//! primitives from `element::binding`.
//!
//! This module is **pure**: every function borrows the [`Scene`] immutably and
//! returns the binding to store (or `None`). The caller owns the mutation — see
//! [`try_bind_endpoint`] for the exact write-back contract.

use crate::geometry::Point;
use crate::scene::Scene;

use super::{bindable_element_at, compute_binding, ElementId, ElementKind, PointBinding};

/// Which end of an arrow/line a binding applies to.
///
/// The interaction layer knows which endpoint the user is dragging; it passes
/// the corresponding [`ArrowEnd`] so the caller can write the resulting binding
/// into the right `LinearData` field (`start_binding` for [`ArrowEnd::Start`],
/// `end_binding` for [`ArrowEnd::End`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowEnd {
    /// The first point of the linear element (`points[0]`).
    Start,
    /// The last point of the linear element (`points[len - 1]`).
    End,
}

/// Find a bindable target under `scene_point` for the given arrow endpoint and
/// return the [`PointBinding`] the caller should store, or `None` if nothing
/// bindable is within `tolerance`.
///
/// This is the function the interaction layer calls on every pointer-move while
/// dragging an arrow endpoint. It:
/// 1. confirms `arrow_id` is a live linear element (arrow or line); otherwise
///    returns `None` — non-linear elements have no endpoints to bind;
/// 2. hit-tests for the topmost bindable element within `tolerance` of
///    `scene_point` via [`bindable_element_at`], **excluding the arrow itself**
///    so an arrow can never bind to itself;
/// 3. computes the focus/gap binding against that target via [`compute_binding`].
///
/// This function is **pure**: it does not mutate the scene. On `Some(binding)`
/// the caller is responsible for two writes that together complete the bind:
///
/// - write `binding` into the arrow's [`LinearData`](super::LinearData):
///   `start_binding` when `which == ArrowEnd::Start`, else `end_binding`; and
/// - push a matching [`BoundElement`](super::BoundElement) with
///   `kind: BoundElementKind::Arrow` and `id == *arrow_id` onto the **target**
///   element's `bound_elements` (skipping any duplicate for this arrow) so the
///   reverse link exists and the target's mover updates this arrow.
///
/// When this returns `None` the caller should *clear* the corresponding
/// endpoint binding (see [`clear_endpoint_binding`]) and remove the arrow from
/// the previously-bound target's `bound_elements`, if it had drifted off a shape.
pub fn try_bind_endpoint(
    scene: &Scene,
    arrow_id: &ElementId,
    which: ArrowEnd,
    scene_point: Point,
    tolerance: f64,
) -> Option<PointBinding> {
    // Only linear elements have bindable endpoints.
    let arrow = scene.get(arrow_id)?;
    if !matches!(arrow.kind, ElementKind::Arrow(_) | ElementKind::Line(_)) {
        return None;
    }
    if arrow.is_deleted {
        return None;
    }
    // `which` selects which endpoint is being dragged; it does not change the
    // hit-test (that is driven by `scene_point`), but it documents intent and
    // lets the caller route the result to the right field.
    let _ = which;

    let target_id = bindable_element_at(scene, scene_point, tolerance)?;
    // Never let an arrow bind to itself.
    if target_id == *arrow_id {
        return None;
    }
    let target = scene.get(&target_id)?;
    Some(compute_binding(scene_point, target))
}

/// The value a caller should assign to an endpoint's binding field when the
/// endpoint is **not** over a bindable target: `None`.
///
/// This exists purely to make the unbind side of the interaction read
/// symmetrically with [`try_bind_endpoint`] at the call site, e.g.
/// `data.end_binding = clear_endpoint_binding();`. There is no state to compute;
/// it always returns `None`. The caller must *also* remove this arrow from the
/// formerly-bound target's `bound_elements`.
#[inline]
pub fn clear_endpoint_binding() -> Option<PointBinding> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Element, LinearData};
    use crate::geometry::Point;

    fn rect(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Rectangle)
    }

    fn arrow(id: &str, points: Vec<Point>) -> Element {
        let data = LinearData::arrow(points);
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

    fn scene(els: Vec<Element>) -> Scene {
        let mut s = Scene::new();
        for e in els {
            s.insert(e);
        }
        s
    }

    #[test]
    fn binds_endpoint_over_target() {
        // Rect (100,0)-(200,50), center (150,25). Drag the arrow's end to (90,25):
        // 10 left of the left edge, within tolerance 20.
        let s = scene(vec![
            rect("r", 100.0, 0.0, 100.0, 50.0),
            arrow("a", vec![Point::new(0.0, 25.0), Point::new(90.0, 25.0)]),
        ]);
        let b = try_bind_endpoint(
            &s,
            &ElementId::from("a"),
            ArrowEnd::End,
            Point::new(90.0, 25.0),
            20.0,
        )
        .expect("should bind to the rect");
        assert_eq!(b.element_id, ElementId::from("r"));
        // Endpoint straight left of center → focus 1, gap 10 (left edge x=100).
        assert!((b.focus - 1.0).abs() < 1e-9, "focus {}", b.focus);
        assert!((b.gap - 10.0).abs() < 1e-9, "gap {}", b.gap);
    }

    #[test]
    fn no_target_returns_none() {
        let s = scene(vec![
            rect("r", 100.0, 0.0, 100.0, 50.0),
            arrow("a", vec![Point::new(0.0, 25.0), Point::new(10.0, 25.0)]),
        ]);
        // Far from the rect, beyond tolerance.
        assert!(try_bind_endpoint(
            &s,
            &ElementId::from("a"),
            ArrowEnd::Start,
            Point::new(0.0, 25.0),
            5.0
        )
        .is_none());
    }

    #[test]
    fn arrow_never_binds_to_itself() {
        // Only the arrow exists; even though the point is on the arrow, arrows are
        // not bindable, and the self-exclusion guards regardless.
        let s = scene(vec![arrow(
            "a",
            vec![Point::new(0.0, 0.0), Point::new(100.0, 0.0)],
        )]);
        assert!(try_bind_endpoint(
            &s,
            &ElementId::from("a"),
            ArrowEnd::End,
            Point::new(50.0, 0.0),
            10.0
        )
        .is_none());
    }

    #[test]
    fn picks_topmost_bindable_target() {
        let s = scene(vec![
            rect("bottom", 0.0, 0.0, 100.0, 100.0),
            rect("top", 0.0, 0.0, 100.0, 100.0),
            arrow("a", vec![Point::new(-50.0, 50.0), Point::new(50.0, 50.0)]),
        ]);
        let b = try_bind_endpoint(
            &s,
            &ElementId::from("a"),
            ArrowEnd::End,
            Point::new(50.0, 50.0),
            5.0,
        )
        .expect("binds");
        assert_eq!(b.element_id, ElementId::from("top"));
    }

    #[test]
    fn non_linear_element_yields_none() {
        let s = scene(vec![rect("r", 0.0, 0.0, 100.0, 50.0)]);
        assert!(try_bind_endpoint(
            &s,
            &ElementId::from("r"),
            ArrowEnd::End,
            Point::new(50.0, 25.0),
            5.0
        )
        .is_none());
    }

    #[test]
    fn deleted_arrow_yields_none() {
        let mut a = arrow("a", vec![Point::new(0.0, 25.0), Point::new(90.0, 25.0)]);
        a.is_deleted = true;
        let s = scene(vec![rect("r", 100.0, 0.0, 100.0, 50.0), a]);
        assert!(try_bind_endpoint(
            &s,
            &ElementId::from("a"),
            ArrowEnd::End,
            Point::new(90.0, 25.0),
            20.0
        )
        .is_none());
    }

    #[test]
    fn clear_endpoint_binding_is_none() {
        assert!(clear_endpoint_binding().is_none());
    }

    #[test]
    fn line_endpoint_also_binds() {
        let data = LinearData::line(vec![Point::new(0.0, 25.0), Point::new(90.0, 25.0)]);
        let line = Element::new(
            ElementId::from("l"),
            1,
            0.0,
            0.0,
            0.0,
            0.0,
            ElementKind::Line(data),
        );
        let s = scene(vec![rect("r", 100.0, 0.0, 100.0, 50.0), line]);
        let b = try_bind_endpoint(
            &s,
            &ElementId::from("l"),
            ArrowEnd::End,
            Point::new(90.0, 25.0),
            20.0,
        )
        .expect("line endpoint binds");
        assert_eq!(b.element_id, ElementId::from("r"));
    }
}
